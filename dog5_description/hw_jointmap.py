"""Joint mapping + encoder state conversion for the DOG5 hardware port.

Bridges three naming/ordering worlds:

  * DOG5 MuJoCo model (`dog5.xml`) - joints `hip_abd_{leg}`, `hip_pitch_{leg}`,
    `knee_{leg}` for leg in FL, FR, RL, RR.  This file's CANONICAL order is the
    MuJoCo body order: [FL, FR, RL, RR] x [abd, pitch, knee], index 0..11.
  * `hil_map.json` (written by `hil_identify.py --wiggle`) - keys
    `{LEG}_{abd|thigh|knee}` with `{can_id, dir}`.  Note `thigh` == DOG5
    `hip_pitch`.  This is the source of truth for motor id + sign.
  * LKMTECH motors on CAN - integer ids 1..12, single-turn encoder on the
    MOTOR shaft behind a 10:1 reducer, so the absolute output angle is only
    known modulo 36 deg.  `EncoderUnwrap` recovers a continuous multi-turn
    output angle referenced to the DOG5 flat calibration pose.

Nothing here talks to CAN or MuJoCo; it is pure bookkeeping so it can be
unit-tested off-hardware (`python hw_jointmap.py`).
"""
from __future__ import annotations

import json
import math
import os

# CANONICAL controller order == MuJoCo body/joint order in dog5.xml.
LEGS = ["FL", "FR", "RL", "RR"]
JOINTS = ["abd", "pitch", "knee"]          # DOG5 short names, per leg
N = 12

# DOG5 MuJoCo joint name for a (leg, short-joint) pair.
_DOG5_PREFIX = {"abd": "hip_abd", "pitch": "hip_pitch", "knee": "knee"}
# hil_map.json uses "thigh" where DOG5 uses "pitch"; abd/knee are shared.
_HILMAP_SUFFIX = {"abd": "abd", "pitch": "thigh", "knee": "knee"}

GEAR = 10.0                                # 10:1 reducer (motor shaft -> output)
ENCODER_GAIN = 360.0 / 65535.0             # MOTOR-deg per 16-bit encoder LSB
_ENC_FULL = 65536                          # single-turn encoder wrap period (LSB)


def canonical_joints():
    """List of 12 (index, leg, short_joint, dog5_name, hilmap_name) tuples."""
    out = []
    for leg in LEGS:
        for jt in JOINTS:
            idx = len(out)
            out.append((
                idx, leg, jt,
                f"{_DOG5_PREFIX[jt]}_{leg}",
                f"{leg}_{_HILMAP_SUFFIX[jt]}",
            ))
    return out


def load_hil_map(path):
    """Read hil_map.json -> {hilmap_name: {'can_id': int, 'dir': +-1}}.

    Ignores the leading `_provisional` note key.  Raises if a real joint entry
    is malformed.
    """
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        out[k] = {"can_id": int(v["can_id"]), "dir": int(v["dir"])}
    return out


class HwMap:
    """Canonical-order motor id + sign vectors, resolved from hil_map.json.

    `mid[i]` / `sign[i]` give the CAN id and direction for canonical joint `i`.
    `sign` folds the hil_map `dir` (encoder-positive == MuJoCo-positive) so it
    is applied symmetrically: MuJoCo-frame value = sign * motor-frame value, and
    motor command = sign * MuJoCo-frame torque.
    """

    def __init__(self, hil_map):
        self.joints = canonical_joints()
        self.mid = [None] * N
        self.sign = [1] * N
        missing = []
        for idx, leg, jt, dog5_name, hil_name in self.joints:
            entry = hil_map.get(hil_name)
            if entry is None or entry["can_id"] is None:
                missing.append(hil_name)
                continue
            self.mid[idx] = entry["can_id"]
            self.sign[idx] = 1 if entry["dir"] >= 0 else -1
        if missing:
            raise ValueError(
                "hil_map.json missing/unmapped joints: %s -- run "
                "`hil_identify.py --wiggle` first." % ", ".join(missing))
        if len(set(self.mid)) != N:
            raise ValueError("hil_map.json has duplicate can_ids: %s" % self.mid)

    @classmethod
    def from_file(cls, path):
        return cls(load_hil_map(path))

    def dirs_for_motorbus(self):
        """{can_id: dir} dict for MotorBus(..., dirs=...).

        MotorBus already applies `dirs` inside torque()/encoders_deg(), so if we
        hand it these signs the values it returns are already in the MuJoCo
        frame and we must NOT sign them again.  We instead keep MotorBus at
        dir=+1 and apply `self.sign` ourselves (see stand_dog5_hw.py), which
        keeps all sign logic in one place.  This helper is provided for callers
        that prefer to push signs into MotorBus.
        """
        return {mid: s for mid, s in zip(self.mid, self.sign)}


class EncoderUnwrap:
    """Continuous multi-turn OUTPUT angle from a wrapped single-turn encoder.

    Feed the raw 16-bit motor-shaft encoder count each tick.  Tracks wraps and
    returns output degrees relative to a zero captured at the flat calibration
    pose.  Mirrors the unwrap logic in run_compliance_2dof.py / trajectory_follow.py.
    """

    def __init__(self):
        self._prev_raw = None
        self._turns = 0            # net motor-shaft wraps
        self._zero_motor_deg = 0.0  # motor-deg at the flat calibration pose

    def reset_zero(self, raw):
        """Capture `raw` (current encoder LSB) as the flat-pose zero."""
        self._prev_raw = raw
        self._turns = 0
        self._zero_motor_deg = raw * ENCODER_GAIN

    def update(self, raw):
        """Return continuous OUTPUT angle in degrees for encoder count `raw`."""
        if self._prev_raw is None:
            self.reset_zero(raw)
        d = raw - self._prev_raw
        if d > _ENC_FULL // 2:
            self._turns -= 1
        elif d < -(_ENC_FULL // 2):
            self._turns += 1
        self._prev_raw = raw
        motor_deg = (raw + self._turns * _ENC_FULL) * ENCODER_GAIN
        return (motor_deg - self._zero_motor_deg) / GEAR

    @property
    def output_deg(self):
        motor_deg = (self._prev_raw + self._turns * _ENC_FULL) * ENCODER_GAIN
        return (motor_deg - self._zero_motor_deg) / GEAR


def deg2rad(d):
    return d * math.pi / 180.0


def rad2deg(r):
    return r * 180.0 / math.pi


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    cj = canonical_joints()
    assert len(cj) == 12
    # spot-check aliasing: canonical index 1 is FL pitch -> hil FL_thigh
    assert cj[1][3] == "hip_pitch_FL" and cj[1][4] == "FL_thigh", cj[1]
    assert cj[2][3] == "knee_FL" and cj[2][4] == "FL_knee", cj[2]
    assert cj[3][1] == "FR" and cj[3][2] == "abd", cj[3]

    # the current placeholder hil_map.json maps FR=1-3, FL=4-6, RR=7-9, RL=10-12
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    mp = os.path.join(repo, "hil_map.json")
    hm = HwMap.from_file(mp)
    # canonical order is FL first, so mid[0..2] are FL abd/pitch/knee == ids 4,5,6
    assert hm.mid[0:3] == [4, 5, 6], hm.mid
    assert hm.mid[3:6] == [1, 2, 3], hm.mid      # FR
    assert hm.mid[6:9] == [10, 11, 12], hm.mid   # RL
    assert hm.mid[9:12] == [7, 8, 9], hm.mid     # RR
    assert all(s == 1 for s in hm.sign)

    # encoder unwrap: forward motion crossing the single-turn wrap.
    # Per-tick deltas must stay < half a period (true at 250 Hz); the only
    # boundary crossing here is 64000 -> 3000 (delta -61000 => +1 turn).
    eu = EncoderUnwrap()
    eu.reset_zero(1000)
    assert abs(eu.update(1000) - 0.0) < 1e-9
    for raw in (9000, 40000, 64000, 3000, 20000):
        v = eu.update(raw)
    exp = ((20000 + _ENC_FULL - 1000) * ENCODER_GAIN) / GEAR   # net +1 turn
    assert abs(v - exp) < 1e-6, (v, exp)
    # step backward through the wrap: 3000 -> 64000 (delta +61000 => -1 turn)
    eu2 = EncoderUnwrap()
    eu2.reset_zero(3000)
    eu2.update(3000)
    v_back = eu2.update(64000)
    exp_back = ((64000 - _ENC_FULL - 3000) * ENCODER_GAIN) / GEAR  # net -1 turn
    assert abs(v_back - exp_back) < 1e-6, (v_back, exp_back)

    print("hw_jointmap self-test OK")
    print("  canonical order:", [c[3] for c in cj])
    print("  can ids        :", hm.mid)
    print("  signs          :", hm.sign)
