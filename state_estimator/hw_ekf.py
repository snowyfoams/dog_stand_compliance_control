#!/usr/bin/env python3
"""Run the real state estimator on DOG5 hardware, READ-ONLY.

Reads the DETA10 IMU (specific force + rate, FLU) and the 12 joint encoders over
CAN, feeds them to `DOG5StateEstimator`, and prints/logs the estimated base
state next to the IMU's own AHRS attitude for cross-checking. **No torque is
ever commanded** -- every motor is held in zero-torque keepalive, so the robot
does not actuate. This is the hardware-validation harness of
`state_estimator/HW_BRINGUP_PLAN.md` (Phase D, online read-only).

Safety: mechanically support / stand the robot. Motors stay passive; you can
hand-manipulate the body (e.g. rock it ±5-10 deg to observe bias) while it runs.

Usage -- MUST use the project venv (it has python-can + pyserial; the system
python3 has neither). From this directory:
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V hw_ekf.py --duration 30
    $V hw_ekf.py --duration 30 --log run.csv
    $V hw_ekf.py --contacts 1,1,1,1             # which feet are on the ground

Encoder decode + CAN patterns mirror dog5_description/stand_dog5_hw.py (the
verified hardware contract); we re-derive the small pieces here so this harness
needs neither mujoco nor the control stack.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DESC = os.path.join(os.path.dirname(_HERE), "dog5_description")
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _DESC, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import motorbus                                    # noqa: E402 (needs python-can)
from dog5_hardware_map import HARDWARE_JOINTS      # noqa: E402
from dog5_state_estimator import DOG5StateEstimator, quat_to_C  # noqa: E402
from imu_ekf_feed import ImuEkfFeed                # noqa: E402
from imu_dog import DEFAULT_PORT                    # noqa: E402

LEGS = ("FL", "FR", "RL", "RR")
MOTOR_IDS = [j.can_id for j in HARDWARE_JOINTS]
MOTOR_DIRECTIONS = {j.can_id: j.direction for j in HARDWARE_JOINTS}
JOINT_DIRECTIONS = np.array([j.direction for j in HARDWARE_JOINTS], dtype=float)
N_JOINTS = len(HARDWARE_JOINTS)
CONTROL_HZ = 250.0
DT_CLAMP = (1.0e-4, 0.05)      # sane per-sample predict dt bounds


class CalibratedEncoderUnwrap:
    """Continuous calibrated motor-output degrees (zero fixed by driver 0x19).

    Copied from stand_dog5_hw.CalibratedEncoderUnwrap: centre the first sample
    into [-180, 180), then track 16-bit wraps. No software offset, no gear
    division -- the value is already output-referenced joint degrees.
    """

    _FULL_TURN = 65536

    def __init__(self):
        self._previous_raw = None
        self._turns = 0

    def update(self, raw: int) -> float:
        raw = int(raw)
        if not 0 <= raw < self._FULL_TURN:
            raise ValueError(f"encoder value outside uint16 range: {raw}")
        if self._previous_raw is None:
            self._turns = -1 if raw >= self._FULL_TURN // 2 else 0
        else:
            delta = raw - self._previous_raw
            if delta > self._FULL_TURN // 2:
                self._turns -= 1
            elif delta < -self._FULL_TURN // 2:
                self._turns += 1
        self._previous_raw = raw
        return (raw + self._turns * self._FULL_TURN) * motorbus.ENCODER_GAIN


def joint_state(mb, unwrap):
    """(q(12), qd(12)) in radians, order [FL,FR,RL,RR]x[abd,pitch,knee].

    Mirrors stand_dog5_hw._joint_state: joint_rad = direction * motoroutput_deg,
    no zero offset, no gearbox division (confirmed hardware relationship).
    """
    raw = [mb.rec(mid).encoder for mid in MOTOR_IDS]
    missing = [mid for mid, v in zip(MOTOR_IDS, raw) if v is None]
    if missing:
        raise RuntimeError(f"no encoder reply from CAN IDs: {missing}")
    motoroutput_deg = np.array([t.update(r) for t, r in zip(unwrap, raw)])
    q = np.deg2rad(JOINT_DIRECTIONS * motoroutput_deg)
    speeds = mb.speeds_dps()
    qd = np.deg2rad(np.array([speeds[mid] for mid in MOTOR_IDS]))
    return q, qd


def roll_pitch_from_q(q_est):
    """Yaw-free roll/pitch (rad) from the estimator quaternion (C maps I->B)."""
    g_b = quat_to_C(q_est) @ np.array([0.0, 0.0, 1.0])
    roll = math.atan2(g_b[1], g_b[2])
    pitch = math.atan2(-g_b[0], math.hypot(g_b[1], g_b[2]))
    return roll, pitch


def run(port, duration, init_secs, contacts, log_path=None, raw_log_path=None):
    contacts = np.asarray(contacts, dtype=bool)
    unwrap = [CalibratedEncoderUnwrap() for _ in MOTOR_IDS]
    est = DOG5StateEstimator()

    # raw-input log for offline replay (gate C8): the two sensor streams with a
    # single monotonic clock, plus the AHRS attitude reference and contacts.
    imu_log = []   # (t_mono, fx,fy,fz, wx,wy,wz)
    enc_log = []   # (t_mono, alpha[12], contacts[4], ahrs_roll, ahrs_pitch)

    writer = None
    log_file = None
    if log_path:
        log_file = open(log_path, "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["t", "rx", "ry", "rz", "vx", "vy", "vz",
                         "roll_est", "pitch_est", "roll_ahrs", "pitch_ahrs",
                         "healthy", "min_eig_P", "imu_hz", "n_imu"])

    print("=" * 70)
    print("DOG5 hardware EKF -- READ-ONLY. Motors held at zero torque (keepalive).")
    print("Support/stand the robot. You may hand-rock the body to observe bias.")
    print("=" * 70)

    with motorbus.MotorBus(MOTOR_IDS, dirs=MOTOR_DIRECTIONS) as mb, \
            ImuEkfFeed(port) as feed:
        if not mb.arm(rate_hz=CONTROL_HZ):
            raise RuntimeError("motor arm failed (check bus / power / terminators)")
        if not feed.wait_for_raw(timeout=3.0):
            raise RuntimeError(
                "no raw IMU (0x40) packets -- enable the DETA10 raw stream "
                "(ImuDog only wires AHRS by default)")

        slot = mb.slot(CONTROL_HZ)
        deadline = time.perf_counter() + slot
        index = 0

        # ---- static init window: gather IMU + a joint pose, then initialise ----
        init_f, init_w = [], []
        alpha0 = None
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < init_secs:
            mid = MOTOR_IDS[index % N_JOINTS]
            mb.poll()
            mb.keepalive(mid)
            if index % N_JOINTS == 0:
                for f, w, tm in feed.drain():
                    init_f.append(f)
                    init_w.append(w)
                    imu_log.append((tm, *f, *w))
                try:
                    q, _qd = joint_state(mb, unwrap)
                    alpha0 = q.reshape(4, 3)
                except RuntimeError:
                    pass
            mb.pace(deadline)
            deadline += slot
            index += 1

        if alpha0 is None or len(init_f) < 10:
            raise RuntimeError("init window collected no encoders or IMU data")
        est.initialise(np.array(init_f), np.array(init_w), alpha0, contacts)
        print(f"[init] {len(init_f)} IMU samples, IMU ~{feed.rate_hz:.0f} Hz, "
              f"gyro bias={np.round(est.state.bw, 5)}  "
              f"accel bias={np.round(est.state.bf, 4)}")

        # ---- main read-only loop ----
        last_t = time.monotonic()
        w_last = init_w[-1]
        loop_start = time.perf_counter()
        last_print = 0.0
        while time.perf_counter() - loop_start < duration:
            mid = MOTOR_IDS[index % N_JOINTS]
            mb.poll()
            mb.keepalive(mid)
            if index % N_JOINTS == 0:
                # batch every buffered IMU sample into predict, then one update
                samples = feed.drain()
                for f, w, tm in samples:
                    dt = min(max(tm - last_t, DT_CLAMP[0]), DT_CLAMP[1])
                    est.predict(f, w, dt, contacts)
                    last_t, w_last = tm, w
                    imu_log.append((tm, *f, *w))
                q_alpha = None
                try:
                    q, _qd = joint_state(mb, unwrap)
                    q_alpha = q
                    est.update(q.reshape(4, 3), contacts)
                except RuntimeError:
                    pass

                out = est.outputs(last_w_meas=w_last)
                roll_e, pitch_e = roll_pitch_from_q(out["q"])
                att = feed.attitude()
                r_a = math.radians(att.roll_deg) if att else float("nan")
                p_a = math.radians(att.pitch_deg) if att else float("nan")
                if q_alpha is not None:
                    enc_log.append((time.monotonic(), *q_alpha,
                                    *contacts.astype(int), r_a, p_a))
                now = time.perf_counter() - loop_start
                if writer:
                    writer.writerow([f"{now:.4f}", *[f"{x:.5f}" for x in out["r"]],
                                     *[f"{x:.5f}" for x in out["v"]],
                                     f"{roll_e:.5f}", f"{pitch_e:.5f}",
                                     f"{r_a:.5f}", f"{p_a:.5f}",
                                     int(out["healthy"]), f"{out['min_eig_P']:.2e}",
                                     f"{feed.rate_hz:.1f}", len(samples)])
                if now - last_print >= 0.5:
                    last_print = now
                    r_mm = out["r"] * 1e3
                    v_mm = out["v"] * 1e3
                    # x,y drift (unobservable) -- shown for completeness; only z,
                    # roll, pitch and the velocity are trustworthy state.
                    print(f"t={now:5.1f} "
                          f"r(mm)=[{r_mm[0]:+7.1f} {r_mm[1]:+7.1f} {r_mm[2]:+6.1f}] "
                          f"v(mm/s)=[{v_mm[0]:+5.1f} {v_mm[1]:+5.1f} {v_mm[2]:+5.1f}] "
                          f"roll={math.degrees(roll_e):+6.2f}(ahrs{math.degrees(r_a):+6.2f}) "
                          f"pitch={math.degrees(pitch_e):+6.2f}(ahrs{math.degrees(p_a):+6.2f}) "
                          f"{'OK' if out['healthy'] else '!!'} "
                          f"imu={feed.rate_hz:.0f}Hz")
            mb.pace(deadline)
            deadline += slot
            index += 1

        mb.stop_all()

    if log_file:
        log_file.close()
        print(f"[log] wrote {log_path}")
    if raw_log_path:
        imu = np.array(imu_log, dtype=float)      # (N, 7): t, f(3), w(3)
        enc = np.array(enc_log, dtype=float)      # (M, 19): t, alpha(12), c(4), rp(2)
        np.savez(raw_log_path,
                 imu_t=imu[:, 0], imu_f=imu[:, 1:4], imu_w=imu[:, 4:7],
                 enc_t=enc[:, 0], enc_alpha=enc[:, 1:13],
                 enc_contacts=enc[:, 13:17], ahrs_rp=enc[:, 17:19],
                 init_secs=float(init_secs))
        print(f"[raw] wrote {raw_log_path}  ({len(imu)} IMU, {len(enc)} encoder frames)")


def _parse_contacts(s):
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("contacts must be 4 comma-separated 0/1")
    return parts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--init-secs", type=float, default=1.0)
    ap.add_argument("--contacts", type=_parse_contacts, default=[1, 1, 1, 1],
                    help="which feet are on the ground, e.g. 1,1,1,1")
    ap.add_argument("--log", default=None, help="CSV of estimator outputs (quick look)")
    ap.add_argument("--raw-log", default=None,
                    help="NPZ of raw IMU+encoder inputs for offline replay (gate C8)")
    args = ap.parse_args()
    run(args.port, args.duration, args.init_secs, args.contacts, args.log, args.raw_log)


if __name__ == "__main__":
    main()
