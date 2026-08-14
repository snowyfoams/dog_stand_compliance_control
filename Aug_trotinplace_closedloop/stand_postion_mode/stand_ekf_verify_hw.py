#!/usr/bin/env python3
"""Step 1 of the closed-loop trot-in-place track: POSITION stand, EKF READ-ONLY.

This script commands NO torque and closes NO loop.  Every joint is held by the
driver's native 0xA4 position loop -- the same proven position stand used by
`vmc/ekf_stand_hw.py` and `crawl_hw2.0/stand3_hold_hw.py`.  The estimator runs
in a worker thread and only observes.

Its single job is to answer the questions step 2 needs answered before the EKF
can be trusted as *feedback*:

  1. Does the EKF agree with the AHRS on roll/pitch while standing still?
     (printed live and summarised as |EKF - AHRS| mean/max at exit)
  2. What is the RESTING roll/pitch of the standing robot?  This is the bias
     the leveling controller must be told about -- the correct setpoint for
     step 2 is this measured resting attitude, NOT a blind zero, because the
     EKF's attitude is gravity-referenced through the IMU's *mechanical mount*.
     (The AHRS path has its own calibrated roll/pitch offsets in imu_dog; the
     EKF path does not, so the mount tilt shows up raw here.)
  3. Is the attitude quiet enough to close a loop on?  A drifting or noisy
     roll/pitch at rest sets the floor on how tight step 2 can hold.

Flow (all POSITION mode, operator-gated):
    ZERO-TORQUE check
      -> CROUCH        native 0xA4 position to the recorded crouch
      -> WAIT_CROUCH   hold still; the EKF initialises on this static pose
                       (quiet stage -- gyro bias is estimated here).  ENTER
                       starts the stand.
      -> STAND         joint interpolation crouch -> precomputed stand pose,
                       streamed as position.  contacts stay ON: the feet never
                       leave the floor, so the EKF keeps its height reference
                       (hardware-confirmed to remove the drift entirely).
      -> HOLD4         hold the stand pose; contacts=ON re-anchor.  Attitude
                       statistics are collected here after a settle delay.
                       Hand-rock the body to exercise attitude/bias.
                       P parks, X stops.
      -> PARK          the STAND ramp run backwards: stand -> crouch over the
                       same T_STAND.
      -> PARKED        hold the crouch.  ENTER stands again -- each HOLD4 visit
                       is scored separately, so a park/stand cycle measures how
                       REPEATABLE the resting attitude is.  X stops.

Q_STAND is precomputed with incremental warm-started IK so it stays on the
crouch branch (a one-shot solve can flip the knee).  The stand streams a
coordinated joint interpolation, so the CAN sweep needs no per-slot IK.

Rates: `MotorBus.slot(rate_hz)` is `1/(rate_hz * n_motors)`, so at CONTROL_HZ
250 with 12 motors every motor is commanded at 250 Hz and a full sweep -- this
loop's `joint_index == 0` block -- also runs at 250 Hz.  The EKF worker runs
off-thread at 100 Hz.

Run:
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V stand_ekf_verify_hw.py --self-test            # no hardware needed
    sudo chrt -f 50 $V stand_ekf_verify_hw.py \
        --log verify.csv --raw-log verify.npz
(sudo is safe here: this script resolves fdilink_imu repo-relative, so the
$HOME=/root import break recorded in 7.28review.md does not apply.)
Then replay for the gate-C8 report:
    $V ../state_estimator/hw_replay.py verify.npz --static
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import threading
import time

import numpy as np

sys.setswitchinterval(0.0005)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_up(name, start=None):
    """Nearest ancestor directory that CONTAINS `name` (falsy if none does).

    The runners were a fixed two levels below the repo root until they moved
    into stand_postion_mode/, which silently broke every one of these imports
    -- `dog5_description` was suddenly three levels up, not two.  Anchoring on
    a directory that is actually there survives the next move.
    """
    d = start or _HERE
    while True:
        if os.path.isdir(os.path.join(d, name)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


# _PARENT is the track root (the directory holding dog5_description and
# state_estimator); _REPO is can_motor_control above it.
_PARENT = _find_up("dog5_description") or os.path.dirname(_HERE)
_DESC = os.path.join(_PARENT, "dog5_description")
_EST = os.path.join(_PARENT, "state_estimator")
_REPO = os.path.dirname(_PARENT)
_IMU = os.path.join(_REPO, "IMU_sensor")
for _p in (_HERE, _DESC, _EST, _REPO, _IMU):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def selftest_dir():
    """Locate self-test/, wherever the runners currently sit relative to it."""
    root = _find_up("self-test")
    return os.path.join(root or os.path.dirname(_HERE), "self-test")


def _add_fdilink_root():
    """Put the `fdilink_imu` package on the path WITHOUT trusting $HOME.

    imu_dog.py resolves it as `Path.home()/Documents/IMU_sensor`, which breaks
    under `sudo` ($HOME becomes /root) -- the failure recorded in 7.28review.md.
    Since RT priority (`chrt`) wants root, resolve it repo-relative first and
    fall back to the invoking user's home, so both invocations work.
    """
    candidates = [os.path.join(os.path.dirname(_REPO), "IMU_sensor"),
                  os.path.join(os.path.expanduser("~"), "Documents", "IMU_sensor")]
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user:
        candidates.append(os.path.join("/home", sudo_user, "Documents", "IMU_sensor"))
    for root in candidates:
        if os.path.isdir(os.path.join(root, "fdilink_imu")):
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    return None


if _add_fdilink_root() is None:
    print("[warn] fdilink_imu package not found; the IMU import will likely fail",
          file=sys.stderr)

import motorbus                                          # noqa: E402
import stand_dog5_hw as base                             # noqa: E402 (low-level lib)
import stand_dog5_recorded_hw as recorded                # noqa: E402 (recorded crouch)
import dog5_kinematics                                   # noqa: E402
from ekf_runtime import EkfShared, ekf_worker, _rp       # noqa: E402
from imu_dog import DEFAULT_PORT                         # noqa: E402

# Every tunable this script has is in stand_params.py -- including the frame
# constants below, which stages 2/2b/A/lift all measure against.
from stand_params import (FOOT_RADIUS_M,                 # noqa: E402
                          IMU_BELOW_TRUNK_ORIGIN_M,
                          T_STAND, EKF_WORKER_HZ, QUIET_STAGES, SETTLE_S,
                          MIN_INIT_QUIET_S, STAND_HEIGHT_STAGE12)

MOTOR_IDS = base.MOTOR_IDS
N_JOINTS = base.N_JOINTS
CONTROL_HZ = base.CONTROL_HZ                             # per motor AND per sweep
LEGS = base.LEGS
Q_CROUCH = recorded.Q_RECORDED_CROUCH
POSITION_TARGET_DEG = recorded.POSITION_TARGET_DEG       # crouch, controller-order deg

assert abs(STAND_HEIGHT_STAGE12 - recorded.DEFAULT_STAND_HEIGHT) < 1e-12, (
    "stand_params.STAND_HEIGHT_STAGE12 has drifted from the recorded default")


def _smoothstep(u):
    u = float(np.clip(u, 0.0, 1.0))
    return 3 * u * u - 2 * u * u * u


def compute_q_stand(stand_height):
    """Stand joint pose: each foot held at its crouch x/y, lowered to
    -stand_height in the trunk frame, via INCREMENTAL warm-started IK."""
    q_stand = Q_CROUCH.copy()
    for i, leg in enumerate(LEGS):
        sl = slice(3 * i, 3 * i + 3)
        q = Q_CROUCH[sl].copy()
        f0 = dog5_kinematics.foot_position(leg, q)
        for n in range(1, 31):
            z = f0[2] + (n / 30.0) * (-stand_height - f0[2])
            tgt = np.array([f0[0], f0[1], z])
            for _ in range(60):
                e = tgt - dog5_kinematics.foot_position(leg, q)
                if np.linalg.norm(e) < 1e-10:
                    break
                J = dog5_kinematics.foot_jacobian(leg, q)
                q = q + 0.5 * J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e)
        q_stand[sl] = q
    lo, hi = base.soft_limits()
    if np.any(q_stand < lo) or np.any(q_stand > hi):
        raise RuntimeError("computed stand pose exceeds soft joint limits")
    return q_stand


def fk_floor_height(q, C=None, ref="imu"):
    """Height above the FLOOR of a trunk reference point, from encoders alone.

    This is the drift-free height reference: pure forward kinematics, no
    integration of anything.  `C` maps inertial -> body (the EKF's), so the
    feet are rotated into the world before averaging and a tilted trunk is
    handled correctly; pass None to assume level.

    `ref` picks WHICH point on the trunk:
        "hip"  the FK trunk origin = the hip-axis plane.  This is the pure
               kinematic quantity and the one a tape measures to the hip pivot.
        "imu"  (default) the IMU board, IMU_BELOW_TRUNK_ORIGIN_M below the
               trunk origin along body -z.  This is the point the EKF's `r`
               actually tracks, so it is the like-for-like comparison.

    The offset is a BODY-frame constant, so its vertical component shrinks as
    the trunk tilts -- hence `C[2, 2]` rather than a flat subtraction.  At the
    couple of degrees this robot stands at the two differ by ~0.02 mm, so the
    choice of `ref` shifts the ABSOLUTE height by 38 mm but leaves the EKF-vs-FK
    drift essentially untouched (a constant cancels in `z_fk - z_fk_at_init`).

    Averaged over all four legs -- valid whenever the four feet are on one
    flat floor, which is true in every stage this script holds still in.
    """
    q = np.asarray(q, dtype=float)
    zs = []
    for i, leg in enumerate(LEGS):
        s = dog5_kinematics.foot_position(leg, q[3 * i:3 * i + 3])   # site in B
        zs.append((C.T @ s)[2] if C is not None else s[2])
    h_hip = -float(np.mean(zs)) + FOOT_RADIUS_M
    if ref == "hip":
        return h_hip
    if ref != "imu":
        raise ValueError(f"ref must be 'imu' or 'hip', got {ref!r}")
    # IMU offset is [0, 0, -d] in BODY coords; its inertial z-component is
    # (C^T @ [0,0,-d])[2] = -d * C[2,2].
    cz = float(C[2, 2]) if C is not None else 1.0
    return h_hip - IMU_BELOW_TRUNK_ORIGIN_M * cz


class StageReport:
    """One row per HOLDING stage (WAIT_CROUCH, HOLD4, PARKED) of everything
    that verifies the EKF: FK height, EKF z, their disagreement, attitude.

    The EKF's `r` is a DISPLACEMENT from wherever the trunk was when
    `initialise()` ran (it sets `st.r = 0`), expressed in gravity-aligned axes.
    So its z is world-frame in DIRECTION but its origin is the crouch, and it
    is only ever observable relative to the currently anchored footholds -- it
    is not a height above the ground.  FK height is.  Lining the two up:

        drift = z_ekf - (z_fk - z_fk_at_init)

    i.e. how far the EKF's integrated rise disagrees with what the legs say.
    Any non-zero drift at PARKED is error the filter accumulated while the feet
    were not in contact and then froze in when the footholds re-anchored.

    WAIT_CROUCH is the baseline row: it is the pose the origin was latched at,
    so its drift is ~0 by construction.  A NON-zero drift there would mean the
    FK height or the origin latch itself is wrong -- which is why it is worth
    printing rather than assuming.
    """

    STAGES = ("WAIT_CROUCH", "HOLD4", "PARKED")

    def __init__(self):
        self.z0_fk = None          # FK floor height at the moment the EKF init'd
        self.rows = {}             # stage -> list of (z_fk, z_ekf, drift, roll, pitch)

    def set_origin(self, z_fk):
        if self.z0_fk is None:
            self.z0_fk = float(z_fk)

    def add(self, stage, z_fk, z_ekf, roll=float("nan"), pitch=float("nan")):
        if self.z0_fk is None:
            return None
        drift = float(z_ekf) - (float(z_fk) - self.z0_fk)
        self.rows.setdefault(stage, []).append(
            (float(z_fk), float(z_ekf), drift, float(roll), float(pitch)))
        return drift

    def summary(self):
        if self.z0_fk is None:
            return ["[stages] EKF never initialised -- no height comparison"]
        lines = [f"[stages] EKF z=0 is the IMU at EKF init: FK floor height "
                 f"{self.z0_fk*1e3:.0f} mm (the crouch)",
                 f"         zFK = IMU board above floor; the hip axis (what a "
                 f"tape measures) is +{IMU_BELOW_TRUNK_ORIGIN_M*1e3:.0f} mm",
                 "  stage          n     zFK      zEKF   drift mean/max"
                 "      roll   pitch"]
        for stage in self.STAGES:
            rows = self.rows.get(stage)
            if not rows:
                continue
            a = np.array(rows)
            worst = a[np.argmax(np.abs(a[:, 2])), 2]
            lines.append(
                f"  {stage:11s} {len(a):5d} {a[:, 0].mean()*1e3:6.0f}mm "
                f"{a[:, 1].mean()*1e3:+7.0f}mm "
                f"{a[:, 2].mean()*1e3:+6.0f}/{worst*1e3:+6.0f}mm "
                f"{math.degrees(np.nanmean(a[:, 3])):+7.2f} "
                f"{math.degrees(np.nanmean(a[:, 4])):+7.2f} deg")
        # the crouch baseline must come back to itself; if it does not, the
        # FK reference or the origin latch is wrong, not the filter
        base_rows = self.rows.get("WAIT_CROUCH")
        if base_rows is not None and abs(np.array(base_rows)[:, 2].mean()) > 0.005:
            lines.append("  -> WAIT_CROUCH drift should be ~0 by construction; "
                         "a non-zero value means the FK height or the origin "
                         "latch is wrong, NOT dead reckoning.")
        crouch = self.rows.get("WAIT_CROUCH")
        parked = self.rows.get("PARKED")
        if crouch is not None and parked is not None:
            d_fk = np.array(parked)[:, 0].mean() - np.array(crouch)[:, 0].mean()
            lines.append(f"  -> legs say PARKED is {d_fk*1e3:+.0f} mm from the "
                         f"crouch baseline (should be ~0: same pose)")
        if parked is not None and abs(np.array(parked)[:, 2].mean()) > 0.02:
            lines.append("  -> so the PARKED EKF z is dead-reckoning error from "
                         "the contacts-OFF ramps, frozen in when the footholds "
                         "re-anchored. See the README 'What EKF z means'.")
        return lines


class StandSequence:
    """CROUCH -> WAIT_CROUCH -> STAND -> HOLD4 -> (PARK -> PARKED -> STAND ...).

    Position commands only.  Pure function of
    (now, q, qd, enter_pressed, park_pressed) so the whole stage machine is
    exercisable in --self-test without a CAN bus.  `update` returns
    (cmd_deg, contacts, event) where event is a one-shot string or None, and
    sets `self.fault` on a timeout.

    PARK is the STAND ramp run backwards over the same duration, so lowering is
    exactly as gentle as rising and needs no separate profile.  It is accepted
    ONLY from HOLD4 -- parking mid-rise would reverse an unsettled pose.
    """

    MOVING_STAGES = ("STAND", "PARK")

    def __init__(self, q_stand, t_stand=T_STAND, contacts_during_ramp=True):
        self.q_stand = np.asarray(q_stand, dtype=float)
        self.stand_deg = np.rad2deg(self.q_stand)
        self.t_stand = float(t_stand)
        self.contacts_during_ramp = bool(contacts_during_ramp)
        self.stage = "CROUCH"
        self.stage_t0 = None
        self.fault = None
        self.n_stands = 0            # completed rises (HOLD4 entries)
        self._settle_since = None

    @property
    def contacts(self):
        """Contact schedule fed identically to the EKF.

        The feet never leave the floor in this script -- both ramps are a pure
        vertical rise/descent with each foot's x/y PINNED to its crouch value
        in the trunk frame, so the contact point is stationary in the world.
        Two schedules, and hardware has now decided between them:

        ON throughout (DEFAULT):
            physically honest for this motion.  The landmarks planted at init
            stay valid the whole way up, `handle_transitions` never sees a
            rising edge, so nothing is ever re-anchored and there is no moment
            at which accumulated error can be baked in.  Height stays observable
            end to end.  MEASURED 2026-08-11: drift essentially vanishes, versus
            +11 mm over the rise and tens of mm after a park with contacts off.

        OFF during the ramps (--dead-reckon-ramps, the old default, inherited
        from vmc/ekf_stand_hw.py):
            conservative.  Any real foot drag would be read as body motion and
            corrupt position AND attitude, so the filter is told to stop
            listening.  The cost is total loss of height observability for the
            5 s ramp -- z becomes open-loop double integration, and the error
            is then frozen in when the feet re-anchor at the next rising edge.
            Kept for A/B comparison and for replaying the older logs.

        !! DO NOT carry "contacts always ON" into the trot.  It is correct HERE
        only because the feet never leave the floor.  A gait that lifts feet
        must feed the real schedule, or the filter will be told a swinging foot
        is planted -- the exact failure this default protects against.
        """
        if self.contacts_during_ramp or self.stage not in self.MOVING_STAGES:
            return np.ones(4, bool)
        return np.zeros(4, bool)

    def update(self, now, q, qd, enter_pressed=False, park_pressed=False):
        if self.stage_t0 is None:
            self.stage_t0 = now
        event = None

        if self.stage == "CROUCH":
            cmd_deg = POSITION_TARGET_DEG
            settled = (float(np.max(np.abs(q - Q_CROUCH))) <= recorded.RECORDED_POSE_TOL
                       and float(np.max(np.abs(qd))) <= recorded.RECORDED_QD_TOL)
            if not settled:
                self._settle_since = None
                if now - self.stage_t0 > recorded.CROUCH_TIMEOUT_S:
                    self.fault = "CROUCH timeout"
            elif self._settle_since is None:
                self._settle_since = now
            elif now - self._settle_since >= recorded.CROUCH_SETTLE_S:
                self.stage, self.stage_t0 = "WAIT_CROUCH", now
                event = "crouch_settled"
        elif self.stage == "WAIT_CROUCH":
            cmd_deg = POSITION_TARGET_DEG
            if enter_pressed:
                self.stage, self.stage_t0 = "STAND", now
                event = "stand_started"
        elif self.stage == "STAND":
            s = _smoothstep((now - self.stage_t0) / self.t_stand)
            cmd_deg = np.rad2deg(Q_CROUCH + s * (self.q_stand - Q_CROUCH))
            if now - self.stage_t0 >= self.t_stand:
                self.stage, self.stage_t0 = "HOLD4", now
                self.n_stands += 1
                event = "stand_complete"
        elif self.stage == "HOLD4":
            cmd_deg = self.stand_deg
            if park_pressed:
                self.stage, self.stage_t0 = "PARK", now
                event = "park_started"
        elif self.stage == "PARK":
            s = _smoothstep((now - self.stage_t0) / self.t_stand)
            cmd_deg = np.rad2deg(self.q_stand + s * (Q_CROUCH - self.q_stand))
            if now - self.stage_t0 >= self.t_stand:
                self.stage, self.stage_t0 = "PARKED", now
                event = "park_complete"
        else:                                     # PARKED
            cmd_deg = POSITION_TARGET_DEG
            if enter_pressed:
                self.stage, self.stage_t0 = "STAND", now
                event = "stand_started"

        return np.asarray(cmd_deg, dtype=float), self.contacts, event


class AttitudeStats:
    """Scores the standing attitude -- the deliverable of this script.

    Two separate things, deliberately not mixed:
      * LEVEL   the resting roll/pitch itself (mean = the bias step 2 must
                use as its setpoint; std/peak-to-peak = the noise floor a
                leveling loop has to live above)
      * AGREE   |EKF - AHRS|, the cross-check that the EKF attitude is sane
    """

    def __init__(self):
        self.roll_e, self.pitch_e = [], []
        self.d_roll, self.d_pitch = [], []
        self._visit = None            # current HOLD4 visit: (rolls, pitches)
        self.visits = []              # per-visit (n, mean_roll, mean_pitch) deg

    def begin_visit(self):
        self._visit = ([], [])

    def end_visit(self):
        """Close a HOLD4 visit; return a one-line summary or None if empty."""
        if not self._visit or not self._visit[0]:
            self._visit = None
            return None
        r = np.degrees(self._visit[0])
        p = np.degrees(self._visit[1])
        self.visits.append((len(r), float(r.mean()), float(p.mean())))
        self._visit = None
        return (f"  stand #{len(self.visits)}: roll={r.mean():+6.2f} "
                f"pitch={p.mean():+6.2f} deg  ({len(r)} samples)")

    def add(self, roll_e, pitch_e, roll_a, pitch_a):
        self.roll_e.append(roll_e)
        self.pitch_e.append(pitch_e)
        if self._visit is not None:
            self._visit[0].append(roll_e)
            self._visit[1].append(pitch_e)
        if not (math.isnan(roll_a) or math.isnan(pitch_a)):
            self.d_roll.append(abs(roll_e - roll_a))
            self.d_pitch.append(abs(pitch_e - pitch_a))

    @property
    def n(self):
        return len(self.roll_e)

    def summary(self):
        if not self.n:
            return ["[attitude] no HOLD4 samples scored (stand never settled?)"]
        r = np.degrees(self.roll_e)
        p = np.degrees(self.pitch_e)
        lines = [
            f"[attitude] scored {self.n} HOLD4 samples across "
            f"{max(len(self.visits), 1)} stand(s) "
            f"(after {SETTLE_S:.1f}s settle each)",
            f"  LEVEL roll  mean={r.mean():+6.2f} std={r.std():5.2f} "
            f"p2p={r.max()-r.min():5.2f} deg",
            f"  LEVEL pitch mean={p.mean():+6.2f} std={p.std():5.2f} "
            f"p2p={p.max()-p.min():5.2f} deg",
        ]
        if len(self.visits) > 1:
            vr = np.array([v[1] for v in self.visits])
            vp = np.array([v[2] for v in self.visits])
            lines.append(f"  REPEAT across {len(self.visits)} stands: "
                         f"roll spread={vr.max()-vr.min():5.2f} "
                         f"pitch spread={vp.max()-vp.min():5.2f} deg")
            lines.extend(f"    stand #{i+1}: roll={v[1]:+6.2f} pitch={v[2]:+6.2f} deg"
                         for i, v in enumerate(self.visits))
        if self.d_roll:
            dr = np.degrees(self.d_roll)
            dp = np.degrees(self.d_pitch)
            lines.append(f"  AGREE |EKF-AHRS| roll  mean={dr.mean():4.2f} "
                         f"max={dr.max():4.2f} deg")
            lines.append(f"  AGREE |EKF-AHRS| pitch mean={dp.mean():4.2f} "
                         f"max={dp.max():4.2f} deg")
        else:
            lines.append("  AGREE  no AHRS attitude packets -- cross-check UNAVAILABLE")
        lines.append("  -> step 2 leveling setpoint = the LEVEL means above, "
                     "not zero (they carry the IMU mount tilt)")
        return lines


def run(port, stand_height, crouch_max_speed_dps, log_path=None, raw_log_path=None,
        contacts_during_ramp=True):
    base.validate_hardware_config()
    q_stand = compute_q_stand(stand_height)

    unwrap = [base.CalibratedEncoderUnwrap() for _ in MOTOR_IDS]
    key = base.KeyPoller()
    gate = base.SafetyGate(tau_cap=base.STAGED_TAU_MAX)   # position mode: limits only
    stats = AttitudeStats()
    report = StageReport()

    print("=" * 74)
    print("DOG5 POSITION STAND + READ-ONLY EKF  (step 1: verify the estimator)")
    print(f"  stand height = {stand_height:.3f} m ; crouch->stand over {T_STAND:.0f}s")
    print("  NO torque is commanded by us; the drivers hold position.")
    print(f"  ramp contacts = {'ON (never re-anchors)' if contacts_during_ramp else 'OFF (dead-reckons the ramps)'}")
    print("  Support the robot, feet on the ground.")
    print("  Keys: ENTER = stand (from WAIT_CROUCH or PARKED), "
          "P = park (from HOLD4), X = stop.")
    print("=" * 74)

    log_file = None
    writer = None
    if log_path:
        log_file = open(log_path, "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["t", "stage", "rz", "vx", "vy", "vz",
                         "roll_est", "pitch_est", "roll_ahrs", "pitch_ahrs",
                         "healthy", "z_fk", "z_drift"])
    enc_log = []          # (t_mono, alpha12, contacts4, roll_ahrs, pitch_ahrs)

    stop_reason = None
    worker = None
    shared = None
    log_t0 = None
    init_secs = 0.0
    try:
        # ImuEkfFeed is imported late so --self-test never needs a serial port.
        from imu_ekf_feed import ImuEkfFeed
        with motorbus.MotorBus(MOTOR_IDS, dirs=base.MOTOR_DIRECTIONS) as mb, \
                ImuEkfFeed(port) as feed:
            if not mb.arm(rate_hz=CONTROL_HZ):
                raise RuntimeError("arm failed (bus / power / terminators)")
            if not feed.wait_for_raw(timeout=3.0):
                raise RuntimeError("no raw IMU (0x40) packets -- enable DETA10 raw mode")

            start_q = base._zero_torque_preflight(mb, key, unwrap)
            if start_q is None:
                print("[abort] preflight not confirmed")
                return

            now = time.perf_counter()
            gate.start(now, start_q)
            shared = EkfShared(start_q)
            worker = threading.Thread(
                target=ekf_worker, args=(shared, feed),
                kwargs=dict(quiet_stages=QUIET_STAGES, control_hz=EKF_WORKER_HZ),
                daemon=True)
            worker.start()
            print(f"[ekf] worker started (read-only, {EKF_WORKER_HZ:.0f} Hz); "
                  f"init happens in {'/'.join(QUIET_STAGES)}")

            seq = StandSequence(q_stand,
                                contacts_during_ramp=contacts_during_ramp)
            miss_monitor = base.CanMissMonitor(mb)
            slot = mb.slot(CONTROL_HZ)
            deadline = time.perf_counter() + slot
            status_period = 1.0 / base.FAULT_STATUS_HZ
            next_fault_status = np.array(
                [status_period + i * status_period / N_JOINTS for i in range(N_JOINTS)])
            last_recover = {mid: -base.RECOVER_PERIOD_S for mid in MOTOR_IDS}
            start = now
            cmd_deg = POSITION_TARGET_DEG.copy()
            index = 0
            last_print = 0.0
            tx_fail = 0
            bias_printed = False
            quiet_t0 = None       # entered a holding stage (HOLD4 / PARKED) at

            while True:
                mb.poll()
                joint_index = index % N_JOINTS
                if joint_index == 0:                 # once per 250 Hz sweep
                    now = time.perf_counter()
                    q, qd = base._joint_state(mb, unwrap)

                    pressed = key.get()
                    if pressed in ("x", "X"):
                        stop_reason = "operator X"
                        break

                    cmd_deg, contacts, event = seq.update(
                        now, q, qd,
                        enter_pressed=pressed in ("\r", "\n"),
                        park_pressed=pressed in ("p", "P"))
                    if seq.fault:
                        stop_reason = seq.fault
                        break

                    shared.q = q
                    shared.qd = qd
                    shared.stage = seq.stage
                    shared.contacts = contacts

                    if event == "crouch_settled":
                        quiet_t0 = now       # the crouch baseline row
                        print("[stage] CROUCH settled; EKF initialising on the "
                              "static pose. ENTER to STAND.")
                    elif event == "stand_started":
                        quiet_t0 = None
                        print("[stage] STAND: streaming crouch -> stand.")
                        # the raw log's static prefix is set by the FIRST rise;
                        # later park/stand cycles must not overwrite it
                        if seq.n_stands == 0 and log_t0 is not None:
                            init_secs = (now - start) - log_t0
                        if seq.n_stands == 0 and log_t0 is not None \
                                and init_secs < MIN_INIT_QUIET_S:
                            print(f"[warn] only {init_secs:.2f}s of static prefix "
                                  f"logged (< {MIN_INIT_QUIET_S:.1f}s) -- hw_replay "
                                  f"init may be poor; hold longer in WAIT_CROUCH "
                                  f"next run")
                    elif event == "stand_complete":
                        quiet_t0 = now
                        stats.begin_visit()
                        print(f"[stage] HOLD4 (stand #{seq.n_stands}): static 4-leg "
                              f"stand. Attitude scored after {SETTLE_S:.1f}s. "
                              "Hand-rock to test the EKF. P parks, X stops.")
                    elif event == "park_started":
                        quiet_t0 = None
                        line = stats.end_visit()
                        if line:
                            print("[attitude] this stand scored:")
                            print(line)
                        print("[stage] PARK: streaming stand -> crouch.")
                    elif event == "park_complete":
                        quiet_t0 = now
                        print("[stage] PARKED: holding the crouch. FK height "
                              "should return to the origin; any EKF z left over "
                              "is drift. ENTER stands again, X stops.")

                    if not bias_printed and shared.est_ready:
                        print(f"[ekf] init: {shared.bias_str}")
                        bias_printed = True
                    # start both logs together, on a clean static prefix
                    if log_t0 is None and shared.est_ready and seq.stage == "WAIT_CROUCH":
                        shared.log_enabled = True
                        log_t0 = now - start

                    # ---- safety (position mode: limits / temp / comms / speed) ----
                    temps = base._temperatures(mb)
                    misses = miss_monitor.update(mb)
                    errors = mb.errors()
                    if stop_reason is None:
                        stop_reason = gate.estop_reason(q, qd, temps, misses, errors,
                                                        now, enforce_position_limits=True)
                    if stop_reason is None:
                        stop_reason = gate.overspeed_reason(qd, q, now)
                    if stop_reason:
                        break
                    latched = [mid for mid, e in errors.items() if e and (e & 0x80)]
                    recover = [mid for mid in latched
                               if (now - start) - last_recover[mid] >= base.RECOVER_PERIOD_S]
                    if recover:
                        base._recover_input_lost(mb, recover, now - start,
                                                 last_recover, next_fault_status)

                    # ---- observe / log ----
                    out = shared.out
                    r_a = p_a = float("nan")
                    att = feed.attitude()
                    if att is not None:
                        r_a, p_a = math.radians(att.roll_deg), math.radians(att.pitch_deg)
                    re_ = pe = z_fk = z_ekf = drift = float("nan")
                    if shared.est_ready and out is not None:
                        re_, pe = _rp(out["C"])
                        # drift-free FK height vs the EKF's integrated z
                        z_fk = fk_floor_height(q, out["C"])
                        z_ekf = float(out["r"][2])
                        report.set_origin(z_fk)     # first call wins (the crouch)
                        drift = z_ekf - (z_fk - (report.z0_fk or z_fk))
                        settled = (quiet_t0 is not None
                                   and now - quiet_t0 >= SETTLE_S)
                        if settled:
                            report.add(seq.stage, z_fk, z_ekf, re_, pe)
                            if seq.stage == "HOLD4":
                                stats.add(re_, pe, r_a, p_a)
                    if log_t0 is not None:
                        enc_log.append((time.monotonic(), *q,
                                        *contacts.astype(int), r_a, p_a))
                    if writer and shared.est_ready and out is not None:
                        writer.writerow([f"{now-start:.4f}", seq.stage,
                                         f"{out['r'][2]:.5f}",
                                         *[f"{x:.5f}" for x in out["v"]],
                                         f"{re_:.5f}", f"{pe:.5f}",
                                         f"{r_a:.5f}", f"{p_a:.5f}",
                                         int(out["healthy"]),
                                         f"{z_fk:.5f}", f"{drift:.5f}"])

                    if now - last_print >= 1.0 / base.STATUS_HZ:
                        last_print = now
                        extra = ""
                        if shared.est_ready and out is not None:
                            extra = (f" zEKF={z_ekf*1e3:+5.0f} zFK={z_fk*1e3:5.0f} "
                                     f"drift={drift*1e3:+5.0f}mm "
                                     f"roll={math.degrees(re_):+5.1f}"
                                     f"(ahrs{math.degrees(r_a):+5.1f}) "
                                     f"pitch={math.degrees(pe):+5.1f}"
                                     f"(ahrs{math.degrees(p_a):+5.1f}) "
                                     f"|v|={np.linalg.norm(out['v'])*1e3:4.0f}mm/s "
                                     f"H={'OK' if out['healthy'] else '!!'}")
                        print(f"[hw] {seq.stage:11s} max|qd|={np.max(np.abs(qd)):.2f} "
                              f"Tmax={int(np.max(temps))}C latched={len(latched)} "
                              f"txfail={tx_fail}{extra}", flush=True)

                # ---- per-slot POSITION command dispatch ----
                mid = MOTOR_IDS[joint_index]
                if (time.perf_counter() - start) >= next_fault_status[joint_index]:
                    mb.status1_req(mid)
                    next_fault_status[joint_index] += status_period
                elif not mb.position(mid, float(cmd_deg[joint_index]), crouch_max_speed_dps):
                    tx_fail += 1
                index += 1
                overrun = mb.pace(deadline)
                deadline += slot
                if overrun and overrun > 2.0 * slot:
                    deadline = time.perf_counter() + slot

            base._soft_stop(mb)
    except KeyboardInterrupt:
        stop_reason = stop_reason or "KeyboardInterrupt"
    finally:
        if shared is not None:
            shared.run = False
        if worker is not None:
            worker.join(timeout=1.0)
        try:
            key.close()
        except Exception:
            pass
        if log_file:
            log_file.close()
            print(f"[log] wrote {log_path}")
        if raw_log_path and shared is not None:
            imu = np.array(shared.imu_log, dtype=float)
            enc = np.array(enc_log, dtype=float)
            if len(imu) and len(enc):
                np.savez(raw_log_path,
                         imu_t=imu[:, 0], imu_f=imu[:, 1:4], imu_w=imu[:, 4:7],
                         enc_t=enc[:, 0], enc_alpha=enc[:, 1:13],
                         enc_contacts=enc[:, 13:17], ahrs_rp=enc[:, 17:19],
                         init_secs=max(init_secs, 0.5))
                print(f"[raw] wrote {raw_log_path} "
                      f"({len(imu)} IMU, {len(enc)} enc frames, "
                      f"init_secs={max(init_secs, 0.5):.2f})")
            else:
                print("[raw] nothing logged (EKF never initialised)")
    stats.end_visit()          # close a HOLD4 still open when we stopped
    for line in stats.summary():
        print(line)
    for line in report.summary():
        print(line)
    print(f"[stop] {stop_reason}")
# ===========================================================================
# offline self-test -> test_stand_ekf_verify.py
# ===========================================================================
# The gates moved out of this file.  They were 371 lines of simulation that no
# hardware run ever executes, and several of them shared a plane plant, a fake
# EKF and a PASS/FAIL harness with the other runners -- all of which now live
# in selftest_common.py.  `--self-test` still runs exactly those gates.
#
#     $V self-test/test_stand_ekf_verify.py     # the same thing, run directly


def self_test(stand_height):
    """Delegate to the suite in self-test/.  Lazy: a hardware run never loads it."""
    _sd = selftest_dir()
    if _sd not in sys.path:
        sys.path.insert(0, _sd)
    from test_stand_ekf_verify import self_test as gates
    return gates(stand_height)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--stand-height", type=float, default=recorded.DEFAULT_STAND_HEIGHT)
    ap.add_argument("--crouch-max-speed-dps", type=float,
                    default=recorded.DEFAULT_CROUCH_MAX_MOTOR_DPS)
    ap.add_argument("--log", default=None, help="CSV of EKF outputs")
    ap.add_argument("--raw-log", default=None,
                    help="NPZ raw IMU+encoder log for hw_replay (gate C8)")
    ap.add_argument("--dead-reckon-ramps", action="store_true",
                    help="drop contacts through STAND/PARK, so the EKF "
                         "dead-reckons them (the old default). Contacts are ON "
                         "throughout by default -- the feet never leave the "
                         "floor here, and hardware showed that removes the "
                         "height drift. Use this only to reproduce older runs.")
    ap.add_argument("--self-test", action="store_true",
                    help="run the offline gates and exit")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test(args.stand_height))
    run(args.port, args.stand_height, args.crouch_max_speed_dps,
        args.log, args.raw_log, not args.dead_reckon_ramps)


if __name__ == "__main__":
    main()
