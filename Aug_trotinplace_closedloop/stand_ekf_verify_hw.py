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
_PARENT = os.path.dirname(_HERE)
_DESC = os.path.join(_PARENT, "dog5_description")
_EST = os.path.join(_PARENT, "state_estimator")
_REPO = os.path.dirname(_PARENT)
_IMU = os.path.join(_REPO, "IMU_sensor")
for _p in (_HERE, _DESC, _EST, _REPO, _IMU):
    if _p not in sys.path:
        sys.path.insert(0, _p)


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

MOTOR_IDS = base.MOTOR_IDS
N_JOINTS = base.N_JOINTS
CONTROL_HZ = base.CONTROL_HZ                             # per motor AND per sweep
LEGS = base.LEGS
Q_CROUCH = recorded.Q_RECORDED_CROUCH
POSITION_TARGET_DEG = recorded.POSITION_TARGET_DEG       # crouch, controller-order deg

# Height reference points (tape-validated 2026-07-31, see vmc/stand_hier_hw.py).
#   * dog5_kinematics.foot_position() returns the foot SITE = the centre of a
#     20 mm contact sphere, so the floor is FOOT_RADIUS_M below it.
#   * The FK trunk origin is the HIP-AXIS plane (dog5.xml puts all four hip
#     bodies at z = 0).
#   * The EKF's r tracks the IMU.  dog5.xml models it AT the trunk origin
#     (`<site name="imu" pos="0 0 0">`, line 43) -- true in sim, WRONG on
#     hardware, where the board sits on the trunk bottom, measured
#     IMU_BELOW_TRUNK_ORIGIN_M lower (hip axis 8.5 in, trunk bottom 7.0 in).
# So FK-to-hip-axis and the EKF's r are NOT the same point on the real robot;
# fk_floor_height(ref="imu") converts between them.
FOOT_RADIUS_M = 0.020
IMU_BELOW_TRUNK_ORIGIN_M = 0.038

T_STAND = 5.0                 # crouch -> stand ramp (s)
EKF_WORKER_HZ = 100.0
QUIET_STAGES = ("WAIT_CROUCH",)   # EKF init happens only while truly still
SETTLE_S = 1.5                # ignore this long after entering a holding stage
                              # (WAIT_CROUCH / HOLD4 / PARKED) before scoring it
MIN_INIT_QUIET_S = 1.0        # warn if the raw log's static prefix is shorter


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


# --------------------------------------------------------------------------
# offline self-test -- no CAN bus, no IMU
# --------------------------------------------------------------------------

class _FakeFeed:
    """Duck-typed ImuEkfFeed serving level-and-still FLU samples."""

    def __init__(self):
        self.lock = threading.Lock()
        self.pending = []

    def push(self, n):
        t = time.monotonic()
        with self.lock:
            for k in range(n):
                self.pending.append((np.array([0.0, 0.0, 9.81]),
                                     np.zeros(3), t + 1e-3 * k))

    def drain(self):
        with self.lock:
            items, self.pending = self.pending, []
        return items

    def attitude(self):
        return None


_FAIL = []


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def _check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _FAIL.append(name)


def _test_q_stand(stand_height):
    q_stand = compute_q_stand(stand_height)
    lo, hi = base.soft_limits()
    _check("stand pose inside soft limits",
           bool(np.all(q_stand >= lo) and np.all(q_stand <= hi)))
    dxy = dz = 0.0
    for i, leg in enumerate(LEGS):
        sl = slice(3 * i, 3 * i + 3)
        f_c = dog5_kinematics.foot_position(leg, Q_CROUCH[sl])
        f_s = dog5_kinematics.foot_position(leg, q_stand[sl])
        dxy = max(dxy, float(np.linalg.norm(f_s[:2] - f_c[:2])))
        dz = max(dz, abs(f_s[2] + stand_height))
    _check("feet keep their crouch x/y", dxy < 1e-6, f"max {dxy*1e3:.4f} mm")
    _check("feet reach -stand_height", dz < 1e-6, f"max err {dz*1e3:.4f} mm")
    return q_stand


def _test_sequence(q_stand):
    """Drive the stage machine with an ideal position plant (stiff servo)."""
    # exercise the OFF schedule here: it is the one with stage-varying contacts
    seq = StandSequence(q_stand, contacts_during_ramp=False)
    t, dt = 0.0, 1.0 / CONTROL_HZ
    q = Q_CROUCH + 0.5                       # start well away from the crouch
    qd = np.zeros(N_JOINTS)
    seen, contacts_by_stage, z_trunk = [], {}, []
    entered_stand = False
    while t < recorded.CROUCH_TIMEOUT_S + T_STAND + 2.0 and seq.stage != "HOLD4":
        enter = seq.stage == "WAIT_CROUCH"   # operator presses ENTER immediately
        cmd, contacts, event = seq.update(t, q, qd, enter_pressed=enter)
        if seq.stage not in seen:
            seen.append(seq.stage)
        contacts_by_stage.setdefault(seq.stage, []).append(contacts.copy())
        q = np.deg2rad(cmd)                  # ideal tracking
        if seq.stage == "STAND":
            entered_stand = True
            z_trunk.append(-dog5_kinematics.foot_position(LEGS[0], q[0:3])[2])
        t += dt
    cmd, contacts, _ = seq.update(t, q, qd)
    contacts_by_stage.setdefault(seq.stage, []).append(contacts.copy())

    _check("stage order CROUCH->WAIT_CROUCH->STAND->HOLD4",
           seen == ["CROUCH", "WAIT_CROUCH", "STAND", "HOLD4"]
           and seq.stage == "HOLD4",
           " -> ".join(seen))
    _check("HOLD4 commands the stand pose",
           bool(np.allclose(cmd, np.rad2deg(q_stand))))
    _check("contacts OFF only during STAND",
           all((not np.any(np.array(v))) if s == "STAND" else bool(np.all(np.array(v)))
               for s, v in contacts_by_stage.items()),
           " ".join(f"{s}={'off' if not np.any(np.array(v)) else 'on'}"
                    for s, v in contacts_by_stage.items()))
    _check("trunk rises monotonically through STAND",
           entered_stand and bool(np.all(np.diff(z_trunk) >= -1e-9)),
           f"{z_trunk[0]*1e3:.0f} -> {z_trunk[-1]*1e3:.0f} mm")

    # a plant that never settles must trip the CROUCH timeout, not hang
    stuck = StandSequence(q_stand, contacts_during_ramp=False)
    t = 0.0
    while t < recorded.CROUCH_TIMEOUT_S + 1.0 and stuck.fault is None:
        stuck.update(t, Q_CROUCH + 5.0, np.zeros(N_JOINTS))
        t += dt
    _check("stuck crouch trips the timeout", stuck.fault == "CROUCH timeout",
           str(stuck.fault))
    return seq, q, t, dt


def _test_park(seq, q, t, dt, q_stand):
    """From the HOLD4 left by _test_sequence: park, then stand again."""
    qd = np.zeros(N_JOINTS)

    # P is accepted only from HOLD4 -- prove it is ignored mid-rise
    mid_rise = StandSequence(q_stand, contacts_during_ramp=False)
    mid_rise.stage, mid_rise.stage_t0 = "STAND", 0.0
    mid_rise.update(0.5 * T_STAND, q, qd, park_pressed=True)
    _check("P ignored outside HOLD4", mid_rise.stage == "STAND", mid_rise.stage)

    _, _, event = seq.update(t, q, qd, park_pressed=True)
    _check("P in HOLD4 starts PARK",
           seq.stage == "PARK" and event == "park_started", seq.stage)

    z_trunk, park_contacts = [], []
    t_park0 = t
    while t - t_park0 < T_STAND + 2 * dt and seq.stage == "PARK":
        cmd, contacts, _ = seq.update(t, q, qd)
        q = np.deg2rad(cmd)
        if seq.stage == "PARK":     # the tick that finishes the ramp is PARKED
            park_contacts.append(contacts.copy())
        z_trunk.append(-dog5_kinematics.foot_position(LEGS[0], q[0:3])[2])
        t += dt
    _check("PARK completes into PARKED", seq.stage == "PARKED", seq.stage)
    _check("trunk descends monotonically through PARK",
           bool(np.all(np.diff(z_trunk) <= 1e-9)),
           f"{z_trunk[0]*1e3:.0f} -> {z_trunk[-1]*1e3:.0f} mm")
    _check("contacts OFF through PARK", not np.any(np.array(park_contacts)))

    cmd, contacts, _ = seq.update(t, q, qd)
    _check("PARKED commands the crouch pose",
           bool(np.allclose(cmd, POSITION_TARGET_DEG)),
           f"max err {np.max(np.abs(cmd - POSITION_TARGET_DEG)):.4f} deg")
    _check("contacts ON in PARKED", bool(np.all(contacts)))

    # ENTER from PARKED must rise again, and the rise must be counted
    _, _, event = seq.update(t, q, qd, enter_pressed=True)
    _check("ENTER in PARKED stands again",
           seq.stage == "STAND" and event == "stand_started", seq.stage)
    t_re0 = t
    while t - t_re0 < T_STAND + 2 * dt and seq.stage == "STAND":
        cmd, _, _ = seq.update(t, q, qd)
        q = np.deg2rad(cmd)
        t += dt
    _check("re-stand reaches HOLD4 at the stand pose",
           seq.stage == "HOLD4" and np.allclose(cmd, np.rad2deg(q_stand), atol=1e-6),
           f"{seq.stage}, n_stands={seq.n_stands}")
    _check("both rises counted", seq.n_stands == 2, str(seq.n_stands))


def _test_worker_wiring(q_stand):
    """The EKF worker must init only in a quiet stage and report level."""
    shared = EkfShared(q_stand)
    feed = _FakeFeed()
    worker = threading.Thread(target=ekf_worker, args=(shared, feed),
                              kwargs=dict(quiet_stages=QUIET_STAGES,
                                          control_hz=200.0),
                              daemon=True)
    worker.start()
    try:
        shared.stage = "CROUCH"               # NOT quiet: must not initialise
        feed.push(60)
        time.sleep(0.15)
        _check("no EKF init outside the quiet stage", not shared.est_ready)

        shared.stage = "WAIT_CROUCH"
        feed.push(60)
        t_end = time.time() + 2.0
        while not shared.est_ready and time.time() < t_end:
            time.sleep(0.01)
        _check("EKF initialises in WAIT_CROUCH", shared.est_ready, shared.bias_str)
        _check("imu_log gated off before log_enabled", len(shared.imu_log) == 0)

        shared.log_enabled = True
        feed.push(40)
        t_end = time.time() + 2.0
        while shared.out is None and time.time() < t_end:
            time.sleep(0.01)
        ok = shared.out is not None and "C" in shared.out
        _check("worker publishes outputs", ok)
        if ok:
            r, p = _rp(shared.out["C"])
            _check("level-and-still reads level",
                   max(abs(r), abs(p)) < math.radians(1.0),
                   f"roll={math.degrees(r):+.2f} pitch={math.degrees(p):+.2f} deg")
            _check("EKF reports healthy", bool(shared.out["healthy"]))
        # the worker may publish outputs before it drains the new samples
        feed.push(40)
        t_end = time.time() + 2.0
        while not shared.imu_log and time.time() < t_end:
            time.sleep(0.01)
        _check("imu_log grows once enabled", len(shared.imu_log) > 0)
    finally:
        shared.run = False
        worker.join(timeout=1.0)


def _test_contact_schedules(q_stand):
    """The default schedule must keep contacts ON everywhere, and that must
    mean the estimator never sees a rising edge (nothing is re-anchored)."""
    dt = 1.0 / CONTROL_HZ
    _check("contacts stay ON through the ramps BY DEFAULT",
           StandSequence(q_stand).contacts_during_ramp is True)
    for during_ramp in (False, True):
        seq = StandSequence(q_stand, contacts_during_ramp=during_ramp)
        q, qd = Q_CROUCH.copy(), np.zeros(N_JOINTS)
        seen = []                                   # (stage, contacts)
        t = 0.0
        while t < recorded.CROUCH_TIMEOUT_S + 3 * T_STAND and seq.stage != "PARKED":
            enter = seq.stage in ("WAIT_CROUCH",)
            park = seq.stage == "HOLD4"
            cmd, contacts, _ = seq.update(t, q, qd, enter_pressed=enter,
                                          park_pressed=park)
            q = np.deg2rad(cmd)
            seen.append((seq.stage, contacts.copy()))
            t += dt
        ramp = [c for s, c in seen if s in StandSequence.MOVING_STAGES]
        assert ramp, "test never entered a ramp"
        label = "ON" if during_ramp else "OFF"
        _check(f"ramp contacts {label} as configured",
               bool(np.all(np.array(ramp))) == during_ramp)

        # a rising edge is what re-anchors a foothold; ON must produce none
        arr = np.array([c for _, c in seen])
        rising = int(np.sum(arr[1:] & ~arr[:-1]))
        _check(f"contacts {label}: {'no' if during_ramp else 'some'} re-anchoring",
               (rising == 0) == during_ramp, f"{rising} rising edges")


def _test_height(q_stand, stand_height):
    """FK height must be absolute and drift-free; drift must isolate the EKF."""
    h_hip = fk_floor_height(q_stand, ref="hip")
    _check("FK hip-axis height = commanded + foot radius",
           abs(h_hip - (stand_height + FOOT_RADIUS_M)) < 1e-9,
           f"{h_hip*1e3:.1f} mm vs {(stand_height+FOOT_RADIUS_M)*1e3:.1f} mm")

    # the default reference is the IMU -- the point the EKF's r actually tracks
    h_crouch = fk_floor_height(Q_CROUCH)
    h_stand = fk_floor_height(q_stand)
    _check("default reference is the IMU, 38 mm below the hip axis",
           abs((h_hip - h_stand) - IMU_BELOW_TRUNK_ORIGIN_M) < 1e-12,
           f"hip {h_hip*1e3:.0f} mm, IMU {h_stand*1e3:.0f} mm")
    _check("an unknown reference is rejected",
           _raises(lambda: fk_floor_height(q_stand, ref="trunk_bottom")))
    _check("FK crouch height is below the stand height",
           h_crouch < h_stand, f"{h_crouch*1e3:.0f} -> {h_stand*1e3:.0f} mm")

    # a level attitude must not change the answer; a tilt must
    eye = np.eye(3)
    _check("identity attitude matches the level shortcut",
           abs(fk_floor_height(q_stand, eye) - h_stand) < 1e-12)
    tilt = math.radians(10.0)
    C_tilt = np.array([[math.cos(tilt), 0.0, -math.sin(tilt)],
                       [0.0, 1.0, 0.0],
                       [math.sin(tilt), 0.0, math.cos(tilt)]])
    _check("a tilted trunk changes the FK height",
           abs(fk_floor_height(q_stand, C_tilt) - h_stand) > 1e-6,
           f"{fk_floor_height(q_stand, C_tilt)*1e3:.1f} mm at 10 deg pitch")
    # the IMU lever shortens the vertical offset as the trunk tilts
    lever = (fk_floor_height(q_stand, C_tilt, ref="hip")
             - fk_floor_height(q_stand, C_tilt))
    _check("IMU vertical offset shrinks with tilt (cos, not constant)",
           lever < IMU_BELOW_TRUNK_ORIGIN_M - 1e-9,
           f"{lever*1e3:.3f} mm at 10 deg vs {IMU_BELOW_TRUNK_ORIGIN_M*1e3:.1f} flat")

    # ...but the DRIFT number must not care: a constant cancels in the delta
    d_imu = ((h_stand - h_crouch))
    d_hip = (fk_floor_height(q_stand, ref="hip")
             - fk_floor_height(Q_CROUCH, ref="hip"))
    _check("reference choice does not move the measured rise",
           abs(d_imu - d_hip) < 1e-12,
           f"IMU {d_imu*1e3:.3f} mm vs hip {d_hip*1e3:.3f} mm")

    # drift bookkeeping: a PERFECT EKF rising by the true amount drifts 0
    ht = StageReport()
    _check("no origin -> add() is a no-op",
           ht.add("WAIT_CROUCH", h_crouch, 0.0) is None)
    ht.set_origin(h_crouch)
    ht.set_origin(999.0)                       # later origins must be ignored
    _check("height origin is latched at init", ht.z0_fk == h_crouch)

    # the crouch baseline row: same pose the origin was latched at -> drift ~0
    d = ht.add("WAIT_CROUCH", h_crouch, 0.0, math.radians(0.3), math.radians(-0.2))
    _check("crouch baseline drifts zero by construction", abs(d) < 1e-12,
           f"{d*1e3:.3f} mm")
    d = ht.add("HOLD4", h_stand, h_stand - h_crouch,
               math.radians(0.3), math.radians(-0.2))
    _check("a perfect EKF shows zero drift", abs(d) < 1e-12, f"{d*1e3:.3f} mm")
    # the reported failure: parked back at the crouch, EKF still says -133 mm
    d = ht.add("PARKED", h_crouch, -0.133, math.radians(0.3), math.radians(-0.2))
    _check("parked-with-stale-z reproduces the observed drift",
           abs(d + 0.133) < 1e-12, f"{d*1e3:.0f} mm")

    txt = "\n".join(ht.summary())
    _check("summary reports every holding stage",
           all(s in txt for s in ("WAIT_CROUCH", "HOLD4", "PARKED")))
    _check("stage rows are chronological, crouch first",
           txt.index("WAIT_CROUCH ") < txt.index("HOLD4") < txt.index("PARKED"))
    _check("summary carries attitude per stage", "+0.30" in txt and "-0.20" in txt)
    _check("summary confirms the legs returned to the crouch",
           "+0 mm from the crouch baseline" in txt,
           next((l.strip() for l in txt.splitlines() if "legs say" in l), "missing"))
    _check("summary explains a large PARKED drift", "dead-reckoning" in txt)
    _check("summary states what EKF z=0 means", "the crouch" in txt)

    # a bad FK reference / origin latch must be called out, not blamed on drift
    bad = StageReport()
    bad.set_origin(h_crouch)
    bad.add("WAIT_CROUCH", h_crouch + 0.030, 0.0)
    _check("a non-zero crouch baseline is flagged as a reference error",
           "NOT dead reckoning" in "\n".join(bad.summary()))


def _test_stats():
    st = AttitudeStats()
    _check("empty stats do not crash", len(st.summary()) == 1)
    st.begin_visit()
    for _ in range(10):
        st.add(math.radians(1.0), math.radians(-2.0),
               math.radians(1.5), math.radians(-2.5))
    line = st.end_visit()
    _check("a HOLD4 visit closes with its own line",
           line is not None and "+1.00" in line, (line or "").strip())
    txt = "\n".join(st.summary())
    _check("stats report the resting means",
           "+1.00" in txt and "-2.00" in txt, txt.splitlines()[1].strip())
    _check("stats report the AHRS disagreement", "0.50" in txt)

    # a second stand at a different attitude must show up as REPEAT spread
    st.begin_visit()
    for _ in range(10):
        st.add(math.radians(1.4), math.radians(-2.0), float("nan"), float("nan"))
    st.end_visit()
    txt = "\n".join(st.summary())
    _check("park/stand cycles report repeatability spread",
           "REPEAT" in txt and "0.40" in txt,
           next((l.strip() for l in txt.splitlines() if "REPEAT" in l), "missing"))


def self_test(stand_height):
    print("stand_ekf_verify_hw self-test (no hardware)")
    print("[1] stand pose")
    q_stand = _test_q_stand(stand_height)
    print("[2] stage machine")
    seq, q, t, dt = _test_sequence(q_stand)
    print("[3] park / re-stand")
    _test_park(seq, q, t, dt, q_stand)
    print("[4] EKF worker wiring")
    _test_worker_wiring(q_stand)
    print("[5] contact schedules")
    _test_contact_schedules(q_stand)
    print("[6] FK height vs EKF z")
    _test_height(q_stand, stand_height)
    print("[7] attitude statistics")
    _test_stats()
    print("self-test " + ("FAIL: " + ", ".join(_FAIL) if _FAIL else "PASS"))
    return 1 if _FAIL else 0


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
