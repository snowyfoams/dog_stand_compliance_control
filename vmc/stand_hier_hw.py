#!/usr/bin/env python3
"""Stage A hierarchical stand: EKF-informed POSITION-mode rise + leveling.

Architecture (agreed 2026-07-30 after the torque-mode VMC stand failed on
hardware -- software torque at the 20.8 Hz per-joint CAN rate cannot
stabilise the legs; native position mode is proven on this rig):

    high level  (Pi, per 250 Hz sweep)
        MotorBus.slot(rate_hz) = 1/(rate_hz * n_motors), so at CONTROL_HZ
        250 with 12 motors a slot is 333 us and a full sweep is 4 ms --
        i.e. EVERY motor is commanded at 250 Hz and this block (the
        joint_index == 0 branch) also runs at 250 Hz.
        EKF attitude (IMU-driven, 100 Hz worker)  ->  per-foot Cartesian
        targets: x/y pinned to the crouch FK anchors, z ramps crouch ->
        -stand_height, plus per-foot leveling z-offsets from the EKF
        roll/pitch (the hardware-proven inplace posture-trim law)
    mid level   (Pi, one leg per 3rd CAN slot)
        warm-started damped-least-squares IK (stand3's proven spread --
        4 legs of IK in one block costs 5-7 ms, far past the 333 us slot
        and past the motor input-timeout window; spread one leg per 3rd
        slot it never blocks long enough to starve a motor)
    low level   (motor drivers, kHz)
        0xA4 native position loops, motor-side speed caps

NO software torque is commanded anywhere.  The EKF is advisory: it feeds
the leveling trim and the displays; every motion gate is encoder-based.

There is NO speed trip (removed 2026-07-31 -- it kept aborting good stands).
Joint speed is bounded by the motor-side max_dps in the 0xA4 frame, and the
commanded path is proven inside the soft position limits offline before any
motor moves.  See the SPEED_WATCH_ONLY note for what still aborts a run.

Stages:
    ZERO-TORQUE preflight -> CROUCH (native 0xA4 to the recorded pose)
      -> WAIT_CROUCH (EKF initialises on the static hold; ENTER)
      -> RISE (z smoothstep over --t-rise; feet never slide)
      -> RISE_SETTLE -> HOLD (leveling active; P parks, X stops)
      -> PARK (reverse ramp, trim decays) -> PARK_SETTLE -> PARKED

Run:
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V stand_hier_hw.py --self-test
    $V stand_hier_hw.py --level-gain 0 --no-ekf      # bisect input-lost:
                                                    # nothing but the CAN loop
    $V stand_hier_hw.py --level-gain 0 --log run1.csv --raw-log run1.npz
    $V stand_hier_hw.py                      # leveling on (0.25/s, +/-12 mm)

Watch `blk=` (worst ms spent in the joint_index == 0 block) and `txfail=` in
the status line: a motor input-lost latch needs a stall far longer than the
333 us slot, so `blk` is the thing that finds it.  `qd=<now>/pk<peak>(trip N)`
shows how much headroom the encoder-speed trip actually has.

The robot must be supported/tethered until the runbook's sign checks pass.
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

# Let the GIL hand off often so the CAN thread keeps pacing while the EKF
# worker is mid-compute (ekf_stand_hw / walk1 proven setting).
sys.setswitchinterval(0.0005)

_HERE = os.path.dirname(os.path.abspath(__file__))                # vmc/
_DESC = os.path.join(os.path.dirname(_HERE), "dog5_description")
_EST = os.path.join(os.path.dirname(_HERE), "state_estimator")
_CRAWL = os.path.join(os.path.dirname(_HERE), "crawl_hw2.0")
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _DESC, _EST, _CRAWL, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stand_dog5_hw as base                             # noqa: E402
import stand_dog5_recorded_hw as recorded                # noqa: E402
import stand_by_position_command as posbase              # noqa: E402
import stand_dog5_inplace_hw as inplace                  # noqa: E402
import dog5_kinematics                                   # noqa: E402
from stand3_hold_hw import (_ik_to_target, _smoothstep,  # noqa: E402
                            IK_SLOT_STRIDE, IK_SLOT_MAX_ITER, IK_SLOT_TOL_M,
                            LEAN_ABORT_DEG, LEAN_STREAK)
from crawl_dog5_hw import foot_load_map, leg_gravity_stack   # noqa: E402
from ekf_runtime import EkfShared, ekf_worker, _rp       # noqa: E402

LEGS = base.LEGS
MOTOR_IDS = base.MOTOR_IDS
MOTOR_DIRECTIONS = base.MOTOR_DIRECTIONS
JOINT_LABELS = base.JOINT_LABELS
N_JOINTS = base.N_JOINTS
Q_RECORDED_CROUCH = recorded.Q_RECORDED_CROUCH   # the EKF-validated crouch
                                                 # (NOT posbase.Q_CROUCH -- a
                                                 # different recording)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# dog5.xml gives every foot a CONTACT SPHERE of radius 0.02 m centred on the
# foot site, and dog5_kinematics.foot_position() returns that SITE -- i.e. the
# sphere centre, not the contact point.  Ground contact is 20 mm below it, so
# the trunk origin stands FOOT_RADIUS_M higher above the floor than every
# "trunk height" this codebase prints.  (Harmless inside the EKF -- a constant
# offset is absorbed when the landmark is planted -- but it makes every height
# comparison against a tape measure wrong by 20 mm.)
FOOT_RADIUS_M = 0.020

# The FK trunk origin is the HIP AXIS PLANE: dog5.xml puts all four hip bodies
# at z = 0 and the IMU site at (0, 0, 0).  Measure to the hip pivot centre, not
# to the chassis top or bottom, when comparing against a tape measure.
#
# VALIDATED 2026-07-31 on hardware at the 0.200 m stand pose:
#   predicted ground -> hip axis  220 mm  (200 foot site + 20 contact sphere)
#   measured  ground -> hip axis  216 mm  (8.5 in)
# 4 mm agreement -- the leg kinematics and the encoder zeros are sound.
#
# The same session measured ground -> trunk bottom / IMU at 178 mm (7 in), so
# the IMU sits ~38 mm BELOW the trunk origin.  dog5.xml places the IMU site at
# (0, 0, 0), and the estimator's propagate_state has no lever-arm term, so:
#   * the EKF's `r` really tracks the IMU, not the trunk origin;
#   * leg_fk() returns feet in the TRUNK frame, so the measurement model
#     s_pred = C (p_i - r) is short by this lever arm.
# The constant part is absorbed when the landmark is planted, leaving only an
# attitude-coupled residual: ~38 mm * sin(tilt), i.e. ~3 mm at 5 deg.  Small,
# but it is a real unmodelled term -- do not chase a few mm of EKF/FK
# disagreement without accounting for it.
IMU_BELOW_TRUNK_ORIGIN_M = 0.038

T_PARK = 8.0                    # reverse ramp duration (s)
SETTLE_TIMEOUT_S = 30.0
LOAD_STATUS_HZ = 4.0            # foot_load_map costs 1.6 ms -- display path ONLY
DEFAULT_LOG_HZ = 20.0           # CSV rows/s.  The sweep is 250 Hz: writing a
                                # row per sweep is 250 rows/s of file I/O in
                                # the CAN loop, and one SD-card flush stall is
                                # enough to starve every motor's input timeout.

# NO SPEED TRIP.  Removed 2026-07-31 at the operator's direction after it kept
# aborting good stands.  Speed is still BOUNDED, just not tripped on:
#
#   * the 0xA4 frame carries a motor-side max_dps the driver enforces in
#     hardware (100 dps during RISE/PARK) -- that cap is untouched and is the
#     real speed limit;
#   * validate_configuration proves the whole commanded path, including the
#     full leveling envelope, stays inside the soft position limits before any
#     motor moves.
#
# What was removed, and why neither was salvageable:
#   * posbase.position_fault's encoder-speed branch -- a host-side finite
#     difference over one 4 ms sweep, then EWMA'd by EncoderVelocity at
#     alpha=0.35.  The filter SMEARS a single raw spike across several sweeps
#     (0.35, 0.2275, 0.148, ... of the spike), so one spike above ~6.8 rad/s
#     clears a 3-sweep streak by itself -- debouncing it does not work.
#   * SafetyGate's driver-side overspeed tier -- neutralised by passing inf
#     thresholds, NOT by editing base (twostand/stand3/walk1 share it).
#
# Still protecting the rig: measured-torque trip, overtemp, CAN miss estop,
# motor hard-fault flags, IMU tip-over + rise lean abort, IK residual watch,
# target-rate clamp, and the motor-side speed cap above.
SPEED_WATCH_ONLY = True

# Leveling trim: port of the hardware-proven inplace.update_posture_trim law
# (Phase-2 leveling stand PASSED), attitude sourced from the EKF instead of
# the AHRS.  All laws are dt-based so they are correct at any sweep rate.
LEVEL_GAIN_PER_S = 0.25         # = inplace.POSTURE_KI_PER_S (closed-loop ~4 s)
LEVEL_CLAMP_M = 0.012           # = inplace.POSTURE_CLAMP_M (+/-2.0 deg pitch,
                                #   +/-6.1 deg roll of authority)
LEVEL_SLEW_M_S = 0.004          # caps response to an EKF attitude step
LEVEL_DEADBAND_DEG = 0.2        # = inplace.POSTURE_DEADBAND_DEG
LEVEL_LPF_FC_HZ = 10.0          # = inplace.POSTURE_LPF_FC_HZ
LEVEL_MAX_TILT_DEG = 10.0       # beyond this leveling is the wrong tool
LEVEL_STALE_S = 0.2             # EKF worker stall -> freeze
LEVEL_XCHECK_DEG = 5.0          # |EKF - AHRS| beyond this -> freeze

# Stream guards (all dt-based)
MAX_TARGET_RATE_M_S = 0.050     # 1.5x the 33 mm/s nominal rise peak
RATE_CLAMP_STUCK_S = 1.0
IK_RESID_TRIP_M = 0.008
IK_RESID_TRIP_S = 0.5


# ---------------------------------------------------------------------------
# Position-mode fault check (torque only -- no speed trip)
# ---------------------------------------------------------------------------
class SpeedWatch:
    """Observes encoder speed for the status line.  NEVER trips.

    Kept purely so the operator can still see what the joints are doing --
    `qd=<now>/pk<peak>` in the status line.  All abort authority was
    removed; see the SPEED_WATCH_ONLY note above for what still protects
    the rig.  `check()` forwards to posbase.position_fault with the speed
    branch disabled (np.inf), so the measured-torque and
    non-finite-feedback faults keep their original immediacy.
    """

    def __init__(self):
        self.peak = 0.0          # worst |qd| since the last peek

    def peek_peak(self, reset=True):
        value = self.peak
        if reset:
            self.peak = 0.0
        return value

    def check(self, qd_encoder, measured_torque, velocity_ready, torque_trip):
        if velocity_ready:
            self.peak = max(self.peak, float(
                np.max(np.abs(np.asarray(qd_encoder, dtype=float)))))
        return posbase.position_fault(qd_encoder, measured_torque,
                                      velocity_ready, np.inf, torque_trip)


# ---------------------------------------------------------------------------
# Leveling trim
# ---------------------------------------------------------------------------
class LevelingTrim:
    """Integrate EKF body tilt into per-foot z offsets (trunk frame, m).

    Law and constants are the proven inplace.update_posture_trim:
        e_i = -x_i * pitch + y_i * roll     (zero-meaned across feet)
        off_i += clip(gain * e_i * dt, +/-slew*dt);  |off_i| <= clamp
    Nose down (+pitch, _rp convention) -> front feet more negative z ->
    front legs extend -> nose rises.  Right side down (+roll) -> right
    legs extend.  Zero-mean so leveling never changes average height.

    `update(..., active=False)` or roll/pitch None freezes (holds, never
    decays); `decay()` ramps the offsets to zero at the slew rate (PARK).
    Pure logic with injected time -- offline-testable and sim-reusable.
    """

    def __init__(self, anchors_xy, gain_per_s=LEVEL_GAIN_PER_S,
                 clamp_m=LEVEL_CLAMP_M, slew_m_s=LEVEL_SLEW_M_S,
                 deadband_deg=LEVEL_DEADBAND_DEG, lpf_fc_hz=LEVEL_LPF_FC_HZ):
        self.anchors = {leg: np.asarray(anchors_xy[leg][:2], dtype=float)
                        for leg in LEGS}
        self.gain = float(gain_per_s)
        self.clamp = float(clamp_m)
        self.slew = float(slew_m_s)
        self.deadband_rad = math.radians(float(deadband_deg))
        self.lpf_fc_hz = float(lpf_fc_hz)
        self.offsets = {leg: 0.0 for leg in LEGS}
        self.frozen_reason = "init"
        self.last_rp_deg = (float("nan"), float("nan"))
        self._lpf = None
        self._last = None

    def reset(self):
        for leg in LEGS:
            self.offsets[leg] = 0.0
        self._lpf = None
        self._last = None
        self.frozen_reason = "reset"

    @property
    def saturated(self):
        return any(abs(v) >= self.clamp - 1e-9 for v in self.offsets.values())

    def _dt(self, now):
        if self._last is None:
            self._last = float(now)
            return None
        dt = float(np.clip(now - self._last, 0.0, 0.1))
        self._last = float(now)
        return dt

    def update(self, now, roll_rad, pitch_rad, active, reason=""):
        if not active or roll_rad is None or pitch_rad is None:
            self._last = float(now)
            self.frozen_reason = reason or "inactive"
            return
        dt = self._dt(now)
        if dt is None:
            self.frozen_reason = ""
            return
        self.frozen_reason = ""
        # LPF the tilt (inplace: measured vibration ~0.03 deg; belt+braces)
        if self._lpf is None:
            self._lpf = [float(roll_rad), float(pitch_rad)]
        else:
            tau_f = 1.0 / (2.0 * math.pi * self.lpf_fc_hz)
            alpha = dt / (tau_f + dt)
            self._lpf[0] += alpha * (roll_rad - self._lpf[0])
            self._lpf[1] += alpha * (pitch_rad - self._lpf[1])
        roll_f, pitch_f = self._lpf
        self.last_rp_deg = (math.degrees(roll_f), math.degrees(pitch_f))
        phi = roll_f if abs(roll_f) >= self.deadband_rad else 0.0
        theta = pitch_f if abs(pitch_f) >= self.deadband_rad else 0.0
        if phi == 0.0 and theta == 0.0:
            return
        # Geometric leveling error per foot (trunk frame, z up in FLU,
        # foot targets have z negative: extend = more negative).
        e = {leg: -self.anchors[leg][0] * theta + self.anchors[leg][1] * phi
             for leg in LEGS}
        mean_e = sum(e.values()) / len(LEGS)
        max_step = self.slew * dt
        for leg in LEGS:
            step = self.gain * (e[leg] - mean_e) * dt
            step = float(np.clip(step, -max_step, max_step))
            self.offsets[leg] = float(
                np.clip(self.offsets[leg] + step, -self.clamp, self.clamp))

    def decay(self, now):
        """PARK: ramp the offsets to zero at the slew rate (never a jump)."""
        dt = self._dt(now)
        self.frozen_reason = "decay"
        if dt is None:
            return
        max_step = self.slew * dt
        for leg in LEGS:
            off = self.offsets[leg]
            self.offsets[leg] = off - float(np.clip(off, -max_step, max_step))


# ---------------------------------------------------------------------------
# Foot-target planner
# ---------------------------------------------------------------------------
class StandHierPlan:
    """Crouch-anchored vertical-rise foot targets (stand-in-place: the feet
    NEVER slide -- x/y pinned to the crouch FK, only z moves)."""

    def __init__(self, stand_height):
        self.stand_height = float(stand_height)
        self.crouch_foot = {
            leg: dog5_kinematics.foot_position(leg, Q_RECORDED_CROUCH[self._sl(leg)])
            for leg in LEGS
        }
        self.anchors_xy = {leg: self.crouch_foot[leg][:2].copy() for leg in LEGS}

    @staticmethod
    def _sl(leg):
        i = LEGS.index(leg)
        return slice(3 * i, 3 * i + 3)

    def foot_targets(self, height_frac, level_off):
        targets = {}
        for leg in LEGS:
            crouch_z = self.crouch_foot[leg][2]
            z = (crouch_z
                 + height_frac * (-self.stand_height - crouch_z)
                 + level_off[leg])
            targets[leg] = np.array(
                [self.anchors_xy[leg][0], self.anchors_xy[leg][1], z])
        return targets

    def q_targets(self, warm_q, height_frac, level_off):
        """Full-convergence IK -- OFFLINE validation only, never in the loop."""
        targets = self.foot_targets(height_frac, level_off)
        q = np.asarray(warm_q, dtype=float).copy()
        for leg in LEGS:
            sl = self._sl(leg)
            q[sl] = _ik_to_target(leg, q[sl], targets[leg])
        return q


# ---------------------------------------------------------------------------
# Stage machine
# ---------------------------------------------------------------------------
STAGES = [
    ("CROUCH", "move"),
    ("WAIT_CROUCH", "wait"),
    ("RISE", "stream"),
    ("RISE_SETTLE", "gate"),
    ("HOLD", "wait"),
    ("PARK", "stream"),
    ("PARK_SETTLE", "gate"),
    ("PARKED", "wait"),
]
STAGE_INDEX = {name: i for i, (name, _) in enumerate(STAGES)}

INSTRUCTION = {
    "CROUCH": "moving to recorded crouch; wait, X stops",
    "WAIT_CROUCH": "inspect crouch; wait for [ekf] init; ENTER rises, X stops",
    "RISE": "rising vertically in place (feet stay put); wait, X stops",
    "RISE_SETTLE": "waiting for the rise to settle",
    "HOLD": "standing; leveling active; P parks, X stops",
    "PARK": "reversing to crouch; wait, X stops",
    "PARK_SETTLE": "waiting for the park to settle",
    "PARKED": "back at crouch; X stops",
}

# EKF contact schedule.
#
# This was inherited from ekf_stand_hw (contacts OFF during its STAND ramp)
# and it is WRONG for this runner.  ekf_stand_hw's ramp could drag its feet;
# StandHierPlan cannot -- x/y are pinned to the crouch anchors and only z
# moves, so the feet are stationary in the INERTIAL frame for the whole rise
# while the trunk climbs over them.  That is precisely the case the foothold
# landmark model is built for: s_meas = leg_fk(alpha) tracks the trunk's climb
# through the encoders, s_pred = C (p_i - r) tracks it through r, and the
# innovation drives r to the encoder-measured height.
#
# Declaring the feet airborne throws that measurement away and leaves r as
# pure accelerometer double-integration for the full 8 s stream.  Only ~10
# mm/s of residual velocity error (or 0.0025 m/s^2 of accel bias) is needed to
# reach 80 mm over 8 s, and its sign varies run to run -- which is exactly the
# +/-70-80 mm sign-flipping close-out error measured on hardware (2 runs,
# 2026-07-31).  Height is unobservable in that window, so nothing corrects it,
# and handle_transitions then replants the landmarks onto the drifted r at
# touchdown, making the error permanent.
#
# Default is therefore all-planted throughout.  `--rise-contacts airborne`
# restores the old schedule for A/B comparison.
_STREAM_STAGES = ("RISE", "PARK")


class StandHierSequence:
    """Encoder-only stage machine emitting joint position targets per sweep.

    Pure logic (no CAN, no time sources of its own) so the offline
    self-test and the MuJoCo sim gate can drive it through every path.
    """

    def __init__(self, now, plan, trim, t_rise=8.0, t_park=T_PARK,
                 stream_feet_planted=True):
        self.plan = plan
        self.trim = trim
        self.stream_feet_planted = bool(stream_feet_planted)
        self.stream_t = {"RISE": float(t_rise), "PARK": float(t_park)}
        self.stage_i = 0
        self.stage_started = float(now)
        self.settle_since = None
        self.wait_since = None
        self.q_cmd = Q_RECORDED_CROUCH.copy()
        self.height_frac = 0.0
        self._stream_from = 0.0
        self._stream_to = 0.0
        self.targets = None
        self._prev_targets = None
        self._prev_target_time = None
        self._rate_clamp_since = None
        self.rate_clamped = False
        self.resid = {leg: 0.0 for leg in LEGS}
        self._resid_since = None
        self.fault = None            # latched guard trip -> runner stop_reason
        self.aborted = None          # lean abort note (stand continues frozen)
        self.events = []
        self.last_settle_detail = ""

    # -- helpers -----------------------------------------------------------
    @property
    def stage(self):
        return STAGES[self.stage_i][0]

    @property
    def kind(self):
        return STAGES[self.stage_i][1]

    @property
    def contacts(self):
        # The feet never leave the ground on this trajectory (x/y pinned), so
        # all-planted is the physically true schedule -- see _STREAM_STAGES.
        if not self.stream_feet_planted and self.stage in _STREAM_STAGES:
            return np.zeros(4, dtype=bool)
        return np.ones(4, dtype=bool)

    def _goto(self, name, now, note=""):
        self.stage_i = STAGE_INDEX[name]
        self.stage_started = float(now)
        self.settle_since = None
        self.wait_since = None
        if self.kind == "stream":
            self._stream_from = float(self.height_frac)
            self._stream_to = 1.0 if name == "RISE" else 0.0
        self.events.append((float(now), name, note))
        return f"{name}: {note}" if note else name

    def _settled(self, now, q_enc, qd, velocity_ready):
        errors = np.abs(np.asarray(q_enc) - self.q_cmd)
        worst = int(np.argmax(errors))
        error = float(errors[worst])
        speed = float(np.max(np.abs(qd))) if velocity_ready else np.inf
        self.last_settle_detail = (
            f"worst {JOINT_LABELS[worst]} err "
            f"{np.rad2deg(error):.1f} deg (tol "
            f"{np.rad2deg(posbase.POSITION_POSE_TOL):.1f}), max|qd| "
            f"{speed:.2f} rad/s (tol {posbase.POSITION_QD_TOL:.2f})"
        )
        if error > posbase.POSITION_POSE_TOL or speed > posbase.POSITION_QD_TOL:
            self.settle_since = None
            return False
        if self.settle_since is None:
            self.settle_since = float(now)
            return False
        return now - self.settle_since >= posbase.POSITION_SETTLE_S

    # -- operator ----------------------------------------------------------
    def request_next(self, now, healthy, ekf_ok=True):
        if self.stage == "HOLD":
            return False, "HOLD: P parks, X stops (ENTER does nothing)."
        if self.stage == "PARKED":
            return False, "already PARKED; X stops."
        if self.kind != "wait":
            return False, f"{self.stage} is busy; ENTER ignored."
        if self.wait_since is None or now - self.wait_since < base.WAIT_DWELL_S:
            return False, f"wait for {self.stage} to settle before ENTER."
        if not healthy:
            return False, "motor latch/fault present; motion refused."
        if not ekf_ok:
            return False, ("EKF not initialised; wait for '[ekf] init' or "
                           "run with --level-gain 0.")
        return True, self._goto("RISE", now, "ENTER accepted")

    def request_park(self, now, healthy):
        if self.stage != "HOLD":
            return False, f"P only parks from HOLD (now {self.stage})."
        if not healthy:
            return False, "motor latch/fault present; motion refused."
        return True, self._goto("PARK", now, "P accepted")

    def abort_freeze(self, now, reason):
        """Lean abort during the rise: freeze the stream where it is and
        drop to HOLD.  Position mode is statically stable at any height on
        this path, so freezing is safer than an automatic reversal; the
        operator decides P or X.  Leveling stays frozen (runner gates on
        self.aborted)."""
        self.aborted = reason
        if self.stage in ("RISE", "RISE_SETTLE"):
            return self._goto("HOLD", now, f"ABORT (frozen): {reason}")
        return None

    # -- per-sweep update --------------------------------------------------
    def sweep(self, now, q_enc, qd, velocity_ready):
        """One per-sweep update.  Returns event string or None."""
        event = None
        name, kind = STAGES[self.stage_i]

        if kind == "move":                                   # CROUCH
            self.q_cmd = Q_RECORDED_CROUCH.copy()
            self.targets = None
            self._prev_targets = None
            if self._settled(now, q_enc, qd, velocity_ready):
                event = self._goto("WAIT_CROUCH", now, "settled")
                self.wait_since = float(now)
            elif now - self.stage_started > recorded.CROUCH_TIMEOUT_S:
                raise RuntimeError(
                    f"CROUCH did not settle in "
                    f"{recorded.CROUCH_TIMEOUT_S:.0f} s: "
                    f"{self.last_settle_detail}")

        elif kind == "stream":                               # RISE / PARK
            u = _smoothstep((now - self.stage_started) / self.stream_t[name])
            self.height_frac = self._stream_from + u * (
                self._stream_to - self._stream_from)
            if now - self.stage_started >= self.stream_t[name]:
                self.height_frac = float(self._stream_to)
                nxt = {"RISE": "RISE_SETTLE", "PARK": "PARK_SETTLE"}[name]
                event = self._goto(nxt, now, "stream complete")

        elif kind == "gate":                                 # *_SETTLE
            if self._settled(now, q_enc, qd, velocity_ready):
                if name == "RISE_SETTLE":
                    event = self._goto("HOLD", now, "stand settled")
                else:
                    event = self._goto("PARKED", now, "park settled")
                    # bumpless snap: IK at height 0 differs from the recorded
                    # crouch only by the IK tolerance (<= 0.1 mm)
                    self.q_cmd = Q_RECORDED_CROUCH.copy()
                    self.targets = None
                    self._prev_targets = None
                self.wait_since = float(now)
            elif now - self.stage_started > SETTLE_TIMEOUT_S:
                raise RuntimeError(
                    f"{name} did not settle in {SETTLE_TIMEOUT_S:.0f} s: "
                    f"{self.last_settle_detail}")

        elif kind == "wait":
            if self.wait_since is None:
                self.wait_since = float(now)

        # Targets: recompute every sweep once past WAIT_CROUCH (cheap
        # arithmetic; the IK toward them is spread across CAN slots by
        # refine_leg).  Trim only moves in its LEVEL stages, height_frac
        # only moves in streams, so this is uniform and bumpless.
        if self.stage in ("RISE", "RISE_SETTLE", "HOLD", "PARK", "PARK_SETTLE"):
            raw = self.plan.foot_targets(self.height_frac, self.trim.offsets)
            self.targets = self._rate_clamp(now, raw)
            self._guard_resid(now)

        return event

    def _rate_clamp(self, now, raw):
        """Bound per-foot target motion to MAX_TARGET_RATE_M_S (dt-based).
        Only z ever moves (x/y are pinned anchors).  Sustained clamping
        means a planner/trim runaway -> latch a fault."""
        if self._prev_targets is None or self._prev_target_time is None:
            self._prev_targets = {leg: raw[leg].copy() for leg in LEGS}
            self._prev_target_time = float(now)
            self.rate_clamped = False
            return self._prev_targets
        dt = float(np.clip(now - self._prev_target_time, 0.0, 0.1))
        self._prev_target_time = float(now)
        max_step = MAX_TARGET_RATE_M_S * dt
        clamped = False
        out = {}
        for leg in LEGS:
            prev = self._prev_targets[leg]
            delta = raw[leg] - prev
            step = np.clip(delta, -max_step, max_step)
            if float(np.max(np.abs(delta - step))) > 1e-12:
                clamped = True
            out[leg] = prev + step
        self._prev_targets = out
        self.rate_clamped = clamped
        if clamped:
            if self._rate_clamp_since is None:
                self._rate_clamp_since = float(now)
            elif (now - self._rate_clamp_since > RATE_CLAMP_STUCK_S
                    and self.fault is None):
                self.fault = ("target rate clamp stuck "
                              f"> {RATE_CLAMP_STUCK_S:.1f} s "
                              "(planner or trim runaway)")
        else:
            self._rate_clamp_since = None
        return out

    def _guard_resid(self, now):
        worst_leg = max(LEGS, key=lambda leg: self.resid[leg])
        worst = self.resid[worst_leg]
        if worst > IK_RESID_TRIP_M:
            if self._resid_since is None:
                self._resid_since = float(now)
            elif (now - self._resid_since > IK_RESID_TRIP_S
                    and self.fault is None):
                self.fault = (f"IK residual {1000 * worst:.1f} mm on "
                              f"{worst_leg} for > {IK_RESID_TRIP_S:.1f} s "
                              "(target unreachable?)")
        else:
            self._resid_since = None

    def refine_leg(self, slot_counter):
        """A few warm-started IK iterations for ONE leg of q_cmd.  Called
        once per IK_SLOT_STRIDE CAN slots so the IK cost never blocks the
        round-robin long enough to trip the 10 ms motor watchdog."""
        if self.targets is None:
            return
        leg = LEGS[slot_counter % len(LEGS)]
        sl = self.plan._sl(leg)
        self.resid[leg] = float(np.linalg.norm(
            self.targets[leg]
            - dog5_kinematics.foot_position(leg, self.q_cmd[sl])))
        self.q_cmd[sl] = _ik_to_target(
            leg, self.q_cmd[sl], self.targets[leg],
            max_iter=IK_SLOT_MAX_ITER, tol=IK_SLOT_TOL_M,
        )

    def speed_cap_dps(self, caps):
        if self.stage in ("CROUCH", "WAIT_CROUCH", "PARKED"):
            return caps["crouch"]
        if self.stage in ("RISE", "PARK"):
            return caps["rise"]
        return caps["stream"]


# ---------------------------------------------------------------------------
# Leveling input selection (runner-side): EKF attitude + freeze rules
# ---------------------------------------------------------------------------
def trim_inputs(shared, stage, aborted, imu_sample, level_gain,
                level_stages, now_mono):
    """Return (roll_rad, pitch_rad, active, reason) for LevelingTrim.update.

    Freeze (hold offsets) whenever the attitude cannot be trusted or the
    stage does not want leveling.  The stand is statically stable without
    leveling, so freezing -- never stopping -- is the correct response.
    """
    if level_gain <= 0.0:
        return None, None, False, "gain 0"
    if stage not in level_stages:
        return None, None, False, f"stage {stage}"
    if aborted:
        return None, None, False, "aborted (frozen)"
    if shared is None or not shared.est_ready:
        return None, None, False, "EKF not ready"
    out = shared.out
    if out is None:
        return None, None, False, "EKF no output"
    if not out.get("healthy", False):
        return None, None, False, "EKF unhealthy"
    if now_mono - shared.tau_stamp > LEVEL_STALE_S:
        return None, None, False, "EKF worker stale"
    roll, pitch = _rp(out["C"])
    if max(abs(math.degrees(roll)), abs(math.degrees(pitch))) \
            > LEVEL_MAX_TILT_DEG:
        return None, None, False, f"tilt > {LEVEL_MAX_TILT_DEG:.0f} deg"
    if imu_sample is not None and imu_sample.age_s <= inplace.POSTURE_STALE_S:
        d_r = abs(math.degrees(roll) - imu_sample.roll_deg)
        d_p = abs(math.degrees(pitch) - imu_sample.pitch_deg)
        if max(d_r, d_p) > LEVEL_XCHECK_DEG:
            return None, None, False, (
                f"EKF/AHRS mismatch {max(d_r, d_p):.1f} deg")
    return roll, pitch, True, ""


# ---------------------------------------------------------------------------
# Offline validation (before arming, every hardware run)
# ---------------------------------------------------------------------------
def validate_configuration(plan, level_clamp, torque_trip):
    """Sweep the whole rise path + leveling envelope with full IK before
    any motor moves.  Model-space targets are trustworthy (the encoder
    zero-offset caveat only corrupts MEASURED q), so soft limits are a
    hard assertion here even though they are report-only at runtime."""
    if not (recorded.MIN_STAND_HEIGHT <= plan.stand_height
            <= inplace.MAX_STAND_HEIGHT):
        raise ValueError(
            f"--stand-height {plan.stand_height:.3f} outside the validated "
            f"[{recorded.MIN_STAND_HEIGHT:.2f}, "
            f"{inplace.MAX_STAND_HEIGHT:.2f}] m")

    low, high = base.soft_limits()
    zero_off = {leg: 0.0 for leg in LEGS}
    support = np.array([0.0, 0.0, -base.DOG5_MASS_KG * 9.81 / 4.0])

    worst_resid = 0.0
    min_margin = np.inf
    worst_tau = 0.0
    q = Q_RECORDED_CROUCH.copy()

    def check(q_full, tag):
        nonlocal worst_resid, min_margin, worst_tau
        if np.any(q_full < low) or np.any(q_full > high):
            bad = int(np.flatnonzero((q_full < low) | (q_full > high))[0])
            raise ValueError(
                f"planned pose exceeds soft limits at {tag}: "
                f"{JOINT_LABELS[bad]}={q_full[bad]:+.3f} rad")
        min_margin = min(min_margin, float(
            np.min(np.minimum(q_full - low, high - q_full))))
        for leg in LEGS:
            sl = plan._sl(leg)
            tau = dog5_kinematics.foot_jacobian(leg, q_full[sl]).T @ support
            worst_tau = max(worst_tau, float(np.max(np.abs(tau))))

    # the nominal rise path (warm-started, like the runtime refiner)
    for frac in np.linspace(0.0, 1.0, 21):
        targets = plan.foot_targets(float(frac), zero_off)
        q = plan.q_targets(q, float(frac), zero_off)
        for leg in LEGS:
            sl = plan._sl(leg)
            resid = float(np.linalg.norm(
                targets[leg] - dog5_kinematics.foot_position(leg, q[sl])))
            worst_resid = max(worst_resid, resid)
        check(q, f"height_frac={frac:.2f}")

    # the leveling envelope at full height: per-leg IK is independent, so
    # per-leg +/-clamp corners bound every tilt pattern the trim can command
    for sign in (+1.0, -1.0):
        off = {leg: sign * float(level_clamp) for leg in LEGS}
        q_c = plan.q_targets(q, 1.0, off)
        targets = plan.foot_targets(1.0, off)
        for leg in LEGS:
            sl = plan._sl(leg)
            resid = float(np.linalg.norm(
                targets[leg] - dog5_kinematics.foot_position(leg, q_c[sl])))
            worst_resid = max(worst_resid, resid)
        check(q_c, f"level offset {sign * level_clamp * 1e3:+.0f} mm")

    if worst_resid > 1.0e-4:
        raise ValueError(f"path IK residual {worst_resid * 1e3:.2f} mm "
                         "> 0.1 mm -- targets not reachable")
    if torque_trip <= worst_tau:
        raise ValueError(
            f"--torque-trip {torque_trip:.2f} N*m must exceed the worst "
            f"planned static stance torque {worst_tau:.2f} N*m")
    if torque_trip >= 9.0:
        raise ValueError("--torque-trip must stay below 9 N*m")

    return {"worst_resid_m": worst_resid, "min_margin_rad": min_margin,
            "worst_static_tau_nm": worst_tau}


def geometry_report(q_meas, q_cmd, tag):
    """Print FK predictions for quantities a tape measure can check.

    Separates the three explanations for a commanded-vs-measured height
    gap.  Compare, in order:
      1. worst |q_cmd - q_meas| -- if it is large the joints simply are
         not tracking (sag, torque limit, speed cap) and the model is
         fine.  If it is small, the encoders believe the pose is correct
         and the gap is in the model or the encoder zeros.
      2. hip-axis height -- the FK trunk origin, +FOOT_RADIUS_M for the
         contact sphere.  This is the number to measure to the hip pivot
         centre.
      3. foot spreads and knee height -- HEIGHT-INDEPENDENT checks of the
         same joint angles.  If the spreads match the tape but the height
         does not, the error is in the pitch/knee chain; if the spreads
         are wrong too, suspect the encoder zeros or the link lengths.
    """
    q_meas = np.asarray(q_meas, dtype=float)
    q_cmd = np.asarray(q_cmd, dtype=float)

    def sl(leg):
        i = LEGS.index(leg)
        return slice(3 * i, 3 * i + 3)

    def geom(q):
        feet = {leg: dog5_kinematics.foot_position(leg, q[sl(leg)])
                for leg in LEGS}
        knees = {leg: dog5_kinematics._state(leg, q[sl(leg)])[1][2]
                 for leg in LEGS}
        h_site = -float(np.mean([feet[leg][2] for leg in LEGS]))
        fore = float(np.mean([feet["FL"][0], feet["FR"][0]]))
        rear = float(np.mean([feet["RL"][0], feet["RR"][0]]))
        left = float(np.mean([feet["FL"][1], feet["RL"][1]]))
        right = float(np.mean([feet["FR"][1], feet["RR"][1]]))
        knee_above_ground = float(np.mean(
            [knees[leg][2] for leg in LEGS])) + h_site + FOOT_RADIUS_M
        return {"hip_axis": h_site + FOOT_RADIUS_M,
                "site": h_site,
                "fore_aft": fore - rear,
                "lateral": left - right,
                "knee": knee_above_ground}

    gm, gc = geom(q_meas), geom(q_cmd)
    worst = int(np.argmax(np.abs(q_meas - q_cmd)))
    print(f"[geom] {tag}: worst joint tracking error "
          f"{JOINT_LABELS[worst]}={np.rad2deg(q_meas[worst] - q_cmd[worst]):+.1f}"
          f" deg (tol {np.rad2deg(posbase.POSITION_POSE_TOL):.1f})")
    hip_in = gm["hip_axis"] / 0.0254
    trunk_bottom = gm["hip_axis"] - IMU_BELOW_TRUNK_ORIGIN_M
    print(f"[geom]   MEASURE ground -> hip pivot centre: predict "
          f"{gm['hip_axis'] * 1e3:.0f} mm = {hip_in:.2f} in "
          f"(foot site {gm['site'] * 1e3:.0f} + {FOOT_RADIUS_M * 1e3:.0f} mm "
          f"contact sphere)")
    print(f"[geom]   MEASURE ground -> trunk bottom / IMU: predict "
          f"{trunk_bottom * 1e3:.0f} mm = {trunk_bottom / 0.0254:.2f} in "
          f"(hip axis - {IMU_BELOW_TRUNK_ORIGIN_M * 1e3:.0f} mm IMU offset)")
    print(f"[geom]   MEASURE ground -> knee pivot centre: predict "
          f"{gm['knee'] * 1e3:.0f} mm")
    print(f"[geom]   MEASURE front-foot to rear-foot spacing: predict "
          f"{gm['fore_aft'] * 1e3:.0f} mm")
    print(f"[geom]   MEASURE left-foot to right-foot spacing: predict "
          f"{gm['lateral'] * 1e3:.0f} mm")
    print(f"[geom]   (from commanded q the same four are "
          f"{gc['hip_axis'] * 1e3:.0f} / {gc['knee'] * 1e3:.0f} / "
          f"{gc['fore_aft'] * 1e3:.0f} / {gc['lateral'] * 1e3:.0f} mm)",
          flush=True)


def report_limit_margins(q, tag):
    """Report-only soft-limit check of a MEASURED pose (never gates: the
    limits are software bounds against the encoder zeros, and a wrong zero
    offset rejects a mechanically fine pose)."""
    low, high = base.soft_limits()
    q = np.asarray(q, dtype=float)
    outside = np.flatnonzero((q < low) | (q > high))
    if not outside.size:
        margin = float(np.min(np.minimum(q - low, high - q)))
        print(f"[limits] {tag}: inside nominal soft limits "
              f"(tightest margin {margin:.2f} rad)")
    else:
        detail = " ".join(f"{JOINT_LABELS[i]}={q[i]:+.2f}" for i in outside)
        print(f"[limits] {tag}: {outside.size} joint(s) outside nominal "
              f"soft limits (allowed, report-only): {detail}")
        print("[limits] if a joint is NOT physically there, its encoder "
              "zero offset is wrong.")


# ---------------------------------------------------------------------------
# Hardware run
# ---------------------------------------------------------------------------
def run_hardware(args):
    plan = StandHierPlan(args.stand_height)
    stats = validate_configuration(plan, args.level_clamp, args.torque_trip)
    crouch_h = -float(np.mean(
        [plan.crouch_foot[leg][2] for leg in LEGS]))
    print("=" * 74)
    print("DOG5 STAGE-A HIERARCHICAL STAND (position mode + EKF leveling)")
    print(f"  rise: crouch h={crouch_h:.3f} -> stand h={plan.stand_height:.3f}"
          f" m over {args.t_rise:.0f} s (feet pinned at the crouch xy)")
    print(f"  validated: IK resid {stats['worst_resid_m'] * 1e3:.2f} mm, "
          f"limit margin {stats['min_margin_rad']:.2f} rad, "
          f"static tau {stats['worst_static_tau_nm']:.2f} N*m "
          f"(trip {args.torque_trip:.1f})")
    if args.level_gain > 0.0:
        stages = ("HOLD", "RISE", "RISE_SETTLE") if args.level_in_rise \
            else ("HOLD",)
        print(f"  leveling: gain {args.level_gain:.2f}/s clamp "
              f"{args.level_clamp * 1e3:.0f} mm slew "
              f"{LEVEL_SLEW_M_S * 1e3:.0f} mm/s in {'+'.join(stages)} "
              f"(EKF attitude; freeze on unhealthy/stale/mismatch)")
    else:
        print("  leveling: OFF (--level-gain 0)")
    print(f"  speed caps (motor dps): crouch {args.crouch_max_speed_dps:.0f} "
          f"rise {args.rise_max_speed_dps:.0f} stream 250")
    print(f"  sweep {base.CONTROL_HZ:.0f} Hz per motor "
          f"({1e6 / (base.CONTROL_HZ * N_JOINTS):.0f} us slot, "
          f"{1e3 / base.CONTROL_HZ:.0f} ms sweep)")
    print("  speed: NO TRIP (encoder + driver overspeed both removed); "
          f"bounded by the motor-side cap only")
    print(f"  position: path validated offline (hard); measured-pose limit "
          f"estop {'ON' if args.enforce_limits else 'OFF (report-only)'}")
    print(f"  torque trip {args.torque_trip:.1f} N*m; overtemp, CAN miss, "
          f"motor faults, IMU tip-over, IK residual all live")
    print(f"  EKF contacts during RISE/PARK: {args.rise_contacts}"
          + ("  (feet never slide here -- height stays observable)"
             if args.rise_contacts == "planted"
             else "  (A/B: dead-reckons r for the whole stream)"))
    if args.log:
        print(f"  CSV log: {args.log} at {args.log_hz:.0f} Hz")
    print("  ROBOT MUST REMAIN SUPPORTED/TETHERED UNTIL THE RUNBOOK "
          "SIGN CHECKS PASS.")
    print("=" * 74)

    level_stages = (("HOLD", "RISE", "RISE_SETTLE") if args.level_in_rise
                    else ("HOLD",))
    caps = {"crouch": args.crouch_max_speed_dps,
            "rise": args.rise_max_speed_dps, "stream": 250.0}
    trim = LevelingTrim(plan.anchors_xy, gain_per_s=args.level_gain,
                        clamp_m=args.level_clamp)
    unwrap = [base.CalibratedEncoderUnwrap() for _ in MOTOR_IDS]
    # inf thresholds disable SafetyGate's overspeed tier without editing base
    # (twostand / stand3 / walk1 share that class).  Every other SafetyGate
    # check -- overtemp, CAN miss estop, motor hard faults -- stays live.
    safety = base.SafetyGate(tau_cap=1.0, qd_estop=np.inf,
                             qd_estop_hard=np.inf)
    speed_watch = SpeedWatch()

    # One serial-port owner: ImuEkfFeed wraps the AHRS (feed.attitude()) and
    # the raw 0x40 stream the EKF predicts on (walk1 pattern).
    from imu_ekf_feed import ImuEkfFeed          # noqa: E402 (serial import)
    feed = None
    ekf_enabled = False
    if not args.no_imu:
        feed = ImuEkfFeed(args.port).start()
        print(f"[imu] tip-over {inplace.TIPOVER_ABORT_DEG:.0f} deg; lean "
              f"watch {LEAN_ABORT_DEG:.0f} deg during RISE")
        ekf_enabled = not args.no_ekf
        if ekf_enabled and not feed.wait_for_raw(timeout=3.0):
            print("[ekf] WARNING: no raw IMU (0x40) packets -- EKF disabled")
            ekf_enabled = False
    if args.level_gain > 0.0 and not ekf_enabled:
        raise RuntimeError("leveling needs the EKF; use --level-gain 0 "
                           "with --no-imu/--no-ekf")
    if args.raw_log and not ekf_enabled:
        raise RuntimeError("--raw-log needs the EKF (raw IMU stream)")

    csv_writer = csv_file = None
    if args.log:
        csv_file = open(args.log, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            # h_fk_m / dh_fk_m are the encoder-side truth for rz_m: with the
            # feet planted the EKF height is essentially leg odometry, so
            # (rz_m - dh_fk_m) is the estimator's running error against a
            # reference that cannot drift.  Plot that, not rz_m alone.
            ["t", "stage", "rz_m", "h_fk_m", "dh_fk_m", "vx", "vy", "vz",
             "roll_est_deg", "pitch_est_deg", "roll_ahrs_deg",
             "pitch_ahrs_deg", "healthy",
             "off_FL_mm", "off_FR_mm", "off_RL_mm", "off_RR_mm",
             "level_frozen"])

    shared = None
    worker = None
    web_stop = None
    enc_log = []
    key = base.KeyPoller()
    try:
        with base.motorbus.MotorBus(MOTOR_IDS, dirs=MOTOR_DIRECTIONS) as mb:
            armed = False
            stop_reason = None
            try:
                print(f"[ARM] arming for up to {posbase.ARM_TIMEOUT_S:.0f}s; "
                      "power the motors on now if they are off, Ctrl-C aborts")
                # verbose=True: stand3/walk1 pass False, which makes a 15 s arm
                # look like a dead hang.  The stream prints which motors are
                # still pending every 0.5 s and names any that need a power
                # cycle -- that is the whole diagnosis, so do not silence it.
                if not mb.arm(rate_hz=base.CONTROL_HZ,
                              timeout_s=posbase.ARM_TIMEOUT_S, verbose=True):
                    raise RuntimeError("arming timed out: "
                                       + posbase.arm_failure_summary(mb))
                armed = True
                start_q = posbase.zero_torque_preflight(mb, key, unwrap)
                report_limit_margins(start_q, "rest pose")
                now = time.perf_counter()
                sequence = StandHierSequence(
                    now, plan, trim, t_rise=args.t_rise,
                    stream_feet_planted=(args.rise_contacts == "planted"))
                safety.start(now, start_q)

                if ekf_enabled:
                    shared = EkfShared(start_q)
                    worker = threading.Thread(
                        target=ekf_worker, args=(shared, feed),
                        kwargs=dict(quiet_stages=("WAIT_CROUCH",),
                                    verbose=args.ekf_verbose),
                        daemon=True)
                    worker.start()
                    print("[ekf] worker started (read-only, 100 Hz); "
                          "initialises during WAIT_CROUCH -- wait for "
                          "'[ekf] init' before ENTER")
                    if args.web:
                        import ekf_web                    # noqa: E402
                        web_stop, urls = ekf_web.start(shared, port=args.web)
                        for u in urls:
                            print(f"[web] live EKF dashboard: {u}")

                slot = mb.slot(base.CONTROL_HZ)
                deadline = time.perf_counter() + slot
                run_started_at = now
                q = start_q.copy()
                miss_monitor = base.CanMissMonitor(mb)
                status_period = 1.0 / base.FAULT_STATUS_HZ
                next_fault_status = np.asarray(
                    [status_period + i * status_period / N_JOINTS
                     for i in range(N_JOINTS)])
                last_recover = {mid: -base.RECOVER_PERIOD_S
                                for mid in MOTOR_IDS}
                last_print = 0.0
                last_load_print = 0.0
                last_csv = 0.0
                csv_period = 1.0 / max(args.log_hz, 1e-6)
                index = 0
                velocity = posbase.EncoderVelocity()
                qd_encoder = np.zeros(N_JOINTS)
                measured_torque = np.zeros(N_JOINTS)
                tau_tare = None
                worst_block_ms = 0.0
                tipover_streak = 0
                lean_baseline = None
                lean_streak = 0
                imu_sample = None
                bias_printed = False
                tx_fail = 0

                while True:
                    mb.poll()
                    joint_index = index % N_JOINTS

                    if joint_index == 0:
                        now = time.perf_counter()
                        elapsed = now - run_started_at
                        q, qd_driver = base._joint_state(mb, unwrap)
                        qd_encoder = velocity.update(q, now)
                        torque_feedback = mb.torques_nm()
                        measured_torque = np.asarray(
                            [torque_feedback[mid] for mid in MOTOR_IDS])

                        imu_sample = None
                        if feed is not None:
                            try:
                                imu_sample = feed.attitude()
                            except Exception:
                                imu_sample = None

                        # leveling trim BEFORE sweep so this sweep's targets
                        # use this sweep's offsets
                        if sequence.stage in ("PARK", "PARK_SETTLE", "PARKED"):
                            trim.decay(now)
                        else:
                            roll, pitch, active, why = trim_inputs(
                                shared, sequence.stage, sequence.aborted,
                                imu_sample, args.level_gain, level_stages,
                                time.monotonic())
                            trim.update(now, roll, pitch, active, why)

                        event = sequence.sweep(now, q, qd_encoder,
                                               velocity.ready)
                        if event:
                            print(f"[stage] {event}")
                            # Close-out metric: WAIT_CROUCH and PARKED are the
                            # SAME physical pose, and the EKF defines r = 0 at
                            # its init in WAIT_CROUCH -- so z at PARKED is the
                            # accumulated drift over one rise+park cycle,
                            # measured against an exactly known truth of 0.
                            # FK height is the encoder-side reference.
                            if sequence.stage in ("HOLD", "PARKED"):
                                geometry_report(q, sequence.q_cmd,
                                                sequence.stage)
                                if (shared is not None
                                        and shared.out is not None):
                                    z_ekf = float(shared.out["r"][2])
                                    h_fk = -float(np.mean(
                                        [dog5_kinematics.foot_position(
                                            leg, q[plan._sl(leg)])[2]
                                         for leg in LEGS]))
                                    d_fk = h_fk - crouch_h
                                    print(
                                        f"[ekf] {sequence.stage} close-out: "
                                        f"z_ekf={z_ekf * 1e3:+.0f}mm "
                                        f"vs FK rise={d_fk * 1e3:+.0f}mm "
                                        f"-> error "
                                        f"{(z_ekf - d_fk) * 1e3:+.0f}mm"
                                        + ("  (PARKED truth is 0)"
                                           if sequence.stage == "PARKED"
                                           else ""), flush=True)
                        if sequence.fault and stop_reason is None:
                            stop_reason = sequence.fault

                        if shared is not None:
                            shared.q = q
                            shared.qd = qd_encoder
                            shared.stage = sequence.stage
                            shared.contacts = sequence.contacts
                            if (not shared.log_enabled
                                    and sequence.stage != "CROUCH"):
                                shared.log_enabled = True
                            if not bias_printed and shared.est_ready:
                                print(f"[ekf] init: {shared.bias_str}")
                                bias_printed = True

                        temps = base._temperatures(mb)
                        misses = miss_monitor.update(mb)
                        errors = mb.errors()
                        if stop_reason is None:
                            stop_reason = safety.estop_reason(
                                q, qd_driver, temps, misses, errors, now,
                                enforce_position_limits=args.enforce_limits)
                        if stop_reason is None:
                            stop_reason = speed_watch.check(
                                qd_encoder, measured_torque, velocity.ready,
                                args.torque_trip)

                        # IMU judges (never steers): tip-over + rise lean
                        if (imu_sample is not None
                                and imu_sample.age_s
                                <= inplace.POSTURE_STALE_S):
                            roll_a = imu_sample.roll_deg
                            pitch_a = imu_sample.pitch_deg
                            if (abs(roll_a) > inplace.TIPOVER_ABORT_DEG
                                    or abs(pitch_a)
                                    > inplace.TIPOVER_ABORT_DEG):
                                tipover_streak += 1
                                if (stop_reason is None and tipover_streak
                                        >= inplace.TIPOVER_CONFIRM_SAMPLES):
                                    stop_reason = (
                                        f"IMU tip-over: roll={roll_a:+.1f} "
                                        f"pitch={pitch_a:+.1f} deg")
                            else:
                                tipover_streak = 0
                            if sequence.stage == "RISE":
                                if lean_baseline is None:
                                    lean_baseline = (roll_a, pitch_a)
                                    lean_streak = 0
                                else:
                                    d_r = roll_a - lean_baseline[0]
                                    d_p = pitch_a - lean_baseline[1]
                                    if max(abs(d_r), abs(d_p)) \
                                            > LEAN_ABORT_DEG:
                                        lean_streak += 1
                                        if lean_streak >= LEAN_STREAK:
                                            msg = sequence.abort_freeze(
                                                now,
                                                f"IMU lean d_roll={d_r:+.1f}"
                                                f" d_pitch={d_p:+.1f} deg")
                                            if msg:
                                                print(f"[stage] {msg}")
                                            lean_baseline = None
                                            lean_streak = 0
                                    else:
                                        lean_streak = 0
                            else:
                                lean_baseline = None
                                lean_streak = 0

                        if (args.raw_log and shared is not None
                                and shared.log_enabled):
                            r_a = p_a = float("nan")
                            if (imu_sample is not None
                                    and imu_sample.age_s
                                    <= inplace.POSTURE_STALE_S):
                                r_a = math.radians(imu_sample.roll_deg)
                                p_a = math.radians(imu_sample.pitch_deg)
                            enc_log.append(
                                (time.monotonic(), *q,
                                 *sequence.contacts.astype(int), r_a, p_a))

                        latched = [mid for mid, err in errors.items()
                                   if err & 0x80]
                        unverified = [mid for mid in MOTOR_IDS
                                      if mb.rec(mid).error is None]
                        healthy = (not latched and not unverified
                                   and stop_reason is None)
                        ekf_ok = (args.level_gain <= 0.0
                                  or (shared is not None
                                      and shared.est_ready))

                        pressed = key.get()
                        if pressed in ("x", "X"):
                            stop_reason = "operator X"
                        elif pressed in ("p", "P"):
                            _, message = sequence.request_park(now, healthy)
                            print(f"[stage] {message}")
                        elif pressed in ("t", "T"):
                            tau_tare = measured_torque - leg_gravity_stack(q)
                            print("[legs] torque tare captured -- valid "
                                  "ONLY if ALL feet were off the ground.")
                        elif base._is_enter(pressed):
                            _, message = sequence.request_next(
                                now, healthy, ekf_ok)
                            print(f"[stage] {message}")
                        if stop_reason:
                            break

                        recover = [mid for mid in latched
                                   if elapsed - last_recover[mid]
                                   >= base.RECOVER_PERIOD_S]
                        if recover:
                            base._recover_input_lost(
                                mb, recover, elapsed, last_recover,
                                next_fault_status)

                        # Throttled: a row per 250 Hz sweep is 250 rows/s of
                        # file I/O inside the CAN loop, and one SD-card flush
                        # stall starves every motor's input timeout.
                        if (csv_writer is not None
                                and now - last_csv >= csv_period):
                            last_csv = now
                            rz = vx = vy = vz = float("nan")
                            r_e = p_e = float("nan")
                            hlt = 0
                            if shared is not None and shared.out is not None:
                                out = shared.out
                                rz = float(out["r"][2])
                                vx, vy, vz = (float(v) for v in out["v"])
                                rr, pp = _rp(out["C"])
                                r_e, p_e = (math.degrees(rr),
                                            math.degrees(pp))
                                hlt = int(bool(out.get("healthy", False)))
                            r_a = p_a = float("nan")
                            if imu_sample is not None:
                                r_a = imu_sample.roll_deg
                                p_a = imu_sample.pitch_deg
                            h_fk = -float(np.mean(
                                [dog5_kinematics.foot_position(
                                    leg, q[plan._sl(leg)])[2]
                                 for leg in LEGS]))
                            csv_writer.writerow(
                                [f"{now:.4f}", sequence.stage, f"{rz:.4f}",
                                 f"{h_fk:.4f}", f"{h_fk - crouch_h:.4f}",
                                 f"{vx:.4f}", f"{vy:.4f}", f"{vz:.4f}",
                                 f"{r_e:.3f}", f"{p_e:.3f}",
                                 f"{r_a:.3f}", f"{p_a:.3f}", hlt,
                                 *(f"{trim.offsets[leg] * 1e3:.2f}"
                                   for leg in LEGS),
                                 trim.frozen_reason])

                        if now - last_load_print >= 1.0 / LOAD_STATUS_HZ:
                            last_load_print = now
                            if sequence.stage in ("HOLD", "RISE_SETTLE",
                                                  "WAIT_CROUCH"):
                                support, com_xy = foot_load_map(
                                    q, measured_torque, tau_tare)
                                fz_text = ",".join(
                                    f"{leg}{support[leg]:+.1f}"
                                    if np.isfinite(support[leg])
                                    else f"{leg}=nan" for leg in LEGS)
                                com_text = ("--" if np.any(~np.isfinite(
                                    com_xy)) else
                                    f"({com_xy[0] * 1e3:+.0f},"
                                    f"{com_xy[1] * 1e3:+.0f})mm")
                                print(f"[legs] fz_N=({fz_text}) "
                                      f"com={com_text}"
                                      f"{' (tared)' if tau_tare is not None else ''}",
                                      flush=True)

                        if now - last_print >= 1.0 / base.STATUS_HZ:
                            lvl_text = " ".join(
                                f"{leg}{trim.offsets[leg] * 1e3:+.1f}"
                                for leg in LEGS)
                            frozen = (f" FROZEN:{trim.frozen_reason}"
                                      if trim.frozen_reason
                                      and args.level_gain > 0 else "")
                            resid_mm = 1000 * max(sequence.resid.values())
                            print(
                                f"[hier] {sequence.stage:<12} "
                                f"{INSTRUCTION[sequence.stage]} | "
                                f"h={100 * sequence.height_frac:3.0f}% "
                                f"qd={np.max(np.abs(qd_encoder)):.2f}"
                                f"/pk{speed_watch.peek_peak():.2f} "
                                f"max|tau|="
                                f"{np.max(np.abs(measured_torque)):.2f} "
                                f"Tmax={int(np.max(temps))}C "
                                f"resid={resid_mm:.1f}mm "
                                f"lvl=({lvl_text})mm{frozen} "
                                f"txfail={tx_fail} "
                                f"blk={worst_block_ms:.1f}ms",
                                flush=True)
                            if shared is not None:
                                out = shared.out
                                if shared.est_ready and out is not None:
                                    rr, pp = _rp(out["C"])
                                    ahrs_text = "--"
                                    if imu_sample is not None:
                                        ahrs_text = (
                                            f"({imu_sample.roll_deg:+.1f},"
                                            f"{imu_sample.pitch_deg:+.1f})")
                                    cbits = "".join(
                                        "1" if c else "0"
                                        for c in sequence.contacts)
                                    print(
                                        f"[ekf] z={out['r'][2] * 1e3:+5.0f}mm"
                                        f" rp=({math.degrees(rr):+5.1f},"
                                        f"{math.degrees(pp):+5.1f})deg "
                                        f"ahrs={ahrs_text} c={cbits} "
                                        f"H={'OK' if out['healthy'] else '!!'}",
                                        flush=True)
                                else:
                                    print("[ekf] initialising (needs static "
                                          "WAIT_CROUCH)...", flush=True)
                            last_print = now
                            worst_block_ms = 0.0
                        worst_block_ms = max(
                            worst_block_ms,
                            1000.0 * (time.perf_counter() - now))

                    if index % IK_SLOT_STRIDE == 0:
                        sequence.refine_leg(index // IK_SLOT_STRIDE)

                    mid = MOTOR_IDS[joint_index]
                    elapsed = time.perf_counter() - run_started_at
                    if elapsed >= next_fault_status[joint_index]:
                        mb.status1_req(mid)
                        next_fault_status[joint_index] += status_period
                    else:
                        # ENOBUFS drops ONE frame; the motor holds and the
                        # next sweep resends -- tolerate it (sustained bus
                        # death reaches MISS_ESTOP anyway).
                        if not mb.position(
                                mid,
                                float(np.rad2deg(
                                    sequence.q_cmd[joint_index])),
                                sequence.speed_cap_dps(caps)):
                            tx_fail += 1

                    index += 1
                    overrun = mb.pace(deadline)
                    deadline += slot
                    if overrun and overrun > 2.0 * slot:
                        # resync after a stall instead of send-bursting
                        deadline = time.perf_counter() + slot

            except KeyboardInterrupt as exc:
                stop_reason = str(exc) or "KeyboardInterrupt"
            except Exception as exc:
                stop_reason = f"error: {exc}"
                raise
            finally:
                if armed:
                    print(f"[STOP] {stop_reason or 'aborted'}")
                    mb.stop_all()
    finally:
        key.close()
        if web_stop is not None:
            web_stop()
        if shared is not None:
            shared.run = False
        if worker is not None:
            worker.join(timeout=1.0)
        if feed is not None:
            try:
                feed.stop()
            except Exception:
                pass
        if csv_file is not None:
            csv_file.close()
            print(f"[log] wrote {args.log}")
        if args.raw_log and shared is not None:
            imu_arr = np.array(shared.imu_log, dtype=float)
            enc_arr = np.array(enc_log, dtype=float)
            if len(imu_arr) and len(enc_arr):
                np.savez(args.raw_log,
                         imu_t=imu_arr[:, 0], imu_f=imu_arr[:, 1:4],
                         imu_w=imu_arr[:, 4:7],
                         enc_t=enc_arr[:, 0], enc_alpha=enc_arr[:, 1:13],
                         enc_contacts=enc_arr[:, 13:17],
                         ahrs_rp=enc_arr[:, 17:19],
                         init_secs=1.0)
                print(f"[raw] wrote {args.raw_log} ({len(imu_arr)} IMU, "
                      f"{len(enc_arr)} enc frames); analyse with "
                      f"hw_replay.py --static")
            else:
                print("[raw] nothing to write")
    return 0


# ---------------------------------------------------------------------------
# Offline self-test (no hardware, no MuJoCo)
# ---------------------------------------------------------------------------
def offline_self_test(args):
    failures = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              f"{'  ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    print("[1] configuration validator")
    plan = StandHierPlan(args.stand_height)
    stats = validate_configuration(plan, args.level_clamp, args.torque_trip)
    print(f"  path IK residual {stats['worst_resid_m'] * 1e3:.3f} mm, "
          f"min limit margin {stats['min_margin_rad']:.3f} rad, "
          f"worst static tau {stats['worst_static_tau_nm']:.2f} N*m")
    check("validator numbers sane",
          stats["worst_resid_m"] < 1e-4
          and 0.05 < stats["min_margin_rad"] < 1.0
          and 0.5 < stats["worst_static_tau_nm"] < 2.0)
    try:
        validate_configuration(StandHierPlan(0.30), args.level_clamp, 6.0)
        check("refuses stand-height 0.30", False)
    except ValueError:
        check("refuses stand-height 0.30", True)
    try:
        validate_configuration(plan, args.level_clamp, 0.5)
        check("refuses torque-trip below static tau", False)
    except ValueError:
        check("refuses torque-trip below static tau", True)

    print("[2] _rp attitude convention (the sign the whole design rests on)")
    for axis, expect_name, idx in ((1, "pitch", 1), (0, "roll", 0)):
        ang = math.radians(3.0)
        c, s = math.cos(ang), math.sin(ang)
        if axis == 1:                     # R_IB = Ry(+3 deg): nose DOWN
            R_IB = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:                             # R_IB = Rx(+3 deg): RIGHT side down
            R_IB = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        got = math.degrees(_rp(R_IB.T)[idx])
        check(f"_rp {expect_name}: R_IB rot +3 deg -> {expect_name} +3 deg",
              abs(got - 3.0) < 1e-9, f"got {got:+.3f}")

    print("[3] LevelingTrim law")
    trim = LevelingTrim(plan.anchors_xy)
    theta = math.radians(2.0)
    expect_front = -plan.anchors_xy["FL"][0] * theta       # -11.9 mm
    now = 0.0
    trim.update(now, 0.0, theta, True)                     # prime dt
    for _ in range(2000):
        now += 0.048
        trim.update(now, 0.0, theta, True)
    check("converges to the geometric error under +2 deg pitch (nose down)",
          abs(trim.offsets["FL"] - expect_front) < 5e-4
          and abs(trim.offsets["FR"] - expect_front) < 5e-4,
          f"FL {trim.offsets['FL'] * 1e3:+.1f} mm vs "
          f"{expect_front * 1e3:+.1f} mm")
    check("front extends (negative z), rear shortens, zero-mean",
          trim.offsets["FL"] < 0 and trim.offsets["RL"] > 0
          and abs(sum(trim.offsets.values())) < 1e-6)
    trim2 = LevelingTrim(plan.anchors_xy)
    trim2.update(0.0, 0.0, math.radians(25.0), True)
    step_ok = True
    prev = dict(trim2.offsets)
    now = 0.0
    for _ in range(200):
        now += 0.048
        trim2.update(now, 0.0, math.radians(25.0), True)
        for leg in LEGS:
            if abs(trim2.offsets[leg] - prev[leg]) > LEVEL_SLEW_M_S * 0.048 + 1e-9:
                step_ok = False
        prev = dict(trim2.offsets)
    check("per-tick step never exceeds the slew limit", step_ok)
    check("clamps at +/-12 mm under 25 deg input",
          abs(min(trim2.offsets.values()) + LEVEL_CLAMP_M) < 1e-9)
    frozen = dict(trim2.offsets)
    trim2.update(now + 0.048, None, None, True)
    trim2.update(now + 0.096, 0.0, theta, False, "test-freeze")
    check("freeze paths hold the offsets",
          trim2.offsets == frozen and trim2.frozen_reason == "test-freeze")
    for _ in range(3000):
        now += 0.048
        trim2.decay(now)
    check("decay ramps offsets to zero",
          max(abs(v) for v in trim2.offsets.values()) < 1e-9)

    # 0.004 s is the real cadence (250 Hz per motor, 12 x 333 us slots);
    # 0.048 s is a 20x-slower stress case, kept to prove the laws are all
    # dt-based and the stage walk is rate-independent.
    print("[4] sequence walk (synthetic instant-tracking plant), dt sweep")
    final = {}
    for dt in (0.048, 0.004):
        trim3 = LevelingTrim(plan.anchors_xy)
        seq = StandHierSequence(0.0, plan, trim3, t_rise=args.t_rise)
        now = 0.0
        stages_seen = [seq.stage]

        def spin(stop_stages, limit_s=60.0, refine=True):
            nonlocal now
            deadline = now + limit_s
            while seq.stage not in stop_stages and now < deadline:
                now += dt
                seq.sweep(now, seq.q_cmd.copy(), np.zeros(N_JOINTS), True)
                if refine:
                    for k in range(4):
                        seq.refine_leg(k)
                if stages_seen[-1] != seq.stage:
                    stages_seen.append(seq.stage)
            return seq.stage in stop_stages

        ok = spin(("WAIT_CROUCH",))
        refused, msg = seq.request_next(now, healthy=True, ekf_ok=True)
        check(f"dt={dt}: ENTER refused before the dwell", not refused, msg)
        now += base.WAIT_DWELL_S + 0.1
        seq.sweep(now, seq.q_cmd.copy(), np.zeros(N_JOINTS), True)
        refused, msg = seq.request_next(now, healthy=True, ekf_ok=False)
        check(f"dt={dt}: ENTER refused while EKF unready",
              not refused and "EKF" in msg, msg)
        refused, _ = seq.request_next(now, healthy=False, ekf_ok=True)
        check(f"dt={dt}: ENTER refused while unhealthy", not refused)
        advanced, _ = seq.request_next(now, healthy=True, ekf_ok=True)
        ok = ok and advanced and spin(("HOLD",))
        h_at_hold = seq.height_frac
        refused, _ = seq.request_next(now, healthy=True, ekf_ok=True)
        check(f"dt={dt}: ENTER in HOLD does nothing", not refused)
        advanced, _ = seq.request_park(now, healthy=True)
        ok = ok and advanced and spin(("PARKED",))
        check(f"dt={dt}: full stage walk completes", ok,
              "->".join(stages_seen))
        check(f"dt={dt}: height_frac 1.0 at HOLD, 0.0 at PARKED",
              abs(h_at_hold - 1.0) < 1e-9
              and abs(seq.height_frac) < 1e-9)
        check(f"dt={dt}: PARKED q_cmd snaps to the recorded crouch",
              float(np.max(np.abs(seq.q_cmd - Q_RECORDED_CROUCH))) < 1e-12)
        final[dt] = stages_seen
    check("stage walk identical at the 250 Hz cadence and 20x slower",
          final[0.048] == final[0.004])

    print("[4b] EKF contact schedule")
    seq_p = StandHierSequence(0.0, plan, LevelingTrim(plan.anchors_xy),
                              stream_feet_planted=True)
    seq_a = StandHierSequence(0.0, plan, LevelingTrim(plan.anchors_xy),
                              stream_feet_planted=False)
    planted_all = airborne_streams = True
    for name, _kind in STAGES:
        seq_p.stage_i = seq_a.stage_i = STAGE_INDEX[name]
        if not seq_p.contacts.all():
            planted_all = False
        expect_off = name in _STREAM_STAGES
        if bool(seq_a.contacts.any()) == expect_off:
            airborne_streams = False
    check("planted schedule keeps all four feet in contact everywhere",
          planted_all)
    check("airborne schedule still drops contact in RISE/PARK only",
          airborne_streams)
    # the x/y pinning that makes 'planted' physically true
    tgt_lo = plan.foot_targets(0.0, {leg: 0.0 for leg in LEGS})
    tgt_hi = plan.foot_targets(1.0, {leg: 0.0 for leg in LEGS})
    check("feet never slide during the rise (x/y identical, only z moves)",
          all(np.allclose(tgt_lo[leg][:2], tgt_hi[leg][:2]) for leg in LEGS)
          and all(abs(tgt_lo[leg][2] - tgt_hi[leg][2]) > 1e-3 for leg in LEGS))

    print("[5] stream guards")
    trim4 = LevelingTrim(plan.anchors_xy)
    seq = StandHierSequence(0.0, plan, trim4, t_rise=args.t_rise)
    seq.stage_i = STAGE_INDEX["HOLD"]
    seq.height_frac = 1.0
    now = 0.0
    seq.sweep(now, seq.q_cmd.copy(), np.zeros(N_JOINTS), True)
    trim4.offsets["FL"] = 0.5            # absurd trim jump -> rate clamp
    for _ in range(40):
        now += 0.048
        seq.sweep(now, seq.q_cmd.copy(), np.zeros(N_JOINTS), True)
    check("target-rate clamp latches a fault when stuck",
          seq.fault is not None and "rate clamp" in seq.fault, str(seq.fault))
    seq2 = StandHierSequence(0.0, plan, LevelingTrim(plan.anchors_xy),
                             t_rise=args.t_rise)
    seq2.stage_i = STAGE_INDEX["HOLD"]
    seq2.height_frac = 1.0
    now = 0.0
    for _ in range(20):
        now += 0.048
        seq2.resid = {leg: 0.02 for leg in LEGS}
        seq2.sweep(now, seq2.q_cmd.copy(), np.zeros(N_JOINTS), True)
    check("IK residual watch latches a fault",
          seq2.fault is not None and "IK residual" in seq2.fault,
          str(seq2.fault))

    print("[6] SpeedWatch (no speed trip) + EncoderVelocity")
    vel = posbase.EncoderVelocity()
    t = 0.0
    for _ in range(6):
        t += 0.004
        vel.update(Q_RECORDED_CROUCH + 1e-4 * t, t)
    check("EncoderVelocity ready after 5 samples", vel.ready)

    zero = np.zeros(N_JOINTS)
    watch = SpeedWatch()
    msg = watch.check(zero, np.full(N_JOINTS, 7.0), True, 6.0)
    check("torque trip survives and is immediate",
          msg is not None and "torque" in msg, msg)
    msg = watch.check(zero, np.full(N_JOINTS, np.nan), True, 6.0)
    check("invalid torque feedback still faults",
          msg is not None and "invalid" in msg, msg)

    # every speed that used to abort a run must now be silent
    watch = SpeedWatch()
    silent = True
    for level in (1.5, 7.0, 13.0, 550.0):        # trip, QD_ESTOP, ceiling,
        qd = np.zeros(N_JOINTS)                  # 360 deg unwrap step
        qd[4] = level
        for _ in range(50):
            if watch.check(qd, zero, True, 6.0) is not None:
                silent = False
    check("no encoder speed ever aborts the run (1.5 -> 550 rad/s)", silent)

    watch = SpeedWatch()
    watch.check(np.full(N_JOINTS, 3.0), zero, True, 6.0)
    check("peak still tracked for the status line",
          abs(watch.peek_peak() - 3.0) < 1e-9)
    check("peek_peak resets", abs(watch.peek_peak()) < 1e-9)
    watch.check(np.full(N_JOINTS, 9.0), zero, False, 6.0)
    check("peak ignores samples before EncoderVelocity is ready",
          abs(watch.peek_peak()) < 1e-9)

    print("[7] SafetyGate overspeed neutralised, other checks intact")
    gate = base.SafetyGate(tau_cap=1.0, qd_estop=np.inf, qd_estop_hard=np.inf)
    q_ok = Q_RECORDED_CROUCH.copy()
    ok_args = dict(temps=np.zeros(N_JOINTS, dtype=int),
                   miss_streaks=np.zeros(N_JOINTS, dtype=int),
                   errors={mid: 0 for mid in MOTOR_IDS})
    gate.start(0.0, q_ok)
    quiet = True
    for k in range(10):
        if gate.estop_reason(q_ok, np.full(N_JOINTS, 500.0), now=0.004 * k,
                             enforce_position_limits=False, **ok_args):
            quiet = False
    check("driver overspeed cannot trip at 500 rad/s", quiet)
    hot = dict(ok_args)
    hot["temps"] = np.full(N_JOINTS, 200, dtype=int)
    check("overtemp still trips",
          gate.estop_reason(q_ok, np.zeros(N_JOINTS), now=1.0,
                            enforce_position_limits=False, **hot) is not None)
    miss = dict(ok_args)
    miss["miss_streaks"] = np.full(N_JOINTS, 10_000, dtype=int)
    check("CAN miss estop still trips",
          gate.estop_reason(q_ok, np.zeros(N_JOINTS), now=1.0,
                            enforce_position_limits=False, **miss) is not None)
    bad = dict(ok_args)
    bad["errors"] = {mid: (0x02 if mid == MOTOR_IDS[0] else 0)
                     for mid in MOTOR_IDS}
    check("motor hard fault still trips",
          gate.estop_reason(q_ok, np.zeros(N_JOINTS), now=1.0,
                            enforce_position_limits=False, **bad) is not None)
    low, high = base.soft_limits()
    q_bad = q_ok.copy()
    q_bad[0] = high[0] + 1.0
    check("position limit trips only when --enforce-limits is on",
          gate.estop_reason(q_bad, np.zeros(N_JOINTS), now=1.0,
                            enforce_position_limits=False, **ok_args) is None
          and gate.estop_reason(q_bad, np.zeros(N_JOINTS), now=1.0,
                                enforce_position_limits=True, **ok_args)
          is not None)

    print()
    if failures:
        print(f"SELF-TEST FAILURES: {failures}")
        return 1
    print("SELF-TEST: all checks passed.")
    return 0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # imu_ekf_feed puts IMU_sensor/ on sys.path and re-exports DEFAULT_PORT;
    # keep it lazy so --self-test never touches pyserial.
    try:
        from imu_ekf_feed import DEFAULT_PORT            # noqa: E402
    except Exception:
        DEFAULT_PORT = "/dev/ttyUSB0"
    ap.add_argument("--port", default=DEFAULT_PORT, help="IMU serial port")
    ap.add_argument("--stand-height", type=float,
                    default=recorded.DEFAULT_STAND_HEIGHT,
                    help="target trunk height m (validated 0.14-0.21)")
    ap.add_argument("--t-rise", type=float, default=8.0,
                    help="rise duration s")
    ap.add_argument("--level-gain", type=float, default=LEVEL_GAIN_PER_S,
                    help="leveling integrator gain 1/s (0 disables; first "
                         "hardware run MUST use 0 per the runbook)")
    ap.add_argument("--level-clamp", type=float, default=LEVEL_CLAMP_M,
                    help="leveling offset clamp m")
    ap.add_argument("--level-in-rise", action="store_true",
                    help="also level during RISE/RISE_SETTLE (default: "
                         "HOLD only)")
    ap.add_argument("--crouch-max-speed-dps", type=float,
                    default=recorded.DEFAULT_CROUCH_MAX_MOTOR_DPS,
                    help="motor-side dps cap for CROUCH (100 = 10 out-deg/s)")
    ap.add_argument("--rise-max-speed-dps", type=float, default=100.0,
                    help="motor-side dps cap for RISE/PARK")
    ap.add_argument("--torque-trip", type=float, default=6.0,
                    help="measured-torque trip N*m (position_fault)")
    ap.add_argument("--enforce-limits", action="store_true",
                    help="also estop on a MEASURED pose outside the soft "
                         "position limits.  Off by default: the limits are "
                         "bounds against the encoder zeros, and a wrong zero "
                         "rejects a mechanically fine pose.  The commanded "
                         "path is validated against them offline either way.")
    ap.add_argument("--rise-contacts", choices=("planted", "airborne"),
                    default="planted",
                    help="EKF contact schedule during RISE/PARK. 'planted' "
                         "(default) is physically true here -- the feet never "
                         "slide -- and keeps height observable. 'airborne' "
                         "restores the old ekf_stand_hw schedule, which "
                         "dead-reckons r for the whole stream (+/-80 mm "
                         "close-out error on hardware); A/B only.")
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--no-ekf", action="store_true")
    ap.add_argument("--web", type=int, default=0,
                    help="EKF live dashboard port (0 = off)")
    ap.add_argument("--ekf-verbose", action="store_true")
    ap.add_argument("--log", default="", help="CSV log path")
    ap.add_argument("--log-hz", type=float, default=DEFAULT_LOG_HZ,
                    help="CSV rows/s (the sweep is 250 Hz; logging every "
                         "sweep puts 250 rows/s of file I/O in the CAN loop)")
    ap.add_argument("--raw-log", default="", help="NPZ raw log for hw_replay")
    ap.add_argument("--self-test", action="store_true",
                    help="offline checks, no hardware")
    args = ap.parse_args()

    if args.self_test:
        return offline_self_test(args)
    return run_hardware(args)


if __name__ == "__main__":
    sys.exit(main())
