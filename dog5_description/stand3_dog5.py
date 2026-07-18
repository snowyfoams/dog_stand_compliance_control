#!/usr/bin/env python3
r"""Three-leg stand for DOG5 in MuJoCo, mirroring the hardware position track.

Weekend sim proof for: *shift the body diagonally, unload one leg, lift it,
and hold a statically stable 3-leg stand* — using only what the real robot
has.  The controller is the position-command hardware pattern
(``stand_by_position_command.py``) plus the crawl gait's shift/unload/lift
gates (``crawl_dog5_hw.py``), both from the ``dog5-live-mirror`` branch:

* motion is native-position-command style only: joint targets + motor-side
  speed caps, no torque law, no dynamics model (kinematic control);
* feedback is encoder-only: quantized joint angles at ~20.8 Hz (one CAN
  round-robin sweep), velocity from the hardware ``EncoderVelocity`` filter,
  and per-joint measured torque (driver telemetry);
* MuJoCo is treated as the physical dog: the control side never reads any
  MuJoCo kinematics (no ``mj_jacSite``, ``site_xpos``, frames, or ``qvel``)
  — all FK/IK/Jacobians come from the NumPy ``dog5_kinematics`` module, and
  the only plant read is joint position (the encoder) inside the driver
  emulation;
* the trunk pose, contact forces, and true CoM are never read by the
  controller — they feed a separate *oracle* that judges the run.

Stages (auto-advance replaces the operator's Enter at each boundary).  The
stand-up is the hardware method: the robot starts placed in the recorded
crouch and CROUCH -> STAND are two native position moves, exactly like
``stand_by_position_command.py`` (``--start flat`` instead runs the sim-only
LIE -> ROLL -> CROUCH escape from the singular calibration pose):

    CROUCH -> STAND -> HOLD4
        -> SHIFT -> SHIFT_SETTLE       body 0.05 m diagonally away from FR
        -> UNLOAD -> UNLOAD_GATE       pre-lift 10 mm; refuse LIFT until the
                                       swing pitch/knee measure unloaded
        -> LIFT -> HOLD3               foot +40 mm, hold the 3-leg stand
        -> LOWER -> RELOAD -> RECENTER -> DONE

A failed gate aborts like the crawl does: lower, recenter, end ABORTED.

Run with the project venv:
    D:\mujoco\.venv\Scripts\python.exe stand3_dog5.py              # viewer
    D:\mujoco\.venv\Scripts\python.exe stand3_dog5.py --self-test  # offline checks
    D:\mujoco\.venv\Scripts\python.exe stand3_dog5.py --headless   # metrics + verdict
    D:\mujoco\.venv\Scripts\python.exe stand3_dog5.py --sweep      # robustness suite
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import dog5_kinematics  # noqa: E402  (hardware FK/Jacobian, no MuJoCo inside)

XML = str(Path(__file__).parent / "dog5.xml")
LEGS = ("FL", "FR", "RL", "RR")
N_JOINTS = 12
JOINT_LABELS = [f"{leg}_{j}" for leg in LEGS for j in ("abd", "pitch", "knee")]

# ---- constants copied from the hardware runners (dog5-live-mirror) ----
CONTROL_HZ = 250.0            # stand_dog5_hw.CONTROL_HZ; one motor per tick
GEAR = 10.0                   # speed field is motor-side; joint side = /GEAR
POSITION_POSE_TOL = 0.08      # stand_by_position_command settle gates
POSITION_QD_TOL = 0.25
POSITION_SETTLE_S = 0.50
WAIT_DWELL_S = 0.50
POSITION_TIMEOUT_S = 30.0
ABD_LIM, PITCH_LIM, KNEE_LIM = 1.75, 2.6, 2.6   # stand_dog5_hw soft limits
LIMIT_ESTOP_MARGIN = 0.05
QD_ESTOP = 7.0                # sustained, encoder-confirmed
QD_ESTOP_HARD = 8.0           # single-sample driver speed
QD_ESTOP_STREAK = 3
TAU_HARD = 9.0                # driver capability; 0xA4 has no commanded cap
MIN_LIFTOFF_MARGIN_M = 0.015  # crawl_dog5_hw liftoff gate
# The crawl's torque-mode unload fades feedforward with the foot planted and
# trips at 0.45 N*m.  In position mode the UNLOAD pre-lift takes the foot off
# the ground, so the unloaded swing pitch/knee still measure leg-gravity
# torque (~0.5 N*m); a loaded leg measures >= 0.9 N*m.  0.7 separates them.
DEFAULT_UNLOAD_TAU_TRIP_NM = 0.7
STATUS_HZ = 2.0

RECORDED_CROUCH_DEG = {       # stand_by_position_command.RECORDED_CROUCH_DEG
    "FL": (90.74, 47.59, -131.02),
    "FR": (-90.39, -51.92, 138.39),
    "RL": (90.23, -44.45, 128.40),
    "RR": (-93.95, 47.06, -125.12),
}
STAND_POSE_DEG = {            # stand_by_position_command.STAND_POSE_DEG
    "FL": (77.0, -8.0, -65.0),
    "FR": (-77.0, 8.0, 65.0),
    "RL": (85.0, 10.0, 65.0),
    "RR": (-85.0, -10.0, -65.0),
}

# ---- three-leg-stand tuning (sim; see SIM_APPROACH_HW.md section 6) ----
DEFAULT_SWING_LEG = "FR"
DEFAULT_SHIFT_M = 0.05        # crawl default is 0.03 for brief swings; the
                              # long static hold uses 0.05 -> ~51 mm FK margin
DEFAULT_LIFT_M = 0.04         # swing-foot raise (crawl swings use 0.025)
PRELIFT_M = 0.010             # UNLOAD pre-lift: exceeds elastic foot sag
T_SHIFT = 2.5                 # s, streamed IK phases
T_UNLOAD = 0.8
T_LIFT = 1.5
T_HOLD3 = 10.0
UNLOAD_GATE_TIMEOUT_S = 6.0
DEFAULT_TORQUE_TRIP_NM = 6.0  # hw default is 2.0; 3-leg stance holds ~4.3 N*m
                              # static on the loaded front hip pitch (see MD)

DEFAULT_CROUCH_DPS = 100.0    # motor-side caps, as on hardware
DEFAULT_STAND_DPS = 100.0
DEFAULT_STREAM_DPS = 250.0

# ---- servo emulation (identify on hardware, SIM_APPROACH_HW.md 2.1) ----
KP_SERVO = 120.0              # N*m/rad   guessed 0xA4 internal position loop
KD_SERVO = 2.5                # N*m*s/rad
ENCODER_QUANTUM_RAD = 0.0005

SWEEPS_PER_STATE = N_JOINTS   # gates/trips run once per round-robin sweep


def _stack_pose(pose_deg):
    return np.deg2rad(
        np.concatenate([np.asarray(pose_deg[leg], dtype=float) for leg in LEGS])
    )


Q_ZERO = np.zeros(N_JOINTS)
Q_CROUCH = _stack_pose(RECORDED_CROUCH_DEG)
Q_STAND = _stack_pose(STAND_POSE_DEG)
Q_ROLL = Q_CROUCH * np.tile([1.0, 0.0, 0.0], 4)   # abd to crouch values first


def soft_limits():
    low = np.tile([-ABD_LIM, -PITCH_LIM, -KNEE_LIM], 4)
    high = np.tile([+ABD_LIM, +PITCH_LIM, +KNEE_LIM], 4)
    return low, high


def support_triangle_margin(points_xy, com_xy=(0.0, 0.0)):
    """Signed distance from ``com_xy`` to the triangle (crawl_dog5_hw copy)."""
    points = np.asarray(points_xy, dtype=float)
    point = np.asarray(com_xy, dtype=float)
    if points.shape != (3, 2):
        raise ValueError(f"support triangle must have shape (3,2), got {points.shape}")
    if point.shape != (2,) or not np.all(np.isfinite(points)):
        raise ValueError("support geometry must be finite")
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    ordered = points[np.argsort(angles)]
    margins = []
    for start, end in zip(ordered, np.roll(ordered, -1, axis=0)):
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length < 1.0e-6:
            raise ValueError("support triangle is degenerate")
        relative = point - start
        cross_z = edge[0] * relative[1] - edge[1] * relative[0]
        margins.append(float(cross_z / length))
    return min(margins)


def _ik_to_target(leg, q_start, target):
    """Damped-least-squares IK (crawl_dog5_hw copy)."""
    q = np.asarray(q_start, dtype=float).copy()
    for _ in range(100):
        error = np.asarray(target) - dog5_kinematics.foot_position(leg, q)
        jacobian = dog5_kinematics.foot_jacobian(leg, q)
        q += 0.35 * jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 1.0e-5 * np.eye(3), error
        )
    return q


class EncoderVelocity:
    """Low-pass velocity from unwrapped angles (stand_by_position_command copy)."""

    def __init__(self, alpha=0.35):
        self.alpha = float(alpha)
        self.previous_q = None
        self.previous_time = None
        self.value = np.zeros(N_JOINTS)
        self.samples = 0

    def update(self, q, now):
        q = np.asarray(q, dtype=float)
        if self.previous_q is not None:
            dt = float(now) - self.previous_time
            if np.isfinite(dt) and 1.0e-4 < dt < 0.25:
                raw = (q - self.previous_q) / dt
                self.value = self.alpha * raw + (1.0 - self.alpha) * self.value
                self.samples += 1
            else:
                self.value.fill(0.0)
                self.samples = 0
        self.previous_q = q.copy()
        self.previous_time = float(now)
        return self.value.copy()

    @property
    def ready(self):
        return self.samples >= 5


class NativePositionServo:
    """Per-motor 0xA4 emulation: speed-capped setpoint slew + stiff PD.

    This is the only control-side object that touches the plant, and the
    only thing it reads is joint position — the physical encoder.  It
    servos on its own quantized encoder value and derives its own speed
    estimate from encoder differences (as the real driver does); MuJoCo's
    qvel and every MuJoCo kinematic quantity stay unread.  Runs at the
    physics rate (the real driver loop is faster still).  The torque clip
    is the driver's capability, not a commandable cap — the native
    position command has no torque field.
    """

    SPEED_ALPHA = 0.4     # driver-internal speed filter at the physics rate

    def __init__(self, q0, kp=KP_SERVO, kd=KD_SERVO, zero_error=None):
        self.setpoint = np.asarray(q0, dtype=float).copy()
        self.target = self.setpoint.copy()
        self.joint_speed = np.full(N_JOINTS, np.deg2rad(DEFAULT_CROUCH_DPS) / GEAR)
        self.kp = float(kp)
        self.kd = float(kd)
        self.tau = np.zeros(N_JOINTS)
        # Calibration zero error: the encoder frame is shifted, so everything
        # the driver measures AND servos on is offset — the plant physically
        # realizes (command - zero_error), exactly like a mis-zeroed motor.
        self.zero_error = np.zeros(N_JOINTS) if zero_error is None \
            else np.asarray(zero_error, dtype=float).copy()
        self.q_enc = self._quantize(self.setpoint)
        self.qd_est = np.zeros(N_JOINTS)

    @staticmethod
    def _quantize(q):
        return np.round(np.asarray(q, dtype=float) / ENCODER_QUANTUM_RAD) \
            * ENCODER_QUANTUM_RAD

    def command(self, index, target_rad, motor_dps):
        """One CAN slot: position + speed for a single motor."""
        self.target[index] = float(target_rad)
        self.joint_speed[index] = np.deg2rad(float(motor_dps)) / GEAR

    def step(self, qpos_joint, dt):
        q_enc = self._quantize(qpos_joint + self.zero_error)
        raw_speed = (q_enc - self.q_enc) / dt
        self.qd_est = self.SPEED_ALPHA * raw_speed \
            + (1.0 - self.SPEED_ALPHA) * self.qd_est
        self.q_enc = q_enc
        step_limit = self.joint_speed * dt
        self.setpoint += np.clip(self.target - self.setpoint, -step_limit, step_limit)
        tau = self.kp * (self.setpoint - q_enc) - self.kd * self.qd_est
        self.tau = np.clip(tau, -TAU_HARD, TAU_HARD)
        return self.tau

    def telemetry(self):
        """What the CAN bus reports: position, driver speed, torque."""
        return self.q_enc.copy(), self.qd_est.copy(), self.tau.copy()


class ThreeLegPlan:
    """Kinematic foot-target planner: neutral -> shifted -> prelift -> lifted.

    Anchors are the FK feet of the commanded stand pose (crawl convention:
    fixed conceptual ground anchors; body motion moves every stance target
    the opposite way in the trunk frame, so stance-foot slip is never
    commanded).
    """

    def __init__(self, swing_leg, shift_m, lift_m,
                 prelift_m=PRELIFT_M, lift_vec=(0.0, 0.0, 1.0)):
        if swing_leg not in LEGS:
            raise ValueError(f"swing leg must be one of {LEGS}")
        self.swing = swing_leg
        self.stance = [leg for leg in LEGS if leg != swing_leg]
        self.anchor = {
            leg: dog5_kinematics.foot_position(leg, Q_STAND[self._sl(leg)])
            for leg in LEGS
        }
        signs = np.sign(self.anchor[swing_leg][:2])
        self.body_shift = -float(shift_m) * signs / np.sqrt(2.0)
        self.lift_m = float(lift_m)
        self.prelift_m = float(prelift_m)
        self.lift_vec = np.asarray(lift_vec, dtype=float)

    @staticmethod
    def _sl(leg):
        i = LEGS.index(leg)
        return slice(3 * i, 3 * i + 3)

    def foot_targets(self, body_frac, prelift_frac, lift_frac):
        """Trunk-frame foot targets for interpolation fractions in [0, 1]."""
        body_xy = body_frac * self.body_shift
        targets = {}
        for leg in LEGS:
            target = self.anchor[leg].copy()
            target[:2] -= body_xy
            if leg == self.swing:
                target[2] += prelift_frac * self.prelift_m
                target += lift_frac * self.lift_m * self.lift_vec
            targets[leg] = target
        return targets

    def q_targets(self, warm_q, body_frac, prelift_frac, lift_frac):
        """IK the current foot targets, warm-started per leg."""
        targets = self.foot_targets(body_frac, prelift_frac, lift_frac)
        q = np.asarray(warm_q, dtype=float).copy()
        for leg in LEGS:
            sl = self._sl(leg)
            q[sl] = _ik_to_target(leg, q[sl], targets[leg])
        return q

    def fk_margin(self, q_measured, use_shifted_body=True):
        """Support margin from measured encoder FK, CoM assumed at origin."""
        feet = {
            leg: dog5_kinematics.foot_position(leg, q_measured[self._sl(leg)])
            for leg in self.stance
        }
        return support_triangle_margin([feet[leg][:2] for leg in self.stance])


# stage -> (kind, duration or None). Streamed stages interpolate targets;
# gated stages hold targets and wait for their condition.
THREE_LEG_STAGES = [
    ("HOLD4", "dwell", 1.0),
    ("SHIFT", "stream", T_SHIFT),
    ("SHIFT_SETTLE", "gate", None),
    ("UNLOAD", "stream", T_UNLOAD),
    ("UNLOAD_GATE", "gate", None),
    ("LIFT", "stream", T_LIFT),
    ("HOLD3", "dwell", T_HOLD3),
    ("LOWER", "stream", T_LIFT),
    ("RELOAD", "dwell", 1.0),
    ("RECENTER", "stream", T_SHIFT),
    ("DONE", "dwell", 1.5),
]


def make_stages(start):
    """Stand-up flow + 3-leg sequence.

    ``crouch`` is the hardware method (stand_by_position_command.py): the
    robot is placed in the crouch, then CROUCH -> STAND are two native
    position moves.  ``flat`` is the sim-only escape from the calibration
    pose (needs ROLL first because that pose is rank-1 singular).
    """
    if start == "crouch":
        stand_up = [("CROUCH", "gate", None), ("STAND", "gate", None)]
    elif start == "flat":
        stand_up = [("LIE", "gate", None), ("ROLL", "gate", None),
                    ("CROUCH", "gate", None), ("STAND", "gate", None)]
    else:
        raise ValueError(f"unknown start {start!r}")
    return stand_up + THREE_LEG_STAGES
# interpolation fractions (body, prelift, lift) at the END of each stage
STAGE_FRACS = {
    "LIE": (0.0, 0.0, 0.0), "ROLL": (0.0, 0.0, 0.0),
    "CROUCH": (0.0, 0.0, 0.0), "STAND": (0.0, 0.0, 0.0),
    "HOLD4": (0.0, 0.0, 0.0),
    "SHIFT": (1.0, 0.0, 0.0), "SHIFT_SETTLE": (1.0, 0.0, 0.0),
    "UNLOAD": (1.0, 1.0, 0.0), "UNLOAD_GATE": (1.0, 1.0, 0.0),
    "LIFT": (1.0, 1.0, 1.0), "HOLD3": (1.0, 1.0, 1.0),
    "LOWER": (1.0, 1.0, 0.0), "RELOAD": (1.0, 1.0, 0.0),
    "RECENTER": (0.0, 0.0, 0.0), "DONE": (0.0, 0.0, 0.0),
}
JOINT_STAGE_TARGETS = {"LIE": Q_ZERO, "ROLL": Q_ROLL, "CROUCH": Q_CROUCH,
                       "STAND": Q_STAND}


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return 3.0 * u * u - 2.0 * u * u * u


class Stand3Controller:
    """Encoder-only stage machine streaming position commands.

    Sees: quantized q at ~20.8 Hz, EncoderVelocity, measured torque.
    Never sees: trunk pose, contacts, true CoM, MuJoCo internals.
    """

    def __init__(self, plan, torque_trip, unload_trip,
                 crouch_dps, stand_dps, stream_dps, hold3_s=T_HOLD3,
                 start="crouch"):
        self.plan = plan
        self.torque_trip = float(torque_trip)
        self.unload_trip = float(unload_trip)
        self.hold3_s = float(hold3_s)
        self.caps = {"crouch": crouch_dps, "stand": stand_dps,
                     "stream": stream_dps}
        self.stages = make_stages(start)
        self.stage_index = {name: i for i, (name, _, _) in enumerate(self.stages)}
        self.stage_i = 0
        self.stage_started = 0.0
        self.settle_since = None
        self.dwell_since = None
        self.velocity = EncoderVelocity()
        self.qd = np.zeros(N_JOINTS)
        self.q_cmd = Q_ZERO.copy()          # last commanded joint targets
        self.stop_reason = None
        self.aborted = None
        self.qd_streak = 0
        self.events = []

    @property
    def stage(self):
        return self.stages[self.stage_i][0]

    def _cap_for_stage(self):
        if self.stage in ("LIE", "ROLL", "CROUCH"):
            return self.caps["crouch"]
        if self.stage == "STAND":
            return self.caps["stand"]
        return self.caps["stream"]

    def _fracs_now(self, now):
        """Interpolation fractions during/after the current stage."""
        name, kind, duration = self.stages[self.stage_i]
        end = np.asarray(STAGE_FRACS[name], dtype=float)
        if kind != "stream":
            return end
        prev = np.asarray(
            STAGE_FRACS[self.stages[self.stage_i - 1][0]], dtype=float
        )
        s = _smoothstep((now - self.stage_started) / duration)
        return prev + s * (end - prev)

    def _advance(self, now, note=""):
        self.stage_i += 1
        self.stage_started = now
        self.settle_since = None
        self.dwell_since = None
        self.events.append((now, self.stage, note))

    def _abort(self, now, reason):
        """Crawl-style graceful abort: lower if airborne, recenter, end."""
        self.aborted = reason
        self.events.append((now, "ABORT", reason))
        self.stage_i = self.stage_index["LOWER"] if self.stage in ("LIFT", "HOLD3") \
            else self.stage_index["RECENTER"]
        self.stage_started = now
        self.settle_since = None

    def _settled(self, now, q_enc):
        error = float(np.max(np.abs(q_enc - self.q_cmd)))
        speed = float(np.max(np.abs(self.qd))) if self.velocity.ready else np.inf
        if error > POSITION_POSE_TOL or speed > POSITION_QD_TOL:
            self.settle_since = None
            return False
        if self.settle_since is None:
            self.settle_since = now
            return False
        return now - self.settle_since >= POSITION_SETTLE_S

    def _check_trips(self, q_enc, qd_reported, tau_measured, now):
        """Mirrors the hardware SafetyGate trips that apply to position mode."""
        low, high = soft_limits()
        if self.stage not in ("LIE", "ROLL", "CROUCH"):
            outside = (q_enc < low - LIMIT_ESTOP_MARGIN) | \
                      (q_enc > high + LIMIT_ESTOP_MARGIN)
            if np.any(outside):
                j = int(np.flatnonzero(outside)[0])
                return f"soft-limit e-stop: {JOINT_LABELS[j]}={q_enc[j]:+.2f} rad"
        if np.any(np.abs(qd_reported) > QD_ESTOP_HARD):
            j = int(np.argmax(np.abs(qd_reported)))
            return (f"hard overspeed: {JOINT_LABELS[j]}="
                    f"{qd_reported[j]:+.1f} rad/s > {QD_ESTOP_HARD}")
        if self.velocity.ready and np.any(np.abs(self.qd) > QD_ESTOP):
            self.qd_streak += 1
            if self.qd_streak >= QD_ESTOP_STREAK:
                j = int(np.argmax(np.abs(self.qd)))
                return (f"sustained overspeed: {JOINT_LABELS[j]}="
                        f"{self.qd[j]:+.1f} rad/s > {QD_ESTOP}")
        else:
            self.qd_streak = 0
        if np.any(np.abs(tau_measured) > self.torque_trip):
            j = int(np.argmax(np.abs(tau_measured)))
            return (f"measured-torque trip: {JOINT_LABELS[j]}="
                    f"{tau_measured[j]:+.2f} N*m > {self.torque_trip:.2f}")
        return None

    def sweep(self, now, q_enc, qd_reported, tau_measured):
        """One per-round-robin update: state estimate, gates, target refresh.

        Returns the 12 joint targets to stream this sweep (rad).
        """
        self.qd = self.velocity.update(q_enc, now)
        self.stop_reason = self._check_trips(q_enc, qd_reported, tau_measured, now)
        if self.stop_reason:
            return self.q_cmd

        name, kind, duration = self.stages[self.stage_i]

        if name in JOINT_STAGE_TARGETS:
            self.q_cmd = JOINT_STAGE_TARGETS[name].copy()
            if self._settled(now, q_enc):
                self._advance(now, "settled")
            elif now - self.stage_started > POSITION_TIMEOUT_S:
                self.stop_reason = f"{name} timed out after {POSITION_TIMEOUT_S:.0f} s"
        elif kind == "stream":
            fracs = self._fracs_now(now)
            self.q_cmd = self.plan.q_targets(self.q_cmd, *fracs)
            if now - self.stage_started >= duration:
                self._advance(now, "stream complete")
        elif name == "SHIFT_SETTLE":
            if self._settled(now, q_enc):
                margin = self.plan.fk_margin(q_enc)
                if margin >= MIN_LIFTOFF_MARGIN_M:
                    self._advance(now, f"FK margin {1000 * margin:.1f} mm")
                elif now - self.stage_started > POSITION_TIMEOUT_S:
                    self._abort(now, f"liftoff margin {1000 * margin:.1f} mm "
                                     f"< {1000 * MIN_LIFTOFF_MARGIN_M:.0f} mm")
            elif now - self.stage_started > POSITION_TIMEOUT_S:
                self._abort(now, "SHIFT did not settle")
        elif name == "UNLOAD_GATE":
            sl = self.plan._sl(self.plan.swing)
            swing_tau = np.abs(tau_measured[sl][1:])   # pitch, knee
            if self._settled(now, q_enc) and np.all(swing_tau <= self.unload_trip):
                self._advance(now, f"swing unloaded "
                                   f"max|tau|={float(np.max(swing_tau)):.2f} N*m")
            elif now - self.stage_started > UNLOAD_GATE_TIMEOUT_S:
                self._abort(now, f"unload gate: swing max|tau|="
                                 f"{float(np.max(swing_tau)):.2f} N*m "
                                 f"> {self.unload_trip:.2f}")
        elif kind == "dwell":
            if name == "HOLD3":
                duration = self.hold3_s
            if self.dwell_since is None:
                self.dwell_since = now
            if now - self.dwell_since >= duration:
                if name == "DONE":
                    self.stop_reason = "sequence complete"
                else:
                    self._advance(now, "dwell complete")
        return self.q_cmd


class Oracle:
    """Privileged sim measurements — judge the run, never steer it.

    This is external instrumentation (a motion-capture rig and force plate
    around the "real dog"), and the only place MuJoCo kinematics or
    contacts are read.  Nothing here feeds back into the controller.
    """

    def __init__(self, model, data, plan):
        import mujoco
        self._mujoco = mujoco
        self.m, self.d = model, data
        self.plan = plan
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.foot_geom = {leg: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}") for leg in LEGS}
        self.swing_geom = self.foot_geom[plan.swing]
        self.site = {leg: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"foot_{leg}") for leg in LEGS}
        self.reset()

    def reset(self):
        self.hold3 = dict(max_swing_force=0.0, min_clearance=np.inf,
                          max_tilt_deg=0.0, min_true_margin=np.inf,
                          min_fk_margin=np.inf, samples=0)
        self.max_abs_tau = np.zeros(N_JOINTS)

    def foot_contact(self, leg):
        """Return (normal, tangential) contact force magnitude on one foot."""
        geom = self.foot_geom[leg]
        normal = 0.0
        tangential = 0.0
        force = np.zeros(6)
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            if geom in (con.geom1, con.geom2):
                self._mujoco.mj_contactForce(self.m, self.d, i, force)
                normal += abs(force[0])
                tangential += float(np.hypot(force[1], force[2]))
        return normal, tangential

    def swing_contact_force(self):
        return self.foot_contact(self.plan.swing)[0]

    def tilt_deg(self):
        R = self.d.xmat[self.trunk].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))

    def true_margin(self):
        com = self.d.subtree_com[self.trunk][:2]
        feet = [self.d.site_xpos[self.site[leg]][:2] for leg in self.plan.stance]
        try:
            return support_triangle_margin(feet, com_xy=com)
        except ValueError:
            return float("nan")

    def sample(self, stage, q_enc, tau):
        self.max_abs_tau = np.maximum(self.max_abs_tau, np.abs(tau))
        if stage != "HOLD3":
            return
        h = self.hold3
        h["max_swing_force"] = max(h["max_swing_force"], self.swing_contact_force())
        clearance = float(
            self.d.site_xpos[self.site[self.plan.swing]][2]) - 0.02
        h["min_clearance"] = min(h["min_clearance"], clearance)
        h["max_tilt_deg"] = max(h["max_tilt_deg"], self.tilt_deg())
        h["min_true_margin"] = min(h["min_true_margin"], self.true_margin())
        h["min_fk_margin"] = min(h["min_fk_margin"], self.plan.fk_margin(q_enc))
        h["samples"] += 1

    def verdict(self, controller):
        h = self.hold3
        checks = [
            ("finished (no abort/e-stop)",
             controller.aborted is None
             and controller.stop_reason == "sequence complete"),
            ("HOLD3 sampled", h["samples"] > 0),
            ("swing foot airborne (force <= 0.05 N)",
             h["max_swing_force"] <= 0.05),
            ("swing clearance >= 10 mm", h["min_clearance"] >= 0.010),
            ("trunk tilt <= 5 deg", h["max_tilt_deg"] <= 5.0),
            ("true CoM margin >= 15 mm", h["min_true_margin"] >= 0.015),
            ("encoder FK margin >= 15 mm", h["min_fk_margin"] >= 0.015),
        ]
        return all(ok for _, ok in checks), checks


def build_sim(mass_scale=1.0, com_offset_y=0.0, friction=None, kp_scale=1.0,
              start="crouch"):
    import mujoco
    model = mujoco.MjModel.from_xml_path(XML)
    if mass_scale != 1.0:
        model.body_mass[:] *= mass_scale
        model.body_inertia[:] *= mass_scale
    if com_offset_y:
        trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        model.body_ipos[trunk][1] += com_offset_y
    if friction is not None:
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        model.geom_friction[floor][0] = friction
        for leg in LEGS:
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}")
            model.geom_friction[g][0] = friction
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    if start == "crouch":
        # The hardware method: the robot is placed in the recorded crouch
        # before READ.  Rest the feet on the floor (NumPy FK, not MuJoCo).
        data.qpos[7:] = Q_CROUCH
        foot_z = min(
            dog5_kinematics.foot_position(
                leg, Q_CROUCH[3 * i:3 * i + 3])[2]
            for i, leg in enumerate(LEGS))
        data.qpos[2] = 0.02 - foot_z + 0.002   # foot sphere radius + drop gap
    mujoco.mj_forward(model, data)
    return model, data


def run(model, data, args, viewer=None, frame_hook=None, quiet=False):
    """Drive one full sequence; returns (controller, oracle)."""
    import mujoco
    plan = ThreeLegPlan(
        args.swing_leg, args.shift, args.lift,
        prelift_m=getattr(args, "prelift", PRELIFT_M),
        lift_vec=getattr(args, "lift_vec", (0.0, 0.0, 1.0)))
    controller = Stand3Controller(
        plan, args.torque_trip, args.unload_trip,
        args.crouch_dps, args.stand_dps, args.stream_dps,
        hold3_s=getattr(args, "hold3", T_HOLD3),
        start=getattr(args, "start", "crouch"))
    oracle = Oracle(model, data, plan)

    # Address wiring only (like the CAN ID map) — never used for kinematics.
    qadr = np.array([model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{j}_{leg}")]
        for leg in LEGS for j in ("hip_abd", "hip_pitch", "knee")])
    assert qadr[0] == 7   # canonical joint order follows the freejoint

    # Arm the drivers at the measured pose (in the encoders' calibrated
    # frame), exactly as the hardware runner does before READ.
    zero_error = getattr(args, "zero_error", None)
    zero_error = np.zeros(N_JOINTS) if zero_error is None \
        else np.asarray(zero_error, dtype=float)
    servo = NativePositionServo(
        data.qpos[qadr] + zero_error, kp=KP_SERVO * args.kp_scale,
        zero_error=zero_error)

    steps_per_tick = max(1, round(1.0 / (CONTROL_HZ * model.opt.timestep)))
    slot = 0
    q_targets = Q_ZERO.copy()
    last_print = -np.inf
    t_wall0 = time.perf_counter()

    while controller.stop_reason is None:
        now = data.time
        if slot % SWEEPS_PER_STATE == 0:
            # The controller sees only CAN telemetry; every kinematic
            # quantity it uses comes from dog5_kinematics (NumPy).
            q_enc, qd_reported, tau_measured = servo.telemetry()
            q_targets = controller.sweep(now, q_enc, qd_reported, tau_measured)
            oracle.sample(controller.stage, q_enc, tau_measured)
            if not quiet and now - last_print >= 1.0 / STATUS_HZ:
                sl = plan._sl(plan.swing)
                print(f"t={now:6.2f}  {controller.stage:<12} "
                      f"tilt={oracle.tilt_deg():4.1f} deg  "
                      f"swingF={oracle.swing_contact_force():5.1f} N  "
                      f"margin={1000 * oracle.true_margin():+6.1f} mm  "
                      f"swing|tau|={np.max(np.abs(tau_measured[sl][1:])):4.2f}",
                      flush=True)
                last_print = now
        motor = slot % N_JOINTS
        servo.command(motor, q_targets[motor], controller._cap_for_stage())
        for _ in range(steps_per_tick):
            data.ctrl[:] = servo.step(data.qpos[qadr], model.opt.timestep)
            mujoco.mj_step(model, data)
        if frame_hook is not None:
            frame_hook(now, controller.stage, oracle)
        if viewer is not None:
            if not viewer.is_running():
                controller.stop_reason = "viewer closed"
                break
            viewer.sync()
            lag = data.time - (time.perf_counter() - t_wall0)
            if lag > 0:
                time.sleep(lag)
            elif lag < -0.5:
                t_wall0 = time.perf_counter() - data.time
        slot += 1
        if data.time > 180.0:
            controller.stop_reason = "wall timeout"
            break
    return controller, oracle


def report(controller, oracle, quiet=False):
    passed, checks = oracle.verdict(controller)
    if not quiet:
        print(f"\nstop: {controller.stop_reason}"
              + (f"  (ABORTED: {controller.aborted})" if controller.aborted else ""))
        for now, stage, note in controller.events:
            print(f"  t={now:6.2f}  -> {stage:<13} {note}")
        h = oracle.hold3
        print(f"\nHOLD3 ({h['samples']} samples):")
        print(f"  swing contact force max : {h['max_swing_force']:.3f} N")
        print(f"  swing clearance min     : {1000 * h['min_clearance']:.1f} mm")
        print(f"  trunk tilt max          : {h['max_tilt_deg']:.2f} deg")
        print(f"  true CoM margin min     : {1000 * h['min_true_margin']:.1f} mm")
        print(f"  encoder FK margin min   : {1000 * h['min_fk_margin']:.1f} mm")
        worst = int(np.argmax(oracle.max_abs_tau))
        print(f"  peak measured torque    : {JOINT_LABELS[worst]} "
              f"{oracle.max_abs_tau[worst]:.2f} N*m "
              f"(hw --position-torque-trip must exceed this)")
        print("\nverdict:")
        for label, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        print(f"\n{'PASS' if passed else 'FAIL'}: 3-leg stand "
              f"({'proven' if passed else 'not proven'} under hw constraints)")
    return passed


def self_test(args):
    """Offline validation: IK reachability, limits, margins, static torque."""
    plan = ThreeLegPlan(args.swing_leg, args.shift, args.lift)
    low, high = soft_limits()
    q = Q_STAND.copy()
    worst_err = 0.0
    for body, pre, lift in [(f, 0, 0) for f in np.linspace(0, 1, 11)] + \
                           [(1, f, 0) for f in np.linspace(0, 1, 5)] + \
                           [(1, 1, f) for f in np.linspace(0, 1, 11)]:
        q = plan.q_targets(q, body, pre, lift)
        targets = plan.foot_targets(body, pre, lift)
        for leg in LEGS:
            sl = plan._sl(leg)
            err = float(np.linalg.norm(
                dog5_kinematics.foot_position(leg, q[sl]) - targets[leg]))
            worst_err = max(worst_err, err)
            if np.any(q[sl] < low[sl]) or np.any(q[sl] > high[sl]):
                raise SystemExit(f"FAIL: {leg} IK outside soft limits at "
                                 f"({body:.2f},{pre:.2f},{lift:.2f}): {q[sl]}")
    if worst_err > 1.0e-4:
        raise SystemExit(f"FAIL: IK residual {worst_err:.2e} m > 1e-4")
    print(f"IK path reachable, in-limits; worst residual {worst_err:.2e} m")

    margin = support_triangle_margin(
        [plan.anchor[leg][:2] - plan.body_shift for leg in plan.stance])
    if margin < 2 * MIN_LIFTOFF_MARGIN_M:
        raise SystemExit(f"FAIL: planned liftoff margin {1000 * margin:.1f} mm "
                         f"< {2000 * MIN_LIFTOFF_MARGIN_M:.0f} mm reserve")
    print(f"planned liftoff FK margin {1000 * margin:.1f} mm "
          f"(gate {1000 * MIN_LIFTOFF_MARGIN_M:.0f} mm)")

    # static stance torques at the shifted pose (barycentric weight shares)
    weight = 5.815 * 9.81
    stance_xy = np.array([plan.anchor[leg][:2] for leg in plan.stance])
    A = np.vstack([np.ones(3), stance_xy.T])
    forces = np.linalg.solve(A, np.array([weight, *(weight * plan.body_shift)]))
    worst_tau = 0.0
    for leg, fz in zip(plan.stance, forces):
        sl = plan._sl(leg)
        tau = dog5_kinematics.foot_jacobian(leg, q[sl]).T @ [0.0, 0.0, fz]
        worst_tau = max(worst_tau, float(np.max(np.abs(tau))))
    if worst_tau > 0.8 * TAU_HARD:
        raise SystemExit(f"FAIL: static stance torque {worst_tau:.1f} N*m "
                         f"too close to driver capability {TAU_HARD} N*m")
    print(f"static stance forces {np.round(forces, 1)} N, "
          f"worst joint torque {worst_tau:.2f} N*m "
          f"(driver capability {TAU_HARD} N*m; set hw torque trip above this)")

    assert support_triangle_margin([(1, 0), (0, 1), (-1, -1)]) > 0
    assert support_triangle_margin([(1, 0), (0, 1), (-1, -1)], (2, 2)) < 0
    velocity = EncoderVelocity()
    for i in range(7):
        v = velocity.update(Q_STAND, 0.05 * i)
    assert velocity.ready and np.allclose(v, 0.0)
    print("stand3_dog5 offline self-test PASS")
    return 0


SWEEP_CASES = [
    dict(label="nominal"),
    dict(label="mu=0.5", friction=0.5),
    dict(label="mu=0.8", friction=0.8),
    dict(label="mass+10%", mass_scale=1.10),
    dict(label="mass-10%", mass_scale=0.90),
    dict(label="com+10mm y", com_offset_y=+0.010),
    dict(label="com-10mm y", com_offset_y=-0.010),
    dict(label="servo kp+50%", kp_scale=1.5),
    dict(label="servo kp-50%", kp_scale=0.5),
    dict(label="mu=0.5 mass+10% com-10mm", friction=0.5, mass_scale=1.10,
         com_offset_y=-0.010),
]


def sweep(args):
    results = []
    for case in SWEEP_CASES:
        case = dict(case)
        label = case.pop("label")
        kp_scale = case.pop("kp_scale", 1.0)
        model, data = build_sim(**case, kp_scale=kp_scale,
                                start=getattr(args, "start", "crouch"))
        case_args = argparse.Namespace(**vars(args))
        case_args.kp_scale = kp_scale
        controller, oracle = run(model, data, case_args, quiet=True)
        passed = report(controller, oracle, quiet=True)
        h = oracle.hold3
        detail = (f"tilt {h['max_tilt_deg']:.1f} deg, "
                  f"margin {1000 * h['min_true_margin']:.0f} mm, "
                  f"clear {1000 * h['min_clearance']:.0f} mm"
                  if h["samples"] else
                  f"stopped: {controller.aborted or controller.stop_reason}")
        results.append((label, passed, detail))
        print(f"[{'PASS' if passed else 'FAIL'}] {label:<28} {detail}",
              flush=True)
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\nsweep: {n_pass}/{len(results)} PASS")
    return 0 if n_pass == len(results) else 1


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--start", default="crouch", choices=("crouch", "flat"),
                        help="crouch = hardware method (placed in crouch, "
                             "CROUCH->STAND native moves); flat = sim-only "
                             "stand-up from the calibration pose")
    parser.add_argument("--swing-leg", default=DEFAULT_SWING_LEG, choices=LEGS)
    parser.add_argument("--shift", type=float, default=DEFAULT_SHIFT_M)
    parser.add_argument("--lift", type=float, default=DEFAULT_LIFT_M)
    parser.add_argument("--torque-trip", type=float,
                        default=DEFAULT_TORQUE_TRIP_NM,
                        help="measured-torque feedback trip, N*m "
                             "(hw default 2.0 is too low for 3-leg stance)")
    parser.add_argument("--unload-trip", type=float,
                        default=DEFAULT_UNLOAD_TAU_TRIP_NM)
    parser.add_argument("--crouch-dps", type=float, default=DEFAULT_CROUCH_DPS)
    parser.add_argument("--stand-dps", type=float, default=DEFAULT_STAND_DPS)
    parser.add_argument("--stream-dps", type=float, default=DEFAULT_STREAM_DPS)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    args = parser.parse_args()

    if args.self_test:
        return self_test(args)
    if args.sweep:
        return sweep(args)

    model, data = build_sim(kp_scale=args.kp_scale, start=args.start)
    if args.headless:
        controller, oracle = run(model, data, args)
        return 0 if report(controller, oracle) else 1

    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        controller, oracle = run(model, data, args, viewer=viewer)
    return 0 if report(controller, oracle) else 1


if __name__ == "__main__":
    raise SystemExit(main())
