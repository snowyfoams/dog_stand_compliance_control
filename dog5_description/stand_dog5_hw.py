#!/usr/bin/env python3
"""Operator-gated DOG5 compliance stand controller for real CAN hardware.

The robot must be securely supported.  The operator advances the sequence with
Enter; X stops immediately:

  ZERO TORQUE CHECK -- inspect live calibrated angles, then Enter starts REST
  REST              -- smooth joint PD moves every joint to calibrated zero
  WAIT REST         -- holds all joints at zero; Enter starts ROLL when settled
  ROLL              -- joint PD moves to the roll pose, then waits
  WAIT ROLL         -- holds the roll pose; Enter starts FOLD when settled
  FOLD              -- joint PD moves to the crouch pose, then waits
  WAIT FOLD         -- holds the crouch; Enter starts Cartesian STAND when settled
  STAND             -- NumPy FK/Jacobian compliance ramps to standing height
  HOLD              -- holds the final Cartesian target until X/Ctrl-C

The calibrated encoder contract is fixed and contains no software zero offset
and no gearbox division::

    motoroutput_deg = raw_encoder * ENCODER_GAIN
    joint_angle_rad = radians(direction * motoroutput_deg)

Directions were confirmed on hardware: CAN 1,3,4,6,9,12 are -1; all other
motors are +1.  MuJoCo is not used for hardware kinematics or dynamics here.

First supported test:

    python stand_dog5_hw.py --tau-max 1.0

The staged runner intentionally limits --tau-max to 3 N*m.  Raise gains or
limits only after reviewing logs from the lower setting.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

import dog5_kinematics
from dog5_hardware_map import HARDWARE_JOINTS
import motorbus
import stand_dog5

LEGS = ("FL", "FR", "RL", "RR")
MOTOR_IDS = [joint.can_id for joint in HARDWARE_JOINTS]
MOTOR_DIRECTIONS = {joint.can_id: joint.direction for joint in HARDWARE_JOINTS}
JOINT_LABELS = [f"{joint.leg}_{joint.joint}" for joint in HARDWARE_JOINTS]
N_JOINTS = len(HARDWARE_JOINTS)
JOINT_DIRECTIONS = np.asarray(
    [joint.direction for joint in HARDWARE_JOINTS], dtype=float
)

# Loop and conservative first-test limits.
CONTROL_HZ = 250.0
DEFAULT_TAU_MAX = 1.0
STAGED_TAU_MAX = 3.0
TAU_HARD = 9.0
TORQUE_RAMP_S = 1.0
TAU_SLEW_NM_S = 5.0
DEFAULT_CART_GAIN_SCALE = 0.7
DEFAULT_SUPPORT_SCALE = 0.25
T_REST = 2.0
T_ROLL_HW = 4.0
T_FOLD_HW = 5.0
T_STAND_HW = 6.0
STAND_CLEARANCE_DROP_M = 0.04
STAND_CLEARANCE_PHASE = 0.25
KP_JOINT_HW = 600.0
KD_JOINT_HW = 90.0

ABD_LIM, PITCH_LIM, KNEE_LIM = 1.75, 2.6, 2.6
LIMIT_ESTOP_MARGIN = 0.05
START_POSE_TOL = 0.12
STAGE_POSE_TOL = 0.20
STAGE_QD_TOL = 0.50
WAIT_DWELL_S = 0.50
# Overspeed is checked once per round-robin (~21 Hz). QD_ESTOP_HARD applies
# immediately to the driver's reported sp
# reaches it. The lower sustained trip must also agree with velocity calculated
# independently from encoder position changes. The sustained threshold is kept
# above the observed 5.9 rad/s nuisance reading; the hard runaway tier remains.
QD_ESTOP = 7.0
QD_ESTOP_HARD = 8.0
QD_ESTOP_STREAK = 3
QD_ESTOP_ENCODER_CONFIRM_RATIO = 0.5
# As with STAGED_TAU_MAX, the staged runner refuses to raise the guard past a
# speed that would cross a joint's whole soft range within one status print.
QD_ESTOP_CEILING = 12.0
TEMP_ESTOP = 80
MISS_ESTOP = 20
STOP_DAMPING = 0.05
STOP_SECONDS = 0.4
STATUS_HZ = 2.0
FAULT_STATUS_HZ = 5.0
RECOVER_PERIOD_S = 0.1

# Total XML body mass, used only for the optional per-foot support feedforward.
DOG5_MASS_KG = 5.3


def _stack_pose(pose):
    return np.concatenate([np.asarray(pose[leg], dtype=float) for leg in LEGS])


# Hardware ROLL is specified only in model joint-angle coordinates, in degrees:
# [hip abduction, hip pitch, knee]. Direction is not baked into these targets.
ROLL_JOINT_TARGET_DEG = {
    "FL": (90.0, 0.0, 0.0),
    "FR": (-90.0, 0.0, 0.0),
    "RL": (90.0, 0.0, 0.0),
    "RR": (-90.0, 0.0, 0.0),
}

# Polished hardware crouch in model joint-angle coordinates, in degrees.
# The magnitudes are averaged from the direction-corrected recorded pose and
# mirrored across all four legs to remove its left/right bias.  These are
# joint-angle targets, not raw motor-output angles.
CROUCH_JOINT_TARGET_DEG = {
    "FL": (88.33, 48.04, -142.11),
    "FR": (-88.33, -48.04, 142.11),
    "RL": (88.33, -48.04, 142.11),
    "RR": (-88.33, 48.04, -142.11),
}

Q_ZERO_ALL = np.zeros(N_JOINTS)
# Controllers use radians internally.
Q_ROLL_ALL = np.deg2rad(_stack_pose(ROLL_JOINT_TARGET_DEG))
Q_CROUCH_ALL = np.deg2rad(_stack_pose(CROUCH_JOINT_TARGET_DEG))


def motoroutput_deg_to_joint_rad(motoroutput_deg):
    """Apply joint_angle = direction * motoroutput and return radians."""
    motoroutput_deg = np.asarray(motoroutput_deg, dtype=float)
    if motoroutput_deg.shape != (N_JOINTS,):
        raise ValueError(
            f"motoroutput must have shape ({N_JOINTS},), "
            f"got {motoroutput_deg.shape}"
        )
    return np.deg2rad(JOINT_DIRECTIONS * motoroutput_deg)


def joint_rad_to_motoroutput_deg(joint_angle_rad):
    """Inverse of the feedback relationship, for checks and diagnostics."""
    joint_angle_rad = np.asarray(joint_angle_rad, dtype=float)
    if joint_angle_rad.shape != (N_JOINTS,):
        raise ValueError(
            f"joint angle must have shape ({N_JOINTS},), "
            f"got {joint_angle_rad.shape}"
        )
    # direction is +/-1, so its inverse is itself.
    return JOINT_DIRECTIONS * np.rad2deg(joint_angle_rad)


def validate_hardware_config() -> None:
    if N_JOINTS != 12:
        raise ValueError(f"expected 12 configured joints, got {N_JOINTS}")
    if len(set(MOTOR_IDS)) != N_JOINTS:
        raise ValueError(f"CAN IDs must be unique: {MOTOR_IDS}")
    if any(mid <= 0 for mid in MOTOR_IDS):
        raise ValueError(f"CAN IDs must be positive: {MOTOR_IDS}")
    if any(direction not in (-1, +1) for direction in MOTOR_DIRECTIONS.values()):
        raise ValueError("every hardware direction must be +1 or -1")
    negative = {mid for mid, direction in MOTOR_DIRECTIONS.items() if direction < 0}
    if negative != {1, 3, 4, 6, 9, 12}:
        raise ValueError(f"direction table does not match verified result: {negative}")
    for joint in HARDWARE_JOINTS:
        if joint.leg not in dog5_kinematics.LEGS:
            raise ValueError(f"unknown leg in hardware map: {joint.leg}")
    expected_roll_deg = np.asarray(
        [90.0, 0.0, 0.0, -90.0, 0.0, 0.0,
         90.0, 0.0, 0.0, -90.0, 0.0, 0.0]
    )
    if not np.allclose(np.rad2deg(Q_ROLL_ALL), expected_roll_deg):
        raise ValueError("hardware ROLL joint-angle target is incorrect")
    expected_crouch_deg = np.asarray(
        [88.33, 48.04, -142.11, -88.33, -48.04, 142.11,
         88.33, -48.04, 142.11, -88.33, 48.04, -142.11]
    )
    if not np.allclose(np.rad2deg(Q_CROUCH_ALL), expected_crouch_deg):
        raise ValueError("hardware CROUCH joint-angle target is incorrect")
    roll_motoroutput = joint_rad_to_motoroutput_deg(Q_ROLL_ALL)
    if not np.allclose(
        motoroutput_deg_to_joint_rad(roll_motoroutput), Q_ROLL_ALL
    ):
        raise ValueError("ROLL joint/motoroutput direction conversion failed")
    crouch_motoroutput = joint_rad_to_motoroutput_deg(Q_CROUCH_ALL)
    if not np.allclose(
        motoroutput_deg_to_joint_rad(crouch_motoroutput), Q_CROUCH_ALL
    ):
        raise ValueError("CROUCH joint/motoroutput direction conversion failed")


class CalibratedEncoderUnwrap:
    """Continuous calibrated motor-output degrees with zero fixed by driver 0x19.

    The first sample is centered into [-180, 180), then subsequent 16-bit wraps
    are tracked.  No current-pose capture, offset subtraction, or gear ratio is
    applied.
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


class HardwareStandController:
    """Joint staging plus independent NumPy Cartesian compliance."""

    def __init__(self, cart_gain_scale, support_scale):
        self.kp_cart = cart_gain_scale * np.asarray(stand_dog5.KP_CART)
        self.kd_cart = cart_gain_scale * np.asarray(stand_dog5.KD_CART)
        self.support_per_foot = support_scale * DOG5_MASS_KG * 9.81 / 4.0
        self.foot_xy = {}
        self.crouch_foot = {}
        self.clearance_foot = {}
        self.stand_foot = {}
        for leg in LEGS:
            hip = np.asarray(dog5_kinematics.LEG_GEOMETRY[leg].hip)
            self.foot_xy[leg] = np.array(
                [hip[0], hip[1] + np.sign(hip[1]) * stand_dog5.STANCE_OUT]
            )
        for leg_index, leg in enumerate(LEGS):
            section = slice(3 * leg_index, 3 * leg_index + 3)
            self.crouch_foot[leg] = dog5_kinematics.foot_position(
                leg, Q_CROUCH_ALL[section]
            )
            self.clearance_foot[leg] = (
                self.crouch_foot[leg]
                + np.array([0.0, 0.0, -STAND_CLEARANCE_DROP_M])
            )
            self.stand_foot[leg] = np.array(
                [*self.foot_xy[leg], -stand_dog5.H_STAND]
            )

    @staticmethod
    def _cubic(t, duration):
        u = np.clip(t / duration, 0.0, 1.0)
        return 3.0 * u**2 - 2.0 * u**3, (6.0 * u - 6.0 * u**2) / duration

    def _two_stage_target(
        self, start, waypoint, final, elapsed, duration, waypoint_phase
    ):
        """Cubic start-to-waypoint-to-final target with zero join velocity."""
        split_time = waypoint_phase * duration
        if elapsed <= split_time:
            s, ds = self._cubic(elapsed, split_time)
            delta = waypoint - start
            return start + s * delta, ds * delta, waypoint_phase * s

        s, ds = self._cubic(elapsed - split_time, duration - split_time)
        delta = final - waypoint
        progress = waypoint_phase + (1.0 - waypoint_phase) * s
        return waypoint + s * delta, ds * delta, progress

    def _joint_move(self, q, qd, q0, q1, elapsed, duration):
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)
        s, ds = self._cubic(elapsed, duration)
        q_des = q0 + s * (q1 - q0)
        qd_des = ds * (q1 - q0)
        # Deliberately omit simulation qfrc_bias on the first hardware path.
        tau = KP_JOINT_HW * (q_des - q) + KD_JOINT_HW * (qd_des - qd)
        if not np.all(np.isfinite(tau)):
            raise RuntimeError("joint controller produced non-finite torque")
        return tau

    def compute_rest(self, q, qd, start_q, elapsed):
        """Smoothly move the measured starting pose to calibrated all-zero."""
        return self._joint_move(
            q, qd, start_q, Q_ZERO_ALL, elapsed, T_REST
        )

    def compute_roll(self, q, qd, elapsed):
        """Slow hardware-only all-zero to ROLL trajectory."""
        return self._joint_move(
            q, qd, Q_ZERO_ALL, Q_ROLL_ALL, elapsed, T_ROLL_HW
        )

    def compute_fold(self, q, qd, elapsed):
        """Slow hardware-only ROLL to crouch trajectory."""
        return self._joint_move(
            q, qd, Q_ROLL_ALL, Q_CROUCH_ALL, elapsed, T_FOLD_HW
        )

    def compute_stand(self, q, qd, elapsed):
        """Hardware-timed Cartesian crouch-to-stand trajectory and hold."""
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        tau = np.zeros(N_JOINTS)

        for leg_index, leg in enumerate(LEGS):
            section = slice(3 * leg_index, 3 * leg_index + 3)
            q_leg, qd_leg = q[section], qd[section]
            foot = dog5_kinematics.foot_position(leg, q_leg)
            jacobian = dog5_kinematics.foot_jacobian(leg, q_leg)
            desired, desired_velocity, _ = self._two_stage_target(
                self.crouch_foot[leg],
                self.clearance_foot[leg],
                self.stand_foot[leg],
                elapsed,
                T_STAND_HW,
                STAND_CLEARANCE_PHASE,
            )
            foot_velocity = jacobian @ qd_leg
            force = (self.kp_cart @ (desired - foot)
                     + self.kd_cart @ (desired_velocity - foot_velocity))
            force[2] -= self.support_per_foot
            tau[section] = (
                jacobian.T @ force - stand_dog5.KD_JOINT_STAND * qd_leg
            )

        if not np.all(np.isfinite(tau)):
            raise RuntimeError("Cartesian controller produced non-finite torque")
        return tau


class StageSequence:
    """Wall-time motion within a stage; Enter is required between stages."""

    def __init__(self, now, rest_start_q):
        self.stage = "REST"
        self.started_at = now
        self.wait_since = None
        self.rest_start_q = np.asarray(rest_start_q, dtype=float).copy()

    def update(self, now):
        if self.stage == "REST" and now - self.started_at >= T_REST:
            self.stage = "WAIT_REST"
            self.wait_since = now
            return "REST trajectory complete; holding all joints at zero."
        if self.stage == "ROLL" and now - self.started_at >= T_ROLL_HW:
            self.stage = "WAIT_ROLL"
            self.wait_since = now
            return "ROLL trajectory complete; holding Q_ROLL."
        if self.stage == "FOLD" and now - self.started_at >= T_FOLD_HW:
            self.stage = "WAIT_FOLD"
            self.wait_since = now
            return "FOLD trajectory complete; holding Q_CROUCH."
        if self.stage == "STAND" and now - self.started_at >= T_STAND_HW:
            self.stage = "HOLD"
            self.wait_since = now
            return "STAND trajectory complete; Cartesian HOLD is active."
        return None

    def request_next(self, now, q, qd, healthy):
        if self.stage not in ("WAIT_REST", "WAIT_ROLL", "WAIT_FOLD"):
            if self.stage == "HOLD":
                return False, "Already in final HOLD; X stops."
            return False, f"{self.stage} is still moving; Enter ignored."
        if now - self.wait_since < WAIT_DWELL_S:
            return False, "Wait for the stage hold to settle before Enter."
        if not healthy:
            return False, "Motor latch/fault present; stage advance refused."

        if self.stage == "WAIT_REST":
            target = Q_ZERO_ALL
            pose_tolerance = START_POSE_TOL
        elif self.stage == "WAIT_ROLL":
            target = Q_ROLL_ALL
            pose_tolerance = STAGE_POSE_TOL
        else:
            target = Q_CROUCH_ALL
            pose_tolerance = STAGE_POSE_TOL
        error = np.abs(np.asarray(q) - target)
        worst_index = int(np.argmax(error))
        worst_error = float(error[worst_index])
        max_speed = float(np.max(np.abs(qd)))
        if worst_error > pose_tolerance:
            return False, (
                f"Pose not reached: {JOINT_LABELS[worst_index]} error "
                f"{worst_error:.2f} rad > {pose_tolerance:.2f}."
            )
        if max_speed > STAGE_QD_TOL:
            return False, (
                f"Robot not settled: max |qd| {max_speed:.2f} rad/s > "
                f"{STAGE_QD_TOL:.2f}."
            )

        if self.stage == "WAIT_REST":
            self.stage = "ROLL"
            message = "ENTER accepted: starting ROLL."
        elif self.stage == "WAIT_ROLL":
            self.stage = "FOLD"
            message = "ENTER accepted: starting FOLD."
        else:
            self.stage = "STAND"
            message = "ENTER accepted: starting Cartesian STAND."
        self.started_at = now
        self.wait_since = None
        return True, message


def soft_limits():
    low = np.tile([-ABD_LIM, -PITCH_LIM, -KNEE_LIM], 4)
    high = np.tile([+ABD_LIM, +PITCH_LIM, +KNEE_LIM], 4)
    return low, high


class SafetyGate:
    """Torque cap, startup ramp, slew limit, joint limits, and e-stop tests."""

    def __init__(self, tau_cap, qd_estop=QD_ESTOP, qd_estop_hard=QD_ESTOP_HARD):
        self.tau_cap = tau_cap
        self.qd_estop = qd_estop
        self.qd_estop_hard = qd_estop_hard
        self.low, self.high = soft_limits()
        self.started_at = None
        self.last_time = None
        self.previous_tau = np.zeros(N_JOINTS)
        self.qd_streaks = np.zeros(N_JOINTS, dtype=int)
        self.qd_peak = np.zeros(N_JOINTS)
        self.encoder_qd = np.zeros(N_JOINTS)
        self.encoder_qd_peak = np.zeros(N_JOINTS)
        self._overspeed_last_q = None
        self._overspeed_last_time = None

    def start(self, now, q=None):
        self.started_at = now
        self.last_time = now
        if q is not None:
            self._overspeed_last_q = np.asarray(q, dtype=float).copy()
            self._overspeed_last_time = float(now)

    def cap_now(self, now):
        fraction = np.clip((now - self.started_at) / TORQUE_RAMP_S, 0.0, 1.0)
        return self.tau_cap * fraction

    def apply(self, tau, q, now):
        cap = self.cap_now(now)
        limited = np.clip(np.asarray(tau), -cap, cap)
        limited = np.clip(limited, -TAU_HARD, TAU_HARD)
        limited = np.where((q >= self.high) & (limited > 0.0), 0.0, limited)
        limited = np.where((q <= self.low) & (limited < 0.0), 0.0, limited)

        dt = np.clip(now - self.last_time, 1.0e-4, 0.05)
        max_change = TAU_SLEW_NM_S * dt
        output = self.previous_tau + np.clip(
            limited - self.previous_tau, -max_change, max_change
        )
        # A directional limit block is immediate, even if the slew limiter would
        # otherwise leave residual torque pointing farther out of bounds.
        output = np.where((q >= self.high) & (output > 0.0), 0.0, output)
        output = np.where((q <= self.low) & (output < 0.0), 0.0, output)
        self.previous_tau = output
        self.last_time = now
        return output

    def overspeed_reason(self, qd, q, now):
        """Two-tier overspeed trip with encoder confirmation of the lower tier.

        Call once per control decision: the streak counters only mean
        "consecutive checks" because each call is one check.
        """
        qd = np.asarray(qd, dtype=float)
        q = np.asarray(q, dtype=float)
        speed = np.abs(qd)
        self.qd_peak = np.maximum(self.qd_peak, speed)

        # The encoder and driver speed field are separate values in the status
        # reply. Position is already unwrapped before it reaches this gate, so
        # finite differencing it gives an independent check on sustained speed.
        if self._overspeed_last_q is None:
            position_confirmed = np.zeros(N_JOINTS, dtype=bool)
            self.encoder_qd.fill(0.0)
        else:
            dt = float(now) - self._overspeed_last_time
            if np.isfinite(dt) and dt > 0.0:
                self.encoder_qd = (q - self._overspeed_last_q) / dt
                confirm_limit = (
                    QD_ESTOP_ENCODER_CONFIRM_RATIO * self.qd_estop
                )
                position_confirmed = np.abs(self.encoder_qd) > confirm_limit
                self.encoder_qd_peak = np.maximum(
                    self.encoder_qd_peak, np.abs(self.encoder_qd)
                )
            else:
                self.encoder_qd.fill(0.0)
                position_confirmed = np.zeros(N_JOINTS, dtype=bool)
        self._overspeed_last_q = q.copy()
        self._overspeed_last_time = float(now)

        if np.any(speed > self.qd_estop_hard):
            index = int(np.argmax(speed))
            return (
                f"overspeed {JOINT_LABELS[index]}: {qd[index]:+.1f} rad/s "
                f"over the {self.qd_estop_hard:.1f} rad/s hard limit"
            )

        over = speed > self.qd_estop
        confirmed_over = over & position_confirmed
        self.qd_streaks = np.where(
            confirmed_over, self.qd_streaks + 1, 0
        )
        tripped = self.qd_streaks >= QD_ESTOP_STREAK
        if np.any(tripped):
            index = int(np.argmax(np.where(tripped, speed, 0.0)))
            return (
                f"confirmed overspeed {JOINT_LABELS[index]}: driver "
                f"{qd[index]:+.1f} rad/s, encoder {self.encoder_qd[index]:+.1f} "
                f"rad/s for {int(self.qd_streaks[index])} consecutive checks "
                f"over the {self.qd_estop:.1f} rad/s driver limit"
            )
        return None

    def estop_reason(
        self,
        q,
        qd,
        temps,
        miss_streaks,
        errors,
        now,
        enforce_position_limits=True,
    ):
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
            return "invalid encoder or velocity state"
        if enforce_position_limits:
            outside = (q < self.low - LIMIT_ESTOP_MARGIN) | (
                q > self.high + LIMIT_ESTOP_MARGIN
            )
            if np.any(outside):
                index = int(np.flatnonzero(outside)[0])
                return (
                    f"joint limit exceeded: {JOINT_LABELS[index]}="
                    f"{q[index]:+.2f} rad"
                )
        overspeed = self.overspeed_reason(qd, q, now)
        if overspeed:
            return overspeed
        if np.any(temps > TEMP_ESTOP):
            index = int(np.argmax(temps))
            return f"overtemp CAN {MOTOR_IDS[index]}: {temps[index]} C"
        if np.any(miss_streaks >= MISS_ESTOP):
            index = int(np.argmax(miss_streaks))
            return (
                f"CAN {MOTOR_IDS[index]} missed {miss_streaks[index]} "
                "consecutive replies"
            )
        hard_faults = {mid: err for mid, err in errors.items() if err & 0x7F}
        if hard_faults:
            detail = ", ".join(
                f"CAN {mid}=0x{err:02x}" for mid, err in hard_faults.items()
            )
            return f"motor fault: {detail}"
        return None


class CanMissMonitor:
    def __init__(self, mb):
        self.previous = np.asarray([mb.rec(mid).missed for mid in MOTOR_IDS])
        self.streaks = np.zeros(N_JOINTS, dtype=int)

    def update(self, mb):
        current = np.asarray([mb.rec(mid).missed for mid in MOTOR_IDS])
        added = current - self.previous
        self.streaks = np.where(added > 0, self.streaks + added, 0)
        self.previous = current
        return self.streaks.copy()


def _raw_encoders(mb):
    raw = [mb.rec(mid).encoder for mid in MOTOR_IDS]
    missing = [mid for mid, value in zip(MOTOR_IDS, raw) if value is None]
    if missing:
        raise RuntimeError(f"no encoder reply from CAN IDs: {missing}")
    return raw


def _joint_state(mb, unwrap):
    motoroutput_deg = np.asarray(
        [tracker.update(raw) for tracker, raw in zip(unwrap, _raw_encoders(mb))]
    )
    # Confirmed relationship: no zero offset and no gearbox division.
    q = motoroutput_deg_to_joint_rad(motoroutput_deg)
    speeds = mb.speeds_dps()  # MotorBus applies the same directions to velocity.
    qd = np.deg2rad(np.asarray([speeds[mid] for mid in MOTOR_IDS]))
    return q, qd


def _temperatures(mb):
    values = mb.temps()
    return np.asarray(
        [values[mid] if values[mid] is not None else 0 for mid in MOTOR_IDS],
        dtype=int,
    )


def _is_enter(key):
    return key in ("\r", "\n")


def _recover_input_lost(mb, targets, elapsed, last_recover,
                        next_fault_status=None):
    """Apply position_14710's in-loop T7 recovery and invalidate stale errors.

    The resumed control stream supplies settle traffic. Clearing the cached
    error is essential: the next verdict must come from a fresh 0x9A reply, not
    the pre-recovery 0x80 value.
    """
    targets = list(targets)
    if not targets:
        return
    detail = ", ".join(
        f"CAN {mid}=0x{(mb.rec(mid).error or 0):02x}" for mid in targets
    )
    print(f"[recover] input-lost latch: {detail}; applying 0x9B -> 0x88")
    mb.recover(targets, settle_s=0.0, verify=False)
    for mid in targets:
        mb.rec(mid).error = None
        last_recover[mid] = elapsed
        if next_fault_status is not None:
            joint_index = MOTOR_IDS.index(mid)
            next_fault_status[joint_index] = elapsed + 1.0 / FAULT_STATUS_HZ


def _zero_torque_preflight(mb, key, unwrap, status=None):
    """Verify a safe, stationary start while every command remains iq=0.

    Return the measured pose used as the start of the REST trajectory.  This
    does not capture or alter the calibrated hardware zero.

    `status` replaces the 2 Hz twelve-joint-angle line with a caller-supplied
    one: it is called as status(q, qd) and whatever string it returns is
    printed instead (return "" to print nothing).  Callers that have a live
    AHRS here use it to stream the trunk attitude, which is the one number an
    operator has to read off the robot by hand at this stage; the twelve
    angles are already covered by the soft-limit refusal below.  None keeps
    the joint line, so every existing caller is unchanged.
    """
    print(
        "[hardware] ZERO-TORQUE CHECK: mechanically support the robot and "
        "hold it still.\n[hardware] No software zero will be captured. Press "
        "ENTER to start REST: all joints will move smoothly to calibrated "
        "zero; X aborts."
    )
    slot = mb.slot(CONTROL_HZ)
    deadline = time.perf_counter() + slot
    last_print = 0.0
    index = 0
    started_at = time.perf_counter()
    next_status = {
        mid: started_at + offset / (FAULT_STATUS_HZ * N_JOINTS)
        for offset, mid in enumerate(MOTOR_IDS)
    }
    last_recover = {mid: -RECOVER_PERIOD_S for mid in MOTOR_IDS}
    while True:
        mb.poll()
        mid = MOTOR_IDS[index % N_JOINTS]
        now = time.perf_counter()
        rec = mb.rec(mid)
        if ((rec.error or 0) & 0x80 and
                now - started_at - last_recover[mid] >= RECOVER_PERIOD_S):
            _recover_input_lost(
                mb, [mid], now - started_at, last_recover
            )
            next_status[mid] = now + 1.0 / FAULT_STATUS_HZ
        elif now >= next_status[mid]:
            mb.status1_req(mid)
            next_status[mid] = now + 1.0 / FAULT_STATUS_HZ
        else:
            mb.keepalive(mid)
        q, qd = _joint_state(mb, unwrap)
        if now - last_print >= 1.0 / STATUS_HZ:
            if status is None:
                groups = []
                for leg_index, leg in enumerate(LEGS):
                    values = q[3 * leg_index:3 * leg_index + 3]
                    groups.append(
                        f"{leg}({values[0]:+.2f},{values[1]:+.2f},"
                        f"{values[2]:+.2f})"
                    )
                print("[zero] " + "  ".join(groups), flush=True)
            else:
                line = status(q, qd)
                if line:
                    print(line, flush=True)
            last_print = now

        pressed = key.get()
        if pressed in ("x", "X"):
            raise KeyboardInterrupt("operator X")
        if _is_enter(pressed):
            unverified = [
                mid for mid in MOTOR_IDS
                if mb.rec(mid).error is None or (mb.rec(mid).error & 0x80)
            ]
            max_speed = float(np.max(np.abs(qd)))
            low, high = soft_limits()
            outside = (q < low) | (q > high)
            if unverified:
                print(
                    f"[hardware] ENTER refused: waiting for fresh clear status "
                    f"from CAN {unverified}."
                )
            elif np.any(outside):
                joint_index = int(np.flatnonzero(outside)[0])
                print(
                    f"[hardware] ENTER refused: {JOINT_LABELS[joint_index]}="
                    f"{q[joint_index]:+.2f} rad is outside the soft limit."
                )
            elif max_speed > STAGE_QD_TOL:
                print(
                    f"[hardware] ENTER refused: max |qd| is {max_speed:.2f} "
                    "rad/s; hold the robot still."
                )
            else:
                return q.copy()

        index += 1
        mb.pace(deadline)
        deadline += slot


def _soft_stop(mb, seconds=STOP_SECONDS):
    slot = mb.slot(CONTROL_HZ)
    deadline = time.perf_counter() + slot
    end = time.perf_counter() + seconds
    index = 0
    while time.perf_counter() < end:
        mb.poll()
        mid = MOTOR_IDS[index % N_JOINTS]
        qd = np.deg2rad(mb.speeds_dps()[mid])
        mb.torque(mid, -STOP_DAMPING * qd)
        index += 1
        mb.pace(deadline)
        deadline += slot


class KeyPoller:
    def __init__(self):
        import select
        import termios
        import tty

        if not sys.stdin.isatty():
            raise RuntimeError("hardware stand requires an interactive terminal")
        self._select = select
        self._termios = termios
        self.fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self._closed = False

    def get(self):
        readable, _, _ = self._select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if readable else ""

    def close(self):
        if not self._closed:
            self._termios.tcsetattr(
                self.fd, self._termios.TCSADRAIN, self._old
            )
            self._closed = True


def run_hardware(tau_max, cart_gain_scale, support_scale,
                 qd_estop=QD_ESTOP, qd_estop_hard=QD_ESTOP_HARD):
    validate_hardware_config()
    controller = HardwareStandController(cart_gain_scale, support_scale)
    gate = SafetyGate(
        tau_cap=tau_max, qd_estop=qd_estop, qd_estop_hard=qd_estop_hard
    )
    unwrap = [CalibratedEncoderUnwrap() for _ in HARDWARE_JOINTS]

    print(f"[hardware] joints = {JOINT_LABELS}")
    print(f"[hardware] CAN IDs = {MOTOR_IDS}")
    print(
        "[hardware] verified directions = "
        f"{[MOTOR_DIRECTIONS[mid] for mid in MOTOR_IDS]}"
    )
    print(
        "[hardware] q = radians(direction * motoroutput); no offset; "
        "no gearbox division"
    )
    print(
        "[hardware] ROLL desired joint angles [abd,pitch,knee] deg: "
        + ", ".join(
            f"{leg}={tuple(ROLL_JOINT_TARGET_DEG[leg])}" for leg in LEGS
        )
    )
    roll_motoroutput = joint_rad_to_motoroutput_deg(Q_ROLL_ALL)
    print(
        "[hardware] ROLL expected motoroutput deg: "
        + ", ".join(
            f"CAN {joint.can_id}={roll_motoroutput[index]:+.0f}"
            for index, joint in enumerate(HARDWARE_JOINTS)
        )
    )
    print(
        "[hardware] CROUCH desired joint angles [abd,pitch,knee] deg: "
        + ", ".join(
            f"{leg}={tuple(CROUCH_JOINT_TARGET_DEG[leg])}" for leg in LEGS
        )
    )
    crouch_motoroutput = joint_rad_to_motoroutput_deg(Q_CROUCH_ALL)
    print(
        "[hardware] CROUCH expected motoroutput deg: "
        + ", ".join(
            f"CAN {joint.can_id}={crouch_motoroutput[index]:+.2f}"
            for index, joint in enumerate(HARDWARE_JOINTS)
        )
    )
    print(
        f"[hardware] tau cap={tau_max:.2f} N*m, cart scale={cart_gain_scale:.2f}, "
        f"support scale={support_scale:.2f}"
    )
    print(
        f"[hardware] motion times: REST={T_REST:.1f}s, ROLL={T_ROLL_HW:.1f}s, "
        f"FOLD={T_FOLD_HW:.1f}s, STAND={T_STAND_HW:.1f}s; "
        f"joint PD=({KP_JOINT_HW:.1f}, {KD_JOINT_HW:.1f})"
    )
    print("[hardware] ROBOT MUST REMAIN MECHANICALLY SUPPORTED. X stops.")

    key = KeyPoller()
    try:
        with motorbus.MotorBus(MOTOR_IDS, dirs=MOTOR_DIRECTIONS) as mb:
            armed = False
            stop_reason = None
            try:
                print("[hardware] Arming with a zero-torque stream...")
                if not mb.arm(rate_hz=CONTROL_HZ):
                    raise RuntimeError("not all motors armed")
                armed = True

                rest_start_q = _zero_torque_preflight(mb, key, unwrap)
                now = time.perf_counter()
                sequence = StageSequence(now, rest_start_q)
                gate.start(now, rest_start_q)
                print(
                    "[hardware] ENTER accepted: starting REST -> all joints "
                    "to calibrated zero. Do not press Enter while moving."
                )

                slot = mb.slot(CONTROL_HZ)
                deadline = time.perf_counter() + slot
                start = now
                q = np.zeros(N_JOINTS)
                tau_command = np.zeros(N_JOINTS)
                miss_monitor = CanMissMonitor(mb)
                status_period = 1.0 / FAULT_STATUS_HZ
                next_fault_status = np.asarray(
                    [status_period + i * status_period / N_JOINTS
                     for i in range(N_JOINTS)]
                )
                last_recover = {mid: -RECOVER_PERIOD_S for mid in MOTOR_IDS}
                last_print = 0.0
                index = 0

                while True:
                    mb.poll()
                    joint_index = index % N_JOINTS

                    if joint_index == 0:
                        now = time.perf_counter()
                        elapsed = now - start
                        q, qd = _joint_state(mb, unwrap)

                        event = sequence.update(now)
                        if event:
                            print(f"[stage] {event}")
                            if sequence.stage == "WAIT_REST":
                                print(
                                    "[stage] Inspect all-zero pose; press "
                                    "ENTER for ROLL when settled."
                                )
                            elif sequence.stage == "WAIT_ROLL":
                                print("[stage] Inspect robot; press ENTER for FOLD when settled.")
                            elif sequence.stage == "WAIT_FOLD":
                                print("[stage] Inspect crouch; press ENTER for STAND when settled.")
                            else:
                                print("[stage] Final HOLD active; press X to stop.")

                        if sequence.stage == "REST":
                            requested_tau = controller.compute_rest(
                                q,
                                qd,
                                sequence.rest_start_q,
                                now - sequence.started_at,
                            )
                        elif sequence.stage == "WAIT_REST":
                            requested_tau = controller.compute_rest(
                                q,
                                qd,
                                sequence.rest_start_q,
                                T_REST,
                            )
                        elif sequence.stage == "ROLL":
                            requested_tau = controller.compute_roll(
                                q, qd, now - sequence.started_at
                            )
                        elif sequence.stage == "WAIT_ROLL":
                            requested_tau = controller.compute_roll(
                                q, qd, T_ROLL_HW
                            )
                        elif sequence.stage == "FOLD":
                            requested_tau = controller.compute_fold(
                                q, qd, now - sequence.started_at
                            )
                        elif sequence.stage == "WAIT_FOLD":
                            requested_tau = controller.compute_fold(
                                q, qd, T_FOLD_HW
                            )
                        elif sequence.stage == "STAND":
                            requested_tau = controller.compute_stand(
                                q, qd, now - sequence.started_at
                            )
                        else:
                            requested_tau = controller.compute_stand(
                                q, qd, T_STAND_HW
                            )
                        tau_command = gate.apply(requested_tau, q, now)

                        temps = _temperatures(mb)
                        misses = miss_monitor.update(mb)
                        errors = mb.errors()
                        stop_reason = gate.estop_reason(
                            q, qd, temps, misses, errors, now
                        )
                        latched = [mid for mid, err in errors.items() if err & 0x80]
                        unverified = [
                            mid for mid in MOTOR_IDS if mb.rec(mid).error is None
                        ]

                        pressed = key.get()
                        if pressed in ("x", "X"):
                            stop_reason = "operator X"
                        elif _is_enter(pressed):
                            _, message = sequence.request_next(
                                now,
                                q,
                                qd,
                                healthy=(not latched and not unverified
                                         and stop_reason is None),
                            )
                            print(f"[stage] {message}")
                        if stop_reason:
                            break

                        recover = [
                            mid for mid in latched
                            if elapsed - last_recover[mid] >= RECOVER_PERIOD_S
                        ]
                        if recover:
                            _recover_input_lost(
                                mb,
                                recover,
                                elapsed,
                                last_recover,
                                next_fault_status,
                            )
                            latched = []

                        if now - last_print >= 1.0 / STATUS_HZ:
                            target = (
                                Q_ZERO_ALL if sequence.stage == "WAIT_REST" else
                                Q_ROLL_ALL if sequence.stage == "WAIT_ROLL" else
                                Q_CROUCH_ALL if sequence.stage == "WAIT_FOLD" else None
                            )
                            pose_error = (float(np.max(np.abs(q - target)))
                                          if target is not None else 0.0)
                            print(
                                f"[hardware] stage={sequence.stage:9s} "
                                f"max|q|={np.max(np.abs(q)):.2f} rad "
                                f"pose_err={pose_error:.2f} "
                                f"max|qd|={np.max(np.abs(qd)):.2f} "
                                f"max|qd_enc|={np.max(np.abs(gate.encoder_qd)):.2f} "
                                f"peak|qd|={np.max(gate.qd_peak):.2f} "
                                f"max|tau|={np.max(np.abs(tau_command)):.2f} "
                                f"Tmax={int(np.max(temps))}C "
                                f"latched={len(latched)}",
                                flush=True,
                            )
                            last_print = now

                    mid = MOTOR_IDS[joint_index]
                    elapsed = time.perf_counter() - start
                    if elapsed >= next_fault_status[joint_index]:
                        mb.status1_req(mid)
                        next_fault_status[joint_index] += status_period
                    else:
                        mb.torque(mid, float(tau_command[joint_index]))

                    index += 1
                    mb.pace(deadline)
                    deadline += slot

            except KeyboardInterrupt as exc:
                stop_reason = str(exc) or "KeyboardInterrupt"
            except Exception as exc:
                stop_reason = f"error: {exc}"
                raise
            finally:
                if armed:
                    print(f"[hardware] stopping: {stop_reason or 'aborted'}")
                    try:
                        _soft_stop(mb)
                    except Exception as exc:
                        print(
                            f"[hardware] soft stop failed: {exc}; sending STOP",
                            file=sys.stderr,
                        )
    finally:
        key.close()
    return 0


def offline_self_test():
    validate_hardware_config()

    near_zero_negative = CalibratedEncoderUnwrap().update(65530)
    assert -0.1 < near_zero_negative < 0.0, near_zero_negative
    direct = CalibratedEncoderUnwrap().update(1000)
    assert abs(direct - 1000 * motorbus.ENCODER_GAIN) < 1.0e-12

    controller = HardwareStandController(0.25, 0.25)
    zero_qd = np.zeros(N_JOINTS)
    for tau in (
        controller.compute_roll(Q_ZERO_ALL, zero_qd, 0.0),
        controller.compute_roll(Q_ROLL_ALL, zero_qd, T_ROLL_HW),
        controller.compute_fold(Q_ROLL_ALL, zero_qd, 0.0),
        controller.compute_fold(Q_CROUCH_ALL, zero_qd, T_FOLD_HW),
        controller.compute_stand(Q_CROUCH_ALL, zero_qd, 0.0),
        controller.compute_stand(Q_CROUCH_ALL, zero_qd, T_STAND_HW),
    ):
        assert tau.shape == (N_JOINTS,) and np.all(np.isfinite(tau))
    stand_start_tau = controller.compute_stand(Q_CROUCH_ALL, zero_qd, 0.0)
    expected_support_tau = np.zeros(N_JOINTS)
    support_force = np.array([0.0, 0.0, -controller.support_per_foot])
    for leg_index, leg in enumerate(LEGS):
        section = slice(3 * leg_index, 3 * leg_index + 3)
        expected_support_tau[section] = (
            dog5_kinematics.foot_jacobian(leg, Q_CROUCH_ALL[section]).T
            @ support_force
        )
    assert np.allclose(stand_start_tau, expected_support_tau), stand_start_tau

    # The slower hardware profile reduces peak ROLL desired speed from the
    # simulation's ~1.96 rad/s to below 0.60 rad/s.
    _, roll_ds = controller._cubic(T_ROLL_HW / 2.0, T_ROLL_HW)
    max_roll_qd_des = float(np.max(np.abs(roll_ds * Q_ROLL_ALL)))
    assert max_roll_qd_des < 0.60, max_roll_qd_des

    rest_start = 0.25 * np.sin(np.arange(N_JOINTS))
    rest_tau = controller.compute_rest(
        rest_start, np.zeros(N_JOINTS), rest_start, 0.0
    )
    assert rest_tau.shape == (N_JOINTS,) and np.all(np.isfinite(rest_tau))

    sequence = StageSequence(0.0, rest_start)
    assert sequence.update(T_REST + 0.01)
    advanced, _ = sequence.request_next(
        T_REST + WAIT_DWELL_S + 0.02,
        Q_ZERO_ALL,
        np.zeros(N_JOINTS),
        True,
    )
    assert advanced and sequence.stage == "ROLL"
    roll_start = sequence.started_at
    assert sequence.update(roll_start + T_ROLL_HW + 0.01)
    advanced, _ = sequence.request_next(
        roll_start + T_ROLL_HW + WAIT_DWELL_S + 0.02,
        Q_ROLL_ALL,
        np.zeros(N_JOINTS),
        True,
    )
    assert advanced and sequence.stage == "FOLD"
    fold_start = sequence.started_at
    assert sequence.update(fold_start + T_FOLD_HW + 0.01)
    advanced, _ = sequence.request_next(
        fold_start + T_FOLD_HW + WAIT_DWELL_S + 0.02,
        Q_CROUCH_ALL,
        np.zeros(N_JOINTS),
        True,
    )
    assert advanced and sequence.stage == "STAND"
    stand_start = sequence.started_at
    assert sequence.update(stand_start + T_STAND_HW + 0.01)
    assert sequence.stage == "HOLD"

    # Overspeed: a lone marginal sample is ridden out, noisy/stuck driver speed
    # without matching encoder motion is ignored, confirmed motion trips, and
    # anything past the hard limit trips on the first sample.
    gate = SafetyGate(tau_cap=1.0)
    knee = JOINT_LABELS.index("RR_knee")
    spike = np.zeros(N_JOINTS)
    spike[knee] = QD_ESTOP + 0.2
    q_stationary = np.zeros(N_JOINTS)
    check_dt = N_JOINTS / CONTROL_HZ
    assert gate.overspeed_reason(spike, q_stationary, 0.0) is None
    assert gate.qd_streaks[knee] == 0
    assert gate.overspeed_reason(
        np.zeros(N_JOINTS), q_stationary, check_dt
    ) is None
    assert gate.qd_streaks[knee] == 0, "streak must reset on a clean sample"
    assert abs(gate.qd_peak[knee] - spike[knee]) < 1.0e-12, (
        "peak must survive the reset"
    )

    noisy = SafetyGate(tau_cap=1.0)
    abd = JOINT_LABELS.index("RR_abd")
    noisy_driver_speed = np.zeros(N_JOINTS)
    noisy_driver_speed[abd] = -(QD_ESTOP + 0.1)
    normal_q = np.zeros(N_JOINTS)
    noisy.start(0.0, normal_q)
    for sample in range(1, 2 * QD_ESTOP_STREAK + 1):
        normal_q[abd] = -0.6 * sample * check_dt
        assert noisy.overspeed_reason(
            noisy_driver_speed, normal_q, sample * check_dt
        ) is None
    assert noisy.qd_streaks[abd] == 0

    # The observed RR_abd nuisance value is below the new default even if the
    # encoder were to agree with it, so it must never accumulate a trip streak.
    accepted = SafetyGate(tau_cap=1.0)
    accepted_driver_speed = np.zeros(N_JOINTS)
    accepted_driver_speed[abd] = -5.9
    accepted_q = np.zeros(N_JOINTS)
    accepted.start(0.0, accepted_q)
    for sample in range(1, 2 * QD_ESTOP_STREAK + 1):
        accepted_q[abd] = -5.9 * sample * check_dt
        assert accepted.overspeed_reason(
            accepted_driver_speed, accepted_q, sample * check_dt
        ) is None
    assert accepted.qd_streaks[abd] == 0

    moving = SafetyGate(tau_cap=1.0)
    moving_q = np.zeros(N_JOINTS)
    moving.start(0.0, moving_q)
    for sample in range(1, QD_ESTOP_STREAK):
        moving_q[knee] = sample * check_dt * spike[knee]
        assert moving.overspeed_reason(
            spike, moving_q, sample * check_dt
        ) is None
    moving_q[knee] = QD_ESTOP_STREAK * check_dt * spike[knee]
    sustained = moving.overspeed_reason(
        spike, moving_q, QD_ESTOP_STREAK * check_dt
    )
    assert sustained and "confirmed overspeed RR_knee" in sustained, sustained

    hard = SafetyGate(tau_cap=1.0)
    runaway = np.zeros(N_JOINTS)
    runaway[knee] = -(QD_ESTOP_HARD + 0.1)
    immediate = hard.overspeed_reason(runaway, q_stationary, 0.0)
    assert immediate and "hard limit" in immediate, immediate
    assert "confirmed" not in immediate, immediate

    # The debounce must not let a joint idle just under the limit forever.
    creep = SafetyGate(tau_cap=1.0, qd_estop=2.0, qd_estop_hard=9.0)
    below = np.full(N_JOINTS, 1.99)
    for sample in range(10 * QD_ESTOP_STREAK):
        assert creep.overspeed_reason(
            below, q_stationary, sample * check_dt
        ) is None

    class _FakeRecord:
        def __init__(self):
            self.error = 0x80

    class _FakeBus:
        def __init__(self):
            self.records = {mid: _FakeRecord() for mid in MOTOR_IDS}
            self.calls = []

        def rec(self, mid):
            return self.records[mid]

        def recover(self, targets, **kwargs):
            self.calls.append((list(targets), kwargs))

    fake = _FakeBus()
    last_recover = {mid: -RECOVER_PERIOD_S for mid in MOTOR_IDS}
    next_status = np.zeros(N_JOINTS)
    _recover_input_lost(fake, [1, 4], 2.0, last_recover, next_status)
    assert fake.calls == [([1, 4], {"settle_s": 0.0, "verify": False})]
    assert fake.rec(1).error is None and fake.rec(4).error is None
    assert last_recover[1] == 2.0 and last_recover[4] == 2.0
    assert next_status[MOTOR_IDS.index(1)] > 2.0
    assert next_status[MOTOR_IDS.index(4)] > 2.0

    print("stand_dog5_hw offline self-test PASS")
    print(
        "  ROLL joint targets deg:",
        {leg: tuple(ROLL_JOINT_TARGET_DEG[leg]) for leg in LEGS},
    )
    print(
        "  ROLL expected motoroutput deg:",
        {
            joint.can_id: float(value)
            for joint, value in zip(
                HARDWARE_JOINTS,
                joint_rad_to_motoroutput_deg(Q_ROLL_ALL),
            )
        },
    )
    print(
        "  CROUCH joint targets deg:",
        {leg: tuple(CROUCH_JOINT_TARGET_DEG[leg]) for leg in LEGS},
    )
    print(
        "  CROUCH expected motoroutput deg:",
        {
            joint.can_id: float(value)
            for joint, value in zip(
                HARDWARE_JOINTS,
                joint_rad_to_motoroutput_deg(Q_CROUCH_ALL),
            )
        },
    )
    print("  negative CAN IDs:", sorted(
        mid for mid, direction in MOTOR_DIRECTIONS.items() if direction < 0
    ))
    print("  encoder conversion: direct output gain, no offset, no gearbox divide")
    print(
        "  stages: REST -> ENTER -> ROLL -> ENTER -> FOLD -> ENTER -> "
        "STAND -> HOLD"
    )
    print(
        f"  hardware times: REST {T_REST:.1f}s, ROLL {T_ROLL_HW:.1f}s, "
        f"FOLD {T_FOLD_HW:.1f}s, STAND {T_STAND_HW:.1f}s"
    )
    print(f"  peak ROLL desired speed: {max_roll_qd_des:.3f} rad/s")
    print(
        f"  overspeed trip: >{QD_ESTOP_HARD:.1f} rad/s immediate, "
        f">{QD_ESTOP:.1f} rad/s plus encoder confirmation for "
        f"{QD_ESTOP_STREAK} consecutive checks "
        f"(~{1000.0 * QD_ESTOP_STREAK * N_JOINTS / CONTROL_HZ:.0f} ms)"
    )
    print("  input-lost recovery: stale 0x80 cleared; fresh status scheduled")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tau-max",
        type=float,
        default=DEFAULT_TAU_MAX,
        help=f"per-joint torque cap in N*m (default: {DEFAULT_TAU_MAX})",
    )
    parser.add_argument(
        "--cart-gain-scale",
        type=float,
        default=DEFAULT_CART_GAIN_SCALE,
        help=f"Cartesian Kp/Kd scale in (0,1] (default: {DEFAULT_CART_GAIN_SCALE})",
    )
    parser.add_argument(
        "--support-scale",
        type=float,
        default=DEFAULT_SUPPORT_SCALE,
        help=f"mg/4 feedforward scale in [0,1] (default: {DEFAULT_SUPPORT_SCALE})",
    )
    parser.add_argument(
        "--qd-estop",
        type=float,
        default=QD_ESTOP,
        help=(
            f"sustained driver-speed trip in rad/s; requires encoder motion "
            f"confirmation for {QD_ESTOP_STREAK} consecutive checks "
            f"(default: {QD_ESTOP})"
        ),
    )
    parser.add_argument(
        "--qd-estop-hard",
        type=float,
        default=QD_ESTOP_HARD,
        help=(
            "single-sample joint-speed trip in rad/s "
            f"(default: {QD_ESTOP_HARD})"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run mapping/controller/stage tests without opening CAN",
    )
    args = parser.parse_args()

    if not 0.0 < args.tau_max <= STAGED_TAU_MAX:
        parser.error(f"--tau-max must be > 0 and <= {STAGED_TAU_MAX} N*m")
    if not 0.0 < args.cart_gain_scale <= 1.0:
        parser.error("--cart-gain-scale must be > 0 and <= 1")
    if not 0.0 <= args.support_scale <= 1.0:
        parser.error("--support-scale must be between 0 and 1")
    if not 0.0 < args.qd_estop <= QD_ESTOP_CEILING:
        parser.error(f"--qd-estop must be > 0 and <= {QD_ESTOP_CEILING} rad/s")
    if not args.qd_estop < args.qd_estop_hard <= QD_ESTOP_CEILING:
        parser.error(
            "--qd-estop-hard must be above --qd-estop and "
            f"<= {QD_ESTOP_CEILING} rad/s"
        )
    if args.self_test:
        return offline_self_test()

    try:
        return run_hardware(
            args.tau_max,
            args.cart_gain_scale,
            args.support_scale,
            args.qd_estop,
            args.qd_estop_hard,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[hardware] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
