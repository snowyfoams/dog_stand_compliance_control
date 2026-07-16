#!/usr/bin/env python3
"""Stand DOG5 with each foot's crouch x/y position held fixed.

This runner starts from a mirrored crouch derived from the direction-corrected
hardware recording in ``dog5_description/stand_dog5_hw.py``.

The operator-gated sequence is deliberately short:

    ZERO TORQUE  read and display the current pose; Enter starts CROUCH
    CROUCH       native 0xA4 position control moves to the recorded pose
    WAIT CROUCH  hold the recorded pose; Enter starts Cartesian STAND
    STAND        Cartesian compliance keeps each crouch foot x/y fixed while
                 ramping only z to the configured standing height
    WAIT STAND   keep the final target until position error and speed settle
    HOLD         hold the full Cartesian target; partial tests report HOLD_PARTIAL
    PARK         press P in HOLD to reverse the Cartesian path back to crouch
    WAIT PARK    settle at crouch, then resume the native crouch hold

The recorded angles are joint coordinates, not raw motor coordinates.  The
confirmed directions in ``dog5_hardware_map.py`` are applied exactly once by
the feedback and torque conversions.

The robot must remain mechanically supported.  This encoder-only controller
does not know trunk attitude or foot contact and is not a balance controller.
The fixed foot targets are in the trunk frame, not the world frame.
Start with a small fraction of the validated path and a 1 N*m torque cap::

    ../.venv/bin/python stand_dog5_fixed_fend_hw.py --self-test
    ../.venv/bin/python stand_dog5_fixed_fend_hw.py \
        --tau-max 1.0 --travel-scale 0.25
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DESCRIPTION = os.path.join(_HERE, "dog5_description")
sys.path.insert(0, _DESCRIPTION)

import dog5_kinematics
import stand_dog5
import stand_dog5_hw as base


LEGS = base.LEGS
MOTOR_IDS = base.MOTOR_IDS
MOTOR_DIRECTIONS = base.MOTOR_DIRECTIONS
JOINT_LABELS = base.JOINT_LABELS
N_JOINTS = base.N_JOINTS

# Preserve the original measured pose for the raw-versus-polished diagnostics.
RAW_RECORDED_CROUCH_DEG = {
    "FL": (89.62, 47.57, -140.92),
    "FR": (-84.59, -53.61, 148.50),
    "RL": (86.16, -44.44, 142.34),
    "RR": (-92.95, 46.54, -136.68),
}

# Use the base controller's polished pose.  Each magnitude is the mean of the
# four measured magnitudes; the signs follow DOG5's mirrored joint axes.  This
# removes the measured left/right bias without inventing a new crouch depth or
# weakening a joint limit.
POLISHED_CROUCH_DEG = {
    leg: tuple(base.CROUCH_JOINT_TARGET_DEG[leg]) for leg in LEGS
}

# Retain the established names used by the shared sequence and safety code.
RECORDED_CROUCH_DEG = POLISHED_CROUCH_DEG
Q_RECORDED_CROUCH = np.deg2rad(base._stack_pose(RECORDED_CROUCH_DEG))
POSITION_TARGET_DEG = np.rad2deg(Q_RECORDED_CROUCH)

T_STAND = 8.0
T_PARK = 8.0
RECORDED_POSE_TOL = 0.08
RECORDED_QD_TOL = 0.25
DEFAULT_CROUCH_MAX_MOTOR_DPS = 100.0
DEFAULT_CROUCH_TORQUE_TRIP_NM = 2.0
DEFAULT_CROUCH_SPEED_TRIP_RAD_S = 1.0
CROUCH_TIMEOUT_S = 30.0
CROUCH_SETTLE_S = 0.50
FINAL_CART_TOL_M = 0.025
FINAL_QD_TOL = 0.30
FINAL_DWELL_S = 0.50
DEFAULT_STAND_HEIGHT = 0.20
MIN_STAND_HEIGHT = 0.14
MAX_STAND_HEIGHT = 0.22
MIN_JACOBIAN_SINGULAR = 0.005
MAX_CART_ERROR_M = 0.15
MAX_CART_FORCE_N = 30.0


class EncoderVelocity:
    """Filtered velocity from position changes for control and settle gates."""

    def __init__(self, alpha=0.35):
        self.alpha = float(alpha)
        self.previous_q = None
        self.previous_time = None
        self.value = np.zeros(N_JOINTS)
        self.samples = 0

    def update(self, q, now):
        q = np.asarray(q, dtype=float)
        if self.previous_q is not None:
            dt = now - self.previous_time
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


class RecordedCrouchController(base.HardwareStandController):
    """Cartesian controller used after native crouch positioning."""

    def __init__(
        self,
        cart_gain_scale,
        support_scale,
        stand_height,
        travel_scale=1.0,
    ):
        super().__init__(cart_gain_scale, support_scale)
        self.stand_height = float(stand_height)
        self.travel_scale = float(travel_scale)
        self.crouch_foot = {}
        self.full_stand_foot = {}
        self.final_foot = {}
        for leg_index, leg in enumerate(LEGS):
            section = slice(3 * leg_index, 3 * leg_index + 3)
            self.crouch_foot[leg] = dog5_kinematics.foot_position(
                leg, Q_RECORDED_CROUCH[section]
            )
            # Preserve the exact crouch foot x/y coordinates.  Standing is a
            # vertical trunk-frame foot motion only; no horizontal sweep to
            # the nominal under-body stance is commanded.
            self.full_stand_foot[leg] = np.array(
                [
                    self.crouch_foot[leg][0],
                    self.crouch_foot[leg][1],
                    -self.stand_height,
                ]
            )
            self.final_foot[leg] = (
                self.crouch_foot[leg]
                + self.travel_scale
                * (self.full_stand_foot[leg] - self.crouch_foot[leg])
            )
            self.clearance_foot[leg] = (
                self.crouch_foot[leg]
                + np.array(
                    [
                        0.0,
                        0.0,
                        -base.STAND_CLEARANCE_DROP_M * self.travel_scale,
                    ]
                )
            )
        self.last_cart_error = 0.0
        self.last_xy_error = 0.0
        self.last_min_singular = np.inf
        self.last_max_force = 0.0

    def cartesian_target(self, leg, elapsed):
        return self._two_stage_target(
            self.crouch_foot[leg],
            self.clearance_foot[leg],
            self.final_foot[leg],
            elapsed,
            T_STAND,
            base.STAND_CLEARANCE_PHASE,
        )

    def parking_target(self, leg, elapsed):
        target, velocity, reverse_progress = self._two_stage_target(
            self.final_foot[leg],
            self.clearance_foot[leg],
            self.crouch_foot[leg],
            elapsed,
            T_PARK,
            1.0 - base.STAND_CLEARANCE_PHASE,
        )
        return target, velocity, 1.0 - reverse_progress

    def _compute_cartesian(self, q, qd, target):
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        tau = np.zeros(N_JOINTS)
        max_error = 0.0
        max_xy_error = 0.0
        min_singular = np.inf
        max_force = 0.0

        for leg_index, leg in enumerate(LEGS):
            section = slice(3 * leg_index, 3 * leg_index + 3)
            q_leg, qd_leg = q[section], qd[section]
            foot = dog5_kinematics.foot_position(leg, q_leg)
            jacobian = dog5_kinematics.foot_jacobian(leg, q_leg)
            singular = float(np.min(np.linalg.svd(jacobian, compute_uv=False)))
            min_singular = min(min_singular, singular)
            if singular < MIN_JACOBIAN_SINGULAR:
                raise RuntimeError(
                    f"{leg} Jacobian near singular: sigma_min={singular:.4f} "
                    f"< {MIN_JACOBIAN_SINGULAR:.4f} m/rad"
                )

            desired, desired_velocity, progress = target(leg)
            error = desired - foot
            error_norm = float(np.linalg.norm(error))
            max_error = max(max_error, error_norm)
            max_xy_error = max(
                max_xy_error,
                float(np.linalg.norm(error[:2])),
            )
            if error_norm > MAX_CART_ERROR_M:
                raise RuntimeError(
                    f"{leg} Cartesian error {error_norm:.3f} m exceeds "
                    f"{MAX_CART_ERROR_M:.3f} m"
                )

            foot_velocity = jacobian @ qd_leg
            force = (
                self.kp_cart @ error
                + self.kd_cart @ (desired_velocity - foot_velocity)
            )
            # No feedforward step at either Cartesian transition. It grows
            # during STAND and fades during PARK.
            force[2] -= (
                progress * self.travel_scale * self.support_per_foot
            )
            force_norm = float(np.linalg.norm(force))
            if force_norm > MAX_CART_FORCE_N:
                force *= MAX_CART_FORCE_N / force_norm
                force_norm = MAX_CART_FORCE_N
            max_force = max(max_force, force_norm)
            tau[section] = (
                jacobian.T @ force - stand_dog5.KD_JOINT_STAND * qd_leg
            )

        self.last_cart_error = max_error
        self.last_xy_error = max_xy_error
        self.last_min_singular = min_singular
        self.last_max_force = max_force
        if not np.all(np.isfinite(tau)):
            raise RuntimeError("Cartesian controller produced non-finite torque")
        return tau

    def compute_stand(self, q, qd, elapsed):
        return self._compute_cartesian(
            q, qd, lambda leg: self.cartesian_target(leg, elapsed)
        )

    def compute_park(self, q, qd, elapsed):
        return self._compute_cartesian(
            q, qd, lambda leg: self.parking_target(leg, elapsed)
        )


class RecordedStandSequence:
    """Stand from crouch, hold, then optionally park back at crouch."""

    def __init__(self, now, start_q, travel_scale=1.0):
        self.stage = "CROUCH"
        self.started_at = float(now)
        self.wait_since = None
        self.start_q = np.asarray(start_q, dtype=float).copy()
        self.travel_scale = float(travel_scale)
        self.crouch_settle_since = None
        self.final_settle_since = None
        self.park_settle_since = None

    def update(self, now, q=None, cart_error=None, qd=None):
        if self.stage == "CROUCH":
            if now - self.started_at > CROUCH_TIMEOUT_S:
                raise RuntimeError(
                    f"native position CROUCH timed out after "
                    f"{CROUCH_TIMEOUT_S:.1f} s"
                )
            if q is None or qd is None:
                self.crouch_settle_since = None
            else:
                pose_error = float(
                    np.max(np.abs(np.asarray(q) - Q_RECORDED_CROUCH))
                )
                speed = float(np.max(np.abs(qd)))
                settled = (
                    pose_error <= RECORDED_POSE_TOL
                    and speed <= RECORDED_QD_TOL
                )
                if not settled:
                    self.crouch_settle_since = None
                elif self.crouch_settle_since is None:
                    self.crouch_settle_since = float(now)
                elif now - self.crouch_settle_since >= CROUCH_SETTLE_S:
                    self.stage = "WAIT_CROUCH"
                    self.wait_since = float(now)
                    return (
                        "Native position CROUCH reached and settled; holding "
                        "the recorded pose."
                    )
        if self.stage == "STAND" and now - self.started_at >= T_STAND:
            self.stage = "WAIT_STAND"
            self.wait_since = float(now)
            self.final_settle_since = None
            return (
                "STAND target trajectory complete; maintaining the final "
                "target until Cartesian tracking settles."
            )
        if self.stage == "WAIT_STAND":
            speed = (
                float(np.max(np.abs(qd))) if qd is not None else np.inf
            )
            settled = (
                cart_error is not None
                and cart_error <= FINAL_CART_TOL_M
                and speed <= FINAL_QD_TOL
            )
            if not settled:
                self.final_settle_since = None
            elif self.final_settle_since is None:
                self.final_settle_since = float(now)
            elif now - self.final_settle_since >= FINAL_DWELL_S:
                self.stage = (
                    "HOLD" if self.travel_scale >= 1.0 else "HOLD_PARTIAL"
                )
                self.wait_since = float(now)
                if self.stage == "HOLD":
                    return (
                        "Full Cartesian target tracked and settled; HOLD is active."
                    )
                return (
                    "Partial Cartesian target tracked and settled; "
                    "HOLD_PARTIAL is active (not a full stand)."
                )
        if self.stage == "PARK" and now - self.started_at >= T_PARK:
            self.stage = "WAIT_PARK"
            self.wait_since = float(now)
            self.park_settle_since = None
            return (
                "PARK target trajectory complete; maintaining the crouch "
                "target until Cartesian tracking settles."
            )
        if self.stage == "WAIT_PARK":
            speed = (
                float(np.max(np.abs(qd))) if qd is not None else np.inf
            )
            settled = (
                cart_error is not None
                and cart_error <= FINAL_CART_TOL_M
                and speed <= FINAL_QD_TOL
            )
            if not settled:
                self.park_settle_since = None
            elif self.park_settle_since is None:
                self.park_settle_since = float(now)
            elif now - self.park_settle_since >= FINAL_DWELL_S:
                self.stage = "WAIT_CROUCH"
                self.wait_since = float(now)
                return (
                    "PARK reached the recorded crouch; native position hold "
                    "is active."
                )
        return None

    def request_next(self, now, q, qd, healthy):
        if self.stage != "WAIT_CROUCH":
            if self.stage in ("HOLD", "HOLD_PARTIAL"):
                return False, "Already in final HOLD; P parks and X stops."
            if self.stage == "WAIT_STAND":
                return False, (
                    "Final Cartesian target has not settled; Enter ignored."
                )
            return False, f"{self.stage} is still moving; Enter ignored."
        if now - self.wait_since < base.WAIT_DWELL_S:
            return False, "Wait for the crouch hold to settle before Enter."
        if not healthy:
            return False, "Motor latch/fault present; STAND refused."

        error = np.abs(np.asarray(q) - Q_RECORDED_CROUCH)
        worst = int(np.argmax(error))
        if error[worst] > RECORDED_POSE_TOL:
            return False, (
                f"Recorded crouch not reached: {JOINT_LABELS[worst]} error "
                f"{error[worst]:.2f} rad > {RECORDED_POSE_TOL:.2f}."
            )
        max_speed = float(np.max(np.abs(qd)))
        if max_speed > RECORDED_QD_TOL:
            return False, (
                f"Robot not settled: max |qd| {max_speed:.2f} rad/s > "
                f"{RECORDED_QD_TOL:.2f}."
            )

        self.stage = "STAND"
        self.started_at = float(now)
        self.wait_since = None
        return True, "ENTER accepted: starting Cartesian STAND."

    def request_park(self, now, healthy):
        if self.stage not in ("HOLD", "HOLD_PARTIAL"):
            return False, (
                "PARK is available only in HOLD/HOLD_PARTIAL; "
                f"stage={self.stage}."
            )
        if not healthy:
            return False, "PARK refused: motor latch/fault present."

        self.stage = "PARK"
        self.started_at = float(now)
        self.wait_since = None
        self.park_settle_since = None
        return True, "P accepted: starting Cartesian PARK back to crouch."


def validate_recorded_configuration(controller):
    base.validate_hardware_config()
    polished_deg = base._stack_pose(POLISHED_CROUCH_DEG).reshape(4, 3)
    expected_polished_deg = np.array(
        [
            [88.33, 48.04, -142.11],
            [-88.33, -48.04, 142.11],
            [88.33, -48.04, 142.11],
            [-88.33, 48.04, -142.11],
        ]
    )
    if not np.allclose(polished_deg, expected_polished_deg):
        raise ValueError("polished crouch is not the expected mirrored pose")

    low, high = base.soft_limits()
    outside = (Q_RECORDED_CROUCH < low) | (Q_RECORDED_CROUCH > high)
    if np.any(outside):
        index = int(np.flatnonzero(outside)[0])
        raise ValueError(
            f"recorded {JOINT_LABELS[index]}="
            f"{Q_RECORDED_CROUCH[index]:+.3f} rad is outside its soft limit"
        )

    expected_motoroutput = {
        1: 88.33,
        2: 48.04,
        3: 142.11,
        4: -88.33,
        5: -48.04,
        6: -142.11,
        7: 88.33,
        8: 48.04,
        9: 142.11,
        10: -88.33,
        11: -48.04,
        12: -142.11,
    }
    actual_motoroutput = base.joint_rad_to_motoroutput_deg(Q_RECORDED_CROUCH)
    for index, mid in enumerate(MOTOR_IDS):
        if not np.isclose(actual_motoroutput[index], expected_motoroutput[mid]):
            raise ValueError(
                f"recorded direction conversion failed for CAN {mid}: "
                f"{actual_motoroutput[index]:+.2f} != "
                f"{expected_motoroutput[mid]:+.2f} deg"
            )

    crouch_feet = np.stack(
        [controller.crouch_foot[leg] for leg in LEGS], axis=0
    )
    left_mean_z = float(np.mean(crouch_feet[[0, 2], 2]))
    right_mean_z = float(np.mean(crouch_feet[[1, 3], 2]))
    if abs(left_mean_z - right_mean_z) > 5.0e-4:
        raise ValueError(
            "polished crouch retains excessive left/right FK height bias: "
            f"{1000.0 * (left_mean_z - right_mean_z):+.3f} mm"
        )
    pair_height_error = max(
        abs(float(crouch_feet[0, 2] - crouch_feet[1, 2])),
        abs(float(crouch_feet[2, 2] - crouch_feet[3, 2])),
    )
    if pair_height_error > 5.0e-4:
        raise ValueError(
            "polished crouch mirrored-pair FK heights differ by "
            f"{1000.0 * pair_height_error:.3f} mm"
        )

    for leg_index, leg in enumerate(LEGS):
        section = slice(3 * leg_index, 3 * leg_index + 3)
        singular = np.min(
            np.linalg.svd(
                dog5_kinematics.foot_jacobian(
                    leg, Q_RECORDED_CROUCH[section]
                ),
                compute_uv=False,
            )
        )
        if singular < MIN_JACOBIAN_SINGULAR:
            raise ValueError(
                f"recorded {leg} crouch is too close to a singularity: "
                f"sigma_min={singular:.4f}"
            )
        if not np.all(np.isfinite(controller.final_foot[leg])):
            raise ValueError(f"non-finite final foot target for {leg}")

        # Follow the complete Cartesian target path with damped least-squares
        # IK.  This is validation only; the hardware controller remains J^T F.
        # Starting each step from the previous solution checks a continuous,
        # in-limit branch exists from this recorded pose to the final stance.
        q_path = Q_RECORDED_CROUCH[section].copy()
        path_segments = (
            (
                "clearance",
                controller.crouch_foot[leg],
                controller.clearance_foot[leg],
            ),
            (
                "stand",
                controller.clearance_foot[leg],
                controller.final_foot[leg],
            ),
        )
        for segment_name, segment_start, segment_end in path_segments:
            for progress in np.linspace(0.05, 1.0, 20):
                target = segment_start + progress * (
                    segment_end - segment_start
                )
                for _ in range(50):
                    error = target - dog5_kinematics.foot_position(leg, q_path)
                    jacobian = dog5_kinematics.foot_jacobian(leg, q_path)
                    q_path += 0.4 * jacobian.T @ np.linalg.solve(
                        jacobian @ jacobian.T + 1.0e-5 * np.eye(3), error
                    )
                residual = np.linalg.norm(
                    target - dog5_kinematics.foot_position(leg, q_path)
                )
                path_singular = np.min(
                    np.linalg.svd(
                        dog5_kinematics.foot_jacobian(leg, q_path),
                        compute_uv=False,
                    )
                )
                leg_low, leg_high = low[section], high[section]
                if residual > 1.0e-4:
                    raise ValueError(
                        f"{leg} {segment_name} path is not reachable at "
                        f"progress {progress:.2f}: residual={residual:.6f} m"
                    )
                if np.any(q_path < leg_low) or np.any(q_path > leg_high):
                    raise ValueError(
                        f"{leg} {segment_name} path leaves a joint limit at "
                        f"progress {progress:.2f}: "
                        f"q={np.rad2deg(q_path)} deg"
                    )
                if path_singular < MIN_JACOBIAN_SINGULAR:
                    raise ValueError(
                        f"{leg} {segment_name} path approaches a singularity "
                        f"at progress {progress:.2f}: sigma_min="
                        f"{path_singular:.4f}"
                    )


def position_stage_fault(
    qd_encoder,
    measured_torque,
    velocity_ready,
    speed_trip,
    torque_trip,
):
    measured_torque = np.asarray(measured_torque, dtype=float)
    if np.any(np.abs(measured_torque) > torque_trip):
        index = int(np.argmax(np.abs(measured_torque)))
        return (
            f"native position measured-torque trip: {JOINT_LABELS[index]}="
            f"{measured_torque[index]:+.2f} N*m exceeds {torque_trip:.2f} N*m"
        )
    if velocity_ready:
        qd_encoder = np.asarray(qd_encoder, dtype=float)
        if np.any(np.abs(qd_encoder) > speed_trip):
            index = int(np.argmax(np.abs(qd_encoder)))
            return (
                f"native position encoder-speed trip: {JOINT_LABELS[index]}="
                f"{qd_encoder[index]:+.2f} rad/s exceeds "
                f"{speed_trip:.2f} rad/s"
            )
    return None


def zero_torque_preflight(mb, key, unwrap):
    """Read the current stationary pose before any nonzero torque command."""
    print(
        "[recorded] ZERO-TORQUE CHECK: support the robot with every leg clear.\n"
        "[recorded] The table shows current model joint degrees and error to "
        "the recorded crouch.\n[recorded] Press ENTER to move current -> "
        "recorded CROUCH; X aborts."
    )
    slot = mb.slot(base.CONTROL_HZ)
    deadline = time.perf_counter() + slot
    last_print = 0.0
    index = 0
    started_at = time.perf_counter()
    next_status = {
        mid: started_at + offset / (base.FAULT_STATUS_HZ * N_JOINTS)
        for offset, mid in enumerate(MOTOR_IDS)
    }
    last_recover = {mid: -base.RECOVER_PERIOD_S for mid in MOTOR_IDS}
    velocity = EncoderVelocity()
    qd_encoder = np.zeros(N_JOINTS)

    while True:
        mb.poll()
        mid = MOTOR_IDS[index % N_JOINTS]
        now = time.perf_counter()
        rec = mb.rec(mid)
        elapsed = now - started_at
        if (
            (rec.error or 0) & 0x80
            and elapsed - last_recover[mid] >= base.RECOVER_PERIOD_S
        ):
            base._recover_input_lost(mb, [mid], elapsed, last_recover)
            next_status[mid] = now + 1.0 / base.FAULT_STATUS_HZ
        elif now >= next_status[mid]:
            mb.status1_req(mid)
            next_status[mid] = now + 1.0 / base.FAULT_STATUS_HZ
        else:
            mb.keepalive(mid)

        q, qd_driver = base._joint_state(mb, unwrap)
        if index % N_JOINTS == 0:
            qd_encoder = velocity.update(q, now)
        if now - last_print >= 1.0 / base.STATUS_HZ:
            groups = []
            for leg_index, leg in enumerate(LEGS):
                section = slice(3 * leg_index, 3 * leg_index + 3)
                current_deg = np.rad2deg(q[section])
                error_deg = np.rad2deg(Q_RECORDED_CROUCH[section] - q[section])
                groups.append(
                    f"{leg} q=({current_deg[0]:+6.1f},{current_deg[1]:+6.1f},"
                    f"{current_deg[2]:+6.1f}) "
                    f"err=({error_deg[0]:+6.1f},{error_deg[1]:+6.1f},"
                    f"{error_deg[2]:+6.1f})"
                )
            print("[current] " + "  ".join(groups), flush=True)
            last_print = now

        pressed = key.get()
        if pressed in ("x", "X"):
            raise KeyboardInterrupt("operator X")
        if base._is_enter(pressed):
            unverified = [
                motor_id
                for motor_id in MOTOR_IDS
                if mb.rec(motor_id).error is None
                or (mb.rec(motor_id).error & 0x80)
            ]
            max_speed = float(np.max(np.abs(qd_encoder)))
            if unverified:
                print(
                    "[recorded] ENTER refused: waiting for fresh clear status "
                    f"from CAN {unverified}."
                )
            elif not velocity.ready:
                print(
                    "[recorded] ENTER refused: collecting stationary encoder "
                    "samples."
                )
            elif max_speed > RECORDED_QD_TOL:
                print(
                    f"[recorded] ENTER refused: max encoder |qd|="
                    f"{max_speed:.2f} "
                    "rad/s; hold the robot still."
                )
            else:
                low, high = base.soft_limits()
                outside = (q < low) | (q > high)
                if np.any(outside):
                    details = ", ".join(
                        f"{JOINT_LABELS[joint_index]}="
                        f"{q[joint_index]:+.2f}rad"
                        for joint_index in np.flatnonzero(outside)
                    )
                    print(
                        "[recorded] WARNING: accepting current pose outside "
                        f"soft limits: {details}. The native position target "
                        "is the in-limit recorded crouch."
                    )
                return q.copy()

        index += 1
        mb.pace(deadline)
        deadline += slot


def run_hardware(
    tau_max,
    cart_gain_scale,
    support_scale,
    stand_height,
    travel_scale,
    crouch_max_speed_dps,
    crouch_torque_trip,
    crouch_speed_trip,
    qd_estop=base.QD_ESTOP,
    qd_estop_hard=base.QD_ESTOP_HARD,
):
    controller = RecordedCrouchController(
        cart_gain_scale, support_scale, stand_height, travel_scale
    )
    validate_recorded_configuration(controller)
    gate = base.SafetyGate(tau_max, qd_estop, qd_estop_hard)
    unwrap = [base.CalibratedEncoderUnwrap() for _ in base.HARDWARE_JOINTS]

    print(
        "[recorded] Flow: CURRENT -> CROUCH -> ENTER -> STAND -> "
        "WAIT_STAND -> HOLD/HOLD_PARTIAL -> P -> PARK -> WAIT_PARK -> "
        "WAIT_CROUCH"
    )
    print(
        "[fixed-fend] polished symmetric crouch joint targets deg: "
        + ", ".join(
            f"{leg}={RECORDED_CROUCH_DEG[leg]}" for leg in LEGS
        )
    )
    print(
        f"[recorded] native position CROUCH: speed cap="
        f"{crouch_max_speed_dps:.0f} motor-deg/s, measured torque trip="
        f"{crouch_torque_trip:.2f}N*m, encoder speed trip="
        f"{crouch_speed_trip:.2f}rad/s, timeout={CROUCH_TIMEOUT_S:.0f}s"
    )
    print(
        f"[recorded] Cartesian STAND time={T_STAND:.1f}s; "
        f"height={stand_height:.3f}m, tau cap={tau_max:.2f}N*m, "
        f"travel={travel_scale:.2f}, cart scale={cart_gain_scale:.2f}, "
        f"support scale={support_scale:.2f}"
    )
    print(
        "[fixed-fend] Each foot keeps its polished crouch x/y target; "
        "only trunk-frame z changes during STAND/PARK."
    )
    print(
        "[recorded] NOTE: native 0xA4 position control has a speed cap but "
        "no commanded torque-cap field; crouch torque is a measured-feedback "
        "trip."
    )
    for leg in LEGS:
        print(
            f"[recorded] {leg} foot: crouch="
            f"{np.array2string(controller.crouch_foot[leg], precision=4)} -> "
            f"stand={np.array2string(controller.final_foot[leg], precision=4)}"
        )
    if travel_scale < 1.0:
        final_heights = [
            -float(controller.final_foot[leg][2]) for leg in LEGS
        ]
        print(
            "[recorded] PARTIAL TEST: --travel-scale "
            f"{travel_scale:.2f} commands only {100.0 * travel_scale:.0f}% "
            f"of the stand path (final leg heights "
            f"{min(final_heights):.3f}..{max(final_heights):.3f} m). "
            "It is not expected to stand fully."
        )
    print(
        "[recorded] ROBOT MUST REMAIN MECHANICALLY SUPPORTED. "
        "P parks from HOLD; X stops."
    )

    key = base.KeyPoller()
    try:
        with base.motorbus.MotorBus(
            MOTOR_IDS, dirs=MOTOR_DIRECTIONS
        ) as mb:
            armed = False
            stop_reason = None
            try:
                print("[recorded] Arming with a zero-torque stream...")
                if not mb.arm(rate_hz=base.CONTROL_HZ):
                    raise RuntimeError("not all motors armed")
                armed = True

                start_q = zero_torque_preflight(mb, key, unwrap)
                now = time.perf_counter()
                sequence = RecordedStandSequence(now, start_q, travel_scale)
                gate.start(now, start_q)
                print(
                    "[recorded] ENTER accepted: native position control moving "
                    "CURRENT -> recorded CROUCH. Do not press Enter while moving."
                )

                slot = mb.slot(base.CONTROL_HZ)
                deadline = time.perf_counter() + slot
                start = now
                q = start_q.copy()
                tau_command = np.zeros(N_JOINTS)
                miss_monitor = base.CanMissMonitor(mb)
                status_period = 1.0 / base.FAULT_STATUS_HZ
                next_fault_status = np.asarray(
                    [
                        status_period + i * status_period / N_JOINTS
                        for i in range(N_JOINTS)
                    ]
                )
                last_recover = {
                    mid: -base.RECOVER_PERIOD_S for mid in MOTOR_IDS
                }
                last_print = 0.0
                index = 0
                velocity = EncoderVelocity()

                while True:
                    mb.poll()
                    joint_index = index % N_JOINTS

                    if joint_index == 0:
                        now = time.perf_counter()
                        elapsed = now - start
                        q, qd_driver = base._joint_state(mb, unwrap)
                        qd_encoder = velocity.update(q, now)

                        event = sequence.update(
                            now,
                            q=q,
                            cart_error=controller.last_cart_error,
                            qd=(qd_encoder if velocity.ready else None),
                        )
                        if event:
                            print(f"[stage] {event}")
                            if sequence.stage == "WAIT_CROUCH":
                                print(
                                    "[stage] Inspect recorded crouch; press "
                                    "ENTER for Cartesian STAND when settled."
                                )
                            elif sequence.stage == "WAIT_STAND":
                                print(
                                    "[stage] Waiting for cart_err <= "
                                    f"{FINAL_CART_TOL_M:.3f} m and "
                                    f"max|qd_enc| <= {FINAL_QD_TOL:.2f} rad/s."
                                )
                            elif sequence.stage == "WAIT_PARK":
                                print(
                                    "[stage] Waiting for crouch cart_err <= "
                                    f"{FINAL_CART_TOL_M:.3f} m and "
                                    f"max|qd_enc| <= {FINAL_QD_TOL:.2f} rad/s."
                                )
                            else:
                                print(
                                    f"[stage] {sequence.stage} active; "
                                    "press P to park or X to stop."
                                )

                        position_stage = sequence.stage in (
                            "CROUCH",
                            "WAIT_CROUCH",
                        )
                        if position_stage:
                            tau_command.fill(0.0)
                            gate.previous_tau.fill(0.0)
                        elif sequence.stage == "STAND":
                            requested_tau = controller.compute_stand(
                                q, qd_encoder, now - sequence.started_at
                            )
                            tau_command = gate.apply(requested_tau, q, now)
                        elif sequence.stage == "PARK":
                            requested_tau = controller.compute_park(
                                q, qd_encoder, now - sequence.started_at
                            )
                            tau_command = gate.apply(requested_tau, q, now)
                        elif sequence.stage == "WAIT_PARK":
                            requested_tau = controller.compute_park(
                                q, qd_encoder, T_PARK
                            )
                            tau_command = gate.apply(requested_tau, q, now)
                        else:
                            requested_tau = controller.compute_stand(
                                q, qd_encoder, T_STAND
                            )
                            tau_command = gate.apply(requested_tau, q, now)

                        temps = base._temperatures(mb)
                        misses = miss_monitor.update(mb)
                        errors = mb.errors()
                        stop_reason = gate.estop_reason(
                            q,
                            qd_driver,
                            temps,
                            misses,
                            errors,
                            now,
                            enforce_position_limits=(
                                sequence.stage
                                in (
                                    "STAND",
                                    "WAIT_STAND",
                                    "HOLD",
                                    "HOLD_PARTIAL",
                                    "PARK",
                                    "WAIT_PARK",
                                )
                            ),
                        )
                        torque_feedback = mb.torques_nm()
                        measured_torque = np.asarray(
                            [torque_feedback[mid] for mid in MOTOR_IDS]
                        )
                        if position_stage and stop_reason is None:
                            stop_reason = position_stage_fault(
                                qd_encoder,
                                measured_torque,
                                velocity.ready,
                                crouch_speed_trip,
                                crouch_torque_trip,
                            )
                        latched = [
                            mid for mid, error in errors.items() if error & 0x80
                        ]
                        unverified = [
                            mid for mid in MOTOR_IDS if mb.rec(mid).error is None
                        ]

                        pressed = key.get()
                        if pressed in ("x", "X"):
                            stop_reason = "operator X"
                        elif pressed in ("p", "P"):
                            _, message = sequence.request_park(
                                now,
                                healthy=(
                                    not latched
                                    and not unverified
                                    and stop_reason is None
                                ),
                            )
                            print(f"[stage] {message}")
                        elif base._is_enter(pressed):
                            _, message = sequence.request_next(
                                now,
                                q,
                                qd_encoder,
                                healthy=(
                                    not latched
                                    and not unverified
                                    and stop_reason is None
                                ),
                            )
                            print(f"[stage] {message}")
                        if stop_reason:
                            break

                        recover = [
                            mid
                            for mid in latched
                            if elapsed - last_recover[mid]
                            >= base.RECOVER_PERIOD_S
                        ]
                        if recover:
                            base._recover_input_lost(
                                mb,
                                recover,
                                elapsed,
                                last_recover,
                                next_fault_status,
                            )
                            latched = []

                        if now - last_print >= 1.0 / base.STATUS_HZ:
                            pose_error = float(
                                np.max(np.abs(q - Q_RECORDED_CROUCH))
                            )
                            low, high = base.soft_limits()
                            outside_count = int(
                                np.count_nonzero((q < low) | (q > high))
                            )
                            actual_heights = []
                            target_heights = []
                            for leg_index, leg in enumerate(LEGS):
                                section = slice(
                                    3 * leg_index, 3 * leg_index + 3
                                )
                                actual_heights.append(
                                    -float(
                                        dog5_kinematics.foot_position(
                                            leg, q[section]
                                        )[2]
                                    )
                                )
                                target_heights.append(
                                    -float(controller.final_foot[leg][2])
                                )
                            print(
                                f"[recorded] stage={sequence.stage:11s} "
                                f"crouch_err={pose_error:.2f}rad "
                                f"cart_err={controller.last_cart_error:.3f}m "
                                f"xy_err={controller.last_xy_error:.3f}m "
                                f"sigma_min={controller.last_min_singular:.4f} "
                                f"force_max={controller.last_max_force:.1f}N "
                                f"h_actual={min(actual_heights):.3f}.."
                                f"{max(actual_heights):.3f}m "
                                f"h_final={min(target_heights):.3f}.."
                                f"{max(target_heights):.3f}m "
                                f"max|qd_enc|={np.max(np.abs(qd_encoder)):.2f} "
                                f"max|qd_driver|="
                                f"{np.max(np.abs(qd_driver)):.2f} "
                                f"safety_qd_enc="
                                f"{np.max(np.abs(gate.encoder_qd)):.2f} "
                                f"max|tau|="
                                f"{np.max(np.abs(tau_command)):.2f}N*m "
                                f"max|tau_fb|="
                                f"{np.max(np.abs(measured_torque)):.2f}N*m "
                                f"Tmax={int(np.max(temps))}C "
                                f"outside_soft={outside_count} "
                                f"latched={len(latched)}",
                                flush=True,
                            )
                            last_print = now

                    mid = MOTOR_IDS[joint_index]
                    elapsed = time.perf_counter() - start
                    if elapsed >= next_fault_status[joint_index]:
                        mb.status1_req(mid)
                        next_fault_status[joint_index] += status_period
                    elif sequence.stage in ("CROUCH", "WAIT_CROUCH"):
                        if not mb.position(
                            mid,
                            float(POSITION_TARGET_DEG[joint_index]),
                            crouch_max_speed_dps,
                        ):
                            raise RuntimeError(
                                f"CAN position transmit failed for CAN {mid}"
                            )
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
                    print(f"[recorded] stopping: {stop_reason or 'aborted'}")
                    try:
                        base._soft_stop(mb)
                    except Exception as exc:
                        print(
                            f"[recorded] soft stop failed: {exc}; sending STOP",
                            file=sys.stderr,
                        )
    finally:
        key.close()
    return 0


def offline_self_test(stand_height=DEFAULT_STAND_HEIGHT, travel_scale=1.0):
    controller = RecordedCrouchController(
        0.25, 0.25, stand_height, travel_scale
    )
    validate_recorded_configuration(controller)
    zero = np.zeros(N_JOINTS)

    start_q = 0.5 * Q_RECORDED_CROUCH
    assert np.allclose(POSITION_TARGET_DEG, np.rad2deg(Q_RECORDED_CROUCH))

    velocity = EncoderVelocity()
    for sample in range(7):
        estimated = velocity.update(start_q, sample * 0.05)
    assert velocity.ready
    assert np.allclose(estimated, 0.0)

    # The recorded runner may start outside the normal knee limit while native
    # position control moves toward its in-limit target.  The shared torque
    # gate still blocks outward Cartesian torque, and position-limit e-stop is
    # enabled again before Cartesian STAND.
    rl_knee = JOINT_LABELS.index("RL_knee")
    outside_q = Q_RECORDED_CROUCH.copy()
    outside_q[rl_knee] = 2.68
    safety = base.SafetyGate(1.0)
    safety.start(0.0, outside_q)
    inward_request = np.zeros(N_JOINTS)
    inward_request[rl_knee] = -1.0
    inward_tau = safety.apply(inward_request, outside_q, 0.1)
    assert inward_tau[rl_knee] < 0.0
    safety.previous_tau.fill(0.0)
    outward_request = np.zeros(N_JOINTS)
    outward_request[rl_knee] = 1.0
    outward_tau = safety.apply(outward_request, outside_q, 0.2)
    assert outward_tau[rl_knee] == 0.0
    no_fault = safety.estop_reason(
        outside_q,
        zero,
        np.zeros(N_JOINTS),
        np.zeros(N_JOINTS, dtype=int),
        {mid: 0 for mid in MOTOR_IDS},
        0.3,
        enforce_position_limits=False,
    )
    assert no_fault is None
    limit_fault = safety.estop_reason(
        outside_q,
        zero,
        np.zeros(N_JOINTS),
        np.zeros(N_JOINTS, dtype=int),
        {mid: 0 for mid in MOTOR_IDS},
        0.4,
        enforce_position_limits=True,
    )
    assert limit_fault and "RL_knee" in limit_fault

    torque_feedback = np.zeros(N_JOINTS)
    torque_feedback[rl_knee] = DEFAULT_CROUCH_TORQUE_TRIP_NM + 0.01
    torque_fault = position_stage_fault(
        zero,
        torque_feedback,
        True,
        DEFAULT_CROUCH_SPEED_TRIP_RAD_S,
        DEFAULT_CROUCH_TORQUE_TRIP_NM,
    )
    assert torque_fault and "RL_knee" in torque_fault
    speed_feedback = np.zeros(N_JOINTS)
    speed_feedback[rl_knee] = DEFAULT_CROUCH_SPEED_TRIP_RAD_S + 0.01
    speed_fault = position_stage_fault(
        speed_feedback,
        zero,
        True,
        DEFAULT_CROUCH_SPEED_TRIP_RAD_S,
        DEFAULT_CROUCH_TORQUE_TRIP_NM,
    )
    assert speed_fault and "RL_knee" in speed_fault

    cart_start = controller.compute_stand(
        Q_RECORDED_CROUCH, zero, 0.0
    )
    assert np.allclose(cart_start, 0.0, atol=1.0e-10)
    assert np.isclose(controller.last_xy_error, 0.0)
    for leg in LEGS:
        target0, velocity0, progress0 = controller.cartesian_target(leg, 0.0)
        target1, velocity1, progress1 = controller.cartesian_target(
            leg, T_STAND
        )
        assert np.allclose(target0, controller.crouch_foot[leg])
        assert np.allclose(target1, controller.final_foot[leg])
        assert np.allclose(
            controller.full_stand_foot[leg][:2],
            controller.crouch_foot[leg][:2],
        )
        assert np.allclose(
            controller.final_foot[leg][:2],
            controller.crouch_foot[leg][:2],
        )
        assert np.allclose(
            controller.clearance_foot[leg][:2],
            controller.crouch_foot[leg][:2],
        )
        assert np.allclose(velocity0, 0.0)
        assert np.allclose(velocity1, 0.0)
        assert progress0 == 0.0 and progress1 == 1.0

        for elapsed in np.linspace(0.0, T_STAND, 21):
            target, _, _ = controller.cartesian_target(leg, elapsed)
            assert np.allclose(target[:2], controller.crouch_foot[leg][:2])

        park0, park_velocity0, park_progress0 = controller.parking_target(
            leg, 0.0
        )
        park1, park_velocity1, park_progress1 = controller.parking_target(
            leg, T_PARK
        )
        assert np.allclose(park0, target1)
        assert np.allclose(park1, target0)
        assert np.allclose(park_velocity0, 0.0)
        assert np.allclose(park_velocity1, 0.0)
        assert park_progress0 == 1.0 and park_progress1 == 0.0
    park_end_tau = controller.compute_park(Q_RECORDED_CROUCH, zero, T_PARK)
    assert np.allclose(park_end_tau, 0.0, atol=1.0e-10)

    sequence = RecordedStandSequence(0.0, start_q, travel_scale)
    assert sequence.update(
        0.10, q=Q_RECORDED_CROUCH, qd=zero
    ) is None
    assert sequence.update(
        0.10 + CROUCH_SETTLE_S + 0.01,
        q=Q_RECORDED_CROUCH,
        qd=zero,
    )
    assert sequence.stage == "WAIT_CROUCH"
    advanced, reason = sequence.request_next(
        sequence.wait_since + base.WAIT_DWELL_S + 0.02,
        Q_RECORDED_CROUCH,
        zero,
        True,
    )
    assert advanced, reason
    assert sequence.stage == "STAND"
    stand_end = sequence.started_at + T_STAND + 0.01
    assert sequence.update(
        stand_end, q=Q_RECORDED_CROUCH, cart_error=0.0, qd=zero
    )
    assert sequence.stage == "WAIT_STAND"
    assert sequence.update(
        stand_end + 0.01,
        q=Q_RECORDED_CROUCH,
        cart_error=0.05,
        qd=zero,
    ) is None
    assert sequence.final_settle_since is None
    settle_start = stand_end + 0.02
    assert sequence.update(
        settle_start,
        q=Q_RECORDED_CROUCH,
        cart_error=0.0,
        qd=zero,
    ) is None
    assert sequence.update(
        settle_start + FINAL_DWELL_S + 0.01,
        q=Q_RECORDED_CROUCH,
        cart_error=0.0,
        qd=zero,
    )
    expected_hold = "HOLD" if travel_scale >= 1.0 else "HOLD_PARTIAL"
    assert sequence.stage == expected_hold
    parked, reason = sequence.request_park(
        settle_start + FINAL_DWELL_S + 0.02, False
    )
    assert not parked and "fault" in reason
    parked, reason = sequence.request_park(
        settle_start + FINAL_DWELL_S + 0.03, True
    )
    assert parked, reason
    assert sequence.stage == "PARK"
    park_end = sequence.started_at + T_PARK + 0.01
    assert sequence.update(
        park_end, q=Q_RECORDED_CROUCH, cart_error=0.0, qd=zero
    )
    assert sequence.stage == "WAIT_PARK"
    assert sequence.update(
        park_end + 0.01,
        q=Q_RECORDED_CROUCH,
        cart_error=0.05,
        qd=zero,
    ) is None
    assert sequence.park_settle_since is None
    park_settle_start = park_end + 0.02
    assert sequence.update(
        park_settle_start,
        q=Q_RECORDED_CROUCH,
        cart_error=0.0,
        qd=zero,
    ) is None
    assert sequence.update(
        park_settle_start + FINAL_DWELL_S + 0.01,
        q=Q_RECORDED_CROUCH,
        cart_error=0.0,
        qd=zero,
    )
    assert sequence.stage == "WAIT_CROUCH"

    print("stand_dog5_fixed_fend_hw offline self-test PASS")
    print("  raw recorded joint targets deg:", RAW_RECORDED_CROUCH_DEG)
    print("  polished symmetric targets deg:", POLISHED_CROUCH_DEG)
    raw_q = np.deg2rad(base._stack_pose(RAW_RECORDED_CROUCH_DEG))
    raw_feet = np.stack(
        [
            dog5_kinematics.foot_position(
                leg, raw_q[3 * leg_index:3 * leg_index + 3]
            )
            for leg_index, leg in enumerate(LEGS)
        ],
        axis=0,
    )
    polished_feet = np.stack(
        [controller.crouch_foot[leg] for leg in LEGS], axis=0
    )
    raw_roll_bias = float(
        np.mean(raw_feet[[0, 2], 2]) - np.mean(raw_feet[[1, 3], 2])
    )
    polished_roll_bias = float(
        np.mean(polished_feet[[0, 2], 2])
        - np.mean(polished_feet[[1, 3], 2])
    )
    print(
        "  modelled left/right crouch height bias: "
        f"raw={1000.0 * raw_roll_bias:+.3f}mm -> "
        f"polished={1000.0 * polished_roll_bias:+.3f}mm"
    )
    print(
        "  expected motoroutput deg:",
        {
            joint.can_id: float(value)
            for joint, value in zip(
                base.HARDWARE_JOINTS,
                base.joint_rad_to_motoroutput_deg(Q_RECORDED_CROUCH),
            )
        },
    )
    for leg in LEGS:
        section = slice(3 * LEGS.index(leg), 3 * LEGS.index(leg) + 3)
        singular = np.min(
            np.linalg.svd(
                dog5_kinematics.foot_jacobian(
                    leg, Q_RECORDED_CROUCH[section]
                ),
                compute_uv=False,
            )
        )
        print(
            f"  {leg}: crouch foot="
            f"{np.array2string(controller.crouch_foot[leg], precision=4)}, "
            f"final={np.array2string(controller.final_foot[leg], precision=4)}, "
            f"sigma_min={singular:.4f}"
        )
    print("  STAND transition torque at t=0: zero")
    print("  STAND/PARK foot x/y targets: fixed at polished crouch values")
    print("  PARK begins at the HOLD target and ends at zero crouch torque")
    print(
        "  flow: CURRENT -> CROUCH -> ENTER -> STAND -> WAIT_STAND -> "
        f"{expected_hold} -> P -> PARK -> WAIT_PARK -> WAIT_CROUCH"
    )
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tau-max",
        type=float,
        default=base.DEFAULT_TAU_MAX,
        help=(
            "Cartesian STAND per-joint torque cap, N*m "
            f"(default: {base.DEFAULT_TAU_MAX})"
        ),
    )
    parser.add_argument(
        "--cart-gain-scale",
        type=float,
        default=base.DEFAULT_CART_GAIN_SCALE,
        help=(
            "Cartesian Kp/Kd scale in (0,1] "
            f"(default: {base.DEFAULT_CART_GAIN_SCALE})"
        ),
    )
    parser.add_argument(
        "--support-scale",
        type=float,
        default=base.DEFAULT_SUPPORT_SCALE,
        help=(
            "full-path mg/4 feedforward scale in [0,1]; partial travel "
            "scales it proportionally "
            f"(default: {base.DEFAULT_SUPPORT_SCALE})"
        ),
    )
    parser.add_argument(
        "--stand-height",
        type=float,
        default=DEFAULT_STAND_HEIGHT,
        help=(
            "final trunk-frame hip-to-foot height, m "
            f"(default: {DEFAULT_STAND_HEIGHT})"
        ),
    )
    parser.add_argument(
        "--travel-scale",
        type=float,
        default=1.0,
        help=(
            "fraction of the complete recorded-to-stand Cartesian path; "
            "use 0.25 for the first supported test (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--crouch-max-speed-dps",
        type=float,
        default=DEFAULT_CROUCH_MAX_MOTOR_DPS,
        help=(
            "native position-command motor-side speed cap, deg/s; output "
            "speed is approximately one tenth "
            f"(default: {DEFAULT_CROUCH_MAX_MOTOR_DPS})"
        ),
    )
    parser.add_argument(
        "--crouch-torque-trip",
        type=float,
        default=DEFAULT_CROUCH_TORQUE_TRIP_NM,
        help=(
            "stop native position motion when measured joint torque exceeds "
            f"this value, N*m (default: {DEFAULT_CROUCH_TORQUE_TRIP_NM})"
        ),
    )
    parser.add_argument(
        "--crouch-speed-trip",
        type=float,
        default=DEFAULT_CROUCH_SPEED_TRIP_RAD_S,
        help=(
            "stop native position motion when encoder-derived joint speed "
            "exceeds this value, rad/s "
            f"(default: {DEFAULT_CROUCH_SPEED_TRIP_RAD_S})"
        ),
    )
    parser.add_argument(
        "--qd-estop",
        type=float,
        default=base.QD_ESTOP,
        help=f"confirmed sustained speed trip, rad/s (default: {base.QD_ESTOP})",
    )
    parser.add_argument(
        "--qd-estop-hard",
        type=float,
        default=base.QD_ESTOP_HARD,
        help=(
            "single-sample driver speed trip, rad/s "
            f"(default: {base.QD_ESTOP_HARD})"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate the recorded mapping, targets, and sequence without CAN",
    )
    args = parser.parse_args()

    if not 0.0 < args.tau_max <= base.STAGED_TAU_MAX:
        parser.error(
            f"--tau-max must be > 0 and <= {base.STAGED_TAU_MAX} N*m"
        )
    if not 0.0 < args.cart_gain_scale <= 1.0:
        parser.error("--cart-gain-scale must be > 0 and <= 1")
    if not 0.0 <= args.support_scale <= 1.0:
        parser.error("--support-scale must be between 0 and 1")
    if not MIN_STAND_HEIGHT <= args.stand_height <= MAX_STAND_HEIGHT:
        parser.error(
            f"--stand-height must be between {MIN_STAND_HEIGHT} and "
            f"{MAX_STAND_HEIGHT} m"
        )
    if not 0.05 <= args.travel_scale <= 1.0:
        parser.error("--travel-scale must be between 0.05 and 1.0")
    if not 1.0 <= args.crouch_max_speed_dps <= 600.0:
        parser.error("--crouch-max-speed-dps must be between 1 and 600")
    if not 0.1 <= args.crouch_torque_trip <= base.STAGED_TAU_MAX:
        parser.error(
            f"--crouch-torque-trip must be between 0.1 and "
            f"{base.STAGED_TAU_MAX} N*m"
        )
    if not 0.1 <= args.crouch_speed_trip <= 3.0:
        parser.error("--crouch-speed-trip must be between 0.1 and 3.0 rad/s")
    if not 0.0 < args.qd_estop <= base.QD_ESTOP_CEILING:
        parser.error(
            f"--qd-estop must be > 0 and <= {base.QD_ESTOP_CEILING} rad/s"
        )
    if not args.qd_estop < args.qd_estop_hard <= base.QD_ESTOP_CEILING:
        parser.error(
            "--qd-estop-hard must be above --qd-estop and "
            f"<= {base.QD_ESTOP_CEILING} rad/s"
        )
    if args.self_test:
        return offline_self_test(args.stand_height, args.travel_scale)

    try:
        return run_hardware(
            args.tau_max,
            args.cart_gain_scale,
            args.support_scale,
            args.stand_height,
            args.travel_scale,
            args.crouch_max_speed_dps,
            args.crouch_torque_trip,
            args.crouch_speed_trip,
            args.qd_estop,
            args.qd_estop_hard,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[recorded] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
