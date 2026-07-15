#!/usr/bin/env python3
"""Stand DOG5 using only native joint-position commands for motion.

The operator-gated hardware sequence is:

    READ -> CROUCH -> WAIT_CROUCH -> STAND -> HOLD -> PARK -> PARKED

``READ`` streams zero torque while displaying the calibrated joint angles.
The first Enter moves from the measured pose to the recorded crouch.  After
the crouch is reached and stationary, inspect the robot and press Enter again
to move to the supplied stand pose.  At HOLD, Enter parks the robot back in
the recorded crouch.  ``X`` or Ctrl-C stops at any time.

All movements use the motor driver's native multi-turn ``0xA4`` command. Targets
are model joint angles; ``motorbus.MotorBus.position`` applies the configured
CAN direction exactly once.  The native position command has a speed field but
no commanded torque-limit field, so the torque option below is a measured
feedback trip, not a driver-side torque cap.

The robot must remain mechanically supported.  This encoder-only program has
no trunk-attitude or foot-contact feedback and is not a balance controller.

First checks::

    python stand_by_position_command.py --self-test
    python stand_by_position_command.py \
        --crouch-max-speed-dps 100 --stand-max-speed-dps 100 \
        --position-torque-trip 2.0 --position-speed-trip 1.0
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import stand_dog5_hw as base


LEGS = base.LEGS
MOTOR_IDS = base.MOTOR_IDS
MOTOR_DIRECTIONS = base.MOTOR_DIRECTIONS
JOINT_LABELS = base.JOINT_LABELS
N_JOINTS = base.N_JOINTS

# Model-positive joint angles in canonical controller order:
# [FL, FR, RL, RR] x [hip abduction, hip pitch, knee].
RECORDED_CROUCH_DEG = {
    "FL": (90.74, 47.59, -131.02),
    "FR": (-90.39, -51.92, 138.39),
    "RL": (90.23, -44.45, 128.40),
    "RR": (-93.95, 47.06, -125.12),
}

STAND_POSE_DEG = {
    "FL": (77.0, -8.0, -65.0),
    "FR": (-77.0, 8.0, 65.0),
    "RL": (85.0, 10.0, 65.0),
    "RR": (-85.0, -10.0, -65.0),
}

Q_CROUCH = np.deg2rad(base._stack_pose(RECORDED_CROUCH_DEG))
Q_STAND = np.deg2rad(base._stack_pose(STAND_POSE_DEG))
CROUCH_TARGET_DEG = np.rad2deg(Q_CROUCH)
STAND_TARGET_DEG = np.rad2deg(Q_STAND)

DEFAULT_CROUCH_MAX_MOTOR_DPS = 100.0
DEFAULT_STAND_MAX_MOTOR_DPS = 100.0
DEFAULT_POSITION_TORQUE_TRIP_NM = 2.0
DEFAULT_POSITION_SPEED_TRIP_RAD_S = 1.0
ARM_TIMEOUT_S = 15.0
POSITION_TIMEOUT_S = 30.0
POSITION_POSE_TOL = 0.08
POSITION_QD_TOL = 0.25
POSITION_SETTLE_S = 0.50


def format_joint_angles(q):
    """Return compact FL/FR/RL/RR joint angles in degrees."""
    q_deg = np.rad2deg(np.asarray(q, dtype=float))
    groups = []
    for leg_index, leg in enumerate(LEGS):
        abd, pitch, knee = q_deg[3 * leg_index:3 * leg_index + 3]
        groups.append(f"{leg}=({abd:+6.1f},{pitch:+6.1f},{knee:+6.1f})")
    return "  ".join(groups)


def stage_instruction(stage):
    """Short operator instruction displayed beside each stage."""
    return {
        "READ": "Keep still; ENTER starts CROUCH, X stops",
        "CROUCH": "Moving to recorded crouch; wait, X stops",
        "WAIT_CROUCH": "Inspect crouch; ENTER starts STAND, X stops",
        "STAND": "Moving to supplied stand pose; wait, X stops",
        "HOLD": "Holding stand; ENTER starts PARK, X stops",
        "PARK": "Returning stand to recorded crouch; wait, X stops",
        "PARKED": "Holding parked crouch; X stops",
    }[stage]


def print_pose(stage, q):
    print(
        f"[{stage}] {stage_instruction(stage)} | {format_joint_angles(q)}",
        flush=True,
    )


def arm_failure_summary(mb):
    """Summarize cached per-motor state after a quiet arming timeout."""
    missing = []
    unverified = []
    latched = []
    faulted = []
    for mid in MOTOR_IDS:
        rec = mb.rec(mid)
        if rec.last_reply_t is None:
            missing.append(mid)
        elif rec.error is None:
            unverified.append(mid)
        elif rec.error & 0x80:
            latched.append(mid)
        elif rec.error & 0x7F:
            faulted.append((mid, rec.error))

    details = []
    if missing:
        details.append(f"no reply CAN {missing}")
    if unverified:
        details.append(f"no fresh status CAN {unverified}")
    if latched:
        details.append(f"input-timeout latch CAN {latched}")
    if faulted:
        details.append(
            "motor fault "
            + ", ".join(f"CAN {mid}=0x{error:02x}" for mid, error in faulted)
        )
    return "; ".join(details) or "motors did not all enter READY"


class EncoderVelocity:
    """Low-pass velocity estimate from unwrapped calibrated joint angles."""

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


class PositionStandSequence:
    """Track CROUCH, STAND, HOLD, PARK, and final PARKED state."""

    def __init__(self, now):
        self.stage = "CROUCH"
        self.started_at = float(now)
        self.wait_since = None
        self.settle_since = None

    @property
    def target_q(self):
        if self.stage in ("CROUCH", "WAIT_CROUCH", "PARK", "PARKED"):
            return Q_CROUCH
        return Q_STAND

    @property
    def target_deg(self):
        if self.stage in ("CROUCH", "WAIT_CROUCH", "PARK", "PARKED"):
            return CROUCH_TARGET_DEG
        return STAND_TARGET_DEG

    def update(self, now, q=None, qd=None):
        if self.stage not in ("CROUCH", "STAND", "PARK"):
            return None
        if now - self.started_at > POSITION_TIMEOUT_S:
            raise RuntimeError(
                f"{self.stage} timed out after {POSITION_TIMEOUT_S:.1f} s"
            )
        if q is None or qd is None:
            self.settle_since = None
            return None

        pose_error = float(
            np.max(np.abs(np.asarray(q, dtype=float) - self.target_q))
        )
        max_speed = float(np.max(np.abs(np.asarray(qd, dtype=float))))
        settled = (
            pose_error <= POSITION_POSE_TOL
            and max_speed <= POSITION_QD_TOL
        )
        if not settled:
            self.settle_since = None
            return None
        if self.settle_since is None:
            self.settle_since = float(now)
            return None
        if now - self.settle_since < POSITION_SETTLE_S:
            return None

        self.settle_since = None
        self.wait_since = float(now)
        if self.stage == "CROUCH":
            self.stage = "WAIT_CROUCH"
            return "CROUCH reached and settled; holding recorded crouch."
        if self.stage == "STAND":
            self.stage = "HOLD"
            return "STAND reached and settled; holding supplied stand pose."
        self.stage = "PARKED"
        return "PARK reached and settled; holding recorded crouch."

    def request_next(self, now, q, qd, healthy):
        if self.stage not in ("WAIT_CROUCH", "HOLD"):
            if self.stage == "PARKED":
                return False, "Already PARKED; X stops."
            return False, f"{self.stage} is still moving; Enter ignored."
        if now - self.wait_since < base.WAIT_DWELL_S:
            return False, f"Wait for {self.stage} to settle before Enter."
        if not healthy:
            return False, "Motor latch/fault present; motion refused."

        start_target = Q_CROUCH if self.stage == "WAIT_CROUCH" else Q_STAND
        error = np.abs(np.asarray(q, dtype=float) - start_target)
        worst = int(np.argmax(error))
        if error[worst] > POSITION_POSE_TOL:
            return False, (
                f"Start pose not reached: {JOINT_LABELS[worst]} error "
                f"{error[worst]:.2f} rad > {POSITION_POSE_TOL:.2f}."
            )
        max_speed = float(np.max(np.abs(qd)))
        if max_speed > POSITION_QD_TOL:
            return False, (
                f"Robot not settled: max |qd| {max_speed:.2f} rad/s > "
                f"{POSITION_QD_TOL:.2f}."
            )

        self.stage = "STAND" if self.stage == "WAIT_CROUCH" else "PARK"
        self.started_at = float(now)
        self.wait_since = None
        self.settle_since = None
        return True, f"ENTER accepted: starting {self.stage}."


def validate_configuration():
    """Validate editable pose limits and direction conversions offline."""
    base.validate_hardware_config()
    low, high = base.soft_limits()
    for name, target in (("crouch", Q_CROUCH), ("stand", Q_STAND)):
        outside = (target < low) | (target > high)
        if np.any(outside):
            index = int(np.flatnonzero(outside)[0])
            raise ValueError(
                f"{name} {JOINT_LABELS[index]}={target[index]:+.3f} rad "
                "is outside its soft limit"
            )
        if not np.allclose(
            base.motoroutput_deg_to_joint_rad(
                base.joint_rad_to_motoroutput_deg(target)
            ),
            target,
        ):
            raise ValueError(f"{name} direction round-trip failed")

def position_fault(
    qd_encoder,
    measured_torque,
    velocity_ready,
    speed_trip,
    torque_trip,
):
    """Return a fault string when position-mode feedback exceeds a trip."""
    measured_torque = np.asarray(measured_torque, dtype=float)
    if not np.all(np.isfinite(measured_torque)):
        return "native position returned invalid measured torque"
    if np.any(np.abs(measured_torque) > torque_trip):
        index = int(np.argmax(np.abs(measured_torque)))
        return (
            f"position measured-torque trip: {JOINT_LABELS[index]}="
            f"{measured_torque[index]:+.2f} N*m exceeds {torque_trip:.2f} N*m"
        )
    if velocity_ready:
        qd_encoder = np.asarray(qd_encoder, dtype=float)
        if np.any(np.abs(qd_encoder) > speed_trip):
            index = int(np.argmax(np.abs(qd_encoder)))
            return (
                f"position encoder-speed trip: {JOINT_LABELS[index]}="
                f"{qd_encoder[index]:+.2f} rad/s exceeds "
                f"{speed_trip:.2f} rad/s"
            )
    return None


def zero_torque_preflight(mb, key, unwrap):
    """Read and display the stationary starting pose before position mode."""
    print("[READ] Keep robot supported and still; ENTER starts CROUCH, X stops.")
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
        elapsed = now - started_at
        rec = mb.rec(mid)
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

        q, _ = base._joint_state(mb, unwrap)
        if index % N_JOINTS == 0:
            qd_encoder = velocity.update(q, now)
        if now - last_print >= 1.0 / base.STATUS_HZ:
            print_pose("READ", q)
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
                    "[position] ENTER refused: waiting for fresh clear status "
                    f"from CAN {unverified}."
                )
            elif not velocity.ready:
                print(
                    "[position] ENTER refused: collecting stationary encoder "
                    "samples."
                )
            elif max_speed > POSITION_QD_TOL:
                print(
                    f"[position] ENTER refused: max encoder |qd|="
                    f"{max_speed:.2f} rad/s; hold the robot still."
                )
            else:
                low, high = base.soft_limits()
                outside = (q < low) | (q > high)
                if np.any(outside):
                    details = ", ".join(
                        f"{JOINT_LABELS[j]}={q[j]:+.2f}rad"
                        for j in np.flatnonzero(outside)
                    )
                    print(
                        "[position] WARNING: starting outside soft limits: "
                        f"{details}. CROUCH targets are inside the limits."
                    )
                return q.copy()

        index += 1
        mb.pace(deadline)
        deadline += slot


def run_hardware(
    crouch_max_speed_dps,
    stand_max_speed_dps,
    position_torque_trip,
    position_speed_trip,
    qd_estop=base.QD_ESTOP,
    qd_estop_hard=base.QD_ESTOP_HARD,
):
    validate_configuration()
    unwrap = [base.CalibratedEncoderUnwrap() for _ in base.HARDWARE_JOINTS]
    safety = base.SafetyGate(
        tau_cap=1.0,
        qd_estop=qd_estop,
        qd_estop_hard=qd_estop_hard,
    )

    key = base.KeyPoller()
    try:
        with base.motorbus.MotorBus(
            MOTOR_IDS, dirs=MOTOR_DIRECTIONS
        ) as mb:
            armed = False
            stop_reason = None
            try:
                print(
                    f"[ARM] Arming motors for up to {ARM_TIMEOUT_S:.0f}s; "
                    "keep supported, Ctrl-C stops."
                )
                if not mb.arm(
                    rate_hz=base.CONTROL_HZ,
                    timeout_s=ARM_TIMEOUT_S,
                    verbose=False,
                ):
                    raise RuntimeError(
                        f"arming timed out after {ARM_TIMEOUT_S:.0f}s: "
                        f"{arm_failure_summary(mb)}"
                    )
                armed = True

                start_q = zero_torque_preflight(mb, key, unwrap)
                now = time.perf_counter()
                sequence = PositionStandSequence(now)
                safety.start(now, start_q)
                print_pose("CROUCH", start_q)

                slot = mb.slot(base.CONTROL_HZ)
                deadline = time.perf_counter() + slot
                run_started_at = now
                q = start_q.copy()
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
                qd_encoder = np.zeros(N_JOINTS)
                measured_torque = np.zeros(N_JOINTS)

                while True:
                    mb.poll()
                    joint_index = index % N_JOINTS

                    if joint_index == 0:
                        now = time.perf_counter()
                        elapsed = now - run_started_at
                        q, qd_driver = base._joint_state(mb, unwrap)
                        qd_encoder = velocity.update(q, now)

                        event = sequence.update(
                            now,
                            q=q,
                            qd=(qd_encoder if velocity.ready else None),
                        )
                        if event:
                            print_pose(sequence.stage, q)

                        temps = base._temperatures(mb)
                        misses = miss_monitor.update(mb)
                        errors = mb.errors()
                        # The initial pose may be outside a normal soft limit.
                        # CROUCH is an in-limit target, so enable the position
                        # limit e-stop after that first inward move completes.
                        stop_reason = safety.estop_reason(
                            q,
                            qd_driver,
                            temps,
                            misses,
                            errors,
                            now,
                            enforce_position_limits=(
                                sequence.stage != "CROUCH"
                            ),
                        )
                        torque_feedback = mb.torques_nm()
                        measured_torque = np.asarray(
                            [torque_feedback[mid] for mid in MOTOR_IDS]
                        )
                        if stop_reason is None:
                            stop_reason = position_fault(
                                qd_encoder,
                                measured_torque,
                                velocity.ready,
                                position_speed_trip,
                                position_torque_trip,
                            )

                        latched = [
                            mid for mid, error in errors.items() if error & 0x80
                        ]
                        unverified = [
                            mid
                            for mid in MOTOR_IDS
                            if mb.rec(mid).error is None
                        ]
                        pressed = key.get()
                        if pressed in ("x", "X"):
                            stop_reason = "operator X"
                        elif base._is_enter(pressed):
                            advanced, message = sequence.request_next(
                                now,
                                q,
                                qd_encoder,
                                healthy=(
                                    not latched
                                    and not unverified
                                    and stop_reason is None
                                ),
                            )
                            if advanced:
                                print_pose(sequence.stage, q)
                            else:
                                print(f"[{sequence.stage}] {message}")
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
                            print_pose(sequence.stage, q)
                            last_print = now

                    mid = MOTOR_IDS[joint_index]
                    elapsed = time.perf_counter() - run_started_at
                    if elapsed >= next_fault_status[joint_index]:
                        mb.status1_req(mid)
                        next_fault_status[joint_index] += status_period
                    else:
                        max_speed = (
                            crouch_max_speed_dps
                            if sequence.stage
                            in ("CROUCH", "WAIT_CROUCH", "PARK", "PARKED")
                            else stand_max_speed_dps
                        )
                        if not mb.position(
                            mid,
                            float(sequence.target_deg[joint_index]),
                            max_speed,
                        ):
                            raise RuntimeError(
                                f"CAN position transmit failed for CAN {mid}"
                            )

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
                    print(f"[STOP] {stop_reason or 'aborted'}")
                    mb.stop_all()
    finally:
        key.close()
    return 0


def offline_self_test():
    validate_configuration()
    zero = np.zeros(N_JOINTS)

    class _ArmRecord:
        def __init__(self, last_reply_t=1.0, error=0):
            self.last_reply_t = last_reply_t
            self.error = error

    class _ArmBus:
        def __init__(self):
            self.records = {mid: _ArmRecord() for mid in MOTOR_IDS}

        def rec(self, mid):
            return self.records[mid]

    arm_bus = _ArmBus()
    arm_bus.rec(MOTOR_IDS[0]).last_reply_t = None
    arm_bus.rec(MOTOR_IDS[1]).error = None
    arm_bus.rec(MOTOR_IDS[2]).error = 0x80
    arm_bus.rec(MOTOR_IDS[3]).error = 0x04
    arm_summary = arm_failure_summary(arm_bus)
    assert f"no reply CAN [{MOTOR_IDS[0]}]" in arm_summary
    assert f"no fresh status CAN [{MOTOR_IDS[1]}]" in arm_summary
    assert f"input-timeout latch CAN [{MOTOR_IDS[2]}]" in arm_summary
    assert f"CAN {MOTOR_IDS[3]}=0x04" in arm_summary

    assert np.allclose(CROUCH_TARGET_DEG, base._stack_pose(RECORDED_CROUCH_DEG))
    assert np.allclose(STAND_TARGET_DEG, base._stack_pose(STAND_POSE_DEG))

    velocity = EncoderVelocity()
    for sample in range(7):
        estimated = velocity.update(0.5 * Q_CROUCH, sample * 0.05)
    assert velocity.ready and np.allclose(estimated, 0.0)

    sequence = PositionStandSequence(0.0)
    assert sequence.update(0.10, q=Q_CROUCH, qd=zero) is None
    assert sequence.update(
        0.10 + POSITION_SETTLE_S + 0.01,
        q=Q_CROUCH,
        qd=zero,
    )
    assert sequence.stage == "WAIT_CROUCH"
    advanced, reason = sequence.request_next(
        sequence.wait_since + base.WAIT_DWELL_S + 0.02,
        Q_CROUCH,
        zero,
        True,
    )
    assert advanced, reason
    assert sequence.stage == "STAND"
    assert sequence.update(
        sequence.started_at + 0.10,
        q=Q_STAND,
        qd=zero,
    ) is None
    assert sequence.update(
        sequence.started_at + 0.10 + POSITION_SETTLE_S + 0.01,
        q=Q_STAND,
        qd=zero,
    )
    assert sequence.stage == "HOLD"
    advanced, reason = sequence.request_next(
        sequence.wait_since + base.WAIT_DWELL_S + 0.02,
        Q_STAND,
        zero,
        True,
    )
    assert advanced, reason
    assert sequence.stage == "PARK"
    assert sequence.update(
        sequence.started_at + 0.10,
        q=Q_CROUCH,
        qd=zero,
    ) is None
    assert sequence.update(
        sequence.started_at + 0.10 + POSITION_SETTLE_S + 0.01,
        q=Q_CROUCH,
        qd=zero,
    )
    assert sequence.stage == "PARKED"

    torque = np.zeros(N_JOINTS)
    torque[3] = DEFAULT_POSITION_TORQUE_TRIP_NM + 0.01
    assert position_fault(
        zero,
        torque,
        True,
        DEFAULT_POSITION_SPEED_TRIP_RAD_S,
        DEFAULT_POSITION_TORQUE_TRIP_NM,
    )

    print("stand_by_position_command offline self-test PASS")
    print(
        "  flow: READ -> CROUCH -> ENTER -> STAND -> HOLD -> "
        "ENTER -> PARK -> PARKED"
    )
    print("  crouch joint targets deg:", RECORDED_CROUCH_DEG)
    print("  stand joint targets deg:", STAND_POSE_DEG)
    print(
        "  stand motoroutput deg by CAN:",
        {
            joint.can_id: float(value)
            for joint, value in zip(
                base.HARDWARE_JOINTS,
                base.joint_rad_to_motoroutput_deg(Q_STAND),
            )
        },
    )
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--crouch-max-speed-dps",
        type=float,
        default=DEFAULT_CROUCH_MAX_MOTOR_DPS,
        help=(
            "CURRENT-to-CROUCH and PARK native 0xA4 motor-side speed cap, deg/s "
            f"(default: {DEFAULT_CROUCH_MAX_MOTOR_DPS})"
        ),
    )
    parser.add_argument(
        "--stand-max-speed-dps",
        type=float,
        default=DEFAULT_STAND_MAX_MOTOR_DPS,
        help=(
            "CROUCH-to-STAND native 0xA4 motor-side speed cap, deg/s "
            f"(default: {DEFAULT_STAND_MAX_MOTOR_DPS})"
        ),
    )
    parser.add_argument(
        "--position-torque-trip",
        type=float,
        default=DEFAULT_POSITION_TORQUE_TRIP_NM,
        help=(
            "stop when measured joint torque exceeds this threshold, N*m; "
            "this is not a commanded torque cap "
            f"(default: {DEFAULT_POSITION_TORQUE_TRIP_NM})"
        ),
    )
    parser.add_argument(
        "--position-speed-trip",
        type=float,
        default=DEFAULT_POSITION_SPEED_TRIP_RAD_S,
        help=(
            "stop when encoder-derived joint speed exceeds this threshold, "
            f"rad/s (default: {DEFAULT_POSITION_SPEED_TRIP_RAD_S})"
        ),
    )
    parser.add_argument(
        "--qd-estop",
        type=float,
        default=base.QD_ESTOP,
        help=f"confirmed sustained driver-speed trip (default: {base.QD_ESTOP})",
    )
    parser.add_argument(
        "--qd-estop-hard",
        type=float,
        default=base.QD_ESTOP_HARD,
        help=(
            "single-sample driver-speed trip, rad/s "
            f"(default: {base.QD_ESTOP_HARD})"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate mappings, targets, and stages without opening CAN",
    )
    args = parser.parse_args()

    for option, value in (
        ("--crouch-max-speed-dps", args.crouch_max_speed_dps),
        ("--stand-max-speed-dps", args.stand_max_speed_dps),
    ):
        if not 1.0 <= value <= 600.0:
            parser.error(f"{option} must be between 1 and 600")
    if not 0.1 <= args.position_torque_trip <= base.TAU_HARD:
        parser.error(
            f"--position-torque-trip must be between 0.1 and "
            f"{base.TAU_HARD} N*m"
        )
    if not 0.1 <= args.position_speed_trip <= 3.0:
        parser.error("--position-speed-trip must be between 0.1 and 3.0 rad/s")
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
        return offline_self_test()

    try:
        return run_hardware(
            args.crouch_max_speed_dps,
            args.stand_max_speed_dps,
            args.position_torque_trip,
            args.position_speed_trip,
            args.qd_estop,
            args.qd_estop_hard,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
