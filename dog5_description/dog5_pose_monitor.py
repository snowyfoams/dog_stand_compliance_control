#!/usr/bin/env python3
"""Capture and hold a manually arranged DOG5 standing pose.

This is a conservative alternative to guessing a standing joint target.  The
robot remains at zero commanded torque while the operator arranges it by hand.
The program displays calibrated joint angles and encoder-derived foot positions.
Once the robot is still, press C to capture the pose, then H to engage a
low-torque joint PD hold around exactly that measured pose.

The saved pose uses model-positive joint coordinates and the confirmed mapping
from ``dog5_hardware_map.py``.  It is not a new encoder zero and this program
never sends the lifetime-affecting firmware command 0x19.

The robot must be supported throughout the test.  Encoder-only control cannot
measure body attitude, foot contact, or absolute body height, so this is a pose
monitor/holder for a supported robot, not a free-standing balance controller.

Typical use::

    python dog5_pose_monitor.py --self-test
    python dog5_pose_monitor.py --tau-max 1.0

Keys:

    C  capture the current stationary pose and save it
    H  hold the captured/loaded pose (only when already close and stationary)
    R  release back to zero torque (support the robot first)
    X  stop and release all motors

To reuse a previously captured pose::

    python dog5_pose_monitor.py --load dog5_stand_pose.json --tau-max 1.0
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO))

import dog5_kinematics
from dog5_hardware_map import HARDWARE_JOINTS
import motorbus


POSE_SCHEMA = "dog5-recorded-stand-pose/v1"
LEGS = ("FL", "FR", "RL", "RR")
MOTOR_IDS = [joint.can_id for joint in HARDWARE_JOINTS]
MOTOR_DIRECTIONS = {
    joint.can_id: joint.direction for joint in HARDWARE_JOINTS
}
JOINT_LABELS = [
    f"{joint.leg}_{joint.joint}" for joint in HARDWARE_JOINTS
]
JOINT_DIRECTIONS = np.asarray(
    [joint.direction for joint in HARDWARE_JOINTS], dtype=float
)
N_JOINTS = len(HARDWARE_JOINTS)

CONTROL_HZ = 250.0
DISPLAY_HZ = 2.0
FAULT_STATUS_HZ = 5.0
RECOVER_PERIOD_S = 0.1
DEFAULT_OUTPUT = _HERE / "dog5_stand_pose.json"

# Capture and hold gates.  Capture stillness is based on encoder position
# changes rather than the driver's occasionally noisy speed field.
CAPTURE_WINDOW_S = 0.75
CAPTURE_RANGE_RAD = np.deg2rad(1.0)
CAPTURE_SPEED_RAD_S = 0.15
HOLD_START_TOL_RAD = np.deg2rad(8.0)
HOLD_ERROR_ESTOP_RAD = np.deg2rad(30.0)

DEFAULT_KP = 8.0
DEFAULT_KD = 0.45
DEFAULT_TAU_MAX = 1.0
TAU_MAX_CEILING = 3.0
TORQUE_RAMP_S = 1.0
TAU_SLEW_NM_S = 5.0

ABD_LIM, PITCH_LIM, KNEE_LIM = 1.75, 2.6, 2.6
LIMIT_ESTOP_MARGIN = 0.05
ENCODER_QD_ESTOP = 8.0
TEMP_ESTOP_C = 80
MISS_ESTOP = 20


def soft_limits() -> tuple[np.ndarray, np.ndarray]:
    low = np.tile([-ABD_LIM, -PITCH_LIM, -KNEE_LIM], 4)
    high = np.tile([+ABD_LIM, +PITCH_LIM, +KNEE_LIM], 4)
    return low, high


def validate_hardware_map() -> None:
    if N_JOINTS != 12:
        raise ValueError(f"expected 12 configured joints, got {N_JOINTS}")
    if len(set(MOTOR_IDS)) != N_JOINTS:
        raise ValueError(f"duplicate CAN IDs in hardware map: {MOTOR_IDS}")
    if set(MOTOR_IDS) != set(range(1, 13)):
        raise ValueError(f"hardware map must contain CAN IDs 1..12: {MOTOR_IDS}")
    if tuple(joint.leg for joint in HARDWARE_JOINTS) != tuple(
        leg for leg in LEGS for _ in range(3)
    ):
        raise ValueError("hardware map is not ordered [FL, FR, RL, RR] x 3")
    expected_joints = ("abd", "pitch", "knee") * 4
    if tuple(joint.joint for joint in HARDWARE_JOINTS) != expected_joints:
        raise ValueError("hardware map joints are not ordered [abd, pitch, knee]")
    if any(direction not in (-1, +1) for direction in MOTOR_DIRECTIONS.values()):
        raise ValueError("every motor direction must be +1 or -1")


class CalibratedEncoderUnwrap:
    """Track the calibrated single-turn register as continuous output degrees."""

    _FULL_TURN = 65536

    def __init__(self):
        self._previous_raw: int | None = None
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


def motoroutput_deg_to_joint_rad(motoroutput_deg) -> np.ndarray:
    motoroutput_deg = np.asarray(motoroutput_deg, dtype=float)
    if motoroutput_deg.shape != (N_JOINTS,):
        raise ValueError(
            f"motor output must have shape ({N_JOINTS},), "
            f"got {motoroutput_deg.shape}"
        )
    return np.deg2rad(JOINT_DIRECTIONS * motoroutput_deg)


def joint_state(mb, unwrap) -> tuple[np.ndarray, np.ndarray]:
    raw = [mb.rec(mid).encoder for mid in MOTOR_IDS]
    missing = [mid for mid, value in zip(MOTOR_IDS, raw) if value is None]
    if missing:
        raise RuntimeError(f"no encoder reply from CAN IDs: {missing}")
    motoroutput_deg = np.asarray(
        [tracker.update(value) for tracker, value in zip(unwrap, raw)]
    )
    q = motoroutput_deg_to_joint_rad(motoroutput_deg)
    driver_dps = mb.speeds_dps()
    qd_driver = np.deg2rad(
        np.asarray([driver_dps[mid] for mid in MOTOR_IDS], dtype=float)
    )
    return q, qd_driver


class EncoderVelocity:
    """Low-pass velocity estimate from the independently measured positions."""

    def __init__(self, alpha: float = 0.35):
        self.alpha = float(alpha)
        self.previous_q: np.ndarray | None = None
        self.previous_time: float | None = None
        self.value = np.zeros(N_JOINTS)

    def update(self, q, now: float) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if self.previous_q is not None:
            dt = now - self.previous_time
            if np.isfinite(dt) and 1.0e-4 < dt < 0.25:
                raw = (q - self.previous_q) / dt
                self.value = self.alpha * raw + (1.0 - self.alpha) * self.value
            else:
                self.value.fill(0.0)
        self.previous_q = q.copy()
        self.previous_time = float(now)
        return self.value.copy()


class PoseWindow:
    """Recent encoder positions used to reject a moving pose capture."""

    def __init__(self, duration_s: float = CAPTURE_WINDOW_S):
        self.duration_s = float(duration_s)
        self.samples: deque[tuple[float, np.ndarray]] = deque()

    def clear(self) -> None:
        self.samples.clear()

    def add(self, now: float, q) -> None:
        self.samples.append((float(now), np.asarray(q, dtype=float).copy()))
        cutoff = now - self.duration_s
        while len(self.samples) > 2 and self.samples[1][0] < cutoff:
            self.samples.popleft()

    def verdict(self) -> tuple[bool, str]:
        if len(self.samples) < 4:
            return False, "collecting encoder samples"
        elapsed = self.samples[-1][0] - self.samples[0][0]
        if elapsed < 0.9 * self.duration_s:
            return False, f"collecting {self.duration_s:.2f} s stillness window"
        values = np.stack([sample for _, sample in self.samples])
        spans = np.ptp(values, axis=0)
        worst = int(np.argmax(spans))
        if spans[worst] > CAPTURE_RANGE_RAD:
            return False, (
                f"{JOINT_LABELS[worst]} moved {np.rad2deg(spans[worst]):.2f} deg "
                f"(limit {np.rad2deg(CAPTURE_RANGE_RAD):.2f} deg)"
            )
        speed = np.abs(values[-1] - values[0]) / max(elapsed, 1.0e-9)
        worst = int(np.argmax(speed))
        if speed[worst] > CAPTURE_SPEED_RAD_S:
            return False, (
                f"{JOINT_LABELS[worst]} encoder speed {speed[worst]:.2f} rad/s "
                f"(limit {CAPTURE_SPEED_RAD_S:.2f})"
            )
        return True, "stationary"

    def mean(self) -> np.ndarray:
        ready, reason = self.verdict()
        if not ready:
            raise RuntimeError(f"pose is not ready to capture: {reason}")
        return np.mean(np.stack([sample for _, sample in self.samples]), axis=0)


def pose_document(q) -> dict:
    q = np.asarray(q, dtype=float)
    if q.shape != (N_JOINTS,) or not np.all(np.isfinite(q)):
        raise ValueError(f"pose must be {N_JOINTS} finite joint angles")
    feet = {}
    for leg_index, leg in enumerate(LEGS):
        section = slice(3 * leg_index, 3 * leg_index + 3)
        feet[leg] = dog5_kinematics.foot_position(leg, q[section]).tolist()
    return {
        "schema": POSE_SCHEMA,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "angle_convention": (
            "q_rad = radians(direction * calibrated_motoroutput_deg); "
            "no software offset; no gearbox division"
        ),
        "joint_order": list(JOINT_LABELS),
        "q_rad": q.tolist(),
        "q_deg": np.rad2deg(q).tolist(),
        "hardware": [
            {
                "leg": joint.leg,
                "joint": joint.joint,
                "can_id": joint.can_id,
                "direction": joint.direction,
            }
            for joint in HARDWARE_JOINTS
        ],
        "feet_trunk_m": feet,
        "encoder_height_m": {
            leg: -float(feet[leg][2]) for leg in LEGS
        },
    }


def validate_pose_document(document: dict) -> np.ndarray:
    if not isinstance(document, dict):
        raise ValueError("pose file root must be a JSON object")
    if document.get("schema") != POSE_SCHEMA:
        raise ValueError(
            f"unsupported pose schema {document.get('schema')!r}; "
            f"expected {POSE_SCHEMA!r}"
        )
    if document.get("joint_order") != JOINT_LABELS:
        raise ValueError("pose joint order does not match current hardware map")
    hardware = document.get("hardware")
    expected_hardware = [
        {
            "leg": joint.leg,
            "joint": joint.joint,
            "can_id": joint.can_id,
            "direction": joint.direction,
        }
        for joint in HARDWARE_JOINTS
    ]
    if hardware != expected_hardware:
        raise ValueError("pose CAN mapping/directions do not match current hardware map")
    q = np.asarray(document.get("q_rad"), dtype=float)
    if q.shape != (N_JOINTS,) or not np.all(np.isfinite(q)):
        raise ValueError(f"pose q_rad must contain {N_JOINTS} finite values")
    low, high = soft_limits()
    outside = (q < low) | (q > high)
    if np.any(outside):
        index = int(np.flatnonzero(outside)[0])
        raise ValueError(
            f"saved pose {JOINT_LABELS[index]}={q[index]:+.3f} rad is "
            "outside the software joint limit"
        )
    return q


def save_pose(path: Path, q) -> None:
    document = pose_document(q)
    validate_pose_document(document)
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")
        stream.flush()
        # Do not fsync here: capture happens while the zero-torque watchdog
        # stream is active, and a synchronous disk flush can stall it.  The
        # atomic rename still prevents readers from seeing partial JSON.
    os.replace(temporary, path)


def load_pose(path: Path) -> np.ndarray:
    path = path.expanduser().resolve()
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read pose file {path}: {exc}") from exc
    return validate_pose_document(document)


class JointPoseHold:
    """Joint PD hold with a ramp, cap, slew limit, and directional limits."""

    def __init__(self, kp: float, kd: float, tau_max: float):
        self.kp = float(kp)
        self.kd = float(kd)
        self.tau_max = float(tau_max)
        self.low, self.high = soft_limits()
        self.started_at: float | None = None
        self.last_time: float | None = None
        self.previous_tau = np.zeros(N_JOINTS)

    def start(self, now: float) -> None:
        self.started_at = float(now)
        self.last_time = float(now)
        self.previous_tau.fill(0.0)

    def release(self) -> None:
        self.started_at = None
        self.last_time = None
        self.previous_tau.fill(0.0)

    def compute(self, target, q, qd_encoder, now: float) -> np.ndarray:
        if self.started_at is None:
            raise RuntimeError("pose hold is not engaged")
        target = np.asarray(target, dtype=float)
        q = np.asarray(q, dtype=float)
        qd_encoder = np.asarray(qd_encoder, dtype=float)
        requested = self.kp * (target - q) - self.kd * qd_encoder
        ramp = np.clip((now - self.started_at) / TORQUE_RAMP_S, 0.0, 1.0)
        limited = np.clip(requested, -self.tau_max * ramp, self.tau_max * ramp)
        limited = np.where((q >= self.high) & (limited > 0.0), 0.0, limited)
        limited = np.where((q <= self.low) & (limited < 0.0), 0.0, limited)
        dt = np.clip(now - self.last_time, 1.0e-4, 0.1)
        max_change = TAU_SLEW_NM_S * dt
        output = self.previous_tau + np.clip(
            limited - self.previous_tau, -max_change, max_change
        )
        output = np.where((q >= self.high) & (output > 0.0), 0.0, output)
        output = np.where((q <= self.low) & (output < 0.0), 0.0, output)
        if not np.all(np.isfinite(output)):
            raise RuntimeError("joint pose hold produced non-finite torque")
        self.previous_tau = output
        self.last_time = float(now)
        return output.copy()


class MissMonitor:
    def __init__(self, mb):
        self.previous = np.asarray([mb.rec(mid).missed for mid in MOTOR_IDS])
        self.streaks = np.zeros(N_JOINTS, dtype=int)

    def update(self, mb) -> np.ndarray:
        current = np.asarray([mb.rec(mid).missed for mid in MOTOR_IDS])
        added = current - self.previous
        self.streaks = np.where(added > 0, self.streaks + added, 0)
        self.previous = current
        return self.streaks.copy()


class KeyPoller:
    def __init__(self):
        import select
        import termios
        import tty

        if not sys.stdin.isatty():
            raise RuntimeError("pose monitor requires an interactive terminal")
        self._select = select
        self._termios = termios
        self.fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self._closed = False

    def get(self) -> str:
        readable, _, _ = self._select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if readable else ""

    def close(self) -> None:
        if not self._closed:
            self._termios.tcsetattr(
                self.fd, self._termios.TCSADRAIN, self._old
            )
            self._closed = True


def temperatures(mb) -> np.ndarray:
    values = mb.temps()
    return np.asarray(
        [values[mid] if values[mid] is not None else 0 for mid in MOTOR_IDS],
        dtype=int,
    )


def safety_reason(mode, q, qd_encoder, target, temps, misses, errors) -> str | None:
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd_encoder)):
        return "invalid encoder state"
    hard_faults = {mid: err for mid, err in errors.items() if err & 0x7F}
    if hard_faults:
        detail = ", ".join(
            f"CAN {mid}=0x{err:02x}" for mid, err in hard_faults.items()
        )
        return f"motor fault: {detail}"
    if np.any(temps > TEMP_ESTOP_C):
        index = int(np.argmax(temps))
        return f"over-temperature CAN {MOTOR_IDS[index]}: {temps[index]} C"
    if np.any(misses >= MISS_ESTOP):
        index = int(np.argmax(misses))
        return (
            f"CAN {MOTOR_IDS[index]} missed {misses[index]} consecutive replies"
        )
    if mode != "HOLD":
        return None
    low, high = soft_limits()
    outside = (q < low - LIMIT_ESTOP_MARGIN) | (q > high + LIMIT_ESTOP_MARGIN)
    if np.any(outside):
        index = int(np.flatnonzero(outside)[0])
        return f"joint limit exceeded: {JOINT_LABELS[index]}={q[index]:+.2f} rad"
    speed = np.abs(qd_encoder)
    if np.any(speed > ENCODER_QD_ESTOP):
        index = int(np.argmax(speed))
        return (
            f"encoder overspeed {JOINT_LABELS[index]}="
            f"{qd_encoder[index]:+.1f} rad/s"
        )
    error = np.abs(np.asarray(target) - q)
    if np.any(error > HOLD_ERROR_ESTOP_RAD):
        index = int(np.argmax(error))
        return (
            f"hold error {JOINT_LABELS[index]}="
            f"{np.rad2deg(error[index]):.1f} deg"
        )
    return None


def capture_reason(window: PoseWindow, q) -> tuple[bool, str]:
    ready, reason = window.verdict()
    if not ready:
        return False, reason
    low, high = soft_limits()
    outside = (q < low) | (q > high)
    if np.any(outside):
        index = int(np.flatnonzero(outside)[0])
        return False, (
            f"{JOINT_LABELS[index]}={q[index]:+.2f} rad is outside its limit"
        )
    return True, "stationary and inside joint limits"


def hold_start_reason(window: PoseWindow, q, target, errors) -> tuple[bool, str]:
    if target is None:
        return False, "capture with C or pass --load before holding"
    ready, reason = capture_reason(window, q)
    if not ready:
        return False, reason
    unverified = [
        mid for mid in MOTOR_IDS
        if mb_error_unknown(errors, mid) or (errors[mid] & 0x80)
    ]
    if unverified:
        return False, (
            f"waiting for a fresh, clear motor status from CAN {unverified}"
        )
    error = np.abs(np.asarray(target) - q)
    worst = int(np.argmax(error))
    if error[worst] > HOLD_START_TOL_RAD:
        return False, (
            f"{JOINT_LABELS[worst]} is {np.rad2deg(error[worst]):.1f} deg "
            f"from target (limit {np.rad2deg(HOLD_START_TOL_RAD):.1f} deg)"
        )
    return True, "ready"


def mb_error_unknown(errors, mid: int) -> bool:
    # ``errors`` contains None when assembled explicitly in run_monitor().
    return errors.get(mid) is None


def print_pose(q, qd_driver, qd_encoder, target, mode, temps, window) -> None:
    fields = []
    heights = []
    for leg_index, leg in enumerate(LEGS):
        section = slice(3 * leg_index, 3 * leg_index + 3)
        q_leg = q[section]
        foot = dog5_kinematics.foot_position(leg, q_leg)
        height = -float(foot[2])
        heights.append(height)
        fields.append(
            f"{leg}[{q_leg[0] * 180 / np.pi:+6.1f} "
            f"{q_leg[1] * 180 / np.pi:+6.1f} "
            f"{q_leg[2] * 180 / np.pi:+6.1f}] "
            f"p=({foot[0]:+.3f},{foot[1]:+.3f},{foot[2]:+.3f})m "
            f"h={height:.3f}m"
        )
    ready, _ = capture_reason(window, q)
    target_error = (
        np.rad2deg(np.max(np.abs(np.asarray(target) - q)))
        if target is not None else None
    )
    line = (
        f"[{mode}] "
        + "  ".join(fields)
        + f"  h_spread={max(heights) - min(heights):.3f}m "
        f"qd(enc/driver)={np.max(np.abs(qd_encoder)):.2f}/"
        f"{np.max(np.abs(qd_driver)):.2f}rad/s "
    )
    if target_error is not None:
        line += f"target_err={target_error:.1f}deg "
    line += (
        f"capture={'ready' if ready else 'wait'} "
        f"Tmax={int(np.max(temps))}C"
    )
    print(line, flush=True)


def run_monitor(output_path: Path, loaded_target, kp, kd, tau_max, rate_hz) -> int:
    validate_hardware_map()
    target = None if loaded_target is None else np.asarray(loaded_target).copy()
    unwrap = [CalibratedEncoderUnwrap() for _ in HARDWARE_JOINTS]
    controller = JointPoseHold(kp, kd, tau_max)

    print("[pose] Confirmed joints:")
    for joint in HARDWARE_JOINTS:
        print(
            f"  {joint.leg}_{joint.joint:<5} CAN {joint.can_id:2d} "
            f"direction {joint.direction:+d}"
        )
    print(
        "[pose] ZERO TORQUE observation starts first. Mechanically support the "
        "robot and arrange the desired pose by hand."
    )
    print("[pose] Keys: C capture, H hold, R release, X stop.")
    if target is not None:
        print("[pose] Loaded target; manually match it before pressing H.")

    key = KeyPoller()
    try:
        with motorbus.MotorBus(
            MOTOR_IDS, dirs=MOTOR_DIRECTIONS
        ) as mb:
            print("[pose] Arming with zero commanded torque...")
            if not mb.arm(rate_hz=rate_hz):
                raise RuntimeError("not all motors armed")

            slot = mb.slot(rate_hz)
            deadline = time.perf_counter() + slot
            now = time.perf_counter()
            status_period = 1.0 / FAULT_STATUS_HZ
            next_status = np.asarray(
                [now + status_period + i * status_period / N_JOINTS
                 for i in range(N_JOINTS)]
            )
            last_recover = {mid: now - RECOVER_PERIOD_S for mid in MOTOR_IDS}
            velocity = EncoderVelocity()
            window = PoseWindow()
            misses = MissMonitor(mb)
            mode = "OBSERVE"
            tau_command = np.zeros(N_JOINTS)
            q = np.zeros(N_JOINTS)
            qd_driver = np.zeros(N_JOINTS)
            qd_encoder = np.zeros(N_JOINTS)
            temp = np.zeros(N_JOINTS, dtype=int)
            last_print = 0.0
            index = 0

            while True:
                mb.poll()
                joint_index = index % N_JOINTS

                if joint_index == 0:
                    now = time.perf_counter()
                    q, qd_driver = joint_state(mb, unwrap)
                    qd_encoder = velocity.update(q, now)
                    window.add(now, q)
                    temp = temperatures(mb)
                    miss_streaks = misses.update(mb)
                    errors = {mid: mb.rec(mid).error for mid in MOTOR_IDS}

                    hard_errors = {
                        mid: (err or 0) for mid, err in errors.items()
                    }
                    stop = safety_reason(
                        mode,
                        q,
                        qd_encoder,
                        target,
                        temp,
                        miss_streaks,
                        hard_errors,
                    )
                    if stop:
                        raise RuntimeError(stop)

                    latched = [
                        mid for mid, err in errors.items()
                        if err is not None and err & 0x80
                    ]
                    recover = [
                        mid for mid in latched
                        if now - last_recover[mid] >= RECOVER_PERIOD_S
                    ]
                    if recover:
                        print(f"[pose] Recovering input-timeout latch: CAN {recover}")
                        mb.recover(recover, settle_s=0.0, verify=False)
                        for mid in recover:
                            mb.rec(mid).error = None
                            last_recover[mid] = now
                            next_status[MOTOR_IDS.index(mid)] = now + status_period

                    pressed = key.get()
                    if pressed in ("x", "X"):
                        print("[pose] Operator X.")
                        break
                    if pressed in ("c", "C"):
                        if mode == "HOLD":
                            print("[pose] Capture refused while HOLD is active; press R first.")
                        else:
                            ready, reason = capture_reason(window, q)
                            if not ready:
                                print(f"[pose] Capture refused: {reason}.")
                            else:
                                target = window.mean()
                                save_pose(output_path, target)
                                # File I/O and its confirmation print can delay
                                # the keep-alive stream.  Force fresh 0x9A
                                # results before H can engage; any resulting
                                # 0x80 latch is recovered in the normal loop.
                                for mid in MOTOR_IDS:
                                    mb.rec(mid).error = None
                                window.clear()
                                print(
                                    f"[pose] Captured stationary pose to "
                                    f"{output_path.expanduser().resolve()}"
                                )
                                print(
                                    "[pose] Inspect the values; wait for "
                                    "capture=ready, then press H only while supported."
                                )
                    if pressed in ("h", "H"):
                        if mode == "HOLD":
                            print("[pose] HOLD is already active.")
                        else:
                            ready, reason = hold_start_reason(
                                window, q, target, errors
                            )
                            if not ready:
                                print(f"[pose] HOLD refused: {reason}.")
                            else:
                                controller.start(now)
                                mode = "HOLD"
                                print(
                                    f"[pose] HOLD active: Kp={kp:.2f}, Kd={kd:.2f}, "
                                    f"tau cap={tau_max:.2f} N*m. X stops."
                                )
                    if pressed in ("r", "R"):
                        if mode == "HOLD":
                            controller.release()
                            tau_command.fill(0.0)
                            mode = "OBSERVE"
                            window.clear()
                            print("[pose] Released to ZERO TORQUE; keep the robot supported.")
                        else:
                            print("[pose] Already at ZERO TORQUE.")

                    if mode == "HOLD":
                        tau_command = controller.compute(
                            target, q, qd_encoder, now
                        )
                    else:
                        tau_command.fill(0.0)

                    if now - last_print >= 1.0 / DISPLAY_HZ:
                        print_pose(
                            q,
                            qd_driver,
                            qd_encoder,
                            target,
                            mode,
                            temp,
                            window,
                        )
                        last_print = now

                mid = MOTOR_IDS[joint_index]
                now = time.perf_counter()
                if now >= next_status[joint_index]:
                    mb.status1_req(mid)
                    next_status[joint_index] += 1.0 / FAULT_STATUS_HZ
                else:
                    mb.torque(mid, float(tau_command[joint_index]))

                index += 1
                mb.pace(deadline)
                deadline += slot
    finally:
        key.close()
    print("[pose] Motors stopped and released.")
    return 0


def offline_self_test() -> int:
    validate_hardware_map()

    near_zero_negative = CalibratedEncoderUnwrap().update(65530)
    assert -0.1 < near_zero_negative < 0.0
    unwrap = CalibratedEncoderUnwrap()
    before = unwrap.update(65530)
    after = unwrap.update(4)
    assert after > before and after - before < 0.1

    q = np.deg2rad(
        [30.0, -35.0, -70.0, -30.0, 35.0, 70.0,
         30.0, 35.0, 70.0, -30.0, -35.0, -70.0]
    )
    document = pose_document(q)
    assert np.allclose(validate_pose_document(document), q)
    assert set(document["feet_trunk_m"]) == set(LEGS)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pose.json"
        save_pose(path, q)
        assert np.allclose(load_pose(path), q)

    window = PoseWindow(duration_s=0.5)
    for sample in range(12):
        window.add(0.05 * sample, q + 1.0e-4 * np.sin(sample))
    ready, reason = window.verdict()
    assert ready, reason
    assert np.max(np.abs(window.mean() - q)) < 2.0e-4
    clear_errors = {mid: 0 for mid in MOTOR_IDS}
    ready, reason = hold_start_reason(window, q, q, clear_errors)
    assert ready, reason
    latched_errors = clear_errors.copy()
    latched_errors[MOTOR_IDS[0]] = 0x80
    ready, _ = hold_start_reason(window, q, q, latched_errors)
    assert not ready

    moving = PoseWindow(duration_s=0.5)
    for sample in range(12):
        moved = q.copy()
        moved[0] += np.deg2rad(0.3 * sample)
        moving.add(0.05 * sample, moved)
    ready, _ = moving.verdict()
    assert not ready

    hold = JointPoseHold(DEFAULT_KP, DEFAULT_KD, DEFAULT_TAU_MAX)
    hold.start(0.0)
    tau0 = hold.compute(q, q, np.zeros(N_JOINTS), 0.0)
    assert np.allclose(tau0, 0.0)
    displaced = q.copy()
    displaced[1] -= 0.1
    tau = hold.compute(q, displaced, np.zeros(N_JOINTS), 0.5)
    assert 0.0 < tau[1] <= DEFAULT_TAU_MAX
    assert np.max(np.abs(tau)) <= DEFAULT_TAU_MAX

    print("dog5_pose_monitor offline self-test PASS")
    print("  mapping:", list(zip(JOINT_LABELS, MOTOR_IDS, JOINT_DIRECTIONS.astype(int))))
    print("  pose schema:", POSE_SCHEMA)
    print("  capture -> validated JSON target -> ramped low-torque joint PD hold")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"captured pose JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--load",
        type=Path,
        default=None,
        help="load an existing captured pose as the hold target",
    )
    parser.add_argument(
        "--tau-max",
        type=float,
        default=DEFAULT_TAU_MAX,
        help=f"per-joint hold torque cap, N*m (default: {DEFAULT_TAU_MAX})",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=DEFAULT_KP,
        help=f"joint hold proportional gain, N*m/rad (default: {DEFAULT_KP})",
    )
    parser.add_argument(
        "--kd",
        type=float,
        default=DEFAULT_KD,
        help=f"joint hold damping gain, N*m*s/rad (default: {DEFAULT_KD})",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=CONTROL_HZ,
        help=f"command rate per motor, Hz (default: {CONTROL_HZ})",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="test mapping, pose validation, capture gate, and hold without CAN",
    )
    args = parser.parse_args()

    if not 0.0 < args.tau_max <= TAU_MAX_CEILING:
        parser.error(f"--tau-max must be > 0 and <= {TAU_MAX_CEILING} N*m")
    if not 0.0 < args.kp <= 30.0:
        parser.error("--kp must be > 0 and <= 30 N*m/rad")
    if not 0.0 <= args.kd <= 3.0:
        parser.error("--kd must be between 0 and 3 N*m*s/rad")
    if not 200.0 <= args.rate <= 300.0:
        parser.error(
            "--rate must be between 200 and 300 Hz per motor so the input "
            "watchdog stays fed without overloading the shared CAN bus"
        )
    if args.self_test:
        return offline_self_test()

    try:
        loaded_target = load_pose(args.load) if args.load is not None else None
        return run_monitor(
            args.output,
            loaded_target,
            args.kp,
            args.kd,
            args.tau_max,
            args.rate,
        )
    except KeyboardInterrupt:
        print("\n[pose] Ctrl-C; motors stopped and released.")
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[pose] STOPPED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
