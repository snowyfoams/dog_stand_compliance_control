#!/usr/bin/env python3
"""Stand DOG5 in place from the recorded crouch on real CAN hardware.

This runner replaces the under-hip stand targets of
``stand_dog5_recorded_hw.py`` with **stand-in-place** targets and fixes the
four failure modes observed on hardware:

1. FOOT SLIP / stick-slip during STAND.  The old targets dragged every
   loaded foot 11-13 cm horizontally (recorded crouch feet sit at
   x = +/-0.34 m, targets at the hips), which needs a friction coefficient
   of ~4.  Here each foot rises straight up from its own recorded crouch
   xy: zero commanded horizontal travel, so no loaded dragging, and the
   fore/aft-splayed stance gives a larger support polygon.
2. UNEXPECTED TRUNK ROLL.  Asymmetric stick-slip is gone with the drag,
   and a slow per-leg vertical-force integral trim levels the trunk
   against uneven loading without an IMU.
3. STUCK IN WAIT_STAND.  With support-scale 0.25 the steady gravity sag is
   ~23 mm against the 25 mm settle tolerance, so HOLD was unreachable
   without a person lifting the robot.  The default support feedforward is
   now 0.65, the integral trim removes the remaining sag, and WAIT_STAND
   times out after 20 s into HOLD_SAG (hold target rebased to the achieved
   foot heights) instead of waiting forever.
4. SPEED-LIMIT ABORTS.  The stock hard overspeed tier tripped on a single
   sample of the noisy driver speed field (a 5.9 rad/s phantom at
   standstill is on record).  The hard tier now needs the same-cycle
   encoder speed to agree, an encoder-alone overspeed, or two consecutive
   driver samples; the sustained tier is unchanged.

The operator-gated sequence matches the recorded runner:

    ZERO TORQUE  read and display the current pose; Enter starts CROUCH
    CROUCH       native 0xA4 position control moves to the recorded pose
    WAIT CROUCH  hold the recorded pose; Enter starts Cartesian STAND
    STAND        Cartesian compliance raises the body straight up over the
                 recorded foot positions (vertical-only path)
    WAIT STAND   keep the final target until tracking settles, at most 20 s
    HOLD         settled at the full target (partial tests: HOLD_PARTIAL)
    HOLD_SAG     20 s timeout fallback: target rebased to achieved heights
    PARK         press P in HOLD/HOLD_PARTIAL/HOLD_SAG to descend to crouch
    WAIT PARK    settle at crouch, then resume the native crouch hold

Static torque to hold mg/4 along the vertical path peaks at ~1.05 N*m per
joint, so the 1.0 N*m default cap cannot support the robot (a startup
warning says so).  Staged bring-up:

    python stand_dog5_inplace_hw.py --self-test
    python stand_dog5_inplace_hw.py --tau-max 1.0 --travel-scale 0.25   # supported
    python stand_dog5_inplace_hw.py --tau-max 1.0 --travel-scale 0.5    # supported
    python stand_dog5_inplace_hw.py --tau-max 3.0                       # full stand

Notes: with --travel-scale < 1 the weight feedforward is scaled down with
the path, so an unloaded rig test shows near-zero trim while a loaded
partial stand may clamp the trim at its +/-6 N ceiling and settle with a
few mm of residual sag.  WAIT_PARK keeps the settle wait of the recorded
runner (X always stops).  The original runners are left untouched; this
file only subclasses them.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import dog5_kinematics
import stand_dog5
import stand_dog5_hw as base
import stand_dog5_recorded_hw as recorded


LEGS = base.LEGS
MOTOR_IDS = base.MOTOR_IDS
MOTOR_DIRECTIONS = base.MOTOR_DIRECTIONS
JOINT_LABELS = base.JOINT_LABELS
N_JOINTS = base.N_JOINTS
Q_RECORDED_CROUCH = recorded.Q_RECORDED_CROUCH
POSITION_TARGET_DEG = recorded.POSITION_TARGET_DEG
T_STAND = recorded.T_STAND
T_PARK = recorded.T_PARK

# Weight support: 0.65 of mg/4 feedforward leaves ~4.5 N per foot to the
# integral trim (clamped well inside its +/-6 N ceiling), instead of asking
# the Cartesian Kp to hold 75% of the robot through a 23 mm sag.
DEFAULT_SUPPORT_SCALE = 0.65
WAIT_STAND_TIMEOUT_S = 20.0
# Trim loop time-constant is kp_z/Ki ~ 0.7*600/150 ~ 2.8 s: settles within
# the WAIT_STAND timeout, far too slow to fight the Cartesian PD.
TRIM_KI_N_PER_M_S = 150.0
TRIM_CLAMP_N = 6.0
TRIM_DEADBAND_M = 0.002
# The vertical path from the recorded RR crouch xy cannot reach 0.22 m.
MAX_STAND_HEIGHT = 0.21
MIN_STAND_HEIGHT = recorded.MIN_STAND_HEIGHT

# Hard overspeed tier confirmation (the sustained tier is inherited).
QD_ESTOP_HARD_STREAK = 2            # consecutive driver-only checks (~96 ms)
QD_ESTOP_HARD_CONFIRM_RATIO = 0.5   # encoder fraction confirming a driver sample
# A single corrupted encoder sample implies an impossible speed; anything
# above this is treated as a glitch, not a runaway.  If phantoms persist,
# requiring two consecutive encoder-alone checks is the next step.
QD_ENC_GLITCH_RAD_S = 25.0


class ConfirmedSafetyGate(base.SafetyGate):
    """SafetyGate with a confirmation requirement on the hard overspeed tier.

    Hard-tier trip conditions (any one, checked ~21 Hz):

      driver+encoder   driver speed above the hard limit and the same-cycle
                       encoder-derived speed above half of it
      encoder          encoder-derived speed above the hard limit but below
                       the glitch ceiling
      driver x2        driver speed above the hard limit for
                       QD_ESTOP_HARD_STREAK consecutive checks

    A single phantom driver sample with a stationary encoder no longer
    stops the run; a real runaway still trips within one or two checks.
    """

    def __init__(self, tau_cap, qd_estop=base.QD_ESTOP,
                 qd_estop_hard=base.QD_ESTOP_HARD):
        super().__init__(tau_cap, qd_estop=qd_estop, qd_estop_hard=qd_estop_hard)
        self.hard_streaks = np.zeros(N_JOINTS, dtype=int)

    def overspeed_reason(self, qd, q, now):
        qd = np.asarray(qd, dtype=float)
        q = np.asarray(q, dtype=float)
        speed = np.abs(qd)
        self.qd_peak = np.maximum(self.qd_peak, speed)

        # Identical independent encoder-speed estimate as the base gate.
        if self._overspeed_last_q is None:
            position_confirmed = np.zeros(N_JOINTS, dtype=bool)
            self.encoder_qd.fill(0.0)
        else:
            dt = float(now) - self._overspeed_last_time
            if np.isfinite(dt) and dt > 0.0:
                self.encoder_qd = (q - self._overspeed_last_q) / dt
                confirm_limit = (
                    base.QD_ESTOP_ENCODER_CONFIRM_RATIO * self.qd_estop
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

        encoder_speed = np.abs(self.encoder_qd)
        hard = self.qd_estop_hard
        self.hard_streaks = np.where(speed > hard, self.hard_streaks + 1, 0)
        hard_confirmed = (speed > hard) & (
            encoder_speed > QD_ESTOP_HARD_CONFIRM_RATIO * hard
        )
        encoder_alone = (encoder_speed > hard) & (
            encoder_speed <= QD_ENC_GLITCH_RAD_S
        )
        driver_persistent = self.hard_streaks >= QD_ESTOP_HARD_STREAK
        tripped_hard = hard_confirmed | encoder_alone | driver_persistent
        if np.any(tripped_hard):
            ranking = np.where(
                tripped_hard, np.maximum(speed, encoder_speed), 0.0
            )
            index = int(np.argmax(ranking))
            if hard_confirmed[index]:
                tier = "driver+encoder"
            elif encoder_alone[index]:
                tier = "encoder"
            else:
                tier = f"driver x{int(self.hard_streaks[index])}"
            return (
                f"hard overspeed {JOINT_LABELS[index]}: driver "
                f"{qd[index]:+.1f} rad/s, encoder "
                f"{self.encoder_qd[index]:+.1f} rad/s ({tier}) over the "
                f"{hard:.1f} rad/s hard limit"
            )

        over = speed > self.qd_estop
        confirmed_over = over & position_confirmed
        self.qd_streaks = np.where(confirmed_over, self.qd_streaks + 1, 0)
        tripped = self.qd_streaks >= base.QD_ESTOP_STREAK
        if np.any(tripped):
            index = int(np.argmax(np.where(tripped, speed, 0.0)))
            return (
                f"confirmed overspeed {JOINT_LABELS[index]}: driver "
                f"{qd[index]:+.1f} rad/s, encoder {self.encoder_qd[index]:+.1f} "
                f"rad/s for {int(self.qd_streaks[index])} consecutive checks "
                f"over the {self.qd_estop:.1f} rad/s driver limit"
            )
        return None


class InPlaceStandController(recorded.RecordedCrouchController):
    """Vertical-only stand over the recorded feet plus support trim."""

    def __init__(self, cart_gain_scale, support_scale, stand_height,
                 travel_scale=1.0):
        super().__init__(
            cart_gain_scale, support_scale, stand_height, travel_scale
        )
        # Stand in place: each foot rises straight up from its recorded
        # crouch xy.  The parent's clearance preload is already vertical.
        for leg in LEGS:
            crouch = self.crouch_foot[leg]
            self.full_stand_foot[leg] = np.array(
                [crouch[0], crouch[1], -self.stand_height]
            )
            self.final_foot[leg] = crouch + self.travel_scale * (
                self.full_stand_foot[leg] - crouch
            )
        self.z_trim_n = {leg: 0.0 for leg in LEGS}
        self.last_leg_sag_m = {leg: 0.0 for leg in LEGS}
        self.last_leg_force_clipped = {leg: False for leg in LEGS}
        self._trim_last_time = None

    def restore_full_targets(self):
        """Recompute the travel-scaled targets (undoes a HOLD_SAG rebase)."""
        for leg in LEGS:
            crouch = self.crouch_foot[leg]
            self.final_foot[leg] = crouch + self.travel_scale * (
                self.full_stand_foot[leg] - crouch
            )

    def rebase_final_z_to_measured(self, q):
        """Adopt the achieved foot heights as the hold target (xy kept)."""
        q = np.asarray(q, dtype=float)
        for leg_index, leg in enumerate(LEGS):
            section = slice(3 * leg_index, 3 * leg_index + 3)
            foot = dog5_kinematics.foot_position(leg, q[section])
            self.final_foot[leg] = np.array(
                [self.final_foot[leg][0], self.final_foot[leg][1], foot[2]]
            )

    def update_support_trim(self, now, mode, saturated_legs=None):
        """Slow per-leg vertical-force integrator against gravity sag.

        mode "reset" (CROUCH/WAIT_CROUCH/STAND) zeroes the trim, "freeze"
        (HOLD_SAG/PARK/WAIT_PARK) keeps it -- PARK fades the applied value
        through the shared progress ramp -- and "integrate"
        (WAIT_STAND/HOLD/HOLD_PARTIAL) runs the loop.  A leg whose torque
        or Cartesian force is saturated may only unwind, never wind up.
        """
        now = float(now)
        if mode == "reset":
            for leg in LEGS:
                self.z_trim_n[leg] = 0.0
            self._trim_last_time = now
            return
        if mode == "freeze":
            self._trim_last_time = now
            return
        if mode != "integrate":
            raise ValueError(f"unknown trim mode: {mode}")
        if self._trim_last_time is None:
            self._trim_last_time = now
            return
        dt = float(np.clip(now - self._trim_last_time, 0.0, 0.1))
        self._trim_last_time = now
        saturated = saturated_legs or set()
        for leg in LEGS:
            sag = self.last_leg_sag_m[leg]
            if abs(sag) < TRIM_DEADBAND_M:
                continue
            increment = TRIM_KI_N_PER_M_S * sag * dt
            trim = self.z_trim_n[leg]
            if leg in saturated and abs(trim + increment) > abs(trim):
                continue
            self.z_trim_n[leg] = float(
                np.clip(trim + increment, -TRIM_CLAMP_N, TRIM_CLAMP_N)
            )

    def _compute_cartesian(self, q, qd, target):
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        tau = np.zeros(N_JOINTS)
        max_error = 0.0
        min_singular = np.inf
        max_force = 0.0

        for leg_index, leg in enumerate(LEGS):
            section = slice(3 * leg_index, 3 * leg_index + 3)
            q_leg, qd_leg = q[section], qd[section]
            foot = dog5_kinematics.foot_position(leg, q_leg)
            jacobian = dog5_kinematics.foot_jacobian(leg, q_leg)
            singular = float(np.min(np.linalg.svd(jacobian, compute_uv=False)))
            min_singular = min(min_singular, singular)
            if singular < recorded.MIN_JACOBIAN_SINGULAR:
                raise RuntimeError(
                    f"{leg} Jacobian near singular: sigma_min={singular:.4f} "
                    f"< {recorded.MIN_JACOBIAN_SINGULAR:.4f} m/rad"
                )

            desired, desired_velocity, progress = target(leg)
            error = desired - foot
            # Positive sag: the body hangs below its target, so the foot
            # sits above the desired point in the trunk frame.
            self.last_leg_sag_m[leg] = -float(error[2])
            error_norm = float(np.linalg.norm(error))
            max_error = max(max_error, error_norm)
            if error_norm > recorded.MAX_CART_ERROR_M:
                raise RuntimeError(
                    f"{leg} Cartesian error {error_norm:.3f} m exceeds "
                    f"{recorded.MAX_CART_ERROR_M:.3f} m"
                )

            foot_velocity = jacobian @ qd_leg
            force = (
                self.kp_cart @ error
                + self.kd_cart @ (desired_velocity - foot_velocity)
            )
            # Weight support and integral trim ride the same progress ramp:
            # both grow through STAND and fade through PARK.
            force[2] -= progress * (
                self.travel_scale * self.support_per_foot
                + self.z_trim_n[leg]
            )
            force_norm = float(np.linalg.norm(force))
            clipped = force_norm > recorded.MAX_CART_FORCE_N
            self.last_leg_force_clipped[leg] = clipped
            if clipped:
                force *= recorded.MAX_CART_FORCE_N / force_norm
                force_norm = recorded.MAX_CART_FORCE_N
            max_force = max(max_force, force_norm)
            tau[section] = (
                jacobian.T @ force - stand_dog5.KD_JOINT_STAND * qd_leg
            )

        self.last_cart_error = max_error
        self.last_min_singular = min_singular
        self.last_max_force = max_force
        if not np.all(np.isfinite(tau)):
            raise RuntimeError("Cartesian controller produced non-finite torque")
        return tau


class InPlaceStandSequence(recorded.RecordedStandSequence):
    """Recorded sequence plus a WAIT_STAND timeout into HOLD_SAG."""

    def __init__(self, now, start_q, travel_scale=1.0):
        super().__init__(now, start_q, travel_scale)
        self.sag_rebased = False

    def update(self, now, q=None, cart_error=None, qd=None):
        if self.stage == "CROUCH":
            if now - self.started_at > recorded.CROUCH_TIMEOUT_S:
                raise RuntimeError(
                    f"native position CROUCH timed out after "
                    f"{recorded.CROUCH_TIMEOUT_S:.1f} s"
                )
            if q is None or qd is None:
                self.crouch_settle_since = None
            else:
                pose_error = float(
                    np.max(np.abs(np.asarray(q) - Q_RECORDED_CROUCH))
                )
                speed = float(np.max(np.abs(qd)))
                settled = (
                    pose_error <= recorded.RECORDED_POSE_TOL
                    and speed <= recorded.RECORDED_QD_TOL
                )
                if not settled:
                    self.crouch_settle_since = None
                elif self.crouch_settle_since is None:
                    self.crouch_settle_since = float(now)
                elif now - self.crouch_settle_since >= recorded.CROUCH_SETTLE_S:
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
                "target until Cartesian tracking settles "
                f"(HOLD_SAG fallback after {WAIT_STAND_TIMEOUT_S:.0f} s)."
            )
        if self.stage == "WAIT_STAND":
            if now - self.wait_since >= WAIT_STAND_TIMEOUT_S:
                self.stage = "HOLD_SAG"
                self.wait_since = float(now)
                self.final_settle_since = None
                self.sag_rebased = True
                error_text = (
                    "unknown" if cart_error is None else f"{cart_error:.3f} m"
                )
                return (
                    "WARNING: WAIT_STAND did not settle within "
                    f"{WAIT_STAND_TIMEOUT_S:.0f} s (cart_err={error_text}); "
                    "rebasing the hold target to the achieved foot heights; "
                    "HOLD_SAG is active."
                )
            speed = float(np.max(np.abs(qd))) if qd is not None else np.inf
            settled = (
                cart_error is not None
                and cart_error <= recorded.FINAL_CART_TOL_M
                and speed <= recorded.FINAL_QD_TOL
            )
            if not settled:
                self.final_settle_since = None
            elif self.final_settle_since is None:
                self.final_settle_since = float(now)
            elif now - self.final_settle_since >= recorded.FINAL_DWELL_S:
                self.stage = (
                    "HOLD" if self.travel_scale >= 1.0 else "HOLD_PARTIAL"
                )
                self.wait_since = float(now)
                if self.stage == "HOLD":
                    return (
                        "Full Cartesian target tracked and settled; HOLD is "
                        "active."
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
            speed = float(np.max(np.abs(qd))) if qd is not None else np.inf
            settled = (
                cart_error is not None
                and cart_error <= recorded.FINAL_CART_TOL_M
                and speed <= recorded.FINAL_QD_TOL
            )
            if not settled:
                self.park_settle_since = None
            elif self.park_settle_since is None:
                self.park_settle_since = float(now)
            elif now - self.park_settle_since >= recorded.FINAL_DWELL_S:
                self.stage = "WAIT_CROUCH"
                self.wait_since = float(now)
                return (
                    "PARK reached the recorded crouch; native position hold "
                    "is active."
                )
        return None

    def request_next(self, now, q, qd, healthy):
        if self.stage == "HOLD_SAG":
            return False, "Already in final HOLD_SAG; P parks and X stops."
        return super().request_next(now, q, qd, healthy)

    def request_park(self, now, healthy):
        if self.stage == "HOLD_SAG":
            if not healthy:
                return False, "PARK refused: motor latch/fault present."
            self.stage = "PARK"
            self.started_at = float(now)
            self.wait_since = None
            self.park_settle_since = None
            return True, "P accepted: starting Cartesian PARK back to crouch."
        return super().request_park(now, healthy)


def validate_inplace_configuration(controller):
    """Validate the vertical path and return its worst static mg/4 torque.

    Reuses the recorded runner's full validation (direction conversion,
    soft limits, damped-least-squares IK walk of crouch -> clearance ->
    final with residual/limit/singularity checks -- the walk follows this
    controller's vertical targets).  Adds the stand-in-place guarantee and
    an IK re-walk that accumulates max |J^T [0,0,-mg/4]| along the path.
    """
    recorded.validate_recorded_configuration(controller)

    weight = np.array([0.0, 0.0, -base.DOG5_MASS_KG * 9.81 / 4.0])
    max_static_tau = 0.0
    for leg_index, leg in enumerate(LEGS):
        if not np.allclose(
            controller.final_foot[leg][:2],
            controller.crouch_foot[leg][:2],
            atol=1.0e-9,
        ):
            raise ValueError(
                f"{leg} final foot target moved horizontally; stand-in-place "
                "requires vertical-only travel"
            )
        section = slice(3 * leg_index, 3 * leg_index + 3)
        q_path = Q_RECORDED_CROUCH[section].copy()
        path_segments = (
            (controller.crouch_foot[leg], controller.clearance_foot[leg]),
            (controller.clearance_foot[leg], controller.final_foot[leg]),
        )
        for segment_start, segment_end in path_segments:
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
                jacobian = dog5_kinematics.foot_jacobian(leg, q_path)
                max_static_tau = max(
                    max_static_tau,
                    float(np.max(np.abs(jacobian.T @ weight))),
                )
    return max_static_tau


def _trim_mode_for_stage(stage):
    if stage in ("WAIT_STAND", "HOLD", "HOLD_PARTIAL"):
        return "integrate"
    if stage in ("HOLD_SAG", "PARK", "WAIT_PARK"):
        return "freeze"
    return "reset"


def _settle_text(sequence, now):
    if sequence.stage == "WAIT_STAND":
        remaining = max(
            0.0, WAIT_STAND_TIMEOUT_S - (now - sequence.wait_since)
        )
        if sequence.final_settle_since is not None:
            dwell = now - sequence.final_settle_since
            return (
                f"dwell {dwell:.1f}/{recorded.FINAL_DWELL_S:.1f}s, "
                f"timeout {remaining:.1f}s"
            )
        return f"not settled, timeout {remaining:.1f}s"
    if sequence.stage in ("HOLD", "HOLD_PARTIAL"):
        return "held"
    if sequence.stage == "HOLD_SAG":
        return "rebased"
    if sequence.stage == "WAIT_PARK":
        if sequence.park_settle_since is not None:
            dwell = now - sequence.park_settle_since
            return f"park dwell {dwell:.1f}/{recorded.FINAL_DWELL_S:.1f}s"
        return "park not settled"
    return "-"


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
    controller = InPlaceStandController(
        cart_gain_scale, support_scale, stand_height, travel_scale
    )
    max_static_tau = validate_inplace_configuration(controller)
    gate = ConfirmedSafetyGate(tau_max, qd_estop, qd_estop_hard)
    unwrap = [base.CalibratedEncoderUnwrap() for _ in base.HARDWARE_JOINTS]

    print(
        "[inplace] Flow: CURRENT -> CROUCH -> ENTER -> STAND -> WAIT_STAND "
        "-> HOLD/HOLD_PARTIAL (or HOLD_SAG on timeout) -> P -> PARK -> "
        "WAIT_PARK -> WAIT_CROUCH"
    )
    print(
        "[inplace] STAND-IN-PLACE: feet rise straight up from the recorded "
        "crouch xy; no commanded horizontal foot travel."
    )
    print(
        "[inplace] crouch joint targets deg: "
        + ", ".join(
            f"{leg}={recorded.RECORDED_CROUCH_DEG[leg]}" for leg in LEGS
        )
    )
    print(
        f"[inplace] native position CROUCH: speed cap="
        f"{crouch_max_speed_dps:.0f} motor-deg/s, measured torque trip="
        f"{crouch_torque_trip:.2f}N*m, encoder speed trip="
        f"{crouch_speed_trip:.2f}rad/s, timeout={recorded.CROUCH_TIMEOUT_S:.0f}s"
    )
    print(
        f"[inplace] Cartesian STAND time={T_STAND:.1f}s; "
        f"height={stand_height:.3f}m, tau cap={tau_max:.2f}N*m, "
        f"travel={travel_scale:.2f}, cart scale={cart_gain_scale:.2f}, "
        f"support scale={support_scale:.2f}"
    )
    print(
        f"[inplace] support trim: Ki={TRIM_KI_N_PER_M_S:.0f} N/(m*s), "
        f"clamp +/-{TRIM_CLAMP_N:.1f} N, deadband "
        f"{1000.0 * TRIM_DEADBAND_M:.0f} mm; active in WAIT_STAND and HOLD"
    )
    print(
        f"[inplace] hard overspeed needs confirmation: driver+encoder same "
        f"check, encoder alone <= {QD_ENC_GLITCH_RAD_S:.0f} rad/s, or driver "
        f"x{QD_ESTOP_HARD_STREAK} consecutive checks"
    )
    print(
        f"[inplace] static mg/4 torque along path: ~{max_static_tau:.2f} N*m"
    )
    if tau_max < max_static_tau:
        print(
            f"[inplace] WARNING: --tau-max {tau_max:.2f} N*m is below the "
            f"~{max_static_tau:.2f} N*m static requirement; the robot cannot "
            "hold its own weight at this cap.  Keep it supported, or use "
            f"--tau-max {base.STAGED_TAU_MAX:.1f} for an unaided full stand."
        )
    for leg in LEGS:
        print(
            f"[inplace] {leg} foot: crouch="
            f"{np.array2string(controller.crouch_foot[leg], precision=4)} -> "
            f"stand={np.array2string(controller.final_foot[leg], precision=4)}"
        )
    if travel_scale < 1.0:
        final_heights = [
            -float(controller.final_foot[leg][2]) for leg in LEGS
        ]
        print(
            "[inplace] PARTIAL TEST: --travel-scale "
            f"{travel_scale:.2f} commands only {100.0 * travel_scale:.0f}% "
            f"of the stand path (final leg heights "
            f"{min(final_heights):.3f}..{max(final_heights):.3f} m). "
            "It is not expected to stand fully."
        )
    print(
        "[inplace] ROBOT MUST REMAIN MECHANICALLY SUPPORTED UNTIL THE FULL "
        "STAND IS VALIDATED. P parks from HOLD; X stops."
    )

    key = base.KeyPoller()
    try:
        with base.motorbus.MotorBus(
            MOTOR_IDS, dirs=MOTOR_DIRECTIONS
        ) as mb:
            armed = False
            stop_reason = None
            try:
                print("[inplace] Arming with a zero-torque stream...")
                if not mb.arm(rate_hz=base.CONTROL_HZ):
                    raise RuntimeError("not all motors armed")
                armed = True

                start_q = recorded.zero_torque_preflight(mb, key, unwrap)
                now = time.perf_counter()
                sequence = InPlaceStandSequence(now, start_q, travel_scale)
                gate.start(now, start_q)
                print(
                    "[inplace] ENTER accepted: native position control moving "
                    "CURRENT -> recorded CROUCH. Do not press Enter while "
                    "moving."
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
                velocity = recorded.EncoderVelocity()

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
                                    f"{recorded.FINAL_CART_TOL_M:.3f} m and "
                                    f"max|qd_enc| <= "
                                    f"{recorded.FINAL_QD_TOL:.2f} rad/s; "
                                    f"HOLD_SAG after "
                                    f"{WAIT_STAND_TIMEOUT_S:.0f} s."
                                )
                            elif sequence.stage == "HOLD_SAG":
                                controller.rebase_final_z_to_measured(q)
                                for leg in LEGS:
                                    print(
                                        f"[stage]   {leg} rebased hold "
                                        "height "
                                        f"{-controller.final_foot[leg][2]:.3f} "
                                        f"m (commanded "
                                        f"{-(controller.crouch_foot[leg][2] + controller.travel_scale * (controller.full_stand_foot[leg][2] - controller.crouch_foot[leg][2])):.3f} m)"
                                    )
                                print(
                                    "[stage] HOLD_SAG: holding the achieved "
                                    "pose, not the commanded height; "
                                    "P parks, X stops."
                                )
                            elif sequence.stage == "WAIT_PARK":
                                print(
                                    "[stage] Waiting for crouch cart_err <= "
                                    f"{recorded.FINAL_CART_TOL_M:.3f} m and "
                                    f"max|qd_enc| <= "
                                    f"{recorded.FINAL_QD_TOL:.2f} rad/s."
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
                        requested_tau = None
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

                        trim_mode = _trim_mode_for_stage(sequence.stage)
                        saturated_legs = set()
                        if trim_mode == "integrate" and requested_tau is not None:
                            cap = gate.cap_now(now)
                            for leg_index, leg in enumerate(LEGS):
                                section = slice(
                                    3 * leg_index, 3 * leg_index + 3
                                )
                                torque_saturated = bool(
                                    np.any(
                                        np.abs(requested_tau[section])
                                        >= cap - 1.0e-9
                                    )
                                )
                                if (
                                    torque_saturated
                                    or controller.last_leg_force_clipped[leg]
                                ):
                                    saturated_legs.add(leg)
                        controller.update_support_trim(
                            now, trim_mode, saturated_legs
                        )

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
                                    "HOLD_SAG",
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
                            stop_reason = recorded.position_stage_fault(
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
                            if advanced and sequence.stage == "STAND":
                                controller.restore_full_targets()
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
                                f"[inplace] stage={sequence.stage:11s} "
                                f"crouch_err={pose_error:.2f}rad "
                                f"cart_err={controller.last_cart_error:.3f}m "
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
                            if not position_stage:
                                sag_text = ",".join(
                                    f"{leg}"
                                    f"{1000.0 * controller.last_leg_sag_m[leg]:+.1f}"
                                    for leg in LEGS
                                )
                                trim_text = ",".join(
                                    f"{controller.z_trim_n[leg]:+.1f}"
                                    for leg in LEGS
                                )
                                print(
                                    f"[legs] sag_mm=({sag_text}) "
                                    f"trim_N=({trim_text}) "
                                    f"settle={_settle_text(sequence, now)}",
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
                    print(f"[inplace] stopping: {stop_reason or 'aborted'}")
                    try:
                        base._soft_stop(mb)
                    except Exception as exc:
                        print(
                            f"[inplace] soft stop failed: {exc}; sending STOP",
                            file=sys.stderr,
                        )
    finally:
        key.close()
    return 0


def offline_self_test(stand_height=recorded.DEFAULT_STAND_HEIGHT,
                      travel_scale=1.0):
    controller = InPlaceStandController(
        0.25, 0.25, stand_height, travel_scale
    )
    max_static = validate_inplace_configuration(controller)
    assert 0.8 < max_static < 1.3, max_static
    zero = np.zeros(N_JOINTS)
    check_dt = N_JOINTS / base.CONTROL_HZ

    # Stand-in-place targets: no horizontal travel, requested height reached.
    for leg in LEGS:
        assert np.allclose(
            controller.final_foot[leg][:2], controller.crouch_foot[leg][:2]
        ), leg
        assert np.isclose(
            controller.full_stand_foot[leg][2], -stand_height
        ), leg
        if travel_scale >= 1.0:
            assert np.isclose(controller.final_foot[leg][2], -stand_height)

    # Height/travel variants must stay reachable and stand in place.
    for variant_height, variant_travel in ((MAX_STAND_HEIGHT, 1.0),
                                           (stand_height, 0.25)):
        variant = InPlaceStandController(
            0.25, 0.25, variant_height, variant_travel
        )
        variant_static = validate_inplace_configuration(variant)
        assert 0.8 < variant_static < 1.3, variant_static

    # No transition step: STAND starts and PARK ends torque-free.
    cart_start = controller.compute_stand(Q_RECORDED_CROUCH, zero, 0.0)
    assert np.allclose(cart_start, 0.0, atol=1.0e-10)
    park_end_tau = controller.compute_park(Q_RECORDED_CROUCH, zero, T_PARK)
    assert np.allclose(park_end_tau, 0.0, atol=1.0e-10)

    # Trim integrator: rate, anti-windup, clamp, reset.
    trim_controller = InPlaceStandController(0.25, 0.25, stand_height, 1.0)
    for leg in LEGS:
        trim_controller.last_leg_sag_m[leg] = 0.020
    trim_controller.update_support_trim(0.0, "integrate")
    assert all(trim_controller.z_trim_n[leg] == 0.0 for leg in LEGS)
    trim_controller.update_support_trim(0.1, "integrate")
    expected = TRIM_KI_N_PER_M_S * 0.020 * 0.1
    for leg in LEGS:
        assert np.isclose(trim_controller.z_trim_n[leg], expected), (
            leg, trim_controller.z_trim_n[leg]
        )
    trim_controller.update_support_trim(0.2, "integrate", {"FL"})
    assert np.isclose(trim_controller.z_trim_n["FL"], expected)
    assert np.isclose(trim_controller.z_trim_n["FR"], 2 * expected)
    trim_controller.last_leg_sag_m["FL"] = -0.020
    trim_controller.update_support_trim(0.3, "integrate", {"FL"})
    assert np.isclose(trim_controller.z_trim_n["FL"], 0.0), (
        "a saturated leg must still be allowed to unwind"
    )
    below_deadband = trim_controller.z_trim_n["FR"]
    trim_controller.last_leg_sag_m["FR"] = 0.5 * TRIM_DEADBAND_M
    trim_controller.update_support_trim(0.4, "integrate")
    assert np.isclose(trim_controller.z_trim_n["FR"], below_deadband)
    for leg in LEGS:
        trim_controller.last_leg_sag_m[leg] = 0.5
    trim_controller.update_support_trim(0.5, "integrate")
    trim_controller.update_support_trim(0.6, "integrate")
    assert np.isclose(trim_controller.z_trim_n["RL"], TRIM_CLAMP_N)
    frozen = dict(trim_controller.z_trim_n)
    trim_controller.update_support_trim(0.7, "freeze")
    assert trim_controller.z_trim_n == frozen
    trim_controller.update_support_trim(0.8, "reset")
    assert all(trim_controller.z_trim_n[leg] == 0.0 for leg in LEGS)

    # Trim rides the progress ramp: full effect at PARK start, none at end.
    fade = InPlaceStandController(0.25, 0.25, stand_height, 0.05)
    q_crouch = Q_RECORDED_CROUCH.copy()
    baseline_start = fade.compute_park(q_crouch, zero, 0.0)
    baseline_end = fade.compute_park(q_crouch, zero, T_PARK)
    for leg in LEGS:
        fade.z_trim_n[leg] = 2.0
    trimmed_start = fade.compute_park(q_crouch, zero, 0.0)
    trimmed_end = fade.compute_park(q_crouch, zero, T_PARK)
    assert not np.allclose(trimmed_start, baseline_start)
    assert np.allclose(trimmed_end, baseline_end, atol=1.0e-10)

    # Rebase: adopt measured heights, keep xy, restore undoes it.
    rebase = InPlaceStandController(0.25, 0.25, stand_height, 1.0)
    rebase.rebase_final_z_to_measured(Q_RECORDED_CROUCH)
    for leg in LEGS:
        assert np.allclose(
            rebase.final_foot[leg][:2], rebase.crouch_foot[leg][:2]
        )
        assert np.isclose(
            rebase.final_foot[leg][2], rebase.crouch_foot[leg][2]
        )
    rebase.restore_full_targets()
    for leg in LEGS:
        assert np.isclose(rebase.final_foot[leg][2], -stand_height)

    # Sequence: normal settle path unchanged, timeout path added.
    sequence = InPlaceStandSequence(0.0, Q_RECORDED_CROUCH, travel_scale)
    assert sequence.update(0.10, q=Q_RECORDED_CROUCH, qd=zero) is None
    assert sequence.update(
        0.10 + recorded.CROUCH_SETTLE_S + 0.01, q=Q_RECORDED_CROUCH, qd=zero
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
    settle_start = stand_end + 0.02
    assert sequence.update(
        settle_start, q=Q_RECORDED_CROUCH, cart_error=0.0, qd=zero
    ) is None
    assert sequence.update(
        settle_start + recorded.FINAL_DWELL_S + 0.01,
        q=Q_RECORDED_CROUCH,
        cart_error=0.0,
        qd=zero,
    )
    expected_hold = "HOLD" if travel_scale >= 1.0 else "HOLD_PARTIAL"
    assert sequence.stage == expected_hold
    assert not sequence.sag_rebased

    timeout_sequence = InPlaceStandSequence(0.0, Q_RECORDED_CROUCH, 1.0)
    timeout_sequence.stage = "WAIT_STAND"
    timeout_sequence.wait_since = 100.0
    assert timeout_sequence.update(
        100.0 + WAIT_STAND_TIMEOUT_S - 0.1,
        q=Q_RECORDED_CROUCH,
        cart_error=0.05,
        qd=zero,
    ) is None
    assert timeout_sequence.stage == "WAIT_STAND"
    event = timeout_sequence.update(
        100.0 + WAIT_STAND_TIMEOUT_S + 0.01,
        q=Q_RECORDED_CROUCH,
        cart_error=0.05,
        qd=zero,
    )
    assert event and "HOLD_SAG" in event and "WARNING" in event, event
    assert timeout_sequence.stage == "HOLD_SAG"
    assert timeout_sequence.sag_rebased
    advanced, reason = timeout_sequence.request_next(
        130.0, Q_RECORDED_CROUCH, zero, True
    )
    assert not advanced and "HOLD_SAG" in reason
    parked, reason = timeout_sequence.request_park(131.0, False)
    assert not parked and "fault" in reason
    parked, reason = timeout_sequence.request_park(132.0, True)
    assert parked, reason
    assert timeout_sequence.stage == "PARK"

    # Confirmed hard overspeed tier.
    knee = JOINT_LABELS.index("RR_knee")
    q_still = np.zeros(N_JOINTS)
    spike = np.zeros(N_JOINTS)
    spike[knee] = base.QD_ESTOP_HARD + 0.1

    persistent = ConfirmedSafetyGate(1.0)
    persistent.start(0.0, q_still)
    assert persistent.overspeed_reason(spike, q_still, check_dt) is None
    second = persistent.overspeed_reason(spike, q_still, 2 * check_dt)
    assert second and "hard limit" in second and "driver x2" in second, second
    assert "confirmed overspeed" not in second

    confirmed = ConfirmedSafetyGate(1.0)
    confirmed.start(0.0, q_still)
    moved = q_still.copy()
    moved[knee] = spike[knee] * check_dt
    first = confirmed.overspeed_reason(spike, moved, check_dt)
    assert first and "hard limit" in first and "driver+encoder" in first, first

    encoder_only = ConfirmedSafetyGate(1.0)
    encoder_only.start(0.0, q_still)
    stepped = q_still.copy()
    stepped[knee] = 9.0 * check_dt
    encoder_trip = encoder_only.overspeed_reason(zero, stepped, check_dt)
    assert encoder_trip and "hard limit" in encoder_trip, encoder_trip
    assert "(encoder)" in encoder_trip, encoder_trip

    phantom = ConfirmedSafetyGate(1.0)
    phantom.start(0.0, q_still)
    assert phantom.overspeed_reason(spike, q_still, check_dt) is None
    assert phantom.overspeed_reason(zero, q_still, 2 * check_dt) is None
    assert int(np.max(phantom.hard_streaks)) == 0
    assert phantom.overspeed_reason(spike, q_still, 3 * check_dt) is None

    glitch = ConfirmedSafetyGate(1.0)
    glitch.start(0.0, q_still)
    jumped = q_still.copy()
    jumped[knee] = 40.0 * check_dt
    assert glitch.overspeed_reason(zero, jumped, check_dt) is None

    # Sustained tier is inherited unchanged: confirmed motion trips after
    # QD_ESTOP_STREAK checks, an unconfirmed noisy driver field never does.
    sustained = ConfirmedSafetyGate(1.0)
    sustained_q = np.zeros(N_JOINTS)
    sustained.start(0.0, sustained_q)
    sustained_speed = np.zeros(N_JOINTS)
    sustained_speed[knee] = base.QD_ESTOP + 0.2
    reason = None
    for sample in range(1, base.QD_ESTOP_STREAK + 1):
        sustained_q[knee] = sample * check_dt * sustained_speed[knee]
        reason = sustained.overspeed_reason(
            sustained_speed, sustained_q, sample * check_dt
        )
        if sample < base.QD_ESTOP_STREAK:
            assert reason is None
    assert reason and "confirmed overspeed RR_knee" in reason, reason

    noisy = ConfirmedSafetyGate(1.0)
    noisy.start(0.0, q_still)
    noisy_speed = np.zeros(N_JOINTS)
    noisy_speed[knee] = base.QD_ESTOP + 0.2
    for sample in range(1, 3 * base.QD_ESTOP_STREAK + 1):
        assert noisy.overspeed_reason(
            noisy_speed, q_still, sample * check_dt
        ) is None

    print("stand_dog5_inplace_hw offline self-test PASS")
    for leg in LEGS:
        print(
            f"  {leg}: crouch foot="
            f"{np.array2string(controller.crouch_foot[leg], precision=4)}, "
            f"final={np.array2string(controller.final_foot[leg], precision=4)}"
            " (vertical-only)"
        )
    print(f"  static mg/4 torque along path: ~{max_static:.2f} N*m")
    print(
        f"  support trim: Ki={TRIM_KI_N_PER_M_S:.0f} N/(m*s), clamp "
        f"+/-{TRIM_CLAMP_N:.1f} N, deadband {1000.0 * TRIM_DEADBAND_M:.0f} mm"
    )
    print(
        f"  WAIT_STAND timeout: {WAIT_STAND_TIMEOUT_S:.0f} s -> HOLD_SAG "
        "(rebased hold, park allowed)"
    )
    print(
        "  hard overspeed: driver+encoder same check, encoder alone <= "
        f"{QD_ENC_GLITCH_RAD_S:.0f} rad/s, or driver x"
        f"{QD_ESTOP_HARD_STREAK}; sustained tier unchanged"
    )
    print(
        "  flow: CURRENT -> CROUCH -> ENTER -> STAND -> WAIT_STAND -> "
        f"{expected_hold} (HOLD_SAG on timeout) -> P -> PARK -> WAIT_PARK "
        "-> WAIT_CROUCH"
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
            "Cartesian STAND per-joint torque cap, N*m; a full unaided "
            f"stand needs about {base.STAGED_TAU_MAX:.1f} "
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
        default=DEFAULT_SUPPORT_SCALE,
        help=(
            "full-path mg/4 feedforward scale in [0,1]; partial travel "
            "scales it proportionally; the integral trim covers the "
            f"remainder (default: {DEFAULT_SUPPORT_SCALE})"
        ),
    )
    parser.add_argument(
        "--stand-height",
        type=float,
        default=recorded.DEFAULT_STAND_HEIGHT,
        help=(
            "final trunk-frame hip-to-foot height, m "
            f"(default: {recorded.DEFAULT_STAND_HEIGHT}, max "
            f"{MAX_STAND_HEIGHT} on the vertical path)"
        ),
    )
    parser.add_argument(
        "--travel-scale",
        type=float,
        default=1.0,
        help=(
            "fraction of the complete crouch-to-stand vertical path; "
            "use 0.25 for the first supported test (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--crouch-max-speed-dps",
        type=float,
        default=recorded.DEFAULT_CROUCH_MAX_MOTOR_DPS,
        help=(
            "native position-command motor-side speed cap, deg/s; output "
            "speed is approximately one tenth "
            f"(default: {recorded.DEFAULT_CROUCH_MAX_MOTOR_DPS})"
        ),
    )
    parser.add_argument(
        "--crouch-torque-trip",
        type=float,
        default=recorded.DEFAULT_CROUCH_TORQUE_TRIP_NM,
        help=(
            "stop native position motion when measured joint torque exceeds "
            "this value, N*m "
            f"(default: {recorded.DEFAULT_CROUCH_TORQUE_TRIP_NM})"
        ),
    )
    parser.add_argument(
        "--crouch-speed-trip",
        type=float,
        default=recorded.DEFAULT_CROUCH_SPEED_TRIP_RAD_S,
        help=(
            "stop native position motion when encoder-derived joint speed "
            "exceeds this value, rad/s "
            f"(default: {recorded.DEFAULT_CROUCH_SPEED_TRIP_RAD_S})"
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
            "hard speed trip, rad/s; needs encoder confirmation, an "
            "encoder-alone overspeed, or two consecutive driver samples "
            f"(default: {base.QD_ESTOP_HARD})"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate targets, trim, sequence, and overspeed without CAN",
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
        print(f"[inplace] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
