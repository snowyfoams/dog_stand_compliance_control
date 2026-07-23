#!/usr/bin/env python3
"""Stand DOG5, then extend the FL and RR foot endpoints on real hardware.

This runner keeps the stand-in-place path from ``stand_dog5_inplace_hw.py``.
After the full standing target reaches HOLD, press Enter once more to move the
front-left and rear-right foot targets downward along trunk Z.  The diagonal
extension is a finite cubic ramp and its final endpoints remain held:

    CURRENT -> CROUCH -> ENTER -> STAND -> WAIT_STAND -> HOLD
                                                    ENTER -> FL_RR_EXTEND
                                                          -> EXTENDED_HOLD

The stage name remains HOLD inside the shared hardware loop so all standing
safety gates and support-trim behavior stay active.  Status messages identify
the extension motion explicitly.  P parks only before the extension starts or
after it finishes; X stops at any time.

"Extend" here means increasing the hip-to-foot distance: the target Z for FL
and RR becomes more negative.  FR and RL keep their normal standing targets,
and no foot receives an x/y command.  This deliberately creates asymmetric
diagonal loading.  Encoder-only control cannot measure body attitude or foot
contact, so keep the robot mechanically supported/tethered.

Suggested bring-up::

    python stand_dog5_fl_rr_extend_hw.py --self-test
    python stand_dog5_fl_rr_extend_hw.py --tau-max 3.0 --extension-mm 10
    python stand_dog5_fl_rr_extend_hw.py --tau-max 3.0 --extension-mm 20
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import stand_dog5_hw as base
import stand_dog5_inplace_hw as inplace
import stand_dog5_recorded_hw as recorded


LEGS = base.LEGS
N_JOINTS = base.N_JOINTS
Q_RECORDED_CROUCH = recorded.Q_RECORDED_CROUCH
EXTEND_LEGS = frozenset(("FL", "RR"))
DEFAULT_EXTENSION_MM = 20.0
DEFAULT_EXTENSION_TIME_S = 3.0
MIN_EXTENSION_MM = 1.0
MAX_EXTENSION_MM = 30.0
MIN_EXTENSION_TIME_S = 0.5
MAX_EXTENSION_TIME_S = 10.0
# The complete continuous IK path from the recorded crouch was validated with
# FL/RR at 0.22 m.  A 0.23 m target fails the reachability residual check.
MAX_EXTENDED_HEIGHT_M = 0.22


class DiagonalExtendController(inplace.InPlaceStandController):
    """In-place stand controller with an operator-started FL/RR Z ramp."""

    def __init__(
        self,
        cart_gain_scale,
        support_scale,
        stand_height,
        travel_scale,
        extension_m,
        extension_time_s,
    ):
        super().__init__(
            cart_gain_scale, support_scale, stand_height, travel_scale
        )
        self.extension_m = float(extension_m)
        self.extension_time_s = float(extension_time_s)
        self.extension_state = "idle"
        self.extension_started_at = None
        self.extension_elapsed_s = 0.0
        self._normal_final_foot = {}
        self.extended_final_foot = {}
        self._rebuild_extension_targets()

    def _rebuild_extension_targets(self):
        self._normal_final_foot = {
            leg: self.final_foot[leg].copy() for leg in LEGS
        }
        self.extended_final_foot = {
            leg: self.final_foot[leg].copy() for leg in LEGS
        }
        for leg in EXTEND_LEGS:
            self.extended_final_foot[leg][2] -= self.extension_m

    @property
    def extension_progress(self):
        if self.extension_state == "idle":
            return 0.0
        if self.extension_state == "held":
            return 1.0
        return float(
            np.clip(
                self.extension_elapsed_s / self.extension_time_s,
                0.0,
                1.0,
            )
        )

    def start_extension(self, now):
        if self.travel_scale < 1.0:
            raise RuntimeError("FL/RR extension requires a full stand")
        if self.extension_state != "idle":
            raise RuntimeError(
                f"FL/RR extension is already {self.extension_state}"
            )
        self.extension_state = "moving"
        self.extension_started_at = float(now)
        self.extension_elapsed_s = 0.0

    def advance_extension(self, now):
        """Advance the target clock and return an event on completion."""
        if self.extension_state != "moving":
            return None
        self.extension_elapsed_s = max(
            0.0, float(now) - self.extension_started_at
        )
        blend, _ = self._cubic(
            self.extension_elapsed_s, self.extension_time_s
        )
        for leg in LEGS:
            delta = (
                self.extended_final_foot[leg]
                - self._normal_final_foot[leg]
            )
            self.final_foot[leg] = (
                self._normal_final_foot[leg] + blend * delta
            )
        if self.extension_elapsed_s < self.extension_time_s:
            return None
        self.extension_elapsed_s = self.extension_time_s
        self.extension_state = "held"
        for leg in LEGS:
            self.final_foot[leg] = self.extended_final_foot[leg].copy()
        return (
            "FL/RR diagonal extension complete; EXTENDED_HOLD is active "
            f"at {1000.0 * self.extension_m:.1f} mm."
        )

    def extension_target(self, leg):
        if self.extension_state == "idle":
            return self._normal_final_foot[leg].copy(), np.zeros(3), 1.0
        if self.extension_state == "held":
            return self.extended_final_foot[leg].copy(), np.zeros(3), 1.0

        blend, blend_rate = self._cubic(
            self.extension_elapsed_s, self.extension_time_s
        )
        delta = self.extended_final_foot[leg] - self._normal_final_foot[leg]
        return (
            self._normal_final_foot[leg] + blend * delta,
            blend_rate * delta,
            1.0,
        )

    def compute_stand(self, q, qd, elapsed):
        if self.extension_state == "idle":
            return super().compute_stand(q, qd, elapsed)
        return self._compute_cartesian(q, qd, self.extension_target)

    def update_support_trim(self, now, mode, saturated_legs=None):
        # A moving target is not gravity sag.  Freeze the slow trim integrator
        # during the ramp so it cannot oppose the commanded extension.
        if self.extension_state == "moving" and mode == "integrate":
            mode = "freeze"
        return super().update_support_trim(now, mode, saturated_legs)

    def additional_status(self, stage, actual_heights, target_heights):
        """Show each leg height so diagonal motion is directly observable."""
        if stage in ("HOLD_PARTIAL", "HOLD_SAG"):
            return (
                f"[diagonal] extension=LOCKED stage={stage}; a settled full "
                "HOLD is required"
            )
        if stage != "HOLD":
            return None
        if self.extension_state == "idle":
            state = "READY (press ENTER)"
        elif self.extension_state == "moving":
            state = f"MOVING {100.0 * self.extension_progress:.0f}%"
        else:
            state = "EXTENDED_HOLD"
        actual = ",".join(
            f"{leg}{actual_heights[leg]:.3f}" for leg in LEGS
        )
        target = ",".join(
            f"{leg}{target_heights[leg]:.3f}" for leg in LEGS
        )
        return (
            f"[diagonal] extension={state} "
            f"h_actual_m=({actual}) h_target_m=({target})"
        )

    def restore_full_targets(self):
        """Restore the normal stand before a later stand/park cycle."""
        super().restore_full_targets()
        self.extension_state = "idle"
        self.extension_started_at = None
        self.extension_elapsed_s = 0.0
        self._rebuild_extension_targets()


class DiagonalExtendSequence(inplace.InPlaceStandSequence):
    """Add an Enter-gated diagonal extension while retaining HOLD safety."""

    def __init__(self, now, start_q, travel_scale, controller):
        super().__init__(now, start_q, travel_scale)
        self.controller = controller

    def update(self, now, q=None, cart_error=None, qd=None):
        event = super().update(now, q=q, cart_error=cart_error, qd=qd)
        if event is not None:
            if (
                self.stage == "HOLD"
                and self.controller.extension_state == "idle"
            ):
                return (
                    event
                    + " Inspect the standing pose, then press ENTER to "
                    "extend FL and RR."
                )
            return event
        if self.stage == "HOLD":
            event = self.controller.advance_extension(now)
            if event is not None:
                self.wait_since = float(now)
                return event
        return None

    def request_next(self, now, q, qd, healthy):
        if self.stage == "HOLD":
            if self.controller.extension_state == "moving":
                return False, (
                    "FL/RR extension is still moving; Enter ignored."
                )
            if self.controller.extension_state == "held":
                return False, (
                    "Already in EXTENDED_HOLD; P parks and X stops."
                )
            if not healthy:
                return False, (
                    "Motor latch/fault present; FL/RR extension refused."
                )
            max_speed = float(np.max(np.abs(qd)))
            if max_speed > recorded.FINAL_QD_TOL:
                return False, (
                    f"Robot not settled: max |qd| {max_speed:.2f} rad/s > "
                    f"{recorded.FINAL_QD_TOL:.2f}."
                )
            if self.controller.last_cart_error > recorded.FINAL_CART_TOL_M:
                return False, (
                    "Standing target is no longer settled: cart_err "
                    f"{self.controller.last_cart_error:.3f} m > "
                    f"{recorded.FINAL_CART_TOL_M:.3f} m."
                )
            self.controller.start_extension(now)
            return True, (
                "ENTER accepted: extending FL and RR downward by "
                f"{1000.0 * self.controller.extension_m:.1f} mm over "
                f"{self.controller.extension_time_s:.1f} s; FR/RL remain "
                "at the standing endpoints."
            )
        if self.stage == "HOLD_PARTIAL":
            return False, (
                "FL/RR extension requires the full standing HOLD; "
                "P parks and X stops."
            )
        if self.stage == "HOLD_SAG":
            return False, (
                "FL/RR extension is locked out from HOLD_SAG because the "
                "normal standing target did not settle; P parks and X stops."
            )
        return super().request_next(now, q, qd, healthy)

    def request_park(self, now, healthy):
        if self.controller.extension_state == "moving":
            return False, (
                "PARK refused while FL/RR targets are moving; wait for "
                "EXTENDED_HOLD or press X to stop."
            )
        return super().request_park(now, healthy)


def validate_diagonal_configuration(controller):
    """Validate the complete crouch-to-extended path, then restore HOLD."""
    if controller.travel_scale < 1.0:
        raise ValueError("FL/RR extension requires --travel-scale 1.0")
    normal_targets = {
        leg: controller.final_foot[leg].copy() for leg in LEGS
    }
    try:
        for leg in LEGS:
            controller.final_foot[leg] = (
                controller.extended_final_foot[leg].copy()
            )
        max_static_tau = inplace.validate_inplace_configuration(controller)
    finally:
        for leg in LEGS:
            controller.final_foot[leg] = normal_targets[leg]

    for leg in LEGS:
        delta = controller.extended_final_foot[leg] - normal_targets[leg]
        expected_z = -controller.extension_m if leg in EXTEND_LEGS else 0.0
        if not np.allclose(delta, [0.0, 0.0, expected_z], atol=1.0e-12):
            raise ValueError(f"incorrect extension target for {leg}: {delta}")
    return max_static_tau


def offline_self_test(
    stand_height=recorded.DEFAULT_STAND_HEIGHT,
    extension_m=DEFAULT_EXTENSION_MM / 1000.0,
    extension_time_s=DEFAULT_EXTENSION_TIME_S,
):
    controller = DiagonalExtendController(
        0.25,
        0.25,
        stand_height,
        1.0,
        extension_m,
        extension_time_s,
    )
    max_static = validate_diagonal_configuration(controller)

    for leg in LEGS:
        normal = controller.final_foot[leg]
        extended = controller.extended_final_foot[leg]
        assert np.allclose(normal[:2], extended[:2]), leg
        expected = extension_m if leg in EXTEND_LEGS else 0.0
        assert np.isclose(normal[2] - extended[2], expected), leg

    zero = np.zeros(N_JOINTS)
    baseline_targets = {
        leg: controller.cartesian_target(leg, recorded.T_STAND)
        for leg in LEGS
    }
    controller.start_extension(10.0)
    controller.advance_extension(10.0)
    for leg in LEGS:
        ramp_start = controller.extension_target(leg)
        assert np.allclose(ramp_start[0], baseline_targets[leg][0]), leg
        assert np.allclose(ramp_start[1], baseline_targets[leg][1]), leg
        assert np.isclose(ramp_start[2], baseline_targets[leg][2]), leg

    controller.advance_extension(10.0 + 0.5 * extension_time_s)
    for leg in LEGS:
        target, velocity, progress = controller.extension_target(leg)
        normal = controller._normal_final_foot[leg]
        extended = controller.extended_final_foot[leg]
        assert np.allclose(target, 0.5 * (normal + extended)), leg
        assert np.allclose(controller.final_foot[leg], target), leg
        assert progress == 1.0
        if leg in EXTEND_LEGS:
            assert velocity[2] < 0.0
        else:
            assert np.allclose(velocity, 0.0)
    height_status = controller.additional_status(
        "HOLD",
        {leg: -controller.final_foot[leg][2] for leg in LEGS},
        {leg: -controller.final_foot[leg][2] for leg in LEGS},
    )
    assert "MOVING 50%" in height_status, height_status
    expected_mid_height = stand_height + 0.5 * extension_m
    assert f"FL{expected_mid_height:.3f}" in height_status, height_status
    assert f"RR{expected_mid_height:.3f}" in height_status, height_status

    finish = controller.advance_extension(10.0 + extension_time_s + 0.01)
    assert finish and "EXTENDED_HOLD" in finish, finish
    assert controller.extension_state == "held"
    for leg in LEGS:
        assert np.allclose(
            controller.final_foot[leg], controller.extended_final_foot[leg]
        ), leg

    # Sequence gating: a fault and unsettled HOLD refuse motion, PARK is
    # locked during the ramp, and the completed extension can park.
    sequence_controller = DiagonalExtendController(
        0.25,
        0.25,
        stand_height,
        1.0,
        extension_m,
        extension_time_s,
    )
    sequence = DiagonalExtendSequence(
        0.0, Q_RECORDED_CROUCH, 1.0, sequence_controller
    )
    sequence.stage = "HOLD"
    sequence.wait_since = 1.0
    sequence_controller.last_cart_error = 0.0
    # WAIT_STAND already enforced the settling dwell before entering HOLD, so
    # the first Enter after the HOLD message must be accepted immediately.
    started_at = 1.0
    advanced, reason = sequence.request_next(
        started_at, Q_RECORDED_CROUCH, zero, False
    )
    assert not advanced and "fault" in reason, reason
    advanced, reason = sequence.request_next(
        started_at, Q_RECORDED_CROUCH, zero, True
    )
    assert advanced and "extending FL and RR" in reason, reason
    parked, reason = sequence.request_park(started_at + 0.1, True)
    assert not parked and "moving" in reason, reason
    event = sequence.update(
        started_at + extension_time_s + 0.01,
        q=Q_RECORDED_CROUCH,
        cart_error=0.0,
        qd=zero,
    )
    assert event and "EXTENDED_HOLD" in event, event
    parked, reason = sequence.request_park(
        started_at + extension_time_s + 0.02, True
    )
    assert parked and sequence.stage == "PARK", reason

    controller.restore_full_targets()
    assert controller.extension_state == "idle"
    for leg in LEGS:
        assert np.isclose(controller.final_foot[leg][2], -stand_height), leg

    print("stand_dog5_fl_rr_extend_hw offline self-test PASS")
    print(
        f"  normal height={stand_height:.3f} m; FL/RR extended height="
        f"{stand_height + extension_m:.3f} m"
    )
    print(
        f"  cubic extension={1000.0 * extension_m:.1f} mm over "
        f"{extension_time_s:.1f} s; x/y targets unchanged"
    )
    print(f"  worst static mg/4 torque along validated path: {max_static:.2f} N*m")
    print(
        "  flow: CURRENT -> CROUCH -> ENTER -> STAND -> WAIT_STAND -> "
        "HOLD -> ENTER -> FL_RR_EXTEND -> EXTENDED_HOLD"
    )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tau-max",
        type=float,
        default=base.DEFAULT_TAU_MAX,
        help=(
            "Cartesian per-joint torque cap, N*m; a full stand normally "
            f"needs about {base.STAGED_TAU_MAX:.1f} "
            f"(default: {base.DEFAULT_TAU_MAX})"
        ),
    )
    parser.add_argument(
        "--extension-mm",
        type=float,
        default=DEFAULT_EXTENSION_MM,
        help=(
            "additional downward FL/RR endpoint travel in mm "
            f"(default: {DEFAULT_EXTENSION_MM:.0f})"
        ),
    )
    parser.add_argument(
        "--extension-time",
        type=float,
        default=DEFAULT_EXTENSION_TIME_S,
        help=(
            "duration of the cubic FL/RR extension ramp in seconds "
            f"(default: {DEFAULT_EXTENSION_TIME_S:.1f})"
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
        default=inplace.DEFAULT_SUPPORT_SCALE,
        help=(
            "mg/4 feedforward scale in [0,1] "
            f"(default: {inplace.DEFAULT_SUPPORT_SCALE})"
        ),
    )
    parser.add_argument(
        "--stand-height",
        type=float,
        default=recorded.DEFAULT_STAND_HEIGHT,
        help=(
            "normal trunk-frame hip-to-foot height, m; height plus "
            f"extension must be <= {MAX_EXTENDED_HEIGHT_M:.2f} "
            f"(default: {recorded.DEFAULT_STAND_HEIGHT})"
        ),
    )
    parser.add_argument(
        "--crouch-max-speed-dps",
        type=float,
        default=recorded.DEFAULT_CROUCH_MAX_MOTOR_DPS,
        help="native crouch motor-side speed cap, deg/s",
    )
    parser.add_argument(
        "--crouch-torque-trip",
        type=float,
        default=recorded.DEFAULT_CROUCH_TORQUE_TRIP_NM,
        help="native crouch measured joint-torque trip, N*m",
    )
    parser.add_argument(
        "--crouch-speed-trip",
        type=float,
        default=recorded.DEFAULT_CROUCH_SPEED_TRIP_RAD_S,
        help="native crouch encoder-derived speed trip, rad/s",
    )
    parser.add_argument(
        "--qd-estop",
        type=float,
        default=base.QD_ESTOP,
        help="confirmed sustained joint-speed trip, rad/s",
    )
    parser.add_argument(
        "--qd-estop-hard",
        type=float,
        default=base.QD_ESTOP_HARD,
        help="confirmed hard joint-speed trip, rad/s",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate reachability, targets, and sequence without CAN",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    extension_m = args.extension_mm / 1000.0

    if not 0.0 < args.tau_max <= base.STAGED_TAU_MAX:
        parser.error(
            f"--tau-max must be > 0 and <= {base.STAGED_TAU_MAX} N*m"
        )
    if not MIN_EXTENSION_MM <= args.extension_mm <= MAX_EXTENSION_MM:
        parser.error(
            f"--extension-mm must be between {MIN_EXTENSION_MM:.0f} and "
            f"{MAX_EXTENSION_MM:.0f}"
        )
    if not MIN_EXTENSION_TIME_S <= args.extension_time <= MAX_EXTENSION_TIME_S:
        parser.error(
            f"--extension-time must be between {MIN_EXTENSION_TIME_S:.1f} "
            f"and {MAX_EXTENSION_TIME_S:.1f} s"
        )
    if not 0.0 < args.cart_gain_scale <= 1.0:
        parser.error("--cart-gain-scale must be > 0 and <= 1")
    if not 0.0 <= args.support_scale <= 1.0:
        parser.error("--support-scale must be between 0 and 1")
    if not recorded.MIN_STAND_HEIGHT <= args.stand_height:
        parser.error(
            f"--stand-height must be >= {recorded.MIN_STAND_HEIGHT} m"
        )
    if args.stand_height + extension_m > MAX_EXTENDED_HEIGHT_M + 1.0e-12:
        parser.error(
            "--stand-height plus --extension-mm must be <= "
            f"{MAX_EXTENDED_HEIGHT_M:.2f} m"
        )
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
            "--qd-estop-hard must be above --qd-estop and <= "
            f"{base.QD_ESTOP_CEILING} rad/s"
        )

    if args.self_test:
        return offline_self_test(
            args.stand_height, extension_m, args.extension_time
        )

    def controller_factory(
        cart_gain_scale, support_scale, stand_height, travel_scale
    ):
        return DiagonalExtendController(
            cart_gain_scale,
            support_scale,
            stand_height,
            travel_scale,
            extension_m,
            args.extension_time,
        )

    print(
        f"[diagonal] FL and RR will extend {args.extension_mm:.1f} mm "
        f"downward over {args.extension_time:.1f} s after Enter in full HOLD."
    )
    print(
        f"[diagonal] normal height={args.stand_height:.3f} m; FL/RR final "
        f"height={args.stand_height + extension_m:.3f} m; FR/RL stay at "
        f"{args.stand_height:.3f} m."
    )
    print(
        "[diagonal] KEEP THE ROBOT MECHANICALLY SUPPORTED/TETHERED: this "
        "encoder-only controller cannot measure trunk attitude or contact."
    )

    try:
        return inplace.run_hardware(
            args.tau_max,
            args.cart_gain_scale,
            args.support_scale,
            args.stand_height,
            1.0,
            args.crouch_max_speed_dps,
            args.crouch_torque_trip,
            args.crouch_speed_trip,
            args.qd_estop,
            args.qd_estop_hard,
            controller_factory=controller_factory,
            sequence_factory=DiagonalExtendSequence,
            configuration_validator=validate_diagonal_configuration,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[diagonal] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
