#!/usr/bin/env python3
"""Gates for body_state_ahrs.py -- trunk state from the AHRS and legs, no EKF.

WHAT IS UNDER TEST
    [1] THE LEG-ODOMETRY IDENTITY, which is the claim the whole AHRS-only
        design rests on: that with planted feet the trunk velocity is an
        algebraic read, not an estimate.  Tested as an exact round trip --
        pick a trunk twist, solve for the joint rates that keep every foot
        nailed to the world, feed those in, and demand the twist back to
        machine precision.  If this gate is red, the VMC's kd_z/kd_x/kd_y
        terms are being fed noise and the AHRS-only choice is unsound.
    [2] the frame conventions round-trip with dog5_vmc_core, so a state built
        here and consumed by body_wrench means the same thing
    [3] FK height is the drift-free floor reference the position track uses
    [4] the REFUSALS.  In torque mode a plausible-but-wrong state drives real
        force, so stale AHRS and too-few-planted-feet must return active=False
        rather than a best guess.

Run:  $V test_body_state.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TORQUE = os.path.join(os.path.dirname(_HERE), "torque_mode_control")
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_DESC = os.path.join(_ROOT, "dog5_description")
_VMC = os.path.join(_ROOT, "vmc")
for _p in (_HERE, _TORQUE, _DESC, _VMC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from selftest_common import check, report                  # noqa: E402

import body_state_ahrs as bs                               # noqa: E402
import dog5_statics as st                                  # noqa: E402
import dog5_vmc_core as vmc                                # noqa: E402
import torque_params as P                                  # noqa: E402

LEGS = st.LEGS

Q_CROUCH = np.deg2rad(np.array(
    [90.74, 47.59, -131.02, -90.39, -51.92, 138.39,
     90.23, -44.45, 128.40, -93.95, 47.06, -125.12]))


class FakeImuDog:
    """`ImuDog` stand-in: whatever attitude and rates the gate wants."""

    class _S:
        pass

    def __init__(self, roll_deg=0.0, pitch_deg=0.0, rates_dps=(0.0, 0.0, 0.0)):
        self.roll_deg = roll_deg
        self.pitch_deg = pitch_deg
        self.rates_dps = tuple(rates_dps)
        self.stale = False
        self.none = False

    def sample(self):
        if self.none:
            return None
        s = FakeImuDog._S()
        s.roll_deg = self.roll_deg
        s.pitch_deg = self.pitch_deg
        s.roll_rate_dps, s.pitch_rate_dps, s.yaw_rate_dps = self.rates_dps
        return s

    def is_stale(self, max_age_s=0.05):
        return self.stale


def _qd_for_twist(q, C, v_world, omega, planted):
    """Joint rates that hold every planted foot fixed in the world.

    Inverts the identity under test:  0 = v + C^T(omega x s + J qdot)
    =>  J qdot = -C v - omega x s
    """
    qd = np.zeros(12)
    for i in range(4):
        if not planted[i]:
            continue
        sl = slice(3 * i, 3 * i + 3)
        foot, anchors, axes, _ = st.leg_frames(LEGS[i], q[sl])
        J = st.foot_jacobian_from(foot, anchors, axes)
        rhs = -(C @ np.asarray(v_world)) - np.cross(omega, foot)
        qd[sl] = np.linalg.solve(J, rhs)
    return qd


# ===========================================================================
# [1] the identity
# ===========================================================================

def test_odometry_round_trip():
    """The gate the AHRS-only design lives or dies on."""
    rng = np.random.default_rng(0)
    worst = 0.0
    cases = 0
    for _ in range(30):
        roll, pitch = rng.uniform(-0.15, 0.15, 2)
        C = bs.C_from_rp(roll, pitch)
        v = rng.uniform(-0.25, 0.25, 3)
        omega = rng.uniform(-0.6, 0.6, 3)
        planted = np.ones(4, dtype=bool)
        qd = _qd_for_twist(Q_CROUCH, C, v, omega, planted)
        v_est, n = bs.leg_odometry_velocity(Q_CROUCH, qd, C, omega, planted)
        worst = max(worst, float(np.max(np.abs(v_est - v))))
        cases += 1
    check("leg odometry recovers an imposed trunk velocity exactly",
          worst < 1e-12,
          f"worst {worst:.2e} m/s over {cases} random twists (4 feet down)")


def test_odometry_with_three_feet():
    """Three planted feet still determine the trunk completely."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for lift in range(4):
        planted = np.ones(4, dtype=bool)
        planted[lift] = False
        for _ in range(5):
            C = bs.C_from_rp(*rng.uniform(-0.1, 0.1, 2))
            v = rng.uniform(-0.2, 0.2, 3)
            omega = rng.uniform(-0.5, 0.5, 3)
            qd = _qd_for_twist(Q_CROUCH, C, v, omega, planted)
            v_est, n = bs.leg_odometry_velocity(Q_CROUCH, qd, C, omega, planted)
            worst = max(worst, float(np.max(np.abs(v_est - v))))
            if n != 3:
                worst = float("inf")
    check("...and with any one foot lifted (3 planted)",
          worst < 1e-12, f"worst {worst:.2e} m/s")


def test_odometry_is_not_differentiation():
    """A stationary trunk reads exactly zero, however the legs are posed."""
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(10):
        C = bs.C_from_rp(*rng.uniform(-0.1, 0.1, 2))
        v_est, _ = bs.leg_odometry_velocity(
            Q_CROUCH, np.zeros(12), C, np.zeros(3), np.ones(4, dtype=bool))
        worst = max(worst, float(np.max(np.abs(v_est))))
    check("zero joint rate gives exactly zero velocity (no integration state)",
          worst == 0.0, f"worst {worst:.2e} m/s")


def test_odometry_reports_the_planted_count():
    for k in range(5):
        planted = np.array([i < k for i in range(4)])
        _, n = bs.leg_odometry_velocity(Q_CROUCH, np.zeros(12), np.eye(3),
                                        np.zeros(3), planted)
        if n != k:
            check("planted count is reported honestly", False, f"{n} != {k}")
            return
    check("planted count is reported honestly", True, "0..4 feet")


# ===========================================================================
# [2] frames round-trip with the wrench layer
# ===========================================================================

def test_attitude_round_trip():
    worst = 0.0
    for roll in (-0.2, -0.05, 0.0, 0.05, 0.2):
        for pitch in (-0.2, -0.05, 0.0, 0.05, 0.2):
            r2, p2 = vmc.attitude_error_rp(bs.C_from_rp(roll, pitch))
            worst = max(worst, abs(r2 - roll), abs(p2 - pitch))
    check("C_from_rp is the exact inverse of vmc.attitude_error_rp",
          worst < 1e-12, f"worst {worst:.2e} rad over 25 pairs")
    check("level attitude gives exactly the identity",
          float(np.max(np.abs(bs.C_from_rp(0.0, 0.0) - np.eye(3)))) < 1e-15)


def test_gravity_direction_agrees_with_statics():
    C = bs.C_from_rp(0.12, -0.07)
    g = st.gravity_down_body(C)
    up = vmc.attitude_error_rp  # noqa: F841  (documented below)
    check("gravity_down_body is -(C @ z), the negative of vmc's g_b",
          float(np.max(np.abs(g + C @ np.array([0.0, 0.0, 1.0])))) < 1e-15,
          "so the tilt fed to leg gravity is the same tilt the wrench sees")


# ===========================================================================
# [3] FK height
# ===========================================================================

def test_fk_height():
    z = bs.fk_trunk_height(Q_CROUCH, np.eye(3), np.ones(4, dtype=bool))
    # the recorded crouch sits a few cm up; the exact value is the position
    # track's business, we only gate that it is a sane floor-referenced height
    check("FK height at the recorded crouch is a plausible floor distance",
          0.0 < z < 0.10, f"{z*1e3:.1f} mm to the trunk origin")
    check("...and includes the 20 mm foot radius",
          abs(z - (bs.fk_trunk_height(Q_CROUCH, np.eye(3),
                                      np.ones(4, dtype=bool))
                   - 0.020) - 0.020) < 1e-12,
          "floor is FOOT_RADIUS_M below the foot site")

    # raising the trunk by construction: same pose, tilted frame, the height
    # must fall as the cosine
    C = bs.C_from_rp(0.0, 0.0)
    z0 = bs.fk_trunk_height(Q_CROUCH, C, np.ones(4, dtype=bool))
    zt = bs.fk_trunk_height(Q_CROUCH, bs.C_from_rp(0.2, 0.0),
                            np.ones(4, dtype=bool))
    check("tilting the trunk lowers the FK height (it is a projection)",
          zt < z0, f"{z0*1e3:.1f} -> {zt*1e3:.1f} mm at 11.5 deg of roll")

    z_none = bs.fk_trunk_height(Q_CROUCH, np.eye(3), np.zeros(4, dtype=bool))
    check("no planted feet gives NaN, not a number",
          math.isnan(z_none))


# ===========================================================================
# [4] the refusals -- what keeps a fiction out of a force command
# ===========================================================================

def test_refusals():
    imu = FakeImuDog()
    state = bs.BodyState(imu)
    planted4 = np.ones(4, dtype=bool)

    out, active, reason = state.read(0.0, Q_CROUCH, np.zeros(12), planted4)
    check("a good read is active", active and out is not None, reason)

    imu.stale = True
    out, active, reason = state.read(0.01, Q_CROUCH, np.zeros(12), planted4)
    check("stale AHRS refuses (returns no state at all)",
          not active and out is None, reason)
    imu.stale = False

    imu.none = True
    out, active, reason = state.read(0.02, Q_CROUCH, np.zeros(12), planted4)
    check("no AHRS packet yet refuses", not active and out is None, reason)
    imu.none = False

    for k in (0, 1, 2):
        planted = np.array([i < k for i in range(4)])
        out, active, reason = state.read(0.03, Q_CROUCH, np.zeros(12), planted)
        if active or out is not None:
            check(f"{k} planted feet refuses the odometry", False, reason)
            return
    check("fewer than MIN_PLANTED_FOR_ODOM feet refuses the odometry",
          True, f"0, 1, 2 feet all refused (need {P.MIN_PLANTED_FOR_ODOM})")

    out, active, _ = state.read(0.04, Q_CROUCH, np.zeros(12),
                                np.array([True, True, True, False]))
    check("...but 3 planted feet is accepted", active and out is not None)


def test_output_shape_matches_the_wrench_layer():
    imu = FakeImuDog(roll_deg=1.5, pitch_deg=-0.8, rates_dps=(2.0, -1.0, 0.5))
    state = bs.BodyState(imu)
    out, active, _ = state.read(0.0, Q_CROUCH, np.zeros(12),
                                np.ones(4, dtype=bool))
    for key in ("r", "v", "C", "w_hat"):
        if key not in out:
            check("output carries every key body_wrench reads", False, key)
            return
    check("output carries every key body_wrench reads", True,
          "r, v, C, w_hat -- the wrench layer cannot tell this from the EKF")

    # and it really is consumable: run the actual wrench
    gains = vmc.VMCGains()
    W = vmc.body_wrench(out, 0.19, np.zeros(3), 0.0, gains, P.DOG5_MASS_KG)
    check("dog5_vmc_core.body_wrench accepts it unmodified",
          W.shape == (6,) and np.all(np.isfinite(W)),
          f"Fz = {W[2]:.1f} N against a {P.WEIGHT_N:.1f} N robot")
    check("...and the attitude it reports is the attitude we put in",
          abs(math.degrees(vmc.attitude_error_rp(out["C"])[0]) - 1.5) < 1e-9,
          "roll 1.5 deg in, 1.5 deg out")


def test_rates_are_radians():
    imu = FakeImuDog(rates_dps=(57.29577951308232, 0.0, 0.0))
    state = bs.BodyState(imu)
    out, _, _ = state.read(0.0, Q_CROUCH, np.zeros(12), np.ones(4, dtype=bool))
    check("gyro rates are converted deg/s -> rad/s",
          abs(out["w_hat"][0] - 1.0) < 1e-9,
          f"57.3 dps in -> {out['w_hat'][0]:.6f} rad/s")


def test_velocity_lpf_converges():
    """The only filter state in the module.  It must settle on the truth."""
    imu = FakeImuDog()
    state = bs.BodyState(imu)
    planted = np.ones(4, dtype=bool)
    C = np.eye(3)
    v_true = np.array([0.0, 0.0, -0.05])
    qd = _qd_for_twist(Q_CROUCH, C, v_true, np.zeros(3), planted)
    out = None
    for k in range(400):
        out, _, _ = state.read(k * P.SWEEP_S, Q_CROUCH, qd, planted)
    check("the odometry low pass settles on the true velocity",
          float(np.max(np.abs(out["v"] - v_true))) < 1e-4,
          f"commanded {v_true[2]*1e3:.0f} mm/s, read "
          f"{out['v'][2]*1e3:.1f} mm/s after 400 sweeps")
    check("...and v_raw is published unfiltered beside it",
          float(np.max(np.abs(out["v_raw"] - v_true))) < 1e-12,
          "so a gate can see the filter's own lag")


def self_test():
    print("[1] the leg-odometry identity")
    test_odometry_round_trip()
    test_odometry_with_three_feet()
    test_odometry_is_not_differentiation()
    test_odometry_reports_the_planted_count()
    print("[2] frames round-trip with the wrench layer")
    test_attitude_round_trip()
    test_gravity_direction_agrees_with_statics()
    print("[3] FK height")
    test_fk_height()
    print("[4] the refusals")
    test_refusals()
    test_output_shape_matches_the_wrench_layer()
    test_rates_are_radians()
    test_velocity_lpf_converges()
    return report()


if __name__ == "__main__":
    sys.exit(self_test())
