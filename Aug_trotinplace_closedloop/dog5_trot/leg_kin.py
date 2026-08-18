#!/usr/bin/env python3
"""Single-leg FK, Jacobian and IK.  Legs are integer indices here, 0..3.

WHY THIS FILE IS A WRAPPER AND NOT A REIMPLEMENTATION
    dog5_description/dog5_kinematics.py already carries DOG5's chain as a
    controlled copy of dog5.xml, and the standing tracks have been checked
    against MuJoCo through it.  Writing a second forward kinematics here would
    mean two copies of the geometry that can disagree silently -- which is the
    exact failure the 2026-08-17 frame bug was.  So FK and the Jacobian are
    delegated, and only the IK, which that module does not provide, is new.

    The one thing this file DOES own is the int-index convention: the rest of
    the trot package addresses legs as 0..3 to index (4,3) arrays, while
    dog5_kinematics takes "FL".."RR".  One translation, in one place.

TWO FRAMES, AND THE ONLY DIFFERENCE IS A CONSTANT
    p_body   foot in the trunk frame, origin at the trunk origin
    p_hip    the same point minus that leg's HIP_OFFSET

    p_hip is what the IK solves in, because a leg's reachable set is fixed in
    it; p_body is what the balance QP and the swing planner use, because they
    compare the four legs against each other.  The offset is constant, so the
    Jacobian is identical in both -- there is only one Jacobian in this file.

RUN
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V dog5_trot/leg_kin.py --self-test
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# Sibling imports go through the PACKAGE when there is one.  Only a direct
# `python dog5_trot/<this>.py --self-test` falls back to a path insert, and
# that insert is what would shadow the repo's own top-level config.py -- see
# the package docstring.  Keeping it off the library path is the point.
if __package__:
    from . import config as cfg
else:
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import config as cfg

# Taken THROUGH config rather than imported again: config is what puts
# dog5_description on sys.path, so a second `import dog5_kinematics` here
# would work only because config happened to run first.  One owner.
kin = cfg.kin


# The pose the IK falls back to when it is given no seed: this leg's own row
# of the measured stance.  PER LEG, not one shared triple -- see config.Q_STAND
# for why the four rows do not agree in sign, and why a seed of zeros is a
# singular, leg-forward pose rather than a neutral one.
Q_NOMINAL = cfg.Q_STAND


def _leg_name(leg: int) -> str:
    if not 0 <= int(leg) < cfg.N_LEGS:
        raise ValueError(f"leg must be 0..{cfg.N_LEGS - 1}, got {leg}")
    return cfg.LEGS[int(leg)]


def leg_state(leg: int, q):
    """(foot_in_trunk_frame, J) from ONE pass down the chain.

    foot_position and foot_jacobian each walk the whole chain, so asking for
    both costs it twice -- measured on this Pi, 42 us and 131 us, and a trot
    sweep wants both for all four legs.  dog5_kinematics computes them from a
    single shared `_state`, so this exposes that pairing rather than paying
    for the chain twice.  Everything in this package that needs both should
    call THIS, not the two separately.

    Reaching for the private `_state` is deliberate: the alternative is a
    second copy of the geometry in this file, and one copy is the whole point
    (see the header).
    """
    foot, anchors, axes = kin._state(_leg_name(leg), q)     # noqa: SLF001
    J = np.column_stack([np.cross(ax, foot - an)
                         for an, ax in zip(anchors, axes)])
    return foot, J


def foot_pos_body(leg: int, q) -> np.ndarray:
    """Foot position in the TRUNK frame (m)."""
    return kin.foot_position(_leg_name(leg), q)


def foot_pos_hip(leg: int, q) -> np.ndarray:
    """Foot position in that leg's HIP frame (m): trunk frame minus the hip."""
    return kin.foot_position(_leg_name(leg), q) - cfg.HIP_OFFSET[int(leg)]


def leg_jacobian(leg: int, q) -> np.ndarray:
    """d(foot)/dq, 3x3.  Identical in the hip and trunk frames."""
    return kin.foot_jacobian(_leg_name(leg), q)


def leg_ik(leg: int, p_hip, q_seed=None, iters: int = 40,
           tol: float = 1.0e-6) -> np.ndarray:
    """Joint angles putting this leg's foot at p_hip, in the HIP frame.

    Damped least squares Newton.  Damping rather than a plain pseudo-inverse
    because the reachable boundary IS reached in a trot -- a Raibert step at
    speed asks for a foot the leg cannot quite meet, and an undamped step
    there produces a huge joint command instead of the nearest pose.  With
    damping the solution degrades to "as close as the leg gets", which is the
    behaviour the swing controller can survive.

    `q_seed` is the warm start; a trot has the previous sweep's answer and
    should pass it, both for speed and because it picks the same elbow branch
    every step.  Without a seed the knee-down stance branch is chosen.

    `tol` IS A DISTANCE IN METRES, not a squared one.  It was written as a
    squared tolerance first, and 1e-10 then meant a 10 um foot error -- a
    number that looks like machine precision and is not.  Compared squared
    internally so the loop still costs no sqrt.

    THE DEFAULT IS SET BY THE SENSOR, NOT BY THE SOLVER.  1 um is already 20x
    finer than the ~20 um a 0.01 deg encoder count moves the foot, so tighter
    buys nothing real and costs the whole iteration budget: at 1 nm this call
    runs all 40 iterations and takes 681 us, which is four legs into 2.7 ms of
    a 4 ms sweep.  At 1 um a warm-started call converges in three.

    Returns the best pose found.  It does NOT raise on an unreachable target:
    the caller sees the residual through foot_pos_hip, and swing.py clamps the
    target before it gets here.
    """
    _leg_name(leg)                       # validates the index
    hip = cfg.HIP_OFFSET[int(leg)]
    target_body = np.asarray(p_hip, dtype=float) + hip
    q = (Q_NOMINAL[int(leg)].copy() if q_seed is None
         else np.asarray(q_seed, dtype=float).copy())

    # LEVENBERG-MARQUARDT, NOT FIXED DAMPING.  A constant lam2 makes every
    # step shorter than the Newton step by the same factor, so convergence is
    # LINEAR and stalls: measured, a fixed 1e-6 sat at 2.5e-7 m after 40
    # iterations and still 7e-8 after 120.  Scaling the damping with the
    # residual gives robustness where it is needed -- far from the target, or
    # near a singular pose -- and hands back the true Newton step once the
    # residual is small, which is quadratic.  Same code, 3-5 iterations.
    idx = int(leg)
    for _ in range(int(iters)):
        foot, J = leg_state(idx, q)          # one chain walk, not two
        err = target_body - foot
        e2 = float(err @ err)
        if e2 < tol * tol:
            break
        lam2 = min(1.0e-4, max(1.0e-14, 1.0e-2 * e2))
        # (J^T J + lam^2 I)^-1 J^T is the damped least squares step.  Solved
        # in the 3x3 joint space rather than the task space so the damping is
        # on the JOINT increment, which is what needs bounding.
        dq = np.linalg.solve(J.T @ J + lam2 * np.eye(3), J.T @ err)
        step = float(np.max(np.abs(dq)))
        if step > 0.3:                  # cap one Newton step at ~17 deg
            dq *= 0.3 / step
        q = q + dq
    return q


def q_ref_for_height(q_ref, z_ref, foot_xy, iters: int = 8):
    """The joint pose that puts every foot on the floor with the trunk at z_ref.

    A PORT OF august_week2/stand_torque_Mode.q_ref_for_height, semantics for
    semantics, because that one has been up and down on this robot and the
    failure it prevents is the one it documents: a q_ref frozen at the crouch
    makes the impedance pull ~150 mm of error against the force law for the
    whole rise, and the two layers fight instead of lifting.

    THE ONE THING THAT CHANGED IN THE PORT IS THE FRAME, and it changed by
    being removed.  Week 2 takes z_des as floor-to-TRUNK-BOTTOM and converts
    with fe.hip_from_imu; this package's z_ref is already the TRUNK ORIGIN
    (see config's height table), so there is no conversion here and no 38 mm
    to lose.  The foot-radius term stays: the site is the centre of a 20 mm
    sphere, so it sits at -(z_ref - FOOT_RADIUS).

    x and y stay pinned at `foot_xy`, so the feet rise straight up and the
    contact point never moves in the world.

    DAMPED least squares and WARM STARTED, both load-bearing: near full leg
    extension the Jacobian is ill-conditioned and a plain Newton step from the
    crouch to the stand height leaves the workspace entirely.  Over a real
    rise each call moves the target ~0.25 mm from the previous solution, so
    one or two steps converge; the damping is what makes the first call after
    a stage change safe anyway.
    """
    q_ref = np.asarray(q_ref, dtype=float).copy().reshape(cfg.N_JOINTS)
    z_site = -(float(z_ref) - cfg.FOOT_RADIUS)
    for i in range(cfg.N_LEGS):
        sl = cfg.JOINT_INDEX[i]
        qi = q_ref[sl].copy()
        tgt = np.array([foot_xy[i][0], foot_xy[i][1], z_site])
        for _ in range(int(iters)):
            foot, J = leg_state(i, qi)
            e = tgt - foot
            if float(e @ e) < 1e-18:
                break
            qi = qi + J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e)
        q_ref[sl] = qi
    return q_ref


def all_foot_pos_body(q) -> np.ndarray:
    """(4,3) foot positions in the trunk frame, from a 12-vector."""
    q = np.asarray(q, dtype=float).reshape(cfg.N_JOINTS)
    return np.stack([foot_pos_body(i, q[cfg.JOINT_INDEX[i]])
                     for i in range(cfg.N_LEGS)])


def all_foot_vel_body(q, qd) -> np.ndarray:
    """(4,3) foot velocities in the trunk frame, J_i qd_i per leg.

    TRUNK frame, so this is the velocity of the foot relative to the trunk,
    NOT the world.  The world velocity needs the trunk's own twist added; the
    balance loop wants that difference and the swing loop does not, so the
    conversion lives at the call sites rather than being smuggled in here.
    """
    q = np.asarray(q, dtype=float).reshape(cfg.N_JOINTS)
    qd = np.asarray(qd, dtype=float).reshape(cfg.N_JOINTS)
    out = np.empty((cfg.N_LEGS, 3))
    for i in range(cfg.N_LEGS):
        sl = cfg.JOINT_INDEX[i]
        out[i] = leg_state(i, q[sl])[1] @ qd[sl]
    return out


# ===========================================================================
# self-test
# ===========================================================================
_PASS = [0, 0]


def check(label, ok, detail=""):
    _PASS[1] += 1
    _PASS[0] += bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))


def self_test():
    rng = np.random.default_rng(7)

    # -- the wrapper really is the shared geometry, not a copy ------------
    q = cfg.Q_STAND[0]
    check("foot_pos_body delegates to dog5_kinematics",
          np.allclose(foot_pos_body(0, q), kin.foot_position("FL", q)))
    check("the hip frame differs from the trunk frame by exactly HIP_OFFSET",
          np.allclose(foot_pos_body(1, q) - foot_pos_hip(1, q),
                      cfg.HIP_OFFSET[1]))

    # -- the Jacobian is the derivative of the FK, checked numerically ----
    worst = 0.0
    for leg in range(4):
        for _ in range(20):
            qq = cfg.Q_STAND[leg] + rng.uniform(-0.35, 0.35, 3)
            J = leg_jacobian(leg, qq)
            for j in range(3):
                h = 1e-6
                dq = np.zeros(3); dq[j] = h
                num = (foot_pos_body(leg, qq + dq)
                       - foot_pos_body(leg, qq - dq)) / (2 * h)
                worst = max(worst, float(np.max(np.abs(num - J[:, j]))))
    check("the Jacobian matches a central difference of the FK",
          worst < 1e-7, f"worst column error {worst:.2e} m/rad")

    # -- IK round-trips FK over the whole working envelope ----------------
    worst, worst_q = 0.0, None
    for leg in range(4):
        for _ in range(150):
            qq = cfg.Q_STAND[leg] + rng.uniform(-0.3, 0.3, 3)
            p = foot_pos_hip(leg, qq)
            got = leg_ik(leg, p, tol=1.0e-9)     # tighter than the default:
            #  this check is about the SOLVER, not about what the loop needs
            e = float(np.linalg.norm(foot_pos_hip(leg, got) - p))
            if e > worst:
                worst, worst_q = e, qq
    check("IK round-trips FK from the DEFAULT seed across the envelope",
          worst < 1e-9,
          f"worst foot error {worst*1e9:.3f} nm at q={np.round(worst_q,3)}")

    # -- warm start: the seed must not change the answer, only the cost ---
    qq = cfg.Q_STAND[2] + np.array([0.05, -0.08, 0.06])
    p = foot_pos_hip(2, qq)
    a = leg_ik(2, p, tol=1.0e-9)
    b = leg_ik(2, p, q_seed=qq + 0.05, tol=1.0e-9)
    check("a warm start reaches the SAME branch as the cold one",
          float(np.max(np.abs(a - b))) < 1e-6,
          f"max joint difference {np.max(np.abs(a-b))*1e9:.2f} nrad")

    # -- an unreachable target degrades, it does not explode --------------
    far = np.array([0.0, 0.0, -10.0])
    q_far = leg_ik(0, far)
    check("an unreachable target returns a BOUNDED pose, not a huge one",
          np.all(np.abs(q_far) < 2 * np.pi),
          f"|q| max {np.max(np.abs(q_far)):.2f} rad, foot reaches "
          f"{np.linalg.norm(foot_pos_hip(0, q_far)):.3f} m of the 10 m asked")

    # -- the batch helpers agree with the per-leg ones --------------------
    q12 = (cfg.Q_STAND + rng.uniform(-0.2, 0.2, (4, 3))).reshape(-1)
    qd12 = rng.normal(0, 0.5, 12)
    check("all_foot_pos_body is the four per-leg calls",
          np.allclose(all_foot_pos_body(q12),
                      [foot_pos_body(i, q12[cfg.JOINT_INDEX[i]]) for i in range(4)]))
    check("leg_state returns exactly foot_pos_body and leg_jacobian",
          all(np.allclose(leg_state(i, q12[cfg.JOINT_INDEX[i]])[0],
                          foot_pos_body(i, q12[cfg.JOINT_INDEX[i]]))
              and np.allclose(leg_state(i, q12[cfg.JOINT_INDEX[i]])[1],
                              leg_jacobian(i, q12[cfg.JOINT_INDEX[i]]))
              for i in range(4)))
    check("all_foot_vel_body is J_i qd_i, leg by leg",
          np.allclose(all_foot_vel_body(q12, qd12),
                      [leg_jacobian(i, q12[cfg.JOINT_INDEX[i]])
                       @ qd12[cfg.JOINT_INDEX[i]] for i in range(4)]))

    # -- Q_STAND is load-bearing: every default seed and the swing planner's
    #    ground plane come off it, so check the POSE, not just the numbers ---
    feet = all_foot_pos_body(cfg.Q_STAND.reshape(-1))
    floor = feet[:, 2] - cfg.FOOT_RADIUS
    check("Q_STAND puts all four foot SITES on one plane",
          float(np.ptp(feet[:, 2])) < 1e-9,
          f"z spread {np.ptp(feet[:, 2])*1e9:.2f} nm")
    check("...and that plane is STAND_HEIGHT below the trunk origin, once "
          "FOOT_RADIUS is taken off",
          abs(float(np.mean(floor)) + cfg.STAND_HEIGHT) < 1e-8,
          f"floor at {np.mean(floor)*1e3:.6f} mm vs "
          f"-{cfg.STAND_HEIGHT*1e3:.0f}; foot SITES at "
          f"{np.mean(feet[:, 2])*1e3:.1f} mm, {cfg.FOOT_RADIUS*1e3:.0f} mm higher")
    check("...with the feet outboard of the hips, never crossed under",
          bool(np.all(np.sign(feet[:, 1]) == cfg.SIDE_SIGN)),
          f"foot y = {np.round(feet[:, 1], 4)} against SIDE_SIGN {cfg.SIDE_SIGN}")
    # The stance is a slight PARALLELOGRAM and that is dog5.xml, not the IK --
    # see config.STANCE_ASYMMETRY_X.  Assert the size of the asymmetry so a
    # future geometry change that makes it worse is caught rather than absorbed.
    dx = abs(abs(feet[0, 0]) - abs(feet[1, 0]))
    check("the front pair's known 2.2 mm x asymmetry is present and no bigger",
          abs(dx - cfg.STANCE_ASYMMETRY_X) < 1e-4,
          f"FL/FR x differ by {dx*1e3:.3f} mm, expected "
          f"{cfg.STANCE_ASYMMETRY_X*1e3:.1f} (2 x the 1.1 mm knee_to_foot y "
          f"offset that dog5.xml does not mirror)")
    check("...while front and rear mirror to within that same asymmetry",
          abs(abs(feet[0, 0]) - abs(feet[3, 0])) < 1e-5
          and abs(abs(feet[1, 0]) - abs(feet[2, 0])) < 1e-5,
          f"FL/RR {abs(feet[0,0]):.6f}/{abs(feet[3,0]):.6f}, "
          f"FR/RL {abs(feet[1,0]):.6f}/{abs(feet[2,0]):.6f} m")

    # -- q_ref_for_height: the function a rise is made of -----------------
    foot_xy = feet[:, :2]
    q_st = q_ref_for_height(cfg.Q_STAND.copy().reshape(-1),
                            cfg.STAND_HEIGHT, foot_xy)
    h = lambda qq: float(np.mean(all_foot_pos_body(qq)[:, 2] - cfg.FOOT_RADIUS))
    check("asking for the stand height at the stand pose changes nothing",
          float(np.max(np.abs(q_st - cfg.Q_STAND.reshape(-1)))) < 1e-6,
          f"max joint move {np.max(np.abs(q_st - cfg.Q_STAND.reshape(-1)))*1e6:.2f} urad")
    check("...and the height it commands measures back correctly",
          abs(h(q_st) + cfg.STAND_HEIGHT) < 1e-9,
          f"FK reads {-h(q_st)*1e3:.6f} mm against {cfg.STAND_HEIGHT*1e3:.0f} "
          f"asked -- the 38 mm frame and the 20 mm foot radius both cancel or "
          f"this line fails")

    # a real crouch, generated rather than hardcoded, then the whole rise
    z_crouch = 0.06
    q_cr = cfg.Q_STAND.copy().reshape(-1)
    for k in range(1, 201):
        q_cr = q_ref_for_height(
            q_cr, cfg.STAND_HEIGHT + (z_crouch - cfg.STAND_HEIGHT) * k / 200,
            foot_xy)
    # ONE call for the whole 130 mm is the case the docstring warns about, and
    # it really does fail -- assert that, so the warning is a measurement.
    q_jump = q_ref_for_height(cfg.Q_STAND.copy().reshape(-1), z_crouch, foot_xy)
    check("a single huge step leaves the workspace, as the docstring says",
          abs(h(q_jump) + z_crouch) > 1e-3,
          f"one call for {(cfg.STAND_HEIGHT-z_crouch)*1e3:.0f} mm lands "
          f"{abs(h(q_jump)+z_crouch)*1e3:.1f} mm out; ramped, it is exact")
    check("a crouch is a real IK solution, far from the stand",
          abs(h(q_cr) + z_crouch) < 1e-9
          and float(np.max(np.abs(q_cr - cfg.Q_STAND.reshape(-1)))) > 0.3,
          f"{np.rad2deg(np.max(np.abs(q_cr - cfg.Q_STAND.reshape(-1)))):.1f} deg "
          f"of joint travel over {(cfg.STAND_HEIGHT - z_crouch)*1e3:.0f} mm")

    n = int(8.0 * 250.0 / 3)                 # 8 s rise at the 83 Hz model rate
    q_r = q_cr.copy()
    worst_lag = 0.0
    for k in range(1, n + 1):
        z = z_crouch + (cfg.STAND_HEIGHT - z_crouch) * k / n
        q_r = q_ref_for_height(q_r, z, foot_xy)
        worst_lag = max(worst_lag, abs(h(q_r) + z))
    check("q_ref tracks a whole 8 s rise without ever lagging",
          worst_lag < 1e-9, f"worst lag over {n} steps: {worst_lag*1e9:.3f} nm")
    check("...and lands on the stand pose it started away from",
          float(np.max(np.abs(q_r - cfg.Q_STAND.reshape(-1)))) < 1e-4,
          f"max joint error at the top "
          f"{np.rad2deg(np.max(np.abs(q_r - cfg.Q_STAND.reshape(-1)))):.5f} deg")
    check("...with the feet never moving in x or y",
          float(np.max(np.abs(all_foot_pos_body(q_r)[:, :2] - foot_xy))) < 1e-9,
          "the contact point is fixed; only the trunk rises")

    print(f"self-test {'PASS' if _PASS[0] == _PASS[1] else 'FAIL'} "
          f"({_PASS[0]}/{_PASS[1]})")
    return 0 if _PASS[0] == _PASS[1] else 1


if __name__ == "__main__":
    sys.exit(self_test() if "--self-test" in sys.argv else self_test())
