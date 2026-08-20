#!/usr/bin/env python3
"""Virtual-model wrench, and the QP that splits it across the planted feet.

    (attitude, height, velocity errors)  ->  W = [F_des; M_des]   world axes
    W + foot geometry + contact set      ->  f (4,3)              world axes

THE MODEL IS A SINGLE RIGID BODY, QUASI-STATIC IN THE SAME SENSE AS THE STAND
    F = spring/damper on {z} + damper on {x,y} + m g
    M = spring/damper on {roll, pitch, yaw}

    No I*alpha and no omega x (I omega).  For a trot IN PLACE the trunk's
    angular acceleration is small and both terms are small with it.  What that
    buys is that INERTIA_BODY is used for exactly one thing -- sizing the
    attitude gains -- rather than sitting inside the control law where a wrong
    tensor becomes a wrong torque.  A trot that TRAVELS wants the inertia
    terms back; this file is honest that it does not have them.

WHY THERE IS A QP AND NOT A PSEUDO-INVERSE
    A is 6x12, so the plain minimum-norm split is one line.  It is also wrong
    the moment a foot is near lifting: it will happily ask a foot for downward
    force it cannot produce, or for tangential force past what friction will
    carry, and the leg then tracks a command the ground refuses.  Every
    constraint in this file exists because the ground can only push:

        fz >= FZ_MIN          a planted foot never unloads to nothing
        fz <= FZ_MAX          and never carries more than the robot plus a push
        |fx|, |fy| <= mu fz   the friction pyramid, linear so the QP stays a QP
        f = 0 on a swing leg  imposed as fz in [0,0], which collapses the
                              tangential rows to |fx|,|fy| <= 0 for free

THE SOLVER IS ADMM, AND IT IS HERE BECAUSE THE PI HAS NOTHING ELSE
    This venv has numpy and no scipy, osqp, quadprog, cvxpy or qpsolvers.  So
    the QP is solved in-file by ADMM in OSQP's form -- min 1/2 x'Px + q'x
    subject to l <= Cx <= u -- which is one 12x12 Cholesky and then a few
    dozen matrix-vector products.  12 variables and 20 constraints is tiny; the
    cost is dominated by the factorisation, which is ~20 us here.

    ONE FIXED ITERATION BUDGET, checked but not early-exited by default: a
    control loop wants the same cost every sweep more than it wants the last
    digit.  `solve` returns the force and records the residuals on the object
    so a caller can log whether the budget was enough.

RUN
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V dog5_trot_quasi_static_model/balance_qp.py --self-test
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# Sibling imports go through the PACKAGE when there is one.  Only a direct
# `python dog5_trot_quasi_static_model/<this>.py --self-test` falls back to a path insert, and
# that insert is what would shadow the repo's own top-level config.py -- see
# the package docstring.  Keeping it off the library path is the point.
if __package__:
    from . import config as cfg
else:
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import config as cfg



# ===========================================================================
# the control law
# ===========================================================================
def _skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _rot_from_rpy(rpy):
    """R (body->world) from intrinsic Z-Y-X yaw/pitch/roll."""
    r, p, y = (float(a) for a in rpy)
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                              np.sin(p), np.cos(y), np.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _so3_log(R):
    """Rotation-vector of R, in the frame R is expressed in.  Handles any angle."""
    c = (np.trace(R) - 1.0) * 0.5
    c = min(1.0, max(-1.0, c))
    ang = np.arccos(c)
    if ang < 1e-9:
        # Small angle: the antisymmetric part IS the rotation vector.
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) * 0.5
    s = np.sin(ang)
    if abs(s) < 1e-9:                       # ang ~ pi; not reachable in a trot
        return np.zeros(3)                  # but must not divide by zero
    return (ang / (2.0 * s)) * np.array([R[2, 1] - R[1, 2],
                                         R[0, 2] - R[2, 0],
                                         R[1, 0] - R[0, 1]])


def desired_wrench(rpy, R_wb, omega, p, v, rpy_ref, z_ref,
                   v_cmd=(0.0, 0.0, 0.0), gains=cfg):
    """The wrench the feet must produce between them, in WORLD axes.

    Returns (F_des (3,), M_des (3,)).

    THE ATTITUDE ERROR IS A ROTATION, NOT AN RPY SUBTRACTION.  rpy_ref - rpy
    is only the rotation error for small angles and it breaks at the yaw wrap:
    a robot at +179 deg with a -179 deg reference reads a 358 deg error and
    the QP is handed a moment pointing the long way round.  The log map of
    R_ref R_wb^T is the exact world-frame error and costs one arccos, so the
    rpy arguments are used ONLY to build R_ref.

    `omega` IS THE BODY-FRAME RATE, which is what a gyro measures.  It is
    rotated to world here.  Feeding a body rate straight into a world-axis
    damper is correct only while the robot is level, which is exactly the
    condition the damper exists to restore.

    m*g IS THE WHOLE POINT of the F term: a robot sitting exactly at z_ref
    would otherwise be told to push with zero force, and would fall.  The
    springs trim; gravity is what holds it up.
    """
    rpy = np.asarray(rpy, dtype=float)
    R_wb = np.asarray(R_wb, dtype=float).reshape(3, 3)
    omega_w = R_wb @ np.asarray(omega, dtype=float)
    p = np.asarray(p, dtype=float)
    v = np.asarray(v, dtype=float)
    v_cmd = np.asarray(v_cmd, dtype=float)

    kp_pos = np.asarray(gains.KP_POS, dtype=float)
    kd_pos = np.asarray(gains.KD_POS, dtype=float)
    kp_ori = np.asarray(gains.KP_ORI, dtype=float)
    kd_ori = np.asarray(gains.KD_ORI, dtype=float)

    # Position: only z has a reference to be stiff to.  KP_POS[:2] is zero, so
    # p_ref[:2] never matters -- it is written as p[:2] to make that explicit
    # rather than leaving a live gain on an unobservable coordinate.
    p_ref = np.array([p[0], p[1], float(z_ref)])
    F_des = kp_pos * (p_ref - p) + kd_pos * (v_cmd - v)
    F_des = F_des + np.array([0.0, 0.0, gains.MASS * gains.GRAVITY])

    R_ref = _rot_from_rpy(rpy_ref)
    e_ori = _so3_log(R_ref @ R_wb.T)                # world-frame rotation error
    M_des = kp_ori * e_ori + kd_ori * (0.0 - omega_w)
    return F_des, M_des


# ===========================================================================
# the force distribution QP
# ===========================================================================
class ForceDistributor:
    """min ||A f - w||^2_W + wf ||f||^2 + ws ||f - f_prev||^2  s.t. the cone.

    `f_prev` is held HERE rather than passed in, because the smoothing term is
    a property of the distributor's own history: a caller that forgot to
    thread it through would silently get an unsmoothed solution, and the whole
    reason the term exists is to stop a contact switch from stepping the
    force.  reset() clears it.
    """

    N_CONE_ROWS = 5 * cfg.N_LEGS            # 20: one fz row + four tangential

    def __init__(self, mass=cfg.MASS, inertia_body=cfg.INERTIA_BODY,
                 mu=cfg.MU, fz_min=cfg.FZ_MIN, fz_max=cfg.FZ_MAX,
                 w_task=cfg.W_TASK, w_force=cfg.W_FORCE,
                 w_smooth=cfg.W_SMOOTH, iters=cfg.QP_ITERS,
                 rho=cfg.QP_RHO, sigma=cfg.QP_SIGMA):
        self.mass = float(mass)
        self.inertia_body = np.asarray(inertia_body, dtype=float).reshape(3, 3)
        self.mu = float(mu)
        self.fz_min = float(fz_min)
        self.fz_max = float(fz_max)
        self.w_task = np.asarray(w_task, dtype=float).reshape(6)
        self.w_force = float(w_force)
        self.w_smooth = float(w_smooth)
        self.iters = int(iters)
        self.rho = float(rho)
        self.sigma = float(sigma)
        # THE PYRAMID IS INSCRIBED IN THE CIRCLE, NOT CIRCUMSCRIBED ABOUT IT.
        # A linear cone written as |fx| <= mu fz and |fy| <= mu fz permits a
        # RESULTANT tangential force of sqrt(2) mu fz on the diagonals -- 41%
        # past the friction the surface actually has, i.e. the QP would call a
        # slipping solution feasible.  Dividing by sqrt(2) puts the pyramid
        # inside the circle, so |f_tan| <= mu fz for every direction.  The
        # cost is up to 29% of the tangential authority on the axes, which is
        # the correct trade: a conservative bound that holds beats an exact
        # bound in two directions and a wrong one in the other two.
        self.mu_axis = self.mu / np.sqrt(2.0)
        if self.mu <= 0.0:
            raise ValueError("mu must be positive; a frictionless foot cannot "
                             "produce the tangential force a trot needs")
        if self.fz_max <= self.fz_min:
            raise ValueError(f"fz_max {self.fz_max} must exceed fz_min "
                             f"{self.fz_min}")
        self.reset()

    def reset(self):
        self.f_prev = np.zeros(3 * cfg.N_LEGS)
        # ADMM's DUAL is warm-started too, and that is where most of the
        # saving is: y encodes which constraints are active, and in a trot
        # that set is unchanged for a whole stance phase.  It is dropped the
        # moment the contact mask changes, because a dual carried across a
        # different constraint set is worse than no warm start at all.
        self._y_prev = np.zeros(self.N_CONE_ROWS)
        self._mask_prev = None
        self.primal_residual = 0.0
        self.dual_residual = 0.0
        self.iters_used = 0

    # -- the two pieces the spec names --------------------------------------
    def _build_A(self, r_feet) -> np.ndarray:
        """(6,12): [I I I I ; [r_0]x [r_1]x [r_2]x [r_3]x].

        r_feet are the foot positions RELATIVE TO THE CoM, in world axes, so
        the bottom block really is the moment each foot force makes about the
        CoM.  Passing trunk-origin-relative vectors here is the classic way to
        get a pitch bias that no gain can remove; the caller owns that
        subtraction and controller.py does it in one place.
        """
        r_feet = np.asarray(r_feet, dtype=float).reshape(cfg.N_LEGS, 3)
        A = np.zeros((6, 3 * cfg.N_LEGS))
        for i in range(cfg.N_LEGS):
            A[0:3, 3 * i:3 * i + 3] = np.eye(3)
            A[3:6, 3 * i:3 * i + 3] = _skew(r_feet[i])
        return A

    def _build_cone(self, contact):
        """(see below).  `contact` may be a bool mask or a WEIGHT in [0,1]."""
        """(C (20,12), l (20,), u (20,)) for  l <= C f <= u.

        Five rows per leg:
            fz                        in [fz_min, fz_max]   planted
                                      in [0, 0]             swinging
            fx - k fz, -fx - k fz     <= 0     k = mu/sqrt(2)
            fy - k fz, -fy - k fz     <= 0

        A SWING LEG NEEDS NO SPECIAL CASE.  Pinning fz to [0,0] makes the four
        tangential rows read |fx| <= 0 and |fy| <= 0, so the whole 3-vector is
        driven to zero by the constraints already present.  One code path for
        both states is one fewer place for the contact mask to be misread.
        """
        # A BOOL MASK IS JUST THE WEIGHTS 0 AND 1.  Taking a weight lets the
        # caller ramp a foot's allowed force in and out of contact instead of
        # switching it, which is what removes the step in the handover; see
        # config.CONTACT_RAMP.  Bools still work, and mean exactly what they
        # used to.
        w = np.clip(np.asarray(contact, dtype=float).reshape(cfg.N_LEGS),
                    0.0, 1.0)
        C = np.zeros((self.N_CONE_ROWS, 3 * cfg.N_LEGS))
        lo = np.full(self.N_CONE_ROWS, -np.inf)
        up = np.zeros(self.N_CONE_ROWS)
        for i in range(cfg.N_LEGS):
            r0, c0 = 5 * i, 3 * i
            C[r0 + 0, c0 + 2] = 1.0                     # fz
            lo[r0 + 0] = self.fz_min * w[i]
            up[r0 + 0] = self.fz_max * w[i]
            C[r0 + 1, c0 + 0], C[r0 + 1, c0 + 2] = +1.0, -self.mu_axis
            C[r0 + 2, c0 + 0], C[r0 + 2, c0 + 2] = -1.0, -self.mu_axis
            C[r0 + 3, c0 + 1], C[r0 + 3, c0 + 2] = +1.0, -self.mu_axis
            C[r0 + 4, c0 + 1], C[r0 + 4, c0 + 2] = -1.0, -self.mu_axis
        return C, lo, up

    # -- the solver ---------------------------------------------------------
    def solve(self, F_des, M_des, r_feet, contact) -> np.ndarray:
        """(4,3) ground reaction forces in WORLD axes, one row per leg."""
        w = np.concatenate([np.asarray(F_des, dtype=float).reshape(3),
                            np.asarray(M_des, dtype=float).reshape(3)])
        A = self._build_A(r_feet)
        C, lo, up = self._build_cone(contact)
        W = np.diag(self.w_task)

        # 1/2 x'Px + q'x, from expanding the three squared terms
        AtW = A.T @ W
        P = 2.0 * (AtW @ A + (self.w_force + self.w_smooth) * np.eye(12))
        q = -2.0 * (AtW @ w + self.w_smooth * self.f_prev)

        # The dual warm start keys on which legs are IN contact at all, not on
        # the exact weight: the active SET is what y encodes, and a weight
        # sliding from 0.4 to 0.5 does not change it.
        mask = tuple(np.asarray(contact, dtype=float).reshape(cfg.N_LEGS) > 0.0)
        y0 = self._y_prev if mask == self._mask_prev else np.zeros(self.N_CONE_ROWS)
        self._mask_prev = mask
        x = self._admm(P, q, C, lo, up, x0=self.f_prev, y0=y0)
        self.f_prev = x
        return x.reshape(cfg.N_LEGS, 3)

    def _admm(self, P, q, C, lo, up, x0, y0=None):
        """OSQP's iteration, dense and without the adaptive rho.

        (P + sigma I + rho C'C) x = sigma x - q + C'(rho z - y)
        z = clip(Cx + y/rho, lo, up)
        y = y + rho (Cx - z)

        sigma keeps the left-hand side positive definite even where P is only
        semidefinite -- and P IS semidefinite here whenever fewer than six
        independent force directions are available, which is every single step
        of a trot, because two planted feet give a rank-6 A only through their
        tangential rows.  Without sigma the Cholesky fails exactly when the
        robot is on two legs.
        """
        rho, sig = self.rho, self.sigma
        Ct = C.T                                  # hoisted: used every sweep
        K = P + sig * np.eye(12) + rho * (Ct @ C)
        # EXPLICIT INVERSE, NOT TWO TRIANGULAR SOLVES.  For a 12x12 that is
        # normally the wrong instinct, but here the matrix is factored ONCE and
        # applied `iters` times, and at this size every numpy call is ~8 us of
        # dispatch overhead regardless of the arithmetic.  Two solve() calls per
        # iteration cost more than the whole inverse.  Cholesky still runs, as
        # the positive-definiteness check that inv() would not give.
        np.linalg.cholesky(K)
        Kinv = np.linalg.inv(K)

        x = np.asarray(x0, dtype=float).copy()
        z = np.clip(C @ x, lo, up)
        y = (np.zeros(C.shape[0]) if y0 is None
             else np.asarray(y0, dtype=float).copy())
        inv_rho = 1.0 / rho
        check_every = 5           # the two reductions cost about as much as a
                                  # step, so convergence is polled, not watched
        for k in range(self.iters):
            x = Kinv @ (sig * x - q + Ct @ (rho * z - y))
            Cx = C @ x
            z_new = np.clip(Cx + y * inv_rho, lo, up)
            y += rho * (Cx - z_new)
            dz = z_new - z
            z = z_new
            self.iters_used = k + 1
            if (k + 1) % check_every == 0 or k + 1 == self.iters:
                self.primal_residual = float(np.max(np.abs(Cx - z)))
                self.dual_residual = float(rho * np.max(np.abs(dz)))
                if (self.primal_residual < 1e-6) and (self.dual_residual < 1e-6):
                    break
        # The iterate can sit a hair outside the box; the CALLER turns this
        # into torque, so hand back something the ground could actually
        # produce rather than something that merely nearly satisfies a KKT
        # system.  Projection can only reduce |f|, never invent force.
        self._y_prev = y
        return self._project(x, lo, up)

    def _project(self, x, lo, up):
        """Clip fz into its box, then scale each foot's tangential part into
        the cone.  Exact for this constraint set: the fz row and the four
        tangential rows of one leg only couple through fz, so fixing fz first
        makes the rest a 2-D radius clamp."""
        f = x.reshape(cfg.N_LEGS, 3).copy()
        for i in range(cfg.N_LEGS):
            fz = float(np.clip(f[i, 2], lo[5 * i], up[5 * i]))
            f[i, 2] = fz
            lim = self.mu_axis * fz
            for j in (0, 1):                    # the pyramid is per-axis
                f[i, j] = float(np.clip(f[i, j], -lim, lim))
        return f.reshape(-1)


# ===========================================================================
# self-test
# ===========================================================================
_PASS = [0, 0]


def check(label, ok, detail=""):
    _PASS[1] += 1
    _PASS[0] += bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))


def _stance_feet():
    """Four feet under a nominal stance, relative to the CoM, world axes."""
    return np.array([[+0.2205, +0.112, -cfg.STAND_HEIGHT],
                     [+0.2205, -0.112, -cfg.STAND_HEIGHT],
                     [-0.2205, +0.112, -cfg.STAND_HEIGHT],
                     [-0.2205, -0.112, -cfg.STAND_HEIGHT]])


def self_test():
    I3 = np.eye(3)

    # -- desired_wrench ----------------------------------------------------
    F, M = desired_wrench(np.zeros(3), I3, np.zeros(3),
                          np.array([0, 0, cfg.STAND_HEIGHT]), np.zeros(3),
                          np.zeros(3), cfg.STAND_HEIGHT)
    check("a level robot at its target height is asked for exactly its weight",
          np.allclose(F, [0, 0, cfg.WEIGHT]) and np.allclose(M, 0),
          f"F = {np.round(F, 3)} N")

    F, _ = desired_wrench(np.zeros(3), I3, np.zeros(3),
                          np.array([0, 0, cfg.STAND_HEIGHT - 0.01]),
                          np.zeros(3), np.zeros(3), cfg.STAND_HEIGHT)
    check("10 mm low asks for kp_z*0.01 MORE than the weight",
          abs(F[2] - (cfg.WEIGHT + cfg.KP_POS[2] * 0.01)) < 1e-9,
          f"{F[2]:.2f} N vs {cfg.WEIGHT:.2f} + {cfg.KP_POS[2]*0.01:.2f}")

    check("x and y have NO stiffness -- only a damper acts there",
          np.allclose(desired_wrench(np.zeros(3), I3, np.zeros(3),
                                     np.array([5.0, -3.0, cfg.STAND_HEIGHT]),
                                     np.zeros(3), np.zeros(3),
                                     cfg.STAND_HEIGHT)[0][:2], 0.0),
          "a 5 m x offset produces no force, because nothing observes x")

    # THE WRAP IS TESTED ON THE ERROR, NOT ON THE MOMENT.  It used to be
    # checked through M_des[2], which silently became vacuous the day
    # KP_ORI[2] went to 0 for the unobservable-yaw reason -- both sides read
    # 0.00 Nm and the check passed by saying nothing.  _so3_log is what has
    # the property, so assert it there and the gain cannot hide it.
    R_a = _rot_from_rpy([0, 0, np.radians(179.0)])
    R_b = _rot_from_rpy([0, 0, np.radians(-179.0)])
    e = _so3_log(R_b @ R_a.T)
    naive = np.radians(-179.0) - np.radians(179.0)
    check("a yaw error across the wrap takes the SHORT way round",
          abs(e[2]) < abs(naive) / 10.0 and abs(abs(e[2]) - np.radians(2.0)) < 1e-9,
          f"log map {np.degrees(e[2]):+.2f} deg vs rpy subtraction "
          f"{np.degrees(naive):+.1f} deg -- the true error is 2 deg")
    check("...and the wrap does not contaminate roll or pitch",
          float(np.max(np.abs(e[:2]))) < 1e-12,
          f"roll/pitch error {np.round(np.degrees(e[:2]), 9)} deg from a pure "
          f"yaw disagreement")

    _, M2 = desired_wrench(np.array([0.05, 0, 0]), _rot_from_rpy([0.05, 0, 0]),
                           np.zeros(3), np.zeros(3), np.zeros(3),
                           np.zeros(3), cfg.STAND_HEIGHT)
    check("...and for small angles it still reduces to -kp*roll",
          abs(M2[0] + cfg.KP_ORI[0] * 0.05) < 1e-3,
          f"{M2[0]:+.4f} vs {-cfg.KP_ORI[0]*0.05:+.4f} Nm")

    # -- the QP's shape ----------------------------------------------------
    fd = ForceDistributor()
    r = _stance_feet()
    A = fd._build_A(r)
    check("A is 6x12 and its top block is four identities",
          A.shape == (6, 12) and np.allclose(A[0:3], np.tile(np.eye(3), 4)))
    ftest = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
    check("A's bottom block really is sum r_i x f_i",
          np.allclose((A @ ftest.reshape(-1))[3:],
                      sum(np.cross(r[i], ftest[i]) for i in range(4))))
    C, lo, up = fd._build_cone(np.array([True, True, False, False]))
    _, lo_h, up_h = fd._build_cone(np.array([0.5, 1.0, 0.0, 0.0]))
    check("a HALF-weight foot gets half the fz box, which is what ramps it in",
          abs(up_h[0] - 0.5 * cfg.FZ_MAX) < 1e-12
          and abs(lo_h[0] - 0.5 * cfg.FZ_MIN) < 1e-12
          and up_h[5] == cfg.FZ_MAX,
          f"leg0 fz in [{lo_h[0]:.2f}, {up_h[0]:.2f}] at weight 0.5")
    check("the cone is 20 rows, five per leg", C.shape == (20, 12))
    check("a swing leg is pinned to fz in [0,0]",
          lo[10] == 0.0 and up[10] == 0.0 and lo[15] == 0.0 and up[15] == 0.0)
    check("a planted leg gets [FZ_MIN, FZ_MAX]",
          lo[0] == cfg.FZ_MIN and up[0] == cfg.FZ_MAX)

    # -- four feet down, level: the split must be feasible and near-exact ---
    fd.reset()
    F, M = desired_wrench(np.zeros(3), I3, np.zeros(3),
                          np.array([0, 0, cfg.STAND_HEIGHT]), np.zeros(3),
                          np.zeros(3), cfg.STAND_HEIGHT)
    # W_SMOOTH IS BIAS-FREE AT STEADY STATE and biased on the FIRST call:
    # ||f - f_prev||^2 has zero gradient once f == f_prev, but f_prev is zero
    # straight out of reset(), so the very first solve is pulled toward zero
    # by the full smoothing weight.  Settling here is not hiding the error, it
    # is measuring the thing a 250 Hz loop actually sees after its first sweep.
    first = fd.solve(F, M, r, np.ones(4, bool))
    for _ in range(20):
        f = fd.solve(F, M, r, np.ones(4, bool))
    res = A @ f.reshape(-1) - np.concatenate([F, M])
    check("standing on four feet reproduces the wrench",
          float(np.max(np.abs(res))) < 0.05,
          f"worst component {np.max(np.abs(res)):.4f} (F in N, M in Nm); "
          f"{fd.iters_used} ADMM iterations")
    check("...and the total vertical force IS the weight",
          abs(f[:, 2].sum() - cfg.WEIGHT) < 0.05,
          f"{f[:, 2].sum():.3f} N vs {cfg.WEIGHT:.3f}")
    check("...which the FIRST solve after reset is not, by W_SMOOTH's design",
          abs(first[:, 2].sum() - cfg.WEIGHT)
          > abs(f[:, 2].sum() - cfg.WEIGHT),
          f"first solve {first[:, 2].sum():.3f} N, settled {f[:, 2].sum():.3f} N "
          f"-- the residual is the f_prev=0 transient, not a modelling error")

    # -- a trot's two-foot stance ------------------------------------------
    for mask, name in [(np.array([1, 0, 0, 1], bool), "FL+RR"),
                       (np.array([0, 1, 1, 0], bool), "FR+RL")]:
        fd.reset()
        f = fd.solve(F, M, r, mask)
        check(f"the {name} diagonal carries the whole weight",
              abs(f[:, 2].sum() - cfg.WEIGHT) < 0.5,
              f"{f[:, 2].sum():.2f} N over two feet; swing feet "
              f"|f| = {np.abs(f[~mask]).max():.2e}")
        check(f"...and the {name} swing feet are EXACTLY zero",
              np.all(f[~mask] == 0.0))

    # -- every constraint, on a hard target the QP cannot meet -------------
    fd.reset()
    f = fd.solve(np.array([200.0, 0.0, cfg.WEIGHT]), np.zeros(3), r,
                 np.array([1, 0, 0, 1], bool))
    fz = f[:, 2]
    tan = np.linalg.norm(f[:, :2], axis=1)
    planted = np.array([1, 0, 0, 1], bool)
    check("an impossible sideways pull still returns a FEASIBLE force",
          np.all(fz[planted] >= cfg.FZ_MIN - 1e-9)
          and np.all(fz[planted] <= cfg.FZ_MAX + 1e-9)
          and np.all(np.abs(f[:, 0]) <= fd.mu_axis * fz + 1e-9)
          and np.all(np.abs(f[:, 1]) <= fd.mu_axis * fz + 1e-9),
          f"asked 200 N sideways, delivered {f[:,0].sum():.1f} N against "
          f"mu*sum(fz) = {cfg.MU*fz.sum():.1f} -- the QP loaded the feet to "
          f"{fz.sum():.0f} N to buy tangential authority, which is legal")
    check("...and the RESULTANT stays inside the friction CIRCLE, not just "
          "the pyramid's axes",
          float(np.max(tan - cfg.MU * fz)) <= 1e-9,
          f"worst |f_tan| - mu*fz = {np.max(tan - cfg.MU*fz):+.3e} N; a "
          f"circumscribed pyramid would allow +{(np.sqrt(2)-1)*cfg.MU*fz.max():.1f}")

    # -- the smoothing term actually smooths -------------------------------
    fd.reset()
    fd.solve(F, M, r, np.array([1, 0, 0, 1], bool))
    a = fd.f_prev.copy()
    fd.solve(F, M, r, np.array([0, 1, 1, 0], bool))
    b = fd.f_prev.copy()
    hard = ForceDistributor(w_smooth=0.0)
    hard.solve(F, M, r, np.array([1, 0, 0, 1], bool))
    hard.solve(F, M, r, np.array([0, 1, 1, 0], bool))
    check("w_smooth reduces the jump across a contact switch",
          np.linalg.norm(b - a) <= np.linalg.norm(hard.f_prev - a) + 1e-9,
          f"|df| {np.linalg.norm(b-a):.2f} N smoothed vs "
          f"{np.linalg.norm(hard.f_prev-a):.2f} N with w_smooth=0")

    # -- the solver is not accidentally solving something else -------------
    # With the cone slack (a light, centred load) the answer must equal the
    # closed-form unconstrained minimiser.
    fd.reset()
    Fs, Ms = np.array([0.0, 0.0, 20.0]), np.zeros(3)
    f = fd.solve(Fs, Ms, r, np.ones(4, bool)).reshape(-1)
    W = np.diag(cfg.W_TASK)
    P = 2 * (A.T @ W @ A + (cfg.W_FORCE + cfg.W_SMOOTH) * np.eye(12))
    qv = -2 * (A.T @ W @ np.concatenate([Fs, Ms]))   # f_prev was zero
    exact = np.linalg.solve(P, -qv)
    inside = np.all(exact.reshape(4, 3)[:, 2] > cfg.FZ_MIN)
    check("with the cone slack it matches the closed-form minimiser",
          inside and float(np.max(np.abs(f - exact))) < 1e-4,
          f"worst |f - f_exact| = {np.max(np.abs(f - exact)):.2e} N")

    # -- rank deficiency: two feet is where a naive solver dies ------------
    fd.reset()
    ok = True
    try:
        for _ in range(50):
            fd.solve(F, M, r, np.array([1, 0, 0, 1], bool))
    except np.linalg.LinAlgError:
        ok = False
    check("fifty consecutive two-foot solves never break the Cholesky", ok,
          "this is what QP_SIGMA is for -- P is only semidefinite on two legs")

    print(f"self-test {'PASS' if _PASS[0] == _PASS[1] else 'FAIL'} "
          f"({_PASS[0]}/{_PASS[1]})")
    return 0 if _PASS[0] == _PASS[1] else 1


if __name__ == "__main__":
    sys.exit(self_test())
