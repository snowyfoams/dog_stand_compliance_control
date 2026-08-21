#!/usr/bin/env python3
"""The convex MPC: the condensed QP over the horizon, and the ADMM that solves it.

    min_U  sum_k || x_k - x_ref_k ||^2_Q  +  w_f ||U - U_grav||^2
                                          +  w_s ||f_0 - f_app||^2
    s.t.   x_{k+1} = Ad x_k + Bd u_k                  (eliminated -- condensed)
           fz_i,k  in [fz_min, fz_max] * weight_i,k   (unilateral, and the gait)
           |fx|, |fy| <= (mu/sqrt2) fz                (friction, per foot)

`U` is the whole horizon of ground-reaction forces, 12 per knot, in world
axes.  Only its first two knots ever leave this file -- the CAN loop reads the
applied force off them by wall clock (mpc_controller.plan_force) -- and the
rest of the horizon exists so that those two know what is coming.  That is the
entire difference between this and the instantaneous grasp map it replaces,
and it is why the load is handed off a foot BEFORE the gait lifts it rather
than after.

=============================================================================
THE SOLVER IS ADMM, IN-FILE, BECAUSE THE PI HAS NOTHING ELSE
=============================================================================
The simulation calls OSQP.  This venv has numpy and no scipy, osqp, quadprog,
cvxpy or qpsolvers -- the same fact dog5_trot/balance_qp.py records and solves
the same way.  So OSQP's iteration is written out:

    (P + sigma I + rho C'C) x = sigma x - q + C'(rho z - y)
    z = clip(C x + y/rho, l, u)
    y = y + rho (C x - z)

sigma keeps the left-hand side positive definite where P is only
semidefinite -- and P IS semidefinite here at every step of a trot, because a
diagonal pair spans fewer than six independent wrench directions.  Without
sigma the factorisation fails exactly when the robot is on two legs.

WHERE THE STRUCTURE IS SPENT
    96 variables and 160 constraint rows would be a real cost dense.  They are
    not, because C is blockdiag(D) with the SAME 5x3 D at every foot of every
    knot:

        C x    reshape to (4N,3), one (4N,3) @ (3,5)          480 MACs
        C' y   reshape to (4N,5), one (4N,5) @ (5,3)          480 MACs
        C'C    kron(I, D'D) -- a 3x3 added to each diagonal block of P

    so an iteration is dominated by one 96x96 matrix-vector product, and the
    whole 60-iteration budget costs less than building the condensed Hessian
    once.

    The Hessian is where the time actually goes: Bqp is 104x96 and Bqp' Q Bqp
    is ~1 million multiply-adds.  That is the number to watch in the runner's
    solve-time report, and mpc_config.N_HORIZON records what happens above 100
    variables.

THE FORCE REGULARISER PULLS TOWARDS GRAVITY, NOT TOWARDS ZERO
    Twelve forces against six wrench rows leaves a six-wide null space at every
    knot, and SOMETHING has to pick a point in it -- that is what the
    regulariser is for.  The simulation writes it as w_f ||U||^2, which picks
    the smallest forces.

    The objection is not that the smallest forces are large; measured at the
    weights this stack ships, the standing support with ||U||^2 comes out
    56.97 N against a 57.05 N robot, a 0.07 N offset nobody would ever see.
    It is that THE REFERENCE STATE IS NOT A ZERO-COST POINT: at x = x_ref the
    optimum is not U_grav, so the solver is always trading a little state error
    against a little force, and how much it trades depends on weights that a
    later bring-up ladder is expected to move.  Driven to a tenth of these
    weights the same probe comes back +0.41 N on the other side, and to a
    hundredth +5.45 N -- not a droop, a wander.

    Regularising towards U_grav -- the even split of the WEIGHT across the feet
    the schedule has in contact at that knot -- removes the trade instead of
    sizing it: at the reference state U_grav is feasible, holds the state
    exactly, and costs zero in both terms, so it IS the optimum and the QP's
    whole job is the correction.  It is the same "m*g is the whole point"
    argument Dynamic_Model.body_wrench makes about its own Fz term, and it
    buys a warm start that already stands.

    The even split is also very nearly moment-free on this stance, which is why
    it is a good null-space pick and not merely a convenient one: the four feet
    straddle the CoM within a millimetre, and so does EITHER trot diagonal
    (+/-0.342 m in x, +/-0.112 m in y about a CoM 0.3 mm off centre).

ONE FIXED ITERATION BUDGET, WITH AN EARLY EXIT IN NEWTONS
    A control loop wants the same cost every solve more than it wants the last
    digit, so the budget is fixed at QP_ITERS and convergence is POLLED every
    fifth iteration rather than watched.  The tolerance is a FORCE -- the
    constraint rows are newtons -- and at 1 mN it is three orders below
    anything this robot can measure.

    WHAT THAT ACTUALLY COSTS, measured, because "converges quickly" is the kind
    of claim that is never true in the case that matters:

        settled four-foot stand    5 iterations, primal residual 0
        trotting, 300 solves       60 every time -- the budget, never the
                                   tolerance -- and it stops at a primal
                                   residual of 1.3 mN mean, 2.4 mN worst

    So a trot pays the whole budget.  That is the honest answer and it is
    affordable (the solve is ~1 ms), and the 2.4 mN it stops at is three
    orders below the friction bound it is a residual on.  Both residuals come
    back on the result so the runner can log it rather than assume it.

AND THE FIRST KNOT IS CLAMPED FEASIBLE ON THE WAY OUT
    An ADMM iterate satisfies the constraints only in the limit.  At 60
    iterations the friction rows are typically violated by milli-newtons --
    but "typically" is not a guarantee, and the force leaving this file is
    about to be multiplied by J^T and sent to a motor.  So the first knot gets
    the same unilateral-and-friction clamp force_totorque applies to its own
    solution, which makes the COMMAND feasible by construction regardless of
    what the solver did.
"""
from __future__ import annotations

import time

import numpy as np

from . import mpc_config as C
from .srb_model import SingleRigidBody, condense, N_STATE, N_FORCE

N_CONE_ROWS = 5          # per foot, per knot: one fz box + four tangential


def _cone_block(mu_axis):
    """D (5,3): the per-foot constraint rows, identical everywhere.

        [ 0   0   1 ]   fz            in [fz_min w, fz_max w]
        [+1   0  -k ]   +fx - k fz    <= 0
        [-1   0  -k ]   -fx - k fz    <= 0
        [ 0  +1  -k ]   +fy - k fz    <= 0
        [ 0  -1  -k ]   -fy - k fz    <= 0

    A SWING FOOT NEEDS NO SPECIAL CASE.  Its weight is 0, so the fz box
    becomes [0, 0] and the four tangential rows read |fx| <= 0 and |fy| <= 0 --
    the whole 3-vector is driven to zero by constraints already present.  One
    code path for both states is one fewer place for the contact mask to be
    misread.
    """
    k = float(mu_axis)
    return np.array([[0.0, 0.0, 1.0],
                     [+1.0, 0.0, -k],
                     [-1.0, 0.0, -k],
                     [0.0, +1.0, -k],
                     [0.0, -1.0, -k]])


def clamp_feasible(f, weight, mu=C.MU, fz_min=C.FZ_MIN, fz_max=C.FZ_MAX):
    """Project one knot of per-foot forces onto the cone.  (4,3)

    Unilateral first (a foot pushes, it never pulls the robot down), then the
    tangential magnitude into mu*fz.  The CIRCLE is used here, not the
    inscribed pyramid the QP was constrained with: the pyramid is the
    conservative linear stand-in the QP needs, and projecting the answer onto
    the true cone can only relax a bound the solver already respected.
    """
    f = np.asarray(f, dtype=float).reshape(4, 3).copy()
    w = np.clip(np.asarray(weight, dtype=float).reshape(4), 0.0, 1.0)
    for i in range(4):
        f[i, 2] = float(np.clip(f[i, 2], fz_min * w[i], fz_max * w[i]))
        t = f[i, :2]
        t_max = mu * f[i, 2]
        t_norm = float(np.linalg.norm(t))
        if t_norm > t_max and t_norm > 1e-12:
            f[i, :2] = t * (t_max / t_norm)
    return f


def restore_support(f_clamped, f_plan, contact, fz_max=C.FZ_MAX, s_max=2.0):
    """Give the load a clamp took off an airborne foot to the feet that are down.

    THE PROBLEM THIS SOLVES.  The plan is read at `now` by interpolating across
    its first knot, and a knot is 50 ms wide -- so near a handover the
    interpolation is already routing load to the foot that is ABOUT to land.
    The contact clamp then zeroes that share, correctly, because the foot is
    still in the air.  What is left is the plan minus a foot: measured over
    four gait cycles, the commanded support fell below 90% of the robot's
    weight on 20 of 133 model ticks and as low as 27 N on a 57 N robot.

    Nothing about that lost force was wrong except WHICH FOOT it was assigned
    to.  The robot still weighs what it weighs, and the feet that are down are
    the only ones that can carry it -- so the surviving stance forces are
    scaled up to the total the plan asked for.  It is the mirror of the
    rescale force_totorque.distribute already does in the other direction, and
    the same argument: "total support is what holds the robot up and must be
    honoured; the moment is a trim".

    SCALING THE WHOLE VECTOR, not just its z, is what keeps every foot inside
    the friction cone it was just clamped into -- a uniform scale moves f and
    mu*fz together.  What it does NOT preserve is the moment, and it cannot:
    the moment the missing foot would have made is not available from feet
    that are somewhere else.  That trim is one model tick old at worst and the
    attitude loop is what removes it.

    `s_max` bounds the repair: if more than half the plan is missing, the honest
    answer is a dip, not a foot pushed twice as hard as anything was planned
    for.
    """
    f = np.asarray(f_clamped, dtype=float).reshape(4, 3).copy()
    plan = np.asarray(f_plan, dtype=float).reshape(4, 3)
    on = np.asarray(contact, dtype=bool).reshape(4)
    if not on.any():
        return f
    target = float(np.sum(plan[:, 2]))
    have = float(np.sum(f[:, 2]))
    if target <= 0.0 or have <= 1e-9 or have >= target:
        return f
    s = min(target / have, s_max)
    f[on] *= s
    # the scale can push a foot past its own box; that bound is physical and
    # the repair is not allowed to break it
    f[on, 2] = np.minimum(f[on, 2], fz_max)
    return f


class ConvexMPC:
    """One horizon of ground-reaction forces, replanned on demand.

    Stateless between solves except for the warm start, which is the point of
    keeping an object at all: the previous horizon SHIFTED BY ONE KNOT is a
    good guess for this one, and the previous dual tells ADMM which
    constraints were active.
    """

    def __init__(self, mass=C.MASS, inertia_body=C.INERTIA_BODY,
                 gravity=C.GRAVITY, n_horizon=C.N_HORIZON, dt=C.MPC_DT,
                 w_att=C.W_ATT, w_pos=C.W_POS, w_omega=C.W_OMEGA,
                 w_vel=C.W_VEL, w_force=C.W_FORCE, w_smooth=C.W_SMOOTH,
                 mu_axis=C.MU_AXIS, fz_min=C.FZ_MIN, fz_max=C.FZ_MAX,
                 iters=C.QP_ITERS, rho=C.QP_RHO, sigma=C.QP_SIGMA,
                 tol=C.QP_TOL_N):
        self.srb = SingleRigidBody(mass, inertia_body, gravity)
        # The discretisation is exact only while A stays nilpotent; assert it
        # once, here, rather than discovering a truncated series in a log.
        self.srb.assert_nilpotent()
        self.N = int(n_horizon)
        self.dt = float(dt)
        self.nu = N_FORCE * self.N
        self.q_state = np.concatenate([
            np.asarray(w_att, float).reshape(3),
            np.asarray(w_pos, float).reshape(3),
            np.asarray(w_omega, float).reshape(3),
            np.asarray(w_vel, float).reshape(3),
            [0.0]])                       # the gravity state is not regulated
        if np.any(self.q_state < 0.0):
            raise ValueError("state weights must be non-negative")
        self.w_force = float(w_force)
        self.w_smooth = float(w_smooth)
        self.weight_total = self.srb.mass * self.srb.gravity   # for U_grav
        self.fz_min, self.fz_max = float(fz_min), float(fz_max)
        self.D = _cone_block(mu_axis)
        self.DtD = self.D.T @ self.D
        self.iters, self.rho, self.sigma = int(iters), float(rho), float(sigma)
        self.tol = float(tol)                # newtons; see mpc_config.QP_TOL_N
        self.n_rows = N_CONE_ROWS * 4 * self.N
        # Q repeated over the horizon, as a DIAGONAL: it is one, and forming
        # the 130x130 matrix to multiply by it would cost more than the solve.
        self.q_bar = np.tile(self.q_state, self.N)
        self.reset()

    def reset(self):
        self.prev_U = np.zeros(self.nu)
        self._y_prev = np.zeros(self.n_rows)
        self._mask_prev = None
        self.last = {}

    # ------------------------------------------------------------------ solve
    def solve(self, x0, x_ref, weight_sched, r_feet, yaw=0.0, f_applied=None):
        """Plan the horizon; return (f0 (4,3), f1 (4,3), info).

        x0            (13,)      current state, world axes, gravity last
        x_ref         (N, 13)    the reference at knots 1..N
        weight_sched  (N, 4)     per-foot contact weight in [0,1]; 0 is swing
        r_feet        (4, 3)     foot MINUS CoM, world axes, at knot 0
        yaw           float      the linearisation yaw (0 here; see srb_model)
        f_applied     (4,3)      the force currently on the robot, for W_SMOOTH

        THE FOOT GEOMETRY IS FROZEN OVER THE HORIZON, exactly as in the
        simulation and in Di Carlo et al.: r_i is taken at knot 0 and held.
        Re-planning at MPC_HZ is what keeps that honest -- a foot moves
        0.5 * T_stance * v = 0.5 mm at this stance's speeds within one replan,
        against a 340 mm lever arm.
        """
        t_start = time.perf_counter()
        N, nu = self.N, self.nu
        x0 = np.asarray(x0, dtype=float).reshape(N_STATE)
        x_ref = np.asarray(x_ref, dtype=float).reshape(N, N_STATE)
        w = np.clip(np.asarray(weight_sched, dtype=float).reshape(N, 4),
                    0.0, 1.0)

        A, B = self.srb.continuous(yaw, r_feet)
        Ad, Bd = SingleRigidBody.discretize(A, B, self.dt)
        Aqp, Bqp = condense(Ad, Bd, N)

        # -- the objective, as 1/2 U' P U + q' U ---------------------------
        err0 = Aqp @ x0 - x_ref.reshape(-1)        # free response minus the ref
        BQ = Bqp * self.q_bar[:, None]             # Q is diagonal; scale rows
        P = 2.0 * (Bqp.T @ BQ)
        P[np.diag_indices(nu)] += 2.0 * self.w_force
        q = 2.0 * (Bqp.T @ (self.q_bar * err0))
        # ...and the regulariser pulls towards the gravity share, not zero.
        # See the header: towards zero it is a standing sag, not a preference.
        u_grav = self.gravity_share(w)
        q -= 2.0 * self.w_force * u_grav
        if self.w_smooth > 0.0 and f_applied is not None:
            fa = np.asarray(f_applied, dtype=float).reshape(N_FORCE)
            P[np.diag_indices(N_FORCE)] += 2.0 * self.w_smooth
            q[:N_FORCE] -= 2.0 * self.w_smooth * fa

        # -- the cone, as l <= C U <= u ------------------------------------
        wf = w.reshape(-1)                          # (4N,) foot-major per knot
        lo = np.zeros((4 * N, N_CONE_ROWS))
        up = np.zeros((4 * N, N_CONE_ROWS))
        lo[:, 0] = self.fz_min * wf
        up[:, 0] = self.fz_max * wf
        lo[:, 1:] = -np.inf                          # tangential rows are <= 0
        lo, up = lo.reshape(-1), up.reshape(-1)

        # -- warm start ----------------------------------------------------
        # THE PRIMAL IS SHIFTED, NOT REUSED.  Knot k of the last horizon is
        # knot k-1 of this one; handing the solver the unshifted vector asks
        # it to undo one knot of gait every time, which is most of the
        # iteration budget.  The tail is repeated rather than zeroed, so the
        # guess stays a plausible stance rather than a sudden unloading.
        x = np.empty(nu)
        x[:nu - N_FORCE] = self.prev_U[N_FORCE:]
        x[nu - N_FORCE:] = self.prev_U[nu - N_FORCE:]
        # The DUAL encodes which constraints are active, and that set is
        # unchanged for a whole stance phase -- but it is meaningless across a
        # different contact pattern, so it is dropped when the pattern moves.
        mask = tuple((w > 0.0).reshape(-1).tolist())
        y = (self._y_prev.copy() if mask == self._mask_prev
             else np.zeros(self.n_rows))
        self._mask_prev = mask

        x, y, info = self._admm(P, q, lo, up, x, y)
        self.prev_U = x
        self._y_prev = y

        f0 = clamp_feasible(x[:N_FORCE].reshape(4, 3), w[0],
                            fz_max=self.fz_max, fz_min=self.fz_min)
        # KNOT 1 COMES BACK TOO, and it is not a diagnostic.  The plan is a
        # trajectory, and a caller that holds only its first sample for a whole
        # replan period throws away exactly the part that describes the load
        # handover -- see mpc_controller.plan_force.
        if N > 1:
            f1 = clamp_feasible(x[N_FORCE:2 * N_FORCE].reshape(4, 3), w[1],
                                fz_max=self.fz_max, fz_min=self.fz_min)
        else:
            f1 = f0.copy()
        info["solve_s"] = time.perf_counter() - t_start
        info["fz_total"] = float(np.sum(f0[:, 2]))
        self.last = info
        return f0, f1, info

    def gravity_share(self, weight_sched):
        """U_grav (12N,): the weight split evenly over the feet in contact.

        Vertical only, and proportional to each foot's contact WEIGHT, so a
        foot ramping in or out takes its share of the load smoothly and the
        total always adds up to m*g.  A knot with no foot down (which a duty
        of 0.6 never produces, but a jump would) gets zeros rather than a
        divide by zero -- the honest answer for a body in flight.
        """
        w = np.asarray(weight_sched, dtype=float).reshape(-1, 4)
        tot = w.sum(axis=1)
        share = np.zeros_like(w)
        live = tot > 1e-9
        share[live] = self.weight_total * w[live] / tot[live, None]
        u = np.zeros((w.shape[0], 4, 3))
        u[:, :, 2] = share
        return u.reshape(-1)

    # ------------------------------------------------------------------ ADMM
    def _Cx(self, x):
        """C @ x, by structure: (4N,3) @ (3,5) -> (4N,5)."""
        return (x.reshape(-1, 3) @ self.D.T).reshape(-1)

    def _Cty(self, y):
        """C^T @ y, by structure: (4N,5) @ (5,3) -> (4N,3)."""
        return (y.reshape(-1, N_CONE_ROWS) @ self.D).reshape(-1)

    def _admm(self, P, q, lo, up, x, y):
        rho, sig = self.rho, self.sigma
        # K = P + sigma I + rho C'C, and C'C is kron(I, D'D): a 3x3 added to
        # each diagonal block.  Building the 200x120 C to form C'C would cost
        # more than the factorisation it feeds.
        K = P.copy()
        K[np.diag_indices(self.nu)] += sig
        rho_DtD = rho * self.DtD
        for b in range(0, self.nu, 3):
            K[b:b + 3, b:b + 3] += rho_DtD
        # Cholesky as the positive-definiteness CHECK, then an explicit
        # inverse to apply: the same trade balance_qp makes and for the same
        # reason -- the matrix is factored once and applied `iters` times, and
        # two triangular solves per iteration cost more than the inverse at
        # this size.
        np.linalg.cholesky(K)
        Kinv = np.linalg.inv(K)

        z = np.clip(self._Cx(x), lo, up)
        inv_rho = 1.0 / rho
        pri = dua = 0.0
        used = 0
        for k in range(self.iters):
            x = Kinv @ (sig * x - q + self._Cty(rho * z - y))
            Cx = self._Cx(x)
            z_new = np.clip(Cx + y * inv_rho, lo, up)
            y = y + rho * (Cx - z_new)
            dz = z_new - z
            z = z_new
            used = k + 1
            if (k + 1) % 5 == 0:                 # polled, not watched: the two
                pri = float(np.max(np.abs(Cx - z)))    # reductions cost about
                dua = rho * float(np.max(np.abs(dz)))  # as much as a step
                if pri < self.tol and dua < self.tol:
                    break
        return x, y, {"iters": used, "primal": pri, "dual": dua}
