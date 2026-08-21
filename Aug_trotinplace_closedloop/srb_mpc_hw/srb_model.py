#!/usr/bin/env python3
"""The single rigid body: its linearised dynamics, and an EXACT discretisation.

    x = [theta(rpy), p(3), omega(3), v(3), g]        13, world axes
    u = [f_FL, f_FR, f_RL, f_RR]                     12, world axes

    theta_dot = Rz(psi)^T omega
    p_dot     = v
    omega_dot = I_world^-1 sum_i (r_i x f_i)
    v_dot     = (1/m) sum_i f_i  -  g e_z

This is Di Carlo, Wensing, Katz, Bledt, Kim, "Dynamic Locomotion in the MIT
Cheetah 3 Through Convex Model-Predictive Control" (IROS 2018), eq. (16)-(17),
and it is the same reduction the simulation uses.  Two properties make it
convex: the inertia is taken constant in the body frame, and the orientation
enters only through the current yaw.

THE GRAVITY STATE IS WHY THERE IS A 13th ROW
    v_z_dot = ... - g is affine, not linear, and a QP wants a linear system.
    Carrying g as a state with g_dot = 0 and A[v_z, g] = -1 makes it linear
    again at the cost of one row that never moves.  Its cost weight is zero.

=============================================================================
THE DISCRETISATION IS EXACT AND HAS NO expm() IN IT
=============================================================================
The Pi's venv has numpy and nothing else -- no scipy, so no
`scipy.linalg.expm`, which is what the simulation calls.  It does not need
one, because A IS NILPOTENT WITH A^3 = 0.  Read the coupling off the four
rows above:

    theta <- omega,  omega <- (nothing)      so A^2 kills the theta row
    p     <- v,      v     <- g,  g <- 0     so p <- g at A^2, and 0 at A^3

so the series terminates after three terms and

    Ad = expm(A dt) = I + A dt + A^2 dt^2 / 2                      EXACTLY.

The input map is the same argument on the augmented matrix M = [[A, B], [0,0]]:
A B is non-zero (a force reaches theta through omega, and p through v), but

    (A^2 B) = A (A B),  and (A B) has ZERO rows at omega and at v,

which are the only rows A reads.  So A^2 B = 0 and

    Bd = B dt + (A B) dt^2 / 2                                     EXACTLY.

Two matrix products and two adds, against expm's Pade approximant and its
scaling-and-squaring -- and no approximation to justify, at any dt.  There is
no accuracy argument to have here: this is not a cheap substitute for the
matrix exponential, it IS the matrix exponential for this A.

WHY THE INERTIA IS ROTATED BY YAW ONLY, AND WHY THAT IS IDENTITY HERE
    The MIT form uses Rz(psi) alone because roll and pitch are small in
    locomotion, and rotating by them would make I_world depend on the state
    the controller is regulating.  On this robot the argument is stronger
    still: the MPC is re-anchored at every solve with yaw = 0 (see
    mpc_controller), so Rz is the identity every time and I_world IS I_body.

    The yaw argument is kept anyway, and honestly: it is what the model is,
    and the day this stack gets an absolute heading it is the one line that
    has to change.
"""
from __future__ import annotations

import numpy as np

N_STATE = 13
N_FORCE = 12
IDX_RPY = slice(0, 3)
IDX_POS = slice(3, 6)
IDX_OMEGA = slice(6, 9)
IDX_VEL = slice(9, 12)
IDX_GRAV = 12


def skew(r):
    """[r]x, so that skew(r) @ f == cross(r, f)."""
    x, y, z = (float(v) for v in r)
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]])


def Rz(psi):
    """Yaw rotation, body -> world."""
    c, s = np.cos(float(psi)), np.sin(float(psi))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class SingleRigidBody:
    """Mass, inertia, and the two matrices the MPC needs from them.

    `inertia_body` is the WHOLE-ROBOT tensor about the WHOLE-ROBOT CoM, in the
    trunk frame -- not the trunk's own.  The legs are 55% of this robot, so
    the difference is not a refinement: dog5_statics carries the same
    distinction for the static case and records the 0.482 Nm it is worth
    there.
    """

    def __init__(self, mass, inertia_body, gravity):
        self.mass = float(mass)
        self.I_body = np.asarray(inertia_body, dtype=float).reshape(3, 3)
        self.gravity = float(gravity)
        # Inverted ONCE.  It is a 3x3 and the loop would otherwise invert it
        # at every solve for a matrix that cannot change.
        self.I_body_inv = np.linalg.inv(self.I_body)
        if not np.all(np.linalg.eigvalsh(self.I_body) > 0.0):
            raise ValueError(
                f"inertia_body is not positive definite:\n{self.I_body}\n"
                f"the MPC's moment rows are its inverse, so a bad tensor is a "
                f"bad torque, not a bad number")

    # -- continuous time ---------------------------------------------------
    def continuous(self, yaw, r_feet):
        """(A (13,13), B (13,12)) about `yaw`, with foot lever arms `r_feet`.

        `r_feet` is (4,3): each foot MINUS THE CoM, in world axes.  Passing
        trunk-origin-relative vectors here is the classic way to get a pitch
        bias that no gain removes, because it is not an error -- it is a wrong
        model.  The caller owns that subtraction; mpc_controller does it in
        one place.
        """
        r_feet = np.asarray(r_feet, dtype=float).reshape(4, 3)
        Rzy = Rz(yaw)
        A = np.zeros((N_STATE, N_STATE))
        A[IDX_RPY, IDX_OMEGA] = Rzy.T              # theta_dot = Rz^T omega
        A[IDX_POS, IDX_VEL] = np.eye(3)            # p_dot = v
        A[11, IDX_GRAV] = -1.0                     # v_z_dot -= g, with x[12]=+g

        I_world_inv = Rzy @ self.I_body_inv @ Rzy.T
        B = np.zeros((N_STATE, N_FORCE))
        inv_m = 1.0 / self.mass
        for i in range(4):
            sl = slice(3 * i, 3 * i + 3)
            B[IDX_OMEGA, sl] = I_world_inv @ skew(r_feet[i])
            B[IDX_VEL, sl] = inv_m * np.eye(3)
        return A, B

    # -- discrete time -----------------------------------------------------
    @staticmethod
    def discretize(A, B, dt):
        """(Ad, Bd) = the EXACT zero-order-hold discretisation.  See the header.

        Ad = I + A dt + A^2 dt^2/2 and Bd = B dt + (A B) dt^2/2, which are the
        full series because A^3 = 0 and A^2 B = 0.  `assert_nilpotent` below
        is the statement of that, callable by anyone who does not believe it.
        """
        dt = float(dt)
        A2 = A @ A
        Ad = np.eye(N_STATE) + A * dt + A2 * (0.5 * dt * dt)
        Bd = B * dt + (A @ B) * (0.5 * dt * dt)
        return Ad, Bd

    def assert_nilpotent(self, yaw=0.0, r_feet=None, tol=1e-12):
        """Raise unless A^3 == 0 and A^2 B == 0 -- the discretisation's premise.

        Cheap (two 13x13 products) and it is called once at construction time
        by the controller, so a future edit that couples, say, omega back into
        the theta row turns the exact series into a silently truncated one and
        this is what catches it.
        """
        if r_feet is None:
            r_feet = np.array([[0.34, 0.11, -0.19], [0.34, -0.11, -0.19],
                               [-0.34, 0.11, -0.19], [-0.34, -0.11, -0.19]])
        A, B = self.continuous(yaw, r_feet)
        a3 = float(np.max(np.abs(A @ A @ A)))
        a2b = float(np.max(np.abs(A @ A @ B)))
        if a3 > tol or a2b > tol:
            raise RuntimeError(
                f"the SRB's A is no longer nilpotent (max|A^3| = {a3:.3e}, "
                f"max|A^2 B| = {a2b:.3e}).  srb_model.discretize is the exact "
                f"matrix exponential ONLY while both are zero -- with a new "
                f"coupling in A it silently becomes a truncated series, and "
                f"the venv has no scipy.linalg.expm to fall back to.")
        return a3, a2b


def condense(Ad, Bd, N):
    """Stack the horizon: X = Aqp x0 + Bqp U.

    Returns (Aqp (13N, 13), Bqp (13N, 12N)) for the standard condensed form,
    with U the whole horizon of forces.  Aqp's k-th block is Ad^(k+1) and
    Bqp's (k, j) block is Ad^(k-j) Bd for j <= k, zero above.

    CONDENSED AND NOT SPARSE, DELIBERATELY.  The sparse (multiple-shooting)
    form is 13N + 12N variables with equality constraints for the dynamics and
    is what a sparse solver wants; this one eliminates the states and leaves
    12N = 120 dense variables.  With numpy alone there is no sparse
    factorisation to exploit, so the dense form -- whose cost is two matrix
    products the BLAS does well -- is the cheaper of the two here.  It is also
    the simulator's form, which keeps the two comparable.
    """
    N = int(N)
    nx, nu = Ad.shape[0], Bd.shape[1]
    powers = [np.eye(nx)]
    for _ in range(N):
        powers.append(Ad @ powers[-1])
    Aqp = np.zeros((nx * N, nx))
    Bqp = np.zeros((nx * N, nu * N))
    for k in range(N):
        Aqp[nx * k:nx * (k + 1)] = powers[k + 1]
        for j in range(k + 1):
            Bqp[nx * k:nx * (k + 1), nu * j:nu * (j + 1)] = powers[k - j] @ Bd
    return Aqp, Bqp
