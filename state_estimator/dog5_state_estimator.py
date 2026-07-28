"""DOG5.0 proprioceptive state estimator — quaternion error-state EKF.

Implementation of the Bloesch et al. (RSS 2013) legged-robot state estimator,
following ``plan.md`` in this directory.  Modules M0 (rotation math), M1 (state
container), M2 (leg-kinematics wrapper over ``dog5_kinematics``), M3
(initialisation), M4 (nominal propagation), M5 (covariance propagation), M6
(contact transitions), M7 (measurement assembly), M8 (update), M9 (outlier
gate), M10 (first-estimates-Jacobian constraint) and M11 (output/health) all
live here.  Every equation number in a comment refers to the paper as
transcribed in Appendix B of ``plan.md``.

Conventions (plan Sec. 2 / Appendix B):
    * quaternion ``[x, y, z, w]`` representing the rotation I -> B (JPL product)
    * ``C(q)`` maps I-coordinates of a vector into B-coordinates
    * attitude error ``q = zeta(dphi) (x) q_hat`` (left multiplication)
    * gravity ``g = [0, 0, -9.81]`` in I; at rest and level ``f ~ [0,0,+9.81]``
    * error-state dimension 27, index map in ``ErrorIndex``
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np

# dog5_kinematics lives one directory up in dog5_description.
_DESC = Path(__file__).resolve().parent.parent / "dog5_description"
if str(_DESC) not in sys.path:
    sys.path.insert(0, str(_DESC))

import dog5_kinematics  # noqa: E402

LEGS = dog5_kinematics.LEGS  # ("FL", "FR", "RL", "RR")
N_LEGS = len(LEGS)
GRAVITY = np.array([0.0, 0.0, -9.81])


# =====================================================================
# M0 — Rotation math library (Appendix B.5, B.11)
# =====================================================================

def skew(v):
    """Return the 3x3 skew-symmetric matrix with ``skew(a) @ b == a x b``."""
    x, y, z = v
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def quat_mul(a, b):
    """JPL quaternion product with ``C(a (x) b) == C(a) @ C(b)`` (B.11).

    Quaternions are ``[x, y, z, w]`` (vector part first, scalar last).
    """
    av, aw = a[:3], a[3]
    bv, bw = b[:3], b[3]
    v = aw * bv + bw * av - np.cross(av, bv)
    w = aw * bw - av @ bv
    return np.array([v[0], v[1], v[2], w])


def quat_to_C(q):
    """Rotation matrix mapping I-coordinates into B-coordinates (JPL)."""
    qv = q[:3]
    qw = q[3]
    vv = qv @ qv
    return (qw * qw - vv) * np.eye(3) - 2.0 * qw * skew(qv) + 2.0 * np.outer(qv, qv)


def zeta(phi):
    """Rotation vector -> quaternion, eq. (13).  ``[x,y,z,w]``."""
    phi = np.asarray(phi, dtype=float)
    n = float(np.linalg.norm(phi))
    if n < 1.0e-8:
        # sin(n/2)/n -> 1/2 as n -> 0; keep it analytic near zero.
        v = 0.5 * phi * (1.0 - n * n / 24.0)
        w = 1.0 - n * n / 8.0
        return np.array([v[0], v[1], v[2], w])
    v = math.sin(n / 2.0) * phi / n
    w = math.cos(n / 2.0)
    return np.array([v[0], v[1], v[2], w])


def zeta_inv(q):
    """Quaternion -> rotation vector, inverse of eq. (13); wrapped to (-pi, pi]."""
    q = normalize(q)
    qv = q[:3]
    qw = q[3]
    nv = float(np.linalg.norm(qv))
    if nv < 1.0e-10:
        return np.zeros(3)
    angle = 2.0 * math.atan2(nv, qw)
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return angle * qv / nv


def normalize(q):
    """Unit-norm quaternion with a non-negative scalar part."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    if q[3] < 0.0:
        q = -q
    return q


def gamma(omega, dt, n, terms=15):
    """Series of eq. (28): ``Gamma_n = sum_j dt^(j+n)/(j+n)! (omega^x)^j``.

    ``gamma(omega, dt, 0)`` equals ``expm(dt * skew(omega))`` (eq. 29).
    """
    W = skew(omega)
    G = np.zeros((3, 3))
    Wj = np.eye(3)  # (omega^x)^j, starting at j = 0
    for j in range(terms):
        G += dt ** (j + n) / math.factorial(j + n) * Wj
        Wj = Wj @ W
    return G


# =====================================================================
# M2 — Leg kinematics wrapper (plan M2)
# =====================================================================
# The heavy lifting (FK and analytic geometric Jacobian, validated against
# MuJoCo to 1e-10 in check_dog5_kinematics.py) lives in dog5_kinematics.  This
# thin wrapper adapts it to the estimator's leg-index API: s_i is the foot
# contact point in the *body/trunk* frame, J_i = d s_i / d alpha_i (3x3).

def leg_fk(leg_index, alpha_i):
    """Foot contact point of one leg in the body frame (eq. 2 lkin_i)."""
    return dog5_kinematics.foot_position(LEGS[leg_index], alpha_i)


def leg_jac(leg_index, alpha_i):
    """Analytic Jacobian ``d s_i / d alpha_i`` (eq. 26)."""
    return dog5_kinematics.foot_jacobian(LEGS[leg_index], alpha_i)


# =====================================================================
# M1 — State container (Appendix B.2)
# =====================================================================
# Nominal state x (28):  r(3) v(3) q(4) p1..p4(12) b_f(3) b_w(3)
# Error state  dx (27):  dr(3) dv(3) dphi(3) dp1..dp4(12) db_f(3) db_w(3)

class NominalIndex:
    R = slice(0, 3)
    V = slice(3, 6)
    Q = slice(6, 10)
    P = slice(10, 22)          # all four footholds
    BF = slice(22, 25)
    BW = slice(25, 28)
    DIM = 28

    @staticmethod
    def p(i):
        return slice(10 + 3 * i, 13 + 3 * i)


class ErrorIndex:
    R = slice(0, 3)
    V = slice(3, 6)
    PHI = slice(6, 9)
    P = slice(9, 21)
    BF = slice(21, 24)
    BW = slice(24, 27)
    DIM = 27

    @staticmethod
    def p(i):
        return slice(9 + 3 * i, 12 + 3 * i)


@dataclass
class State:
    """Nominal state x and covariance P with named accessors."""

    x: np.ndarray = field(default_factory=lambda: np.zeros(NominalIndex.DIM))
    P: np.ndarray = field(default_factory=lambda: np.zeros((ErrorIndex.DIM, ErrorIndex.DIM)))

    def __post_init__(self):
        # A fresh state must carry a valid (identity) quaternion.
        if np.linalg.norm(self.x[NominalIndex.Q]) == 0.0:
            self.x[NominalIndex.Q] = np.array([0.0, 0.0, 0.0, 1.0])

    # --- nominal accessors ---
    @property
    def r(self):
        return self.x[NominalIndex.R]

    @r.setter
    def r(self, val):
        self.x[NominalIndex.R] = val

    @property
    def v(self):
        return self.x[NominalIndex.V]

    @v.setter
    def v(self, val):
        self.x[NominalIndex.V] = val

    @property
    def q(self):
        return self.x[NominalIndex.Q]

    @q.setter
    def q(self, val):
        self.x[NominalIndex.Q] = val

    @property
    def bf(self):
        return self.x[NominalIndex.BF]

    @bf.setter
    def bf(self, val):
        self.x[NominalIndex.BF] = val

    @property
    def bw(self):
        return self.x[NominalIndex.BW]

    @bw.setter
    def bw(self, val):
        self.x[NominalIndex.BW] = val

    def p(self, i):
        return self.x[NominalIndex.p(i)]

    def set_p(self, i, val):
        self.x[NominalIndex.p(i)] = val

    @property
    def C(self):
        """Rotation matrix I -> B for the current attitude."""
        return quat_to_C(self.q)

    def copy(self):
        return State(self.x.copy(), self.P.copy())


# =====================================================================
# Parameters (plan Sec. 9)
# =====================================================================

@dataclass
class EstimatorParams:
    # IMU continuous noise densities (units so that the discrete Q of B.7 is
    # dimensionally correct: sigma_f in (m/s^2)/sqrt(Hz), sigma_w in
    # (rad/s)/sqrt(Hz), bias random-walk in per-sqrt(Hz) units).
    sigma_f: float = 5.0e-3        # accel white noise density (m/s^2 / sqrt(Hz))
    sigma_w: float = 5.0e-4        # gyro white noise density (rad/s / sqrt(Hz))
    sigma_bf: float = 1.0e-3       # accel bias random walk
    sigma_bw: float = 1.0e-4       # gyro bias random walk

    # Kinematic measurement noise.
    sigma_alpha: float = 0.002     # encoder noise (rad)
    sigma_s: float = 0.010         # kinematic model error (m)

    # Foothold process noise (body frame, in contact).
    sigma_p_xy: float = 1.0e-3     # tangential slip (m / sqrt(s))
    sigma_p_z: float = 5.0e-4      # normal slip
    swing_p: float = 1.0e4         # swing-foot process noise (paper Sec. III-B)

    # Contact transition / init.
    sigma_reset: float = 0.05      # fresh landmark uncertainty at touchdown (m)

    # Initial covariance diagonal.
    p0_r: float = 1.0e-6           # position is defined, not measured
    p0_v: float = 0.1              # m/s
    p0_att: float = math.radians(1.0)   # 1 deg
    p0_p: float = 0.05             # foothold
    p0_bf: float = 0.1
    p0_bw: float = 0.01

    # Outlier gate (M9).
    mahalanobis_gate: float = 9.0  # chi^2, 3 DOF, ~97 %

    @property
    def Qf(self):
        return self.sigma_f ** 2 * np.eye(3)

    @property
    def Qw(self):
        return self.sigma_w ** 2 * np.eye(3)

    @property
    def Qbf(self):
        return self.sigma_bf ** 2 * np.eye(3)

    @property
    def Qbw(self):
        return self.sigma_bw ** 2 * np.eye(3)

    @property
    def Rs(self):
        return self.sigma_s ** 2 * np.eye(3)

    @property
    def Ralpha(self):
        return self.sigma_alpha ** 2 * np.eye(3)

    def Qp_contact(self):
        """Foothold process noise in the body frame, foot in contact (eq. 21)."""
        return np.diag([self.sigma_p_xy ** 2, self.sigma_p_xy ** 2, self.sigma_p_z ** 2])

    def Qp_swing(self):
        return self.swing_p ** 2 * np.eye(3)


# =====================================================================
# The estimator
# =====================================================================

class DOG5StateEstimator:
    """Quaternion error-state EKF for DOG5.0."""

    def __init__(self, params: EstimatorParams | None = None):
        self.params = params or EstimatorParams()
        self.state = State()
        self.prev_contacts = np.zeros(N_LEGS, dtype=bool)
        # FEJ anchors (M10): foothold position frozen at most-recent touchdown.
        self.fej_anchor = [None] * N_LEGS
        self.use_fej = False
        self.use_gate = False
        self.estimate_bias = True
        self.initialised = False

    # ---------------- M3 — Initialisation ----------------
    def initialise(self, f_samples, w_samples, alpha, contacts):
        """Build x0, P0 from a static hold (plan M3).

        ``f_samples`` and ``w_samples`` are (K, 3) arrays of static specific
        force / angular rate; ``alpha`` is the (4, 3) joint-angle array;
        ``contacts`` is a length-4 boolean mask.
        """
        p = self.params
        f_mean = np.mean(np.asarray(f_samples), axis=0)
        w_mean = np.mean(np.asarray(w_samples), axis=0)

        bw0 = w_mean.copy()                              # static gyro mean = bias
        g_hat = f_mean / np.linalg.norm(f_mean)          # measured "up" in B
        bf0 = f_mean - 9.81 * g_hat                      # magnitude residual

        # q0: rotation taking inertial +z onto the measured up-direction g_hat.
        # C(q0) maps I -> B, so C(q0) @ [0,0,1] must equal g_hat.
        q0 = self._quat_from_z_to(g_hat)

        st = State()
        st.r = np.zeros(3)
        st.v = np.zeros(3)
        st.q = q0
        st.bf = bf0
        st.bw = bw0
        C = quat_to_C(q0)
        for i in range(N_LEGS):
            s_i = leg_fk(i, np.asarray(alpha)[i])
            st.set_p(i, st.r + C.T @ s_i)                # p_i = r + C^T s_i

        # P0 diagonal (plan Sec. 9).
        P0 = np.zeros((ErrorIndex.DIM, ErrorIndex.DIM))
        P0[ErrorIndex.R, ErrorIndex.R] = p.p0_r ** 2 * np.eye(3)
        P0[ErrorIndex.V, ErrorIndex.V] = p.p0_v ** 2 * np.eye(3)
        P0[ErrorIndex.PHI, ErrorIndex.PHI] = p.p0_att ** 2 * np.eye(3)
        for i in range(N_LEGS):
            P0[ErrorIndex.p(i), ErrorIndex.p(i)] = p.p0_p ** 2 * np.eye(3)
        P0[ErrorIndex.BF, ErrorIndex.BF] = p.p0_bf ** 2 * np.eye(3)
        P0[ErrorIndex.BW, ErrorIndex.BW] = p.p0_bw ** 2 * np.eye(3)
        st.P = P0

        self.state = st
        self.prev_contacts = np.asarray(contacts, dtype=bool).copy()
        for i in range(N_LEGS):
            self.fej_anchor[i] = st.p(i).copy() if contacts[i] else None
        self.initialised = True
        return st

    @staticmethod
    def _quat_from_z_to(g_hat):
        """Smallest-rotation quaternion whose C maps inertial +z onto g_hat.

        We need C(q) @ e_z = g_hat with C mapping I -> B.  Equivalently the
        rotation that carries e_z to g_hat, expressed as a JPL I->B quaternion.
        """
        ez = np.array([0.0, 0.0, 1.0])
        g_hat = g_hat / np.linalg.norm(g_hat)
        axis = np.cross(ez, g_hat)
        s = float(np.linalg.norm(axis))
        c = float(ez @ g_hat)
        if s < 1.0e-9:
            if c > 0.0:
                return np.array([0.0, 0.0, 0.0, 1.0])
            return np.array([1.0, 0.0, 0.0, 0.0])  # 180 deg about x
        angle = math.atan2(s, c)
        # rotation vector that carries ez -> g_hat in inertial coords
        rotvec = angle * axis / s
        # C(q) maps I->B and must satisfy C @ ez = g_hat.  With the JPL
        # small-angle form C(zeta(v)) ~ I - v^x, the rotation vector inside
        # zeta is the *negative* of the inertial-frame rotation.
        return zeta(-rotvec)

    # ---------------- M4 — Nominal propagation ----------------
    def propagate_state(self, st: State, f_meas, w_meas, dt) -> State:
        """Dead-reckon the nominal state one IMU step (eqs. 30-37)."""
        f_hat = np.asarray(f_meas) - st.bf          # (36)
        w_hat = np.asarray(w_meas) - st.bw          # (37)
        C = st.C
        a_I = C.T @ f_hat + GRAVITY                 # from (3)

        out = st.copy()
        out.r = st.r + dt * st.v + 0.5 * dt * dt * a_I     # (30)
        out.v = st.v + dt * a_I                             # (31)
        out.q = normalize(quat_mul(zeta(dt * w_hat), st.q))  # (32)
        # footholds (33) and biases (34-35) unchanged
        return out, f_hat, w_hat

    # ---------------- M5 — Covariance propagation ----------------
    def build_FQ(self, st: State, f_hat, w_hat, dt, contacts):
        """Discrete error-state transition F and process noise Q (B.7)."""
        p = self.params
        C = st.C
        Ct = C.T
        fx = skew(f_hat)
        I3 = np.eye(3)

        G0 = gamma(w_hat, dt, 0)
        G1 = gamma(w_hat, dt, 1)
        G2 = gamma(w_hat, dt, 2)
        G3 = gamma(w_hat, dt, 3)

        n = ErrorIndex.DIM
        F = np.eye(n)
        # dr rows
        F[ErrorIndex.R, ErrorIndex.V] = dt * I3
        F[ErrorIndex.R, ErrorIndex.PHI] = -0.5 * dt * dt * Ct @ fx
        F[ErrorIndex.R, ErrorIndex.BF] = -0.5 * dt * dt * Ct
        # dv rows
        F[ErrorIndex.V, ErrorIndex.PHI] = -dt * Ct @ fx
        F[ErrorIndex.V, ErrorIndex.BF] = -dt * Ct
        # dphi rows
        F[ErrorIndex.PHI, ErrorIndex.PHI] = G0.T
        F[ErrorIndex.PHI, ErrorIndex.BW] = -G1.T
        # footholds and biases stay identity

        # ---- Q (B.7) ----
        Q = np.zeros((n, n))
        Qf, Qbf, Qw, Qbw = p.Qf, p.Qbf, p.Qw, p.Qbw

        Q[ErrorIndex.R, ErrorIndex.R] = dt ** 3 / 3.0 * Qf + dt ** 5 / 20.0 * Qbf
        Q[ErrorIndex.R, ErrorIndex.V] = dt ** 2 / 2.0 * Qf + dt ** 4 / 8.0 * Qbf
        Q[ErrorIndex.R, ErrorIndex.BF] = -dt ** 3 / 6.0 * Ct @ Qbf
        Q[ErrorIndex.V, ErrorIndex.V] = dt * Qf + dt ** 3 / 3.0 * Qbf
        Q[ErrorIndex.V, ErrorIndex.BF] = -dt ** 2 / 2.0 * Ct @ Qbf
        Q[ErrorIndex.PHI, ErrorIndex.PHI] = dt * Qw + (G3 + G3.T) @ Qbw
        Q[ErrorIndex.PHI, ErrorIndex.BW] = -G2.T @ Qbw
        Q[ErrorIndex.BF, ErrorIndex.BF] = dt * Qbf
        Q[ErrorIndex.BW, ErrorIndex.BW] = dt * Qbw

        for i in range(N_LEGS):
            if contacts[i]:
                Qp = p.Qp_contact()
            else:
                Qp = p.Qp_swing()
            Q[ErrorIndex.p(i), ErrorIndex.p(i)] = dt * Ct @ Qp @ C

        # Mirror the off-diagonal blocks (Q symmetric).
        Q[ErrorIndex.V, ErrorIndex.R] = Q[ErrorIndex.R, ErrorIndex.V].T
        Q[ErrorIndex.BF, ErrorIndex.R] = Q[ErrorIndex.R, ErrorIndex.BF].T
        Q[ErrorIndex.BF, ErrorIndex.V] = Q[ErrorIndex.V, ErrorIndex.BF].T
        Q[ErrorIndex.BW, ErrorIndex.PHI] = Q[ErrorIndex.PHI, ErrorIndex.BW].T

        return F, Q

    def propagate_covariance(self, st: State, F, Q):
        """P- = F P F^T + Q (eq. 45), symmetrised."""
        P = F @ st.P @ F.T + Q
        st.P = 0.5 * (P + P.T)
        return st.P

    def predict(self, f_meas, w_meas, dt, contacts):
        """One IMU-rate prediction step (M4 + M5)."""
        st = self.state
        new_st, f_hat, w_hat = self.propagate_state(st, f_meas, w_meas, dt)
        F, Q = self.build_FQ(st, f_hat, w_hat, dt, contacts)
        if not self.estimate_bias:
            # Freeze biases: no random walk, no cross-coupling growth.
            Q[ErrorIndex.BF, :] = 0.0
            Q[:, ErrorIndex.BF] = 0.0
            Q[ErrorIndex.BW, :] = 0.0
            Q[:, ErrorIndex.BW] = 0.0
        new_st.P = st.P  # carry P into new_st for propagate_covariance
        self.propagate_covariance(new_st, F, Q)
        self.state = new_st
        return new_st

    # ---------------- M6 — Contact transitions ----------------
    def handle_transitions(self, contacts, alpha):
        """Re-anchor footholds on rising edges (touchdown), plan M6."""
        st = self.state
        C = st.C
        contacts = np.asarray(contacts, dtype=bool)
        for i in range(N_LEGS):
            rising = contacts[i] and not self.prev_contacts[i]
            if rising:
                s_i = leg_fk(i, np.asarray(alpha)[i])
                st.set_p(i, st.r + C.T @ s_i)            # plant new landmark
                idx = ErrorIndex.p(i)
                st.P[idx, :] = 0.0                       # forget old correlations
                st.P[:, idx] = 0.0
                st.P[idx, idx] = self.params.sigma_reset ** 2 * np.eye(3)
                self.fej_anchor[i] = st.p(i).copy()      # FEJ anchor (eq. 76)
            elif not contacts[i]:
                self.fej_anchor[i] = None
        self.prev_contacts = contacts.copy()

    # ---------------- M7 — Measurement assembly ----------------
    def build_measurement(self, alpha, contacts):
        """Innovation y, Jacobian H, noise R for legs in contact (eqs. 46-48)."""
        st = self.state
        C = st.C
        alpha = np.asarray(alpha)
        legs = [i for i in range(N_LEGS) if contacts[i]]
        if not legs:
            return legs, np.zeros(0), np.zeros((0, ErrorIndex.DIM)), np.zeros((0, 0))

        p = self.params
        y_blocks = []
        H = np.zeros((3 * len(legs), ErrorIndex.DIM))
        R = np.zeros((3 * len(legs), 3 * len(legs)))
        for row, i in enumerate(legs):
            s_meas = leg_fk(i, alpha[i])                 # (22)
            lever = st.p(i) - st.r
            s_pred = C @ lever                           # predicted foot in body
            y_blocks.append(s_meas - s_pred)             # (46)

            # FEJ: freeze the lever arm inside H at its touchdown anchor (76).
            lever_H = lever
            if self.use_fej and self.fej_anchor[i] is not None:
                lever_H = self.fej_anchor[i] - st.r
            rr = slice(3 * row, 3 * row + 3)
            H[rr, ErrorIndex.R] = -C                     # (47) dr
            H[rr, ErrorIndex.PHI] = skew(C @ lever_H)    # (47) dphi
            H[rr, ErrorIndex.p(i)] = C                   # (47) dp_i

            J = leg_jac(i, alpha[i])
            R[rr, rr] = p.Rs + J @ p.Ralpha @ J.T        # (25)

        return legs, np.concatenate(y_blocks), H, R

    # ---------------- M9 — Outlier gate ----------------
    def gate(self, legs, y, H, R):
        """Drop legs with impossible innovation (Mahalanobis, plan M9)."""
        if not self.use_gate or len(legs) == 0:
            return legs, y, H, R
        st = self.state
        S = H @ st.P @ H.T + R
        keep = []
        for row, i in enumerate(legs):
            rr = slice(3 * row, 3 * row + 3)
            yi = y[rr]
            Sii = S[rr, rr]
            d2 = float(yi @ np.linalg.solve(Sii, yi))
            if d2 <= self.params.mahalanobis_gate:
                keep.append(row)
        if len(keep) == len(legs):
            return legs, y, H, R
        rows = np.concatenate([np.arange(3 * r, 3 * r + 3) for r in keep]) if keep else np.zeros(0, int)
        legs2 = [legs[r] for r in keep]
        return legs2, y[rows], H[rows], R[np.ix_(rows, rows)]

    # ---------------- M8 — Update ----------------
    def update(self, alpha, contacts):
        """Fuse the leg measurement into x, P (eqs. 49-53, Joseph form)."""
        self.handle_transitions(contacts, alpha)
        legs, y, H, R = self.build_measurement(alpha, contacts)
        legs, y, H, R = self.gate(legs, y, H, R)
        if len(legs) == 0:
            return self.state  # flight phase: dead-reckon

        st = self.state
        P = st.P
        S = H @ P @ H.T + R                               # (49)
        K = P @ H.T @ np.linalg.solve(S, np.eye(S.shape[0])).T  # (50): P H^T S^-1
        # (np.linalg.solve(S, I).T == S^-T == S^-1 since S symmetric)
        dx = K @ y                                        # (51)

        if not self.estimate_bias:
            K[ErrorIndex.BF, :] = 0.0
            K[ErrorIndex.BW, :] = 0.0
            dx = K @ y

        self._inject(st, dx)

        IKH = np.eye(ErrorIndex.DIM) - K @ H
        P = IKH @ P @ IKH.T + K @ R @ K.T                 # Joseph form of (52)
        st.P = 0.5 * (P + P.T)
        return st

    @staticmethod
    def _inject(st: State, dx):
        """Additive injection of dx into x, except attitude (multiplicative, 53)."""
        st.r = st.r + dx[ErrorIndex.R]
        st.v = st.v + dx[ErrorIndex.V]
        dphi = dx[ErrorIndex.PHI]
        st.q = normalize(quat_mul(zeta(dphi), st.q))      # (53)
        for i in range(N_LEGS):
            st.set_p(i, st.p(i) + dx[ErrorIndex.p(i)])
        st.bf = st.bf + dx[ErrorIndex.BF]
        st.bw = st.bw + dx[ErrorIndex.BW]

    # ---------------- M10 — switches ----------------
    def set_bias_estimation(self, enable: bool):
        self.estimate_bias = bool(enable)

    def set_fej(self, enable: bool):
        self.use_fej = bool(enable)

    def set_gate(self, enable: bool):
        self.use_gate = bool(enable)

    # ---------------- M11 — Output / health ----------------
    def outputs(self, last_w_meas=None):
        """Controller-facing outputs and health flag (plan M11)."""
        st = self.state
        sigma = np.sqrt(np.clip(np.diag(st.P), 0.0, None))
        w_hat = None
        if last_w_meas is not None:
            w_hat = np.asarray(last_w_meas) - st.bw
        C = st.C
        sigma_v = sigma[ErrorIndex.V]
        finite = bool(np.all(np.isfinite(st.x)) and np.all(np.isfinite(st.P)))
        min_eig = float(np.min(np.linalg.eigvalsh(0.5 * (st.P + st.P.T))))
        healthy = bool(np.all(sigma_v < 0.15) and finite and min_eig > -1.0e-9)
        return {
            "r": st.r.copy(),
            "v": st.v.copy(),
            "q": st.q.copy(),
            "w_hat": w_hat,
            "v_body": C @ st.v,
            "sigma": sigma,
            "min_eig_P": min_eig,
            "healthy": healthy,
        }
