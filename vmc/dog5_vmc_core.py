"""Whole-body Virtual Model Controller (VMC) for DOG5 — platform-agnostic core.

A virtual spring/damper acts on the TRUNK pose (height + attitude) and twist,
producing a body wrench W = [F; M]. The wrench is distributed across the feet
currently in contact (grasp-map / damped least squares) into per-foot ground
reaction forces, and each stance leg realises its share with tau = J^T f. Swing
legs track a Cartesian foot trajectory with their own gravity compensation.

Feedback comes from the proprioceptive state estimator
(`dog5_state_estimator.DOG5StateEstimator.outputs()`), which observes height z,
velocity v, and roll/pitch. Because x, y and yaw are unobservable (they drift),
those axes get **damping only** — never absolute-position stiffness — so the
controller can never chase the estimator's drift.

Frames (all locked to the rest of the repo):
    * trunk/body frame is FLU (x fwd, y left, z up); `dog5_kinematics` lives here
    * estimator quaternion is JPL [x,y,z,w] with C(q) mapping INERTIAL -> BODY
    * the inertial frame I is gravity-aligned (z up); wrench + distribution are
      done in the world/inertial frame (world-up is the clean friction normal),
      then each foot force is rotated into the body frame with C for the J^T map.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np

_DESC = Path(__file__).resolve().parent.parent / "dog5_description"
_EST = Path(__file__).resolve().parent.parent / "state_estimator"
for _p in (_DESC, _EST):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dog5_kinematics  # noqa: E402
from dog5_state_estimator import quat_to_C, skew  # noqa: E402

LEGS = list(dog5_kinematics.LEGS)          # ["FL", "FR", "RL", "RR"]
N_LEGS = len(LEGS)
GRAVITY_MAG = 9.81


# =====================================================================
# Gains and configuration
# =====================================================================

@dataclass
class VMCGains:
    # Trunk virtual spring/damper. Stiffness ONLY on {z, roll, pitch}.
    kp_z: float = 1200.0       # N/m   height stiffness
    kd_z: float = 120.0        # Ns/m
    kp_roll: float = 120.0     # Nm/rad
    kd_roll: float = 8.0       # Nms/rad
    kp_pitch: float = 120.0
    kd_pitch: float = 8.0
    # Damping ONLY on the unobservable axes {x, y, yaw}.
    kd_x: float = 60.0         # Ns/m
    kd_y: float = 60.0
    kd_yaw: float = 4.0        # Nms/rad
    # Swing-leg Cartesian PD (task space, per foot).
    kp_swing: float = 500.0    # N/m
    kd_swing: float = 12.0     # Ns/m
    # Stance joint-space damping (adds robustness, as in stand_dog5.py).
    kd_joint: float = 0.15     # Nms/rad


@dataclass
class VMCConfig:
    z_des: float = 0.20            # desired trunk height (m), overridden by scheduler
    mu: float = 0.6               # friction coefficient (tangential clamp)
    fz_min: float = 1.0           # min per-foot normal force in contact (N)
    lam: float = 1.0e-3           # damped-LS regulariser for the grasp map
    tau_max: float = 8.0          # per-joint torque clamp (Nm), matches stand_dog5
    min_jac_singular: float = 5.0e-3   # leg-Jacobian singularity guard
    com_body: np.ndarray = field(default_factory=lambda: np.zeros(3))  # CoM in body


# =====================================================================
# Attitude (yaw-free) and wrench
# =====================================================================

def attitude_error_rp(C):
    """Roll and pitch (rad) from C (I->B), ignoring yaw.

    g_b is the inertial up-axis expressed in the body frame; a level robot has
    g_b = [0, 0, 1] -> roll = pitch = 0.
    """
    g_b = C @ np.array([0.0, 0.0, 1.0])
    roll = math.atan2(g_b[1], g_b[2])
    pitch = math.atan2(-g_b[0], math.hypot(g_b[1], g_b[2]))
    return roll, pitch


def body_wrench(est_out, z_des, v_cmd_world, yawrate_cmd, gains: VMCGains, mass):
    """Virtual trunk spring/damper -> world-frame wrench W = [F(3); M(3)].

    F is a world-frame force at the CoM; M are world-axis moments (valid for the
    small tilts a level-hold maintains). Height and attitude are stiff; the
    unobservable x, y, yaw axes are damping-only.
    """
    r = np.asarray(est_out["r"])
    v = np.asarray(est_out["v"])
    C = est_out["C"]
    w = np.asarray(est_out["w_hat"])          # bias-corrected body rates
    roll, pitch = attitude_error_rp(C)
    v_cmd = np.asarray(v_cmd_world, dtype=float)

    Fz = gains.kp_z * (z_des - r[2]) + gains.kd_z * (0.0 - v[2]) + mass * GRAVITY_MAG
    Fx = gains.kd_x * (v_cmd[0] - v[0])
    Fy = gains.kd_y * (v_cmd[1] - v[1])
    Mx = gains.kp_roll * (0.0 - roll) + gains.kd_roll * (0.0 - w[0])
    My = gains.kp_pitch * (0.0 - pitch) + gains.kd_pitch * (0.0 - w[1])
    Mz = gains.kd_yaw * (yawrate_cmd - w[2])
    return np.array([Fx, Fy, Fz, Mx, My, Mz])


# =====================================================================
# Grasp map and wrench distribution
# =====================================================================

def grasp_map(foot_pos_body, stance, C, com_body):
    """Assemble G (6 x 3k) mapping stacked world-frame foot forces to the wrench.

    Foot lever arms are taken about the CoM and expressed in the world frame:
    r_i^w = C^T (s_i^body - com_body).  G [F_stack] = [sum F ; sum r x F].
    """
    k = len(stance)
    G = np.zeros((6, 3 * k))
    for j, leg in enumerate(stance):
        i = LEGS.index(leg)
        r_w = C.T @ (foot_pos_body[i] - com_body)
        G[0:3, 3 * j:3 * j + 3] = np.eye(3)
        G[3:6, 3 * j:3 * j + 3] = skew(r_w)
    return G


def distribute_wrench(W, foot_pos_body, stance, C, gains: VMCGains, cfg: VMCConfig):
    """Distribute the body wrench across stance feet -> {leg: force in BODY frame}.

    Damped least squares (minimum-norm) solve of G f = W; for a rank-deficient
    stance (e.g. a 2-foot diagonal, which has no moment authority about the
    support line) this automatically drops the uncontrollable component. A light
    unilateral/friction clamp guards against pathological pulls; for a symmetric
    stand/diagonal it essentially never triggers.
    """
    if not stance:
        return {}
    G = grasp_map(foot_pos_body, stance, C, cfg.com_body)
    # f_world_stack = G^T (G G^T + lam I)^-1 W
    S = G @ G.T + cfg.lam * np.eye(6)
    f_stack = G.T @ np.linalg.solve(S, W)

    forces = {}
    for j, leg in enumerate(stance):
        f_w = f_stack[3 * j:3 * j + 3].copy()
        # unilateral + friction-cone guard (world up = normal)
        if f_w[2] < cfg.fz_min:
            f_w[2] = cfg.fz_min
        t = f_w[:2]
        t_max = cfg.mu * f_w[2]
        t_norm = float(np.linalg.norm(t))
        if t_norm > t_max and t_norm > 1e-9:
            f_w[:2] = t * (t_max / t_norm)
        forces[leg] = C @ f_w        # into body frame for the J^T map
    return forces


# =====================================================================
# Full torque assembly
# =====================================================================

def _leg_slice(i):
    return slice(3 * i, 3 * i + 3)


def compute_vmc_torques(q, qd, est_out, sched, gains: VMCGains, cfg: VMCConfig, mass):
    """Whole-body VMC torque vector (12,) for the current tick.

    Inputs
    ------
    q, qd : (12,) joint angles / rates, order LEGS x [hip_abd, hip_pitch, knee].
    est_out : dict from the estimator's outputs() plus key "C" = quat_to_C(q).
    sched : dict with keys
        contacts : (4,) bool     stance mask (fed identically to the estimator)
        z_des : float
        v_cmd_world : (3,)       linear velocity command (world); 0 for in-place
        yawrate_cmd : float
        swing_targets : {leg: (p_des(3), v_des(3))}  body-frame foot targets
    mass : total body mass (kg) — pass model.body_mass.sum(), never hard-code.

    Returns (tau(12), diag_dict).
    """
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    contacts = np.asarray(sched["contacts"], dtype=bool)
    stance = [LEGS[i] for i in range(N_LEGS) if contacts[i]]
    swing_targets = sched.get("swing_targets", {})

    # Foot positions (body frame) for every leg — needed for the grasp map.
    foot_pos_body = [dog5_kinematics.foot_position(LEGS[i], q[_leg_slice(i)])
                     for i in range(N_LEGS)]

    W = body_wrench(est_out, sched["z_des"], sched.get("v_cmd_world", np.zeros(3)),
                    sched.get("yawrate_cmd", 0.0), gains, mass)
    forces = distribute_wrench(W, foot_pos_body, stance, est_out["C"], gains, cfg)

    tau = np.zeros(12)
    singular = []
    for i in range(N_LEGS):
        leg = LEGS[i]
        q_i = q[_leg_slice(i)]
        qd_i = qd[_leg_slice(i)]
        J = dog5_kinematics.foot_jacobian(leg, q_i)
        smin = float(np.linalg.svd(J, compute_uv=False)[-1])
        if smin < cfg.min_jac_singular:
            singular.append(leg)

        if contacts[i]:
            # stance: realise the distributed ground reaction. `forces[leg]` is
            # the reaction ON the body (up, +z); tau = J^T F needs F = the force
            # the foot exerts ON the ground (down) = -reaction, so the leg pushes
            # the foot into the floor and the reaction carries the body. NO
            # gravity torque here — the m*g term in the wrench already routes
            # total weight to the feet (double-count guard, matches stand_dog5).
            tau_i = -J.T @ forces[leg] - gains.kd_joint * qd_i
        else:
            # swing: track the foot trajectory + carry the leg's own weight.
            p_des, v_des = swing_targets.get(
                leg, (foot_pos_body[i], np.zeros(3)))
            p = foot_pos_body[i]
            v_foot = J @ qd_i
            F = gains.kp_swing * (np.asarray(p_des) - p) + \
                gains.kd_swing * (np.asarray(v_des) - v_foot)
            tau_i = J.T @ F + dog5_kinematics.leg_gravity_torque(leg, q_i)

        tau[_leg_slice(i)] = tau_i

    tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)
    diag = {
        "W": W,
        "forces": forces,
        "stance": stance,
        "singular_legs": singular,
        "roll_pitch": attitude_error_rp(est_out["C"]),
    }
    return tau, diag
