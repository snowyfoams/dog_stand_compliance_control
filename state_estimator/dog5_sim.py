"""M12 — simulated data generator for the DOG5 state estimator (plan M12).

Produces ground-truth trajectories with known noise and known biases, plus the
synthetic IMU / encoder / contact streams a real robot would emit.  This is the
only environment in which "correct" is defined exactly, so it exists before the
filter is trusted.

Critical construction detail (plan M12): stance feet are genuinely stationary
in the inertial frame.  Joint angles for a stance leg are synthesised by IK
from the (moving) body to the (fixed) inertial foot position.  A foot that
slid along with the body would violate the filter's core assumption and
manufacture fake velocity error.

Truth is generated with the *same* discrete integrator the filter uses (eqs.
30-32), so a noiseless / unbiased / perfectly-initialised run reproduces truth
to machine precision — any residual error is then unambiguously a filter bug.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from dog5_state_estimator import (
    GRAVITY,
    LEGS,
    N_LEGS,
    leg_fk,
    leg_jac,
    normalize,
    quat_mul,
    quat_to_C,
    zeta,
)

# Nominal stance pose (valid within joint limits; feet non-colinear).  Copied
# from check_dog5_kinematics.Q_CROUCH.
Q_STANCE = {
    "FL": np.array([1.219, -0.796, -2.328]),
    "FR": np.array([-1.219, 0.796, 2.345]),
    "RL": np.array([1.219, 0.796, 2.345]),
    "RR": np.array([-1.219, -0.796, -2.328]),
}


class UnreachableFootError(RuntimeError):
    """Raised when a scenario asks a foot to go outside the leg workspace.

    Silently returning a bad IK solution would manufacture huge fake encoder
    errors — the exact failure the plan's M12 warns about (a foot that "slides"
    or teleports).  Scenarios must keep every foot reachable.
    """


def leg_ik(leg_index, target_body, q_start, max_iter=200, tol=1.0e-10,
           strict=False, hard_tol=1.0e-4):
    """Damped-least-squares IK: joint angles placing the foot at target_body.

    Same law as crawl_hw2.0/stand3_hold_hw.py, tightened for offline use.  With
    ``strict=True`` an unconverged solve (residual > ``hard_tol``) raises
    ``UnreachableFootError`` instead of returning garbage.
    """
    q = np.asarray(q_start, dtype=float).copy()
    err = np.asarray(target_body) - leg_fk(leg_index, q)
    for _ in range(max_iter):
        if float(np.linalg.norm(err)) < tol:
            break
        J = leg_jac(leg_index, q)
        q = q + J.T @ np.linalg.solve(J @ J.T + 1.0e-8 * np.eye(3), err)
        err = np.asarray(target_body) - leg_fk(leg_index, q)
    if strict and float(np.linalg.norm(err)) > hard_tol:
        raise UnreachableFootError(
            f"leg {LEGS[leg_index]} target {np.round(target_body, 3)} "
            f"unreachable (residual {np.linalg.norm(err)*1e3:.1f} mm)"
        )
    return q


@dataclass
class SimConfig:
    dt: float = 1.0 / 400.0
    duration: float = 2.0
    seed: int = 0
    # discrete measurement-noise std (per sample).  0 => noiseless.
    noise_f: float = 0.0        # m/s^2
    noise_w: float = 0.0        # rad/s
    noise_alpha: float = 0.0    # rad
    bias_f: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bias_w: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class SimData:
    t: np.ndarray
    f_meas: np.ndarray          # (K, 3) specific force, body frame
    w_meas: np.ndarray          # (K, 3) angular rate, body frame
    alpha: np.ndarray           # (K, 4, 3) joint angles
    contacts: np.ndarray        # (K, 4) bool
    # truth
    r: np.ndarray               # (K, 3)
    v: np.ndarray               # (K, 3)
    q: np.ndarray               # (K, 4)
    bias_f: np.ndarray          # (3,)
    bias_w: np.ndarray          # (3,)
    foot_I: np.ndarray          # (K, 4, 3) truth inertial foot positions

    @property
    def n(self):
        return len(self.t)


def _integrate_truth(accel_I, omega_B, r0, v0, q0, dt):
    """Integrate the discrete nominal model (eqs. 30-32) given driving inputs.

    accel_I[k] is absolute acceleration in I applied over step k; omega_B[k]
    the body rate over step k.  Returns r, v, q arrays of length K+1... we
    return length K aligned so index k is the state *at* sample k.
    """
    K = len(accel_I)
    r = np.zeros((K, 3))
    v = np.zeros((K, 3))
    q = np.zeros((K, 4))
    r[0], v[0], q[0] = r0, v0, normalize(q0)
    for k in range(K - 1):
        a = accel_I[k]
        w = omega_B[k]
        r[k + 1] = r[k] + dt * v[k] + 0.5 * dt * dt * a
        v[k + 1] = v[k] + dt * a
        q[k + 1] = normalize(quat_mul(zeta(dt * w), q[k]))
    return r, v, q


def _foot_positions_body(alpha_row):
    return np.stack([leg_fk(i, alpha_row[i]) for i in range(N_LEGS)], axis=0)


def _initial_feet_inertial(r0, q0, alpha0):
    """Inertial positions of the four feet at the initial pose."""
    C0 = quat_to_C(q0)
    feet = np.zeros((N_LEGS, 3))
    for i in range(N_LEGS):
        s_i = leg_fk(i, alpha0[i])
        feet[i] = r0 + C0.T @ s_i
    return feet


def _synthesize(cfg, accel_I, omega_B, foot_I_traj, contacts, encoders=True):
    """Common back-end: given truth base drivers and inertial foot trajectories,
    produce IMU + encoder streams.  foot_I_traj is (K, 4, 3); contacts (K, 4).

    ``encoders=False`` skips the (expensive) per-sample IK and fills the joint
    angles with the nominal stance — for propagation-only tests that never read
    the encoders.
    """
    dt = cfg.dt
    K = len(accel_I)
    alpha0 = np.stack([Q_STANCE[leg] for leg in LEGS], axis=0)

    r0 = np.zeros(3)
    v0 = np.zeros(3)
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    r, v, q = _integrate_truth(accel_I, omega_B, r0, v0, q0, dt)

    rng = np.random.default_rng(cfg.seed)

    f_meas = np.zeros((K, 3))
    w_meas = np.zeros((K, 3))
    alpha = np.zeros((K, N_LEGS, 3))
    prev_alpha = alpha0.copy()
    for k in range(K):
        C = quat_to_C(q[k])
        # specific force truth (eq. 3): f = C (a - g)
        f_true = C @ (accel_I[k] - GRAVITY)
        f_meas[k] = f_true + cfg.bias_f + cfg.noise_f * rng.standard_normal(3)
        w_meas[k] = omega_B[k] + cfg.bias_w + cfg.noise_w * rng.standard_normal(3)
        if not encoders:
            alpha[k] = alpha0
            continue
        # encoders: IK each foot from body to its inertial target.  strict=True
        # turns an out-of-reach scenario into a loud error instead of silent
        # garbage encoders (plan M12).
        for i in range(N_LEGS):
            target_body = C @ (foot_I_traj[k, i] - r[k])
            a_i = leg_ik(i, target_body, prev_alpha[i], strict=True)
            prev_alpha[i] = a_i
            alpha[k, i] = a_i + cfg.noise_alpha * rng.standard_normal(3)

    return SimData(
        t=np.arange(K) * dt,
        f_meas=f_meas,
        w_meas=w_meas,
        alpha=alpha,
        contacts=contacts.astype(bool),
        r=r,
        v=v,
        q=q,
        bias_f=np.asarray(cfg.bias_f, dtype=float),
        bias_w=np.asarray(cfg.bias_w, dtype=float),
        foot_I=foot_I_traj,
    )


# ---------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------

def scenario_imu_only(cfg: SimConfig) -> SimData:
    """Propagation-only truth: an oscillating specific force plus a constant
    body rotation, feet planted but encoders NOT synthesised.  Exercises the
    full M4 integrator (accel + rotation) for the dead-reckoning gate C3
    without the leg-reach limits of a long planted-foot translation.
    """
    K = int(round(cfg.duration / cfg.dt))
    t = np.arange(K) * cfg.dt
    accel_I = np.zeros((K, 3))
    accel_I[:, 0] = 0.3 * np.sin(2 * math.pi * 0.5 * t)
    accel_I[:, 1] = 0.2 * np.cos(2 * math.pi * 0.3 * t)
    accel_I[:, 2] = 0.1 * np.sin(2 * math.pi * 0.7 * t)
    omega_B = np.zeros((K, 3))
    omega_B[:] = np.array([0.05, -0.03, 0.08])   # constant body rate
    foot_I = np.zeros((K, N_LEGS, 3))
    contacts = np.zeros((K, N_LEGS), dtype=bool)
    return _synthesize(cfg, accel_I, omega_B, foot_I, contacts, encoders=False)


def scenario_static(cfg: SimConfig) -> SimData:
    """(a) Static hold: no motion, all four feet planted."""
    K = int(round(cfg.duration / cfg.dt))
    accel_I = np.zeros((K, 3))
    omega_B = np.zeros((K, 3))
    alpha0 = np.stack([Q_STANCE[leg] for leg in LEGS], axis=0)
    feet0 = _initial_feet_inertial(np.zeros(3), np.array([0, 0, 0, 1.0]), alpha0)
    foot_I = np.broadcast_to(feet0, (K, N_LEGS, 3)).copy()
    contacts = np.ones((K, N_LEGS), dtype=bool)
    return _synthesize(cfg, accel_I, omega_B, foot_I, contacts)


def scenario_yaw(cfg: SimConfig, rate_dps: float = 15.0) -> SimData:
    """(b) Pure yaw rotation in place, all feet planted (fixed in I)."""
    K = int(round(cfg.duration / cfg.dt))
    rate = math.radians(rate_dps)
    accel_I = np.zeros((K, 3))
    omega_B = np.tile(np.array([0.0, 0.0, rate]), (K, 1))
    alpha0 = np.stack([Q_STANCE[leg] for leg in LEGS], axis=0)
    feet0 = _initial_feet_inertial(np.zeros(3), np.array([0, 0, 0, 1.0]), alpha0)
    foot_I = np.broadcast_to(feet0, (K, N_LEGS, 3)).copy()
    contacts = np.ones((K, N_LEGS), dtype=bool)
    return _synthesize(cfg, accel_I, omega_B, foot_I, contacts)


def scenario_translate(cfg: SimConfig, speed: float = 0.05) -> SimData:
    """(c) Forward translation at constant velocity, feet planted.

    Accelerate smoothly to `speed` over the first 0.3 s, then cruise, so the
    velocity is genuinely nonzero and observable through the planted feet.
    """
    K = int(round(cfg.duration / cfg.dt))
    dt = cfg.dt
    t = np.arange(K) * dt
    t_ramp = 0.3
    # Piecewise-constant accel during a short ramp, then cruise at `speed`.
    a_ramp = speed / t_ramp
    accel_I = np.zeros((K, 3))
    accel_I[t < t_ramp, 0] = a_ramp
    omega_B = np.zeros((K, 3))
    alpha0 = np.stack([Q_STANCE[leg] for leg in LEGS], axis=0)
    feet0 = _initial_feet_inertial(np.zeros(3), np.array([0, 0, 0, 1.0]), alpha0)
    foot_I = np.broadcast_to(feet0, (K, N_LEGS, 3)).copy()
    contacts = np.ones((K, N_LEGS), dtype=bool)
    return _synthesize(cfg, accel_I, omega_B, foot_I, contacts)


def scenario_excite(cfg: SimConfig, amp_deg: float = 5.0,
                    freq: float = 0.7, settle_s: float = 1.0) -> SimData:
    """Startup excitation: body rolls/pitches +/- amp_deg with feet planted.

    This is the deliberate manoeuvre of plan M10 / Table I: a nonzero omega
    lifts the observability rank so the accelerometer bias separates from the
    tilt (which is impossible at rest).  Used to validate bias estimation.

    The first ``settle_s`` seconds are held static so the estimator's
    static-alignment init (M3) has a valid stationary window to work with,
    exactly as a real bring-up sequence would.
    """
    K = int(round(cfg.duration / cfg.dt))
    dt = cfg.dt
    t = np.arange(K) * dt
    amp = math.radians(amp_deg)
    w = 2 * math.pi * freq
    # Body angular rate: roll and pitch oscillate out of phase after a static
    # settle window; base does not translate (a = 0), feet stay planted.
    moving = (t >= settle_s).astype(float)
    tm = np.maximum(t - settle_s, 0.0)
    omega_B = np.zeros((K, 3))
    omega_B[:, 0] = moving * amp * w * np.cos(w * tm)
    omega_B[:, 1] = moving * amp * w * np.cos(w * tm + math.pi / 2)
    accel_I = np.zeros((K, 3))
    alpha0 = np.stack([Q_STANCE[leg] for leg in LEGS], axis=0)
    feet0 = _initial_feet_inertial(np.zeros(3), np.array([0, 0, 0, 1.0]), alpha0)
    foot_I = np.broadcast_to(feet0, (K, N_LEGS, 3)).copy()
    contacts = np.ones((K, N_LEGS), dtype=bool)
    return _synthesize(cfg, accel_I, omega_B, foot_I, contacts)


def scenario_trot(cfg: SimConfig, speed: float = 0.04,
                  period: float = 0.6, step_len: float | None = None,
                  lift: float = 0.03) -> SimData:
    """(d) Trot: diagonal pairs alternate stance / swing while the body creeps
    forward.  Stance feet are fixed in I; swing feet arc forward to a new plant.

    Diagonal pair A = {FL, RR}, pair B = {FR, RL}.  During a stance half-period
    a pair is planted; during its swing half-period it lifts and moves forward
    by `step_len`, replanting at the end.
    """
    K = int(round(cfg.duration / cfg.dt))
    dt = cfg.dt
    t = np.arange(K) * dt
    if step_len is None:
        # Match the average foot advance to the body cruise so the planted feet
        # neither run ahead of nor fall behind the body (body-frame offset stays
        # bounded, within leg reach): each foot steps once per period.
        step_len = speed * period

    # Body: gentle forward cruise (const velocity after a short ramp).
    t_ramp = 0.3
    a_ramp = speed / t_ramp
    accel_I = np.zeros((K, 3))
    accel_I[t < t_ramp, 0] = a_ramp
    omega_B = np.zeros((K, 3))

    r0 = np.zeros(3)
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    _r, _v, _q = _integrate_truth(accel_I, omega_B, r0, np.zeros(3), q0, dt)

    alpha0 = np.stack([Q_STANCE[leg] for leg in LEGS], axis=0)
    feet0 = _initial_feet_inertial(r0, q0, alpha0)

    pair = {0: (0, 3), 1: (1, 2)}  # A={FL,RR}, B={FR,RL} by LEG index
    half = period / 2.0

    foot_I = np.zeros((K, N_LEGS, 3))
    contacts = np.ones((K, N_LEGS), dtype=bool)
    planted = feet0.copy()
    prev_swinging = np.zeros(N_LEGS, dtype=bool)  # everyone starts in stance

    for k in range(K):
        phase = t[k] % period
        swing_pair = 0 if phase < half else 1
        for i in range(N_LEGS):
            swinging = i in pair[swing_pair]
            if swinging:
                u = (phase / half) if swing_pair == 0 else ((phase - half) / half)
                u = min(max(u, 0.0), 1.0)
                start = planted[i]                       # swing begins at old plant
                foot_I[k, i] = np.array([
                    start[0] + step_len * u,
                    start[1],
                    start[2] + lift * math.sin(math.pi * u),
                ])
                contacts[k, i] = False
            else:
                # Commit the new plant exactly at the swing->stance transition,
                # so the inertial foot trajectory is continuous (no ~cm jump).
                if prev_swinging[i]:
                    planted[i] = planted[i] + np.array([step_len, 0.0, 0.0])
                foot_I[k, i] = planted[i]
                contacts[k, i] = True
            prev_swinging[i] = swinging

    return _synthesize(cfg, accel_I, omega_B, foot_I, contacts)
