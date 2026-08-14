#!/usr/bin/env python3
"""The torque law, split by cost: what runs at 100 Hz and what runs at 250 Hz.

WHY THE SPLIT
    Measured on this Pi, per 4-leg sweep:
        dog5_vmc_core.compute_vmc_torques      998 us
        the corrected stance law (dog5_statics) 822 us
    Either is 3-4x over the 333 us CAN slot, so the wrench and the grasp map
    can never run in the sweep.  But the terms that actually STABILISE a
    joint are three array ops, and they want the freshest qd they can get.
    So:

        stance worker, 100 Hz   wrench -> GRF -> -J^T f + leg gravity
                                publishes tau_ff and q_ref
        CAN sweep,     250 Hz   tau = tau_ff + kp(q_ref - q) - kd qd
                                -> TorqueSafetyGate -> RunawayBrake -> bus

    vmc_stand_hw fused both into one 100 Hz thread, so the CAN loop held a
    torque up to 10 ms stale.  Here the feedback terms run on qd that is at
    most one 4 ms sweep old.  That is the entire benefit of the corrected
    loop rate, spent where it buys stability.

WHAT IS HERE
    JointImpedance      the 250 Hz floor.  The reason no joint can run away.
    TorqueSafetyGate    base.SafetyGate with torque-mode ramp and slew
    compute_stance      the 100 Hz half: wrench, distribution, corrected tau

RUN
    A library: no bus, no IMU.  Constructing JointImpedance with gains past
    the measured stable envelope raises rather than clipping, so a bad CLI
    value fails before the motors are armed.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    cd /home/robot01/Documents/can_motor_control/dog_stand_compliance_control/Aug_trotinplace_closedloop

    $V self-test/test_stance_law.py        # 41 gates, no hardware
      # [1] simulates the real sampled loop (zero-order hold + one sweep of
      # measurement delay) and asserts the gains are stable at the true 4 ms
      # AND divergent at the 48 ms the 2026-07-30 verdict assumed.
      # [2] reproduces the RR_abd runaway and shows the impedance bounding it.

    On hardware, this layer is what the three runners are made of:
    $V torque_mode_control/torque_hold_hw.py  --kp 15 --kd 0.6   # impedance only
    $V torque_mode_control/torque_stand_hw.py --force-frac 0     # + q_ref
    $V torque_mode_control/torque_stand_hw.py --force-frac 1     # + force law
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUG = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_AUG)
_DESC = os.path.join(_ROOT, "dog5_description")
_VMC = os.path.join(_ROOT, "vmc")
for _p in (_HERE, _DESC, _VMC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as _np                                        # noqa: E402,F401
import dog5_statics as st                                  # noqa: E402
import dog5_vmc_core as vmc                                # noqa: E402
import stand_dog5_hw as base                               # noqa: E402
import torque_params as P                                  # noqa: E402

LEGS = st.LEGS
N_LEGS = len(LEGS)
N_JOINTS = P.N_JOINTS


def _leg_slice(i):
    return slice(3 * i, 3 * i + 3)


# ===========================================================================
# the 250 Hz floor
# ===========================================================================

class JointImpedance:
    """tau = kp (q_ref - q) - kd qd, evaluated in the CAN sweep.

    THE POINT, stated once because it is the whole safety argument:

        A pure force law tau = -J^T f is a VELOCITY-level system.  Take the
        ground away from a foot and there is nothing left holding that joint
        anywhere -- qdd integrates without bound, and a damper only sets the
        terminal speed qd = tau/kd.  The joint does not stop, it coasts.  On
        2026-07-30 RR_abd did exactly that: it settled at tau/0.4 = 1.87 rad/s
        and the run ended on the overspeed e-stop.

        kp (q_ref - q) makes it a POSITION-level system with a fixed point.
        The same fault becomes a bounded offset dq = tau/kp -- 13 mrad at the
        corrected abduction torque and kp = 20.  There is no runaway left for
        a brake to catch.

    It is NOT a position servo fighting the balance loop.  q_ref is the IK of
    the same foot trajectory the wrench loop is trying to realise, so in
    nominal stance this term is ~0 and the force law does the work.  It only
    develops torque when the force law's premise ("this foot is loaded") is
    false.  A model-mismatch term, not a stiffness.

    Gains are bounded by the sampled-PD limit at the TRUE 4 ms sweep, not the
    48 ms that the abandoned attempt assumed -- see torque_params' header.
    """

    def __init__(self, kp=P.KP_JOINT_IMP, kd=P.KD_JOINT_IMP):
        if not 0.0 <= kp <= P.KP_IMP_MAX:
            raise ValueError(
                f"kp={kp} outside the measured stable envelope "
                f"[0, {P.KP_IMP_MAX}] at a {P.SWEEP_S*1e3:.0f} ms sweep")
        if not 0.0 <= kd <= P.KD_IMP_MAX:
            raise ValueError(
                f"kd={kd} outside [0, {P.KD_IMP_MAX}]; the sampled-damper "
                f"bound is 2J/dt = {2*P.J_MIN/P.SWEEP_S:.1f} Nms/rad")
        self.kp = float(kp)
        self.kd = float(kd)
        self.dq = np.zeros(N_JOINTS)          # last error, for the status line

    def tau(self, q, qd, q_ref):
        """The floor torque.  Three array ops -- this is the in-sweep budget."""
        self.dq = np.asarray(q_ref, dtype=float) - np.asarray(q, dtype=float)
        return self.kp * self.dq - self.kd * np.asarray(qd, dtype=float)

    def worst_dq(self):
        """max |q_ref - q| (rad).  Large here means the impedance is carrying
        load the force law believes is on the ground -- i.e. a foot is not
        where the model thinks it is."""
        return float(np.max(np.abs(self.dq))) if self.dq.size else 0.0


class TorqueSafetyGate(base.SafetyGate):
    """base.SafetyGate with the torque-mode ramp and slew.

    base.apply() reads the MODULE-level TORQUE_RAMP_S (1.0 s) and
    TAU_SLEW_NM_S (5.0 Nm/s), which are position-track numbers and are wrong
    here in both directions:

      * the slew is wall-clock, so 5 Nm/s caps a disturbance response at
        0.25 Nm per 50 ms.  A kp=15 impedance at 0.1 rad of error is asking
        for 1.5 Nm and would take 0.3 s to get there -- by which time the leg
        has folded.
      * with a 1.0 s ramp on top, no leg can take weight for a full second
        after the gate starts.

    Overridden by subclass rather than by editing stand_dog5_hw, which the
    position track, the crawl and the EKF harness all import (see
    CONTROL_ROADMAP.md's standing constraints).  Everything else -- the hard
    clip, the directional joint-limit zeroing, the two-tier overspeed, the
    temp/miss/fault e-stop tests -- is inherited untouched.
    """

    def __init__(self, tau_cap, qd_estop=base.QD_ESTOP,
                 qd_estop_hard=base.QD_ESTOP_HARD,
                 ramp_s=P.TORQUE_RAMP_S, slew_nm_s=P.TAU_SLEW_NM_S):
        super().__init__(tau_cap, qd_estop, qd_estop_hard)
        self.ramp_s = float(ramp_s)
        self.slew_nm_s = float(slew_nm_s)

    def cap_now(self, now):
        fraction = np.clip((now - self.started_at) / self.ramp_s, 0.0, 1.0)
        return self.tau_cap * fraction

    def apply(self, tau, q, now):
        cap = self.cap_now(now)
        limited = np.clip(np.asarray(tau, dtype=float), -cap, cap)
        limited = np.clip(limited, -P.TAU_HARD_NM, P.TAU_HARD_NM)
        limited = np.where((q >= self.high) & (limited > 0.0), 0.0, limited)
        limited = np.where((q <= self.low) & (limited < 0.0), 0.0, limited)

        dt = np.clip(now - self.last_time, 1.0e-4, 0.05)
        max_change = self.slew_nm_s * dt
        output = self.previous_tau + np.clip(
            limited - self.previous_tau, -max_change, max_change)
        # a directional limit block is immediate, even where the slew limiter
        # would otherwise leave torque pointing further out of bounds
        output = np.where((q >= self.high) & (output > 0.0), 0.0, output)
        output = np.where((q <= self.low) & (output < 0.0), 0.0, output)
        self.previous_tau = output
        self.last_time = now
        return output


# ===========================================================================
# the 100 Hz half
# ===========================================================================

def gravity_stack(q, C=None, frames=None):
    """Stacked (12,) leg-gravity torque, tilt-aware.

    `dog5_description/crawl_dog5_hw.leg_gravity_stack` does the same thing for
    a level trunk; this one takes the estimator's C so a leveling stand's
    permanent mount tilt is not silently ignored.  At 8 deg of roll that is
    worth 0.05 Nm on abduction -- 11% of the term.
    """
    q = np.asarray(q, dtype=float)
    g_down = None if C is None else st.gravity_down_body(C)
    out = np.zeros(N_JOINTS)
    for i, leg in enumerate(LEGS):
        fr = None if frames is None else frames[i]
        out[_leg_slice(i)] = st.leg_gravity_torque_tilted(
            leg, q[_leg_slice(i)], g_down, frames=fr)
    return out


def compute_stance(q, qd, est_out, sched, gains, cfg, mass,
                   leg_gravity=P.STANCE_LEG_GRAVITY,
                   force_frac=P.FORCE_FRAC_DEFAULT):
    """Feedforward stance torque for the whole robot.  Runs in the worker.

    Reuses `dog5_vmc_core` for everything above the leg -- `body_wrench`,
    `grasp_map`, `distribute_wrench` are sim-proven (gates V1-V4) and are
    imported unchanged.  Only the stance branch is replaced, and only to add
    the term dog5_vmc_core.py:224-227 omits.

    `force_frac` scales the distributed GRF -- the only term that holds the
    TRUNK up.  Leg gravity is deliberately NOT scaled by it: an unloaded leg
    still has to hold its own links up.

        1.0  full VMC, impedance only as a floor.  The normal setting.
        0.0  no body-support force at all.  With the feet ON THE FLOOR this
             is NOT a benign "plumbing test" -- the trunk is then carried
             solely by the impedance's positional error, so the legs fold
             until kp*dq balances 57 N and the brake latches on the way down
             (hardware 2026-08-14).  Safe only with the feet in the air.

    Returns (tau_ff(12,), diag).
    """
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    contacts = np.asarray(sched["contacts"], dtype=bool)
    stance = [LEGS[i] for i in range(N_LEGS) if contacts[i]]

    frames = [st.leg_frames(LEGS[i], q[_leg_slice(i)]) for i in range(N_LEGS)]
    foot_pos_body = [fr[0] for fr in frames]

    if cfg.com_body is not None and P.CONFIG_DEPENDENT_COM:
        cfg.com_body = st.com_body(q.reshape(4, 3), frames_all=frames)

    W = vmc.body_wrench(est_out, sched["z_des"],
                        sched.get("v_cmd_world", np.zeros(3)),
                        sched.get("yawrate_cmd", 0.0), gains, mass)
    forces = vmc.distribute_wrench(W, foot_pos_body, stance, est_out["C"],
                                   gains, cfg)

    g_down = st.gravity_down_body(est_out["C"])
    tau = np.zeros(N_JOINTS)
    singular = []
    for i in range(N_LEGS):
        leg = LEGS[i]
        q_i = q[_leg_slice(i)]
        foot, anchors, axes, _ = frames[i]
        J = st.foot_jacobian_from(foot, anchors, axes)
        if float(np.linalg.svd(J, compute_uv=False)[-1]) < cfg.min_jac_singular:
            singular.append(leg)

        tau_i = np.zeros(3)
        if contacts[i] and leg in forces:
            # forces[leg] is the reaction ON THE BODY (up); the foot pushes
            # DOWN on the ground with its negative, hence the sign.  This part
            # matches dog5_vmc_core.py:227 and was never the bug.
            tau_i = -J.T @ (force_frac * forces[leg])
            tau_i -= gains.kd_joint * qd[_leg_slice(i)]
        if leg_gravity:
            # the omitted internal term -- see dog5_statics' header
            tau_i += st.leg_gravity_torque_tilted(leg, q_i, g_down,
                                                  frames=frames[i])
        tau[_leg_slice(i)] = tau_i

    diag = {
        "W": W,
        "forces": forces,
        "stance": stance,
        "singular_legs": singular,
        "roll_pitch": vmc.attitude_error_rp(est_out["C"]),
        "com_body": cfg.com_body,
    }
    return tau, diag


def default_gains():
    """VMCGains carrying torque_params' values, not the sim-tuned defaults."""
    g = vmc.VMCGains()
    g.kp_z, g.kd_z = P.KP_Z, P.KD_Z
    g.kp_roll, g.kd_roll = P.KP_ROLL, P.KD_ROLL
    g.kp_pitch, g.kd_pitch = P.KP_PITCH, P.KD_PITCH
    g.kd_x, g.kd_y, g.kd_yaw = P.KD_X, P.KD_Y, P.KD_YAW
    g.kd_joint = P.KD_JOINT_VMC
    return g


def default_cfg(stand_height=P.STAND_HEIGHT_DEFAULT, tau_max=P.TAU_START_MAX):
    c = vmc.VMCConfig()
    c.z_des = stand_height
    c.mu = P.MU_FRICTION
    c.fz_min = P.FZ_MIN_N
    c.lam = P.GRASP_LAMBDA
    c.tau_max = tau_max
    c.min_jac_singular = P.MIN_JAC_SINGULAR
    c.com_body = np.zeros(3)
    return c
