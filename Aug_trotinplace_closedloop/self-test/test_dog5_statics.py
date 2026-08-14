#!/usr/bin/env python3
"""Gates for dog5_statics.py -- the corrected quasi-static model.

WHAT IS UNDER TEST
    One claim, from four directions: that `-J^T f + leg_gravity` is the EXACT
    rigid-body stance torque for a static pose, and that `-J^T f` alone (what
    dog5_vmc_core.py:224-227 commands today) is not.

    [1] the model literals agree with dog5.xml -- the DOG5_MASS_KG = 5.3 class
        of bug, which is still live in stand_dog5_hw.py:106
    [2] the fused chain walk agrees with dog5_kinematics, so `leg_frames` is a
        speed optimisation and not a second implementation that can drift
    [3] the level-trunk identity: leg_gravity_torque_tilted reduces EXACTLY to
        the stock dog5_kinematics.leg_gravity_torque.  This is what lets the
        old callers keep the old function while we generalise
    [4] MuJoCo floating-base inverse dynamics as ground truth -- and, the gate
        that actually protects the fix, that REMOVING the leg-gravity term
        makes the error large again.  If someone "simplifies" stance_torque
        back to -J^T f, test_leg_gravity_is_load_bearing goes red.

Run:  $V test_dog5_statics.py
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TORQUE = os.path.join(os.path.dirname(_HERE), "torque_mode_control")
_DESC = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                     "dog5_description")
for _p in (_HERE, _TORQUE, _DESC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from selftest_common import check, report, C_from_rp, SLOT_BUDGET_S  # noqa: E402

import dog5_kinematics as kin                            # noqa: E402
import dog5_statics as st                                # noqa: E402
import torque_params as P                                # noqa: E402

# The recorded hardware crouch (stand_dog5_recorded_hw.Q_RECORDED_CROUCH) and
# a stand pose reached from it.  Real poses matter here: a random pose can put
# the abduction axis somewhere the leg never goes, and the whole point is the
# size of the term at poses the robot actually holds.
Q_CROUCH = np.deg2rad(np.array([
    [90.74, 47.59, -131.02],
    [-90.39, -51.92, 138.39],
    [90.23, -44.45, 128.40],
    [-93.95, 47.06, -125.12],
]))


def _random_poses(n, seed=0):
    """Poses spanning the reachable range, plus the recorded crouch."""
    rng = np.random.default_rng(seed)
    out = [Q_CROUCH]
    for _ in range(n - 1):
        out.append(np.column_stack([
            rng.uniform(-1.75, 1.75, 4),      # abd, the soft limit
            rng.uniform(-2.6, 2.6, 4),        # pitch
            rng.uniform(-2.6, 2.6, 4),        # knee
        ]))
    return out


def _tilts():
    """(roll, pitch) in radians -- level, plus the range a stand ever holds."""
    return [(0.0, 0.0), (0.14, 0.0), (0.0, -0.09), (0.10, 0.06), (-0.05, 0.12)]


# ===========================================================================
# [1] the literals agree with dog5.xml
# ===========================================================================

def test_mass():
    m_model = st.model_total_mass()
    check("total_mass equals MjModel.body_mass.sum()",
          abs(st.total_mass() - m_model) < 1e-9,
          f"{st.total_mass():.8f} vs {m_model:.8f} kg")
    check("torque_params.DOG5_MASS_KG agrees with the model",
          abs(P.DOG5_MASS_KG - m_model) < 1e-3,
          f"{P.DOG5_MASS_KG} vs {m_model:.4f} kg")
    # the number this whole file exists to keep out of the torque path
    import stand_dog5_hw as base                          # noqa: PLC0415
    check("stand_dog5_hw.DOG5_MASS_KG is NOT used here (it is 8.9% low)",
          abs(base.DOG5_MASS_KG - m_model) > 0.3
          and abs(P.DOG5_MASS_KG - base.DOG5_MASS_KG) > 0.3,
          f"base has {base.DOG5_MASS_KG}, we use {P.DOG5_MASS_KG}")
    check("legs are the majority of the mass (why leg gravity matters)",
          0.5 < 4 * st.leg_mass() / st.total_mass() < 0.6,
          f"{100 * 4 * st.leg_mass() / st.total_mass():.1f}%")
    check("WEIGHT_N and PER_FOOT_GRF_N are consistent with the mass",
          abs(P.WEIGHT_N - P.DOG5_MASS_KG * P.GRAVITY_M_S2) < 0.02
          and abs(P.PER_FOOT_GRF_N - P.WEIGHT_N / 4) < 0.02,
          f"{P.WEIGHT_N:.2f} N, {P.PER_FOOT_GRF_N:.2f} N/foot")


def test_trunk_inertial_matches_xml():
    import mujoco                                         # noqa: PLC0415
    model = mujoco.MjModel.from_xml_path(os.path.join(_DESC, "dog5.xml"))
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    check("TRUNK_MASS_KG is a faithful copy of dog5.xml",
          abs(st.TRUNK_MASS_KG - float(model.body_mass[tid])) < 1e-9,
          f"{st.TRUNK_MASS_KG} vs {float(model.body_mass[tid])}")
    check("TRUNK_COM_BODY is a faithful copy of dog5.xml",
          float(np.max(np.abs(st.TRUNK_COM_BODY - model.body_ipos[tid]))) < 1e-9)


# ===========================================================================
# [2] the fused walk is not a second implementation
# ===========================================================================

def test_leg_frames_matches_kinematics():
    worst_foot = worst_jac = 0.0
    for q_all in _random_poses(20):
        for i, leg in enumerate(kin.LEGS):
            foot, anchors, axes, _ = st.leg_frames(leg, q_all[i])
            worst_foot = max(worst_foot, float(np.max(np.abs(
                foot - kin.foot_position(leg, q_all[i])))))
            worst_jac = max(worst_jac, float(np.max(np.abs(
                st.foot_jacobian_from(foot, anchors, axes)
                - kin.foot_jacobian(leg, q_all[i])))))
    check("leg_frames foot == dog5_kinematics.foot_position",
          worst_foot < 1e-12, f"worst {worst_foot:.2e} m")
    check("foot_jacobian_from == dog5_kinematics.foot_jacobian",
          worst_jac < 1e-12, f"worst {worst_jac:.2e}")


# ===========================================================================
# [3] the level-trunk identity -- what protects every existing caller
# ===========================================================================

def test_level_identity():
    worst = 0.0
    for q_all in _random_poses(50, seed=1):
        for i, leg in enumerate(kin.LEGS):
            worst = max(worst, float(np.max(np.abs(
                st.leg_gravity_torque_tilted(leg, q_all[i])
                - kin.leg_gravity_torque(leg, q_all[i])))))
    check("leg_gravity_torque_tilted(level) IS dog5_kinematics'",
          worst == 0.0, f"worst {worst:.2e} Nm over 50 poses x 4 legs")
    check("...and explicitly passing g=(0,0,-1) is the same",
          float(np.max(np.abs(
              st.leg_gravity_torque_tilted("FL", Q_CROUCH[0], [0, 0, -1])
              - kin.leg_gravity_torque("FL", Q_CROUCH[0])))) == 0.0)


def test_gravity_direction():
    check("gravity_down_body(I) is -z",
          float(np.max(np.abs(
              st.gravity_down_body(np.eye(3)) - [0, 0, -1]))) < 1e-15)
    g = st.gravity_down_body(C_from_rp(0.14, -0.09))
    check("gravity_down_body stays a unit vector under tilt",
          abs(float(np.linalg.norm(g)) - 1.0) < 1e-12,
          f"|g| = {np.linalg.norm(g):.12f}")
    # the reason the tilted form exists at all
    d = st.leg_gravity_torque_tilted("FL", Q_CROUCH[0], g) \
        - st.leg_gravity_torque_tilted("FL", Q_CROUCH[0])
    check("tilt materially moves the abduction gravity torque",
          abs(d[0]) > 0.02,
          f"8 deg roll / -5 deg pitch shifts abd by {d[0]:+.3f} Nm")


# ===========================================================================
# [4] MuJoCo inverse dynamics as ground truth
# ===========================================================================

def test_matches_mujoco_inverse_dynamics():
    worst = 0.0
    for q_all in _random_poses(20, seed=2):
        tau_model, tau_ours = st.verify_against_model(q_all)
        worst = max(worst, float(np.max(np.abs(tau_model - tau_ours))))
    check("stance_torque == MuJoCo floating-base inverse dynamics",
          worst < 1e-9, f"worst {worst:.2e} Nm over 20 poses")


def test_matches_mujoco_with_feet_up():
    """A leg with no ground under it must command exactly its own weight."""
    worst = 0.0
    for contacts in ([True, True, True, False], [True, False, False, True],
                     [False, False, False, False]):
        tau_model, tau_ours = st.verify_against_model(
            Q_CROUCH, contacts=np.array(contacts))
        worst = max(worst, float(np.max(np.abs(tau_model - tau_ours))))
    check("...also with one, two and four feet unloaded",
          worst < 1e-9, f"worst {worst:.2e} Nm")


def test_leg_gravity_is_load_bearing():
    """THE anti-regression gate.  If stance_torque is ever 'simplified' back
    to -J^T f, this goes red with the number that motivated the change."""
    share = st.total_mass() * P.GRAVITY_M_S2 / 4.0
    f = np.array([0.0, 0.0, share])
    for name, q_all in (("recorded crouch", Q_CROUCH),):
        tau_model, tau_ours = st.verify_against_model(q_all)
        tau_bare = np.zeros(12)
        for i, leg in enumerate(kin.LEGS):
            foot, anchors, axes, _ = st.leg_frames(leg, q_all[i])
            J = st.foot_jacobian_from(foot, anchors, axes)
            tau_bare[3 * i:3 * i + 3] = -J.T @ f
        err_bare = float(np.max(np.abs(tau_model - tau_bare)))
        err_ours = float(np.max(np.abs(tau_model - tau_ours)))
        check(f"-J^T f ALONE is materially wrong at the {name}",
              err_bare > 0.3,
              f"max err {err_bare:.3f} Nm (rms "
              f"{float(np.sqrt(np.mean((tau_model - tau_bare) ** 2))):.3f})")
        check("...and adding leg gravity removes it entirely",
              err_ours < 1e-9,
              f"{err_bare:.3f} Nm -> {err_ours:.1e} Nm")

    # the per-joint split, printed because it is the argument
    tau_bare_fl = -st.foot_jacobian_from(
        *st.leg_frames("FL", Q_CROUCH[0])[:3]).T @ f
    g_fl = kin.leg_gravity_torque("FL", Q_CROUCH[0])
    frac = np.abs(g_fl) / np.abs(tau_bare_fl)
    check("abduction is the joint the omission hurts most",
          frac[0] > 0.5 and frac[0] > frac[1] > frac[2],
          f"FL abd {100*frac[0]:.0f}%  pitch {100*frac[1]:.0f}%  "
          f"knee {100*frac[2]:.0f}% of |J^T f|")
    check("...and it OPPOSES the J^T f term (so the error adds, not cancels)",
          float(np.sign(g_fl[0]) * np.sign(tau_bare_fl[0])) < 0,
          f"J^T f {tau_bare_fl[0]:+.3f} vs gravity {g_fl[0]:+.3f} Nm")


def test_matches_mujoco_under_tilt():
    """Gravity in the body frame is the only thing that changes when the trunk
    tilts, so the tilted form must track it without a second MuJoCo pose."""
    worst = 0.0
    for roll, pitch in _tilts():
        C = C_from_rp(roll, pitch)
        g_down = st.gravity_down_body(C)
        for i, leg in enumerate(kin.LEGS):
            tau = st.leg_gravity_torque_tilted(leg, Q_CROUCH[i], g_down)
            # independent recomputation: rotate the pose's link CoMs and take
            # the moment about each axis directly
            _, anchors, axes, com_pts = st.leg_frames(leg, Q_CROUCH[i])
            ref = np.zeros(3)
            for link, inertial in enumerate(kin.LINK_INERTIALS[leg]):
                w = inertial.mass * P.GRAVITY_M_S2 * g_down
                for j in range(link + 1):
                    ref[j] -= float(np.dot(
                        axes[j], np.cross(com_pts[link] - anchors[j], w)))
            worst = max(worst, float(np.max(np.abs(tau - ref))))
    check("tilted leg gravity == an independent moment sum",
          worst < 1e-12, f"worst {worst:.2e} Nm over 5 tilts x 4 legs")


# ===========================================================================
# whole-body CoM
# ===========================================================================

def test_com_body():
    import mujoco                                         # noqa: PLC0415
    model = mujoco.MjModel.from_xml_path(os.path.join(_DESC, "dog5.xml"))
    data = mujoco.MjData(model)
    worst = 0.0
    for q_all in _random_poses(10, seed=3):
        data.qpos[:3] = 0.0
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[7:] = q_all.reshape(-1)
        mujoco.mj_forward(model, data)
        com_world = ((model.body_mass[1:, None] * data.xipos[1:]).sum(0)
                     / model.body_mass[1:].sum())
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        ref = com_world - data.xpos[tid]           # into the trunk frame
        worst = max(worst, float(np.max(np.abs(st.com_body(q_all) - ref))))
    check("com_body == MuJoCo sum(m*xipos)/sum(m), trunk frame",
          worst < 1e-9, f"worst {worst:.2e} m over 10 poses")

    c = st.com_body(Q_CROUCH)
    check("CoM is well off the trunk origin in z (and config-dependent)",
          abs(c[2]) > 0.015,
          f"crouch CoM ({c[0]*1000:+.1f}, {c[1]*1000:+.1f}, "
          f"{c[2]*1000:+.1f}) mm")
    check("...but negligible in x/y, which is what the grasp map uses",
          abs(c[0]) < 0.005 and abs(c[1]) < 0.005,
          "r_z cancels in r x (0,0,fz); only x/y set the vertical-force lever")


# ===========================================================================
# cost -- this runs in the 100 Hz worker, not the CAN sweep, but it is the
# term whose naive form (860 us for 4 legs) is why leg_frames exists
# ===========================================================================

def test_cost():
    q = Q_CROUCH
    f = np.array([0.0, 0.0, P.PER_FOOT_GRF_N])
    g_down = st.gravity_down_body(C_from_rp(0.02, -0.01))

    def sweep():
        for i, leg in enumerate(kin.LEGS):
            fr = st.leg_frames(leg, q[i])
            st.stance_torque(leg, q[i], f, g_down, frames=fr)

    sweep()
    t0 = time.perf_counter()
    n = 200
    for _ in range(n):
        sweep()
    dt = (time.perf_counter() - t0) / n
    budget = 1.0 / P.CONTROL_UPDATE_HZ
    check("4-leg corrected stance law fits the 100 Hz worker",
          dt < 0.25 * budget,
          f"{dt*1e6:.0f} us per sweep (worker budget {budget*1e3:.0f} ms)")
    check("...and is reported against the CAN slot for reference",
          True, f"{dt*1e6:.0f} us vs the {SLOT_BUDGET_S*1e6:.0f} us slot "
                f"budget -- this is WHY it runs off-thread")


def self_test():
    print("[1] model literals vs dog5.xml")
    test_mass()
    test_trunk_inertial_matches_xml()
    print("[2] the fused chain walk")
    test_leg_frames_matches_kinematics()
    print("[3] the level-trunk identity")
    test_level_identity()
    test_gravity_direction()
    print("[4] MuJoCo inverse dynamics as ground truth")
    test_matches_mujoco_inverse_dynamics()
    test_matches_mujoco_with_feet_up()
    test_leg_gravity_is_load_bearing()
    test_matches_mujoco_under_tilt()
    print("[5] whole-body CoM")
    test_com_body()
    print("[6] cost")
    test_cost()
    return report()


if __name__ == "__main__":
    sys.exit(self_test())
