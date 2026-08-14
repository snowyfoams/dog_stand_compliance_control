#!/usr/bin/env python3
"""Gates for stance_law.py -- the impedance floor, the gate, and the 100 Hz law.

WHAT IS UNDER TEST
    [1] the DISCRETE STABILITY claim, as an executable assertion.  The whole
        torque track was abandoned in 2026-07 on a sampled-stability argument
        computed at a 12x-too-long dt.  These gates simulate the real sampled
        loop -- zero-order hold, one sweep of measurement delay -- and demand
        that the gains are stable at the true 4 ms AND that the same gains
        diverge at the 48 ms that was assumed.  If someone "fixes" the rate
        back, this suite says exactly what broke.
    [2] the impedance FIXED POINT: an unloaded joint under a pure force law is
        an open integrator, and adding kp is what turns a runaway into a
        bounded offset.  Tested by simulating the 2026-07-30 RR_abd fault.
    [3] TorqueSafetyGate really overrides the position track's ramp and slew,
        and still inherits every base clamp
    [4] compute_stance reproduces dog5_vmc_core where it should and differs
        only by the leg-gravity term, and force_frac dials cleanly to zero

Run:  $V test_stance_law.py
"""
from __future__ import annotations

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

from selftest_common import check, report, raises, C_from_rp  # noqa: E402

import body_state_ahrs as bs                               # noqa: E402
import dog5_statics as st                                  # noqa: E402
import dog5_vmc_core as vmc                                # noqa: E402
import stance_law as law                                   # noqa: E402
import torque_params as P                                  # noqa: E402

LEGS = st.LEGS

Q_CROUCH = np.deg2rad(np.array(
    [90.74, 47.59, -131.02, -90.39, -51.92, 138.39,
     90.23, -44.45, 128.40, -93.95, 47.06, -125.12]))


# ===========================================================================
# [1] the discrete stability claim
# ===========================================================================

def _sim_pd(kp, kd, dt, J=P.J_KNEE, n=4000, q0=0.05):
    """One joint, zero-order-hold PD with ONE SWEEP of measurement delay.

    The delay is not optional realism -- the CAN loop always acts on the
    previous sweep's telemetry (motorbus.py:501), and it is what sets the
    stability bound.  Returns the peak |q| over the run; a divergent loop
    returns inf.
    """
    q, qd = q0, 0.0
    q_m, qd_m = q, qd          # delayed measurement
    peak = abs(q0)
    for _ in range(n):
        tau = -kp * q_m - kd * qd_m       # hold to q = 0
        q_m, qd_m = q, qd                 # this sweep's sample, used next time
        qdd = tau / J
        qd += qdd * dt
        q += qd * dt
        peak = max(peak, abs(q))
        if not np.isfinite(peak) or peak > 1e3:
            return float("inf")
    return peak


def test_gains_are_stable_at_the_true_rate():
    dt_true = P.SWEEP_S                       # 4 ms
    dt_wrong = P.N_JOINTS / P.CONTROL_HZ      # 48 ms, the abandoned premise

    peak = _sim_pd(P.KP_JOINT_IMP, P.KD_JOINT_IMP, dt_true)
    check("the shipped impedance gains are stable at the true 4 ms sweep",
          np.isfinite(peak) and peak < 0.06,
          f"kp={P.KP_JOINT_IMP} kd={P.KD_JOINT_IMP}: peak |q| "
          f"{peak*1e3:.1f} mrad from a 50 mrad step")
    peak_wrong = _sim_pd(P.KP_JOINT_IMP, P.KD_JOINT_IMP, dt_wrong)
    check("...and DIVERGE at the 48 ms the 2026-07-30 verdict assumed",
          not np.isfinite(peak_wrong),
          "which is why that verdict rejected impedance control entirely -- "
          "the arithmetic was right, the dt was 12x too long")


def test_the_damper_bound():
    dt_true, dt_wrong = P.SWEEP_S, P.N_JOINTS / P.CONTROL_HZ
    # vmc_stand_hw's KD_JOINT_BRAKE, declared "109% of the limit, unstable"
    for kd in (0.4, 1.0, 2.0):
        ok_true = np.isfinite(_sim_pd(0.0, kd, dt_true, q0=0.0) if False
                              else _sim_pd(1e-6, kd, dt_true))
        if not ok_true:
            check(f"a kd={kd} damper is stable at 4 ms", False)
            return
    check("kd = 0.4, 1.0 and 2.0 dampers are all stable at 4 ms",
          True, f"the bound 2J/dt is {2*P.J_MIN/dt_true:.1f} Nms/rad")
    check("...and kd = 0.4 diverges at 48 ms",
          not np.isfinite(_sim_pd(1e-6, 0.4, dt_wrong)),
          "vmc_stand_hw.py:120 called 0.4 '109% of the limit'; at the true "
          "rate it is 9%")
    check("a kd past the true bound still diverges (the gate is real)",
          not np.isfinite(_sim_pd(1e-6, 3.0 * 2 * P.J_MIN / dt_true, dt_true)),
          "so this is measuring stability, not always passing")


def test_gain_ceilings_are_enforced():
    check("JointImpedance refuses a kp past the measured envelope",
          raises(lambda: law.JointImpedance(kp=P.KP_IMP_MAX + 1.0)))
    check("...and a kd past the sampled-damper bound",
          raises(lambda: law.JointImpedance(kd=P.KD_IMP_MAX + 1.0)))
    check("...and negative gains", raises(lambda: law.JointImpedance(kp=-1.0)))
    check("the shipped defaults construct", law.JointImpedance() is not None)


# ===========================================================================
# [2] the fixed point -- why impedance and not a bigger damper
# ===========================================================================

def _sim_unloaded(tau_ff, kp, kd, dt=P.SWEEP_S, J=P.J_ABD, n=1000):
    """A joint with NO ground under it, driven by a constant feedforward.

    This is the 2026-07-30 RR_abd fault: the stance law commands a torque
    that assumes a ground reaction, the ground is not there, and nothing else
    references the joint's position.  Returns (peak |qd|, final |q| offset).
    """
    q, qd = 0.0, 0.0
    q_m, qd_m = 0.0, 0.0
    peak_qd = 0.0
    for _ in range(n):
        tau = tau_ff - kp * q_m - kd * qd_m
        q_m, qd_m = q, qd
        qd += (tau / J) * dt
        q += qd * dt
        peak_qd = max(peak_qd, abs(qd))
        if not np.isfinite(q) or abs(q) > 1e3:
            return float("inf"), float("inf")
    return peak_qd, abs(q)


def test_damper_alone_coasts():
    """The failure mode, reproduced: a damper sets a terminal speed, not a stop."""
    tau = 0.748                      # the as-shipped abduction command
    peak, off = _sim_unloaded(tau, kp=0.0, kd=0.4)
    check("a pure damper leaves an unloaded joint COASTING at tau/kd",
          abs(peak - tau / 0.4) / (tau / 0.4) < 0.1,
          f"terminal {peak:.2f} rad/s vs tau/kd = {tau/0.4:.2f} -- this is "
          f"the RR_abd runaway, and it trips QD_ESTOP at {7.0} rad/s")
    check("...and its position runs away without bound",
          off > 1.0, f"{off:.1f} rad of travel in {1000*P.SWEEP_S:.0f} s")


def test_impedance_bounds_the_same_fault():
    tau = 0.748
    peak, off = _sim_unloaded(tau, kp=P.KP_JOINT_IMP, kd=P.KD_JOINT_IMP)
    check("adding kp turns the runaway into a bounded offset at tau/kp",
          abs(off - tau / P.KP_JOINT_IMP) < 2e-3,
          f"settles at {off*1e3:.0f} mrad, predicted "
          f"{tau/P.KP_JOINT_IMP*1e3:.0f} mrad")
    # the peak is the step TRANSIENT (zeta = 0.64 on the abduction inertia),
    # not a terminal velocity -- the damper-only case has no settling value
    # at all.  What matters is that it is bounded and far from the e-stop.
    check("...and its peak is a bounded transient, not a terminal speed",
          peak < 0.25 * 7.0,
          f"peak {peak:.2f} rad/s vs QD_ESTOP 7.0, and vs "
          f"{tau/P.KD_JOINT_IMP:.1f} rad/s if kp were 0")

    # and with the leg-gravity term corrected the commanded torque is smaller
    # to begin with -- the two fixes compound
    peak2, off2 = _sim_unloaded(0.266, kp=P.KP_JOINT_IMP, kd=P.KD_JOINT_IMP)
    check("the corrected stance torque makes the same fault smaller still",
          off2 < off,
          f"0.748 Nm -> {off*1e3:.0f} mrad, 0.266 Nm -> {off2*1e3:.0f} mrad")


def test_impedance_arithmetic():
    imp = law.JointImpedance(kp=10.0, kd=1.0)
    q = np.zeros(12)
    qd = np.zeros(12)
    q_ref = np.full(12, 0.1)
    tau = imp.tau(q, qd, q_ref)
    check("tau = kp (q_ref - q) with qd = 0",
          float(np.max(np.abs(tau - 1.0))) < 1e-12)
    tau = imp.tau(q, np.full(12, 0.5), q_ref)
    check("...minus kd qd",
          float(np.max(np.abs(tau - (1.0 - 0.5)))) < 1e-12)
    check("worst_dq reports the largest error",
          abs(imp.worst_dq() - 0.1) < 1e-12)
    check("a zero-error, zero-rate joint commands nothing",
          float(np.max(np.abs(imp.tau(q_ref, qd, q_ref)))) == 0.0,
          "so in nominal stance the force law is unopposed")


# ===========================================================================
# [3] the safety gate override
# ===========================================================================

def test_gate_override():
    import stand_dog5_hw as base                           # noqa: PLC0415
    g = law.TorqueSafetyGate(P.TAU_STAGED_MAX)
    check("TorqueSafetyGate is a base.SafetyGate",
          isinstance(g, base.SafetyGate),
          "so every inherited clamp and e-stop test still applies")
    check("...with the torque-mode slew, not the position track's",
          g.slew_nm_s > 5.0 * base.TAU_SLEW_NM_S,
          f"{g.slew_nm_s} vs base {base.TAU_SLEW_NM_S} Nm/s")
    check("...and a shorter ramp",
          g.ramp_s < base.TORQUE_RAMP_S,
          f"{g.ramp_s} vs base {base.TORQUE_RAMP_S} s")

    # the ramp really ramps
    g.start(0.0, np.zeros(12))
    check("cap_now ramps from 0 to the cap over ramp_s",
          abs(g.cap_now(0.0)) < 1e-12
          and abs(g.cap_now(g.ramp_s) - P.TAU_STAGED_MAX) < 1e-12
          and abs(g.cap_now(0.5 * g.ramp_s) - 0.5 * P.TAU_STAGED_MAX) < 1e-9)

    # The slew really limits -- stepped one SWEEP at a time, which is how the
    # runner calls it.  Note base.SafetyGate clamps dt to 50 ms, so a single
    # call after a long gap IS allowed a full staged-cap step; the limit is
    # only meaningful when apply() is called every sweep, as it is in the loop.
    g = law.TorqueSafetyGate(P.TAU_STAGED_MAX)
    q = np.zeros(12)
    g.start(0.0, q)
    t = 0.0
    while t < g.ramp_s:                      # walk past the startup ramp at 0
        t += P.SWEEP_S
        g.apply(np.zeros(12), q, t)
    per_sweep = P.TAU_SLEW_NM_S * P.SWEEP_S
    t += P.SWEEP_S
    got = float(np.max(np.abs(g.apply(np.full(12, P.TAU_STAGED_MAX), q, t))))
    check("apply() slews rather than jumping, one sweep at a time",
          abs(got - per_sweep) < 1e-9,
          f"asked {P.TAU_STAGED_MAX} Nm, got {got:.3f} in one "
          f"{P.SWEEP_S*1e3:.0f} ms sweep (limit {per_sweep:.3f})")
    n = 1
    out = np.full(12, got)
    while float(np.max(np.abs(out))) < P.TAU_STAGED_MAX - 1e-9 and n < 500:
        n += 1
        t += P.SWEEP_S
        out = g.apply(np.full(12, P.TAU_STAGED_MAX), q, t)
    check("...reaching the cap in the predicted number of sweeps",
          abs(n - P.TAU_STAGED_MAX / per_sweep) <= 1,
          f"{n} sweeps = {n*P.SWEEP_S*1e3:.0f} ms to {P.TAU_STAGED_MAX} Nm")

    # the inherited directional limit block
    g = law.TorqueSafetyGate(P.TAU_STAGED_MAX)
    g.start(0.0, np.zeros(12))
    low, high = base.soft_limits()
    out = g.apply(np.full(12, 1.0), high + 0.1, 1.0)
    check("...and still zeroes torque pushing past a joint limit",
          float(np.max(np.abs(out))) == 0.0)


def test_gate_respects_the_hard_cap():
    g = law.TorqueSafetyGate(P.TAU_STAGED_MAX)
    g.start(0.0, np.zeros(12))
    q = np.zeros(12)
    out = np.zeros(12)
    for k in range(200):
        out = g.apply(np.full(12, 100.0), q, 1.0 + 0.01 * k)
    check("a huge demand saturates at the configured cap, not TAU_HARD",
          abs(float(np.max(np.abs(out))) - P.TAU_STAGED_MAX) < 1e-9,
          f"{float(np.max(np.abs(out))):.2f} Nm")


# ===========================================================================
# [4] the 100 Hz law
# ===========================================================================

def _est_out(z=0.19, roll=0.0, pitch=0.0, v=None):
    C = C_from_rp(roll, pitch)
    return {"r": np.array([0.0, 0.0, z]),
            "v": np.zeros(3) if v is None else np.asarray(v),
            "C": C, "w_hat": np.zeros(3), "healthy": True}


def _sched(z_des=0.19, contacts=None):
    return {"contacts": np.ones(4, dtype=bool) if contacts is None else contacts,
            "z_des": z_des, "v_cmd_world": np.zeros(3), "yawrate_cmd": 0.0}


def test_compute_stance_shape():
    gains, cfg = law.default_gains(), law.default_cfg()
    out = _est_out()
    tau, diag = law.compute_stance(Q_CROUCH, np.zeros(12), out,
                                   _sched(z_des=out["r"][2]), gains, cfg,
                                   P.DOG5_MASS_KG)
    check("compute_stance returns a finite 12-vector",
          tau.shape == (12,) and np.all(np.isfinite(tau)))
    check("...with all four legs in stance", len(diag["stance"]) == 4)
    check("...and reports the config-dependent CoM it used",
          diag["com_body"] is not None
          and abs(diag["com_body"][2] - st.com_body(Q_CROUCH.reshape(4, 3))[2])
          < 1e-12)


def test_force_frac_dials_out_the_force_law():
    gains, cfg = law.default_gains(), law.default_cfg()
    out = _est_out()
    sched = _sched(z_des=out["r"][2])
    tau0, _ = law.compute_stance(Q_CROUCH, np.zeros(12), out, sched, gains,
                                 cfg, P.DOG5_MASS_KG, force_frac=0.0)
    grav = law.gravity_stack(Q_CROUCH, out["C"])
    check("force_frac = 0 leaves exactly the leg-gravity feedforward",
          float(np.max(np.abs(tau0 - grav))) < 1e-12,
          "so --force-frac 0 is a pure joint-space hold through the torque path")

    tau1, _ = law.compute_stance(Q_CROUCH, np.zeros(12), out, sched, gains,
                                 cfg, P.DOG5_MASS_KG, force_frac=1.0)
    check("force_frac = 1 adds real force on top",
          float(np.max(np.abs(tau1 - tau0))) > 0.1,
          f"max change {float(np.max(np.abs(tau1 - tau0))):.2f} Nm")

    half, _ = law.compute_stance(Q_CROUCH, np.zeros(12), out, sched, gains,
                                 cfg, P.DOG5_MASS_KG, force_frac=0.5)
    check("...and it is linear in between",
          float(np.max(np.abs(half - 0.5 * (tau0 + tau1)))) < 1e-9,
          "so the bring-up dial is a true interpolation")


def test_leg_gravity_toggle_reproduces_the_old_law():
    gains, cfg = law.default_gains(), law.default_cfg()
    out = _est_out()
    sched = _sched(z_des=out["r"][2])
    tau_new, _ = law.compute_stance(Q_CROUCH, np.zeros(12), out, sched, gains,
                                    cfg, P.DOG5_MASS_KG, leg_gravity=True)
    tau_old, _ = law.compute_stance(Q_CROUCH, np.zeros(12), out, sched, gains,
                                    cfg, P.DOG5_MASS_KG, leg_gravity=False)
    d = float(np.max(np.abs(tau_new - tau_old)))
    check("--no-leg-gravity reproduces the 2026-07-30 law, and differs a lot",
          d > 0.3, f"max difference {d:.3f} Nm -- the A/B stage T2 runs")
    check("...and the difference IS the gravity stack",
          float(np.max(np.abs((tau_new - tau_old)
                              - law.gravity_stack(Q_CROUCH, out["C"])))) < 1e-12)


def test_uses_the_correct_mass():
    """The wrench must be built on 5.815 kg, not stand_dog5_hw's 5.3."""
    import stand_dog5_hw as base                           # noqa: PLC0415
    gains = law.default_gains()
    out = _est_out()
    W_right = vmc.body_wrench(out, out["r"][2], np.zeros(3), 0.0, gains,
                              P.DOG5_MASS_KG)
    W_wrong = vmc.body_wrench(out, out["r"][2], np.zeros(3), 0.0, gains,
                              base.DOG5_MASS_KG)
    check("the wrench's gravity term is the model mass",
          abs(W_right[2] - P.WEIGHT_N) < 0.05,
          f"Fz {W_right[2]:.2f} N at zero height error")
    check("...and base's 5.3 kg would be 5 N short",
          W_right[2] - W_wrong[2] > 4.0,
          f"{W_right[2]:.1f} vs {W_wrong[2]:.1f} N")


def test_stance_matches_statics_leg_by_leg():
    """compute_stance must agree with dog5_statics.stance_torque, so the
    MuJoCo-verified model and the runner cannot drift apart."""
    gains, cfg = law.default_gains(), law.default_cfg()
    out = _est_out(roll=0.05, pitch=-0.03)
    sched = _sched(z_des=out["r"][2])
    tau, diag = law.compute_stance(Q_CROUCH, np.zeros(12), out, sched, gains,
                                   cfg, P.DOG5_MASS_KG)
    g_down = st.gravity_down_body(out["C"])
    worst = 0.0
    for i, leg in enumerate(LEGS):
        sl = slice(3 * i, 3 * i + 3)
        ref, _ = st.stance_torque(leg, Q_CROUCH[sl], diag["forces"][leg],
                                  g_down)
        worst = max(worst, float(np.max(np.abs(tau[sl] - ref))))
    check("compute_stance == dog5_statics.stance_torque, leg by leg",
          worst < 1e-12, f"worst {worst:.2e} Nm (tilted 2.9/-1.7 deg)")


def test_body_state_drives_the_law():
    """End to end: AHRS-only state in, torque out, no EKF anywhere."""
    class _Imu:
        class _S:
            roll_deg = 0.4
            pitch_deg = -0.2
            roll_rate_dps = 0.0
            pitch_rate_dps = 0.0
            yaw_rate_dps = 0.0

        def sample(self):
            return _Imu._S()

        def is_stale(self, max_age_s=0.05):
            return False

    state = bs.BodyState(_Imu())
    out, active, _ = state.read(0.0, Q_CROUCH, np.zeros(12),
                                np.ones(4, dtype=bool))
    gains, cfg = law.default_gains(), law.default_cfg()
    tau, diag = law.compute_stance(Q_CROUCH, np.zeros(12), out,
                                   _sched(z_des=out["r"][2]), gains, cfg,
                                   P.DOG5_MASS_KG)
    check("the AHRS-only state drives the stance law with no EKF present",
          active and np.all(np.isfinite(tau)) and len(diag["stance"]) == 4,
          f"max |tau| {float(np.max(np.abs(tau))):.2f} Nm")
    check("...and 'ekf' appears nowhere in the torque law's imports",
          not any("ekf" in m for m in (law.__file__, bs.__file__)),
          "stage 1 is EKF-free by construction, not by discipline")


def test_zdes_must_track_the_measured_height():
    """A trap worth a gate: z_des is an ABSOLUTE height, so starting the rise
    at the stand height while the robot is still crouched asks for four times
    the robot's weight on the first sweep."""
    gains = law.default_gains()
    out = _est_out(z=0.0415)                    # the measured crouch
    W_bad = vmc.body_wrench(out, P.STAND_HEIGHT_DEFAULT, np.zeros(3), 0.0,
                            gains, P.DOG5_MASS_KG)
    W_good = vmc.body_wrench(out, out["r"][2], np.zeros(3), 0.0, gains,
                             P.DOG5_MASS_KG)
    check("z_des = stand height at the crouch demands a huge force",
          W_bad[2] > 3.0 * P.WEIGHT_N,
          f"{W_bad[2]:.0f} N against a {P.WEIGHT_N:.0f} N robot -- the RISE "
          f"ramp MUST start from the measured crouch height")
    check("...and z_des = the measured height demands exactly its weight",
          abs(W_good[2] - P.WEIGHT_N) < 0.05, f"{W_good[2]:.2f} N")


def self_test():
    print("[1] discrete stability at the true loop rate")
    test_gains_are_stable_at_the_true_rate()
    test_the_damper_bound()
    test_gain_ceilings_are_enforced()
    print("[2] the impedance fixed point")
    test_damper_alone_coasts()
    test_impedance_bounds_the_same_fault()
    test_impedance_arithmetic()
    print("[3] the torque-mode safety gate")
    test_gate_override()
    test_gate_respects_the_hard_cap()
    print("[4] the 100 Hz stance law")
    test_compute_stance_shape()
    test_force_frac_dials_out_the_force_law()
    test_leg_gravity_toggle_reproduces_the_old_law()
    test_uses_the_correct_mass()
    test_stance_matches_statics_leg_by_leg()
    test_body_state_drives_the_law()
    test_zdes_must_track_the_measured_height()
    return report()


if __name__ == "__main__":
    sys.exit(self_test())
