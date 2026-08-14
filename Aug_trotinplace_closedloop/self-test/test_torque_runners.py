#!/usr/bin/env python3
"""Gates for the three torque runners' offline logic.

    tau_calib_hw    ScaleSweep, _crossing        (stage T0)
    torque_hold_hw  HoldStats, GapWatch          (stage T1)
    torque_stand_hw StandSequence, RunawayBrake, LoadWatch  (stages T2-T4)

WHAT IS UNDER TEST
    [1] the T0 estimator: sweeping a gravity SCALE and reading off where the
        impedance error crosses zero really does recover the true torque
        scale, and degrades honestly under stiction
    [2] the T1 discriminator: a drifting hold and an oscillating hold must be
        told apart, because they have different causes and different fixes
    [3] THE HEIGHT REFERENCE.  z_des is absolute, so the rise must start from
        the MEASURED crouch height.  Starting it at the stand height demands
        three times the robot's weight on the first sweep.
    [4] the corrected RunawayBrake: latches, releases with hysteresis, ramps
        drive back on TIME (not on qd, which would be feedback), and its hold
        torque is now large enough to actually stop a joint
    [5] LIMP semantics -- the abort that keeps the keep-alives running

Run:  $V test_torque_runners.py
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

from selftest_common import check, report                  # noqa: E402

import torque_params as P                                  # noqa: E402

# The runners import motorbus/ImuDog at module scope, which is fine offline
# (python-can and pyserial are installed; nothing opens a bus until run()).
import tau_calib_hw as t0                                  # noqa: E402
import torque_hold_hw as t1                                # noqa: E402
import torque_stand_hw as t2                               # noqa: E402


# ===========================================================================
# [1] the T0 gravity-scale estimator
# ===========================================================================

def test_crossing_recovers_the_scale():
    kp, tau_model = 15.0, 0.4
    scales = np.arange(0.7, 1.31, 0.1)
    for k in (0.85, 1.0, 1.12, 1.25):
        errs = [(k * tau_model - s * tau_model) / kp for s in scales]
        s_star, band = t0._crossing(scales, errs)
        if abs(s_star - k) > 1e-6 or band > 1e-9:
            check("the scale sweep recovers a noise-free torque scale",
                  False, f"true {k}, got {s_star:.4f}")
            return
    check("the scale sweep recovers a noise-free torque scale",
          True, "0.85, 1.00, 1.12, 1.25 all exact, band 0")


def test_crossing_under_stiction():
    """Friction must show up as a band, and must not wreck the estimate."""
    kp, tau_model, k, f = 15.0, 0.4, 1.12, 0.03
    scales = np.arange(0.7, 1.31, 0.1)
    errs = []
    for s in scales:
        e = k * tau_model - s * tau_model
        e = np.sign(e) * max(abs(e) - f, 0.0)      # deadband
        errs.append(e / kp)
    s_star, band = t0._crossing(scales, errs)
    check("stiction biases the scale only slightly",
          abs(s_star - k) < 0.05, f"true {k}, got {s_star:.3f}")
    check("...and shows up as a non-zero band",
          band > 0.02, f"band {band:.3f} (a lower bound on the true "
                       f"{2*f/tau_model:.3f})")


def test_crossing_refuses_degenerate_input():
    s, b = t0._crossing([1.0], [0.0])
    check("a single scale cannot give a crossing", np.isnan(s))
    s, b = t0._crossing([0.8, 0.9, 1.0], [0.0, 0.0, 0.0])
    check("a flat sweep gives NaN, not a divide-by-zero", np.isnan(s))


def test_scale_sweep_waits_for_stillness():
    """The dwell is gated on joint speed, not a timer: stiction means some
    scales do not move at all and a timer would sample others mid-drift."""
    sw = t0.ScaleSweep([0.9, 1.0], dwell_s=1.0, settle_qd=0.05)
    sw.update(0.0, np.zeros(12), np.zeros(12), np.zeros(12), np.zeros(12))
    moving = np.full(12, 0.5)
    msg = sw.update(2.0, np.zeros(12), moving, np.zeros(12), np.zeros(12))
    check("a still-moving leg does not end the dwell", msg == "" and not sw.records)
    msg = sw.update(2.1, np.zeros(12), np.zeros(12), np.zeros(12), np.zeros(12))
    check("...and a settled one does", "settled" in msg and len(sw.records) == 1)
    check("the sweep advances to the next scale", abs(sw.scale - 1.0) < 1e-9)


# ===========================================================================
# [2] the T1 drift/oscillation discriminator
# ===========================================================================

def _hold_stats(series, settle_s=0.0):
    s = t1.HoldStats(settle_s=settle_s)
    for k, v in enumerate(series):
        s.add(k * P.SWEEP_S, np.full(12, v))
    return s.report()


def test_hold_discriminates_drift_from_oscillation():
    n = 400
    steady = _hold_stats(np.full(n, 0.004))
    check("a steady hold is reported as steady",
          "steady" in steady, steady.strip().splitlines()[-1].strip())

    drift = _hold_stats(np.linspace(0.002, 0.020, n))
    check("a slow creep is called a feedforward error, not a gain problem",
          "DRIFTING" in drift, drift.strip().splitlines()[-1].strip())

    osc = _hold_stats(0.010 + 0.008 * np.sin(np.arange(n) * 0.5))
    check("a fixed-amplitude ring is called the discrete stability bound",
          "OSCILLATING" in osc, osc.strip().splitlines()[-1].strip())
    check("...and names the inertias, not the loop rate, as the suspect",
          "J_KNEE" in osc,
          "because the rate is now measured, and a ring at 4 ms means J is off")


def test_hold_needs_settled_samples():
    s = t1.HoldStats(settle_s=5.0)
    for k in range(10):
        s.add(k * P.SWEEP_S, np.zeros(12))
    check("too few settled samples refuses to score",
          "too few" in s.report())


def test_gap_watch():
    g = t1.GapWatch()
    mid = P.N_JOINTS and 7
    for k in range(20):
        g.mark(mid, k * P.SWEEP_S)
    check("a clean stream reports a gap of one sweep",
          abs(g.worst_overall() - P.SWEEP_S) < 1e-9,
          f"{g.worst_overall()*1e3:.1f} ms")
    g.mark(mid, 20 * P.SWEEP_S + 0.012)
    check("a 12 ms stall is caught",
          g.worst_overall() > P.WATCHDOG_S,
          f"{g.worst_overall()*1e3:.1f} ms vs a "
          f"{P.WATCHDOG_S*1e3:.0f} ms watchdog")
    check("...and named in the report",
          "over threshold" in g.report(), g.report().splitlines()[-1].strip())


# ===========================================================================
# [3] the height reference -- the trap that would command 3x body weight
# ===========================================================================

def test_sequence_height_reference():
    seq = t2.StandSequence(0.0, stand_height=0.19)
    check("z_des is None before the crouch height is measured",
          seq.z_des(0.0) is None,
          "so nothing can command a height we have not been told")

    seq.enter_wait(1.0, z_measured=0.0415)
    check("WAIT_CROUCH holds the MEASURED crouch height",
          abs(seq.z_des(1.0) - 0.0415) < 1e-12,
          "not the stand height -- see test_stance_law's 176 N gate")

    check("the rise refuses to start before the crouch is measured",
          not t2.StandSequence(0.0, 0.19).start_rise(0.0))

    seq.start_rise(2.0)
    check("the rise starts AT the crouch height (no step)",
          abs(seq.z_des(2.0) - 0.0415) < 1e-9,
          f"{seq.z_des(2.0)*1e3:.1f} mm")
    mid = seq.z_des(2.0 + 0.5 * P.T_RISE)
    check("...passes through the middle",
          0.0415 < mid < 0.19, f"{mid*1e3:.0f} mm at half time")
    seq.stage = "HOLD"
    check("...and HOLD is exactly the commanded stand height",
          abs(seq.z_des(99.0) - 0.19) < 1e-12)

    # monotone and continuous through the whole ramp
    seq2 = t2.StandSequence(0.0, 0.19)
    seq2.enter_wait(0.0, 0.0415)
    seq2.start_rise(0.0)
    zs = [seq2.z_des(t) for t in np.linspace(0.0, P.T_RISE, 200)]
    check("the rise is monotone and smooth",
          all(b >= a - 1e-12 for a, b in zip(zs, zs[1:]))
          and max(abs(b - a) for a, b in zip(zs, zs[1:])) < 0.005,
          f"max step {max(abs(b-a) for a, b in zip(zs, zs[1:]))*1e3:.2f} mm")


def test_sequence_stages():
    seq = t2.StandSequence(0.0, 0.19)
    check("CROUCH is not a torque stage", not seq.torque_active)
    seq.enter_wait(0.0, 0.0415)
    check("WAIT_CROUCH is not a torque stage", not seq.torque_active)
    seq.start_rise(0.0)
    check("RISE is", seq.torque_active)
    seq.update(P.T_RISE + 0.01)
    check("RISE ends at HOLD", seq.stage == "HOLD" and seq.torque_active)
    check("park is refused outside HOLD... ",
          not t2.StandSequence(0.0, 0.19).park(0.0))
    check("...and accepted from HOLD", seq.park(0.0))
    seq.update(P.T_RISE + 0.01)
    check("PARK ends at PARKED, which is not a torque stage",
          seq.stage == "PARKED" and not seq.torque_active)


def test_crouch_settle_is_on_speed():
    seq = t2.StandSequence(0.0, 0.19)
    fast = np.full(12, 1.0)
    check("a moving crouch is not settled",
          not seq.crouch_settled(0.0, fast, np.zeros(12)))
    still = np.zeros(12)
    check("...and stillness alone is not enough either (it must persist)",
          not seq.crouch_settled(1.0, still, np.zeros(12)))
    check("...but persistent stillness is",
          seq.crouch_settled(1.0 + P.T_CROUCH_SETTLE_S + 0.01, still,
                             np.zeros(12)))


# ===========================================================================
# [4] the corrected brake
# ===========================================================================

def test_brake_latches_and_releases():
    b = t2.RunawayBrake()
    tau = np.full(12, 2.0)
    qd = np.zeros(12)
    out, n = b.apply(tau, qd, 0.0)
    check("a slow joint passes the controller torque through untouched",
          float(np.max(np.abs(out - tau))) < 1e-12 and n == 0)

    qd[0] = P.BRAKE_LATCH_RAD_S + 0.5
    out, n = b.apply(tau, qd, 1.0)
    check("a fast joint latches and gets a CONSTANT opposing torque",
          n == 1 and abs(out[0] + P.BRAKE_HOLD_NM) < 1e-12,
          f"{out[0]:+.2f} Nm against qd {qd[0]:+.1f} rad/s")
    check("...and the other joints are unaffected",
          float(np.max(np.abs(out[1:] - 2.0))) < 1e-12)

    # hysteresis: between release and latch it stays latched
    qd[0] = 0.5 * (P.BRAKE_RELEASE_RAD_S + P.BRAKE_LATCH_RAD_S)
    out, n = b.apply(tau, qd, 1.0 + P.BRAKE_MIN_LATCH_S + 0.01)
    check("it does not release inside the hysteresis band", n == 1)

    qd[0] = 0.5 * P.BRAKE_RELEASE_RAD_S
    out, n = b.apply(tau, qd, 1.0 + P.BRAKE_MIN_LATCH_S + 0.02)
    check("...and releases below the release speed", n == 0)


def test_brake_minimum_dwell():
    b = t2.RunawayBrake()
    qd = np.zeros(12)
    qd[3] = P.BRAKE_LATCH_RAD_S + 1.0
    b.apply(np.zeros(12), qd, 0.0)
    qd[3] = 0.0
    _, n = b.apply(np.zeros(12), qd, 0.01)      # inside the dwell
    check("a latch holds for at least BRAKE_MIN_LATCH_S", n == 1,
          f"{P.BRAKE_MIN_LATCH_S} s")


def test_brake_recovery_is_a_time_ramp():
    """Restoring drive as a function of qd would be velocity feedback, and
    that is exactly what the zero-incremental-gain design avoids."""
    b = t2.RunawayBrake()
    qd = np.zeros(12)
    qd[0] = P.BRAKE_LATCH_RAD_S + 1.0
    b.apply(np.zeros(12), qd, 0.0)
    qd[0] = 0.0
    t_rel = P.BRAKE_MIN_LATCH_S + 0.01
    b.apply(np.zeros(12), qd, t_rel)            # releases here
    tau = np.full(12, 2.0)
    frac = []
    for k in range(1, 6):
        out, _ = b.apply(tau, qd, t_rel + k * P.BRAKE_RECOVER_S / 5.0)
        frac.append(out[0] / 2.0)
    check("drive is restored on a time ramp, not a velocity ramp",
          all(b_ > a_ for a_, b_ in zip(frac, frac[1:]))
          and frac[-1] > 0.9 and frac[0] < 0.5,
          f"{[round(f,2) for f in frac]} over {P.BRAKE_RECOVER_S} s")


def test_brake_hold_is_strong_enough_now():
    """The shipped 48 ms constant gave 0.128 Nm, less than a leg's own weight."""
    old = (P.BRAKE_DEADBEAT_FRAC * P.J_MIN * P.BRAKE_RELEASE_RAD_S
           / (P.N_JOINTS / P.CONTROL_HZ))
    check("the corrected hold torque exceeds abduction leg gravity",
          P.BRAKE_HOLD_NM > 0.47 > old,
          f"{P.BRAKE_HOLD_NM:.2f} Nm now, {old:.3f} Nm as shipped, against "
          f"~0.47 Nm of leg weight")


def test_brake_never_exceeds_the_operator_cap():
    """Hardware 2026-08-14: `--tau-max 1.0` and the status line printed
    |tau|=1.54 -- exactly BRAKE_HOLD_NM.

    The brake runs AFTER SafetyGate on purpose (it must be neither ramped nor
    slewed), which makes it the one path that can outrun the cap.  That was
    invisible while the constant was the 48 ms value of 0.128 Nm, because
    that was below every cap -- and below anything useful.  Correcting it to
    1.54 made it effective and made it an override at the same time.
    """
    b = t2.RunawayBrake()
    qd = np.zeros(12)
    qd[0] = P.BRAKE_LATCH_RAD_S + 1.0
    out, _ = b.apply(np.zeros(12), qd, 0.0, cap=1.0)
    check("the brake honours a --tau-max below its hold torque",
          abs(out[0]) <= 1.0 + 1e-12,
          f"cap 1.0 Nm -> {abs(out[0]):.2f} Nm (BRAKE_HOLD_NM is "
          f"{P.BRAKE_HOLD_NM})")

    b2 = t2.RunawayBrake()
    out, _ = b2.apply(np.zeros(12), qd, 0.0, cap=P.TAU_STAGED_MAX)
    check("...and uses its full hold torque when the cap allows it",
          abs(abs(out[0]) - P.BRAKE_HOLD_NM) < 1e-12,
          f"cap {P.TAU_STAGED_MAX} Nm -> {abs(out[0]):.2f} Nm")

    b3 = t2.RunawayBrake()
    out, _ = b3.apply(np.zeros(12), qd, 0.0)
    check("...and an uncapped call is unchanged (the old signature still works)",
          abs(abs(out[0]) - P.BRAKE_HOLD_NM) < 1e-12)


def test_force_frac_zero_is_not_a_free_plumbing_test():
    """Hardware 2026-08-14: `--force-frac 0` with the feet on the floor folded
    the legs and latched seven joints inside 2 s of RISE.

    force_frac scales the distributed GRF, which is the ONLY term holding the
    trunk up -- leg gravity carries each leg's own links and nothing more.  So
    at 0 the trunk is supported purely by the impedance's positional error.
    Gate the arithmetic that makes that predictable, so the docs cannot drift
    back to calling it safe.
    """
    sag = P.WEIGHT_N * 0.12 / P.KP_JOINT_IMP        # ~0.12 m lever per leg
    check("at force_frac 0 the predicted trunk sag is large, not negligible",
          sag > 0.05,
          f"~{sag*1e3:.0f} mrad per joint at kp={P.KP_JOINT_IMP:.0f} "
          f"({P.WEIGHT_N:.0f} N over a ~0.12 m lever)")
    check("...far past the notice threshold the status line flags",
          sag > 3.0 * P.IMP_DQ_NOTICE_RAD,
          f"{sag*1e3:.0f} mrad vs a {P.IMP_DQ_NOTICE_RAD*1e3:.0f} mrad notice")
    check("torque_params carries the warn threshold the runner uses",
          0.0 < P.FORCE_FRAC_WARN_BELOW < 1.0,
          f"warns below force_frac {P.FORCE_FRAC_WARN_BELOW}")
    check("...and the default is full force, not the dial-down",
          P.FORCE_FRAC_DEFAULT == 1.0)

    import pathlib                                        # noqa: PLC0415
    src = pathlib.Path(_TORQUE, "torque_stand_hw.py").read_text()
    check("the runner warns at startup instead of failing silently",
          "force_frac" in src and "WARNING" in src,
          "prints the predicted sag and says a stand is required")


def test_brake_stop_conditions():
    b = t2.RunawayBrake()
    qd = np.zeros(12)
    check("a fresh brake has no stop reason", b.stop_reason(0.0) is None)
    for k in range(P.BRAKE_MAX_TRIPS + 1):
        qd[5] = P.BRAKE_LATCH_RAD_S + 1.0
        b.apply(np.zeros(12), qd, 10.0 * k)
        qd[5] = 0.0
        b.apply(np.zeros(12), qd, 10.0 * k + P.BRAKE_MIN_LATCH_S + 0.01)
    check("repeated latching of one joint stops the run with a diagnosis",
          b.stop_reason(100.0) is not None
          and "not bearing load" in b.stop_reason(100.0),
          b.stop_reason(100.0))

    b2 = t2.RunawayBrake()
    qd = np.zeros(12)
    qd[2] = P.BRAKE_LATCH_RAD_S + 5.0
    b2.apply(np.zeros(12), qd, 0.0)
    b2.apply(np.zeros(12), qd, P.BRAKE_LATCH_TIMEOUT_S + 0.1)
    check("...as does one latch that never slows down",
          b2.stop_reason(P.BRAKE_LATCH_TIMEOUT_S + 0.1) is not None)


# ===========================================================================
# [5] load watch and LIMP semantics
# ===========================================================================

def test_load_watch_bounds():
    w = t2.LoadWatch()
    w.samples.append(P.WEIGHT_N)
    check("a correct load sum is in bounds", not w.out_of_bounds())
    w.samples.append(P.WEIGHT_N * (1.0 + P.LOAD_SUM_TOL_FRAC + 0.05))
    check("a load sum far from the robot's weight limps the run",
          w.out_of_bounds(),
          f"beyond +/-{100*P.LOAD_SUM_TOL_FRAC:.0f}% of {P.WEIGHT_N:.1f} N")
    w.samples.append(float("nan"))
    check("a non-finite sum does NOT limp (a singular leg is not a fault)",
          not w.out_of_bounds(),
          "foot_load_map returns NaN at a folded pose by design")


def test_miss_monitor_is_polled_every_sweep():
    """Hardware 2026-08-14: the T2 run died on 'CAN 12 missed 1447
    consecutive replies' the instant RISE armed.

    base.CanMissMonitor DIFFERENCES a cumulative counter, so a call that is
    skipped does not skip those misses -- it hands the whole backlog to the
    next call, and `streaks` (documented as CONSECUTIVE) becomes "everything
    since I last looked".  torque_stand_hw polled it only inside the torque
    branch, so ~25 s of crouch accumulated silently and tripped MISS_ESTOP
    (20) on the first torque sweep.

    Gate the CALL SITE, not the arithmetic: update() must not be reachable
    only from inside a conditional.
    """
    import pathlib                                        # noqa: PLC0415
    bad = []
    for name in ("torque_stand_hw.py", "torque_hold_hw.py", "tau_calib_hw.py"):
        src = pathlib.Path(_TORQUE, name).read_text()
        # the call must not appear as an ARGUMENT to estop_reason -- that is
        # what buries it inside whatever branch the estop lives in
        if "miss.update(mb)," in src.replace(" ", "").replace("\n", ""):
            bad.append(f"{name}: miss.update() called as an estop argument")
        if "miss.update(" not in src:
            bad.append(f"{name}: never polls the miss monitor")
    check("the CAN miss monitor is polled unconditionally, not inside a branch",
          not bad, "; ".join(bad) or "all three runners poll it every sweep")

    # and the arithmetic itself, so the reason the call site matters is explicit
    class _Rec:
        def __init__(self):
            self.missed = 0

    class _MB:
        def __init__(self):
            self.r = {m: _Rec() for m in range(1, 13)}

        def rec(self, m):
            return self.r[m]

    import stand_dog5_hw as base                          # noqa: PLC0415
    mb = _MB()
    mon = base.CanMissMonitor(mb)
    for _ in range(500):                     # 500 sweeps of one miss each
        mb.r[12].missed += 1
    streaks = mon.update(mb)                 # ...polled once at the end
    check("...because one skipped poll turns 500 misses into one 500 streak",
          int(streaks[base.MOTOR_IDS.index(12)]) == 500,
          f"streak {int(streaks[base.MOTOR_IDS.index(12)])} vs MISS_ESTOP "
          f"{base.MISS_ESTOP} -- an instant trip")

    mb2 = _MB()
    mon2 = base.CanMissMonitor(mb2)
    worst = 0
    for k in range(500):                     # the same misses, polled properly
        mb2.r[12].missed += (k % 3 == 0)     # an intermittent 33% miss rate
        worst = max(worst, int(mon2.update(mb2)[base.MOTOR_IDS.index(12)]))
    check("...while polling every sweep leaves an intermittent rate harmless",
          worst < base.MISS_ESTOP,
          f"worst streak {worst} at a 33% miss rate -- the streak resets on "
          f"every clean sweep, which is what 'consecutive' is supposed to mean")


def test_limp_is_not_stop():
    check("LIMP and STOP are distinct keys",
          P.KEY_LIMP != P.KEY_STOP and P.KEY_LIMP == " ",
          "SPACE zeroes torque but keeps the round-robin and the keep-alives "
          "running; killing the process latches all twelve drivers")
    check("a stance-leg latch limps rather than recovering in place",
          P.LATCH_LIMPS_ROBOT,
          "under 0xA1 a latched motor stops pushing and that leg collapses")


def self_test():
    print("[1] T0 gravity-scale estimator")
    test_crossing_recovers_the_scale()
    test_crossing_under_stiction()
    test_crossing_refuses_degenerate_input()
    test_scale_sweep_waits_for_stillness()
    print("[2] T1 hold discriminator")
    test_hold_discriminates_drift_from_oscillation()
    test_hold_needs_settled_samples()
    test_gap_watch()
    print("[3] the height reference")
    test_sequence_height_reference()
    test_sequence_stages()
    test_crouch_settle_is_on_speed()
    print("[4] the corrected runaway brake")
    test_brake_latches_and_releases()
    test_brake_minimum_dwell()
    test_brake_recovery_is_a_time_ramp()
    test_brake_hold_is_strong_enough_now()
    test_brake_never_exceeds_the_operator_cap()
    test_force_frac_zero_is_not_a_free_plumbing_test()
    test_brake_stop_conditions()
    print("[5] load watch, the miss monitor, and LIMP")
    test_load_watch_bounds()
    test_miss_monitor_is_polled_every_sweep()
    test_limp_is_not_stop()
    return report()


if __name__ == "__main__":
    sys.exit(self_test())
