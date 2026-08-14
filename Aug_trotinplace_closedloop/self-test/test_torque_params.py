#!/usr/bin/env python3
"""Gates for torque_params.py -- the torque track's single source of truth.

WHAT IS UNDER TEST
    [1] THE RATE.  `250/12 = 20.8 Hz, 48 ms` is wrong and is written in three
        places in this repo.  On 2026-07-30 it was used to prove that software
        torque could not stabilise these legs, and the whole torque track was
        abandoned on it.  Every stability bound in torque_params is derived
        from the true 4 ms sweep, so if anyone ever "corrects" it back, these
        gates go red carrying the story.
    [2] the gains sit inside the bounds the corrected rate implies -- and the
        bounds are recomputed here from J and dt rather than copied, so a gain
        and its justification cannot drift apart
    [3] the torque-mode overrides really do differ from the position track's,
        because silently inheriting base.TAU_SLEW_NM_S is the failure mode
    [4] no torque runner has grown its own copy of a tunable (the property
        test_stand_params.py exists to enforce, applied to this directory)

Run:  $V test_torque_params.py
"""
from __future__ import annotations

import ast
import os
import pathlib
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TORQUE = os.path.join(os.path.dirname(_HERE), "torque_mode_control")
_RUNNERS = os.path.join(os.path.dirname(_HERE), "stand_postion_mode")
for _p in (_HERE, _TORQUE, _RUNNERS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from selftest_common import check, report                # noqa: E402

import torque_params as P                                # noqa: E402
import stand_dog5_hw as base                             # noqa: E402

ALLOWED_ALIASES = {"MOTOR_IDS", "N_JOINTS", "CONTROL_HZ", "LEGS", "Q_CROUCH",
                   "POSITION_TARGET_DEG", "STAND_HEIGHT_DEFAULT", "LEGS"}


# ===========================================================================
# [1] the rate -- the gate this whole file is built around
# ===========================================================================

def test_the_rate():
    check("SWEEP_S is 1/CONTROL_HZ, NOT N_JOINTS/CONTROL_HZ",
          abs(P.SWEEP_S - 1.0 / P.CONTROL_HZ) < 1e-12
          and abs(P.SWEEP_S - 0.004) < 1e-9,
          f"{P.SWEEP_S*1e3:.1f} ms.  The wrong value is "
          f"{P.N_JOINTS/P.CONTROL_HZ*1e3:.0f} ms -- see the module header")
    check("SLOT_S is the per-frame share of a sweep",
          abs(P.SLOT_S - 1.0 / (P.CONTROL_HZ * P.N_JOINTS)) < 1e-12,
          f"{P.SLOT_S*1e6:.0f} us")
    check("a full sweep fits inside the driver watchdog with margin",
          P.SWEEP_S < 0.5 * P.WATCHDOG_S,
          f"{P.SWEEP_S*1e3:.0f} ms sweep vs {P.WATCHDOG_S*1e3:.0f} ms watchdog")
    check("BRAKE_PERIOD_S is the sweep, not the 48 ms that killed this track",
          abs(P.BRAKE_PERIOD_S - P.SWEEP_S) < 1e-12,
          "vmc_stand_hw.py:148 has N_JOINTS/CONTROL_HZ = 48 ms; every brake "
          "constant derived from it is 12x off, and BRAKE_HOLD_NM came out "
          "at 0.128 Nm -- less than a leg's own weight, i.e. a no-op")


def test_stability_bounds_are_derived_not_copied():
    """Recompute the bounds from J and dt so a gain cannot outlive its reason."""
    kd_bound = 2.0 * P.J_MIN / P.SWEEP_S            # sampled-damper limit
    check("the sampled-damper bound is ~4.4 Nms/rad at the true rate",
          4.0 < kd_bound < 5.0,
          f"2*J_MIN/SWEEP_S = {kd_bound:.2f} (at the wrong 48 ms it is "
          f"{2*P.J_MIN/(P.N_JOINTS/P.CONTROL_HZ):.2f}, which is what made "
          f"KD_JOINT_BRAKE=0.4 look unstable)")
    check("KD_JOINT_IMP is comfortably inside it",
          P.KD_JOINT_IMP < 0.25 * kd_bound,
          f"{P.KD_JOINT_IMP} = {100*P.KD_JOINT_IMP/kd_bound:.0f}% of the bound")
    check("KD_IMP_MAX (the CLI ceiling) is still inside it",
          P.KD_IMP_MAX < 0.5 * kd_bound,
          f"{P.KD_IMP_MAX} = {100*P.KD_IMP_MAX/kd_bound:.0f}% of the bound")

    # a discrete PD with one sample of delay wants omega_n*dt well under ~0.3
    wn = (P.KP_JOINT_IMP / P.J_KNEE) ** 0.5
    zeta = P.KD_JOINT_IMP / (2.0 * (P.KP_JOINT_IMP * P.J_KNEE) ** 0.5)
    check("KP_JOINT_IMP keeps omega_n*dt inside the delay limit",
          wn * P.SWEEP_S < 0.3,
          f"omega_n {wn:.0f} rad/s, omega_n*dt = {wn*P.SWEEP_S:.3f}")
    check("...and the pair is well damped, not ringing",
          0.5 < zeta < 1.5, f"zeta = {zeta:.2f}")
    wn_max = (P.KP_IMP_MAX / P.J_KNEE) ** 0.5
    check("KP_IMP_MAX is the edge of the measured envelope, not past it",
          wn_max * P.SWEEP_S < 0.5,
          f"kp {P.KP_IMP_MAX} -> omega_n*dt = {wn_max*P.SWEEP_S:.3f}")


def test_brake_constants_follow_the_period():
    expect = (P.BRAKE_DEADBEAT_FRAC * P.J_MIN * P.BRAKE_RELEASE_RAD_S
              / P.BRAKE_PERIOD_S)
    check("BRAKE_HOLD_NM matches its own formula at the true period",
          abs(P.BRAKE_HOLD_NM - expect) < 0.02,
          f"{P.BRAKE_HOLD_NM} vs {expect:.3f} Nm")
    check("...and is now big enough to hold a leg's own weight",
          P.BRAKE_HOLD_NM > 0.5,
          f"{P.BRAKE_HOLD_NM} Nm vs ~0.47 Nm of abduction leg gravity; the "
          f"48 ms version was 0.128 Nm")
    check("the brake stays sub-deadbeat at the release speed",
          P.BRAKE_HOLD_NM <= P.J_MIN * P.BRAKE_RELEASE_RAD_S / P.BRAKE_PERIOD_S,
          "so it can never reverse a joint on the way down")
    check("the hysteresis band is ordered",
          P.BRAKE_RELEASE_RAD_S < P.BRAKE_LATCH_RAD_S,
          f"{P.BRAKE_RELEASE_RAD_S} -> {P.BRAKE_LATCH_RAD_S} rad/s")
    check("the brake latches well below the overspeed e-stop",
          P.BRAKE_LATCH_RAD_S < 0.5 * base.QD_ESTOP,
          f"latch {P.BRAKE_LATCH_RAD_S} vs QD_ESTOP {base.QD_ESTOP} rad/s")


# ===========================================================================
# [2] torque caps and the slew override
# ===========================================================================

def test_torque_caps():
    check("the caps are ordered start < staged < hard",
          P.TAU_START_MAX < P.TAU_STAGED_MAX < P.TAU_HARD_NM,
          f"{P.TAU_START_MAX} < {P.TAU_STAGED_MAX} < {P.TAU_HARD_NM} Nm")
    check("TAU_HARD_NM is below where the current command saturates",
          P.TAU_HARD_NM < 2048 / 206.04,
          f"{P.TAU_HARD_NM} Nm vs iq saturation at "
          f"{2048/206.04:.2f} Nm (config.torque_gain = 206.04)")
    check("the staged ceiling agrees with the shared safety library",
          abs(P.TAU_STAGED_MAX - base.STAGED_TAU_MAX) < 1e-9
          and abs(P.TAU_HARD_NM - base.TAU_HARD) < 1e-9,
          f"base has {base.STAGED_TAU_MAX} / {base.TAU_HARD}")
    check("one stance leg's share fits well inside the staged cap",
          P.PER_FOOT_GRF_N * 0.15 < P.TAU_STAGED_MAX,
          f"~{P.PER_FOOT_GRF_N*0.15:.2f} Nm at a 150 mm lever vs "
          f"{P.TAU_STAGED_MAX} Nm")


def test_slew_override_is_real():
    check("TAU_SLEW_NM_S is NOT the position track's number",
          P.TAU_SLEW_NM_S > 5.0 * base.TAU_SLEW_NM_S,
          f"torque {P.TAU_SLEW_NM_S} vs base {base.TAU_SLEW_NM_S} Nm/s -- "
          f"base's caps a disturbance response at "
          f"{base.TAU_SLEW_NM_S*0.05:.2f} Nm per 50 ms")
    dqd = P.TAU_SLEW_NM_S * P.SWEEP_S * P.SWEEP_S / P.J_MIN
    check("...and is bounded in the units that matter (joint speed per sweep)",
          dqd < 0.2,
          f"{P.TAU_SLEW_NM_S} Nm/s -> {P.TAU_SLEW_NM_S*P.SWEEP_S:.2f} Nm per "
          f"sweep -> {dqd:.3f} rad/s per sweep")
    check("a full staged-cap command is reachable inside a settle window",
          P.TAU_STAGED_MAX / P.TAU_SLEW_NM_S < 0.1,
          f"{1e3*P.TAU_STAGED_MAX/P.TAU_SLEW_NM_S:.0f} ms to slew to "
          f"{P.TAU_STAGED_MAX} Nm")
    check("TORQUE_RAMP_S is shorter than base's position-preflight ramp",
          P.TORQUE_RAMP_S < base.TORQUE_RAMP_S,
          f"{P.TORQUE_RAMP_S} s vs base {base.TORQUE_RAMP_S} s -- this is the "
          f"window in which the legs have no authority")


def test_in_sweep_budget():
    check("the in-sweep block is budgeted under one CAN slot",
          P.INSWEEP_BUDGET_S < P.SLOT_S,
          f"{P.INSWEEP_BUDGET_S*1e6:.0f} us vs a {P.SLOT_S*1e6:.0f} us slot")
    check("the worker runs slower than the sweep, so torque is held between",
          P.CONTROL_UPDATE_HZ < P.CONTROL_HZ,
          f"worker {P.CONTROL_UPDATE_HZ:.0f} Hz, sweep "
          f"{P.CONTROL_HZ:.0f} Hz -- the round-robin still feeds the watchdog")
    check("WORKER_STALE_S is many worker periods, so it is not a nuisance trip",
          P.WORKER_STALE_S > 10.0 / P.CONTROL_UPDATE_HZ,
          f"{P.WORKER_STALE_S*1e3:.0f} ms vs a "
          f"{1e3/P.CONTROL_UPDATE_HZ:.0f} ms worker period")


# ===========================================================================
# [3] the safety semantics that differ from position mode
# ===========================================================================

def test_torque_mode_safety_semantics():
    check("a stance-leg latch limps the robot (it does NOT recover in place)",
          P.LATCH_LIMPS_ROBOT is True,
          "0xA1 latch = that leg stops producing torque and collapses while "
          "the other three push; 0xA4 latch merely freezes a target")
    check("LIMP and STOP are different keys",
          P.KEY_LIMP != P.KEY_STOP,
          f"limp={P.KEY_LIMP!r} stop={P.KEY_STOP!r} -- killing the process "
          f"stops the keep-alives and latches all twelve drivers")
    check("the gap abort straddles the 10 ms watchdog from below",
          P.GAP_ABORT_S < P.WATCHDOG_S and P.GAP_ABORT_S in P.GAP_BUCKETS,
          f"{P.GAP_ABORT_S*1e3:.0f} ms abort vs "
          f"{P.WATCHDOG_S*1e3:.0f} ms watchdog")
    check("MIN_PLANTED_FOR_ODOM refuses a degenerate leg-odometry solve",
          P.MIN_PLANTED_FOR_ODOM >= 3,
          f"{P.MIN_PLANTED_FOR_ODOM} feet -- below this the trunk velocity is "
          f"not determined and the worker must freeze, not publish a fiction")
    check("the load-sum tolerance is loose enough not to nuisance-trip",
          0.2 < P.LOAD_SUM_TOL_FRAC < 0.6,
          f"+/-{100*P.LOAD_SUM_TOL_FRAC:.0f}% of {P.WEIGHT_N:.1f} N")
    check("TILT_STOP_DEG matches the position track's convention",
          abs(P.TILT_STOP_DEG - 12.0) < 1e-9)


def test_defaults_are_conservative():
    check("--tau-max defaults to the low first-run cap",
          abs(P.TAU_START_MAX - 1.0) < 1e-9)
    check("leg gravity is ON by default",
          P.STANCE_LEG_GRAVITY is True,
          "--no-leg-gravity exists only to reproduce the 2026-07-30 failure")
    check("the torque rise is gentler than the position track's ramp",
          P.T_RISE > 5.0,
          f"T_RISE {P.T_RISE} s vs the position track's T_STAND 5.0 s")
    check("KP_Z is backed off from the sim-tuned value",
          P.KP_Z < 1200.0,
          f"{P.KP_Z} N/m -- the sim had no slew limit, no ZOH and no delay")


# ===========================================================================
# [4] no runner has grown its own copy
# ===========================================================================

def test_no_redeclaration():
    params = {k for k, v in vars(P).items()
              if k.isupper() and not k.startswith("_")}
    runners = sorted(pathlib.Path(_TORQUE).glob("*.py"))
    runners = [p for p in runners if p.name != "torque_params.py"]
    offenders = []
    for path in runners:
        for n in ast.parse(path.read_text()).body:
            if not (isinstance(n, ast.Assign) and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name)):
                continue
            name = n.targets[0].id
            if name in params and name not in ALLOWED_ALIASES \
                    and isinstance(n.value, ast.Constant):
                offenders.append(f"{path.name}:{n.lineno} {name}")
    check("no torque runner re-declares a torque_params literal",
          not offenders, "; ".join(offenders) or f"{len(runners)} files scanned")


def test_params_is_import_free():
    """The contract that makes this file safe to read from anywhere."""
    src = pathlib.Path(_TORQUE, "torque_params.py").read_text()
    bad = [n.lineno for n in ast.parse(src).body
           if isinstance(n, (ast.Import, ast.ImportFrom))]
    check("torque_params imports nothing (same contract as stand_params)",
          not bad, f"imports at lines {bad}" if bad else
          "safe to read from a test, a notebook or a plotting script")


def self_test():
    print("[1] the loop rate, and every bound derived from it")
    test_the_rate()
    test_stability_bounds_are_derived_not_copied()
    test_brake_constants_follow_the_period()
    print("[2] torque caps and the slew override")
    test_torque_caps()
    test_slew_override_is_real()
    test_in_sweep_budget()
    print("[3] torque-mode safety semantics")
    test_torque_mode_safety_semantics()
    test_defaults_are_conservative()
    print("[4] single source of truth")
    test_no_redeclaration()
    test_params_is_import_free()
    return report()


if __name__ == "__main__":
    sys.exit(self_test())
