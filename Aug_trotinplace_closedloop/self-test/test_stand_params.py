#!/usr/bin/env python3
"""Gates for stand_params.py -- the single source of truth for every tunable.

WHAT IS UNDER TEST
    Not values (those are engineering choices) but the PROPERTY that makes the
    module worth having: that no runner has quietly grown its own copy of a
    tunable again.  That is how the mess started -- `T_STAND = 5.0` in stage 1
    AND stage 2, `STAND_HEIGHT_DEFAULT = 0.19` in stage 2b AND the lift, and
    `LEVEL_SLEW_M_S` meaning 4 mm/s in one module and 10 mm/s in another.  Each
    was correct on the day it was written and a trap the day after.

    [1] catches a re-declared literal, [2] catches a runner whose value has
    drifted from the shared one, [3] is the arithmetic that ties the numbers to
    each other, and [4] checks the two values that must agree with modules
    outside this directory.

Run:  $V test_stand_params.py
"""
from __future__ import annotations

import argparse
import ast
import os
import pathlib
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNERS = os.path.join(os.path.dirname(_HERE), "stand_postion_mode")
# ^ the hardware scripts live in stand_postion_mode/, beside this directory
for _p in (_HERE, _RUNNERS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from selftest_common import check, report               # noqa: E402

import stand_params as P                                # noqa: E402
import stand_dog5_recorded_hw as recorded               # noqa: E402
import stand_dog5_hw as base                            # noqa: E402

RUNNERS = ["stand_ekf_verify_hw.py", "stand_ekf_height_hw.py",
           "stand_ekf_level_hw.py", "stand_ekf_schedcontact_hw.py",
           "stand_ahrs_level_hw.py", "lift_ekf_contact_hw.py",
           "trot_fk_switch_hw.py"]

# names a runner may legitimately bind itself: pure aliases of another
# module's single definition, so there is nothing to drift
ALLOWED_ALIASES = {"MOTOR_IDS", "N_JOINTS", "CONTROL_HZ", "LEGS", "Q_CROUCH",
                   "POSITION_TARGET_DEG"}

PARAMS = {k: v for k, v in vars(P).items()
          if k.isupper() and not k.startswith("_")}


def test_no_redeclaration():
    """No runner may assign a LITERAL to a name stand_params already owns."""
    offenders = []
    for fn in RUNNERS:
        path = pathlib.Path(_RUNNERS) / fn
        for n in ast.parse(path.read_text()).body:
            if not (isinstance(n, ast.Assign) and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name)):
                continue
            name = n.targets[0].id
            if name in PARAMS and isinstance(n.value,
                                             (ast.Constant, ast.Tuple,
                                              ast.Dict, ast.List)):
                offenders.append(f"{fn}:{n.lineno} {name}")
    check("no runner re-declares a stand_params tunable as a literal",
          not offenders, "; ".join(offenders) or f"{len(PARAMS)} tunables clean")


def test_no_shadowing():
    """A runner that exports a tunable must export stand_params' VALUE.

    Importing the modules is the honest check: it catches a re-declaration
    that [1] would miss because it was written as an expression rather than a
    literal, and it catches an import that was renamed to something else.
    """
    import importlib
    drift = []
    for fn in RUNNERS:
        mod = importlib.import_module(fn[:-3])
        for name, want in PARAMS.items():
            got = getattr(mod, name, None)
            if got is None or name in ALLOWED_ALIASES:
                continue
            if got != want:
                drift.append(f"{fn[:-3]}.{name}={got} vs params {want}")
    check("every runner's exported tunable equals stand_params'",
          not drift, "; ".join(drift) or "no shadowing")


def test_relations():
    """The arithmetic between the numbers, which no single value can express."""
    # a slew limit is only meaningful if the clamp is several steps away
    check("height clamp is many slew-seconds deep",
          P.HEIGHT_CLAMP_M > 4 * P.HEIGHT_SLEW_M_S,
          f"{P.HEIGHT_CLAMP_M*1e3:.0f} mm at {P.HEIGHT_SLEW_M_S*1e3:.0f} mm/s "
          f"= {P.HEIGHT_CLAMP_M/P.HEIGHT_SLEW_M_S:.0f} s to saturate")
    check("leveling clamp is many slew-seconds deep",
          P.LEVEL_CLAMP_M > 2 * P.LEVEL_SLEW_M_S,
          f"{P.LEVEL_CLAMP_M*1e3:.0f} mm at {P.LEVEL_SLEW_M_S*1e3:.0f} mm/s "
          f"= {P.LEVEL_CLAMP_M/P.LEVEL_SLEW_M_S:.1f} s to saturate")
    # the deadband must be reachable: smaller than one sweep of integrator
    check("height deadband is inside the clamp", P.HEIGHT_DEADBAND_M < P.HEIGHT_CLAMP_M)
    # the FK watchdog must be no tighter than the authority the loop may use,
    # or the loop would veto itself the moment it did its job
    check("EKF-FK veto is not tighter than the height authority",
          P.HEIGHT_XCHECK_M >= P.HEIGHT_CLAMP_M,
          f"veto {P.HEIGHT_XCHECK_M*1e3:.0f} mm, clamp {P.HEIGHT_CLAMP_M*1e3:.0f} mm")
    # experiment A's whole argument: the pole is ki/(1+kp), so its tau must
    # actually beat the EKF script's or the P term was pointless
    tau_ahrs = (1.0 + P.AHRS_KP) / P.AHRS_GAIN_PER_S
    tau_ekf = 1.0 / P.LEVEL_GAIN_PER_S
    check("experiment A's tau beats the EKF script's",
          tau_ahrs < tau_ekf, f"{tau_ahrs:.1f}s vs {tau_ekf:.1f}s")
    check("experiment A's slew is the faster one",
          P.AHRS_SLEW_M_S > P.LEVEL_SLEW_M_S,
          f"{P.AHRS_SLEW_M_S*1e3:.0f} vs {P.LEVEL_SLEW_M_S*1e3:.0f} mm/s")
    # the lift must clear the contact threshold by a wide margin, or a lifted
    # foot would still read as planted
    check("commanded lift clears the contact-off threshold",
          P.LIFT_M > 3 * P.CONTACT_OFF_M,
          f"{P.LIFT_M*1e3:.0f} mm lift vs {P.CONTACT_OFF_M*1e3:.0f} mm threshold")
    # the tilt stop must be well outside the leveling authority, or the run
    # would trip before the loop could correct anything
    check("tilt-stop is outside the leveling authority",
          P.TILT_STOP_DEG > 2 * P.AGREE_VETO_DEG,
          f"stop {P.TILT_STOP_DEG:.0f} deg, agree-veto {P.AGREE_VETO_DEG:.0f} deg")
    # the trot's swing must outrun the loops it runs on top of, or the
    # leveling offsets would chase the gait instead of the floor
    check("the trot swing is the fastest foot motion of any loop",
          P.TROT_SLEW_M_S > max(P.LIFT_SLEW_M_S, P.AHRS_SLEW_M_S,
                                P.HEIGHT_SLEW_M_S, P.LEVEL_SLEW_M_S),
          f"{P.TROT_SLEW_M_S*1e3:.0f} mm/s vs the next fastest "
          f"{max(P.LIFT_SLEW_M_S, P.AHRS_SLEW_M_S)*1e3:.0f} mm/s")
    # the FK switch must be able to resolve a return more finely than the
    # clearance threshold, or a foot could read 'back' while still airborne
    check("the FK return tolerance is finer than the contact threshold",
          P.TROT_SWITCH_TOL_M < P.TROT_CONTACT_CLEAR_M,
          f"{P.TROT_SWITCH_TOL_M*1e3:.1f} mm return band inside a "
          f"{P.TROT_CONTACT_CLEAR_M*1e3:.1f} mm clearance threshold")
    # the gait's own abort must fire well before the run-stop does
    check("the trot lean-abort is well inside the run's tilt-stop",
          P.TROT_LEAN_ABORT_DEG < 0.5 * P.TILT_STOP_DEG,
          f"abort {P.TROT_LEAN_ABORT_DEG:.1f} deg, stop {P.TILT_STOP_DEG:.0f} deg")
    # a swing has to fit inside its own timeout with room for the ramps
    swing_s = 2 * P.TROT_LIFT_M / P.TROT_SLEW_M_S + P.TROT_AIR_DWELL_S
    check("the FK switch timeout is longer than a whole swing",
          P.TROT_SWITCH_TIMEOUT_S > 2 * swing_s,
          f"timeout {P.TROT_SWITCH_TIMEOUT_S:.1f}s vs a {swing_s:.2f}s swing")
    check("the settle window is shorter than the stand ramp",
          P.SETTLE_S < P.T_STAND, f"{P.SETTLE_S}s of {P.T_STAND}s")
    # the trot stands lower for a reason that is pure arithmetic: reach
    check("the trot stand height leaves room for sag + leveling + push",
          (0.2213 - P.TROT_STAND_HEIGHT_M) >= 0.013 + P.LEVEL_CLAMP_M + P.TROT_PUSH_M,
          f"{(0.2213-P.TROT_STAND_HEIGHT_M)*1e3:.0f} mm of authority vs "
          f"{(0.013+P.LEVEL_CLAMP_M+P.TROT_PUSH_M)*1e3:.0f} mm needed")
    check("... which is why it stands lower than the rest of the track",
          P.TROT_STAND_HEIGHT_M < P.STAND_HEIGHT_DEFAULT,
          f"{P.TROT_STAND_HEIGHT_M*1e3:.0f} mm vs "
          f"{P.STAND_HEIGHT_DEFAULT*1e3:.0f} mm")
    check("the two stand heights are ordered as documented",
          P.STAND_HEIGHT_DEFAULT < P.STAND_HEIGHT_STAGE12,
          f"{P.STAND_HEIGHT_DEFAULT*1e3:.0f} mm vs "
          f"{P.STAND_HEIGHT_STAGE12*1e3:.0f} mm")


def test_external_agreement():
    """The values that must match modules outside this directory."""
    check("STAND_HEIGHT_STAGE12 equals the recorded default",
          abs(P.STAND_HEIGHT_STAGE12 - recorded.DEFAULT_STAND_HEIGHT) < 1e-12,
          f"{P.STAND_HEIGHT_STAGE12} vs {recorded.DEFAULT_STAND_HEIGHT}")
    check("the temperature notice sits below base's e-stop",
          P.TEMP_NOTICE_C < base.TEMP_ESTOP,
          f"notice {P.TEMP_NOTICE_C}C, e-stop {base.TEMP_ESTOP}C")
    check("the gap histogram straddles the 10 ms driver watchdog",
          min(P.GAP_BUCKETS) < 0.010 <= max(P.GAP_BUCKETS),
          str([f"{b*1e3:.0f}" for b in P.GAP_BUCKETS]))


def self_test(stand_height=None):
    print("stand_params self-test (no hardware)")
    print(f"[1] no re-declaration ({len(PARAMS)} tunables, "
          f"{len(RUNNERS)} runners)")
    test_no_redeclaration()
    print("[2] no shadowing (values agree after import)")
    test_no_shadowing()
    print("[3] relations between the numbers")
    test_relations()
    print("[4] agreement with modules outside this directory")
    test_external_agreement()
    return report()


def main():
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    sys.exit(self_test())


if __name__ == "__main__":
    main()
