#!/usr/bin/env python3
"""FL-deep-shift gait: give ONLY the FL step a bigger CoM shift.

Findings that motivate this (2026-07-23, hardware):
  * Isolated single-leg tests (step_order_hw.py): FR, RL, RR each step
    cleanly even alone from a fresh stand.  FL is the ONLY problem leg
    -- at UNLOAD its foot slips back and the body wobbles, fresh stand
    or in-gait (static left-front load bias: knee torques FL 1.46/RL
    1.01 vs FR 0.06/RR 0.04 N*m, IMU -1.3 deg left-down at HOLD4).
  * The global 32 mm shift cap came from OTHER legs' constraints; the
    binding constraint for a DEEPER FL shift is FL's OWN abduction
    limit, and it tightens with pre-lift height (raising the stretched
    foot folds the hip).

The package (offline-validated envelope, step 0.030, H = 0.175):

    order RR -> FR -> RL -> FL    three proven steps first; the body
                                  advance (+22.5 mm) also buys FL reach
    FL shift: lateral 25 mm       (was 22.6 -- abd wall at 26)
              back    20 mm       (was 15)
    pre-lift ladder capped 25 mm  (start 20 + one 5 mm retry; the 30 mm
                                  ladder tail is what blocked lateral
                                  >25 -- hardware has never needed it:
                                  every clear gate passed FIRST TRY at
                                  20, corner sag 0.2 mm)
    => cycle-worst planned margin 27.3 mm (baseline 25.8)

Everything else is walk1_hw.py verbatim (stage machine, gates, trips,
IMU watch); the custom plan/order/cap are validated by walk1's
validate_cycle BEFORE the bus is armed.

Usage (other walk1 args pass through; --step defaults to 0.030 here):

    walk2_fl_hw.py                          # the package above
    walk2_fl_hw.py --legs FL --fl-back 0.015   # isolated FL re-test (a
                                  # fresh stand lacks the body advance
                                  # that pays for back 20; lat 25 stays)
    walk2_fl_hw.py --fl-lat 0.024           # back off if abd complains
    walk2_fl_hw.py --check [--step 0.030]   # offline feasibility only
    walk2_fl_hw.py --auto --cycles 1 --time-scale 0.8

What to watch on the FL step: foot slip by eye, UNLOAD/CLEAR_GATE lines
(rise should reach ~20 mm first try), and rp= roll vs the -7.2 deg /
"slip and return" baseline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import walk1_hw  # noqa: E402

DEFAULT_ORDER = ("RR", "FR", "RL", "FL")
DEFAULT_FL_LAT_M = 0.025      # abd wall at 26 (validator, prelift cap 25)
DEFAULT_FL_BACK_M = 0.020     # front reach allows it at step 0.030
DEFAULT_PRELIFT_CAP_M = 0.025


class FLShiftPlan(walk1_hw.WalkPlan):
    """WalkPlan with an FL-specific shift; other legs keep the proven one.

    fl_lat_m is the literal lateral component (the base class uses
    shift_m/sqrt(2) = 22.6 mm); fl_back_m mirrors --front-back-shift
    but for FL only.
    """

    fl_lat_m = DEFAULT_FL_LAT_M
    fl_back_m = DEFAULT_FL_BACK_M

    def shift_vec_for(self, swing):
        if swing == "FL":
            rel = self.anchors[swing] - self.neutral
            return np.array([
                -self.fl_back_m * np.sign(rel[0]),
                -self.fl_lat_m * np.sign(rel[1]),
            ])
        return super().shift_vec_for(swing)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fl-lat", type=float, default=DEFAULT_FL_LAT_M)
    parser.add_argument("--fl-back", type=float, default=DEFAULT_FL_BACK_M)
    parser.add_argument("--prelift-cap", type=float,
                        default=DEFAULT_PRELIFT_CAP_M)
    parser.add_argument("--legs", type=str,
                        default=",".join(DEFAULT_ORDER))
    parser.add_argument("--check", action="store_true",
                        help="offline validate_cycle only; no hardware")
    args, rest = parser.parse_known_args()

    if "--help" in rest or "-h" in rest:
        print(__doc__)
        print("---- underlying walk1_hw options: ----\n")
        sys.argv = [sys.argv[0], "--help"]
        return walk1_hw.main()
    if "--self-test" in rest:
        print("error: run walk1_hw.py --self-test directly (asserts the "
              "stock 4-step cycle); use --check here for the custom "
              "package.", file=sys.stderr)
        return 2

    order = tuple(s.strip().upper() for s in args.legs.split(",") if s.strip())
    if (not 1 <= len(order) <= 8
            or any(leg not in walk1_hw.LEGS for leg in order)):
        print(f"error: bad --legs {args.legs!r}: legs from "
              f"{'/'.join(walk1_hw.LEGS)}, 1-8 entries", file=sys.stderr)
        return 2
    if not 0.020 <= args.fl_lat <= 0.030:
        print("error: --fl-lat must be 0.020-0.030 (abd wall ~0.026)",
              file=sys.stderr)
        return 2
    if not 0.0 <= args.fl_back <= 0.030:
        print("error: --fl-back must be 0-0.030", file=sys.stderr)
        return 2
    if not walk1_hw.s3.PRELIFT_START_M <= args.prelift_cap <= 0.030:
        print(f"error: --prelift-cap must be "
              f"{walk1_hw.s3.PRELIFT_START_M}-0.030", file=sys.stderr)
        return 2

    # One override point each, applied before anything runs -- walk1's
    # validator, stage machine, retry ladder and banners all read these
    # module globals live.
    FLShiftPlan.fl_lat_m = args.fl_lat
    FLShiftPlan.fl_back_m = args.fl_back
    walk1_hw.WalkPlan = FLShiftPlan
    walk1_hw.GAIT_ORDER = order
    walk1_hw.s3.PRELIFT_MAX_M = args.prelift_cap

    print(f"[walk2-fl] FL-DEEP-SHIFT PACKAGE: order {'->'.join(order)}; "
          f"FL shift lat {1000 * args.fl_lat:.1f} mm / back "
          f"{1000 * args.fl_back:.1f} mm (others keep 22.6/15); pre-lift "
          f"ladder capped {1000 * args.prelift_cap:.0f} mm (all legs)")

    if "--step" not in rest:
        rest = ["--step", "0.030"] + rest

    if args.check:
        i = rest.index("--step")
        step = float(rest[i + 1])
        plan_args = dict(step_m=step, shift_m=walk1_hw.DEFAULT_SHIFT_M,
                         lift_m=walk1_hw.DEFAULT_LIFT_M,
                         stand_height=walk1_hw.DEFAULT_STAND_HEIGHT_M,
                         front_back_shift=walk1_hw.FRONT_BACK_SHIFT_M)
        try:
            margin = walk1_hw.validate_cycle(plan_args)
        except ValueError as exc:
            print(f"[walk2-fl] INFEASIBLE at step {step}: {exc}",
                  file=sys.stderr)
            return 2
        print(f"[walk2-fl] feasible at step {step}: cycle-worst planned "
              f"margin {1000 * margin:.1f} mm (gate "
              f"{1000 * walk1_hw.MIN_STEP_MARGIN_M:.0f} mm)")
        return 0

    sys.argv = [sys.argv[0]] + rest
    return walk1_hw.main()


if __name__ == "__main__":
    raise SystemExit(main())
