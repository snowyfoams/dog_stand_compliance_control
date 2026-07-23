#!/usr/bin/env python3
"""Step-order experiments -- isolate gait-history effects on FL's UNLOAD.

Hypothesis (2026-07-23): FL's UNLOAD slip/instability may be caused by
the step ORDER, not (only) by static asymmetry.  In the walk FL always
steps IMMEDIATELY AFTER RR -- RR's foot has just moved 30-40 mm forward
and every leg's load share has changed -- so FL has never been tested
from a fresh symmetric stand inside the gait machinery.  (stand3's
3-leg holds passed for all four legs, but those had no SWING, no
TOUCHDOWN anchor commit and no prior step.)

This driver reuses walk1_hw.py UNCHANGED -- same stage machine, gates,
trips, IMU watch, offline validation -- and only chooses which legs step
and in what order:

    A. FL alone, fresh stand:       step_order_hw.py --legs FL
    B. walk context reproduced:     step_order_hw.py --legs RR,FL
    C. control, other legs after RR: step_order_hw.py --legs RR,RL
                                     step_order_hw.py --legs RR,FR

How to read the A/B result (watch FL's UNLOAD/CLEAR_GATE lines, foot
slip by eye, and the rp= roll):

    A clean, B slips  -> gait history confirmed: RR's step changes FL's
                         starting load; the shift planner needs to know
                         about it (load-aware shift or re-settle dwell).
    A slips too       -> static cause confirmed (left-side load bias /
                         tilt); order is innocent.
    both, B worse     -> both contribute; fix static first.

All other walk1 arguments pass through verbatim, e.g.:

    step_order_hw.py --legs RR,FL --step 0.030 --time-scale 0.8
    step_order_hw.py --legs FL --auto --cycles 1

Notes:
  * The custom order is validated offline (IK/limits/margins) by
    walk1's validate_cycle BEFORE the bus is armed; an infeasible
    combination refuses to run.
  * Without --auto, each ENTER performs the next listed step; after the
    list is exhausted the list repeats.  With --auto --cycles N the run
    stops after N passes over the list.
  * Body advance per step stays step/4 (walk1's 4-step convention), so
    short lists advance the body less than a full cycle would --
    irrelevant for 1-2 step experiments, slightly conservative margins.
  * The [walk1] status line's "step=k/4" label keeps walk1's 4-step
    text; the [stage] lines show the true "step k/len(list)".
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import walk1_hw  # noqa: E402


def main():
    argv = sys.argv[1:]
    if "--legs" not in argv:
        print(__doc__)
        print("error: --legs LEG[,LEG...] is required (e.g. --legs RR,FL)",
              file=sys.stderr)
        return 2
    i = argv.index("--legs")
    if i + 1 >= len(argv):
        print("error: --legs needs a value (e.g. --legs RR,FL)",
              file=sys.stderr)
        return 2
    order = tuple(s.strip().upper() for s in argv[i + 1].split(",") if s.strip())
    rest = argv[:i] + argv[i + 2:]
    bad = [leg for leg in order if leg not in walk1_hw.LEGS]
    if bad or not 1 <= len(order) <= 8:
        print(f"error: bad --legs {argv[i + 1]!r}: legs must be from "
              f"{'/'.join(walk1_hw.LEGS)}, 1-8 entries", file=sys.stderr)
        return 2
    if "--self-test" in rest:
        print("error: run walk1_hw.py --self-test directly (its dry-run "
              "asserts the full 4-step cycle); this driver validates the "
              "custom order offline before arming instead.", file=sys.stderr)
        return 2

    # walk1 reads the module-global GAIT_ORDER everywhere (validator,
    # stage machine, --auto stop count, banners) -- one override, applied
    # before any of it runs, keeps every consumer consistent.
    walk1_hw.GAIT_ORDER = order
    print(f"[step-order] EXPERIMENT: steps {'->'.join(order)} "
          f"(walk1 machinery, custom order)")

    sys.argv = [sys.argv[0]] + rest
    return walk1_hw.main()


if __name__ == "__main__":
    raise SystemExit(main())
