#!/usr/bin/env python3
"""Offline gates for stand_ekf_schedcontact_hw (experiment B: honest contacts).

WHAT IS UNDER TEST
    The contact SCHEDULE -- which feet the EKF is told are on the floor, as a
    function of stage and ramp progress -- and the two consumers that have to
    agree with it: the schedule-masked FK height and the FK attitude.

    The control laws themselves belong to stand_ekf_level_hw and are gated in
    test_stand_ekf_level.py; nothing is re-tested here.

Run:  $V test_stand_ekf_schedcontact.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNERS = os.path.join(os.path.dirname(_HERE), "stand_postion_mode")
# ^ the hardware scripts live in stand_postion_mode/, beside this directory
for _p in (_HERE, _RUNNERS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import selftest_common as tc                             # noqa: E402
from selftest_common import check, report                # noqa: E402

import stand_ekf_level_hw as lv                          # noqa: E402
import stand_ekf_schedcontact_hw as sc                   # noqa: E402

LEGS = tc.LEGS
N_JOINTS = tc.N_JOINTS


def test_schedule():
    s = sc.ContactSchedule(("FL", "FR"), touch_frac=0.3)
    m = s.mask("CROUCH", 0.0)
    check("crouch mask plants only FL,FR",
          list(m) == [True, True, False, False], str(m.astype(int)))
    check("WAIT_CROUCH (EKF init) uses the same honest mask",
          np.array_equal(s.mask("WAIT_CROUCH", 0.0), m))
    check("early STAND keeps the rear feet up",
          np.array_equal(s.mask("STAND", 0.29), m))
    check("STAND flips the rear feet ON at touch-frac",
          bool(np.all(s.mask("STAND", 0.30))))
    check("HOLD4 is all planted", bool(np.all(s.mask("HOLD4", 0.0))))
    check("early PARK keeps all planted",
          bool(np.all(s.mask("PARK", 0.69))))
    check("late PARK lifts the rear feet (mirror of the touchdown)",
          np.array_equal(s.mask("PARK", 0.71), m))
    check("PARKED returns to the crouch set",
          np.array_equal(s.mask("PARKED", 0.0), m))

    # exactly ONE rising edge per stand: walk a full stage sequence
    edges = 0
    prev = s.mask("WAIT_CROUCH", 0.0)
    for stage, steps in (("STAND", 50), ("HOLD4", 20), ("PARK", 50),
                         ("PARKED", 5)):
        for k in range(steps):
            cur = s.mask(stage, (k + 1) / steps)
            if np.any(cur & ~prev):
                edges += 1
            prev = cur
    check("exactly one touchdown edge per stand cycle", edges == 1,
          f"{edges} rising edges")

    check("fewer than 2 planted feet is refused",
          tc.raises(lambda: sc.ContactSchedule(("FL",))))
    check("unknown leg names are refused",
          tc.raises(lambda: sc.ContactSchedule(("FL", "XX"))))


def test_masked_height(stand_height):
    """The height veto must average only the schedule's planted feet."""
    tables = tc.tables_for(stand_height, clamp_m=lv.LEVEL_CLAMP_M)
    q = np.zeros(N_JOINTS)
    for i, leg in enumerate(LEGS):
        q[3 * i:3 * i + 3] = tables[leg].q_at(-stand_height)
    # rear feet 20 mm short of the floor (the crouch situation, exaggerated)
    for i, leg in enumerate(LEGS):
        if leg in ("RL", "RR"):
            q[3 * i:3 * i + 3] = tables[leg].q_at(-stand_height + 0.020)
    mask = sc.ContactSchedule(("FL", "FR")).crouch_mask
    h_masked = lv.fk_floor_height_planted(q, None, mask, ref="imu")
    h_all = lv.fk_floor_height_planted(q, None, None, ref="imu")
    # a foot raised TOWARD the trunk pulls the all-four average DOWN by
    # lift * n_air / 4; the masked height must not move
    check("masked FK height ignores the airborne rear feet",
          abs((h_masked - h_all) - 0.020 / 2) < 1e-4,
          f"all-four biased {(h_all-h_masked)*1e3:+.1f} mm (20 mm x 2/4 legs)")
    r_fk, p_fk = lv.fk_attitude(q, mask)
    check("2-foot plane fit refuses an attitude (dashes, not garbage)",
          math.isnan(r_fk) and math.isnan(p_fk))
    return tables


def test_sequence_wiring(tables, stand_height):
    """The mask follows the stage machine through a full stand cycle."""
    sched = sc.ContactSchedule(("FL", "FR"), touch_frac=0.3)
    seq = lv.LevelStandSequence(tables, stand_height)
    seen = {}

    def sample(sq, t, cmd, contacts):
        m = sched.mask(sq.stage, sc._progress(sq, t))
        seen.setdefault(sq.stage, []).append(int(m.sum()))

    tc.walk_stages(seq, after_tick=sample)
    check("WAIT_CROUCH never reports 4 planted",
          max(seen.get("WAIT_CROUCH", [0])) == 2, str(set(seen["WAIT_CROUCH"])))
    check("STAND transitions 2 -> 4 planted",
          seen["STAND"][0] == 2 and seen["STAND"][-1] == 4)
    check("HOLD4 is 4 planted throughout",
          set(seen["HOLD4"]) == {4})
    check("PARK transitions 4 -> 2 planted",
          seen["PARK"][0] == 4 and seen["PARK"][-1] == 2)


def self_test(stand_height=lv.STAND_HEIGHT_DEFAULT):
    print("stand_ekf_schedcontact_hw self-test (no hardware)")
    print("[1] contact schedule")
    test_schedule()
    print("[2] schedule-masked FK height / attitude")
    tables = test_masked_height(stand_height)
    print("[3] schedule follows the stage machine")
    test_sequence_wiring(tables, stand_height)
    return report()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    tc.stand_height_arg(ap, lv.STAND_HEIGHT_DEFAULT)
    sys.exit(self_test(ap.parse_args().stand_height))


if __name__ == "__main__":
    main()
