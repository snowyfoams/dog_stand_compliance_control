#!/usr/bin/env python3
"""Offline gates for lift_ekf_contact_hw (one foot off the floor, EKF in loop).

WHAT IS UNDER TEST
    `LiftManager` (the ramp, the contact threshold, one-leg-at-a-time), the
    zEKF/zFK comparison block that is this runner's whole deliverable, the
    --fake-contacts semantics, and the stage-2b control stack held together
    on a THREE-leg stance.

    This runner owns no control law -- leveling and height are
    stand_ekf_level_hw's and are gated in test_stand_ekf_level.py.  What is
    new here is the 3-leg stance and the contact bookkeeping.

Run:  $V test_lift_ekf_contact.py
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

import stand_ekf_height_hw as s2                         # noqa: E402
import stand_ekf_level_hw as lv                          # noqa: E402
import lift_ekf_contact_hw as lift_hw                    # noqa: E402
from lift_ekf_contact_hw import (LiftManager,            # noqa: E402
                                 height_compare_block, LIFT_M,
                                 LIFT_SLEW_M_S, CONTACT_OFF_M,
                                 STAND_HEIGHT_DEFAULT)

LEGS = tc.LEGS
N_JOINTS = tc.N_JOINTS
DT = tc.DT
CONTROL_HZ = tc.CONTROL_HZ

# the span this runner needs: stage-2's height clamp plus the leveling clamp,
# and (gate [1]) room for the full lift ABOVE the stand
TABLE_CLAMP_M = s2.HEIGHT_CLAMP_M + lv.LEVEL_CLAMP_M


def test_lift():
    lm = LiftManager(lift_m=0.015)
    check("lift starts inactive, all planted",
          not lm.active and all(lm.planted().values()))
    msg = lm.request("FL")
    check("lift request starts a ramp", lm.active and "lifting FL" in msg)
    t, worst_rate, prev = 0.0, 0.0, 0.0
    flipped_at = None
    while t < 5.0:
        lm.advance(t)
        worst_rate = max(worst_rate, abs(lm.cur - prev) / DT)
        prev = lm.cur
        if flipped_at is None and not lm.planted()["FL"]:
            flipped_at = lm.cur
        t += DT
    check("ramp reaches the commanded lift", abs(lm.cur - 0.015) < 1e-9)
    check("ramp respects the lift slew", worst_rate <= LIFT_SLEW_M_S + 1e-9,
          f"{worst_rate*1e3:.2f} mm/s")
    check("contact flips OFF just past the threshold",
          flipped_at is not None
          and CONTACT_OFF_M < flipped_at < CONTACT_OFF_M + 2 * LIFT_SLEW_M_S * DT,
          f"at {flipped_at*1e3:.2f} mm")
    check("only the lifted leg is airborne",
          lm.planted() == tc.planted_except("FL"))
    msg = lm.request("RR")
    check("a second leg is refused while one is lifted",
          "refused" in msg and lm.leg == "FL")
    msg = lm.request("FL")
    check("the same key lowers", "lowering" in msg and lm.goal == 0.0)
    while t < 10.0:
        lm.advance(t)
        t += DT
    check("lowered leg re-plants and clears",
          not lm.active and all(lm.planted().values()))
    lm2 = LiftManager()
    lm2.request("RL")
    for k in range(100):
        lm2.advance(k * DT)
    msg = lm2.lower("EKF stale")
    check("auto-lower reports its reason",
          msg is not None and "EKF stale" in msg and lm2.goal == 0.0)


def test_closed_loop_3leg(tables, stand_height, anchors):
    """End to end: stand, latch, lift FL under a load-shift disturbance;
    leveling + height hold on the 3 planted legs; lower; re-plant."""
    mount = (math.radians(1.5), math.radians(-0.5))
    plant = tc.PlanePlant(anchors, mount=mount)
    lvl = lv.LevelingLoop(anchors)
    hctrl = s2.HeightController()
    latch = lv.SetpointLatch()
    lifter = LiftManager(lift_m=0.015)
    target_h = stand_height + tc.PLANT_TO_BOTTOM
    SAG = 0.010
    base_z = tc.flat_base_z(stand_height)

    def measure(fz, planted):
        h, roll, pitch = plant.solve(fz, planted)
        return h + tc.PLANT_TO_BOTTOM - SAG, roll + mount[0], pitch + mount[1]

    t = 0.0
    events = []
    max_err_during_lift = 0.0
    settled_err = float("nan")
    while t < 90.0:
        lifter.advance(t)
        planted = lifter.planted()
        fz = {leg: base_z[leg] - hctrl.offset + lvl.offsets[leg]
              + (lifter.cur if leg == lifter.leg else 0.0) for leg in LEGS}
        h, r, p = measure(fz, planted)
        # setpoint latch on the settled stand
        if not latch.ready and t > lift_hw.SETTLE_S:
            if latch.add(t, r, p):
                events.append("latched")
        engaged = latch.ready
        lvl.update(t, r, p, latch.sp_roll if engaged else 0.0,
                   latch.sp_pitch if engaged else 0.0, planted, engaged)
        hctrl.update(t, h, target_h, True)
        if engaged and lifter.active:
            max_err_during_lift = max(
                max_err_during_lift,
                abs(r - latch.sp_roll), abs(p - latch.sp_pitch))
            if lifter.goal != 0.0:      # settled reading just before lowering
                settled_err = max(abs(r - latch.sp_roll),
                                  abs(p - latch.sp_pitch))
        # script: lift FL at t=10 with a 1.5 deg load-shift disturbance,
        # lower at t=55, disturbance gone once planted again
        if "latched" in events and lifter.leg is None and t < 40.0 \
                and "lifted" not in events:
            lifter.request("FL")
            plant.disturb = (math.radians(1.0), math.radians(-1.1))
            events.append("lifted")
        if "lifted" in events and t > 55.0 and lifter.goal != 0.0:
            lifter.request("FL")        # toggle -> lower
        if "lifted" in events and not lifter.active \
                and "replanted" not in events:
            plant.disturb = (0.0, 0.0)
            events.append("replanted")
        t += DT

    fz = {leg: base_z[leg] - hctrl.offset + lvl.offsets[leg] for leg in LEGS}
    h, r, p = measure(fz, tc.all_planted())
    check("sequence ran: latch -> lift -> lower -> re-plant",
          events == ["latched", "lifted", "replanted"], str(events))
    # the step disturbance transits through in full before the integrator
    # winds it out -- the claim is bounded transient + settled recovery
    check("3-leg transient stays bounded (no runaway)",
          max_err_during_lift < math.radians(2.5),
          f"max err {math.degrees(max_err_during_lift):.2f} deg")
    check("leveling settles the 3-leg stance back to the setpoint",
          settled_err < math.radians(0.35),
          f"settled err {math.degrees(settled_err):.2f} deg")
    check("attitude back at the setpoint after re-plant",
          abs(r - latch.sp_roll) < math.radians(0.3)
          and abs(p - latch.sp_pitch) < math.radians(0.3),
          f"{math.degrees(r-latch.sp_roll):+.3f}/"
          f"{math.degrees(p-latch.sp_pitch):+.3f} deg")
    check("height loop recovered the sag through it all",
          abs(h - target_h) <= s2.HEIGHT_DEADBAND_M + 1e-6,
          f"{(h-target_h)*1e3:+.2f} mm")
    check("leveling stayed inside its clamp (no saturation)",
          not lvl.saturated,
          f"max |off| {max(abs(v) for v in lvl.offsets.values())*1e3:.1f} mm")


def test_compare_block():
    """The printed comparison is the deliverable -- check what it asserts."""
    baseline = (0.2103, 0.2098)
    lines = height_compare_block("x", 0.2111, 0.2099, 3, baseline)
    check("compare block reports both heights and their gap",
          "zEKF(bottom)" in lines[1] and "zFK(bottom, 3 planted)" in lines[1]
          and "+1.2 mm" in lines[1], lines[1].strip())
    check("compare block reports the change vs the all-4 baseline",
          "+0.7 mm" in lines[2], lines[2].strip())
    check("compare block omits the baseline row when there is none",
          len(height_compare_block("x", 0.21, 0.21, 4)) == 2)
    nan = float("nan")
    check("a non-finite baseline does not crash the block",
          len(height_compare_block("x", 0.21, 0.21, 3, (nan, nan))) == 2)


def test_fake_contacts_semantics(tables):
    """--fake-contacts must change ONLY what the EKF is told."""
    lm = LiftManager(lift_m=0.020)
    lm.request("FL")
    for k in range(int(4.0 * CONTROL_HZ)):
        lm.advance(k / CONTROL_HZ)
    honest = np.array([lm.planted()[leg] for leg in LEGS])
    faked = np.ones(4, dtype=bool)
    check("honest schedule marks the lifted leg airborne",
          not honest[LEGS.index("FL")] and honest.sum() == 3)
    check("faked schedule claims all four planted", bool(faked.all()))
    # zFK must follow the HONEST set either way -- that is what keeps the
    # comparison a real measurement rather than a self-fulfilling one
    q = np.zeros(N_JOINTS)
    for i, leg in enumerate(LEGS):
        q[3*i:3*i+3] = tables[leg].q_at(-0.19)
    q[0:3] = tables["FL"].q_at(-0.19 + lm.cur)
    h_honest = lv.fk_floor_height_planted(q, None, honest, ref="hip")
    h_faked = lv.fk_floor_height_planted(q, None, faked, ref="hip")
    check("zFK on the honest set ignores the raised foot",
          abs(h_honest - lv.fk_floor_height_planted(
              np.concatenate([tables[l].q_at(-0.19) for l in LEGS]),
              None, honest, ref="hip")) < 1e-9)
    check("zFK would be biased by lift/4 if it used the faked set",
          abs((h_honest - h_faked) - lm.cur / 4) < 2e-4,
          f"{(h_honest-h_faked)*1e3:+.2f} mm vs {lm.cur/4*1e3:.2f} mm")


def test_timing(tables, anchors):
    lvl = lv.LevelingLoop(anchors)
    planted = tc.all_planted()
    lm = LiftManager()

    def sweep(k):
        lm.advance(k * 0.004)
        lvl.update(k * 0.004, 0.01, -0.01, 0.0, 0.0, planted, True)
        lz = lm.lift_z()
        for leg in LEGS:
            tables[leg].q_at(-0.19 + lvl.offsets[leg] + lz[leg])

    tc.time_sweep("leveling + lift + 4-leg lookup fit the CAN slot", sweep)


def self_test(stand_height=STAND_HEIGHT_DEFAULT):
    print("lift_ekf_contact_hw self-test (no hardware)")
    print("  (control laws are stand_ekf_level_hw's and are gated there)")
    print("[1] height tables -- room for the lift above the stand")
    tables = tc.tables_for(stand_height, clamp_m=TABLE_CLAMP_M)
    anchors = tc.anchors_of(tables)
    check("table span covers the full lift above the stand",
          all(tables[leg].z_max >= -stand_height + LIFT_M for leg in LEGS),
          "  ".join(f"{leg}[{tables[leg].z_min*1e3:.0f},"
                    f"{tables[leg].z_max*1e3:.0f}]" for leg in LEGS))
    print("[2] lift manager (ramp, contact threshold, one leg at a time)")
    test_lift()
    print("[3] the zEKF/zFK comparison block")
    test_compare_block()
    print("[4] --fake-contacts changes only what the EKF is told")
    test_fake_contacts_semantics(tables)
    print("[5] closed loop end to end, 3-leg stance")
    test_closed_loop_3leg(tables, stand_height, anchors)
    print("[6] timing (leveling + lift + 4-leg lookup)")
    test_timing(tables, anchors)
    return report()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    tc.stand_height_arg(ap, STAND_HEIGHT_DEFAULT)
    sys.exit(self_test(ap.parse_args().stand_height))


if __name__ == "__main__":
    main()
