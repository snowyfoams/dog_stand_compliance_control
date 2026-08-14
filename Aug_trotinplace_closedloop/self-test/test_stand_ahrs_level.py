#!/usr/bin/env python3
"""Offline gates for stand_ahrs_level_hw (experiment A: AHRS-only leveling).

WHAT IS UNDER TEST
    `PiLevelingLoop` -- the P+I leveling law this script adds on top of
    stand_ekf_level_hw's pure integrator -- and the startup banner that tells
    the operator what tuning is about to run.

    The gates that matter are the TUNING ones: a P term bolted onto the EKF
    script's integral gain makes the robot SLOWER, and [2] measures that
    rather than asserting it.  The stage machine and the plane plant belong to
    stage 2b and are gated in test_stand_ekf_level.py.

Run:  $V test_stand_ahrs_level.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNERS = os.path.join(os.path.dirname(_HERE), "stand_postion_mode")
# ^ the hardware scripts live in stand_postion_mode/, beside this directory
for _p in (_HERE, _RUNNERS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import selftest_common as tc                             # noqa: E402
from selftest_common import check, report                # noqa: E402

import stand_ekf_level_hw as lv                          # noqa: E402
import stand_ahrs_level_hw as ah                         # noqa: E402
from stand_ahrs_level_hw import (PiLevelingLoop, AHRS_KP,  # noqa: E402
                                 AHRS_GAIN_PER_S, AHRS_SLEW_M_S)

LEGS = tc.LEGS
DT = tc.DT

DISTURB_DEG = (2.0, -1.5)        # the standard test tilt, roll/pitch


def sim(anchors, stand_height, kp, ki, slew=AHRS_SLEW_M_S,
        tol_deg=0.3, tmax=60.0, **kw):
    """Close `PiLevelingLoop` on the plane plant; measure how fast it recovers.

    The plant sees the offsets from the PREVIOUS sweep, so the one-sample
    delay a real loop has is in here too -- without it a P term would look
    unconditionally stable, which it is not.

    Returns (loop, t_settle, worst_rate, (roll, pitch)).  `t_settle` is the
    first time the attitude enters +/-tol and STAYS inside for the rest of the
    run (None if it never does), so an overshoot back out is not scored as
    settled.
    """
    plant = tc.PlanePlant(anchors)               # mount = (0,0): AHRS = truth
    plant.disturb = tuple(math.radians(d) for d in DISTURB_DEG)
    loop = PiLevelingLoop(anchors, kp=kp, gain_per_s=ki, slew_m_s=slew, **kw)
    all_on = tc.all_planted()
    base_z = tc.flat_base_z(stand_height)
    tol = math.radians(tol_deg)
    rate = tc.RateWatch(loop.offsets)
    t_settle, t = None, 0.0
    while t < tmax:
        fz = {leg: base_z[leg] + loop.offsets[leg] for leg in LEGS}
        roll, pitch = plant.ekf_attitude(fz, all_on)
        loop.update(t, roll, pitch, 0.0, 0.0, all_on, True)
        rate.add(loop.offsets)
        t_settle = (t if t_settle is None else t_settle) \
            if (abs(roll) < tol and abs(pitch) < tol) else None
        t += DT
    fz = {leg: base_z[leg] + loop.offsets[leg] for leg in LEGS}
    return loop, t_settle, rate.worst, plant.ekf_attitude(fz, all_on)


def test_closed_loop(stand_height, anchors):
    """Plane plant + AHRS feedback + setpoint (0,0) -> physical level.

    The AHRS is modelled as TRUTH (mount calibrated out, this experiment's
    premise), so driving it to zero must level the physical trunk exactly.
    """
    loop, t_pi, rate, (roll, pitch) = sim(
        anchors, stand_height, AHRS_KP, AHRS_GAIN_PER_S)
    check("AHRS feedback levels a 2.0/-1.5 deg disturbance",
          abs(roll) < math.radians(0.3) and abs(pitch) < math.radians(0.3),
          f"residual {math.degrees(roll):+.3f}/{math.degrees(pitch):+.3f} deg")
    check("offsets stayed inside the clamp", not loop.saturated,
          f"max {max(abs(v) for v in loop.offsets.values())*1e3:.1f} mm")
    check("the rate limit bounds the P+I TOTAL, not just the integrator",
          rate <= AHRS_SLEW_M_S + 1e-9,
          f"worst {rate*1e3:.1f} mm/s of {AHRS_SLEW_M_S*1e3:.0f}")

    # --- the tuning claim, measured rather than asserted -------------------
    # Same disturbance, three laws.  If (b) is not slower than (a) the
    # ki/(1+kp) reasoning in PiLevelingLoop is wrong and the defaults with it.
    _, t_i, _, _ = sim(anchors, stand_height, 0.0, lv.LEVEL_GAIN_PER_S,
                       slew=lv.LEVEL_SLEW_M_S)
    _, t_naive, _, _ = sim(anchors, stand_height, AHRS_KP,
                           lv.LEVEL_GAIN_PER_S, slew=lv.LEVEL_SLEW_M_S)

    def _s(t):
        return f"{t:.2f}s" if t is not None else "never"
    print(f"      settle to 0.3 deg:  I-only {_s(t_i)} |  naive PI (kp on, "
          f"ki untouched) {_s(t_naive)} |  shipped {_s(t_pi)}")
    check("a P term on the OLD ki is slower, not faster (the trap)",
          t_i is not None and (t_naive is None or t_naive > t_i),
          f"{_s(t_naive)} vs {_s(t_i)} -- the pole is ki/(1+kp)")
    check("the shipped kp+ki+slew at least halves the recovery time",
          t_pi is not None and t_i is not None and t_pi < 0.5 * t_i,
          f"{_s(t_pi)} vs {_s(t_i)}")

    # --- P alone cannot finish the job -------------------------------------
    # Unity plant gain + soft deadband => err_ss = (d + kp*db) / (1 + kp).
    # This is why the integrator stays even though P does the fast work.
    _, _, _, (r_p, p_p) = sim(anchors, stand_height, AHRS_KP, 0.0)
    db = lv.LEVEL_DEADBAND_DEG
    pred = [(abs(d) + AHRS_KP * db) / (1.0 + AHRS_KP) for d in DISTURB_DEG]
    check("P-only parks at (d + kp*db)/(1+kp) -- so I is not optional",
          abs(abs(math.degrees(r_p)) - pred[0]) < 0.05
          and abs(abs(math.degrees(p_p)) - pred[1]) < 0.05,
          f"{math.degrees(r_p):+.2f}/{math.degrees(p_p):+.2f} deg, "
          f"predicted {pred[0]:+.2f}/{-pred[1]:+.2f}")

    # --- kp=0 must be the EKF script's law, bit for bit --------------------
    # The A/B against stand_ekf_level_hw only means something if this holds.
    ref = lv.LevelingLoop(anchors)
    mine = PiLevelingLoop(anchors, kp=0.0, gain_per_s=lv.LEVEL_GAIN_PER_S,
                          slew_m_s=lv.LEVEL_SLEW_M_S)
    plant = tc.PlanePlant(anchors)
    plant.disturb = tuple(math.radians(d) for d in DISTURB_DEG)
    all_on = tc.all_planted()
    base_z = tc.flat_base_z(stand_height)
    worst = 0.0
    for k in range(int(30 / DT)):
        for lp in (ref, mine):
            fz = {leg: base_z[leg] + lp.offsets[leg] for leg in LEGS}
            r, p = plant.ekf_attitude(fz, all_on)
            lp.update(k * DT, r, p, 0.0, 0.0, all_on, True)
        worst = max(worst, max(abs(mine.offsets[l] - ref.offsets[l])
                               for l in LEGS))
    check("--level-kp 0 reproduces lv.LevelingLoop exactly",
          worst < 1e-12, f"max divergence {worst*1e12:.3f} pm")


def test_freeze(anchors):
    """Stale AHRS must freeze, not unwind, WITH A LIVE P TERM.

    The P part is the new hazard, and it must be live for the test to mean
    anything: P exists only while the error does, so a freeze that simply
    dropped it would step the legs by kp*e the instant the AHRS died.  Hold
    the loop mid-transient (big error, P at full stretch) and pull the AHRS
    out from under it.  `_hold` folds P into the integrator instead.
    """
    all_on = tc.all_planted()
    loop = PiLevelingLoop(anchors)
    src = tc.FakeAhrs()
    src.roll, src.pitch = math.radians(3.0), math.radians(-2.0)
    for k in range(int(0.5 / DT)):
        r, p, active, why = src.read()
        loop.update(k * DT, r, p, 0.0, 0.0, all_on, active, why)
    p_live = max(abs(v) for v in loop.p_offsets.values())
    check("the transient really is P-dominated (the freeze test has teeth)",
          p_live > 5 * max(abs(v) for v in loop.i_offsets.values()),
          f"P {p_live*1e3:.2f} mm vs I "
          f"{max(abs(v) for v in loop.i_offsets.values())*1e3:.2f} mm")
    held = dict(loop.offsets)
    src.stale = True
    for k in range(int(2 / DT)):
        r, p, active, why = src.read()
        loop.update(0.5 + k * DT, r, p, 0.0, 0.0, all_on, active, why)
    check("stale AHRS freezes the offsets (no unwind)",
          all(abs(loop.offsets[l] - held[l]) < 1e-12 for l in LEGS)
          and loop.frozen_reason == "AHRS stale")
    check("freezing folds P into I -- held output stays the integrator state",
          all(abs(loop.i_offsets[l] - held[l]) < 1e-12 for l in LEGS)
          and all(v == 0.0 for v in loop.p_offsets.values()),
          f"P was {p_live*1e3:.2f} mm at the freeze")

    # re-engaging must not step the legs either: the AHRS comes back with the
    # error inverted, so the P target flips sign -- the rate limit is all that
    # stands between that and a jump at the foot.
    src.stale = False
    src.roll, src.pitch = math.radians(-3.0), math.radians(2.0)
    prev, worst = dict(loop.offsets), 0.0
    for k in range(int(1.0 / DT)):
        r, p, active, why = src.read()
        loop.update(2.5 + k * DT, r, p, 0.0, 0.0, all_on, active, why)
        worst = max(worst, max(abs(loop.offsets[l] - prev[l]) for l in LEGS))
        prev = dict(loop.offsets)
    check("re-engaging on an inverted error is bumpless (rate-limited)",
          worst <= AHRS_SLEW_M_S * DT + 1e-12,
          f"worst step {worst*1e6:.1f} um, limit {AHRS_SLEW_M_S*DT*1e6:.1f} um")


def test_sequence(tables, stand_height):
    seq = lv.LevelStandSequence(tables, stand_height)

    def add_extra(sq, t):
        sq.extra_z = {leg: 0.005 for leg in LEGS}

    seen, t, _, _ = tc.walk_stages(seq, before_tick=add_extra)
    check("stage order incl. park", seen == tc.STAGE_ORDER, " -> ".join(seen))
    check("PARKED lands on the recorded crouch despite leveling extra",
          float(max(abs(seq.q_cmd(t) - tc.Q_CROUCH))) < 1e-5)


def test_timing(tables, anchors):
    loop = PiLevelingLoop(anchors)
    planted = tc.all_planted()

    def sweep(k):
        loop.update(k * 0.004, 0.01, -0.01, 0.0, 0.0, planted, True)
        for leg in LEGS:
            tables[leg].q_at(-0.19 + loop.offsets[leg])

    tc.time_sweep("leveling + 4-leg lookup fit the CAN slot", sweep)


def test_banner(anchors, stand_height, clamp):
    """Render the banner at every tuning corner: it must not crash or lie.

    The corners are the ones an operator actually types -- P off, I off, both
    off -- and each takes a different branch through the string building.
    """
    corners = [("shipped PI", AHRS_KP, AHRS_GAIN_PER_S, AHRS_SLEW_M_S),
               ("EKF law, I only", 0.0, lv.LEVEL_GAIN_PER_S,
                lv.LEVEL_SLEW_M_S),
               ("P only", AHRS_KP, 0.0, AHRS_SLEW_M_S),
               ("open loop", 0.0, 0.0, AHRS_SLEW_M_S)]
    ok = True
    for name, kp, ki, slew in corners:
        args = argparse.Namespace(level_kp=kp, level_gain=ki, level_slew=slew,
                                  level_clamp=clamp,
                                  level_deadband=lv.LEVEL_DEADBAND_DEG)
        lines = ah.leveling_banner(args, anchors, stand_height)
        # only the fully open-loop corner may collapse to the summary line
        ok &= (len(lines) == 1) == (kp == 0.0 and ki == 0.0)
        ok &= all(isinstance(s, str) and s.strip() for s in lines)
        if name == "shipped PI":
            for line in lines:
                print("     " + line)
    check("the banner renders at every tuning corner and hides nothing", ok)


def self_test(stand_height=ah.STAND_HEIGHT_DEFAULT):
    print("stand_ahrs_level_hw self-test (no hardware)")
    print("[1] tables + physical-reach clamp")
    # ask for far more span than the legs have -- the table march stops at the
    # physical reach, and THAT becomes the clamp (as the runner does it)
    tables = tc.tables_for(stand_height, clamp_m=0.080)
    anchors = tc.anchors_of(tables)
    clamp = ah.auto_clamp(tables, stand_height)
    ext = min(-tables[leg].z_min - stand_height for leg in LEGS)
    check("auto clamp = extension headroom minus the margin",
          abs(clamp - (ext - 0.002)) < 1e-9,
          f"{clamp*1e3:.0f} mm of {ext*1e3:.0f} mm reach")
    check("the 80 mm table request was capped by the legs, not granted",
          ext < 0.079, f"physical reach ceiling {ext*1e3:.0f} mm below stand")
    wb = abs(anchors["FL"][0] - anchors["RL"][0])
    check("pitch authority at this stand exceeds the old 12 mm clamp's 2.0deg",
          math.degrees(math.atan2(2 * clamp, wb)) > 2.5,
          f"{math.degrees(math.atan2(2*clamp, wb)):.1f} deg")
    print("[2] AHRS-fed closed loop on the plane plant")
    test_closed_loop(stand_height, anchors)
    test_freeze(anchors)
    print("[3] stage machine")
    test_sequence(tables, stand_height)
    print("[4] timing")
    test_timing(tables, anchors)
    print("[5] startup banner (the tuning as the operator will read it)")
    test_banner(anchors, stand_height, clamp)
    return report()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    tc.stand_height_arg(ap, ah.STAND_HEIGHT_DEFAULT)
    sys.exit(self_test(ap.parse_args().stand_height))


if __name__ == "__main__":
    main()
