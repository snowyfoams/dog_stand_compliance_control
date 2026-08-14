#!/usr/bin/env python3
"""Offline gates for trot_fk_switch_hw (stage 3: FK-switched trot in place).

WHAT IS UNDER TEST
    The two things this runner owns and nothing else does:

    [2] TrotFKSwitch -- return detection from measured foot z against a
        pre-swing baseline.  The gates that matter are the ones about WHY it
        is a baseline: a measured-vs-COMMANDED test can never fire under load
        sag, and the baseline test must not fire early just because the
        command got home.
    [3] TrotGait -- the swing state machine.  Phase order, exactly one
        diagonal in the air at a time, the 4-foot overlap window actually
        existing, graceful stop vs hard abort, and the timeout that stops a
        stalled leg from hanging the gait with a foot up.
    [4] the contact schedule handed to the EKF, which must be honest
    [5] geometry: the default lift cannot rock the trunk past --lean-abort
    [6] the per-sweep cost, against the CAN slot

    The control laws are stage 2/2b's and are gated in their own suites;
    nothing is re-tested here.

THE PLANT
    `LagPlant` is deliberately crude -- a first-order lag from commanded foot
    z to measured, plus a constant sag while the foot carries load.  It is not
    a dynamics model and does not try to be: what these gates need is a
    measured z that (a) trails the command and (b) carries a sag that only
    exists in stance, because those are exactly the two effects the FK switch
    claims to handle.

Run:  $V test_trot_fk_switch.py
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
import trot_fk_switch_hw as tr                           # noqa: E402
import stand_params as P                                 # noqa: E402

LEGS = tc.LEGS
N_JOINTS = tc.N_JOINTS
DT = tc.DT

SAG_M = 0.013          # measured foot sits this far ABOVE the command in
                       # stance (the leg compresses under load).  This is
                       # stage 1's measured trunk sag, and it is the number
                       # that makes a 5 mm commanded lift buy no clearance --
                       # the plant reproduces that, so the gates see it too.
TAU_S = 0.030          # position-loop lag, commanded -> measured


def clearance_of(lift_m, push_m, sag_m=SAG_M):
    """Ground clearance a commanded (lift, push) actually buys, first order.

    Hardware 2026-08-14 found the commanded lift alone does nothing: the feet
    move relative to the TRUNK, and lifting a diagonal moves the trunk too.
    Two terms subtract, each about one sag deep --

        the swinging legs unload and extend by ~sag, reaching back down;
        the stance pair's load doubles, compressing ~sag, sinking the trunk;

    -- so clearance = lift + push - 2*sag.  Sag is taken linear in load, which
    is the crude part: it puts the two terms at one sag each rather than
    pretending to know the leg's real compliance curve.
    """
    return lift_m + push_m - 2.0 * sag_m


class LagPlant:
    """Commanded foot z -> measured joint angles, with lag and stance sag."""

    def __init__(self, tables, stand_height):
        self.tables = tables
        self.base_z = -float(stand_height)
        self.z = {leg: self.base_z + SAG_M for leg in LEGS}   # settled stance

    def step(self, offsets, planted, dt=DT):
        """Advance one sweep.  `offsets` is the gait's per-leg z shift.

        A planted leg carries load and sags; an unloaded one extends.  That
        asymmetry is the whole reason the commanded lift is not clearance.
        """
        for leg in LEGS:
            target = self.base_z + offsets[leg] + (SAG_M if planted[leg] else 0.0)
            a = dt / (TAU_S + dt)
            self.z[leg] += a * (target - self.z[leg])
        return self.q()

    def q(self):
        out = np.zeros(N_JOINTS)
        for i, leg in enumerate(LEGS):
            out[3 * i:3 * i + 3] = self.tables[leg].q_at(self.z[leg])
        return out


def drive(gait, plant, t_max=30.0, dt=DT, on_event=None, stop_at=None):
    """Run the gait against the plant.  Returns the per-sweep trace."""
    t = 0.0
    trace = []
    while t < t_max:
        q = plant.q()
        swing = gait.swing()
        offsets = gait.foot_offsets()
        # The offline plant has no EKF, so stand in for SwingClearance with
        # the physical model above: a foot is off the floor once the lift plus
        # the stance push beats twice the sag.
        push_now = -min(offsets.values()) if swing else 0.0
        air = ({leg for leg in swing}
               if clearance_of(gait.cur, push_now) > 0.0 else set())
        ev = gait.update(t, q, air)
        planted = gait.planted(air)
        plant.step(offsets, planted, dt)
        trace.append({"t": t, "phase": gait.phase, "pair": gait.pair,
                      "lift": dict(offsets), "planted": dict(planted),
                      "n_planted": sum(planted.values()), "event": ev,
                      "cycles": gait.cycles, "steps": gait.steps})
        if ev is not None and on_event is not None:
            on_event(gait, t, ev)
        if stop_at is not None and stop_at(gait, t, ev):
            break
        t += dt
    return trace


def _q_at(tables, z):
    """A full 12-joint pose from a per-leg commanded foot z."""
    q = np.zeros(N_JOINTS)
    for i, leg in enumerate(LEGS):
        q[3 * i:3 * i + 3] = tables[leg].q_at(z[leg])
    return q


# ===========================================================================
# [1] construction
# ===========================================================================

def test_construction():
    g = tr.TrotGait()
    check("default pairs are the two diagonals",
          g.pairs == (("FL", "RR"), ("FR", "RL")), str(g.pairs))
    check("a pair that is not two legs is refused",
          tc.raises(lambda: tr.TrotGait(pairs=(("FL", "RR", "FR"), ("RL",)))))
    check("an unknown leg name is refused",
          tc.raises(lambda: tr.TrotGait(pairs=(("FL", "XX"), ("FR", "RL")))))
    check("pairs that do not cover all four legs are refused",
          tc.raises(lambda: tr.TrotGait(pairs=(("FL", "RR"), ("FL", "RL")))))
    check("a fresh gait is IDLE and commands nothing",
          g.phase == "IDLE" and not g.running
          and all(v == 0.0 for v in g.foot_offsets().values())
          and g.swing() == () and sum(g.planted().values()) == 4)
    check("planted() reports the MEASURED airborne set, not the command",
          g.planted(("FL", "RR")) == {"FL": False, "FR": True,
                                      "RL": True, "RR": False},
          "a commanded lift alone can no longer flag a foot airborne")


def test_stance_push(tables):
    """The stance pair must extend while the other diagonal swings.

    Hardware 2026-08-14: without this the knees moved and the feet stayed on
    the floor.  Lifting a diagonal hands its load to the other one, which
    compresses further and sinks the trunk by about what the lift asked for.
    """
    g = tr.TrotGait(lift_m=0.020, push_m=0.015)
    check("an idle gait pushes nothing",
          all(v == 0.0 for v in g.foot_offsets().values()))

    g.phase, g.cur, g.pair_i = "AIR", 0.020, 0        # FL+RR up, full lift
    off = g.foot_offsets()
    check("the swing pair goes UP by the lift",
          off["FL"] == 0.020 and off["RR"] == 0.020)
    check("the STANCE pair goes DOWN by the push (legs extend, trunk rises)",
          off["FR"] == -0.015 and off["RL"] == -0.015,
          "sign convention: more negative foot z = leg extends")
    check("the two are zero-sum in neither direction -- the trunk nets up",
          sum(off.values()) > 0.0,
          f"{sum(off.values())*1e3:+.0f} mm summed over four legs")

    g.cur = 0.010                                     # halfway up the ramp
    half = g.foot_offsets()
    check("the push tracks the SAME ramp as the lift (no second slew to sync)",
          abs(half["FR"] / off["FR"] - half["FL"] / off["FL"]) < 1e-12,
          f"lift at {half['FL']/0.020*100:.0f}%, push at "
          f"{half['FR']/-0.015*100:.0f}%")

    g2 = tr.TrotGait(lift_m=0.020, push_m=0.0)
    g2.phase, g2.cur, g2.pair_i = "AIR", 0.020, 0
    check("--push 0 reproduces the 2026-08-14 failure exactly",
          g2.foot_offsets()["FR"] == 0.0
          and clearance_of(0.020, 0.0) < 0.0,
          f"{clearance_of(0.020, 0.0)*1e3:+.0f} mm of clearance at a 20 mm "
          "lift: the knees move and the feet stay down")
    check("... and the default push turns that same lift into a step",
          clearance_of(0.020, P.TROT_PUSH_M) > P.TROT_CONTACT_CLEAR_M,
          f"{clearance_of(0.020, P.TROT_PUSH_M)*1e3:+.0f} mm of clearance")


def test_body_trim(stand_height):
    """A body trim slides the trunk over the footprint, feet the other way.

    This is the answer to the 2026-08-14 asymmetry (RR cleared, FL did not):
    while a diagonal swings, rotation about the stance line is unactuated, so
    the only fix is to put the CoM on that line -- and since the two diagonals
    cross at the polygon centre, one constant trim serves both.
    """
    shift = 0.010                                   # body 10 mm forward
    plain = tc.tables_for(stand_height, clamp_m=lv.LEVEL_CLAMP_M,
                          announce=False)
    moved = tr.s2.build_tables(stand_height, clamp_m=lv.LEVEL_CLAMP_M,
                               xy_offset=(-shift, 0.0))
    dx = [moved[leg].xy[0] - plain[leg].xy[0] for leg in LEGS]
    check("every foot moves by the same x, opposite the body",
          all(abs(v + shift) < 1e-12 for v in dx),
          f"feet {dx[0]*1e3:+.1f} mm for a body shift of {shift*1e3:+.0f} mm")
    check("... and y is untouched",
          all(abs(moved[leg].xy[1] - plain[leg].xy[1]) < 1e-12 for leg in LEGS))

    # the shifted table must still be a valid table at the stand pose
    for leg in LEGS:
        q = moved[leg].q_at(-stand_height)
        f = tr.dog5_kinematics.foot_position(leg, q)
        ok = (abs(f[0] - moved[leg].xy[0]) < 5e-4
              and abs(f[2] + stand_height) < 5e-4)
        if not ok:
            check(f"{leg}: shifted table still solves the stand pose", False,
                  f"FK {f*1e3} vs anchor {moved[leg].xy*1e3}")
            break
    else:
        check("the shifted tables still solve the stand pose exactly",
              True, "FK of q_at(-stand) lands on the shifted anchor")
    check("a trim far past the leg's reach is refused, not silently clipped",
          tc.raises(lambda: tr.s2.build_tables(stand_height,
                                               xy_offset=(0.30, 0.0))))


def test_load_balance(tables, stand_height):
    """Per-leg load share and the CoM, from sag alone."""
    anchors = tc.anchors_of(tables)
    bal = tr.StanceLoadBalance(anchors_xy=anchors)
    q_cmd = _q_at(tables, {leg: -stand_height for leg in LEGS})

    # equal sag on all four -> CoM at the polygon centre -> no trim wanted
    q_even = _q_at(tables, {leg: -stand_height + 0.013 for leg in LEGS})
    t = 0.0
    while not bal.note_stance(q_even, q_cmd, t):
        t += DT
    check("equal sag reads as an equal load share",
          all(abs(bal.share[leg] - 0.25) < 0.02 for leg in LEGS),
          "  ".join(f"{leg} {bal.share[leg]*100:.0f}%" for leg in LEGS))
    check("... and asks for no trim",
          max(abs(v) for v in bal.trim()) < 1e-3,
          f"trim {bal.trim()*1e3} mm")

    # now load the FRONT: FL and FR sag more.  The CoM must move forward, and
    # the trim must point BACKWARD to recentre it.
    bal.reset()
    q_front = _q_at(tables, {leg: -stand_height
                             + (0.020 if leg in ("FL", "FR") else 0.006)
                             for leg in LEGS})
    t = 0.0
    while not bal.note_stance(q_front, q_cmd, t):
        t += DT
    front = bal.share["FL"] + bal.share["FR"]
    check("more sag on the front legs reads as more load on the front",
          front > 0.7, f"front carries {front*100:.0f}%")
    check("... so the CoM sits forward of the polygon centre",
          -bal.trim()[0] > 0.0, f"CoM x {-bal.trim()[0]*1e3:+.1f} mm forward")
    check("... and the suggested trim moves the BODY back to recentre it",
          bal.trim()[0] < 0.0,
          f"--com-shift-x {bal.trim()[0]:+.4f}")

    # a left-loaded robot: the 2026-08-14 symptom is a lean toward FL
    bal.reset()
    q_left = _q_at(tables, {leg: -stand_height
                            + (0.020 if leg in ("FL", "RL") else 0.006)
                            for leg in LEGS})
    t = 0.0
    while not bal.note_stance(q_left, q_cmd, t):
        t += DT
    check("a left-loaded robot asks to be trimmed right",
          -bal.trim()[1] > 0.0 and bal.trim()[1] < 0.0,
          f"CoM y {-bal.trim()[1]*1e3:+.1f} mm left -> "
          f"--com-shift-y {bal.trim()[1]:+.4f}")

    # no measurable sag must not produce a confident answer
    bal.reset()
    t = 0.0
    for _ in range(200):
        bal.note_stance(q_cmd, q_cmd, t)
        t += DT
    check("no measurable sag yields no estimate rather than a wrong one",
          bal.share is None and bal.trim() is None,
          "; ".join(bal.report()))

    # --- the three guards the 2026-08-14 run needed -----------------------
    # [a] A MOVING command must produce nothing.  This is the one that
    # failed on hardware: run while the height loop winds 13 mm and the
    # leveling loop winds +/-12 mm, and `sag` is velocity lag on a moving
    # reference, not load.  It reported "RL 100%, trim the body 342 mm".
    bal.reset()
    t = 0.0
    for k in range(400):
        creep = -stand_height - k * (P.HEIGHT_SLEW_M_S * DT)   # the height loop
        moving = _q_at(tables, {leg: creep for leg in LEGS})
        settled = _q_at(tables, {leg: creep + 0.013 for leg in LEGS})
        bal.note_stance(settled, moving, t)
        t += DT
    check("a command that is still ramping yields NO estimate",
          bal.share is None and bal.moved,
          "; ".join(bal.report()))
    check("... and the run is told to wait rather than given a number",
          "ramping" in bal.report()[0], bal.report()[0])

    # [b] A negative share must be refused, not floored to zero -- flooring
    # is what let one leg reach 100% and the trim reach a full leg anchor.
    bal.reset()
    q_neg = _q_at(tables, {leg: -stand_height
                           + (0.013 if leg != "FL" else -0.004)
                           for leg in LEGS})
    t = 0.0
    for _ in range(400):
        bal.note_stance(q_neg, q_cmd, t)
        t += DT
    check("a leg reading NEGATIVE sag refuses the whole estimate",
          bal.share is None and "negative" in " ".join(bal.report()),
          "; ".join(bal.report()))

    # [c] a trim can never exceed the footprint it is trimming within
    for case in ("front", "left"):
        bal.reset()
        legs = ("FL", "FR") if case == "front" else ("FL", "RL")
        q = _q_at(tables, {leg: -stand_height
                           + (0.020 if leg in legs else 0.006)
                           for leg in LEGS})
        t = 0.0
        while not bal.note_stance(q, q_cmd, t):
            t += DT
        half = max(abs(anchors[leg][0]) for leg in LEGS)
        check(f"the {case}-loaded trim stays inside the footprint",
              max(abs(v) for v in bal.trim()) <= half + 1e-9,
              f"trim {np.round(bal.trim()*1e3, 1)} mm vs a "
              f"{half*1e3:.0f} mm half-footprint")

    # [d] the print gate: once, then only on a real change
    bal.reset()
    q_even2 = _q_at(tables, {leg: -stand_height + 0.013 for leg in LEGS})
    t = 0.0
    while not bal.note_stance(q_even2, q_cmd, t):
        t += DT
    first = bal.worth_printing()
    again = bal.worth_printing()
    check("the load report prints once, not every window",
          first and not again,
          "the hardware log was mostly this block repeating")


def test_reach_budget(tables, stand_height):
    """The push spends extension authority, and it is the term that runs out.

    A push that does not fit fails SILENTLY -- the z table clips and the feet
    quietly do not clear, which looks exactly like having no push at all.  So
    gate the budget, and gate that the documented remedy (stand lower) works.
    """
    deep = min(tc.tables_for(stand_height, clamp_m=0.120,
                             announce=False)[leg].z_min for leg in LEGS)
    check("the leg's reach is the documented -221 mm", abs(deep + 0.2213) < 1e-3,
          f"deepest commanded foot z {deep*1e3:.1f} mm")

    need = 0.013 + P.LEVEL_CLAMP_M + P.TROT_PUSH_M     # sag + leveling + push
    auth_19 = -deep - 0.19
    auth_17 = -deep - 0.17
    check("at the 0.19 default stand the push does NOT fit",
          auth_19 < need,
          f"{auth_19*1e3:.0f} mm of authority vs {need*1e3:.0f} mm needed "
          f"(13 sag + {P.LEVEL_CLAMP_M*1e3:.0f} leveling + "
          f"{P.TROT_PUSH_M*1e3:.0f} push) -- the documented remedy is to "
          "stand lower, not to shrink the push")
    check("... and at 0.17 it does",
          auth_17 >= need,
          f"{auth_17*1e3:.0f} mm of authority vs {need*1e3:.0f} mm needed")


# ===========================================================================
# [2] the FK switch
# ===========================================================================

def test_fk_switch(tables, stand_height):
    plant = LagPlant(tables, stand_height)
    sw = tr.TrotFKSwitch()
    pair = ("FL", "RR")

    # settle, then arm in stance -- the baseline carries the sag
    for _ in range(200):
        plant.step({leg: 0.0 for leg in LEGS}, tc.all_planted())
    q0 = plant.q()
    sw.arm(q0, pair, 0.0)
    base_fl = sw.baseline["FL"]
    cmd_fl = -stand_height
    check("the baseline is the MEASURED stance z, sag included",
          abs((base_fl - cmd_fl) - SAG_M) < 2e-4,
          f"baseline {base_fl*1e3:.2f} mm vs command {cmd_fl*1e3:.2f} mm "
          f"= {(base_fl-cmd_fl)*1e3:+.2f} mm of sag")
    # the hold timer starts at the first in-band SAMPLE, not at arm(): a leg
    # that never left still has to be observed home for the full hold
    check("an unmoved leg fires after one hold, not instantly",
          sw.update(q0, 0.0) is None
          and sw.update(q0, P.TROT_SWITCH_HOLD_S + DT) == "returned")

    # the point of the baseline: measured-vs-COMMANDED never comes inside tol
    check("a measured-vs-commanded test could never fire (why the baseline "
          "exists)", abs(base_fl - cmd_fl) > 3 * P.TROT_SWITCH_TOL_M,
          f"{abs(base_fl-cmd_fl)*1e3:.2f} mm of sag vs a "
          f"{P.TROT_SWITCH_TOL_M*1e3:.1f} mm tolerance")

    # lift the pair, then bring it back, and watch WHEN it fires
    sw = tr.TrotFKSwitch()
    sw.arm(plant.q(), pair, 0.0)
    lift = {leg: (P.TROT_LIFT_M if leg in pair else 0.0) for leg in LEGS}
    planted = {leg: leg not in pair for leg in LEGS}
    fired_up = False
    for k in range(60):
        plant.step(lift, planted)
        if sw.update(plant.q(), k * DT) == "returned":
            fired_up = True
    check("it does not fire while the pair is up", not fired_up,
          f"{sw.error_m(plant.q())*1e3:.1f} mm from baseline at full lift")

    # command home; the leg lags, and the sag only returns as it re-loads
    zero = {leg: 0.0 for leg in LEGS}
    t = 60 * DT
    fired_t = None
    for k in range(400):
        plant.step(zero, tc.all_planted())
        t += DT
        if fired_t is None and sw.update(plant.q(), t) == "returned":
            fired_t = t
    check("it fires once the pair is back under load", fired_t is not None,
          f"at t={fired_t:.3f}s" if fired_t else "never fired")
    if fired_t is not None:
        # it must not fire the instant the COMMAND got home: the plant needs
        # a few lag time constants to close the last millimetre
        check("it waits for the leg, not for the command",
              fired_t - 60 * DT > TAU_S,
              f"{(fired_t-60*DT)*1e3:.0f} ms after the command came home "
              f"(lag tau {TAU_S*1e3:.0f} ms)")

    # debounce: leaving the band restarts the hold
    sw = tr.TrotFKSwitch(tol_m=0.001, hold_s=0.10)
    q_home = plant.q()
    sw.arm(q_home, pair, 0.0)
    sw.update(q_home, 0.05)                       # 50 ms inside the band
    far = tables["FL"].q_at(plant.z["FL"] + 0.010)
    q_far = q_home.copy()
    q_far[0:3] = far
    sw.update(q_far, 0.06)                        # popped back out
    check("leaving the band restarts the hold timer",
          sw.update(q_home, 0.12) is None,
          "0.12 s armed but only 0.06 s continuously inside")
    check("... and it fires once the hold is served uninterrupted",
          sw.update(q_home, 0.23) == "returned")

    # timeout
    sw = tr.TrotFKSwitch(timeout_s=0.5)
    sw.arm(q_home, pair, 0.0)
    check("a leg that never returns times out", sw.update(q_far, 0.51) == "timeout")
    sw2 = tr.TrotFKSwitch(timeout_s=0.5)
    sw2.arm(q_home, pair, 0.0)
    check("... with nothing reported early", sw2.update(q_far, 0.49) is None)


# ===========================================================================
# [3] the swing state machine
# ===========================================================================

CLEARING_LIFT_M = 0.020   # with the default 15 mm push this clears by 9 mm,
                          # so the plant reports feet in the air.  The default
                          # 5 mm lift does not, which is its own gate below.


def test_phases(tables, stand_height):
    plant = LagPlant(tables, stand_height)
    gait = tr.TrotGait(lift_m=CLEARING_LIFT_M)
    gait.start(plant.q(), 0.0)
    trace = drive(gait, plant, t_max=12.0,
                  stop_at=lambda g, t, ev: g.cycles >= 2)

    phases = []
    for row in trace:
        if not phases or phases[-1] != row["phase"]:
            phases.append(row["phase"])
    check("phase order is LIFT -> AIR -> LOWER -> SETTLE -> OVERLAP",
          phases[:5] == ["LIFT", "AIR", "LOWER", "SETTLE", "OVERLAP"],
          " -> ".join(phases[:6]))

    pairs_stepped = [row["pair"] for row in trace if row["event"] == "step_done"]
    check("the pairs alternate, diagonal by diagonal",
          pairs_stepped[:4] == [("FL", "RR"), ("FR", "RL"),
                                ("FL", "RR"), ("FR", "RL")],
          " ".join("+".join(p) for p in pairs_stepped[:4]))
    check("two steps make one cycle",
          gait.cycles == 2 and gait.steps == 4,
          f"{gait.steps} steps, {gait.cycles} cycles")

    # never more than one diagonal in the air, and always the right one
    worst = 4
    wrong = 0
    for row in trace:
        up = {leg for leg in LEGS if not row["planted"][leg]}
        worst = min(worst, row["n_planted"])
        if up and up != set(row["pair"]):
            wrong += 1
    check("never fewer than 2 planted feet", worst == 2, f"min {worst}")
    check("the airborne set is always exactly the active diagonal", wrong == 0,
          f"{wrong} sweeps disagreed")

    # the 4-foot overlap window has to actually exist
    four = [row for row in trace if row["n_planted"] == 4]
    check("a 4-foot window exists between steps", len(four) > 0,
          f"{len(four)*DT:.2f}s of the {len(trace)*DT:.2f}s run has all 4 down")
    # ... and each SETTLE+OVERLAP run of it is at least the commanded overlap
    runs, cur = [], 0
    for row in trace:
        if row["n_planted"] == 4:
            cur += 1
        elif cur:
            runs.append(cur * DT)
            cur = 0
    check("every 4-foot window is at least the commanded overlap",
          all(r >= P.TROT_OVERLAP_S - 2 * DT for r in runs[1:-1] or [1e9]),
          f"windows {[f'{r:.2f}' for r in runs[:4]]}s, "
          f"commanded {P.TROT_OVERLAP_S:.2f}s")

    peak = max(max(row["lift"].values()) for row in trace)
    check("the commanded lift reaches its target and no further",
          abs(peak - CLEARING_LIFT_M) < 1e-9, f"{peak*1e3:.2f} mm")

    # the same walk at the DEFAULT lift must never claim a foot is airborne
    plant2 = LagPlant(tables, stand_height)
    gait2 = tr.TrotGait()
    gait2.start(plant2.q(), 0.0)
    t2 = drive(gait2, plant2, t_max=12.0, stop_at=lambda g, t, ev: g.cycles >= 1)
    check("at the DEFAULT lift the gait runs with all four feet planted",
          all(row["n_planted"] == 4 for row in t2)
          and gait2.steps >= 2,
          f"{gait2.steps} steps taken, never fewer than "
          f"{min(row['n_planted'] for row in t2)} planted -- a rock, not a step")
    return trace


def test_stop_and_abort(tables, stand_height):
    # graceful stop: the CURRENT step finishes and the feet come down
    plant = LagPlant(tables, stand_height)
    gait = tr.TrotGait()
    gait.start(plant.q(), 0.0)
    asked = {"t": None}

    def maybe_stop(g, t, ev):
        if asked["t"] is None and g.phase == "AIR":
            gait.stop()
            asked["t"] = t

    trace = drive(gait, plant, t_max=15.0, on_event=None,
                  stop_at=lambda g, t, ev: (maybe_stop(g, t, ev)
                                            or ev == "stopped"))
    check("a graceful stop lands IDLE with every foot down",
          gait.phase == "IDLE" and gait.swing() == ()
          and all(v == 0.0 for v in gait.foot_offsets().values()),
          f"phase {gait.phase}, swing {gait.swing()}")
    check("... after finishing the cycle it was in the middle of",
          gait.cycles >= 1 and trace[-1]["event"] == "stopped",
          f"{gait.steps} steps, {gait.cycles} cycles")

    # hard abort mid-air: the feet ramp down at the slew limit, no jump
    plant = LagPlant(tables, stand_height)
    gait = tr.TrotGait()
    gait.start(plant.q(), 0.0)
    t = 0.0
    while gait.phase != "AIR":
        gait.update(t, plant.q())
        plant.step(gait.foot_offsets(), gait.planted())
        t += DT
    mid = gait.cur
    gait.abort("lean 5.0deg")
    check("abort from mid-air enters ABORT holding the current lift",
          gait.phase == "ABORT" and abs(gait.cur - mid) < 1e-12)
    prev, worst_rate, ev = gait.cur, 0.0, None
    for _ in range(2000):
        ev = gait.update(t, plant.q()) or ev
        worst_rate = max(worst_rate, abs(gait.cur - prev) / DT)
        prev = gait.cur
        plant.step(gait.foot_offsets(), gait.planted())
        t += DT
        if gait.phase == "IDLE":
            break
    check("abort ramps the feet down at the slew limit, never jumps",
          worst_rate <= P.TROT_SLEW_M_S + 1e-9,
          f"worst {worst_rate*1e3:.1f} mm/s vs limit "
          f"{P.TROT_SLEW_M_S*1e3:.0f} mm/s")
    check("abort ends IDLE, feet down, and says why",
          gait.phase == "IDLE" and gait.cur == 0.0
          and str(ev).startswith("aborted:lean"), str(ev))

    # a leg that never returns must abort, not hang with a foot up
    frozen = LagPlant(tables, stand_height)
    gait = tr.TrotGait(switch=tr.TrotFKSwitch(timeout_s=0.5))
    gait.start(frozen.q(), 0.0)
    t, saw_timeout = 0.0, False
    stuck_q = None
    for _ in range(4000):
        q = frozen.q() if stuck_q is None else stuck_q
        ev = gait.update(t, q)
        if gait.phase == "SETTLE" and stuck_q is None:
            # freeze the encoders 10 mm off baseline: the leg never comes back
            stuck_q = frozen.q().copy()
            stuck_q[0:3] = tables["FL"].q_at(frozen.z["FL"] + 0.010)
        if ev == "timeout":
            saw_timeout = True
        frozen.step(gait.foot_offsets(), gait.planted())
        t += DT
        if gait.phase == "IDLE" and saw_timeout:
            break
    check("a stalled leg times out instead of hanging the gait", saw_timeout)
    check("... and the timeout brings every foot down",
          gait.phase == "IDLE" and gait.swing() == () and gait.cur == 0.0)


def test_cycle_limit(tables, stand_height):
    plant = LagPlant(tables, stand_height)
    gait = tr.TrotGait(max_cycles=2)
    gait.start(plant.q(), 0.0)
    drive(gait, plant, t_max=20.0, stop_at=lambda g, t, ev: ev == "stopped")
    check("--cycles stops the gait after exactly that many cycles",
          gait.cycles == 2 and gait.phase == "IDLE",
          f"{gait.cycles} cycles, phase {gait.phase}")


# ===========================================================================
# [4] SwingClearance -- the measured contact schedule
# ===========================================================================

def test_clearance(tables, stand_height):
    """The measurement that decides what the EKF is told about contacts."""
    H = stand_height
    pair = ("FL", "RR")
    eye = np.eye(3)
    stance_z = {leg: -H + SAG_M for leg in LEGS}
    q_stance = _q_at(tables, stance_z)

    sc = tr.SwingClearance()
    check("with no stance reference yet, nothing is reported airborne",
          sc.airborne(q_stance, eye, pair) == set()
          and sc.clearance(q_stance, eye, pair) is None,
          "conservative default: the schedule stages 1/2/2b validated")

    # the reference latches only after TROT_STANCE_REF_S of 4-foot stance
    t = 0.0
    latched_at = None
    while t < 2 * P.TROT_STANCE_REF_S:
        if sc.note_stance(q_stance, eye, t) and latched_at is None:
            latched_at = t
        t += DT
    check("the stance reference latches after the settling window",
          latched_at is not None
          and abs(latched_at - P.TROT_STANCE_REF_S) < 2 * DT,
          f"latched at {latched_at:.3f}s, window {P.TROT_STANCE_REF_S:.2f}s")

    sc2 = tr.SwingClearance()
    for k in range(500):
        sc2.note_stance(q_stance, None, k * DT)
    check("no EKF attitude -> no reference -> everything stays planted",
          sc2.ref is None and sc2.airborne(q_stance, None, pair) == set())

    def clear_at(lift_m, C=eye):
        """Swing pair commanded up `lift_m`, unloaded so it extends by SAG."""
        z = dict(stance_z)
        for leg in pair:
            z[leg] = -H + lift_m          # no sag: the leg carries no load
        return sc.clearance(_q_at(tables, z), C, pair)

    # THE headline result: the default lift buys negative clearance
    c_default = clear_at(P.TROT_LIFT_M)
    worst = min(c_default.values())
    check("the DEFAULT lift measures NEGATIVE clearance (the feet stay down)",
          worst < 0.0,
          f"{P.TROT_LIFT_M*1e3:.0f} mm commanded - {SAG_M*1e3:.0f} mm of "
          f"unload extension = {worst*1e3:+.1f} mm above the stance plane")
    check("... so no foot is reported airborne at the default lift",
          sc.airborne(_q_at(tables, {**stance_z,
                                     **{leg: -H + P.TROT_LIFT_M for leg in pair}}),
                      eye, pair) == set(),
          "a command-derived flag would have lied to the EKF here")

    # and a lift that beats the sag does clear, by the difference
    big = 0.025
    c_big = clear_at(big)
    check("a lift past the sag measures positive clearance, lift - sag",
          abs(min(c_big.values()) - (big - SAG_M)) < 5e-4,
          f"{big*1e3:.0f} mm commanded -> {min(c_big.values())*1e3:+.1f} mm "
          f"measured (expected {(big-SAG_M)*1e3:+.1f})")
    check("... and both feet of the pair are then reported airborne",
          sc.airborne(_q_at(tables, {**stance_z,
                                     **{leg: -H + big for leg in pair}}),
                      eye, pair) == set(pair))

    # the trunk's own height must cancel -- only attitude matters
    sunk = {leg: v - 0.010 for leg, v in stance_z.items()}
    for leg in pair:
        sunk[leg] = -H + big - 0.010
    c_sunk = sc.clearance(_q_at(tables, sunk), eye, pair)
    # 0.1 mm, not exact: the z->q tables are sampled at TABLE_STEP_M and
    # interpolated, so FK of q_at(z) lands a few microns off z
    worst_shift = max(abs(c_sunk[leg] - c_big[leg]) for leg in pair)
    check("clearance is independent of trunk height (r cancels)",
          worst_shift < 1e-4,
          f"10 mm of trunk sink moved it {worst_shift*1e6:.0f} um "
          "(table interpolation, not the measurement)")

    # ... and the tip is the term FK alone could never see
    z_big = {**stance_z, **{leg: -H + big for leg in pair}}
    q_big = _q_at(tables, z_big)
    c_level = min(sc.clearance(q_big, eye, pair).values())
    c_tip = min(sc.clearance(q_big, tc.C_from_rp(0.0, math.radians(2.0)),
                             pair).values())
    check("trunk tip eats clearance -- the term only the EKF can supply",
          c_tip < c_level - 1e-4,
          f"level {c_level*1e3:+.1f} mm -> 2 deg pitched {c_tip*1e3:+.1f} mm")


# ===========================================================================
# [4b] the contact schedule handed to the EKF
# ===========================================================================

def test_contacts(trace):
    """Honest contacts: 2 planted through a swing, 4 otherwise, no chatter."""
    edges = 0
    prev = None
    for row in trace:
        mask = np.array([row["planted"][leg] for leg in LEGS])
        if prev is not None and np.any(mask & ~prev):
            edges += 1
        prev = mask
    steps = sum(1 for row in trace if row["event"] == "step_done")
    check("exactly one touchdown edge per step (no contact chatter)",
          edges == steps, f"{edges} rising edges over {steps} steps")

    # a lifted foot must be reported airborne for most of its travel
    air = sum(1 for row in trace if row["n_planted"] == 2)
    lifted = sum(1 for row in trace
                 if max(row["lift"].values()) > 1e-9)
    check("a swinging foot reads airborne only above the unload extension",
          0.0 < air < lifted,
          f"{air*DT:.2f}s airborne of {lifted*DT:.2f}s with a commanded lift "
          f"-- the first and last {SAG_M*1e3:.0f} mm of every ramp are still "
          "on the floor, which is the physics, not a threshold")

    # the 2-planted mask is what stage 2b's FK attitude legitimately refuses
    mask2 = np.array([True, False, False, True])       # FL, RR down
    r_fk, p_fk = lv.fk_attitude(np.zeros(N_JOINTS), mask2)
    check("FK refuses an attitude on a diagonal pair (why roll is EKF-only "
          "here)", math.isnan(r_fk) and math.isnan(p_fk))


# ===========================================================================
# [5] geometry -- what the default lift can do to the trunk
# ===========================================================================

def test_rock_geometry(tables):
    """Lifting a diagonal on stiff legs tips the trunk about the other one.

    The trunk rotates about the line through the two stance feet until the
    lifted feet touch down again, so the peak attitude is bounded by
    atan(lift / d), where d is the lifted foot's distance from that line.
    This is the gate that says the DEFAULT lift cannot, geometrically, rock
    the robot past its own lean-abort threshold -- i.e. the abort exists to
    catch something going wrong, not to catch the gait working.
    """
    a = tc.anchors_of(tables)
    p, r = np.array(a["FL"][:2]), np.array(a["RR"][:2])
    axis = (r - p) / np.linalg.norm(r - p)
    # perpendicular distance from the FL-RR line, by the 2-D cross product
    # (numpy 2 dropped the 2-vector form of np.cross)
    def _off(leg):
        v = np.array(a[leg][:2]) - p
        return abs(axis[0] * v[1] - axis[1] * v[0])
    d = min(_off(leg) for leg in ("FR", "RL"))
    rock = math.degrees(math.atan2(P.TROT_LIFT_M, d))
    check("the default lift's worst-case rock is inside the lean-abort",
          rock < P.TROT_LEAN_ABORT_DEG,
          f"{P.TROT_LIFT_M*1e3:.0f} mm at {d*1e3:.0f} mm from the stance "
          f"diagonal = {rock:.2f} deg, abort at {P.TROT_LEAN_ABORT_DEG:.1f} deg")
    check("... and well inside the run's tilt-stop",
          rock < 0.5 * P.TILT_STOP_DEG,
          f"{rock:.2f} deg vs {P.TILT_STOP_DEG:.0f} deg")
    # the lift the abort WOULD catch, for the operator raising --lift
    lift_at_abort = d * math.tan(math.radians(P.TROT_LEAN_ABORT_DEG))
    check("the lean-abort corresponds to a sane maximum --lift",
          lift_at_abort > 2 * P.TROT_LIFT_M,
          f"--lean-abort {P.TROT_LEAN_ABORT_DEG:.1f} deg trips at about "
          f"{lift_at_abort*1e3:.0f} mm of lift")
    # THE gate that justifies the stance push existing.  Without it clearance
    # is lift - 26 mm, so the smallest stepping lift is 26 mm -- whose rocking
    # bound is already past the abort.  Hardware 2026-08-14 hit exactly this:
    # the knees moved and the feet stayed on the floor.
    push_free_lift = 2 * SAG_M
    check("with NO stance push there is no lift that steps inside the abort",
          push_free_lift > lift_at_abort,
          f"needs {push_free_lift*1e3:.0f} mm of lift "
          f"({math.degrees(math.atan2(push_free_lift, d)):.1f} deg bound) but "
          f"the abort trips at {lift_at_abort*1e3:.0f} mm -- the push is not "
          "an optimisation, it is what makes stepping reachable")
    # ... and with the default push, a usable window opens
    for lift in (0.020, 0.025):
        c = clearance_of(lift, P.TROT_PUSH_M)
        rock_real = math.degrees(math.atan2(c, d))
        check(f"a {lift*1e3:.0f} mm lift + {P.TROT_PUSH_M*1e3:.0f} mm push "
              "steps inside the abort",
              c > 0.0 and rock_real < P.TROT_LEAN_ABORT_DEG,
              f"{c*1e3:.0f} mm of clearance = {rock_real:.2f} deg of rock "
              f"(worst-case bound {math.degrees(math.atan2(lift, d)):.2f})")


def test_leveling_freezes(tables):
    """The leveling loop must hold its offsets while a diagonal is up.

    Not a mechanism this runner adds -- `LevelingLoop.update` refuses to
    integrate below 3 planted feet.  Gated here because the gait is the first
    thing that actually produces a 2-planted state in flight, and a future
    edit to that rule would break the gait silently.
    """
    lvl = lv.LevelingLoop(anchors_xy=tc.anchors_of(tables))
    t = 0.0
    for _ in range(200):                       # wind something in on 4 feet
        lvl.update(t, math.radians(2.0), math.radians(1.0), 0.0, 0.0,
                   tc.all_planted(), True)
        t += DT
    wound = dict(lvl.offsets)
    check("leveling winds an offset with all four feet down",
          max(abs(v) for v in wound.values()) > 1e-4,
          "  ".join(f"{leg}{wound[leg]*1e3:+.2f}" for leg in LEGS))
    diag = {leg: leg in ("FL", "RR") for leg in LEGS}
    for _ in range(200):
        lvl.update(t, math.radians(2.0), math.radians(1.0), 0.0, 0.0,
                   diag, True)
        t += DT
    check("... and holds them frozen on a diagonal pair",
          all(abs(lvl.offsets[leg] - wound[leg]) < 1e-12 for leg in LEGS),
          lvl.frozen_reason)


# ===========================================================================
# [6] the per-sweep cost
# ===========================================================================

def test_timing(tables, stand_height):
    plant = LagPlant(tables, stand_height)
    gait = tr.TrotGait()
    gait.start(plant.q(), 0.0)
    state = {"t": 0.0}

    def body(k):
        q = plant.q()
        gait.update(state["t"], q)
        gait.foot_offsets()
        gait.planted()
        plant.step(gait.foot_offsets(), gait.planted())
        state["t"] += DT

    tc.time_sweep("gait + FK switch fit inside the CAN slot", body, n=400)


# ===========================================================================

def self_test(stand_height=tr.TROT_STAND_HEIGHT_M):
    print("trot_fk_switch_hw self-test (no hardware)")
    tables = tc.tables_for(stand_height, clamp_m=lv.LEVEL_CLAMP_M)
    print("[1] construction, the pair schedule and the stance push")
    test_construction()
    test_stance_push(tables)
    test_reach_budget(tables, stand_height)
    print("[1b] body trim and the load balance that sizes it")
    test_body_trim(stand_height)
    test_load_balance(tables, stand_height)
    print("[2] the FK switch (baseline, debounce, timeout)")
    test_fk_switch(tables, stand_height)
    print("[3] the swing state machine")
    trace = test_phases(tables, stand_height)
    test_stop_and_abort(tables, stand_height)
    test_cycle_limit(tables, stand_height)
    print("[4] SwingClearance -- the measured contact schedule")
    test_clearance(tables, stand_height)
    test_contacts(trace)
    print("[5] rocking geometry and the leveling freeze")
    test_rock_geometry(tables)
    test_leveling_freezes(tables)
    print("[6] per-sweep cost")
    test_timing(tables, stand_height)
    return report()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    tc.stand_height_arg(ap, tr.TROT_STAND_HEIGHT_M)
    sys.exit(self_test(ap.parse_args().stand_height))


if __name__ == "__main__":
    main()
