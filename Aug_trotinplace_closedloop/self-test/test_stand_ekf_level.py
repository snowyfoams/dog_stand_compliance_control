#!/usr/bin/env python3
"""Offline gates for stand_ekf_level_hw (stage 2b: the attitude-holding stand).

WHAT IS UNDER TEST
    The leveling law and everything that decides when it is allowed to run:
    the rotation convention the plane plant rests on, the zero-mean per-foot
    trim, the setpoint latch and the height-settle gate that keeps that latch
    off the sag ramp, the AHRS cross-check vetoes, the planted-aware FK height
    and the IMU-free FK attitude, and finally leveling + height together on
    all four feet.

    This is the module the other runners borrow their control law from, so
    these gates are the ones that hold the whole stack up.

Run:  $V test_stand_ekf_level.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNERS = os.path.join(os.path.dirname(_HERE), "stand_postion_mode")
# ^ the hardware scripts live in stand_postion_mode/, beside this directory
for _p in (_HERE, _RUNNERS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import selftest_common as tc                             # noqa: E402
from selftest_common import check, report                # noqa: E402

import dog5_kinematics                                   # noqa: E402
import stand_ekf_verify_hw as s1                         # noqa: E402
import stand_ekf_height_hw as s2                         # noqa: E402
import stand_ekf_level_hw as lv                          # noqa: E402
from stand_ekf_level_hw import (LevelingLoop,            # noqa: E402
                                SetpointLatch, HeightSettleGate,
                                LevelStandSequence, level_inputs,
                                fk_attitude, fk_attitude_from_feet,
                                fk_floor_height_planted,
                                LEVEL_CLAMP_M, LEVEL_SLEW_M_S, LATCH_S,
                                HEIGHT_SETTLE_S, AGREE_VETO_DEG,
                                LEVEL_STALE_S, SETTLE_S)
from ekf_runtime import _rp                              # noqa: E402

LEGS = tc.LEGS
N_JOINTS = tc.N_JOINTS
DT = tc.DT
Q_CROUCH = tc.Q_CROUCH


def test_convention():
    """The _rp convention the plane plant relies on."""
    for r, p in [(0.03, -0.01), (-0.05, 0.02), (0.0, 0.04)]:
        C = tc.C_from_rp(r, p)
        rr, pp = _rp(C)
        check(f"C_from_rp round-trips ({math.degrees(r):+.1f},"
              f"{math.degrees(p):+.1f}) deg",
              abs(rr - r) < 1e-9 and abs(pp - p) < 1e-9)
        break  # one detailed line; assert the rest silently
    ok = True
    for r, p in [(-0.05, 0.02), (0.0, 0.04), (0.1, -0.08)]:
        C = tc.C_from_rp(r, p)
        rr, pp = _rp(C)
        ok &= abs(rr - r) < 1e-9 and abs(pp - p) < 1e-9
    check("C_from_rp round-trips across the range", ok)


def test_leveling(anchors):
    all_on = tc.all_planted()

    # pure roll error converges to the setpoint on the plane plant
    plant = tc.PlanePlant(anchors,
                          mount=(math.radians(1.5), math.radians(-0.5)))
    plant.disturb = (math.radians(1.2), math.radians(-0.8))
    lvl = LevelingLoop(anchors)
    sp_r, sp_p = plant.mount        # setpoint = resting attitude, no disturb
    base_z = tc.flat_base_z(0.22)
    for k in range(int(60 / DT)):
        fz = {leg: base_z[leg] + lvl.offsets[leg] for leg in LEGS}
        r, p = plant.ekf_attitude(fz, all_on)
        lvl.update(k * DT, r, p, sp_r, sp_p, all_on, True)
    fz = {leg: base_z[leg] + lvl.offsets[leg] for leg in LEGS}
    r, p = plant.ekf_attitude(fz, all_on)
    check("leveling drives a 1.2/-0.8 deg disturbance to the setpoint",
          abs(r - sp_r) < math.radians(0.25)
          and abs(p - sp_p) < math.radians(0.25),
          f"residual {math.degrees(r-sp_r):+.3f}/"
          f"{math.degrees(p-sp_p):+.3f} deg")
    h0 = plant.solve(base_z, all_on)[0]
    h1 = plant.solve(fz, all_on)[0]
    check("zero-mean leveling leaves the trunk height alone",
          abs(h1 - h0) < 5e-4, f"{(h1-h0)*1e3:+.2f} mm")

    # deadband: tiny errors do not wind
    d = LevelingLoop(anchors)
    for k in range(int(5 / DT)):
        d.update(k * DT, math.radians(0.1), math.radians(-0.1), 0.0, 0.0,
                 all_on, True)
    check("errors inside the deadband do not wind the offsets",
          all(v == 0.0 for v in d.offsets.values()))

    # clamp + slew under a huge permanent error
    c = LevelingLoop(anchors)
    rate = tc.RateWatch()
    for k in range(int(20 / DT)):
        c.update(k * DT, math.radians(30), 0.0, 0.0, 0.0, all_on, True)
        rate.add(c.offsets)
    check("offsets respect the clamp",
          max(abs(v) for v in c.offsets.values()) <= LEVEL_CLAMP_M + 1e-9
          and c.saturated)
    check("offsets respect the slew limit",
          rate.worst <= LEVEL_SLEW_M_S + 1e-9, f"{rate.worst*1e3:.2f} mm/s")

    # freeze holds, never unwinds
    f = LevelingLoop(anchors)
    for k in range(int(10 / DT)):
        f.update(k * DT, math.radians(3), 0.0, 0.0, 0.0, all_on, True)
    held = dict(f.offsets)
    check("offsets wound in before the freeze",
          max(abs(v) for v in held.values()) > 1e-4)
    for k in range(int(5 / DT)):
        f.update(10 + k * DT, 0.0, 0.0, 0.0, 0.0, all_on, False, "EKF stale")
    check("frozen leveling holds its offsets, does not unwind",
          all(abs(f.offsets[leg] - held[leg]) < 1e-12 for leg in LEGS)
          and f.frozen_reason == "EKF stale")

    # planted-set awareness: lifted leg's offset frozen, mean over planted.
    # Run CLOSED LOOP on the 3-leg plant (open-loop constant error would just
    # saturate the clamp, where zero-mean cannot hold by construction).
    p3 = tc.planted_except("FL")
    plant3 = tc.PlanePlant(anchors)
    plant3.disturb = (math.radians(0.8), math.radians(-0.6))
    g = LevelingLoop(anchors)
    for k in range(int(60 / DT)):
        fz = {leg: base_z[leg] + g.offsets[leg] for leg in LEGS}
        r3, p3_att = plant3.ekf_attitude(fz, p3)
        g.update(k * DT, r3, p3_att, 0.0, 0.0, p3, True)
    check("a lifted leg's offset stays frozen",
          g.offsets["FL"] == 0.0, "FL " + f"{g.offsets['FL']*1e3:.2f} mm")
    mean3 = np.mean([g.offsets[leg] for leg in LEGS if leg != "FL"])
    check("converged offsets stay zero-mean over the planted set",
          abs(mean3) < 5e-4, f"{mean3*1e3:+.3f} mm")
    two = {leg: leg in ("FL", "FR") for leg in LEGS}
    h2 = LevelingLoop(anchors)
    h2.update(0.0, 0.0, 0.0, 0.0, 0.0, two, True)
    h2.update(0.1, math.radians(3), 0.0, 0.0, 0.0, two, True)
    check("fewer than 3 planted feet freezes leveling",
          all(v == 0.0 for v in h2.offsets.values())
          and "planted" in h2.frozen_reason)


def test_latch():
    latch = SetpointLatch()
    check("latch starts un-ready", not latch.ready)
    t = 0.0
    fired = False
    while t < LATCH_S + 0.1:
        fired |= latch.add(t, 0.02 + 0.001 * math.sin(t * 40), -0.01)
        t += DT
    check("latch fires after the window and lands on the mean",
          fired and latch.ready and abs(latch.sp_roll - 0.02) < 5e-4
          and abs(latch.sp_pitch + 0.01) < 1e-6,
          f"sp=({math.degrees(latch.sp_roll):+.3f},"
          f"{math.degrees(latch.sp_pitch):+.3f}) deg")
    fixed = SetpointLatch(0.01, 0.02)
    check("fixed CLI setpoint skips the latch",
          fixed.ready and fixed.sp_roll == 0.01)
    latch.reset()
    check("reset un-latches (fresh setpoint per stand)", not latch.ready)


def test_settle_gate():
    """The gate that keeps the setpoint off the sag ramp."""
    g = HeightSettleGate()
    # a height loop still ramping (13 mm of sag, 5 mm/s) must NOT open it
    t, err, opened_during_ramp = 0.0, 0.013, False
    while err > s2.HEIGHT_DEADBAND_M:
        opened_during_ramp |= g.update(t, True, err)
        err -= s2.HEIGHT_SLEW_M_S * DT
        t += DT
    check("gate stays shut while the height loop is still ramping",
          not opened_during_ramp, f"ramp took {t:.1f}s")
    # ... and opens HEIGHT_SETTLE_S after it arrives, not instantly
    opened_at = None
    t_arrive = t
    while t < t_arrive + HEIGHT_SETTLE_S + 0.5:
        if g.update(t, True, 0.0002) and opened_at is None:
            opened_at = t
        t += DT
    check("gate opens only after the loop holds its target",
          opened_at is not None
          and abs((opened_at - t_arrive) - HEIGHT_SETTLE_S) < 5 * DT,
          f"opened {opened_at-t_arrive:.2f}s after arrival, "
          f"want {HEIGHT_SETTLE_S:.1f}s")
    # a fresh excursion outside the band must re-arm it
    g.update(t, True, 0.020)
    check("leaving the band re-arms the gate", not g.update(t + DT, True, 0.020))
    # --height-gain 0 / frozen: nothing to settle, so do not block leveling
    g0 = HeightSettleGate()
    t = 0.0
    opened = False
    while t < HEIGHT_SETTLE_S + 0.5:
        opened |= g0.update(t, False, 0.013)     # loop not running, big error
        t += DT
    check("a height loop that is not running counts as settled", opened)


def test_level_vetoes():
    now = time.monotonic()
    veto = math.radians(AGREE_VETO_DEG)
    r, p, active, why = level_inputs(tc.FakeShared(roll=0.02, pitch=-0.01),
                                     (0.02, -0.01), now, veto)
    check("healthy agreeing EKF is accepted",
          active and abs(r - 0.02) < 1e-9, str(why))
    _, _, active, why = level_inputs(tc.FakeShared(ready=False), (0, 0), now,
                                     veto)
    check("un-initialised EKF is vetoed", not active)
    _, _, active, why = level_inputs(tc.FakeShared(healthy=False), (0, 0), now,
                                     veto)
    check("unhealthy EKF is vetoed", not active and why == "EKF unhealthy")
    stale = tc.FakeShared()
    stale.tau_stamp = now - 10 * LEVEL_STALE_S
    _, _, active, why = level_inputs(stale, (0, 0), now, veto)
    check("stale EKF is vetoed", not active and why == "EKF stale")
    _, _, active, why = level_inputs(tc.FakeShared(), None, now, veto)
    check("missing AHRS is vetoed (no cross-check, no loop)",
          not active and why == "no AHRS attitude")
    _, _, active, why = level_inputs(tc.FakeShared(roll=math.radians(5)),
                                     (0.0, 0.0), now, veto)
    check("EKF-vs-AHRS divergence vetoes the loop",
          not active and "disagree" in (why or ""), str(why))
    _, _, active, _ = level_inputs(tc.FakeShared(roll=math.radians(1)),
                                   (0.0, 0.0), now, veto)
    check("small EKF-AHRS disagreement is tolerated", active)


def test_sequence(tables, stand_height):
    seq = LevelStandSequence(tables, stand_height)
    seen, t, _, _ = tc.walk_stages(seq, offset=0.005, q_from="q_cmd")
    check("stage order incl. park", seen == tc.STAGE_ORDER, " -> ".join(seen))
    check("PARKED lands on the recorded crouch",
          float(np.max(np.abs(seq.q_cmd(t) - Q_CROUCH))) < 1e-5)

    s = LevelStandSequence(tables, stand_height)
    s.stage, s.stage_t0 = "HOLD4", 0.0
    q0 = s.q_cmd(0.0)
    s.extra_z = {"FL": 0.004, "FR": -0.002, "RL": 0.0, "RR": -0.002}
    q1 = s.q_cmd(0.0)
    ok = True
    for i, leg in enumerate(LEGS):
        f0 = dog5_kinematics.foot_position(leg, q0[3 * i:3 * i + 3])
        f1 = dog5_kinematics.foot_position(leg, q1[3 * i:3 * i + 3])
        ok &= abs((f1[2] - f0[2]) - s.extra_z[leg]) < 1e-4
    check("per-leg extra z moves each commanded foot by its own amount", ok)

    s.planted_mask = np.array([True, True, False, True])
    check("contact mask reaches the sequence in HOLD4",
          list(s.contacts) == [True, True, False, True])
    parked = LevelStandSequence(tables, stand_height)
    parked.stage, parked.stage_t0 = "PARK", 0.0
    parked._park_from = dict(parked.z_crouch)
    parked.planted_mask = np.array([True, True, False, True])
    check("contacts forced ON outside HOLD4", bool(np.all(parked.contacts)))

    # PARK ramps from the held pose extra included
    s3 = LevelStandSequence(tables, stand_height)
    s3.stage, s3.stage_t0 = "HOLD4", 0.0
    s3.offset = 0.008
    s3.extra_z = {leg: 0.003 for leg in LEGS}
    held = s3.q_cmd(0.0)
    s3.update(1.0, Q_CROUCH, np.zeros(N_JOINTS), park_pressed=True,
              offset=0.008)
    check("PARK starts from the pose actually being held (extra included)",
          float(np.max(np.abs(s3.q_cmd(1.0) - held))) < 1e-9)


def test_fk_planted(tables, stand_height):
    q = np.zeros(N_JOINTS)
    for i, leg in enumerate(LEGS):
        q[3 * i:3 * i + 3] = tables[leg].q_at(-stand_height)
    h4 = fk_floor_height_planted(q, None, None, ref="hip")
    check("planted FK with all feet down matches the stage-1 function",
          abs(h4 - s1.fk_floor_height(q, None, ref="hip")) < 1e-12)
    # lift FL by 15 mm: the all-four average is biased by lift/4, planted isn't
    ql = q.copy()
    ql[0:3] = tables["FL"].q_at(-stand_height + 0.015)
    h_all = s1.fk_floor_height(ql, None, ref="hip")
    h_planted = fk_floor_height_planted(ql, None,
                                        np.array([False, True, True, True]),
                                        ref="hip")
    # raising a foot makes the trunk look LOWER to the all-four average
    check("planted FK ignores the lifted foot",
          abs(h_planted - h4) < 1e-4 and abs((h4 - h_all) - 0.015 / 4) < 2e-4,
          f"all-four biased {(h_all-h4)*1e3:+.2f} mm, planted "
          f"{(h_planted-h4)*1e3:+.2f} mm")


def test_fk_attitude(tables, stand_height, anchors):
    """The IMU-free attitude estimate and what its gap to the EKF means."""
    # 1. the plane fit must invert _rp exactly, at any angle
    worst = 0.0
    for r_deg, p_deg in [(0, 0), (1.5, -0.5), (-3, 2), (8, -6), (0, 12)]:
        r, p = math.radians(r_deg), math.radians(p_deg)
        u = tc.C_from_rp(r, p) @ np.array([0.0, 0.0, 1.0])
        b, c = -u[0] / u[2], -u[1] / u[2]
        feet = [(x, y, -stand_height + b * x + c * y)
                for x, y in (anchors[leg] for leg in LEGS)]
        rr, pp = fk_attitude_from_feet(feet)
        worst = max(worst, abs(rr - r), abs(pp - p))
    check("plane-fit attitude inverts the EKF's _rp convention",
          worst < 1e-9, f"worst {math.degrees(worst)*1e6:.2f} udeg")

    # 2. THE claim the shim test rests on: EKF - FK is the floor slope, and
    #    the trunk's own attitude drops out of it (so mount tilt and encoder
    #    zeros, being constants, cancel in the CHANGE across a shim)
    for slope_deg, axis in [(2.0, "roll"), (3.0, "pitch")]:
        s = math.radians(slope_deg)
        n = (np.array([0.0, -math.sin(s), math.cos(s)]) if axis == "roll"
             else np.array([math.sin(s), 0.0, math.cos(s)]))
        for trunk in [(0.0, 0.0), (1.0, -0.5), (-2.0, 1.5)]:
            C = tc.C_from_rp(math.radians(trunk[0]), math.radians(trunk[1]))
            u = C @ n                       # floor normal in body coords
            b, c = -u[0] / u[2], -u[1] / u[2]
            feet = [(x, y, -stand_height + b * x + c * y)
                    for x, y in (anchors[leg] for leg in LEGS)]
            r_fk, p_fk = fk_attitude_from_feet(feet)
            r_ek, p_ek = _rp(C)
            got = (r_ek - r_fk) if axis == "roll" else (p_ek - p_fk)
            off = (p_ek - p_fk) if axis == "roll" else (r_ek - r_fk)
            ok = abs(got - s) < math.radians(0.01) \
                and abs(off) < math.radians(0.01)
            if not ok:
                break
        check(f"EKF-FK attitude = the floor slope ({slope_deg:.0f} deg "
              f"{axis}), whatever the trunk does", ok,
              f"read {math.degrees(got):+.4f} deg")

    # 3. fewer than 3 planted feet cannot define a plane
    r, p = fk_attitude_from_feet([(0.2, 0.1, -0.19), (-0.2, -0.1, -0.19)])
    check("two feet do not define an attitude", math.isnan(r))
    q = np.zeros(N_JOINTS)
    for i, leg in enumerate(LEGS):
        q[3 * i:3 * i + 3] = tables[leg].q_at(-stand_height)
    r3, p3 = fk_attitude(q, np.array([True, True, True, False]))
    check("three planted feet still give an attitude", math.isfinite(r3))

    # 4. the commanded stand is a horizontal foot plane by construction, so
    #    FK attitude reads zero there -- it is only non-zero once the legs
    #    differ, which on hardware means sag
    r0, p0 = fk_attitude(q)
    check("FK attitude of the commanded stand is level",
          abs(r0) < 1e-9 and abs(p0) < 1e-9,
          f"{math.degrees(r0):+.2e}/{math.degrees(p0):+.2e} deg")

    # 5. differential sag is what it actually measures: drop the two front
    #    feet 5 mm and the nose must read DOWN (+pitch, _rp convention)
    q_sag = q.copy()
    for i, leg in enumerate(LEGS):
        if anchors[leg][0] > 0:                    # front legs
            q_sag[3 * i:3 * i + 3] = tables[leg].q_at(-stand_height + 0.005)
    r_s, p_s = fk_attitude(q_sag)
    span = abs(anchors["FL"][0] - anchors["RL"][0])
    check("5 mm of front-leg sag reads as nose-down pitch",
          p_s > 0 and abs(p_s - math.atan2(0.005, span)) < math.radians(0.05),
          f"{math.degrees(p_s):+.3f} deg over a {span*1e3:.0f} mm wheelbase")


def test_closed_loop(stand_height, anchors):
    """End to end on all four feet: stand, latch the resting attitude, take a
    tilt disturbance, and hold both attitude and height through it."""
    mount = (math.radians(1.5), math.radians(-0.5))
    plant = tc.PlanePlant(anchors, mount=mount)
    lvl = LevelingLoop(anchors)
    hctrl = s2.HeightController()
    latch = SetpointLatch()
    all_on = tc.all_planted()
    target_h = stand_height + tc.PLANT_TO_BOTTOM
    SAG = 0.010
    base_z = tc.flat_base_z(stand_height)

    def measure(fz):
        h, roll, pitch = plant.solve(fz, all_on)
        return (h + tc.PLANT_TO_BOTTOM - SAG,
                roll + mount[0], pitch + mount[1])

    t, events, max_err = 0.0, [], 0.0
    while t < 90.0:
        fz = {leg: base_z[leg] - hctrl.offset + lvl.offsets[leg]
              for leg in LEGS}
        h, roll, pitch = measure(fz)
        if not latch.ready and t > SETTLE_S:
            if latch.add(t, roll, pitch):
                events.append("latched")
        engaged = latch.ready
        lvl.update(t, roll, pitch, latch.sp_roll if engaged else 0.0,
                   latch.sp_pitch if engaged else 0.0, all_on, engaged)
        hctrl.update(t, h, target_h, True)
        # a 1.2/-0.9 deg disturbance arrives once the setpoint is latched
        if "latched" in events and "disturbed" not in events and t > 20.0:
            plant.disturb = (math.radians(1.2), math.radians(-0.9))
            events.append("disturbed")
        if "disturbed" in events:
            max_err = max(max_err, abs(roll - latch.sp_roll),
                          abs(pitch - latch.sp_pitch))
        t += DT

    fz = {leg: base_z[leg] - hctrl.offset + lvl.offsets[leg] for leg in LEGS}
    h, roll, pitch = measure(fz)
    check("sequence ran: latch -> disturbance",
          events == ["latched", "disturbed"], str(events))
    check("setpoint latched onto the resting (mount-tilted) attitude",
          abs(latch.sp_roll - mount[0]) < math.radians(0.05)
          and abs(latch.sp_pitch - mount[1]) < math.radians(0.05),
          f"sp=({math.degrees(latch.sp_roll):+.2f},"
          f"{math.degrees(latch.sp_pitch):+.2f}) deg")
    check("disturbance transient stayed bounded",
          max_err < math.radians(2.0), f"max {math.degrees(max_err):.2f} deg")
    check("leveling drove the disturbance back to the setpoint",
          abs(roll - latch.sp_roll) < math.radians(0.3)
          and abs(pitch - latch.sp_pitch) < math.radians(0.3),
          f"{math.degrees(roll-latch.sp_roll):+.3f}/"
          f"{math.degrees(pitch-latch.sp_pitch):+.3f} deg")
    check("height loop held the trunk-bottom target through it",
          abs(h - target_h) <= s2.HEIGHT_DEADBAND_M + 1e-6,
          f"{(h-target_h)*1e3:+.2f} mm")
    # the offset must equal the sag -- not the clamp.  Before the frame fix
    # this test's target was a foot radius off, so it converged only because
    # the required 30 mm happened to equal the clamp exactly.
    check("height offset recovered the sag, and is nowhere near the clamp",
          abs(hctrl.offset - SAG) <= s2.HEIGHT_DEADBAND_M + 1e-6
          and not hctrl.saturated,
          f"{hctrl.offset*1e3:.1f} mm vs {SAG*1e3:.0f} mm sag, clamp "
          f"{s2.HEIGHT_CLAMP_M*1e3:.0f} mm")


def test_timing(tables, anchors):
    lvl = LevelingLoop(anchors)
    planted = tc.all_planted()

    def sweep(k):
        lvl.update(k * 0.004, 0.01, -0.01, 0.0, 0.0, planted, True)
        for leg in LEGS:
            tables[leg].q_at(-0.22 + lvl.offsets[leg])

    tc.time_sweep("leveling + 4-leg lookup fit the CAN slot", sweep)


def self_test(stand_height=lv.STAND_HEIGHT_DEFAULT):
    print("stand_ekf_level_hw self-test (no hardware)")
    print("[1] rotation convention")
    test_convention()
    print("[2] height tables (stage 2's, wider span)")
    tables = tc.tables_for(stand_height,
                           clamp_m=s2.HEIGHT_CLAMP_M + LEVEL_CLAMP_M)
    anchors = tc.anchors_of(tables)
    # The legs' physical reach caps the downward span (~-221 mm) well before
    # the requested clamp sum, so assert the authority that matters: room for
    # the measured sag (~13 mm) + the leveling clamp below the stand, and the
    # full lift above it.
    span_ok = all(tables[leg].z_min <= -stand_height - (0.013 + LEVEL_CLAMP_M)
                  for leg in LEGS)
    check("table span covers sag + leveling authority below the stand",
          span_ok,
          "  ".join(f"{leg}[{tables[leg].z_min*1e3:.0f},"
                    f"{tables[leg].z_max*1e3:.0f}]" for leg in LEGS))
    print("[3] leveling loop on the plane plant")
    test_leveling(anchors)
    print("[4] setpoint latch")
    test_latch()
    test_settle_gate()
    print("[5] leveling vetoes (AHRS watchdog)")
    test_level_vetoes()
    print("[6] stage machine with per-leg extra z")
    test_sequence(tables, stand_height)
    print("[7] FK attitude (IMU-free) vs the EKF")
    test_fk_attitude(tables, stand_height, anchors)
    print("[7b] planted-aware FK height")
    test_fk_planted(tables, stand_height)
    print("[8] closed loop end to end, four feet")
    test_closed_loop(stand_height, anchors)
    print("[9] timing")
    test_timing(tables, anchors)
    return report()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    tc.stand_height_arg(ap, lv.STAND_HEIGHT_DEFAULT)
    sys.exit(self_test(ap.parse_args().stand_height))


if __name__ == "__main__":
    main()
