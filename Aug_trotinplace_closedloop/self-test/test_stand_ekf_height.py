#!/usr/bin/env python3
"""Offline gates for stand_ekf_height_hw (stage 2: closed-loop stand height).

WHAT IS UNDER TEST
    The per-leg z->q tables that replace IK in the CAN sweep, the integral
    height loop, the EKF vetoes that FK stands watchdog over, the stage
    machine, and the four of them together against a sagging plant.

Run:  $V test_stand_ekf_height.py
"""
from __future__ import annotations

import argparse
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

import stand_dog5_hw as base                             # noqa: E402
import stand_dog5_recorded_hw as recorded                # noqa: E402
import dog5_kinematics                                   # noqa: E402
import stand_ekf_verify_hw as s1                         # noqa: E402
from stand_ekf_height_hw import (HeightController,       # noqa: E402
                                 HeightStandSequence, height_inputs,
                                 fk_floor_height, HEIGHT_DEADBAND_M,
                                 HEIGHT_STALE_S)

LEGS = tc.LEGS
N_JOINTS = tc.N_JOINTS
DT = tc.DT
Q_CROUCH = tc.Q_CROUCH
FOOT_RADIUS_M = tc.FOOT_RADIUS_M
IMU_BELOW_TRUNK_ORIGIN_M = tc.IMU_BELOW_TRUNK_ORIGIN_M


def test_tables(tables, stand_height):
    for leg in LEGS:
        t = tables[leg]
        i = LEGS.index(leg)
        q_crouch = Q_CROUCH[3 * i:3 * i + 3]
        z_crouch = float(dog5_kinematics.foot_position(leg, q_crouch)[2])
        check(f"{leg}: table reproduces the recorded crouch",
              float(np.max(np.abs(t.q_at(z_crouch) - q_crouch))) < 1e-6,
              f"max {np.max(np.abs(t.q_at(z_crouch)-q_crouch)):.2e} rad")
        check(f"{leg}: table spans the stand height",
              t.z_min <= -stand_height <= t.z_max,
              f"[{t.z_min*1e3:.0f}, {t.z_max*1e3:.0f}] mm")

    # interpolation must reproduce a direct IK solve away from the grid points
    worst = 0.0
    for leg in LEGS:
        t = tables[leg]
        for z in np.linspace(-stand_height - 0.02, -stand_height + 0.02, 9):
            q = t.q_at(z)
            f = dog5_kinematics.foot_position(leg, q)
            worst = max(worst, abs(f[2] - z),
                        float(np.linalg.norm(f[:2] - t.xy)))
    check("interpolated poses hit the commanded foot position",
          worst < 5e-5, f"worst {worst*1e6:.1f} um")

    lo, hi = base.soft_limits()
    ok = True
    for i, leg in enumerate(LEGS):
        t = tables[leg]
        for z in np.linspace(t.z_min, t.z_max, 41):
            q = t.q_at(z)
            ok &= bool(np.all(q >= lo[3*i:3*i+3])
                       and np.all(q <= hi[3*i:3*i+3]))
    check("every reachable table pose is inside the soft limits", ok)

    tc.time_sweep("full 4-leg lookup fits the CAN slot",
                  lambda k: [tables[leg].q_at(-stand_height) for leg in LEGS])


def test_controller():
    """Integral loop against a position-servo plant with a constant sag."""
    SAG = 0.013
    target = 0.220
    ctrl = HeightController(gain_per_s=0.5)
    h = target - SAG
    t = 0.0
    for _ in range(int(30 / DT)):
        off = ctrl.update(t, h, target, True)
        h = (target + off) - SAG              # plant: commanded minus sag
        t += DT
    # the deadband is a DESIGN limit, not slop: the integrator deliberately
    # stops inside it so EKF noise cannot make the pose chatter.  So the loop
    # converges to within the deadband, never exactly onto the target.
    check("loop converges to within its deadband",
          abs(h - target) <= HEIGHT_DEADBAND_M + 1e-9,
          f"{(h-target)*1e3:+.2f} mm vs {HEIGHT_DEADBAND_M*1e3:.1f} mm deadband")
    check("loop recovers the sag to within the deadband",
          abs(ctrl.offset - SAG) <= HEIGHT_DEADBAND_M + 1e-9,
          f"offset {ctrl.offset*1e3:.1f} mm vs {SAG*1e3:.0f} mm sag")
    check("loop removes most of the sag",
          abs(h - target) < 0.1 * SAG,
          f"{(h-target)*1e3:+.2f} mm left of {SAG*1e3:.0f} mm")

    # gain 0 must be a true no-op (the open-loop A/B)
    z = HeightController(gain_per_s=0.0)
    for k in range(1000):
        z.update(k * DT, 0.1, target, True)
    check("gain 0 is open loop", z.offset == 0.0)

    # clamp and slew
    c = HeightController(gain_per_s=5.0, clamp_m=0.010, slew_m_s=0.004)
    prev, worst_rate = 0.0, 0.0
    for k in range(int(20 / DT)):
        o = c.update(k * DT, 0.0, 1.0, True)          # huge, permanent error
        worst_rate = max(worst_rate, abs(o - prev) / DT)
        prev = o
    check("offset respects the clamp", abs(c.offset - 0.010) < 1e-9,
          f"{c.offset*1e3:.2f} mm")
    check("offset respects the slew limit", worst_rate <= 0.004 + 1e-9,
          f"{worst_rate*1e3:.2f} mm/s")

    # freeze must HOLD, never unwind
    f = HeightController(gain_per_s=0.5)
    for k in range(int(5 / DT)):
        f.update(k * DT, 0.200, target, True)
    held = f.offset
    check("integrator wound in before the freeze", held > 1e-3,
          f"{held*1e3:.1f} mm")
    for k in range(int(5 / DT)):
        o = f.update(5 + k * DT, 0.0, target, False, "EKF stale")
    check("frozen loop holds its offset, does not unwind",
          abs(o - held) < 1e-12 and f.frozen_reason == "EKF stale")

    # deadband
    d = HeightController(gain_per_s=0.5, deadband_m=0.002)
    for k in range(int(5 / DT)):
        d.update(k * DT, target - 0.001, target, True)
    check("errors inside the deadband do not wind the integrator",
          d.offset == 0.0)


def test_vetoes(stand_height):
    q_stand = s1.compute_q_stand(stand_height)
    z0 = fk_floor_height(Q_CROUCH, np.eye(3), ref="imu")
    rise = (fk_floor_height(q_stand, np.eye(3), ref="imu") - z0)
    now = time.monotonic()

    h, hfk, active, why = height_inputs(tc.FakeShared(rz=rise), q_stand, z0,
                                        now)
    check("healthy EKF at the stand pose is accepted", active, str(why))
    check("measurement lands on the FK trunk-bottom height",
          abs(h - hfk) < 1e-9, f"EKF {h*1e3:.1f} vs FK {hfk*1e3:.1f} mm")
    check("measured height is the commanded stand at the TRUNK BOTTOM",
          abs(h - (stand_height + FOOT_RADIUS_M
                   - IMU_BELOW_TRUNK_ORIGIN_M)) < 1e-9, f"{h*1e3:.1f} mm")
    check("... which is the abd axis minus the IMU lever",
          abs((h + IMU_BELOW_TRUNK_ORIGIN_M)
              - fk_floor_height(q_stand, np.eye(3), ref="hip")) < 1e-9,
          f"abd {(h+IMU_BELOW_TRUNK_ORIGIN_M)*1e3:.1f} mm")

    _, _, active, why = height_inputs(tc.FakeShared(rz=rise, healthy=False),
                                      q_stand, z0, now)
    check("unhealthy EKF is vetoed", not active and why == "EKF unhealthy")

    _, _, active, why = height_inputs(tc.FakeShared(rz=rise, ready=False),
                                      q_stand, z0, now)
    check("un-initialised EKF is vetoed", not active)

    stale = tc.FakeShared(rz=rise)
    stale.tau_stamp = now - 10 * HEIGHT_STALE_S
    _, _, active, why = height_inputs(stale, q_stand, z0, now)
    check("stale EKF is vetoed", not active and why == "EKF stale")

    _, _, active, why = height_inputs(tc.FakeShared(rz=rise), q_stand, None,
                                      now)
    check("missing height origin is vetoed", not active)

    # the headline safety net: EKF drifting away from FK must stop the loop
    drifted = tc.FakeShared(rz=rise + 0.050)
    _, _, active, why = height_inputs(drifted, q_stand, z0, now)
    check("EKF-vs-FK divergence vetoes the loop",
          not active and "disagree" in (why or ""), str(why))
    _, _, active, _ = height_inputs(tc.FakeShared(rz=rise + 0.010), q_stand,
                                    z0, now)
    check("small EKF-FK disagreement is tolerated", active)


def test_sequence(tables, stand_height):
    seq = HeightStandSequence(tables, stand_height)
    OFF = 0.012
    every_contact = []
    seen, t, _, _ = tc.walk_stages(
        seq, offset=OFF,
        after_tick=lambda sq, tt, cmd, ct: every_contact.append(ct.copy()))
    check("stage order incl. park", seen == tc.STAGE_ORDER, " -> ".join(seen))
    check("contacts stay ON in every stage",
          bool(np.all(np.array(every_contact))),
          f"{len(every_contact)} sweeps")

    # the offset must actually raise the trunk in HOLD4
    hold = HeightStandSequence(tables, stand_height)
    hold.stage, hold.stage_t0 = "HOLD4", 0.0
    q0 = hold.q_cmd(0.0)
    hold.offset = OFF
    q1 = hold.q_cmd(0.0)
    h0 = fk_floor_height(q0, ref="hip")
    h1 = fk_floor_height(q1, ref="hip")
    check("positive offset raises the commanded trunk height",
          abs((h1 - h0) - OFF) < 2e-4,
          f"{(h1-h0)*1e3:+.2f} mm for {OFF*1e3:.0f} mm")

    # PARK must unwind whatever the loop wound in, landing on the crouch
    cmd_parked = seq.q_cmd(t)
    check("PARKED lands on the recorded crouch despite the offset",
          float(np.max(np.abs(cmd_parked - Q_CROUCH))) < 1e-5,
          f"max {np.max(np.abs(cmd_parked-Q_CROUCH))*1e3:.3f} mrad")

    # park ramp must START from the held (offset) pose, not the nominal one
    s3 = HeightStandSequence(tables, stand_height)
    s3.stage, s3.stage_t0, s3.offset = "HOLD4", 0.0, OFF
    held = s3.q_cmd(0.0)
    s3.update(1.0, Q_CROUCH, np.zeros(N_JOINTS), park_pressed=True, offset=OFF)
    check("PARK starts from the pose actually being held",
          float(np.max(np.abs(s3.q_cmd(1.0) - held))) < 1e-9)

    # offset is ignored outside HOLD4
    s4 = HeightStandSequence(tables, stand_height)
    s4.stage, s4.stage_t0 = "WAIT_CROUCH", 0.0
    s4.update(0.0, Q_CROUCH, np.zeros(N_JOINTS), offset=0.05)
    check("offset is ignored outside HOLD4",
          float(np.max(np.abs(s4.q_cmd(0.0) - Q_CROUCH))) < 1e-6)


def sagged_pose(tables, h_hip):
    """Joint angles whose FK hip height is h_hip (the 'measured' encoders)."""
    q = np.zeros(N_JOINTS)
    for i, leg in enumerate(LEGS):
        q[3 * i:3 * i + 3] = tables[leg].q_at(-(h_hip - FOOT_RADIUS_M))
    return q


def test_closed_loop(tables, stand_height):
    """End to end: sagging plant + EKF + veto + tables -> holds the target.

    The plant is simulated at the abd axis (that is where the leg tables and
    FK live); the loop only ever sees trunk-bottom heights, so the conversion
    happens once, here, exactly as the lever does on hardware.
    """
    target = stand_height + FOOT_RADIUS_M - IMU_BELOW_TRUNK_ORIGIN_M
    SAG = 0.013
    seq = HeightStandSequence(tables, stand_height)
    seq.stage, seq.stage_t0 = "HOLD4", 0.0
    ctrl = HeightController(gain_per_s=0.5)
    # EKF origin: the IMU height at init, i.e. at the crouch, level trunk
    z0 = fk_floor_height(Q_CROUCH, np.eye(3), ref="imu")
    h = fk_floor_height(seq.q_cmd(0.0), ref="hip") - SAG      # abd axis
    for k in range(int(25 / DT)):
        t = k * DT
        # encoders report the ACTUAL pose, so FK sees the sag too
        q_meas = sagged_pose(tables, h)
        # the EKF reports the IMU's displacement from init: (h - lever) - z0
        r_z = (h - IMU_BELOW_TRUNK_ORIGIN_M) - z0
        h_ekf, h_fk, active, why = height_inputs(tc.FakeShared(rz=r_z), q_meas,
                                                 z0, time.monotonic())
        off = ctrl.update(t, h_ekf, target, active and seq.loop_engaged, why)
        seq.update(t, q_meas, np.zeros(N_JOINTS), offset=off)
        # plant: the legs reach the commanded pose minus a constant sag
        h = fk_floor_height(seq.q_cmd(t), ref="hip") - SAG
    h_bot = h - IMU_BELOW_TRUNK_ORIGIN_M           # what the loop regulates
    check("closed loop holds the commanded trunk-bottom height (sagging plant)",
          abs(h_bot - target) <= HEIGHT_DEADBAND_M + 1e-9,
          f"{(h_bot-target)*1e3:+.2f} mm")
    check("the abd axis lands one IMU lever above it",
          abs(h - (target + IMU_BELOW_TRUNK_ORIGIN_M))
          <= HEIGHT_DEADBAND_M + 1e-9, f"abd {h*1e3:.1f} mm")
    check("recovered offset equals the sag",
          abs(ctrl.offset - SAG) <= HEIGHT_DEADBAND_M + 1e-9,
          f"{ctrl.offset*1e3:.1f} mm vs {SAG*1e3:.0f} mm")
    check("EKF and FK agree throughout the run (watchdog never fired)",
          abs(h_ekf - h_fk) < 1e-5, f"{abs(h_ekf-h_fk)*1e6:.1f} um")


def self_test(stand_height=recorded.DEFAULT_STAND_HEIGHT):
    print("stand_ekf_height_hw self-test (no hardware)")
    print("[1] height tables")
    tables = tc.tables_for(stand_height)
    test_tables(tables, stand_height)
    print("[2] integral height controller")
    test_controller()
    print("[3] EKF vetoes / FK watchdog")
    test_vetoes(stand_height)
    print("[4] stage machine + park")
    test_sequence(tables, stand_height)
    print("[5] closed loop end to end")
    test_closed_loop(tables, stand_height)
    return report()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    tc.stand_height_arg(ap, recorded.DEFAULT_STAND_HEIGHT)
    sys.exit(self_test(ap.parse_args().stand_height))


if __name__ == "__main__":
    main()
