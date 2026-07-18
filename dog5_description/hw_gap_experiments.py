#!/usr/bin/env python3
r"""Reproduce the hardware failure "CoM shifts fine but the leg won't move".

Three experiments on the hardware-faithful sim (see SIM_APPROACH_HW.md):

A. FRICTION DEMAND — during SHIFT/RECENTER every foot is loaded and must
   stick.  Measure each foot's tangential/normal force ratio: the minimum
   floor friction the method requires.

B. SWING WHILE LOADED — bypass the unload gate (as an impatient hardware
   test would) and command a 30 mm horizontal foot move at pre-lift heights
   0..20 mm, at floor friction 1.0 and 2.0 (rubber).  Records the residual
   normal force when the horizontal move starts and how far the foot
   actually travels: the "leg won't shift" failure appears whenever the
   foot still carries load.

C. CALIBRATION ERROR — inject +/-2 deg joint-zero errors (mis-zeroed
   encoders).  Shows the four-leg load asymmetry this creates at neutral
   stance and finds the pre-lift height that still passes the full 3-leg
   sequence.

Run with the project venv (about 10 minutes total):
    D:\mujoco\.venv\Scripts\python.exe hw_gap_experiments.py
    D:\mujoco\.venv\Scripts\python.exe hw_gap_experiments.py A   # one experiment
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import stand3_dog5 as S

# Stage speed caps 3x the hardware default: the stand-up ramp is not under
# test here and quasi-static behaviour is preserved (streamed phases keep
# their normal rate).
FAST = dict(crouch_dps=300.0, stand_dps=300.0)


def base_args(**overrides):
    ns = argparse.Namespace(
        swing_leg=S.DEFAULT_SWING_LEG, shift=S.DEFAULT_SHIFT_M,
        lift=S.DEFAULT_LIFT_M, torque_trip=S.DEFAULT_TORQUE_TRIP_NM,
        unload_trip=S.DEFAULT_UNLOAD_TAU_TRIP_NM,
        crouch_dps=S.DEFAULT_CROUCH_DPS, stand_dps=S.DEFAULT_STAND_DPS,
        stream_dps=S.DEFAULT_STREAM_DPS, kp_scale=1.0)
    for key, value in {**FAST, **overrides}.items():
        setattr(ns, key, value)
    return ns


def experiment_a():
    print("=" * 72)
    print("A. FRICTION DEMAND of the body shift (all feet loaded and sticking)")
    print("=" * 72)
    usage = {leg: {} for leg in S.LEGS}   # leg -> stage -> max Ft/N

    def hook(now, stage, oracle):
        if stage not in ("SHIFT", "RECENTER"):
            return
        for leg in S.LEGS:
            normal, tangential = oracle.foot_contact(leg)
            if normal > 2.0:               # ignore grazing/unloaded feet
                ratio = tangential / normal
                usage[leg][stage] = max(usage[leg].get(stage, 0.0), ratio)

    args = base_args(hold3=2.0)
    model, data = S.build_sim()
    controller, oracle = S.run(model, data, args, frame_hook=hook, quiet=True)
    ok = S.report(controller, oracle, quiet=True)
    print(f"run: {'PASS' if ok else 'FAIL'}   max tangential/normal per foot:")
    worst = 0.0
    for leg in S.LEGS:
        cells = "  ".join(f"{stage}={usage[leg].get(stage, 0.0):.3f}"
                          for stage in ("SHIFT", "RECENTER"))
        worst = max(worst, *usage[leg].values()) if usage[leg] else worst
        print(f"  {leg}: {cells}")
    print(f"-> minimum floor friction required ~= {worst:.2f} "
          f"(plus margin; any normal floor exceeds this)")


def experiment_b():
    print("=" * 72)
    print("B. SWING WHILE LOADED (unload gate bypassed, 30 mm horizontal move)")
    print("=" * 72)
    print("The swing command starts the moment the pre-lift ends; whatever")
    print("normal force remains on the foot, friction resists the move.\n")
    swing_mm = 30.0
    header = (f"{'mu':>4} {'prelift':>8} {'N at start':>11} {'foot moved':>11} "
              f"{'body pushed':>12} {'peak Ft':>8}  outcome")
    print(header)
    for mu in (1.0, 2.0):
        for prelift_mm in (0.0, 2.0, 5.0, 10.0, 15.0, 20.0):
            record = {"n0": None, "foot0": None, "foot1": None,
                      "body0": None, "body1": None, "ft": 0.0}

            def hook(now, stage, oracle, record=record):
                if stage != "LIFT":
                    return
                foot = oracle.d.site_xpos[oracle.site[oracle.plan.swing]]
                body = oracle.d.xpos[oracle.trunk]
                if record["foot0"] is None:
                    record["n0"] = oracle.foot_contact(oracle.plan.swing)[0]
                    record["foot0"] = foot[:2].copy()
                    record["body0"] = body[:2].copy()
                record["foot1"] = foot[:2].copy()
                record["body1"] = body[:2].copy()
                record["ft"] = max(
                    record["ft"], oracle.foot_contact(oracle.plan.swing)[1])

            args = base_args(
                prelift=prelift_mm / 1000.0,
                lift=swing_mm / 1000.0,
                lift_vec=(-1.0, 0.0, 0.0),    # horizontal, toward the body
                unload_trip=99.0,             # gate bypassed on purpose
                hold3=1.0)
            model, data = S.build_sim(friction=mu)
            controller, oracle = S.run(model, data, args,
                                       frame_hook=hook, quiet=True)
            if record["foot0"] is None:
                print(f"{mu:4.1f} {prelift_mm:6.0f}mm    (run stopped early: "
                      f"{controller.aborted or controller.stop_reason})")
                continue
            moved = 1000.0 * float(
                np.linalg.norm(record["foot1"] - record["foot0"]))
            pushed = 1000.0 * float(
                np.linalg.norm(record["body1"] - record["body0"]))
            outcome = ("moves freely" if moved >= 0.8 * swing_mm and
                       record["n0"] <= 1.0 else
                       "DRAGGED under load" if moved >= 0.5 * swing_mm else
                       "PINNED by friction")
            print(f"{mu:4.1f} {prelift_mm:6.0f}mm {record['n0']:9.1f} N "
                  f"{moved:8.1f} mm {pushed:9.1f} mm {record['ft']:6.1f} N"
                  f"  {outcome}")
    print("\n-> a leg only 'shifts' when its normal force is ~zero first;")
    print("   friction never resists the vertical pre-lift itself.")


def experiment_c():
    print("=" * 72)
    print("C. CALIBRATION ZERO ERROR (+/-2 deg on all 12 joints)")
    print("=" * 72)
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        zero_error = np.deg2rad(rng.uniform(-2.0, 2.0, S.N_JOINTS))
        print(f"\nseed {seed}: worst zero error "
              f"{np.rad2deg(np.max(np.abs(zero_error))):.1f} deg")

        loads = {}

        def hook(now, stage, oracle, loads=loads):
            if stage == "HOLD4":
                for leg in S.LEGS:
                    loads[leg] = oracle.foot_contact(leg)[0]

        for prelift_mm in (10.0, 15.0, 20.0):
            args = base_args(zero_error=zero_error,
                             prelift=prelift_mm / 1000.0, hold3=3.0)
            model, data = S.build_sim()
            controller, oracle = S.run(model, data, args,
                                       frame_hook=hook, quiet=True)
            ok = S.report(controller, oracle, quiet=True)
            h = oracle.hold3
            if prelift_mm == 10.0:
                stance = "  ".join(f"{leg}={loads.get(leg, 0.0):.1f}N"
                                   for leg in S.LEGS)
                print(f"  4-leg stance load (ideal would be ~14.3 N each): "
                      f"{stance}")
            state = (f"clear {1000 * h['min_clearance']:.0f} mm, "
                     f"tilt {h['max_tilt_deg']:.1f} deg"
                     if h["samples"] else
                     f"{controller.aborted or controller.stop_reason}")
            print(f"  prelift {prelift_mm:4.0f} mm: "
                  f"{'PASS' if ok else 'FAIL'}  ({state})")
    print("\n-> the load asymmetry at 4-leg stance is the mis-calibration")
    print("   signature to look for on hardware before any shift test.")


def main():
    chosen = [c.upper() for c in sys.argv[1:]] or ["A", "B", "C"]
    for name in chosen:
        {"A": experiment_a, "B": experiment_b, "C": experiment_c}[name]()
        print()


if __name__ == "__main__":
    main()
