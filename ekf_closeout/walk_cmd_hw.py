#!/usr/bin/env python3
"""World-frame displacement command for the crawl -- "walk to (dx, dy)".

Purpose: give the EKF close-out a commanded ground-truth motion.  You ask for
a world-frame displacement, the dog crawls it, you tape-measure what actually
happened, and `replay_full.py --measured` compares all three (commanded, EKF,
tape).  That is the no-mocap substitute for a motion-capture room.

    walk_cmd_hw.py --goto 0.16 0 --raw-log walk_A1.npz     # 160 mm forward
    walk_cmd_hw.py --goto 0.12 0.03 --check                # feasibility only
    walk_cmd_hw.py --goto 0 0 --height 0.19 --raw-log h_190.npz   # stand only
    walk_cmd_hw.py --vx 0.0005 --duration 300              # velocity form
    walk_cmd_hw.py --self-test                             # offline, no robot

Everything below the command layer is walk1_hw.py verbatim -- the same stage
machine, gates, trips, IMU watch and EKF logging.  This file only:
  * turns (dx, dy) into a heading + cycle count + step length,
  * redirects the gait's two hard-coded +x advances along that heading,
  * validates the WHOLE cycle at the commanded heading before the bus arms,
  * writes a <raw-log>.cmd.json sidecar so the replay knows what was asked.

Frames and honest limits (printed in the banner too):
  * "world" is the STARTUP frame -- there is no yaw/turning gait, the trunk
    never rotates, so world == body heading for the whole run.  EKF yaw drifts,
    which is another reason to keep runs short.
  * displacement is quantized: the body advances step_m per 4-step cycle, so
    the granularity is one cycle (~40 mm) and the cap is 20 cycles.
  * speed is NOT freely commandable.  A cycle is ~75 s, dominated by settle
    gates that --time-scale never shortens, so |v| lands near 0.5 mm/s.  The
    velocity form exists to express "walk this far, roughly this fast"; the
    heading is exact, the speed is best-effort.
  * lateral headings are abduction-limited.  --self-test prints the feasible
    fan; infeasible headings are refused by walk1's own validator before the
    bus is armed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CRAWL = os.path.join(os.path.dirname(_HERE), "crawl_hw2.0")
for _p in (_HERE, _CRAWL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import walk1_hw  # noqa: E402

STEP_MIN_M = 0.025
STEP_MAX_M = 0.045
MAX_CYCLES = 20              # walk1's own --cycles cap
CYCLE_S = 75.0               # measured: walk_0729_1748.npz, 4 steps at scale 1
STAND_S = 45.0               # crouch -> stand -> settle before the first step


class CmdPlan(walk1_hw.WalkPlan):
    """WalkPlan that knows the commanded heading.

    `step_dir` is a class attribute because walk1's `run_hardware` and
    `validate_cycle` construct `WalkPlan(**plan_args)` with a fixed kwarg set
    (same override point walk2_fl_hw.py uses for its FL shift).
    """

    step_dir = np.array([1.0, 0.0])


class CmdSequence(walk1_hw.WalkSequence):
    """WalkSequence with the two +x advances redirected along step_dir.

    walk1 hard-codes the heading in exactly two runtime places:
      * SWING endpoint      swing_anchor + [step_m, 0]      (walk1_hw.py:319)
      * TOUCHDOWN commit    neutral + [step_m/4, 0]         (walk1_hw.py:541)
    Both are intercepted here rather than edited in place, so walk1 stays the
    validated file it is today.  The TOUCHDOWN interception reads the delta
    walk1 actually applied and re-points it, so it keeps working if that
    advance is ever retuned.
    """

    def _stream_endpoint(self, name):
        end = super()._stream_endpoint(name)
        if name == "SWING":
            end["swing_anchor"] = (self.swing_anchor
                                   + self.plan.step_m * self.plan.step_dir)
        return end

    def sweep(self, *a, **kw):
        idx0 = self.step_index
        neutral0 = np.asarray(self.plan.neutral, float).copy()
        out = super().sweep(*a, **kw)
        if self.step_index != idx0:
            # walk1 just committed a step and advanced `neutral` along +x.
            # Re-point that same magnitude along the commanded heading.
            delta = np.asarray(self.plan.neutral, float) - neutral0
            self.plan.neutral = neutral0 + np.linalg.norm(delta) * self.plan.step_dir
        return out


_STAND_ONLY = False     # set by main() for a zero-displacement command


def validate_cycle_dir(plan_args, steps=None):
    """walk1_hw.validate_cycle with the swing/advance rotated to step_dir.

    Mirrors walk1_hw.py:583-637 (IK residual + soft limits over every scripted
    waypoint of a full cycle, at the pre-lift cap) and differs only in the
    three heading terms at :609-618.  Kept here instead of edited there so
    walk1 is untouched; if that validator changes, re-sync this copy.

    `steps=False` (a stand-only command) validates just the rise -- checking a
    stepping cycle that will never run would refuse legitimate tall stands:
    H = 0.19 stands fine but cannot take a 40 mm step.  walk1's own
    run_hardware calls this through the module global, so the stand-only flag
    has to be honoured here rather than only at the call site.
    """
    steps = (not _STAND_ONLY) if steps is None else steps
    s3 = walk1_hw.s3
    plan = walk1_hw.WalkPlan(**plan_args)        # patched to CmdPlan
    u_dir = np.asarray(plan.step_dir, float)
    low, high = walk1_hw.base.soft_limits()
    q = walk1_hw.Q_CROUCH.copy()
    worst_margin = np.inf
    for h in np.linspace(0.0, 1.0, 9):
        t = plan.targets(np.zeros(2), h, None, None, 0.0, 0.0)
        for leg in walk1_hw.LEGS:
            sl = plan._sl(leg)
            q[sl] = s3._ik_to_target(leg, q[sl], t[leg])
            if not steps:      # stand-only: the rise is the whole command
                err = float(np.linalg.norm(
                    walk1_hw.dog5_kinematics.foot_position(leg, q[sl]) - t[leg]))
                if err > 1.0e-4:
                    raise ValueError(
                        f"stand at h={h:.2f}: {leg} IK residual {err:.4f} m")
                if np.any(q[sl] < low[sl]) or np.any(q[sl] > high[sl]):
                    raise ValueError(
                        f"stand at h={h:.2f}: {leg} outside soft limits")
    if not steps:
        return np.inf
    for step_i, swing in enumerate(walk1_hw.GAIT_ORDER):
        plan.swing = swing
        anchor = plan.anchors[swing].copy()
        shift_body = plan.neutral + plan.shift_vec_for(swing)
        worst_margin = min(worst_margin, plan.planned_shift_margin(swing))
        waypoints = []
        for u in np.linspace(0.0, 1.0, 6):
            body = plan.neutral + u * (shift_body - plan.neutral)
            waypoints.append((body, anchor, 0.0, 0.0))
        for pre in np.linspace(0.0, s3.PRELIFT_MAX_M, 4):
            waypoints.append((shift_body, anchor, pre, 0.0))
        waypoints.append((shift_body, anchor, s3.PRELIFT_MAX_M, 1.0))
        for u in np.linspace(0.0, 1.0, 5):
            a = anchor + u * plan.step_m * u_dir
            waypoints.append((shift_body, a, s3.PRELIFT_MAX_M, 1.0))
        landed = anchor + plan.step_m * u_dir
        waypoints.append((shift_body, landed, 0.0, 0.0))
        plan.anchors[swing] = landed
        new_neutral = plan.neutral + (plan.step_m / 4.0) * u_dir
        for u in np.linspace(0.0, 1.0, 6):
            body = shift_body + u * (new_neutral - shift_body)
            waypoints.append((body, landed, 0.0, 0.0))
        plan.neutral = new_neutral
        for body, a, pre, lift in waypoints:
            t = plan.targets(body, 1.0, swing, a, pre, lift)
            for leg in walk1_hw.LEGS:
                sl = plan._sl(leg)
                q[sl] = s3._ik_to_target(leg, q[sl], t[leg])
                err = float(np.linalg.norm(
                    walk1_hw.dog5_kinematics.foot_position(leg, q[sl]) - t[leg]
                ))
                if err > 1.0e-4:
                    raise ValueError(
                        f"step {step_i + 1} ({swing}): {leg} IK residual "
                        f"{err:.4f} m")
                if np.any(q[sl] < low[sl]) or np.any(q[sl] > high[sl]):
                    raise ValueError(
                        f"step {step_i + 1} ({swing}): {leg} outside soft "
                        f"limits at body={body}, pre={1000 * pre:.0f}mm")
    return worst_margin


def decompose(dx, dy):
    """(dx, dy) -> (unit heading, cycles, step_m, quantized displacement).

    The body advances step_m per cycle, so cycles = round(dist / 40 mm) and
    step_m absorbs the remainder within the validated 25-45 mm envelope.
    """
    dist = math.hypot(dx, dy)
    if dist < 1.0e-9:
        return np.array([1.0, 0.0]), 0, walk1_hw.DEFAULT_STEP_M, np.zeros(2)
    u = np.array([dx, dy], float) / dist
    cycles = int(round(dist / walk1_hw.DEFAULT_STEP_M))
    cycles = max(1, min(MAX_CYCLES, cycles))
    step_m = min(STEP_MAX_M, max(STEP_MIN_M, dist / cycles))
    return u, cycles, step_m, cycles * step_m * u


def plan_args_for(step_m, height):
    return dict(step_m=step_m, shift_m=walk1_hw.DEFAULT_SHIFT_M,
                lift_m=walk1_hw.DEFAULT_LIFT_M, stand_height=height,
                front_back_shift=walk1_hw.FRONT_BACK_SHIFT_M)


def feasible_fan(step_m, height, degrees=None):
    """Headings whose full cycle validates.  Returns [(deg, margin_m|None)].

    The envelope is narrow (see FEASIBLE_CONE_NOTE), so the default sweep is
    fine near forward and coarse elsewhere.
    """
    if degrees is None:
        degrees = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 20, 45, 90, 180, 270]
    saved = CmdPlan.step_dir
    out = []
    try:
        for deg in degrees:
            th = math.radians(deg)
            CmdPlan.step_dir = np.array([math.cos(th), math.sin(th)])
            try:
                out.append((deg, validate_cycle_dir(plan_args_for(step_m, height))))
            except ValueError:
                out.append((deg, None))
    finally:
        CmdPlan.step_dir = saved
    return out


# Measured 2026-07-30 with this validator (H = 0.175, pre-lift cap 30 mm):
# feasible headings are 0..+2 deg at 40 mm steps and 0..+4 deg at 30 mm.
# Everything else fails as "FL outside soft limits at pre=30mm" -- the FL
# abduction wall that walk2_fl_hw.py was written for.  Dropping the pre-lift
# cap to 20 mm widens it only to about -4..+6 deg.  So this planner commands
# a straight forward walk in practice; the (dx, dy) interface is honest about
# what it refuses rather than pretending to omnidirectional motion.
FEASIBLE_CONE_NOTE = ("validated envelope is a narrow forward cone "
                      "(~0..+4 deg); lateral/backward headings hit the FL "
                      "abduction wall and are refused")


def write_sidecar(raw_log, cmd):
    path = os.path.splitext(raw_log)[0] + ".cmd.json"
    with open(path, "w") as fh:
        json.dump(cmd, fh, indent=2)
    print(f"[cmd] wrote {path} (replay_full.py reads it for the commanded "
          "displacement)")


def self_test():
    """Offline checks: decomposition, heading redirection, feasibility fan."""
    print("== decomposition ==")
    u, n, s, q = decompose(0.12, 0.0)
    assert n == 3 and abs(s - 0.040) < 1e-9, (n, s)
    assert np.allclose(u, [1, 0]) and np.allclose(q, [0.12, 0]), (u, q)
    u, n, s, q = decompose(0.0, 0.08)
    assert np.allclose(u, [0, 1]) and n == 2, (u, n)
    u, n, s, q = decompose(0.10, 0.10)          # 141 mm at 45 deg
    assert n == 4 and STEP_MIN_M <= s <= STEP_MAX_M, (n, s)
    assert np.linalg.norm(q - [0.10, 0.10]) < 0.006, q
    u, n, s, q = decompose(-0.16, 0.0)          # backward
    assert np.allclose(u, [-1, 0]) and n == 4, (u, n)
    u, n, s, q = decompose(5.0, 0.0)            # capped at 20 cycles
    assert n == MAX_CYCLES and s == STEP_MAX_M, (n, s)
    print("  decomposition OK (forward/lateral/diagonal/backward/capped)")

    print("== feasibility fan (step 40/30 mm, H = 0.175) ==")
    cone = {}
    for step_m in (0.040, 0.030):
        fan = feasible_fan(step_m, 0.175)
        good = [d for d, m in fan if m is not None]
        cone[step_m] = good
        assert 0 in good, "forward must be feasible"
        assert any(m is None for _, m in fan), (
            "expected some heading to be refused (FL abduction wall)")
        print(f"  step {1000 * step_m:.0f} mm: feasible "
              + ", ".join(f"{d:+d}" for d in good) + " deg")
    print(f"  {FEASIBLE_CONE_NOTE}")
    for deg in (90, 180):
        th = math.radians(deg)
        CmdPlan.step_dir = np.array([math.cos(th), math.sin(th)])
        try:
            validate_cycle_dir(plan_args_for(0.030, 0.175))
            raise AssertionError(f"{deg} deg should be refused")
        except ValueError:
            pass
    CmdPlan.step_dir = np.array([1.0, 0.0])
    print("  90/180 deg refused with ValueError (clean refusal, no crash)")

    print("== heading redirection (dry run, no hardware) ==")
    walk1_hw.WalkPlan, walk1_hw.WalkSequence = CmdPlan, CmdSequence
    # 0 deg reproduces stock walk1 exactly; the widest feasible non-zero
    # heading proves the redirection actually steers the gait.
    step_m = 0.030          # the wider cone; 0 deg still reproduces stock walk1
    for deg in (0.0, float(max(cone[step_m]))):
        th = math.radians(deg)
        CmdPlan.step_dir = np.array([math.cos(th), math.sin(th)])
        plan = walk1_hw.WalkPlan(**plan_args_for(step_m, 0.175))
        seq = walk1_hw.WalkSequence(0.0, plan)
        now = 0.0

        def spin(stop):
            nonlocal now
            deadline = now + 120.0
            while seq.stage not in stop:
                now += 0.048
                seq.sweep(now, seq.q_cmd.copy(),
                          np.zeros(walk1_hw.N_JOINTS), True,
                          np.zeros(walk1_hw.N_JOINTS))
                for k in range(4):
                    seq.refine_leg(k)
                assert now < deadline, f"stuck in {seq.stage}"

        def enter():
            nonlocal now
            now += walk1_hw.base.WAIT_DWELL_S + 0.1
            ok, msg = seq.request_next(now, seq.q_cmd,
                                       np.zeros(walk1_hw.N_JOINTS), True)
            assert ok, msg

        spin(("WAIT_CROUCH",))
        enter()
        spin(("HOLD4",))
        for _ in range(4):
            enter(); spin(("WAIT_UNLOAD",))
            enter(); spin(("WAIT_SWING",))
            enter(); spin(("HOLD4",))
        assert seq.step_index == 4 and seq.aborted is None, seq.aborted
        want = step_m * CmdPlan.step_dir
        assert np.linalg.norm(plan.neutral - want) < 1e-9, (plan.neutral, want)
        for leg in walk1_hw.LEGS:
            crouch = walk1_hw.dog5_kinematics.foot_position(
                leg, walk1_hw.Q_CROUCH[plan._sl(leg)])[:2]
            moved = plan.anchors[leg] - crouch
            # anchors are committed from MEASURED FK, so they carry the IK
            # tolerance (~0.1 mm), not machine precision (walk1_hw.py:1186)
            assert np.linalg.norm(moved - step_m * CmdPlan.step_dir) < 5.0e-4, (
                leg, moved)
        print(f"  heading {deg:+5.1f} deg: body walked "
              f"{1000 * np.linalg.norm(plan.neutral):.1f} mm along "
              f"({CmdPlan.step_dir[0]:+.3f},{CmdPlan.step_dir[1]:+.3f}); "
              "every anchor followed")
    CmdPlan.step_dir = np.array([1.0, 0.0])
    print("  0 deg reproduces stock walk1 (neutral +step along +x); the "
          "non-zero heading steers both the swing and the body advance")
    print("\nself-test PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--goto", type=float, nargs=2, metavar=("DX", "DY"),
                    help="world-frame displacement in metres (startup frame)")
    ap.add_argument("--vx", type=float, default=None,
                    help="world-frame velocity m/s (with --duration)")
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds, for the --vx/--vy form")
    ap.add_argument("--height", type=float, default=None,
                    help="stand height m (0.15-0.19); forwarded as "
                         "--stand-height")
    ap.add_argument("--check", action="store_true",
                    help="offline feasibility of the command; no hardware")
    ap.add_argument("--self-test", action="store_true",
                    help="offline unit tests; no hardware")
    args, rest = ap.parse_known_args()

    if "--help" in rest or "-h" in rest:
        print(__doc__)
        print("---- underlying walk1_hw options: ----\n")
        sys.argv = [sys.argv[0], "--help"]
        return walk1_hw.main()
    if args.self_test:
        walk1_hw.WalkPlan, walk1_hw.WalkSequence = CmdPlan, CmdSequence
        walk1_hw.validate_cycle = validate_cycle_dir
        return self_test()

    height = args.height if args.height is not None \
        else walk1_hw.DEFAULT_STAND_HEIGHT_M
    if not 0.15 <= height <= 0.19:
        print("error: --height must be 0.15-0.19 m", file=sys.stderr)
        return 2

    time_scale = None
    if args.vx is not None:
        if args.duration is None or args.duration <= 0:
            print("error: --vx/--vy need a positive --duration",
                  file=sys.stderr)
            return 2
        dx, dy = args.vx * args.duration, args.vy * args.duration
        speed = math.hypot(args.vx, args.vy)
        print(f"[cmd] velocity form: ({1000 * args.vx:+.2f}, "
              f"{1000 * args.vy:+.2f}) mm/s for {args.duration:.0f} s "
              f"-> goto ({dx:+.3f}, {dy:+.3f}) m")
    elif args.goto is not None:
        dx, dy = args.goto
        speed = None
    else:
        print("error: give --goto DX DY (or --vx/--vy with --duration)",
              file=sys.stderr)
        return 2

    u, cycles, step_m, quantized = decompose(dx, dy)
    heading_deg = math.degrees(math.atan2(u[1], u[0]))
    dist = math.hypot(dx, dy)
    walking = dist > 1.0e-9

    # Install the command layer before anything validates or runs: walk1's
    # run_hardware builds WalkPlan/WalkSequence and calls validate_cycle by
    # module-global name, so these assignments cover the whole run.
    global _STAND_ONLY
    _STAND_ONLY = not walking
    CmdPlan.step_dir = u
    walk1_hw.WalkPlan = CmdPlan
    walk1_hw.WalkSequence = CmdSequence
    walk1_hw.validate_cycle = validate_cycle_dir

    print("[cmd] ==== world-frame command ====")
    if walking:
        est_s = STAND_S + cycles * CYCLE_S * (
            float(rest[rest.index("--time-scale") + 1])
            if "--time-scale" in rest else 1.0)
        print(f"[cmd] request ({dx:+.3f}, {dy:+.3f}) m = "
              f"{1000 * dist:.0f} mm at heading {heading_deg:+.1f} deg")
        print(f"[cmd] plan    {cycles} cycle(s) x {1000 * step_m:.1f} mm = "
              f"({1000 * quantized[0]:+.1f}, {1000 * quantized[1]:+.1f}) mm; "
              f"quantization residual "
              f"{1000 * np.linalg.norm(quantized - [dx, dy]):.1f} mm")
        print(f"[cmd] ~{est_s / 60.0:.0f} min of motion "
              f"(mean |v| ~ {1000 * dist / max(est_s, 1):.2f} mm/s -- a crawl "
              "cycle is gate-dominated, speed is not commandable)")
        if speed is not None:
            print(f"[cmd] requested speed {1000 * speed:.2f} mm/s is "
                  "best-effort: heading is exact, |v| is what the gates allow")
    else:
        print("[cmd] zero displacement: STAND ONLY (crouch -> stand -> HOLD4, "
              "no steps).  Press X at HOLD4 to stop and write the log.")
    print(f"[cmd] stand height {1000 * height:.0f} mm; world frame = STARTUP "
          "frame (no turning gait, trunk never yaws)")

    try:
        margin = validate_cycle_dir(plan_args_for(step_m, height))
    except ValueError as exc:
        print(f"[cmd] INFEASIBLE at heading {heading_deg:+.1f} deg, step "
              f"{1000 * step_m:.1f} mm: {exc}", file=sys.stderr)
        print("[cmd] try a heading closer to +/-x, a shorter step, or run "
              "--self-test to print the feasible fan.", file=sys.stderr)
        return 2
    if walking:
        print(f"[cmd] cycle validated at this heading: worst planned margin "
              f"{1000 * margin:.1f} mm (gate "
              f"{1000 * walk1_hw.MIN_STEP_MARGIN_M:.0f} mm)")
    else:
        print(f"[cmd] stand pose validated at {1000 * height:.0f} mm "
              "(no steps to validate)")

    if args.check:
        return 0

    raw_log = rest[rest.index("--raw-log") + 1] if "--raw-log" in rest else None
    if raw_log:
        write_sidecar(raw_log, {
            "goto_m": [dx, dy],
            "heading_deg": heading_deg,
            "n_cycles": cycles if walking else 0,
            "step_m": step_m,
            "quantized_dxy_m": [float(quantized[0]), float(quantized[1])],
            "stand_height_m": height,
            "time_scale": (float(rest[rest.index("--time-scale") + 1])
                           if "--time-scale" in rest else 1.0),
        })

    forwarded = ["--stand-height", f"{height}"]
    if walking:
        forwarded += ["--step", f"{step_m}", "--cycles", f"{cycles}"]
        if "--auto" not in rest:
            forwarded.append("--auto")
    sys.argv = [sys.argv[0]] + forwarded + rest
    return walk1_hw.main()


if __name__ == "__main__":
    raise SystemExit(main())
