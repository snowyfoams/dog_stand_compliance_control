#!/usr/bin/env python3
r"""Kinematic statically-stable crawl for DOG5 in MuJoCo — no dynamics.

Walking as position commands only, built on the proven pieces of
``stand3_dog5.py`` and mirroring the hardware crawl plan
(``crawl_dog5_hw.py`` on the ``dog5-live-mirror`` branch):

* gait order RR -> FR -> RL -> FL, one foot in the air at a time;
* every step: SHIFT (body diagonally away from the swing corner)
  -> SHIFT_GATE (encoder-FK margin >= 15 mm) -> UNLOAD (pre-lift 10 mm)
  -> UNLOAD_GATE (swing pitch/knee measure unloaded) -> LIFT -> SWING
  (foot advances one step length) -> LOWER -> LOAD -> RECENTER (body
  returns to neutral advanced by step/4);
* stance feet have fixed conceptual ground anchors — body motion is
  commanded by moving every foot target the opposite way in the trunk
  frame, so foot slip is never commanded;
* the controller sees only CAN telemetry (quantized encoders, driver
  speed, measured torque); all FK/IK from NumPy ``dog5_kinematics``.

Run with the project venv:
    D:\mujoco\.venv\Scripts\python.exe walk_dog5.py --self-test
    D:\mujoco\.venv\Scripts\python.exe walk_dog5.py --headless
    D:\mujoco\.venv\Scripts\python.exe walk_dog5.py              # viewer
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import dog5_kinematics  # noqa: E402
import stand3_dog5 as S  # noqa: E402

LEGS = S.LEGS
N_JOINTS = S.N_JOINTS

GAIT_ORDER = ("RR", "FR", "RL", "FL")     # crawl_dog5_hw lateral sequence

# (phase, kind, duration s).  Streamed phases interpolate targets with a
# smoothstep; gates hold targets until their condition (crawl-style).
WALK_PHASES = [
    ("SHIFT", "stream", 1.2),
    ("SHIFT_GATE", "gate", None),
    ("UNLOAD", "stream", 0.5),
    ("UNLOAD_GATE", "gate", None),
    ("LIFT", "stream", 0.4),
    ("SWING", "stream", 0.8),
    ("LOWER", "stream", 0.5),
    ("LOAD", "dwell", 0.5),
    ("RECENTER", "stream", 1.2),
]
PHASE_INDEX = {name: i for i, (name, _, _) in enumerate(WALK_PHASES)}
AIRBORNE = ("LIFT", "SWING", "LOWER")

DEFAULT_STEP_M = 0.03          # crawl_dog5_hw default forward step
DEFAULT_SHIFT_M = 0.04         # 0.03 in the crawl; sprawl rear-swing margin
                               # is thinner, 0.04 gives ~25 mm planned
DEFAULT_PRELIFT_M = 0.010      # proven in hw_gap_experiments B (mu = 1.0)
DEFAULT_SWING_H_M = 0.025      # crawl swing clearance, on top of the pre-lift
DEFAULT_CYCLES = 2
# The sprawl stand has the front legs near full extension: stepping forward
# from it is unreachable (self-test residual 14 mm).  One GATHER cycle first
# re-anchors each foot 50 mm inboard with the same statically-stable step
# machinery (no planted-foot dragging), then the walk cycles begin.
DEFAULT_GATHER_M = 0.05
GATE_TIMEOUT_S = 6.0


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return 3.0 * u * u - 2.0 * u * u * u


class WalkPlan:
    """World foot anchors + body path, all from NumPy FK (crawl planner)."""

    def __init__(self, step_m, shift_m, prelift_m, swing_h_m, cycles,
                 gather_m=DEFAULT_GATHER_M):
        self.step_m = float(step_m)
        self.shift_m = float(shift_m)
        self.prelift_m = float(prelift_m)
        self.swing_h_m = float(swing_h_m)
        self.gather_m = float(gather_m)
        self.n_steps = 4 + 4 * int(cycles)    # one GATHER cycle, then walk
        self.anchor = {}
        self.z = {}
        for i, leg in enumerate(LEGS):
            foot = dog5_kinematics.foot_position(
                leg, S.Q_STAND[3 * i:3 * i + 3])
            self.anchor[leg] = foot[:2].copy()
            self.z[leg] = float(foot[2])
        self.body = np.zeros(2)
        self.step_i = 0
        self._begin_step()

    def _begin_step(self):
        self.swing = GAIT_ORDER[self.step_i % 4]
        self.stance = [leg for leg in LEGS if leg != self.swing]
        signs = np.sign(self.anchor[self.swing] - self.body)
        self.shifted = self.body - self.shift_m * signs / np.sqrt(2.0)
        if self.step_i < 4:
            # GATHER: re-anchor this foot inboard (front back, rear forward);
            # the body does not advance.
            inboard = -np.sign(self.anchor[self.swing][0] - self.body[0])
            displacement = np.array([inboard * self.gather_m, 0.0])
            self.next_body = self.body.copy()
        else:
            displacement = np.array([self.step_m, 0.0])
            self.next_body = self.body + np.array([self.step_m / 4.0, 0.0])
        self.old_anchor = self.anchor[self.swing].copy()
        self.new_anchor = self.old_anchor + displacement

    def end_step(self):
        self.anchor[self.swing] = self.new_anchor.copy()
        self.body = self.next_body.copy()
        self.step_i += 1
        if not self.done:
            self._begin_step()

    @property
    def done(self):
        return self.step_i >= self.n_steps

    def targets(self, phase, s):
        """Trunk-frame foot targets during ``phase`` at smoothstep ``s``."""
        if phase == "SHIFT":
            body = self.body + s * (self.shifted - self.body)
        elif phase == "RECENTER":
            body = self.shifted + s * (self.next_body - self.shifted)
        else:
            body = self.shifted

        swing_xy = self.old_anchor.copy()
        if phase == "SWING":
            swing_xy = self.old_anchor + s * (self.new_anchor - self.old_anchor)
        elif phase in ("LOWER", "LOAD", "RECENTER"):
            swing_xy = self.new_anchor.copy()

        dz = 0.0
        if phase == "UNLOAD":
            dz = s * self.prelift_m
        elif phase == "LIFT":
            dz = self.prelift_m + s * self.swing_h_m
        elif phase == "SWING":
            dz = self.prelift_m + self.swing_h_m
        elif phase == "LOWER":
            dz = (self.prelift_m + self.swing_h_m) * (1.0 - s)
        elif phase in ("SHIFT_GATE", "UNLOAD_GATE"):
            dz = self.prelift_m if phase == "UNLOAD_GATE" else 0.0

        targets = {}
        for leg in LEGS:
            xy = swing_xy if leg == self.swing else self.anchor[leg]
            z = self.z[leg] + (dz if leg == self.swing else 0.0)
            targets[leg] = np.array([*(xy - body), z])
        return targets

    def q_targets(self, warm_q, phase, s):
        targets = self.targets(phase, s)
        q = np.asarray(warm_q, dtype=float).copy()
        for i, leg in enumerate(LEGS):
            sl = slice(3 * i, 3 * i + 3)
            q[sl] = S._ik_to_target(leg, q[sl], targets[leg])
        return q

    def fk_margin(self, q_enc):
        """Stance-triangle margin from measured encoders, CoM at origin."""
        feet = []
        for i, leg in enumerate(LEGS):
            if leg in self.stance:
                feet.append(dog5_kinematics.foot_position(
                    leg, q_enc[3 * i:3 * i + 3])[:2])
        return S.support_triangle_margin(feet)


class WalkController(S.Stand3Controller):
    """Stand up with the hardware method, then crawl kinematically."""

    def __init__(self, plan, torque_trip, unload_trip,
                 crouch_dps, stand_dps, stream_dps):
        super().__init__(plan, torque_trip, unload_trip,
                         crouch_dps, stand_dps, stream_dps, start="crouch")
        # Linear stand-up prefix only; the walk cycle replaces the rest.
        self.stages = [("CROUCH", "gate", None), ("STAND", "gate", None),
                       ("HOLD4", "dwell", 1.0)]
        self.stage_index = {}
        self.walking = False
        self.phase_i = 0
        self.phase_started = 0.0
        self.aborting = False

    @property
    def stage(self):
        if self.walking or self.stage_i >= len(self.stages):
            return f"{WALK_PHASES[self.phase_i][0]}:{self.plan.swing}"
        return self.stages[self.stage_i][0]

    def _cap_for_stage(self):
        if self.walking:
            return self.caps["stream"]
        return super()._cap_for_stage()

    def _phase(self, name, now):
        self.phase_i = PHASE_INDEX[name]
        self.phase_started = now
        self.settle_since = None

    def _walk_abort(self, now, reason):
        """Crawl-style graceful abort: ground the foot, recenter, stop."""
        self.aborting = True
        self.events.append((now, "ABORT", reason))
        name = WALK_PHASES[self.phase_i][0]
        self._phase("LOWER" if name in AIRBORNE + ("UNLOAD", "UNLOAD_GATE")
                    else "RECENTER", now)

    def sweep(self, now, q_enc, qd_reported, tau_measured):
        if not self.walking:
            q_cmd = super().sweep(now, q_enc, qd_reported, tau_measured)
            if self.stage_i >= len(self.stages) and self.stop_reason is None:
                self.walking = True
                self.phase_started = now
                self.events.append((now, f"WALK:{self.plan.swing}", "start"))
            return q_cmd

        self.qd = self.velocity.update(q_enc, now)
        self.stop_reason = self._check_trips(q_enc, qd_reported,
                                             tau_measured, now)
        if self.stop_reason:
            return self.q_cmd

        name, kind, duration = WALK_PHASES[self.phase_i]

        # loss of support during swing is a hard stop (crawl e-stop)
        if name in AIRBORNE and self.plan.fk_margin(q_enc) < 0.0:
            self.stop_reason = f"support margin lost during {name}"
            return self.q_cmd

        if kind == "stream":
            s = _smoothstep((now - self.phase_started) / duration)
            self.q_cmd = self.plan.q_targets(self.q_cmd, name, s)
            if now - self.phase_started >= duration:
                if name == "RECENTER":
                    self._finish_step(now)
                else:
                    self._phase(WALK_PHASES[self.phase_i + 1][0], now)
        elif name == "SHIFT_GATE":
            if self._settled(now, q_enc):
                margin = self.plan.fk_margin(q_enc)
                if margin >= S.MIN_LIFTOFF_MARGIN_M:
                    self.events.append(
                        (now, self.stage, f"margin {1000 * margin:.0f} mm"))
                    self._phase("UNLOAD", now)
                elif now - self.phase_started > GATE_TIMEOUT_S:
                    self._walk_abort(now, f"liftoff margin "
                                          f"{1000 * margin:.1f} mm too small")
            elif now - self.phase_started > GATE_TIMEOUT_S:
                self._walk_abort(now, "SHIFT did not settle")
        elif name == "UNLOAD_GATE":
            i = LEGS.index(self.plan.swing)
            swing_tau = np.abs(tau_measured[3 * i + 1:3 * i + 3])
            if self._settled(now, q_enc) and np.all(swing_tau <= self.unload_trip):
                self._phase("LIFT", now)
            elif now - self.phase_started > GATE_TIMEOUT_S:
                self._walk_abort(now, f"unload gate: swing max|tau|="
                                      f"{float(np.max(swing_tau)):.2f} N*m")
        elif kind == "dwell":
            if now - self.phase_started >= duration:
                self._phase("RECENTER", now)
        return self.q_cmd

    def _finish_step(self, now):
        if self.aborting:
            self.stop_reason = "aborted step; recentered and stopped"
            self.aborted = self.events[-1][2] if self.aborted is None \
                else self.aborted
            return
        self.plan.end_step()
        if self.plan.done:
            self.stop_reason = "sequence complete"
        else:
            self.events.append((now, f"WALK:{self.plan.swing}",
                                f"step {self.plan.step_i + 1}/{self.plan.n_steps}"))
            self._phase("SHIFT", now)


class WalkOracle(S.Oracle):
    """External instrumentation for the gait — judges, never steers."""

    def reset(self):
        super().reset()
        self.walk = dict(min_true_margin=np.inf, min_fk_margin=np.inf,
                         min_swing_clearance=np.inf, max_swing_contact=0.0,
                         max_tilt_deg=0.0, swing_samples=0)
        self.body_x0 = None

    def sample(self, stage, q_enc, tau):
        self.max_abs_tau = np.maximum(self.max_abs_tau, np.abs(tau))
        if ":" not in stage:
            return
        if self.body_x0 is None:
            self.body_x0 = float(self.d.xpos[self.trunk][0])
        w = self.walk
        w["max_tilt_deg"] = max(w["max_tilt_deg"], self.tilt_deg())
        phase = stage.split(":")[0]
        if phase in AIRBORNE:
            w["min_true_margin"] = min(w["min_true_margin"], self.true_margin())
            w["min_fk_margin"] = min(w["min_fk_margin"],
                                     self.plan.fk_margin(q_enc))
        if phase == "SWING":
            swing = self.plan.swing
            clearance = float(
                self.d.site_xpos[self.site[swing]][2]) - 0.02
            w["min_swing_clearance"] = min(w["min_swing_clearance"], clearance)
            w["max_swing_contact"] = max(w["max_swing_contact"],
                                         self.foot_contact(swing)[0])
            w["swing_samples"] += 1

    def travel_x(self):
        if self.body_x0 is None:
            return 0.0
        return float(self.d.xpos[self.trunk][0]) - self.body_x0

    def verdict(self, controller):
        w = self.walk
        expected = self.plan.step_m * (self.plan.n_steps - 4) / 4.0
        travel = self.travel_x()
        checks = [
            ("finished all steps (no abort/e-stop)",
             controller.aborted is None
             and controller.stop_reason == "sequence complete"),
            ("swing phases sampled", w["swing_samples"] > 0),
            ("swing foot airborne during SWING (<= 0.05 N)",
             w["max_swing_contact"] <= 0.05),
            ("swing clearance >= 10 mm", w["min_swing_clearance"] >= 0.010),
            ("trunk tilt <= 5 deg", w["max_tilt_deg"] <= 5.0),
            ("true CoM margin >= 10 mm in swing", w["min_true_margin"] >= 0.010),
            ("encoder FK margin >= 10 mm in swing", w["min_fk_margin"] >= 0.010),
            (f"body advanced >= 60% of {1000 * expected:.0f} mm",
             travel >= 0.6 * expected),
        ]
        return all(ok for _, ok in checks), checks


def run(model, data, args, viewer=None, frame_hook=None, quiet=False):
    import mujoco
    plan = WalkPlan(args.step, args.shift, args.prelift, args.swing_height,
                    args.cycles)
    controller = WalkController(plan, args.torque_trip, args.unload_trip,
                                args.crouch_dps, args.stand_dps,
                                args.stream_dps)
    oracle = WalkOracle(model, data, plan)

    qadr = np.array([model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{j}_{leg}")]
        for leg in LEGS for j in ("hip_abd", "hip_pitch", "knee")])
    servo = S.NativePositionServo(data.qpos[qadr],
                                  kp=S.KP_SERVO * args.kp_scale)

    steps_per_tick = max(1, round(1.0 / (S.CONTROL_HZ * model.opt.timestep)))
    slot = 0
    q_targets = data.qpos[qadr].copy()
    last_print = -np.inf
    t_wall0 = time.perf_counter()

    while controller.stop_reason is None:
        now = data.time
        if slot % N_JOINTS == 0:
            q_enc, qd_reported, tau_measured = servo.telemetry()
            q_targets = controller.sweep(now, q_enc, qd_reported, tau_measured)
            oracle.sample(controller.stage, q_enc, tau_measured)
            if not quiet and now - last_print >= 1.0 / S.STATUS_HZ:
                print(f"t={now:6.2f}  {controller.stage:<15} "
                      f"tilt={oracle.tilt_deg():4.1f} deg  "
                      f"travel={1000 * oracle.travel_x():6.1f} mm  "
                      f"margin={1000 * oracle.true_margin():+6.1f} mm",
                      flush=True)
                last_print = now
        motor = slot % N_JOINTS
        servo.command(motor, q_targets[motor], controller._cap_for_stage())
        for _ in range(steps_per_tick):
            data.ctrl[:] = servo.step(data.qpos[qadr], model.opt.timestep)
            mujoco.mj_step(model, data)
        if frame_hook is not None:
            frame_hook(now, controller.stage, oracle)
        if viewer is not None:
            if not viewer.is_running():
                controller.stop_reason = "viewer closed"
                break
            viewer.sync()
            lag = data.time - (time.perf_counter() - t_wall0)
            if lag > 0:
                time.sleep(lag)
            elif lag < -0.5:
                t_wall0 = time.perf_counter() - data.time
        slot += 1
        if data.time > 400.0:
            controller.stop_reason = "wall timeout"
            break
    return controller, oracle


def report(controller, oracle, quiet=False):
    passed, checks = oracle.verdict(controller)
    if not quiet:
        print(f"\nstop: {controller.stop_reason}"
              + (f"  (ABORTED: {controller.aborted})"
                 if controller.aborted else ""))
        for now, stage, note in controller.events:
            print(f"  t={now:6.2f}  {stage:<16} {note}")
        w = oracle.walk
        walk_steps = max(0, oracle.plan.step_i - 4)
        print(f"\ngait metrics over {oracle.plan.step_i} steps "
              f"(4 gather + {walk_steps} walk):")
        print(f"  body travel            : {1000 * oracle.travel_x():.1f} mm "
              f"(commanded {1000 * oracle.plan.step_m * walk_steps / 4:.0f} mm)")
        print(f"  swing contact max      : {w['max_swing_contact']:.3f} N")
        print(f"  swing clearance min    : {1000 * w['min_swing_clearance']:.1f} mm")
        print(f"  trunk tilt max         : {w['max_tilt_deg']:.2f} deg")
        print(f"  true CoM margin min    : {1000 * w['min_true_margin']:.1f} mm")
        print(f"  encoder FK margin min  : {1000 * w['min_fk_margin']:.1f} mm")
        worst = int(np.argmax(oracle.max_abs_tau))
        print(f"  peak measured torque   : {S.JOINT_LABELS[worst]} "
              f"{oracle.max_abs_tau[worst]:.2f} N*m")
        print("\nverdict:")
        for label, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        print(f"\n{'PASS' if passed else 'FAIL'}: kinematic crawl "
              f"({'proven' if passed else 'not proven'} under hw constraints)")
    return passed


def self_test(args):
    """Offline: every planned waypoint IK-reachable inside soft limits."""
    plan = WalkPlan(args.step, args.shift, args.prelift, args.swing_height,
                    args.cycles)
    low, high = S.soft_limits()
    q = S.Q_STAND.copy()
    worst_err = 0.0
    worst_at = None
    min_margin = np.inf
    while not plan.done:
        for name, kind, _ in WALK_PHASES:
            if kind != "stream":
                continue
            for s in np.linspace(0.0, 1.0, 7):
                q = plan.q_targets(q, name, s)
                targets = plan.targets(name, s)
                for i, leg in enumerate(LEGS):
                    sl = slice(3 * i, 3 * i + 3)
                    err = float(np.linalg.norm(dog5_kinematics.foot_position(
                        leg, q[sl]) - targets[leg]))
                    if err > worst_err:
                        worst_err, worst_at = err, (plan.step_i, name, leg)
                    if np.any(q[sl] < low[sl]) or np.any(q[sl] > high[sl]):
                        raise SystemExit(
                            f"FAIL: {leg} outside soft limits in step "
                            f"{plan.step_i} {name} s={s:.2f}: {q[sl]}")
        # planned liftoff margin for this step (CoM at origin, shifted body)
        feet = [plan.anchor[leg] - plan.shifted for leg in plan.stance]
        min_margin = min(min_margin, S.support_triangle_margin(feet))
        plan.end_step()
    if worst_err > 1.0e-3:
        raise SystemExit(f"FAIL: IK residual {1000 * worst_err:.2f} mm "
                         f"at {worst_at} (step too long for the sprawl reach)")
    print(f"all {plan.n_steps} steps IK-reachable, in-limits; "
          f"worst residual {1000 * worst_err:.3f} mm at {worst_at}")
    print(f"worst planned liftoff margin {1000 * min_margin:.1f} mm "
          f"(gate {1000 * S.MIN_LIFTOFF_MARGIN_M:.0f} mm)")
    print("walk_dog5 offline self-test PASS")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--step", type=float, default=DEFAULT_STEP_M)
    parser.add_argument("--shift", type=float, default=DEFAULT_SHIFT_M)
    parser.add_argument("--prelift", type=float, default=DEFAULT_PRELIFT_M)
    parser.add_argument("--swing-height", type=float,
                        default=DEFAULT_SWING_H_M)
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    parser.add_argument("--torque-trip", type=float,
                        default=S.DEFAULT_TORQUE_TRIP_NM)
    parser.add_argument("--unload-trip", type=float,
                        default=S.DEFAULT_UNLOAD_TAU_TRIP_NM)
    parser.add_argument("--crouch-dps", type=float,
                        default=S.DEFAULT_CROUCH_DPS)
    parser.add_argument("--stand-dps", type=float,
                        default=S.DEFAULT_STAND_DPS)
    parser.add_argument("--stream-dps", type=float,
                        default=S.DEFAULT_STREAM_DPS)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    args = parser.parse_args()

    if args.self_test:
        return self_test(args)

    model, data = S.build_sim(kp_scale=args.kp_scale, start="crouch")
    if args.headless:
        controller, oracle = run(model, data, args)
        return 0 if report(controller, oracle) else 1

    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        controller, oracle = run(model, data, args, viewer=viewer)
    return 0 if report(controller, oracle) else 1


if __name__ == "__main__":
    raise SystemExit(main())
