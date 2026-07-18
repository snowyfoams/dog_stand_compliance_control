#!/usr/bin/env python3
r"""Static crawl for DOG5 in MuJoCo, mirroring the hardware position track.

Sim proof for the crawl of ``crawl_dog5_hw.py`` (dog5-live-mirror) under the
SIM_APPROACH_HW.md rules: the controller runs the hardware's logic, sees only
the hardware's signals, and commands only native ``0xA4`` position targets.
The plant interface, encoder discipline, and CAN round-robin are reused
verbatim from ``stand3_dog5.py``:

* motion is native-position-command style only (joint targets + motor-side
  speed caps, no torque law);
* feedback is encoder-only at ~20.8 Hz (one round-robin sweep): quantized q,
  ``EncoderVelocity``, and per-joint measured torque (driver telemetry);
* every control-side FK/IK/Jacobian comes from ``dog5_kinematics`` (NumPy);
  trunk pose, contacts, and true CoM feed only the *oracle*.

Each step is the crawl's phase machine, in the position-mode form that the
hardware-gap experiments (SIM_APPROACH_HW.md section 7) selected:

    SHIFT -> gate: settle + encoder FK margin >= 15 mm
    PRELIFT (15 mm) -> gate: encoder-FK stance-plane rise AND swing
                             pitch/knee |tau| <= 0.7 N*m  (clearance proof)
    LIFT -> SWING -> LOWER   (horizontal motion only after the clear gate)
    TOUCHDOWN gate: encoder-FK foot back on the stance plane; the MEASURED
                    foot position is committed as the new anchor
    LOAD -> RECENTER -> gate: settle    (then the next leg)

Gate failures abort like the crawl does: lower the foot where it is, commit
the measured anchor, recenter, end the batch ABORTED (only support-triangle
loss and the safety trips stop the run).

The hardware stand pose is a sprawl (feet x ~ +/-0.42 m) from which forward
steps are IK-unreachable for the front legs.  ``--swing-test`` therefore
crawls in place on the sprawl (the first hardware test), while walk mode
first runs a REPOSE cycle -- each leg's first swing re-places its foot on a
tucked stance rectangle (``--walk-x/--walk-y``), using the same shift/
pre-lift/swing machinery -- and then walks forward.

Run with the project venv:
    D:\mujoco\.venv\Scripts\python.exe crawl_dog5_sim.py              # viewer
    D:\mujoco\.venv\Scripts\python.exe crawl_dog5_sim.py --self-test  # offline
    D:\mujoco\.venv\Scripts\python.exe crawl_dog5_sim.py --headless   # verdict
    D:\mujoco\.venv\Scripts\python.exe crawl_dog5_sim.py --headless --swing-test
    D:\mujoco\.venv\Scripts\python.exe crawl_dog5_sim.py --sweep      # robustness
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import dog5_kinematics  # noqa: E402
import stand3_dog5 as s3  # noqa: E402  (servo emulation + hw constants)

LEGS = s3.LEGS
N_JOINTS = s3.N_JOINTS
JOINT_LABELS = s3.JOINT_LABELS
Q_CROUCH = s3.Q_CROUCH
Q_STAND = s3.Q_STAND

# ---- constants copied from crawl_dog5_hw.py (dog5-live-mirror) ----
GAIT_ORDER = ("RR", "FR", "RL", "FL")
# The REPOSE cycle tucks legs in diagonal order: lateral order leaves a
# 2 mm support margin when both same-side legs are already tucked, diagonal
# order keeps every mixed sprawl/tucked triangle >= ~20 mm at 0.03 shift.
REPOSE_ORDER = ("RR", "FL", "RL", "FR")
MIN_LIFTOFF_MARGIN_M = s3.MIN_LIFTOFF_MARGIN_M          # 0.015
SUPPORT_MARGIN_ESTOP_M = 0.0
SUPPORT_MARGIN_ESTOP_STREAK = 3
TOUCHDOWN_PLANE_TOL_M = 0.008
MIN_JACOBIAN_SINGULAR = 0.005   # stand_dog5_recorded_hw.MIN_JACOBIAN_SINGULAR

# ---- position-mode crawl tuning (SIM_APPROACH_HW.md sections 6-7) ----
# The crawl hw default shift is 0.03 m, but on the sprawl the rear-leg
# steps then plan only ~16 mm of margin -- one CoM-offset away from the
# 15 mm gate, and a -10 mm CoM leaves single-digit true margins.  0.04
# plans ~25 mm.  (0.04 only fits the rear abduction soft limit with the
# walk-stance y at 0.10; at y=0.112 it exceeds -1.75 rad in the repose
# swing.)
DEFAULT_SHIFT_M = 0.04
DEFAULT_STEP_M = 0.03
DEFAULT_SWING_HEIGHT_M = 0.025  # crawl default swing clearance
# Unloading a corner tilts the trunk toward it (~1.4 deg measured), which
# physically eats ~12 mm of the commanded pre-lift on this sprawl.  20 mm
# (section 7B's upper value) leaves a real clearance the gate can prove.
DEFAULT_PRELIFT_M = 0.020
# Clear gate: encoder-FK plane rise must exceed this ABSOLUTE clearance.
# (A fraction of the commanded pre-lift misreads corner sag as failure.)
CLEAR_PLANE_RISE_MIN_M = 0.005
# If the clear gate times out, raise the pre-lift and retry before giving
# up: corner sag scales with servo compliance (unknown per section 2.1),
# and the measured plane rise tells the truth either way.  The whole gait
# is limit/reach-validated offline up to PRELIFT_MAX_M.
PRELIFT_RETRY_STEP_M = 0.008
PRELIFT_MAX_M = 0.036
# The swing height is adapted per step from the clear gate's measurement:
# command (measured corner sag + this clearance), so grippy floors or soft
# servos that tilt the trunk more automatically swing higher.  Bounded by
# the same offline-validated envelope (PRELIFT_MAX_M + 5 mm).  30 mm keeps
# >= 8 mm of real clearance at mu=2.0, where lateral elastic wind-up
# (feet gripping through the shift) releases mid-swing and costs up to
# ~19 mm on the first swing off the fully wound-up sprawl.
SWING_CLEARANCE_MIN_M = 0.030
DEFAULT_UNLOAD_TAU_TRIP_NM = s3.DEFAULT_UNLOAD_TAU_TRIP_NM   # 0.7 (position mode)
DEFAULT_TORQUE_TRIP_NM = s3.DEFAULT_TORQUE_TRIP_NM           # 6.0 (3-leg stance)
DEFAULT_GAIT_CYCLES = 1
WALK_STANCE_X_M = 0.34          # tucked walk rectangle ("stand less sprawled");
                                # x matches the crawl_dog5_hw self-test stance
WALK_STANCE_Y_M = 0.10          # y stays at the sprawl's natural track width:
                                # 0.112 breaks the rear abd soft limit in the
                                # repose swing (RR -100.3 deg at the limit)

T_SHIFT = 2.0                   # streamed phase durations, s (250 motor-dps cap)
T_PRELIFT = 1.0
T_LIFT = 0.8
T_SWING = 1.2
T_LOWER = 1.0
T_LOAD = 0.8
T_RECENTER = 2.0
GATE_TIMEOUT_S = 6.0

PHASES = (
    ("SHIFT", "stream", T_SHIFT),
    ("SHIFT_GATE", "gate", GATE_TIMEOUT_S),
    ("PRELIFT", "stream", T_PRELIFT),
    ("CLEAR_GATE", "gate", GATE_TIMEOUT_S),
    ("LIFT", "stream", T_LIFT),
    ("SWING", "stream", T_SWING),
    ("LOWER", "stream", T_LOWER),
    ("TOUCHDOWN_GATE", "gate", GATE_TIMEOUT_S),
    ("LOAD", "dwell", T_LOAD),
    ("RECENTER", "stream", T_RECENTER),
    ("RECENTER_GATE", "gate", GATE_TIMEOUT_S),
)
PHASE_KIND = {name: kind for name, kind, _ in PHASES}
PHASE_DURATION = {name: duration for name, _, duration in PHASES}
PHASE_INDEX = {name: i for i, (name, _, _) in enumerate(PHASES)}
# Phases in which the swing leg carries no (or reduced) load: the continuous
# support-triangle e-stop applies (crawl REDUCED_SUPPORT_PHASES analogue).
REDUCED_SUPPORT_PHASES = frozenset(
    ("PRELIFT", "CLEAR_GATE", "LIFT", "SWING", "LOWER", "TOUCHDOWN_GATE",
     "LOAD", "ABORT_LOWER")
)
AIRBORNE_PHASES = frozenset(("LIFT", "SWING", "LOWER"))

support_triangle_margin = s3.support_triangle_margin
_ik_to_target = s3._ik_to_target


def signed_distance_to_stance_plane(point, stance_points):
    """Signed point/plane distance, normal toward body +z (crawl copy)."""
    point = np.asarray(point, dtype=float)
    stance = np.asarray(stance_points, dtype=float)
    if point.shape != (3,) or stance.shape != (3, 3):
        raise ValueError("stance-plane geometry must be one point and 3x3 feet")
    normal = np.cross(stance[1] - stance[0], stance[2] - stance[0])
    norm = float(np.linalg.norm(normal))
    if norm < 1.0e-6:
        raise ValueError("stance-foot plane is degenerate")
    normal /= norm
    if normal[2] < 0.0:
        normal *= -1.0
    return float(normal @ (point - stance[0]))


def _sl(leg):
    i = LEGS.index(leg)
    return slice(3 * i, 3 * i + 3)


class CrawlPlan:
    """Ground-anchor bookkeeping + per-phase trunk-frame foot targets.

    Crawl convention: every foot has a conceptual ground anchor; body motion
    moves every stance target the opposite way in the trunk frame, so
    stance-foot slip is never commanded.  Anchors are committed from the
    MEASURED encoder FK at each touchdown.
    """

    def __init__(self, step_length, shift, swing_height, prelift,
                 gait_cycles, walk_x=WALK_STANCE_X_M, walk_y=WALK_STANCE_Y_M):
        self.step_length = float(step_length)
        self.shift = float(shift)
        self.swing_height = float(swing_height)
        self.prelift = float(prelift)
        self.prelift_from = 0.0     # PRELIFT streams from here (retry support)
        self.swing_height_step = float(swing_height)  # adapted per step
        self.gait_cycles = int(gait_cycles)
        self.mode = "SWING_TEST" if abs(self.step_length) < 1e-12 else "WALK"
        self.anchor0 = {
            leg: dog5_kinematics.foot_position(leg, Q_STAND[_sl(leg)])
            for leg in LEGS
        }
        self.nominal_z = {leg: float(self.anchor0[leg][2]) for leg in LEGS}
        # Tucked walk rectangle, same per-leg height as the sprawl stand.
        sign_x = {"FL": +1, "FR": +1, "RL": -1, "RR": -1}
        sign_y = {"FL": +1, "FR": -1, "RL": +1, "RR": -1}
        self.walk_anchor_xy = {
            leg: np.array([sign_x[leg] * float(walk_x),
                           sign_y[leg] * float(walk_y)]) for leg in LEGS
        }
        # Step schedule: walk mode reposes each leg onto the rectangle first.
        self.repose_steps = len(GAIT_ORDER) if self.mode == "WALK" else 0
        self.total_steps = self.repose_steps + len(GAIT_ORDER) * self.gait_cycles
        self.reset()

    def reset(self):
        self.world_foot_xy = {
            leg: self.anchor0[leg][:2].copy() for leg in LEGS
        }
        self.body_xy = np.zeros(2)
        self.step_index = 0
        self.step_ctx = None

    # -- step planning ---------------------------------------------------
    def swing_of(self, step_index):
        if step_index < self.repose_steps:
            return REPOSE_ORDER[step_index]
        return GAIT_ORDER[(step_index - self.repose_steps) % len(GAIT_ORDER)]

    def begin_step(self):
        """Compute this step's shift/advance geometry from current anchors."""
        swing = self.swing_of(self.step_index)
        relative = self.world_foot_xy[swing] - self.body_xy
        signs = np.sign(relative)
        if np.any(signs == 0.0):
            raise RuntimeError(f"cannot choose diagonal shift for {swing}")
        shifted = self.body_xy - self.shift * signs / np.sqrt(2.0)
        if self.step_index < self.repose_steps:
            new_anchor = self.body_xy + self.walk_anchor_xy[swing]
            body_end = self.body_xy.copy()
        elif self.mode == "WALK":
            new_anchor = self.world_foot_xy[swing] + np.array(
                [self.step_length, 0.0])
            body_end = self.body_xy + np.array(
                [self.step_length / len(GAIT_ORDER), 0.0])
        else:
            new_anchor = self.world_foot_xy[swing].copy()
            body_end = self.body_xy.copy()
        self.prelift_from = 0.0
        self.swing_height_step = self.swing_height
        self.step_ctx = dict(
            swing=swing, body_start=self.body_xy.copy(), shifted=shifted,
            body_end=body_end, new_anchor=new_anchor,
            kind="REPOSE" if self.step_index < self.repose_steps else self.mode,
        )
        return self.step_ctx

    def stance_legs(self):
        return [leg for leg in LEGS if leg != self.step_ctx["swing"]]

    # -- targets ---------------------------------------------------------
    def _body_now(self, phase, s):
        ctx = self.step_ctx
        if phase == "SHIFT":
            return ctx["body_start"] + s * (ctx["shifted"] - ctx["body_start"])
        if phase in ("RECENTER", "RECENTER_GATE"):
            return ctx["shifted"] + s * (ctx["body_end"] - ctx["shifted"])
        if phase == "ABORT_RECENTER":
            return ctx["shifted"] + s * (ctx["body_start"] - ctx["shifted"])
        return ctx["shifted"]

    def _swing_target(self, phase, s, abort_from=None):
        ctx = self.step_ctx
        swing = ctx["swing"]
        z0 = self.nominal_z[swing]
        anchor = self.world_foot_xy[swing]
        if phase in ("SHIFT", "SHIFT_GATE"):
            xy, z = anchor, z0
        elif phase == "PRELIFT":
            xy = anchor
            z = z0 + self.prelift_from + s * (self.prelift - self.prelift_from)
        elif phase == "CLEAR_GATE":
            xy, z = anchor, z0 + self.prelift
        elif phase == "LIFT":
            xy = anchor
            z = z0 + self.prelift + s * (self.swing_height_step - self.prelift)
        elif phase == "SWING":
            xy = anchor + s * (ctx["new_anchor"] - anchor)
            z = z0 + self.swing_height_step
        elif phase == "LOWER":
            xy = ctx["new_anchor"]
            z = z0 + (1.0 - s) * self.swing_height_step
        elif phase in ("TOUCHDOWN_GATE", "LOAD"):
            xy, z = ctx["new_anchor"], z0
        elif phase == "ABORT_LOWER":
            start = np.asarray(abort_from, dtype=float)
            end = np.array([start[0], start[1], z0])
            return start + s * (end - start)
        else:   # RECENTER / RECENTER_GATE / ABORT_RECENTER: planted
            xy, z = anchor, z0
        return np.array([*xy, z])

    def foot_targets(self, phase, s, abort_from=None):
        """Trunk-frame foot targets for smoothed fraction ``s`` in [0, 1]."""
        body = self._body_now(phase, s)
        targets = {}
        for leg in LEGS:
            if leg == self.step_ctx["swing"]:
                target = self._swing_target(phase, s, abort_from=abort_from)
                if phase != "ABORT_LOWER":
                    target = target.copy()
                    target[:2] -= body
                targets[leg] = target
            else:
                targets[leg] = np.array([
                    *(self.world_foot_xy[leg] - body), self.nominal_z[leg]])
        return targets

    def q_targets(self, warm_q, phase, s, abort_from=None):
        targets = self.foot_targets(phase, s, abort_from=abort_from)
        q = np.asarray(warm_q, dtype=float).copy()
        for leg in LEGS:
            q[_sl(leg)] = _ik_to_target(leg, q[_sl(leg)], targets[leg])
        return q

    # -- encoder-FK measurements (control side) --------------------------
    def fk_feet(self, q_measured):
        return {leg: dog5_kinematics.foot_position(leg, q_measured[_sl(leg)])
                for leg in LEGS}

    def fk_margin(self, q_measured):
        feet = self.fk_feet(q_measured)
        return support_triangle_margin(
            [feet[leg][:2] for leg in self.stance_legs()])

    def fk_plane_distance(self, q_measured):
        feet = self.fk_feet(q_measured)
        stance = [feet[leg] for leg in self.stance_legs()]
        return signed_distance_to_stance_plane(
            feet[self.step_ctx["swing"]], stance)


class CrawlController:
    """Encoder-only stage machine streaming position commands.

    Stand-up is the hardware method (recorded CROUCH -> STAND native moves),
    then the planner's phase machine runs ``total_steps`` crawl steps with
    auto-advance replacing the operator's ENTER at each boundary.

    Sees: quantized q at ~20.8 Hz, EncoderVelocity, measured torque.
    Never sees: trunk pose, contacts, true CoM, MuJoCo internals.
    """

    def __init__(self, plan, torque_trip, unload_trip,
                 crouch_dps, stand_dps, stream_dps, start="crouch"):
        self.plan = plan
        self.torque_trip = float(torque_trip)
        self.unload_trip = float(unload_trip)
        self.caps = {"crouch": crouch_dps, "stand": stand_dps,
                     "stream": stream_dps}
        if start == "crouch":
            self.standup = ["CROUCH", "STAND"]
        elif start == "flat":
            self.standup = ["LIE", "ROLL", "CROUCH", "STAND"]
        else:
            raise ValueError(f"unknown start {start!r}")
        self.standup_targets = dict(s3.JOINT_STAGE_TARGETS)
        self.stage = self.standup[0]        # standup stages, HOLD4, CRAWL, DONE
        self.phase = None                    # planner phase while stage==CRAWL
        self.stage_started = 0.0
        self.settle_since = None
        self.dwell_since = None
        self.velocity = s3.EncoderVelocity()
        self.qd = np.zeros(N_JOINTS)
        self.q_cmd = s3.Q_ZERO.copy()
        self.stop_reason = None
        self.aborted = None
        self.abort_from = None
        self.plane_baseline = 0.0
        self.qd_streak = 0
        self.margin_streak = 0
        self.events = []
        self.step_events = []

    # -- helpers ---------------------------------------------------------
    def _cap_for_stage(self):
        if self.stage in ("LIE", "ROLL", "CROUCH"):
            return self.caps["crouch"]
        if self.stage == "STAND":
            return self.caps["stand"]
        return self.caps["stream"]

    def _phase_duration(self):
        if self.phase in ("ABORT_LOWER",):
            return T_LOWER
        if self.phase in ("ABORT_RECENTER",):
            return T_RECENTER
        return PHASE_DURATION[self.phase]

    def _enter_phase(self, phase, now, note=""):
        self.phase = phase
        self.stage_started = now
        self.settle_since = None
        self.dwell_since = None
        if note:
            self.events.append((now, phase, note))

    def _settled(self, now, q_enc):
        error = float(np.max(np.abs(q_enc - self.q_cmd)))
        speed = float(np.max(np.abs(self.qd))) if self.velocity.ready else np.inf
        if error > s3.POSITION_POSE_TOL or speed > s3.POSITION_QD_TOL:
            self.settle_since = None
            return False
        if self.settle_since is None:
            self.settle_since = now
            return False
        return now - self.settle_since >= s3.POSITION_SETTLE_S

    def _check_trips(self, q_enc, qd_reported, tau_measured):
        """Hardware SafetyGate trips that apply to position mode (s3 mirror)."""
        low, high = s3.soft_limits()
        if self.stage not in ("LIE", "ROLL", "CROUCH"):
            outside = (q_enc < low - s3.LIMIT_ESTOP_MARGIN) | \
                      (q_enc > high + s3.LIMIT_ESTOP_MARGIN)
            if np.any(outside):
                j = int(np.flatnonzero(outside)[0])
                return f"soft-limit e-stop: {JOINT_LABELS[j]}={q_enc[j]:+.2f} rad"
        if np.any(np.abs(qd_reported) > s3.QD_ESTOP_HARD):
            j = int(np.argmax(np.abs(qd_reported)))
            return (f"hard overspeed: {JOINT_LABELS[j]}="
                    f"{qd_reported[j]:+.1f} rad/s > {s3.QD_ESTOP_HARD}")
        if self.velocity.ready and np.any(np.abs(self.qd) > s3.QD_ESTOP):
            self.qd_streak += 1
            if self.qd_streak >= s3.QD_ESTOP_STREAK:
                j = int(np.argmax(np.abs(self.qd)))
                return (f"sustained overspeed: {JOINT_LABELS[j]}="
                        f"{self.qd[j]:+.1f} rad/s > {s3.QD_ESTOP}")
        else:
            self.qd_streak = 0
        if np.any(np.abs(tau_measured) > self.torque_trip):
            j = int(np.argmax(np.abs(tau_measured)))
            return (f"measured-torque trip: {JOINT_LABELS[j]}="
                    f"{tau_measured[j]:+.2f} N*m > {self.torque_trip:.2f}")
        return None

    def _begin_step(self, now):
        ctx = self.plan.begin_step()
        self._enter_phase(
            "SHIFT", now,
            f"step {self.plan.step_index + 1}/{self.plan.total_steps} "
            f"[{ctx['kind']}] swing={ctx['swing']}")

    def _abort_step(self, now, q_enc, reason):
        """Crawl-style graceful abort: lower where it is, recenter, end.

        Aborts only fire from gate phases, so targets are at well-defined
        endpoints and the body is at the shifted position.
        """
        self.aborted = reason
        self.events.append((now, "ABORT", reason))
        if self.phase == "CLEAR_GATE":
            # Foot is (at least commanded) airborne: freeze the trunk-frame
            # target where it was and stream it down; the measured anchor is
            # committed when the foot lands.
            start = self.plan._swing_target(self.phase, 1.0)
            start[:2] -= self.plan._body_now(self.phase, 1.0)
            self.abort_from = start
            self._enter_phase("ABORT_LOWER", now, "lowering swing foot")
        else:
            # SHIFT_GATE / TOUCHDOWN_GATE: foot is on (or commanded to) the
            # ground; commit where it measures and return the body.
            self._commit_anchor(q_enc)
            self._enter_phase("ABORT_RECENTER", now, "returning body to start")

    def _commit_anchor(self, q_enc):
        """Commit the MEASURED foot position as the swing anchor (crawl rule).

        Anchor commits always happen with the body at the shifted position
        (touchdown, or any abort), matching crawl_dog5_hw's
        ``feet[leg][:2] + self.shifted_body_xy``.
        """
        ctx = self.plan.step_ctx
        feet = self.plan.fk_feet(q_enc)
        self.plan.world_foot_xy[ctx["swing"]] = \
            feet[ctx["swing"]][:2] + ctx["shifted"]

    # -- the per-sweep update -------------------------------------------
    def sweep(self, now, q_enc, qd_reported, tau_measured):
        """One round-robin state update; returns 12 joint targets (rad)."""
        self.qd = self.velocity.update(q_enc, now)
        self.stop_reason = self._check_trips(q_enc, qd_reported, tau_measured)
        if self.stop_reason:
            return self.q_cmd

        # Continuous support-triangle e-stop during reduced support (crawl).
        if self.stage == "CRAWL" and self.phase in REDUCED_SUPPORT_PHASES:
            margin = self.plan.fk_margin(q_enc)
            if margin < SUPPORT_MARGIN_ESTOP_M:
                self.margin_streak += 1
                if self.margin_streak >= SUPPORT_MARGIN_ESTOP_STREAK:
                    self.stop_reason = (
                        f"support triangle lost during "
                        f"{self.plan.step_ctx['swing']} {self.phase}: "
                        f"margin={1000 * margin:.1f} mm")
                    return self.q_cmd
            else:
                self.margin_streak = 0
        else:
            self.margin_streak = 0

        if self.stage in self.standup_targets:
            self.q_cmd = self.standup_targets[self.stage].copy()
            if self._settled(now, q_enc):
                nxt = self.standup[self.standup.index(self.stage) + 1] \
                    if self.stage != "STAND" else "HOLD4"
                self.events.append((now, nxt, "settled"))
                self.stage = nxt
                self.stage_started = now
                self.settle_since = None
                self.dwell_since = None
            elif now - self.stage_started > s3.POSITION_TIMEOUT_S:
                self.stop_reason = f"{self.stage} timed out"
        elif self.stage == "HOLD4":
            if self.dwell_since is None:
                self.dwell_since = now
            if now - self.dwell_since >= 1.0:
                self.stage = "CRAWL"
                self.events.append((now, "CRAWL", f"mode={self.plan.mode}"))
                self._begin_step(now)
        elif self.stage == "CRAWL":
            self._crawl_sweep(now, q_enc, tau_measured)
        elif self.stage == "DONE":
            if self.dwell_since is None:
                self.dwell_since = now
            if now - self.dwell_since >= 1.5:
                self.stop_reason = "sequence complete"
        return self.q_cmd

    def _crawl_sweep(self, now, q_enc, tau_measured):
        phase = self.phase
        duration = self._phase_duration()
        kind = PHASE_KIND.get(phase, "stream")
        elapsed = now - self.stage_started

        if kind == "stream" or phase in ("ABORT_LOWER", "ABORT_RECENTER"):
            s = s3._smoothstep(elapsed / duration)
            self.q_cmd = self.plan.q_targets(
                self.q_cmd, phase, s, abort_from=self.abort_from)
            if elapsed < duration:
                return
            # stream complete -> next phase (or resolve the abort chain)
            if phase == "ABORT_LOWER":
                if self._settled(now, q_enc):
                    self._commit_anchor(q_enc)
                    self.abort_from = None
                    self._enter_phase("ABORT_RECENTER", now,
                                      "swing foot down; recentering")
                elif elapsed > duration + GATE_TIMEOUT_S:
                    self.stop_reason = "abort lower did not settle"
                return
            if phase == "ABORT_RECENTER":
                if self._settled(now, q_enc):
                    self._finish_batch(now, aborted=True)
                elif elapsed > duration + GATE_TIMEOUT_S:
                    self.stop_reason = "abort recenter did not settle"
                return
            nxt = PHASES[PHASE_INDEX[phase] + 1][0]
            self._enter_phase(nxt, now)
            return

        if kind == "dwell":            # LOAD
            self.q_cmd = self.plan.q_targets(self.q_cmd, phase, 1.0)
            if elapsed >= duration:
                self._enter_phase("RECENTER", now)
            return

        # gates: targets held at the phase endpoint
        self.q_cmd = self.plan.q_targets(self.q_cmd, phase, 1.0)
        ctx = self.plan.step_ctx
        swing = ctx["swing"]

        if phase == "SHIFT_GATE":
            if self._settled(now, q_enc):
                margin = self.plan.fk_margin(q_enc)
                if margin >= MIN_LIFTOFF_MARGIN_M:
                    # Plane baseline with the swing foot still planted: the
                    # clear/touchdown gates measure rise relative to this.
                    self.plane_baseline = self.plan.fk_plane_distance(q_enc)
                    self._enter_phase(
                        "PRELIFT", now,
                        f"{swing} FK margin {1000 * margin:.1f} mm, plane "
                        f"baseline {1000 * self.plane_baseline:+.1f} mm")
                    return
                if elapsed > duration:
                    self._abort_step(now, q_enc,
                                     f"{swing} liftoff margin "
                                     f"{1000 * margin:.1f} mm < "
                                     f"{1000 * MIN_LIFTOFF_MARGIN_M:.0f} mm")
            elif elapsed > duration:
                self._abort_step(now, q_enc, "SHIFT did not settle")
        elif phase == "CLEAR_GATE":
            rise = self.plan.fk_plane_distance(q_enc) - self.plane_baseline
            swing_tau = np.abs(tau_measured[_sl(swing)][1:])   # pitch, knee
            clear = rise >= CLEAR_PLANE_RISE_MIN_M
            unloaded = bool(np.all(swing_tau <= self.unload_trip))
            if self._settled(now, q_enc) and clear and unloaded:
                # Swing at (measured corner sag + required clearance): a
                # grippy floor or soft servo that tilts the trunk more
                # automatically swings higher, inside the validated envelope.
                sag = self.plan.prelift - rise
                self.plan.swing_height_step = float(np.clip(
                    sag + SWING_CLEARANCE_MIN_M,
                    max(self.plan.swing_height, self.plan.prelift + 0.003),
                    PRELIFT_MAX_M + 0.005))
                self._enter_phase(
                    "LIFT", now,
                    f"{swing} clear: plane rise {1000 * rise:.1f} mm, "
                    f"max|tau| {float(np.max(swing_tau)):.2f} N*m; swing "
                    f"height {1000 * self.plan.swing_height_step:.0f} mm")
            elif elapsed > duration:
                if self.plan.prelift < PRELIFT_MAX_M - 1e-9:
                    # The measured rise says the corner sagged more than the
                    # pre-lift; raise it and re-prove instead of giving up.
                    self.plan.prelift_from = self.plan.prelift
                    self.plan.prelift = min(
                        self.plan.prelift + PRELIFT_RETRY_STEP_M,
                        PRELIFT_MAX_M)
                    self._enter_phase(
                        "PRELIFT", now,
                        f"{swing} clear retry: rise {1000 * rise:.1f} mm "
                        f"short, pre-lift -> "
                        f"{1000 * self.plan.prelift:.0f} mm")
                    return
                self._abort_step(
                    now, q_enc,
                    f"{swing} clear gate: plane rise {1000 * rise:.1f} mm "
                    f"(need {1000 * CLEAR_PLANE_RISE_MIN_M:.1f}),"
                    f" max|tau| {float(np.max(swing_tau)):.2f} N*m "
                    f"(trip {self.unload_trip:.2f}) -- foot did not lift")
        elif phase == "TOUCHDOWN_GATE":
            plane = self.plan.fk_plane_distance(q_enc) - self.plane_baseline
            if self._settled(now, q_enc) and abs(plane) <= TOUCHDOWN_PLANE_TOL_M:
                self._commit_anchor(q_enc)
                planned = ctx["new_anchor"]
                shortfall = float(np.linalg.norm(
                    self.plan.world_foot_xy[swing] - planned))
                self._enter_phase(
                    "LOAD", now,
                    f"{swing} touchdown: plane {1000 * plane:+.1f} mm, "
                    f"anchored {1000 * shortfall:.1f} mm from planned")
            elif elapsed > duration:
                self._abort_step(now, q_enc,
                                 f"{swing} touchdown: plane "
                                 f"{1000 * plane:+.1f} mm not within "
                                 f"{1000 * TOUCHDOWN_PLANE_TOL_M:.0f} mm")
        elif phase == "RECENTER_GATE":
            if self._settled(now, q_enc):
                self._end_step(now)
            elif elapsed > duration:
                # Crawl behavior: all four feet are planted and the target is
                # at its endpoint -- adopt it and end the batch aborted.
                self.aborted = f"{swing} recenter did not settle"
                self.plan.body_xy = ctx["body_end"].copy()
                self._finish_batch(now, aborted=True)

    def _end_step(self, now):
        ctx = self.plan.step_ctx
        self.plan.body_xy = ctx["body_end"].copy()
        self.plan.step_index += 1
        self.step_events.append((now, ctx["swing"], ctx["kind"], "ok"))
        if self.plan.step_index >= self.plan.total_steps:
            self._finish_batch(now, aborted=False)
        else:
            self._begin_step(now)

    def _finish_batch(self, now, aborted):
        if aborted:
            ctx = self.plan.step_ctx
            self.plan.body_xy = ctx["body_start"].copy() \
                if self.phase == "ABORT_RECENTER" else self.plan.body_xy
            self.step_events.append((now, ctx["swing"], ctx["kind"], "ABORT"))
        self.stage = "DONE"
        self.phase = None
        self.dwell_since = None
        self.events.append(
            (now, "DONE", "batch ABORTED" if aborted else
             f"{self.plan.step_index}/{self.plan.total_steps} steps complete"))


class CrawlOracle:
    """Privileged sim measurements — judge the run, never steer it."""

    def __init__(self, model, data, plan):
        import mujoco
        self._mujoco = mujoco
        self.m, self.d = model, data
        self.plan = plan
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.foot_geom = {leg: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}") for leg in LEGS}
        self.site = {leg: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"foot_{leg}") for leg in LEGS}
        self.max_abs_tau = np.zeros(N_JOINTS)
        self.max_tilt_deg = 0.0
        self.min_true_margin = np.inf
        self.min_fk_margin = np.inf
        self.step_stats = []            # one dict per crawl step
        self._step_key = None
        self.start_x = None
        self.final_x = None

    def foot_force(self, leg):
        geom = self.foot_geom[leg]
        total = 0.0
        force = np.zeros(6)
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            if geom in (con.geom1, con.geom2):
                self._mujoco.mj_contactForce(self.m, self.d, i, force)
                total += abs(force[0])
        return total

    def tilt_deg(self):
        R = self.d.xmat[self.trunk].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))

    def true_margin(self, stance_legs):
        com = self.d.subtree_com[self.trunk][:2]
        feet = [self.d.site_xpos[self.site[leg]][:2] for leg in stance_legs]
        try:
            return support_triangle_margin(feet, com_xy=com)
        except ValueError:
            return float("nan")

    def _current(self, controller):
        key = (self.plan.step_index, controller.phase)
        if self._step_key is None or self._step_key[0] != self.plan.step_index:
            ctx = self.plan.step_ctx
            self.step_stats.append(dict(
                swing=ctx["swing"], kind=ctx["kind"],
                swing_force_max=0.0, clearance_min=np.inf,
                true_margin_min=np.inf, fk_margin_min=np.inf,
                stance_slip_max=0.0, aborted=False,
                stance0={leg: self.d.site_xpos[self.site[leg]][:2].copy()
                         for leg in LEGS if leg != ctx["swing"]}))
        self._step_key = key
        return self.step_stats[-1]

    def sample(self, controller, q_enc, tau):
        self.max_abs_tau = np.maximum(self.max_abs_tau, np.abs(tau))
        if controller.stage == "HOLD4" and self.start_x is None:
            self.start_x = float(self.d.qpos[0])
        if controller.stage != "CRAWL" or self.plan.step_ctx is None:
            return
        stats = self._current(controller)
        self.max_tilt_deg = max(self.max_tilt_deg, self.tilt_deg())
        phase = controller.phase
        stance = self.plan.stance_legs()
        if phase in REDUCED_SUPPORT_PHASES or phase in (
                "SHIFT_GATE", "TOUCHDOWN_GATE"):
            margin = self.true_margin(stance)
            fk = self.plan.fk_margin(q_enc)
            stats["true_margin_min"] = min(stats["true_margin_min"], margin)
            stats["fk_margin_min"] = min(stats["fk_margin_min"], fk)
            self.min_true_margin = min(self.min_true_margin, margin)
            self.min_fk_margin = min(self.min_fk_margin, fk)
        if phase == "SWING":
            swing = self.plan.step_ctx["swing"]
            stats["swing_force_max"] = max(
                stats["swing_force_max"], self.foot_force(swing))
            clearance = float(self.d.site_xpos[self.site[swing]][2]) - 0.02
            stats["clearance_min"] = min(stats["clearance_min"], clearance)
        for leg, xy0 in stats["stance0"].items():
            slip = float(np.linalg.norm(
                self.d.site_xpos[self.site[leg]][:2] - xy0))
            stats["stance_slip_max"] = max(stats["stance_slip_max"], slip)
        if controller.aborted and not stats["aborted"]:
            stats["aborted"] = True

    def finish(self):
        self.final_x = float(self.d.qpos[0])

    def verdict(self, controller):
        plan = self.plan
        swung = [s for s in self.step_stats if not s["aborted"]]
        checks = [
            ("finished (no abort/e-stop)",
             controller.aborted is None
             and controller.stop_reason == "sequence complete"),
            (f"all {plan.total_steps} steps completed",
             plan.step_index >= plan.total_steps),
            ("every swing airborne (force <= 0.05 N)",
             len(swung) > 0 and all(
                 s["swing_force_max"] <= 0.05 for s in swung)),
            ("swing clearance >= 8 mm",
             len(swung) > 0 and all(
                 s["clearance_min"] >= 0.008 for s in swung)),
            ("trunk tilt <= 5 deg", self.max_tilt_deg <= 5.0),
            ("true CoM margin >= 10 mm during 3-leg support",
             self.min_true_margin >= 0.010),
            ("encoder FK margin >= 15 mm at gates",
             self.min_fk_margin >= MIN_LIFTOFF_MARGIN_M),
        ]
        return all(ok for _, ok in checks), checks


def run(model, data, args, viewer=None, frame_hook=None, quiet=False):
    """Drive one full stand-up + crawl batch; returns (controller, oracle)."""
    import mujoco
    plan = CrawlPlan(
        0.0 if args.swing_test else args.step_length,
        args.shift, args.swing_height, args.prelift, args.gait_cycles,
        walk_x=args.walk_x, walk_y=args.walk_y)
    controller = CrawlController(
        plan, args.torque_trip, args.unload_trip,
        args.crouch_dps, args.stand_dps, args.stream_dps,
        start=getattr(args, "start", "crouch"))
    oracle = CrawlOracle(model, data, plan)

    qadr = np.array([model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{j}_{leg}")]
        for leg in LEGS for j in ("hip_abd", "hip_pitch", "knee")])
    assert qadr[0] == 7

    zero_error = getattr(args, "zero_error", None)
    zero_error = np.zeros(N_JOINTS) if zero_error is None \
        else np.asarray(zero_error, dtype=float)
    servo = s3.NativePositionServo(
        data.qpos[qadr] + zero_error, kp=s3.KP_SERVO * args.kp_scale,
        zero_error=zero_error)

    steps_per_tick = max(1, round(1.0 / (s3.CONTROL_HZ * model.opt.timestep)))
    slot = 0
    q_targets = s3.Q_ZERO.copy()
    last_print = -np.inf
    t_wall0 = time.perf_counter()

    while controller.stop_reason is None:
        now = data.time
        if slot % s3.SWEEPS_PER_STATE == 0:
            q_enc, qd_reported, tau_measured = servo.telemetry()
            q_targets = controller.sweep(now, q_enc, qd_reported, tau_measured)
            oracle.sample(controller, q_enc, tau_measured)
            if not quiet and now - last_print >= 1.0 / s3.STATUS_HZ:
                phase = controller.phase or "-"
                swing = plan.step_ctx["swing"] if plan.step_ctx else "-"
                step = min(plan.step_index + 1, plan.total_steps) \
                    if controller.stage == "CRAWL" else 0
                print(f"t={now:6.2f}  {controller.stage:<6} "
                      f"{phase:<15} step={step}/{plan.total_steps} "
                      f"swing={swing:2s} tilt={oracle.tilt_deg():4.1f} deg  "
                      f"max|tau|={float(np.max(np.abs(tau_measured))):4.2f}",
                      flush=True)
                last_print = now
        motor = slot % N_JOINTS
        servo.command(motor, q_targets[motor], controller._cap_for_stage())
        for _ in range(steps_per_tick):
            data.ctrl[:] = servo.step(data.qpos[qadr], model.opt.timestep)
            mujoco.mj_step(model, data)
        if frame_hook is not None:
            frame_hook(now, controller, oracle)
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
    oracle.finish()
    return controller, oracle


def report(controller, oracle, quiet=False):
    passed, checks = oracle.verdict(controller)
    if quiet:
        return passed
    print(f"\nstop: {controller.stop_reason}"
          + (f"  (ABORTED: {controller.aborted})" if controller.aborted else ""))
    for now, stage, note in controller.events:
        print(f"  t={now:6.2f}  -> {stage:<15} {note}")
    print(f"\nper-step (swing force N / clearance mm / true margin mm / "
          f"FK margin mm / stance slip mm):")
    for s in oracle.step_stats:
        clr = 1000 * s["clearance_min"] if np.isfinite(s["clearance_min"]) else float("nan")
        print(f"  {s['kind']:<10} {s['swing']}  "
              f"F={s['swing_force_max']:5.2f}  clear={clr:5.1f}  "
              f"true={1000 * s['true_margin_min']:5.1f}  "
              f"fk={1000 * s['fk_margin_min']:5.1f}  "
              f"slip={1000 * s['stance_slip_max']:4.1f}"
              + ("  ABORTED" if s["aborted"] else ""))
    progress = (oracle.final_x - oracle.start_x) \
        if oracle.start_x is not None else float("nan")
    worst = int(np.argmax(oracle.max_abs_tau))
    print(f"\ntrunk forward progress : {1000 * progress:+.1f} mm "
          f"(planned {1000 * oracle.plan.step_length * oracle.plan.gait_cycles:+.0f})")
    print(f"trunk tilt max         : {oracle.max_tilt_deg:.2f} deg")
    print(f"true CoM margin min    : {1000 * oracle.min_true_margin:.1f} mm")
    print(f"encoder FK margin min  : {1000 * oracle.min_fk_margin:.1f} mm")
    print(f"peak measured torque   : {JOINT_LABELS[worst]} "
          f"{oracle.max_abs_tau[worst]:.2f} N*m "
          f"(hw --position-torque-trip must exceed this)")
    print("\nverdict:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\n{'PASS' if passed else 'FAIL'}: crawl "
          f"({'proven' if passed else 'not proven'} under hw constraints)")
    return passed


# ---------------------------------------------------------------------------
def _validate_gait(plan):
    """Walk the whole planned gait; raise SystemExit on any violation."""
    low, high = s3.soft_limits()
    q = Q_STAND.copy()
    worst_err = 0.0
    worst_sigma = np.inf
    worst_margin = np.inf
    worst_tau = 0.0
    weight = 5.815 * 9.81

    stream_phases = [name for name, kind, _ in PHASES if kind != "gate"]
    while plan.step_index < plan.total_steps:
        ctx = plan.begin_step()
        margin = support_triangle_margin(
            [plan.world_foot_xy[leg] - ctx["shifted"]
             for leg in LEGS if leg != ctx["swing"]])
        worst_margin = min(worst_margin, margin)
        if margin < MIN_LIFTOFF_MARGIN_M:
            raise SystemExit(
                f"FAIL: step {plan.step_index + 1} ({ctx['swing']}) planned "
                f"margin {1000 * margin:.1f} mm < "
                f"{1000 * MIN_LIFTOFF_MARGIN_M:.0f} mm")
        for phase in stream_phases:
            for s in np.linspace(0.0, 1.0, 8):
                targets = plan.foot_targets(phase, float(s))
                for leg in LEGS:
                    sl = _sl(leg)
                    q[sl] = _ik_to_target(leg, q[sl], targets[leg])
                    err = float(np.linalg.norm(
                        dog5_kinematics.foot_position(leg, q[sl]) - targets[leg]))
                    worst_err = max(worst_err, err)
                    if err > 1.0e-4:
                        raise SystemExit(
                            f"FAIL: step {plan.step_index + 1} "
                            f"({ctx['swing']}) {phase} s={s:.2f} {leg} IK "
                            f"residual {err:.4f} m (unreachable)")
                    if np.any(q[sl] < low[sl]) or np.any(q[sl] > high[sl]):
                        raise SystemExit(
                            f"FAIL: {leg} outside soft limits in {phase}: "
                            f"{np.rad2deg(q[sl])} deg")
                    sigma = float(np.min(np.linalg.svd(
                        dog5_kinematics.foot_jacobian(leg, q[sl]),
                        compute_uv=False)))
                    worst_sigma = min(worst_sigma, sigma)
                    if sigma < MIN_JACOBIAN_SINGULAR:
                        raise SystemExit(
                            f"FAIL: {leg} near singular in {phase}: "
                            f"sigma_min={sigma:.4f}")
                    if leg != ctx["swing"]:
                        tau = dog5_kinematics.foot_jacobian(leg, q[sl]).T @ \
                            [0.0, 0.0, weight / 3.0]
                        worst_tau = max(worst_tau, float(np.max(np.abs(tau))))
        # commit the planned endpoint (ideal tracking)
        plan.world_foot_xy[ctx["swing"]] = ctx["new_anchor"].copy()
        plan.body_xy = ctx["body_end"].copy()
        plan.step_index += 1

    if worst_tau > 0.8 * s3.TAU_HARD:
        raise SystemExit(f"FAIL: static mg/3 torque {worst_tau:.1f} N*m too "
                         f"close to driver capability {s3.TAU_HARD} N*m")
    return worst_err, worst_sigma, worst_margin, worst_tau


def self_test(args):
    """Offline validation of the whole planned gait (no MuJoCo)."""
    step = 0.0 if args.swing_test else args.step_length
    plan = CrawlPlan(step, args.shift, args.swing_height, args.prelift,
                     args.gait_cycles, walk_x=args.walk_x, walk_y=args.walk_y)
    worst_err, worst_sigma, worst_margin, worst_tau = _validate_gait(plan)

    # The clear-gate retry may raise the pre-lift up to PRELIFT_MAX_M at any
    # step; the whole gait must stay reachable/in-limits at that envelope.
    plan_max = CrawlPlan(
        step, args.shift,
        max(args.swing_height, PRELIFT_MAX_M + 0.005), PRELIFT_MAX_M,
        args.gait_cycles, walk_x=args.walk_x, walk_y=args.walk_y)
    _validate_gait(plan_max)

    # plane-distance invariance + degenerate rejection (crawl self-test copy)
    plane = np.array([[-0.3, -0.1, 0.0], [0.3, -0.1, 0.02], [-0.3, 0.1, 0.01]])
    normal = np.cross(plane[1] - plane[0], plane[2] - plane[0])
    normal /= np.linalg.norm(normal)
    point = np.mean(plane, axis=0) + 0.003 * normal
    distance = signed_distance_to_stance_plane(point, plane)
    angle = np.deg2rad(20.0)
    rot = np.array([[1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)]])
    shift = np.array([0.2, -0.3, 0.1])
    assert np.isclose(signed_distance_to_stance_plane(
        rot @ point + shift, plane @ rot.T + shift), distance)
    try:
        signed_distance_to_stance_plane(np.zeros(3), np.zeros((3, 3)))
    except ValueError as exc:
        assert "degenerate" in str(exc)
    else:
        raise AssertionError("degenerate stance plane was accepted")

    # anchors: stance targets are constant in the ground frame during body
    # motion; only the swing target moves during SWING
    plan2 = CrawlPlan(args.step_length or 0.03, args.shift,
                      args.swing_height, args.prelift, 1)
    ctx = plan2.begin_step()
    for phase in ("SHIFT", "RECENTER"):
        for leg in LEGS:
            if leg == ctx["swing"]:
                continue
            g0 = plan2.foot_targets(phase, 0.0)[leg][:2] + \
                plan2._body_now(phase, 0.0)
            g1 = plan2.foot_targets(phase, 1.0)[leg][:2] + \
                plan2._body_now(phase, 1.0)
            assert np.allclose(g0, g1), (phase, leg)
    d0 = plan2.foot_targets("SWING", 0.0)[ctx["swing"]][:2]
    d1 = plan2.foot_targets("SWING", 1.0)[ctx["swing"]][:2]
    assert np.allclose(
        d1 - d0, ctx["new_anchor"] - plan2.world_foot_xy[ctx["swing"]])

    mode = plan.mode + (" (+REPOSE)" if plan.repose_steps else "")
    print("crawl_dog5_sim offline self-test PASS")
    print(f"  mode={mode}, order={','.join(GAIT_ORDER)}, "
          f"steps={plan.total_steps}, step={1000 * plan.step_length:.0f} mm, "
          f"shift={1000 * plan.shift:.0f} mm, "
          f"prelift={1000 * plan.prelift:.0f} mm, "
          f"swing height={1000 * plan.swing_height:.0f} mm")
    print(f"  whole gait IK-reachable, in-limits; worst residual "
          f"{worst_err:.2e} m, sigma_min {worst_sigma:.4f} m/rad")
    print(f"  minimum planned shifted margin {1000 * worst_margin:.1f} mm "
          f"(gate {1000 * MIN_LIFTOFF_MARGIN_M:.0f} mm)")
    print(f"  worst static mg/3 stance torque ~{worst_tau:.2f} N*m "
          f"(driver capability {s3.TAU_HARD} N*m)")
    print(f"  clear-gate retry envelope validated to pre-lift "
          f"{1000 * PRELIFT_MAX_M:.0f} mm")
    return 0


SWEEP_CASES = [
    dict(label="nominal"),
    dict(label="mu=0.5", friction=0.5),
    dict(label="mu=0.8", friction=0.8),
    dict(label="mu=2.0 rubber", friction=2.0),
    dict(label="mass+10%", mass_scale=1.10),
    dict(label="mass-10%", mass_scale=0.90),
    dict(label="com+10mm y", com_offset_y=+0.010),
    dict(label="com-10mm y", com_offset_y=-0.010),
    dict(label="servo kp+50%", kp_scale=1.5),
    dict(label="servo kp-50%", kp_scale=0.5),
    dict(label="zero error +/-2deg", zero_deg=2.0),
    dict(label="mu=0.5 mass+10% com-10mm", friction=0.5, mass_scale=1.10,
         com_offset_y=-0.010),
]


def _zero_error_vec(zero_deg):
    """Deterministic +/- alternating mis-zero on all 12 joints (exp C)."""
    signs = np.array([+1, -1] * (N_JOINTS // 2), dtype=float)
    return np.deg2rad(float(zero_deg)) * signs


def sweep(args):
    results = []
    for case in SWEEP_CASES:
        case = dict(case)
        label = case.pop("label")
        kp_scale = case.pop("kp_scale", 1.0)
        zero_deg = case.pop("zero_deg", 0.0)
        model, data = s3.build_sim(**case, kp_scale=kp_scale,
                                   start=getattr(args, "start", "crouch"))
        case_args = argparse.Namespace(**vars(args))
        case_args.kp_scale = kp_scale
        case_args.zero_error = _zero_error_vec(zero_deg) if zero_deg else None
        controller, oracle = run(model, data, case_args, quiet=True)
        passed = report(controller, oracle, quiet=True)
        aborts = sum(1 for s in oracle.step_stats if s["aborted"])
        if oracle.step_stats:
            detail = (f"steps {oracle.plan.step_index}/"
                      f"{oracle.plan.total_steps}, tilt "
                      f"{oracle.max_tilt_deg:.1f} deg, true margin "
                      f"{1000 * oracle.min_true_margin:.0f} mm"
                      + (f", {aborts} ABORT" if aborts else ""))
        else:
            detail = f"stopped: {controller.aborted or controller.stop_reason}"
        results.append((label, passed, detail))
        print(f"[{'PASS' if passed else 'FAIL'}] {label:<28} {detail}",
              flush=True)
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\nsweep: {n_pass}/{len(results)} PASS")
    return 0 if n_pass == len(results) else 1


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--start", default="crouch", choices=("crouch", "flat"))
    parser.add_argument("--swing-test", action="store_true",
                        help="crawl in place on the sprawl stand (step=0), "
                             "the first hardware test")
    parser.add_argument("--step-length", type=float, default=DEFAULT_STEP_M)
    parser.add_argument("--shift", type=float, default=DEFAULT_SHIFT_M)
    parser.add_argument("--prelift", type=float, default=DEFAULT_PRELIFT_M)
    parser.add_argument("--swing-height", type=float,
                        default=DEFAULT_SWING_HEIGHT_M)
    parser.add_argument("--gait-cycles", type=int, default=DEFAULT_GAIT_CYCLES)
    parser.add_argument("--walk-x", type=float, default=WALK_STANCE_X_M)
    parser.add_argument("--walk-y", type=float, default=WALK_STANCE_Y_M)
    parser.add_argument("--torque-trip", type=float,
                        default=DEFAULT_TORQUE_TRIP_NM)
    parser.add_argument("--unload-trip", type=float,
                        default=DEFAULT_UNLOAD_TAU_TRIP_NM)
    parser.add_argument("--crouch-dps", type=float, default=s3.DEFAULT_CROUCH_DPS)
    parser.add_argument("--stand-dps", type=float, default=s3.DEFAULT_STAND_DPS)
    parser.add_argument("--stream-dps", type=float, default=s3.DEFAULT_STREAM_DPS)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--friction", type=float, default=None,
                        help="override floor+foot friction coefficient")
    parser.add_argument("--mass-scale", type=float, default=1.0)
    parser.add_argument("--com-offset-y", type=float, default=0.0)
    parser.add_argument("--zero-error-deg", type=float, default=0.0,
                        help="calibration mis-zero on all 12 joints "
                             "(alternating sign), degrees")
    return parser


def main():
    args = build_parser().parse_args()
    if args.swing_height < args.prelift:
        raise SystemExit("--swing-height must be >= --prelift")
    args.zero_error = _zero_error_vec(args.zero_error_deg) \
        if args.zero_error_deg else None

    if args.self_test:
        return self_test(args)
    if args.sweep:
        return sweep(args)

    model, data = s3.build_sim(kp_scale=args.kp_scale,
                               friction=args.friction,
                               mass_scale=args.mass_scale,
                               com_offset_y=args.com_offset_y,
                               start=getattr(args, "start", "crouch"))
    if args.headless:
        controller, oracle = run(model, data, args)
        return 0 if report(controller, oracle) else 1

    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        controller, oracle = run(model, data, args, viewer=viewer)
    return 0 if report(controller, oracle) else 1


if __name__ == "__main__":
    raise SystemExit(main())
