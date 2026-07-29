#!/usr/bin/env python3
"""One slow gait cycle on DOG5 hardware -- position-command track.

Builds directly on the hardware-proven ``stand3_hold_hw.py`` (all four legs
lifted and held on 2026-07-23): every step reuses the same shift ->
pre-lift -> clear-gate ladder, then adds the walk pieces from
``crawl_dog5_sim.py`` (SIM_APPROACH_HW.md section 8): a horizontal SWING,
a TOUCHDOWN gate that commits the MEASURED anchor, and a RECENTER to a
forward-advanced neutral.  One cycle = 4 steps in the crawl's proven order
RR -> FL -> RL -> FR (hind, then diagonal fore); each step advances its
foot +30 mm and the body +7.5 mm, so a full cycle walks +30 mm.

Discipline unchanged: native 0xA4 position commands only, encoder-only
state at ~20.8 Hz, NumPy kinematics, IK spread one-leg-per-3rd-CAN-slot
(10 ms input-watchdog budget), IMU judges but never steers (tip-over
run-stop, lean watch during LIFT/SWING).

Flow (ENTER at every boundary, X stops, P parks from HOLD4):

    READ -> CROUCH -> WAIT_CROUCH -ENTER-> STAND (vertical rise) -> HOLD4
    -ENTER-> step k: SHIFT -> SHIFT_SETTLE (margin >= 15 mm)
        -ENTER-> UNLOAD -> CLEAR_GATE (rise >= 4 mm ABS + |tau| <= 0.7;
                            ladder 20 -> 30 mm)
        -ENTER-> LIFT -> SWING (+30 mm) -> LOWER
              -> TOUCHDOWN (plane +-8 mm; MEASURED anchor committed)
              -> LOAD -> RECENTER (body +7.5 mm fwd) -> HOLD4
    ... 4 steps -> cycle complete; ENTER starts another cycle, P parks.

A failed gate aborts gracefully (lower if airborne, reload, recenter,
HOLD4 with the reason) -- the run does not stop.

First checks:
    python walk1_hw.py --self-test
Hardware (robot supported until proven):
    python walk1_hw.py
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Fast GIL handoff so the EKF worker thread cannot stall the 250 Hz CAN loop
# (validated in vmc/ekf_stand_hw.py against the 10 ms motor watchdog).
sys.setswitchinterval(0.0005)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dog5_description"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "state_estimator"))

import dog5_kinematics                      # noqa: E402
import stand3_hold_hw as s3                 # noqa: E402
import stand_dog5_hw as base                # noqa: E402
import stand_by_position_command as posbase  # noqa: E402
import stand_dog5_inplace_hw as inplace     # noqa: E402
from crawl_dog5_hw import (                 # noqa: E402
    signed_distance_to_stance_plane,
    support_triangle_margin,
)
from ekf_runtime import EkfShared, ekf_worker, _rpy  # noqa: E402
from imu_ekf_feed import ImuEkfFeed         # noqa: E402
from imu_dog import DEFAULT_PORT as IMU_DEFAULT_PORT  # noqa: E402
import ekf_web                              # noqa: E402

LEGS = s3.LEGS
MOTOR_IDS = s3.MOTOR_IDS
MOTOR_DIRECTIONS = s3.MOTOR_DIRECTIONS
JOINT_LABELS = s3.JOINT_LABELS
N_JOINTS = s3.N_JOINTS
Q_CROUCH = s3.Q_CROUCH

# ---- gait tuning (offline-validated full cycle: worst margin 20.6 mm) ----
GAIT_ORDER = ("RR", "FL", "RL", "FR")
DEFAULT_STEP_M = 0.040
DEFAULT_SHIFT_M = 0.032         # larger shifts are RR-abd infeasible
# Front-leg steps: the diagonal shift's backward component subtracts
# directly from the leg's forward reach, capping the step at 30 mm.  The
# front shift therefore keeps the proven ~22.6 mm lateral component but
# only 15 mm backward -> 40 mm steps clear (45 mm is a hard reach wall).
FRONT_BACK_SHIFT_M = 0.015
DEFAULT_LIFT_M = s3.DEFAULT_LIFT_M              # 14 mm above the pre-lift
DEFAULT_STAND_HEIGHT_M = s3.DEFAULT_STAND_HEIGHT_M   # 0.175
TOUCHDOWN_TOL_M = 0.008         # crawl's plane tolerance
T_SWING = 2.0
T_LOWER = 1.5
T_LOAD = 1.0

MIN_STEP_MARGIN_M = s3.MIN_LIFTOFF_MARGIN_M     # 15 mm gate per shift


class WalkPlan:
    """Ground-frame anchor bookkeeping for one gait cycle.

    World frame: body starts at the origin; ``anchors`` are each foot's
    ground xy (crouch feet), fixed except when a TOUCHDOWN commits the
    measured landing point.  Trunk-frame targets are anchor - body_xy at
    z = -H, plus the swing leg's vertical offsets.
    """

    def __init__(self, step_m=DEFAULT_STEP_M, shift_m=DEFAULT_SHIFT_M,
                 lift_m=DEFAULT_LIFT_M, stand_height=DEFAULT_STAND_HEIGHT_M,
                 front_back_shift=FRONT_BACK_SHIFT_M):
        self.step_m = float(step_m)
        self.shift_m = float(shift_m)
        self.front_back_shift = float(front_back_shift)
        self.lift_m = float(lift_m)
        self.stand_height = float(stand_height)
        self.crouch_foot = {
            leg: dog5_kinematics.foot_position(
                leg, Q_CROUCH[self._sl(leg)])
            for leg in LEGS
        }
        self.anchors = {leg: self.crouch_foot[leg][:2].copy() for leg in LEGS}
        self.neutral = np.zeros(2)
        self.swing = None

    @staticmethod
    def _sl(leg):
        i = LEGS.index(leg)
        return slice(3 * i, 3 * i + 3)

    def shift_vec_for(self, swing):
        rel = self.anchors[swing] - self.neutral
        if rel[0] > 0:   # front leg: protect forward reach
            return np.array([
                -self.front_back_shift * np.sign(rel[0]),
                -self.shift_m / np.sqrt(2.0) * np.sign(rel[1]),
            ])
        return -self.shift_m * np.sign(rel) / np.sqrt(2.0)

    def planned_shift_margin(self, swing):
        body = self.neutral + self.shift_vec_for(swing)
        stance = [leg for leg in LEGS if leg != swing]
        return support_triangle_margin(
            [self.anchors[leg] - body for leg in stance]
        )

    def targets(self, body_xy, height_frac, swing, swing_anchor_xy,
                pre_m, lift_frac):
        """Trunk-frame foot targets.  ``swing_anchor_xy`` overrides the
        stored anchor for the swing leg (moves during SWING)."""
        out = {}
        for leg in LEGS:
            axy = swing_anchor_xy if leg == swing else self.anchors[leg]
            crouch_z = self.crouch_foot[leg][2]
            z = crouch_z + height_frac * (-self.stand_height - crouch_z)
            if leg == swing:
                z += pre_m + lift_frac * self.lift_m
            out[leg] = np.array([axy[0] - body_xy[0],
                                 axy[1] - body_xy[1], z])
        return out

    def fk_margin(self, q, body_is_shifted_stance):
        feet = [
            dog5_kinematics.foot_position(leg, q[self._sl(leg)])[:2]
            for leg in LEGS if leg != self.swing
        ]
        return support_triangle_margin(feet)

    def plane_distance(self, q, swing):
        feet = {
            leg: dog5_kinematics.foot_position(leg, q[self._sl(leg)])
            for leg in LEGS
        }
        return signed_distance_to_stance_plane(
            feet[swing], [feet[leg] for leg in LEGS if leg != swing]
        )


STAGES = [
    ("CROUCH", "move"), ("WAIT_CROUCH", "wait"),
    ("STAND", "stream"), ("STAND_SETTLE", "gate"), ("HOLD4", "wait"),
    ("SHIFT", "stream"), ("SHIFT_SETTLE", "gate"), ("WAIT_UNLOAD", "wait"),
    ("UNLOAD", "stream"), ("CLEAR_GATE", "gate"), ("WAIT_SWING", "wait"),
    ("LIFT", "stream"), ("SWING", "stream"), ("LOWER", "stream"),
    ("TOUCHDOWN", "gate"), ("LOAD", "dwell"), ("RECENTER", "stream"),
    ("PARK", "move"), ("PARKED", "wait"),
]
STAGE_INDEX = {name: i for i, (name, _) in enumerate(STAGES)}

# ---- EKF contact schedule -------------------------------------------------
# Stages in which the swing foot is treated as AIRBORNE by the estimator.
# Conservative from UNLOAD entry (foot commanded up 20-30 mm; a false planted
# constraint is strictly worse than 3-leg dead-reckoning) through the
# TOUCHDOWN *stage* (descending/settling).  The contact rising edge fires at
# the TOUCHDOWN->LOAD transition -- the same sweep that commits the MEASURED
# anchor, so the EKF re-anchors from exactly the alpha that defined it.
EKF_AIRBORNE_STAGES = frozenset(
    ("UNLOAD", "CLEAR_GATE", "WAIT_SWING", "LIFT", "SWING", "LOWER",
     "TOUCHDOWN"))
# z-offset guard: after a CLEAR_GATE-exhausted abort the sequence jumps to
# RECENTER with pre_m still ~30 mm (the foot stays pre-lifted through the
# following HOLD4) -- the stage name alone would falsely say "planted".
EKF_SWING_Z_EPS_M = 0.003
STREAM_T = {"STAND": s3.T_STAND, "SHIFT": s3.T_SHIFT, "UNLOAD": s3.T_UNLOAD,
            "LIFT": s3.T_LIFT, "SWING": T_SWING, "LOWER": T_LOWER,
            "RECENTER": s3.T_SHIFT}
INSTRUCTION = {
    "CROUCH": "moving to recorded crouch; wait, X stops",
    "WAIT_CROUCH": "inspect crouch; ENTER stands, X stops",
    "STAND": "rising vertically in place; wait",
    "STAND_SETTLE": "waiting for the stand to settle",
    "HOLD4": "4-leg stand; ENTER steps, P parks, X stops",
    "SHIFT": "shifting body off the swing corner; wait",
    "SHIFT_SETTLE": "waiting for settle + margin >= 15 mm",
    "WAIT_UNLOAD": "shift OK; ENTER pre-lifts, X stops",
    "UNLOAD": "pre-lifting the swing foot; wait",
    "CLEAR_GATE": "waiting for plane rise + torque drop (auto-retries)",
    "WAIT_SWING": "clear gate OK; ENTER lifts+swings, X stops",
    "LIFT": "lifting the swing foot; wait",
    "SWING": "swinging the foot forward; wait",
    "LOWER": "lowering the foot; wait",
    "TOUCHDOWN": "waiting for touchdown on the stance plane",
    "LOAD": "reloading the foot; wait",
    "RECENTER": "recentering body (advanced fwd); wait",
    "PARK": "returning to recorded crouch; wait, X stops",
    "PARKED": "holding parked crouch; X stops",
}


class WalkSequence:
    """Encoder-only stage machine for the gait (4 steps per cycle)."""

    def __init__(self, now, plan, unload_trip=s3.DEFAULT_UNLOAD_TRIP_NM,
                 time_scale=1.0):
        self.plan = plan
        self.unload_trip = float(unload_trip)
        # Faster walking: scale the stream durations (gates and settle
        # checks stay untouched -- speed never bypasses a gate).
        self.stream_t = {k: v * float(time_scale) for k, v in STREAM_T.items()}
        self.stream_t["STAND"] = STREAM_T["STAND"]   # the rise stays slow
        self.stage_i = 0
        self.stage_started = float(now)
        self.settle_since = None
        self.wait_since = None
        self.q_cmd = Q_CROUCH.copy()
        self.targets = None
        self.step_index = 0          # completed steps this cycle
        self.cycles_done = 0
        # stream interpolation endpoints
        self._from = {}
        self._to = {}
        self.height_frac = 0.0
        self.body_xy = np.zeros(2)
        self.swing_anchor = None
        self.pre_m = 0.0
        self.lift_frac = 0.0
        self.prelift_m = s3.PRELIFT_START_M
        self.plane_baseline_m = None
        self.touchdown_target_m = None
        self.last_margin_m = np.nan
        self.last_rise_m = np.nan
        self.last_swing_tau_nm = np.nan
        self.last_settle_detail = ""
        self.aborted = None
        self.events = []
        self.torque_table_printed = False

    # -- helpers -----------------------------------------------------------
    @property
    def stage(self):
        return STAGES[self.stage_i][0]

    @property
    def kind(self):
        return STAGES[self.stage_i][1]

    @property
    def contacts(self):
        """EKF contact schedule (bool[4], LEGS order FL,FR,RL,RR).

        All planted, except: STAND (feet drag during the vertical rise --
        the EKF dead-reckons through it, matching the validated
        ekf_stand_hw choice) and the swing leg while airborne (stage set
        OR residual commanded z-offset -- covers the CLEAR_GATE-abort path
        where pre_m persists into RECENTER/HOLD4).
        """
        if self.stage == "STAND":
            return np.zeros(4, dtype=bool)
        c = np.ones(4, dtype=bool)
        swing = self.plan.swing
        if swing is not None:
            airborne = (
                self.stage in EKF_AIRBORNE_STAGES
                or self.pre_m + self.lift_frac * self.plan.lift_m
                > EKF_SWING_Z_EPS_M
            )
            if airborne:
                c[LEGS.index(swing)] = False
        return c

    def _snapshot(self):
        return {
            "height": self.height_frac,
            "body": self.body_xy.copy(),
            "swing_anchor": (None if self.swing_anchor is None
                             else self.swing_anchor.copy()),
            "pre": self.pre_m,
            "lift": self.lift_frac,
        }

    def _stream_endpoint(self, name):
        plan = self.plan
        end = self._snapshot()
        if name == "STAND":
            end["height"] = 1.0
        elif name == "SHIFT":
            end["body"] = plan.neutral + plan.shift_vec_for(plan.swing)
        elif name == "UNLOAD":
            end["pre"] = self.prelift_m
        elif name == "LIFT":
            end["lift"] = 1.0
        elif name == "SWING":
            end["swing_anchor"] = self.swing_anchor + [plan.step_m, 0.0]
        elif name == "LOWER":
            end["pre"] = 0.0
            end["lift"] = 0.0
        elif name == "RECENTER":
            end["body"] = plan.neutral.copy()
        else:
            raise ValueError(name)
        return end

    def _goto(self, name, now, note=""):
        self.stage_i = STAGE_INDEX[name]
        self.stage_started = float(now)
        self.settle_since = None
        self.wait_since = None
        if self.kind == "stream":
            self._from = self._snapshot()
            self._to = self._stream_endpoint(name)
        self.events.append((float(now), name, note))
        return f"{name}: {note}" if note else name

    def _abort(self, now, reason):
        self.aborted = reason
        note = f"ABORT step {self.step_index + 1} ({self.plan.swing}): {reason}"
        if self.stage in ("LIFT", "SWING"):
            # foot is airborne: lower it where it stands, then touchdown
            # commits the measured anchor -- same reload path as a normal
            # step, just without the forward progress.
            return self._goto("LOWER", now, note)
        return self._goto("RECENTER", now, note)

    def _settled(self, now, q_enc, qd, velocity_ready):
        errors = np.abs(np.asarray(q_enc) - self.q_cmd)
        worst = int(np.argmax(errors))
        error = float(errors[worst])
        speed = float(np.max(np.abs(qd))) if velocity_ready else np.inf
        self.last_settle_detail = (
            f"worst {JOINT_LABELS[worst]} err {np.rad2deg(error):.1f} deg, "
            f"max|qd| {speed:.2f} rad/s"
        )
        if error > posbase.POSITION_POSE_TOL or speed > posbase.POSITION_QD_TOL:
            self.settle_since = None
            return False
        if self.settle_since is None:
            self.settle_since = float(now)
            return False
        return now - self.settle_since >= posbase.POSITION_SETTLE_S

    # -- operator ----------------------------------------------------------
    def request_next(self, now, q, qd, healthy):
        if self.kind != "wait":
            return False, f"{self.stage} is busy; ENTER ignored."
        if self.stage == "PARKED":
            return False, "already PARKED; X stops."
        if self.wait_since is None or now - self.wait_since < base.WAIT_DWELL_S:
            return False, f"wait for {self.stage} to settle before ENTER."
        if not healthy:
            return False, "motor latch/fault present; motion refused."
        if self.stage == "WAIT_CROUCH":
            return True, self._goto("STAND", now, "ENTER accepted")
        if self.stage == "HOLD4":
            if self.step_index >= len(GAIT_ORDER):
                self.step_index = 0
                self.cycles_done += 1
            swing = GAIT_ORDER[self.step_index]
            self.plan.swing = swing
            self.swing_anchor = self.plan.anchors[swing].copy()
            self.aborted = None
            self.prelift_m = s3.PRELIFT_START_M
            self.plane_baseline_m = None
            return True, self._goto(
                "SHIFT", now,
                f"step {self.step_index + 1}/{len(GAIT_ORDER)} swing={swing}"
            )
        if self.stage == "WAIT_UNLOAD":
            return True, self._goto("UNLOAD", now, "ENTER accepted")
        if self.stage == "WAIT_SWING":
            return True, self._goto("LIFT", now, "ENTER accepted")
        return False, f"no ENTER action in {self.stage}"

    def request_park(self, now, healthy):
        if self.stage != "HOLD4":
            return False, f"P only parks from HOLD4 (now {self.stage})."
        if not healthy:
            return False, "motor latch/fault present; motion refused."
        return True, self._goto("PARK", now, "P accepted")

    # -- per-sweep update --------------------------------------------------
    def sweep(self, now, q_enc, qd, velocity_ready, tau_measured):
        event = None
        name, kind = STAGES[self.stage_i]
        plan = self.plan

        if kind == "move":
            self.q_cmd = Q_CROUCH.copy()
            self.targets = None
            if self._settled(now, q_enc, qd, velocity_ready):
                nxt = {"CROUCH": "WAIT_CROUCH", "PARK": "PARKED"}[name]
                event = self._goto(nxt, now, "settled")
                self.wait_since = float(now)
            elif now - self.stage_started > s3.SETTLE_TIMEOUT_S:
                raise RuntimeError(
                    f"{name} did not settle: {self.last_settle_detail}"
                )

        elif kind == "stream":
            u = s3._smoothstep((now - self.stage_started) / self.stream_t[name])
            f, t = self._from, self._to
            self.height_frac = f["height"] + u * (t["height"] - f["height"])
            self.body_xy = f["body"] + u * (t["body"] - f["body"])
            self.pre_m = f["pre"] + u * (t["pre"] - f["pre"])
            self.lift_frac = f["lift"] + u * (t["lift"] - f["lift"])
            if f["swing_anchor"] is not None:
                self.swing_anchor = f["swing_anchor"] + u * (
                    t["swing_anchor"] - f["swing_anchor"]
                )
            self.targets = plan.targets(
                self.body_xy, self.height_frac, plan.swing,
                self.swing_anchor, self.pre_m, self.lift_frac,
            )
            if now - self.stage_started >= self.stream_t[name]:
                nxt = {"STAND": "STAND_SETTLE", "SHIFT": "SHIFT_SETTLE",
                       "UNLOAD": "CLEAR_GATE", "LIFT": "SWING",
                       "SWING": "LOWER", "LOWER": "TOUCHDOWN",
                       "RECENTER": "HOLD4"}[name]
                if name == "LIFT" and self.aborted:
                    nxt = "LOWER"      # aborted mid-lift: skip the swing
                event = self._goto(nxt, now, "stream complete")
                if nxt == "HOLD4":
                    self.wait_since = float(now)
                    if self.aborted:
                        event += f"  [step ABORTED: {self.aborted}]"
                    elif self.step_index >= len(GAIT_ORDER):
                        event += ("  [GAIT CYCLE COMPLETE: body walked "
                                  f"{1000 * plan.neutral[0]:.0f} mm; ENTER "
                                  "starts another cycle, P parks]")

        elif name == "STAND_SETTLE":
            if self._settled(now, q_enc, qd, velocity_ready):
                event = self._goto("HOLD4", now, "stand settled")
                self.wait_since = float(now)
            elif now - self.stage_started > s3.SETTLE_TIMEOUT_S:
                raise RuntimeError(
                    f"STAND did not settle: {self.last_settle_detail}"
                )

        elif name == "SHIFT_SETTLE":
            self.last_margin_m = plan.fk_margin(q_enc, True)
            if self._settled(now, q_enc, qd, velocity_ready):
                if self.last_margin_m >= MIN_STEP_MARGIN_M:
                    self.plane_baseline_m = plan.plane_distance(
                        q_enc, plan.swing)
                    event = self._goto(
                        "WAIT_UNLOAD", now,
                        f"margin {1000 * self.last_margin_m:.1f} mm, plane "
                        f"baseline {1000 * self.plane_baseline_m:+.1f} mm",
                    )
                    self.wait_since = float(now)
                elif now - self.stage_started > s3.GATE_TIMEOUT_S:
                    event = self._abort(
                        now,
                        f"margin {1000 * self.last_margin_m:.1f} mm < "
                        f"{1000 * MIN_STEP_MARGIN_M:.0f} mm",
                    )
            elif now - self.stage_started > s3.SETTLE_TIMEOUT_S:
                event = self._abort(now, "SHIFT did not settle")

        elif name == "CLEAR_GATE":
            sl = plan._sl(plan.swing)
            self.last_swing_tau_nm = float(
                np.max(np.abs(np.asarray(tau_measured)[sl][1:]))
            )
            self.last_rise_m = (
                plan.plane_distance(q_enc, plan.swing) - self.plane_baseline_m
            )
            clear = (
                self.last_rise_m >= s3.CLEAR_RISE_MIN_M
                and self.last_swing_tau_nm <= self.unload_trip
            )
            if self._settled(now, q_enc, qd, velocity_ready) and clear:
                event = self._goto(
                    "WAIT_SWING", now,
                    f"rise {1000 * self.last_rise_m:+.1f} mm, swing |tau| "
                    f"{self.last_swing_tau_nm:.2f} N*m at pre-lift "
                    f"{1000 * self.prelift_m:.0f} mm",
                )
                self.wait_since = float(now)
            elif now - self.stage_started > s3.GATE_TIMEOUT_S:
                detail = (
                    f"rise {1000 * self.last_rise_m:+.1f} mm, swing |tau| "
                    f"{self.last_swing_tau_nm:.2f} N*m"
                )
                if (self.prelift_m + s3.PRELIFT_RETRY_STEP_M
                        <= s3.PRELIFT_MAX_M + 1.0e-9):
                    self.prelift_m = min(
                        self.prelift_m + s3.PRELIFT_RETRY_STEP_M,
                        s3.PRELIFT_MAX_M,
                    )
                    event = self._goto(
                        "UNLOAD", now,
                        f"clear gate timeout ({detail}); pre-lift to "
                        f"{1000 * self.prelift_m:.0f} mm",
                    )
                else:
                    event = self._abort(
                        now, f"clear gate exhausted at "
                        f"{1000 * self.prelift_m:.0f} mm: {detail}",
                    )

        elif name == "TOUCHDOWN":
            distance = plan.plane_distance(q_enc, plan.swing)
            if self._settled(now, q_enc, qd, velocity_ready) \
                    and abs(distance) <= TOUCHDOWN_TOL_M:
                # Commit the MEASURED anchor: the foot stays wherever it
                # actually landed (crawl convention).
                foot = dog5_kinematics.foot_position(
                    plan.swing, q_enc[plan._sl(plan.swing)]
                )
                landed = foot[:2] + self.body_xy
                plan.anchors[plan.swing] = landed.copy()
                self.swing_anchor = landed.copy()
                if not self.aborted:
                    plan.neutral = plan.neutral + [plan.step_m / 4.0, 0.0]
                    self.step_index += 1
                event = self._goto(
                    "LOAD", now,
                    f"touchdown at plane {1000 * distance:+.1f} mm; anchor "
                    f"committed ({1000 * landed[0]:+.0f},"
                    f"{1000 * landed[1]:+.0f}) mm",
                )
            elif now - self.stage_started > s3.SETTLE_TIMEOUT_S:
                raise RuntimeError(
                    f"TOUCHDOWN never settled on the plane: distance "
                    f"{1000 * distance:+.1f} mm, {self.last_settle_detail}"
                )

        elif kind == "dwell":  # LOAD
            if now - self.stage_started >= T_LOAD:
                event = self._goto("RECENTER", now, "loaded")

        elif kind == "wait":
            if self.wait_since is None:
                self.wait_since = float(now)

        return self.q_cmd, event

    def refine_leg(self, slot_counter):
        if self.targets is None:
            return
        leg = LEGS[slot_counter % len(LEGS)]
        sl = self.plan._sl(leg)
        self.q_cmd[sl] = s3._ik_to_target(
            leg, self.q_cmd[sl], self.targets[leg],
            max_iter=s3.IK_SLOT_MAX_ITER, tol=s3.IK_SLOT_TOL_M,
        )

    def speed_cap_dps(self, caps):
        if self.stage in ("CROUCH", "WAIT_CROUCH", "PARK", "PARKED"):
            return caps["crouch"]
        if self.stage == "STAND":
            return caps["stand"]
        return caps["stream"]


def validate_cycle(plan_args):
    """Offline: IK/limits/margins over the whole scripted cycle at the
    pre-lift cap.  Raises on the first infeasible waypoint."""
    plan = WalkPlan(**plan_args)
    low, high = base.soft_limits()
    q = Q_CROUCH.copy()
    worst_margin = np.inf
    # stand up
    for h in np.linspace(0.0, 1.0, 9):
        t = plan.targets(np.zeros(2), h, None, None, 0.0, 0.0)
        for leg in LEGS:
            sl = plan._sl(leg)
            q[sl] = s3._ik_to_target(leg, q[sl], t[leg])
    for step_i, swing in enumerate(GAIT_ORDER):
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
            a = anchor + [u * plan.step_m, 0.0]
            waypoints.append((shift_body, a, s3.PRELIFT_MAX_M, 1.0))
        landed = anchor + [plan.step_m, 0.0]
        waypoints.append((shift_body, landed, 0.0, 0.0))
        plan.anchors[swing] = landed
        new_neutral = plan.neutral + [plan.step_m / 4.0, 0.0]
        for u in np.linspace(0.0, 1.0, 6):
            body = shift_body + u * (new_neutral - shift_body)
            waypoints.append((body, landed, 0.0, 0.0))
        plan.neutral = new_neutral
        for body, a, pre, lift in waypoints:
            t = plan.targets(body, 1.0, swing, a, pre, lift)
            for leg in LEGS:
                sl = plan._sl(leg)
                q[sl] = s3._ik_to_target(leg, q[sl], t[leg])
                err = float(np.linalg.norm(
                    dog5_kinematics.foot_position(leg, q[sl]) - t[leg]
                ))
                if err > 1.0e-4:
                    raise ValueError(
                        f"step {step_i + 1} ({swing}): {leg} IK residual "
                        f"{err:.4f} m"
                    )
                if np.any(q[sl] < low[sl]) or np.any(q[sl] > high[sl]):
                    raise ValueError(
                        f"step {step_i + 1} ({swing}): {leg} outside soft "
                        f"limits at body={body}, pre={1000 * pre:.0f}mm"
                    )
    return worst_margin


def run_hardware(args):
    plan_args = dict(step_m=args.step, shift_m=args.shift, lift_m=args.lift,
                     stand_height=args.stand_height,
                     front_back_shift=args.front_back_shift)
    worst_margin = validate_cycle(plan_args)
    plan = WalkPlan(**plan_args)
    caps = {"crouch": args.crouch_dps, "stand": args.stand_dps,
            "stream": args.stream_dps}
    unwrap = [base.CalibratedEncoderUnwrap() for _ in base.HARDWARE_JOINTS]
    safety = base.SafetyGate(
        tau_cap=1.0, qd_estop=args.qd_estop, qd_estop_hard=args.qd_estop_hard
    )

    print(f"[walk1] ONE GAIT CYCLE: order {'->'.join(GAIT_ORDER)}, step "
          f"{1000 * args.step:.0f} mm, body +{1000 * args.step / 4:.1f} mm "
          f"per step; planned worst shift margin {1000 * worst_margin:.1f} mm")
    print(f"[walk1] front-leg back-shift {1000 * args.front_back_shift:.1f} mm "
          f"(lateral stays {1000 * args.shift / np.sqrt(2.0):.1f} mm)")
    print(f"[walk1] per-step machinery = the proven stand3 ladder: shift "
          f"{1000 * args.shift:.0f} mm, pre-lift "
          f"{1000 * s3.PRELIFT_START_M:.0f}->"
          f"{1000 * s3.PRELIFT_MAX_M:.0f} mm, lift {1000 * args.lift:.0f} mm, "
          f"touchdown tol +-{1000 * TOUCHDOWN_TOL_M:.0f} mm")
    if args.auto:
        print(f"[walk1] AUTO: {args.cycles} cycle(s) = "
              f"{args.cycles * len(GAIT_ORDER)} steps, walking "
              f"{1000 * args.cycles * args.step:.0f} mm hands-free; an abort "
              "pauses for ENTER; X stops, P parks at any HOLD4")
    if args.time_scale != 1.0:
        print(f"[walk1] time scale {args.time_scale:.2f}: motion phases "
              f"{'faster' if args.time_scale < 1 else 'slower'}; gates and "
              "settle checks unchanged")
    print("[walk1] ROBOT MUST REMAIN MECHANICALLY SUPPORTED UNTIL PROVEN.")

    # One serial-port owner: ImuEkfFeed REPLACES inplace._open_imu() -- it
    # wraps the same ImuDog (AHRS attitude via feed.attitude()) and adds the
    # raw 0x40 stream the EKF predicts on.
    feed = None
    ekf_enabled = False
    if not args.no_imu:
        feed = ImuEkfFeed(args.imu_port).start()
        print(f"[imu] tip-over {inplace.TIPOVER_ABORT_DEG:.0f} deg; lean "
              f"watch {s3.LEAN_ABORT_DEG:.0f} deg in LIFT/SWING")
        ekf_enabled = not args.no_ekf
        if ekf_enabled and not feed.wait_for_raw(timeout=3.0):
            print("[ekf] WARNING: no raw IMU (0x40) packets -- EKF disabled "
                  "(walk continues on AHRS posture checks only)")
            ekf_enabled = False
    if args.raw_log and not ekf_enabled:
        raise RuntimeError("--raw-log needs the EKF (raw IMU stream) enabled")

    shared = None
    worker = None
    web_stop = None
    enc_log = []       # (t_mono, alpha12, contacts4, roll_ahrs, pitch_ahrs)
    key = base.KeyPoller()
    try:
        with base.motorbus.MotorBus(MOTOR_IDS, dirs=MOTOR_DIRECTIONS) as mb:
            armed = False
            stop_reason = None
            try:
                print(f"[ARM] arming for up to {posbase.ARM_TIMEOUT_S:.0f}s")
                if not mb.arm(rate_hz=base.CONTROL_HZ,
                              timeout_s=posbase.ARM_TIMEOUT_S, verbose=False):
                    raise RuntimeError(
                        "arming timed out: " + posbase.arm_failure_summary(mb)
                    )
                armed = True
                start_q = posbase.zero_torque_preflight(mb, key, unwrap)
                now = time.perf_counter()
                sequence = WalkSequence(now, plan,
                                        unload_trip=args.unload_trip,
                                        time_scale=args.time_scale)
                safety.start(now, start_q)

                if ekf_enabled:
                    shared = EkfShared(start_q)
                    worker = threading.Thread(
                        target=ekf_worker, args=(shared, feed),
                        kwargs=dict(quiet_stages=("WAIT_CROUCH",),
                                    verbose=args.ekf_verbose),
                        daemon=True)
                    worker.start()
                    print("[ekf] worker started (read-only, 100 Hz); "
                          "initialises during WAIT_CROUCH -- wait for "
                          "'[ekf] init' before ENTER")
                    if args.web:
                        web_stop, urls = ekf_web.start(shared, port=args.web)
                        for u in urls:
                            print(f"[web] live EKF dashboard: {u}")

                slot = mb.slot(base.CONTROL_HZ)
                deadline = time.perf_counter() + slot
                run_started_at = now
                q = start_q.copy()
                miss_monitor = base.CanMissMonitor(mb)
                status_period = 1.0 / base.FAULT_STATUS_HZ
                next_fault_status = np.asarray([
                    status_period + i * status_period / N_JOINTS
                    for i in range(N_JOINTS)
                ])
                last_recover = {
                    mid: -base.RECOVER_PERIOD_S for mid in MOTOR_IDS
                }
                last_print = 0.0
                index = 0
                velocity = posbase.EncoderVelocity()
                qd_encoder = np.zeros(N_JOINTS)
                worst_block_ms = 0.0
                tipover_streak = 0
                lean_baseline = None
                lean_streak = 0
                imu_sample = None
                initial_yaw_deg = None
                last_yaw_deg = None
                bias_printed = False

                while True:
                    mb.poll()
                    joint_index = index % N_JOINTS

                    if joint_index == 0:
                        now = time.perf_counter()
                        elapsed = now - run_started_at
                        q, qd_driver = base._joint_state(mb, unwrap)
                        qd_encoder = velocity.update(q, now)
                        torque_feedback = mb.torques_nm()
                        measured_torque = np.asarray(
                            [torque_feedback[mid] for mid in MOTOR_IDS]
                        )

                        _, event = sequence.sweep(
                            now, q, qd_encoder, velocity.ready, measured_torque
                        )
                        if event:
                            print(f"[stage] {event}")
                        if shared is not None:
                            shared.q = q
                            shared.qd = qd_encoder
                            shared.stage = sequence.stage
                            shared.contacts = sequence.contacts
                            if (not shared.log_enabled
                                    and sequence.stage != "CROUCH"):
                                # raw log starts at the static WAIT_CROUCH ->
                                # clean all-contact init prefix for hw_replay
                                shared.log_enabled = True
                            if not bias_printed and shared.est_ready:
                                print(f"[ekf] init: {shared.bias_str}")
                                bias_printed = True
                        if (sequence.stage == "HOLD4"
                                and not sequence.torque_table_printed):
                            s3._print_torque_table(measured_torque)
                            sequence.torque_table_printed = True

                        temps = base._temperatures(mb)
                        misses = miss_monitor.update(mb)
                        errors = mb.errors()
                        stop_reason = safety.estop_reason(
                            q, qd_driver, temps, misses, errors, now,
                            enforce_position_limits=(
                                sequence.stage != "CROUCH"
                            ),
                        )
                        if stop_reason is None:
                            stop_reason = posbase.position_fault(
                                qd_encoder, measured_torque, velocity.ready,
                                args.speed_trip, args.torque_trip,
                            )

                        imu_sample = None
                        if feed is not None:
                            try:
                                imu_sample = feed.attitude()
                            except Exception:
                                imu_sample = None
                            fresh = (
                                imu_sample is not None
                                and imu_sample.age_s <= inplace.POSTURE_STALE_S
                            )
                            if fresh:
                                roll = imu_sample.roll_deg
                                pitch = imu_sample.pitch_deg
                                last_yaw_deg = imu_sample.yaw_deg
                                if initial_yaw_deg is None:
                                    initial_yaw_deg = imu_sample.yaw_deg
                                    print(f"[imu] initial yaw "
                                          f"{initial_yaw_deg:+.1f} deg (mag)")
                                if (abs(roll) > inplace.TIPOVER_ABORT_DEG
                                        or abs(pitch)
                                        > inplace.TIPOVER_ABORT_DEG):
                                    tipover_streak += 1
                                    if (stop_reason is None
                                            and tipover_streak
                                            >= inplace.TIPOVER_CONFIRM_SAMPLES):
                                        stop_reason = (
                                            f"IMU tip-over: roll={roll:+.1f} "
                                            f"pitch={pitch:+.1f} deg"
                                        )
                                else:
                                    tipover_streak = 0
                                if sequence.stage in ("LIFT", "SWING"):
                                    if lean_baseline is None:
                                        lean_baseline = (roll, pitch)
                                        lean_streak = 0
                                    else:
                                        d_r = roll - lean_baseline[0]
                                        d_p = pitch - lean_baseline[1]
                                        if max(abs(d_r), abs(d_p)) \
                                                > s3.LEAN_ABORT_DEG:
                                            lean_streak += 1
                                            if lean_streak >= s3.LEAN_STREAK:
                                                print("[stage] " +
                                                      sequence._abort(
                                                          now,
                                                          f"IMU lean d_roll="
                                                          f"{d_r:+.1f} "
                                                          f"d_pitch="
                                                          f"{d_p:+.1f} deg"))
                                                lean_baseline = None
                                                lean_streak = 0
                                        else:
                                            lean_streak = 0
                                else:
                                    lean_baseline = None
                                    lean_streak = 0

                        if shared is not None:
                            # display state for the web dashboard: REBIND a
                            # fresh dict (the sampler thread reads it lock-free)
                            shared.status = {
                                "stage": sequence.stage,
                                "step": f"{min(sequence.step_index + 1, 4)}/4",
                                "swing": plan.swing or "--",
                                "blk_ms": round(worst_block_ms, 2),
                                "ahrs_roll": (
                                    None if imu_sample is None
                                    else round(imu_sample.roll_deg, 3)),
                                "ahrs_pitch": (
                                    None if imu_sample is None
                                    else round(imu_sample.pitch_deg, 3)),
                            }

                        if (args.raw_log and shared is not None
                                and shared.log_enabled):
                            r_a = p_a = float("nan")
                            if (imu_sample is not None
                                    and imu_sample.age_s
                                    <= inplace.POSTURE_STALE_S):
                                r_a = math.radians(imu_sample.roll_deg)
                                p_a = math.radians(imu_sample.pitch_deg)
                            enc_log.append(
                                (time.monotonic(), *q,
                                 *shared.contacts.astype(int), r_a, p_a))

                        latched = [
                            mid for mid, error in errors.items()
                            if error & 0x80
                        ]
                        unverified = [
                            mid for mid in MOTOR_IDS
                            if mb.rec(mid).error is None
                        ]
                        pressed = key.get()
                        healthy = (not latched and not unverified
                                   and stop_reason is None)
                        if pressed in ("x", "X"):
                            stop_reason = "operator X"
                        elif pressed in ("p", "P"):
                            _, message = sequence.request_park(now, healthy)
                            print(f"[stage] {message}")
                        elif base._is_enter(pressed):
                            _, message = sequence.request_next(
                                now, q, qd_encoder, healthy
                            )
                            print(f"[stage] {message}")
                        elif (
                            args.auto
                            and healthy
                            and sequence.aborted is None
                            and sequence.stage in (
                                "HOLD4", "WAIT_UNLOAD", "WAIT_SWING"
                            )
                            and sequence.wait_since is not None
                            and now - sequence.wait_since
                            >= base.WAIT_DWELL_S + 0.2
                        ):
                            # --auto: advance the step gates by itself while
                            # steps remain; an abort always pauses for the
                            # operator.  WAIT_CROUCH stays manual.
                            steps_done = (
                                sequence.cycles_done * len(GAIT_ORDER)
                                + sequence.step_index
                            )
                            if (sequence.stage != "HOLD4"
                                    or steps_done
                                    < args.cycles * len(GAIT_ORDER)):
                                advanced, message = sequence.request_next(
                                    now, q, qd_encoder, healthy
                                )
                                if advanced:
                                    print(f"[stage] (auto) {message}")
                        if stop_reason:
                            break

                        recover = [
                            mid for mid in latched
                            if elapsed - last_recover[mid]
                            >= base.RECOVER_PERIOD_S
                        ]
                        if recover:
                            base._recover_input_lost(
                                mb, recover, elapsed, last_recover,
                                next_fault_status,
                            )
                            latched = []

                        if now - last_print >= 1.0 / base.STATUS_HZ:
                            sl = plan._sl(plan.swing) if plan.swing else None
                            swing_tau = (
                                float(np.max(np.abs(measured_torque[sl][1:])))
                                if sl else 0.0
                            )
                            margin = ("-" if not np.isfinite(
                                sequence.last_margin_m)
                                else f"{1000 * sequence.last_margin_m:.1f}mm")
                            imu_text = ""
                            if imu_sample is not None:
                                imu_text = (
                                    f" rp=({imu_sample.roll_deg:+.2f},"
                                    f"{imu_sample.pitch_deg:+.2f})deg"
                                )
                            print(
                                f"[walk1] step="
                                f"{min(sequence.step_index + 1, 4)}/4 "
                                f"swing={plan.swing or '--'} "
                                f"{sequence.stage:<12} "
                                f"{INSTRUCTION[sequence.stage]} | "
                                f"margin={margin} "
                                f"prelift={1000 * sequence.prelift_m:.0f}mm "
                                f"swing|tau|={swing_tau:.2f} "
                                f"max|tau|="
                                f"{np.max(np.abs(measured_torque)):.2f}N*m "
                                f"blk={worst_block_ms:.1f}ms"
                                f"{imu_text}",
                                flush=True,
                            )
                            if shared is not None:
                                out = shared.out
                                if shared.est_ready and out is not None:
                                    re_, pe, ye = _rpy(out["C"])
                                    vb = out["v_body"] * 1e3
                                    cbits = "".join(
                                        "1" if c else "0"
                                        for c in shared.contacts)
                                    ahrs_text = "--"
                                    if imu_sample is not None:
                                        ahrs_text = (
                                            f"({imu_sample.roll_deg:+.1f},"
                                            f"{imu_sample.pitch_deg:+.1f})")
                                    print(
                                        f"[ekf] z={out['r'][2]*1e3:+5.0f}mm "
                                        f"vb=({vb[0]:+4.0f},{vb[1]:+4.0f},"
                                        f"{vb[2]:+4.0f})mm/s "
                                        f"rpy=({math.degrees(re_):+5.1f},"
                                        f"{math.degrees(pe):+5.1f},"
                                        f"{math.degrees(ye):+6.1f})deg "
                                        f"ahrs={ahrs_text} "
                                        f"|bf|={np.linalg.norm(out['bf']):.3f} "
                                        f"|bw|={np.linalg.norm(out['bw']):.1e} "
                                        f"c={cbits} "
                                        f"H={'OK' if out['healthy'] else '!!'}",
                                        flush=True,
                                    )
                                    if args.ekf_verbose and shared.extra:
                                        ex = shared.extra
                                        foot_text = " ".join(
                                            f"{leg}=({p[0]*1e3:+4.0f},"
                                            f"{p[1]*1e3:+4.0f})"
                                            for leg, p in ex["foot"].items())
                                        innov_text = " ".join(
                                            f"{leg}:{e:.1f}"
                                            for leg, e in ex["innov"].items())
                                        print(f"[ekf+] foot_xy_mm {foot_text}"
                                              f" | innov_sig {innov_text}",
                                              flush=True)
                                else:
                                    print("[ekf] initialising"
                                          " (needs static WAIT_CROUCH)...",
                                          flush=True)
                            last_print = now
                            worst_block_ms = 0.0
                        worst_block_ms = max(
                            worst_block_ms,
                            1000.0 * (time.perf_counter() - now),
                        )

                    if index % s3.IK_SLOT_STRIDE == 0:
                        sequence.refine_leg(index // s3.IK_SLOT_STRIDE)

                    mid = MOTOR_IDS[joint_index]
                    elapsed = time.perf_counter() - run_started_at
                    if elapsed >= next_fault_status[joint_index]:
                        mb.status1_req(mid)
                        next_fault_status[joint_index] += status_period
                    else:
                        if not mb.position(
                            mid,
                            float(np.rad2deg(sequence.q_cmd[joint_index])),
                            sequence.speed_cap_dps(caps),
                        ):
                            raise RuntimeError(
                                f"CAN position transmit failed for CAN {mid}"
                            )

                    index += 1
                    overrun = mb.pace(deadline)
                    deadline += slot
                    if overrun and overrun > 2.0 * slot:
                        # resync after a stall (GIL/IO hiccup) instead of
                        # send-bursting to catch up (ENOBUFS lesson)
                        deadline = time.perf_counter() + slot

            except KeyboardInterrupt as exc:
                stop_reason = str(exc) or "KeyboardInterrupt"
            except Exception as exc:
                stop_reason = f"error: {exc}"
                raise
            finally:
                if armed:
                    print(f"[STOP] {stop_reason or 'aborted'}")
                    if initial_yaw_deg is not None and last_yaw_deg is not None:
                        d = last_yaw_deg - initial_yaw_deg
                        d = (d + 180.0) % 360.0 - 180.0
                        print(f"[imu] yaw: initial {initial_yaw_deg:+.1f} -> "
                              f"final {last_yaw_deg:+.1f} deg, net rotation "
                              f"{d:+.1f} deg (mag heading; + = nose swung "
                              f"LEFT over the run)")
                    mb.stop_all()
    finally:
        key.close()
        if web_stop is not None:
            web_stop()
        if shared is not None:
            shared.run = False
        if worker is not None:
            worker.join(timeout=1.0)
        if feed is not None:
            try:
                feed.stop()
            except Exception:
                pass
        if args.raw_log and shared is not None:
            imu_arr = np.array(shared.imu_log, dtype=float)
            enc_arr = np.array(enc_log, dtype=float)
            if len(imu_arr) and len(enc_arr):
                np.savez(args.raw_log,
                         imu_t=imu_arr[:, 0], imu_f=imu_arr[:, 1:4],
                         imu_w=imu_arr[:, 4:7],
                         enc_t=enc_arr[:, 0], enc_alpha=enc_arr[:, 1:13],
                         enc_contacts=enc_arr[:, 13:17],
                         ahrs_rp=enc_arr[:, 17:19],
                         init_secs=1.0)
                print(f"[raw] wrote {args.raw_log} ({len(imu_arr)} IMU, "
                      f"{len(enc_arr)} enc frames); analyse with "
                      f"hw_replay.py --gait")
            else:
                print("[raw] nothing to write (no IMU/enc frames logged)")
    return 0


def offline_self_test(args):
    plan_args = dict(step_m=args.step, shift_m=args.shift, lift_m=args.lift,
                     stand_height=args.stand_height,
                     front_back_shift=args.front_back_shift)
    worst_margin = validate_cycle(plan_args)
    print(f"full cycle IK-feasible; worst planned shift margin "
          f"{1000 * worst_margin:.1f} mm (gate "
          f"{1000 * MIN_STEP_MARGIN_M:.0f} mm)")
    assert worst_margin >= MIN_STEP_MARGIN_M + 0.005

    # Dry-run the sequence through one full cycle with a synthetic plant.
    plan = WalkPlan(**plan_args)
    seq = WalkSequence(0.0, plan, unload_trip=args.unload_trip)
    now = 0.0
    observed = set()      # (stage, swing, contact-bits) seen by the EKF feed

    def spin(stop_stages, limit_s=90.0, tau=None):
        nonlocal now
        deadline = now + limit_s
        while seq.stage not in stop_stages:
            now += 0.048
            tt = np.zeros(N_JOINTS) if tau is None else tau
            seq.sweep(now, seq.q_cmd.copy(), np.zeros(N_JOINTS), True, tt)
            for k in range(4):
                seq.refine_leg(k)
            observed.add((seq.stage, plan.swing,
                          tuple(int(c) for c in seq.contacts)))
            assert now < deadline, f"stuck in {seq.stage}"

    def enter():
        nonlocal now
        now += base.WAIT_DWELL_S + 0.1
        ok, msg = seq.request_next(now, seq.q_cmd, np.zeros(N_JOINTS), True)
        assert ok, msg

    spin(("WAIT_CROUCH",))
    assert tuple(seq.contacts) == (1, 1, 1, 1)
    enter()
    spin(("HOLD4",))
    for step in range(4):
        enter()                      # HOLD4 -> SHIFT
        spin(("WAIT_UNLOAD",))
        assert tuple(seq.contacts) == (1, 1, 1, 1), (
            "shifted stance must be all-planted", seq.contacts)
        enter()                      # -> UNLOAD
        spin(("WAIT_SWING",))
        swing_i = LEGS.index(plan.swing)
        assert not seq.contacts[swing_i], "pre-lifted foot must be airborne"
        enter()                      # -> LIFT/SWING/LOWER/TOUCHDOWN...
        spin(("HOLD4",))
        assert tuple(seq.contacts) == (1, 1, 1, 1), (
            "post-step HOLD4 must be all-planted", seq.contacts)
        assert seq.aborted is None, seq.aborted
        assert seq.step_index == step + 1, (seq.step_index, step)

    # EKF contact-schedule oracle over everything the dry run visited:
    # STAND dead-reckons (all off); otherwise exactly the swing leg is off in
    # the airborne stages and never anywhere else; support is always >= 3.
    for stage, swing, bits in observed:
        if stage == "STAND":
            assert bits == (0, 0, 0, 0), (stage, bits)
            continue
        assert sum(bits) >= 3, ("support < 3 outside STAND", stage, bits)
        if stage in EKF_AIRBORNE_STAGES:
            assert bits[LEGS.index(swing)] == 0, (stage, swing, bits)
    for leg in GAIT_ORDER:
        expect = tuple(0 if l == leg else 1 for l in LEGS)
        for st in ("UNLOAD", "SWING", "TOUCHDOWN"):
            assert (st, leg, expect) in observed, (st, leg)
        assert ("LOAD", leg, (1, 1, 1, 1)) in observed, leg
    print("EKF contact schedule: STAND dead-reckons, exactly the swing leg "
          "airborne UNLOAD->TOUCHDOWN, re-planted from LOAD")
    assert np.isclose(plan.neutral[0], args.step), plan.neutral
    for leg in LEGS:
        expected = (dog5_kinematics.foot_position(
            leg, Q_CROUCH[plan._sl(leg)])[0] + args.step)
        # anchors are committed from MEASURED FK, so they carry the IK
        # tolerance (~0.1 mm), not machine precision
        assert abs(plan.anchors[leg][0] - expected) < 5.0e-4, (
            leg, plan.anchors[leg][0], expected
        )
    print(f"dry run: 4 steps complete, body walked "
          f"{1000 * plan.neutral[0]:.1f} mm, every anchor advanced "
          f"{1000 * args.step:.0f} mm")

    # Lean-abort from SWING routes through LOWER and still commits an anchor.
    plan2 = WalkPlan(**plan_args)
    seq2 = WalkSequence(0.0, plan2, unload_trip=args.unload_trip)
    now = 0.0

    def spin2(stop_stages, limit_s=90.0):
        nonlocal now
        deadline = now + limit_s
        while seq2.stage not in stop_stages:
            now += 0.048
            seq2.sweep(now, seq2.q_cmd.copy(), np.zeros(N_JOINTS), True,
                       np.zeros(N_JOINTS))
            for k in range(4):
                seq2.refine_leg(k)
            assert now < deadline, f"stuck in {seq2.stage}"

    spin2(("WAIT_CROUCH",))
    now += base.WAIT_DWELL_S + 0.1
    seq2.request_next(now, seq2.q_cmd, np.zeros(N_JOINTS), True)
    spin2(("HOLD4",))
    now += base.WAIT_DWELL_S + 0.1
    seq2.request_next(now, seq2.q_cmd, np.zeros(N_JOINTS), True)
    spin2(("WAIT_UNLOAD",))
    now += base.WAIT_DWELL_S + 0.1
    seq2.request_next(now, seq2.q_cmd, np.zeros(N_JOINTS), True)
    spin2(("WAIT_SWING",))
    now += base.WAIT_DWELL_S + 0.1
    seq2.request_next(now, seq2.q_cmd, np.zeros(N_JOINTS), True)
    spin2(("SWING",))
    print("[synthetic lean abort] " + seq2._abort(now, "test lean"))
    assert seq2.stage == "LOWER"
    assert not seq2.contacts[LEGS.index(plan2.swing)], \
        "aborted swing foot still airborne through LOWER"
    spin2(("HOLD4",))
    assert seq2.aborted is not None
    assert seq2.step_index == 0, "aborted step must not count"
    assert tuple(seq2.contacts) == (1, 1, 1, 1), \
        "abort-via-LOWER re-plants the foot by HOLD4"
    print("abort mid-SWING: lowered, anchor committed, recentered, "
          "step not counted")

    # CLEAR_GATE exhaustion: high swing torque defeats the gate; the pre-lift
    # ladder (20->25->30 mm) times out and aborts straight to RECENTER with
    # the foot STILL pre-lifted -- the z-offset guard (not the stage name)
    # must keep the EKF contact off through RECENTER and the ensuing HOLD4.
    plan3 = WalkPlan(**plan_args)
    seq3 = WalkSequence(0.0, plan3, unload_trip=args.unload_trip)
    now = 0.0
    tau_high = np.full(N_JOINTS, 5.0)

    def spin3(stop_stages, limit_s=90.0, tau=None):
        nonlocal now
        deadline = now + limit_s
        while seq3.stage not in stop_stages:
            now += 0.048
            tt = np.zeros(N_JOINTS) if tau is None else tau
            seq3.sweep(now, seq3.q_cmd.copy(), np.zeros(N_JOINTS), True, tt)
            for k in range(4):
                seq3.refine_leg(k)
            assert now < deadline, f"stuck in {seq3.stage}"

    def enter3():
        nonlocal now
        now += base.WAIT_DWELL_S + 0.1
        ok, msg = seq3.request_next(now, seq3.q_cmd, np.zeros(N_JOINTS), True)
        assert ok, msg

    spin3(("WAIT_CROUCH",))
    enter3()
    spin3(("HOLD4",))
    enter3()                         # -> SHIFT (swing = RR)
    spin3(("WAIT_UNLOAD",))
    enter3()                         # -> UNLOAD; gate never clears
    spin3(("RECENTER",), limit_s=120.0, tau=tau_high)
    swing3 = LEGS.index(plan3.swing)
    assert seq3.aborted is not None and "exhausted" in seq3.aborted, \
        seq3.aborted
    assert seq3.pre_m > EKF_SWING_Z_EPS_M, seq3.pre_m
    assert not seq3.contacts[swing3], \
        "z-guard: pre-lifted foot must stay airborne through the abort RECENTER"
    spin3(("HOLD4",), tau=tau_high)
    assert not seq3.contacts[swing3], \
        "z-guard: foot still pre-lifted in post-abort HOLD4 -> still airborne"
    print("CLEAR_GATE exhaustion: abort keeps the pre-lifted foot airborne "
          "for the EKF through RECENTER and HOLD4 (z-offset guard)")
    # Time scale plumbing: streams scale, the stand rise does not.
    fast = WalkSequence(0.0, WalkPlan(**plan_args), time_scale=0.5)
    assert np.isclose(fast.stream_t["SWING"], 0.5 * T_SWING)
    assert np.isclose(fast.stream_t["SHIFT"], 0.5 * s3.T_SHIFT)
    assert np.isclose(fast.stream_t["STAND"], s3.T_STAND)
    print("time-scale: streams scale, STAND rise fixed")
    print("walk1_hw offline self-test PASS")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--auto", action="store_true",
        help="advance the step gates automatically (HOLD4/UNLOAD/SWING) "
             "while steps remain; aborts always pause for the operator; "
             "X and P still work; the crouch->stand ENTER stays manual")
    parser.add_argument(
        "--cycles", type=int, default=1,
        help="gait cycles to walk with --auto (4 steps, +step_m each)")
    parser.add_argument(
        "--time-scale", type=float, default=1.0,
        help="scale the motion-phase durations (0.5 = twice as fast); "
             "gates and settle checks are never shortened")
    parser.add_argument("--step", type=float, default=DEFAULT_STEP_M)
    parser.add_argument("--shift", type=float, default=DEFAULT_SHIFT_M)
    parser.add_argument(
        "--front-back-shift", type=float, default=FRONT_BACK_SHIFT_M,
        help="backward shift component for FRONT-leg steps (m).  The proven "
             "default 0.015 protects forward reach for 40 mm steps; shorter "
             "steps can afford more (deeper CoM shift off the FL/FR corner "
             "-> larger real margin during UNLOAD).  Always validated by "
             "the offline cycle check before hardware motion.")
    parser.add_argument("--lift", type=float, default=DEFAULT_LIFT_M)
    parser.add_argument("--stand-height", type=float,
                        default=DEFAULT_STAND_HEIGHT_M)
    parser.add_argument("--torque-trip", type=float,
                        default=s3.DEFAULT_TORQUE_TRIP_NM)
    parser.add_argument("--speed-trip", type=float,
                        default=s3.DEFAULT_SPEED_TRIP_RAD_S)
    parser.add_argument("--unload-trip", type=float,
                        default=s3.DEFAULT_UNLOAD_TRIP_NM)
    parser.add_argument("--crouch-dps", type=float,
                        default=s3.DEFAULT_CROUCH_DPS)
    parser.add_argument("--stand-dps", type=float,
                        default=s3.DEFAULT_STAND_DPS)
    parser.add_argument("--stream-dps", type=float,
                        default=s3.DEFAULT_STREAM_DPS)
    parser.add_argument("--qd-estop", type=float, default=base.QD_ESTOP)
    parser.add_argument("--qd-estop-hard", type=float,
                        default=base.QD_ESTOP_HARD)
    parser.add_argument("--no-imu", action="store_true")
    parser.add_argument("--no-ekf", action="store_true",
                        help="skip the read-only EKF worker (IMU still "
                             "drives the posture checks)")
    parser.add_argument("--ekf-verbose", action="store_true",
                        help="extra [ekf+] line: per-leg foothold xy + max "
                             "normalised innovation")
    parser.add_argument("--imu-port", default=IMU_DEFAULT_PORT)
    parser.add_argument("--raw-log", default=None,
                        help="NPZ raw IMU+enc+contacts log for "
                             "hw_replay.py --gait")
    parser.add_argument("--web", type=int, nargs="?", const=8080, default=None,
                        metavar="PORT",
                        help="serve the live EKF dashboard on PORT "
                             "(default 8080); open it in a browser on any "
                             "machine that can reach this Pi")
    args = parser.parse_args()

    if args.raw_log and args.no_imu:
        parser.error("--raw-log needs the IMU (drop --no-imu)")

    if not 0.4 <= args.time_scale <= 1.5:
        parser.error("--time-scale must be between 0.4 and 1.5")
    if not 1 <= args.cycles <= 20:
        parser.error("--cycles must be between 1 and 20")
    if not 0.0 <= args.front_back_shift <= 0.030:
        parser.error("--front-back-shift must be between 0 and 0.030")
    if args.self_test:
        return offline_self_test(args)
    try:
        return run_hardware(args)
    except (RuntimeError, ValueError) as exc:
        print(f"[walk1] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
