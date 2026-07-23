#!/usr/bin/env python3
"""Three-leg stand HOLD on DOG5 hardware -- port of the sim-proven
``stand3_dog5.py`` (branch ``dog5-crawl-sim``) onto the native-position
command track, with the ``crawl_dog5_sim.py`` clear-gate upgrades.

Everything the sim proved is kept: kinematic position control only (0xA4
targets + motor-side speed caps, no torque law), encoder-only feedback at
one state update per CAN round-robin sweep (~20.8 Hz), all FK/IK/Jacobians
from the NumPy ``dog5_kinematics`` module.  The controller never reads the
IMU -- it only judges (display, tip-over run-stop, lean-watch step abort),
exactly like the sim's oracle.

The stand is IN-PLACE: the body rises vertically over the crouch feet (the
feet never slide -- the sim's sprawl stand needs an ~8 cm outward foot
slide, which real floor grip blocks; observed as a knee stalling 46 deg
short with no torque).  All four swing legs are IK-feasible from this
stance, rear legs included.

Operator-gated flow (ENTER at every phase boundary, X stops anywhere):

    READ -> CROUCH -> WAIT_CROUCH  -ENTER-> STAND (vertical rise) -> HOLD4
        (HOLD4 prints the 12 measured torques: a big left/right or diagonal
         asymmetry is the mis-calibration signature of SIM_APPROACH_HW #7-C)
    -ENTER-> SHIFT (body 32 mm diagonally away from the swing leg)
          -> SHIFT_SETTLE   gate: settle + encoder-FK margin >= 15 mm
    -ENTER-> UNLOAD (pre-lift, starts 20 mm)
          -> CLEAR_GATE     gate: ABSOLUTE stance-plane rise >= 4 mm
                            AND swing pitch/knee |tau| <= 0.7 N*m;
                            timeout raises the pre-lift +5 mm and retries,
                            up to 30 mm, then aborts gracefully
    -ENTER-> LIFT (+14 mm) -> HOLD3 (holds until ENTER; lean watch active)
    -ENTER-> LOWER -> RELOAD -> RECENTER -> HOLD4   (cycle can repeat)
    HOLD4 -P-> PARK -> PARKED

A failed gate aborts like the crawl: lower if airborne, recenter, back to
HOLD4 with the reason printed.  Expect the plane-rise readout to be about
(pre-lift - 12 mm): unloading the corner tilts the trunk 1.4-1.6 deg and
physically lowers it ~12 mm.  That is sag, not a stuck leg.

First checks:
    python stand3_hold_hw.py --self-test
Hardware (robot supported until proven):
    python stand3_hold_hw.py                  # defaults: FR, sim-tuned gates
    python stand3_hold_hw.py --swing-leg RL --torque-trip 6.0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dog5_description"))

import dog5_kinematics                      # noqa: E402
import stand_dog5_hw as base                # noqa: E402
import stand_by_position_command as posbase  # noqa: E402
import stand_dog5_inplace_hw as inplace     # noqa: E402
from crawl_dog5_hw import (                 # noqa: E402
    signed_distance_to_stance_plane,
    support_triangle_margin,
)

LEGS = base.LEGS
MOTOR_IDS = base.MOTOR_IDS
MOTOR_DIRECTIONS = base.MOTOR_DIRECTIONS
JOINT_LABELS = base.JOINT_LABELS
N_JOINTS = base.N_JOINTS

Q_CROUCH = posbase.Q_CROUCH
Q_STAND = posbase.Q_STAND

# ---- tuning: sim gates (SIM_APPROACH_HW.md sections 6-8) on the IN-PLACE
# stance.  The sim's sprawl stand needs the feet to slide ~8 cm outward
# during crouch->stand; on the real high-grip floor the feet stick and a
# knee stalls below the torque trip (observed: FR_knee 45.9 deg short, qd 0,
# no trip -- the sim's own experiment B).  The in-place stance rises
# VERTICALLY from the crouch xy (like every proven torque-mode stand), so
# nothing slides -- and all four swing legs become IK-feasible (the sprawl
# blocked RL/RR on the rear abduction limit).  Offline sweep at H=0.17,
# shift 35 mm: margin 31 mm for every leg, pre-lift ladder to 28 mm valid.
DEFAULT_SWING_LEG = "FR"
# H=0.175 frees abduction travel vs 0.17: total swing-foot raise grows
# from 36 to 44 mm for ALL four legs (validated per leg incl. the ladder).
DEFAULT_STAND_HEIGHT_M = 0.175
DEFAULT_SHIFT_M = 0.032         # ~26-31 mm planned FK margin (gate 15); all
                                # legs IK-clean through the full ladder
DEFAULT_LIFT_M = 0.014          # above the pre-lift; abd budget is tight
PRELIFT_START_M = 0.020         # corner sag eats ~12 mm of it
PRELIFT_RETRY_STEP_M = 0.005
PRELIFT_MAX_M = 0.030
CLEAR_RISE_MIN_M = 0.004        # ABSOLUTE stance-plane rise, not a fraction
DEFAULT_UNLOAD_TRIP_NM = 0.7    # airborne ~0.5 (leg gravity) vs loaded >= 0.9
MIN_LIFTOFF_MARGIN_M = 0.015
DEFAULT_TORQUE_TRIP_NM = 6.0    # measured peak 4.27 static / 4.50 crawl
DEFAULT_SPEED_TRIP_RAD_S = 1.0
T_STAND = 8.0                   # vertical rise, streamed
T_SHIFT = 2.5
T_UNLOAD = 0.8
T_LIFT = 1.5
T_RELOAD = 1.0
GATE_TIMEOUT_S = 6.0
SETTLE_TIMEOUT_S = 30.0

DEFAULT_CROUCH_DPS = 100.0      # motor-side speed caps
DEFAULT_STAND_DPS = 100.0
DEFAULT_STREAM_DPS = 250.0

# IMU lean watch during the 3-leg phases (judging only, never steering)
LEAN_ABORT_DEG = 6.0
LEAN_STREAK = 3


def _ik_to_target(leg, q_start, target, max_iter=100, tol=1.0e-5):
    """Damped-least-squares IK (crawl_dog5_hw law + early exit for the Pi)."""
    q = np.asarray(q_start, dtype=float).copy()
    for _ in range(max_iter):
        error = np.asarray(target) - dog5_kinematics.foot_position(leg, q)
        if float(np.linalg.norm(error)) < tol:
            break
        jacobian = dog5_kinematics.foot_jacobian(leg, q)
        q += 0.35 * jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 1.0e-5 * np.eye(3), error
        )
    return q


# Real-time budget: the motors latch "input lost" if any of them goes
# > 10 ms without a command.  At 250 Hz/motor x 12 the per-motor refresh
# is 4 ms, so control-side compute must stay well under ~6 ms per cycle.
# Solving 4 legs of IK in one sweep block costs 5-7 ms on the Pi (measured)
# and trips the watchdog -- so the IK is spread across the round-robin:
# one leg every third CAN slot (4 refines per 12-slot cycle), 2 damped
# Newton iterations each.  Targets move < 1.5 mm per sweep, so the ~1 mm
# tracking lag mid-stream is harmless and the refiner converges fully
# while the gates hold the targets still.  Measured: ~1.5 ms per cycle.
IK_SLOT_STRIDE = 3              # refine on every 3rd slot
IK_SLOT_MAX_ITER = 2
IK_SLOT_TOL_M = 1.0e-4


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return float(3.0 * u * u - 2.0 * u * u * u)


class ThreeLegPlan:
    """Kinematic foot-target planner: crouch -> stand-in-place -> shifted
    -> prelift -> lifted.

    Anchors are the FK feet of the recorded crouch, xy kept (stand-in-place
    convention: the feet NEVER slide -- the body rises vertically, then
    moves over the fixed feet).  Vertical offsets are metres, so the
    clear-gate retry can raise the pre-lift without re-scaling.
    """

    def __init__(self, swing_leg, shift_m, lift_m,
                 stand_height=DEFAULT_STAND_HEIGHT_M):
        if swing_leg not in LEGS:
            raise ValueError(f"swing leg must be one of {LEGS}")
        self.swing = swing_leg
        self.stance = [leg for leg in LEGS if leg != swing_leg]
        self.stand_height = float(stand_height)
        self.crouch_foot = {
            leg: dog5_kinematics.foot_position(leg, Q_CROUCH[self._sl(leg)])
            for leg in LEGS
        }
        # stand anchor: crouch xy, body height H (foot z = -H in trunk frame)
        self.anchor = {
            leg: np.array([self.crouch_foot[leg][0],
                           self.crouch_foot[leg][1],
                           -self.stand_height])
            for leg in LEGS
        }
        signs = np.sign(self.anchor[swing_leg][:2])
        self.body_shift = -float(shift_m) * signs / np.sqrt(2.0)
        self.lift_m = float(lift_m)

    @staticmethod
    def _sl(leg):
        i = LEGS.index(leg)
        return slice(3 * i, 3 * i + 3)

    def foot_targets(self, height_frac, body_frac, pre_m, lift_frac):
        body_xy = body_frac * self.body_shift
        targets = {}
        for leg in LEGS:
            crouch_z = self.crouch_foot[leg][2]
            target = self.anchor[leg].copy()
            target[2] = crouch_z + height_frac * (-self.stand_height - crouch_z)
            target[:2] -= body_xy
            if leg == self.swing:
                target[2] += pre_m + lift_frac * self.lift_m
            targets[leg] = target
        return targets

    def q_targets(self, warm_q, height_frac, body_frac, pre_m, lift_frac):
        targets = self.foot_targets(height_frac, body_frac, pre_m, lift_frac)
        q = np.asarray(warm_q, dtype=float).copy()
        for leg in LEGS:
            sl = self._sl(leg)
            q[sl] = _ik_to_target(leg, q[sl], targets[leg])
        return q

    def fk_margin(self, q_measured):
        """Stance-triangle margin from measured encoder FK, CoM at origin."""
        feet = [
            dog5_kinematics.foot_position(leg, q_measured[self._sl(leg)])[:2]
            for leg in self.stance
        ]
        return support_triangle_margin(feet)

    def plane_distance(self, q_measured):
        """Swing-foot signed distance to the measured stance plane."""
        feet = {
            leg: dog5_kinematics.foot_position(leg, q_measured[self._sl(leg)])
            for leg in LEGS
        }
        return signed_distance_to_stance_plane(
            feet[self.swing], [feet[leg] for leg in self.stance]
        )


# (stage name, kind).  "move" = joint-pose target with settle gate;
# "stream" = interpolated plan targets; "wait" = hold + operator ENTER;
# "gate" = hold targets until the condition passes; "dwell" = timed hold.
STAGES = [
    ("CROUCH", "move"),
    ("WAIT_CROUCH", "wait"),
    ("STAND", "stream"),
    ("STAND_SETTLE", "gate"),
    ("HOLD4", "wait"),
    ("SHIFT", "stream"),
    ("SHIFT_SETTLE", "gate"),
    ("WAIT_UNLOAD", "wait"),
    ("UNLOAD", "stream"),
    ("CLEAR_GATE", "gate"),
    ("WAIT_LIFT", "wait"),
    ("LIFT", "stream"),
    ("HOLD3", "wait"),
    ("LOWER", "stream"),
    ("RELOAD", "dwell"),
    ("RECENTER", "stream"),
    ("PARK", "move"),
    ("PARKED", "wait"),
]
STAGE_INDEX = {name: i for i, (name, _) in enumerate(STAGES)}
STREAM_T = {"STAND": T_STAND, "SHIFT": T_SHIFT, "UNLOAD": T_UNLOAD,
            "LIFT": T_LIFT, "LOWER": T_LIFT, "RECENTER": T_SHIFT}
INSTRUCTION = {
    "CROUCH": "moving to recorded crouch; wait, X stops",
    "WAIT_CROUCH": "inspect crouch; ENTER stands, X stops",
    "STAND": "rising vertically in place (feet stay put); wait, X stops",
    "STAND_SETTLE": "waiting for the stand to settle",
    "HOLD4": "4-leg stand; ENTER shifts, P parks, X stops",
    "SHIFT": "shifting body off the swing corner; wait",
    "SHIFT_SETTLE": "waiting for settle + FK margin >= 15 mm",
    "WAIT_UNLOAD": "shift gated OK; ENTER pre-lifts, X stops",
    "UNLOAD": "pre-lifting the swing foot; wait",
    "CLEAR_GATE": "waiting for plane rise + torque drop (auto-retries)",
    "WAIT_LIFT": "clear gate OK; ENTER lifts, X stops",
    "LIFT": "lifting the swing foot; wait",
    "HOLD3": "3-LEG HOLD; ENTER lowers, X stops",
    "LOWER": "lowering the swing foot; wait",
    "RELOAD": "reloading the foot; wait",
    "RECENTER": "recentering the body; wait",
    "PARK": "returning to recorded crouch; wait, X stops",
    "PARKED": "holding parked crouch; X stops",
}


class Stand3HoldSequence:
    """Encoder-only stage machine emitting joint targets once per sweep.

    Pure logic (no CAN, no time sources of its own) so the offline
    self-test can drive it through every path.
    """

    def __init__(self, now, plan, unload_trip=DEFAULT_UNLOAD_TRIP_NM):
        self.plan = plan
        self.unload_trip = float(unload_trip)
        self.stage_i = 0
        self.stage_started = float(now)
        self.settle_since = None
        self.wait_since = None
        self.q_cmd = Q_CROUCH.copy()
        # interpolation state: (height_frac, body_frac, pre_m, lift_frac)
        self.state = np.zeros(4)
        self._stream_from = np.zeros(4)
        self._stream_to = np.zeros(4)
        self.prelift_m = PRELIFT_START_M
        self.targets = None            # foot targets refined per CAN slot
        self.plane_baseline_m = None
        self.last_margin_m = np.nan
        self.last_rise_m = np.nan
        self.last_swing_tau_nm = np.nan
        self.aborted = None
        self.events = []
        self.torque_table_printed = False
        self.last_settle_detail = ""

    # -- helpers -----------------------------------------------------------
    @property
    def stage(self):
        return STAGES[self.stage_i][0]

    @property
    def kind(self):
        return STAGES[self.stage_i][1]

    def _goto(self, name, now, note=""):
        self.stage_i = STAGE_INDEX[name]
        self.stage_started = float(now)
        self.settle_since = None
        self.wait_since = None
        if self.kind == "stream":
            self._stream_from = self.state.copy()
            self._stream_to = self._stream_target(name)
        self.events.append((float(now), name, note))
        return f"{name}: {note}" if note else name

    def _stream_target(self, name):
        if name == "STAND":
            return np.array([1.0, 0.0, 0.0, 0.0])
        if name == "SHIFT":
            return np.array([1.0, 1.0, 0.0, 0.0])
        if name == "UNLOAD":
            return np.array([1.0, 1.0, self.prelift_m, 0.0])
        if name == "LIFT":
            return np.array([1.0, 1.0, self.prelift_m, 1.0])
        if name == "LOWER":
            return np.array([1.0, 1.0, self.prelift_m, 0.0])
        if name == "RECENTER":
            return np.array([1.0, 0.0, 0.0, 0.0])
        raise ValueError(name)

    def _abort(self, now, reason):
        self.aborted = reason
        note = f"ABORT: {reason}"
        if self.stage in ("LIFT", "HOLD3"):
            return self._goto("LOWER", now, note)
        return self._goto("RECENTER", now, note)

    def _settled(self, now, q_enc, qd, velocity_ready):
        errors = np.abs(np.asarray(q_enc) - self.q_cmd)
        worst = int(np.argmax(errors))
        error = float(errors[worst])
        speed = float(np.max(np.abs(qd))) if velocity_ready else np.inf
        self.last_settle_detail = (
            f"worst {JOINT_LABELS[worst]} err "
            f"{np.rad2deg(error):.1f} deg (tol "
            f"{np.rad2deg(posbase.POSITION_POSE_TOL):.1f}), max|qd| "
            f"{speed:.2f} rad/s (tol {posbase.POSITION_QD_TOL:.2f})"
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
        advance = {
            "WAIT_CROUCH": "STAND",
            "HOLD4": "SHIFT",
            "WAIT_UNLOAD": "UNLOAD",
            "WAIT_LIFT": "LIFT",
            "HOLD3": "LOWER",
        }[self.stage]
        if advance == "SHIFT":
            self.aborted = None
            self.prelift_m = PRELIFT_START_M
            self.plane_baseline_m = None
        return True, self._goto(advance, now, "ENTER accepted")

    def request_park(self, now, healthy):
        if self.stage != "HOLD4":
            return False, f"P only parks from HOLD4 (now {self.stage})."
        if not healthy:
            return False, "motor latch/fault present; motion refused."
        return True, self._goto("PARK", now, "P accepted")

    # -- per-sweep update --------------------------------------------------
    def sweep(self, now, q_enc, qd, velocity_ready, tau_measured):
        """One ~20.8 Hz update.  Returns (q_cmd_rad, event_or_None)."""
        event = None
        name, kind = STAGES[self.stage_i]

        if kind == "move":
            self.q_cmd = Q_CROUCH.copy()
            self.targets = None
            if self._settled(now, q_enc, qd, velocity_ready):
                nxt = {"CROUCH": "WAIT_CROUCH", "PARK": "PARKED"}[name]
                event = self._goto(nxt, now, "settled")
                self.wait_since = float(now)
            elif now - self.stage_started > SETTLE_TIMEOUT_S:
                raise RuntimeError(
                    f"{name} did not settle in {SETTLE_TIMEOUT_S:.0f} s: "
                    f"{self.last_settle_detail}"
                )

        elif kind == "stream":
            u = _smoothstep((now - self.stage_started) / STREAM_T[name])
            self.state = self._stream_from + u * (
                self._stream_to - self._stream_from
            )
            # Only the (cheap) targets update here; the IK toward them is
            # spread across CAN slots by refine_leg().
            self.targets = self.plan.foot_targets(*self.state)
            if now - self.stage_started >= STREAM_T[name]:
                self.state = self._stream_to.copy()
                nxt = {"STAND": "STAND_SETTLE", "SHIFT": "SHIFT_SETTLE",
                       "UNLOAD": "CLEAR_GATE", "LIFT": "HOLD3",
                       "LOWER": "RELOAD", "RECENTER": "HOLD4"}[name]
                event = self._goto(nxt, now, "stream complete")
                if nxt in ("HOLD3", "HOLD4"):
                    self.wait_since = float(now)
                if nxt == "HOLD4" and self.aborted:
                    event += f"  [batch ABORTED: {self.aborted}]"

        elif name == "STAND_SETTLE":
            if self._settled(now, q_enc, qd, velocity_ready):
                event = self._goto("HOLD4", now, "stand settled")
                self.wait_since = float(now)
            elif now - self.stage_started > SETTLE_TIMEOUT_S:
                raise RuntimeError(
                    f"STAND did not settle in {SETTLE_TIMEOUT_S:.0f} s: "
                    f"{self.last_settle_detail}"
                )

        elif name == "SHIFT_SETTLE":
            self.last_margin_m = self.plan.fk_margin(q_enc)
            if self._settled(now, q_enc, qd, velocity_ready):
                if self.last_margin_m >= MIN_LIFTOFF_MARGIN_M:
                    self.plane_baseline_m = self.plan.plane_distance(q_enc)
                    event = self._goto(
                        "WAIT_UNLOAD", now,
                        f"FK margin {1000 * self.last_margin_m:.1f} mm, "
                        f"plane baseline "
                        f"{1000 * self.plane_baseline_m:+.1f} mm",
                    )
                    self.wait_since = float(now)
                elif now - self.stage_started > GATE_TIMEOUT_S:
                    event = self._abort(
                        now,
                        f"liftoff margin {1000 * self.last_margin_m:.1f} mm "
                        f"< {1000 * MIN_LIFTOFF_MARGIN_M:.0f} mm",
                    )
            elif now - self.stage_started > SETTLE_TIMEOUT_S:
                event = self._abort(now, "SHIFT did not settle")

        elif name == "CLEAR_GATE":
            sl = self.plan._sl(self.plan.swing)
            self.last_swing_tau_nm = float(
                np.max(np.abs(np.asarray(tau_measured)[sl][1:]))
            )
            self.last_rise_m = (
                self.plan.plane_distance(q_enc) - self.plane_baseline_m
            )
            clear = (
                self.last_rise_m >= CLEAR_RISE_MIN_M
                and self.last_swing_tau_nm <= self.unload_trip
            )
            if self._settled(now, q_enc, qd, velocity_ready) and clear:
                event = self._goto(
                    "WAIT_LIFT", now,
                    f"plane rise {1000 * self.last_rise_m:+.1f} mm, swing "
                    f"|tau| {self.last_swing_tau_nm:.2f} N*m at pre-lift "
                    f"{1000 * self.prelift_m:.0f} mm",
                )
                self.wait_since = float(now)
            elif now - self.stage_started > GATE_TIMEOUT_S:
                detail = (
                    f"rise {1000 * self.last_rise_m:+.1f} mm (need "
                    f"{1000 * CLEAR_RISE_MIN_M:.0f}), swing |tau| "
                    f"{self.last_swing_tau_nm:.2f} N*m (max "
                    f"{self.unload_trip:.2f})"
                )
                if (self.prelift_m + PRELIFT_RETRY_STEP_M
                        <= PRELIFT_MAX_M + 1.0e-9):
                    self.prelift_m = min(
                        self.prelift_m + PRELIFT_RETRY_STEP_M, PRELIFT_MAX_M
                    )
                    event = self._goto(
                        "UNLOAD", now,
                        f"clear gate timeout ({detail}); raising pre-lift "
                        f"to {1000 * self.prelift_m:.0f} mm",
                    )
                else:
                    event = self._abort(
                        now,
                        f"clear gate exhausted at pre-lift "
                        f"{1000 * self.prelift_m:.0f} mm: {detail}",
                    )

        elif kind == "dwell":  # RELOAD
            if now - self.stage_started >= T_RELOAD:
                event = self._goto("RECENTER", now, "reloaded")

        elif kind == "wait":
            if self.wait_since is None:
                self.wait_since = float(now)
            if self.stage in ("SHIFT_SETTLE",):
                pass
            if self.stage in ("HOLD3",):
                self.last_margin_m = self.plan.fk_margin(q_enc)

        return self.q_cmd, event

    def refine_leg(self, slot_counter):
        """A few warm-started IK iterations for ONE leg of q_cmd.

        Called once per CAN slot from the run loop, cycling through the
        legs, so the IK cost never blocks the round-robin long enough to
        trip the 10 ms motor input watchdog.
        """
        if self.targets is None:
            return
        leg = LEGS[slot_counter % len(LEGS)]
        sl = self.plan._sl(leg)
        self.q_cmd[sl] = _ik_to_target(
            leg, self.q_cmd[sl], self.targets[leg],
            max_iter=IK_SLOT_MAX_ITER, tol=IK_SLOT_TOL_M,
        )

    def speed_cap_dps(self, caps):
        if self.stage in ("CROUCH", "WAIT_CROUCH", "PARK", "PARKED"):
            return caps["crouch"]
        if self.stage == "STAND":
            return caps["stand"]
        return caps["stream"]


def _print_torque_table(measured_torque):
    """SIM_APPROACH_HW #7-C: mis-calibration shows as load-share asymmetry."""
    print("[HOLD4] measured torques (mis-zero signature = large "
          "left/right or diagonal asymmetry):")
    for i, leg in enumerate(LEGS):
        abd, pitch, knee = measured_torque[3 * i:3 * i + 3]
        print(f"        {leg}: abd={abd:+5.2f} pitch={pitch:+5.2f} "
              f"knee={knee:+5.2f} N*m")


def validate_configuration(plan):
    posbase.validate_configuration()
    low, high = base.soft_limits()
    q = Q_CROUCH.copy()
    worst_err = 0.0
    path = (
        [(h, 0.0, 0.0, 0.0) for h in np.linspace(0.0, 1.0, 11)]
        + [(1.0, f, 0.0, 0.0) for f in np.linspace(0.0, 1.0, 11)]
        + [(1.0, 1.0, m, 0.0) for m in np.linspace(0.0, PRELIFT_MAX_M, 5)]
        + [(1.0, 1.0, PRELIFT_MAX_M, f) for f in np.linspace(0.0, 1.0, 6)]
    )
    for height, body, pre, lift in path:
        q = plan.q_targets(q, height, body, pre, lift)
        targets = plan.foot_targets(height, body, pre, lift)
        for leg in LEGS:
            sl = plan._sl(leg)
            err = float(np.linalg.norm(
                dog5_kinematics.foot_position(leg, q[sl]) - targets[leg]
            ))
            worst_err = max(worst_err, err)
            if np.any(q[sl] < low[sl]) or np.any(q[sl] > high[sl]):
                raise ValueError(
                    f"{leg} IK outside soft limits at (h={height:.2f},"
                    f"{body:.2f},{1000 * pre:.0f}mm,{lift:.2f}): {q[sl]}"
                )
    if worst_err > 1.0e-4:
        raise ValueError(f"IK residual {worst_err:.2e} m > 1e-4")

    planned_margin = support_triangle_margin(
        [plan.anchor[leg][:2] - plan.body_shift for leg in plan.stance]
    )
    # Reserve = gate + 10 mm: a margin that survives 10 mm of foot-position
    # error is backlash-proof (SIM_APPROACH_HW section 4).
    reserve = MIN_LIFTOFF_MARGIN_M + 0.010
    if planned_margin < reserve:
        raise ValueError(
            f"planned liftoff margin {1000 * planned_margin:.1f} mm < "
            f"{1000 * reserve:.0f} mm reserve"
        )

    # Static stance torques at the shifted stand pose (barycentric shares).
    q_shifted = plan.q_targets(Q_CROUCH.copy(), 1.0, 1.0, 0.0, 0.0)
    weight = base.DOG5_MASS_KG * 9.81
    stance_xy = np.array([plan.anchor[leg][:2] for leg in plan.stance])
    A = np.vstack([np.ones(3), stance_xy.T])
    forces = np.linalg.solve(A, np.array([weight, *(weight * plan.body_shift)]))
    worst_tau = 0.0
    for leg, fz in zip(plan.stance, forces):
        sl = plan._sl(leg)
        tau = dog5_kinematics.foot_jacobian(leg, q_shifted[sl]).T \
            @ [0.0, 0.0, fz]
        worst_tau = max(worst_tau, float(np.max(np.abs(tau))))
    return planned_margin, forces, worst_tau


def run_hardware(args):
    plan = ThreeLegPlan(args.swing_leg, args.shift, args.lift,
                        stand_height=args.stand_height)
    planned_margin, forces, worst_tau = validate_configuration(plan)
    if args.torque_trip <= worst_tau:
        raise RuntimeError(
            f"--torque-trip {args.torque_trip:.1f} N*m is below the "
            f"~{worst_tau:.2f} N*m static 3-leg requirement -- it WILL fire"
        )
    caps = {"crouch": args.crouch_dps, "stand": args.stand_dps,
            "stream": args.stream_dps}
    unwrap = [base.CalibratedEncoderUnwrap() for _ in base.HARDWARE_JOINTS]
    safety = base.SafetyGate(
        tau_cap=1.0, qd_estop=args.qd_estop, qd_estop_hard=args.qd_estop_hard
    )

    print(f"[stand3] STAND-IN-PLACE: feet keep their crouch xy and never "
          f"slide; the body rises to H={args.stand_height:.2f}m over "
          f"{T_STAND:.0f}s, then moves over the fixed feet")
    print(f"[stand3] swing={args.swing_leg} shift={1000 * args.shift:.0f}mm "
          f"lift={1000 * args.lift:.0f}mm; planned liftoff FK margin "
          f"{1000 * planned_margin:.1f}mm")
    print(f"[stand3] static stance forces {np.round(forces, 1)} N, worst "
          f"joint torque ~{worst_tau:.2f} N*m (trip {args.torque_trip:.1f})")
    print(f"[stand3] pre-lift {1000 * PRELIFT_START_M:.0f}mm, clear gate: "
          f"plane rise >= {1000 * CLEAR_RISE_MIN_M:.0f}mm ABS + swing |tau| "
          f"<= {args.unload_trip:.2f} N*m; timeout +"
          f"{1000 * PRELIFT_RETRY_STEP_M:.0f}mm retries to "
          f"{1000 * PRELIFT_MAX_M:.0f}mm")
    print("[stand3] expect plane rise ~ (pre-lift - 12mm): corner unload "
          "sags the trunk; that is not a stuck leg")
    print("[stand3] ROBOT MUST REMAIN MECHANICALLY SUPPORTED UNTIL PROVEN.")

    imu = None if args.no_imu else inplace._open_imu()
    if imu is not None:
        print(f"[imu] tip-over abort {inplace.TIPOVER_ABORT_DEG:.0f} deg; "
              f"lean watch {LEAN_ABORT_DEG:.0f} deg during 3-leg phases "
              "(judging only -- control stays encoder-only)")

    key = base.KeyPoller()
    try:
        with base.motorbus.MotorBus(MOTOR_IDS, dirs=MOTOR_DIRECTIONS) as mb:
            armed = False
            stop_reason = None
            try:
                print(f"[ARM] arming for up to {posbase.ARM_TIMEOUT_S:.0f}s; "
                      "keep supported, Ctrl-C stops.")
                if not mb.arm(rate_hz=base.CONTROL_HZ,
                              timeout_s=posbase.ARM_TIMEOUT_S, verbose=False):
                    raise RuntimeError(
                        "arming timed out: " + posbase.arm_failure_summary(mb)
                    )
                armed = True

                start_q = posbase.zero_torque_preflight(mb, key, unwrap)
                now = time.perf_counter()
                sequence = Stand3HoldSequence(
                    now, plan, unload_trip=args.unload_trip
                )
                safety.start(now, start_q)

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
                        if (sequence.stage == "HOLD4"
                                and not sequence.torque_table_printed):
                            _print_torque_table(measured_torque)
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

                        # IMU: judge, never steer.
                        imu_sample = None
                        if imu is not None:
                            try:
                                imu_sample = imu.sample()
                            except Exception:
                                imu_sample = None
                            fresh = (
                                imu_sample is not None
                                and imu_sample.age_s <= inplace.POSTURE_STALE_S
                            )
                            if fresh:
                                roll = imu_sample.roll_deg
                                pitch = imu_sample.pitch_deg
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
                                if sequence.stage in ("LIFT", "HOLD3"):
                                    if lean_baseline is None:
                                        lean_baseline = (roll, pitch)
                                        lean_streak = 0
                                    else:
                                        d_r = roll - lean_baseline[0]
                                        d_p = pitch - lean_baseline[1]
                                        if max(abs(d_r), abs(d_p)) \
                                                > LEAN_ABORT_DEG:
                                            lean_streak += 1
                                            if lean_streak >= LEAN_STREAK:
                                                print("[stage] " +
                                                      sequence._abort(
                                                          now,
                                                          f"IMU lean d_roll="
                                                          f"{d_r:+.1f} d_pitch="
                                                          f"{d_p:+.1f} deg"))
                                                lean_baseline = None
                                                lean_streak = 0
                                        else:
                                            lean_streak = 0
                                else:
                                    lean_baseline = None
                                    lean_streak = 0

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
                            sl = plan._sl(plan.swing)
                            swing_tau = float(
                                np.max(np.abs(measured_torque[sl][1:]))
                            )
                            rise = ("-" if sequence.plane_baseline_m is None
                                    else f"{1000 * (plan.plane_distance(q) - sequence.plane_baseline_m):+.1f}mm")
                            margin = ("-" if not np.isfinite(
                                sequence.last_margin_m)
                                else f"{1000 * sequence.last_margin_m:.1f}mm")
                            imu_text = ""
                            if imu_sample is not None:
                                imu_text = (
                                    f" rp=({imu_sample.roll_deg:+.2f},"
                                    f"{imu_sample.pitch_deg:+.2f})deg"
                                )
                            settle_text = ""
                            if (sequence.kind in ("move", "gate")
                                    and sequence.last_settle_detail):
                                settle_text = (
                                    f" | {sequence.last_settle_detail}"
                                )
                            print(
                                f"[stand3] {sequence.stage:<12} "
                                f"{INSTRUCTION[sequence.stage]} | "
                                f"margin={margin} rise={rise} "
                                f"prelift={1000 * sequence.prelift_m:.0f}mm "
                                f"swing|tau|={swing_tau:.2f} "
                                f"max|tau|={np.max(np.abs(measured_torque)):.2f}N*m "
                                f"max|qd|={np.max(np.abs(qd_encoder)):.2f} "
                                f"blk={worst_block_ms:.1f}ms"
                                f"{imu_text}{settle_text}",
                                flush=True,
                            )
                            last_print = now
                            worst_block_ms = 0.0
                        worst_block_ms = max(
                            worst_block_ms,
                            1000.0 * (time.perf_counter() - now),
                        )

                    # Spread IK: one leg every third slot, never a block.
                    if index % IK_SLOT_STRIDE == 0:
                        sequence.refine_leg(index // IK_SLOT_STRIDE)

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
                    mb.pace(deadline)
                    deadline += slot

            except KeyboardInterrupt as exc:
                stop_reason = str(exc) or "KeyboardInterrupt"
            except Exception as exc:
                stop_reason = f"error: {exc}"
                raise
            finally:
                if armed:
                    print(f"[STOP] {stop_reason or 'aborted'}")
                    mb.stop_all()
    finally:
        key.close()
        if imu is not None:
            try:
                imu.stop()
            except Exception:
                pass
    return 0


def offline_self_test(args):
    plan = ThreeLegPlan(args.swing_leg, args.shift, args.lift,
                        stand_height=args.stand_height)
    planned_margin, forces, worst_tau = validate_configuration(plan)
    print(f"IK path reachable in-limits up to pre-lift "
          f"{1000 * PRELIFT_MAX_M:.0f} mm")
    print(f"planned liftoff FK margin {1000 * planned_margin:.1f} mm "
          f"(gate {1000 * MIN_LIFTOFF_MARGIN_M:.0f} mm)")
    print(f"static stance forces {np.round(forces, 1)} N, worst torque "
          f"{worst_tau:.2f} N*m (trip {args.torque_trip:.1f}, driver cap 9)")
    assert worst_tau < args.torque_trip < 9.0

    # Drive the sequence machine through the full nominal path with a
    # synthetic plant that instantly tracks commands.
    def synth(seq, tau=None, q=None):
        qq = seq.q_cmd.copy() if q is None else q
        tt = np.zeros(N_JOINTS) if tau is None else tau
        return qq, np.zeros(N_JOINTS), True, tt

    now = 0.0
    seq = Stand3HoldSequence(now, plan, unload_trip=args.unload_trip)

    def advance_until(stage, limit_s=60.0, tau=None, q_fn=None):
        nonlocal now
        deadline = now + limit_s
        while seq.stage != stage:
            now += 0.048  # ~20.8 Hz
            q = q_fn(seq) if q_fn else None
            seq.sweep(now, *synth(seq, tau=tau, q=q))
            for k in range(4):  # the per-slot IK refiner, as the loop runs it
                seq.refine_leg(k)
            assert now < deadline, (
                f"never reached {stage}; stuck in {seq.stage}"
            )

    def enter():
        nonlocal now
        now += base.WAIT_DWELL_S + 0.1
        ok, msg = seq.request_next(now, seq.q_cmd, np.zeros(N_JOINTS), True)
        assert ok, msg

    advance_until("WAIT_CROUCH")
    enter()
    advance_until("HOLD4")
    enter()  # SHIFT
    advance_until("WAIT_UNLOAD")
    assert seq.last_margin_m >= MIN_LIFTOFF_MARGIN_M
    assert seq.plane_baseline_m is not None
    enter()  # UNLOAD
    advance_until("WAIT_LIFT")   # synthetic plant tracks -> rise == prelift
    assert seq.last_rise_m >= CLEAR_RISE_MIN_M
    enter()  # LIFT
    advance_until("HOLD3")
    enter()  # LOWER
    advance_until("HOLD4")
    assert seq.aborted is None
    print("nominal sequence: CROUCH -> ... -> HOLD3 -> ... -> HOLD4 OK")

    # Clear-gate retry ladder: a swing leg that always measures loaded must
    # raise the pre-lift to the cap and then abort gracefully to RECENTER.
    seq2 = Stand3HoldSequence(0.0, plan, unload_trip=args.unload_trip)
    now = 0.0

    def advance2(stage, tau=None, limit_s=120.0):
        nonlocal now
        deadline = now + limit_s
        while seq2.stage != stage:
            now += 0.048
            qq = seq2.q_cmd.copy()
            tt = np.zeros(N_JOINTS) if tau is None else tau
            seq2.sweep(now, qq, np.zeros(N_JOINTS), True, tt)
            for k in range(4):
                seq2.refine_leg(k)
            assert now < deadline, f"stuck in {seq2.stage} not {stage}"

    loaded = np.zeros(N_JOINTS)
    sl = plan._sl(plan.swing)
    loaded[sl] = [0.0, 2.0, 2.0]  # swing pitch/knee clearly loaded
    advance2("WAIT_CROUCH")
    now += base.WAIT_DWELL_S + 0.1
    seq2.request_next(now, seq2.q_cmd, np.zeros(N_JOINTS), True)
    advance2("HOLD4")
    now += base.WAIT_DWELL_S + 0.1
    seq2.request_next(now, seq2.q_cmd, np.zeros(N_JOINTS), True)
    advance2("WAIT_UNLOAD")
    now += base.WAIT_DWELL_S + 0.1
    seq2.request_next(now, seq2.q_cmd, np.zeros(N_JOINTS), True)
    advance2("HOLD4", tau=loaded)   # retries exhaust, abort recenters
    assert seq2.aborted is not None and "clear gate" in seq2.aborted
    assert np.isclose(seq2.prelift_m, PRELIFT_MAX_M)
    print(f"clear-gate ladder: retried to {1000 * seq2.prelift_m:.0f} mm, "
          f"then graceful abort ({seq2.aborted})")

    # Abort from HOLD3 routes through LOWER (foot down before recenter).
    seq3 = Stand3HoldSequence(0.0, plan)
    seq3.stage_i = STAGE_INDEX["HOLD3"]
    seq3.state = np.array([1.0, 1.0, PRELIFT_START_M, 1.0])
    seq3._abort(1.0, "test")
    assert seq3.stage == "LOWER"
    print("HOLD3 abort routes through LOWER")

    velocity = posbase.EncoderVelocity()
    for i in range(7):
        v = velocity.update(Q_STAND, 0.05 * i)
    assert velocity.ready and np.allclose(v, 0.0)
    print("stand3_hold_hw offline self-test PASS")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--swing-leg", default=DEFAULT_SWING_LEG, choices=LEGS)
    parser.add_argument("--shift", type=float, default=DEFAULT_SHIFT_M)
    parser.add_argument("--lift", type=float, default=DEFAULT_LIFT_M)
    parser.add_argument("--stand-height", type=float,
                        default=DEFAULT_STAND_HEIGHT_M,
                        help="in-place stand height, m (offline-validated "
                             f"envelope centred on {DEFAULT_STAND_HEIGHT_M})")
    parser.add_argument("--torque-trip", type=float,
                        default=DEFAULT_TORQUE_TRIP_NM,
                        help="measured-torque trip N*m (sim: 3-leg stance "
                             "needs ~4.3-4.5; the old 2.0 default WILL fire)")
    parser.add_argument("--speed-trip", type=float,
                        default=DEFAULT_SPEED_TRIP_RAD_S)
    parser.add_argument("--unload-trip", type=float,
                        default=DEFAULT_UNLOAD_TRIP_NM,
                        help="clear-gate swing pitch/knee |tau| N*m "
                             "(airborne ~0.5 leg gravity vs loaded >= 0.9)")
    parser.add_argument("--crouch-dps", type=float, default=DEFAULT_CROUCH_DPS)
    parser.add_argument("--stand-dps", type=float, default=DEFAULT_STAND_DPS)
    parser.add_argument("--stream-dps", type=float, default=DEFAULT_STREAM_DPS)
    parser.add_argument("--qd-estop", type=float, default=base.QD_ESTOP)
    parser.add_argument("--qd-estop-hard", type=float,
                        default=base.QD_ESTOP_HARD)
    parser.add_argument("--no-imu", action="store_true",
                        help="skip the IMU display / tip-over / lean watch")
    args = parser.parse_args()

    if args.self_test:
        return offline_self_test(args)
    try:
        return run_hardware(args)
    except (RuntimeError, ValueError) as exc:
        print(f"[stand3] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
