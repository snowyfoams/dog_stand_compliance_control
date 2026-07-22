#!/usr/bin/env python3
"""Manual CoM-swing / leg-lift jog tool for DOG5.

Nothing here is on a timer.  You swing the CoM off one corner, watch the
measured foot load yourself for as long as you like, and only lift when YOU
decide the leg is unloaded.

After the usual stand (CROUCH -> ENTER -> STAND -> HOLD), ENTER arms manual
mode and the robot holds flat on four feet.  Then:

    1 = RR   2 = FL   3 = FR   4 = RL
        swing the CoM away from that corner and HOLD there indefinitely.
        The leg's mg/4 support feedforward fades out at the same time, so
        its measured load should fall toward zero.  Watch the [load] line.

    + / -   grow / shrink the CoM swing by 2 mm (live, rate-limited).  Use
            this to find the corner's swing limit: keep pressing + until
            the leg's fz reaches ~0 or the safe cap is hit.
    A       LIFT the selected foot off the ground (only once the CoM swing
            has settled; refused if the three-foot support margin is below
            the 15 mm floor -- press + first).  Holds a 3-leg stand
            indefinitely.
    D       put the foot back Down, reload it, and return to neutral.
            From the CoM swing (foot still down) it just reloads/recenters.
    T       tare the fz display (press with ALL feet off the ground).
    P       park (only from the four-foot neutral).
    X       stop.

The lift is deliberately NOT gated on the measured load -- that judgement is
yours.  What IS enforced: the support-margin floor before liftoff, the
support-triangle e-stop while airborne, and every standard e-stop
(joint-limit, overspeed, temperature, CAN-miss, motor fault).  While the
foot is airborne a still-loaded reading is reported as a WARNING, not an
abort, so you can see for yourself whether the foot really left the ground.

    python manual_swing_hw.py --self-test
    python manual_swing_hw.py --tau-max 3.0
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import dog5_kinematics
import stand_dog5_hw as base
import stand_dog5_inplace_hw as inplace
import stand_dog5_recorded_hw as recorded
import crawl_dog5_hw as crawl


LEGS = base.LEGS
MOTOR_IDS = base.MOTOR_IDS
MOTOR_DIRECTIONS = base.MOTOR_DIRECTIONS
N_JOINTS = base.N_JOINTS
POSITION_TARGET_DEG = recorded.POSITION_TARGET_DEG
T_PARK = recorded.T_PARK
T_STAND = recorded.T_STAND

KEY_TO_LEG = {"1": "RR", "2": "FL", "3": "FR", "4": "RL"}
KEY_HINT = "1=RR 2=FL 3=FR 4=RL"

# Jog states.  Every transition is either a key press or a follower
# reaching its target -- there are no timeouts and no gates that abort.
NEUTRAL = "NEUTRAL"
SWING_COM = "SWING_COM"
AIRBORNE = "AIRBORNE"
LOWERING = "LOWERING"
RELOADING = "RELOADING"
RECENTERING = "RECENTERING"

# Rate-limited followers: the state machine only ever sets a target, and
# these rates decide how fast the robot actually gets there.  Slow enough
# to watch, fast enough not to be tedious.
SHIFT_RATE_M_S = 0.015
UNLOAD_RATE_PER_S = 0.5
LIFT_RATE_M_S = 0.020
# One + / - press.
SHIFT_STEP_M = 0.002
# Hard ceiling on the manual swing regardless of what the safe cap says.
MAX_MANUAL_SHIFT_M = 0.045
# Keep the linearized joint excursion this far inside the soft limits.
LIMIT_MARGIN_RAD = 0.02
# Follower convergence (a target is "reached" within this).
SHIFT_EPS_M = 1.0e-4
FRACTION_EPS = 1.0e-3

DEFAULT_TAU_MAX = base.STAGED_TAU_MAX
DEFAULT_STAND_HEIGHT_M = 0.15
DEFAULT_SHIFT_DISTANCE_M = 0.026
# 20 mm, not more: at the 0.15 m stand height the LIFTED leg's abduction sits
# ~0.02 deg inside its +-100.27 deg soft limit, so a 25 mm lift fails
# validation outright.  Raise --stand-height for more lift headroom.
DEFAULT_SWING_HEIGHT_M = 0.020

MIN_LIFTOFF_MARGIN_M = crawl.MIN_LIFTOFF_MARGIN_M
SUPPORT_MARGIN_ESTOP_M = crawl.SUPPORT_MARGIN_ESTOP_M
SUPPORT_MARGIN_ESTOP_STREAK = crawl.SUPPORT_MARGIN_ESTOP_STREAK
# Airborne foot still measuring this much ground torque = it did not leave
# the ground.  Reported, never acted on (this tool exists to find out).
DRAG_WARN_TAU_NM = crawl.SWING_DRAG_TAU_TRIP_NM
DRAG_WARN_PERIOD_S = 2.0

LOAD_STATUS_HZ = 4.0
HEALTH_STATUS_HZ = 1.0


def _move_toward(value, target, step):
    if value < target:
        return min(target, value + step), min(step, target - value)
    if value > target:
        return max(target, value - step), -min(step, value - target)
    return target, 0.0


def shift_direction_cap(q, world_foot_xy, swing_leg, base_body_xy, direction):
    """Furthest body shift along ``direction`` (m) that stays safe.

    Two bounds, both cheap enough for the control loop (no IK).  The
    binding one is the linearized joint excursion (dq = J^-1 dx per leg,
    accurate over a few cm), which must stay inside the soft limits with
    margin -- at this stance width the abduction joints run out first.

    The support margin is the second, and it is NOT a bound on getting
    started: with all four feet down the body origin sits ON the diagonal
    edge of any three-foot triangle, so the margin begins at ~0 and GROWS
    with the shift -- that is the whole point of swinging the CoM.  It only
    bounds the far side, where shifting further would leave the triangle
    again.  Being below the floor near zero shift is normal and is what the
    liftoff check (not this cap) is there to catch.
    """
    q = np.asarray(q, dtype=float)
    direction = np.asarray(direction, dtype=float)
    low, high = base.soft_limits()
    cap = MAX_MANUAL_SHIFT_M
    # Every foot target moves by -direction per metre of body shift.
    move_dir = np.array([-direction[0], -direction[1], 0.0])
    for index, leg in enumerate(LEGS):
        section = slice(3 * index, 3 * index + 3)
        jacobian = dog5_kinematics.foot_jacobian(leg, q[section])
        if (
            np.linalg.svd(jacobian, compute_uv=False)[-1]
            < recorded.MIN_JACOBIAN_SINGULAR
        ):
            return 0.0
        dq_per_m = np.linalg.solve(jacobian, move_dir)
        for joint in range(3):
            rate = float(dq_per_m[joint])
            if rate > 1.0e-9:
                room = (high[section][joint] - LIMIT_MARGIN_RAD) - q[section][joint]
                cap = min(cap, max(0.0, room / rate))
            elif rate < -1.0e-9:
                room = q[section][joint] - (low[section][joint] + LIMIT_MARGIN_RAD)
                cap = min(cap, max(0.0, room / -rate))
    # Far-side margin bound: the margin is concave in the shift, so keep the
    # LAST probe that still clears the floor.  If nothing clears it, the
    # joint cap stands and the liftoff check reports why A is refused.
    stance_pts = [world_foot_xy[name] for name in LEGS if name != swing_leg]
    base_body_xy = np.asarray(base_body_xy, dtype=float)
    last_clear = None
    probe = 0.0
    while probe <= cap + 1.0e-9:
        margin = crawl.support_triangle_margin(
            stance_pts, base_body_xy + probe * direction
        )
        if margin >= MIN_LIFTOFF_MARGIN_M:
            last_clear = probe
        probe += SHIFT_STEP_M
    if last_clear is not None:
        cap = min(cap, last_clear)
    return max(0.0, cap)


def best_shift_direction(q, world_foot_xy, swing_leg, base_body_xy):
    """Pick the swing direction that can actually unload ``swing_leg``.

    Straight away from the foot is the obvious choice and the wrong one:
    the stance is only ~12 cm wide, so a lateral shift saturates the
    abduction joints within a few mm.  Score every candidate by how far it
    can safely go TIMES how much of it points along the inward normal of
    the support-triangle edge that limits the margin -- the same trade the
    crawl's force-feedback growth makes, which is why a mostly longitudinal
    lean usually wins.

    Returns ``(unit_direction, cap_m)``.
    """
    body_xy = np.asarray(base_body_xy, dtype=float)
    stance_pts = [world_foot_xy[name] for name in LEGS if name != swing_leg]
    _, normal = crawl.adaptive_body_shift(
        world_foot_xy[swing_leg], stance_pts, body_xy, 0.0
    )
    best_dir, best_cap, best_score = normal, 0.0, -np.inf
    for degrees in range(-75, 76, 15):
        angle = np.deg2rad(degrees)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        candidate = rotation @ normal
        cap = shift_direction_cap(
            q, world_foot_xy, swing_leg, body_xy, candidate
        )
        score = cap * float(candidate @ normal)
        if score > best_score:
            best_dir, best_cap, best_score = candidate, cap, score
    return best_dir, best_cap


class ManualSwingPlanner:
    """Key-driven CoM swing and leg lift.

    Implements the one method the Cartesian controller needs --
    ``target(leg, now) -> (position, velocity, support_share, trim_scale)``
    -- so ``crawl.CrawlController`` can drive it unchanged.  All motion is
    produced by three rate-limited followers (body shift, support-share
    unload, foot lift); the state machine only sets their targets.
    """

    def __init__(self, controller, shift_distance, swing_height):
        self.controller = controller
        self.default_shift_m = float(shift_distance)
        self.swing_height = float(swing_height)
        self.active = False
        self.state = NEUTRAL
        self.selected_leg = None
        self.world_foot_xy = {}
        self.nominal_z = {}
        self.shift_m = 0.0
        self.shift_target_m = 0.0
        self.shift_cap_m = 0.0
        self.shift_normal = np.zeros(2)
        self.unload_u = 0.0
        self.unload_target = 0.0
        self.lift_h = 0.0
        self.lift_target = 0.0
        self.body_velocity_xy = np.zeros(2)
        self.lift_velocity_z = 0.0
        self.last_tau_ext_nm = np.nan
        self._last_update = None
        self._drag_warned_at = -np.inf

    # ---- geometry -------------------------------------------------------
    @property
    def body_xy(self):
        return self.shift_m * self.shift_normal

    @property
    def airborne(self):
        return self.active and self.lift_h > FRACTION_EPS

    @property
    def three_leg_support(self):
        return self.active and self.selected_leg is not None and (
            self.unload_u > 0.5 or self.airborne
        )

    @property
    def trim_stance_legs(self):
        if self.active and self.selected_leg is not None and self.unload_u > 0.0:
            return set(LEGS) - {self.selected_leg}
        return set(LEGS)

    @property
    def context(self):
        return (
            f"state={self.state} leg={self.selected_leg or '--'} "
            f"shift={1000.0 * self.shift_m:.1f}mm"
        )

    def start(self, now):
        """Snapshot the settled stand as the neutral this tool jogs around."""
        self.world_foot_xy = {
            leg: np.asarray(
                self.controller.final_foot[leg][:2], dtype=float
            ).copy()
            for leg in LEGS
        }
        self.nominal_z = {
            leg: float(self.controller.final_foot[leg][2]) for leg in LEGS
        }
        self.active = True
        self.state = NEUTRAL
        self.selected_leg = None
        self.shift_m = self.shift_target_m = self.shift_cap_m = 0.0
        self.shift_normal = np.zeros(2)
        self.unload_u = self.unload_target = 0.0
        self.lift_h = self.lift_target = 0.0
        self.body_velocity_xy = np.zeros(2)
        self.lift_velocity_z = 0.0
        self._last_update = float(now)

    def actual_foot_positions(self, q):
        q = np.asarray(q, dtype=float)
        return {
            leg: dog5_kinematics.foot_position(leg, q[3 * i : 3 * i + 3])
            for i, leg in enumerate(LEGS)
        }

    def support_margin(self, q):
        """Margin of the body origin inside the CURRENT support polygon --
        the three planted feet once a leg is selected, otherwise all four."""
        feet = self.actual_foot_positions(q)
        if self.selected_leg is None:
            points = np.asarray([feet[leg][:2] for leg in LEGS], dtype=float)
            return _polygon_margin(points)
        stance = [feet[leg][:2] for leg in LEGS if leg != self.selected_leg]
        return crawl.support_triangle_margin(stance)

    def liftoff_margin(self, q):
        """Margin of the three-foot triangle that WOULD support a lift."""
        feet = self.actual_foot_positions(q)
        leg = self.selected_leg
        if leg is None:
            return np.nan
        return crawl.support_triangle_margin(
            [feet[name][:2] for name in LEGS if name != leg]
        )

    def swing_leg_tau_ext(self, q, measured_tau, tau_offset=None):
        """Gravity-compensated pitch/knee |tau| of the selected leg.

        The per-tick drag check.  Deliberately does NOT go through
        foot_load_map: that costs ~1.6 ms (twelve 3x3 SVDs and solves) and
        would eat most of the 4 ms control tick, putting the 10 ms CAN
        watchdog at risk.  This is one leg's gravity model and a subtraction.
        """
        leg = self.selected_leg
        if leg is None or measured_tau is None:
            return np.nan
        index = 3 * LEGS.index(leg)
        external = np.asarray(measured_tau, dtype=float)[
            index : index + 3
        ] - dog5_kinematics.leg_gravity_torque(
            leg, np.asarray(q, dtype=float)[index : index + 3]
        )
        if tau_offset is not None:
            external = external - np.asarray(tau_offset, dtype=float)[
                index : index + 3
            ]
        return float(max(abs(external[1]), abs(external[2])))

    def swing_leg_load(self, q, measured_tau, tau_offset=None):
        """(fz, |tau_ext|) of the selected foot -- the DISPLAY path.

        Runs foot_load_map, so keep it at the status rate, never per tick.
        """
        leg = self.selected_leg
        if leg is None or measured_tau is None:
            return np.nan, np.nan
        support, _ = crawl.foot_load_map(q, measured_tau, tau_offset)
        return support[leg], self.swing_leg_tau_ext(q, measured_tau, tau_offset)

    # ---- the controller's interface -------------------------------------
    def target(self, leg, now):
        del now
        if not self.active:
            target = np.asarray(self.controller.final_foot[leg], dtype=float)
            return target.copy(), np.zeros(3), 0.25, 1.0
        desired = np.array(
            [*(self.world_foot_xy[leg] - self.body_xy), self.nominal_z[leg]]
        )
        velocity = np.array(
            [-self.body_velocity_xy[0], -self.body_velocity_xy[1], 0.0]
        )
        share = 0.25
        trim_scale = 1.0
        if leg == self.selected_leg:
            desired[2] += self.lift_h * self.swing_height
            velocity[2] = self.lift_velocity_z
            share = 0.25 * (1.0 - self.unload_u)
            trim_scale = 1.0 - self.unload_u
        elif self.selected_leg is not None:
            share = 0.25 + self.unload_u * (1.0 / 3.0 - 0.25)
        return desired, velocity, share, trim_scale

    # ---- per-tick -------------------------------------------------------
    def update(self, now, q, measured_tau=None):
        """Advance the followers; auto-sequence only the return to neutral."""
        if not self.active:
            return None
        now = float(now)
        dt = float(np.clip(now - (self._last_update or now), 0.0, 0.1))
        self._last_update = now

        self.shift_m, delta = _move_toward(
            self.shift_m, self.shift_target_m, SHIFT_RATE_M_S * dt
        )
        self.body_velocity_xy = (
            (delta / dt) * self.shift_normal if dt > 0.0 else np.zeros(2)
        )
        self.unload_u, _ = _move_toward(
            self.unload_u, self.unload_target, UNLOAD_RATE_PER_S * dt
        )
        lift_rate = LIFT_RATE_M_S / max(self.swing_height, 1.0e-6)
        self.lift_h, lift_delta = _move_toward(
            self.lift_h, self.lift_target, lift_rate * dt
        )
        self.lift_velocity_z = (
            (lift_delta / dt) * self.swing_height if dt > 0.0 else 0.0
        )

        if measured_tau is not None and self.selected_leg is not None:
            self.last_tau_ext_nm = self.swing_leg_tau_ext(q, measured_tau)

        leg = self.selected_leg
        if self.state == LOWERING and self.lift_h <= FRACTION_EPS:
            self.state = RELOADING
            self.unload_target = 0.0
            return f"{leg} is back on the ground; restoring its support share"
        if self.state == RELOADING and self.unload_u <= FRACTION_EPS:
            self.state = RECENTERING
            self.shift_target_m = 0.0
            return f"{leg} reloaded; returning the body to neutral"
        if self.state == RECENTERING and self.shift_m <= SHIFT_EPS_M:
            self.state = NEUTRAL
            self.selected_leg = None
            self.shift_normal = np.zeros(2)
            self.shift_cap_m = 0.0
            return (
                f"{leg} done; flat on four feet at neutral -- press "
                f"{KEY_HINT} for another corner"
            )
        return None

    def drag_warning(self, now, tau_ext_nm):
        """Airborne but still measuring ground load: say so, do not act."""
        if not self.airborne or not np.isfinite(tau_ext_nm):
            return None
        if tau_ext_nm <= DRAG_WARN_TAU_NM:
            return None
        if now - self._drag_warned_at < DRAG_WARN_PERIOD_S:
            return None
        self._drag_warned_at = float(now)
        return (
            f"WARNING: {self.selected_leg} is commanded "
            f"{1000.0 * self.lift_h * self.swing_height:.0f} mm up but still "
            f"measures |tau_ext|={tau_ext_nm:.2f} N*m (> "
            f"{DRAG_WARN_TAU_NM:.2f}): the foot has NOT left the ground. "
            "Press D to reload, then + for more swing."
        )

    # ---- keys -----------------------------------------------------------
    def _absolute_cap(self, q):
        """Safe swing limit measured from neutral, at the CURRENT pose.

        ``shift_direction_cap`` reports how much further the body may travel
        from where it is now, so the limit from neutral is that plus the
        shift already taken.
        """
        return self.shift_m + shift_direction_cap(
            q,
            self.world_foot_xy,
            self.selected_leg,
            self.body_xy,
            self.shift_normal,
        )

    def select(self, key_leg, now, q):
        del now
        if self.state != NEUTRAL:
            return False, (
                f"corner {key_leg} refused: {self.context}; press D to return "
                "to neutral first"
            )
        q = np.asarray(q, dtype=float)
        direction, cap = best_shift_direction(
            q, self.world_foot_xy, key_leg, np.zeros(2)
        )
        if cap < SHIFT_STEP_M:
            return False, (
                f"corner {key_leg} refused: no safe swing direction (joint "
                "limits/support margin leave 0 mm of travel)"
            )
        self.selected_leg = key_leg
        self.shift_normal = direction
        self.shift_cap_m = cap
        self.shift_target_m = min(self.default_shift_m, cap)
        self.unload_target = 1.0
        self.state = SWING_COM
        heading = np.rad2deg(np.arctan2(direction[1], direction[0]))
        return True, (
            f"corner {key_leg}: swinging the CoM "
            f"{1000.0 * self.shift_target_m:.0f} mm along "
            f"{heading:+.0f} deg (safe cap {1000.0 * cap:.0f} mm) and fading "
            f"{key_leg}'s support share to zero -- watch [load], press + for "
            "more swing, A to lift"
        )

    def nudge(self, grow, now, q):
        del now
        if self.state != SWING_COM:
            return False, (
                f"+/- refused: only while holding the CoM swing ({self.context})"
            )
        if grow:
            self.shift_cap_m = self._absolute_cap(q)
            if self.shift_target_m >= self.shift_cap_m - 1.0e-9:
                return False, (
                    f"+ refused: {1000.0 * self.shift_target_m:.0f} mm is the "
                    f"safe cap for {self.selected_leg} (joint limits/support "
                    f"margin). This corner's swing limit is "
                    f"{1000.0 * self.shift_cap_m:.0f} mm."
                )
            self.shift_target_m = min(
                self.shift_target_m + SHIFT_STEP_M, self.shift_cap_m
            )
        else:
            if self.shift_target_m <= 0.0:
                return False, "- refused: already at neutral"
            self.shift_target_m = max(self.shift_target_m - SHIFT_STEP_M, 0.0)
        return True, (
            f"CoM swing target {1000.0 * self.shift_target_m:.0f} mm "
            f"(cap {1000.0 * self.shift_cap_m:.0f} mm)"
        )

    def lift(self, now, q):
        del now
        if self.state != SWING_COM:
            return False, f"A refused: not holding a CoM swing ({self.context})"
        if abs(self.shift_m - self.shift_target_m) > SHIFT_EPS_M:
            return False, (
                f"A refused: the CoM is still moving "
                f"({1000.0 * self.shift_m:.1f} of "
                f"{1000.0 * self.shift_target_m:.0f} mm) -- wait for it to "
                "settle"
            )
        if self.unload_u < 1.0 - FRACTION_EPS:
            return False, (
                f"A refused: {self.selected_leg}'s support share is still "
                f"fading ({100.0 * (1.0 - self.unload_u):.0f}% left)"
            )
        margin = self.liftoff_margin(q)
        if not np.isfinite(margin) or margin < MIN_LIFTOFF_MARGIN_M:
            return False, (
                f"A refused: support margin {1000.0 * margin:.1f} mm is below "
                f"the {1000.0 * MIN_LIFTOFF_MARGIN_M:.0f} mm floor -- the "
                "robot would topple. Press + to swing the CoM further."
            )
        self.lift_target = 1.0
        self.state = AIRBORNE
        return True, (
            f"A accepted: lifting {self.selected_leg} "
            f"{1000.0 * self.swing_height:.0f} mm (support margin "
            f"{1000.0 * margin:.1f} mm). D puts it back down."
        )

    def down(self, now, q):
        del now, q
        if self.state == AIRBORNE:
            self.lift_target = 0.0
            self.state = LOWERING
            return True, f"D accepted: lowering {self.selected_leg}"
        if self.state == SWING_COM:
            self.lift_target = 0.0
            self.state = RELOADING
            self.unload_target = 0.0
            return True, (
                f"D accepted: reloading {self.selected_leg} and returning to "
                "neutral"
            )
        return False, f"D refused: {self.context}"


def _polygon_margin(points):
    """Signed distance from the body origin to a convex support polygon."""
    points = np.asarray(points, dtype=float)
    if not np.all(np.isfinite(points)) or len(points) < 3:
        return np.nan
    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    ordered = points[np.argsort(angles)]
    margins = []
    for start, end in zip(ordered, np.roll(ordered, -1, axis=0)):
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length < 1.0e-6:
            return np.nan
        margins.append(
            float((edge[0] * -start[1] - edge[1] * -start[0]) / length)
        )
    return min(margins)


class ManualSequence(inplace.InPlaceStandSequence):
    """Stand sequence with a key-driven MANUAL stage after HOLD."""

    def __init__(self, now, start_q, planner, travel_scale=1.0):
        super().__init__(now, start_q, travel_scale)
        self.planner = planner

    def update(self, now, q=None, cart_error=None, qd=None):
        if self.stage == "MANUAL":
            return None  # the planner is driven by keys, not by this clock
        return super().update(now, q=q, cart_error=cart_error, qd=qd)

    def request_next(self, now, q, qd, healthy):
        if self.stage == "MANUAL":
            return False, (
                f"already in manual mode: press {KEY_HINT} to swing a corner, "
                "A lifts, D returns to neutral, X stops"
            )
        if self.stage == "HOLD":
            if self.travel_scale < 1.0:
                return False, "MANUAL refused: a full-height HOLD is required."
            if not healthy:
                return False, "MANUAL refused: motor latch/fault present."
            speed = float(np.max(np.abs(qd)))
            if speed > recorded.FINAL_QD_TOL:
                return False, (
                    f"MANUAL refused: max |qd| {speed:.2f} rad/s > "
                    f"{recorded.FINAL_QD_TOL:.2f}."
                )
            self.planner.start(now)
            self.stage = "MANUAL"
            self.started_at = float(now)
            self.wait_since = None
            return True, (
                f"ENTER accepted: manual jog armed. Press {KEY_HINT} to swing "
                "the CoM off a corner; watch the [load] line; + / - adjust "
                "the swing; A lifts the foot; D returns to neutral."
            )
        if self.stage in ("HOLD_PARTIAL", "HOLD_SAG"):
            return False, "MANUAL refused: full commanded stand must settle first."
        return super().request_next(now, q, qd, healthy)

    def request_park(self, now, healthy):
        if self.stage == "MANUAL":
            if self.planner.state != NEUTRAL:
                return False, (
                    "PARK refused: return to the four-foot neutral first "
                    f"(press D) -- {self.planner.context}"
                )
            self.planner.active = False
            self.stage = "HOLD"
        return super().request_park(now, healthy)


def run_hardware(args):
    controller = crawl.CrawlController(
        args.cart_gain_scale,
        args.support_scale,
        args.stand_height,
        args.travel_scale,
    )
    # The in-place stand + a full four-corner in-place swing at this shift is
    # a superset of what the jog commands, so validating it up front covers
    # the reachability/torque envelope.  The live cap (joint limits + support
    # margin, recomputed from encoders) is what bounds the manual + key.
    max_static_tau, planned_margin = crawl.validate_crawl_configuration(
        controller, args.shift_distance, 0.0, args.swing_height
    )
    planner = ManualSwingPlanner(controller, args.shift_distance, args.swing_height)
    gate = crawl.CrawlSafetyGate(
        args.tau_max,
        args.qd_estop,
        args.qd_estop_hard,
        overspeed_enabled=not args.no_overspeed,
    )
    unwrap = [base.CalibratedEncoderUnwrap() for _ in base.HARDWARE_JOINTS]

    print(
        "[manual] Flow: CURRENT -> CROUCH -> ENTER -> STAND -> HOLD -> ENTER "
        "-> MANUAL JOG"
    )
    print(
        f"[manual] Keys once armed: {KEY_HINT} swing the CoM off that corner "
        "(and fade its support share to zero); +/- adjust the swing by "
        f"{1000.0 * SHIFT_STEP_M:.0f} mm; A lifts the foot; D puts it down "
        "and returns to neutral; T tares the fz display; P parks; X stops."
    )
    print(
        f"[manual] default swing {1000.0 * args.shift_distance:.0f} mm, lift "
        f"{1000.0 * args.swing_height:.0f} mm, hard swing ceiling "
        f"{1000.0 * MAX_MANUAL_SHIFT_M:.0f} mm; planned in-place margin "
        f"{1000.0 * planned_margin:.1f} mm; static mg/3 peak torque "
        f"~{max_static_tau:.2f} N*m, cap {args.tau_max:.2f} N*m"
    )
    print(
        "[manual] The LIFT is not gated on the measured load -- that is your "
        f"call. Enforced: the {1000.0 * MIN_LIFTOFF_MARGIN_M:.0f} mm support "
        "margin before liftoff, the support-triangle e-stop while airborne, "
        "and all standard e-stops. A still-loaded airborne foot is WARNED "
        "about, not aborted."
    )
    if args.tau_max < max_static_tau:
        print(
            f"[manual] WARNING: --tau-max {args.tau_max:.2f} N*m is below the "
            f"~{max_static_tau:.2f} N*m static mg/3 requirement; keep supported."
        )
    if args.no_overspeed:
        print(
            "[manual] !! OVERSPEED E-STOP DISABLED (--no-overspeed): a real "
            "runaway will NOT be caught -- keep a hand on X."
        )
    print("[manual] ROBOT MUST REMAIN MECHANICALLY SUPPORTED DURING FIRST TESTS.")

    imu = None if args.no_imu else inplace._open_imu()
    key = base.KeyPoller()
    try:
        with base.motorbus.MotorBus(MOTOR_IDS, dirs=MOTOR_DIRECTIONS) as mb:
            armed = False
            stop_reason = None
            try:
                print("[manual] Arming with a zero-torque stream...")
                if not mb.arm(rate_hz=base.CONTROL_HZ):
                    raise RuntimeError("not all motors armed")
                armed = True

                start_q = recorded.zero_torque_preflight(mb, key, unwrap)
                now = time.perf_counter()
                sequence = ManualSequence(now, start_q, planner, args.travel_scale)
                gate.start(now, start_q)
                print(
                    "[manual] ENTER accepted: native position control moving "
                    "CURRENT -> recorded CROUCH."
                )

                slot = mb.slot(base.CONTROL_HZ)
                deadline = time.perf_counter() + slot
                start = now
                q = start_q.copy()
                tau_command = np.zeros(N_JOINTS)
                miss_monitor = base.CanMissMonitor(mb)
                status_period = 1.0 / base.FAULT_STATUS_HZ
                next_fault_status = np.asarray(
                    [
                        status_period + i * status_period / N_JOINTS
                        for i in range(N_JOINTS)
                    ]
                )
                last_recover = {mid: -base.RECOVER_PERIOD_S for mid in MOTOR_IDS}
                last_load_print = 0.0
                last_health_print = 0.0
                index = 0
                velocity = recorded.EncoderVelocity()
                margin_trip_streak = 0
                tau_tare = None

                while True:
                    mb.poll()
                    joint_index = index % N_JOINTS

                    if joint_index == 0:
                        now = time.perf_counter()
                        elapsed = now - start
                        q, qd_driver = base._joint_state(mb, unwrap)
                        qd_encoder = velocity.update(q, now)
                        torque_feedback = mb.torques_nm()
                        measured_torque = np.asarray(
                            [torque_feedback[mid] for mid in MOTOR_IDS]
                        )

                        old_stage = sequence.stage
                        event = sequence.update(
                            now,
                            q=q,
                            cart_error=controller.last_cart_error,
                            qd=(qd_encoder if velocity.ready else None),
                        )
                        if event:
                            print(f"[stage] {event}")
                            if sequence.stage == "WAIT_CROUCH":
                                print(
                                    "[stage] Press ENTER for Cartesian STAND "
                                    "when settled."
                                )
                            elif sequence.stage == "HOLD":
                                print(
                                    "[stage] Full stand held; press ENTER to "
                                    "arm the manual jog."
                                )
                            elif sequence.stage == "HOLD_SAG":
                                controller.rebase_final_z_to_measured(q)
                                print(
                                    "[stage] HOLD_SAG cannot jog; P parks or "
                                    "X stops."
                                )
                        if (
                            old_stage != sequence.stage
                            and sequence.stage == "WAIT_STAND"
                        ):
                            print(
                                "[stage] Waiting for stand settle; HOLD_SAG "
                                f"after {inplace.WAIT_STAND_TIMEOUT_S:.0f}s."
                            )
                        if old_stage == "STAND" and sequence.stage != "STAND":
                            controller.restore_full_targets()

                        if sequence.stage == "MANUAL":
                            event = planner.update(
                                now, q, measured_tau=measured_torque
                            )
                            if event:
                                print(f"[jog] {event}")

                        position_stage = sequence.stage in ("CROUCH", "WAIT_CROUCH")
                        requested_tau = None
                        if position_stage:
                            tau_command.fill(0.0)
                            gate.previous_tau.fill(0.0)
                        elif sequence.stage == "STAND":
                            requested_tau = controller.compute_stand(
                                q, qd_encoder, now - sequence.started_at
                            )
                            tau_command = gate.apply(requested_tau, q, now)
                        elif sequence.stage == "MANUAL":
                            requested_tau = controller.compute_crawl(
                                q, qd_encoder, planner, now
                            )
                            tau_command = gate.apply(requested_tau, q, now)
                        elif sequence.stage == "PARK":
                            requested_tau = controller.compute_park(
                                q, qd_encoder, now - sequence.started_at
                            )
                            tau_command = gate.apply(requested_tau, q, now)
                        elif sequence.stage == "WAIT_PARK":
                            requested_tau = controller.compute_park(
                                q, qd_encoder, T_PARK
                            )
                            tau_command = gate.apply(requested_tau, q, now)
                        else:
                            requested_tau = controller.compute_stand(
                                q, qd_encoder, T_STAND
                            )
                            tau_command = gate.apply(requested_tau, q, now)

                        if sequence.stage == "MANUAL":
                            stance = planner.trim_stance_legs
                            saturated = crawl._saturated_legs(
                                requested_tau, gate, now, controller, stance
                            )
                            controller.update_gait_trim(now, stance, saturated)
                        else:
                            trim_mode = inplace._trim_mode_for_stage(sequence.stage)
                            candidates = (
                                set(LEGS) if trim_mode == "integrate" else set()
                            )
                            saturated = crawl._saturated_legs(
                                requested_tau, gate, now, controller, candidates
                            )
                            controller.update_support_trim(
                                now, trim_mode, saturated
                            )

                        temps = base._temperatures(mb)
                        misses = miss_monitor.update(mb)
                        errors = mb.errors()
                        stop_reason = gate.estop_reason(
                            q,
                            qd_driver,
                            temps,
                            misses,
                            errors,
                            now,
                            enforce_position_limits=not position_stage,
                        )
                        support_margin = np.nan
                        if sequence.stage == "MANUAL":
                            support_margin = planner.support_margin(q)
                            if (
                                planner.three_leg_support
                                and support_margin < SUPPORT_MARGIN_ESTOP_M
                            ):
                                margin_trip_streak += 1
                            else:
                                margin_trip_streak = 0
                            if (
                                stop_reason is None
                                and margin_trip_streak
                                >= SUPPORT_MARGIN_ESTOP_STREAK
                            ):
                                stop_reason = (
                                    f"support triangle lost ({planner.context}): "
                                    f"margin={1000.0 * support_margin:.1f} mm"
                                )
                        else:
                            margin_trip_streak = 0

                        if position_stage and stop_reason is None:
                            stop_reason = recorded.position_stage_fault(
                                qd_encoder,
                                measured_torque,
                                velocity.ready,
                                args.crouch_speed_trip,
                                args.crouch_torque_trip,
                            )
                        latched = [
                            mid for mid, error in errors.items() if error & 0x80
                        ]
                        unverified = [
                            mid for mid in MOTOR_IDS if mb.rec(mid).error is None
                        ]
                        healthy = (
                            not latched and not unverified and stop_reason is None
                        )

                        pressed = key.get()
                        if pressed in ("x", "X"):
                            stop_reason = "operator X"
                        elif pressed in ("t", "T"):
                            tau_tare = measured_torque - crawl.leg_gravity_stack(q)
                            print(
                                "[legs] fz TARED at the current pose -- valid "
                                "only if ALL feet were off the ground "
                                f"(residual max {np.max(np.abs(tau_tare)):.2f} "
                                "N*m); press T again to re-tare"
                            )
                        elif pressed in ("p", "P"):
                            _, message = sequence.request_park(now, healthy)
                            print(f"[stage] {message}")
                        elif base._is_enter(pressed):
                            advanced, message = sequence.request_next(
                                now, q, qd_encoder, healthy
                            )
                            if advanced and sequence.stage == "STAND":
                                controller.restore_full_targets()
                            print(f"[stage] {message}")
                        elif pressed in KEY_TO_LEG:
                            leg = KEY_TO_LEG[pressed]
                            if sequence.stage != "MANUAL":
                                print(
                                    f"[jog] corner key {pressed}={leg} "
                                    f"ignored: manual jog is not armed "
                                    f"(stage={sequence.stage}); stand to HOLD "
                                    "and press ENTER first"
                                )
                            elif not healthy:
                                print(
                                    f"[jog] corner {leg} refused: motor "
                                    "latch/fault present"
                                )
                            else:
                                print(f"[jog] {planner.select(leg, now, q)[1]}")
                        elif pressed in ("+", "="):
                            if sequence.stage == "MANUAL":
                                print(f"[jog] {planner.nudge(True, now, q)[1]}")
                        elif pressed in ("-", "_"):
                            if sequence.stage == "MANUAL":
                                print(f"[jog] {planner.nudge(False, now, q)[1]}")
                        elif pressed in ("a", "A"):
                            if sequence.stage == "MANUAL":
                                print(f"[jog] {planner.lift(now, q)[1]}")
                        elif pressed in ("d", "D"):
                            if sequence.stage == "MANUAL":
                                print(f"[jog] {planner.down(now, q)[1]}")

                        if stop_reason:
                            if sequence.stage == "MANUAL":
                                stop_reason = f"{planner.context}: {stop_reason}"
                            break

                        recover = [
                            mid
                            for mid in latched
                            if elapsed - last_recover[mid] >= base.RECOVER_PERIOD_S
                        ]
                        if recover:
                            base._recover_input_lost(
                                mb, recover, elapsed, last_recover,
                                next_fault_status,
                            )
                            latched = []

                        if sequence.stage == "MANUAL":
                            warning = planner.drag_warning(
                                now, planner.last_tau_ext_nm
                            )
                            if warning:
                                print(f"[jog] {warning}")

                        if now - last_load_print >= 1.0 / LOAD_STATUS_HZ:
                            last_load_print = now
                            support, com_xy = crawl.foot_load_map(
                                q, measured_torque, tau_tare
                            )
                            fz_text = ",".join(
                                f"{leg}{support[leg]:+5.1f}" for leg in LEGS
                            )
                            com_text = (
                                f"({1000.0 * com_xy[0]:+.0f},"
                                f"{1000.0 * com_xy[1]:+.0f})mm"
                                if np.all(np.isfinite(com_xy))
                                else "(-)"
                            )
                            tare_text = " (tared)" if tau_tare is not None else ""
                            focus = ""
                            if sequence.stage == "MANUAL" and planner.selected_leg:
                                leg = planner.selected_leg
                                fz, tau_ext = planner.swing_leg_load(
                                    q, measured_torque, tau_tare
                                )
                                focus = (
                                    f"  <<{leg} fz={fz:+5.1f}N "
                                    f"|tau_ext|={tau_ext:4.2f}N*m>>"
                                )
                            imu_text = ""
                            if imu is not None:
                                try:
                                    d = imu.sample()
                                except Exception:
                                    d = None
                                if d is not None:
                                    stale = (
                                        " STALE"
                                        if d.age_s > inplace.IMU_STALE_S
                                        else ""
                                    )
                                    imu_text = (
                                        f" rp=({d.roll_deg:+.2f},"
                                        f"{d.pitch_deg:+.2f})deg{stale}"
                                    )
                            print(
                                f"[load] fz_N=({fz_text}) com={com_text}"
                                f"{tare_text}{imu_text}{focus}",
                                flush=True,
                            )

                        if now - last_health_print >= 1.0 / HEALTH_STATUS_HZ:
                            last_health_print = now
                            margin_text = (
                                "-"
                                if not np.isfinite(support_margin)
                                else f"{1000.0 * support_margin:.1f}mm"
                            )
                            jog_text = "-"
                            if sequence.stage == "MANUAL":
                                jog_text = (
                                    f"{planner.state} leg="
                                    f"{planner.selected_leg or '--'} shift="
                                    f"{1000.0 * planner.shift_m:.1f}/"
                                    f"{1000.0 * planner.shift_target_m:.0f}"
                                    f"(cap{1000.0 * planner.shift_cap_m:.0f})mm"
                                    f" lift="
                                    f"{1000.0 * planner.lift_h * planner.swing_height:.0f}mm"
                                )
                            print(
                                f"[manual] stage={sequence.stage:11s} "
                                f"jog={jog_text} margin={margin_text} "
                                f"cart_err={controller.last_cart_error:.3f}m "
                                f"force_max={controller.last_max_force:.1f}N "
                                f"max|qd_enc|={np.max(np.abs(qd_encoder)):.2f} "
                                f"max|tau|={np.max(np.abs(tau_command)):.2f}N*m "
                                f"Tmax={int(np.max(temps))}C "
                                f"latched={len(latched)}",
                                flush=True,
                            )

                    mid = MOTOR_IDS[joint_index]
                    elapsed = time.perf_counter() - start
                    if elapsed >= next_fault_status[joint_index]:
                        mb.status1_req(mid)
                        next_fault_status[joint_index] += status_period
                    elif sequence.stage in ("CROUCH", "WAIT_CROUCH"):
                        if not mb.position(
                            mid,
                            float(POSITION_TARGET_DEG[joint_index]),
                            args.crouch_max_speed_dps,
                        ):
                            raise RuntimeError(
                                f"CAN position transmit failed for CAN {mid}"
                            )
                    else:
                        mb.torque(mid, float(tau_command[joint_index]))

                    index += 1
                    mb.pace(deadline)
                    deadline += slot

            except KeyboardInterrupt as exc:
                stop_reason = str(exc) or "KeyboardInterrupt"
            except Exception as exc:
                stop_reason = f"error: {exc}"
                raise RuntimeError(str(exc)) from exc
            finally:
                if armed:
                    print(f"[manual] stopping: {stop_reason or 'aborted'}")
                    try:
                        base._soft_stop(mb)
                    except Exception as exc:
                        print(
                            f"[manual] soft stop failed: {exc}; sending STOP",
                            file=sys.stderr,
                        )
    finally:
        key.close()
        if imu is not None:
            try:
                imu.stop()
            except Exception:
                pass
    return 0


def offline_self_test(args):
    controller = crawl.CrawlController(
        args.cart_gain_scale, args.support_scale, args.stand_height, 1.0
    )
    crawl.validate_crawl_configuration(
        controller, args.shift_distance, 0.0, args.swing_height
    )
    planner = ManualSwingPlanner(controller, args.shift_distance, args.swing_height)

    q = recorded.Q_RECORDED_CROUCH.copy()
    for index, leg in enumerate(LEGS):
        section = slice(3 * index, 3 * index + 3)
        q[section] = crawl._ik_to_target(
            leg, q[section], controller.final_foot[leg]
        )
    hold_before = {leg: controller.final_foot[leg].copy() for leg in LEGS}
    now = [0.0]

    def track():
        for index, leg in enumerate(LEGS):
            desired = planner.target(leg, now[0])[0]
            section = slice(3 * index, 3 * index + 3)
            q[section] = crawl._ik_to_target(leg, q[section], desired)

    def synth_tau(ground_leg_load):
        """Joint torque holding each leg's own links, plus the ground load
        each foot carries (``ground_leg_load`` maps leg -> fz)."""
        tau = np.zeros(N_JOINTS)
        for index, leg in enumerate(LEGS):
            section = slice(3 * index, 3 * index + 3)
            tau[section] = dog5_kinematics.leg_gravity_torque(leg, q[section])
            fz = ground_leg_load.get(leg, 0.0)
            if fz:
                jac = dog5_kinematics.foot_jacobian(leg, q[section])
                tau[section] += jac.T @ np.array([0.0, 0.0, -fz])
        return tau

    def settle(seconds=12.0, dt=0.05):
        """Run the followers to convergence, then snap q onto the targets.

        The followers are open-loop (they never read q), so tracking only
        needs solving once at the end -- a full-robot IK round costs ~80 ms
        and doing it per step made this test take minutes.
        """
        events = []
        for _ in range(int(seconds / dt)):
            now[0] += dt
            event = planner.update(now[0], q)
            if event:
                events.append(event)
        track()
        return events

    # ---- share conservation across every reachable follower state --------
    planner.start(now[0])
    assert planner.state == NEUTRAL and planner.selected_leg is None
    for u in (0.0, 0.3, 1.0):
        for h in (0.0, 0.5, 1.0):
            planner.selected_leg = "RR"
            planner.unload_u, planner.lift_h = u, h
            shares = [planner.target(leg, 0.0)[2] for leg in LEGS]
            assert np.isclose(sum(shares), 1.0), (u, h, shares)
            assert shares[LEGS.index("RR")] >= 0.0
    planner.start(now[0])

    # ---- keys are refused in the wrong state ----------------------------
    assert not planner.lift(now[0], q)[0]
    assert not planner.nudge(True, now[0], q)[0]
    assert not planner.down(now[0], q)[0]

    # ---- RR: swing the CoM, watch it unload, lift, come back ------------
    ok, message = planner.select("RR", now[0], q)
    assert ok and "RR" in message, message
    assert planner.state == SWING_COM
    assert planner.shift_cap_m >= planner.shift_target_m > 0.0
    # A is refused while the body is still moving.
    assert not planner.lift(now[0], q)[0]
    settle()
    assert abs(planner.shift_m - planner.shift_target_m) < SHIFT_EPS_M
    assert planner.unload_u > 1.0 - FRACTION_EPS
    # The CoM swing alone must not lift the foot, and no stance foot moves.
    assert np.isclose(planner.target("RR", now[0])[0][2], planner.nominal_z["RR"])
    # The selected leg's support share is gone; the other three share it.
    assert np.isclose(planner.target("RR", now[0])[2], 0.0)
    assert np.isclose(planner.target("FL", now[0])[2], 1.0 / 3.0)
    assert planner.trim_stance_legs == set(LEGS) - {"RR"}
    # Synthesized load: RR unloaded, the other three carry mg/3.
    third = base.DOG5_MASS_KG * 9.81 / 3.0
    loads = {"FL": third, "FR": third, "RL": third}
    fz, tau_ext = planner.swing_leg_load(q, synth_tau(loads))
    assert abs(fz) < 1.0e-6, fz
    assert tau_ext < 1.0e-6, tau_ext

    # + grows the swing, - shrinks it, both bounded.
    before = planner.shift_target_m
    ok, message = planner.nudge(True, now[0], q)
    assert ok and np.isclose(planner.shift_target_m, before + SHIFT_STEP_M), message
    ok, _ = planner.nudge(False, now[0], q)
    assert ok and np.isclose(planner.shift_target_m, before)
    grow_guard = 0
    while planner.nudge(True, now[0], q)[0]:
        grow_guard += 1
        assert grow_guard < 100
    assert planner.shift_target_m <= planner.shift_cap_m + 1.0e-9
    assert planner.shift_target_m <= MAX_MANUAL_SHIFT_M + 1.0e-9
    refused, message = planner.nudge(True, now[0], q)
    assert not refused and "swing limit" in message, message
    planner.shift_target_m = min(before, planner.shift_cap_m)
    settle()

    # A lifts only with margin in hand; the foot then leaves the ground.
    margin = planner.liftoff_margin(q)
    assert margin >= MIN_LIFTOFF_MARGIN_M, margin
    ok, message = planner.lift(now[0], q)
    assert ok and "lifting RR" in message, message
    assert planner.state == AIRBORNE
    settle()
    assert np.isclose(planner.lift_h, 1.0)
    lifted = planner.target("RR", now[0])[0]
    assert np.isclose(lifted[2], planner.nominal_z["RR"] + args.swing_height)
    assert planner.airborne and planner.three_leg_support
    # It holds there indefinitely -- no timeout takes the foot back down.
    settle(seconds=120.0, dt=0.05)
    assert planner.state == AIRBORNE and np.isclose(planner.lift_h, 1.0)

    # A foot that is airborne but still measuring load WARNS, never aborts.
    stuck = planner.drag_warning(1.0e6, DRAG_WARN_TAU_NM + 0.5)
    assert stuck and "NOT left the ground" in stuck, stuck
    assert planner.state == AIRBORNE
    assert planner.drag_warning(1.0e6, 0.0) is None

    # D returns: lower -> reload -> recenter -> neutral, all automatic.
    ok, message = planner.down(now[0], q)
    assert ok and planner.state == LOWERING, message
    events = settle(seconds=30.0)
    assert planner.state == NEUTRAL, planner.state
    assert planner.selected_leg is None
    assert any("back on the ground" in e for e in events), events
    assert any("reloaded" in e for e in events), events
    assert any("neutral" in e for e in events), events
    assert planner.shift_m <= SHIFT_EPS_M and planner.unload_u <= FRACTION_EPS
    assert planner.trim_stance_legs == set(LEGS)
    for leg in LEGS:
        assert np.allclose(
            controller.final_foot[leg], hold_before[leg], atol=1.0e-9
        ), leg
        actual = dog5_kinematics.foot_position(
            leg, q[3 * LEGS.index(leg) : 3 * LEGS.index(leg) + 3]
        )
        assert np.allclose(actual, controller.final_foot[leg], atol=1.0e-4), leg

    # ---- D from the CoM swing (foot never lifted) also returns cleanly ---
    planner.select("FL", now[0], q)
    settle()
    ok, _ = planner.down(now[0], q)
    assert ok and planner.state == RELOADING
    settle(seconds=30.0)
    assert planner.state == NEUTRAL and planner.selected_leg is None

    # ---- every corner: swing -> clear the floor -> lift -> return -------
    caps = {}
    margins = {}
    for leg in LEGS:
        ok, message = planner.select(leg, now[0], q)
        assert ok, message
        # Right after select the body has not moved yet, so the three-foot
        # margin is still ~0 (the origin sits on the triangle's diagonal).
        # It is the settled swing that has to clear the liftoff floor.
        assert planner.liftoff_margin(q) < MIN_LIFTOFF_MARGIN_M
        settle()
        caps[leg] = planner.shift_cap_m
        margins[leg] = planner.liftoff_margin(q)
        assert margins[leg] >= MIN_LIFTOFF_MARGIN_M, (leg, margins[leg])
        ok, message = planner.lift(now[0], q)
        assert ok, (leg, message)
        settle()
        assert np.isclose(planner.lift_h, 1.0), leg
        ok, _ = planner.down(now[0], q)
        assert ok, leg
        settle(seconds=30.0)
        assert planner.state == NEUTRAL, (leg, planner.state)

    # ---- per-tick cost stays inside the control budget ------------------
    # The loop runs this block at CONTROL_HZ and a stall risks the motors'
    # 10 ms CAN watchdog, so the planner's per-tick work must stay small.
    # (foot_load_map costs ~1.6 ms and belongs on the status path only.)
    planner.select("RR", now[0], q)
    settle()
    tau_tick = synth_tau({leg: third for leg in LEGS if leg != "RR"})
    ticks = 200
    started = time.perf_counter()
    for step in range(ticks):
        planner.update(now[0] + 0.004 * step, q, measured_tau=tau_tick)
    per_tick_us = 1.0e6 * (time.perf_counter() - started) / ticks
    budget_us = 1.0e6 / base.CONTROL_HZ
    assert per_tick_us < 0.15 * budget_us, (
        f"planner.update costs {per_tick_us:.0f} us of the "
        f"{budget_us:.0f} us tick -- too much for the CAN watchdog"
    )
    planner.down(now[0], q)
    settle(seconds=30.0)
    assert planner.state == NEUTRAL

    print(
        f"[manual] planner.update costs {per_tick_us:.0f} us per tick "
        f"({budget_us:.0f} us budget at {base.CONTROL_HZ:.0f} Hz); "
        "foot_load_map stays on the 4 Hz display path"
    )
    print(
        "[manual] self-test OK: CoM swing holds indefinitely with the leg's "
        "share faded out (shares always sum to 1); +/- bounded by the live "
        "joint-limit/margin cap; A refused without margin, lifts otherwise "
        "and holds a 3-leg stand for 2 min; a still-loaded airborne foot "
        "only WARNS; D always returns to the exact neutral."
    )
    print(
        f"[manual] per-corner safe swing cap / liftoff margin at the default "
        f"{1000.0 * args.shift_distance:.0f} mm swing: "
        + ", ".join(
            f"{leg} {1000.0 * caps[leg]:.0f}mm/{1000.0 * margins[leg]:.0f}mm"
            for leg in LEGS
        )
    )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tau-max", type=float, default=DEFAULT_TAU_MAX)
    parser.add_argument(
        "--cart-gain-scale", type=float, default=base.DEFAULT_CART_GAIN_SCALE
    )
    parser.add_argument(
        "--support-scale", type=float, default=inplace.DEFAULT_SUPPORT_SCALE
    )
    parser.add_argument(
        "--stand-height", type=float, default=DEFAULT_STAND_HEIGHT_M
    )
    parser.add_argument("--travel-scale", type=float, default=1.0)
    parser.add_argument(
        "--shift-distance",
        type=float,
        default=DEFAULT_SHIFT_DISTANCE_M,
        help=(
            "CoM swing a corner key starts with, m; + / - adjust it live "
            f"(default: {DEFAULT_SHIFT_DISTANCE_M})"
        ),
    )
    parser.add_argument(
        "--swing-height",
        type=float,
        default=DEFAULT_SWING_HEIGHT_M,
        help=f"foot lift height for A, m (default: {DEFAULT_SWING_HEIGHT_M})",
    )
    parser.add_argument(
        "--crouch-max-speed-dps",
        type=float,
        default=recorded.DEFAULT_CROUCH_MAX_MOTOR_DPS,
    )
    parser.add_argument(
        "--crouch-torque-trip",
        type=float,
        default=recorded.DEFAULT_CROUCH_TORQUE_TRIP_NM,
    )
    parser.add_argument(
        "--crouch-speed-trip",
        type=float,
        default=recorded.DEFAULT_CROUCH_SPEED_TRIP_RAD_S,
    )
    parser.add_argument("--qd-estop", type=float, default=base.QD_ESTOP)
    parser.add_argument("--qd-estop-hard", type=float, default=base.QD_ESTOP_HARD)
    parser.add_argument("--no-overspeed", action="store_true")
    parser.add_argument(
        "--no-imu",
        action="store_true",
        help="skip the trunk IMU roll/pitch on the [load] line (display only)",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="validate without opening CAN"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not 0.0 < args.tau_max <= base.STAGED_TAU_MAX:
        parser.error(f"--tau-max must be > 0 and <= {base.STAGED_TAU_MAX} N*m")
    if not 0.0 < args.cart_gain_scale <= 1.0:
        parser.error("--cart-gain-scale must be > 0 and <= 1")
    if not 0.0 <= args.support_scale <= 1.0:
        parser.error("--support-scale must be between 0 and 1")
    if not crawl.MIN_CRAWL_STAND_HEIGHT_M <= args.stand_height <= inplace.MAX_STAND_HEIGHT:
        parser.error(
            f"--stand-height must be between {crawl.MIN_CRAWL_STAND_HEIGHT_M} "
            f"and {inplace.MAX_STAND_HEIGHT} m"
        )
    if not 0.05 <= args.travel_scale <= 1.0:
        parser.error("--travel-scale must be between 0.05 and 1.0")
    if not 0.0 < args.shift_distance <= MAX_MANUAL_SHIFT_M:
        parser.error(
            f"--shift-distance must be > 0 and <= {MAX_MANUAL_SHIFT_M} m"
        )
    if not 0.015 <= args.swing_height <= 0.05:
        parser.error("--swing-height must be between 0.015 and 0.05 m")
    if not 0.0 < args.qd_estop <= base.QD_ESTOP_CEILING:
        parser.error(f"--qd-estop must be > 0 and <= {base.QD_ESTOP_CEILING}")
    if not args.qd_estop < args.qd_estop_hard <= base.QD_ESTOP_CEILING:
        parser.error(
            "--qd-estop-hard must be above --qd-estop and <= "
            f"{base.QD_ESTOP_CEILING} rad/s"
        )
    try:
        if args.self_test:
            return offline_self_test(args)
        return run_hardware(args)
    except (RuntimeError, ValueError) as exc:
        print(f"[manual] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
