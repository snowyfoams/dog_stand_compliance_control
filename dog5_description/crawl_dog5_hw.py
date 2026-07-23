#!/usr/bin/env python3
"""Statically stable encoder-gated crawl for DOG5 on real CAN hardware.

This runner builds on :mod:`stand_dog5_inplace_hw`.  It first performs the
same recorded-crouch to vertical stand, then ENTER starts a finite crawl.
Only one foot swings at a time.  Every step is:

    SHIFT -> UNLOAD -> LIFT -> SWING -> LOWER -> LOAD -> RECENTER

Before liftoff, the trunk shifts by ``--shift-distance`` perpendicular to the
nearest edge of the current stance triangle -- the edge that limits the
support margin -- so the shift adapts to whatever (possibly skewed) geometry
the previous steps left behind, instead of a fixed diagonal that only suits
the symmetric rectangle.  The command is expressed by moving every stance-foot
target the opposite way in the trunk frame.  Each stance foot has a fixed
conceptual ground anchor, so changing the body target cannot command
stance-foot slip.

The support triangle is calculated directly from encoder FK.  LIFT cannot
begin until the trunk-frame CoM projection (the origin) is at least 15 mm
inside the other three feet, and leaving the triangle during swing stops the
run.  The geometric margin assumes the CoM sits at the body origin, so the
UNLOAD phase adds a physical check: the swing leg's support feedforward
fades to zero while the foot is still planted, and LIFT is refused until the
leg MEASURES unloaded (gravity-compensated pitch/knee |tau_ext| below
``--unload-tau-trip`` and relative sag below 8 mm).  While the leg still
measures loaded, the body keeps shifting deeper along the same inward normal
(force-feedback, up to +20 mm extra, bounded by the planned support margin
and the joint soft limits), so a real CoM offset that geometry cannot see is
walked out by measurement instead of failing the step.  Only when that safe
budget is exhausted does the gate abort.  Once airborne, a drag watch aborts
gracefully if the swing foot still measures ground load (foot scraping).

Mid-step gate failures (liftoff margin, unload gate, touchdown, recenter) no
longer stop the run: the planner reloads the swing leg where it stands,
returns the body to neutral on four planted feet, and ends the batch in
CRAWL_HOLD with an ABORTED message.  Only real support-triangle loss, the
safety-gate e-stops, and X stop the run.  The three stance legs receive an
mg/3 support share; the swing leg's trim integrator is frozen from UNLOAD
through LOAD and resumes once the foot is fully reloaded.
Touchdown is inferred only after encoder FK puts the slow swing foot back on
the plane through the three stance feet; gravity load is then restored.

The default timing is a conservative 3 s slot per leg (12 s per four-leg gait
cycle) with nominal duty factor 0.90.  One complete gait cycle is run by
default, after which targets are again the original symmetric stand targets.
ENTER repeats another batch; P parks only from a stationary hold; X stops at
any time.  T tares the fz load-map display (press while holding the robot
with ALL feet off the ground; removes the gear-friction/model-bias residual,
display only -- gates are never tared).

Mechanically support the robot for initial tests::

    python crawl_dog5_hw.py --self-test
    python crawl_dog5_hw.py --tau-max 3.0 --step-length 0.0
    python crawl_dog5_hw.py --tau-max 3.0 --step-length 0.03

To identify whether a problem is leg swing or forward body motion, first run
the zero-step swing test and pause at every phase boundary::

    python crawl_dog5_hw.py --swing-test --observe-phases
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import dog5_kinematics
import stand_dog5
import stand_dog5_hw as base
import stand_dog5_inplace_hw as inplace
import stand_dog5_recorded_hw as recorded


LEGS = base.LEGS
MOTOR_IDS = base.MOTOR_IDS
MOTOR_DIRECTIONS = base.MOTOR_DIRECTIONS
JOINT_LABELS = base.JOINT_LABELS
N_JOINTS = base.N_JOINTS
Q_RECORDED_CROUCH = recorded.Q_RECORDED_CROUCH
POSITION_TARGET_DEG = recorded.POSITION_TARGET_DEG
T_STAND = recorded.T_STAND
T_PARK = recorded.T_PARK

# A diagonal-sequence static crawl.  Every leg is in stance while any other
# leg is swinging, and the diagonal body shift makes each liftoff static.
# The order is each hind leg followed by the DIAGONALLY OPPOSITE front leg
# (RR->FL, then RL->FR), which is the maximum-stability crawl sequence: the
# swing legs alternate across the diagonal so the loaded three-foot triangle
# stays well-centred instead of collapsing to one side.
GAIT_ORDER = ("RR", "FL", "RL", "FR")

# Insert a mandatory, ENTER-gated four-foot hold after every this-many steps,
# i.e. between the two diagonal pairs (RR FL <hold> RL FR <hold> ...).  Unlike
# --observe-phases (which pauses at every phase boundary), this hold is always
# active: the robot settles flat on all four feet and waits for the operator.
MANDATORY_HOLD_EVERY = 2

# Fractions of one per-leg slot.  Only LIFT/SWING/LOWER count as swing, so one
# leg is airborne for 0.40 of its slot and for 0.10 of a four-slot gait cycle.
# The ABORT_* phases run only when a gate fails and extend that slot.
PHASE_FRACTIONS = {
    "SHIFT": 0.20,
    "UNLOAD": 0.10,
    "LIFT": 0.15,
    "SWING": 0.10,
    "LOWER": 0.15,
    "LOAD": 0.10,
    "RECENTER": 0.20,
    "ABORT_LOAD": 0.10,
    "ABORT_RECENTER": 0.20,
}
SWING_PHASES = frozenset(("LIFT", "SWING", "LOWER"))
# Phases in which the swing leg carries no (or reduced) support and its trim
# integrator must stay frozen.
REDUCED_SUPPORT_PHASES = SWING_PHASES.union(
    ("UNLOAD", "LOAD", "ABORT_LOAD")
)
PHASE_MOTION_KIND = {
    "SHIFT": "BODY_MOTION",
    "UNLOAD": "WEIGHT_TRANSFER",
    "LIFT": "LEG_SWING",
    "SWING": "LEG_SWING",
    "LOWER": "LEG_SWING",
    "LOAD": "TOUCHDOWN_LOAD",
    "RECENTER": "BODY_MOTION",
    "ABORT_LOAD": "WEIGHT_TRANSFER",
    "ABORT_RECENTER": "BODY_MOTION",
    "IDLE": "HOLD",
}
NOMINAL_DUTY_FACTOR = 1.0 - sum(
    PHASE_FRACTIONS[phase] for phase in SWING_PHASES
) / len(GAIT_ORDER)

DEFAULT_TAU_MAX = base.STAGED_TAU_MAX
# Keep the trunk low for a lower CoM.  At 0.10 m, the old 30 mm body shift and
# 25 mm lift leave the abduction workspace, so the coupled defaults below use
# a 22.5 mm shift and 20 mm lift.  The full 30 mm forward gait remains
# reachable and retains about 20 mm of planned support-triangle margin.
MIN_CRAWL_STAND_HEIGHT_M = 0.10
# 0.15 m (was 0.10): at 0.10 the abduction joint sits within ~0.1 deg of its
# soft limit during the body shift, so the gate clamps the outward torque and
# the shift never completes -- the swing leg then cannot unload.  Letting the
# legs hang lower (knee -116 -> -90 deg) frees abduction headroom so the shift
# finishes; conditioning and mg/3 torque stay healthy.  --stand-height still
# spans 0.10..0.21 for experiments.
DEFAULT_STAND_HEIGHT_M = 0.15
# 26 mm (was 22.5): with the adaptive shift and the headroom from 0.15 m, this
# keeps the worst (skewed second-pair) support-triangle margin near +21 mm --
# well above the 15 mm liftoff floor -- while leaving ~1.4 deg of abduction.
DEFAULT_SHIFT_DISTANCE_M = 0.026
DEFAULT_STEP_LENGTH_M = 0.03
DEFAULT_SWING_HEIGHT_M = 0.020
DEFAULT_LEG_CYCLE_S = 3.0
DEFAULT_GAIT_CYCLES = 1

# UNLOAD gate: the swing leg must physically measure unloaded before LIFT.
# GRAVITY-COMPENSATED torque feedback on its pitch and knee (measured minus
# the leg's own link-weight holding torque, ~0.4 N*m at the stand pose) must
# stay below the trip and the RELATIVE sag (swing minus mean stance -- moving
# 4 -> 3 loaded legs sags the whole trunk, which absolute sag would misread)
# below the maximum, sustained for the settle time.  Timing out aborts the
# step instead of lifting.
DEFAULT_UNLOAD_TAU_TRIP_NM = 0.45
UNLOAD_REL_SAG_MAX_M = 0.008
UNLOAD_GATE_SETTLE_S = 0.20
# Force-feedback shift: geometry says WHERE to shift, but the measured swing
# torque says WHETHER the leg actually unloaded -- with a real CoM offset the
# planned shift can complete and still leave the leg loaded (foot drags).  So
# while the gate reads loaded, keep shifting deeper along the same inward
# normal, up to an extra budget bounded by the planned support margin and the
# (linearized) joint limits.  Timeout raised 1.5 -> 3.5 s to give the extra
# shift time to act; the abort path is unchanged.
UNLOAD_GATE_TIMEOUT_S = 3.5
UNLOAD_EXTRA_RATE_M_S = 0.012
UNLOAD_EXTRA_MAX_M = 0.020

# IMU lean watch (Phase 3): during 3-leg support the encoder-based support
# triangle assumes rigid geometry, but the body measurably tilts beyond it
# (Phase 1: stands sag ~1 deg, ramps ~4 deg).  A tilt CHANGE beyond this
# many degrees from the liftoff attitude means the body is falling toward
# the airborne corner -> graceful step abort (same reload+recenter path as
# the drag watch), not a run stop.
CRAWL_LEAN_ABORT_DEG = 6.0
CRAWL_LEAN_STREAK = 3           # consecutive fresh samples to confirm
# Keep this below the abduction headroom left by the base shift (~1.4 deg at
# H=0.15), or the lateral growth budget collapses to zero and only the
# longitudinal fallback direction remains.
UNLOAD_LIMIT_MARGIN_RAD = 0.02
# Drag watch: once airborne (LIFT/SWING) the compensated swing torque should
# read ~0; sustained torque means the foot is scraping the ground.  Abort
# gracefully instead of dragging through the step.  Threshold is deliberately
# above the unload trip to ignore swing dynamics.
SWING_DRAG_TAU_TRIP_NM = 0.90
SWING_DRAG_STREAK = 3

MIN_LIFTOFF_MARGIN_M = 0.015
SUPPORT_MARGIN_ESTOP_M = 0.0
SUPPORT_MARGIN_ESTOP_STREAK = 3
SHIFT_CART_TOL_M = 0.025
SHIFT_QD_TOL_RAD_S = 0.35
SHIFT_SETTLE_S = 0.20
SHIFT_SETTLE_TIMEOUT_S = 3.0
TOUCHDOWN_CART_TOL_M = 0.012
# A planted foot can stop above its unloaded body-frame target when the trunk
# is sagged.  Infer contact geometrically: the encoder-FK foot must be close to
# the plane through the other three feet, laterally placed, and nearly still.
# This is not a force/contact measurement, but it rejects a stationary foot
# that stalled in midair above the stance plane.
# A foot that contacts early is pinned by friction with residual xy error
# (hardware showed 8.7 mm on a 30 mm step); the MEASURED foot position is
# committed as the anchor on touchdown, so the tolerance only needs to bound
# how far the bookkeeping may step at once, not tracking accuracy.
TOUCHDOWN_CONTACT_XY_TOL_M = 0.015
TOUCHDOWN_PLANE_TOL_M = 0.008
TOUCHDOWN_TARGET_ABOVE_PLANE_TOL_M = 0.004
TOUCHDOWN_TARGET_BELOW_PLANE_MAX_M = 0.030
TOUCHDOWN_FOOT_SPEED_TOL_M_S = 0.020
TOUCHDOWN_QD_TOL_RAD_S = 0.30
TOUCHDOWN_SETTLE_S = 0.20
TOUCHDOWN_TIMEOUT_S = 2.0
RECENTER_CART_TOL_M = 0.025
RECENTER_QD_TOL_RAD_S = 0.35
RECENTER_SETTLE_S = 0.20
RECENTER_TIMEOUT_S = 3.0
OBSERVE_RESUME_CART_TOL_M = 0.025
OBSERVE_RESUME_QD_TOL_RAD_S = 0.35


def touchdown_accepted(
    total_error,
    xy_error,
    foot_plane_distance,
    target_plane_distance,
    joint_speed,
    foot_speed,
):
    """Return whether encoder geometry consistently indicates touchdown."""
    tracked = total_error <= TOUCHDOWN_CART_TOL_M
    on_plane = abs(foot_plane_distance) <= TOUCHDOWN_PLANE_TOL_M
    target_reaches_plane = (
        -TOUCHDOWN_TARGET_BELOW_PLANE_MAX_M
        <= target_plane_distance
        <= TOUCHDOWN_TARGET_ABOVE_PLANE_TOL_M
    )
    accepted = (
        xy_error <= TOUCHDOWN_CONTACT_XY_TOL_M
        and on_plane
        and target_reaches_plane
        and joint_speed <= TOUCHDOWN_QD_TOL_RAD_S
        and foot_speed <= TOUCHDOWN_FOOT_SPEED_TOL_M_S
    )
    return accepted, tracked, on_plane


def signed_distance_to_stance_plane(point, stance_points):
    """Signed point/plane distance with the normal oriented toward body +z."""
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


def support_triangle_margin(points_xy, com_xy=(0.0, 0.0)):
    """Return signed distance from ``com_xy`` to a three-point triangle.

    The result is positive inside, zero on an edge, and negative outside.
    Points are sorted counter-clockwise, so each edge's left normal points
    inward.  Degenerate triangles are rejected.
    """
    points = np.asarray(points_xy, dtype=float)
    point = np.asarray(com_xy, dtype=float)
    if points.shape != (3, 2):
        raise ValueError(f"support triangle must have shape (3,2), got {points.shape}")
    if point.shape != (2,) or not np.all(np.isfinite(points)):
        raise ValueError("support geometry must be finite")

    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    ordered = points[np.argsort(angles)]
    margins = []
    for start, end in zip(ordered, np.roll(ordered, -1, axis=0)):
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length < 1.0e-6:
            raise ValueError("support triangle is degenerate")
        relative = point - start
        cross_z = edge[0] * relative[1] - edge[1] * relative[0]
        margins.append(float(cross_z / length))
    return min(margins)


def leg_gravity_stack(q):
    """Stacked per-joint torque holding every leg's own links, at pose q."""
    q = np.asarray(q, dtype=float)
    stacked = np.zeros(N_JOINTS)
    for index, leg in enumerate(LEGS):
        section = slice(3 * index, 3 * index + 3)
        stacked[section] = dog5_kinematics.leg_gravity_torque(leg, q[section])
    return stacked


def foot_load_map(q, measured_tau, tau_offset=None):
    """Per-leg vertical support force (N, positive down into the ground) and
    the CoM xy implied by the load distribution, from measured joint torque
    via f = J^-T (tau - tau_leg_gravity - tau_offset).  Subtracting each
    leg's own link weight isolates the external ground load, so an airborne
    leg reads ~0 instead of the ~0.4-0.5 N*m it holds against gravity.

    The remaining noise floor is gearbox static friction and small mass-model
    errors, a few 0.01 N*m per joint -- amplified by the folded-pose Jacobian
    (sigma_min ~0.018 at crouch) to ~1 N of phantom force.  ``tau_offset``
    (the operator 'T' tare, captured with all feet OFF the ground) removes
    the repeatable part of that bias for the display."""
    q = np.asarray(q, dtype=float)
    measured_tau = np.asarray(measured_tau, dtype=float)
    support = {}
    weighted_xy = np.zeros(2)
    total = 0.0
    for index, leg in enumerate(LEGS):
        section = slice(3 * index, 3 * index + 3)
        jacobian = dog5_kinematics.foot_jacobian(leg, q[section])
        # Near-singular legs (folded crouch) turn small torque noise into
        # huge force estimates; report unknown instead of garbage.
        if np.linalg.svd(jacobian, compute_uv=False)[-1] < 1.0e-3:
            support[leg] = np.nan
            continue
        external_tau = measured_tau[section] - dog5_kinematics.leg_gravity_torque(
            leg, q[section]
        )
        if tau_offset is not None:
            external_tau = external_tau - np.asarray(
                tau_offset, dtype=float
            )[section]
        try:
            force = np.linalg.solve(jacobian.T, external_tau)
        except np.linalg.LinAlgError:
            support[leg] = np.nan
            continue
        support[leg] = -float(force[2])
        if support[leg] > 0.0:
            weighted_xy += support[leg] * dog5_kinematics.foot_position(
                leg, q[section]
            )[:2]
            total += support[leg]
    com_xy = weighted_xy / total if total > 1.0 else np.full(2, np.nan)
    return support, com_xy


def adaptive_body_shift(swing_xy, stance_points, body_xy, shift_distance):
    """Body target that moves the CoM away from the current triangle's edge.

    After the first diagonal pair steps forward the four feet are no longer a
    symmetric rectangle, so a fixed 45-degree diagonal aims the CoM at the
    wrong place.  This instead shifts along the inward normal of the stance
    triangle edge NEAREST the CoM -- the edge that limits the support margin --
    so the shift is always perpendicular to whatever is about to be violated,
    for any (even skewed, multi-cycle) geometry.  Falls back to the old
    away-from-swing diagonal only if the triangle is degenerate.

    Returns ``(shifted_body_xy, unit_normal)``; the normal lets the UNLOAD
    force-feedback extend the shift along the same direction.
    """
    body_xy = np.asarray(body_xy, dtype=float)
    pts = [np.asarray(p, dtype=float) for p in stance_points]
    centroid = np.mean(pts, axis=0)
    best_distance, best_normal = np.inf, None
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 1.0e-9:
            continue
        normal = np.array([-edge[1], edge[0]]) / length
        if np.dot(normal, centroid - a) < 0.0:
            normal = -normal
        distance = float(np.dot(normal, body_xy - a))
        if distance < best_distance:
            best_distance, best_normal = distance, normal
    if best_normal is None:
        signs = np.sign(np.asarray(swing_xy, dtype=float) - body_xy)
        signs[signs == 0.0] = 1.0
        best_normal = -signs / np.sqrt(2.0)
    return body_xy + shift_distance * best_normal, best_normal


class CrawlGaitPlanner:
    """Stateful body/foot-anchor planner for a finite static crawl."""

    phases = tuple(PHASE_FRACTIONS)

    def __init__(
        self,
        controller,
        shift_distance,
        step_length,
        swing_height,
        leg_cycle,
        gait_cycles,
        observe_phases=False,
        unload_tau_trip=DEFAULT_UNLOAD_TAU_TRIP_NM,
    ):
        self.controller = controller
        self.shift_distance = float(shift_distance)
        self.step_length = float(step_length)
        self.swing_height = float(swing_height)
        self.leg_cycle = float(leg_cycle)
        self.gait_cycles = int(gait_cycles)
        self.observe_phases = bool(observe_phases)
        self.unload_tau_trip = float(unload_tau_trip)
        self.total_steps = len(GAIT_ORDER) * self.gait_cycles
        self.nominal_xy = {
            leg: np.asarray(controller.final_foot[leg][:2], dtype=float).copy()
            for leg in LEGS
        }
        self.nominal_z = {
            leg: float(controller.final_foot[leg][2]) for leg in LEGS
        }
        self.active = False
        self.finished = False
        self.aborted = False
        self.abort_reason = None
        self.paused = False
        self.mandatory_hold = False
        self.phase = "IDLE"
        self.phase_started_at = 0.0
        self.phase_settle_since = None
        self.phase_wait_started = None
        self.gate_timeout_reason = None
        self.step_index = 0
        self.world_foot_xy = {}
        self.body_xy = np.zeros(2)
        self.shifted_body_xy = np.zeros(2)
        self.next_body_xy = np.zeros(2)
        self.new_swing_anchor_xy = np.zeros(2)
        self.touchdown_plane_baseline_m = 0.0
        self.last_unload_tau_nm = np.nan
        self.last_unload_rel_sag_m = np.nan
        self.shift_normal = np.zeros(2)
        self.base_shifted_body_xy = np.zeros(2)
        self.unload_extra_m = 0.0
        self.unload_extra_cap_m = None
        self.unload_growth_dir = np.zeros(2)
        self.unload_grow_at = None
        self.swing_drag_streak = 0

    @property
    def swing_leg(self):
        if not self.active:
            return None
        return GAIT_ORDER[self.step_index % len(GAIT_ORDER)]

    @property
    def airborne(self):
        return self.active and self.phase in SWING_PHASES

    @property
    def three_leg_support(self):
        return self.active and self.phase in REDUCED_SUPPORT_PHASES

    @property
    def gait_mode(self):
        return "SWING_TEST" if abs(self.step_length) < 1.0e-12 else "WALK"

    @property
    def motion_kind(self):
        return PHASE_MOTION_KIND.get(self.phase, "UNKNOWN")

    @property
    def diagnostic_context(self):
        leg = self.swing_leg or "-"
        step = min(self.step_index + 1, self.total_steps)
        return (
            f"mode={self.gait_mode} step={step}/{self.total_steps} leg={leg} "
            f"phase={self.phase} motion={self.motion_kind}"
        )

    @property
    def trim_stance_legs(self):
        if self.active and self.phase in REDUCED_SUPPORT_PHASES:
            return set(LEGS) - {self.swing_leg}
        return set(LEGS)

    def duration(self, phase):
        return PHASE_FRACTIONS[phase] * self.leg_cycle

    def start(self, now):
        # Re-snapshot the hold targets: an aborted batch commits its end
        # state into controller.final_foot, and a restart must begin there.
        self.nominal_xy = {
            leg: np.asarray(
                self.controller.final_foot[leg][:2], dtype=float
            ).copy()
            for leg in LEGS
        }
        self.nominal_z = {
            leg: float(self.controller.final_foot[leg][2]) for leg in LEGS
        }
        self.active = True
        self.finished = False
        self.aborted = False
        self.abort_reason = None
        self.paused = False
        self.mandatory_hold = False
        self.step_index = 0
        self.world_foot_xy = {
            leg: xy.copy() for leg, xy in self.nominal_xy.items()
        }
        self.body_xy = np.zeros(2)
        self.touchdown_plane_baseline_m = 0.0
        self._begin_step(float(now), pause=False)

    def _begin_step(self, now, pause=False, mandatory_pause=False):
        leg = GAIT_ORDER[self.step_index % len(GAIT_ORDER)]
        # Shift the CoM away from the nearest edge of the ACTUAL stance
        # triangle (the three planted feet), so the shift adapts to whatever
        # geometry the previous steps left behind.
        stance_pts = [self.world_foot_xy[s] for s in LEGS if s != leg]
        self.shifted_body_xy, self.shift_normal = adaptive_body_shift(
            self.world_foot_xy[leg], stance_pts, self.body_xy, self.shift_distance
        )
        self.base_shifted_body_xy = self.shifted_body_xy.copy()
        self.unload_extra_m = 0.0
        self.unload_extra_cap_m = None
        self.unload_growth_dir = np.asarray(self.shift_normal, dtype=float)
        self.unload_grow_at = None
        self.swing_drag_streak = 0
        self.next_body_xy = self.body_xy + np.array(
            [self.step_length / len(GAIT_ORDER), 0.0]
        )
        self.new_swing_anchor_xy = self.world_foot_xy[leg] + np.array(
            [self.step_length, 0.0]
        )
        self.phase = "SHIFT"
        self.phase_started_at = float(now)
        self.phase_settle_since = None
        self.phase_wait_started = None
        self.gate_timeout_reason = None
        # A mandatory pause holds at the SHIFT start, whose target is the
        # neutral four-foot stance (elapsed 0 => zero body shift), so the
        # robot waits flat and stable until ENTER.
        self.mandatory_hold = bool(mandatory_pause)
        self.paused = bool(mandatory_pause or (self.observe_phases and pause))

    def _set_phase(self, phase, now, pause=True):
        self.phase = phase
        self.phase_started_at = float(now)
        self.phase_settle_since = None
        self.phase_wait_started = None
        self.gate_timeout_reason = None
        self.mandatory_hold = False
        self.paused = bool(self.observe_phases and pause)

    def _begin_abort(self, now, q, reason):
        """Recover to a four-foot neutral hold instead of stopping the run.

        Every gated phase completes its commanded motion before its gate can
        time out, so aborts always start from a well-defined endpoint: body
        at the shifted position, swing foot planted (UNLOAD) or commanded to
        the ground (LOWER).  The chain is ABORT_LOAD (restore the mg/4 share
        on the standing swing foot) then ABORT_RECENTER (body back to the
        pre-step neutral), after which the batch ends aborted.
        """
        self.aborted = True
        self.abort_reason = reason
        leg = self.swing_leg
        # Drag aborts (LIFT/SWING) fire precisely because the foot is on the
        # ground, so they take the same commit-and-reload path as LOWER.
        if self.phase in ("LIFT", "SWING", "LOWER") and q is not None:
            # The foot physically stands wherever the swing ended; commit the
            # measured position as its anchor so no target drags it sideways.
            feet = self.actual_foot_positions(q)
            self.world_foot_xy[leg] = feet[leg][:2] + self.shifted_body_xy
        if self.phase in ("UNLOAD", "LIFT", "SWING", "LOWER"):
            self._set_phase("ABORT_LOAD", now, pause=False)
        else:
            self._set_phase("ABORT_RECENTER", now, pause=False)
        return (
            f"ABORT step {min(self.step_index + 1, self.total_steps)} "
            f"({leg}): {reason}; reloading and returning to neutral"
        )

    def _finish(self, aborted, message):
        """End the batch and commit the commanded state as the new hold."""
        for leg in LEGS:
            self.controller.final_foot[leg] = np.array(
                [
                    *(self.world_foot_xy[leg] - self.body_xy),
                    self.nominal_z[leg],
                ]
            )
        self.active = False
        self.finished = True
        self.aborted = bool(aborted)
        self.paused = False
        self.phase = "IDLE"
        return message

    def resume(self, now):
        if not self.active:
            return False, "crawl is not active"
        if not self.paused:
            return False, f"{self.phase} is already running"
        self.paused = False
        self.mandatory_hold = False
        self.phase_started_at = float(now)
        self.phase_settle_since = None
        self.phase_wait_started = None
        return True, (
            f"ENTER accepted: resuming {self.diagnostic_context}"
        )

    def _smooth(self, now):
        elapsed = 0.0 if self.paused else float(now) - self.phase_started_at
        return base.HardwareStandController._cubic(
            elapsed, self.duration(self.phase)
        )

    def target(self, leg, now):
        """Return desired position, velocity, weight share, trim scale."""
        if not self.active:
            target = np.asarray(self.controller.final_foot[leg], dtype=float)
            return target.copy(), np.zeros(3), 0.25, 1.0

        phase = self.phase
        swing = self.swing_leg
        s, ds = self._smooth(now)
        anchor = self.world_foot_xy[leg]
        z_ground = self.nominal_z[leg]
        desired = np.array([*(anchor - self.body_xy), z_ground])
        velocity = np.zeros(3)
        share = 0.25
        trim_scale = 1.0

        if phase == "SHIFT":
            delta_body = self.shifted_body_xy - self.body_xy
            desired[:2] = anchor - (self.body_xy + s * delta_body)
            velocity[:2] = -ds * delta_body
        elif phase == "UNLOAD":
            # Feet frozen at the shifted stance; only the weight moves.  The
            # swing foot stays planted so the gate can measure it unloaded.
            desired[:2] = anchor - self.shifted_body_xy
            if leg == swing:
                share = 0.25 * (1.0 - s)
                trim_scale = 1.0 - s
            else:
                share = 0.25 + s * (1.0 / 3.0 - 0.25)
        elif phase == "LIFT":
            desired[:2] = anchor - self.shifted_body_xy
            if leg == swing:
                desired[2] = z_ground + s * self.swing_height
                velocity[2] = ds * self.swing_height
                share = 0.0
                trim_scale = 0.0
            else:
                share = 1.0 / 3.0
        elif phase == "SWING":
            desired[:2] = anchor - self.shifted_body_xy
            if leg == swing:
                delta = self.new_swing_anchor_xy - anchor
                desired[:2] += s * delta
                velocity[:2] = ds * delta
                desired[2] = z_ground + self.swing_height
                share = 0.0
                trim_scale = 0.0
            else:
                share = 1.0 / 3.0
        elif phase == "LOWER":
            desired[:2] = anchor - self.shifted_body_xy
            if leg == swing:
                desired[:2] = self.new_swing_anchor_xy - self.shifted_body_xy
                desired[2] = z_ground + (1.0 - s) * self.swing_height
                velocity[2] = -ds * self.swing_height
                share = 0.0
                trim_scale = 0.0
            else:
                share = 1.0 / 3.0
        elif phase == "LOAD":
            desired[:2] = anchor - self.shifted_body_xy
            if leg == swing:
                share = 0.25 * s
                trim_scale = s
            else:
                share = 1.0 / 3.0 + s * (0.25 - 1.0 / 3.0)
        elif phase == "RECENTER":
            delta_body = self.next_body_xy - self.shifted_body_xy
            desired[:2] = anchor - (self.shifted_body_xy + s * delta_body)
            velocity[:2] = -ds * delta_body
        elif phase == "ABORT_LOAD":
            # Swing foot is planted (its anchor already reflects reality);
            # restore its quarter share where it stands.
            desired[:2] = anchor - self.shifted_body_xy
            if leg == swing:
                share = s * 0.25
                trim_scale = s
            else:
                share = 1.0 / 3.0 + s * (0.25 - 1.0 / 3.0)
        elif phase == "ABORT_RECENTER":
            delta_body = self.body_xy - self.shifted_body_xy
            desired[:2] = anchor - (self.shifted_body_xy + s * delta_body)
            velocity[:2] = -ds * delta_body
        else:
            raise RuntimeError(f"unknown crawl phase {phase}")
        return desired, velocity, share, trim_scale

    def actual_foot_positions(self, q):
        q = np.asarray(q, dtype=float)
        return {
            leg: dog5_kinematics.foot_position(
                leg, q[3 * index : 3 * index + 3]
            )
            for index, leg in enumerate(LEGS)
        }

    def actual_support_margin(self, q):
        feet = self.actual_foot_positions(q)
        stance = [leg for leg in LEGS if leg != self.swing_leg]
        return support_triangle_margin([feet[leg][:2] for leg in stance])

    def _tracking_metrics(self, q, qd, now, legs):
        feet = self.actual_foot_positions(q)
        max_error = 0.0
        max_speed = 0.0
        qd = np.asarray(qd, dtype=float)
        for leg in legs:
            desired, _, _, _ = self.target(leg, now)
            max_error = max(max_error, float(np.linalg.norm(desired - feet[leg])))
            index = LEGS.index(leg)
            max_speed = max(
                max_speed,
                float(np.max(np.abs(qd[3 * index : 3 * index + 3]))),
            )
        return max_error, max_speed

    def _touchdown_metrics(self, q, qd, now, leg):
        """Return tracking, stance-plane, and speed metrics at touchdown."""
        feet = self.actual_foot_positions(q)
        desired, _, _, _ = self.target(leg, now)
        error = desired - feet[leg]
        index = LEGS.index(leg)
        qd_leg = np.asarray(qd)[3 * index : 3 * index + 3]
        joint_speed = float(np.max(np.abs(qd_leg)))
        foot_speed = float(
            np.linalg.norm(
                dog5_kinematics.foot_jacobian(leg, q[3 * index : 3 * index + 3])
                @ qd_leg
            )
        )
        stance_points = [feet[name] for name in LEGS if name != leg]
        foot_plane_distance = signed_distance_to_stance_plane(
            feet[leg], stance_points
        )
        target_plane_distance = signed_distance_to_stance_plane(
            desired, stance_points
        )
        return (
            float(np.linalg.norm(error)),
            float(np.linalg.norm(error[:2])),
            float(error[2]),
            foot_plane_distance - self.touchdown_plane_baseline_m,
            target_plane_distance - self.touchdown_plane_baseline_m,
            joint_speed,
            foot_speed,
        )

    def _settled(self, now, condition, dwell, timeout, error_message):
        """Sustained-condition gate.  A timeout no longer raises: it stores
        ``gate_timeout_reason`` so the caller can start a graceful abort."""
        if self.phase_wait_started is None:
            self.phase_wait_started = float(now)
        if condition:
            if self.phase_settle_since is None:
                self.phase_settle_since = float(now)
            if now - self.phase_settle_since >= dwell:
                return True
        else:
            self.phase_settle_since = None
        if now - self.phase_wait_started > timeout:
            self.gate_timeout_reason = error_message
        return False

    def _unload_metrics(self, q, measured_tau):
        """Swing-leg gravity-compensated pitch/knee torque and relative sag.

        The leg's own link weight holds ~0.4 N*m on the pitch joint even when
        fully airborne -- nearly the whole 0.45 N*m trip -- so the external
        (ground) torque is what must be compared, not the raw feedback.
        """
        leg = self.swing_leg
        index = 3 * LEGS.index(leg)
        if measured_tau is None:
            tau_worst = 0.0
        else:
            measured_tau = np.asarray(measured_tau, dtype=float)
            external = measured_tau[
                index : index + 3
            ] - dog5_kinematics.leg_gravity_torque(
                leg, np.asarray(q, dtype=float)[index : index + 3]
            )
            tau_worst = float(max(abs(external[1]), abs(external[2])))
        sag = self.controller.last_leg_sag_m
        stance_mean = float(
            np.mean([sag[name] for name in LEGS if name != leg])
        )
        rel_sag = float(sag[leg]) - stance_mean
        return tau_worst, rel_sag

    def _direction_cap(self, q, low, high, direction):
        """Furthest extra shift along ``direction`` that stays safe.

        Two bounds, both cheap enough for the control loop (no IK): the
        planned support margin of the stance triangle must stay above the
        liftoff floor, and the linearized joint excursion (dq = J^-1 dx per
        leg, exact enough over <= 20 mm) must stay inside the soft limits
        with margin.
        """
        cap = UNLOAD_EXTRA_MAX_M
        move_dir = np.array([-direction[0], -direction[1], 0.0])
        for index, leg in enumerate(LEGS):
            section = slice(3 * index, 3 * index + 3)
            jacobian = dog5_kinematics.foot_jacobian(leg, q[section])
            if (
                np.linalg.svd(jacobian, compute_uv=False)[-1]
                < recorded.MIN_JACOBIAN_SINGULAR
            ):
                return 0.0
            # Feet targets move by -direction per metre of body shift.
            dq_per_m = np.linalg.solve(jacobian, move_dir)
            for joint in range(3):
                rate = float(dq_per_m[joint])
                if rate > 1.0e-9:
                    room = (
                        high[section][joint] - UNLOAD_LIMIT_MARGIN_RAD
                    ) - q[section][joint]
                    cap = min(cap, max(0.0, room / rate))
                elif rate < -1.0e-9:
                    room = q[section][joint] - (
                        low[section][joint] + UNLOAD_LIMIT_MARGIN_RAD
                    )
                    cap = min(cap, max(0.0, room / -rate))
        stance_pts = [
            self.world_foot_xy[name] for name in LEGS if name != self.swing_leg
        ]
        extra = 0.002
        while extra <= cap + 1.0e-9:
            candidate = self.base_shifted_body_xy + extra * np.asarray(
                direction
            )
            if (
                support_triangle_margin(stance_pts, candidate)
                < MIN_LIFTOFF_MARGIN_M
            ):
                cap = extra - 0.002
                break
            extra += 0.002
        return max(0.0, cap)

    def _unload_growth_plan(self, q):
        """Pick the growth direction that buys the most support margin.

        The inward normal is the most efficient direction per millimetre, but
        at the shifted stance the abduction joints are within ~1.4 deg of
        their soft limit, so the lateral budget can be a few mm at best.  The
        longitudinal axis has huge pitch/knee headroom and still gains margin
        at normal_x per metre, so a mostly-x fallback often buys MORE total
        margin.  Score = cap x (direction . normal); best wins.
        """
        q = np.asarray(q, dtype=float)
        low, high = base.soft_limits()
        candidates = [np.asarray(self.shift_normal, dtype=float)]
        if abs(self.shift_normal[0]) > 1.0e-6:
            longitudinal = np.array(
                [np.sign(self.shift_normal[0]), 0.0]
            )
            blend = self.shift_normal + longitudinal
            candidates.append(blend / np.linalg.norm(blend))
            candidates.append(longitudinal)
        best_dir = candidates[0]
        best_cap = 0.0
        best_score = -np.inf
        for direction in candidates:
            cap = self._direction_cap(q, low, high, direction)
            score = cap * float(direction @ self.shift_normal)
            if score > best_score:
                best_dir, best_cap, best_score = direction, cap, score
        return best_dir, best_cap

    def update(self, now, q, qd, measured_tau=None):
        """Advance phases only after geometric/tracking gates are satisfied."""
        if not self.active:
            return None
        now = float(now)
        # Drag watch: an airborne leg should measure ~0 external torque.
        # Sustained torque means the foot is scraping the ground (the CoM
        # never truly left it, or the trunk sagged back onto it) -- abort
        # gracefully instead of dragging.  Runs every tick, pause included,
        # because the foot is (supposedly) airborne the whole time.
        if self.phase in ("LIFT", "SWING") and measured_tau is not None:
            tau_worst, rel_sag = self._unload_metrics(q, measured_tau)
            self.last_unload_tau_nm = tau_worst
            self.last_unload_rel_sag_m = rel_sag
            if tau_worst > SWING_DRAG_TAU_TRIP_NM:
                self.swing_drag_streak += 1
                if self.swing_drag_streak >= SWING_DRAG_STREAK:
                    return self._begin_abort(
                        now,
                        q,
                        f"{self.swing_leg} foot still loaded while airborne "
                        f"(|tau_ext|={tau_worst:.2f} N*m > "
                        f"{SWING_DRAG_TAU_TRIP_NM:.2f}): dragging",
                    )
            else:
                self.swing_drag_streak = 0
        if self.paused:
            return None
        if now - self.phase_started_at < self.duration(self.phase):
            return None

        leg = self.swing_leg
        if self.phase == "SHIFT":
            margin = self.actual_support_margin(q)
            error, speed = self._tracking_metrics(q, qd, now, LEGS)
            good = (
                margin >= MIN_LIFTOFF_MARGIN_M
                and error <= SHIFT_CART_TOL_M
                and speed <= SHIFT_QD_TOL_RAD_S
            )
            if not self._settled(
                now,
                good,
                SHIFT_SETTLE_S,
                SHIFT_SETTLE_TIMEOUT_S,
                f"{leg} liftoff gate timed out: margin={1000.0 * margin:.1f} "
                f"mm, cart_err={error:.3f} m, max|qd|={speed:.2f} rad/s",
            ):
                if self.gate_timeout_reason:
                    return self._begin_abort(now, q, self.gate_timeout_reason)
                return None
            self._set_phase("UNLOAD", now)
            return (
                f"{leg} shift settled: encoder support margin "
                f"{1000.0 * margin:.1f} mm; unloading {leg} for the "
                "pre-lift gate"
            )

        if self.phase == "UNLOAD":
            tau_worst, rel_sag = self._unload_metrics(q, measured_tau)
            self.last_unload_tau_nm = tau_worst
            self.last_unload_rel_sag_m = rel_sag
            good = (
                tau_worst < self.unload_tau_trip
                and abs(rel_sag) < UNLOAD_REL_SAG_MAX_M
            )
            # Force-feedback shift: the planned shift completed but the leg
            # still measures loaded (real CoM is not where geometry assumes),
            # so keep moving the body along the same inward normal until the
            # torque drops, the safe budget runs out, or the gate times out.
            if self.unload_extra_cap_m is None:
                (
                    self.unload_growth_dir,
                    self.unload_extra_cap_m,
                ) = self._unload_growth_plan(q)
                self.unload_grow_at = now
            if not good and self.unload_extra_m < self.unload_extra_cap_m:
                grow_dt = max(0.0, now - self.unload_grow_at)
                self.unload_extra_m = min(
                    self.unload_extra_m + UNLOAD_EXTRA_RATE_M_S * grow_dt,
                    self.unload_extra_cap_m,
                )
                self.shifted_body_xy = (
                    self.base_shifted_body_xy
                    + self.unload_extra_m * self.unload_growth_dir
                )
            self.unload_grow_at = now
            if not self._settled(
                now,
                good,
                UNLOAD_GATE_SETTLE_S,
                UNLOAD_GATE_TIMEOUT_S,
                f"{leg} unload gate timed out: pitch/knee |tau_ext|="
                f"{tau_worst:.2f} N*m (trip {self.unload_tau_trip:.2f}), "
                f"rel_sag={1000.0 * rel_sag:+.1f} mm (max "
                f"{1000.0 * UNLOAD_REL_SAG_MAX_M:.0f}), extra shift "
                f"{1000.0 * self.unload_extra_m:.1f}/"
                f"{1000.0 * self.unload_extra_cap_m:.1f} mm exhausted -- the "
                "leg still carries load beyond the safe shift budget",
            ):
                if self.gate_timeout_reason:
                    return self._begin_abort(now, q, self.gate_timeout_reason)
                return None
            feet = self.actual_foot_positions(q)
            stance_points = [feet[name] for name in LEGS if name != leg]
            self.touchdown_plane_baseline_m = signed_distance_to_stance_plane(
                feet[leg], stance_points
            )
            self._set_phase("LIFT", now)
            return (
                f"{leg} measured unloaded (|tau_ext|={tau_worst:.2f} N*m, "
                f"rel_sag={1000.0 * rel_sag:+.1f} mm, extra shift "
                f"{1000.0 * self.unload_extra_m:.1f} mm); liftoff enabled, "
                f"plane baseline "
                f"{1000.0 * self.touchdown_plane_baseline_m:+.1f} mm"
            )

        if self.phase == "LIFT":
            self._set_phase("SWING", now)
            return f"{leg} lifted; advancing foot {1000.0 * self.step_length:.0f} mm"
        if self.phase == "SWING":
            self._set_phase("LOWER", now)
            return f"{leg} advance complete; lowering"
        if self.phase == "LOWER":
            (
                error,
                xy_error,
                z_error,
                plane_error,
                target_plane_error,
                joint_speed,
                foot_speed,
            ) = self._touchdown_metrics(q, qd, now, leg)
            good, tracked, _ = touchdown_accepted(
                error,
                xy_error,
                plane_error,
                target_plane_error,
                joint_speed,
                foot_speed,
            )
            if not self._settled(
                now,
                good,
                TOUCHDOWN_SETTLE_S,
                TOUCHDOWN_TIMEOUT_S,
                f"{leg} touchdown timed out: err="
                f"({1000.0 * xy_error:.1f} mm xy, "
                f"{1000.0 * z_error:+.1f} mm z, "
                f"{1000.0 * error:.1f} mm total), "
                f"plane={1000.0 * plane_error:+.1f} mm, "
                f"target_plane={1000.0 * target_plane_error:+.1f} mm, "
                f"foot_speed={1000.0 * foot_speed:.1f} mm/s, "
                f"max|qd|={joint_speed:.2f} rad/s",
            ):
                if self.gate_timeout_reason:
                    return self._begin_abort(now, q, self.gate_timeout_reason)
                return None
            # Commit the MEASURED foot position as the anchor: an early
            # contact pinned by friction lands short of the commanded
            # advance, and idealized bookkeeping would drag every later
            # target by that error.
            feet = self.actual_foot_positions(q)
            measured_anchor = feet[leg][:2] + self.shifted_body_xy
            shortfall = float(
                np.linalg.norm(measured_anchor - self.new_swing_anchor_xy)
            )
            self.world_foot_xy[leg] = measured_anchor
            self._set_phase("LOAD", now)
            touchdown_mode = "tracked" if tracked else "ground contact inferred"
            return (
                f"{leg} touchdown {touchdown_mode} "
                f"(xy={1000.0 * xy_error:.1f} mm, "
                f"z={1000.0 * z_error:+.1f} mm, "
                f"plane={1000.0 * plane_error:+.1f} mm); anchored at the "
                f"measured foot ({1000.0 * shortfall:.1f} mm from planned); "
                "restoring stance load and trim"
            )
        if self.phase == "LOAD":
            self._set_phase("RECENTER", now)
            return f"{leg} loaded; recentering body with fixed stance anchors"
        if self.phase == "RECENTER":
            error, speed = self._tracking_metrics(q, qd, now, LEGS)
            good = (
                error <= RECENTER_CART_TOL_M
                and speed <= RECENTER_QD_TOL_RAD_S
            )
            if not self._settled(
                now,
                good,
                RECENTER_SETTLE_S,
                RECENTER_TIMEOUT_S,
                f"{leg} recenter timed out: cart_err={error:.3f} m, "
                f"max|qd|={speed:.2f} rad/s",
            ):
                if self.gate_timeout_reason:
                    # All four feet are planted and the body target is already
                    # at its endpoint: adopt it as the new neutral and end the
                    # batch instead of moving anything else.
                    reason = self.gate_timeout_reason
                    self.aborted = True
                    self.abort_reason = reason
                    self.body_xy = self.next_body_xy.copy()
                    return self._finish(
                        True,
                        f"ABORT: {reason}; holding the recentered stance "
                        "(batch ended early)",
                    )
                return None
            self.body_xy = self.next_body_xy.copy()
            self.step_index += 1
            if self.step_index >= self.total_steps:
                return self._finish(
                    False,
                    f"crawl batch complete ({self.gait_cycles} gait "
                    "cycle(s)); neutral Cartesian HOLD restored",
                )
            completed = leg
            # Hold flat on four feet between diagonal pairs (after RR FL, and
            # after RL FR) and wait for ENTER, regardless of observe mode.
            mandatory = self.step_index % MANDATORY_HOLD_EVERY == 0
            self._begin_step(now, pause=True, mandatory_pause=mandatory)
            if mandatory:
                return (
                    f"{completed} step complete; diagonal pair done -- "
                    "HOLDING on four feet, press ENTER for the next pair "
                    f"(starting with {self.swing_leg})"
                )
            return f"{completed} step complete; shifting for {self.swing_leg}"

        if self.phase == "ABORT_LOAD":
            self._set_phase("ABORT_RECENTER", now, pause=False)
            return (
                f"{leg} reloaded where it stands; returning body to neutral"
            )
        if self.phase == "ABORT_RECENTER":
            return self._finish(
                True,
                f"step ABORTED ({self.abort_reason}); four-foot neutral "
                "hold restored -- ENTER starts a fresh batch",
            )
        raise RuntimeError(f"unknown crawl phase {self.phase}")


class CrawlController(inplace.InPlaceStandController):
    """In-place stand controller plus stance-aware crawl compliance."""

    def __init__(self, cart_gain_scale, support_scale, stand_height, travel_scale=1.0):
        super().__init__(cart_gain_scale, support_scale, stand_height, travel_scale)
        self.support_scale = float(support_scale)
        self.weight_n = base.DOG5_MASS_KG * 9.81

    def compute_crawl(self, q, qd, planner, now):
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        tau = np.zeros(N_JOINTS)
        max_error = 0.0
        min_singular = np.inf
        max_force = 0.0

        for leg_index, leg in enumerate(LEGS):
            section = slice(3 * leg_index, 3 * leg_index + 3)
            q_leg, qd_leg = q[section], qd[section]
            foot = dog5_kinematics.foot_position(leg, q_leg)
            jacobian = dog5_kinematics.foot_jacobian(leg, q_leg)
            singular = float(np.min(np.linalg.svd(jacobian, compute_uv=False)))
            min_singular = min(min_singular, singular)
            if singular < recorded.MIN_JACOBIAN_SINGULAR:
                raise RuntimeError(
                    f"{leg} Jacobian near singular: sigma_min={singular:.4f} "
                    f"< {recorded.MIN_JACOBIAN_SINGULAR:.4f} m/rad"
                )

            desired, desired_velocity, support_share, trim_scale = planner.target(
                leg, now
            )
            error = desired - foot
            self.last_leg_sag_m[leg] = -float(error[2])
            error_norm = float(np.linalg.norm(error))
            max_error = max(max_error, error_norm)
            if error_norm > recorded.MAX_CART_ERROR_M:
                raise RuntimeError(
                    f"{leg} Cartesian error {error_norm:.3f} m exceeds "
                    f"{recorded.MAX_CART_ERROR_M:.3f} m"
                )

            foot_velocity = jacobian @ qd_leg
            force = self.kp_cart @ error + self.kd_cart @ (
                desired_velocity - foot_velocity
            )
            force[2] -= (
                self.support_scale * self.weight_n * support_share
                + trim_scale * self.z_trim_n[leg]
            )
            force_norm = float(np.linalg.norm(force))
            clipped = force_norm > recorded.MAX_CART_FORCE_N
            self.last_leg_force_clipped[leg] = clipped
            if clipped:
                force *= recorded.MAX_CART_FORCE_N / force_norm
                force_norm = recorded.MAX_CART_FORCE_N
            max_force = max(max_force, force_norm)
            tau[section] = jacobian.T @ force - stand_dog5.KD_JOINT_STAND * qd_leg

        self.last_cart_error = max_error
        self.last_min_singular = min_singular
        self.last_max_force = max_force
        if not np.all(np.isfinite(tau)):
            raise RuntimeError("crawl controller produced non-finite torque")
        return tau

    def update_gait_trim(self, now, stance_legs, saturated_legs=None):
        """Integrate stance trims and freeze (without clearing) swing trim."""
        now = float(now)
        if self._trim_last_time is None:
            self._trim_last_time = now
            return
        dt = float(np.clip(now - self._trim_last_time, 0.0, 0.1))
        self._trim_last_time = now
        stance = set(stance_legs)
        saturated = saturated_legs or set()
        for leg in LEGS:
            if leg not in stance:
                continue
            sag = self.last_leg_sag_m[leg]
            if abs(sag) < inplace.TRIM_DEADBAND_M:
                continue
            increment = inplace.TRIM_KI_N_PER_M_S * sag * dt
            trim = self.z_trim_n[leg]
            if leg in saturated and abs(trim + increment) > abs(trim):
                continue
            self.z_trim_n[leg] = float(
                np.clip(
                    trim + increment,
                    -inplace.TRIM_CLAMP_N,
                    inplace.TRIM_CLAMP_N,
                )
            )


class CrawlSequence(inplace.InPlaceStandSequence):
    """Stand sequence with an ENTER-gated finite crawl after HOLD."""

    def __init__(self, now, start_q, planner, travel_scale=1.0):
        super().__init__(now, start_q, travel_scale)
        self.planner = planner

    def update(self, now, q=None, cart_error=None, qd=None, measured_tau=None):
        if self.stage == "CRAWL":
            if q is None or qd is None:
                return None
            event = self.planner.update(now, q, qd, measured_tau=measured_tau)
            if self.planner.finished:
                self.stage = "CRAWL_HOLD"
                self.wait_since = float(now)
            return event
        return super().update(now, q=q, cart_error=cart_error, qd=qd)

    def request_next(self, now, q, qd, healthy):
        if self.stage == "CRAWL" and self.planner.paused:
            if not healthy:
                return False, "phase resume refused: motor latch/fault present."
            if self.planner.phase == "LOAD":
                metrics = self.planner._touchdown_metrics(
                    q, qd, now, self.planner.swing_leg
                )
                accepted = touchdown_accepted(
                    metrics[0],
                    metrics[1],
                    metrics[3],
                    metrics[4],
                    metrics[5],
                    metrics[6],
                )[0]
                if not accepted:
                    return False, (
                        "phase resume refused: touchdown geometry no longer "
                        f"valid (xy={1000.0 * metrics[1]:.1f} mm, "
                        f"plane={1000.0 * metrics[3]:+.1f} mm)"
                    )
            else:
                error, speed = self.planner._tracking_metrics(q, qd, now, LEGS)
                if error > OBSERVE_RESUME_CART_TOL_M:
                    return False, (
                        f"phase resume refused: cart_err {error:.3f} m > "
                        f"{OBSERVE_RESUME_CART_TOL_M:.3f} m"
                    )
                if speed > OBSERVE_RESUME_QD_TOL_RAD_S:
                    return False, (
                        f"phase resume refused: max |qd| {speed:.2f} rad/s > "
                        f"{OBSERVE_RESUME_QD_TOL_RAD_S:.2f} rad/s"
                    )
            return self.planner.resume(now)
        if self.stage in ("HOLD", "CRAWL_HOLD"):
            if self.travel_scale < 1.0:
                return False, "CRAWL refused: a full-height HOLD is required."
            if not healthy:
                return False, "CRAWL refused: motor latch/fault present."
            speed = float(np.max(np.abs(qd)))
            if speed > recorded.FINAL_QD_TOL:
                return False, (
                    f"CRAWL refused: max |qd| {speed:.2f} rad/s > "
                    f"{recorded.FINAL_QD_TOL:.2f}."
                )
            self.planner.start(now)
            self.stage = "CRAWL"
            self.started_at = float(now)
            self.wait_since = None
            return True, (
                f"ENTER accepted: {self.planner.gait_mode} starting with "
                f"{self.planner.swing_leg}; {self.planner.total_steps} steps "
                f"planned, phase={self.planner.phase} "
                f"motion={self.planner.motion_kind}"
            )
        if self.stage in ("HOLD_PARTIAL", "HOLD_SAG"):
            return False, "CRAWL refused: full commanded stand must settle first."
        return super().request_next(now, q, qd, healthy)

    def request_park(self, now, healthy):
        if self.stage == "CRAWL":
            return False, "PARK refused while crawling; X stops immediately."
        if self.stage == "CRAWL_HOLD":
            if not healthy:
                return False, "PARK refused: motor latch/fault present."
            self.stage = "HOLD"
        return super().request_park(now, healthy)


def _ik_to_target(leg, q_start, target):
    q = np.asarray(q_start, dtype=float).copy()
    for _ in range(100):
        error = np.asarray(target) - dog5_kinematics.foot_position(leg, q)
        jacobian = dog5_kinematics.foot_jacobian(leg, q)
        q += 0.35 * jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 1.0e-5 * np.eye(3), error
        )
    return q


def _planned_gait_segments(controller, shift_distance, step_length, swing_height):
    """Yield exact one-cycle target segments for validation/self-test."""
    anchors = {
        leg: np.asarray(controller.final_foot[leg][:2]).copy() for leg in LEGS
    }
    z = {leg: float(controller.final_foot[leg][2]) for leg in LEGS}
    body = np.zeros(2)
    for step_index, swing in enumerate(GAIT_ORDER):
        stance_pts = [anchors[s] for s in LEGS if s != swing]
        shifted, _ = adaptive_body_shift(
            anchors[swing], stance_pts, body, shift_distance
        )
        body_end = body + np.array([step_length / len(GAIT_ORDER), 0.0])
        new_anchor = anchors[swing] + np.array([step_length, 0.0])

        neutral = {leg: np.array([*(anchors[leg] - body), z[leg]]) for leg in LEGS}
        shifted_targets = {
            leg: np.array([*(anchors[leg] - shifted), z[leg]]) for leg in LEGS
        }
        yield "SHIFT", swing, neutral, shifted_targets

        lifted = {leg: target.copy() for leg, target in shifted_targets.items()}
        lifted[swing][2] += swing_height
        yield "LIFT", swing, shifted_targets, lifted

        advanced = {leg: target.copy() for leg, target in lifted.items()}
        advanced[swing][:2] = new_anchor - shifted
        yield "SWING", swing, lifted, advanced

        lowered = {leg: target.copy() for leg, target in advanced.items()}
        lowered[swing][2] = z[swing]
        yield "LOWER", swing, advanced, lowered

        anchors[swing] = new_anchor
        recentered = {
            leg: np.array([*(anchors[leg] - body_end), z[leg]]) for leg in LEGS
        }
        yield "RECENTER", swing, lowered, recentered
        body = body_end


def validate_crawl_configuration(controller, shift_distance, step_length, swing_height):
    """Validate stand/gait reachability, support margin, and mg/3 torque."""
    inplace.validate_inplace_configuration(controller)
    low, high = base.soft_limits()
    q_path = {}
    for index, leg in enumerate(LEGS):
        section = slice(3 * index, 3 * index + 3)
        q_path[leg] = _ik_to_target(
            leg, Q_RECORDED_CROUCH[section], controller.final_foot[leg]
        )

    max_static_tau = 0.0
    minimum_planned_margin = np.inf
    for phase, swing, starts, ends in _planned_gait_segments(
        controller, shift_distance, step_length, swing_height
    ):
        if phase == "SHIFT":
            stance = [leg for leg in LEGS if leg != swing]
            margin = support_triangle_margin([ends[leg][:2] for leg in stance])
            minimum_planned_margin = min(minimum_planned_margin, margin)
            if margin < MIN_LIFTOFF_MARGIN_M:
                raise ValueError(
                    f"{swing} planned support margin {1000.0 * margin:.1f} mm "
                    f"is below {1000.0 * MIN_LIFTOFF_MARGIN_M:.0f} mm"
                )

        for progress in np.linspace(0.1, 1.0, 10):
            for index, leg in enumerate(LEGS):
                target = starts[leg] + progress * (ends[leg] - starts[leg])
                q_path[leg] = _ik_to_target(leg, q_path[leg], target)
                residual = float(
                    np.linalg.norm(
                        target - dog5_kinematics.foot_position(leg, q_path[leg])
                    )
                )
                singular = float(
                    np.min(
                        np.linalg.svd(
                            dog5_kinematics.foot_jacobian(leg, q_path[leg]),
                            compute_uv=False,
                        )
                    )
                )
                section = slice(3 * index, 3 * index + 3)
                if residual > 1.0e-4:
                    raise ValueError(
                        f"{leg} {phase} path unreachable: residual={residual:.6f} m"
                    )
                if np.any(q_path[leg] < low[section]) or np.any(q_path[leg] > high[section]):
                    raise ValueError(
                        f"{leg} {phase} path leaves joint limits: "
                        f"q={np.rad2deg(q_path[leg])} deg"
                    )
                if singular < recorded.MIN_JACOBIAN_SINGULAR:
                    raise ValueError(
                        f"{leg} {phase} path approaches singularity: "
                        f"sigma_min={singular:.4f}"
                    )
                if leg != swing or phase in ("SHIFT", "RECENTER"):
                    jacobian = dog5_kinematics.foot_jacobian(leg, q_path[leg])
                    force = np.array([0.0, 0.0, -base.DOG5_MASS_KG * 9.81 / 3.0])
                    max_static_tau = max(
                        max_static_tau,
                        float(np.max(np.abs(jacobian.T @ force))),
                    )

    for leg in LEGS:
        final_position = dog5_kinematics.foot_position(leg, q_path[leg])
        if not np.allclose(
            final_position, controller.final_foot[leg], atol=1.0e-4
        ):
            raise ValueError(
                f"{leg} gait cycle did not return to its neutral target"
            )
    return max_static_tau, minimum_planned_margin


class CrawlSafetyGate(inplace.ConfirmedSafetyGate):
    """ConfirmedSafetyGate whose overspeed e-stop can be switched off.

    With ``overspeed_enabled=False`` (operator ``--no-overspeed``) the driver
    and encoder speed tiers never stop the run.  The encoder/driver
    bookkeeping still updates every cycle so the status line stays truthful
    and the guard can be re-enabled by simply not passing the flag; ALL other
    e-stops (joint-limit, over-temp, CAN-miss, motor-fault) remain active.
    """

    def __init__(self, tau_cap, qd_estop=base.QD_ESTOP,
                 qd_estop_hard=base.QD_ESTOP_HARD, overspeed_enabled=True):
        super().__init__(tau_cap, qd_estop=qd_estop, qd_estop_hard=qd_estop_hard)
        self.overspeed_enabled = bool(overspeed_enabled)

    def overspeed_reason(self, qd, q, now):
        # Run the real check for its side effects (encoder_qd, peaks, streaks),
        # then suppress the trip verdict when the operator disabled it.
        reason = super().overspeed_reason(qd, q, now)
        return reason if self.overspeed_enabled else None


def _saturated_legs(requested_tau, gate, now, controller, candidate_legs):
    saturated = set()
    if requested_tau is None:
        return saturated
    cap = gate.cap_now(now)
    for index, leg in enumerate(LEGS):
        if leg not in candidate_legs:
            continue
        section = slice(3 * index, 3 * index + 3)
        if (
            np.any(np.abs(requested_tau[section]) >= cap - 1.0e-9)
            or controller.last_leg_force_clipped[leg]
        ):
            saturated.add(leg)
    return saturated


def run_hardware(args):
    controller = CrawlController(
        args.cart_gain_scale,
        args.support_scale,
        args.stand_height,
        args.travel_scale,
    )
    max_static_tau, planned_margin = validate_crawl_configuration(
        controller, args.shift_distance, args.step_length, args.swing_height
    )
    planner = CrawlGaitPlanner(
        controller,
        args.shift_distance,
        args.step_length,
        args.swing_height,
        args.leg_cycle,
        args.gait_cycles,
        observe_phases=args.observe_phases,
        unload_tau_trip=args.unload_tau_trip,
    )
    gate = CrawlSafetyGate(
        args.tau_max,
        args.qd_estop,
        args.qd_estop_hard,
        overspeed_enabled=not args.no_overspeed,
    )
    unwrap = [base.CalibratedEncoderUnwrap() for _ in base.HARDWARE_JOINTS]

    print(
        "[crawl] Flow: CURRENT -> CROUCH -> ENTER -> STAND -> HOLD -> "
        "ENTER -> CRAWL -> CRAWL_HOLD; P parks from a hold, X stops"
    )
    print(
        f"[crawl] mode={planner.gait_mode}, order={','.join(GAIT_ORDER)} "
        f"(hind then diagonal fore; ENTER-gated hold every "
        f"{MANDATORY_HOLD_EVERY} steps), cycles={args.gait_cycles}, "
        f"leg slot={args.leg_cycle:.2f}s, nominal duty={NOMINAL_DUTY_FACTOR:.2f}"
    )
    if args.observe_phases:
        print(
            "[crawl] OBSERVE PHASES: motion pauses at every boundary; inspect "
            "the robot and press ENTER to start the displayed next phase."
        )
    elif planner.gait_mode == "WALK":
        print(
            "[crawl] To isolate leg swing from forward walking, rerun with "
            "--swing-test --observe-phases."
        )
    print(
        f"[crawl] adaptive body shift={1000.0 * args.shift_distance:.0f}mm "
        f"perp. to the nearest stance-triangle edge; planned minimum margin="
        f"{1000.0 * planned_margin:.1f}mm"
    )
    print(
        f"[crawl] forward step={1000.0 * args.step_length:.0f}mm, "
        f"swing lift={1000.0 * args.swing_height:.0f}mm, "
        f"static mg/3 peak torque~{max_static_tau:.2f}N*m, cap={args.tau_max:.2f}N*m"
    )
    print(
        "[crawl] stance anchors remain fixed in the ground frame; only the "
        "active swing anchor changes. The swing trim is frozen from UNLOAD "
        "through LOAD."
    )
    print(
        f"[crawl] pre-lift UNLOAD gate: swing pitch/knee gravity-compensated "
        f"|tau_ext| < {args.unload_tau_trip:.2f} N*m and relative sag < "
        f"{1000.0 * UNLOAD_REL_SAG_MAX_M:.0f} mm, sustained "
        f"{UNLOAD_GATE_SETTLE_S:.1f}s within {UNLOAD_GATE_TIMEOUT_S:.1f}s; "
        f"while loaded the shift keeps growing (force feedback) up to "
        f"+{1000.0 * UNLOAD_EXTRA_MAX_M:.0f} mm within margin/joint budgets; "
        "a failure reloads the leg and ends the batch instead of stopping "
        "the run."
    )
    if args.tau_max < max_static_tau:
        print(
            f"[crawl] WARNING: --tau-max {args.tau_max:.2f}N*m is below the "
            f"~{max_static_tau:.2f}N*m static mg/3 requirement; keep supported."
        )
    if args.travel_scale < 1.0:
        print(
            "[crawl] PARTIAL STAND TEST: crawl will remain locked out until "
            "--travel-scale is 1.0."
        )
    if args.no_overspeed:
        print(
            "[crawl] !! OVERSPEED E-STOP DISABLED (--no-overspeed): neither "
            "the driver nor the encoder speed tier will stop the run. A real "
            "runaway will NOT be caught -- keep a hand on X and the robot "
            "supported. Joint-limit/temp/CAN-miss/fault e-stops still apply."
        )
    print("[crawl] ROBOT MUST REMAIN MECHANICALLY SUPPORTED DURING FIRST TESTS.")

    imu = None if args.no_imu else inplace._open_imu()
    if imu is not None:
        print(
            f"[imu] tip-over abort armed: |roll| or |pitch| > "
            f"{inplace.TIPOVER_ABORT_DEG:.0f} deg for "
            f"{inplace.TIPOVER_CONFIRM_SAMPLES} consecutive fresh samples"
        )
        print(
            f"[imu] lean watch armed: tilt change > "
            f"{CRAWL_LEAN_ABORT_DEG:.0f} deg from the liftoff attitude "
            "during LIFT/SWING -> graceful step abort (reload + recenter)"
        )

    key = base.KeyPoller()
    try:
        with base.motorbus.MotorBus(MOTOR_IDS, dirs=MOTOR_DIRECTIONS) as mb:
            armed = False
            stop_reason = None
            try:
                print("[crawl] Arming with a zero-torque stream...")
                if not mb.arm(rate_hz=base.CONTROL_HZ):
                    raise RuntimeError("not all motors armed")
                armed = True

                start_q = recorded.zero_torque_preflight(mb, key, unwrap)
                now = time.perf_counter()
                sequence = CrawlSequence(now, start_q, planner, args.travel_scale)
                gate.start(now, start_q)
                print(
                    "[crawl] ENTER accepted: native position control moving "
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
                last_print = 0.0
                index = 0
                velocity = recorded.EncoderVelocity()
                margin_trip_streak = 0
                tau_tare = None
                tipover_streak = 0
                lean_baseline = None
                lean_streak = 0

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
                        old_phase = planner.phase
                        old_step_index = planner.step_index
                        event = sequence.update(
                            now,
                            q=q,
                            cart_error=controller.last_cart_error,
                            qd=(qd_encoder if velocity.ready else None),
                            measured_tau=measured_torque,
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
                                    "start crawl."
                                )
                            elif sequence.stage == "CRAWL_HOLD":
                                if planner.aborted:
                                    print(
                                        "[stage] Crawl batch ABORTED "
                                        f"({planner.abort_reason}); holding "
                                        "on four feet. Inspect, then ENTER "
                                        "retries, P parks, X stops."
                                    )
                                else:
                                    print(
                                        "[stage] Crawl held neutral; ENTER "
                                        "repeats, P parks, X stops."
                                    )
                            elif sequence.stage == "HOLD_SAG":
                                controller.rebase_final_z_to_measured(q)
                                print("[stage] HOLD_SAG cannot crawl; P parks or X stops.")
                        phase_changed = (
                            old_stage == "CRAWL"
                            and (
                                planner.phase != old_phase
                                or planner.step_index != old_step_index
                                or sequence.stage != "CRAWL"
                            )
                        )
                        if phase_changed:
                            if sequence.stage == "CRAWL":
                                print(
                                    f"[phase] completed={old_phase} -> "
                                    f"{planner.diagnostic_context} "
                                    f"paused={'YES' if planner.paused else 'NO'}"
                                )
                                if planner.mandatory_hold:
                                    print(
                                        "[phase] DIAGONAL-PAIR HOLD: flat on "
                                        "four feet. Inspect, then press ENTER "
                                        "for the next pair."
                                    )
                                elif planner.paused:
                                    print(
                                        "[phase] HOLDING endpoint safely; "
                                        "inspect motion, then press ENTER."
                                    )
                            else:
                                print(
                                    f"[phase] completed={old_phase}; "
                                    f"crawl batch is now {sequence.stage}"
                                )
                        if (
                            old_stage != sequence.stage
                            and sequence.stage == "WAIT_STAND"
                        ):
                            print(
                                f"[stage] Waiting for stand settle; HOLD_SAG after "
                                f"{inplace.WAIT_STAND_TIMEOUT_S:.0f}s."
                            )

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
                        elif sequence.stage == "CRAWL":
                            requested_tau = controller.compute_crawl(q, qd_encoder, planner, now)
                            tau_command = gate.apply(requested_tau, q, now)
                        elif sequence.stage == "PARK":
                            requested_tau = controller.compute_park(
                                q, qd_encoder, now - sequence.started_at
                            )
                            tau_command = gate.apply(requested_tau, q, now)
                        elif sequence.stage == "WAIT_PARK":
                            requested_tau = controller.compute_park(q, qd_encoder, T_PARK)
                            tau_command = gate.apply(requested_tau, q, now)
                        else:
                            requested_tau = controller.compute_stand(q, qd_encoder, T_STAND)
                            tau_command = gate.apply(requested_tau, q, now)

                        if sequence.stage == "CRAWL":
                            stance = planner.trim_stance_legs
                            saturated = _saturated_legs(
                                requested_tau, gate, now, controller, stance
                            )
                            controller.update_gait_trim(now, stance, saturated)
                        else:
                            trim_stage = (
                                "HOLD" if sequence.stage == "CRAWL_HOLD" else sequence.stage
                            )
                            trim_mode = inplace._trim_mode_for_stage(trim_stage)
                            candidates = set(LEGS) if trim_mode == "integrate" else set()
                            saturated = _saturated_legs(
                                requested_tau, gate, now, controller, candidates
                            )
                            controller.update_support_trim(now, trim_mode, saturated)

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
                        if sequence.stage == "CRAWL":
                            support_margin = planner.actual_support_margin(q)
                            if (
                                planner.three_leg_support
                                and support_margin < SUPPORT_MARGIN_ESTOP_M
                            ):
                                margin_trip_streak += 1
                            else:
                                margin_trip_streak = 0
                            if (
                                stop_reason is None
                                and margin_trip_streak >= SUPPORT_MARGIN_ESTOP_STREAK
                            ):
                                stop_reason = (
                                    f"support triangle lost during {planner.swing_leg} "
                                    f"{planner.phase}: margin="
                                    f"{1000.0 * support_margin:.1f} mm"
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

                        imu_sample = None
                        if imu is not None:
                            try:
                                imu_sample = imu.sample()
                            except Exception:
                                imu_sample = None
                            if (
                                imu_sample is not None
                                and imu_sample.age_s <= inplace.POSTURE_STALE_S
                            ):
                                if (
                                    abs(imu_sample.roll_deg)
                                    > inplace.TIPOVER_ABORT_DEG
                                    or abs(imu_sample.pitch_deg)
                                    > inplace.TIPOVER_ABORT_DEG
                                ):
                                    tipover_streak += 1
                                    if (
                                        stop_reason is None
                                        and tipover_streak
                                        >= inplace.TIPOVER_CONFIRM_SAMPLES
                                    ):
                                        stop_reason = (
                                            f"IMU tip-over: roll="
                                            f"{imu_sample.roll_deg:+.1f} pitch="
                                            f"{imu_sample.pitch_deg:+.1f} deg "
                                            f"exceeds "
                                            f"{inplace.TIPOVER_ABORT_DEG:.0f} deg"
                                        )
                                else:
                                    tipover_streak = 0
                                # Lean watch: real (measured) body tilt drift
                                # away from the liftoff attitude while only
                                # three feet support -- the encoder support
                                # triangle cannot see this because the body
                                # tilts beyond the rigid geometry it assumes.
                                if (
                                    sequence.stage == "CRAWL"
                                    and planner.active
                                    and planner.phase in ("LIFT", "SWING")
                                ):
                                    if lean_baseline is None:
                                        lean_baseline = (
                                            imu_sample.roll_deg,
                                            imu_sample.pitch_deg,
                                        )
                                        lean_streak = 0
                                    else:
                                        d_roll = (
                                            imu_sample.roll_deg
                                            - lean_baseline[0]
                                        )
                                        d_pitch = (
                                            imu_sample.pitch_deg
                                            - lean_baseline[1]
                                        )
                                        if (
                                            max(abs(d_roll), abs(d_pitch))
                                            > CRAWL_LEAN_ABORT_DEG
                                        ):
                                            lean_streak += 1
                                            if lean_streak >= CRAWL_LEAN_STREAK:
                                                event = planner._begin_abort(
                                                    now,
                                                    q,
                                                    f"IMU lean: d_roll="
                                                    f"{d_roll:+.1f} d_pitch="
                                                    f"{d_pitch:+.1f} deg from "
                                                    f"liftoff exceeds "
                                                    f"{CRAWL_LEAN_ABORT_DEG:.0f}",
                                                )
                                                print(f"[stage] {event}")
                                                lean_baseline = None
                                                lean_streak = 0
                                        else:
                                            lean_streak = 0
                                else:
                                    lean_baseline = None
                                    lean_streak = 0
                        latched = [mid for mid, error in errors.items() if error & 0x80]
                        unverified = [mid for mid in MOTOR_IDS if mb.rec(mid).error is None]

                        pressed = key.get()
                        if pressed in ("t", "T"):
                            # Tare the load map: capture the current external
                            # torque residual (friction + model error) as the
                            # display offset.  Only meaningful when every
                            # foot is OFF the ground (robot held up).
                            tau_tare = measured_torque - leg_gravity_stack(q)
                            print(
                                "[legs] fz TARED at current pose -- valid "
                                "only if ALL feet were off the ground "
                                f"(residual max {np.max(np.abs(tau_tare)):.2f}"
                                " N*m); press T again to re-tare"
                            )
                        elif pressed in ("x", "X"):
                            stop_reason = "operator X"
                        elif pressed in ("p", "P"):
                            _, message = sequence.request_park(
                                now,
                                healthy=not latched and not unverified and stop_reason is None,
                            )
                            print(f"[stage] {message}")
                        elif base._is_enter(pressed):
                            advanced, message = sequence.request_next(
                                now,
                                q,
                                qd_encoder,
                                healthy=not latched and not unverified and stop_reason is None,
                            )
                            if advanced and sequence.stage == "STAND":
                                controller.restore_full_targets()
                            print(f"[stage] {message}")
                        if stop_reason:
                            if (
                                sequence.stage == "CRAWL"
                                and stop_reason != "operator X"
                            ):
                                stop_reason = (
                                    f"{planner.diagnostic_context}: {stop_reason}"
                                )
                            break

                        recover = [
                            mid
                            for mid in latched
                            if elapsed - last_recover[mid] >= base.RECOVER_PERIOD_S
                        ]
                        if recover:
                            base._recover_input_lost(
                                mb, recover, elapsed, last_recover, next_fault_status
                            )
                            latched = []

                        if now - last_print >= 1.0 / base.STATUS_HZ:
                            crawling = sequence.stage == "CRAWL"
                            phase = planner.phase if crawling else "-"
                            swing = planner.swing_leg if crawling else "-"
                            motion = planner.motion_kind if crawling else "-"
                            paused = "YES" if crawling and planner.paused else "NO"
                            margin_text = (
                                "-"
                                if not np.isfinite(support_margin)
                                else f"{1000.0 * support_margin:.1f}mm"
                            )
                            if crawling and phase in (
                                "UNLOAD",
                                "LIFT",
                                "SWING",
                            ):
                                margin_text += (
                                    f" tau_ext={planner.last_unload_tau_nm:.2f}"
                                    f"N*m rel_sag="
                                    f"{1000.0 * planner.last_unload_rel_sag_m:+.1f}mm"
                                    f" extra_shift="
                                    f"{1000.0 * planner.unload_extra_m:.1f}mm"
                                )
                            print(
                                f"[crawl] mode={planner.gait_mode:10s} "
                                f"stage={sequence.stage:11s} phase={phase:14s} "
                                f"motion={motion:14s} paused={paused:3s} "
                                f"swing={swing:2s} step="
                                f"{planner.step_index + 1 if sequence.stage == 'CRAWL' else 0}/"
                                f"{planner.total_steps} margin={margin_text} "
                                f"cart_err={controller.last_cart_error:.3f}m "
                                f"sigma_min={controller.last_min_singular:.4f} "
                                f"force_max={controller.last_max_force:.1f}N "
                                f"max|qd_enc|={np.max(np.abs(qd_encoder)):.2f} "
                                f"max|qd_driver|={np.max(np.abs(qd_driver)):.2f} "
                                f"max|tau|={np.max(np.abs(tau_command)):.2f}N*m "
                                f"max|tau_fb|={np.max(np.abs(measured_torque)):.2f}N*m "
                                f"Tmax={int(np.max(temps))}C latched={len(latched)}",
                                flush=True,
                            )
                            # Measured load map: per-leg vertical support
                            # (J^-T tau, all three joints) plus the CoM xy the
                            # distribution implies.  Printed in every stage --
                            # the CROUCH map vs the STAND map separates joint-
                            # zero/geometry error (asymmetry flips or a foot
                            # unloads between poses) from real mass asymmetry
                            # (same-side in both).  trim_N is the commanded
                            # integral top-up, NOT measured load.
                            support, com_xy = foot_load_map(
                                q, measured_torque, tau_tare
                            )
                            fz_text = ",".join(
                                f"{leg}{support[leg]:+.1f}" for leg in LEGS
                            )
                            com_text = (
                                f"({1000.0 * com_xy[0]:+.0f},"
                                f"{1000.0 * com_xy[1]:+.0f})mm"
                                if np.all(np.isfinite(com_xy))
                                else "(-)"
                            )
                            trim_text = (
                                "-"
                                if position_stage
                                else ",".join(
                                    f"{leg}{controller.z_trim_n[leg]:+.1f}"
                                    for leg in LEGS
                                )
                            )
                            tare_text = " (tared)" if tau_tare is not None else ""
                            imu_text = ""
                            if imu_sample is not None:
                                stale = (
                                    " STALE"
                                    if imu_sample.age_s
                                    > inplace.POSTURE_STALE_S
                                    else ""
                                )
                                imu_text = (
                                    f" rp=({imu_sample.roll_deg:+.2f},"
                                    f"{imu_sample.pitch_deg:+.2f})deg{stale}"
                                )
                            print(
                                f"[legs] trim_N=({trim_text}) "
                                f"fz_N=({fz_text}) com={com_text}{tare_text}"
                                f"{imu_text}",
                                flush=True,
                            )
                            last_print = now

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
                            raise RuntimeError(f"CAN position transmit failed for CAN {mid}")
                    else:
                        mb.torque(mid, float(tau_command[joint_index]))

                    index += 1
                    mb.pace(deadline)
                    deadline += slot

            except KeyboardInterrupt as exc:
                stop_reason = str(exc) or "KeyboardInterrupt"
            except Exception as exc:
                detail = str(exc)
                if (
                    "sequence" in locals()
                    and sequence.stage == "CRAWL"
                    and not detail.startswith("mode=")
                ):
                    detail = f"{planner.diagnostic_context}: {detail}"
                stop_reason = f"error: {detail}"
                raise RuntimeError(detail) from exc
            finally:
                if armed:
                    print(f"[crawl] stopping: {stop_reason or 'aborted'}")
                    try:
                        base._soft_stop(mb)
                    except Exception as exc:
                        print(
                            f"[crawl] soft stop failed: {exc}; sending STOP",
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
    controller = CrawlController(
        args.cart_gain_scale,
        args.support_scale,
        args.stand_height,
        1.0,
    )
    max_static, planned_margin = validate_crawl_configuration(
        controller, args.shift_distance, args.step_length, args.swing_height
    )
    assert max_static < base.STAGED_TAU_MAX, max_static
    assert planned_margin >= MIN_LIFTOFF_MARGIN_M, planned_margin
    assert NOMINAL_DUTY_FACTOR >= 0.75

    # foot_load_map round-trip: joint torques synthesized from a known
    # per-leg vertical load PLUS each leg's own gravity-holding torque must
    # reproduce that load and the CoM implied by the distribution (here 60/40
    # left-heavy at the neutral stand pose).  A leg holding only its own
    # weight (airborne) must read ~0.
    q_stand = np.zeros(N_JOINTS)
    known_fz = {"FL": 15.0, "FR": 10.0, "RL": 15.0, "RR": 10.0}
    tau_synth = np.zeros(N_JOINTS)
    for check_index, check_leg in enumerate(LEGS):
        section = slice(3 * check_index, 3 * check_index + 3)
        q_stand[section] = _ik_to_target(
            check_leg,
            recorded.Q_RECORDED_CROUCH[section],
            np.asarray(controller.final_foot[check_leg], dtype=float),
        )
        jac = dog5_kinematics.foot_jacobian(check_leg, q_stand[section])
        tau_synth[section] = dog5_kinematics.leg_gravity_torque(
            check_leg, q_stand[section]
        ) + jac.T @ np.array([0.0, 0.0, -known_fz[check_leg]])
    support_check, com_check = foot_load_map(q_stand, tau_synth)
    for check_leg in LEGS:
        assert abs(support_check[check_leg] - known_fz[check_leg]) < 1.0e-6
    tau_airborne = tau_synth.copy()
    tau_airborne[0:3] = dog5_kinematics.leg_gravity_torque("FL", q_stand[0:3])
    support_air, _ = foot_load_map(q_stand, tau_airborne)
    assert abs(support_air["FL"]) < 1.0e-6, support_air
    # A friction-like torque bias reads as phantom force; the tare offset
    # removes exactly that bias.
    bias = np.full(N_JOINTS, 0.05)
    support_biased, _ = foot_load_map(q_stand, tau_synth + bias)
    assert any(
        abs(support_biased[leg] - known_fz[leg]) > 0.1 for leg in LEGS
    ), support_biased
    support_tared, _ = foot_load_map(q_stand, tau_synth + bias, bias)
    for check_leg in LEGS:
        assert abs(support_tared[check_leg] - known_fz[check_leg]) < 1.0e-6
    feet_check = {
        leg: dog5_kinematics.foot_position(
            leg, q_stand[3 * i : 3 * i + 3]
        )[:2]
        for i, leg in enumerate(LEGS)
    }
    com_expected = sum(
        known_fz[leg] * feet_check[leg] for leg in LEGS
    ) / sum(known_fz.values())
    assert np.allclose(com_check, com_expected, atol=1.0e-9), com_check
    assert com_check[1] > 0.005, com_check  # left-heavy load -> CoM left

    # Touchdown accepts the reported hardware signature when the actual foot
    # is on the stance plane and the target is 16 mm below it.  Lateral miss,
    # free-space/midair endpoints, motion, and excessive penetration reject.
    accepted, tracked, on_plane = touchdown_accepted(
        0.016, 0.0, 0.0, -0.016, 0.01, 0.002
    )
    assert accepted and not tracked and on_plane
    assert not touchdown_accepted(
        0.016, 0.016, 0.0, -0.016, 0.01, 0.002
    )[0]
    assert not touchdown_accepted(
        0.016, 0.0, -0.016, -0.016, 0.01, 0.002
    )[0]
    assert not touchdown_accepted(
        0.016, 0.0, +0.016, -0.016, 0.01, 0.002
    )[0]
    assert not touchdown_accepted(
        0.016, 0.0, 0.0, -0.016, 0.31, 0.002
    )[0]
    assert not touchdown_accepted(
        0.016, 0.0, 0.0, -0.016, 0.01, 0.021
    )[0]
    assert not touchdown_accepted(
        0.035, 0.0, 0.0, -0.035, 0.01, 0.002
    )[0]
    assert not touchdown_accepted(
        0.010, 0.004, 0.020, 0.0, 0.01, 0.002
    )[0]
    assert touchdown_accepted(
        0.010, 0.004, 0.002, 0.0, 0.01, 0.002
    )[0]
    # Regression: the exact hardware signature that used to time out -- a
    # real early contact pinned 8.7 mm short of the planned advance, on the
    # stance plane, target 14 mm below it, foot stationary.
    assert touchdown_accepted(
        0.0142, 0.0087, -0.003, -0.0141, 0.0, 0.0
    )[0]

    # Signed plane distance is invariant to a rigid body-frame transform and
    # rejects a degenerate three-foot plane explicitly.
    plane = np.array(
        [[-0.3, -0.1, 0.0], [0.3, -0.1, 0.02], [-0.3, 0.1, 0.01]]
    )
    normal = np.cross(plane[1] - plane[0], plane[2] - plane[0])
    normal /= np.linalg.norm(normal)
    point = np.mean(plane, axis=0) + 0.003 * normal
    distance = signed_distance_to_stance_plane(point, plane)
    angle = np.deg2rad(20.0)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    translation = np.array([0.2, -0.3, 0.1])
    transformed_plane = plane @ rotation.T + translation
    transformed_point = rotation @ point + translation
    assert np.isclose(
        signed_distance_to_stance_plane(transformed_point, transformed_plane),
        distance,
    )
    try:
        signed_distance_to_stance_plane(np.zeros(3), np.zeros((3, 3)))
    except ValueError as exc:
        assert "degenerate" in str(exc)
    else:
        raise AssertionError("degenerate stance plane was accepted")

    # A symmetric rectangle has zero margin at liftoff; the adaptive shift must
    # put the origin strictly inside every three-foot triangle, and (the whole
    # point of the change) it must do at least as well as the old fixed diagonal
    # on the SKEWED geometry after a diagonal pair has stepped forward.
    symmetric = {
        "FL": np.array([+0.34, +0.112]),
        "FR": np.array([+0.34, -0.112]),
        "RL": np.array([-0.34, +0.112]),
        "RR": np.array([-0.34, -0.112]),
    }
    for swing in LEGS:
        stance = [symmetric[leg] for leg in LEGS if leg != swing]
        unshifted = support_triangle_margin(stance)
        shifted_body, _ = adaptive_body_shift(
            symmetric[swing], stance, np.zeros(2), args.shift_distance
        )
        shifted = support_triangle_margin(stance, com_xy=shifted_body)
        assert abs(unshifted) < 1.0e-12, (swing, unshifted)
        assert shifted >= MIN_LIFTOFF_MARGIN_M, (swing, shifted)

    # Skewed geometry: FL and RR have stepped 30 mm forward (the first diagonal
    # pair).  The adaptive shift must still clear the liftoff floor for the
    # second pair, where a fixed diagonal left the thinnest margin.
    skewed = dict(symmetric)
    skewed["FL"] = symmetric["FL"] + np.array([0.030, 0.0])
    skewed["RR"] = symmetric["RR"] + np.array([0.030, 0.0])
    body_skew = np.array([0.015, 0.0])
    for swing in ("RL", "FR"):
        stance = [skewed[leg] for leg in LEGS if leg != swing]
        shifted_body, _ = adaptive_body_shift(
            skewed[swing], stance, body_skew, args.shift_distance
        )
        margin = support_triangle_margin(stance, com_xy=shifted_body)
        assert margin >= MIN_LIFTOFF_MARGIN_M, (swing, margin)

    # Stance-anchor invariant: target_xy + body_xy is constant during SHIFT
    # and RECENTER; only the swing anchor changes during SWING.
    segments = list(
        _planned_gait_segments(
            controller, args.shift_distance, args.step_length, args.swing_height
        )
    )
    for phase, swing, starts, ends in segments:
        if phase in ("LIFT", "LOWER"):
            for leg in LEGS:
                assert np.allclose(starts[leg][:2], ends[leg][:2])
        if phase == "SWING":
            for leg in LEGS:
                delta = ends[leg][:2] - starts[leg][:2]
                if leg == swing:
                    assert np.allclose(delta, [args.step_length, 0.0])
                else:
                    assert np.allclose(delta, 0.0)
    final_targets = segments[-1][3]
    for leg in LEGS:
        assert np.allclose(final_targets[leg], controller.final_foot[leg], atol=1.0e-10)

    # Trim gating: swing trim freezes while all three stance trims integrate,
    # then resumes once the foot is declared loaded.
    trim = CrawlController(0.25, 0.65, args.stand_height, 1.0)
    for leg in LEGS:
        trim.last_leg_sag_m[leg] = 0.020
        trim.z_trim_n[leg] = 1.0
    trim._trim_last_time = 0.0
    trim.update_gait_trim(0.1, set(LEGS) - {"RR"})
    assert np.isclose(trim.z_trim_n["RR"], 1.0)
    assert all(trim.z_trim_n[leg] > 1.0 for leg in LEGS if leg != "RR")
    trim.update_gait_trim(0.2, set(LEGS))
    assert trim.z_trim_n["RR"] > 1.0

    # Planner phase targets conserve total weight share through unload/load.
    planner = CrawlGaitPlanner(
        controller,
        args.shift_distance,
        args.step_length,
        args.swing_height,
        args.leg_cycle,
        args.gait_cycles,
    )
    planner.start(0.0)
    for phase in planner.phases:
        planner.phase = phase
        planner.phase_started_at = 0.0
        for fraction in (0.0, 0.5, 1.0):
            now = fraction * planner.duration(phase)
            shares = [planner.target(leg, now)[2] for leg in LEGS]
            assert np.isclose(sum(shares), 1.0), (phase, fraction, shares)

    # Observation mode freezes the next phase at its continuous start target
    # and resumes its clock only after ENTER.
    observed = CrawlGaitPlanner(
        controller,
        args.shift_distance,
        0.0,
        args.swing_height,
        args.leg_cycle,
        1,
        observe_phases=True,
    )
    observed.start(0.0)
    assert observed.gait_mode == "SWING_TEST"
    assert observed.motion_kind == "BODY_MOTION"
    assert not observed.paused
    observed._set_phase("LIFT", 1.0)
    assert observed.paused and observed.motion_kind == "LEG_SWING"
    held_target = observed.target(observed.swing_leg, 100.0)[0]
    start_target = observed.target(observed.swing_leg, 1.0)[0]
    assert np.allclose(held_target, start_target)
    resumed, message = observed.resume(101.0)
    assert resumed and "LEG_SWING" in message
    assert not observed.paused and np.isclose(observed.phase_started_at, 101.0)

    # Drive the real phase machine with exact encoder-FK endpoint poses.  This
    # exercises both settle gates, anchor commits, and the final neutral hold.
    state_planner = CrawlGaitPlanner(
        controller,
        args.shift_distance,
        args.step_length,
        args.swing_height,
        args.leg_cycle,
        1,
    )
    q_state = Q_RECORDED_CROUCH.copy()
    for index, leg in enumerate(LEGS):
        section = slice(3 * index, 3 * index + 3)
        q_state[section] = _ik_to_target(
            leg, q_state[section], controller.final_foot[leg]
        )
    state_planner.start(0.0)
    zero_qd = np.zeros(N_JOINTS)
    transitions = 0
    mandatory_holds = 0
    now_t = 0.0
    while not state_planner.finished:
        # A mandatory diagonal-pair hold pauses at the SHIFT start; the
        # operator (here, the test) resumes it with settled feet.
        if state_planner.paused:
            assert state_planner.mandatory_hold, "only mandatory holds here"
            mandatory_holds += 1
            for index, leg in enumerate(LEGS):
                desired = state_planner.target(leg, now_t)[0]
                section = slice(3 * index, 3 * index + 3)
                q_state[section] = _ik_to_target(leg, q_state[section], desired)
            resumed, _ = state_planner.resume(now_t)
            assert resumed
        phase = state_planner.phase
        endpoint_time = (
            state_planner.phase_started_at + state_planner.duration(phase)
            + 1.0e-6
        )
        for index, leg in enumerate(LEGS):
            desired = state_planner.target(leg, endpoint_time)[0]
            section = slice(3 * index, 3 * index + 3)
            q_state[section] = _ik_to_target(leg, q_state[section], desired)
        event = state_planner.update(endpoint_time, q_state, zero_qd)
        if phase in ("SHIFT", "UNLOAD", "LOWER", "RECENTER"):
            dwell = {
                "SHIFT": SHIFT_SETTLE_S,
                "UNLOAD": UNLOAD_GATE_SETTLE_S,
                "LOWER": TOUCHDOWN_SETTLE_S,
                "RECENTER": RECENTER_SETTLE_S,
            }[phase]
            event = state_planner.update(
                endpoint_time + dwell + 0.01, q_state, zero_qd
            )
        now_t = endpoint_time + 1.0
        assert event is not None, (phase, state_planner.phase)
        transitions += 1
        assert transitions <= len(GAIT_ORDER) * (len(state_planner.phases) + 1)
    assert state_planner.step_index == len(GAIT_ORDER)
    assert not state_planner.aborted
    # One cycle (RR FL RL FR) holds once, between the two diagonal pairs.
    assert mandatory_holds == 1, mandatory_holds
    for index, leg in enumerate(LEGS):
        section = slice(3 * index, 3 * index + 3)
        actual = dog5_kinematics.foot_position(leg, q_state[section])
        assert np.allclose(actual, controller.final_foot[leg], atol=1.0e-4)

    # ---- graceful aborts ---------------------------------------------------
    def _track_endpoint(planner, q, endpoint):
        for index, leg in enumerate(LEGS):
            desired = planner.target(leg, endpoint)[0]
            section = slice(3 * index, 3 * index + 3)
            q[section] = _ik_to_target(leg, q[section], desired)

    def _drive(planner, q, measured_tau=None, freeze_q_in=None,
               stop_phase=None, guard=60):
        """Advance the phase machine; returns collected event text."""
        events = []
        while not planner.finished and guard > 0:
            guard -= 1
            phase = planner.phase
            if stop_phase is not None and phase == stop_phase:
                break
            endpoint = (
                planner.phase_started_at + planner.duration(phase) + 1.0e-6
            )
            if freeze_q_in is None or phase not in freeze_q_in:
                _track_endpoint(planner, q, endpoint)
            times = [endpoint]
            if phase == "SHIFT":
                times += [endpoint + SHIFT_SETTLE_S + 0.01,
                          endpoint + SHIFT_SETTLE_TIMEOUT_S + 0.01]
            elif phase == "UNLOAD":
                times += [endpoint + UNLOAD_GATE_SETTLE_S + 0.01,
                          endpoint + UNLOAD_GATE_TIMEOUT_S + 0.01]
            elif phase == "LOWER":
                times += [endpoint + TOUCHDOWN_SETTLE_S + 0.01,
                          endpoint + TOUCHDOWN_TIMEOUT_S + 0.01]
            elif phase == "RECENTER":
                times += [endpoint + RECENTER_SETTLE_S + 0.01,
                          endpoint + RECENTER_TIMEOUT_S + 0.01]
            for tick in times:
                event = planner.update(
                    tick, q, zero_qd, measured_tau=measured_tau
                )
                if event:
                    events.append(event)
                if planner.finished or planner.phase != phase:
                    break
        assert guard > 0, "abort drive did not converge"
        return " | ".join(events)

    # UNLOAD gate failure (a loaded swing leg) aborts the step gracefully and
    # restores the exact pre-step neutral hold; nothing raises.
    abort_controller = CrawlController(
        args.cart_gain_scale, args.support_scale, args.stand_height, 1.0
    )
    abort_planner = CrawlGaitPlanner(
        abort_controller,
        args.shift_distance,
        args.step_length,
        args.swing_height,
        args.leg_cycle,
        1,
    )
    q_abort = Q_RECORDED_CROUCH.copy()
    for index, leg in enumerate(LEGS):
        section = slice(3 * index, 3 * index + 3)
        q_abort[section] = _ik_to_target(
            leg, q_abort[section], abort_controller.final_foot[leg]
        )
    hold_before = {
        leg: abort_controller.final_foot[leg].copy() for leg in LEGS
    }
    abort_planner.start(0.0)
    loaded_tau = np.zeros(N_JOINTS)
    loaded_tau[3 * LEGS.index(abort_planner.swing_leg) + 1] = 1.0
    text = _drive(abort_planner, q_abort, measured_tau=loaded_tau)
    assert abort_planner.finished and abort_planner.aborted, text
    assert "unload gate timed out" in text, text
    assert "ABORT" in text and "neutral" in text, text
    for leg in LEGS:
        assert np.allclose(
            abort_controller.final_foot[leg], hold_before[leg], atol=1.0e-9
        ), leg

    # Force-feedback shift: a persistently loaded swing leg grows the body
    # shift along the stored inward normal (never beyond the safe cap)
    # before the gate finally aborts.
    grow_planner = CrawlGaitPlanner(
        abort_controller,
        args.shift_distance,
        args.step_length,
        args.swing_height,
        args.leg_cycle,
        1,
    )
    grow_planner.start(0.0)
    q_grow = q_abort.copy()
    _drive(grow_planner, q_grow, stop_phase="UNLOAD")
    assert grow_planner.phase == "UNLOAD"
    loaded_grow = np.zeros(N_JOINTS)
    loaded_grow[3 * LEGS.index(grow_planner.swing_leg) + 1] = 2.0
    tick = grow_planner.phase_started_at + grow_planner.duration("UNLOAD")
    grow_planner.update(tick + 0.01, q_grow, zero_qd, measured_tau=loaded_grow)
    cap = grow_planner.unload_extra_cap_m
    assert cap is not None and cap > 0.001, cap
    grow_planner.update(tick + 0.51, q_grow, zero_qd, measured_tau=loaded_grow)
    grown = grow_planner.unload_extra_m
    assert 0.001 < grown <= cap + 1.0e-9, (grown, cap)
    # The chosen direction must still gain support margin (positive
    # projection on the inward normal) even when the lateral budget is
    # abduction-starved and the longitudinal fallback wins.
    assert float(
        grow_planner.unload_growth_dir @ grow_planner.shift_normal
    ) > 0.1
    assert np.allclose(
        grow_planner.shifted_body_xy,
        grow_planner.base_shifted_body_xy
        + grown * grow_planner.unload_growth_dir,
    )
    text = _drive(grow_planner, q_grow, measured_tau=loaded_grow)
    assert grow_planner.finished and grow_planner.aborted, text
    assert "safe shift budget" in text, text

    # Airborne drag watch: sustained external torque during SWING aborts
    # gracefully and commits the measured foot position as the anchor.
    drag_planner = CrawlGaitPlanner(
        abort_controller,
        args.shift_distance,
        args.step_length,
        args.swing_height,
        args.leg_cycle,
        1,
    )
    drag_planner.start(0.0)
    q_drag = q_abort.copy()
    _drive(drag_planner, q_drag, stop_phase="SWING")
    assert drag_planner.phase == "SWING"
    drag_leg = drag_planner.swing_leg
    drag_tau = np.zeros(N_JOINTS)
    drag_tau[3 * LEGS.index(drag_leg) + 2] = 2.0
    drag_event = None
    for streak in range(SWING_DRAG_STREAK):
        drag_event = drag_planner.update(
            drag_planner.phase_started_at + 0.01 * (streak + 1),
            q_drag,
            zero_qd,
            measured_tau=drag_tau,
        )
    assert drag_event and "dragging" in drag_event, drag_event
    assert drag_planner.phase == "ABORT_LOAD"
    drag_section = slice(
        3 * LEGS.index(drag_leg), 3 * LEGS.index(drag_leg) + 3
    )
    expected_drag_anchor = (
        dog5_kinematics.foot_position(drag_leg, q_drag[drag_section])[:2]
        + drag_planner.shifted_body_xy
    )
    assert np.allclose(
        drag_planner.world_foot_xy[drag_leg], expected_drag_anchor
    )
    text = _drive(drag_planner, q_drag)
    assert drag_planner.finished and drag_planner.aborted, text

    # SHIFT gate failure (robot never tracks the shift) also aborts cleanly.
    shift_planner = CrawlGaitPlanner(
        abort_controller,
        args.shift_distance,
        args.step_length,
        args.swing_height,
        args.leg_cycle,
        1,
    )
    shift_planner.start(0.0)
    text = _drive(shift_planner, q_abort, freeze_q_in=("SHIFT",))
    assert shift_planner.finished and shift_planner.aborted, text
    assert "liftoff gate timed out" in text, text
    for leg in LEGS:
        assert np.allclose(
            abort_controller.final_foot[leg], hold_before[leg], atol=1.0e-9
        ), leg

    # Touchdown timeout aborts, commits the MEASURED foot position as the
    # swing anchor (no sideways drag), and ends in a consistent hold.
    lower_planner = CrawlGaitPlanner(
        abort_controller,
        args.shift_distance,
        args.step_length,
        args.swing_height,
        args.leg_cycle,
        1,
    )
    lower_planner.start(0.0)
    q_lower = q_abort.copy()
    _drive(lower_planner, q_lower, stop_phase="LOWER")
    assert lower_planner.phase == "LOWER"
    swing = lower_planner.swing_leg
    swing_index = LEGS.index(swing)
    frozen_fk = dog5_kinematics.foot_position(
        swing, q_lower[3 * swing_index : 3 * swing_index + 3]
    )
    text = _drive(
        lower_planner, q_lower, freeze_q_in=("LOWER",)
    )
    assert lower_planner.finished and lower_planner.aborted, text
    assert "touchdown timed out" in text, text
    expected_anchor = frozen_fk[:2] + lower_planner.shifted_body_xy
    assert np.allclose(
        lower_planner.world_foot_xy[swing], expected_anchor, atol=1.0e-9
    )
    assert np.allclose(
        abort_controller.final_foot[swing][:2], expected_anchor, atol=1.0e-9
    ), "aborted swing hold target must match the measured foot"
    abort_controller.final_foot[swing] = hold_before[swing].copy()

    # start() re-snapshots the hold targets so a retry after an abort begins
    # from the committed state instead of the stale construction-time one.
    resnap_planner = CrawlGaitPlanner(
        abort_controller,
        args.shift_distance,
        args.step_length,
        args.swing_height,
        args.leg_cycle,
        1,
    )
    abort_controller.final_foot["FL"] = hold_before["FL"] + np.array(
        [0.005, 0.0, 0.0]
    )
    resnap_planner.start(0.0)
    assert np.isclose(
        resnap_planner.nominal_xy["FL"][0],
        abort_controller.final_foot["FL"][0],
    )
    abort_controller.final_foot["FL"] = hold_before["FL"].copy()

    print("crawl_dog5_hw offline self-test PASS")
    print(
        f"  order={','.join(GAIT_ORDER)}, leg slot={args.leg_cycle:.2f}s, "
        f"nominal duty factor={NOMINAL_DUTY_FACTOR:.2f}"
    )
    print(
        f"  body shift={1000.0 * args.shift_distance:.0f} mm adaptive (perp. to nearest edge), "
        f"minimum planned support margin={1000.0 * planned_margin:.1f} mm"
    )
    print(
        f"  forward step={1000.0 * args.step_length:.0f} mm, "
        f"swing height={1000.0 * args.swing_height:.0f} mm"
    )
    print(f"  worst static mg/3 joint torque along gait: ~{max_static:.2f} N*m")
    print(
        "  stance anchors fixed; swing trim frozen from UNLOAD through LOAD"
    )
    print(
        f"  pre-lift UNLOAD gate: gravity-compensated |tau_ext| < "
        f"{args.unload_tau_trip:.2f} N*m and rel sag < "
        f"{1000.0 * UNLOAD_REL_SAG_MAX_M:.0f} mm; while loaded the shift "
        f"grows on force feedback (up to +{1000.0 * UNLOAD_EXTRA_MAX_M:.0f} "
        "mm within margin/joint budgets), else the step aborts"
    )
    print(
        f"  airborne drag watch: |tau_ext| > {SWING_DRAG_TAU_TRIP_NM:.2f} "
        "N*m during LIFT/SWING aborts the step (foot scraping)"
    )
    print(
        "  gate timeouts reload + recenter to a four-foot hold (no run stop);"
        " only triangle loss / e-stops / X stop the run"
    )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tau-max",
        type=float,
        default=DEFAULT_TAU_MAX,
        help=f"per-joint torque cap, N*m (default: {DEFAULT_TAU_MAX})",
    )
    parser.add_argument(
        "--cart-gain-scale",
        type=float,
        default=base.DEFAULT_CART_GAIN_SCALE,
        help=f"Cartesian Kp/Kd scale in (0,1] (default: {base.DEFAULT_CART_GAIN_SCALE})",
    )
    parser.add_argument(
        "--support-scale",
        type=float,
        default=inplace.DEFAULT_SUPPORT_SCALE,
        help=f"gravity feedforward scale in [0,1] (default: {inplace.DEFAULT_SUPPORT_SCALE})",
    )
    parser.add_argument(
        "--stand-height",
        type=float,
        default=DEFAULT_STAND_HEIGHT_M,
        help=f"standing hip-to-foot height, m (default: {DEFAULT_STAND_HEIGHT_M})",
    )
    parser.add_argument(
        "--travel-scale",
        type=float,
        default=1.0,
        help="stand path fraction for supported tests; crawl requires 1.0",
    )
    parser.add_argument(
        "--shift-distance",
        type=float,
        default=DEFAULT_SHIFT_DISTANCE_M,
        help=(
            "diagonal pre-liftoff body shift, m "
            f"(default: {DEFAULT_SHIFT_DISTANCE_M})"
        ),
    )
    parser.add_argument(
        "--step-length",
        type=float,
        default=DEFAULT_STEP_LENGTH_M,
        help=(
            "forward placement per foot, m; 0 crawls in place "
            f"(default: {DEFAULT_STEP_LENGTH_M})"
        ),
    )
    parser.add_argument(
        "--swing-test",
        action="store_true",
        help=(
            "isolate leg swing by forcing zero forward step length; body "
            "SHIFT/RECENTER remains active for static stability"
        ),
    )
    parser.add_argument(
        "--observe-phases",
        action="store_true",
        help="pause at every crawl phase boundary until ENTER",
    )
    parser.add_argument(
        "--swing-height",
        type=float,
        default=DEFAULT_SWING_HEIGHT_M,
        help=f"swing-foot clearance, m (default: {DEFAULT_SWING_HEIGHT_M})",
    )
    parser.add_argument(
        "--unload-tau-trip",
        type=float,
        default=DEFAULT_UNLOAD_TAU_TRIP_NM,
        help=(
            "pre-lift gate: swing pitch/knee |tau_fb| must stay below this "
            f"before liftoff, N*m (default: {DEFAULT_UNLOAD_TAU_TRIP_NM})"
        ),
    )
    parser.add_argument(
        "--leg-cycle",
        type=float,
        default=DEFAULT_LEG_CYCLE_S,
        help=f"time slot per leg, seconds (default: {DEFAULT_LEG_CYCLE_S})",
    )
    parser.add_argument(
        "--gait-cycles",
        type=int,
        default=DEFAULT_GAIT_CYCLES,
        help=f"four-leg gait cycles per ENTER (default: {DEFAULT_GAIT_CYCLES})",
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
    parser.add_argument(
        "--no-overspeed",
        action="store_true",
        help=(
            "disable the overspeed e-stop (both driver and encoder speed "
            "tiers) for this run; the fast leg swing near the crouched "
            "geometry can trip the encoder tier.  Joint-limit, over-temp, "
            "CAN-miss and motor-fault e-stops stay active."
        ),
    )
    parser.add_argument(
        "--no-imu",
        action="store_true",
        help=(
            "skip the trunk IMU: no rp=() on the [legs] line and no "
            f"{inplace.TIPOVER_ABORT_DEG:.0f} deg tip-over abort"
        ),
    )
    parser.add_argument(
        "--self-test", action="store_true", help="validate gait without opening CAN"
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
    if not MIN_CRAWL_STAND_HEIGHT_M <= args.stand_height <= inplace.MAX_STAND_HEIGHT:
        parser.error(
            f"--stand-height must be between {MIN_CRAWL_STAND_HEIGHT_M} and "
            f"{inplace.MAX_STAND_HEIGHT} m"
        )
    if not 0.05 <= args.travel_scale <= 1.0:
        parser.error("--travel-scale must be between 0.05 and 1.0")
    if not 0.02 <= args.shift_distance <= 0.05:
        parser.error("--shift-distance must be between 0.02 and 0.05 m")
    if not 0.0 <= args.step_length <= 0.06:
        parser.error("--step-length must be between 0 and 0.06 m")
    if args.swing_test:
        args.step_length = 0.0
    if not 0.015 <= args.swing_height <= 0.05:
        parser.error("--swing-height must be between 0.015 and 0.05 m")
    if not 0.1 <= args.unload_tau_trip <= 2.0:
        parser.error("--unload-tau-trip must be between 0.1 and 2.0 N*m")
    if not 2.0 <= args.leg_cycle <= 4.0:
        parser.error("--leg-cycle must be between 2 and 4 seconds")
    if not 1 <= args.gait_cycles <= 20:
        parser.error("--gait-cycles must be between 1 and 20")
    if not 1.0 <= args.crouch_max_speed_dps <= 600.0:
        parser.error("--crouch-max-speed-dps must be between 1 and 600")
    if not 0.1 <= args.crouch_torque_trip <= base.STAGED_TAU_MAX:
        parser.error(
            f"--crouch-torque-trip must be between 0.1 and {base.STAGED_TAU_MAX} N*m"
        )
    if not 0.1 <= args.crouch_speed_trip <= 3.0:
        parser.error("--crouch-speed-trip must be between 0.1 and 3.0 rad/s")
    if not 0.0 < args.qd_estop <= base.QD_ESTOP_CEILING:
        parser.error(
            f"--qd-estop must be > 0 and <= {base.QD_ESTOP_CEILING} rad/s"
        )
    if not args.qd_estop < args.qd_estop_hard <= base.QD_ESTOP_CEILING:
        parser.error(
            "--qd-estop-hard must be above --qd-estop and "
            f"<= {base.QD_ESTOP_CEILING} rad/s"
        )
    try:
        if args.self_test:
            return offline_self_test(args)
        return run_hardware(args)
    except (RuntimeError, ValueError) as exc:
        print(f"[crawl] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
