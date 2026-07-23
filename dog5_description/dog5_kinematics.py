"""Independent NumPy forward kinematics and foot Jacobians for DOG5.

All positions are expressed in the trunk frame and all joint angles are in
radians.  The joint vector for one leg is always::

    q = [hip_abduction, hip_pitch, knee]

This module intentionally does not import MuJoCo.  Its fixed geometry is copied
from the body and foot-site transforms in ``dog5.xml``; visual mesh transforms
are not part of the kinematic chain.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LEGS = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class LegGeometry:
    """Fixed translations for one three-hinge leg, in parent-body frames."""

    hip: tuple[float, float, float]
    hip_to_pitch: tuple[float, float, float]
    pitch_to_knee: tuple[float, float, float]
    knee_to_foot: tuple[float, float, float]


# Controlled copy of these dog5.xml fields:
#   hip body pos, thigh body pos, shin body pos, and foot site pos.
LEG_GEOMETRY = {
    "FL": LegGeometry(
        hip=(0.2205, 0.060000, 0.0),
        hip_to_pitch=(0.0484776, 0.0, -0.052859),
        pitch_to_knee=(0.1200000, 0.0, 0.0),
        knee_to_foot=(0.1124000, -0.0011, -0.0003),
    ),
    "FR": LegGeometry(
        hip=(0.2205, -0.060000, 0.0),
        hip_to_pitch=(0.0484776, 0.0, -0.052859),
        pitch_to_knee=(0.1200000, 0.0, 0.0),
        knee_to_foot=(0.1124000, -0.0011, -0.0003),
    ),
    "RL": LegGeometry(
        hip=(-0.2205, 0.059999, 0.0),
        hip_to_pitch=(-0.0484786, 0.0, -0.052859),
        pitch_to_knee=(-0.1200000, 0.0, 0.0),
        knee_to_foot=(-0.1124000, 0.0011, -0.0003),
    ),
    "RR": LegGeometry(
        hip=(-0.2205, -0.060001, 0.0),
        hip_to_pitch=(-0.0484786, 0.0, -0.052859),
        pitch_to_knee=(-0.1200000, 0.0, 0.0),
        knee_to_foot=(-0.1124000, 0.0011, -0.0003),
    ),
}

_X_AXIS = np.array([1.0, 0.0, 0.0])
_Z_AXIS = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class LinkInertial:
    mass: float
    com: tuple[float, float, float]


# Controlled copy of the dog5.xml hip/thigh/shin body inertials (mass and
# inertial pos in the owning body frame).  Front and rear pairs mirror in x,
# exactly as in the XML.
_FRONT_LINK_INERTIALS = (
    LinkInertial(0.39152916, (0.04378628, -0.00000166, -0.05507725)),
    LinkInertial(0.36932519, (0.11348071, 0.00014012, -0.01397932)),
    LinkInertial(0.0381955, (0.07623007, -0.00009226, -0.00002971)),
)
_REAR_LINK_INERTIALS = (
    LinkInertial(0.39152916, (-0.04378728, 0.00000166, -0.05507725)),
    LinkInertial(0.36932519, (-0.11348071, -0.00014012, -0.01397932)),
    LinkInertial(0.0381955, (-0.07623007, 0.00009226, -0.00002971)),
)
LINK_INERTIALS = {
    "FL": _FRONT_LINK_INERTIALS,
    "FR": _FRONT_LINK_INERTIALS,
    "RL": _REAR_LINK_INERTIALS,
    "RR": _REAR_LINK_INERTIALS,
}
_GRAVITY_M_S2 = 9.81


def _checked_q(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (3,):
        raise ValueError(f"q must have shape (3,), got {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError("q must contain only finite values")
    return q


def _geometry(leg: str) -> LegGeometry:
    try:
        return LEG_GEOMETRY[leg]
    except KeyError as exc:
        raise ValueError(f"unknown leg {leg!r}; expected one of {LEGS}") from exc


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _state(leg: str, q) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return foot position, three joint anchors, and three joint axes."""
    q_abd, q_pitch, q_knee = _checked_q(q)
    geom = _geometry(leg)

    hip = np.asarray(geom.hip)
    hip_to_pitch = np.asarray(geom.hip_to_pitch)
    pitch_to_knee = np.asarray(geom.pitch_to_knee)
    knee_to_foot = np.asarray(geom.knee_to_foot)

    # Each body has one hinge at its origin.  Child-body translations are
    # rotated by the accumulated orientation of their parent body.
    r_hip = _rot_x(q_abd)
    pitch = hip + r_hip @ hip_to_pitch

    r_thigh = r_hip @ _rot_z(q_pitch)
    knee = pitch + r_thigh @ pitch_to_knee

    r_shin = r_thigh @ _rot_z(q_knee)
    foot = knee + r_shin @ knee_to_foot

    anchors = np.stack((hip, pitch, knee), axis=0)
    axes = np.stack(
        (
            _X_AXIS,
            r_hip @ _Z_AXIS,
            r_thigh @ _Z_AXIS,
        ),
        axis=0,
    )
    return foot, anchors, axes


def foot_position(leg: str, q) -> np.ndarray:
    """Return the foot-site position in the trunk frame, in metres."""
    foot, _, _ = _state(leg, q)
    return foot


def foot_jacobian(leg: str, q) -> np.ndarray:
    """Return ``d(foot_position)/dq`` as a 3x3 geometric Jacobian."""
    foot, anchors, axes = _state(leg, q)
    return np.column_stack(
        [np.cross(axis, foot - anchor) for anchor, axis in zip(anchors, axes)]
    )


def leg_gravity_torque(leg: str, q) -> np.ndarray:
    """Torque the three motors must hold against the leg's own link weights.

    Statics with no foot contact: measured joint torque equals this value, so
    subtracting it from torque feedback isolates the external (ground) load.
    Assumes the trunk is horizontal (gravity along trunk -z).
    """
    q_abd, q_pitch, q_knee = _checked_q(q)
    geom = _geometry(leg)

    hip = np.asarray(geom.hip)
    r_hip = _rot_x(q_abd)
    pitch = hip + r_hip @ np.asarray(geom.hip_to_pitch)
    r_thigh = r_hip @ _rot_z(q_pitch)
    knee = pitch + r_thigh @ np.asarray(geom.pitch_to_knee)
    r_shin = r_thigh @ _rot_z(q_knee)

    anchors = (hip, pitch, knee)
    axes = (_X_AXIS, r_hip @ _Z_AXIS, r_thigh @ _Z_AXIS)
    com_frames = ((hip, r_hip), (pitch, r_thigh), (knee, r_shin))

    tau = np.zeros(3)
    for link_index, inertial in enumerate(LINK_INERTIALS[leg]):
        origin, rotation = com_frames[link_index]
        com = origin + rotation @ np.asarray(inertial.com)
        weight = inertial.mass * _GRAVITY_M_S2
        # A joint only carries links at or below it in the chain.
        for joint_index in range(link_index + 1):
            dcom = np.cross(axes[joint_index], com - anchors[joint_index])
            tau[joint_index] += weight * dcom[2]
    return tau
