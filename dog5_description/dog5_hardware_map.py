"""Confirmed DOG5 physical CAN-to-joint mapping.

This is the single executable source of truth shared by the zero-torque
direction verifier and the hardware stand controller.  Canonical controller
order is ``[FL, FR, RL, RR] x [abd, pitch, knee]``.

The CAN assignment was confirmed on the assembled robot:

* RR: 1, 2, 3
* RL: 4, 5, 6
* FL: 7, 8, 9
* FR: 10, 11, 12

Direction values are the confirmed hardware values and are intentionally kept
separate from the physical-ID confirmation.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareJoint:
    leg: str
    joint: str
    model_name: str
    can_id: int
    direction: int


HARDWARE_JOINTS = (
    HardwareJoint("FL", "abd",   "hip_abd_FL",    7, +1),
    HardwareJoint("FL", "pitch", "hip_pitch_FL",  8, +1),
    HardwareJoint("FL", "knee",  "knee_FL",       9, -1),
    HardwareJoint("FR", "abd",   "hip_abd_FR",   10, +1),
    HardwareJoint("FR", "pitch", "hip_pitch_FR", 11, +1),
    HardwareJoint("FR", "knee",  "knee_FR",      12, -1),
    HardwareJoint("RL", "abd",   "hip_abd_RL",    4, -1),
    HardwareJoint("RL", "pitch", "hip_pitch_RL",  5, +1),
    HardwareJoint("RL", "knee",  "knee_RL",       6, -1),
    HardwareJoint("RR", "abd",   "hip_abd_RR",    1, -1),
    HardwareJoint("RR", "pitch", "hip_pitch_RR",  2, +1),
    HardwareJoint("RR", "knee",  "knee_RR",       3, -1),
)

MOTOR_JOINT_MAP = {
    joint.can_id: (joint.model_name, joint.direction)
    for joint in HARDWARE_JOINTS
}


def _validate() -> None:
    can_ids = [joint.can_id for joint in HARDWARE_JOINTS]
    if sorted(can_ids) != list(range(1, 13)):
        raise ValueError(f"DOG5 hardware map must contain CAN IDs 1..12: {can_ids}")
    if any(joint.direction not in (-1, +1) for joint in HARDWARE_JOINTS):
        raise ValueError("DOG5 hardware directions must be +1 or -1")


_validate()
