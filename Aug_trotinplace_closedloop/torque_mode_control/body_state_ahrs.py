#!/usr/bin/env python3
"""Trunk state from the AHRS and the encoders alone -- no EKF, nothing to drift.

WHY THIS CAN WORK AT ALL
    The VMC needs r_z, v, C and omega.  The obvious objection to dropping the
    EKF is velocity: nothing else in this tree estimates it, and the Aug
    README says so ("FK gives no velocity without differentiating noisy
    encoders").  That is true of differentiating FK POSITIONS.  It is not the
    only way to get velocity out of the legs.

    A planted foot is fixed in the world.  Its world position is
        p_i = r + C^T s_i        s_i = foot_position(leg, q_i), body frame
    Differentiate with p_i constant, using d(C^T)/dt = C^T [omega]x :
        0 = rdot + C^T (omega x s_i) + C^T (J_i qdot_i)
        => v_world = -C^T ( omega x s_i + J_i qdot_i )
    averaged over the planted set.  qdot comes from the motors' own speed
    field, which arrives in the same 0xA1 reply as the encoder, so this is a
    direct algebraic read at 250 Hz -- no integration, no filter state, and
    nothing that can drift.  It is the same information the EKF's leg-odometry
    measurement uses; with four feet down it determines the trunk completely.

    Height is the drift-free FK height the position track already trusts, and
    attitude comes from the DETA10's own fusion, which experiment A
    (stand_ahrs_level_hw) already closed a leveling loop on successfully --
    on the day the EKF's attitude split 4-5 degrees during a ramp.

THE BOUNDARY, STATED LOUDLY
    ALL OF THIS ASSUMES PLANTED FEET.  Below MIN_PLANTED_FOR_ODOM the trunk
    velocity is no longer determined and `BodyState.read()` returns
    active=False rather than a plausible-looking fiction.  A flight phase, or
    a stance that degenerates to a diagonal pair, is where the EKF stops being
    optional -- see the Aug README's "which estimate to trust for what".
    Stage 1 stands on four feet, so stage 1 does not need it.  Stage 3 will.

WHAT IT PRODUCES
    The dict `dog5_vmc_core.body_wrench` already consumes -- r, v, C, w_hat --
    so the wrench layer cannot tell the difference between this and the EKF.

RUN
    A library: it opens no bus, commands nothing, and is safe to import from
    a notebook.  What exercises it:

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    cd /home/robot01/Documents/can_motor_control/dog_stand_compliance_control/Aug_trotinplace_closedloop

    $V self-test/test_body_state.py        # 22 gates, no hardware
      # [1] is the one that matters: it imposes a trunk twist, solves for the
      # joint rates that keep every foot nailed to the world, and demands the
      # twist back.  If that goes red the AHRS-only design is unsound and the
      # VMC's kd_z/kd_x/kd_y terms are being fed noise.

    On hardware it is driven by torque_stand_hw.py (via torque_worker), never
    on its own.  To watch it against a hand-tilted robot at zero torque, the
    read-only harness is:
    $V torque_mode_control/tau_calib_hw.py --hang --tau-max 0.1
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUG = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_AUG)
_DESC = os.path.join(_ROOT, "dog5_description")
for _p in (_HERE, _DESC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dog5_statics as st                                  # noqa: E402
import torque_params as P                                  # noqa: E402

LEGS = st.LEGS
N_LEGS = len(LEGS)


def C_from_rp(roll, pitch):
    """I->B rotation whose (roll, pitch) is exactly the pair given.

    Same construction as selftest_common.C_from_rp and the exact inverse of
    dog5_vmc_core.attitude_error_rp, so a state built here and read back by
    the wrench layer round-trips.  Yaw is deliberately absent: the AHRS's yaw
    is magnetometer-derived and untrusted next to twelve motors and a steel
    frame, and the VMC only ever damps yaw RATE, which comes from the gyro.
    """
    g = np.array([-math.sin(pitch),
                  math.cos(pitch) * math.sin(roll),
                  math.cos(pitch) * math.cos(roll)])
    e = np.array([0.0, 0.0, 1.0])
    v = np.cross(e, g)
    c = float(np.dot(e, g))
    if float(np.linalg.norm(v)) < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]],
                   [v[2], 0.0, -v[0]],
                   [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def leg_odometry_velocity(q, qd, C, omega, planted, frames=None):
    """World-frame trunk velocity from the planted legs.  See the header.

    Returns (v_world(3,), n_planted).  Raises nothing on an empty set -- the
    caller decides what too-few-feet means; this just reports the count.
    """
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    omega = np.asarray(omega, dtype=float)
    planted = np.asarray(planted, dtype=bool)

    acc = np.zeros(3)
    n = 0
    for i in range(N_LEGS):
        if not planted[i]:
            continue
        sl = slice(3 * i, 3 * i + 3)
        fr = st.leg_frames(LEGS[i], q[sl]) if frames is None else frames[i]
        foot, anchors, axes, _ = fr
        J = st.foot_jacobian_from(foot, anchors, axes)
        # body-frame velocity of the foot relative to the trunk, plus the
        # part the trunk's own rotation contributes at that lever arm
        acc += -(np.cross(omega, foot) + J @ qd[sl])
        n += 1
    if n == 0:
        return np.zeros(3), 0
    return C.T @ (acc / n), n


def fk_trunk_height(q, C, planted, frames=None):
    """Height of the trunk origin above the plane of the planted feet (m).

    Drift-free by construction: encoders and attitude only, no integration.
    The foot SITE is the centre of a 20 mm sphere, so the floor is
    FOOT_RADIUS_M below it -- the same convention the position track uses, and
    the reason every "trunk height" printed in this repo needs to say which
    point it means.  This one is FLOOR to TRUNK ORIGIN (the hip-axis plane).
    """
    q = np.asarray(q, dtype=float)
    planted = np.asarray(planted, dtype=bool)
    drops = []
    for i in range(N_LEGS):
        if not planted[i]:
            continue
        sl = slice(3 * i, 3 * i + 3)
        fr = st.leg_frames(LEGS[i], q[sl]) if frames is None else frames[i]
        # foot site in the WORLD-aligned frame, relative to the trunk origin
        drops.append(-(C.T @ fr[0])[2])
    if not drops:
        return float("nan")
    return float(np.mean(drops)) + _FOOT_RADIUS_M


# the position track's constant; imported rather than redeclared where it is
# available, but this module must stay importable without the runners
_FOOT_RADIUS_M = 0.020


class BodyState:
    """AHRS + encoders -> the est_out dict the VMC layer already speaks.

    `read()` returns (out, active, reason).  `active` is False -- and the
    caller must freeze rather than command -- when the AHRS has gone stale or
    too few feet are planted for the odometry to be determined.  It never
    returns a stale-but-plausible state: in torque mode a plausible fiction
    drives real force.
    """

    def __init__(self, ahrs, setpoint_roll=0.0, setpoint_pitch=0.0,
                 lpf_fc_hz=P.ODOM_LPF_FC_HZ, stale_s=P.AHRS_STALE_S,
                 min_planted=P.MIN_PLANTED_FOR_ODOM):
        self.ahrs = ahrs
        self.sp_roll = float(setpoint_roll)
        self.sp_pitch = float(setpoint_pitch)
        self.lpf_fc_hz = float(lpf_fc_hz)
        self.stale_s = float(stale_s)
        self.min_planted = int(min_planted)
        self.v = np.zeros(3)              # low-passed odometry velocity
        self.z0 = None                    # FK height at the first good read
        self._last_t = None
        self.n_planted = 0
        self.roll = self.pitch = 0.0

    def read(self, now, q, qd, planted):
        planted = np.asarray(planted, dtype=bool)
        sample = self.ahrs.sample()
        if sample is None or self.ahrs.is_stale(self.stale_s):
            return None, False, "AHRS stale"
        if int(planted.sum()) < self.min_planted:
            return None, False, (f"only {int(planted.sum())} planted feet; "
                                 f"leg odometry needs {self.min_planted}")

        self.roll = math.radians(sample.roll_deg) - self.sp_roll
        self.pitch = math.radians(sample.pitch_deg) - self.sp_pitch
        C = C_from_rp(self.roll, self.pitch)
        omega = np.deg2rad([sample.roll_rate_dps,
                            sample.pitch_rate_dps,
                            sample.yaw_rate_dps])

        q = np.asarray(q, dtype=float)
        frames = [st.leg_frames(LEGS[i], q[3 * i:3 * i + 3])
                  for i in range(N_LEGS)]
        v_raw, n = leg_odometry_velocity(q, qd, C, omega, planted, frames)
        z = fk_trunk_height(q, C, planted, frames)
        self.n_planted = n

        # first-order low pass.  The motors' speed field is quantised at
        # 0.1 dps and twelve of them sum into kd_z, so the raw odometry is
        # noticeably grainy; this is the only filter state in the module and
        # it is on a velocity, not a position, so it cannot integrate an error.
        dt = P.SWEEP_S if self._last_t is None else max(now - self._last_t,
                                                        1e-4)
        self._last_t = now
        alpha = 1.0 - math.exp(-2.0 * math.pi * self.lpf_fc_hz * dt)
        self.v += alpha * (v_raw - self.v)

        if self.z0 is None:
            self.z0 = z

        out = {
            # r is only ever read for r[2] by body_wrench, and z is absolute
            # floor-referenced FK, so x/y stay at zero rather than pretending
            # to an origin nothing observes
            "r": np.array([0.0, 0.0, z]),
            "v": self.v.copy(),
            "C": C,
            "w_hat": omega,
            "healthy": True,
            "z_fk": z,
            "n_planted": n,
            "v_raw": v_raw,
        }
        return out, True, ""

    def status(self):
        return (f"roll{math.degrees(self.roll):+5.2f} "
                f"pitch{math.degrees(self.pitch):+5.2f} deg  "
                f"v({self.v[0]*1e3:+.0f},{self.v[1]*1e3:+.0f},"
                f"{self.v[2]*1e3:+.0f})mm/s  planted{self.n_planted}")
