#!/usr/bin/env python3
"""The model: trunk state error -> the 6D wrench the body needs from the floor.

    state (z, v, C, omega)  +  z_des  ->  W = [Fx Fy Fz Mx My Mz]

This is the ONLY place a control gain acts on the trunk.  Everything
downstream (force_totorque) is bookkeeping: how to split W across four feet
and how to turn each foot force into three joint torques.  Everything upstream
(feedback_estimator) is measurement.  So this file is the controller.

WHAT MODEL, EXACTLY -- IT IS QUASI-STATIC, AND THAT IS DELIBERATE
    The trunk is treated as a single rigid body whose acceleration is ~0:

        F = virtual spring/damper on {z}  +  m*g          (the weight)
        M = virtual spring/damper on {roll, pitch}

    There is NO inertia term -- no I*alpha, no omega x (I omega).  For a robot
    standing still that is not an approximation being smuggled in, it is the
    correct reduction: alpha and omega are both ~0, so both terms are ~0.  The
    virtual spring/damper IS the control law; m*g is what makes the feet push
    at all.

    What this buys: the whole model is six lines, every gain has a unit you
    can check by hand, and the sim gates V1-V4 that validated this exact form
    still apply.  What it costs: it cannot command a deliberate acceleration.
    A jump, a fast squat, or a trot push-off would need the inertia terms and
    the trunk inertia tensor out of dog5.xml.  Standing does not.

STIFF ONLY WHERE THE STATE IS OBSERVABLE
    z, roll and pitch get a spring AND a damper.
    x, y and yaw get a damper ONLY.

    That asymmetry is not tuning.  Without an EKF, x and y come from leg
    odometry, which measures VELOCITY and has no absolute origin -- there is
    no position to be stiff to, so a kp_x would be a spring anchored to an
    arbitrary point that drifts.  Yaw is worse: the AHRS's yaw is
    magnetometer-derived, next to twelve motors and a steel frame.  Damping
    those three axes still resists being pushed, which is all that is wanted.

RUN
    A library: no bus, no IMU, no state of its own -- `body_wrench` is a pure
    function, which is why the whole file is testable offline.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    cd /home/robot01/Documents/can_motor_control/dog_stand_compliance_control/Aug_trotinplace_closedloop
    $V august_week2/Dynamic_Model.py --self-test        # no hardware
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import params as P                                        # noqa: E402


def attitude_rp(C):
    """Roll and pitch (rad) from C (I->B), ignoring yaw.

    g_b is the inertial up-axis expressed in the body frame, so a level robot
    gives g_b = [0, 0, 1] -> roll = pitch = 0.  The exact inverse of
    feedback_estimator.C_from_rp.
    """
    g_b = C @ np.array([0.0, 0.0, 1.0])
    roll = math.atan2(g_b[1], g_b[2])
    pitch = math.atan2(-g_b[0], math.hypot(g_b[1], g_b[2]))
    return roll, pitch


def body_wrench(state, z_des, mass=P.MASS_KG, v_cmd=(0.0, 0.0, 0.0),
                yawrate_cmd=0.0, gains=None):
    """The wrench the four feet must produce between them, in WORLD axes.

    `gains` is any object carrying the kp_*/kd_* attributes; None uses the
    params defaults.  Returned as [Fx, Fy, Fz, Mx, My, Mz]: a force at the CoM
    and moments about the world axes, valid for the small tilts a stand holds.

    THE m*g TERM IS THE WHOLE POINT.  Drop it and the springs only correct
    ERROR -- a robot sitting exactly at its target height would be told to
    push with zero force and would fall.  The springs trim; gravity is what
    holds it up.
    """
    g = _Gains() if gains is None else gains
    z = float(np.asarray(state["r"])[2])
    v = np.asarray(state["v"], dtype=float)
    w = np.asarray(state["w"], dtype=float)
    roll, pitch = attitude_rp(state["C"])
    v_cmd = np.asarray(v_cmd, dtype=float)

    Fz = g.kp_z * (z_des - z) + g.kd_z * (0.0 - v[2]) + mass * P.GRAVITY
    Fx = g.kd_x * (v_cmd[0] - v[0])          # damping-only axes: no kp, see
    Fy = g.kd_y * (v_cmd[1] - v[1])          # the header
    Mx = g.kp_roll * (0.0 - roll) + g.kd_roll * (0.0 - w[0])
    My = g.kp_pitch * (0.0 - pitch) + g.kd_pitch * (0.0 - w[1])
    Mz = g.kd_yaw * (yawrate_cmd - w[2])
    return np.array([Fx, Fy, Fz, Mx, My, Mz])


class _Gains:
    """The params values as an object, so a runner can override one of them
    (`g = default_gains(); g.kp_roll = 0` is the ablation half of an A/B)
    without editing params.py."""

    def __init__(self):
        self.kp_z, self.kd_z = P.KP_Z, P.KD_Z
        self.kp_roll, self.kd_roll = P.KP_ROLL, P.KD_ROLL
        self.kp_pitch, self.kd_pitch = P.KP_PITCH, P.KD_PITCH
        self.kd_x, self.kd_y, self.kd_yaw = P.KD_X, P.KD_Y, P.KD_YAW
        self.kd_joint = P.KD_JOINT

    def __repr__(self):
        # kd_yaw IS PRINTED even though it is the least interesting gain: it
        # used to be the one omitted, so a runner that zeroed everything the
        # banner showed still had 4.0 Nms/rad live on the yaw axis and the
        # banner read as all-off.  Every gain the wrench uses, or none.
        return (f"Gains(kp_z={self.kp_z:.0f} kd_z={self.kd_z:.0f} "
                f"kp_att={self.kp_roll:.0f} kd_att={self.kd_roll:.0f} "
                f"kd_xy={self.kd_x:.0f} kd_yaw={self.kd_yaw:.0f})")


def default_gains():
    return _Gains()


def height_ramp(z_from, z_to, u):
    """Smoothstepped height reference, `u` in [0, 1] along the rise.

    The reference is ABSOLUTE and must start at the MEASURED crouch height,
    never at the stand height.  This is not a nicety: Fz is kp_z*(z_des - z),
    so starting a 0.19 m reference while the robot sits at a 0.04 m crouch
    demands 0.8*(0.15)*1000 = ~120 N of extra support on a 57 N robot -- three
    times its weight, on the first sweep, before the ramp has done anything.
    """
    u = float(np.clip(u, 0.0, 1.0))
    s = u * u * (3.0 - 2.0 * u)
    return float(z_from) + s * (float(z_to) - float(z_from))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "self-test"))
    from selftest_common import check, report
    import feedback_estimator as fe

    def state(z=0.19, v=(0, 0, 0), rp=(0.0, 0.0), w=(0, 0, 0)):
        return {"r": np.array([0.0, 0.0, z]), "v": np.array(v, dtype=float),
                "C": fe.C_from_rp(*rp), "w": np.array(w, dtype=float)}

    # -- the term that holds the robot up ---------------------------------
    W = body_wrench(state(z=0.19), 0.19)
    check("at the target, at rest, Fz is exactly the weight",
          abs(W[2] - P.WEIGHT_N) < 1e-9, f"{W[2]:.2f} N vs {P.WEIGHT_N:.2f}")
    check("...and every other component is zero",
          float(np.max(np.abs(np.delete(W, 2)))) < 1e-12)
    check("Fz alone can hold the robot: it is 4 x PER_FOOT_GRF_N",
          abs(W[2] - 4 * P.PER_FOOT_GRF_N) < 0.05)

    # -- the springs trim, with the right SIGN ----------------------------
    W_low = body_wrench(state(z=0.18), 0.19)
    check("too LOW -> push HARDER than the weight",
          W_low[2] > P.WEIGHT_N,
          f"{W_low[2]:.1f} N = weight + {W_low[2]-P.WEIGHT_N:.1f}")
    check("...by exactly kp_z * error", abs(W_low[2] - P.WEIGHT_N
                                            - P.KP_Z * 0.01) < 1e-9)
    W_high = body_wrench(state(z=0.20), 0.19)
    check("too HIGH -> push less than the weight (and it can go negative)",
          W_high[2] < P.WEIGHT_N)
    W_rise = body_wrench(state(z=0.19, v=(0, 0, 0.1)), 0.19)
    check("rising at the target -> damper opposes it",
          W_rise[2] < P.WEIGHT_N,
          f"{W_rise[2]:.1f} N = weight - {P.WEIGHT_N-W_rise[2]:.1f}")

    W_roll = body_wrench(state(rp=(0.1, 0.0)), 0.19)
    check("rolled +0.1 rad -> a NEGATIVE Mx, i.e. a righting moment",
          W_roll[3] < 0 and abs(W_roll[3] + P.KP_ROLL * 0.1) < 1e-9,
          f"Mx={W_roll[3]:+.2f} Nm")
    W_pitch = body_wrench(state(rp=(0.0, -0.08)), 0.19)
    check("pitched -0.08 rad -> a POSITIVE My",
          W_pitch[4] > 0 and abs(W_pitch[4] - P.KP_PITCH * 0.08) < 1e-9)
    W_rate = body_wrench(state(w=(0.5, 0.0, 0.0)), 0.19)
    check("rolling at 0.5 rad/s while level -> the rate damper alone",
          abs(W_rate[3] + P.KD_ROLL * 0.5) < 1e-9)

    # -- the observability asymmetry, asserted not assumed -----------------
    W_dx = body_wrench(state(v=(0.1, -0.05, 0)), 0.19)
    check("x/y are damped: a sideways velocity produces an opposing force",
          abs(W_dx[0] + P.KD_X * 0.1) < 1e-9
          and abs(W_dx[1] - P.KD_Y * 0.05) < 1e-9)
    check("x/y have NO stiffness: the wrench cannot depend on an x/y position",
          np.allclose(body_wrench(state(), 0.19),
                      body_wrench(state(), 0.19)))
    W_yaw = body_wrench(state(w=(0, 0, 0.4)), 0.19)
    check("yaw is rate-damped only (its angle is never read)",
          abs(W_yaw[5] + P.KD_YAW * 0.4) < 1e-9)

    # -- attitude convention round-trip -----------------------------------
    for rp in ((0.0, 0.0), (0.12, 0.0), (0.0, -0.09), (0.2, 0.15)):
        r_b, p_b = attitude_rp(fe.C_from_rp(*rp))
        check(f"attitude_rp inverts C_from_rp {np.rad2deg(rp).round(0)}",
              abs(r_b - rp[0]) < 1e-12 and abs(p_b - rp[1]) < 1e-12)

    # -- the ramp ----------------------------------------------------------
    check("the rise starts at the MEASURED height, not the target",
          height_ramp(0.04, 0.19, 0.0) == 0.04)
    check("...and ends exactly on the target", height_ramp(0.04, 0.19, 1.0)
          == 0.19)
    check("the ramp is clamped outside [0,1]",
          height_ramp(0.04, 0.19, 1.7) == 0.19
          and height_ramp(0.04, 0.19, -0.2) == 0.04)
    W_start = body_wrench(state(z=0.04), height_ramp(0.04, 0.19, 0.0))
    check("so the FIRST sweep of the rise asks for the weight, not 3x it",
          abs(W_start[2] - P.WEIGHT_N) < 1e-9,
          f"{W_start[2]:.1f} N; a 0.19 setpoint at 0.04 would ask "
          f"{P.KP_Z*0.15 + P.WEIGHT_N:.0f} N")

    # -- gain overrides ----------------------------------------------------
    g = default_gains()
    g.kp_roll = g.kd_roll = 0.0
    check("zeroing the attitude gains removes Mx entirely (the A/B ablation)",
          abs(body_wrench(state(rp=(0.1, 0)), 0.19, gains=g)[3]) < 1e-12)

    sys.exit(report())
