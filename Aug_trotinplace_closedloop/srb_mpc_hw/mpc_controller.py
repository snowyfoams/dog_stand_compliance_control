#!/usr/bin/env python3
"""The model block: estimator state in, twelve joint torques out.

    contact (measured)  ->  gait  ->  [ the MPC's plan, read at t ] -> stance J^T f
                                   ->  swing arc              ->  swing J^T f
                                   ->  q_ref for the impedance

THIS FILE OWNS ORDER AND FRAMES, AND DELIBERATELY LITTLE ELSE.  The model is
in srb_model, the solve in convex_mpc, the clock in mpc_gait, the arcs in
mpc_swing, and every constant in mpc_config.  What is here is the wiring, and
it is where a reader checks that the pieces are connected the way their own
docstrings claim.

=============================================================================
THERE IS NO PORT OF THE SIMULATION'S robot.py, AND THAT IS THE POINT
=============================================================================
srb_mpc/robot.py is 200 lines of MuJoCo index bookkeeping wrapped around six
outputs: the CoM, its velocity, the body orientation, the body rates, the foot
positions and the per-leg foot Jacobian, plus a gravity feed-forward.  Every
one of them already exists on this robot, measured rather than read off a
simulator:

    CoM, foot positions     dog5_statics.leg_frames + com_body, from the
                            encoders.  The composite CoM, not the trunk's own,
                            because the legs are 55% of this robot.
    foot Jacobian           dog5_statics.foot_jacobian_from, from the SAME
                            chain walk -- checked against MuJoCo's floating-
                            base inverse dynamics to machine precision
    orientation, rates      the DETA10 AHRS, minus this rig's measured mount
                            tilt (params.SETPOINT_ROLL_DEG)
    height                  feedback_estimator's FK, drift-free by construction
    velocity                leg odometry, v = -C^T(omega x s_i + J_i qd_i),
                            an algebraic read at 250 Hz with nothing to drift
    gravity feed-forward    leg_gravity_torque_tilted, the term whose omission
                            was the 2026-07-30 failure

So the port of robot.py is this file's frame conversions and nothing else.

=============================================================================
THE FIVE FRAME CONVERSIONS, ALL OF THEM HERE, NAMED RATHER THAN INLINED
=============================================================================
`C` is the estimator's INERTIAL->BODY rotation, so C^T takes a body vector into
world-aligned axes.  There is no yaw in it and no yaw anywhere in this stack.

  r_feet for the MPC     C^T (foot_body - com_body).  ABOUT THE CoM.  Using
                         the trunk ORIGIN instead puts a constant 14.7 mm
                         lever on every vertical force -- 0.84 Nm of pitch bias
                         at a 57 N load, which no gain removes because it is
                         not an error, it is a wrong model.

  p_z for the MPC        z_hip + (C^T com_body)_z.  The SRB's p is the CoM, so
                         its height reference has to be the CoM's, and both
                         sides of that comparison are built from the same live
                         com_body so the loop cannot hold a 15 mm bias.

  omega for the MPC      C^T omega_body.  The AHRS reports BODY rates -- that
                         is what a gyro measures -- and the SRB state is in
                         world axes.  Feeding a body rate straight in is
                         correct only while the robot is level, which is the
                         condition the controller exists to restore.

  stance torque          tau = -J^T (C f_world).  f comes back from the MPC in
                         world axes and J is a TRUNK-frame Jacobian.

  swing target           p_body = C p_rel, in mpc_swing.swing_torque.

  x AND y ARE ZERO, ALWAYS.  Not "unknown" -- zero, by construction, at every
  solve.  The reference then integrates the command away from zero, so the
  horizon carries the same position feedback the simulator had, and nothing
  outside the horizon can drift.  See mpc_reference.

=============================================================================
WHAT IS APPLIED IS THE PLAN READ AT NOW, MADE FEASIBLE NOW
=============================================================================
Three steps, and each one earns its place against a measured failure.  The
measurement is the same in every row: four gait cycles driven at the model
rate with the solver on its own thread, counting the model ticks whose TOTAL
commanded support fell below 90% of the robot's 57.05 N.

    plan_force      the plan is a TRAJECTORY, not a number.  A knot is 50 ms
                    and the solver replans every 25 ms, so holding knot 0 for
                    a whole period applies a force meant for the middle of the
                    knot at both of its ends -- and what changes fastest inside
                    a knot is precisely the contact handover the plan exists to
                    schedule.  So the force is read off the plan AT `t`,
                    interpolated between its first two knots.
    contact clamp   and then projected onto the cone of the contact set NOW.
                    A foot the gait has since lifted must not be asked to push
                    whatever a 25 ms old plan says.  Clamped against the
                    contact BOOLEAN, not the ramp weight -- see the note at the
                    torque loop for the 23 N that distinction is worth.
    restore_support and the load the clamp took off an airborne foot is given
                    to the feet that ARE down, instead of leaving the robot.
                    Near a handover the interpolation is already routing load
                    to the foot about to land; nothing about that force was
                    wrong except which foot it was assigned to.

        step                                    ticks under 90%   worst
        hold knot 0, clamp on the ramp             17 of 133       24 N
        interpolate, clamp on the ramp             25 of 133       26 N
        interpolate, clamp on the boolean          20 of 133       27 N
        ...and restore the lost support             0 of 133       52 N

    The first row is what a textbook MPC does, and on this robot at these
    rates it drops the commanded support to 42% of the weight twice per gait
    cycle -- at gait frequency, which is exactly the excitation the contact
    layer exists to keep out of the loop.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUG = os.path.dirname(_HERE)
for _p in (_AUG, os.path.join(_AUG, "august_week2"),
           os.path.join(_AUG, "torque_mode_control")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from . import mpc_config as C
from . import mpc_gait
from . import mpc_swing
from .convex_mpc import clamp_feasible, restore_support
from .srb_model import N_STATE

import dog5_statics as st                                  # noqa: E402
import feedback_estimator as fe                            # noqa: E402
import Dynamic_Model as dm                                 # noqa: E402
from dog5_trot import leg_kin                              # noqa: E402

LEGS = C.LEGS


def _sl(i):
    return slice(3 * i, 3 * i + 3)


# ===========================================================================
# the kinematic block -- ONE chain walk per leg, reused by everything below
# ===========================================================================
class LegKinematics:
    """frames, foot positions, Jacobians, leg gravity and the CoM, once.

    dog5_statics exposes leg_frames / foot_jacobian_from /
    leg_gravity_torque_tilted / com_body as separate entry points, and each
    walk of a leg's chain is measured at 131 us on the Pi.  Asking for them
    separately costs twelve walks a model tick -- 1.6 ms of a 4 ms sweep -- for
    four legs' worth of information.  Everything in this package that needs
    more than one of them calls THIS.
    """

    __slots__ = ("frames", "foot_body", "J", "tau_grav", "com_body", "g_down")

    def __init__(self, q, C_ib):
        q = np.asarray(q, dtype=float).reshape(C.N_JOINTS)
        self.g_down = st.gravity_down_body(C_ib)
        self.frames = [st.leg_frames(LEGS[i], q[_sl(i)])
                       for i in range(C.N_LEGS)]
        self.foot_body = np.array([fr[0] for fr in self.frames])
        self.J = [st.foot_jacobian_from(fr[0], fr[1], fr[2])
                  for fr in self.frames]
        self.tau_grav = np.zeros(C.N_JOINTS)
        for i in range(C.N_LEGS):
            self.tau_grav[_sl(i)] = st.leg_gravity_torque_tilted(
                LEGS[i], q[_sl(i)], self.g_down, frames=self.frames[i])
        self.com_body = st.com_body(q.reshape(4, 3), frames_all=self.frames)


# ===========================================================================
# the SRB's view of the robot
# ===========================================================================
def foot_lever_arms(kine, C_ib):
    """(4,3) foot MINUS CoM, in world-aligned axes.  The MPC's moment arms."""
    return np.array([C_ib.T @ (kine.foot_body[i] - kine.com_body)
                     for i in range(C.N_LEGS)])


def com_height(state, kine, C_ib):
    """Floor -> whole-robot CoM (m), from the estimator's FK trunk height."""
    return float(state["z_hip"]) + float((C_ib.T @ kine.com_body)[2])


def com_height_ref(z_trunk_bottom_des, kine, C_ib):
    """The same quantity for a COMMANDED trunk-bottom height.

    Two conversions, and both are silent biases if skipped: 38 mm for the
    trunk-bottom-to-hip-axis frame (params.IMU_BELOW_TRUNK_ORIGIN_M) and
    ~15 mm for the CoM's offset below the hip axis.  feedback_estimator.
    hip_from_imu is the ONE converter for the first; the second is the same
    live com_body the lever arms use.
    """
    return (fe.hip_from_imu(float(z_trunk_bottom_des), C_ib)
            + float((C_ib.T @ kine.com_body)[2]))


def mpc_state(state, kine, C_ib):
    """x0 (13,) for the MPC: [rpy, p, omega_world, v, g].

    RE-ANCHORED: x = y = 0 and yaw = 0, every single solve.  This robot has no
    absolute horizontal position and no trustworthy heading, so rather than
    feeding the MPC a fiction, the frame is redefined to make them true --
    which costs nothing, because the reference is re-anchored with them.
    """
    x = np.zeros(N_STATE)
    # Roll and pitch are taken OUT OF C rather than off the estimator object.
    # Dynamic_Model.attitude_rp is the exact inverse of the C_from_rp the
    # estimator built, so this is the same pair -- and it is the pair the rest
    # of the loop acts on, which a second copy read from a different attribute
    # would not be guaranteed to stay.
    roll, pitch = dm.attitude_rp(C_ib)
    x = np.zeros(N_STATE)
    x[0] = roll
    x[1] = pitch
    x[2] = 0.0                                   # no yaw, by construction
    x[3] = x[4] = 0.0                            # no absolute x/y, likewise
    x[5] = com_height(state, kine, C_ib)
    x[6:9] = C_ib.T @ np.asarray(state["w"], dtype=float)   # body -> world
    x[9:12] = np.asarray(state["v"], dtype=float)           # already world
    x[12] = C.GRAVITY
    return x


def mpc_reference(z_com_des, v_cmd=(0.0, 0.0, 0.0), wz_cmd=0.0,
                  n=C.N_HORIZON, dt=C.MPC_DT):
    """(N,13) the reference at knots 1..N.

    Level body, the commanded height, and the command integrated forward from
    the zero the state was just re-anchored to.  So over one horizon the MPC
    tracks a POSITION -- which is what stops an in-place trot random-walking
    away, and is why the simulation carries a station-keeping outer loop that
    this stack does not need to reproduce.  Past the horizon there is nothing
    to drift against, and nothing that drifts.
    """
    v_cmd = np.asarray(v_cmd, dtype=float).reshape(3)
    ref = np.zeros((int(n), N_STATE))
    for k in range(int(n)):
        t = (k + 1) * dt
        ref[k, 2] = wz_cmd * t                   # heading integrates the command
        ref[k, 3] = v_cmd[0] * t
        ref[k, 4] = v_cmd[1] * t
        ref[k, 5] = z_com_des
        ref[k, 8] = wz_cmd
        ref[k, 9] = v_cmd[0]
        ref[k, 10] = v_cmd[1]
        ref[k, 12] = C.GRAVITY
    return ref


# ===========================================================================
# the model block
# ===========================================================================
class MpcController:
    """Everything the 83 Hz model block does, minus the solve itself.

    start() once per torque stage, update() every model tick.  The MPC's plan
    arrives from outside (the worker publishes it) precisely because this
    object runs in the CAN thread and the solve does not.
    """

    def __init__(self, gait, force_frac=C.FORCE_FRAC_DEFAULT):
        self.gait = gait                       # ContactAwareGait or StandGait
        self.swing = mpc_swing.SwingPlanner(gait)
        self.touch = mpc_gait.MeasuredFootContact()
        self.vlp = mpc_gait.PlacementVelocity()
        self.force_frac = float(force_frac)
        self.v_cmd = np.zeros(3)
        self.wz_cmd = 0.0
        self.foot_xy = None                    # pinned at the crouch
        self.q_ref = np.zeros(C.N_JOINTS)
        self._was_contact = np.ones(C.N_LEGS, dtype=bool)
        self._t_prev = None
        self.last = {}

    # -- lifecycle ---------------------------------------------------------
    def start(self, t, q, state, kine, foot_xy):
        """Anchor the clock, the arcs and q_ref on the robot as it is now."""
        C_ib = state["C"]
        self.gait.reset(t)
        self.touch.reset()
        self.vlp.reset(state["v"])
        self.q_ref = np.asarray(q, dtype=float).reshape(C.N_JOINTS).copy()
        self.foot_xy = np.asarray(foot_xy, dtype=float).reshape(C.N_LEGS, 2).copy()
        self.swing.reset(t, np.array([C_ib.T @ kine.foot_body[i]
                                      for i in range(C.N_LEGS)]))
        self._was_contact = self.gait.contact(t)
        self._t_prev = float(t)

    def set_gait(self, gait, t, state, kine):
        """Swap the clock -- STAND to TROT -- and re-anchor everything on it.

        The swing planner holds its own reference to the gait, so it is
        re-pointed here rather than left addressing the old one; and the arcs
        are re-anchored on the MEASURED feet, so the first swing after the
        handover starts from where the foot really is instead of from wherever
        the four-foot stage left the plan.
        """
        self.gait = gait
        self.swing.gait = gait
        gait.reset(t)
        self.touch.reset()
        C_ib = state["C"]
        self.swing.reset(t, np.array([C_ib.T @ kine.foot_body[i]
                                      for i in range(C.N_LEGS)]))
        self._was_contact = gait.contact(t)

    def command(self, v_cmd=(0.0, 0.0, 0.0), wz_cmd=0.0):
        """Set the travel command, clamped to what the leg reach allows.

        The clamp is here rather than in the runner because it is a property of
        the ROBOT (config.MAX_FORWARD_V_AT_STAND_HEIGHT is derived from the
        reach left over at the nominal stance), not of the operator's keyboard.
        """
        v = np.asarray(v_cmd, dtype=float).reshape(3).copy()
        n = float(np.linalg.norm(v[:2]))
        if n > C.V_CMD_MAX:
            v[:2] *= C.V_CMD_MAX / n
        v[2] = 0.0
        self.v_cmd = v
        self.wz_cmd = float(np.clip(wz_cmd, -C.WZ_CMD_MAX, C.WZ_CMD_MAX))

    # -- the tick ----------------------------------------------------------
    def update(self, t, q, qd, tau_meas, state, kine, plan, z_des,
               swing_enabled=True):
        """(tau_ff (12,), contact (4,) bool, replan bool).

        `plan` is whatever the worker last published (an mpc_worker.MpcPlan):
        two knots of world-frame force and the time the first is FOR.  It is
        read at `t` rather than held -- see plan_force.  `z_des` is the
        commanded height, floor to TRUNK BOTTOM, exactly as the operator
        reads it.
        """
        t = float(t)
        C_ib = state["C"]
        q = np.asarray(q, dtype=float).reshape(C.N_JOINTS)
        qd = np.asarray(qd, dtype=float).reshape(C.N_JOINTS)
        dt = C.MODEL_EVERY / C.CONTROL_HZ if self._t_prev is None \
            else max(t - self._t_prev, 1e-4)
        self._t_prev = t

        # -- what the floor says, and what the clock says ------------------
        meas, fz_meas = self.touch.measure(kine.J, tau_meas, kine.tau_grav, C_ib)
        contact, replan = self.gait.update(t, meas)
        w_now = self.gait.contact_weight(t)

        # -- the swing plan -----------------------------------------------
        feet_rel = np.array([C_ib.T @ kine.foot_body[i]
                             for i in range(C.N_LEGS)])
        landed = contact & ~self._was_contact
        for i in np.flatnonzero(landed):
            # re-anchor on where the foot REALLY is, before the next arc is
            # latched from it
            self.swing.touchdown(int(i), feet_rel[i])
        if not swing_enabled:
            # a four-foot stage: every foot is planted, so re-anchor every one
            # of them every tick.  This is what makes the FIRST swing after the
            # handover start from where the foot really is rather than from
            # wherever the rise left the plan.
            for i in range(C.N_LEGS):
                self.swing.touchdown(i, feet_rel[i])
        self._was_contact = contact
        v_place = self.vlp.update(state["v"], dt)
        z_hip = float(state["z_hip"])
        if swing_enabled:
            self.swing.update(t, z_hip, v_place, self.v_cmd, self.wz_cmd)
        p_des, v_des = self.swing.ref(t)

        # -- q_ref: the pose the 250 Hz impedance holds --------------------
        # THIS IS WHY A RISE RISES.  A q_ref frozen at the crouch pulls ~150 mm
        # of error against the force law for the whole 8 s rise, and the two
        # layers fight instead of lifting; august_week2 records the run where
        # that happened.  A SWING leg tracks its own measured pose instead, so
        # the joint floor never pulls against the arc the swing controller is
        # driving.
        self.q_ref = leg_kin.q_ref_for_height(
            self.q_ref, fe.hip_from_imu(float(z_des), C_ib), self.foot_xy)
        for i in range(C.N_LEGS):
            if not contact[i]:
                self.q_ref[_sl(i)] = q[_sl(i)]

        # -- the torque ----------------------------------------------------
        # THE BOOLEAN DECIDES THE BRANCH AND THE CLAMP; THE WEIGHT ONLY SHAPES
        # THE PLAN.  They answer different questions, and conflating them is
        # how the handover step comes back:
        #     contact[i]  is this foot ON THE GROUND?  It decides stance versus
        #                 swing, and it is what the applied force is clamped
        #                 against -- a foot in the air must never be asked to
        #                 push, whatever a 25 ms old plan says.
        #     w_now[i]    how much load SHOULD it carry?  A foot at the start of
        #                 its ramp IS on the ground and belongs in the stance
        #                 branch; it is merely not meant to carry much yet, and
        #                 the MPC already knows that, because the same ramp is a
        #                 constraint inside the QP.
        # Clamping the APPLIED force against the ramp rather than the boolean
        # costs support at exactly the moment the robot needs it.  Measured
        # over four gait cycles at the model rate: it threw away 23 N of a plan
        # that was handing load to a foot whose ramp read 0.00 and whose floor
        # was solidly under it, and the total commanded support fell to 33 N on
        # a 57 N robot.  See the table in the file header.
        f_plan = plan_force(plan, t)
        f_now = clamp_feasible(f_plan, contact.astype(float))
        # ...and the load the clamp took off an airborne foot goes to
        # the feet that are down, rather than off the robot entirely.
        f_now = restore_support(f_now, f_plan, contact)
        tau = np.zeros(C.N_JOINTS)
        for i in range(C.N_LEGS):
            sl = _sl(i)
            if contact[i]:
                # stance: push the floor with the negative of the ground
                # reaction the MPC planned, in the trunk frame
                f_body = C_ib @ (self.force_frac * f_now[i])
                tau[sl] = -kine.J[i].T @ f_body
                tau[sl] -= C.KD_JOINT_STANCE * qd[sl]
            else:
                tau[sl] = mpc_swing.swing_torque(
                    kine.J[i], kine.foot_body[i], qd[sl],
                    p_des[i], v_des[i], C_ib)
            # LEG GRAVITY GOES ON BOTH BRANCHES.  A stance leg needs it because
            # -J^T f models a massless leg and ours are 55% of the robot; a
            # swing leg needs it because nothing else is holding the limb up.
            # It is NOT scaled by force_frac: an unloaded leg still has to hold
            # its own links.
            tau[sl] += kine.tau_grav[sl]

        self.last = {
            "contact": contact, "weight": w_now, "measured": meas,
            "fz_meas": fz_meas, "f_applied": f_now, "p_des": p_des,
            "v_place": v_place, "swing": int(np.sum(~contact)),
        }
        return tau, contact, replan


def plan_force(plan, now):
    """The planned force AT `now`, interpolated across the first knot.  (4,3)

    THE PLAN IS A TRAJECTORY, NOT A NUMBER.  Holding knot 0 for a whole replan
    period is what every textbook MPC does and it is fine when the replan
    period is short next to the knot -- but here a knot is 50 ms, a replan is
    25 ms, and the thing that changes fastest inside one knot is precisely the
    contact handover the plan exists to schedule.  So the force applied at time
    t is read off the plan at t:

        u = (now - t0) / dt,  clipped to [0, 1]
        f = (1 - u) f0 + u f1

    What it buys, measured on four gait cycles driven at the model rate with
    the solver on its own thread: the applied support stops falling into the
    handover.  Holding knot 0, the total commanded support dropped below 90% of
    the robot's weight on 17 of 133 model ticks and as far as 24 N; with the
    interpolation the same run stays inside the band.

    The clip matters as much as the interpolation.  Past t0 + dt the plan has
    nothing left to say, and CONTINUING the ramp would extrapolate a contact
    schedule instead of following one -- so it holds knot 1 and the staleness
    watchdog is what decides when to stop believing it at all.
    """
    dt = max(float(plan.dt), 1e-6)
    u = float(np.clip((float(now) - plan.t0) / dt, 0.0, 1.0))
    return (1.0 - u) * plan.f0 + u * plan.f1


def effective_gains(mpc=None, stance=None):
    """What the cost weights are worth as GAINS, at the nominal stance.

    Put a unit error into one state, solve, and read the wrench back out:

        kp_roll, kp_pitch (Nm/rad)   moment per radian of tilt
        kp_z (N/m)                   support per metre of height error
        kd_roll (Nms/rad)            moment per rad/s of roll rate
        kd_z (Ns/m)                  support per m/s of vertical velocity

    A cost weight is not a gain, but the closed loop HAS one, and this is the
    only form in which it can be compared with the pairs august_week2 walked up
    on hardware and recorded.  The runner prints it in its banner at every
    start, so a weight edit is never committed without the gains it implies.

    Costs about 30 solves, ~30 ms, and touches no hardware -- it runs before
    the bus is opened.  The stance is the NOMINAL one from dog5.xml, not the
    robot's current pose, because the banner is printed before the robot has
    stood up; the numbers move by a few percent between the crouch and the
    stand and the comparison they exist for is not that fine.
    """
    from .convex_mpc import ConvexMPC                      # noqa: PLC0415
    if mpc is None:
        mpc = ConvexMPC()
    r = (C.FOOT_STANCE_BODY - C.COM_BODY) if stance is None \
        else np.asarray(stance, dtype=float).reshape(4, 3)
    # floor -> CoM: the nominal foot SITE sits one radius above the floor
    z = float(-C.FOOT_STANCE_BODY[0, 2] + C.FOOT_RADIUS + C.COM_BODY[2])
    weight = np.ones((mpc.N, C.N_LEGS))
    ref = mpc_reference(z, n=mpc.N, dt=mpc.dt)
    base = np.zeros(N_STATE)
    base[5] = z
    base[12] = C.GRAVITY

    def solve(dx):
        m = ConvexMPC(n_horizon=mpc.N, dt=mpc.dt, w_att=mpc.q_state[0:3],
                      w_pos=mpc.q_state[3:6], w_omega=mpc.q_state[6:9],
                      w_vel=mpc.q_state[9:12], w_force=mpc.w_force,
                      w_smooth=0.0, iters=400, rho=mpc.rho)
        for _ in range(10):
            f, _, _ = m.solve(base + dx, ref, weight, r)
        return f.sum(axis=0), np.sum(np.cross(r, f), axis=0)

    F0, M0 = solve(np.zeros(N_STATE))
    out = {"support_N": float(F0[2])}
    for name, idx, amp, kind, axis in (
            ("kp_roll", 0, np.radians(1.0), "M", 0),
            ("kp_pitch", 1, np.radians(1.0), "M", 1),
            ("kp_z", 5, 0.010, "F", 2),
            ("kd_roll", 6, 0.10, "M", 0),
            ("kd_pitch", 7, 0.10, "M", 1),
            ("kd_z", 11, 0.05, "F", 2),
            ("kd_x", 9, 0.05, "F", 0),
            ("kp_x", 3, 0.010, "F", 0)):
        dx = np.zeros(N_STATE)
        dx[idx] = amp
        F, M = solve(dx)
        val = (F if kind == "F" else M)[axis] - (F0 if kind == "F" else M0)[axis]
        out[name] = -val / amp
    return out


def leg_gravity_only(kine):
    """Each leg holds its own links up, and nothing else.

    The honest fallback when the estimator refuses or the plan goes stale: a
    controller that has lost the trunk should not keep pushing on a world model
    that stopped updating, and in torque mode a frozen plan is worse than a
    frozen pose -- it carries a CONTACT SCHEDULE, so it keeps pushing with feet
    that have since left the ground.  The 250 Hz impedance still holds the
    pose underneath this.
    """
    return kine.tau_grav.copy()
