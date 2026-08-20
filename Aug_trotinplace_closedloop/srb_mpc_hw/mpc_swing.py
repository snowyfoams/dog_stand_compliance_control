#!/usr/bin/env python3
"""Where a swinging foot goes, and the torque that takes it there.

    SwingPlanner   the reference: a per-leg arc from where the foot left the
                   ground to where it should next land
    swing_torque   a Cartesian impedance that tracks that arc

Split so that "where to step" can be wrong without the tracking being wrong,
and the other way round.

=============================================================================
THE FRAME, WHICH IS THE ONE THING THIS PORT HAD TO CHANGE
=============================================================================
The simulation plans footsteps in the WORLD: it reads the floating base's
absolute x, y and yaw off the simulator and places feet against them.  This
robot has none of those.  There is no absolute horizontal position anywhere in
the stack, and the only yaw available is a magnetometer's, sitting next to
twelve motors and a steel frame.

So the arcs live in the TRUNK-ANCHORED, GRAVITY-ALIGNED frame:

    origin      the trunk origin (the hip-axis plane), wherever it is
    axes        world axes de-rotated by the AHRS attitude, so +z is up and
                the floor is flat -- but NOT yaw-rotated, because there is no
                yaw to rotate by
    a point     p_rel = C^T p_body,  and back again p_body = C p_rel

Everything the planner needs is expressible in it: the nominal stance foot is
a body constant, the Raibert terms are velocities the estimator measures, and
the ground is a plane at the measured trunk height.  Nothing in it can drift,
because nothing in it is integrated.

WHAT THE FRAME COSTS, IN MILLIMETRES, STATED RATHER THAN HIDDEN
    The frame TRANSLATES with the robot, so a point latched at liftoff travels
    with the body during the swing instead of staying put on the floor.  Over
    one 160 ms swing that is v * T_swing:

        trotting in place       under 1 mm
        at V_CMD_MAX, 43 mm/s   6.9 mm

    and it enters as a landing point 6.9 mm behind where the world-frame plan
    would have put it -- against a Raibert forward term of 0.5 * T_st * v =
    5.2 mm at the same speed.  That is the honest statement of what this stack
    is: a trot IN PLACE with a small velocity command, which is also all the
    leg reach at this stance height allows (config.MAX_FORWARD_V_AT_STAND_
    HEIGHT is 43 mm/s, and it is derived from the geometry, not chosen).
    A trot that travels needs the world frame back, and that needs the EKF.

THE LANDING POINT IS RAIBERT'S, AND EACH TERM EARNS ITS PLACE
        p_land = p_nominal + (T_st/2)(v + w_cmd x r) + k_v (v - v_cmd)

    p_nominal       THE STANCE FOOT POSITION, not the point under the hip.
                    The textbook form uses the hip, which is right for a leg
                    that hangs vertically at rest.  DOG5's does not: its
                    abduction sits at ~90 deg, so the nominal foot is 120 mm
                    forward and 52 mm outboard of its own hip, and aiming at
                    the hip commands a 120 mm backwards lunge from standing.
                    dog5_trot/swing.py records catching exactly that.
    (T_st/2) v      the SYMMETRY term: land where the body WILL be halfway
                    through the coming stance, so the stance sweeps the foot
                    front to back symmetrically and nets no push
    w_cmd x r       the same term for the commanded YAW RATE: a turning body
                    carries each hip sideways, and a foot placed as if it were
                    not fights the turn.  The simulation folds this into its
                    world-frame yaw instead; with no yaw here it is explicit.
    k_v (v - v_cmd) the only FEEDBACK term, and what makes v_cmd a command
                    rather than a description

    Its z is where a foot SITE rests, which is the floor plus FOOT_RADIUS --
    the site is the centre of a 20 mm contact sphere, so landing the site ON
    the floor drives the foot 20 mm into it every step.  Flat ground, stated:
    this package has no terrain map and does not pretend to.

THE ARC HAS ZERO VELOCITY AT BOTH ENDS, AND THAT IS THE WHOLE DESIGN
    Horizontal and vertical are both smoothstep, 3u^2 - 2u^3, whose derivative
    vanishes at u = 0 and u = 1.  So the foot leaves with no impulsive tug
    against a foot still carrying load, and arrives with zero velocity in ALL
    THREE axes -- no vertical impact, and no horizontal scuff dragging the
    robot sideways at the instant the MPC starts trusting that foot.  A sine
    bump, the usual shortcut, has a vertical speed of pi*h/T = 0.79 m/s at
    both ends, straight into the floor.
"""
from __future__ import annotations

import numpy as np

from . import mpc_config as C


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _smoothstep_d(u):
    u = np.clip(u, 0.0, 1.0)
    return 6.0 * u * (1.0 - u)


class SwingPlanner:
    """Per-leg swing arcs in the trunk-anchored, gravity-aligned frame.

    update() is called every model tick and latches a new arc for any leg that
    has just entered swing; ref() is a pure evaluation of the latched arcs, so
    it can be called as often as wanted and never moves the plan underneath
    the controller.  That separation is why a swing does not wander when the
    control rate changes.
    """

    def __init__(self, gait, step_height=C.SWING_HEIGHT,
                 raibert_kv=C.RAIBERT_KV, reach_frac=C.REACH_FRAC):
        self.gait = gait
        self.step_height = float(step_height)
        self.raibert_kv = float(raibert_kv)
        self.reach_frac = float(reach_frac)
        self.p_lift = np.zeros((C.N_LEGS, 3))     # latched at liftoff
        self.p_land = np.zeros((C.N_LEGS, 3))     # latched at liftoff
        self.p_anchor = np.zeros((C.N_LEGS, 3))   # last known real foot
        self.z_site = -0.17                       # ground plane in this frame
        self._was_swing = np.zeros(C.N_LEGS, dtype=bool)
        self._armed = False

    # -- setup -------------------------------------------------------------
    def reset(self, t, feet_rel):
        """Latch the MEASURED feet as anchors and clear the swing edges.

        `feet_rel` is (4,3) in this module's frame -- C^T applied to the FK
        foot positions.  Without it the first swing would start from the
        origin, which is inside the robot.
        """
        self.p_anchor = np.asarray(feet_rel, dtype=float).reshape(C.N_LEGS, 3).copy()
        self.p_lift = self.p_anchor.copy()
        self.p_land = self.p_anchor.copy()
        self.z_site = float(np.mean(self.p_anchor[:, 2]))
        self._was_swing = ~self.gait.contact(t)
        self._armed = True

    def touchdown(self, leg, p_rel):
        """Re-anchor one leg on the foot position actually measured at contact.

        A foot lands early on a bump and late in a hollow, and either way the
        NEXT swing must start from where the foot really is.  Nothing in this
        file infers contact -- the estimator owns that, and a planner that
        guessed would be a second, disagreeing contact detector.
        """
        self.p_anchor[int(leg)] = np.asarray(p_rel, dtype=float).reshape(3)

    # -- per model tick ----------------------------------------------------
    def update(self, t, z_hip, v_place, v_cmd=(0.0, 0.0, 0.0), wz_cmd=0.0):
        """Latch an arc for every leg that entered swing since the last call.

        z_hip    floor -> TRUNK ORIGIN (m), from the estimator's FK.  The one
                 place the ground enters, and it is measured, not assumed.
        v_place  the LOW-PASSED trunk velocity, this frame's axes.  Raw
                 velocity oscillates at gait frequency and wobbles the
                 touchdown targets in resonance; see mpc_gait.PlacementVelocity
        v_cmd    commanded travel (m/s), same axes
        wz_cmd   commanded yaw rate (rad/s)
        """
        if not self._armed:
            raise RuntimeError(
                "SwingPlanner.reset(t, feet_rel) must be called before "
                "update(): without measured feet the first swing starts from "
                "the trunk origin, which is inside the robot")
        v_place = np.asarray(v_place, dtype=float).reshape(3)
        v_cmd = np.asarray(v_cmd, dtype=float).reshape(3)
        # The foot SITE rests one radius above the floor; z_hip is measured to
        # the trunk origin, so the site sits that far below it.
        self.z_site = -(float(z_hip) - C.FOOT_RADIUS)

        swinging = ~self.gait.contact(t)
        T_st = self.gait.stance_duration
        w_vec = np.array([0.0, 0.0, float(wz_cmd)])
        for i in range(C.N_LEGS):
            if not (swinging[i] and not self._was_swing[i]):
                continue
            # rising edge: this leg has just lifted
            self.p_lift[i] = self.p_anchor[i].copy()
            nom = np.array([C.FOOT_STANCE_BODY[i, 0],
                            C.FOOT_STANCE_BODY[i, 1], self.z_site])
            v_hip = v_place + np.cross(w_vec, nom)      # yaw carries the hip
            land = nom + 0.5 * T_st * v_hip \
                + self.raibert_kv * (v_place - v_cmd)
            land[2] = self.z_site
            self.p_land[i] = self._clamp_reachable(i, land)
        self._was_swing = swinging

    def _clamp_reachable(self, leg, land):
        """Pull a landing point back inside the leg's reach, from its own hip.

        A Raibert step at speed, or a large v_cmd, asks for a foot the leg
        cannot meet.  Clamping HERE means the impedance is never handed an
        impossible target -- the failure becomes a SHORTER STEP, which the gait
        survives, instead of a saturated joint command, which it does not.
        """
        d = land - C.HIP_OFFSET[leg]
        reach = float(np.linalg.norm(d))
        max_reach = self.reach_frac * C.LEG_REACH      # off the singularity
        if reach > max_reach:
            return C.HIP_OFFSET[leg] + d * (max_reach / reach)
        return land

    def ref(self, t):
        """(p_des (4,3), v_des (4,3)) in this module's frame.

        A STANCE LEG RETURNS ITS ANCHOR AND ZERO VELOCITY, not a garbage value.
        The caller masks by contact anyway, but a NaN or a stale arc here would
        be one indexing slip away from being applied as torque.
        """
        s = self.gait.swing_phase(t)
        stance = self.gait.contact(t)
        T_sw = self.gait.swing_duration
        p = np.empty((C.N_LEGS, 3))
        v = np.zeros((C.N_LEGS, 3))
        for i in range(C.N_LEGS):
            if stance[i]:
                p[i] = self.p_anchor[i]
                continue
            u = float(s[i])
            g, gd = _smoothstep(u), _smoothstep_d(u) / T_sw
            span = self.p_land[i] - self.p_lift[i]
            p[i] = self.p_lift[i] + span * g
            v[i] = span * gd
            # the vertical bump: up over the first half, down over the second,
            # each half its own smoothstep so the APEX has zero speed too
            if u < 0.5:
                b, bd = _smoothstep(2 * u), _smoothstep_d(2 * u) * 2.0 / T_sw
            else:
                b, bd = (_smoothstep(2 - 2 * u),
                         -_smoothstep_d(2 - 2 * u) * 2.0 / T_sw)
            p[i, 2] += self.step_height * b
            v[i, 2] += self.step_height * bd
        return p, v


def swing_torque(J, foot_body, qd_leg, p_des_rel, v_des_rel, C_ib,
                 kp=C.KP_SWING, kd=C.KD_SWING):
    """Cartesian impedance on one swinging foot.  (3,) joint torque.

        f_body = kp (C p_des - p_foot) + kd (C v_des - J qd)
        tau    = J^T f_body

    THE ROTATION IS THE WHOLE DIFFERENCE between a swing that tracks and one
    that drifts as the trunk pitches: the arc is planned in gravity-aligned
    axes and J is a TRUNK-frame Jacobian, so the target has to come back
    through C before it meets J^T.

    NOTE tau = +J^T f HERE, against the -J^T f of a stance leg, and both are
    right.  A stance leg is handed the GROUND REACTION and must push the floor
    with its negative; a swing leg is handed the force it should apply to its
    own foot.  dog5_trot/controller.py records the same pair of signs.

    THE TRUNK'S ANGULAR VELOCITY IS NOT IN THE FOOT VELOCITY.  J qd is the
    foot's velocity RELATIVE TO THE TRUNK, which is exactly what v_des_rel is
    the derivative of, so the two are consistent and no omega term is missing
    from the difference.  What IS approximated is that C is held constant
    across the derivative; at the 0.3 rad/s a trot in place turns, the dropped
    dC/dt term is under 0.08 m/s at a 0.25 m lever, which against KD_SWING is
    0.6 N -- 1% of the weight.  It is the first thing to add when yaw rate
    goes up.
    """
    C_ib = np.asarray(C_ib, dtype=float).reshape(3, 3)
    p_des = C_ib @ np.asarray(p_des_rel, dtype=float).reshape(3)
    v_des = C_ib @ np.asarray(v_des_rel, dtype=float).reshape(3)
    v_foot = J @ np.asarray(qd_leg, dtype=float).reshape(3)
    f = kp * (p_des - np.asarray(foot_body, dtype=float).reshape(3)) \
        + kd * (v_des - v_foot)
    return J.T @ f
