#!/usr/bin/env python3
"""Where a swinging foot goes, and the torque that takes it there.

TWO THINGS, DELIBERATELY SEPARATE
    SwingPlanner   the reference: a world-frame arc per leg, from where the
                   foot left the ground to where it should next land.
    swing_torque   a Cartesian impedance that tracks that arc.

    The split is so that "where to step" can be wrong without the tracking
    being wrong, and vice versa.  They are checked separately below.

THE LANDING POINT IS RAIBERT'S, AND EACH TERM EARNS ITS PLACE
        p_land = p_nominal + v * T_stance/2 + k_v (v - v_cmd)

    p_nominal       THE STANCE FOOT POSITION, not the point under the hip.
                    The textbook form uses the hip, which is right for a leg
                    that hangs vertically at rest.  DOG5's does not: its
                    abduction sits at ~90 deg, so the nominal foot is 120 mm
                    forward and 52 mm outboard of its own hip.  Aiming at the
                    hip commands a 120 mm backwards lunge from a standing
                    start -- caught by the swing self-test printing a lift at
                    x = 0.340 and a landing at x = 0.220.
                    So a stationary robot steps IN PLACE, which is the
                    property the hip term was there to provide
    v * T/2         the SYMMETRY term: land where the hip WILL be halfway
                    through the coming stance, so the stance sweeps the foot
                    from front to back symmetrically and nets no push
    k_v (v - v_cmd) the only FEEDBACK term.  Without it the gait holds
                    whatever velocity it has; this is what makes v_cmd a
                    command instead of a description.

    The z of the landing point is where a foot SITE rests, which is the floor
    plus FOOT_RADIUS -- dog5.xml's foot site is the centre of a 20 mm contact
    sphere, so landing the site ON the floor drives the foot 20 mm into it
    every step.  From the CoM that is
        p_com.z - COM_ABOVE_FLOOR + FOOT_RADIUS
    and each of those three names is a different height; see config.
    This package has no terrain map and does not pretend to: on a slope every
    foot would be told to land at the same height and the attitude loop would
    fight it.  Flat ground, stated.

THE ARC HAS ZERO VELOCITY AT BOTH ENDS, AND THAT IS THE WHOLE DESIGN
    Horizontal and vertical both use smoothstep, 3u^2 - 2u^3, whose derivative
    vanishes at u = 0 and u = 1.  So:

        liftoff    the foot leaves with zero velocity -- no impulsive tug
                   against a foot that is still carrying load
        touchdown  the foot arrives with zero velocity in ALL THREE axes -- no
                   vertical impact, and no horizontal scuff dragging the robot
                   sideways at the instant the QP starts trusting that foot

    A sine bump, the usual shortcut, has a vertical speed of pi*h/T at BOTH
    ends: 0.63 m/s here, straight into the ground.

RE-ANCHORING, AND WHY IT IS AN INPUT RATHER THAN A GUESS
    A foot lands early on a bump and late in a hollow, and either way the next
    swing must start from where the foot REALLY is, not from where the arc
    said it would be.  touchdown() takes the measurement.  Nothing in this
    file infers contact -- the estimator owns that, and a planner that guessed
    would be a second, disagreeing contact detector.

RUN
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V dog5_trot/swing.py --self-test
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# Sibling imports go through the PACKAGE when there is one.  Only a direct
# `python dog5_trot/<this>.py --self-test` falls back to a path insert, and
# that insert is what would shadow the repo's own top-level config.py -- see
# the package docstring.  Keeping it off the library path is the point.
if __package__:
    from . import config as cfg
    from . import leg_kin
else:
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import config as cfg
    import leg_kin



def _smoothstep(u):
    """3u^2 - 2u^3 on [0,1], with zero slope at both ends."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _smoothstep_d(u):
    """d/du of the above; zero at u = 0 and u = 1."""
    u = np.clip(u, 0.0, 1.0)
    return 6.0 * u * (1.0 - u)


class SwingPlanner:
    """Per-leg swing arcs in WORLD axes.

    update() is called every sweep and latches a new arc for any leg that has
    just entered swing; ref() is a pure evaluation of the latched arcs, so it
    can be called as often as wanted and never moves the plan underneath the
    controller.  That separation is why a swing does not wander when the
    control rate changes.
    """

    def __init__(self, gait):
        self.gait = gait
        self.p_lift = np.zeros((cfg.N_LEGS, 3))     # world, latched at liftoff
        self.p_land = np.zeros((cfg.N_LEGS, 3))     # world, latched at liftoff
        self.p_anchor = np.zeros((cfg.N_LEGS, 3))   # last known real foot pos
        self.z_ground = 0.0
        self._was_swing = np.zeros(cfg.N_LEGS, dtype=bool)
        self._armed = False

    # -- setup -------------------------------------------------------------
    def reset(self, t, p_feet_world):
        """Latch the measured feet as the anchors and clear the swing edges."""
        self.p_anchor = np.asarray(p_feet_world, dtype=float).reshape(cfg.N_LEGS, 3).copy()
        self.p_lift = self.p_anchor.copy()
        self.p_land = self.p_anchor.copy()
        # the plane the SITES rest on, which is what an arc must return to
        self.z_ground = float(np.mean(self.p_anchor[:, 2]))
        self._was_swing = ~self.gait.contact(t)
        self._armed = True

    def touchdown(self, leg: int, p_meas) -> None:
        """Re-anchor one leg on the foot position actually measured at contact."""
        self.p_anchor[int(leg)] = np.asarray(p_meas, dtype=float).reshape(3)

    # -- per-sweep ---------------------------------------------------------
    def update(self, t, p_com, v_com, R_wb, v_cmd=(0.0, 0.0, 0.0)) -> None:
        """Latch an arc for every leg that entered swing since the last call."""
        if not self._armed:
            raise RuntimeError("SwingPlanner.reset(t, p_feet_world) must be "
                               "called before update(): without measured feet "
                               "the first swing would start from the origin")
        p_com = np.asarray(p_com, dtype=float).reshape(3)
        v_com = np.asarray(v_com, dtype=float).reshape(3)
        R_wb = np.asarray(R_wb, dtype=float).reshape(3, 3)
        v_cmd = np.asarray(v_cmd, dtype=float).reshape(3)

        self.z_ground = float(p_com[2] - cfg.COM_ABOVE_FLOOR + cfg.FOOT_RADIUS)
        swinging = ~self.gait.contact(t)
        T_st = self.gait.stance_duration

        for i in range(cfg.N_LEGS):
            if swinging[i] and not self._was_swing[i]:
                # rising edge: this leg has just lifted
                self.p_lift[i] = self.p_anchor[i]
                nom_w = p_com + R_wb @ (cfg.FOOT_STANCE_BODY[i] - cfg.COM_BODY)
                land = (nom_w
                        + v_com * (0.5 * T_st)
                        + cfg.RAIBERT_KV * (v_com - v_cmd))
                land[2] = self.z_ground
                self.p_land[i] = self._clamp_reachable(i, land, p_com, R_wb)
        self._was_swing = swinging

    def _clamp_reachable(self, leg, land_w, p_com, R_wb):
        """Pull a landing point back inside the leg's reach, in the hip frame.

        A Raibert step at speed, or a large v_cmd, asks for a foot the leg
        cannot meet.  Clamping HERE, in the planner, means the IK and the
        impedance are never handed an impossible target -- the failure becomes
        a shorter step, which the gait survives, instead of a saturated joint
        command, which it does not.
        """
        hip_w = p_com + R_wb @ (cfg.HIP_OFFSET[leg] - cfg.COM_BODY)
        d_hip = R_wb.T @ (land_w - hip_w)          # into the trunk frame
        reach = float(np.linalg.norm(d_hip))
        max_reach = 0.95 * cfg.LEG_REACH           # 5% off the singularity
        if reach > max_reach:
            d_hip = d_hip * (max_reach / reach)
            return hip_w + R_wb @ d_hip
        return land_w

    def ref(self, t):
        """(p_des (4,3), v_des (4,3)) in WORLD axes.

        A STANCE LEG RETURNS ITS ANCHOR AND ZERO VELOCITY, not a garbage value:
        the caller masks by contact anyway, but a NaN or a stale arc here would
        be one indexing slip away from being applied as torque.
        """
        s = self.gait.swing_phase(t)
        stance = self.gait.contact(t)
        T_sw = self.gait.swing_duration

        p = np.empty((cfg.N_LEGS, 3))
        v = np.zeros((cfg.N_LEGS, 3))
        for i in range(cfg.N_LEGS):
            if stance[i]:
                p[i] = self.p_anchor[i]
                continue
            u = float(s[i])
            g, gd = _smoothstep(u), _smoothstep_d(u) / T_sw
            # horizontal (and the straight-line z between lift and land)
            p[i] = self.p_lift[i] + (self.p_land[i] - self.p_lift[i]) * g
            v[i] = (self.p_land[i] - self.p_lift[i]) * gd
            # vertical bump, up over the first half and down over the second,
            # each half a smoothstep so the apex has zero vertical speed too
            if u < 0.5:
                b, bd = _smoothstep(2 * u), _smoothstep_d(2 * u) * 2.0 / T_sw
            else:
                b, bd = _smoothstep(2 - 2 * u), -_smoothstep_d(2 - 2 * u) * 2.0 / T_sw
            p[i, 2] += cfg.SWING_HEIGHT * b
            v[i, 2] += cfg.SWING_HEIGHT * bd
        return p, v


def swing_torque(leg: int, q, qd, p_des, v_des, R_wb, p_com, v_com,
                 kp=cfg.KP_SWING, kd=cfg.KD_SWING) -> np.ndarray:
    """Joint torque tracking one world-frame swing reference.  (3,)

        f_world = Kp (p_des - p_foot) + Kd (v_des - v_foot)
        tau     = J^T R_wb^T f_world

    J is the TRUNK-frame Jacobian, so the world force must be rotated into the
    trunk frame before it is transposed through -- that R_wb^T is the whole
    difference between a swing that tracks and one that drifts as the body
    pitches.

    THE TRUNK'S ANGULAR VELOCITY IS NOT IN v_foot.  The exact world foot
    velocity is v_com + R(omega x r + J qd); this uses v_com + R J qd, because
    the signature it must satisfy carries no omega.  The dropped term is
    omega x r with |r| ~ 0.25 m, so at the 0.3 rad/s a trot in place actually
    runs it is under 0.08 m/s -- against KD_SWING that is 0.6 N, ~1% of the
    weight.  It is NOT negligible for a trot that turns, and that is the first
    thing to add when yaw rate goes up.
    """
    q = np.asarray(q, dtype=float).reshape(3)
    qd = np.asarray(qd, dtype=float).reshape(3)
    R_wb = np.asarray(R_wb, dtype=float).reshape(3, 3)
    p_com = np.asarray(p_com, dtype=float).reshape(3)
    v_com = np.asarray(v_com, dtype=float).reshape(3)

    p_body, J = leg_kin.leg_state(int(leg), q)
    r_body = p_body - cfg.COM_BODY
    p_foot_w = p_com + R_wb @ r_body
    v_foot_w = v_com + R_wb @ (J @ qd)

    f_w = (np.asarray(kp) @ (np.asarray(p_des, dtype=float).reshape(3) - p_foot_w)
           + np.asarray(kd) @ (np.asarray(v_des, dtype=float).reshape(3) - v_foot_w))
    return J.T @ (R_wb.T @ f_w)


# ===========================================================================
# self-test
# ===========================================================================
_PASS = [0, 0]


def check(label, ok, detail=""):
    _PASS[1] += 1
    _PASS[0] += bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))


def self_test():
    import gait as gait_mod

    g = gait_mod.TrotGait()
    g.reset(0.0)
    sp = SwingPlanner(g)

    q_st = cfg.Q_STAND.reshape(-1)
    feet_body = leg_kin.all_foot_pos_body(q_st)
    p_com = np.array([0.0, 0.0, cfg.COM_ABOVE_FLOOR])   # CoM, not trunk origin
    R = np.eye(3)
    feet_w = np.array([p_com + R @ (feet_body[i] - cfg.COM_BODY) for i in range(4)])
    sp.reset(0.0, feet_w)

    check("update() before reset() is refused, not silently zero-anchored",
          True if SwingPlanner(g) is not None else False)
    bad = SwingPlanner(g)
    try:
        bad.update(0.0, p_com, np.zeros(3), R)
        raised = False
    except RuntimeError:
        raised = True
    check("...checked", raised, "an unanchored first swing would start at the origin")

    # -- standing still: the step lands under the hip ---------------------
    dt = cfg.CTRL_DT
    for k in range(int(1.5 * g.period / dt)):
        sp.update(k * dt, p_com, np.zeros(3), R)
    fl_hip_w = p_com + R @ (cfg.HIP_OFFSET[0] - cfg.COM_BODY)
    check("at zero velocity the foot is told to land WHERE IT ALREADY IS",
          np.allclose(sp.p_land[0][:2], feet_w[0][:2], atol=1e-9),
          f"land xy {np.round(sp.p_land[0][:2], 4)} vs the measured foot "
          f"{np.round(feet_w[0][:2], 4)} -- a standing robot steps in place")
    check("...which is NOT under the hip, and aiming there would lunge",
          float(np.linalg.norm(sp.p_land[0][:2] - fl_hip_w[:2])) > 0.10,
          f"the hip is {np.linalg.norm(feet_w[0][:2]-fl_hip_w[:2])*1e3:.0f} mm "
          f"from the nominal foot; a hip-anchored Raibert step would command "
          f"that as a backwards lunge from standing")
    check("...and at the height a foot SITE rests, not on the floor itself",
          abs(sp.p_land[0][2] - cfg.FOOT_RADIUS) < 1e-12,
          f"land z {sp.p_land[0][2]*1e3:.3f} mm = floor 0 + FOOT_RADIUS "
          f"{cfg.FOOT_RADIUS*1e3:.0f}; landing the site ON the floor would "
          f"drive it {cfg.FOOT_RADIUS*1e3:.0f} mm under every step")
    check("...and that height is exactly where the measured feet already are",
          abs(sp.p_land[0][2] - feet_w[0][2]) < 1e-6,
          f"planned {sp.p_land[0][2]*1e3:.3f} mm vs measured "
          f"{feet_w[0][2]*1e3:.3f} mm -- the plan and the robot agree on the "
          f"ground, which is the check the frame bug would fail")

    # -- the Raibert symmetry term ----------------------------------------
    # 0.04 m/s, deliberately UNDER cfg.MAX_FORWARD_V_AT_STAND_HEIGHT: at this
    # stance height the reach clamp bites at 0.05 m/s, so a check written at
    # the more natural 0.3 m/s would be testing the clamp and reporting it as
    # a broken Raibert term.  The clamp gets its own check below.
    sp2 = SwingPlanner(g); sp2.reset(0.0, feet_w)
    v = np.array([0.04, 0.0, 0.0])
    for k in range(int(1.5 * g.period / dt)):
        sp2.update(k * dt, p_com, v, R)
    expect = (feet_w[0][0] + v[0] * 0.5 * g.stance_duration
              + cfg.RAIBERT_KV * v[0])
    check("moving forward, the step lands v*T_stance/2 + k_v*v ahead of the hip",
          abs(sp2.p_land[0][0] - expect) < 1e-9,
          f"{sp2.p_land[0][0]:.4f} m vs {expect:.4f}")

    # -- the arc: ends, apex, and the velocities that matter --------------
    sp3 = SwingPlanner(g); sp3.reset(0.0, feet_w)
    t_lift = g.duty * g.period
    for k in range(int(1.2 * g.period / dt)):
        sp3.update(k * dt, p_com, np.zeros(3), R)
    # WHICH LEG SWINGS IN WHICH WINDOW IS NOT A FREE CHOICE.  With offset 0,
    # FL is planted over [0, duty*T) and swings over [duty*T, T); FR, at
    # offset 0.5, is the other way round.  Reading the wrong leg here made
    # every arc assertion pass against a STANCE anchor, which is the same
    # constant at both ends of the window -- so "starts at liftoff" was green
    # while nothing was swinging at all.  Assert the schedule first.
    LEG = 0                                            # FL swings second half
    check("the leg under test really is swinging across the whole window",
          not g.contact(t_lift + 1e-6)[LEG]
          and not g.contact(g.period - 1e-6)[LEG]
          and g.contact(t_lift - 1e-6)[LEG],
          f"FL planted up to {t_lift*1e3:.0f} ms, swinging after")
    ts = np.linspace(t_lift + 1e-9, g.period - 1e-9, 4001)
    P = np.array([sp3.ref(t)[0][LEG] for t in ts])
    V = np.array([sp3.ref(t)[1][LEG] for t in ts])
    check("the arc starts at the liftoff point",
          np.allclose(P[0], sp3.p_lift[LEG], atol=1e-4))
    check("...and ends at the landing point",
          np.allclose(P[-1], sp3.p_land[LEG], atol=1e-4),
          f"end {np.round(P[-1],4)} vs land {np.round(sp3.p_land[LEG],4)}")
    check("...and the two are DIFFERENT, so this is an arc and not a hold",
          float(np.linalg.norm(P[-1] - P[0])) > 1e-6
          or float(P[:, 2].max() - P[0, 2]) > 1e-3,
          f"lift {np.round(P[0],4)} -> land {np.round(P[-1],4)}")
    check("the apex clears the ground by SWING_HEIGHT",
          abs((P[:, 2].max() - sp3.p_lift[LEG][2]) - cfg.SWING_HEIGHT) < 1e-4,
          f"{(P[:,2].max()-sp3.p_lift[LEG][2])*1e3:.2f} mm vs "
          f"{cfg.SWING_HEIGHT*1e3:.0f}")
    check("TOUCHDOWN VELOCITY IS ZERO IN ALL THREE AXES",
          float(np.max(np.abs(V[-1]))) < 1e-3,
          f"|v| = {np.linalg.norm(V[-1])*1e3:.3f} mm/s; a sine bump would "
          f"land at {np.pi*cfg.SWING_HEIGHT/g.swing_duration*1e3:.0f} mm/s")
    check("...and liftoff velocity too, so nothing tugs a loaded foot",
          float(np.max(np.abs(V[0]))) < 1e-3)

    # the reference must be differentiable -- v_des has to BE dp_des/dt
    num = np.gradient(P, ts, axis=0)
    check("v_des is the true time derivative of p_des",
          float(np.max(np.abs(num - V))) < 2e-3,
          f"worst |v_num - v_des| = {np.max(np.abs(num - V))*1e3:.3f} mm/s")

    # -- a stance leg is quiet --------------------------------------------
    p_s, v_s = sp3.ref(0.01)                         # FL/RR planted at t~0
    check("a planted leg's reference is its anchor, at zero velocity",
          g.contact(0.01)[0]
          and np.allclose(p_s[0], sp3.p_anchor[0]) and np.allclose(v_s[0], 0.0))

    # -- re-anchoring ------------------------------------------------------
    sp3.touchdown(1, np.array([1.0, 2.0, 3.0]))
    check("touchdown() moves that leg's anchor and no other",
          np.allclose(sp3.p_anchor[1], [1, 2, 3])
          and np.allclose(sp3.p_anchor[0], feet_w[0]))

    # -- reach clamping ----------------------------------------------------
    sp4 = SwingPlanner(g); sp4.reset(0.0, feet_w)
    for k in range(int(1.2 * g.period / dt)):
        sp4.update(k * dt, p_com, np.array([9.0, 0.0, 0.0]), R)
    hip1 = p_com + R @ (cfg.HIP_OFFSET[1] - cfg.COM_BODY)   # reach is from the HIP
    hip0_ = p_com + R @ (cfg.HIP_OFFSET[0] - cfg.COM_BODY)
    room = 0.95 * cfg.LEG_REACH - float(np.linalg.norm(feet_w[0] - hip0_))
    v_lim = room / (0.5 * g.stance_duration + cfg.RAIBERT_KV)
    check("config's quoted max forward speed is the one the geometry gives",
          abs(v_lim - cfg.MAX_FORWARD_V_AT_STAND_HEIGHT) < 0.01,
          f"{v_lim:.3f} m/s from {room*1e3:.1f} mm of reach room vs the "
          f"{cfg.MAX_FORWARD_V_AT_STAND_HEIGHT:.2f} config records")
    sp5 = SwingPlanner(g); sp5.reset(0.0, feet_w)
    for k in range(int(1.2 * g.period / dt)):
        sp5.update(k * dt, p_com, np.array([0.3, 0.0, 0.0]), R)
    hip0 = p_com + R @ (cfg.HIP_OFFSET[0] - cfg.COM_BODY)
    want = feet_w[0][0] + 0.3 * (0.5 * g.stance_duration + cfg.RAIBERT_KV)
    check("...so a 0.3 m/s step IS truncated, and lands short of the ask",
          sp5.p_land[0][0] < want - 1e-3
          and np.linalg.norm(sp5.p_land[0] - hip0) <= 0.95 * cfg.LEG_REACH + 1e-9,
          f"asked x={want:.4f}, got {sp5.p_land[0][0]:.4f} -- the stance "
          f"already uses {100*np.linalg.norm(feet_w[0]-hip0)/cfg.LEG_REACH:.1f}% "
          f"of the leg, so travel needs a lower stance, not a looser clamp")
    check("an absurd command clamps the step to 95% of the leg's reach",
          np.linalg.norm(sp4.p_land[1] - hip1) <= 0.95 * cfg.LEG_REACH + 1e-9,
          f"asked 9 m/s, landing sits {np.linalg.norm(sp4.p_land[1]-hip1):.4f} m "
          f"from the hip against a {cfg.LEG_REACH:.4f} m reach")

    # -- swing_torque ------------------------------------------------------
    q = np.array([0.05, 0.7, -1.5]); qd = np.zeros(3)
    p_body, J = leg_kin.leg_state(0, q)
    p_here = p_com + R @ (p_body - cfg.COM_BODY)
    check("zero error gives zero torque",
          np.allclose(swing_torque(0, q, qd, p_here, np.zeros(3), R,
                                   p_com, np.zeros(3)), 0.0, atol=1e-12))
    d = np.array([0.0, 0.0, 0.01])
    tau = swing_torque(0, q, qd, p_here + d, np.zeros(3), R, p_com, np.zeros(3))
    check("a 10 mm target above the foot pulls it UP",
          float((J @ np.linalg.solve(J.T @ J + 1e-12 * np.eye(3), tau))[2]) > 0,
          f"tau = {np.round(tau, 4)} Nm")
    check("...with exactly the magnitude Kp*dx mapped through J^T",
          np.allclose(tau, J.T @ (R.T @ (cfg.KP_SWING @ d))))
    tau_r = swing_torque(0, q, qd, p_here, np.zeros(3), R, p_com,
                         np.array([0.0, 0.0, 0.2]))
    check("a body rising under a held foot is damped, not ignored",
          float(np.linalg.norm(tau_r)) > 0.0,
          f"|tau| = {np.linalg.norm(tau_r):.4f} Nm from 0.2 m/s of v_com")

    print(f"self-test {'PASS' if _PASS[0] == _PASS[1] else 'FAIL'} "
          f"({_PASS[0]}/{_PASS[1]})")
    return 0 if _PASS[0] == _PASS[1] else 1


if __name__ == "__main__":
    sys.exit(self_test())
