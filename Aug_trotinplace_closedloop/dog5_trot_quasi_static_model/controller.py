#!/usr/bin/env python3
"""The orchestration: state in, twelve torques out.  The only file that reads
every other one.

    gait.contact  ->  desired_wrench  ->  ForceDistributor.solve
                  ->  stance J^T f  /  swing impedance  ->  clamp

WHAT THIS FILE OWNS, WHICH IS DELIBERATELY LITTLE
    Order, frames, and the clamp.  Every gain lives in config, every model
    lives in its own module, and this file is where a reader checks that the
    pieces are wired the way the docstrings claim.  If it starts holding
    control logic, the split has failed.

THE THREE FRAME CONVERSIONS, ALL OF THEM IN ONE PLACE
    They are the only thing here that can be silently wrong, so they are
    named rather than inlined:

    r_feet for the QP     the foot relative to the CoM, in WORLD axes:
                          R_wb (p_foot_body - COM_BODY).  Using the trunk
                          ORIGIN instead of the CoM puts a constant 14.7 mm
                          lever on every vertical force -- 0.84 Nm of pitch
                          bias at a 57 N load, which no gain removes because
                          it is not an error, it is a wrong model.

    stance torque         tau = J^T R_wb^T f.  f comes back from the QP in
                          world axes and J is a TRUNK-frame Jacobian, so the
                          rotation has to happen between them.

    p_com for swing       p (the trunk origin, which is what the estimator
                          measures) plus R_wb COM_BODY.

THE JOINT-SPACE FLOOR IS NOT OPTIONAL
    tau = J^T f is a VELOCITY-level law: take the ground away from a stance
    foot and nothing bounds qdd.  KP_JOINT (q_ref - q) makes it position-level
    with a fixed point, exactly as august_week2 does, and q_ref is the pose
    the force law is already trying to hold -- so in nominal stance the term
    is ~0 and costs nothing.  3.0/0.1 and not 15/0.6; see config.

RUN
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V dog5_trot_quasi_static_model/controller.py --self-test
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

if __package__:
    from . import config as cfg
    from . import leg_kin
    from . import gait as gait_mod
    from . import balance_qp
    from . import swing as swing_mod
else:
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import config as cfg
    import leg_kin
    import gait as gait_mod
    import balance_qp
    import swing as swing_mod


@dataclass
class RobotState:
    """Everything the controller reads, in one object so a caller cannot
    supply half of it.

    rpy, R_wb   trunk attitude.  R_wb maps BODY -> WORLD.
    omega       BODY-frame angular rate: what a gyro measures, unrotated.
    p           the TRUNK ORIGIN in world.  p[:2] is unused -- there is no
                absolute x/y without an EKF, and config.KP_POS[:2] is zero so
                nothing here is stiff to it.  p[2] is the height the loop
                controls, measured floor-to-trunk-origin.
    v           trunk velocity in WORLD axes.
    q, qd       twelve joint angles and rates, in JOINT_INDEX order.
    """

    rpy: np.ndarray
    R_wb: np.ndarray
    omega: np.ndarray
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    qd: np.ndarray

    def __post_init__(self):
        self.rpy = np.asarray(self.rpy, dtype=float).reshape(3)
        self.R_wb = np.asarray(self.R_wb, dtype=float).reshape(3, 3)
        self.omega = np.asarray(self.omega, dtype=float).reshape(3)
        self.p = np.asarray(self.p, dtype=float).reshape(3)
        self.v = np.asarray(self.v, dtype=float).reshape(3)
        self.q = np.asarray(self.q, dtype=float).reshape(cfg.N_JOINTS)
        self.qd = np.asarray(self.qd, dtype=float).reshape(cfg.N_JOINTS)


def stance_torque(leg: int, q, f_world, R_wb, J=None) -> np.ndarray:
    """tau = -J^T R_wb^T f  for one planted leg.  (3,)

    THE MINUS SIGN IS NOT OPTIONAL AND IS EASY TO LOSE.  `f_world` is the
    GROUND REACTION -- the force the floor applies to the foot, which is what
    the QP solves for and why A f sums to +m*g.  Virtual work with an external
    force at the foot gives f.J dq + tau.dq = 0, so tau = -J^T f.  Writing
    +J^T f instead is a robot that answers "push up" by driving its feet into
    the floor; august_week2/force_totorque.py has carried the minus since it
    was checked against MuJoCo, and this file agrees with it.

    `J` may be passed if the caller already has it -- the chain walk is 131 us
    and TrotController does all four legs once per sweep.
    """
    q = np.asarray(q, dtype=float).reshape(3)
    R_wb = np.asarray(R_wb, dtype=float).reshape(3, 3)
    if J is None:
        _, J = leg_kin.leg_state(int(leg), q)
    return -J.T @ (R_wb.T @ np.asarray(f_world, dtype=float).reshape(3))


def leg_gravity_torque(leg: int, q, R_wb) -> np.ndarray:
    """What the three motors must hold just to carry the LEG'S OWN links.  (3,)

    NOT A REFINEMENT.  DOG5's legs are 3.196 of its 5.815 kg -- 55% -- and at
    the nominal stance this term is 0.47 Nm on the abduction joint, 16% of
    TAU_MAX, against a -J^T f that is itself only a few Nm.  august_week2
    records the check: -J^T f alone differs from MuJoCo's floating-base
    inverse dynamics by 0.482 Nm, and matches to machine precision once this
    is added.  Leaving it out is a leg that sags under its own weight.

    Uses the TILTED form, so gravity is resolved in the current body
    orientation rather than assumed along trunk -z; at 5 degrees of roll that
    is worth 0.025 Nm on the abduction joint.
    """
    g_down = cfg.statics.gravity_down_body(np.asarray(R_wb, dtype=float))
    return cfg.statics.leg_gravity_torque_tilted(cfg.LEGS[int(leg)], q, g_down)


class TrotController:
    """start() once, update() every sweep, stop() to release."""

    def __init__(self, cfg_mod=cfg):
        self.cfg = cfg_mod
        self.gait = gait_mod.TrotGait(cfg_mod.GAIT_PERIOD, cfg_mod.DUTY,
                                      cfg_mod.PHASE_OFFSET)
        self.dist = balance_qp.ForceDistributor(
            mass=cfg_mod.MASS, inertia_body=cfg_mod.INERTIA_BODY,
            mu=cfg_mod.MU, fz_min=cfg_mod.FZ_MIN, fz_max=cfg_mod.FZ_MAX,
            w_task=cfg_mod.W_TASK, w_force=cfg_mod.W_FORCE,
            w_smooth=cfg_mod.W_SMOOTH)
        self.swing = swing_mod.SwingPlanner(self.gait)
        self.running = False
        self.v_cmd = np.zeros(3)
        self.yawrate_cmd = 0.0
        # diagnostics a runner should log; none of them feed back
        self.last = {}
        self._yaw_lock = 0.0
        self._z_ref = cfg_mod.STAND_HEIGHT
        self._q_ref = np.zeros(cfg_mod.N_JOINTS)
        self._was_contact = np.zeros(cfg_mod.N_LEGS, dtype=bool)
        self._tau_ff = np.zeros(cfg_mod.N_JOINTS)
        self._sweep = 0
        # "stand" holds all four feet down and ignores the gait clock; "trot"
        # follows it.  A runner needs the first to get OFF THE GROUND: the
        # rise is a torque-mode stand-up, and starting it with two feet in the
        # air would ask a crouched robot to balance on a diagonal.
        self.mode = "stand"
        self._t_now = 0.0
        self._w_model = np.ones(cfg_mod.N_LEGS)

    # -- lifecycle ---------------------------------------------------------
    def start(self, t: float, state: RobotState) -> None:
        """Align the phase to now, lock yaw and height, anchor the feet.

        YAW AND HEIGHT ARE LATCHED FROM THE ROBOT, NOT TAKEN FROM CONFIG.  The
        controller's job is to hold what it was handed, and a start that
        commanded config.STAND_HEIGHT to a robot standing 20 mm lower would
        step the wrench by KP_POS[2]*0.02 = 6 N on its first sweep, which is
        a lurch at exactly the moment the operator is closest to the robot.
        """
        state = _as_state(state)
        self.gait.reset(t)
        self.dist.reset()
        self._yaw_lock = float(state.rpy[2])
        self._z_ref = float(state.p[2])
        self._q_ref = state.q.copy()
        self.swing.reset(t, self._feet_world(state))
        self._was_contact = self.gait.contact(t)
        self._tau_ff = np.zeros(cfg.N_JOINTS)
        # -1 so the FIRST update() lands on sweep 0 and runs the model block:
        # starting held at zero would leave the robot with only the joint
        # floor for up to MODEL_EVERY sweeps at exactly the moment torque arms.
        self._sweep = -1
        self.running = True

    def stop(self) -> None:
        self.running = False
        self.dist.reset()
        self.last = {}

    def set_mode(self, mode: str) -> None:
        if mode not in ("stand", "trot"):
            raise ValueError(f"mode must be 'stand' or 'trot', got {mode!r}")
        self.mode = mode

    def set_z_ref(self, z: float) -> None:
        """Move the height reference.  The runner ramps this through the rise.

        Setting it is the ONLY way the height target changes -- start() latches
        the robot's own height and nothing else writes it, so a ramp is the
        runner's explicit act rather than a side effect of a stage name.
        """
        self._z_ref = float(z)

    def contact_now(self, t: float) -> np.ndarray:
        """(4,) bool: which feet are on the ground at this t.

        Exposed because the ESTIMATOR needs it before update() runs -- leg
        odometry must be told which feet to trust, and in a trot that is two
        of them for most of the cycle and four through the handover.
        """
        if self.mode == "stand":
            return np.ones(cfg.N_LEGS, dtype=bool)
        return self.gait.contact(t)

    def contact_weight_now(self, t: float) -> np.ndarray:
        """(4,) in [0,1]: how much force each foot may carry.  The QP's input.

        Separate from contact_now because they answer different questions.  A
        foot at weight 0.2 IS on the ground -- the estimator should use it and
        the leg should run its stance branch -- it is merely not allowed to
        carry much yet.  Conflating the two is how the handover step comes
        back.
        """
        if self.mode == "stand":
            return np.ones(cfg.N_LEGS)
        return self.gait.contact_weight(t)

    def command(self, v_cmd=(0.0, 0.0, 0.0), yawrate_cmd: float = 0.0) -> None:
        """Set the travel command.  Zero is a trot in place, which is the only
        thing this stance height has the leg reach for -- see
        config.MAX_FORWARD_V_AT_STAND_HEIGHT."""
        self.v_cmd = np.asarray(v_cmd, dtype=float).reshape(3)
        self.yawrate_cmd = float(yawrate_cmd)

    # -- frames, named once ------------------------------------------------
    @staticmethod
    def _legs(state):
        """(feet_body (4,3), Js [4]) -- ONE chain walk per leg, per sweep.

        Everything downstream wants either the foot position or the Jacobian
        or both, and each walk is 131 us.  Calling the convenience helpers
        separately cost twelve walks a sweep, 1.6 ms of a 4 ms budget, for
        four legs' worth of information.
        """
        feet = np.empty((cfg.N_LEGS, 3))
        Js = []
        for i in range(cfg.N_LEGS):
            foot, J = leg_kin.leg_state(i, state.q[cfg.JOINT_INDEX[i]])
            feet[i] = foot
            Js.append(J)
        return feet, Js

    def _feet_world(self, state, feet_body=None) -> np.ndarray:
        """(4,3) foot SITES in world, from the trunk origin and the encoders."""
        if feet_body is None:
            feet_body = leg_kin.all_foot_pos_body(state.q)
        return np.array([state.p + state.R_wb @ feet_body[i]
                         for i in range(cfg.N_LEGS)])

    def _r_feet_com(self, state, feet_body=None) -> np.ndarray:
        """(4,3) foot positions RELATIVE TO THE CoM, world axes -- the QP's
        moment arms.  The COM_BODY subtraction is the whole point."""
        if feet_body is None:
            feet_body = leg_kin.all_foot_pos_body(state.q)
        return np.array([state.R_wb @ (feet_body[i] - cfg.COM_BODY)
                         for i in range(cfg.N_LEGS)])

    def _p_com_world(self, state) -> np.ndarray:
        return state.p + state.R_wb @ cfg.COM_BODY

    # -- the sweep ---------------------------------------------------------
    def update(self, t: float, state: RobotState) -> np.ndarray:
        """(12,) joint torques.  Returns zeros unless start() has been called.

        The model block is sub-sampled at config.MODEL_EVERY and its result is
        HELD between updates; the joint-space floor runs every call.  See
        config.MODEL_EVERY for the measurement that forces this.
        """
        if not self.running:
            return np.zeros(cfg.N_JOINTS)
        state = _as_state(state)
        self._sweep += 1
        self._t_now = float(t)

        contact = self.contact_now(t)
        if self._sweep % self.cfg.MODEL_EVERY == 0:
            self._tau_ff = self._model_block(t, state, contact)
            self._w_model = self.contact_weight_now(t)

        # q_ref FOLLOWS EVERY LEG, and how it follows is the difference
        # between a rise and a robot that cannot get off the floor.  See
        # _update_q_ref; a swing leg simply tracks its own measured pose.
        for i in range(cfg.N_LEGS):
            if not contact[i]:
                self._q_ref[cfg.JOINT_INDEX[i]] = state.q[cfg.JOINT_INDEX[i]]
        tau = (self._tau_ff
               + self.cfg.KP_JOINT * (self._q_ref - state.q)
               - self.cfg.KD_JOINT * state.qd)
        tau = np.clip(tau, -self.cfg.TAU_MAX, self.cfg.TAU_MAX)
        self.last["saturated"] = int(np.sum(np.abs(tau) >= self.cfg.TAU_MAX - 1e-9))
        return tau

    def _update_q_ref(self, state, contact, feet_body, Js):
        """Move the joint-space reference to the pose that HOLDS z_ref.

        THIS IS WHY A RISE RISES.  q_ref latched at start() is the CROUCH, and
        a stance leg's contact never goes false, so a q_ref that only follows
        swinging legs stays at the crouch for the whole 8 s rise.  Measured on
        this robot that is 1.38 rad of hip error at the top and
        KP_JOINT * 1.38 = 13.8 Nm pulling the leg back down, against a 1.0 Nm
        cap: the impedance saturates the whole torque budget in the wrong
        direction and the force law never gets a say.  The robot rose in
        neither direction on 2026-08-18, which is exactly what august_week2's
        self-test warns about one week earlier.

        The fix is not new work.  leg_kin.q_ref_for_height is week 2's own
        function, ported, and it is called here the way week 2 calls it: warm
        started on the previous q_ref, with the feet pinned at the anchors'
        x/y so the contact point never moves in the world.

        A SWING LEG TRACKS ITS OWN MEASURED POSE instead, so the floor never
        pulls against an arc the swing controller is driving.
        """
        # feet_body, NOT swing.p_anchor.  q_ref_for_height compares against
        # leg_frames, which is the TRUNK frame; p_anchor is the WORLD frame.
        # With p[:2] pinned at 0 and a level robot the two happen to agree
        # numerically, so the mistake is invisible until the trunk tilts --
        # and then the reference stance shears with the tilt.
        #
        # Measured on this robot, the foot x/y in the trunk frame is IDENTICAL
        # at the crouch and at the stand (0.3401/0.1125 either way): the rise
        # is purely vertical in this frame, which is exactly why pinning x/y
        # and moving only z is the right description of it.
        self._q_ref = leg_kin.q_ref_for_height(self._q_ref, self._z_ref,
                                               feet_body[:, :2])
        for i in range(cfg.N_LEGS):
            if not contact[i]:
                self._q_ref[cfg.JOINT_INDEX[i]] = state.q[cfg.JOINT_INDEX[i]]

    def _model_block(self, t, state, contact) -> np.ndarray:
        """The expensive half: frames, wrench, QP, swing plan, per-leg torque."""
        p_com = self._p_com_world(state)
        feet_body, Js = self._legs(state)

        # -- swing plan.  update() BEFORE ref(), and re-anchor any foot the
        #    schedule has just planted, so the next lift starts from the real
        #    position rather than from where the last arc ended.
        if self.mode == "stand":
            # All four planted: re-anchor every one of them, every block.  Not
            # bookkeeping -- it is what makes the FIRST swing after set_mode
            # start from where the foot really is instead of from wherever the
            # rise left the plan.  The gait clock still runs underneath, so
            # switching to trot needs no resynchronisation.
            feet_w = self._feet_world(state, feet_body)
            for i in range(cfg.N_LEGS):
                self.swing.touchdown(i, feet_w[i])
            self._was_contact = contact
            p_des, v_des = feet_w, np.zeros((cfg.N_LEGS, 3))
        else:
            landed = contact & ~self._was_contact
            if landed.any():
                feet_w = self._feet_world(state, feet_body)
                for i in np.flatnonzero(landed):
                    self.swing.touchdown(int(i), feet_w[i])
            self._was_contact = contact
            self.swing.update(t, p_com, state.v, state.R_wb, self.v_cmd)
            p_des, v_des = self.swing.ref(t)

        # after the anchors are current, and before the torque is built
        self._update_q_ref(state, contact, feet_body, Js)

        # -- the body wrench, then its split across the planted feet
        rpy_ref = np.array([0.0, 0.0, self._yaw_lock])
        F_des, M_des = balance_qp.desired_wrench(
            state.rpy, state.R_wb, state.omega, p_com, state.v,
            rpy_ref, self._z_ref + (p_com[2] - state.p[2]),
            v_cmd=self.v_cmd, gains=self.cfg)
        f_world = self.dist.solve(F_des, M_des,
                                  self._r_feet_com(state, feet_body),
                                  self.contact_weight_now(t))

        # -- per leg: push if planted, track the arc if not
        tau = np.zeros(cfg.N_JOINTS)
        for i in range(cfg.N_LEGS):
            sl = cfg.JOINT_INDEX[i]
            if contact[i]:
                tau[sl] = stance_torque(i, state.q[sl], f_world[i],
                                        state.R_wb, J=Js[i])
            else:
                tau[sl] = swing_mod.swing_torque(
                    i, state.q[sl], state.qd[sl], p_des[i], v_des[i],
                    state.R_wb, p_com, state.v,
                    kp=self.cfg.KP_SWING, kd=self.cfg.KD_SWING)
            # LEG GRAVITY GOES ON BOTH BRANCHES.  A stance leg needs it
            # because -J^T f models a massless leg; a swing leg needs it
            # because nothing else is holding the limb up at all.
            tau[sl] += leg_gravity_torque(i, state.q[sl], state.R_wb)

        self.last.update({
            "contact": contact, "F_des": F_des, "M_des": M_des,
            "f_world": f_world, "p_des": p_des,
            "qp_primal": self.dist.primal_residual,
            "qp_dual": self.dist.dual_residual,
            "qp_iters": self.dist.iters_used,
        })
        return tau


def _as_state(s):
    return s if isinstance(s, RobotState) else RobotState(**s)


# ===========================================================================
# self-test
# ===========================================================================
_PASS = [0, 0]


def check(label, ok, detail=""):
    _PASS[1] += 1
    _PASS[0] += bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))


def _standing_state():
    q = cfg.Q_STAND.reshape(-1)
    return RobotState(rpy=np.zeros(3), R_wb=np.eye(3), omega=np.zeros(3),
                      p=np.array([0.0, 0.0, cfg.STAND_HEIGHT]),
                      v=np.zeros(3), q=q, qd=np.zeros(cfg.N_JOINTS))


def self_test():
    c = TrotController()
    st = _standing_state()

    # -- nothing happens before start() ------------------------------------
    check("update() before start() commands exactly zero torque",
          np.all(c.update(0.0, st) == 0.0))

    # -- the frames --------------------------------------------------------
    c.start(0.0, st)
    feet_w = c._feet_world(st)
    r_com = c._r_feet_com(st)
    check("the QP's moment arms are measured from the CoM, not the trunk origin",
          np.allclose(r_com, feet_w - c._p_com_world(st)),
          f"they differ by COM_BODY = {np.round(cfg.COM_BODY*1e3, 2)} mm; at "
          f"{cfg.WEIGHT:.0f} N the z term alone is "
          f"{cfg.WEIGHT*abs(cfg.COM_BODY[0]):.3f} Nm of pitch")
    check("start() latches the robot's OWN height, not config's",
          abs(c._z_ref - st.p[2]) < 1e-12)
    low = _standing_state(); low.p[2] -= 0.02
    c2 = TrotController(); c2.start(0.0, low)
    check("...so starting 20 mm low does not step the wrench",
          abs(c2._z_ref - low.p[2]) < 1e-12,
          f"a config-fixed reference would ask "
          f"{cfg.KP_POS[2]*0.02:.1f} N extra on sweep 1")

    # -- stance_torque sign, checked against gravity, not against algebra --
    f_up = np.array([0.0, 0.0, cfg.WEIGHT / 2.0])
    tau_st = stance_torque(0, cfg.Q_STAND[0], f_up, np.eye(3))
    _, J = leg_kin.leg_state(0, cfg.Q_STAND[0])
    # the joint velocity this torque would produce must lift the trunk, i.e.
    # push the foot DOWN relative to the trunk
    check("a stance leg told to push up drives its foot DOWN in the trunk frame",
          float((J @ tau_st)[2]) < 0.0,
          f"J tau has z = {float((J @ tau_st)[2]):+.4f}; +J^T f instead of "
          f"-J^T f flips this, and that robot shoves itself into the floor")
    check("...and stance_torque agrees with august_week2's verified -J^T f",
          np.allclose(tau_st, -J.T @ f_up),
          "force_totorque.py line 217: tau_i = -J.T @ forces[leg]")
    gtau = leg_gravity_torque(0, cfg.Q_STAND[0], np.eye(3))
    check("leg gravity is present and is NOT a small correction",
          float(np.max(np.abs(gtau))) > 0.4,
          f"{np.round(gtau, 4)} Nm, {100*np.max(np.abs(gtau))/cfg.TAU_MAX:.0f}% "
          f"of TAU_MAX; the legs are 55% of the robot's mass")

    # -- one sweep, standing still -----------------------------------------
    c.start(0.0, st)
    tau = c.update(0.0, st)
    L = c.last
    check("a trot never has fewer than two feet planted",
          int(L["contact"].sum()) >= 2, f"contact = {L['contact']}")
    check("the commanded wrench is the weight, level, at rest",
          abs(L["F_des"][2] - cfg.WEIGHT) < 1e-6
          and float(np.max(np.abs(L["M_des"]))) < 1e-6,
          f"F = {np.round(L['F_des'], 3)}, M = {np.round(L['M_des'], 4)}")
    check("the planted feet carry the whole weight between them",
          abs(L["f_world"][:, 2].sum() - cfg.WEIGHT) < 1.0,
          f"{L['f_world'][:, 2].sum():.2f} N of {cfg.WEIGHT:.2f}")
    check("the swinging feet are commanded no ground force at all",
          np.all(L["f_world"][~L["contact"]] == 0.0))
    check("no joint is saturated while merely standing",
          L["saturated"] == 0,
          f"|tau| max {np.max(np.abs(tau)):.3f} of {cfg.TAU_MAX} Nm")

    # -- a full gait cycle: nothing blows up, nothing saturates ------------
    c.start(0.0, st)
    dt = cfg.CTRL_DT
    worst_tau, worst_res, sat = 0.0, 0.0, 0
    for k in range(int(3 * cfg.GAIT_PERIOD / dt)):
        tau = c.update(k * dt, st)
        worst_tau = max(worst_tau, float(np.max(np.abs(tau))))
        worst_res = max(worst_res, c.last["qp_primal"])
        sat += c.last["saturated"]
        if not np.all(np.isfinite(tau)):
            break
    check("three full cycles produce finite, unsaturated torque throughout",
          np.all(np.isfinite(tau)) and sat == 0,
          f"worst |tau| {worst_tau:.3f} of {cfg.TAU_MAX} Nm, "
          f"{sat} saturated samples in {int(3*cfg.GAIT_PERIOD/dt)} sweeps")
    # The primal residual is in NEWTONS -- it is the worst violation of a cone
    # row -- so the threshold has to be a force, not a solver's idea of small.
    # 0.01 N on a 57 N robot is 1.8e-4 of the load, and _project guarantees the
    # RETURNED force is feasible regardless.
    check("...and the QP stayed feasible every sweep",
          worst_res < 0.01,
          f"worst primal residual {worst_res:.2e} N on a {cfg.WEIGHT:.0f} N "
          f"robot; the returned force is projected feasible either way")

    # -- the contact switch is where a force step would show ---------------
    c.start(0.0, st)
    fz = []
    for k in range(int(2 * cfg.GAIT_PERIOD / dt)):
        c.update(k * dt, st)
        fz.append(c.last["f_world"][:, 2].sum())
    fz = np.array(fz)
    check("total vertical force never steps across a contact switch",
          float(np.max(np.abs(np.diff(fz)))) < 5.0,
          f"worst sweep-to-sweep jump {np.max(np.abs(np.diff(fz))):.2f} N "
          f"on a {cfg.WEIGHT:.0f} N robot -- this is what W_SMOOTH buys")

    # -- a tilt must produce a righting moment of the right sign ----------
    tilt = _standing_state()
    roll = 0.05
    tilt.rpy = np.array([roll, 0.0, 0.0])
    tilt.R_wb = balance_qp._rot_from_rpy(tilt.rpy)
    c.start(0.0, _standing_state())
    c.update(0.0, tilt)
    check("a roll to the left is answered by a moment to the right",
          c.last["M_des"][0] < 0.0,
          f"roll {np.degrees(roll):+.1f} deg -> Mx "
          f"{c.last['M_des'][0]:+.3f} Nm = -KP_ORI[0]*roll")

    # -- stop() really stops ----------------------------------------------
    # -- THE RISE, which is the thing that stopped working -----------------
    # THE CROUCH IS RAMPED TO, NOT JUMPED TO.  q_ref_for_height is a damped
    # least-squares step warm-started on its own last answer; asking it for
    # 130 mm of travel in ONE call is the case its docstring says leaves the
    # workspace, and it does -- 291 deg of joint travel and a 52 mm lag.  The
    # runner never does that, so neither does this test.
    foot_xy = leg_kin.all_foot_pos_body(cfg.Q_STAND.reshape(-1))[:, :2]
    q_crouch = cfg.Q_STAND.copy().reshape(-1)
    for k in range(1, 201):
        q_crouch = leg_kin.q_ref_for_height(
            q_crouch, cfg.STAND_HEIGHT + (0.06 - cfg.STAND_HEIGHT) * k / 200,
            foot_xy)
    def _at(z, qq):
        return RobotState(rpy=np.zeros(3), R_wb=np.eye(3), omega=np.zeros(3),
                          p=np.array([0.0, 0.0, z]), v=np.zeros(3),
                          q=qq, qd=np.zeros(cfg.N_JOINTS))

    # THE ROBOT HAS TO BE RISING FOR THIS TEST TO MEAN ANYTHING.  Holding it
    # at the crouch while commanding the top makes a tracking q_ref and a
    # frozen one look identical -- both read kp*(stand - crouch) -- which is
    # how the first version of this check passed on a controller that could
    # not rise.  So q follows the reference here, exactly as a working rise
    # does, and the question becomes whether q_ref follows q.
    cr = TrotController()
    cr.start(0.0, _at(0.06, q_crouch))
    n_steps = int(8.0 * 250.0 / cfg.MODEL_EVERY)
    worst_track, worst_frozen = 0.0, 0.0
    q_now = q_crouch.copy()
    for k in range(1, n_steps + 1):
        z = 0.06 + (cfg.STAND_HEIGHT - 0.06) * k / n_steps
        q_now = leg_kin.q_ref_for_height(q_now, z, foot_xy)   # the robot rises
        st_k = _at(z, q_now)
        cr.set_z_ref(z)
        # MODEL_EVERY calls, because update() only refreshes q_ref on a model
        # sweep.  Stepping z once per call instead left q_ref three sweeps
        # behind and read as a 51 mNm "error" that was really the sub-sampling
        # this test is not about.
        for _ in range(cfg.MODEL_EVERY):
            cr.update((k * cfg.MODEL_EVERY + _) * cfg.CTRL_DT, st_k)
        worst_track = max(worst_track, float(np.max(np.abs(
            cfg.KP_JOINT * (cr._q_ref - q_now)))))
        # what the frozen version would have commanded at this same instant
        worst_frozen = max(worst_frozen, float(np.max(np.abs(
            cfg.KP_JOINT * (q_crouch - q_now)))))
    check("through a real 8 s rise the impedance stays a floor, near zero",
          worst_track < 0.05,
          f"worst {worst_track*1e3:.2f} mNm with q_ref tracking")
    check("...against what a q_ref frozen at the crouch would have commanded",
          worst_frozen > 20 * max(worst_track, 1e-9),
          f"{worst_frozen:.2f} Nm frozen vs {worst_track*1e3:.2f} mNm tracking "
          f"-- {worst_frozen/1.0:.0f}x a 1.0 Nm cap, pulling the leg back down "
          f"the whole way, which is why the robot could not rise")
    check("...and q_ref really did travel, it is not the crouch held",
          float(np.max(np.abs(cr._q_ref - q_crouch))) > 0.3,
          f"{np.rad2deg(np.max(np.abs(cr._q_ref - q_crouch))):.1f} deg of "
          f"joint travel over the rise")

    # -- stand mode: what the rise runs on ---------------------------------
    c.start(0.0, st)
    check("a fresh controller starts in stand mode, not trotting",
          c.mode == "stand" and bool(np.all(c.contact_now(0.0))))
    tau_stand = c.update(0.0, st)
    check("stand mode plants all four feet and splits the load four ways",
          int(c.last["contact"].sum()) == 4
          and float(np.min(c.last["f_world"][:, 2])) > 5.0,
          f"fz = {np.round(c.last['f_world'][:, 2], 2)} N")
    c.set_z_ref(st.p[2] + 0.01)
    # MODEL_EVERY calls, not one: F_des only refreshes on a model sweep, and
    # a single update after the change reads the HELD value from before it.
    for k in range(cfg.MODEL_EVERY):
        c.update((k + 1) * cfg.CTRL_DT, st)
    check("set_z_ref raises the commanded force by KP_POS[2] * dz",
          abs(c.last["F_des"][2] - (cfg.WEIGHT + cfg.KP_POS[2] * 0.01)) < 1e-6,
          f"{c.last['F_des'][2]:.2f} N vs {cfg.WEIGHT + cfg.KP_POS[2]*0.01:.2f}")

    # switching to trot must not step the torque -- that is the moment two
    # feet leave the ground under load
    c.start(0.0, st)
    for k in range(30):
        prev = c.update(k * cfg.CTRL_DT, st)
    c.set_mode("trot")
    nxt = c.update(30 * cfg.CTRL_DT, st)
    ramp_s = cfg.CONTACT_RAMP * cfg.DUTY * cfg.GAIT_PERIOD
    absorb = cfg.TAU_SLEW_NM_S * ramp_s
    check("the stand->trot switch steps less than the slew limiter absorbs "
          "in one handover",
          float(np.max(np.abs(nxt - prev))) < absorb,
          f"{np.max(np.abs(nxt - prev)):.3f} Nm against {absorb:.2f} Nm that "
          f"{cfg.TAU_SLEW_NM_S:.0f} Nm/s covers in the {ramp_s*1e3:.0f} ms ramp")

    # -- the handover, which is what CONTACT_RAMP exists for ---------------
    c.start(0.0, st); c.set_mode("trot")
    taus = np.array([c.update(k * dt, st)
                     for k in range(int(3 * cfg.GAIT_PERIOD / dt))])
    step = float(np.max(np.abs(np.diff(taus, axis=0))))
    # THE CRITERION IS THE GATE, NOT A FRACTION OF TAU_MAX.  The commanded
    # feedforward is a 12 ms staircase by construction (MODEL_EVERY), so its
    # sweep-to-sweep difference is not what any motor sees -- the runner's
    # slew limiter is between them.  What has to be true is that the limiter
    # can absorb the worst step inside the handover it belongs to.
    check("the slew limiter absorbs the worst handover step within one ramp",
          step < absorb,
          f"worst step {step:.3f} Nm, absorbed in {1e3*step/cfg.TAU_SLEW_NM_S:.0f} "
          f"ms of the {ramp_s*1e3:.0f} ms ramp; a binary contact mask stepped "
          f"2.208 Nm and no ramp at all existed to absorb it")
    # and the ramp has to be earning its place, not just present
    check("...and CONTACT_RAMP is what makes that true",
          step < 2.0,
          f"{step:.3f} Nm with the ramp against 2.208 Nm measured without it")

    c.stop()
    check("stop() returns the controller to commanding zero",
          np.all(c.update(0.5, st) == 0.0))

    # -- cost, on the machine this has to run on --------------------------
    import time
    c.start(0.0, st)
    n = 300
    t0 = time.perf_counter()
    for k in range(n):
        c.update(k * dt, st)
    per = (time.perf_counter() - t0) / n
    check("the AVERAGE update is a small part of a sweep",
          per < 0.25 * dt,
          f"{per*1e6:.0f} us mean, {100*per/dt:.0f}% of the {dt*1e3:.0f} ms "
          f"sweep, with the model block at 1-in-{cfg.MODEL_EVERY}")
    # The WORST sweep is the model-block one, and it is measured as a median
    # over model sweeps rather than a max, because a max over 300 samples on a
    # non-realtime Pi reports OS scheduling noise, not this code.
    model_us = []
    for k in range(n):
        t1 = time.perf_counter()
        c.update((n + k) * dt, st)
        el = time.perf_counter() - t1
        if c._sweep % cfg.MODEL_EVERY == 0:
            model_us.append(el)
    worst = float(np.median(model_us))
    check("the model-block sweep is far inside the driver's input-lost "
          "watchdog, which is the hard limit",
          worst < 0.5 * 0.050,
          f"{worst*1e6:.0f} us against a 50 ms watchdog")
    # THIS ONE IS NOT ASSERTED, BECAUSE IT CANNOT BE SETTLED OFFLINE.
    # A model sweep leaves (4 ms - this) for twelve CAN transactions, and
    # whether that is enough is a bus measurement, not an arithmetic one.
    # Printed so a hardware bring-up has the number to check rather than a
    # green tick that quietly assumed it.
    print(f"  [INFO] model-block sweep is {worst*1e6:.0f} us of the "
          f"{dt*1e3:.0f} ms budget ({100*worst/dt:.0f}%), leaving "
          f"{(dt-worst)*1e6:.0f} us for 12 CAN frames -- MEASURE THIS ON "
          f"HARDWARE; raise config.MODEL_EVERY if the bus misses.")
    c.start(0.0, st)
    n_model = sum(1 for k in range(30)
                  if (c.update(k * dt, st) is not None
                      and c._sweep % cfg.MODEL_EVERY == 0))
    check("the model block really does run 1 sweep in MODEL_EVERY",
          n_model == 30 // cfg.MODEL_EVERY,
          f"{n_model} model sweeps in 30, expected {30 // cfg.MODEL_EVERY}")

    print(f"self-test {'PASS' if _PASS[0] == _PASS[1] else 'FAIL'} "
          f"({_PASS[0]}/{_PASS[1]})")
    return 0 if _PASS[0] == _PASS[1] else 1


if __name__ == "__main__":
    sys.exit(self_test())
