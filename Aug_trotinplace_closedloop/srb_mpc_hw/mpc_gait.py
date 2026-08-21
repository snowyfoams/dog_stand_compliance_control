#!/usr/bin/env python3
"""The contact layer: the clock, what the motors say about it, and the filter
that keeps foot placement out of resonance with itself.

    TrotGait            NOT redefined here -- dog5_trot.gait, imported
    MeasuredFootContact per-foot support force from measured iq -> bool(4)
    ContactAwareGait    the clock, with a LATE measured touchdown promoted
    PlacementVelocity   a time-constant low pass for the Raibert target

WHY THE CLOCK IS NOT REDEFINED
    dog5_trot/gait.py is already the contact schedule for this robot, in this
    leg order, with the smoothstep contact_weight that removes the 2.2 Nm
    handover step -- and it has a self-test that asserts the diagonals by NAME
    rather than by four numbers.  A second trot clock in this package would be
    a second thing to keep in phase with it.  So this module wraps it and adds
    only what the MPC needs on top.

WHY PROMOTION EXISTS AT ALL, WHICH IS A SIMULATION RESULT
    The simulation's contact_gait.py records what a purely clock-driven trot
    does over time: as the body oscillates, swing feet touch down EARLY, the
    swing controller keeps force-tracking its arc into the floor, and the MPC
    treats the foot as force-free.  Each early contact is an unmodelled impulse
    at gait frequency, the rocking pumps itself, and the robot skates away
    leaning -- 31 deg within 12 s in place.  Promoting a late-swing measured
    contact to stance breaks the loop.

    That module's own sim2real note says exactly what to replace and with what:
    "a class exposing the same measure() -> bool(4) contract from foot force
    sensors, sole switches, or joint-current spikes".  This robot has no foot
    sensors and no switches.  It has the third one.

THE MEASUREMENT IS THE MOTOR CURRENT, AND IT IS NEARLY FREE HERE
    force_totorque.foot_load_from_torque already inverts measured iq through
    J^-T with each leg's own weight removed -- it is THE number the exit report
    checks against 57 N.  Called as it stands it re-walks each chain,
    recomputes leg gravity and takes an SVD per leg -- 1029 us for four legs on
    the Pi (params.LOAD_EVERY records that measurement), which is why the load
    watch only runs at 21 Hz.

    Every one of those is already computed in the caller's model block, for the
    stance torque.  Handed the frames, the Jacobian and the gravity torque it
    already has, the same measurement is one 3x3 solve per leg.  Measured on
    the development box, all four legs:

        force_totorque.foot_load_from_torque      406 us
        MeasuredFootContact.measure                24 us

    -- and it is the SAME NUMBER, not an approximation of it: across 30 random
    poses and trunk tilts the two agree to 0.0 N, because the only thing
    removed is the recomputation.  17x cheaper means it can run at the model
    rate instead of the 21 Hz the load watch runs at, and that is the whole
    point: 48 ms is half a swing, 12 ms is not.

    THE SVD GUARD IS DROPPED HERE AND KEPT THERE, DELIBERATELY.  In
    force_totorque the inverse feeds the load watchdog that LIMPS the robot, so
    a near-singular leg must be excluded rather than believed.  Here it feeds a
    boolean whose only effect is whether a foot is promoted early, and the
    failure is handled by the answer itself: a singular solve raises or returns
    a wild number, both of which come back as "no contact", i.e. no promotion,
    i.e. the unmodified clock.  Fail-safe by construction, at a 17th of the
    cost.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUG = os.path.dirname(_HERE)
if _AUG not in sys.path:
    sys.path.insert(0, _AUG)

from . import mpc_config as C
from dog5_trot_quasi_static_model import gait as gait_mod  # noqa: E402

TrotGait = gait_mod.TrotGait


# ===========================================================================
# what the motors say about the floor
# ===========================================================================
class MeasuredFootContact:
    """Per-foot support force from measured joint torque, and a latched bool.

    `measure` takes what the caller's model block has already produced, so it
    adds four 3x3 solves and nothing else:

        J        (4,) list of 3x3 trunk-frame foot Jacobians
        tau_meas (12,) measured joint torque, actuator order
        tau_grav (12,) each leg's own link-weight torque, tilted -- the term
                 that must come OFF before what is left can be called a GRF
        C_ib     the estimator's I->B rotation, to read the WORLD vertical

    Returns (contact bool(4), fz (4,) N).  Hysteresis on the force, not on
    time: CONTACT_FZ_ON to latch on, CONTACT_FZ_OFF to let go.  Every change
    of the contact set forces the MPC to replan, so a boundary that chatters
    is a solver that never warm starts.

    =======================================================================
    ON A SWING LEG THIS READS BACK THE SWING CONTROLLER'S OWN COMMAND
    =======================================================================
    The runner sends a swinging leg  tau = J^T f_swing + tau_grav.  Take
    tau_grav off and invert through J^-T and you get -f_swing exactly: the two
    operations are inverses, and there is no ground anywhere near the foot.
    With the actuator gain params records (tau_meas/tau_cmd = 0.76), measured:

        swing command       detector reads      contact at 8 N?
        -5 N  (downward)        +3.80 N              no
        -8 N                    +6.08 N              no
        -12 N                   +9.12 N              YES, foot in the air

    AND THE SIGN IS THE WRONG WAY ROUND FOR US.  Late in the swing the foot
    must be driven DOWN -- leg gravity is already fed forward, so the impedance
    does that work -- which makes the phantom force POSITIVE, i.e. it looks
    like ground reaction, in exactly the window PROMOTE_AFTER opens.

    A 12 N downward command is a ~45 mm tracking error at KP_SWING = 200 N/m,
    and this arc produces errors of that order: following it exactly would need
    482 rad/s^2 at the joint, which against the 10:1 reflected rotor is 4-7 Nm
    -- past the 1-3 Nm the gate allows.  The leg CANNOT track the reference, so
    the error is not an anomaly, it is the operating point.

    SO tau_cmd IS SUBTRACTED, and it is the previous model tick's swing torque
    because that is what tau_meas is reporting: params.LOOP_DELAY_S measures
    +3 sweeps of transport, and 3 sweeps is exactly one MODEL_EVERY.  What that
    leaves:

        swing command   raw      minus cmd    minus cmd, at the 0.76 gain
        -5 N            +5.00       0.00              -1.20
        -8 N            +8.00       0.00              -1.92
        -12 N          +12.00       0.00              -2.88
        -20 N          +20.00       0.00              -4.80

    and a real 14.26 N ground reaction under a stance foot still reads 14.26 N
    and latches, because a stance leg's tau_cmd is zero in this vector.

    THE RESIDUAL'S SIGN IS THE SAFE ONE, AND THAT IS WORTH SAYING.  The
    actuator delivers LESS than it is asked for, so subtracting the whole
    command over-subtracts and the leftover is NEGATIVE.  The detector
    therefore under-reports contact rather than over-reporting it -- it can
    miss a touchdown carrying under ~5 N, and it will not invent one.  For a
    layer whose dangerous failure is the false positive, biasing late is the
    right way to be wrong.  (It is also the direction a state estimator wants,
    and the opposite of what promotion would prefer.)

    WHAT IS STILL NOT IN IT: the leg's own inertia.  The inversion is
    quasi-static, and the reflected rotor -- 0.0085 kgm^2 of the 0.0088-0.0147
    params records per joint -- dominates M(q) on this robot, so the missing
    term is about  J_diag * qdd  with qdd differenced from the encoder.  That
    is cheap and it is the obvious next term; CONTROL_ROADMAP Phase 4 already
    names swing-leg inertia compensation as work.  It is not written here
    because nobody has yet measured how much of the residual it accounts for
    with this robot's legs actually in the air.

    AND THE SCALE IS ONLY AS GOOD AS iq -> tau.  Every number above is a
    newton derived from a current.  CONTROL_ROADMAP calls torque fidelity "the
    big unknown of Phase 2" and budgets a per-joint calibration that week 2
    then dropped; a threshold in newtons inherits all of that.  The exit
    report's foot-load sum against 57 N is the only end-to-end check there is.

    WHICH IS WHY PROMOTION IS OFF BY DEFAULT.  See ContactAwareGait.
    """

    def __init__(self, fz_on=C.CONTACT_FZ_ON, fz_off=C.CONTACT_FZ_OFF):
        if not fz_off < fz_on:
            raise ValueError(f"fz_off ({fz_off}) must be below fz_on ({fz_on}) "
                             f"or the hysteresis is inverted and the contact "
                             f"flag chatters at exactly the threshold")
        self.fz_on = float(fz_on)
        self.fz_off = float(fz_off)
        self.state = np.zeros(4, dtype=bool)
        self.fz = np.zeros(4)

    def measure(self, J, tau_meas, tau_grav, C_ib, tau_cmd=None):
        """See the class docstring.  `tau_cmd` is the SWING command to
        subtract, zero on stance legs; without it this reads back the swing
        controller's own force."""
        tau_meas = np.asarray(tau_meas, dtype=float).reshape(C.N_JOINTS)
        tau_grav = np.asarray(tau_grav, dtype=float).reshape(C.N_JOINTS)
        C_ib = np.asarray(C_ib, dtype=float).reshape(3, 3)
        tau_cmd = (np.zeros(C.N_JOINTS) if tau_cmd is None
                   else np.asarray(tau_cmd, dtype=float).reshape(C.N_JOINTS))
        for i in range(C.N_LEGS):
            sl = C.JOINT_INDEX[i]
            # The foot pushes DOWN on the ground with -f, so tau = -J^T f and
            # the GRF is the solution of J^T f = -(tau_meas - tau_grav).  Same
            # sign convention as dog5_statics.stance_torque, which is where
            # the commanded direction comes from.  tau_cmd comes off first --
            # see the class docstring for the 6 N of phantom contact it is.
            try:
                f_body = np.linalg.solve(
                    J[i].T, -(tau_meas[sl] - tau_grav[sl] - tau_cmd[sl]))
            except np.linalg.LinAlgError:
                self.fz[i] = np.nan
                self.state[i] = False           # fail-safe: no promotion
                continue
            fz = float((C_ib.T @ f_body)[2])    # WORLD vertical, not body z
            if not np.isfinite(fz) or abs(fz) > 10.0 * C.WEIGHT:
                self.fz[i] = np.nan
                self.state[i] = False
                continue
            self.fz[i] = fz
            if self.state[i]:
                self.state[i] = fz > self.fz_off
            else:
                self.state[i] = fz > self.fz_on
        return self.state.copy(), self.fz.copy()

    def reset(self):
        self.state[:] = False
        self.fz[:] = 0.0


# ===========================================================================
# the clock, reconciled with the measurement
# ===========================================================================
class ContactAwareGait:
    """dog5_trot's clock, with a late measured touchdown promoted to stance.

    THE CLOCK IS NOT MODIFIED, AND THAT IS THE WHOLE DESIGN.  `gait` still
    answers "which feet are SUPPOSED to be down", unchanged and untouched by
    any measurement; this class answers "and which one is down anyway".
    Mixing the two inside the schedule would make the plan depend on the thing
    the plan is supposed to drive.

    PROMOTION IS OFF BY DEFAULT ON THIS ROBOT, AND THAT IS NOT TIMIDITY.
    MeasuredFootContact's own docstring shows the detector reading +9 N of
    phantom ground reaction off a foot in the air, in exactly this window.
    Subtracting the swing command removes the dominant term but leaves the
    current loop's ~24% tracking error, and nobody has yet measured what that
    is worth with this robot's legs actually swinging.  A FALSE promotion is
    not a missed optimisation: the gait plants a foot that is airborne, the MPC
    allocates it force, and that share of the weight is pushed into nothing
    while the other legs carry less.

    THE MEASUREMENT THAT TURNS IT ON, and it needs no new code:
        run to HOLD, lift one foot clear of the floor BY HAND, and watch the
        detector's fz for that leg in the --log npz across a full swing arc.
        If it stays under CONTACT_FZ_OFF for the whole arc, the detector is
        clean and --promote is earned.  If it does not, the threshold or the
        swing-inertia term is what needs work -- not the gait.
    Without it, this stack trots on the clock alone, which is what
    dog5_trot/trot_hw already does.

    A leg is promoted when all three hold:
        the schedule says swing
        the swing is LATE           (phase > promote_after)
        the foot measures contact   (and has been airborne this swing)

    The airborne test is the simulation's, and it is load-bearing: without it
    a slow liftoff latches straight back to stance and the foot never leaves.

    A PROMOTED LEG RAMPS ITS LOAD IN, it does not take it in one step.  The
    simulation flips a boolean, which is fine when contact is exact and the
    solver runs every 2 ms.  Here the promotion is a measurement with a
    threshold, arriving up to one model tick late, and the force it admits
    lands on a slew-limited actuator -- so the promoted leg's weight follows
    the same smoothstep the schedule's own ramp uses, from the instant it was
    promoted.  The weight taken is the MAX of the two, so when the clock
    catches up and starts its own ramp at zero, the leg does not dip.
    """

    def __init__(self, gait, promote_after=C.PROMOTE_AFTER,
                 ramp=None, enabled=C.PROMOTE_ENABLED):
        self.gait = gait
        self.enabled = bool(enabled)
        self.promote_after = float(promote_after)
        # the ramp duration in SECONDS, taken from the schedule's own fraction
        # of stance so the two cannot drift apart
        frac = gait_mod.cfg.CONTACT_RAMP if ramp is None else float(ramp)
        self.ramp_s = max(frac * gait.stance_duration, 1e-3)
        self.early = np.zeros(4, dtype=bool)
        self.t_promote = np.zeros(4)
        self._lifted = np.zeros(4, dtype=bool)

    # -- lifecycle ---------------------------------------------------------
    def reset(self, t):
        self.gait.reset(t)
        self.early[:] = False
        self._lifted[:] = False
        self.t_promote[:] = 0.0

    # -- per model tick ----------------------------------------------------
    def update(self, t, measured):
        """Reconcile clock with measurement.  -> (contact (4,) bool, changed).

        `changed` flags a latch transition, so the caller can force a replan:
        the contact set is a CONSTRAINT of the QP, and a plan built for the
        old set is wrong about which foot may push.
        """
        measured = np.asarray(measured, dtype=bool).reshape(4)
        if not self.enabled:
            # OFF is the same schedule dog5_trot/trot_hw already trots on: the
            # clock, unmodified.  The measurement is still taken and still
            # logged; it is simply not acted on.
            return self.gait.contact(t), False
        sched = self.gait.contact(t)
        sp = self.gait.swing_phase(t)
        prev = self.early.copy()
        # a foot only counts as having TOUCHED DOWN if it has been airborne in
        # this swing; latched off again the moment the schedule plants it
        self._lifted = (~sched) & (self._lifted | ~measured)
        newly = (~sched) & (~self.early) & (sp > self.promote_after) \
            & measured & self._lifted
        if newly.any():
            self.t_promote[newly] = float(t)
        self.early = (~sched) & (self.early | newly)
        return sched | self.early, bool(np.any(self.early != prev))

    # -- what the MPC and the torque map read ------------------------------
    def contact(self, t):
        return self.gait.contact(t) | self.early

    def contact_weight(self, t):
        """(4,) in [0,1]: how much force each foot may carry, now."""
        w = self.gait.contact_weight(t)
        return np.maximum(w, self._promote_weight(t))

    def contact_weight_schedule(self, t, n, dt):
        """(n,4) the horizon's weights, promotion carried forward.

        KNOT k IS EVALUATED AT t + k*dt, NOT t + (k+1)*dt.  Force u_k acts
        across the interval [t_k, t_k+1] and produces the state x_{k+1}, so the
        contact that constrains u_0 is the contact at the START of the window
        the force is applied over -- which is now, or rather now plus the lead
        the caller has already folded into `t`.  The STATE reference is the
        other way round and is built that way in mpc_controller.mpc_reference;
        getting the two the same way round is a half-knot phase error in the
        handover, on a schedule whose whole job is the handover.

        The promotion is held across the WHOLE horizon rather than folded into
        knot 0 alone.  A promoted foot is on the ground and its own schedule is
        about to plant it anyway, so a plan that unloads it again at knot 1 is
        planning around a lift that will not happen.
        """
        out = np.empty((int(n), 4))
        for k in range(int(n)):
            out[k] = self.contact_weight(t + k * dt)
        return out

    def _promote_weight(self, t):
        w = np.zeros(4)
        if not self.early.any():
            return w
        u = np.clip((float(t) - self.t_promote) / self.ramp_s, 0.0, 1.0)
        w[self.early] = (u * u * (3.0 - 2.0 * u))[self.early]
        return w

    def swing_phase(self, t):
        """(4,) swing progress; forced to 0 for a promoted leg.

        A promoted foot is ON THE GROUND, so its arc is over: leaving the phase
        running would have the swing controller keep pushing the trajectory
        into the floor, which is the impulse the promotion exists to remove.
        """
        sp = self.gait.swing_phase(t)
        return np.where(self.early, 0.0, sp)

    # -- passthroughs, so a caller needs one handle ------------------------
    @property
    def stance_duration(self):
        return self.gait.stance_duration

    @property
    def swing_duration(self):
        return self.gait.swing_duration

    @property
    def period(self):
        return self.gait.period

    def phase(self, t):
        return self.gait.phase(t)

    def __repr__(self):
        return f"ContactAware({self.gait!r}, promote>{self.promote_after:.2f})"


class StandGait:
    """All four feet planted, for ever.  The same handle, without a clock.

    The RISE and the HOLD stages need every foot down and the gait clock
    ignored -- a crouched robot asked to balance on a diagonal is not a
    bring-up step -- and the MPC needs a weight schedule either way.  A tiny
    class beats a `if stage == ...` inside the controller, because the
    controller then has exactly one thing to ask.
    """

    def __init__(self, period=C.GAIT_PERIOD, duty=C.DUTY):
        self.period = float(period)
        self.duty = float(duty)

    def reset(self, t):
        pass

    def update(self, t, measured):
        return np.ones(4, dtype=bool), False

    def contact(self, t):
        return np.ones(4, dtype=bool)

    def contact_weight(self, t):
        return np.ones(4)

    def contact_weight_schedule(self, t, n, dt):
        return np.ones((int(n), 4))

    def swing_phase(self, t):
        return np.zeros(4)

    @property
    def stance_duration(self):
        return self.period * self.duty

    @property
    def swing_duration(self):
        return self.period * (1.0 - self.duty)

    def phase(self, t):
        return np.zeros(4)

    def __repr__(self):
        return "StandGait(all four planted)"


# ===========================================================================
# the placement filter
# ===========================================================================
class PlacementVelocity:
    """First-order low pass on the velocity the Raibert target is built from.

    PARAMETRISED BY A TIME CONSTANT, NOT A BLEND FACTOR, so it means the same
    thing when the model rate changes -- which it does between the 83 Hz model
    block and the 40 Hz solver.  The simulation makes the same choice for the
    same reason.

    WHY IT IS NEEDED ON TOP OF THE ESTIMATOR'S OWN 5 Hz FILTER.  The trunk
    velocity oscillates at GAIT frequency, 2.5 Hz here, which a 5 Hz corner
    passes essentially untouched.  Fed raw into the touchdown target, that
    oscillation moves the footholds in resonance with the rocking that
    produced it, and the limit cycle sustains itself.  At a 0.10 s time
    constant the placement sees 1.6 Hz, well under the gait.

    It filters the FOOT PLACEMENT ONLY.  The MPC and the wrench still act on
    the estimator's velocity, unfiltered by this: a controller wants the
    freshest state it can get, while a footstep two hundred milliseconds in
    the future wants the trend.
    """

    def __init__(self, tau=C.RAIBERT_V_TAU):
        self.tau = float(tau)
        self.v = np.zeros(3)

    def update(self, v, dt):
        dt = float(np.clip(dt, 1e-4, 0.5))
        alpha = dt / (self.tau + dt)
        self.v += alpha * (np.asarray(v, dtype=float).reshape(3) - self.v)
        return self.v.copy()

    def reset(self, v0=None):
        self.v = (np.zeros(3) if v0 is None
                  else np.asarray(v0, dtype=float).reshape(3).copy())
