"""In-place trot scheduler with closed-loop foot placement (Gate D1).

This is `dog5_gait.GaitScheduler`'s DIAGONAL mode made *dynamic* and made
*closed-loop*.  Two changes, and the second one is the point of the file.

1. Timing.  `GaitParams` defaults to period 2.0 s with `ds_frac = 0.6`, i.e.
   0.4 s of single-diagonal support per half-cycle.  That is quasi-static
   stepping.  A trot runs the same {FL,RR} <-> {FR,RL} footfall at ~0.4 s with a
   short double-support window.

2. Foot placement is closed on the estimator.  `GaitScheduler.swing_targets()`
   is pure feedforward -- nominal + a sin lift arc -- so nothing regulates where
   the body ends up.  A trot with no velocity feedback into footfall location
   trots and drifts away.  Here the swing foot is placed using the EKF's body
   velocity (Raibert neutral point + a corrective term), which is what actually
   makes "in place" a controlled quantity rather than a hope.

Why this is safe to close on, given x/y/yaw are unobservable
------------------------------------------------------------
Foot placement uses **velocity only** (`est_out["v_body"]`), never `r[:2]`.
Velocity is observable with >= 1 contact and is hardware-validated (static
median 1.4 mm/s, touchdown jumps <= 1.1 mm/s, CONTROL_ROADMAP.md Sec. 1).
Position x/y drifts by construction and is never read here.  This mirrors the
discipline already in `dog5_vmc_core.body_wrench`: stiffness on {z, roll, pitch},
damping only on {x, y, yaw}.

Why a trot survives a support line a static 2-leg stand cannot
--------------------------------------------------------------
On a 2-foot diagonal the stance is rank-deficient: there is no moment authority
about the support line, and `distribute_wrench` silently drops that component.
That is exactly the Stage 8/9 failure (`twostand_dog5.py --sweep`: 0/9, one leg
lifts clean and the other never leaves the ground).  A trot does not resist the
tip -- it outruns it.  Uncontrolled tip about the diagonal grows as

    theta(T_ss) ~ 0.5 * (m*g*d / I_axis) * T_ss^2

with d the CoM offset from the support line.  The double-support window then
restores full 4-foot rank so the wrench can null what accumulated.

That is the mechanism, but it is NOT the binding constraint at this scale, and
the sweep says so plainly: across nine (period, ds_frac) cells the measured
tilt/T_ss^2 ranges from 10 to 225 deg/s^2 instead of collapsing to a constant.
What actually governs is torque authority.  Below ds_frac ~ 0.30 the wrench the
VMC asks for stops fitting inside the per-joint clamp, the controller silently
stops delivering it, and tilt/drift/velocity error all degrade together.  The
tip relation is why a trot is *possible* where a static 2-leg stand is not; it
is not what sets the parameters.  See `TrotParams` and TROT_D1.md.

Contact flags
-------------
The flags are fed identically to the estimator and to the VMC.  A scheduled
swing foot is reported AIRBORNE for its whole swing and is re-planted only when
the estimate says it actually landed (`early_td_*`), rather than on the clock.

The asymmetry is deliberate and it is the opposite of the crawl's rule.  The
crawl (`walk1_hw.py`, `EKF_SWING_Z_EPS_M`) biases toward *planted*, because
`stand_hier_hw.py:384-406` records what the other bias cost there: calling an
inertially-stationary foot airborne dead-reckoned the base through an 8 s
zero-contact window and produced +-70-80 mm of z error.  A trot inverts both
halves of that trade.  There are always two other feet down, so the filter never
dead-reckons and losing one measurement is nearly free -- while the swing is so
fast that calling a moving foot "planted" asserts that a point travelling at
~0.1-0.4 m/s is fixed in inertial space, which goes straight into z and v.
Measured: the naive port of the crawl's rule cost 68.8 mm/s of EKF velocity
error, against a 50 mm/s gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from dog5_gait import LEGS, DIAG_A, DIAG_B

N_LEGS = len(LEGS)


@dataclass
class TrotParams:
    # --- timing -------------------------------------------------------------
    # period 0.40 / ds_frac 0.34 -> half-cycle 0.200 s = 0.068 s double support
    # + 0.132 s single support; per-leg duty factor 0.67.  Flight-free by
    # construction: the swing pair is never larger than one diagonal, so at
    # least two feet are always down.
    #
    # Both numbers are MEASURED, not chosen (see TROT_D1.md).  Two effects
    # bracket them from opposite sides:
    #
    #  * ds_frac below ~0.30 saturates the VMC.  Peak joint torque pins at
    #    exactly tau_max = 8.0 N.m, so the controller stops delivering the wrench
    #    it asked for, and tilt / drift / velocity error all degrade together.
    #    Measured at period 0.40: ds 0.18 -> tilt 4.64 deg, drift 3.04 mm/cycle,
    #    tau 8.00 (saturated);  ds 0.34 -> tilt 0.18 deg, drift 0.17 mm/cycle,
    #    tau 2.40 (unsaturated).  Raising tau_max is not the fix -- 8 N.m is
    #    already past the 6.0 N.m hardware torque trip.
    #  * period below ~0.35 is limited by swing dynamics rather than tipping:
    #    the same lift must be flown in less time, so foot accelerations and
    #    torque climb again (period 0.30 -> tilt 2.20 deg, tau 4.02) even though
    #    the shorter single support should help the tip.
    #
    # The flat optimum is period 0.40-0.45 with ds_frac 0.30-0.46.
    period: float = 0.40         # s, full cycle (both diagonals step once)
    ds_frac: float = 0.34        # double-support fraction of each half-cycle
    lift: float = 0.022          # m, swing apex above the nominal foot
    step_len: float = 0.0        # m, forward travel per step (0 = in place)

    # --- closed-loop foot placement ----------------------------------------
    # dp = (T_st/2)*v_err  (Raibert neutral point: the foot that neither
    #      accelerates nor decelerates the body at the current speed)
    #    + k_place*v_err   (corrective: drives v toward v_cmd)
    # With T_st = 0.2 s the neutral term alone is 0.10 s of gain; k_place adds
    # the correction on top.  Effective 0.16 s -> a 0.1 m/s drift moves the
    # foothold 16 mm, just inside the clamp.
    k_place: float = 0.12        # s, corrective velocity gain
    place_clamp: float = 0.018   # m, per-axis foothold offset limit
    # The clamp is not a tuning knob -- abduction range caps lateral authority
    # at ~2 cm (CONTROL_ROADMAP.md:184), so a larger lateral offset is not
    # reachable and would only wind up the IK.
    v_lpf_alpha: float = 0.25    # low-pass on the estimated velocity used for
                                 # placement; touchdown transients are real but
                                 # should not snap the next foothold

    # --- contact flags ------------------------------------------------------
    early_td_eps: float = 0.002  # m, estimated foot height below which a
                                 # descending swing foot is declared landed
    early_td_from: float = 0.6   # only look for early touchdown after this
                                 # fraction of the swing (foot is descending)

    # --- body targets -------------------------------------------------------
    z_des: float = 0.0           # trunk height target in the ESTIMATOR frame
                                 # (0 = hold the height at estimator init)
    v_cmd_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yawrate_cmd: float = 0.0

    def __post_init__(self):
        if not 0.0 < self.period:
            raise ValueError("period must be positive")
        if not 0.0 <= self.ds_frac < 1.0:
            raise ValueError("ds_frac must be in [0, 1)")

    @property
    def half(self):
        return self.period / 2.0

    @property
    def t_ds(self):
        """Double-support duration at the start of each half-cycle (s)."""
        return self.ds_frac * self.half

    @property
    def t_sw(self):
        """Single-support (one diagonal swinging) duration per half-cycle (s)."""
        return self.half - self.t_ds

    @property
    def stance_time(self):
        """How long a given foot stays down between its own swings (s)."""
        return self.period - self.t_sw


class TrotScheduler:
    """Emit contacts, swing foot targets and body targets vs time.

    Same output contract as `dog5_gait.GaitScheduler.sched_state` -- the dict is
    consumed unchanged by `dog5_vmc_core.compute_vmc_torques`, so the VMC core
    needs no modification.  The one difference is the signature: `sched_state`
    takes the estimator output, because foot placement is closed on it.
    """

    def __init__(self, nominal_feet_body, params: TrotParams | None = None,
                 mode: str = "TROT"):
        # nominal_feet_body: {leg: (3,)} foot positions in the body frame captured
        # at the standing hand-off; swings lift from and return to these.
        self.nominal = {l: np.asarray(nominal_feet_body[l], dtype=float)
                        for l in LEGS}
        self.p = params or TrotParams()
        # HOLD keeps all four feet planted with no swing -- the same four-foot
        # hold that gate V1 already proves, reused so the hand-off into TROT is
        # a mode flip rather than a different object.
        self.mode = mode
        self._v_filt = np.zeros(3)
        self._v_ready = False
        self._ground_z = None      # inertial z of the contact plane, lazily set
        self._landed = set()       # legs confirmed landed early this swing
        self._swing_id = None      # which (half-cycle) swing the set refers to

    # ------------------------------------------------------------------ timing
    def _swing_state(self, t):
        """Return (swing_pair or None, u in [0,1], swing_index) for time t."""
        p = self.p
        if self.mode != "TROT":
            return None, 0.0, None
        phase = t % p.period
        cycle = int(t // p.period)
        if phase < p.half:
            local, pair, k = phase, DIAG_A, 0
        else:
            local, pair, k = phase - p.half, DIAG_B, 1
        if local < p.t_ds or p.t_sw <= 0.0:
            return None, 0.0, None                 # double support
        u = (local - p.t_ds) / p.t_sw
        return pair, min(max(u, 0.0), 1.0), (cycle, k)

    # ------------------------------------------------------- foot placement
    def _placement_offset(self, est_out):
        """Raibert foothold offset in the BODY frame xy, from EKF velocity.

        Returns (dp_xy(2,), dv_xy(2,)) where dv is the offset's contribution to
        the commanded foot velocity over one swing.
        """
        p = self.p
        v_b = np.asarray(est_out["v_body"], dtype=float)[:2]
        if not self._v_ready:
            self._v_filt[:2] = v_b
            self._v_ready = True
        else:
            a = p.v_lpf_alpha
            self._v_filt[:2] = a * v_b + (1.0 - a) * self._v_filt[:2]
        # v_cmd is expressed in the world frame; for an in-place trot it is zero,
        # and zero is frame-invariant, so no rotation is needed in that case.
        # A non-zero command is rotated by the caller before it gets here.
        v_err = self._v_filt[:2] - np.asarray(p.v_cmd_world, dtype=float)[:2]
        dp = (0.5 * p.stance_time + p.k_place) * v_err
        dp = np.clip(dp, -p.place_clamp, p.place_clamp)
        return dp

    # ---------------------------------------------------------------- contacts
    @staticmethod
    def _arc_z(u):
        """Vertical swing profile, normalised: 0 at u=0 and u=1, 1 at u=0.5.

        Raised cosine, NOT sin(pi*u).  A sine arc has its MAXIMUM vertical foot
        speed exactly at liftoff and touchdown -- for a 22 mm lift in a 0.164 s
        swing that is 0.42 m/s at both ends, which slams the foot down and makes
        the contact flag wrong at the worst possible moment.  The raised cosine
        has zero derivative at both ends: the foot leaves and lands softly, so
        touchdown injects almost no impulse into the estimator.
        """
        return 0.5 * (1.0 - math.cos(2.0 * math.pi * u))

    @staticmethod
    def _arc_z_du(u):
        return math.pi * math.sin(2.0 * math.pi * u)

    @staticmethod
    def _smoothstep(u):
        """Horizontal blend with zero slope at both ends (no touchdown scuff)."""
        return u * u * (3.0 - 2.0 * u)

    @staticmethod
    def _smoothstep_du(u):
        return 6.0 * u * (1.0 - u)

    def _commanded_lift(self, t, leg):
        """Commanded z offset above nominal for `leg` at time t (m)."""
        pair, u, _ = self._swing_state(t)
        if pair is None or leg not in pair:
            return 0.0
        return self.p.lift * self._arc_z(u)

    def contacts(self, t, est_out=None, foot_pos_body=None):
        """(4,) bool stance mask, fed identically to the estimator and the VMC.

        A scheduled-stance foot is always planted.  A scheduled-swing foot is
        airborne only while its commanded lift clears `swing_z_eps`, and is
        re-planted early if the estimate says it already touched down.
        """
        p = self.p
        pair, u, sid = self._swing_state(t)
        c = np.ones(N_LEGS, dtype=bool)
        if pair is None:
            self._landed.clear()
            self._swing_id = None
            # Double support: all four feet are down, so this is the only moment
            # the contact plane can honestly be measured.  Latching it mid-swing
            # would average in two airborne feet and bias every later height
            # test by roughly half the lift.
            if est_out is not None and foot_pos_body is not None:
                self._refresh_ground_z(est_out, foot_pos_body)
            return c
        if sid != self._swing_id:                  # new swing -> forget the old
            self._swing_id = sid
            self._landed.clear()
        # A scheduled-swing foot is reported AIRBORNE for its whole swing, and is
        # re-planted only when the estimate says it has actually touched down.
        #
        # The asymmetry is deliberate.  Calling a moving foot "planted" tells the
        # filter a point that is travelling at ~0.1-0.4 m/s is fixed in inertial
        # space, and that error goes straight into z and v.  Calling a resting
        # foot "airborne" merely forgoes one measurement, and with a trot there
        # are always two other feet down, so the filter never dead-reckons -- the
        # condition that made the opposite bias expensive in
        # `stand_hier_hw.py:384-406` (an 8 s window with ZERO contacts) simply
        # cannot arise here.  So: bias airborne during swing, and re-plant on
        # evidence rather than on the clock.
        for leg in pair:
            if leg in self._landed:
                continue                            # already re-planted
            if (u >= p.early_td_from and est_out is not None
                    and foot_pos_body is not None
                    and self._foot_height(est_out, foot_pos_body, leg)
                    <= p.early_td_eps):
                self._landed.add(leg)               # touchdown detected
                continue
            c[LEGS.index(leg)] = False
        return c

    def _foot_z_inertial(self, est_out, foot_pos_body):
        """Inertial z of all four feet, from the estimate + encoder FK only.

        No privileged state: r and C come from the EKF, the foot vector from
        `dog5_kinematics.foot_position`.  Only *differences* of these are ever
        used, so the unobservable x/y drift cancels.
        """
        r = np.asarray(est_out["r"], dtype=float)
        C = est_out["C"]
        return np.array([float((r + C.T @ np.asarray(foot_pos_body[i]))[2])
                         for i in range(N_LEGS)])

    def _refresh_ground_z(self, est_out, foot_pos_body):
        """Latch the contact plane from all four feet (double support only)."""
        self._ground_z = float(np.mean(
            self._foot_z_inertial(est_out, foot_pos_body)))

    def _foot_height(self, est_out, foot_pos_body, leg):
        """Estimated height of `leg`'s foot above the contact plane (m).

        Returns +inf until the plane has been latched during a double-support
        window, so early-touchdown detection stays disarmed rather than firing
        on an unreferenced measurement.
        """
        if self._ground_z is None:
            return float("inf")
        z = self._foot_z_inertial(est_out, foot_pos_body)
        return float(z[LEGS.index(leg)]) - self._ground_z

    def stance_set(self, t, est_out=None, foot_pos_body=None):
        c = self.contacts(t, est_out, foot_pos_body)
        return [LEGS[i] for i in range(N_LEGS) if c[i]]

    # ----------------------------------------------------------- swing targets
    def swing_targets(self, t, est_out=None):
        """{leg: (p_des(3), v_des(3))} in the body frame for swinging legs."""
        p = self.p
        pair, u, _ = self._swing_state(t)
        if pair is None:
            return {}
        dp = (self._placement_offset(est_out) if est_out is not None
              else np.zeros(2))
        du_dt = 1.0 / p.t_sw if p.t_sw > 0 else 0.0
        out = {}
        s, ds_du = self._smoothstep(u), self._smoothstep_du(u)
        for leg in pair:
            p0 = self.nominal[leg]
            # z: raised-cosine lift arc -- back to nominal at u = 1 with zero
            # vertical speed, so the foot lands softly instead of slamming.
            dz = p.lift * self._arc_z(u)
            ddz_du = p.lift * self._arc_z_du(u)
            # xy: the scripted step plus the closed-loop placement offset, both
            # blended with a smoothstep so the foot neither jumps at liftoff nor
            # scuffs sideways at touchdown, and both are fully applied on landing.
            dx = (p.step_len + dp[0]) * s
            ddx_du = (p.step_len + dp[0]) * ds_du
            dy = dp[1] * s
            ddy_du = dp[1] * ds_du
            p_des = p0 + np.array([dx, dy, dz])
            v_des = np.array([ddx_du, ddy_du, ddz_du]) * du_dt
            out[leg] = (p_des, v_des)
        return out

    # ------------------------------------------------------------------ bundle
    def sched_state(self, t, est_out=None, foot_pos_body=None):
        """Everything the VMC needs this tick (same keys as GaitScheduler)."""
        p = self.p
        return {
            "contacts": self.contacts(t, est_out, foot_pos_body),
            "swing_targets": self.swing_targets(t, est_out),
            "z_des": p.z_des,
            "v_cmd_world": p.v_cmd_world,
            "yawrate_cmd": p.yawrate_cmd,
        }
