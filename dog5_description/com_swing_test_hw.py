#!/usr/bin/env python3
"""Operator-selected 3-leg-stand test for DOG5: swing the CoM, lift ONE
chosen foot, and hold on three legs.

After the normal stand-up, ENTER arms the test and the robot waits flat on
four feet.  The operator then picks a corner with a number key:

    1 = RR    2 = FL    3 = FR    4 = RL

Each selection runs one full episode of the crawl step with zero forward
travel:

    SHIFT -> UNLOAD (leg must MEASURE unloaded) -> LIFT -> 3-LEG STAND HOLD
    -> ENTER -> LOWER -> LOAD -> RECENTER -> back to the four-foot wait

The hold pauses at the SWING phase start: the selected foot is airborne at
full swing height and the dog stands on the other three legs indefinitely
(support-margin e-stop and the foot-drag watch stay live) until ENTER lowers
it.  Foot-end load is monitored throughout via the measured-torque load map
(f = J^-T tau, gravity-compensated).

The unload gate does NOT abort the episode by default: if the leg still
measures loaded when the gate times out (shift + extra force-feedback
budget exhausted), the foot is lifted ANYWAY -- the goal of this test is
to make the foot leave the ground -- and the airborne drag watch
(|tau_ext| > 0.90 N*m sustained) plus the support-margin e-stop decide
whether it really did.  A completed forced episode is scored FORCED
instead of PASS.  Pass --strict-unload to restore the abort-on-timeout
behaviour.  Real failures (drag while "airborne", touchdown or recenter
timeout, SHIFT never settling) still run the crawl's graceful abort --
reload the foot where it stands, body back to neutral -- and the test
CONTINUES: the corner is scored FAIL and the robot returns to the
four-foot wait for the next key.  ENTER at the wait ends the batch and
prints the per-corner summary.

To focus on one corner, just press its key repeatedly -- failures do not
consume the episode budget.

Everything else (stand-up flow, safety gates, e-stops, status printout, CLI
flags) is inherited from crawl_dog5_hw; this file only swaps in a planner,
sequence, and key-poller subclass -- crawl_dog5_hw.py itself is not edited.

    python com_swing_test_hw.py --self-test
    python com_swing_test_hw.py --tau-max 3.0
"""
from __future__ import annotations

import sys
import time

import numpy as np

import dog5_kinematics
import crawl_dog5_hw as crawl

LEGS = crawl.LEGS
N_JOINTS = crawl.N_JOINTS

# Operator corner keys (user-specified mapping).
KEY_TO_LEG = {"1": "RR", "2": "FL", "3": "FR", "4": "RL"}
KEY_HINT = "1=RR 2=FL 3=FR 4=RL"

# Sampling period for the measured foot-load map used by the per-corner
# records (planner.update can be called at the CAN slot rate; the 3x3 SVDs
# are cheap but there is no reason to run them faster than this).
FZ_SAMPLE_PERIOD_S = 0.02

# A corner key pressed outside the CRAWL stage sits in the mailbox unseen;
# drop it instead of firing a stale selection when the stage is entered.
KEY_STALE_S = 1.0

# Single-slot mailbox filled by KeySpy (same thread as the control loop).
_MAILBOX = {"leg": None, "at": -np.inf}

# Live objects for key feedback, set by the subclass constructors below.
ACTIVE_PLANNER = None
ACTIVE_SEQUENCE = None


def _capture_corner_key(ch, stamp):
    """Handle one polled character.  Corner digits always get an immediate
    printed response -- silently swallowing a key press reads as a dead
    keyboard to the operator -- and are queued only when the planner is at
    the four-foot wait and can actually act on them."""
    if ch not in KEY_TO_LEG:
        return ch
    leg = KEY_TO_LEG[ch]
    planner, sequence = ACTIVE_PLANNER, ACTIVE_SEQUENCE
    if (
        planner is not None
        and planner.active
        and getattr(planner, "waiting", False)
    ):
        _MAILBOX["leg"] = leg
        _MAILBOX["at"] = stamp
        print(f"[com-swing] corner key {ch}={leg} received; starting episode")
    else:
        stage = sequence.stage if sequence is not None else "-"
        if planner is not None and planner.active:
            why = f"episode in progress ({planner.diagnostic_context})"
        else:
            why = (
                f"test not armed (stage={stage}); stand to HOLD, press "
                "ENTER to arm the test, THEN press corner keys"
            )
        print(f"[com-swing] corner key {ch}={leg} ignored: {why}")
    return ""


class KeySpy(crawl.base.KeyPoller):
    """KeyPoller that captures the corner digits and passes the rest on."""

    def get(self):
        return _capture_corner_key(super().get(), time.perf_counter())


class ComSwingTestPlanner(crawl.CrawlGaitPlanner):
    """Crawl planner driven one operator-selected corner at a time.

    Instead of stepping through GAIT_ORDER, the planner idles in a paused
    four-foot WAIT until a corner key selects the swing leg, runs that one
    episode (forcing a pause at the SWING start = the 3-leg stand hold),
    then returns to the WAIT.  Graceful aborts also return to the WAIT so
    a failed corner never ends the batch; only the operator's ENTER at the
    WAIT (or exhausting the episode budget) finishes it.

    With force_lift_on_timeout (the default), an unload-gate timeout lifts
    the foot anyway instead of aborting -- the drag watch and the margin
    e-stop stay armed to catch a foot that truly cannot leave the ground.
    """

    force_lift_on_timeout = True  # main() clears this for --strict-unload

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global ACTIVE_PLANNER
        ACTIVE_PLANNER = self
        self.results = []
        self.waiting = False
        self.selected_leg = None
        self._now = 0.0
        self._was_waiting = False
        self._episode_recorded = True
        self._forced_lift = False
        self._at_hold = False
        self._hold_started = None
        self._hold_s = np.nan
        self._hold_margin_m = np.nan
        self._hold_fz_air_n = np.nan
        self._hold_support = None
        self._fz_last_sample = -np.inf
        self._fz_swing_first = None
        self._fz_swing_min = np.inf
        self._last_support = None

    @property
    def gait_mode(self):
        return "COM_SWING"

    @property
    def swing_leg(self):
        if not self.active:
            return None
        # "--" (a string, not None) while waiting: the hardware status line
        # formats this with ":2s" and no leg name ever equals it, so every
        # foot counts as stance.
        return self.selected_leg if self.selected_leg is not None else "--"

    def actual_support_margin(self, q):
        if self.selected_leg is not None:
            return super().actual_support_margin(q)
        # Four-foot wait: the base helper insists on exactly three stance
        # feet.  Same signed edge-distance margin of the body origin, over
        # the support QUADRILATERAL instead.
        feet = self.actual_foot_positions(q)
        points = np.asarray([feet[leg][:2] for leg in LEGS], dtype=float)
        if not np.all(np.isfinite(points)):
            return np.nan
        centroid = points.mean(axis=0)
        angles = np.arctan2(
            points[:, 1] - centroid[1], points[:, 0] - centroid[0]
        )
        ordered = points[np.argsort(angles)]
        margins = []
        for start, end in zip(ordered, np.roll(ordered, -1, axis=0)):
            edge = end - start
            length = float(np.linalg.norm(edge))
            if length < 1.0e-6:
                return np.nan
            cross_z = edge[0] * (-start[1]) - edge[1] * (-start[0])
            margins.append(float(cross_z / length))
        return min(margins)

    def start(self, now):
        self.results = []
        super().start(now)

    def _reset_episode_trackers(self):
        self._forced_lift = False
        self._fz_swing_first = None
        self._fz_swing_min = np.inf
        self._last_support = None
        self._at_hold = False
        self._hold_started = None
        self._hold_s = np.nan
        self._hold_margin_m = np.nan
        self._hold_fz_air_n = np.nan
        self._hold_support = None

    def _begin_step(self, now, pause=False, mandatory_pause=False):
        """Enter the four-foot WAIT instead of auto-starting a GAIT_ORDER
        step.  Called by start() and after every completed episode."""
        if self.selected_leg is not None and not self._episode_recorded:
            self._record("FORCED" if self._forced_lift else "PASS")
        self.selected_leg = None
        self.waiting = True
        self.phase = "SHIFT"  # paused at s=0 => neutral four-foot targets
        self.phase_started_at = float(now)
        self.phase_settle_since = None
        self.phase_wait_started = None
        self.gate_timeout_reason = None
        self.mandatory_hold = False
        self.paused = True
        self.shifted_body_xy = self.body_xy.copy()
        self.base_shifted_body_xy = self.body_xy.copy()
        self.next_body_xy = self.body_xy.copy()
        self.unload_extra_m = 0.0
        self.unload_extra_cap_m = None
        self.unload_growth_dir = np.zeros(2)
        self.unload_grow_at = None
        self.swing_drag_streak = 0
        self._reset_episode_trackers()

    def select_leg(self, leg, now, q, qd):
        """Start one episode for the operator-chosen corner."""
        if not self.active or not self.waiting:
            return False, f"corner key {leg} ignored: {self.diagnostic_context}"
        error, speed = self._tracking_metrics(q, qd, now, LEGS)
        if error > crawl.OBSERVE_RESUME_CART_TOL_M:
            return False, (
                f"{leg} refused: stance not settled "
                f"(cart_err {error:.3f} m > "
                f"{crawl.OBSERVE_RESUME_CART_TOL_M:.3f} m)"
            )
        if speed > crawl.OBSERVE_RESUME_QD_TOL_RAD_S:
            return False, (
                f"{leg} refused: still moving (max |qd| {speed:.2f} rad/s > "
                f"{crawl.OBSERVE_RESUME_QD_TOL_RAD_S:.2f} rad/s)"
            )
        self.selected_leg = leg
        self._episode_recorded = False
        self._reset_episode_trackers()
        # Same bookkeeping as the base _begin_step, for the chosen leg and
        # zero forward travel.
        stance_pts = [self.world_foot_xy[s] for s in LEGS if s != leg]
        self.shifted_body_xy, self.shift_normal = crawl.adaptive_body_shift(
            self.world_foot_xy[leg], stance_pts, self.body_xy,
            self.shift_distance,
        )
        self.base_shifted_body_xy = self.shifted_body_xy.copy()
        self.unload_extra_m = 0.0
        self.unload_extra_cap_m = None
        self.unload_growth_dir = np.asarray(self.shift_normal, dtype=float)
        self.unload_grow_at = None
        self.swing_drag_streak = 0
        self.next_body_xy = self.body_xy.copy()
        self.new_swing_anchor_xy = self.world_foot_xy[leg].copy()
        self.waiting = False
        self.paused = False
        self.mandatory_hold = False
        self.phase = "SHIFT"
        self.phase_started_at = float(now)
        self.phase_settle_since = None
        self.phase_wait_started = None
        self.gate_timeout_reason = None
        return True, (
            f"corner {leg} selected (episode "
            f"{min(self.step_index + 1, self.total_steps)}/"
            f"{self.total_steps}): shifting the CoM off {leg}"
        )

    def request_finish(self, now):
        if not self.active or not self.waiting:
            return False, "test can only be ended from the four-foot wait"
        del now
        return True, self._finish(
            False, "CoM swing test ended by operator; neutral HOLD restored"
        )

    def _set_phase(self, phase, now, pause=True):
        super()._set_phase(phase, now, pause)
        if phase == "SWING":
            # The 3-leg stand: hold with the foot airborne at full swing
            # height until ENTER, regardless of --observe-phases.
            self.paused = True
            self._at_hold = True
            self._hold_started = float(now)

    def resume(self, now):
        if self._at_hold and self.phase == "SWING":
            self._hold_s = float(now) - self._hold_started
            self._at_hold = False
            ok, _ = super().resume(now)
            if not ok:
                return ok, _
            return True, (
                f"ENTER accepted: lowering {self.swing_leg} after "
                f"{self._hold_s:.1f}s of 3-leg stand"
            )
        return super().resume(now)

    def _sample_loads(self, now, q, measured_tau):
        if measured_tau is None:
            return
        if self.phase not in ("SHIFT", "UNLOAD", "LIFT", "SWING"):
            return
        if now - self._fz_last_sample < FZ_SAMPLE_PERIOD_S:
            return
        self._fz_last_sample = float(now)
        support, _ = crawl.foot_load_map(np.asarray(q, dtype=float), measured_tau)
        self._last_support = support
        fz = support.get(self.swing_leg, np.nan)
        if not np.isfinite(fz):
            return
        if self.phase == "SHIFT" and self._fz_swing_first is None:
            self._fz_swing_first = fz
        elif self.phase == "UNLOAD":
            self._fz_swing_min = min(self._fz_swing_min, fz)
        elif self.phase == "SWING" and self._at_hold:
            self._hold_fz_air_n = fz
            self._hold_support = dict(support)

    def _support_text(self, support):
        if not support:
            return "-"
        return ",".join(
            f"{leg}{support[leg]:+.1f}"
            for leg in LEGS
            if np.isfinite(support[leg])
        )

    def _record(self, verdict, reason=None):
        self._episode_recorded = True
        self.results.append(
            {
                "episode": len(self.results) + 1,
                "leg": self.selected_leg,
                "verdict": verdict,
                "forced": self._forced_lift,
                "reason": reason,
                "fz_first_n": self._fz_swing_first,
                "fz_min_n": self._fz_swing_min,
                "tau_ext_nm": self.last_unload_tau_nm,
                "rel_sag_m": self.last_unload_rel_sag_m,
                "base_shift_m": float(
                    np.linalg.norm(self.base_shifted_body_xy - self.body_xy)
                ),
                "extra_shift_m": self.unload_extra_m,
                "held": bool(self._hold_started is not None),
                "hold_s": self._hold_s,
                "hold_margin_m": self._hold_margin_m,
                "hold_fz_air_n": self._hold_fz_air_n,
                "hold_support": self._support_text(self._hold_support),
            }
        )

    def update(self, now, q, qd, measured_tau=None):
        self._now = float(now)
        if self.active and self.waiting:
            leg = _MAILBOX["leg"]
            if leg is not None:
                _MAILBOX["leg"] = None
                if now - _MAILBOX["at"] <= KEY_STALE_S:
                    return self.select_leg(leg, now, q, qd)[1]
        elif _MAILBOX["leg"] is not None:
            leg = _MAILBOX["leg"]
            _MAILBOX["leg"] = None
            return (
                f"corner key {leg} ignored ({self.diagnostic_context}); "
                "keys work only at the four-foot wait"
            )
        if self.active and not self.finished:
            self._sample_loads(self._now, q, measured_tau)
        was_waiting = self.waiting
        message = super().update(now, q, qd, measured_tau=measured_tau)
        if self.active and self._at_hold and self.phase == "SWING":
            if not np.isfinite(self._hold_margin_m):
                self._hold_margin_m = self.actual_support_margin(
                    np.asarray(q, dtype=float)
                )
                stance = [s for s in LEGS if s != self.swing_leg]
                forced = " (FORCED lift, gate timed out)" if self._forced_lift else ""
                message = (
                    f"3-LEG STAND{forced}: {self.swing_leg} airborne at "
                    f"{1000.0 * self.swing_height:.0f}mm, standing on "
                    f"{'/'.join(stance)} (margin="
                    f"{1000.0 * self._hold_margin_m:.1f}mm, unload "
                    f"|tau_ext|={self.last_unload_tau_nm:.2f}N*m, extra "
                    f"shift {1000.0 * self.unload_extra_m:.1f}mm); holding "
                    "until ENTER"
                )
        if self.waiting and not was_waiting and self.active:
            # The base message ends with GAIT_ORDER bookkeeping ("shifting
            # for None"); keep only the completion part and add the key hint.
            lead = (message or "episode done").split(";")[0]
            message = (
                f"{lead} -- four-foot wait: press {KEY_HINT} for the next "
                "corner, ENTER to end the test"
            )
        return message

    def _begin_abort(self, now, q, reason):
        if (
            self.phase == "UNLOAD"
            and self.force_lift_on_timeout
            and not self._forced_lift
        ):
            # The leg never measured unloaded, but the point of this test
            # is to make the foot leave the ground: lift anyway.  The drag
            # watch (LIFT/SWING, |tau_ext| > SWING_DRAG_TAU_TRIP sustained)
            # and the support-margin e-stop stay armed; if the foot truly
            # cannot leave the ground they abort the episode for real.
            self._forced_lift = True
            q = np.asarray(q, dtype=float)
            feet = self.actual_foot_positions(q)
            stance_points = [
                feet[name] for name in LEGS if name != self.selected_leg
            ]
            self.touchdown_plane_baseline_m = (
                crawl.signed_distance_to_stance_plane(
                    feet[self.selected_leg], stance_points
                )
            )
            self._set_phase("LIFT", now)
            return (
                f"{self.selected_leg} unload gate timed out "
                f"(|tau_ext|={self.last_unload_tau_nm:.2f}N*m, extra shift "
                f"{1000.0 * self.unload_extra_m:.1f}mm exhausted) -- "
                f"FORCING liftoff anyway; drag watch armed at "
                f"{crawl.SWING_DRAG_TAU_TRIP_NM:.2f}N*m "
                "(--strict-unload restores abort-on-timeout)"
            )
        self._record("FAIL", reason=reason)
        return super()._begin_abort(now, q, reason)

    def _finish(self, aborted, message):
        if aborted and self.active and self.step_index < self.total_steps:
            # The crawl ends its batch here; the swing test scores the
            # corner FAIL (already recorded in _begin_abort) and returns to
            # the four-foot wait for the next key instead.
            self.aborted = False
            self.abort_reason = None
            self._begin_step(self._now)
            return (
                f"{message.split(' -- ')[0]} -- corner FAILED; four-foot "
                f"wait: press {KEY_HINT} to test another corner, ENTER to "
                "end the test"
            )
        if self.selected_leg is not None and not self._episode_recorded:
            if aborted:
                self._record("FAIL", self.abort_reason)
            else:
                self._record("FORCED" if self._forced_lift else "PASS")
        result = super()._finish(aborted, message)
        self._print_summary()
        return result

    def _print_summary(self):
        print("[com-swing] ========== 3-LEG STAND TEST SUMMARY ==========")
        if not self.results:
            print("[com-swing] no corners were tested")
            return
        counts = {"PASS": 0, "FORCED": 0, "FAIL": 0}
        lifted_legs = set()
        for record in self.results:
            fz_min = record["fz_min_n"]
            fz_first = record["fz_first_n"]
            fz_min_text = "-" if not np.isfinite(fz_min) else f"{fz_min:.1f}N"
            fz_first_text = "-" if fz_first is None else f"{fz_first:.1f}N"
            detail = (
                f"unload fz {fz_first_text} -> {fz_min_text}, "
                f"|tau_ext|={record['tau_ext_nm']:.2f}N*m, "
                f"rel_sag={1000.0 * record['rel_sag_m']:+.1f}mm, "
                f"shift {1000.0 * record['base_shift_m']:.0f}"
                f"+{1000.0 * record['extra_shift_m']:.1f}mm"
            )
            counts[record["verdict"]] = counts.get(record["verdict"], 0) + 1
            if record["verdict"] in ("PASS", "FORCED"):
                lifted_legs.add(record["leg"])
                if record["verdict"] == "FORCED":
                    detail += "; gate timed out, LIFTED ANYWAY"
                detail += (
                    f"; 3-leg stand held {record['hold_s']:.1f}s, "
                    f"margin={1000.0 * record['hold_margin_m']:.1f}mm, "
                    f"airborne fz={record['hold_fz_air_n']:.1f}N"
                )
            else:
                if record["forced"]:
                    detail += "; FORCED lift attempted"
                detail += f" -- {record['reason']}"
            print(
                f"[com-swing] episode {record['episode']} {record['leg']}: "
                f"{record['verdict']:6s} ({detail})"
            )
            if (
                record["verdict"] in ("PASS", "FORCED")
                and record["hold_support"] != "-"
            ):
                print(
                    f"[com-swing]   load map during hold "
                    f"fz_N=({record['hold_support']})"
                )
        unlifted = [leg for leg in LEGS if leg not in lifted_legs]
        print(
            f"[com-swing] {counts['PASS'] + counts['FORCED']}/"
            f"{len(self.results)} episodes lifted into a 3-leg stand "
            f"(PASS={counts['PASS']} FORCED={counts['FORCED']} "
            f"FAIL={counts['FAIL']})"
            + (
                ""
                if not unlifted
                else f"; corners not lifted yet: {','.join(unlifted)}"
            )
        )


class ComSwingSequence(crawl.CrawlSequence):
    """ENTER at the four-foot wait ends the test; everywhere else it keeps
    the crawl sequence behaviour (resume pauses, start batches, ...)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global ACTIVE_SEQUENCE
        ACTIVE_SEQUENCE = self

    def request_next(self, now, q, qd, healthy):
        planner = self.planner
        if (
            self.stage == "CRAWL"
            and planner.active
            and getattr(planner, "waiting", False)
        ):
            return planner.request_finish(now)
        advanced, message = super().request_next(now, q, qd, healthy)
        if advanced and self.stage == "CRAWL" and planner.waiting:
            message = (
                f"ENTER accepted: CoM swing / 3-leg stand test armed -- "
                f"press {KEY_HINT} to swing that corner, ENTER to end "
                f"(episode budget {planner.total_steps})"
            )
        return advanced, message


def offline_self_test(args):
    """Drive the operator-selected planner offline through all four
    outcomes: clean PASS, forced-lift-then-drag FAIL, forced-lift success
    (FORCED), and strict-mode gate FAIL."""

    def make_rig(force_lift):
        controller = crawl.CrawlController(
            args.cart_gain_scale, args.support_scale, args.stand_height, 1.0
        )
        planner = ComSwingTestPlanner(
            controller,
            args.shift_distance,
            0.0,
            args.swing_height,
            args.leg_cycle,
            2,  # episode budget 8: room for 4 corners + retries
            unload_tau_trip=args.unload_tau_trip,
        )
        planner.force_lift_on_timeout = force_lift
        q = crawl.Q_RECORDED_CROUCH.copy()
        for index, leg in enumerate(LEGS):
            section = slice(3 * index, 3 * index + 3)
            q[section] = crawl._ik_to_target(
                leg, q[section], controller.final_foot[leg]
            )
        return controller, planner, q

    zero_qd = np.zeros(N_JOINTS)
    now_t = [0.0]

    def make_driver(planner, q):
        def synth_tau(load_mode):
            """Gravity-holding torque plus ground load: mg/3 on stance feet
            and, on the selected foot, a 15 N ground load per load_mode --
            "never" (clean unload), "always" (foot physically cannot leave
            the ground), or "until_lift" (breaks contact once lifted)."""
            tau = np.zeros(N_JOINTS)
            swing = planner.swing_leg
            for index, leg in enumerate(LEGS):
                section = slice(3 * index, 3 * index + 3)
                tau[section] = dog5_kinematics.leg_gravity_torque(
                    leg, q[section]
                )
                if leg == swing:
                    loaded = load_mode == "always" or (
                        load_mode == "until_lift"
                        and planner.phase in ("SHIFT", "UNLOAD")
                    )
                    if not loaded:
                        continue
                fz = (
                    15.0
                    if leg == swing
                    else crawl.base.DOG5_MASS_KG * 9.81 / 3.0
                )
                jac = dog5_kinematics.foot_jacobian(leg, q[section])
                tau[section] += jac.T @ np.array([0.0, 0.0, -fz])
            return tau

        def track(endpoint):
            for index, leg in enumerate(LEGS):
                desired = planner.target(leg, endpoint)[0]
                if leg != planner.swing_leg:
                    assert abs(desired[2] - planner.nominal_z[leg]) < 1.0e-9, (
                        "stance foot commanded off the ground",
                        planner.phase,
                        leg,
                    )
                section = slice(3 * index, 3 * index + 3)
                q[section] = crawl._ik_to_target(leg, q[section], desired)

        def tick(load_mode):
            return planner.update(
                now_t[0], q, zero_qd, measured_tau=synth_tau(load_mode)
            )

        def drive_episode(leg, load_mode="never", via_key=None):
            if via_key is not None:
                passed_on = _capture_corner_key(via_key, now_t[0])
                assert passed_on == "", passed_on
                event = tick(load_mode)
            else:
                ok, event = planner.select_leg(leg, now_t[0], q, zero_qd)
                assert ok, event
            assert leg in event and not planner.waiting, event
            assert planner.swing_leg == leg
            guard = 60
            while not planner.waiting and not planner.finished and guard > 0:
                guard -= 1
                if planner.paused and planner.phase == "SWING":
                    # The 3-leg stand hold: no timeout while held; a still-
                    # loaded forced lift is aborted here by the drag watch.
                    assert planner.three_leg_support
                    desired = planner.target(leg, now_t[0])[0]
                    assert abs(
                        desired[2]
                        - (planner.nominal_z[leg] + planner.swing_height)
                    ) < 1.0e-9, desired
                    now_t[0] += 30.0
                    # Each tick may abort a still-loaded forced lift (drag
                    # watch runs even while paused); a clean hold must stay
                    # paused at SWING no matter how much time passes.
                    tick(load_mode)
                    if planner.phase == "SWING" and planner.paused:
                        tick(load_mode)
                    if planner.phase == "SWING" and planner.paused:
                        ok, message = planner.resume(now_t[0])
                        assert ok and "3-leg stand" in message, message
                    continue
                phase = planner.phase
                endpoint = (
                    planner.phase_started_at + planner.duration(phase) + 1.0e-6
                )
                track(endpoint)
                times = [endpoint]
                if phase == "SHIFT":
                    times += [endpoint + crawl.SHIFT_SETTLE_S + 0.01]
                elif phase == "UNLOAD":
                    times += [
                        endpoint + crawl.UNLOAD_GATE_SETTLE_S + 0.01,
                        endpoint + crawl.UNLOAD_GATE_TIMEOUT_S + 0.01,
                    ]
                elif phase == "LOWER":
                    times += [endpoint + crawl.TOUCHDOWN_SETTLE_S + 0.01]
                elif phase == "RECENTER":
                    times += [endpoint + crawl.RECENTER_SETTLE_S + 0.01]
                for t in times:
                    now_t[0] = t
                    tick(load_mode)
                    if (
                        planner.waiting
                        or planner.finished
                        or planner.phase != phase
                    ):
                        break
            assert guard > 0, "episode did not converge"
            now_t[0] += 1.0

        return drive_episode

    # ---- default rig: forced lift enabled --------------------------------
    controller, planner, q = make_rig(force_lift=True)
    hold_before = {leg: controller.final_foot[leg].copy() for leg in LEGS}
    drive_episode = make_driver(planner, q)

    # Corner keys pressed before the test is armed must respond (printed
    # hint) but queue nothing -- a silent swallow reads as a dead keyboard.
    assert _capture_corner_key("1", 0.0) == ""
    assert _MAILBOX["leg"] is None
    assert _capture_corner_key("\n", 0.0) == "\n"  # other keys pass through

    planner.start(now_t[0])
    assert planner.waiting and planner.paused
    # The hardware loop, every tick of the four-foot wait, formats the swing
    # leg with ":2s" and computes the support margin -- both crashed on the
    # rig with no leg selected (None format / 4-point triangle).  Exercise
    # the exact expressions.
    assert "{:2s}".format(planner.swing_leg) == "--"
    wait_margin = planner.actual_support_margin(q)
    assert wait_margin > crawl.MIN_LIFTOFF_MARGIN_M, wait_margin

    drive_episode("RR", via_key="1")  # clean PASS with a 30 s 3-leg hold
    # A corner key mid-episode is refused at capture time, loudly.
    planner.waiting = False  # simulate mid-episode
    assert _capture_corner_key("2", now_t[0]) == ""
    assert _MAILBOX["leg"] is None
    planner.waiting = True
    # FL physically cannot unload: gate times out, lift is FORCED anyway,
    # then the airborne drag watch catches the still-loaded foot -> FAIL.
    drive_episode("FL", load_mode="always", via_key="2")
    assert planner.waiting and not planner.finished and not planner.aborted
    # FR stays loaded through the whole CoM swing but breaks contact once
    # lifted: gate times out, forced lift succeeds -> FORCED with a hold.
    drive_episode("FR", load_mode="until_lift", via_key="3")
    drive_episode("RL", via_key="4")  # clean PASS
    ok, message = planner.request_finish(now_t[0])
    assert ok and planner.finished and not planner.aborted, message

    assert len(planner.results) == 4, planner.results
    by_leg = {record["leg"]: record for record in planner.results}
    assert set(by_leg) == set(LEGS)
    for leg in ("RR", "RL"):
        record = by_leg[leg]
        assert record["verdict"] == "PASS" and not record["forced"], record
        assert record["tau_ext_nm"] < args.unload_tau_trip, record
        assert record["fz_min_n"] < 1.0, record
        assert record["held"] and record["hold_s"] > 29.0, record
        assert record["hold_margin_m"] > crawl.MIN_LIFTOFF_MARGIN_M, record
        assert abs(record["hold_fz_air_n"]) < 1.0, record
    record = by_leg["FL"]
    assert record["verdict"] == "FAIL" and record["forced"], record
    assert "still loaded while airborne" in record["reason"], record
    record = by_leg["FR"]
    assert record["verdict"] == "FORCED" and record["forced"], record
    assert record["fz_min_n"] > 10.0, record  # never unloaded on the ground
    assert record["held"] and record["hold_s"] > 29.0, record
    assert abs(record["hold_fz_air_n"]) < 1.0, record  # airborne once lifted
    # The neutral hold is exactly restored after PASS/FORCED/FAIL alike.
    for leg in LEGS:
        assert np.allclose(
            controller.final_foot[leg], hold_before[leg], atol=1.0e-9
        ), leg
        actual = dog5_kinematics.foot_position(
            leg, q[3 * LEGS.index(leg) : 3 * LEGS.index(leg) + 3]
        )
        assert np.allclose(actual, controller.final_foot[leg], atol=1.0e-4), leg

    # ---- strict rig: --strict-unload restores abort-on-timeout -----------
    _, strict, q_strict = make_rig(force_lift=False)
    drive_strict = make_driver(strict, q_strict)
    strict.start(now_t[0])
    drive_strict("FL", load_mode="always")
    assert strict.waiting and not strict.finished
    record = strict.results[0]
    assert record["verdict"] == "FAIL" and not record["forced"], record
    assert "unload gate timed out" in record["reason"], record

    print(
        "[com-swing] self-test OK: clean corners PASS with a 30 s 3-leg "
        "hold; a gate timeout FORCES liftoff (drag watch FAILs a foot that "
        "stays loaded, FORCED verdict when it breaks contact); "
        "--strict-unload restores abort-on-timeout; neutral restored"
    )
    return 0


def main():
    strict = "--strict-unload" in sys.argv
    if strict:
        # Our flag only; crawl's parser must not see it.
        sys.argv = [arg for arg in sys.argv if arg != "--strict-unload"]
        ComSwingTestPlanner.force_lift_on_timeout = False
    if "--self-test" in sys.argv:
        args = crawl.build_parser().parse_args()
        return offline_self_test(args)
    print(
        "[com-swing] 3-LEG STAND TEST: after the crawl HOLD, ENTER arms the "
        f"test; then {KEY_HINT} swings the CoM off that corner and LIFTS it "
        "(SHIFT -> UNLOAD -> LIFT -> 3-LEG HOLD -> ENTER lowers -> LOAD -> "
        "RECENTER).  Gate failures reload, recenter, score FAIL, and wait "
        "for the next key.  ENTER at the four-foot wait ends the test and "
        "prints the summary."
    )
    if strict:
        print(
            "[com-swing] STRICT UNLOAD: an unload-gate timeout aborts the "
            "episode (no forced lift)."
        )
    else:
        print(
            "[com-swing] FORCED LIFT (default): if the unload gate times "
            "out, the foot is lifted ANYWAY (scored FORCED); the drag watch "
            f"(|tau_ext| > {crawl.SWING_DRAG_TAU_TRIP_NM:.2f}N*m) and the "
            "support-margin e-stop catch a foot that cannot leave the "
            "ground.  --strict-unload restores abort-on-timeout."
        )
    crawl.CrawlGaitPlanner = ComSwingTestPlanner
    crawl.CrawlSequence = ComSwingSequence
    crawl.base.KeyPoller = KeySpy
    if "--swing-test" not in sys.argv:
        sys.argv.append("--swing-test")  # force zero forward step length
    if not any(arg.startswith("--gait-cycles") for arg in sys.argv):
        sys.argv += ["--gait-cycles", "5"]  # episode budget 20; ENTER ends
    return crawl.main()


if __name__ == "__main__":
    sys.exit(main())
