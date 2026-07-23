#!/usr/bin/env python3
"""Two-leg DIAGONAL stand on DOG5 hardware -- position-command track.

The multi-cycle walk sessions produced a surprise: the dog held a 2-leg
diagonal stand (FL+RR) "for a while" when a step teetered -- the support
pattern of a TROT.  This script makes that deliberate and safe: rise on
four legs, then lift one diagonal PAIR and balance on the other.

Physics to know before running:
  * A 2-leg diagonal stand is STATICALLY UNSTABLE (the support "polygon"
    is a line).  It holds for seconds through foot-rubber compliance and
    wide feet, not through control -- position mode has no balance law.
  * HARDWARE RESULT (first run, 2026-07-23): lifting FR+RL (standing on
    FL+RR) does NOT produce a 2-leg stand -- the CoM sits LEFT of the
    FL-RR diagonal, so the body rolled to a steady -8.5 deg and rested
    on RL as a third support; RL's commanded 29 mm raise was eaten by
    the sag and its |tau| read ~0.2 "airborne" (the tau gate is blind
    to load whose force line passes the sensed joints).
  * The CoM sits NEAR the FR-RL diagonal (also seen in walk1: lifting
    FL pops RR up -- the dog spontaneously stands on FR+RL).  So the
    DEFAULT here lifts FL+RR and stands on FR+RL, the diagonal that can
    actually balance.  Use --bias-x/--bias-y (a few mm) to trim the CoM
    onto the line if it keeps settling the same direction.
  * A steady large roll in HOLD2 = resting on a third leg, NOT a 2-leg
    stand; the HOLD2 entry line prints the lean delta so you can tell.

Safety net: IMU is REQUIRED (override with --no-imu at your own risk);
25 deg tip-over run-stop, --lean-abort-deg (default 8, delta from the
lift baseline) puts the feet straight back down (graceful, no e-stop);
--hold-s (default 10 s, 0 = until ENTER) auto-puts-down; ENTER during
the 2-leg hold puts down immediately; torque/speed/CAN trips unchanged
from the proven walk scripts.

Flow (ENTER at each boundary, X stops, P parks from HOLD4):

    READ -> CROUCH -> WAIT_CROUCH -ENTER-> STAND (vertical) -> HOLD4
    -ENTER-> SHIFT (optional --bias-x/--bias-y starting offset)
          -> WAIT_LIFT -ENTER-> UNLOAD2 (both feet +15 mm)
          -> BALANCE: the body creeps perpendicular to the stance line,
             AWAY from whichever lifted leg still carries torque, until
             BOTH read free (cap 30 mm, timeout 25 s -> put-down).
             This is the CoM shift the open-loop attempts were missing.
          -> WAIT_LIFT2 -ENTER-> LIFT2 (+14 mm) -> HOLD2 (the 2-leg
             stand: timer / ENTER / lean-abort -> put-down)
          -> LOWER2 (feet down at the balanced body position)
          -> RELOAD -> RECENTER -> HOLD4 (repeat or P parks)

First checks:
    python twostand_hw.py --self-test
Hardware (robot supported until proven):
    python twostand_hw.py                      # lift FL,RR (default)
    python twostand_hw.py --bias-y -0.005      # trim CoM right if it
                                               # settles left, and v.v.
    python twostand_hw.py --hold-s 0           # hold until ENTER
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dog5_description"))

import dog5_kinematics                      # noqa: E402
import stand3_hold_hw as s3                 # noqa: E402
import stand_dog5_hw as base                # noqa: E402
import stand_by_position_command as posbase  # noqa: E402
import stand_dog5_inplace_hw as inplace     # noqa: E402

LEGS = s3.LEGS
MOTOR_IDS = s3.MOTOR_IDS
MOTOR_DIRECTIONS = s3.MOTOR_DIRECTIONS
JOINT_LABELS = s3.JOINT_LABELS
N_JOINTS = s3.N_JOINTS
Q_CROUCH = s3.Q_CROUCH

DIAGONALS = (frozenset(("FL", "RR")), frozenset(("FR", "RL")))
DEFAULT_LIFT_PAIR = ("FL", "RR")     # stand on FR+RL, the CoM's diagonal
DEFAULT_STAND_HEIGHT_M = s3.DEFAULT_STAND_HEIGHT_M      # 0.175
PRE2_START_M = 0.015
PRE2_MAX_M = 0.025                   # validator envelope headroom
LIFT2_M = 0.014                      # above the pre-lift, same as walk
T_LOWER2 = 1.5
T_RELOAD = 1.0
DEFAULT_HOLD_S = 10.0
DEFAULT_LEAN_ABORT_DEG = 8.0         # 2-leg WILL lean; delta from baseline
BIAS_MAX_M = 0.020
# BALANCE: closed-loop CoM shift (hardware lesson 2026-07-23: BOTH
# diagonals fail open-loop -- the second foot never lifts until the CoM
# is moved onto the stance line, and the offset is invisible to
# geometry).  During the pre-lift the body creeps PERPENDICULAR to the
# stance diagonal, away from whichever lifted leg still carries torque,
# until both read free.
SHIFT2_RATE_M_S = 0.004              # body creep speed
SHIFT2_MAX_M = 0.025                 # travel cap -- the abd wall binds
                                     # at 30 (validator, both diagonals);
                                     # CoM offset to find is ~9 mm
BALANCE_TIMEOUT_S = 25.0
# Primary balance feedback is the IMU TILT DELTA from the 4-leg
# baseline, not torque: hardware run 3 proved the tau reading is blind
# in exactly the failure state (RL resting under the tipped body read
# 0.13-0.28 N*m "airborne" -- force line through the sensed joints).
# A third-leg rest = constant large tilt; balanced = tilt near the
# baseline.  The body creeps UPHILL until the tilt collapses.
TILT_BALANCED_DEG = 1.2


class TwoPlan:
    """Feet fixed at crouch xy; body optionally biased; one diagonal pair
    gets vertical pre-lift/lift offsets."""

    def __init__(self, lift_pair, stand_height=DEFAULT_STAND_HEIGHT_M,
                 bias_xy=(0.0, 0.0)):
        self.lifted = tuple(lift_pair)
        self.stance = tuple(leg for leg in LEGS if leg not in self.lifted)
        self.stand_height = float(stand_height)
        self.bias = np.array(bias_xy, dtype=float)
        self.crouch_foot = {
            leg: dog5_kinematics.foot_position(leg, Q_CROUCH[self._sl(leg)])
            for leg in LEGS
        }
        # Stance-diagonal geometry for the BALANCE stage: unit vector
        # PERPENDICULAR to the stance line, and each lifted foot's side.
        p0 = self.crouch_foot[self.stance[0]][:2]
        p1 = self.crouch_foot[self.stance[1]][:2]
        t = (p1 - p0) / np.linalg.norm(p1 - p0)
        self.perp = np.array([-t[1], t[0]])
        self.lift_side = {
            leg: float(np.sign(np.dot(self.crouch_foot[leg][:2] - p0,
                                      self.perp)))
            for leg in self.lifted
        }

    @staticmethod
    def _sl(leg):
        i = LEGS.index(leg)
        return slice(3 * i, 3 * i + 3)

    def targets(self, body_xy, height_frac, pre_m, lift_frac):
        out = {}
        for leg in LEGS:
            xy = self.crouch_foot[leg][:2]
            crouch_z = self.crouch_foot[leg][2]
            z = crouch_z + height_frac * (-self.stand_height - crouch_z)
            if leg in self.lifted:
                z += pre_m + lift_frac * LIFT2_M
            out[leg] = np.array([xy[0] - body_xy[0],
                                 xy[1] - body_xy[1], z])
        return out


STAGES = [
    ("CROUCH", "move"), ("WAIT_CROUCH", "wait"),
    ("STAND", "stream"), ("STAND_SETTLE", "gate"), ("HOLD4", "wait"),
    ("SHIFT", "stream"), ("SHIFT_SETTLE", "gate"), ("WAIT_LIFT", "wait"),
    ("UNLOAD2", "stream"), ("BALANCE", "gate"), ("WAIT_LIFT2", "wait"),
    ("LIFT2", "stream"), ("HOLD2", "wait"),
    ("LOWER2", "stream"), ("RELOAD", "dwell"), ("RECENTER", "stream"),
    ("PARK", "move"), ("PARKED", "wait"),
]
STAGE_INDEX = {name: i for i, (name, _) in enumerate(STAGES)}
STREAM_T = {"STAND": s3.T_STAND, "SHIFT": s3.T_SHIFT,
            "UNLOAD2": s3.T_UNLOAD, "LIFT2": s3.T_LIFT,
            "LOWER2": T_LOWER2, "RECENTER": s3.T_SHIFT}
LEAN_STAGES = ("UNLOAD2", "BALANCE", "WAIT_LIFT2", "LIFT2", "HOLD2")
INSTRUCTION = {
    "CROUCH": "moving to recorded crouch; wait, X stops",
    "WAIT_CROUCH": "inspect crouch; ENTER stands, X stops",
    "STAND": "rising vertically in place; wait",
    "STAND_SETTLE": "waiting for the stand to settle",
    "HOLD4": "4-leg stand; ENTER starts 2-leg cycle, P parks, X stops",
    "SHIFT": "applying body bias; wait",
    "SHIFT_SETTLE": "waiting for bias to settle",
    "WAIT_LIFT": "ready; ENTER pre-lifts the pair, X stops",
    "UNLOAD2": "pre-lifting BOTH feet of the pair; wait",
    "BALANCE": "shifting CoM onto the stance line (torque-steered)",
    "WAIT_LIFT2": "pair unloaded; ENTER lifts to full height, X stops",
    "LIFT2": "lifting the pair; wait",
    "HOLD2": "2-LEG STAND: ENTER puts down, lean/timer auto put-down",
    "LOWER2": "putting the pair back down; wait",
    "RELOAD": "feet re-taking load; wait",
    "RECENTER": "removing body bias; wait",
    "PARK": "returning to recorded crouch; wait, X stops",
    "PARKED": "holding parked crouch; X stops",
}


class TwoSequence:
    """Encoder-only stage machine for the diagonal 2-leg stand."""

    def __init__(self, now, plan, unload_trip=s3.DEFAULT_UNLOAD_TRIP_NM,
                 hold_s=DEFAULT_HOLD_S):
        self.plan = plan
        self.unload_trip = float(unload_trip)
        self.hold_s = float(hold_s)
        self.stage_i = 0
        self.stage_started = float(now)
        self.settle_since = None
        self.wait_since = None
        self.q_cmd = Q_CROUCH.copy()
        self.targets = None
        self._from = {}
        self._to = {}
        self.height_frac = 0.0
        self.body_xy = np.zeros(2)
        self.pre_m = 0.0
        self.lift_frac = 0.0
        self.prelift_m = PRE2_START_M
        self.balance_origin = None
        self.balance_last_t = None
        self.balance_shift = np.zeros(2)
        self.tilt_baseline = None
        self.last_tilt_deg = np.nan
        self.last_pair_tau_nm = np.nan
        self.last_settle_detail = ""
        self.aborted = None
        self.holds_done = 0
        self.torque_table_printed = False

    @property
    def stage(self):
        return STAGES[self.stage_i][0]

    @property
    def kind(self):
        return STAGES[self.stage_i][1]

    def _snapshot(self):
        return {"height": self.height_frac, "body": self.body_xy.copy(),
                "pre": self.pre_m, "lift": self.lift_frac}

    def _stream_endpoint(self, name):
        end = self._snapshot()
        if name == "STAND":
            end["height"] = 1.0
        elif name == "SHIFT":
            end["body"] = self.plan.bias.copy()
        elif name == "UNLOAD2":
            end["pre"] = self.prelift_m
        elif name == "LIFT2":
            end["lift"] = 1.0
        elif name == "LOWER2":
            end["pre"] = 0.0
            end["lift"] = 0.0
        elif name == "RECENTER":
            end["body"] = np.zeros(2)
        else:
            raise ValueError(name)
        return end

    def _goto(self, name, now, note=""):
        self.stage_i = STAGE_INDEX[name]
        self.stage_started = float(now)
        self.settle_since = None
        self.wait_since = None
        if self.kind == "stream":
            self._from = self._snapshot()
            self._to = self._stream_endpoint(name)
        return f"{name}: {note}" if note else name

    def _abort(self, now, reason):
        self.aborted = reason
        note = f"PUT-DOWN ({reason})"
        if self.stage in LEAN_STAGES:
            return self._goto("LOWER2", now, note)
        return self._goto("RECENTER", now, note)

    def _settled(self, now, q_enc, qd, velocity_ready):
        errors = np.abs(np.asarray(q_enc) - self.q_cmd)
        worst = int(np.argmax(errors))
        error = float(errors[worst])
        speed = float(np.max(np.abs(qd))) if velocity_ready else np.inf
        self.last_settle_detail = (
            f"worst {JOINT_LABELS[worst]} err {np.rad2deg(error):.1f} deg, "
            f"max|qd| {speed:.2f} rad/s"
        )
        if error > posbase.POSITION_POSE_TOL or speed > posbase.POSITION_QD_TOL:
            self.settle_since = None
            return False
        if self.settle_since is None:
            self.settle_since = float(now)
            return False
        return now - self.settle_since >= posbase.POSITION_SETTLE_S

    def request_next(self, now, healthy):
        if self.kind != "wait":
            return False, f"{self.stage} is busy; ENTER ignored."
        if self.stage == "PARKED":
            return False, "already PARKED; X stops."
        if self.wait_since is None or now - self.wait_since < base.WAIT_DWELL_S:
            return False, f"wait for {self.stage} to settle before ENTER."
        if not healthy:
            return False, "motor latch/fault present; motion refused."
        if self.stage == "WAIT_CROUCH":
            return True, self._goto("STAND", now, "ENTER accepted")
        if self.stage == "HOLD4":
            self.aborted = None
            self.prelift_m = PRE2_START_M
            self.balance_origin = None
            self.balance_shift = np.zeros(2)
            self.tilt_baseline = None
            self.last_tilt_deg = np.nan
            return True, self._goto(
                "SHIFT", now,
                f"2-leg cycle: lifting {'+'.join(self.plan.lifted)}, "
                f"standing on {'+'.join(self.plan.stance)}"
            )
        if self.stage == "WAIT_LIFT":
            return True, self._goto("UNLOAD2", now, "ENTER accepted")
        if self.stage == "WAIT_LIFT2":
            return True, self._goto("LIFT2", now, "ENTER accepted")
        if self.stage == "HOLD2":
            return True, self._goto("LOWER2", now, "ENTER put-down")
        return False, f"no ENTER action in {self.stage}"

    def request_park(self, now, healthy):
        if self.stage != "HOLD4":
            return False, f"P only parks from HOLD4 (now {self.stage})."
        if not healthy:
            return False, "motor latch/fault present; motion refused."
        return True, self._goto("PARK", now, "P accepted")

    def sweep(self, now, q_enc, qd, velocity_ready, tau_measured,
              rp_deg=None):
        event = None
        name, kind = STAGES[self.stage_i]
        plan = self.plan

        if name in ("SHIFT_SETTLE", "WAIT_LIFT") and rp_deg is not None:
            # last 4-leg attitude before the unload = the BALANCE
            # tilt-steering reference
            self.tilt_baseline = (float(rp_deg[0]), float(rp_deg[1]))

        if kind == "move":
            self.q_cmd = Q_CROUCH.copy()
            self.targets = None
            if self._settled(now, q_enc, qd, velocity_ready):
                nxt = {"CROUCH": "WAIT_CROUCH", "PARK": "PARKED"}[name]
                event = self._goto(nxt, now, "settled")
                self.wait_since = float(now)
            elif now - self.stage_started > s3.SETTLE_TIMEOUT_S:
                raise RuntimeError(
                    f"{name} did not settle: {self.last_settle_detail}"
                )

        elif kind == "stream":
            u = s3._smoothstep((now - self.stage_started) / STREAM_T[name])
            f, t = self._from, self._to
            self.height_frac = f["height"] + u * (t["height"] - f["height"])
            self.body_xy = f["body"] + u * (t["body"] - f["body"])
            self.pre_m = f["pre"] + u * (t["pre"] - f["pre"])
            self.lift_frac = f["lift"] + u * (t["lift"] - f["lift"])
            self.targets = plan.targets(
                self.body_xy, self.height_frac, self.pre_m, self.lift_frac
            )
            if now - self.stage_started >= STREAM_T[name]:
                nxt = {"STAND": "STAND_SETTLE", "SHIFT": "SHIFT_SETTLE",
                       "UNLOAD2": "BALANCE", "LIFT2": "HOLD2",
                       "LOWER2": "RELOAD", "RECENTER": "HOLD4"}[name]
                event = self._goto(nxt, now, "stream complete")
                if nxt == "HOLD2":
                    self.wait_since = float(now)
                    event += ("  [2-LEG STAND on "
                              f"{'+'.join(plan.stance)}: "
                              + ("hold until ENTER"
                                 if self.hold_s <= 0
                                 else f"{self.hold_s:.0f}s timer")
                              + ", lean watch armed]")
                if nxt == "HOLD4":
                    self.wait_since = float(now)
                    if self.aborted:
                        event += f"  [cycle ended early: {self.aborted}]"
                    else:
                        self.holds_done += 1
                        event += (f"  [2-LEG CYCLE {self.holds_done} "
                                  "COMPLETE: ENTER repeats, P parks]")

        elif name == "STAND_SETTLE":
            if self._settled(now, q_enc, qd, velocity_ready):
                event = self._goto("HOLD4", now, "stand settled")
                self.wait_since = float(now)
            elif now - self.stage_started > s3.SETTLE_TIMEOUT_S:
                raise RuntimeError(
                    f"STAND did not settle: {self.last_settle_detail}"
                )

        elif name == "SHIFT_SETTLE":
            if self._settled(now, q_enc, qd, velocity_ready):
                event = self._goto("WAIT_LIFT", now, "bias settled")
                self.wait_since = float(now)
            elif now - self.stage_started > s3.SETTLE_TIMEOUT_S:
                event = self._abort(now, "SHIFT did not settle")

        elif name == "BALANCE":
            if self.balance_origin is None:
                self.balance_origin = self.body_xy.copy()
                self.balance_last_t = float(now)
            taus = {}
            for leg in plan.lifted:
                sl = plan._sl(leg)
                taus[leg] = float(
                    np.max(np.abs(np.asarray(tau_measured)[sl][1:]))
                )
            self.last_pair_tau_nm = max(taus.values())
            self.balance_shift = self.body_xy - self.balance_origin
            tau_list = [taus[leg] for leg in plan.lifted]
            tau_ok = self.last_pair_tau_nm <= self.unload_trip
            # Tilt delta from the 4-leg baseline: the DOWNHILL vector in
            # body xy (FLU: +pitch = nose down -> +x is low; -roll =
            # left down -> +y is low).  A steady tilt = resting on a
            # third leg; tau alone cannot see that (hardware-proven).
            tilt = None
            if rp_deg is not None and self.tilt_baseline is not None:
                d_r = rp_deg[0] - self.tilt_baseline[0]
                d_p = rp_deg[1] - self.tilt_baseline[1]
                tilt = np.array([d_p, -d_r])
                self.last_tilt_deg = float(np.linalg.norm(tilt))
            level_ok = (self.last_tilt_deg <= TILT_BALANCED_DEG
                        if tilt is not None else True)
            if tau_ok and level_ok:
                # balanced -- hold the body here and let it settle
                if self._settled(now, q_enc, qd, velocity_ready):
                    tilt_text = ("" if tilt is None else
                                 f", tilt {self.last_tilt_deg:.1f} deg")
                    event = self._goto(
                        "WAIT_LIFT2", now,
                        f"pair free: |tau| {tau_list[0]:.2f}/"
                        f"{tau_list[1]:.2f} N*m{tilt_text} after CoM "
                        f"shift ({1000 * self.balance_shift[0]:+.1f},"
                        f"{1000 * self.balance_shift[1]:+.1f}) mm",
                    )
                    self.wait_since = float(now)
            else:
                # creep the body UPHILL (tilt steering, primary) or away
                # from the loaded lifted leg (tau fallback, no IMU)
                self.settle_since = None
                if tilt is not None and self.last_tilt_deg > 1.0e-6 \
                        and not level_ok:
                    direction = -tilt / np.linalg.norm(tilt)
                else:
                    err = sum(plan.lift_side[leg] * taus[leg]
                              for leg in plan.lifted)
                    direction = -plan.perp * float(np.sign(err))
                dt = max(0.0, now - self.balance_last_t)
                self.body_xy = self.body_xy + direction \
                    * SHIFT2_RATE_M_S * dt
                self.targets = plan.targets(
                    self.body_xy, self.height_frac, self.pre_m,
                    self.lift_frac,
                )
                if (np.linalg.norm(self.body_xy - self.balance_origin)
                        > SHIFT2_MAX_M):
                    event = self._abort(
                        now,
                        f"balance travel cap {1000 * SHIFT2_MAX_M:.0f} mm "
                        f"reached, still tilted "
                        f"{self.last_tilt_deg:.1f} deg / |tau| "
                        f"{tau_list[0]:.2f}/{tau_list[1]:.2f} N*m",
                    )
            self.balance_last_t = float(now)
            if (event is None
                    and now - self.stage_started > BALANCE_TIMEOUT_S):
                event = self._abort(
                    now,
                    f"balance timeout: tilt {self.last_tilt_deg:.1f} deg, "
                    f"|tau| {tau_list[0]:.2f}/{tau_list[1]:.2f} N*m at "
                    f"shift ({1000 * self.balance_shift[0]:+.1f},"
                    f"{1000 * self.balance_shift[1]:+.1f}) mm",
                )

        elif kind == "dwell":  # RELOAD
            if now - self.stage_started >= T_RELOAD:
                event = self._goto("RECENTER", now, "reloaded")

        elif kind == "wait":
            if self.wait_since is None:
                self.wait_since = float(now)
            if (name == "HOLD2" and self.hold_s > 0
                    and now - self.wait_since >= self.hold_s):
                event = self._goto(
                    "LOWER2", now,
                    f"held {self.hold_s:.0f} s -- putting down"
                )

        return self.q_cmd, event

    def refine_leg(self, slot_counter):
        if self.targets is None:
            return
        leg = LEGS[slot_counter % len(LEGS)]
        sl = self.plan._sl(leg)
        self.q_cmd[sl] = s3._ik_to_target(
            leg, self.q_cmd[sl], self.targets[leg],
            max_iter=s3.IK_SLOT_MAX_ITER, tol=s3.IK_SLOT_TOL_M,
        )

    def speed_cap_dps(self, caps):
        if self.stage in ("CROUCH", "WAIT_CROUCH", "PARK", "PARKED"):
            return caps["crouch"]
        if self.stage == "STAND":
            return caps["stand"]
        return caps["stream"]


def validate_configuration(plan):
    """Offline: IK/limits over the whole scripted sequence.  Raises on
    the first infeasible waypoint."""
    posbase.validate_configuration()
    low, high = base.soft_limits()
    q = Q_CROUCH.copy()
    path = (
        [(np.zeros(2), h, 0.0, 0.0) for h in np.linspace(0.0, 1.0, 9)]
        + [(u * plan.bias, 1.0, 0.0, 0.0) for u in np.linspace(0.0, 1.0, 6)]
        + [(plan.bias, 1.0, p, 0.0)
           for p in np.linspace(0.0, PRE2_MAX_M, 5)]
        + [(plan.bias, 1.0, PRE2_MAX_M, f)
           for f in np.linspace(0.0, 1.0, 5)]
        # BALANCE can carry the body up to SHIFT2_MAX_M perpendicular to
        # the stance line, at any pre-lift/lift state:
        + [(plan.bias + s * u * SHIFT2_MAX_M * plan.perp, 1.0, p, f)
           for s in (+1.0, -1.0)
           for u in (0.5, 1.0)
           for p, f in ((PRE2_START_M, 0.0), (PRE2_START_M, 1.0),
                        (PRE2_MAX_M, 1.0))]
    )
    for body, height, pre, lift in path:
        targets = plan.targets(body, height, pre, lift)
        for leg in LEGS:
            sl = plan._sl(leg)
            q[sl] = s3._ik_to_target(leg, q[sl], targets[leg])
            err = float(np.linalg.norm(
                dog5_kinematics.foot_position(leg, q[sl]) - targets[leg]
            ))
            if err > 1.0e-4:
                raise ValueError(
                    f"{leg} IK residual {err:.4f} m at body={body}, "
                    f"pre={1000 * pre:.0f}mm, lift={lift:.1f}"
                )
            if np.any(q[sl] < low[sl]) or np.any(q[sl] > high[sl]):
                raise ValueError(
                    f"{leg} outside soft limits at body={body}, "
                    f"pre={1000 * pre:.0f}mm, lift={lift:.1f}"
                )


def run_hardware(args):
    plan = TwoPlan(args.lift_pair, stand_height=args.stand_height,
                   bias_xy=(args.bias_x, args.bias_y))
    validate_configuration(plan)
    caps = {"crouch": args.crouch_dps, "stand": args.stand_dps,
            "stream": args.stream_dps}
    unwrap = [base.CalibratedEncoderUnwrap() for _ in base.HARDWARE_JOINTS]
    safety = base.SafetyGate(
        tau_cap=1.0, qd_estop=args.qd_estop, qd_estop_hard=args.qd_estop_hard
    )

    print(f"[2leg] DIAGONAL 2-LEG STAND: lift {'+'.join(plan.lifted)}, "
          f"stand on {'+'.join(plan.stance)}; pre-lift "
          f"{1000 * PRE2_START_M:.0f}->{1000 * PRE2_MAX_M:.0f} mm ladder, "
          f"lift {1000 * LIFT2_M:.0f} mm, hold "
          + ("until ENTER" if args.hold_s <= 0 else f"{args.hold_s:.0f} s"))
    if abs(args.bias_x) > 0 or abs(args.bias_y) > 0:
        print(f"[2leg] body bias ({1000 * args.bias_x:+.0f},"
              f"{1000 * args.bias_y:+.0f}) mm before the lift")
    if frozenset(plan.lifted) == frozenset(("FR", "RL")):
        print("[2leg] WARNING: standing on FL+RR was shown NOT to work "
              "(2026-07-23): the CoM sits left of that diagonal -- the "
              "body rolled -8.5 deg onto RL and only LOOKED 2-legged "
              "(tau gate false-passed).  Expect the same unless you add "
              "a strong bias.")
    print("[2leg] 2-leg support is statically unstable -- it balances on "
          "compliance, not control.  ROBOT MUST BE SUPPERVISED, hands "
          "ready; X stops, ENTER in HOLD2 puts down immediately.")

    imu = None if args.no_imu else inplace._open_imu()
    if imu is None and not args.no_imu:
        raise RuntimeError(
            "IMU required for the 2-leg stand (lean watch is the only "
            "tip warning); pass --no-imu to override deliberately"
        )
    if imu is not None:
        print(f"[imu] tip-over {inplace.TIPOVER_ABORT_DEG:.0f} deg "
              f"run-stop; lean put-down {args.lean_abort_deg:.0f} deg "
              f"(delta from lift baseline) in {'/'.join(LEAN_STAGES)}")
    else:
        print("[2leg] WARNING: running WITHOUT the IMU lean watch.")

    key = base.KeyPoller()
    try:
        with base.motorbus.MotorBus(MOTOR_IDS, dirs=MOTOR_DIRECTIONS) as mb:
            armed = False
            stop_reason = None
            try:
                print(f"[ARM] arming for up to {posbase.ARM_TIMEOUT_S:.0f}s")
                if not mb.arm(rate_hz=base.CONTROL_HZ,
                              timeout_s=posbase.ARM_TIMEOUT_S, verbose=False):
                    raise RuntimeError(
                        "arming timed out: " + posbase.arm_failure_summary(mb)
                    )
                armed = True
                start_q = posbase.zero_torque_preflight(mb, key, unwrap)
                now = time.perf_counter()
                sequence = TwoSequence(now, plan,
                                       unload_trip=args.unload_trip,
                                       hold_s=args.hold_s)
                safety.start(now, start_q)

                slot = mb.slot(base.CONTROL_HZ)
                deadline = time.perf_counter() + slot
                run_started_at = now
                q = start_q.copy()
                miss_monitor = base.CanMissMonitor(mb)
                status_period = 1.0 / base.FAULT_STATUS_HZ
                next_fault_status = np.asarray([
                    status_period + i * status_period / N_JOINTS
                    for i in range(N_JOINTS)
                ])
                last_recover = {
                    mid: -base.RECOVER_PERIOD_S for mid in MOTOR_IDS
                }
                last_print = 0.0
                index = 0
                velocity = posbase.EncoderVelocity()
                qd_encoder = np.zeros(N_JOINTS)
                worst_block_ms = 0.0
                tipover_streak = 0
                lean_baseline = None
                lean_streak = 0
                imu_sample = None

                while True:
                    mb.poll()
                    joint_index = index % N_JOINTS

                    if joint_index == 0:
                        now = time.perf_counter()
                        elapsed = now - run_started_at
                        q, qd_driver = base._joint_state(mb, unwrap)
                        qd_encoder = velocity.update(q, now)
                        torque_feedback = mb.torques_nm()
                        measured_torque = np.asarray(
                            [torque_feedback[mid] for mid in MOTOR_IDS]
                        )

                        imu_sample = None
                        imu_fresh = False
                        if imu is not None:
                            try:
                                imu_sample = imu.sample()
                            except Exception:
                                imu_sample = None
                            imu_fresh = (
                                imu_sample is not None
                                and imu_sample.age_s <= inplace.POSTURE_STALE_S
                            )
                        rp_deg = (
                            (imu_sample.roll_deg, imu_sample.pitch_deg)
                            if imu_fresh else None
                        )

                        _, event = sequence.sweep(
                            now, q, qd_encoder, velocity.ready,
                            measured_torque, rp_deg=rp_deg,
                        )
                        if event:
                            print(f"[stage] {event}")
                            if (sequence.stage == "HOLD2"
                                    and lean_baseline is not None
                                    and imu_sample is not None):
                                d_r = imu_sample.roll_deg - lean_baseline[0]
                                d_p = imu_sample.pitch_deg - lean_baseline[1]
                                verdict = (
                                    "large steady lean = resting on a "
                                    "THIRD leg, not a 2-leg stand; bias "
                                    "toward the high side and retry"
                                    if max(abs(d_r), abs(d_p)) > 5.0 else
                                    "small lean -- plausibly a true "
                                    "2-leg stand; watch for wander"
                                )
                                print(f"[2leg] settle check: lean since "
                                      f"pre-lift d_roll={d_r:+.1f} "
                                      f"d_pitch={d_p:+.1f} deg -> {verdict}")
                        if (sequence.stage == "HOLD4"
                                and not sequence.torque_table_printed):
                            s3._print_torque_table(measured_torque)
                            sequence.torque_table_printed = True

                        temps = base._temperatures(mb)
                        misses = miss_monitor.update(mb)
                        errors = mb.errors()
                        stop_reason = safety.estop_reason(
                            q, qd_driver, temps, misses, errors, now,
                            enforce_position_limits=(
                                sequence.stage != "CROUCH"
                            ),
                        )
                        if stop_reason is None:
                            stop_reason = posbase.position_fault(
                                qd_encoder, measured_torque, velocity.ready,
                                args.speed_trip, args.torque_trip,
                            )

                        if imu is not None:
                            if imu_fresh:
                                roll = imu_sample.roll_deg
                                pitch = imu_sample.pitch_deg
                                if (abs(roll) > inplace.TIPOVER_ABORT_DEG
                                        or abs(pitch)
                                        > inplace.TIPOVER_ABORT_DEG):
                                    tipover_streak += 1
                                    if (stop_reason is None
                                            and tipover_streak
                                            >= inplace.TIPOVER_CONFIRM_SAMPLES):
                                        stop_reason = (
                                            f"IMU tip-over: roll={roll:+.1f} "
                                            f"pitch={pitch:+.1f} deg"
                                        )
                                else:
                                    tipover_streak = 0
                                if sequence.stage in LEAN_STAGES:
                                    if lean_baseline is None:
                                        lean_baseline = (roll, pitch)
                                        lean_streak = 0
                                    else:
                                        d_r = roll - lean_baseline[0]
                                        d_p = pitch - lean_baseline[1]
                                        if max(abs(d_r), abs(d_p)) \
                                                > args.lean_abort_deg:
                                            lean_streak += 1
                                            if lean_streak >= s3.LEAN_STREAK:
                                                print("[stage] " +
                                                      sequence._abort(
                                                          now,
                                                          f"IMU lean d_roll="
                                                          f"{d_r:+.1f} "
                                                          f"d_pitch="
                                                          f"{d_p:+.1f} deg"))
                                                lean_baseline = None
                                                lean_streak = 0
                                        else:
                                            lean_streak = 0
                                else:
                                    lean_baseline = None
                                    lean_streak = 0

                        latched = [
                            mid for mid, error in errors.items()
                            if error & 0x80
                        ]
                        unverified = [
                            mid for mid in MOTOR_IDS
                            if mb.rec(mid).error is None
                        ]
                        pressed = key.get()
                        healthy = (not latched and not unverified
                                   and stop_reason is None)
                        if pressed in ("x", "X"):
                            stop_reason = "operator X"
                        elif pressed in ("p", "P"):
                            _, message = sequence.request_park(now, healthy)
                            print(f"[stage] {message}")
                        elif base._is_enter(pressed):
                            _, message = sequence.request_next(now, healthy)
                            print(f"[stage] {message}")
                        if stop_reason:
                            break

                        recover = [
                            mid for mid in latched
                            if elapsed - last_recover[mid]
                            >= base.RECOVER_PERIOD_S
                        ]
                        if recover:
                            base._recover_input_lost(
                                mb, recover, elapsed, last_recover,
                                next_fault_status,
                            )
                            latched = []

                        if now - last_print >= 1.0 / base.STATUS_HZ:
                            taus = [
                                float(np.max(np.abs(
                                    measured_torque[plan._sl(leg)][1:])))
                                for leg in plan.lifted
                            ]
                            hold_text = ""
                            if (sequence.stage == "HOLD2"
                                    and sequence.hold_s > 0
                                    and sequence.wait_since is not None):
                                left = max(0.0, sequence.hold_s
                                           - (now - sequence.wait_since))
                                hold_text = f" hold_left={left:.1f}s"
                            if (sequence.stage == "BALANCE"
                                    and np.isfinite(sequence.last_tilt_deg)):
                                hold_text += (
                                    f" tilt={sequence.last_tilt_deg:.1f}deg"
                                )
                            imu_text = ""
                            if imu_sample is not None:
                                imu_text = (
                                    f" rp=({imu_sample.roll_deg:+.2f},"
                                    f"{imu_sample.pitch_deg:+.2f})deg"
                                )
                            print(
                                f"[2leg] {sequence.stage:<12} "
                                f"{INSTRUCTION[sequence.stage]} | "
                                f"pair|tau|={taus[0]:.2f}/{taus[1]:.2f} "
                                f"body=({1000 * sequence.body_xy[0]:+.0f},"
                                f"{1000 * sequence.body_xy[1]:+.0f})mm "
                                f"max|tau|="
                                f"{np.max(np.abs(measured_torque)):.2f}N*m "
                                f"blk={worst_block_ms:.1f}ms"
                                f"{hold_text}{imu_text}",
                                flush=True,
                            )
                            last_print = now
                            worst_block_ms = 0.0
                        worst_block_ms = max(
                            worst_block_ms,
                            1000.0 * (time.perf_counter() - now),
                        )

                    if index % s3.IK_SLOT_STRIDE == 0:
                        sequence.refine_leg(index // s3.IK_SLOT_STRIDE)

                    mid = MOTOR_IDS[joint_index]
                    elapsed = time.perf_counter() - run_started_at
                    if elapsed >= next_fault_status[joint_index]:
                        mb.status1_req(mid)
                        next_fault_status[joint_index] += status_period
                    else:
                        if not mb.position(
                            mid,
                            float(np.rad2deg(sequence.q_cmd[joint_index])),
                            sequence.speed_cap_dps(caps),
                        ):
                            raise RuntimeError(
                                f"CAN position transmit failed for CAN {mid}"
                            )

                    index += 1
                    mb.pace(deadline)
                    deadline += slot

            except KeyboardInterrupt as exc:
                stop_reason = str(exc) or "KeyboardInterrupt"
            except Exception as exc:
                stop_reason = f"error: {exc}"
                raise
            finally:
                if armed:
                    print(f"[STOP] {stop_reason or 'aborted'} "
                          f"(2-leg holds completed: "
                          f"{sequence.holds_done if 'sequence' in dir() else 0})")
                    mb.stop_all()
    finally:
        key.close()
        if imu is not None:
            try:
                imu.stop()
            except Exception:
                pass
    return 0


def offline_self_test(args):
    plan = TwoPlan(args.lift_pair, stand_height=args.stand_height,
                   bias_xy=(args.bias_x, args.bias_y))
    validate_configuration(plan)
    print(f"sequence IK-feasible: lift {'+'.join(plan.lifted)}, bias "
          f"({1000 * args.bias_x:+.0f},{1000 * args.bias_y:+.0f}) mm, "
          f"raise to {1000 * (PRE2_MAX_M + LIFT2_M):.0f} mm")

    seq = TwoSequence(0.0, plan, unload_trip=args.unload_trip, hold_s=2.0)
    now = 0.0

    def spin(stop_stages, limit_s=90.0, tau=None):
        nonlocal now
        deadline = now + limit_s
        while seq.stage not in stop_stages:
            now += 0.048
            tt = np.zeros(N_JOINTS) if tau is None else tau
            seq.sweep(now, seq.q_cmd.copy(), np.zeros(N_JOINTS), True, tt)
            for k in range(4):
                seq.refine_leg(k)
            assert now < deadline, f"stuck in {seq.stage}"

    def enter():
        nonlocal now
        now += base.WAIT_DWELL_S + 0.1
        ok, msg = seq.request_next(now, True)
        assert ok, msg

    spin(("WAIT_CROUCH",))
    enter()
    spin(("HOLD4",))
    enter()                          # HOLD4 -> SHIFT
    spin(("WAIT_LIFT",))
    enter()                          # -> UNLOAD2
    spin(("WAIT_LIFT2",))
    enter()                          # -> LIFT2 -> HOLD2 (2s) -> down
    spin(("HOLD4",))
    assert seq.aborted is None, seq.aborted
    assert seq.holds_done == 1
    print("dry run: full 2-leg cycle (timer put-down) complete")

    # BALANCE closed loop: a lifted leg stays loaded until the body has
    # shifted 8 mm away from its side -- the stage must steer there.
    loaded_leg = plan.lifted[0]
    away = -plan.lift_side[loaded_leg] * plan.perp

    def tau_for(seq_obj):
        tt = np.zeros(N_JOINTS)
        moved = 0.0
        if seq_obj.balance_origin is not None:
            moved = float(np.dot(
                seq_obj.body_xy - seq_obj.balance_origin, away))
        sl = plan._sl(loaded_leg)
        tt[sl.start + 2] = 0.2 if moved >= 0.008 else 2.0
        return tt

    seq2 = TwoSequence(0.0, plan, unload_trip=args.unload_trip, hold_s=2.0)
    now = 0.0

    def spin2(stop_stages, limit_s=200.0, tau_fn=None):
        nonlocal now
        deadline = now + limit_s
        while seq2.stage not in stop_stages:
            now += 0.048
            tt = (np.zeros(N_JOINTS) if tau_fn is None
                  else tau_fn(seq2))
            seq2.sweep(now, seq2.q_cmd.copy(), np.zeros(N_JOINTS), True, tt)
            for k in range(4):
                seq2.refine_leg(k)
            assert now < deadline, f"stuck in {seq2.stage}"

    spin2(("WAIT_CROUCH",))
    now += base.WAIT_DWELL_S + 0.1
    seq2.request_next(now, True)
    spin2(("HOLD4",))
    now += base.WAIT_DWELL_S + 0.1
    seq2.request_next(now, True)
    spin2(("WAIT_LIFT",))
    now += base.WAIT_DWELL_S + 0.1
    seq2.request_next(now, True)
    spin2(("WAIT_LIFT2",), tau_fn=tau_for)
    moved = float(np.dot(seq2.body_xy - seq2.balance_origin, away))
    assert moved >= 0.008 - 1.0e-6, moved
    assert seq2.aborted is None
    now += base.WAIT_DWELL_S + 0.1
    seq2.request_next(now, True)
    spin2(("HOLD4",), tau_fn=tau_for)   # LIFT2 -> HOLD2 (2s) -> down
    assert seq2.holds_done == 1
    print(f"balance loop: steered {1000 * moved:.1f} mm off {loaded_leg}'s "
          "side, pair freed, full cycle complete")

    # Never balances (one leg loaded no matter how far the body goes)
    # -> travel cap -> graceful put-down, cycle not counted.
    seq2 = TwoSequence(0.0, plan, unload_trip=args.unload_trip, hold_s=2.0)
    now = 0.0
    heavy = np.zeros(N_JOINTS)
    heavy[plan._sl(loaded_leg).start + 2] = 2.0
    spin2(("WAIT_CROUCH",))
    now += base.WAIT_DWELL_S + 0.1
    seq2.request_next(now, True)
    spin2(("HOLD4",))
    now += base.WAIT_DWELL_S + 0.1
    seq2.request_next(now, True)
    spin2(("WAIT_LIFT",))
    now += base.WAIT_DWELL_S + 0.1
    seq2.request_next(now, True)
    spin2(("HOLD4",), tau_fn=lambda s: heavy)
    assert seq2.aborted is not None and "cap" in seq2.aborted
    assert seq2.holds_done == 0
    print("never-balances: travel cap hit, graceful put-down, "
          "cycle not counted")

    # THE hardware failure mode (run 3): tau reads LOW (blind) while the
    # body rests tilted on a third leg.  Tau-only logic exits with zero
    # shift; tilt steering must keep creeping uphill until level.
    tseq = TwoSequence(0.0, plan, unload_trip=args.unload_trip, hold_s=2.0)
    now = 0.0

    def rp_for(s):
        # left tip (roll -2.5) until the body has moved 9 mm uphill (-y)
        if s.stage in ("UNLOAD2", "BALANCE") and -s.body_xy[1] < 0.009:
            return (-2.5, 0.0)
        return (0.0, 0.0)

    def spint(stop_stages, limit_s=200.0):
        nonlocal now
        deadline = now + limit_s
        while tseq.stage not in stop_stages:
            now += 0.048
            tseq.sweep(now, tseq.q_cmd.copy(), np.zeros(N_JOINTS), True,
                       np.zeros(N_JOINTS), rp_deg=rp_for(tseq))
            for k in range(4):
                tseq.refine_leg(k)
            assert now < deadline, f"stuck in {tseq.stage}"

    spint(("WAIT_CROUCH",))
    now += base.WAIT_DWELL_S + 0.1
    tseq.request_next(now, True)
    spint(("HOLD4",))
    now += base.WAIT_DWELL_S + 0.1
    tseq.request_next(now, True)
    spint(("WAIT_LIFT",))
    now += base.WAIT_DWELL_S + 0.1
    tseq.request_next(now, True)
    spint(("WAIT_LIFT2",))
    assert tseq.body_xy[1] <= -0.009 + 1.0e-6, tseq.body_xy
    assert tseq.aborted is None
    print(f"tau-blind tilt scenario: tau read 'free' throughout, tilt "
          f"steering still shifted {-1000 * tseq.body_xy[1]:.1f} mm "
          "uphill before declaring balance")

    # Lean abort from HOLD2 routes through LOWER2 and recovers to HOLD4.
    seq3 = TwoSequence(0.0, plan, unload_trip=args.unload_trip, hold_s=0.0)
    now = 0.0

    def spin3(stop_stages, limit_s=90.0):
        nonlocal now
        deadline = now + limit_s
        while seq3.stage not in stop_stages:
            now += 0.048
            seq3.sweep(now, seq3.q_cmd.copy(), np.zeros(N_JOINTS), True,
                       np.zeros(N_JOINTS))
            for k in range(4):
                seq3.refine_leg(k)
            assert now < deadline, f"stuck in {seq3.stage}"

    spin3(("WAIT_CROUCH",))
    now += base.WAIT_DWELL_S + 0.1
    seq3.request_next(now, True)
    spin3(("HOLD4",))
    now += base.WAIT_DWELL_S + 0.1
    seq3.request_next(now, True)
    spin3(("WAIT_LIFT",))
    now += base.WAIT_DWELL_S + 0.1
    seq3.request_next(now, True)
    spin3(("WAIT_LIFT2",))
    now += base.WAIT_DWELL_S + 0.1
    seq3.request_next(now, True)
    spin3(("HOLD2",))
    print("[synthetic lean] " + seq3._abort(now, "test lean"))
    assert seq3.stage == "LOWER2"
    spin3(("HOLD4",))
    assert seq3.aborted is not None
    assert seq3.holds_done == 0
    print("lean put-down from HOLD2: lowered, reloaded, recentered")
    print("twostand_hw offline self-test PASS")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--lift", type=str, default=",".join(DEFAULT_LIFT_PAIR),
        help="the diagonal PAIR to lift (default FL,RR -- stands on "
             "FR+RL, the diagonal the CoM actually sits near; FR,RL "
             "was hardware-shown to rest on a third leg instead)")
    parser.add_argument(
        "--hold-s", type=float, default=DEFAULT_HOLD_S,
        help="2-leg hold time before auto put-down (0 = until ENTER)")
    parser.add_argument(
        "--bias-x", type=float, default=0.0,
        help="body x offset (m) applied before the lift (+ = forward)")
    parser.add_argument(
        "--bias-y", type=float, default=0.0,
        help="body y offset (m) applied before the lift (+ = left)")
    parser.add_argument(
        "--lean-abort-deg", type=float, default=DEFAULT_LEAN_ABORT_DEG,
        help="lean delta (deg from lift baseline) that puts the feet "
             "back down")
    parser.add_argument("--stand-height", type=float,
                        default=DEFAULT_STAND_HEIGHT_M)
    parser.add_argument("--torque-trip", type=float,
                        default=s3.DEFAULT_TORQUE_TRIP_NM)
    parser.add_argument("--speed-trip", type=float,
                        default=s3.DEFAULT_SPEED_TRIP_RAD_S)
    parser.add_argument("--unload-trip", type=float,
                        default=s3.DEFAULT_UNLOAD_TRIP_NM)
    parser.add_argument("--crouch-dps", type=float,
                        default=s3.DEFAULT_CROUCH_DPS)
    parser.add_argument("--stand-dps", type=float,
                        default=s3.DEFAULT_STAND_DPS)
    parser.add_argument("--stream-dps", type=float,
                        default=s3.DEFAULT_STREAM_DPS)
    parser.add_argument("--qd-estop", type=float, default=base.QD_ESTOP)
    parser.add_argument("--qd-estop-hard", type=float,
                        default=base.QD_ESTOP_HARD)
    parser.add_argument("--no-imu", action="store_true")
    args = parser.parse_args()

    pair = tuple(s.strip().upper() for s in args.lift.split(",") if s.strip())
    if frozenset(pair) not in DIAGONALS or len(pair) != 2:
        parser.error(f"--lift must be a DIAGONAL pair (FR,RL or FL,RR); "
                     f"got {args.lift!r} -- a same-side pair cannot stand")
    args.lift_pair = pair
    if not 0.0 < args.hold_s <= 60.0 and args.hold_s != 0.0:
        parser.error("--hold-s must be 0 (until ENTER) or 0-60 s")
    if abs(args.bias_x) > BIAS_MAX_M or abs(args.bias_y) > BIAS_MAX_M:
        parser.error(f"--bias-x/--bias-y limited to +-{BIAS_MAX_M} m")
    if not 2.0 <= args.lean_abort_deg <= inplace.TIPOVER_ABORT_DEG:
        parser.error("--lean-abort-deg must be between 2 and the "
                     f"{inplace.TIPOVER_ABORT_DEG:.0f} deg tip-over")

    if args.self_test:
        return offline_self_test(args)
    try:
        return run_hardware(args)
    except (RuntimeError, ValueError) as exc:
        print(f"[2leg] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
