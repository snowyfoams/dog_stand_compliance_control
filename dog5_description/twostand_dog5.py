#!/usr/bin/env python3
r"""Two-leg DIAGONAL stand for DOG5 in MuJoCo -- real2sim port of
``crawl_hw2.0/twostand_hw.py`` (branch ``dog5-live-mirror``, 2026-07-23).

Today's hardware session (IMU freshly assembled and wired) could not hold a
2-leg stand: the pre-lift raises both feet of a diagonal pair, but the
BALANCE creep only ever frees ONE of the two -- the other stays loaded on
the ground no matter how long it waits.  This is the mirror-image of the
earlier 3-leg work: that time sim proved the approach *before* hardware;
this time hardware found a failure and sim is where it gets diagnosed
before the next hardware attempt.

Everything reused is the proven real2sim discipline from ``stand3_dog5.py``:
motion is native-position-command style only (joint targets + motor-side
speed caps, no torque law); the only plant read is joint position (via
``NativePositionServo``, standing in for the real 0xA4 driver + encoder);
all FK/IK/Jacobians come from the NumPy ``dog5_kinematics`` module; trunk
pose, contact forces and true CoM are read only by the ``Oracle``, which
judges the run and never steers it.

New in this file, because the hardware BALANCE stage is not encoder-only:
it also reads the IMU (roll/pitch tilt), a legitimate physical sensor, not
a privileged one -- ``SimIMU`` reproduces it from the trunk's orientation.
The sign convention (roll = atan2(R[2,1], R[2,2]), pitch =
atan2(-R[2,0], hypot(R[2,1], R[2,2]))) is derived to match
``twostand_hw.py``'s own comments ("+pitch = nose down -> +x is low",
"-roll = left down -> +y is low"); ``self_test`` checks it algebraically.

BALANCE status (2026-07-23, this file): the original hardware law -- creep
the body horizontally, sign-of-tilt direction, fixed rate -- was verified
in sim to be a genuine limit cycle (still hadn't converged after 90s at
3.3x slower gain).  It was replaced here with a PER-LEG VERTICAL trim
(same PI structure as the hardware-proven walk `--imu-level` stance trim:
filtered tilt, deadband, Ki integration, clamp -- but raising, never
lowering, each lifted foot's own pre-lift).  That fix is real and
validated: it removed the oscillation and lets one lifted leg (RR, for
the default pair) reach genuine, sustained ground clearance.  It is NOT
sufficient by itself, though -- once that leg is genuinely airborne the
robot is standing on a true 2-point line and starts to physically ROTATE
about it under gravity if the CoM isn't exactly on that line; a vertical
trim on the other lifted leg (FL) cannot counter a rigid-body tip it isn't
touching anything to resist.  Closing that gap needs a horizontal CoM
correction too -- properly damped this time, not the bang-bang law that
oscillated -- which is not implemented here.  Treat this file as: vertical
half solved, horizontal half still open, before the next hardware attempt.

CONTROL RATE (corrected 2026-08-10): every runner used to command ONE motor
per 250 Hz tick, so each joint was really refreshed at 250/12 = 20.8 Hz --
12x slower than the robot.  A tick is now a whole 12-motor CAN sweep and
the stage machine runs at a true 250 Hz (``--can-roundrobin`` restores the
old model).  Re-running this file at the corrected rate changes nothing
that matters: RR still reaches 23.5 mm of clean clearance while FL sits at
-0.4 mm under 9.4 N, BALANCE still times out at 7.2 deg, and --sweep is
still 0/9.  The diagnosis above is a rigid-body limit, not a bandwidth one.

Run with the project venv:
    D:\mujoco\.venv\Scripts\python.exe twostand_dog5.py              # viewer
    D:\mujoco\.venv\Scripts\python.exe twostand_dog5.py --self-test  # offline checks
    D:\mujoco\.venv\Scripts\python.exe twostand_dog5.py --headless   # metrics + verdict
    D:\mujoco\.venv\Scripts\python.exe twostand_dog5.py --sweep      # diagonal x bias grid
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import dog5_kinematics                 # noqa: E402
import stand3_dog5 as base             # noqa: E402  (real2sim primitives)

LEGS = base.LEGS
N_JOINTS = base.N_JOINTS
JOINT_LABELS = base.JOINT_LABELS
CONTROL_HZ = base.CONTROL_HZ            # per-motor rate; see base's CAN budget
SWEEP_DT = 1.0 / CONTROL_HZ             # one whole-bus sweep per control tick
STATUS_HZ = base.STATUS_HZ
Q_CROUCH = base.Q_CROUCH
KP_SERVO = base.KP_SERVO
KD_SERVO = base.KD_SERVO
NativePositionServo = base.NativePositionServo
_smoothstep = base._smoothstep
_ik_to_target = base._ik_to_target
soft_limits = base.soft_limits
build_sim = base.build_sim
POSITION_POSE_TOL = base.POSITION_POSE_TOL
POSITION_QD_TOL = base.POSITION_QD_TOL
POSITION_SETTLE_S = base.POSITION_SETTLE_S

DIAGONALS = (frozenset(("FL", "RR")), frozenset(("FR", "RL")))
DEFAULT_LIFT_PAIR = ("FL", "RR")        # stands on FR+RL -- the diagonal the
                                        # CoM actually sits near (hw 2026-07-23)

# ---- crawl_hw2.0 in-place stance (superseded the sprawl stand3_dog5 used) --
DEFAULT_STAND_HEIGHT_M = 0.175
T_STAND = 8.0
T_SHIFT = 2.5
T_UNLOAD = 0.8
T_LIFT = 1.5
T_LOWER2 = 1.5
T_RELOAD = 1.0
HOLD4_S = 1.0
DONE_S = 0.5
SETTLE_TIMEOUT_S = 30.0

PRE2_START_M = 0.015            # hw baseline, kept deliberately gentle: a
                                # shared 30mm synchronous pre-lift on BOTH
                                # legs (tested 2026-07-23) tips the trunk
                                # fast enough (tilt already -6deg the
                                # instant UNLOAD2 completes) to trip the
                                # lean-abort before BALANCE's PI trim gets
                                # a chance to respond.  Let the per-leg
                                # trim below climb from a safe start instead
                                # of starting both legs high and hoping.
LIFT2_M = 0.014
DEFAULT_HOLD_S = 10.0
DEFAULT_LEAN_ABORT_DEG = 8.0
BIAS_MAX_M = 0.020

# BALANCE (redesigned 2026-07-23): the original port's horizontal bang-bang
# CoM creep (sign-of-tilt direction, fixed rate) was verified in sim to be a
# genuine limit cycle -- it still hadn't converged after 90s at 3.3x slower
# gain, always ending back near zero net shift.  It also can't explain why
# raising pre-lift consistently freed ONE lifted leg (RR) while leaving the
# other (FL) stuck at ~-0.4mm clearance regardless of pre-lift amount (0.03,
# 0.04, 0.05m all IK-feasible, nowhere near a joint limit) -- that is a
# per-leg sag asymmetry, not something a single shared body_xy offset can
# fix.  This replaces the horizontal creep with a PER-LEG vertical trim,
# reusing the exact PI structure already hardware-proven for the walk's
# `--imu-level` stance trim (work_review.md Step 2): filtered tilt, a
# deadband, per-leg error from that leg's trunk-frame xy, Ki integration,
# a clamp -- except here it raises (never lowers) each lifted foot's own
# pre-lift, directly compensating for "this specific corner won't clear."
BALANCE_TRIM_KI = 0.25          # 1/s, same gain as the proven walk trim
BALANCE_TRIM_CLAMP_M = 0.015    # max extra lift a single leg can accrue
BALANCE_TILT_DEADBAND_DEG = 0.2
BALANCE_TILT_LPF_TAU_S = 0.135  # tilt LPF time constant.  Was a fixed
                                # per-sweep alpha of 0.3, which only meant
                                # 0.135 s while the sweep rate was (wrongly)
                                # 20.8 Hz.  Held as a time constant so the
                                # sim-tuned BALANCE law is unchanged by the
                                # move to a true 250 Hz per-motor rate --
                                # the command rate is the only variable.
BALANCE_TIMEOUT_S = 25.0
TILT_BALANCED_DEG = 1.2

DEFAULT_TORQUE_TRIP_NM = 6.0
DEFAULT_UNLOAD_TRIP_NM = 0.7
DEFAULT_CROUCH_DPS = 100.0
DEFAULT_STAND_DPS = 100.0
DEFAULT_STREAM_DPS = 250.0

TIPOVER_ABORT_DEG = 25.0
# Confirmation windows in SECONDS, not sweeps: at the old (wrong) 20.8 Hz
# sweep rate "3 samples" was 0.144 s; at a true 250 Hz it would be 0.012 s,
# which would make both aborts 12x twitchier for reasons that have nothing
# to do with the control rate.  Holding the wall time keeps them comparable.
TIPOVER_CONFIRM_S = 0.144
LEAN_CONFIRM_S = 0.144


def _sl(leg):
    i = LEGS.index(leg)
    return slice(3 * i, 3 * i + 3)


def _ik_all(q_warm, targets):
    q = np.asarray(q_warm, dtype=float).copy()
    for leg in LEGS:
        sl = _sl(leg)
        q[sl] = _ik_to_target(leg, q[sl], targets[leg])
    return q


class TwoPlan:
    """Feet fixed at crouch xy; body optionally biased; one diagonal pair
    gets vertical pre-lift/lift offsets.  Direct port of
    ``crawl_hw2.0/twostand_hw.py::TwoPlan``."""

    def __init__(self, lift_pair, stand_height=DEFAULT_STAND_HEIGHT_M,
                 bias_xy=(0.0, 0.0)):
        self.lifted = tuple(lift_pair)
        self.stance = tuple(leg for leg in LEGS if leg not in self.lifted)
        self.stand_height = float(stand_height)
        self.bias = np.array(bias_xy, dtype=float)
        self.crouch_foot = {
            leg: dog5_kinematics.foot_position(leg, Q_CROUCH[_sl(leg)])
            for leg in LEGS
        }
        p0 = self.crouch_foot[self.stance[0]][:2]
        p1 = self.crouch_foot[self.stance[1]][:2]
        t = (p1 - p0) / np.linalg.norm(p1 - p0)
        self.perp = np.array([-t[1], t[0]])   # used only by Oracle.line_margin

    def targets(self, body_xy, height_frac, pre_m, lift_frac, trim=None):
        trim = trim or {}
        out = {}
        for leg in LEGS:
            xy = self.crouch_foot[leg][:2]
            crouch_z = self.crouch_foot[leg][2]
            z = crouch_z + height_frac * (-self.stand_height - crouch_z)
            if leg in self.lifted:
                z += pre_m + lift_frac * LIFT2_M + trim.get(leg, 0.0)
            out[leg] = np.array([xy[0] - body_xy[0],
                                 xy[1] - body_xy[1], z])
        return out


# stage -> kind.  "move" = fixed joint pose w/ settle gate; "stream" =
# interpolated plan targets; "gate" = hold targets until a condition passes;
# "dwell" = timed hold.  No operator ENTER in sim -- auto-advance, mirroring
# stand3_dog5.py's own simplification of the hardware wait-stages.
STAGES = [
    ("CROUCH", "move"), ("STAND", "stream"), ("STAND_SETTLE", "gate"),
    ("HOLD4", "dwell"), ("SHIFT", "stream"), ("SHIFT_SETTLE", "gate"),
    ("UNLOAD2", "stream"), ("BALANCE", "gate"),
    ("LIFT2", "stream"), ("HOLD2", "dwell"),
    ("LOWER2", "stream"), ("RELOAD", "dwell"), ("RECENTER", "stream"),
    ("DONE", "dwell"),
]
STAGE_INDEX = {name: i for i, (name, _) in enumerate(STAGES)}
STREAM_T = {"STAND": T_STAND, "SHIFT": T_SHIFT, "UNLOAD2": T_UNLOAD,
            "LIFT2": T_LIFT, "LOWER2": T_LOWER2, "RECENTER": T_SHIFT}
LEAN_STAGES = ("UNLOAD2", "BALANCE", "LIFT2", "HOLD2")


def _roll_pitch_deg(R):
    """Trunk roll/pitch from a body->world rotation matrix R (FLU frame).

    Matches the sign convention documented in twostand_hw.py's BALANCE
    comments: +roll = right side down, +pitch = nose down.  Derivation:
    R[:, 1] is the body's +y (left) axis expressed in world coordinates, so
    R[2, 1] is that axis's world-z component; when the left side is
    physically lower, R[2, 1] < 0, and atan2(R[2,1], R[2,2]) < 0 to match
    -- i.e. "-roll = left down".  Symmetric argument for pitch about x.
    """
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))
    return np.degrees(roll), np.degrees(pitch)


class SimIMU:
    """Trunk roll/pitch sensor standing in for the physical IMU.

    A physical sensor, not a privileged Oracle read -- the real BALANCE
    stage legitimately consumes IMU tilt (twostand_hw.py).  Optional bias
    models an uncalibrated mounting offset; optional noise models sensor
    jitter.  The IMU-absent / stale-sample fallback path is reproduced at
    the call site via ``--no-imu`` (``run()`` passes ``rp_deg=None``),
    matching how twostand_hw.py drops to tau-only steering when the
    sample is stale.
    """

    def __init__(self, model, data, trunk_body, noise_std_deg=0.0,
                 bias_deg=(0.0, 0.0)):
        self.m, self.d = model, data
        self.trunk = trunk_body
        self.noise_std = np.deg2rad(noise_std_deg)
        self.bias = np.deg2rad(np.asarray(bias_deg, dtype=float))

    def sample(self):
        R = self.d.xmat[self.trunk].reshape(3, 3)
        roll, pitch = _roll_pitch_deg(R)
        roll += np.degrees(self.bias[0])
        pitch += np.degrees(self.bias[1])
        if self.noise_std > 0:
            roll += np.degrees(np.random.normal(0.0, self.noise_std))
            pitch += np.degrees(np.random.normal(0.0, self.noise_std))
        return roll, pitch


class TwoSequence:
    """Encoder(+IMU)-only stage machine for the diagonal 2-leg stand.

    Pure logic (no MuJoCo, no wall-clock source of its own) so the offline
    self-test can drive it with synthetic telemetry.  Port of
    ``crawl_hw2.0/twostand_hw.py::TwoSequence`` with the operator-gated
    WAIT_* stages collapsed into auto-advancing dwells/gates, matching how
    ``stand3_dog5.py`` already simplifies its hardware counterpart.
    """

    def __init__(self, now, plan, unload_trip=DEFAULT_UNLOAD_TRIP_NM,
                 hold_s=DEFAULT_HOLD_S):
        self.plan = plan
        self.unload_trip = float(unload_trip)
        self.hold_s = float(hold_s)
        self.stage_i = 0
        self.stage_started = float(now)
        self.settle_since = None
        self.dwell_since = None
        self.q_cmd = Q_CROUCH.copy()
        self.targets = None
        self.height_frac = 0.0
        self.body_xy = np.zeros(2)
        self.pre_m = 0.0
        self.lift_frac = 0.0
        self._from = self._snapshot()
        self._to = self._snapshot()
        self.prelift_m = PRE2_START_M
        self.balance_last_t = None
        self.leg_trim = {}
        self._tilt_lpf = (0.0, 0.0)
        self.tilt_baseline = None
        self.last_tilt_deg = np.nan
        self.last_pair_tau_nm = np.nan
        self.last_settle_detail = ""
        self.aborted = None
        self.stop_reason = None
        self.torque_table_printed = False
        self.events = []

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
        self.dwell_since = None
        if name == "LOWER2":
            # the foot is returning to the ground -- any accrued per-leg
            # trim would just hold it that much off the true stance level
            self.leg_trim = {leg: 0.0 for leg in self.leg_trim}
        if self.kind == "stream":
            self._from = self._snapshot()
            self._to = self._stream_endpoint(name)
        self.events.append((float(now), name, note))
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
        if error > POSITION_POSE_TOL or speed > POSITION_QD_TOL:
            self.settle_since = None
            return False
        if self.settle_since is None:
            self.settle_since = float(now)
            return False
        return now - self.settle_since >= POSITION_SETTLE_S

    def sweep(self, now, q_enc, qd, velocity_ready, tau_measured, rp_deg=None):
        """One 250 Hz update.  Returns (q_cmd_rad, event_or_None)."""
        event = None
        name, kind = STAGES[self.stage_i]
        plan = self.plan

        if name == "SHIFT_SETTLE" and rp_deg is not None:
            self.tilt_baseline = (float(rp_deg[0]), float(rp_deg[1]))

        if kind == "move":
            self.q_cmd = Q_CROUCH.copy()
            self.targets = None
            if self._settled(now, q_enc, qd, velocity_ready):
                event = self._goto("STAND", now, "settled")
            elif now - self.stage_started > SETTLE_TIMEOUT_S:
                self.stop_reason = (f"{name} did not settle: "
                                    f"{self.last_settle_detail}")

        elif kind == "stream":
            u = _smoothstep((now - self.stage_started) / STREAM_T[name])
            f, t = self._from, self._to
            self.height_frac = f["height"] + u * (t["height"] - f["height"])
            self.body_xy = f["body"] + u * (t["body"] - f["body"])
            self.pre_m = f["pre"] + u * (t["pre"] - f["pre"])
            self.lift_frac = f["lift"] + u * (t["lift"] - f["lift"])
            self.targets = plan.targets(
                self.body_xy, self.height_frac, self.pre_m, self.lift_frac,
                trim=self.leg_trim,
            )
            self.q_cmd = _ik_all(self.q_cmd, self.targets)
            if now - self.stage_started >= STREAM_T[name]:
                nxt = {"STAND": "STAND_SETTLE", "SHIFT": "SHIFT_SETTLE",
                       "UNLOAD2": "BALANCE", "LIFT2": "HOLD2",
                       "LOWER2": "RELOAD", "RECENTER": "DONE"}[name]
                event = self._goto(nxt, now, "stream complete")

        elif name == "STAND_SETTLE":
            if self._settled(now, q_enc, qd, velocity_ready):
                event = self._goto("HOLD4", now, "stand settled")
            elif now - self.stage_started > SETTLE_TIMEOUT_S:
                self.stop_reason = (f"STAND did not settle: "
                                    f"{self.last_settle_detail}")

        elif name == "SHIFT_SETTLE":
            if self._settled(now, q_enc, qd, velocity_ready):
                event = self._goto("UNLOAD2", now, "bias settled")
            elif now - self.stage_started > SETTLE_TIMEOUT_S:
                event = self._abort(now, "SHIFT did not settle")

        elif name == "BALANCE":
            if self.balance_last_t is None:
                self.balance_last_t = float(now)
                self.leg_trim = {leg: 0.0 for leg in plan.lifted}
                self._tilt_lpf = (0.0, 0.0)
            taus = {}
            for leg in plan.lifted:
                sl = _sl(leg)
                taus[leg] = float(
                    np.max(np.abs(np.asarray(tau_measured)[sl][1:]))
                )
            self.last_pair_tau_nm = max(taus.values())
            tau_list = [taus[leg] for leg in plan.lifted]
            tau_ok = self.last_pair_tau_nm <= self.unload_trip
            tilt = None
            if rp_deg is not None and self.tilt_baseline is not None:
                d_r = rp_deg[0] - self.tilt_baseline[0]
                d_p = rp_deg[1] - self.tilt_baseline[1]
                tilt = np.array([d_p, -d_r])
                self.last_tilt_deg = float(np.linalg.norm(tilt))
            level_ok = (self.last_tilt_deg <= TILT_BALANCED_DEG
                        if tilt is not None else True)
            trim_text = "/".join(
                f"{1000 * self.leg_trim[leg]:.1f}" for leg in plan.lifted
            )
            if tau_ok and level_ok:
                if self._settled(now, q_enc, qd, velocity_ready):
                    tilt_text = ("" if tilt is None else
                                 f", tilt {self.last_tilt_deg:.1f} deg")
                    event = self._goto(
                        "LIFT2", now,
                        f"pair free: |tau| {tau_list[0]:.2f}/"
                        f"{tau_list[1]:.2f} N*m{tilt_text}, per-leg trim "
                        f"{trim_text} mm",
                    )
            else:
                self.settle_since = None
                dt = max(0.0, now - self.balance_last_t)
                # PER-LEG vertical trim (replaces the old horizontal creep):
                # same PI structure as the hardware-proven walk `--imu-level`
                # stance trim, but it only ever raises a lifted leg's own
                # pre-lift -- directly compensating "this specific corner
                # won't clear" instead of hoping a shared body offset does.
                if tilt is not None:
                    # rate-independent LPF: alpha from the elapsed dt, so the
                    # filter has the same 0.135 s time constant whether the
                    # sweep runs at 20.8 Hz or the true 250 Hz.
                    a = 1.0 - np.exp(-dt / BALANCE_TILT_LPF_TAU_S) if dt > 0 \
                        else 0.0
                    self._tilt_lpf = (a * d_r + (1.0 - a) * self._tilt_lpf[0],
                                      a * d_p + (1.0 - a) * self._tilt_lpf[1])
                    d_r_f, d_p_f = self._tilt_lpf
                    if max(abs(d_r_f), abs(d_p_f)) > BALANCE_TILT_DEADBAND_DEG:
                        tilt_rad = np.deg2rad([d_p_f, -d_r_f])
                        for leg in plan.lifted:
                            raw = float(np.dot(
                                plan.crouch_foot[leg][:2], tilt_rad))
                            self.leg_trim[leg] = float(np.clip(
                                self.leg_trim[leg]
                                + BALANCE_TRIM_KI * max(0.0, raw) * dt,
                                0.0, BALANCE_TRIM_CLAMP_M,
                            ))
                else:
                    # no IMU: fall back to each leg's own measured torque --
                    # targeted, but still blind exactly where the hardware
                    # docs warn tau is blind (a leg resting on a third point
                    # can read near-zero even while physically unloaded).
                    for leg in plan.lifted:
                        over = max(0.0, taus[leg] - self.unload_trip)
                        self.leg_trim[leg] = float(np.clip(
                            self.leg_trim[leg] + BALANCE_TRIM_KI * over * dt,
                            0.0, BALANCE_TRIM_CLAMP_M,
                        ))
                self.targets = plan.targets(
                    self.body_xy, self.height_frac, self.pre_m,
                    self.lift_frac, trim=self.leg_trim,
                )
                self.q_cmd = _ik_all(self.q_cmd, self.targets)
            self.balance_last_t = float(now)
            if (event is None
                    and now - self.stage_started > BALANCE_TIMEOUT_S):
                event = self._abort(
                    now,
                    f"balance timeout: tilt {self.last_tilt_deg:.1f} deg, "
                    f"|tau| {tau_list[0]:.2f}/{tau_list[1]:.2f} N*m, "
                    f"per-leg trim {trim_text} mm",
                )

        elif kind == "dwell":
            if self.dwell_since is None:
                self.dwell_since = float(now)
            duration = {"HOLD4": HOLD4_S, "HOLD2": self.hold_s,
                       "RELOAD": T_RELOAD, "DONE": DONE_S}[name]
            if now - self.dwell_since >= duration:
                if name == "HOLD4":
                    event = self._goto("SHIFT", now, "hold4 complete")
                elif name == "HOLD2":
                    event = self._goto(
                        "LOWER2", now, f"held {duration:.0f}s -- putting down"
                    )
                elif name == "RELOAD":
                    event = self._goto("RECENTER", now, "reloaded")
                elif name == "DONE":
                    self.stop_reason = (
                        "sequence complete" if self.aborted is None
                        else f"aborted: {self.aborted}"
                    )

        return self.q_cmd, event

    def speed_cap_dps(self, caps):
        if self.stage == "CROUCH":
            return caps["crouch"]
        if self.stage == "STAND":
            return caps["stand"]
        return caps["stream"]


def _print_torque_table(measured_torque):
    print("[HOLD4] measured torques (mis-zero signature = large left/right "
          "or diagonal asymmetry):")
    for i, leg in enumerate(LEGS):
        abd, pitch, knee = measured_torque[3 * i:3 * i + 3]
        print(f"        {leg}: abd={abd:+5.2f} pitch={pitch:+5.2f} "
              f"knee={knee:+5.2f} N*m")


def validate_configuration(plan):
    """Offline: IK/limits over the whole scripted sequence, including the
    worst-case per-leg BALANCE trim (both lifted legs maxed out at once,
    a stricter check than either alone).  Raises on the first infeasible
    waypoint."""
    low, high = soft_limits()
    q = Q_CROUCH.copy()
    max_pre = PRE2_START_M + BALANCE_TRIM_CLAMP_M
    path = (
        [(np.zeros(2), h, 0.0, 0.0) for h in np.linspace(0.0, 1.0, 9)]
        + [(u * plan.bias, 1.0, 0.0, 0.0) for u in np.linspace(0.0, 1.0, 6)]
        + [(plan.bias, 1.0, p, 0.0)
           for p in np.linspace(0.0, max_pre, 6)]
        + [(plan.bias, 1.0, max_pre, f)
           for f in np.linspace(0.0, 1.0, 5)]
    )
    for body, height, pre, lift in path:
        targets = plan.targets(body, height, pre, lift)
        q = _ik_all(q, targets)
        for leg in LEGS:
            sl = _sl(leg)
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


class Oracle:
    """Privileged sim measurements -- judge the run, never steer it.

    The stance "polygon" for 2 legs is a LINE, not a triangle: the honest
    stability measure is the true CoM's perpendicular distance to that
    line (0 = balanced), not a triangle margin.  Also tracks each lifted
    foot's contact force (0 = genuinely airborne, i.e. a real 2-leg stand
    rather than a tilted rest on a third leg).
    """

    def __init__(self, model, data, plan):
        import mujoco
        self._mujoco = mujoco
        self.m, self.d = model, data
        self.plan = plan
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                       "trunk")
        self.foot_geom = {leg: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}") for leg in LEGS}
        self.site = {leg: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"foot_{leg}") for leg in LEGS}
        self.reset()

    def reset(self):
        self.hold2 = dict(
            max_lift_force={leg: 0.0 for leg in self.plan.lifted},
            min_clearance={leg: np.inf for leg in self.plan.lifted},
            max_tilt_deg=0.0, min_line_margin_abs=np.inf,
            last_line_margin=np.nan, samples=0,
        )
        self.max_abs_tau = np.zeros(N_JOINTS)

    def foot_contact(self, leg):
        geom = self.foot_geom[leg]
        normal = 0.0
        force = np.zeros(6)
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            if geom in (con.geom1, con.geom2):
                self._mujoco.mj_contactForce(self.m, self.d, i, force)
                normal += abs(force[0])
        return normal

    def tilt_deg(self):
        R = self.d.xmat[self.trunk].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))

    def line_margin(self):
        """Signed perpendicular distance from the true CoM to the stance
        line, along ``plan.perp`` (0 = balanced on the line)."""
        com = self.d.subtree_com[self.trunk][:2]
        p0 = self.d.site_xpos[self.site[self.plan.stance[0]]][:2]
        return float(np.dot(com - p0, self.plan.perp))

    def sample(self, stage, tau):
        self.max_abs_tau = np.maximum(self.max_abs_tau, np.abs(tau))
        self.hold2["last_line_margin"] = self.line_margin()
        if stage not in ("BALANCE", "LIFT2", "HOLD2"):
            return
        h = self.hold2
        for leg in self.plan.lifted:
            f = self.foot_contact(leg)
            h["max_lift_force"][leg] = max(h["max_lift_force"][leg], f)
            clearance = float(self.d.site_xpos[self.site[leg]][2]) - 0.02
            h["min_clearance"][leg] = min(h["min_clearance"][leg], clearance)
        h["max_tilt_deg"] = max(h["max_tilt_deg"], self.tilt_deg())
        h["min_line_margin_abs"] = min(h["min_line_margin_abs"],
                                       abs(self.line_margin()))
        h["samples"] += 1

    def verdict(self, controller):
        h = self.hold2
        checks = [
            ("finished (no abort)", controller.aborted is None),
            ("HOLD2 sampled", h["samples"] > 0),
        ]
        for leg in self.plan.lifted:
            checks.append((f"{leg} airborne (force <= 0.05 N)",
                          h["max_lift_force"][leg] <= 0.05))
            checks.append((f"{leg} clearance >= 5 mm",
                          h["min_clearance"][leg] >= 0.005))
        checks.append(("trunk tilt <= 10 deg", h["max_tilt_deg"] <= 10.0))
        return all(ok for _, ok in checks), checks


def run(model, data, args, viewer=None, quiet=False):
    """Drive one full sequence; returns (controller, oracle)."""
    import mujoco
    plan = TwoPlan(args.lift_pair, stand_height=args.stand_height,
                   bias_xy=(args.bias_x, args.bias_y))
    controller = TwoSequence(0.0, plan, unload_trip=args.unload_trip,
                             hold_s=args.hold_s)
    oracle = Oracle(model, data, plan)
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    imu = SimIMU(model, data, trunk, noise_std_deg=args.imu_noise_deg,
                bias_deg=(args.imu_bias_roll_deg, args.imu_bias_pitch_deg))

    qadr = np.array([model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{j}_{leg}")]
        for leg in LEGS for j in ("hip_abd", "hip_pitch", "knee")])
    assert qadr[0] == 7

    servo = NativePositionServo(data.qpos[qadr],
                                kp=KP_SERVO * args.kp_scale, kd=KD_SERVO)

    steps_per_tick = max(1, round(1.0 / (CONTROL_HZ * model.opt.timestep)))
    # One control tick = one whole-bus sweep: all 12 drivers commanded, all
    # 12 replying, so every motor runs at CONTROL_HZ (250 Hz, 78% of a
    # 1 Mbit/s bus).  --can-roundrobin restores the old one-motor-per-tick
    # model, which refreshed each joint at only 250/12 = 20.8 Hz.
    roundrobin = bool(getattr(args, "can_roundrobin", False))
    ticks_per_sweep = N_JOINTS if roundrobin else 1
    caps = {"crouch": args.crouch_dps, "stand": args.stand_dps,
            "stream": args.stream_dps}
    slot = 0
    q_targets = Q_CROUCH.copy()
    last_print = -np.inf
    t_wall0 = time.perf_counter()
    tipover_since = None
    lean_baseline = None
    lean_since = None
    velocity = base.EncoderVelocity()

    while controller.stop_reason is None:
        now = data.time
        if slot % ticks_per_sweep == 0:
            q_enc, _qd_driver, tau_measured = servo.telemetry()
            qd_encoder = velocity.update(q_enc, now)
            roll_deg, pitch_deg = imu.sample()
            rp_deg = None if args.no_imu else (roll_deg, pitch_deg)

            q_targets, event = controller.sweep(
                now, q_enc, qd_encoder, velocity.ready, tau_measured,
                rp_deg=rp_deg,
            )
            oracle.sample(controller.stage, tau_measured)

            if event and not quiet:
                print(f"t={now:6.2f}  [stage] {event}")
            if (controller.stage == "HOLD4"
                    and not controller.torque_table_printed):
                _print_torque_table(tau_measured)
                controller.torque_table_printed = True

            if abs(roll_deg) > TIPOVER_ABORT_DEG \
                    or abs(pitch_deg) > TIPOVER_ABORT_DEG:
                if tipover_since is None:
                    tipover_since = now
                elif (now - tipover_since >= TIPOVER_CONFIRM_S
                        and controller.stop_reason is None):
                    controller.stop_reason = (
                        f"IMU tip-over: roll={roll_deg:+.1f} "
                        f"pitch={pitch_deg:+.1f} deg"
                    )
            else:
                tipover_since = None
            if controller.stage in LEAN_STAGES:
                if lean_baseline is None:
                    lean_baseline = (roll_deg, pitch_deg)
                    lean_since = None
                else:
                    d_r = roll_deg - lean_baseline[0]
                    d_p = pitch_deg - lean_baseline[1]
                    if max(abs(d_r), abs(d_p)) > args.lean_abort_deg:
                        if lean_since is None:
                            lean_since = now
                        elif now - lean_since >= LEAN_CONFIRM_S:
                            note = controller._abort(
                                now, f"IMU lean d_roll={d_r:+.1f} "
                                    f"d_pitch={d_p:+.1f} deg")
                            if not quiet:
                                print(f"t={now:6.2f}  [stage] {note}")
                            lean_baseline = None
                            lean_since = None
                    else:
                        lean_since = None
            else:
                lean_baseline = None
                lean_since = None

            if not quiet and now - last_print >= 1.0 / STATUS_HZ:
                taus = [float(np.max(np.abs(
                    tau_measured[_sl(leg)][1:]))) for leg in plan.lifted]
                trim_text = "/".join(
                    f"{1000 * controller.leg_trim.get(leg, 0.0):.1f}"
                    for leg in plan.lifted
                )
                print(f"t={now:6.2f}  {controller.stage:<12} "
                      f"pair|tau|={taus[0]:.2f}/{taus[1]:.2f} "
                      f"trim=({trim_text})mm "
                      f"rp=({roll_deg:+.2f},{pitch_deg:+.2f})deg "
                      f"lineErr={1000 * oracle.hold2['last_line_margin']:+.1f}mm",
                      flush=True)
                last_print = now

        cap_dps = controller.speed_cap_dps(caps)
        motors = (slot % N_JOINTS,) if roundrobin else range(N_JOINTS)
        for motor in motors:
            servo.command(motor, q_targets[motor], cap_dps)
        for _ in range(steps_per_tick):
            data.ctrl[:] = servo.step(data.qpos[qadr], model.opt.timestep)
            mujoco.mj_step(model, data)
        if viewer is not None:
            if not viewer.is_running():
                controller.stop_reason = "viewer closed"
                break
            viewer.sync()
            lag = data.time - (time.perf_counter() - t_wall0)
            if lag > 0:
                time.sleep(lag)
            elif lag < -0.5:
                t_wall0 = time.perf_counter() - data.time
        slot += 1
        if data.time > 120.0:
            controller.stop_reason = "wall timeout"
            break
    return controller, oracle


def report(controller, oracle, quiet=False):
    passed, checks = oracle.verdict(controller)
    if not quiet:
        print(f"\nstop: {controller.stop_reason}"
              + (f"  (ABORTED: {controller.aborted})"
                 if controller.aborted else ""))
        for now, stage, note in controller.events:
            print(f"  t={now:6.2f}  -> {stage:<12} {note}")
        h = oracle.hold2
        print(f"\nBALANCE/HOLD2 window ({h['samples']} samples):")
        for leg in oracle.plan.lifted:
            print(f"  {leg} lifted-foot force max : "
                  f"{h['max_lift_force'][leg]:.3f} N")
            print(f"  {leg} clearance min         : "
                  f"{1000 * h['min_clearance'][leg]:.1f} mm")
        print(f"  trunk tilt max              : {h['max_tilt_deg']:.2f} deg")
        print(f"  true CoM-to-line margin min : "
              f"{1000 * h['min_line_margin_abs']:.1f} mm (0 = on the line)")
        worst = int(np.argmax(oracle.max_abs_tau))
        print(f"  peak measured torque        : {JOINT_LABELS[worst]} "
              f"{oracle.max_abs_tau[worst]:.2f} N*m")
        print("\nverdict:")
        for label, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        print(f"\n{'PASS' if passed else 'FAIL'}: 2-leg diagonal stand on "
              f"{'+'.join(oracle.plan.stance)} "
              f"({'proven' if passed else 'not proven'} under sim)")
    return passed


def self_test(args):
    """Offline validation: IK reachability + scripted BALANCE scenarios,
    ported from twostand_hw.py's offline_self_test."""
    # -- IMU sign convention regression guard (pure algebra, no MuJoCo) --
    rx = lambda a: np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)],
                             [0, np.sin(a), np.cos(a)]])
    ry = lambda a: np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                             [-np.sin(a), 0, np.cos(a)]])
    r, p = _roll_pitch_deg(rx(np.deg2rad(-5.0)))
    assert r < -1.0 and abs(p) < 1.0e-6, (r, p)          # left-down -> -roll
    r, p = _roll_pitch_deg(ry(np.deg2rad(5.0)))
    assert p > 1.0 and abs(r) < 1.0e-6, (r, p)           # nose-down -> +pitch
    print("IMU sign convention: left-down -> -roll, nose-down -> +pitch  OK")

    plan = TwoPlan(args.lift_pair, stand_height=args.stand_height,
                   bias_xy=(args.bias_x, args.bias_y))
    validate_configuration(plan)
    print(f"sequence IK-feasible: lift {'+'.join(plan.lifted)}, bias "
          f"({1000 * args.bias_x:+.0f},{1000 * args.bias_y:+.0f}) mm, "
          f"raise to {1000 * (PRE2_START_M + LIFT2_M):.0f} mm")

    def spin(seq, stop_stages, tau_fn=None, rp_fn=None, limit_s=200.0):
        now = seq.stage_started
        while seq.stage not in stop_stages:
            now += SWEEP_DT
            tau = np.zeros(N_JOINTS) if tau_fn is None else tau_fn(seq)
            rp = None if rp_fn is None else rp_fn(seq)
            seq.sweep(now, seq.q_cmd.copy(), np.zeros(N_JOINTS), True,
                     tau, rp_deg=rp)
            assert now < seq.stage_started + limit_s, f"stuck in {seq.stage}"
        return now

    # 1) dry run, no load anywhere -> full cycle, no abort.
    seq = TwoSequence(0.0, plan, unload_trip=args.unload_trip, hold_s=1.0)
    now = seq.stage_started
    while seq.stop_reason is None:
        now += 0.048
        seq.sweep(now, seq.q_cmd.copy(), np.zeros(N_JOINTS), True,
                 np.zeros(N_JOINTS))
        assert now < seq.stage_started + 200.0, f"stuck in {seq.stage}"
    assert seq.aborted is None, seq.aborted
    assert seq.stop_reason == "sequence complete"
    print("dry run: full 2-leg cycle (timer put-down) complete, no abort")

    # 2) BALANCE closed loop: a lifted leg stays loaded until ITS OWN trim
    # (not a shared body offset) has accrued enough -- the no-IMU tau
    # fallback drives this leg-by-leg now.
    loaded_leg = plan.lifted[0]

    def tau_for(seq_obj):
        tt = np.zeros(N_JOINTS)
        trim = seq_obj.leg_trim.get(loaded_leg, 0.0)
        sl = _sl(loaded_leg)
        tt[sl.start + 2] = 0.2 if trim >= 0.008 else 2.0
        return tt

    seq2 = TwoSequence(0.0, plan, unload_trip=args.unload_trip, hold_s=1.0)
    spin(seq2, ("LIFT2",), tau_fn=tau_for)
    accrued_mm = 1000 * seq2.leg_trim[loaded_leg]   # LOWER2 zeroes this later
    assert seq2.leg_trim[loaded_leg] >= 0.008 - 1.0e-6, seq2.leg_trim
    assert seq2.aborted is None
    spin(seq2, ("DONE",), tau_fn=tau_for)
    assert seq2.aborted is None
    print(f"balance loop: accrued {accrued_mm:.1f} mm of trim on "
          f"{loaded_leg}, pair freed, full cycle complete")

    # 3) Never balances (one leg stays loaded no matter how much trim it
    # gets) -> trim clamps, stage times out, graceful put-down.
    seq3 = TwoSequence(0.0, plan, unload_trip=args.unload_trip, hold_s=1.0)
    heavy = np.zeros(N_JOINTS)
    heavy[_sl(loaded_leg).start + 2] = 2.0
    spin(seq3, ("DONE",), tau_fn=lambda s: heavy)
    assert seq3.aborted is not None and "timeout" in seq3.aborted, seq3.aborted
    print(f"never-balances (tau stays loaded): trim clamps at "
          f"{1000 * BALANCE_TRIM_CLAMP_M:.0f} mm, stage times out, "
          "graceful put-down")

    # 4) THE tau-blind scenario (twostand_hw.py's documented hardware bug):
    # tau reads LOW throughout (as it does when the body rests tilted on a
    # third leg -- the force line passes through the sensed joints) while
    # the IMU shows a real, uncorrected tilt.  WITH the IMU, tilt steering
    # keeps trimming loaded_leg until level; WITHOUT it (rp_fn=None,
    # --no-imu on hardware or a stale sample), the blind tau law falsely
    # declares the pair free at ZERO trim -- this is the reproduction of
    # today's "second leg never lifts" symptom.
    mag_deg = 2.5
    dirn = plan.crouch_foot[loaded_leg][:2]
    dirn = dirn / np.linalg.norm(dirn)   # tilt = [d_p,-d_r] aligned with
                                         # loaded_leg's own crouch position
    def rp_for(s):
        trim = s.leg_trim.get(loaded_leg, 0.0)
        if s.stage in ("UNLOAD2", "BALANCE") and trim < 0.009:
            return (-mag_deg * dirn[1], mag_deg * dirn[0])   # (roll, pitch)
        return (0.0, 0.0)

    tseq = TwoSequence(0.0, plan, unload_trip=args.unload_trip, hold_s=1.0)
    spin(tseq, ("LIFT2",), rp_fn=rp_for)
    assert tseq.leg_trim[loaded_leg] >= 0.009 - 1.0e-6, tseq.leg_trim
    assert tseq.aborted is None
    print(f"tau-blind scenario WITH IMU: tau read 'free' throughout, tilt "
          f"steering still accrued {1000 * tseq.leg_trim[loaded_leg]:.1f} mm "
          f"of trim on {loaded_leg} before declaring balance")

    bseq = TwoSequence(0.0, plan, unload_trip=args.unload_trip, hold_s=1.0)
    spin(bseq, ("LIFT2",))    # rp_fn=None: no IMU, tau reads near-zero/free
    assert all(v == 0.0 for v in bseq.leg_trim.values()), bseq.leg_trim
    print("tau-blind scenario WITHOUT IMU: BALANCE declared the pair free "
          "at ZERO per-leg trim -- this is the reproduction of the hardware "
          "'second leg never lifts' symptom (LIFT2 is commanded on a leg "
          "that never actually got the extra lift it needed)")

    print("twostand_dog5 offline self-test PASS")
    return 0


SWEEP_CASES = [
    dict(label="FL+RR lift, bias 0",       lift=("FL", "RR"), bias=(0.0, 0.0)),
    dict(label="FR+RL lift, bias 0",       lift=("FR", "RL"), bias=(0.0, 0.0)),
    dict(label="FL+RR lift, bias y-5mm",   lift=("FL", "RR"), bias=(0.0, -0.005)),
    dict(label="FL+RR lift, bias y+5mm",   lift=("FL", "RR"), bias=(0.0, +0.005)),
    dict(label="FR+RL lift, bias y-5mm",   lift=("FR", "RL"), bias=(0.0, -0.005)),
    dict(label="FR+RL lift, bias y+5mm",   lift=("FR", "RL"), bias=(0.0, +0.005)),
    dict(label="FL+RR lift, bias x+5mm",   lift=("FL", "RR"), bias=(+0.005, 0.0)),
    dict(label="FL+RR lift, bias x-5mm",   lift=("FL", "RR"), bias=(-0.005, 0.0)),
    dict(label="FL+RR lift, no IMU",       lift=("FL", "RR"), bias=(0.0, 0.0), no_imu=True),
]


def sweep(args):
    results = []
    for case in SWEEP_CASES:
        case = dict(case)
        label = case.pop("label")
        case_args = argparse.Namespace(**vars(args))
        case_args.lift_pair = case.pop("lift")
        case_args.bias_x, case_args.bias_y = case.pop("bias")
        case_args.no_imu = case.pop("no_imu", args.no_imu)
        model, data = build_sim(kp_scale=args.kp_scale)
        controller, oracle = run(model, data, case_args, quiet=True)
        passed = report(controller, oracle, quiet=True)
        h = oracle.hold2
        detail = (
            f"tilt {h['max_tilt_deg']:.1f} deg, line-err "
            f"{1000 * h['min_line_margin_abs']:.0f} mm, forces "
            + "/".join(f"{h['max_lift_force'][leg]:.2f}N"
                      for leg in oracle.plan.lifted)
            if h["samples"] else
            f"stopped: {controller.aborted or controller.stop_reason}"
        )
        results.append((label, passed, detail))
        print(f"[{'PASS' if passed else 'FAIL'}] {label:<24} {detail}",
              flush=True)
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\nsweep: {n_pass}/{len(results)} PASS")
    return 0 if n_pass == len(results) else 1


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument(
        "--can-roundrobin", action="store_true",
        help="legacy CAN model: one motor per 250 Hz tick, so each joint is "
             "only refreshed at 250/12 = 20.8 Hz.  Kept to reproduce the "
             "pre-2026-08-10 results; the real bus carries all 12 motors at "
             "250 Hz each (78%% load)")
    parser.add_argument(
        "--lift", type=str, default=",".join(DEFAULT_LIFT_PAIR),
        help="the diagonal PAIR to lift (default FL,RR -- stands on FR+RL)")
    parser.add_argument("--hold-s", type=float, default=DEFAULT_HOLD_S)
    parser.add_argument("--bias-x", type=float, default=0.0)
    parser.add_argument("--bias-y", type=float, default=0.0)
    parser.add_argument("--lean-abort-deg", type=float,
                        default=DEFAULT_LEAN_ABORT_DEG)
    parser.add_argument("--stand-height", type=float,
                        default=DEFAULT_STAND_HEIGHT_M)
    parser.add_argument("--torque-trip", type=float,
                        default=DEFAULT_TORQUE_TRIP_NM)
    parser.add_argument("--unload-trip", type=float,
                        default=DEFAULT_UNLOAD_TRIP_NM)
    parser.add_argument("--crouch-dps", type=float, default=DEFAULT_CROUCH_DPS)
    parser.add_argument("--stand-dps", type=float, default=DEFAULT_STAND_DPS)
    parser.add_argument("--stream-dps", type=float, default=DEFAULT_STREAM_DPS)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--imu-noise-deg", type=float, default=0.0)
    parser.add_argument("--imu-bias-roll-deg", type=float, default=0.0)
    parser.add_argument("--imu-bias-pitch-deg", type=float, default=0.0)
    parser.add_argument(
        "--no-imu", action="store_true",
        help="disable the simulated IMU -- BALANCE falls back to the "
             "tau-only law hardware-shown to be blind to a third-leg rest")
    args = parser.parse_args()

    pair = tuple(s.strip().upper() for s in args.lift.split(",") if s.strip())
    if frozenset(pair) not in DIAGONALS or len(pair) != 2:
        parser.error(f"--lift must be a DIAGONAL pair (FR,RL or FL,RR); "
                     f"got {args.lift!r}")
    args.lift_pair = pair
    if abs(args.bias_x) > BIAS_MAX_M or abs(args.bias_y) > BIAS_MAX_M:
        parser.error(f"--bias-x/--bias-y limited to +-{BIAS_MAX_M} m")

    if args.self_test:
        return self_test(args)
    if args.sweep:
        return sweep(args)

    model, data = build_sim(kp_scale=args.kp_scale)
    if args.headless:
        controller, oracle = run(model, data, args)
        return 0 if report(controller, oracle) else 1

    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        controller, oracle = run(model, data, args, viewer=viewer)
    return 0 if report(controller, oracle) else 1


if __name__ == "__main__":
    raise SystemExit(main())
