#!/usr/bin/env python3
"""The runner: trot in place on commanded force.  One thread.

    CROUCH (0xA4) -> WAIT -> RISE (torque) -> HOLD (torque) -> TROT -> PARK

THIS FILE IS august_week2/stand_torque_Mode.py WITH ONE STAGE ADDED.
    CROUCH, WAIT, RISE and HOLD are that file, unchanged -- same gains, same
    q_ref_for_height, same TorqueGate, same estimator, same e-stop set, same
    everything that has been up and down on this robot.  A trot that cannot
    stand first is not a trot problem, and rewriting the part that already
    works is how the last attempt ended up unable to rise at all.

    What is new is TROT, and only TROT:
        dog5_trot_quasi_static_model.gait   the contact clock: which diagonal is down, and the
                         weight ramp that hands the load over
        a swing arc      in the BODY frame, because this trots IN PLACE: the
                         foot's x/y target is its own crouch value, exactly
                         the one q_ref_for_height already pins, and only z
                         moves.  No world frame, so no frame to get wrong.

    The force path is week 2's too.  force_totorque.distribute already takes
    an arbitrary stance list and already carries the unilateral clamp, the
    friction cone and the total-Fz rescale; stance_torque already gives a
    non-planted leg its own leg-gravity torque and nothing else.  Feeding it
    a time-varying contact mask IS the trot.  There is no second force
    distributor in this file.

WHAT THIS ANSWERS
    In position mode the drivers' own loops hold the pose, so once the robot
    settles nothing the upper layer computes has any observable effect -- you
    cannot tell a working feedback loop from an ignored one.  Here the trunk is
    held up by forces this code computes, so the loop is the only thing keeping
    it up.  Push it, and the recovery IS the feedback.

THE FIVE FILES, AND WHY THE SPLIT IS WHERE IT IS
    crouch_and_park     0xA4 bookends.  No torque, no model.
    feedback_estimator  sensors -> (z, v, C, omega).  Measurement only.
    Dynamic_Model       state error -> 6D wrench.  THE CONTROLLER: the only
                        place a gain touches the trunk.
    force_totorque      wrench -> per-foot GRF -> 12 joint torques.  Pure
                        transmission; no trunk feedback at all.
    stand_torque_Mode   when to do which, at what rate, and every way to
                        stop.  THIS FILE is that runner plus the TROT stage,
                        plus dog5_trot_quasi_static_model.gait and the body-frame swing arc.

ONE THREAD, WITH THE MODEL SUB-SAMPLED
    MEASURED on this Pi, not assumed: the per-sweep work (impedance + gate) is
    28 us, 8% of one 333 us slot.  The model block (wrench + grasp map + J^T +
    warm-started IK) is 1384 us and the load check is 1029 us -- neither can
    run every sweep.  Last week's answer was a 100 Hz worker thread with a
    mailbox, atomic publishing and a staleness watchdog: ~200 lines that exist
    only to make a thread safe.

    Here both run IN the sweep, sub-sampled and STAGGERED so they never land
    in the same one -- model at 83 Hz, load check at 21 Hz, worst single-sweep
    delay 1.38 ms rather than the 2.41 ms of both together.  That is inside
    the driver's input-lost watchdog with margin either way it is read (see
    the note on params.WATCHDOG_S).  The term that actually stabilises a joint
    -- the impedance -- still runs EVERY sweep on qd at most 4 ms old.  Only
    the feedforward is held between updates.

SAFETY, IN THE ORDER IT APPLIES
    joint impedance (250 Hz)     no joint is ever an open integrator
    abduction sign check         no leg swings through the body
    TorqueGate                   ramp, cap, slew, overspeed, temp, CAN miss
    LoadWatch                    measured iq must add up to the robot
    LIMP                         SPACE, or automatic on any refusal

    NOTE the joint SOFT LIMITS are off, as in crouch_and_park: the symmetric
    +/-2.6 rad box sits 6.9 deg from the resting crouch on the knees while
    allowing a knee 298 deg of travel.  The abduction sign check replaces it.

WHAT THE STATUS LINE SHOWS, AND WHY IT IS ONLY THESE
    A line printed at 2 Hz is for the human holding the robot, and a human can
    act on exactly two things: is it at the right height, and is it level.  So
    the stream carries height, tilt, the IMU-free cross-check on tilt, trunk
    velocity and the planted count -- and a legend for them is printed once, at
    the top, rather than left as initials.

    |tau|, trk and |dq| USED TO STREAM HERE AND NO LONGER DO.  They were
    per-sweep control internals sampled at 2 Hz, which is neither a
    measurement (too slow to see the 10 Hz behaviour that mattered) nor
    actionable (nothing an operator does changes them).  Every one is in the
    --log npz at the full 250 Hz -- tau_cmd, tau_meas, and q_ref alongside q
    so the impedance error is recoverable -- which is where they were actually
    read from anyway when the 2026-08-17 shake was diagnosed.

    THE FOOT-LOAD SUM MOVED TO THE EXIT REPORT, and it is still the number
    that matters most.  Torque calibration was dropped for week 2, so it is
    the ONLY end-to-end evidence that commanded torque becomes real force:
    measured iq, inverted through J^-T with each leg's own weight removed, and
    it must read ~57 N.  If it does not, the grasp map is fantasy and a
    good-looking attitude proves nothing.  It also still LIMPS the robot
    in-run when it disagrees with the weight by more than LOAD_SUM_TOL_FRAC,
    and that prints when it fires -- an alarm, rather than a number to watch.

THE TWO NUMBERS THAT WERE DEAD RECKONING UNTIL 2026-08-17
    Height and attitude both came out of FK and the AHRS with nothing to check
    them against, so a constant error in either was invisible from inside the
    loop.  Two were sitting there:

    zI= IS FLOOR TO TRUNK BOTTOM, and it did not used to be.  It was the
    hip-axis plane -- a plane nothing physical sits on -- so the runner printed
    191 mm where a ruler read ~160.  38 mm of that was the frame.  zI is the
    point the ruler reaches; the (H...) beside it is the hip axis, which is
    where the leg tables and the IK work.  The height knob (config.py's
    STAND_TRUNK_BOTTOM_M) means zI now, and
    STAND_HEIGHT moved 0.190 -> 0.152 so the POSE is unchanged.

    rp:d IS THE ONLY INDEPENDENT CHECK ON THE AHRS.  rp:fk is a least-squares
    plane through the four measured feet, no IMU in it, in the same convention
    as the AHRS branch -- so rp:d = ahrs - fk is floor slope + IMU mount tilt
    + encoder zeros.  It printed ~0.5 deg with the robot standing still, and
    with kp_roll = 120 that is a 1.05 Nm standing moment (16% of this stance's
    6.4 Nm roll capacity) spent holding the robot off true level.

    THE SETPOINT PROCEDURE, because rp:d is three constants added together:
        1. crouch and read the rp: block the WAIT stage prints
        2. set SETPOINT_ROLL_DEG / SETPOINT_PITCH_DEG to rp:d and re-run
        3. to split mount from floor: rotate the robot 180 deg on the SAME
           spot and read again.  The floor's share flips sign, the mount's
           does not, so half the sum is the mount and half the difference is
           the floor.  Only the mount half belongs in a setpoint.
    It defaults to 0/0 -- i.e. wrong -- on purpose: a measured pair from one
    floor is not a property of the robot, and baking it in would tilt the
    robot by the floor's share everywhere else.

RUN -- supported robot, hand on SPACE
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    cd /home/robot01/Documents/can_motor_control/dog_stand_compliance_control/Aug_trotinplace_closedloop

    $V dog5_trot_quasi_static_model/trot_hw.py --self-test      # offline, no hardware
    $V dog5_trot_quasi_static_model/gait.py    --self-test      # the contact clock on its own
    $V dog5_trot_quasi_static_model/swing.py   --self-test      # the swing arc on its own
    THIS FILE'S OWN --self-test IS THE WEEK-2 STAND SET, PLUS THE THREE
    THINGS THIS FILE OWNS RATHER THAN dog5_trot_quasi_static_model/: the sweep-rate torque map
    (held to force_totorque's own answer, exactly), the swing arc's horizontal
    half, and the Raibert step that plans it.  The gait clock and the
    world-frame swing are still tested a module at a time, by the two above.

    PARAMETERS ARE EDITS, NOT FLAGS (workflow changed 2026-08-21).  Open
    dog5_trot_quasi_static_model/config.py, change the constant, save, and every hardware run
    is the SAME one line:

    sudo HOME=$HOME chrt -f 50 $V dog5_trot_quasi_static_model/trot_hw.py

    Every run logs itself: with no --log it writes run_<date>_<time>.npz to
    the cwd, and the npz embeds cfg.snapshot() -- every constant as imported
    plus config.py's own source text -- so the log says what flew without
    anyone writing it down.  `mv` a run worth keeping to a real name
    afterwards, and promote its values into config.py's provenance tables.
    --no-log is for torque-free bench fiddling and nothing else.

    --tau-max IS THE ONE TUNING FLAG LEFT: the staged cap, raised between
    otherwise-identical runs (and the one number worth lowering from the
    terminal in a hurry).  Every other parameter flag is GONE -- a flag that
    shadows config.py is a value the npz's cfg_* snapshot lies about.  What
    remains besides it is plumbing: --port, --log/--no-log, --park-on-stop,
    --self-test.

    THE STAND COMES FIRST, and its procedure is week 2's because the code is:
    tune on THIS runner, not on stand_torque_Mode.py, so that what you tuned
    is what then trots.  Do not press T until HOLD is quiet.  Each step below
    is an edit to config.py followed by the one line above:

    1. first run after any edit: leave TAU_START_MAX at 1.0, the low cap, and
       raise it toward TAU_STAGED_MAX (3.0) only after a quiet HOLD.  Do NOT
       zero FORCE_FRAC_DEFAULT with the feet on the floor -- that removes the
       only term holding the trunk up.
    2. the rig's resting attitude: run, read the rp: block the crouch prints,
       X out, and write rp:d into SETPOINT_ROLL_DEG / SETPOINT_PITCH_DEG.
       Do this BEFORE any run with attitude gains live.  (Done 2026-08-20:
       -0.29 / 0.12 is this rig; a different floor moves it.)
    3. the A/B that makes the STAND a measurement rather than a demo: one run
       as-is, then zero the roll/pitch entries of KP_ORI and KD_ORI and run
       again, pushing the trunk the SAME way in both.  Keep the setpoints
       identical across the pair or the ablation is confounded.  Two npz,
       each carrying its own config -- their cfg_* diff IS the record.
    4. only then the trot: ENTER, wait for HOLD, T.  The gait knobs are
       GAIT_PERIOD / DUTY / SWING_HEIGHT; move ONE per run, and duty 0.5
       removes the double support the contact ramp hands the load over in.

    THE YAW SPRING (new 2026-08-21) is KP_ORI[2], and it is the newest and
    least-tested loop in this file -- treat it the way the 2026-08-18 ladder
    treated the attitude pair.  Its reference is a LOCK, latched from the
    DETA10 heading the first sweep torque is live and retaken on every
    re-engage, so the error is zero at arming by construction and there is no
    step to ramp through.  Two things to know before turning it up:

      * yaw authority is 100% FRICTION.  A trot's stance is a diagonal pair
        and a pair makes a moment about the vertical only out of tangential
        force, so asking for more yaw moment spends the same cone budget the
        roll and pitch corrections are drawing on.  This is the coupling to
        watch: a yaw demand that the distributor has to clip shows up as
        every foot's tangential force moving at once.
      * the banner prints the ceiling (kp_yaw * YAW_ERR_MAX_RAD) because the
        error is clamped BEFORE it is multiplied.  Past the clamp a bigger
        heading error buys nothing at all.

    The A/B is KP_ORI[2] = 0 against the value under test, same floor, same
    session; the npz carries yaw, yaw_ref and kp_yaw, and tools_npz_to_csv
    emits yaw_err_deg already wrapped, so the two logs are directly
    comparable.  Note that yaw_err_deg is NOT readable in any npz written
    before 2026-08-21: those runs have no yaw arrays at all, and their
    cfg_KP_ORI[2] records a number that reached no multiply.

    FOOT PLACEMENT -- the only horizontal feedback this runner has -- is
    RAIBERT_ON / RAIBERT_KV in config.py, and OFF is the verified state.
    Off, the swing is purely vertical and nothing acts on x/y at all: the
    wrench has no kp there (no EKF, so no origin for a spring), so drift
    integrates unopposed -- t1..t8 all did.  The three-way A/B is off, on
    with RAIBERT_KV = 0 (symmetry term alone), and on with 0.03 -- three
    runs, same floor, same session, or the comparison is between two carpets.
    Then read step_xy against v in the CSV.  IF step_xy SITS ON THE CLAMP
    the banner prints, the height is the limit and not the gain: at the
    default 152 mm there are 10.2 mm of reach room, which corrects 0.068 m/s
    of drift and no more.  The height is ONE number: STAND_TRUNK_BOTTOM_M in
    config.py, floor to trunk bottom, what a ruler reads (0.140 buys 23.6 mm
    and 0.157 m/s).  STAND_HEIGHT is the geometry anchor, not the run height.

Keys: ENTER = rise / re-engage from limp, T = trot (FROM HOLD ONLY),
      SPACE = LIMP, P = park (FROM HOLD ONLY), X = stop.
      P IS DELIBERATELY DEAD ONCE TROTTING, and SPACE does not re-open it --
      LIMP does not leave the TROT stage.  0xA4 mid-swing is a lurch onto a
      diagonal, so the way out of a trot is X, and X alone cuts torque at
      height.  RUN THE TROT WITH --park-on-stop if you want X to put it down
      in position mode instead of handing you the weight.
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import sys
import time

import numpy as np

sys.setswitchinterval(0.0005)

_HERE = os.path.dirname(os.path.abspath(__file__))          # dog5_trot_quasi_static_model/
_AUG = os.path.dirname(_HERE)                               # Aug_trotinplace_...
_WEEK2 = os.path.join(_AUG, "august_week2")
_ROOT = os.path.dirname(_AUG)
_REPO = os.path.dirname(_ROOT)
# _HERE IS DELIBERATELY NOT ON THE PATH.  This directory holds a config.py and
# motorbus.py imports the repo's top-level one; putting ours in front breaks
# the CAN layer with `module 'config' has no attribute 'encoder_gain'`.  The
# week-2 modules this file is built on live in _WEEK2, and dog5_trot_quasi_static_model is
# reached as a PACKAGE from _AUG.
for _p in (_WEEK2, _AUG, os.path.join(_AUG, "torque_mode_control"),
           os.path.join(_ROOT, "dog5_description"), _REPO,
           os.path.join(_REPO, "IMU_sensor")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _dup in ("", ".", _HERE):
    while _dup in sys.path:
        sys.path.remove(_dup)


def _add_fdilink_root():
    """Find the `fdilink_imu` package WITHOUT trusting $HOME.

    imu_dog resolves it as Path.home()/Documents/IMU_sensor, which breaks under
    sudo ($HOME becomes /root).  RT priority wants root, so resolve it
    repo-relative first and fall back to the invoking user's home.
    """
    cands = [os.path.join(os.path.dirname(_REPO), "IMU_sensor"),
             os.path.join(_REPO, "IMU_sensor"),
             os.path.join(os.path.expanduser("~"), "Documents", "IMU_sensor")]
    if os.environ.get("SUDO_USER"):
        cands.append(os.path.join("/home", os.environ["SUDO_USER"],
                                  "Documents", "IMU_sensor"))
    for root in cands:
        if os.path.isdir(os.path.join(root, "fdilink_imu")):
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    return None


_add_fdilink_root()

import motorbus                                            # noqa: E402
import stand_dog5_hw as base                               # noqa: E402
import dog5_statics as st                                  # noqa: E402
# params.py IS NO LONGER THIS RUNNER'S TABLE.  dog5_trot_quasi_static_model/config.py is.
# It is imported only so cfg.assert_shared() can hold it to the constants
# the week-2 modules read INTERNALLY and no argument can override.
import params as _week2_params                             # noqa: E402
import crouch_and_park as cap                              # noqa: E402

from dog5_trot_quasi_static_model import gait as gait_mod                     # noqa: E402
from dog5_trot_quasi_static_model import config as cfg                        # noqa: E402
import feedback_estimator as fe                            # noqa: E402
import Dynamic_Model as dm                                 # noqa: E402
import force_totorque as ft                                # noqa: E402

MOTOR_IDS = base.MOTOR_IDS
N_JOINTS = cfg.N_JOINTS
LEGS = ft.LEGS
Q_CROUCH = cap.Q_CROUCH
ALL4 = np.ones(4, dtype=bool)


class JointImpedance:
    """tau = kp (q_ref - q) - kd qd, evaluated EVERY sweep.

    THE POINT, stated once because it is the whole safety argument:

        A pure force law tau = -J^T f is a VELOCITY-level system.  Take the
        ground away from a foot and there is nothing holding that joint
        anywhere -- qdd integrates without bound, and a damper only sets the
        terminal speed qd = tau/kd.  The joint does not stop, it coasts.  On
        2026-07-30 RR_abd did exactly that and the run ended on the overspeed
        e-stop.

        kp (q_ref - q) makes it POSITION-level with a fixed point.  The same
        fault becomes a bounded offset dq = tau/kp, ~13 mrad.

    It is NOT a position servo fighting the balance loop: q_ref tracks the
    same foot trajectory the wrench is trying to realise, so in nominal stance
    this term is ~0.  A model-mismatch term, not a stiffness.

    The gains are bounded by the sampled-PD limit at the MEASURED 16 ms loop
    delay, NOT at the 4 ms command interval: see params.LOOP_DELAY_S.  Using
    the command interval is what made kp=15/kd=0.6 look like 19% of the bound
    when it was 68% of it, and shook the robot at 9-12 Hz.
    """

    def __init__(self, kp=cfg.KP_IMP, kd=cfg.KD_IMP):
        if not 0.0 <= kp <= cfg.KP_IMP_MAX:
            raise ValueError(f"kp={kp} outside the measured stable envelope "
                             f"[0, {cfg.KP_IMP_MAX}] at a "
                             f"{cfg.SWEEP_S*1e3:.0f} ms sweep")
        if not 0.0 <= kd <= cfg.KD_IMP_MAX:
            raise ValueError(f"kd={kd} outside [0, {cfg.KD_IMP_MAX}]; the "
                             f"sampled-damper bound is 2J/dt = "
                             f"{2*cfg.J_MIN/cfg.LOOP_DELAY_S:.1f} Nms/rad at the "
                             f"MEASURED {cfg.LOOP_DELAY_S*1e3:.0f} ms loop "
                             f"delay, not {2*cfg.J_MIN/cfg.SWEEP_S:.1f} at the "
                             f"{cfg.SWEEP_S*1e3:.0f} ms command interval")
        self.kp, self.kd = float(kp), float(kd)
        self.dq = np.zeros(N_JOINTS)

    def tau(self, q, qd, q_ref):
        self.dq = np.asarray(q_ref) - np.asarray(q)
        return self.kp * self.dq - self.kd * np.asarray(qd)

    def worst(self):
        return float(np.max(np.abs(self.dq)))


class TorqueGate(base.SafetyGate):
    """base.SafetyGate with the TORQUE-mode ramp and slew.

    base.apply() reads module-level TORQUE_RAMP_S (1.0 s) and TAU_SLEW_NM_S
    (5.0 Nm/s), which are position-track numbers and are wrong here in both
    directions: 5 Nm/s is slow enough that a leg folds while the impedance is
    still ramping towards the torque it asked for, and a 1.0 s ramp on top
    means no leg can take weight for a full second.  See params.TAU_SLEW_NM_S
    for why the sizing argument is now the joint-speed one and not the old
    kp = 15 one.

    Subclassed rather than edited, because the position track, the crawl and
    the EKF harness all import stand_dog5_hw.
    """

    def overspeed_reason(self, qd, q, now):
        """DISABLED for week 2: record the peak, never stop the run on it.

        Removed on request after it ended a hardware run at
        `overspeed FL_abd: +8.1 rad/s over the 8.0 rad/s hard limit`.

        The hard tier is the one that fired, and it trips on a SINGLE reading
        with no encoder confirmation -- unlike the 7.0 rad/s sustained tier,
        which needs three consecutive checks agreeing with a finite-difference
        of the encoder.  stand_dog5_hw's own comment records a 5.9 rad/s
        nuisance reading from the driver's speed field, so a lone 8.1 is far
        more likely to be that field glitching than a joint genuinely turning
        at 464 deg/s.

        WHAT STILL BOUNDS JOINT SPEED, with this gone:
            the impedance      -kd*qd every sweep, so speed is opposed
                               continuously rather than watched for
            the slew limit     one sweep at 60 Nm/s adds 0.11 rad/s
            the torque cap     --tau-max, 1.0 Nm by default
            the abduction sign a leg that runs away crosses zero and stops
            the tilt e-stop    a leg that runs away tips the trunk
        What is NOT covered any more: a joint spinning fast while the trunk
        stays level and the abduction sign stays correct.

        The peak is still accumulated so the exit report can show what the
        speeds actually were -- that is the number that says whether 8.1 was a
        glitch or real, and it is worth having even with the trip gone.
        """
        self.qd_peak = np.maximum(self.qd_peak, np.abs(np.asarray(qd, float)))
        return None

    def cap_now(self, now):
        f = np.clip((now - self.started_at) / cfg.TORQUE_RAMP_S, 0.0, 1.0)
        return self.tau_cap * f

    def apply(self, tau, q, now):
        cap = self.cap_now(now)
        out = np.clip(np.asarray(tau, dtype=float), -cap, cap)
        out = np.clip(out, -cfg.TAU_HARD_NM, cfg.TAU_HARD_NM)
        dt = np.clip(now - self.last_time, 1e-4, 0.05)
        step = cfg.TAU_SLEW_NM_S * dt
        out = self.previous_tau + np.clip(out - self.previous_tau, -step, step)
        self.previous_tau = out
        self.last_time = now
        return out


def _smoothstep(u):
    """3u^2 - 2u^3 on [0,1].  Zero SLOPE at both ends, which is the point."""
    u = min(1.0, max(0.0, float(u)))
    return u * u * (3.0 - 2.0 * u)


def raibert_step_body(i, v_world, C, t_stance, kv, z_des, foot_xy,
                      v_cmd=(0.0, 0.0)):
    """Trunk-frame x/y offset from a foot's resting site to where it should
    LAND, clamped to the leg's reach.  (2,)

    THIS IS THE ONLY THING IN THE RUNNER THAT ACTS ON HORIZONTAL DRIFT, and
    until it existed nothing did.  The wrench cannot: Dynamic_Model.body_wrench
    has no kp on x or y -- leg odometry measures VELOCITY and has no origin, so
    there is no position for a spring to pull against -- and every t*.npz was
    logged with kd_xy = 0, which makes W[0] and W[1] identically zero in all of
    them.  (W[5] was zero in the t*.npz too, but is NOT any more: yaw got a
    spring on 2026-08-21 and x/y did not, because a heading is measured and an
    x/y position still is not.)  A foot placed DOWNSTREAM of the drift is the
    standard answer
    and needs no absolute position: it needs the velocity, which IS measured.

        step = v * (t_stance / 2)  +  kv * (v - v_cmd)

    The first term is Raibert's symmetry term and is NOT a gain: it is where
    the foot has to land for the stance to be symmetric about the hip, so the
    body decelerates over the second half by as much as it accelerated over the
    first.  It follows from t_stance, which is DUTY * period and already known.
    kv is the only tunable number here, and `kv = 0` -- symmetry term alone --
    is therefore the honest A/B against RAIBERT_ON = False entirely.

    v IS ROTATED INTO THE TRUNK FRAME because the arc and `foot_xy` live there.
    C is the estimator's roll/pitch-only rotation (I->B), so this rotation
    carries NO YAW.  That stayed true when the heading became a controlled
    axis on 2026-08-21: the wrench reads the yaw ANGLE beside C, never inside
    it, precisely so that this frame -- and the leg odometry that shares it --
    does not start rotating with the heading.  Placement still cannot correct
    a yaw drift, only an x/y one; the yaw spring is what answers that now.

    THE CLAMP HOLDS z AND SHORTENS THE STEP, which swing.py's does not: that
    one scales the whole hip-to-foot vector, and so lifts the landing point off
    the floor by the same fraction it pulls it in.  On a flat floor the height
    is not negotiable and the length is, so the reachable set is intersected at
    the LANDING PLANE -- a horizontal disc of radius sqrt((0.95 R)^2 - dz^2)
    about the hip.  At the nominal stand height that disc leaves ~6 mm
    (config.MAX_FORWARD_V_AT_STAND_HEIGHT), which is why the honest way to buy
    a longer step is to crouch and not to raise the clamp.
    """
    C = np.asarray(C, dtype=float).reshape(3, 3)
    v_b = C @ np.asarray(v_world, dtype=float).reshape(3)
    step = (v_b[:2] * (0.5 * float(t_stance))
            + float(kv) * (v_b[:2] - np.asarray(v_cmd, dtype=float).reshape(2)))

    z_site = -(fe.hip_from_imu(z_des) - cfg.FOOT_RADIUS_M)
    hip = np.asarray(cfg.HIP_OFFSET[i], dtype=float)
    dz = z_site - hip[2]
    r_max_sq = (0.95 * cfg.LEG_REACH) ** 2 - dz * dz
    if r_max_sq <= 0.0:
        # The leg is already at full stretch straight down; there is no
        # horizontal room at all.  Not reachable at any sane height, but a
        # negative sqrt here would be a NaN target fed to a torque.
        return np.zeros(2)

    # INWARD BEFORE REACH: a leg's reachable disc extends far PAST the
    # midline, so the reach clamp alone admits a stance-narrowing step five
    # times the outward room.  The inward cap is a stability bound, not a
    # workspace one, and it binds first.
    inward = -float(cfg.SIDE_SIGN[i]) * step[1]
    if inward > cfg.STEP_INWARD_MAX_M:
        step[1] = -float(cfg.SIDE_SIGN[i]) * cfg.STEP_INWARD_MAX_M
    land = np.asarray(foot_xy[i], dtype=float) + step
    d = land - hip[:2]
    r = float(np.linalg.norm(d))
    r_max = math.sqrt(r_max_sq)
    if r > r_max:
        land = hip[:2] + d * (r_max / r)
    return land - np.asarray(foot_xy[i], dtype=float)


def _step_room(z_des, i=0, foot_xy=None):
    """How far, in m, a foot at `z_des` can be placed from its resting site
    before raibert_step_body's clamp truncates it.

    DERIVED FROM THE HEIGHT ACTUALLY BEING FLOWN, not read from
    config.MAX_FORWARD_V_AT_STAND_HEIGHT: that constant is computed once at the
    nominal stand pose and the height is a knob, and the number more than doubles
    over 12 mm of crouch.  A banner quoting the nominal figure during a lower
    run would be worse than quoting nothing.

    z_des IS THE TRUNK BOTTOM, as everywhere in this file.  config.py's step
    table is in the OLD HIP-AXIS FRAME -- its 0.190 row is this frame's 0.152,
    the same 38 mm that renamed STAND_HEIGHT -- so its heights must not be fed
    here.  The room is computed from geometry rather than looked up in it.

    THE FIGURE IS THE WORST CASE, radially outward from the hip: the clamp
    admits any landing point inside the disc, so a step INWARD has more room
    than this.  The banner should under-promise.
    """
    if foot_xy is None:
        foot_xy = [st.leg_frames(LEGS[k], Q_CROUCH[3*k:3*k+3])[0][:2]
                   for k in range(4)]
    z_site = -(fe.hip_from_imu(z_des) - cfg.FOOT_RADIUS_M)
    hip = np.asarray(cfg.HIP_OFFSET[i], dtype=float)
    dz = z_site - hip[2]
    r_max_sq = (0.95 * cfg.LEG_REACH) ** 2 - dz * dz
    if r_max_sq <= 0.0:
        return 0.0
    rest = float(np.linalg.norm(np.asarray(foot_xy[i], dtype=float) - hip[:2]))
    return max(0.0, math.sqrt(r_max_sq) - rest)


def swing_foot_body(i, s, z_des, foot_xy, height=cfg.SWING_HEIGHT,
                    step_xy=(0.0, 0.0)):
    """Where a swinging foot should be, in the TRUNK frame.  (p (3,), v (3,))

    THE FRAME IS THE BODY'S, and the arc is vertical unless `step_xy` says
    otherwise.  The foot's x/y target starts at its own crouch value -- the
    same `foot_xy` q_ref_for_height pins the stance feet at -- and travels
    `step_xy` over the swing, so a zero step is exactly the old behaviour:
    the foot lifts and lands on the spot it left.  step_xy defaults to zero so
    that a caller that does not plan a step gets the arc every t*.npz was
    logged with, bit for bit, rather than a subtly different one.

    THE HORIZONTAL IS A SMOOTHSTEP OVER THE WHOLE SWING, not a constant offset
    applied from s = 0.  A constant offset steps p_des at liftoff, and p_des
    goes straight into a 200 N/m impedance: a 6 mm jump is 1.2 N appearing in
    one sweep, on the leg with the least authority to absorb it.  Ramping also
    puts the horizontal speed at zero at BOTH ends, which is what keeps the
    step from scuffing on touchdown.

    The bump is a smoothstep up over the first half and down over the second,
    so the vertical speed is ZERO at liftoff, at the apex and at TOUCHDOWN.
    A sine bump, the usual shortcut, arrives at pi*h/T = 0.63 m/s straight
    into the floor.

    z_des is FLOOR to TRUNK BOTTOM, as everywhere else in this file, so the
    resting site sits at -(hip_from_imu(z_des) - FOOT_RADIUS_M) -- the same
    two conversions q_ref_for_height makes, made once more here rather than
    assumed.
    """
    z_site = -(fe.hip_from_imu(z_des) - cfg.FOOT_RADIUS_M)
    if s < 0.5:
        b, db = _smoothstep(2 * s), 6 * (2 * s) * (1 - 2 * s) * 2
    else:
        u = 2 - 2 * s
        b, db = _smoothstep(u), -6 * u * (1 - u) * 2
    u = min(1.0, max(0.0, float(s)))
    g, dg = _smoothstep(u), 6 * u * (1 - u)
    dx, dy = float(step_xy[0]), float(step_xy[1])
    p = np.array([foot_xy[i][0] + dx * g,
                  foot_xy[i][1] + dy * g,
                  z_site + height * b])
    v = np.array([dx * dg, dy * dg, height * db])  # per unit swing phase
    return p, v


def swing_torque(i, q_i, qd_i, p_des, v_des_phase, dsdt,
                 kp=cfg.KP_SWING[0, 0], kd=cfg.KD_SWING[0, 0]):
    """Cartesian impedance on one swinging foot, in the TRUNK frame.  (3,)

    THE LOOP NO LONGER CALLS THIS -- leg_torque_block inlines it, so that one
    leg_frames call serves the Jacobian, the impedance and leg gravity instead
    of three.  It is kept because it is the REFERENCE the self-test holds that
    inlining to: two expressions of the same law, checked against each other,
    beat one expression checked against nothing.

        tau = J^T ( kp (p_des - p) + kd (v_des - J qd) )

    ON TOP OF the leg-gravity term force_totorque already gives a non-planted
    leg, and INSTEAD of the -J^T f a planted one gets.  It is a scalar kp/kd
    rather than a 3x3 because a trot in place has no reason to be stiffer in
    one direction than another, and a diagonal matrix that is secretly one
    number invites the reader to think otherwise.
    """
    fr = st.leg_frames(LEGS[i], q_i)
    J = st.foot_jacobian_from(fr[0], fr[1], fr[2])
    v = J @ qd_i
    f = kp * (p_des - fr[0]) + kd * (v_des_phase * dsdt - v)
    return J.T @ f


def leg_torque_block(q, qd, f_foot, planted, g_down, p_des=None,
                     v_des_phase=None, dsdt=0.0, leg_gravity=True,
                     kd_joint=cfg.KD_JOINT_STANCE,
                     kp_sw=cfg.KP_SWING[0, 0], kd_sw=cfg.KD_SWING[0, 0]):
    """Every leg's feedforward torque from THIS SWEEP'S q.  (12,)

    THE FORCE IS WHAT THE MODEL BLOCK HOLDS, NOT THE TORQUE.  tau = -J(q)^T f
    was being computed once per model block and reused for three sweeps, which
    froze J(q) for 12 ms.  J is a function of the joint angles ALONE and costs
    four leg_frames -- 606 us for all four legs, measured -- so there is no
    reason for it to be stale when q is not.  The wrench, the QP-free
    distribution and the friction clamp all still run at 83 Hz, because they
    need the ESTIMATOR; the map from their answer onto the joints does not.

    WHY THIS MATTERS MORE FOR A TROT THAN FOR A STAND: a stance leg on a stand
    moves microns between model blocks and a stale J is worth nothing.  A trot
    lands a foot and unloads another inside those same 12 ms, and the swing
    leg -- whose Cartesian impedance is a DAMPER, and a damper fed a 12 ms old
    velocity is a phase lag, not a damper -- moves a centimetre.

    The three branches are exactly the three the 83 Hz code had, kept together
    so a leg cannot fall through all of them:
      planted   -J^T f  and the stance joint damper, as force_totorque does
      swinging  the Cartesian impedance, as swing_torque does
      always    its own links' weight, which neither branch may skip

    `f_foot` is (4, 3) in the BODY frame and is ALREADY scaled by force_frac:
    that scaling belongs to the force, and leg gravity must not inherit it --
    an unloaded leg still has to hold itself up.  A swinging leg's row is
    zero and is never read.

    `g_down` None means UNTILTED gravity, which is what the estimator-refused
    fallback wants: with f_foot zeroed as well this returns exactly
    force_totorque.leg_gravity_only(q), and the self-test holds it to that.
    """
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    f_foot = np.asarray(f_foot, dtype=float).reshape(4, 3)
    tau = np.zeros(N_JOINTS)
    for i in range(4):
        sl = slice(3 * i, 3 * i + 3)
        leg = LEGS[i]
        fr = st.leg_frames(leg, q[sl])
        if planted[i]:
            J = st.foot_jacobian_from(fr[0], fr[1], fr[2])
            tau[sl] = -J.T @ f_foot[i] - kd_joint * qd[sl]
        elif p_des is not None:
            J = st.foot_jacobian_from(fr[0], fr[1], fr[2])
            v = J @ qd[sl]
            f = kp_sw * (p_des[i] - fr[0]) + kd_sw * (v_des_phase[i] * dsdt - v)
            tau[sl] = J.T @ f
        if leg_gravity:
            tau[sl] += st.leg_gravity_torque_tilted(leg, q[sl], g_down,
                                                    frames=fr)
    return tau


def q_ref_for_height(q_ref, z_des, foot_xy):
    """Warm-started IK: the joint pose that puts every foot at -z_des.

    This is what makes the impedance a MODEL-MISMATCH term rather than a
    stiffness fighting the rise.  Last week's runner froze q_ref at the crouch
    for the whole 8 s rise, so by the top the impedance was pulling ~150 mm of
    error against the force law -- the two layers were fighting.

    `z_des` is FLOOR to TRUNK BOTTOM, the same quantity the estimator's
    fk_trunk_height now returns and therefore the same one body_wrench takes
    the error of.  The legs, though, are tabulated from the HIP AXIS, so this
    is the one place the two frames meet and fe.hip_from_imu is the only
    converter.  Then the foot SITE is the centre of a 20 mm sphere, so it sits
    at -(z_hip - FOOT_RADIUS_M).

    Two silent biases live in those two lines -- 38 mm if the frame is skipped,
    20 mm if the foot radius is -- and neither shows up as anything but a
    height that reads right and measures wrong.  That is exactly what happened
    on 2026-08-17 with the frame.

    WARM-STARTED, AND IT HAS TO BE.  The step is damped least squares, not a
    plain Newton step: near full leg extension the Jacobian is ill-conditioned
    and Newton diverges -- from the crouch straight to the stand height it
    overshoots by 330 mm and leaves the workspace entirely.  Over a real rise
    that never arises, because at 83 Hz across 8 s each call moves the target
    0.25 mm and starts from the previous solution.  The damping is what makes
    the first call after a stage change safe anyway.

    x and y stay pinned at their crouch values, so the feet rise straight up
    and the contact point never moves in the world.
    """
    z_site = -(fe.hip_from_imu(z_des) - cfg.FOOT_RADIUS_M)
    for i, leg in enumerate(LEGS):
        sl = slice(3 * i, 3 * i + 3)
        qi = q_ref[sl].copy()
        tgt = np.array([foot_xy[i][0], foot_xy[i][1], z_site])
        for _ in range(8):
            fr = st.leg_frames(leg, qi)
            e = tgt - fr[0]
            if float(np.linalg.norm(e)) < 1e-9:
                break
            J = st.foot_jacobian_from(fr[0], fr[1], fr[2])
            qi = qi + J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e)
        q_ref[sl] = qi
    return q_ref


def _zero_rp_streamer(imu, args, window_s=5.0):
    """The ZERO-TORQUE stage streams the AHRS, not the twelve joint angles.

    The joint line that used to print here was the wrong thing to look at:
    the encoders are calibrated, the preflight's own soft-limit check already
    REFUSES ENTER on a bad pose, and there is nothing an operator does with
    twelve numbers at 2 Hz.  The one number that has to be read off this robot
    by hand is the IMU mount tilt -- and this is the stage to read it in, with
    no torque anywhere and the trunk wherever the operator puts it.

    So this prints raw roll/pitch, a mean over the last `window_s`, and the
    SETPOINT_* pair that cancels the mean.  The SPREAD is printed with it
    because it is the part that says whether the mean is trustworthy: held by
    hand it runs a degree or more, and a mean that moves by a degree is not a
    mount tilt.

    THIS IS THE MOUNT TILT ONLY IF THE TRUNK IS ACTUALLY LEVEL -- put a spirit
    level on it, or the reading is whatever the hands are doing.  The reading
    that needs no level is rp:d in the crouch (ahrs minus the foot plane),
    which cancels the floor as well; see _print_attitude_report.
    """
    hist = collections.deque(maxlen=max(2, int(window_s * 2.0)))  # 2 Hz prints

    def line(q, qd):
        sample = imu.sample()
        if sample is None:
            return "[zero] rp: no AHRS sample"
        r, p_ = float(sample.roll_deg), float(sample.pitch_deg)
        hist.append((r, p_))
        rs = [h[0] for h in hist]
        ps = [h[1] for h in hist]
        rm, pm = sum(rs) / len(rs), sum(ps) / len(ps)
        spread = max(max(rs) - min(rs), max(ps) - min(ps))
        # The printed pair is ABSOLUTE (raw AHRS, no setpoint in it): paste it
        # into config.py as written, do not add it to the old values.
        return (f"[zero] rp {r:+6.2f} / {p_:+6.2f} deg   "
                f"mean({len(hist)/2:.1f}s) {rm:+6.2f} / {pm:+6.2f} "
                f"+/-{spread:.2f}   config.py: SETPOINT_ROLL_DEG = {rm:.2f}  "
                f"SETPOINT_PITCH_DEG = {pm:.2f}")

    return line


def _print_attitude_report(est, args):
    """The one place an operator can read the mount tilt off the robot.

    Printed once, in the crouch, with the feet on the floor and no torque
    anywhere -- the only moment in the run when "the trunk is not tilted" is
    something we can assume rather than something the loop is enforcing.

        rp:ahrs   the DETA10, with the SETPOINT_* pair already subtracted
        rp:fk     a least-squares plane through the four measured feet.  No
                  IMU in it at all, so it is the independent number.
        rp:d      ahrs - fk = floor slope + IMU mount tilt + encoder zeros

    rp:d is what a setpoint should cancel, and it is three constants added
    together, which is why this prints the arithmetic instead of latching it:
    rotate the robot 180 deg on the same spot and read again -- the floor's
    share flips sign, the mount's does not.
    """
    dr, dp = est.attitude_residual()
    print("[stand] " + "-" * 66)
    print(f"[stand] rp:ahrs {np.degrees(est.roll):+6.2f} / "
          f"{np.degrees(est.pitch):+6.2f} deg"
          + (f"   (raw {np.degrees(est.roll_raw):+.2f} / "
             f"{np.degrees(est.pitch_raw):+.2f}, setpoint "
             f"{args.setpoint_roll:+.2f} / {args.setpoint_pitch:+.2f})"
             if (args.setpoint_roll or args.setpoint_pitch) else
             "   (no setpoint subtracted)"))
    print(f"[stand] rp:fk   {np.degrees(est.roll_fk):+6.2f} / "
          f"{np.degrees(est.pitch_fk):+6.2f} deg   "
          f"(foot plane, no IMU)")
    print(f"[stand] rp:d    {np.degrees(dr):+6.2f} / {np.degrees(dp):+6.2f} "
          f"deg   = floor slope + mount tilt + encoder zeros")
    if max(abs(np.degrees(dr)), abs(np.degrees(dp))) > 0.2:
        print(f"[stand] ^ to make the wrench see THIS as level, set in "
              f"config.py and re-run:")
        print(f"[stand]     SETPOINT_ROLL_DEG = "
              f"{args.setpoint_roll + np.degrees(dr):.2f}   "
              f"SETPOINT_PITCH_DEG = "
              f"{args.setpoint_pitch + np.degrees(dp):.2f}")
        print(f"[stand]   but that folds the FLOOR in too.  Rotate the robot "
              f"180 deg on the same")
        print(f"[stand]   spot and read again to split them: half the sum is "
              f"the mount.")
    print("[stand] " + "-" * 66)


def run(args):
    base.validate_hardware_config()
    gains = cfg.gains()
    if args.kp_att is not None:
        gains.kp_roll = gains.kp_pitch = args.kp_att
    if args.kd_att is not None:
        gains.kd_roll = gains.kd_pitch = args.kd_att
    if args.kp_z is not None:
        gains.kp_z = args.kp_z
    if args.kd_z is not None:
        gains.kd_z = args.kd_z
    if args.kd_xy is not None:
        gains.kd_x = gains.kd_y = args.kd_xy
    if args.kd_yaw is not None:
        gains.kd_yaw = args.kd_yaw
    if args.open_loop:
        # ALL outer feedback off: W = [0, 0, m*g, 0, 0, 0], split evenly by
        # the grasp map, + leg gravity + the joint impedance tracking the
        # z ramp.  That is the old Cartesian compliance stand, expressed in
        # joint space -- the known-good baseline.  This is the diagnostic A/B
        # for the 2026-08-17 shake: if THIS shakes, the fault is below the
        # model (torque gain, current loop, impedance, CAN timing) and no
        # controller swap can fix it; if it is smooth, add loops back one at
        # a time (kp_z first, attitude last) to find the one that chatters.
        gains.kp_z = gains.kd_z = 0.0
        gains.kp_roll = gains.kd_roll = 0.0
        gains.kp_pitch = gains.kd_pitch = 0.0
        gains.kd_x = gains.kd_y = gains.kd_yaw = 0.0
        # kp_yaw with the rest of them.  OPEN_LOOP's whole claim is
        # W = [0, 0, m*g, 0, 0, 0], and a live yaw spring would leave W[5]
        # non-zero while the banner said every loop was off -- the exact
        # failure kd_yaw's own comment in Dynamic_Model records.
        gains.kp_yaw = 0.0
    ablated = gains.kp_roll == 0.0 and gains.kd_roll == 0.0

    print("=" * 74)
    print("DOG5 CLOSED-LOOP TORQUE STAND  (week 2)")
    print(f"  mass {args.mass:.4f} kg = {args.mass*cfg.GRAVITY:.1f} N, "
          f"stand height {args.height*1e3:.0f} mm, rise {cfg.T_RISE:.0f} s")
    # Say the frame every run.  The 2026-08-17 gap between a printed 191 mm
    # and a measured 160 was this line not existing.
    print(f"  height is FLOOR to TRUNK BOTTOM (the IMU board) -- a ruler "
          f"reaches it.  That")
    print(f"  is {fe.hip_from_imu(args.height)*1e3:.0f} mm at the hip axis, "
          f"{cfg.IMU_BELOW_TRUNK_ORIGIN_M*1e3:.0f} mm higher, which is the "
          f"frame the leg IK uses.")
    if args.setpoint_roll or args.setpoint_pitch:
        print(f"  AHRS setpoint {args.setpoint_roll:+.2f} / "
              f"{args.setpoint_pitch:+.2f} deg subtracted from every reading")
    else:
        print(f"  AHRS setpoint 0.00 / 0.00 -- the IMU MOUNT'S OWN TILT is "
              f"being read as a real")
        print(f"  attitude.  The crouch prints rp:d; write it into config.py "
              f"as SETPOINT_ROLL_DEG / SETPOINT_PITCH_DEG.")
    print(f"  {gains}, impedance kp={args.kp} kd={args.kd}, "
          f"tau cap {args.tau_max} Nm, force_frac {args.force_frac}")
    print(f"  model at {cfg.CONTROL_HZ/cfg.MODEL_EVERY:.0f} Hz "
          f"(every {cfg.MODEL_EVERY} sweeps), impedance at {cfg.CONTROL_HZ:.0f} Hz")
    # SAY WHICH SIDE OF THE HORIZONTAL A/B THIS RUN IS ON, every run.  The
    # gains line above prints kd_xy, and a reader who sees kd_xy = 0 has been
    # told that x/y is open loop -- which stopped being the whole story the
    # moment placement existed.  Both halves or neither.
    _room = _step_room(args.height)
    if args.raibert is None:
        print(f"  FOOT PLACEMENT OFF: the swing is purely vertical, so with "
              f"kd_xy = {gains.kd_x:.0f} there is")
        print(f"  NOTHING acting on x or y -- that drift integrates. "
              f"RAIBERT_ON = True in config.py turns it on.")
    else:
        _vmax = _room / (0.5 * args.duty * args.period + args.raibert)
        print(f"  foot placement ON: kv={args.raibert} s, "
              f"step = v*{0.5*args.duty*args.period*1e3:.0f}ms + "
              f"kv*(v-v_cmd), clamped at {_room*1e3:.1f} mm")
        print(f"  at this height -- i.e. it stops correcting above "
              f"{_vmax:.2f} m/s of drift.  Crouch to buy more.")
    # The roll axis saturates first, and on this rig it saturates EARLY.
    Mx_max, My_max = ft.moment_capacity(
        [st.leg_frames(LEGS[i], Q_CROUCH[3*i:3*i+3])[0] for i in range(4)],
        list(LEGS), args.mass * cfg.GRAVITY)
    print(f"  stance moment capacity: roll {Mx_max:.1f} Nm "
          f"(kp_roll saturates at {np.rad2deg(Mx_max/max(gains.kp_roll,1e-9)):.1f} deg), "
          f"pitch {My_max:.1f} Nm")
    # THE YAW AXIS SAYS OUT LOUD WHICH OF ITS TWO STATES IT IS IN.  It was
    # damping-only for every run before 2026-08-21, and a log from either era
    # looks the same from the outside, so the banner names the era.
    if gains.kp_yaw:
        print(f"  yaw: SPRING LIVE, kp_yaw {gains.kp_yaw:.0f} Nm/rad on the "
              f"DETA10 heading, error clamped")
        print(f"  at {np.degrees(gains.yaw_err_max):.0f} deg -> at most "
              f"{gains.kp_yaw*gains.yaw_err_max:.2f} Nm, made entirely of "
              f"tangential force.")
        print(f"  The lock is latched when torque arms, NOT at ENTER, and is "
              f"retaken on every re-engage.")
    else:
        print(f"  yaw: damping only (kd_yaw {gains.kd_yaw:.1f}), heading not "
              f"held -- KP_ORI[2] turns the spring on.")
    if args.open_loop:
        print("  OPEN LOOP: W = [0, 0, m*g, 0, 0, 0] always -- even split + "
              "leg gravity + impedance.")
        print("  This is the old Cartesian compliance stand in joint space. "
              "If THIS shakes, the fault is below the model.")
    elif ablated:
        print("  ATTITUDE GAINS ZEROED -- the ablation half of the A/B; "
              "the trunk will NOT push back")
    if args.force_frac < 0.2:
        print(f"  WARNING: force_frac {args.force_frac} removes the term that "
              f"holds the TRUNK up.  Feet on the floor, the legs fold until "
              f"kp*dq carries {args.mass*cfg.GRAVITY:.0f} N.")
    print(f"  trot: period {args.period:.2f} s, duty {args.duty:.2f} "
          f"(stance {args.period*args.duty*1e3:.0f} ms, swing "
          f"{args.period*(1-args.duty)*1e3:.0f} ms), swing height "
          f"{args.swing_height*1e3:.0f} mm")
    print("  Support the robot.  ENTER = rise, T = trot, SPACE = LIMP, "
          "P = park, X = stop")
    print("-" * 74)
    print("  what the status line means:")
    print(fe.BodyState.status_legend())
    print("  torque, tracking and impedance error are no longer streamed -- "
          "they say nothing")
    print("  a human can act on at 2 Hz.  The foot-load sum is in the EXIT "
          "REPORT, and every")
    print("  one of them is in the --log npz per sweep.")
    print("=" * 74)

    from imu_dog import ImuDog, DEFAULT_PORT                # noqa: PLC0415
    imu = ImuDog(port=args.port or DEFAULT_PORT)
    imp = JointImpedance(args.kp, args.kd)
    key = base.KeyPoller()
    log = []
    stop = None
    web, web_stop = None, None

    try:
        imu.start()
        if args.web:
            # Display only: daemon threads that READ the AHRS and a status
            # dict this loop rebinds.  A dead port must not cost a run.
            from dog5_trot_quasi_static_model import att_web  # noqa: PLC0415
            web = att_web.AttShared(imu)
            try:
                web_stop, _web_urls = att_web.start(web)
            except OSError as exc:
                print(f"[web] page disabled ({exc}); the run continues")
                web = None
            else:
                web.status = {"stage": "PREFLIGHT"}
                for _u in _web_urls:
                    print(f"[web] attitude stream at {_u}")
        with motorbus.MotorBus(MOTOR_IDS, dirs=base.MOTOR_DIRECTIONS) as mb:
            if not mb.arm(rate_hz=cfg.CONTROL_HZ):
                raise RuntimeError("arm failed (bus / power / terminators)")
            unwrap = [base.CalibratedEncoderUnwrap() for _ in MOTOR_IDS]
            if not imu.wait_for_data(3.0):
                raise RuntimeError("no AHRS data")
            base._zero_torque_preflight(
                mb, key, unwrap, status=_zero_rp_streamer(imu, args))

            # ---- CROUCH: position mode, week-2 bookend -------------------
            q = cap.crouch(mb, unwrap, key, max_speed_dps=args.crouch_dps)

            # min_planted = 2, AND THAT IS THE ONE ESTIMATOR CHANGE A TROT
            # NEEDS.  params.MIN_PLANTED = 3 is a STAND's alarm: a stand that
            # has lifted a foot has already gone wrong.  A trot lifting two is
            # the trot working, and leg odometry does not need three anyway --
            # v = -C^T(omega x s_i + J_i qd_i) is solved by ONE planted foot,
            # more only average.  Left at 3 the estimator refuses every sweep
            # of the trot and the runner limps instantly.
            #
            # WHAT IS LOST: fk_attitude needs 3+ feet for a plane and returns
            # NaN on two, so the IMU-free attitude cross-check is unavailable
            # DURING the trot.  It still reads in the crouch and in HOLD,
            # where all four are down, which is where it is acted on.  The NaN
            # is diagnostic only -- state["C"] is the AHRS's, so nothing NaN
            # reaches a torque.
            est = fe.BodyState(imu,
                               setpoint_roll=np.radians(args.setpoint_roll),
                               setpoint_pitch=np.radians(args.setpoint_pitch),
                               min_planted=2)
            gate = TorqueGate(tau_cap=args.tau_max)
            miss = base.CanMissMonitor(mb)
            foot_xy = [st.leg_frames(LEGS[i], Q_CROUCH[3*i:3*i+3])[0][:2]
                       for i in range(4)]

            # HALF A CYCLE IS THE WHOLE SWAP.  Derived from PHASE_OFFSET
            # rather than written out, so retuning that array cannot leave the
            # two choices disagreeing about which legs are a diagonal.
            offsets = np.mod(cfg.PHASE_OFFSET
                             + (0.5 if args.lead_diagonal == "fl-rr" else 0.0),
                             1.0)
            gait = gait_mod.TrotGait(args.period, args.duty, offsets)
            planted = ALL4
            stage, t0 = "WAIT", time.perf_counter()
            z_crouch = None
            q_ref = q.copy()
            tau_ff = np.zeros(N_JOINTS)
            W = np.zeros(6)
            tau_cmd = np.zeros(N_JOINTS)
            tau_des = np.zeros(N_JOINTS)
            # The foot-placement latch.  step_xy is the trunk-frame offset
            # THIS swing was planned with, held for the whole swing; was_swing
            # is what turns a swinging leg into a liftoff EDGE.  Both are also
            # logged, because a placement run whose log cannot say what step
            # was asked for is only readable as "it drifted less, somehow".
            step_xy = np.zeros((4, 2))
            was_swing = np.zeros(4, dtype=bool)
            # the placement's own slow velocity -- see RAIBERT_V_TAU_S
            v_place = np.zeros(3)
            t_place = None
            # WHAT THE MODEL BLOCK HANDS THE SWEEP.  f_foot is the per-foot
            # BODY-frame force the distributor produced, already scaled by
            # force_frac; g_down is the tilt leg gravity is taken against;
            # model_live says the estimator was answering when they were set.
            # The torque itself is no longer held -- see leg_torque_block.
            f_foot = np.zeros((4, 3))
            g_down = None
            model_live = False
            z_des = None
            diag = {"fz": [0.0] * 4, "stance": [], "singular": []}
            state, act, why = None, False, "starting"
            limp = False
            armed = False
            load_sum = float("nan")
            # THE HEADING THE YAW SPRING PULLS TOWARDS.  None means "not
            # latched yet"; the model block takes it the first sweep torque is
            # live, and LIMP puts it back to None so a re-engage locks onto
            # wherever the robot ended up rather than fighting back to a
            # heading it was carried away from.
            yaw_ref = None

            slot = mb.slot(cfg.CONTROL_HZ)
            deadline = time.perf_counter() + slot
            index = sweep = 0
            last_print = 0.0
            # qd comes from the ENCODER, not the driver's speed field -- see
            # params.QD_ALPHA for the hardware run that made that necessary.
            qd = np.zeros(N_JOINTS)
            q_prev, t_prev = None, time.perf_counter()
            qd_drv = np.zeros(N_JOINTS)
            glitches = np.zeros(N_JOINTS, dtype=int)
            print("[stand] WAIT: hold still, then ENTER to rise under torque")

            while stop is None:
                mb.poll()
                j = index % N_JOINTS
                if j == 0:
                    now = time.perf_counter()
                    q, qd_drv = base._joint_state(mb, unwrap)
                    tau_meas = np.array([mb.torques_nm()[m] for m in MOTOR_IDS])

                    # ENCODER-DIFFERENCED velocity, low-passed -- the same
                    # thing stand_dog5_recorded_hw.EncoderVelocity does, and
                    # the reason that controller stands without shaking.  The
                    # driver's field is still read, but only to COUNT how
                    # often it disagrees, so the exit report can say whether
                    # the glitching is still happening.
                    if q_prev is not None:
                        dt_v = now - t_prev
                        if 1e-4 < dt_v < 0.05:
                            qd += cfg.QD_ALPHA * ((q - q_prev) / dt_v - qd)
                    q_prev, t_prev = q.copy(), now
                    glitches += (np.abs(qd_drv - qd)
                                 > np.maximum(2.0 * np.abs(qd), 1.0))

                    pressed = key.get()
                    if pressed in (cfg.KEY_STOP, cfg.KEY_STOP.upper()):
                        stop = "operator X"
                        break
                    if pressed == cfg.KEY_LIMP and not limp:
                        # Zeroing a held tau_ff used to be what LIMP did.
                        # The feedforward is rebuilt every sweep now, so the
                        # flag alone is what stops it -- see the torque_stage
                        # gate below.  Writing the zero as well would read as
                        # load-bearing and is not.
                        limp = True
                        print("[stand] LIMP")
                    elif pressed in ("\r", "\n"):
                        if limp:
                            limp = False
                            gate.start(now, q)
                            q_ref = q.copy()
                            # RE-LATCH THE HEADING, for the same reason q_ref
                            # is retaken: a limp is where the robot gets moved.
                            # Holding the old lock would ask the first torqued
                            # sweep for the whole accumulated yaw error at
                            # once, which is precisely the step the ramp and
                            # the slew limit exist to prevent elsewhere.
                            yaw_ref = None
                            print("[stand] re-engaged")
                        elif stage == "WAIT" and z_crouch is not None:
                            stage, t0, armed = "RISE", now, True
                            gate.start(now, q)
                            q_ref = q.copy()
                            print(f"[stand] RISE: {z_crouch*1e3:.0f} -> "
                                  f"{args.height*1e3:.0f} mm over "
                                  f"{cfg.T_RISE:.0f} s under torque")
                    elif pressed in (cfg.KEY_PARK, cfg.KEY_PARK.upper()) \
                            and stage == "HOLD":
                        stop = "park"
                        break

                    if pressed in ("t", "T") and stage == "HOLD" \
                            and not limp:
                        stage, t0 = "TROT", now
                        gait.reset(now)
                        # gait.reset restarts the clock, so every leg is about
                        # to produce a liftoff edge.  Clearing here rather than
                        # relying on the stance branch means a second T after a
                        # limp cannot re-apply a step planned before it.
                        step_xy[:] = 0.0
                        was_swing[:] = False
                        print(f"[trot] TROT: {gait}.  Feet are leaving the "
                              f"ground.  SPACE limps, X stops.  P is dead "
                              f"until you are back in HOLD.")

                    if stage == "RISE" and now - t0 >= cfg.T_RISE:
                        stage, t0 = "HOLD", now
                        print(f"[stand] HOLD: closed loop at "
                              f"{args.height*1e3:.0f} mm.  Push the trunk, "
                              f"then T to trot.  P parks.")

                    torque_stage = stage in ("RISE", "HOLD", "TROT") and armed

                    # THE CONTACT MASK IS THE WHOLE TROT.  Everything below is
                    # week 2's, and it already takes a stance list: distribute
                    # splits the wrench over whatever feet it is given, and
                    # stance_torque gives a non-planted leg its own leg
                    # gravity and nothing else.  Handing it a clock instead of
                    # ALL4 is the change.
                    planted = gait.contact(now) if stage == "TROT" else ALL4

                    # ---- the sub-sampled block: estimator + model + map ---
                    # ~1.5 ms, so it runs every MODEL_EVERY sweeps and that
                    # sweep's FL frames are late by that much.  Inside the
                    # 10 ms watchdog with margin; see params.MODEL_EVERY.
                    if sweep % cfg.MODEL_EVERY == 0:
                        state, act, why = est.read(now, q, qd, planted)
                        if act:
                            if z_crouch is None:
                                z_crouch = float(state["r"][2])
                                print(f"[stand] crouch height {z_crouch*1e3:.0f}"
                                      f" mm to the TRUNK BOTTOM "
                                      f"({state['z_hip']*1e3:.0f} mm to the hip "
                                      f"axis).  FK, drift-free -- put a ruler "
                                      f"on it.")
                                _print_attitude_report(est, args)
                                print("[stand] ENTER to rise.")
                            z_des = (dm.height_ramp(z_crouch, args.height,
                                                    (now - t0) / cfg.T_RISE)
                                     if stage == "RISE" else
                                     args.height
                                     if stage in ("HOLD", "TROT")
                                     else z_crouch)
                            if torque_stage:
                                q_ref = q_ref_for_height(q_ref, z_des, foot_xy)
                                # THE YAW LOCK, LATCHED THE FIRST TIME TORQUE
                                # IS LIVE AND CLEARED ON EVERY LIMP.  Roll and
                                # pitch have a reference that is a fact about
                                # the world -- level -- and yaw does not, so
                                # the reference is "wherever the robot was
                                # pointing when it started pushing".  Latching
                                # it HERE rather than at the ENTER that begins
                                # RISE matters: the estimator may not have
                                # answered yet at ENTER, and a lock taken from
                                # a stale heading would be a constant offset
                                # the spring then fights for the whole run.
                                # By construction the error is exactly zero on
                                # the sweep it is taken, so arming can never
                                # step the wrench.
                                if yaw_ref is None:
                                    yaw_ref = float(est.yaw)
                                    print(f"[stand] yaw lock "
                                          f"{np.degrees(yaw_ref):+.1f} deg "
                                          f"(kp_yaw {gains.kp_yaw:.0f} "
                                          f"Nm/rad, error clamped at "
                                          f"{np.degrees(gains.yaw_err_max):.0f}"
                                          f" deg)")
                                W = dm.body_wrench(state, z_des, args.mass,
                                                   gains=gains,
                                                   yaw_ref=yaw_ref)
                                # stance_torque IS STILL CALLED, and its
                                # torque IS thrown away.  What is wanted is
                                # diag: the clamped per-foot force, the
                                # singular-Jacobian watch and the fz readout.
                                # Calling ft.distribute directly would save
                                # ~1 ms here, and would fork a shared week-2
                                # module that stand_torque_Mode.py also flies.
                                # The 1 ms is affordable (measured 1066 us per
                                # sweep total, 27% of the 4 ms budget); a
                                # second copy of the clamp is not.
                                # THE HANDOVER IS A RAMP NOW, NOT A STEP.
                                # contact_weight smoothsteps each foot's share
                                # over the first and last 15% of its stance;
                                # binary planted took half the robot in one
                                # model step, and on a diagonal pair that step
                                # is an unopposed roll impulse -- the roll's
                                # sign FOLLOWED the lead diagonal in the
                                # t1/t5-vs-t8 A/B, which is this mechanism's
                                # fingerprint.
                                cw = (gait.contact_weight(now)
                                      if stage == "TROT" else None)
                                _, diag = ft.stance_torque(
                                    q, qd, state, W, planted, gains,
                                    force_frac=args.force_frac,
                                    leg_gravity=args.leg_gravity,
                                    contact_weight=cw)
                                f_foot[:] = 0.0
                                for _k, _leg in enumerate(LEGS):
                                    if planted[_k] and _leg in diag["forces"]:
                                        f_foot[_k] = (args.force_frac
                                                      * diag["forces"][_leg])
                                g_down = st.gravity_down_body(state["C"])
                                model_live = True
                                dt_p = (0.0 if t_place is None
                                        else now - t_place)
                                t_place = now
                                a_p = 1.0 - math.exp(
                                    -dt_p / cfg.RAIBERT_V_TAU_S)
                                v_place += a_p * (est.v - v_place)
                                if stage == "TROT":
                                    # ONLY THE LATCH IS LEFT HERE.  The arc and
                                    # the impedance moved to the sweep, where
                                    # q is current; what stays is the once-per-
                                    # swing decision that needs the ESTIMATOR
                                    # -- est.v and state["C"] -- and so cannot
                                    # run faster than the estimator does.
                                    for i in range(4):
                                        if planted[i]:
                                            # BACK ON THE FLOOR: forget the
                                            # step, or the next swing reuses an
                                            # offset planned for a velocity the
                                            # robot no longer has.
                                            step_xy[i] = 0.0
                                            was_swing[i] = False
                                            continue
                                        if args.raibert is not None \
                                                and not was_swing[i]:
                                            # LIFTOFF EDGE.  The step is
                                            # LATCHED here, not recomputed
                                            # every block: a target chasing the
                                            # low-passed odometry through the
                                            # swing is a moving reference, and
                                            # the impedance tracks it as a
                                            # disturbance.  The edge is seen up
                                            # to MODEL_EVERY sweeps (~12 ms of
                                            # a ~120 ms swing) late, which
                                            # costs nothing -- the horizontal
                                            # ramp is a smoothstep and is still
                                            # within 4% of zero there.
                                            step_xy[i] = raibert_step_body(
                                                i, v_place, state["C"],
                                                gait.stance_duration,
                                                args.raibert, z_des, foot_xy)
                                        was_swing[i] = True
                        else:
                            # honest fallback: hold each leg up, command no
                            # body wrench.  A frozen wrench would keep pushing
                            # on a world model that has stopped updating.
                            # ZEROING THE FORCE IS THE WHOLE FALLBACK NOW: with
                            # f_foot 0 and g_down None, leg_torque_block returns
                            # exactly force_totorque.leg_gravity_only(q), and
                            # the self-test pins it to that.
                            f_foot[:] = 0.0
                            g_down = None
                            model_live = False
                            if torque_stage and not limp:
                                limp = True
                                print(f"[stand] LIMP: estimator refused ({why})")

                    # ---- the 250 Hz law + floor + safety -----------------
                    # THE FEEDFORWARD IS BUILT HERE NOW, not held from the
                    # model block: -J(q)^T f and the swing impedance both see
                    # THIS sweep's q and qd.  The self-test TIMES this block
                    # rather than quoting it -- 0.85 ms for four legs on the
                    # Pi it was written on, which puts the worst sweep (model
                    # block + this) at 2.23 of the 4 ms.
                    if torque_stage and not limp \
                            and sweep % args.map_every == 0:
                        p_des = v_des = None
                        dsdt = 0.0
                        if (model_live and stage == "TROT"
                                and z_des is not None):
                            # The arc is re-evaluated every sweep, so the
                            # reference the impedance chases advances smoothly
                            # instead of in 12 ms steps -- a 40 mm apex over a
                            # 120 ms swing moves 2.6 mm in one model block, and
                            # a 200 N/m stiffness reads that as half a newton
                            # of staircase.
                            sph = gait.swing_phase(now)
                            dsdt = 1.0 / gait.swing_duration
                            p_des = np.zeros((4, 3))
                            v_des = np.zeros((4, 3))
                            for i in range(4):
                                if planted[i]:
                                    continue
                                sl = slice(3 * i, 3 * i + 3)
                                p_des[i], v_des[i] = swing_foot_body(
                                    i, float(sph[i]), z_des, foot_xy,
                                    args.swing_height, step_xy[i])
                                # q_ref FOLLOWS THE LEG, every sweep.  Left at
                                # the stance pose the joint floor fights the
                                # swing; left at the 83 Hz value it fights it
                                # 12 ms late, which is worse than either.
                                q_ref[sl] = q[sl]
                        tau_ff = leg_torque_block(
                            q, qd, f_foot, planted, g_down,
                            p_des, v_des, dsdt,
                            leg_gravity=args.leg_gravity,
                            kd_joint=gains.kd_joint)
                        # tau_des IS THE CONTROLLER'S ANSWER, tau_cmd is what
                        # the gate let through (ramp, cap, slew).  Both are
                        # logged: their difference is the only record of how
                        # much of the control law the hardware actually ran,
                        # and a law that is right but 60% clipped looks
                        # exactly like a law that is wrong.
                        tau_des = tau_ff + imp.tau(q, qd, q_ref)
                        tau_cmd = gate.apply(tau_des, q, now)
                    elif torque_stage and not limp:
                        # a held map: tau_ff is stale by < map_every sweeps,
                        # exactly the pre-refactor behaviour; the impedance
                        # floor still sees this sweep's q and qd.
                        tau_des = tau_ff + imp.tau(q, qd, q_ref)
                        tau_cmd = gate.apply(tau_des, q, now)
                    else:
                        tau_des = np.zeros(N_JOINTS)
                        tau_cmd = np.zeros(N_JOINTS)
                        if not torque_stage:
                            q_ref = q.copy()     # never step on re-engage

                    reason = gate.estop_reason(
                        q, qd, base._temperatures(mb), miss.update(mb),
                        mb.errors(), now, enforce_position_limits=False)
                    if reason:
                        stop = reason
                        break
                    crossed = cap.ABD_SIGN * q[cap.ABD] < 0.0
                    if crossed.any():
                        k = int(np.argmax(crossed))
                        stop = (f"{base.JOINT_LABELS[cap.ABD[k]]} "
                                f"(CAN {MOTOR_IDS[cap.ABD[k]]}) crossed 0 at "
                                f"{np.rad2deg(q[cap.ABD[k]]):+.1f} deg")
                        break

                    # in TORQUE mode a latched driver stops producing torque:
                    # that leg collapses while the others keep pushing, which
                    # is an active tip-over, not a stall
                    latched = [m for m, e in mb.errors().items() if e & 0x80]
                    if latched and torque_stage and cfg.LATCH_LIMPS_ROBOT \
                            and not limp:
                        limp = True
                        print(f"[stand] LIMP: input-lost latch on {latched}")

                    if state is not None and act:
                        tilt = max(abs(np.degrees(est.roll)),
                                   abs(np.degrees(est.pitch)))
                        if args.tilt_stop > 0 and tilt > args.tilt_stop:
                            stop = f"tilt {tilt:.1f} deg"
                            break

                    # THE measurement: measured iq -> foot force -> the robot.
                    # Four Jacobians, four SVDs, four solves -- measured
                    # 1029 us, so it cannot run every 4 ms sweep, and it is
                    # STAGGERED against the model block so the two never land
                    # in the same sweep (see params.LOAD_OFFSET).  At 21 Hz it
                    # still catches a collapsing leg within 48 ms.
                    if sweep % cfg.LOAD_EVERY == cfg.LOAD_OFFSET:
                        support, ok = ft.foot_load_from_torque(
                            q, tau_meas, None if state is None else state["C"])
                        load_sum = float(np.nansum(support)) if ok.any() \
                            else float("nan")
                        if stage == "HOLD" and not limp \
                                and np.isfinite(load_sum) \
                                and abs(load_sum - args.mass * cfg.GRAVITY) \
                                / (args.mass * cfg.GRAVITY) > cfg.LOAD_SUM_TOL_FRAC:
                            limp = True
                            print(f"[stand] LIMP: measured foot load "
                                  f"{load_sum:.1f} N against "
                                  f"{args.mass*cfg.GRAVITY:.1f} N expected -- the "
                                  f"force loop is not doing what it says")

                    if args.log:
                        # w, v and W are the OUTER LOOP'S OWN SIGNALS: the two
                        # it feeds back on and the one it produces.  Without
                        # them a log of an outer-loop shake shows the result
                        # and neither the cause nor the command -- roll is an
                        # ANGLE, and kd_att acts on the RATE, which nothing
                        # else in this file records.  W and est.* are held
                        # between model steps; tau_cmd is not, and neither is
                        # the feedforward inside it any more.
                        log.append((now, q.copy(), qd.copy(), qd_drv.copy(),
                                    tau_cmd.copy(), tau_meas.copy(),
                                    np.array(diag["fz"]),
                                    est.roll, est.pitch, est.z, load_sum,
                                    stage, est.roll_fk, est.pitch_fk,
                                    est.z_hip, est.roll_raw, est.pitch_raw,
                                    q_ref.copy(),
                                    np.zeros(3) if state is None
                                    else np.asarray(state["w"]).copy(),
                                    est.v.copy(), W.copy(),
                                    np.asarray(planted, float).copy(),
                                    step_xy.copy(), f_foot.copy(),
                                    float("nan") if z_des is None else z_des,
                                    # THE HEADING AND THE LOCK IT IS PULLED
                                    # TOWARDS.  Both, because neither is
                                    # readable alone: yaw is absolute and
                                    # wraps, the lock is whatever the run
                                    # happened to arm at, and the ERROR --
                                    # the only quantity the wrench acts on --
                                    # is the wrapped difference.  Logging the
                                    # error instead would hide a lock that
                                    # was latched off a stale sample.
                                    est.yaw,
                                    float("nan") if yaw_ref is None
                                    else yaw_ref,
                                    # the pre-gate request, beside tau_cmd
                                    # (r[4], the post-gate command) and
                                    # tau_meas (r[5], the measured iq): the
                                    # three stations of one torque.
                                    tau_des.copy()))

                    if now - last_print >= 1.0 / base.STATUS_HZ:
                        last_print = now
                        nsw = ("" if stage != "TROT"
                               else f"  swing={4 - int(np.sum(planted))}")
                        print(f"[stand] {'LIMP' if limp else stage:5s} "
                              f"{est.status() if state is not None else why}"
                              f"{nsw}", flush=True)

                    if web is not None:
                        # One whole-dict REBIND per sweep, never an in-place
                        # mutation -- att_web's sampler reads this without a
                        # lock and must never see a half-write.
                        web.status = {
                            "stage": "LIMP" if limp else stage,
                            "swing": (4 - int(np.sum(planted))
                                      if stage == "TROT" else 0),
                            "planted": [bool(b) for b in planted],
                            "est_roll": (round(float(np.degrees(est.roll)), 3)
                                         if state is not None else None),
                            "est_pitch": (round(float(np.degrees(est.pitch)), 3)
                                          if state is not None else None),
                            # THE LOCK AND THE ERROR, not just the heading.
                            # att_web already streams raw yaw off its own
                            # sampler; what it cannot know is where the spring
                            # is pulling, and the error is the only number
                            # that says whether the loop is winning.  None
                            # before the lock is latched, which the page shows
                            # as "--" rather than as a zero error.
                            "yaw_lock": (round(float(np.degrees(yaw_ref)), 2)
                                         if yaw_ref is not None else None),
                            "yaw_err": (round(float(np.degrees(
                                dm.wrap_pi(yaw_ref - est.yaw))), 2)
                                if yaw_ref is not None and state is not None
                                else None),
                            # THE THREE STATIONS OF EVERY JOINT'S TORQUE,
                            # per joint and NOT reduced: a worst-joint scalar
                            # was tried first and told the operator neither
                            # WHICH joint nor how the other eleven were
                            # doing.  The page renders these as a 12-row
                            # table -- des (the law's request), cmd (what the
                            # gate let through), meas (what the driver
                            # reports) -- and derives the clip marks itself.
                            "tau_des": np.round(tau_des, 3).tolist(),
                            "tau_cmd": np.round(tau_cmd, 3).tolist(),
                            "tau_meas": np.round(tau_meas, 3).tolist(),
                            # None until torque arms: cap_now needs
                            # gate.started_at, which WAIT has not set yet --
                            # and the page's "--" is honest there, where a
                            # 0.00 would read as a zeroed cap.
                            "tau_cap": (round(float(gate.cap_now(now)), 3)
                                        if gate.started_at is not None
                                        else None),
                        }

                mid = MOTOR_IDS[j]
                if stage == "WAIT":
                    mb.position(mid, float(cap.POSITION_TARGET_DEG[j]),
                                args.crouch_dps)
                else:
                    mb.torque(mid, float(tau_cmd[j]))
                index += 1
                if j == N_JOINTS - 1:
                    sweep += 1
                overrun = mb.pace(deadline)
                deadline += slot
                if overrun and overrun > 2.0 * slot:
                    deadline = time.perf_counter() + slot

            print(f"[stand] stopped: {stop}")
            # Park in POSITION mode from wherever the torque stand left it.
            # Cutting torque at height would drop the robot; 0xA4 takes the
            # weight on the drivers' own loops first.
            if stop == "park" or args.park_on_stop:
                try:
                    cap.parked(mb, unwrap, key, max_speed_dps=args.crouch_dps)
                except Exception as exc:                    # noqa: BLE001
                    print(f"[stand] park failed: {exc}")
            print("[stand] soft stop -- HOLD THE ROBOT")
            base._soft_stop(mb)
    except KeyboardInterrupt as exc:
        print(f"\n[stand] aborted: {exc}")
    except RuntimeError as exc:
        print(f"\n[stand] fault: {exc}")
    finally:
        if web_stop is not None:
            web_stop()
        try:
            imu.stop()
        except Exception:                                   # noqa: BLE001
            pass
        key.close()

    if log:
        a = np.array([r[10] for r in log], dtype=float)
        a = a[np.isfinite(a)]
        print()
        print("=" * 74)
        if a.size:
            print(f"  foot load sum: mean {a.mean():.1f} N, min {a.min():.1f}, "
                  f"max {a.max():.1f}, against "
                  f"{args.mass*cfg.GRAVITY:.1f} N")
            print("  ^ THE number: this is measured current, not a command.")

        # The control loop runs on the encoder now, so this block answers a
        # different question than it used to: is the driver's speed field
        # STILL glitching, i.e. would the old path still have shaken us?
        enc = np.max(np.abs(np.array([r[2] for r in log])), axis=0)
        drv = np.max(np.abs(np.array([r[3] for r in log])), axis=0)
        i = int(np.argmax(drv))
        print(f"  peak |qd|: encoder (used by the loop) {enc.max():.2f} rad/s, "
              f"driver field (not used) {drv[i]:.2f} rad/s on "
              f"{base.JOINT_LABELS[i]}")
        if glitches.any():
            worst = int(np.argmax(glitches))
            print(f"  driver/encoder disagreements: {int(glitches.sum())} total, "
                  f"worst {base.JOINT_LABELS[worst]} ({int(glitches[worst])}) "
                  f"-- these no longer reach the force law")
            print(f"  ^ through the OLD path each would have injected up to "
                  f"{cfg.KD_POS[2] * drv[i] * 0.0425:.0f} N of phantom force")
        else:
            print("  driver and encoder agreed throughout -- no glitching this "
                  "run, so the shake had another cause")
        if args.log:
            np.savez_compressed(
                args.log,
                t=np.array([r[0] for r in log]), q=np.array([r[1] for r in log]),
                qd=np.array([r[2] for r in log]),        # encoder, used
                qd_drv=np.array([r[3] for r in log]),    # driver field, unused
                tau_cmd=np.array([r[4] for r in log]),
                tau_meas=np.array([r[5] for r in log]),
                # THE PRE-GATE REQUEST.  tau_des is what the control law asked
                # for THIS sweep; tau_cmd is what survived the gate's ramp,
                # cap and slew; tau_meas is what the drivers report doing.
                # des-vs-cmd is the gate's bite (how much law was clipped),
                # cmd-vs-meas is the drivers' tracking -- two different
                # failures, separable only if all three are here.
                tau_des=np.array([r[27] for r in log]),
                fz=np.array([r[6] for r in log]),
                roll=np.array([r[7] for r in log]),      # AHRS, setpoint out
                pitch=np.array([r[8] for r in log]),
                z=np.array([r[9] for r in log]),         # floor->trunk bottom
                load=np.array([r[10] for r in log]),
                stage=np.array([r[11] for r in log]),
                # the independent attitude and the frame's other half: with
                # these a log can be re-read for a mount tilt or a height
                # frame after the fact, which the 2026-08-17 logs could not
                roll_fk=np.array([r[12] for r in log]),
                pitch_fk=np.array([r[13] for r in log]),
                z_hip=np.array([r[14] for r in log]),
                roll_raw=np.array([r[15] for r in log]),
                pitch_raw=np.array([r[16] for r in log]),
                # the impedance error |q_ref - q| used to be streamed as |dq|;
                # it is recoverable only if q_ref is here, so it is
                q_ref=np.array([r[17] for r in log]),
                # the outer loop's inputs (w = AHRS gyro, UNFILTERED; v = the
                # low-passed leg odometry) and its output (W, the 6D wrench)
                w=np.array([r[18] for r in log]),
                v=np.array([r[19] for r in log]),
                W=np.array([r[20] for r in log]),
                # WHICH FEET THE LOOP BELIEVED WERE DOWN.  Without it a trot
                # log cannot separate "this leg was commanded no force" from
                # "the QP gave it none", and every per-foot number below is
                # ambiguous.
                planted=np.array([r[21] for r in log]),
                # THE STEP EACH SWING WAS PLANNED WITH (4,2), trunk frame,
                # zero on a planted leg and zero everywhere with placement
                # off.  This is the whole readout of the placement loop: with
                # v beside it a log says whether the step tracked the drift,
                # and whether the reach clamp was the thing limiting it.
                # THE FORCE EACH FOOT WAS ACTUALLY GIVEN (n,4,3), BODY
                # frame, force_frac already in it, zero on a swinging leg.
                # W above is the wrench the PD law ASKED for; this is what
                # survived distribute()'s min-norm solve, its unilateral and
                # friction clamps and its saturation rescale.  With q and
                # planted beside it the ACHIEVED wrench is G f -- which is the
                # only way to see a residual Mz, since the yaw couple lives
                # entirely in the tangential components and W[5] is identically
                # zero while kd_yaw is.
                f_foot=np.array([r[23] for r in log]),
                # THE HEIGHT REFERENCE, beside the measured z two lines up.
                # The only setpoint in this loop that MOVES: over RISE it is
                # dm.height_ramp(z_crouch -> the height knob) and z_crouch is
                # MEASURED, so a log without it cannot be replotted as
                # reference-vs-actual -- the ramp would have to be rebuilt
                # from t0 and a crouch height that was never written down.
                # NaN before the estimator's first answer, which is honest:
                # there was no reference yet.
                z_des=np.array([r[24] for r in log]),
                # THE ATTITUDE REFERENCE IS IDENTICALLY ZERO and is written
                # out so a plot can say so.  Dynamic_Model.body_wrench is
                # Mx = kp_roll*(0 - roll): the target is level, always.  What
                # makes that mean level on THIS rig is setpoint_roll_deg,
                # already subtracted from `roll` below -- so `roll` IS the
                # error the law acts on, and roll_raw - setpoint reproduces
                # it.  Three arrays, no arithmetic, no chance of subtracting
                # the mount tilt twice.
                roll_ref=np.zeros(len(log)),
                pitch_ref=np.zeros(len(log)),
                # YAW IS THE ONE ATTITUDE REFERENCE THAT IS NOT ZERO and not
                # even constant across runs -- it is latched per run, and NaN
                # on every sweep before torque armed.  Kept in radians like
                # roll/pitch; the error the spring acted on is
                # wrap_pi(yaw_ref - yaw), which Dynamic_Model.wrap_pi will
                # reproduce exactly.
                yaw=np.array([r[25] for r in log]),
                yaw_ref=np.array([r[26] for r in log]),
                kp_yaw=gains.kp_yaw,
                yaw_err_max=gains.yaw_err_max,
                step_xy=np.array([r[22] for r in log]),
                raibert_kv=(float("nan") if args.raibert is None
                            else float(args.raibert)),
                period=args.period, duty=args.duty,
                swing_height=args.swing_height,
                glitches=glitches,
                # every OUTER gain, not just the attitude pair: a kd_z or
                # kd_xy A/B is unreadable afterwards if the npz cannot say
                # which half of it this file is.
                kp_att=gains.kp_roll, kd_att=gains.kd_roll,
                kp_z=gains.kp_z, kd_z=gains.kd_z,
                kd_xy=gains.kd_x, kd_yaw=gains.kd_yaw,
                force_frac=args.force_frac, mass=args.mass,
                setpoint_roll_deg=args.setpoint_roll,
                setpoint_pitch_deg=args.setpoint_pitch,
                imu_below_trunk_origin_m=cfg.IMU_BELOW_TRUNK_ORIGIN_M,
                kp_imp=args.kp, kd_imp=args.kd, tau_max=args.tau_max,
                map_every=args.map_every,
                # THE WHOLE PARAMETER TABLE, EVERY RUN.  The workflow is
                # edit-config.py-and-run, so the log must carry the table it
                # flew: cfg_* is every constant as imported, cfg_source is the
                # file's own text for diffing two runs.  The unprefixed keys
                # above stay authoritative for anything a CLI flag can still
                # move, since a flag wins over the file.
                **cfg.snapshot())
            print(f"  log -> {args.log} ({len(log)} sweeps, full config.py "
                  f"snapshot inside)")
    return 0


def fxy_st():
    """The four crouch foot x/y, as the runner builds them.  Self-test only."""
    return [st.leg_frames(LEGS[k], Q_CROUCH[3*k:3*k+3])[0][:2]
            for k in range(4)]


def self_test():
    sys.path.insert(0, os.path.join(_AUG, "self-test"))
    from selftest_common import check, report, raises      # noqa: PLC0415

    # -- the impedance refuses gains past the measured envelope ------------
    check("impedance refuses kp past the stable envelope, before arming",
          raises(lambda: JointImpedance(kp=cfg.KP_IMP_MAX + 1)))
    check("...and kd past the sampled-damper bound",
          raises(lambda: JointImpedance(kd=cfg.KD_IMP_MAX + 1)))
    imp = JointImpedance()
    t = imp.tau(np.zeros(12), np.zeros(12), np.full(12, 0.1))
    check("impedance pulls TOWARDS q_ref", bool(np.all(t > 0))
          and abs(t[0] - cfg.KP_IMP * 0.1) < 1e-12)
    t = imp.tau(np.zeros(12), np.full(12, 1.0), np.zeros(12))
    check("...and opposes velocity", abs(t[0] + cfg.KD_IMP) < 1e-12)
    # THE DAMPER IS CHECKED AT THE MEASURED DELAY, NOT THE COMMAND INTERVAL.
    # The old pair of checks used SWEEP_S and passed at kd=0.6, which is the
    # exact reason the offline gate was green while the robot shook: the
    # inequality had no delay term in it.  kd_joint is in the sum because it
    # damps the same joint through the same delay.
    bound = 2 * cfg.J_MIN / cfg.LOOP_DELAY_S
    check("the damper is inside the bound at the MEASURED loop delay",
          cfg.KD_IMP + cfg.KD_JOINT_STANCE < 0.5 * bound,
          f"kd+kd_joint={cfg.KD_IMP+cfg.KD_JOINT_STANCE:.2f} vs bound {bound:.2f} "
          f"Nms/rad at {cfg.LOOP_DELAY_S*1e3:.0f} ms "
          f"({100*(cfg.KD_IMP+cfg.KD_JOINT_STANCE)/bound:.0f}%)")
    # the whole reason this file can exist -- recompute, do not trust
    check("...and the OLD default would not have been (it shook, 9-12 Hz)",
          cfg.KD_IMP_MAX + cfg.KD_JOINT_STANCE > 0.5 * bound,
          f"the old kd={cfg.KD_IMP_MAX}+{cfg.KD_JOINT_STANCE} is "
          f"{100*(cfg.KD_IMP_MAX+cfg.KD_JOINT_STANCE)/bound:.0f}% of the same bound, and "
          f"{100*(cfg.KD_IMP_MAX+cfg.KD_JOINT_STANCE)/(2*cfg.J_MIN/cfg.SWEEP_S):.0f}% of the "
          f"4 ms one this file used to check -- same gain, both answers")
    # the stiffness half: what a delay does is set by WHERE the loop crosses
    f_c = math.sqrt(cfg.KP_IMP / cfg.J_MIN) / (2 * math.pi)
    check("the stiffness puts the crossover where 16 ms is a small phase",
          360.0 * f_c * cfg.LOOP_DELAY_S < 25.0,
          f"kp={cfg.KP_IMP} -> {f_c:.1f} Hz -> "
          f"{360*f_c*cfg.LOOP_DELAY_S:.0f} deg of delay phase; kp="
          f"{cfg.KP_IMP_MAX} gives "
          f"{360*math.sqrt(cfg.KP_IMP_MAX/cfg.J_MIN)/(2*math.pi)*cfg.LOOP_DELAY_S:.0f}"
          f" deg, which is the shake")

    # -- the torque gate ---------------------------------------------------
    g = TorqueGate(tau_cap=1.0)
    g.start(0.0, np.zeros(12))
    check("the ramp opens over TORQUE_RAMP_S, not base's 1.0 s",
          abs(g.cap_now(cfg.TORQUE_RAMP_S) - 1.0) < 1e-12
          and g.cap_now(cfg.TORQUE_RAMP_S / 2) < 1.0)
    # measure the SLEW, which means letting the ramp finish first -- otherwise
    # this reads the cap opening, not the rate limiter
    g.last_time = 10.0
    out = g.apply(np.full(12, 5.0), np.zeros(12), 10.0 + cfg.SWEEP_S)
    check("the slew is the torque-mode 60 Nm/s, not the position track's 5",
          abs(float(out[0]) - cfg.TAU_SLEW_NM_S * cfg.SWEEP_S) < 1e-9,
          f"{float(out[0])*1e3:.1f} mNm in one sweep; base's 5 Nm/s would give "
          f"{base.TAU_SLEW_NM_S*cfg.SWEEP_S*1e3:.1f}")
    dqd = cfg.TAU_SLEW_NM_S * cfg.SWEEP_S ** 2 / cfg.J_MIN
    check("...which is bounded: one sweep at the slew adds ~0.1 rad/s",
          dqd < 0.2, f"{dqd:.3f} rad/s per sweep vs the 7.0 e-stop")
    g2 = TorqueGate(tau_cap=1.0)
    g2.start(0.0, np.zeros(12))
    g2.last_time = 0.0
    check("the cap is honoured after the ramp",
          float(np.max(np.abs(g2.apply(np.full(12, 9.0), np.zeros(12), 10.0))))
          <= 1.0 + 1e-9)

    # -- the overspeed trip is GONE, and the rest of estop_reason is not -----
    g3 = TorqueGate(tau_cap=1.0)
    g3.start(0.0, np.zeros(12))
    fast = np.full(12, 20.0)          # far past both old tiers
    clean = {m: 0 for m in MOTOR_IDS}
    check("20 rad/s no longer stops the run (the 8.0 hard trip is removed)",
          g3.estop_reason(np.zeros(12), fast, np.zeros(12, int),
                          np.zeros(12, int), clean, 0.01,
                          enforce_position_limits=False) is None)
    check("...but the peak is still recorded, so the log can show it",
          float(np.max(g3.qd_peak)) == 20.0)
    check("overtemp still stops the run",
          "overtemp" in (g3.estop_reason(
              np.zeros(12), np.zeros(12), np.full(12, base.TEMP_ESTOP + 1),
              np.zeros(12, int), clean, 0.02,
              enforce_position_limits=False) or ""))
    check("a stuck CAN reply still stops the run",
          "missed" in (g3.estop_reason(
              np.zeros(12), np.zeros(12), np.zeros(12, int),
              np.full(12, base.MISS_ESTOP), clean, 0.03,
              enforce_position_limits=False) or ""))
    check("a driver hard fault still stops the run",
          "motor fault" in (g3.estop_reason(
              np.zeros(12), np.zeros(12), np.zeros(12, int),
              np.zeros(12, int), {m: (0x01 if m == MOTOR_IDS[0] else 0)
                                  for m in MOTOR_IDS}, 0.04,
              enforce_position_limits=False) or ""))
    check("a NaN in the state still stops the run",
          g3.estop_reason(np.full(12, np.nan), np.zeros(12),
                          np.zeros(12, int), np.zeros(12, int), clean, 0.05,
                          enforce_position_limits=False) is not None)

    # -- q_ref tracks the rise (last week's runner froze it) ---------------
    foot_xy = [st.leg_frames(LEGS[i], Q_CROUCH[3*i:3*i+3])[0][:2]
               for i in range(4)]

    def height_of(qv):
        """Floor-to-TRUNK-BOTTOM height of a pose -- the estimator's quantity,
        so this gate checks the two definitions agree rather than assuming."""
        return fe.fk_trunk_height(qv, np.eye(3), ALL4)

    # -- THE FRAME.  This is the gate that would have caught 2026-08-17 -----
    # q_ref_for_height and fk_trunk_height are the two ends of the height
    # loop: one turns a number into joint angles, the other turns joint angles
    # back into a number.  If they disagree about WHICH POINT the number means,
    # the loop still closes, the log still looks right, and the robot stands
    # 38 mm off.  Nothing else in the stack can see that.  So: ask for a
    # height, and demand the pose that comes back MEASURES that height.
    worst_frame = 0.0
    for z_ask in (0.05, 0.09, 0.12, cfg.STAND_TRUNK_BOTTOM_M):
        q_at = q_ref_for_height(Q_CROUCH.copy(), z_ask, foot_xy)
        worst_frame = max(worst_frame, abs(height_of(q_at) - z_ask))
    check("ask for a height and the pose MEASURES that height -- the IK and "
          "the estimator agree on which point it is",
          worst_frame < 1e-9,
          f"worst {worst_frame*1e6:.3f} um over 4 heights; a frame mismatch "
          f"here is a silent {cfg.IMU_BELOW_TRUNK_ORIGIN_M*1e3:.0f} mm and the "
          f"log cannot show it")
    # PINNED TO THE NOMINAL, NOT THE KNOB.  These two checks verify the
    # 2026-08-17 frame relabelling (0.152 bottom == 0.190 hip, same pose), so
    # they must solve at the nominal derived from STAND_HEIGHT -- config's
    # STAND_TRUNK_BOTTOM_M is the user's height KNOB now and may be anything.
    z_nom = cfg.STAND_HEIGHT - cfg.IMU_BELOW_TRUNK_ORIGIN_M
    q_st_f = q_ref_for_height(Q_CROUCH.copy(), z_nom, foot_xy)
    check("...and it is the TRUNK BOTTOM, not the hip axis (the 38 mm)",
          abs(fe.fk_trunk_height(q_st_f, np.eye(3), ALL4, ref="hip")
              - z_nom - cfg.IMU_BELOW_TRUNK_ORIGIN_M) < 1e-9,
          f"stand pose measures {z_nom*1e3:.0f} mm at the trunk "
          f"bottom, "
          f"{fe.fk_trunk_height(q_st_f, np.eye(3), ALL4, ref='hip')*1e3:.0f} "
          f"mm at the hip axis")
    check("the STAND_HEIGHT number moved with the frame and the POSE did not",
          abs(fe.fk_trunk_height(q_st_f, np.eye(3), ALL4, ref="hip")
              - cfg.STAND_HEIGHT) < 1e-9,
          "0.152 at the trunk bottom is the same joint angles 0.190 at the "
          "hip axis used to command -- the change is a relabelling, not a "
          "different stand")

    # The recorded crouch's four feet are NOT exactly coplanar, and the
    # estimator's height is their MEAN.  So q_ref at that mean height cannot
    # reproduce the crouch exactly -- it levels the feet, by exactly the
    # out-of-plane amount.  Gate the size of that correction rather than
    # asserting an equality that is false for a real recorded pose.
    z0 = height_of(Q_CROUCH)
    spread = max(-float(st.leg_frames(LEGS[i], Q_CROUCH[3*i:3*i+3])[0][2])
                 for i in range(4)) \
        - min(-float(st.leg_frames(LEGS[i], Q_CROUCH[3*i:3*i+3])[0][2])
              for i in range(4))
    q_ref = q_ref_for_height(Q_CROUCH.copy(), z0, foot_xy)
    moved = max(abs(height_of(Q_CROUCH) - height_of(q_ref)), 0.0)
    check("q_ref at the crouch's own mean height levels the feet and "
          "nothing more",
          float(np.max(np.abs(q_ref - Q_CROUCH))) < 5e-3 and moved < 1e-9,
          f"joints move {np.max(np.abs(q_ref-Q_CROUCH))*1e3:.2f} mrad to close "
          f"a {spread*1e6:.0f} um out-of-plane spread; mean height unchanged")

    # Walk the ACTUAL rise: 83 Hz for T_RISE seconds, warm-started throughout.
    # A single jump from crouch to stand is not how this is ever called, and
    # with a plain Newton step it diverges by 330 mm -- the damped step and the
    # warm start are both load-bearing.
    n = int(cfg.T_RISE * cfg.CONTROL_HZ / cfg.MODEL_EVERY)
    q_ref = Q_CROUCH.copy()
    worst_lag = 0.0
    for k in range(n + 1):
        z_des = dm.height_ramp(z0, cfg.STAND_TRUNK_BOTTOM_M, k / n)
        q_ref = q_ref_for_height(q_ref, z_des, foot_xy)
        worst_lag = max(worst_lag, abs(height_of(q_ref) - z_des))
    check("q_ref tracks the whole 8 s rise, never lagging the reference",
          worst_lag < 1e-6, f"worst lag over {n} steps: {worst_lag*1e6:.3f} um")
    check("...and lands exactly on the stand height",
          abs(height_of(q_ref) - cfg.STAND_TRUNK_BOTTOM_M) < 1e-6,
          f"{height_of(q_ref)*1e3:.4f} mm vs {cfg.STAND_TRUNK_BOTTOM_M*1e3:.1f}")
    xy_err = max(float(np.max(np.abs(
        st.leg_frames(LEGS[i], q_ref[3*i:3*i+3])[0][:2] - foot_xy[i])))
        for i in range(4))
    check("...with x/y PINNED, so the contact point never moves in the world",
          xy_err < 1e-6, f"worst {xy_err*1e6:.3f} um")
    crossed = cap.ABD_SIGN * q_ref[cap.ABD] < 0.0
    check("the stand pose never crosses the abduction sign limit",
          not bool(crossed.any()),
          f"abd {np.rad2deg(q_ref[cap.ABD]).round(1)} deg")
    check("q_ref is a real IK solution, not the crouch held (it MOVED)",
          float(np.max(np.abs(q_ref - Q_CROUCH))) > 0.3,
          f"max joint travel {np.rad2deg(np.max(np.abs(q_ref-Q_CROUCH))):.1f} "
          f"deg -- last week's runner froze q_ref here, so the impedance "
          f"fought the rise all the way up")
    step = (cfg.STAND_TRUNK_BOTTOM_M - z0) / n
    check("one 83 Hz step of the rise is a fraction of a mm",
          step < 5e-4, f"{step*1e3:.3f} mm per model update")

    # -- open loop really is open: no state error can reach the wrench ------
    g0 = cfg.gains()
    g0.kp_z = g0.kd_z = g0.kp_roll = g0.kd_roll = 0.0
    g0.kp_pitch = g0.kd_pitch = g0.kd_x = g0.kd_y = g0.kd_yaw = 0.0
    g0.kp_yaw = 0.0
    # THE STATES CARRY A HEADING AND THE CALL PASSES A LOCK, so this is a real
    # test of the yaw spring's OPEN_LOOP zeroing and not an accident of the
    # key being absent.  Written the hard way on purpose: the version of this
    # test that omitted "yaw" passed before kp_yaw existed and would have gone
    # on passing after, which is the shape of a test that checks nothing.
    rng_states = [
        {"r": np.array([0., 0., z]), "v": np.array(v, dtype=float),
         "C": fe.C_from_rp(*rp), "w": np.array(w, dtype=float), "yaw": yw}
        for z, v, rp, w, yw in (
            (0.10, (0.2, -0.1, 0.3), (0.2, -0.15), (1, 2, -1), 0.35),
            (0.30, (-.5, 0.4, -0.2), (-0.1, 0.3), (-2, 1, 3), -2.90))]
    check("the OPEN_LOOP wrench is [0,0,mg,0,0,0] whatever the state does",
          all(np.allclose(dm.body_wrench(s, 0.19, gains=g0, yaw_ref=0.0),
                          [0, 0, cfg.WEIGHT, 0, 0, 0]) for s in rng_states),
          "the diagnostic baseline cannot be perturbed by any estimate")

    # -- the yaw spring, end to end through the runner's own gains ----------
    gy = cfg.gains()
    s_yaw = {"r": np.array([0., 0., 0.19]), "v": np.zeros(3),
             "C": fe.C_from_rp(0.0, 0.0), "w": np.zeros(3), "yaw": 0.05}
    check("no lock latched (yaw_ref None) means no yaw moment at all",
          abs(dm.body_wrench(s_yaw, 0.19, gains=gy)[5]) < 1e-12,
          "every sweep before torque arms must be spring-free")
    check("the lock is zero-error on the sweep it is taken, so arming cannot "
          "step the wrench",
          abs(dm.body_wrench(s_yaw, 0.19, gains=gy,
                             yaw_ref=s_yaw["yaw"])[5]) < 1e-12)
    if gy.kp_yaw:
        # A heading error must produce a RESTORING moment: imu_dog's frame has
        # yaw > 0 = nose left, so drifting left must be answered clockwise.
        check("a nose-left drift is answered with a clockwise moment",
              dm.body_wrench(s_yaw, 0.19, gains=gy, yaw_ref=0.0)[5] < 0.0)
        s_big = dict(s_yaw, yaw=-2.0)        # 115 deg off, past the clamp
        check("the yaw moment is bounded by kp_yaw * yaw_err_max",
              abs(dm.body_wrench(s_big, 0.19, gains=gy, yaw_ref=0.0)[5])
              <= gy.kp_yaw * gy.yaw_err_max + 1e-9,
              f"ceiling {gy.kp_yaw*gy.yaw_err_max:.3f} Nm")
        # The bound has to be worth something against the axis it acts on.
        # Yaw comes entirely out of friction, so the honest comparison is the
        # tangential budget mu*W, not the roll/pitch capacity above.
        check("...and that ceiling is a small fraction of the friction budget",
              gy.kp_yaw * gy.yaw_err_max < 0.5 * cfg.MU * cfg.WEIGHT * 0.2,
              f"{gy.kp_yaw*gy.yaw_err_max:.2f} Nm against a "
              f"{cfg.MU*cfg.WEIGHT*0.2:.2f} Nm scale")

    # -- the velocity path that caused the 2026-08-17 shake -----------------
    # Replay the measured glitch through both paths and demand the fix holds.
    q_st = q_ref_for_height(Q_CROUCH.copy(), cfg.STAND_TRUNK_BOTTOM_M, foot_xy)
    spike = np.zeros(12)
    spike[0] = 8.1                      # what the driver field reported
    v_bad, _ = fe.leg_odometry_velocity(q_st, spike, np.eye(3), np.zeros(3),
                                        ALL4)
    F_bad = abs(cfg.KD_POS[2] * v_bad[2]) + abs(cfg.KD_POS[0] * v_bad[0]) \
        + abs(cfg.KD_POS[1] * v_bad[1])
    # THE THRESHOLD IS THE GAIN'S, NOT A FIXED FRACTION OF THE WEIGHT.  This
    # asserted F_bad > 20% of the weight, which was written when kd_z was 120
    # and the glitch was worth 33 N.  At the verified kd_z = 40 the same
    # glitch is 4.3 N and the check failed -- not because the argument for
    # encoder differencing got weaker, but because it was pinned to a gain
    # that has since moved.  What matters is that ONE bad reading is worth
    # more than the height loop's own resolution, so compare it to what a
    # millimetre of real height error commands.
    F_mm = cfg.KP_POS[2] * 0.001
    check("the measured 8.1 rad/s glitch is worth more than a mm of real error",
          F_bad > 3.0 * F_mm,
          f"{F_bad:.1f} N from one bad reading, against {F_mm:.2f} N for a "
          f"whole millimetre of height -- {F_bad/F_mm:.0f} mm of phantom "
          f"height, on a {cfg.WEIGHT:.0f} N robot")

    # the encoder path, driven by a joint that is NOT moving
    qd_enc = np.zeros(12)
    for _ in range(200):                # a stationary joint, quantised encoder
        qd_enc += cfg.QD_ALPHA * (0.0 - qd_enc)
    check("a stationary joint differences to zero -- an encoder cannot glitch",
          float(np.max(np.abs(qd_enc))) < 1e-9)
    # worst case: one encoder LSB of jitter every sweep
    lsb = np.deg2rad(0.01)
    qd_j = 0.0
    for k in range(400):
        qd_j += cfg.QD_ALPHA * ((lsb if k % 2 else -lsb) / cfg.SWEEP_S - qd_j)
    spike_j = np.zeros(12)
    spike_j[0] = abs(qd_j)
    v_j, _ = fe.leg_odometry_velocity(q_st, spike_j, np.eye(3), np.zeros(3),
                                      ALL4)
    F_j = abs(cfg.KD_POS[2] * v_j[2]) + abs(cfg.KD_POS[0] * v_j[0]) + abs(cfg.KD_POS[1] * v_j[1])
    check("encoder quantisation dither is a NEGLIGIBLE force by comparison",
          F_j < 0.01 * cfg.WEIGHT and F_j < F_bad / 50,
          f"{F_j:.2f} N vs {F_bad:.1f} N -- {F_bad/max(F_j,1e-9):.0f}x better")
    check("...and negligible through the impedance damper too",
          cfg.KD_IMP * abs(qd_j) < 0.05 * cfg.TAU_START_MAX,
          f"{cfg.KD_IMP*abs(qd_j):.4f} Nm vs {cfg.KD_IMP*8.1:.2f} Nm from the glitch")

    # the low pass that was not one
    dt_outer = cfg.MODEL_EVERY / cfg.CONTROL_HZ
    alpha = 1.0 - np.exp(-2 * np.pi * cfg.ODOM_LPF_FC_HZ * dt_outer)
    check("the odometry low pass is actually a low pass at the OUTER rate",
          alpha < 0.5,
          f"alpha={alpha:.2f} at {1/dt_outer:.0f} Hz; the old 20 Hz corner "
          f"gave {1-np.exp(-2*np.pi*20*dt_outer):.2f}, i.e. no filtering")
    check("...while still far above the height loop it feeds",
          cfg.ODOM_LPF_FC_HZ > 2.0 * np.sqrt(cfg.KP_POS[2] / cfg.MASS) / (2 * np.pi),
          f"{cfg.ODOM_LPF_FC_HZ:.0f} Hz filter vs "
          f"{np.sqrt(cfg.KP_POS[2]/cfg.MASS)/2/np.pi:.2f} Hz height loop")

    # -- the weighted handover ---------------------------------------------
    # None must be the EXACT old solve -- every stand flies through this code.
    fx = [st.leg_frames(l, Q_CROUCH[3*k:3*k+3])[0] for k, l in enumerate(LEGS)]
    Wt = np.array([0.0, 0.0, 57.05, 0.0, 0.0, 0.0])
    f_none = ft.distribute(Wt, fx, list(LEGS), np.eye(3))
    f_ones = ft.distribute(Wt, fx, list(LEGS), np.eye(3),
                           weights=[1.0, 1.0, 1.0, 1.0])
    check("weights of 1.0 ARE the unweighted solve, bit for bit",
          all(np.array_equal(f_none[l], f_ones[l]) for l in LEGS))
    f_ramp = ft.distribute(Wt, fx, list(LEGS), np.eye(3),
                           weights=[0.1, 1.0, 1.0, 0.1])
    check("a foot at 10% weight volunteers a small share, not half the robot",
          f_ramp["FL"][2] < 0.35 * f_ramp["FR"][2],
          f"FL {f_ramp['FL'][2]:.1f} N vs FR {f_ramp['FR'][2]:.1f} N")
    check("...and the ramped split still adds up to the commanded weight",
          abs(sum(f_ramp[l][2] for l in LEGS) - 57.05) < 0.1,
          f"{sum(f_ramp[l][2] for l in LEGS):.2f} N of 57.05")

    # -- the 250 Hz law is the SAME law, just evaluated later ---------------
    # THIS IS A REFACTOR AND MUST PROVE IT.  The 83 Hz code that shipped t1..t8
    # built tau from f inside ft.stance_torque and added swing_torque on top;
    # leg_torque_block does both from a q that is one sweep old instead of
    # three.  If the two disagree at the SAME q, the move changed the law and
    # not just when it is evaluated -- so they are held to machine precision.
    g_st = cfg.gains()
    st_state = {"r": np.array([0.0, 0.0, cfg.STAND_TRUNK_BOTTOM_M]),
                "v": np.zeros(3), "C": np.eye(3), "w": np.zeros(3),
                "z_hip": 0.0}
    q_s = q_ref_for_height(Q_CROUCH.copy(), cfg.STAND_TRUNK_BOTTOM_M, fxy_st())
    qd_s = np.full(N_JOINTS, 0.3)          # NOT zero: the kd_joint term must
    W_s = dm.body_wrench(st_state, cfg.STAND_TRUNK_BOTTOM_M, cfg.MASS, gains=g_st)
    for frac in (1.0, 0.6):
        tau_old, dg = ft.stance_torque(q_s, qd_s, st_state, W_s, ALL4, g_st,
                                       force_frac=frac, leg_gravity=True)
        ff = np.array([frac * dg["forces"][l] for l in LEGS])
        tau_new = leg_torque_block(q_s, qd_s, ff, ALL4,
                                   st.gravity_down_body(st_state["C"]),
                                   kd_joint=g_st.kd_joint)
        check(f"the sweep-rate stance law IS force_totorque's, force_frac={frac}",
              float(np.max(np.abs(tau_new - tau_old))) < 1e-12,
              f"worst joint differs by {np.max(np.abs(tau_new-tau_old)):.2e} Nm "
              f"over 12 joints")

    # A SWINGING LEG: leg gravity from stance_torque PLUS swing_torque was the
    # old sum, and swing_torque is kept in the file precisely to be this half.
    planted_t = np.array([True, False, False, True])
    tau_old, dg = ft.stance_torque(q_s, qd_s, st_state, W_s, planted_t, g_st,
                                   force_frac=1.0, leg_gravity=True)
    p_d = np.zeros((4, 3))
    v_d = np.zeros((4, 3))
    for i in (1, 2):
        p_d[i], v_d[i] = swing_foot_body(i, 0.37, cfg.STAND_TRUNK_BOTTOM_M, fxy_st(),
                                         0.02, (0.003, -0.002))
        sl = slice(3 * i, 3 * i + 3)
        tau_old[sl] += swing_torque(i, q_s[sl], qd_s[sl], p_d[i], v_d[i], 8.3)
    ff = np.zeros((4, 3))
    for k, leg in enumerate(LEGS):
        if planted_t[k] and leg in dg["forces"]:
            ff[k] = dg["forces"][leg]
    tau_new = leg_torque_block(q_s, qd_s, ff, planted_t,
                               st.gravity_down_body(st_state["C"]),
                               p_d, v_d, 8.3, kd_joint=g_st.kd_joint)
    check("...and the sweep-rate SWING law is swing_torque + leg gravity",
          float(np.max(np.abs(tau_new - tau_old))) < 1e-12,
          f"worst {np.max(np.abs(tau_new-tau_old)):.2e} Nm across a diagonal "
          f"stance with two legs in the air")

    # The estimator-refused fallback must not have become something else.
    check("zero force and no tilt IS force_totorque.leg_gravity_only",
          float(np.max(np.abs(
              leg_torque_block(q_s, qd_s, np.zeros((4, 3)),
                               np.zeros(4, dtype=bool), None)
              - ft.leg_gravity_only(q_s)))) < 1e-12,
          "the LIMP branch is the same three torques it always was")

    # THE POINT OF THE MOVE: J must not be frozen.  Perturb q by one model
    # block's worth of swing travel and the torque has to move with it.
    q_late = q_s.copy()
    q_late[3:6] += 0.02                      # ~1 deg on a leg, 12 ms of swing
    d = np.max(np.abs(leg_torque_block(q_late, qd_s, ff, planted_t, None,
                                       p_d, v_d, 8.3)
                      - leg_torque_block(q_s, qd_s, ff, planted_t, None,
                                         p_d, v_d, 8.3)))
    check("a stale q would have been worth real torque, so J is not frozen",
          d > 0.01,
          f"{d:.3f} Nm between a fresh q and one 12 ms old -- what the 83 Hz "
          f"map was silently commanding")

    # -- the logged force must reconstruct the ACHIEVED wrench --------------
    # W in the npz is what the PD law ASKED for.  The whole reason f_foot is
    # logged is that the answer after the clamps is a different wrench, and
    # the yaw half of it cannot be seen any other way.
    tau_old, dg = ft.stance_torque(q_s, qd_s, st_state, W_s, ALL4, g_st)
    feet = [st.leg_frames(l, q_s[3*k:3*k+3])[0] for k, l in enumerate(LEGS)]
    G = ft.grasp_map(feet, list(LEGS), st_state["C"], dg["com_body"])
    f_w = np.concatenate([st_state["C"].T @ dg["forces"][l] for l in LEGS])
    W_ach = G @ f_w
    check("f_foot + q + planted reconstruct the wrench the feet PRODUCED",
          float(np.max(np.abs(W_ach - W_s))) < 0.05,
          f"|G f - W| = {np.max(np.abs(W_ach - W_s)):.4f} N/Nm on an "
          f"unclamped four-foot stand (GRASP_LAMBDA leaves the residual)")
    # THIS STATE IS AT REST WITH NO YAW LOCK LATCHED, so the commanded Mz is
    # zero for both of the reasons it can be -- no rate for the damper and no
    # reference for the spring.  It is NOT zero because the axis is
    # uncommandable: since 2026-08-21 a latched lock makes W[5] non-zero, and
    # this check would then be reading the spring rather than the point it
    # exists to make.  The point is that the ACHIEVED Mz is non-zero even when
    # the commanded one is not, which is why f_foot has to be logged.
    check("...and that reconstruction is what carries Mz, which W cannot",
          abs(W_s[5]) < 1e-12,
          f"commanded Mz is identically {W_s[5]:.1e} at rest with no lock "
          f"(kd_yaw={g_st.kd_yaw:.1f}, kp_yaw={g_st.kp_yaw:.1f}); the achieved "
          f"Mz is {W_ach[5]:+.4f} Nm and only the tangential forces know it")

    # -- foot placement: the horizontal loop, and the arc it rides on ------
    fxy = [st.leg_frames(LEGS[k], Q_CROUCH[3*k:3*k+3])[0][:2] for k in range(4)]
    zh = cfg.STAND_TRUNK_BOTTOM_M

    # THE REGRESSION GUARD FIRST.  Every t*.npz was flown with the vertical
    # arc, and they stay comparable to the next run only if the default
    # argument reproduces that arc EXACTLY -- not to a tolerance.
    same = all(
        np.array_equal(swing_foot_body(i, s, zh, fxy)[0][:2],
                       np.array([fxy[i][0], fxy[i][1]]))
        and swing_foot_body(i, s, zh, fxy)[1][0] == 0.0
        and swing_foot_body(i, s, zh, fxy)[1][1] == 0.0
        for i in range(4) for s in np.linspace(0, 1, 41))
    check("the default step reproduces the t1..t8 vertical arc exactly", same,
          "x/y bit-identical to foot_xy and horizontal v exactly 0")

    # v_des has to BE dp_des/ds or the impedance damps against a lie -- the
    # same check swing.py makes of its own reference, made here because this
    # is a DIFFERENT reference and inherits none of that proof.
    step = np.array([0.004, -0.003])
    ss = np.linspace(0.001, 0.999, 401)
    P_ = np.array([swing_foot_body(0, s, zh, fxy, 0.02, step)[0] for s in ss])
    V_ = np.array([swing_foot_body(0, s, zh, fxy, 0.02, step)[1] for s in ss])
    # INTERIOR ONLY: np.gradient is second-order in the middle and FIRST
    # order at the two ends, so the edge samples measure the stencil rather
    # than the arc -- they read 0.59 mm/phase on a reference that is exact.
    num = np.gradient(P_, ss, axis=0)[1:-1]
    err = float(np.max(np.abs(num - V_[1:-1])))
    check("v_des is the true d(p_des)/ds, horizontal included", err < 1e-4,
          f"worst |v_num - v_des| = {err*1e3:.4f} mm/phase over 399 interior "
          f"samples")

    # A step that arrives with horizontal speed scuffs; one that STARTS with
    # horizontal speed is a p_des discontinuity into a 200 N/m impedance.
    v0 = swing_foot_body(0, 0.0, zh, fxy, 0.02, step)[1]
    v1 = swing_foot_body(0, 1.0, zh, fxy, 0.02, step)[1]
    check("the horizontal starts and ends at zero speed",
          abs(v0[0]) + abs(v0[1]) + abs(v1[0]) + abs(v1[1]) < 1e-12)
    p0 = swing_foot_body(0, 0.0, zh, fxy, 0.02, step)[0]
    p1 = swing_foot_body(0, 1.0, zh, fxy, 0.02, step)[0]
    check("...and it travels exactly the planned step, lift to land",
          np.allclose(p1[:2] - p0[:2], step, atol=1e-12),
          f"{(p1[:2]-p0[:2])*1e3} mm vs {step*1e3} mm asked")

    # TROTTING IN PLACE MUST BE UNAFFECTED, or the flag is not an A/B: at zero
    # drift and zero command the step is zero and the arc is the old one.
    T_st = cfg.DUTY * cfg.GAIT_PERIOD
    z = raibert_step_body(0, np.zeros(3), np.eye(3), T_st, cfg.RAIBERT_KV,
                          zh, fxy)
    check("a robot with no drift plans no step at all",
          float(np.max(np.abs(z))) == 0.0, f"{z*1e3} mm")

    # The sign is the whole mechanism: drifting +x, the foot goes +x, AHEAD of
    # the body, and the stance that follows pushes it back.  A sign slip here
    # is a positive feedback that empties the workspace in two cycles, and it
    # would look exactly like the drift it was meant to fix.
    s_fwd = raibert_step_body(0, np.array([0.02, 0.0, 0.0]), np.eye(3), T_st,
                              cfg.RAIBERT_KV, zh, fxy)
    check("the foot is placed DOWNSTREAM of the drift, not against it",
          s_fwd[0] > 0.0 and abs(s_fwd[1]) < 1e-12,
          f"+20 mm/s of x drift -> {s_fwd[0]*1e3:+.1f} mm step")
    check("...and the symmetry term alone is v*t_stance/2, to the mm",
          abs(raibert_step_body(0, np.array([0.02, 0.0, 0.0]), np.eye(3),
                                T_st, 0.0, zh, fxy)[0]
              - 0.02 * 0.5 * T_st) < 1e-9,
          f"{0.02*0.5*T_st*1e3:.2f} mm at t_stance = {T_st*1e3:.0f} ms")

    # THE INWARD CAP, on the exact failure f_ramp_raibert.npz recorded: the
    # trunk rolls left, v_y says left, and the LEFT foot must not answer by
    # stepping under the body.  Outward from the same velocity stays free.
    v_left = np.array([0.0, -0.30, 0.0])           # 300 mm/s leftward slosh
    s_fl = raibert_step_body(0, v_left, np.eye(3), T_st, cfg.RAIBERT_KV, zh, fxy)
    s_fr = raibert_step_body(1, v_left, np.eye(3), T_st, cfg.RAIBERT_KV, zh, fxy)
    check("a left foot answering a leftward fall cannot cross the midline",
          -cfg.SIDE_SIGN[0] * s_fl[1] <= cfg.STEP_INWARD_MAX_M + 1e-12,
          f"FL inward {-cfg.SIDE_SIGN[0]*s_fl[1]*1e3:.1f} mm, cap "
          f"{cfg.STEP_INWARD_MAX_M*1e3:.0f}; the log showed -57 mm")
    # "in full" up to the REACH clamp, which for a pure-y step bites near
    # 28 mm: the rest pose is 92% of reach almost entirely in x-z, so lateral
    # motion adds in quadrature and gets ~3x the radial room.
    check("...while the right foot, stepping OUTWARD, is capped by reach only",
          -cfg.SIDE_SIGN[1] * s_fr[1] < 0
          and abs(s_fr[1]) > 2 * cfg.STEP_INWARD_MAX_M,
          f"FR steps {s_fr[1]*1e3:+.1f} mm outward vs the 10 mm inward cap")

    # The clamp, at the height it actually bites: a 1 m/s step is absurd and
    # is exactly what a single bad odometry sample looks like.
    big = raibert_step_body(0, np.array([1.0, 0.0, 0.0]), np.eye(3), T_st,
                            cfg.RAIBERT_KV, zh, fxy)
    hip0 = np.asarray(cfg.HIP_OFFSET[0], dtype=float)
    z_site = -(fe.hip_from_imu(zh) - cfg.FOOT_RADIUS_M)
    land = np.array([fxy[0][0] + big[0], fxy[0][1] + big[1], z_site])
    check("a wild velocity is clamped inside 95% of the leg's reach",
          float(np.linalg.norm(land - hip0)) <= 0.95 * cfg.LEG_REACH + 1e-9,
          f"1 m/s asks {1.0*0.5*T_st*1e3:.0f} mm, clamp gives "
          f"{big[0]*1e3:.1f} mm")
    check("...and the clamp holds the LANDING HEIGHT, not just the length",
          abs(land[2] - z_site) < 1e-15,
          "a foot pulled in must still arrive at the floor")

    # THE ROOM MUST BE POSITIVE AT THE HEIGHT ACTUALLY FLOWN, or the flag is
    # a no-op that still prints a banner: every step would clamp to zero and
    # the run would look like placement was on and did nothing.
    check("there IS step room at the stand height this runner flies",
          _step_room(zh) > 0.005,
          f"{_step_room(zh)*1e3:.1f} mm at {zh*1e3:.0f} mm trunk-bottom "
          f"({fe.hip_from_imu(zh)*1e3:.0f} mm at the hip axis)")
    # config.py's table is in the OLD HIP-AXIS FRAME and z_des here is the
    # TRUNK BOTTOM -- the same 38 mm that made STAND_HEIGHT 0.152, and the
    # reason 0.190 is quoted nowhere in this block.  Compare heights this
    # runner can actually be given.
    # AT THE NOMINAL, NOT THE KNOB: the doubling is a property of how cramped
    # the 0.152 stance is.  A user who has already dialled the knob lower is
    # standing where the room is no longer scarce, and the ratio there is
    # honestly smaller -- that is the knob working, not this claim failing.
    z_nom_room = cfg.STAND_HEIGHT - cfg.IMU_BELOW_TRUNK_ORIGIN_M
    check("crouching buys step room, which is the only way to buy it",
          _step_room(z_nom_room - 0.012) > 2.0 * _step_room(z_nom_room),
          f"{_step_room(z_nom_room)*1e3:.1f} mm at {z_nom_room*1e3:.0f}, "
          f"{_step_room(z_nom_room-0.012)*1e3:.1f} mm 12 mm lower")

    # The banner quotes a drift speed the loop can still correct; it has to be
    # the speed at which the clamp starts truncating, not a nearby number.
    kv = cfg.RAIBERT_KV
    v_edge = _step_room(zh) / (0.5 * T_st + kv)
    s_edge = raibert_step_body(0, np.array([v_edge * 0.98, 0.0, 0.0]),
                               np.eye(3), T_st, kv, zh, fxy)
    check("the banner's max correctable drift is where the clamp starts",
          abs(s_edge[0] - v_edge * 0.98 * (0.5 * T_st + kv)) < 1e-6,
          f"{v_edge:.3f} m/s at {zh*1e3:.0f} mm -- unclamped just below it")

    # -- the loop's own timing budget --------------------------------------
    check("the sub-sampled model still updates faster than the body moves",
          cfg.CONTROL_HZ / cfg.MODEL_EVERY > 50.0,
          f"{cfg.CONTROL_HZ/cfg.MODEL_EVERY:.0f} Hz model, "
          f"{cfg.CONTROL_HZ/cfg.LOAD_EVERY:.0f} Hz load check")
    # The staggering is the whole reason a single thread is safe here: without
    # it the two heavy blocks coincide and one sweep is 2.41 ms late instead
    # of 1.38.  Prove they cannot collide, do not assert it in a comment.
    collide = [s for s in range(cfg.MODEL_EVERY * cfg.LOAD_EVERY)
               if s % cfg.MODEL_EVERY == 0 and s % cfg.LOAD_EVERY == cfg.LOAD_OFFSET]
    check("the model and load blocks can NEVER land in the same sweep",
          not collide, f"checked all {cfg.MODEL_EVERY*cfg.LOAD_EVERY} phases")
    # THE WORST SWEEP GREW when the map moved to the sweep rate, and this
    # check has to grow with it or it is measuring last week's loop.  Time the
    # block rather than quote a number: a slower Pi, or a fifth leg, moves it.
    _t0 = time.perf_counter()
    for _ in range(50):
        leg_torque_block(q_s, qd_s, ff, planted_t, None, p_d, v_d, 8.3)
    _blk = (time.perf_counter() - _t0) / 50
    check("the sweep-rate map fits in a sweep, beside the model block",
          0.00138 + _blk < cfg.SWEEP_S,
          f"1.38 ms model + {_blk*1e3:.2f} ms map = {(0.00138+_blk)*1e3:.2f} ms "
          f"of the {cfg.SWEEP_S*1e3:.0f} ms sweep; SLOT_S is {cfg.SLOT_S*1e6:.0f} us "
          f"and mb.pace borrows across slots, which is what makes this legal")
    check("...and the worst sweep is still inside half the driver watchdog",
          0.00138 + _blk < cfg.WATCHDOG_S / 2,
          f"{(0.00138+_blk)*1e3:.2f} ms vs {cfg.WATCHDOG_S*1e3:.0f} ms watchdog")

    # -- the stage machine's one irreversible rule -------------------------
    # z_crouch is MEASURED, so recompute it here rather than carrying the
    # 0.044 literal this gate used to hold -- that number was the hip-axis
    # crouch and the frame change made it silently wrong by 38 mm.
    z_cr = height_of(Q_CROUCH)
    check("RISE starts at the MEASURED crouch height, never at the target",
          dm.height_ramp(z_cr, cfg.STAND_TRUNK_BOTTOM_M, 0.0) == z_cr,
          f"a {cfg.STAND_TRUNK_BOTTOM_M:.3f} setpoint at the {z_cr:.3f} crouch would ask "
          f"{cfg.KP_POS[2]*(cfg.STAND_TRUNK_BOTTOM_M-z_cr) + cfg.WEIGHT:.0f} N on sweep 1, "
          f"{(cfg.KP_POS[2]*(cfg.STAND_TRUNK_BOTTOM_M-z_cr)+cfg.WEIGHT)/cfg.WEIGHT:.1f}x weight")
    # The NOMINAL again, not the knob: the claim is about the frame
    # relabelling, and it must hold whatever height the user has dialled in.
    z_nom2 = cfg.STAND_HEIGHT - cfg.IMU_BELOW_TRUNK_ORIGIN_M
    check("...and the rise TRAVEL is unchanged by the frame change",
          abs((z_nom2 - z_cr)
              - (cfg.STAND_HEIGHT - fe.fk_trunk_height(Q_CROUCH, np.eye(3),
                                                       ALL4, ref="hip"))) < 1e-9,
          f"{(z_nom2 - z_cr)*1e3:.1f} mm of travel either way -- a "
          f"constant offset cancels in a difference, which is why the shake "
          f"work was unaffected by the frame being wrong")

    return report()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # --tau-max IS THE ONE TUNING FLAG LEFT.  Every other parameter flag was
    # deleted 2026-08-21: the workflow is edit-config.py-and-run, and a flag
    # that shadows the file is a value the cfg_* snapshot in the npz lies
    # about.  The cap is different in kind -- the staged-verification
    # procedure raises it between otherwise-IDENTICAL runs, and it is the one
    # number worth being able to lower from the terminal in a hurry.
    ap.add_argument("--tau-max", type=float, default=cfg.TAU_START_MAX,
                    help=f"per-joint torque cap (Nm), the staged-verification "
                         f"knob: {cfg.TAU_START_MAX} first, toward "
                         f"{cfg.TAU_STAGED_MAX} only after a quiet HOLD.  "
                         f"The ONLY parameter flag; everything else is an "
                         f"edit to dog5_trot_quasi_static_model/config.py.")
    # The rest is run plumbing, not parameters.
    ap.add_argument("--port", default=None, help="AHRS serial port override")
    ap.add_argument("--web", action="store_true",
                    help="serve a live roll/pitch/yaw page on :8080 (att_web; "
                         "AHRS + the loop's setpoint-subtracted view, mag yaw "
                         "display-only).  Run plumbing, not a parameter.")
    ap.add_argument("--park-on-stop", action="store_true",
                    help="park in position mode after ANY stop, not just P")
    ap.add_argument("--log", default=None,
                    help="npz path.  EVERY hardware run logs: left unset, a "
                         "run_<date>_<time>.npz is written to the cwd.  "
                         "--no-log is the only way to not leave a record.")
    ap.add_argument("--no-log", action="store_true",
                    help="run without writing an npz (bench fiddling only)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    # THE TABLE FILLS THE NAMESPACE, in one place.  run() still reads args.*
    # so the loop is untouched; every parameter lands here from config.py and
    # nowhere else, so there is exactly one way a number reaches the loop --
    # and the npz's cfg_* snapshot is that number, always.
    args.height = cfg.STAND_TRUNK_BOTTOM_M
    args.mass = cfg.MASS
    args.kp, args.kd = cfg.KP_IMP, cfg.KD_IMP
    args.force_frac = cfg.FORCE_FRAC_DEFAULT
    args.leg_gravity = cfg.LEG_GRAVITY_ON
    args.open_loop = cfg.OPEN_LOOP
    # The per-gain overrides are gone WITH their flags: run() treats None as
    # "config.py as written", i.e. KP_ORI/KD_ORI/KP_POS/KD_POS.
    args.kp_att = args.kd_att = None
    args.kp_z = args.kd_z = args.kd_xy = args.kd_yaw = None
    args.setpoint_roll = cfg.SETPOINT_ROLL_DEG
    args.setpoint_pitch = cfg.SETPOINT_PITCH_DEG
    args.period, args.duty = cfg.GAIT_PERIOD, cfg.DUTY
    args.swing_height = cfg.SWING_HEIGHT
    args.map_every = cfg.MAP_EVERY
    args.raibert = cfg.RAIBERT_KV if cfg.RAIBERT_ON else None
    args.lead_diagonal = cfg.LEAD_DIAGONAL
    args.tilt_stop = cfg.TILT_STOP_DEG
    args.crouch_dps = cap.MAX_DPS

    # EVERY RUN LEAVES AN NPZ unless --no-log says otherwise.  The parameter
    # workflow is edit-config.py-and-run, and an edit whose run wrote no log
    # is a change with no record -- the exact hole the snapshot exists to
    # close.  Auto-named by wall clock so back-to-back runs cannot clobber
    # each other; name it yourself the moment a run is worth keeping.
    if args.no_log:
        args.log = None
    elif args.log is None:
        args.log = time.strftime("run_%Y%m%d_%H%M%S.npz")
        print(f"[log] no --log given: this run -> {args.log}")

    # BEFORE ANYTHING ARMS.  config.py owns this runner's table, but three
    # week-2 modules read a handful of constants out of params.py that no
    # argument can override.  A divergence there is a wrong number inside a
    # live force loop, so it is a refusal to start instead.
    cfg.assert_shared(_week2_params)
    if not 0.0 < args.tau_max <= cfg.TAU_STAGED_MAX:
        ap.error(f"--tau-max must be in (0, {cfg.TAU_STAGED_MAX}]")
    if not 0.0 <= args.force_frac <= 1.0:
        raise SystemExit(f"config.py FORCE_FRAC_DEFAULT = {args.force_frac} "
                         f"must be in [0, 1]")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
