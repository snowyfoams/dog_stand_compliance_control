#!/usr/bin/env python3
"""Stage 3: TROT IN PLACE, position mode, with the gait TIMED BY THE ROBOT.

WHAT IT DOES
    From a settled, leveled HOLD4 stand, alternate the two diagonal pairs --
    FL+RR, then FR+RL -- by lifting each pair a few millimetres and putting it
    back.  There is no gait clock and no phase variable.  The next pair is
    released only once forward kinematics confirms the last one has RETURNED,
    so the cadence is an output of the run, not an input to it.

    Everything below the gait is imported: the height loop is stage 2's, the
    leveling law and the stage machine are stage 2b's, the per-leg z->q tables
    are stage 2's.  This runner owns the swing state machine, the FK switch,
    and the contact schedule that goes with them -- nothing else.

WHY FK, AND WHAT IT CAN AND CANNOT TELL US
    FK CANNOT see touchdown.  In position mode the encoder tracks its
    reference whether or not the foot has met the floor; there is no contact
    event anywhere in `foot_position()`.  The geometric alternative -- "the
    swing foot crossed the plane of the stance feet" -- goes blind exactly
    when it is needed, because a diagonal support pair is two points and
    `fk_attitude` needs three non-collinear ones (it returns NaN at two).

    What FK CAN do is judge RETURN.  `TrotFKSwitch` samples each swinging
    leg's measured foot z at the instant its swing begins, and calls the pair
    back once the measured z is within TROT_SWITCH_TOL_M of that baseline and
    has stayed there for TROT_SWITCH_HOLD_S.  Measuring against the leg's own
    pre-swing baseline is what makes the tolerance meaningful: the commanded
    and measured z differ by several millimetres of load sag in stance, so
    comparing measured-to-COMMANDED would never come inside 1 mm.  Comparing
    measured-to-BASELINE cancels the sag -- and since the sag only comes back
    as the leg re-loads, the test waits for load transfer rather than for the
    encoder to reach a number.

    That is a weaker claim than "the foot has landed", and it is deliberately
    the claim this script makes.  What it buys over a fixed schedule is real:
    the gait can never outrun the motors, because it does not advance until
    the hardware says the last step finished.  TROT_SWITCH_TIMEOUT_S is the
    other half of that bargain -- a leg that never confirms aborts the gait
    instead of hanging it with a foot in the air.

LIFTING IS NOT ENOUGH -- THE STANCE PAIR HAS TO PUSH
    Hardware, 2026-08-14: commanded FL+RR up and the knees moved while the
    feet stayed on the floor.  Raising --lift does not fix it, and the reason
    is that a lift moves the feet relative to the TRUNK, while clearance is
    relative to the FLOOR, and lifting a diagonal moves the trunk too:

      * the swinging legs UNLOAD.  A leg that sags 13 mm under load (stage 1's
        measured figure) extends by about that much when the weight comes off,
        so the foot reaches back DOWN toward the floor.
      * the stance pair's load DOUBLES, so it compresses by roughly another
        13 mm and the trunk SINKS -- carrying the swinging feet down with it.

    Both terms subtract, and they are each about the size of the lift:

        clearance ~ lift + push - 2 * 13 mm

    With no push, the lift alone has to beat 26 mm -- and atan(26 / 214 mm) is
    7 deg of rocking bound, well past --lean-abort.  There is no push-free
    lift that both clears the floor and stays inside the abort.  So the stance
    pair extends by `--push` for exactly as long as the other pair is up,
    ramped off the same variable so the two can never fall out of step.  The
    height loop cannot stand in for it: at HEIGHT_SLEW_M_S it moves 2.5 mm in
    a 0.5 s swing, and being feedback it arrives after the trunk has dropped.

    The push spends EXTENSION AUTHORITY -- how far below the stand pose the
    legs can still reach -- and so do the sag the height loop winds out and
    the leveling clamp.  The reach bottoms out at a commanded foot z of
    -221 mm, so the stand height sets the budget:

        --stand-height 0.19   31 mm of authority   13 sag + 12 level = 6 left
        --stand-height 0.17   51 mm of authority   ... 26 mm left for the push

    The startup banner prints the budget and says if it is short.  A push that
    does not fit does not fail loudly -- the table simply clips and the feet
    quietly do not clear, which looks exactly like the bug this section is
    about.  Stand lower before raising --push.

    MEASURED on hardware 2026-08-14 at --push 0.015, reading `clear=` off the
    ramp of a --lift 0.020 run as it swept through each height:

        lift    clear (FL/RR)     rock     what happened
        11 mm   +12.5 / +12.0    0.09 deg  both feet off, body still
        12 mm   +13.4 / +14.4    0.26 deg  both feet off, body still
        14 mm   +14.9 / +15.0    0.42 deg  both feet off, body still
        16 mm    +7.4 / +41.9    5.69 deg  TIPPING -- see below
        20 mm     --              4-5 deg  --lean-abort stops the gait

    So the sag model above is PESSIMISTIC: 11 mm of commanded lift bought
    about 12 mm of real clearance, not the 11 + 8 - 26 = -7 mm it predicts.
    The 13 mm figure is stage 1's TRUNK sag across four legs; the per-leg
    unload extension is evidently smaller, and the push covers the rest.
    Trust `clear=`, not the arithmetic.

    The interesting number is the CLIFF between 14 and 16 mm.  Below it both
    feet clear together and the trunk barely moves; above it FL's clearance
    COLLAPSES while RR's more than doubles, which is not a lift at all -- it
    is the trunk rotating about the FR-RL line and dropping FL back onto the
    floor.  RR at +42 mm on a 16 mm command is the giveaway.  That is the CoM
    problem in the next section, and it sets the usable lift until the body is
    trimmed: run at 12 mm.

    --lean-abort is sized against the worst-case bound, so at a working lift
    it idles with 2-3 deg of headroom and fires only once the trunk starts
    taking the lift as tip.  It caught every one of the above at 4.0-4.3 deg.
    It is the "this has stopped being a step" detector, and on 2026-08-14 that
    is exactly what it detected.

    Between steps all four feet are commanded down for TROT_OVERLAP_S, which
    is what makes this a STEP IN PLACE rather than a true trot: support never
    hands straight from one diagonal to the other.  That 4-foot window is also
    the only time the leveling loop runs -- `LevelingLoop.update` freezes
    itself below 3 planted feet, so a diagonal-support phase holds its offsets
    instead of winding them against the rock -- and it is where the clearance
    reference is taken.  `--overlap 0` removes the window and gives a real
    trot.  That is stage 3 proper; do not start there.

ONE FOOT LIFTS AND THE OTHER DOES NOT -- THAT IS THE CoM, NOT THE LIFT
    Hardware, 2026-08-14, with the stance push already in: FL+RR commanded up,
    RR left the floor, FL did not, and the body leaned toward FL.

    While a diagonal swings, the robot stands on TWO points, and rotation of
    the body about the line through them is an UNACTUATED degree of freedom.
    Both stance legs lie ON that axis, so extending them -- together or
    differentially -- moves the axis but cannot rotate the body about it.
    Gravity alone decides that rotation.  With the CoM off to one side, the
    body falls that way until the near swing foot touches down and becomes a
    third support; that foot then carries load and cannot lift.  Commanding it
    higher does not help, because the body simply tips further to meet it.
    It is a statics problem, and no lift, push or gain fixes it.

    The remedy is to put the CoM ON the support diagonal.  The two diagonals
    cross at the centre of the foot polygon, so ONE CONSTANT trim serves both
    swings -- there is no per-step weight shift to schedule.  `--com-shift-x`
    and `--com-shift-y` slide the body over the footprint (the feet move the
    opposite way, folded into the z tables at build time, so the CAN sweep
    still costs one interpolation).

    `StanceLoadBalance` measures which trim to use, from encoders only: a
    driver's position loop holds against load with finite stiffness, so each
    leg's foot sits above its commanded z by an amount proportional to what it
    carries.  Load share is that sag normalised; the CoM is the load-weighted
    mean of the foot anchors.  It prints in HOLD4 and at exit:

        [com] load share from stance sag:  FL 31.2%  FR 21.4%  RL 27.9%  RR 19.5%
        [com] CoM offset from the foot-polygon centre: x  +9.8 mm  y  +6.1 mm
        [com] -> try --com-shift-x -0.0098 --com-shift-y -0.0061

    Believe the DIRECTION, bisect the magnitude.  Backlash, per-leg stiffness
    spread and encoder-zero error all land in `sag` and none of them are load,
    so the honest test is whether both feet of a diagonal clear -- which
    `SwingClearance` reports per leg, and which is why the status line shows
    the swinging pair separately rather than as one number.

CONTACTS ARE MEASURED, NOT COMMANDED
    Every earlier runner derives its contact schedule from what it asked for.
    That is not safe here, because of the sag above: a commanded lift is not
    a lifted foot, so a command-derived flag would tell the filter that a foot
    still carrying weight is in the air -- the exact failure the OFF schedule
    exists to prevent (README, stage-3 warning).

    `SwingClearance` measures it instead.  The stance feet ARE the floor, so
    rotating the feet into world-vertical with the EKF's C and subtracting the
    stance mean gives each swinging foot's height above the floor plane, with
    the trunk's own position cancelling out.  Only ATTITUDE is needed, and
    attitude is the EKF's strongest, AHRS-cross-checked output.  Below
    --contact-clear the foot stays flagged planted; with no EKF attitude and
    no reference yet, everything stays planted, which is the schedule stages
    1/2/2b validated.  `--fake-contacts` forces all-planted on purpose as the
    A/B.

Flow:
    ZERO-TORQUE -> CROUCH -> WAIT_CROUCH (EKF inits) -> STAND -> HOLD4
      T starts the gait, once the leveling setpoint has latched
      T again stops it -- the current step always finishes, feet down
      P parks (refused mid-gait), X stops

Run:
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    cd ~/Documents/can_motor_control/dog_stand_compliance_control/Aug_trotinplace_closedloop

    $V stand_postion_mode/trot_fk_switch_hw.py --self-test    # offline gates
    sudo chrt -f 50 $V stand_postion_mode/trot_fk_switch_hw.py

    --lift 0.012          the measured stepping setting (hardware 2026-08-14:
                          both feet clear ~12 mm with the trunk near still;
                          16 mm tips the body onto FL instead)
    --push 0.0            NO stance push: reproduces the 2026-08-14 failure
                          (knees move, feet stay down) as a deliberate A/B
    --overlap 0.0         no 4-foot window: a real trot, diagonal support only
    --cycles 4            stop the gait after this many full FL+RR/FR+RL cycles
    --fake-contacts       tell the EKF all four feet stay planted (the A/B)
    --raw-log trot.npz    raw IMU+encoder for state_estimator/hw_replay

WHAT TO WATCH
    [trot] AIR     lift=20.0mm clear=FL +8.7/RR +9.1mm 2 planted  fk=20.31mm
                   rock=2.31deg  zEKF-zFK +0.9mm  cyc=3
    [trot] step   7 FL+RR  0.71s (air 0.38s)  swing 20.3mm  rock 2.31deg

    `clear` is the measured height of each swinging foot above the stance
    plane, PER LEG.  Negative means that foot never left the floor at this
    --lift; `air` in the step line then stays at 0.00s, saying the same thing.

    A SPLIT between the two -- one positive, one negative -- is the CoM, not
    the lift: the body is leaning onto the low foot.  Trim it with
    --com-shift-x/y and read the [com] lines for the suggestion.

    The per-step time is the second deliverable: the cadence the ROBOT chose,
    with the exit summary reporting its spread -- that spread is how
    repeatable the FK switch is.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time

import numpy as np

sys.setswitchinterval(0.0005)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import stand_ekf_verify_hw as s1                         # noqa: E402
import stand_ekf_height_hw as s2                         # noqa: E402
import stand_ekf_level_hw as lv                          # noqa: E402

import motorbus                                          # noqa: E402
import stand_dog5_hw as base                             # noqa: E402
import stand_dog5_recorded_hw as recorded                # noqa: E402
import dog5_kinematics                                   # noqa: E402
from ekf_runtime import EkfShared, ekf_worker, _rp       # noqa: E402
from imu_dog import DEFAULT_PORT                         # noqa: E402

# Every tunable this script has is in stand_params.py -- the TROT_* block is
# this gait's; the loops it drives are tuned by LEVEL_* and HEIGHT_*.
from stand_params import (FOOT_RADIUS_M,                 # noqa: E402
                          IMU_BELOW_TRUNK_ORIGIN_M,
                          T_STAND, EKF_WORKER_HZ, QUIET_STAGES, SETTLE_S,
                          TROT_STAND_HEIGHT_M,
                          LEVEL_GAIN_PER_S, LEVEL_CLAMP_M,
                          LATCH_S, AGREE_VETO_DEG,
                          TROT_PAIRS, TROT_LIFT_M, TROT_PUSH_M, TROT_SLEW_M_S,
                          TROT_COM_SHIFT_X, TROT_COM_SHIFT_Y,
                          TROT_AIR_DWELL_S, TROT_OVERLAP_S,
                          TROT_CONTACT_CLEAR_M, TROT_STANCE_REF_S,
                          TROT_LOAD_STILL_M,
                          TROT_SWITCH_TOL_M,
                          TROT_SWITCH_HOLD_S, TROT_SWITCH_TIMEOUT_S,
                          TROT_LEAN_ABORT_DEG, TILT_STOP_DEG)

MOTOR_IDS = base.MOTOR_IDS
N_JOINTS = base.N_JOINTS
CONTROL_HZ = base.CONTROL_HZ
LEGS = base.LEGS
Q_CROUCH = recorded.Q_RECORDED_CROUCH

# ---- imported control laws -- this runner owns none of them ---------------
fk_floor_height_planted = lv.fk_floor_height_planted
height_inputs = lv.height_inputs
LevelingLoop = lv.LevelingLoop
level_inputs = lv.level_inputs
SetpointLatch = lv.SetpointLatch
LevelStandSequence = lv.LevelStandSequence


def measured_foot_z(q, leg):
    """This leg's foot z in the TRUNK frame, from measured encoders.

    Trunk frame, not world: the FK switch compares a leg against its own
    earlier self, and both samples carry the same trunk attitude to first
    order, so rotating into the world would add EKF noise to a test that does
    not need it.
    """
    i = LEGS.index(leg)
    return float(dog5_kinematics.foot_position(leg, np.asarray(q)[3 * i:3 * i + 3])[2])


# ===========================================================================
# Where the weight actually sits -- the CoM, from load sag alone
# ===========================================================================

class StanceLoadBalance:
    """Per-leg load share and the resulting CoM offset, from encoders only.

    WHY THIS EXISTS
        Hardware 2026-08-14: with FL+RR commanded up, RR left the floor and FL
        did not, and the body leaned toward FL.  That is not a lift problem and
        no amount of stance push fixes it.  While a diagonal swings, the robot
        is supported at TWO points, and rotation about the line through them is
        an UNACTUATED degree of freedom -- both stance legs lie on that axis,
        so extending them moves the axis but cannot rotate the body about it.
        Gravity decides that rotation.  If the CoM sits on FL's side of the
        FR-RL line, the body falls toward FL until FL touches down and takes
        load, and commanding FL higher only makes it tip further to meet it.

        The fix is to put the CoM ON the diagonal.  Conveniently, the two
        diagonals cross at the centre of the foot polygon, so ONE constant
        body trim serves both -- `--com-shift-x/y`, applied once at table
        build.  This class measures which trim to use.

    HOW
        A driver's position loop holds against load with a finite stiffness, so
        each leg's foot sits above where it was told to be by an amount
        proportional to what it carries:

            sag_i = z_measured_i - z_commanded_i     (both from FK, no IMU)

        Load share is sag_i / sum(sag), and the CoM is the load-weighted mean
        of the foot anchors.  Both are in the TRUNK frame, which is the frame
        the trim is applied in, so no attitude is needed anywhere.

    HOW MUCH TO BELIEVE IT
        The proportionality is the weak link: backlash, per-leg stiffness
        spread and encoder-zero error all land in `sag` and none of them are
        load.  So treat the DIRECTION as solid and the magnitude as a starting
        point to bisect from -- the honest test is whether both feet of a
        diagonal clear, which `SwingClearance` reports directly.
    """

    def __init__(self, anchors_xy, window_s=TROT_STANCE_REF_S,
                 still_m=TROT_LOAD_STILL_M):
        self.anchors = {leg: np.asarray(anchors_xy[leg][:2], dtype=float)
                        for leg in LEGS}
        self.window_s = float(window_s)
        self.still_m = float(still_m)
        self._acc = None
        self._shown = None
        self.share = None            # per-leg fraction of the total load
        self.com = None              # (x, y) in the trunk frame, metres
        self.moved = False           # the command was ramping: no reading
        self.reason = "no reading yet"

    def reset(self):
        self._acc = None
        self._shown = None
        self.share = None
        self.com = None
        self.moved = False
        self.reason = "no reading yet"

    def note_stance(self, q_meas, q_cmd, now):
        """Feed a sweep with ALL FOUR feet down; latch after `window_s`.

        THE COMMAND MUST BE STILL, and that is not a detail -- hardware
        2026-08-14 produced "RL carries 100% of the weight, trim the body
        342 mm" because this ran while the loops were winding.  `sag` is only
        a load proxy against a STATIONARY target: once the height loop is
        moving the command at HEIGHT_SLEW_M_S and leveling is winding +/-12 mm
        differentially, most of `z_measured - z_commanded` is velocity-lag on
        a moving reference, and the estimate reads the integrators instead of
        the weight.  So a window is thrown away the moment any commanded foot
        z moves more than `still_m`, and the reading is refused outright if a
        leg comes back negative -- a leg cannot carry less than nothing, so a
        negative share means something other than load is dominating and the
        honest answer is no answer.
        """
        cmd = {leg: measured_foot_z(q_cmd, leg) for leg in LEGS}
        sag = {leg: measured_foot_z(q_meas, leg) - cmd[leg] for leg in LEGS}
        if self._acc is None:
            self._acc = [float(now), {leg: [] for leg in LEGS}, dict(cmd)]
        elif any(abs(cmd[leg] - self._acc[2][leg]) > self.still_m
                 for leg in LEGS):
            self.moved = True                    # the command is still ramping
            self._acc = None
            return False
        for leg in LEGS:
            self._acc[1][leg].append(sag[leg])
        if now - self._acc[0] < self.window_s:
            return False
        mean = {leg: float(np.mean(self._acc[1][leg])) for leg in LEGS}
        self._acc = None
        self.moved = False
        if min(mean.values()) < -self.still_m:
            self.share = self.com = None
            self.reason = ("a leg reads negative sag "
                           f"({min(mean.values())*1e3:+.2f} mm) -- not load")
            return False
        total = sum(max(0.0, v) for v in mean.values())
        if total <= 4 * self.still_m:
            self.share = self.com = None
            self.reason = f"sag too small to divide ({total*1e3:.2f} mm total)"
            return False
        self.share = {leg: max(0.0, mean[leg]) / total for leg in LEGS}
        self.com = sum(self.share[leg] * self.anchors[leg] for leg in LEGS)
        self.reason = None
        return True

    def worth_printing(self, change_m=0.003):
        """True the first time, and thereafter only on a real change.

        The 2026-08-14 log is mostly this block repeating at STATUS_HZ with
        the same numbers, which buried the run.
        """
        t = self.trim()
        if t is None:
            return False
        if self._shown is None or np.max(np.abs(t - self._shown)) > change_m:
            self._shown = np.array(t, dtype=float)
            return True
        return False

    def trim(self):
        """The `--com-shift-x/y` that would centre the CoM, or None.

        Returns the BODY shift.  The feet move the other way, which is what
        `build_tables(xy_offset=...)` takes -- the runner negates it there, in
        one place, so this stays in the frame an operator thinks in ("move the
        body forward 8 mm").
        """
        if self.com is None:
            return None
        centre = sum(self.anchors[leg] for leg in LEGS) / len(LEGS)
        return centre - self.com

    def report(self):
        if self.share is None:
            why = ("the command is still ramping -- hold in HOLD4 until the "
                   "height and leveling loops settle" if self.moved
                   else self.reason)
            return [f"[com] no load estimate: {why}"]
        t = self.trim()
        off = -t          # trim recentres, so the offset is its negation
        return [
            "[com] load share from stance sag:  "
            + "  ".join(f"{leg} {self.share[leg]*100:4.1f}%" for leg in LEGS),
            f"[com] CoM offset from the foot-polygon centre: "
            f"x {off[0]*1e3:+5.1f} mm  y {off[1]*1e3:+5.1f} mm",
            f"[com] -> try --com-shift-x {t[0]:+.4f} --com-shift-y {t[1]:+.4f}"
            "   (moves the BODY; direction is solid, magnitude is a start)",
        ]


# ===========================================================================
# Measured clearance -- "did this foot actually leave the floor?"
# ===========================================================================

class SwingClearance:
    """Per-foot height above the stance plane, from encoders + EKF attitude.

    WHY THIS EXISTS INSTEAD OF A COMMANDED CONTACT FLAG
        Every earlier runner derives its contact schedule from what it asked
        for: `lift_ekf_contact_hw` calls a foot airborne once the COMMANDED
        lift passes CONTACT_OFF_M.  That is fair at a 20 mm lift with three
        feet planted.  It is wrong here, twice over:

          * a swinging leg unloads, and a leg that sags 13 mm under load
            (stage 1's measured figure) extends by about that much when the
            weight comes off -- so several millimetres of commanded lift buy
            no ground clearance at all;
          * the stance legs are stiff position holds, so lifting a diagonal
            tips the trunk about the stance diagonal and brings the lifted
            feet back down to meet the floor.

        A command-derived flag would therefore tell the filter that a foot
        carrying weight is in the air -- the exact failure the OFF schedule
        was invented to prevent (README, stage-3 warning).  So measure it.

    HOW
        Both stance feet are on the floor by definition, so the floor is the
        plane through them.  Rotating each foot into the WORLD-VERTICAL axis
        with the EKF's C and subtracting the stance mean gives a height above
        that plane in which the trunk's own position cancels -- only attitude
        is needed, and attitude is the EKF's strongest, AHRS-cross-checked
        output.  Subtracting the same quantity measured in the last 4-foot
        window cancels the constant part (unequal leg zeros, the stand pose's
        own geometry), leaving the CHANGE, which is the clearance.

        This is also the one thing FK cannot do alone: with two stance feet
        the support is a line, and the trunk's rotation about it is invisible
        to the legs (`fk_attitude` returns NaN below three feet).  The tip has
        to come from the EKF.  That is stage 3's whole thesis.

    WHEN IT CANNOT ANSWER
        No EKF attitude, or no stance reference captured yet -> `None`, and
        the caller reports the foot PLANTED.  That is the conservative
        direction: it is the schedule stages 1/2/2b ran with and validated,
        and its failure mode is a bounded, visible zEKF/zFK gap rather than
        throwing away a good leg measurement.
    """

    def __init__(self, clear_m=TROT_CONTACT_CLEAR_M, ref_s=TROT_STANCE_REF_S):
        self.clear_m = float(clear_m)
        self.ref_s = float(ref_s)
        self.ref = None              # per-leg height above the stance mean
        self._acc = None             # (t0, {leg: [samples]}) while all 4 down
        self.last = {leg: float("nan") for leg in LEGS}

    def reset(self):
        self.ref = None
        self._acc = None
        self.last = {leg: float("nan") for leg in LEGS}

    @staticmethod
    def _world_z(q, C):
        """Each foot's height along the WORLD vertical, trunk position aside."""
        q = np.asarray(q, dtype=float)
        out = {}
        for i, leg in enumerate(LEGS):
            s = dog5_kinematics.foot_position(leg, q[3 * i:3 * i + 3])
            out[leg] = float((C.T @ s)[2]) if C is not None else float(s[2])
        return out

    def note_stance(self, q, C, now):
        """Feed a sweep in which ALL FOUR feet are down; refresh the reference.

        Accumulates over `ref_s` and then latches, so the reference is the
        settled 4-foot pose rather than the touchdown transient.
        """
        if C is None:
            self._acc = None
            return False
        z = self._world_z(q, C)
        mean = sum(z.values()) / len(LEGS)
        rel = {leg: z[leg] - mean for leg in LEGS}
        if self._acc is None:
            self._acc = [float(now), {leg: [] for leg in LEGS}]
        for leg in LEGS:
            self._acc[1][leg].append(rel[leg])
        if now - self._acc[0] >= self.ref_s:
            self.ref = {leg: float(np.mean(self._acc[1][leg])) for leg in LEGS}
            self._acc = None
            return True
        return False

    def clearance(self, q, C, swing_legs):
        """Height of each swinging foot above the stance plane (m), or None."""
        if C is None or self.ref is None or not swing_legs:
            return None
        stance = [leg for leg in LEGS if leg not in swing_legs]
        if len(stance) < 2:
            return None
        z = self._world_z(q, C)
        # the stance feet ARE the floor; measure everything against their mean
        base = sum(z[leg] for leg in stance) / len(stance)
        ref_base = sum(self.ref[leg] for leg in stance) / len(stance)
        out = {}
        for leg in swing_legs:
            out[leg] = (z[leg] - base) - (self.ref[leg] - ref_base)
        self.last = {leg: out.get(leg, float("nan")) for leg in LEGS}
        return out

    def airborne(self, q, C, swing_legs):
        """The subset of `swing_legs` measurably off the floor (may be empty)."""
        c = self.clearance(q, C, swing_legs)
        if c is None:
            return set()
        return {leg for leg, v in c.items() if v > self.clear_m}


# ===========================================================================
# The FK switch -- "has this pair come back?"
# ===========================================================================

class TrotFKSwitch:
    """Per-leg return detection from measured foot z, against a baseline.

    `arm(q, legs, now)` samples the baseline at the moment a swing starts.
    `update(q, now)` is called every sweep and returns one of:

        None       still out (or still settling back)
        "returned" every armed leg is within `tol` of its baseline and has
                   been for `hold_s`
        "timeout"  `timeout_s` elapsed without that -- the caller must abort

    WHY A BASELINE AND NOT THE COMMAND: in stance the measured foot z sits
    several millimetres above the commanded one (load sag), so a
    measured-vs-commanded test can never come inside a 1 mm tolerance.  The
    baseline is taken in the same loaded stance the leg will return to, so the
    sag cancels -- and because the sag only reappears as the leg takes weight
    again, the test waits for LOAD TRANSFER, not merely for the encoder to
    arrive.  That is the strongest statement FK can make here; it is still not
    a touchdown sensor (see the module docstring).
    """

    def __init__(self, tol_m=TROT_SWITCH_TOL_M, hold_s=TROT_SWITCH_HOLD_S,
                 timeout_s=TROT_SWITCH_TIMEOUT_S):
        self.tol = float(tol_m)
        self.hold_s = float(hold_s)
        self.timeout_s = float(timeout_s)
        self.baseline = {}
        self.legs = ()
        self.t_arm = None
        self._inside_t0 = None
        self.worst_m = float("nan")

    def arm(self, q, legs, now):
        self.legs = tuple(legs)
        self.baseline = {leg: measured_foot_z(q, leg) for leg in self.legs}
        self.t_arm = float(now)
        self._inside_t0 = None
        self.worst_m = 0.0

    def disarm(self):
        self.legs = ()
        self.baseline = {}
        self.t_arm = None
        self._inside_t0 = None

    @property
    def armed(self):
        return bool(self.legs)

    def error_m(self, q):
        """Worst |measured - baseline| over the armed legs (m)."""
        if not self.legs:
            return float("nan")
        return max(abs(measured_foot_z(q, leg) - self.baseline[leg])
                   for leg in self.legs)

    def update(self, q, now):
        if not self.armed:
            return None
        err = self.error_m(q)
        self.worst_m = max(self.worst_m, err)
        if err <= self.tol:
            if self._inside_t0 is None:
                self._inside_t0 = float(now)
            elif now - self._inside_t0 >= self.hold_s:
                return "returned"
        else:
            self._inside_t0 = None          # left the band: start the hold over
        if now - self.t_arm >= self.timeout_s:
            return "timeout"
        return None


# ===========================================================================
# The gait -- a swing state machine over the two diagonals
# ===========================================================================

class TrotGait:
    """Alternating diagonal pairs, advanced by the FK switch.

    Phases, per pair:

        LIFT      ramp the pair's commanded z up to `lift_m` at `slew`
        AIR       hold there for `air_dwell_s`
        LOWER     ramp back to 0 -- and, once the command reaches 0, ARM the
                  FK switch on the pair
        SETTLE    the FK switch is armed and waiting: the command is home, the
                  legs are not yet.  Nothing moves.
        OVERLAP   the pair has RETURNED, confirmed by FK.  All four feet down
                  for `overlap_s`; this is the only window with 4 planted, so
                  it is the only window the leveling loop runs in.
        -> LIFT on the other pair

    `stop()` is graceful: the current step always finishes and the feet come
    down.  `abort(reason)` is not -- it drops straight to all-feet-down at the
    slew rate, which is what a lean or an FK timeout gets.

    The state machine holds NO absolute times.  Every phase either ramps at a
    slew limit or waits on a duration measured from its own entry, so pausing
    or a slow sweep cannot desynchronise it.
    """

    PHASES = ("IDLE", "LIFT", "AIR", "LOWER", "SETTLE", "OVERLAP", "ABORT")

    def __init__(self, pairs=TROT_PAIRS, lift_m=TROT_LIFT_M,
                 push_m=TROT_PUSH_M, slew_m_s=TROT_SLEW_M_S,
                 air_dwell_s=TROT_AIR_DWELL_S,
                 overlap_s=TROT_OVERLAP_S, switch=None, max_cycles=0):
        self.pairs = tuple(tuple(p) for p in pairs)
        for p in self.pairs:
            if len(p) != 2 or any(leg not in LEGS for leg in p):
                raise ValueError(f"bad trot pair {p}: two known legs required")
        if len({leg for p in self.pairs for leg in p}) != len(LEGS):
            raise ValueError("the pairs must cover all four legs exactly once")
        self.lift_m = float(lift_m)
        self.push_m = float(push_m)
        self.slew = float(slew_m_s)
        self.air_dwell_s = float(air_dwell_s)
        self.overlap_s = float(overlap_s)
        self.switch = switch if switch is not None else TrotFKSwitch()
        self.max_cycles = int(max_cycles)

        self.phase = "IDLE"
        self.pair_i = 0
        self.cur = 0.0               # commanded lift of the ACTIVE pair (m)
        self.cycles = 0              # completed full FL+RR / FR+RL cycles
        self.steps = 0               # completed single-pair steps
        self.stop_requested = False
        self.abort_reason = None
        self._phase_t0 = None
        self._last = None
        self._step_t0 = None
        self._air_s = 0.0
        self.last_step = None        # (pair, total_s, air_s) of the last step

    # -- queries -----------------------------------------------------------
    @property
    def running(self):
        return self.phase not in ("IDLE",)

    @property
    def pair(self):
        return self.pairs[self.pair_i]

    def swing(self):
        """The legs currently commanded off the stance pose (may be empty).

        Note what this is NOT: a claim that these feet are off the floor.
        Whether they cleared is a MEASUREMENT -- see SwingClearance -- and the
        run loop asks for it separately.  Keeping the two apart is the point:
        this method knows what was asked for, and only that.
        """
        if self.phase not in ("LIFT", "AIR", "LOWER", "ABORT") or self.cur <= 0.0:
            return ()
        return self.pair

    def foot_offsets(self):
        """Per-leg z shift (m) for `extra_z`: swing legs UP, stance legs DOWN.

        The stance term is the one hardware insisted on (2026-08-14: the knees
        moved and the feet stayed on the floor).  Lifting a diagonal hands its
        load to the other one, which compresses further and sinks the trunk by
        roughly what the lift asked for, so the swinging feet come back down to
        meet the floor.  Extending the stance pair by `push_m` puts that height
        back, feedforward -- the height loop cannot, at HEIGHT_SLEW_M_S it
        moves 2.5 mm in a swing and it only reacts after the trunk has dropped.

        The push tracks the SAME ramp variable as the lift, so the two are in
        step by construction: no second slew limit, no phase to keep aligned,
        and both are exactly zero whenever the gait is idle.

        Sign: foot z more negative = leg extends = that corner of the trunk
        rises, the convention the height loop and the z tables already use.
        """
        active = self.swing()
        if not active:
            return {leg: 0.0 for leg in LEGS}
        frac = self.cur / self.lift_m if self.lift_m > 0.0 else 0.0
        push = frac * self.push_m
        return {leg: (self.cur if leg in active else -push) for leg in LEGS}

    @staticmethod
    def planted(airborne=()):
        """Contact dict from the MEASURED airborne set."""
        air = set(airborne)
        return {leg: leg not in air for leg in LEGS}

    # -- commands ----------------------------------------------------------
    def start(self, q, now):
        if self.running:
            return "already running"
        self.phase = "LIFT"
        self.pair_i = 0
        self.cur = 0.0
        self.stop_requested = False
        self.abort_reason = None
        self._phase_t0 = float(now)
        self._last = float(now)
        self._step_t0 = float(now)
        self._air_s = 0.0
        return f"gait started on {'+'.join(self.pair)}"

    def stop(self):
        """Graceful: finish this step, put the feet down, then idle."""
        if not self.running:
            return None
        self.stop_requested = True
        return "gait will stop after this step"

    def abort(self, reason):
        """Hard: bring the feet down now and idle.  Lean / FK timeout / veto."""
        if not self.running or self.phase == "ABORT":
            return None
        self.phase = "ABORT"
        self.abort_reason = reason
        self.switch.disarm()
        return f"gait ABORTING: {reason}"

    def reset(self):
        self.phase = "IDLE"
        self.cur = 0.0
        self.pair_i = 0
        self.cycles = self.steps = 0
        self.stop_requested = False
        self.abort_reason = None
        self.switch.disarm()
        self._phase_t0 = self._last = self._step_t0 = None
        self._air_s = 0.0

    # -- the state machine -------------------------------------------------
    def update(self, now, q, airborne=()):
        """Advance one sweep.  Returns an event string, or None.

        `airborne` is the MEASURED set of feet off the floor (SwingClearance),
        used only to score air time -- the state machine itself never branches
        on it, so a clearance measurement that never fires slows nothing down.

        Events: "step_done", "cycle_done", "stopped", "aborted:<reason>",
        "timeout".
        """
        if self._last is None:
            self._last = float(now)
        dt = float(np.clip(now - self._last, 0.0, 0.1))
        self._last = float(now)
        if self.phase == "IDLE":
            return None
        if airborne:
            self._air_s += dt

        step = self.slew * dt

        if self.phase == "ABORT":
            self.cur = max(0.0, self.cur - step)
            if self.cur <= 0.0:
                reason = self.abort_reason or "unspecified"
                self.phase = "IDLE"
                self.cur = 0.0
                self.abort_reason = None
                return f"aborted:{reason}"
            return None

        if self.phase == "LIFT":
            self.cur = min(self.lift_m, self.cur + step)
            if self.cur >= self.lift_m - 1e-12:
                self.cur = self.lift_m
                self._enter("AIR", now)
            return None

        if self.phase == "AIR":
            if now - self._phase_t0 >= self.air_dwell_s:
                self._enter("LOWER", now)
            return None

        if self.phase == "LOWER":
            self.cur = max(0.0, self.cur - step)
            if self.cur <= 1e-12:
                self.cur = 0.0
                # The COMMAND is home; the leg is not necessarily.  Arming
                # here (not at lift) is what makes the baseline the stance
                # pose the leg has to come back to.
                self.switch.arm(q, self.pair, now)
                self._enter("SETTLE", now)
            return None

        if self.phase == "SETTLE":
            verdict = self.switch.update(q, now)
            if verdict == "timeout":
                self.abort(f"FK switch timed out on {'+'.join(self.pair)} "
                           f"({self.switch.error_m(q)*1e3:.1f} mm from baseline)")
                return "timeout"
            if verdict == "returned":
                self.switch.disarm()
                self.steps += 1
                self.last_step = (self.pair, now - self._step_t0, self._air_s)
                self._enter("OVERLAP", now)
                return "step_done"
            return None

        if self.phase == "OVERLAP":
            if now - self._phase_t0 < self.overlap_s:
                return None
            was_last = self.pair_i == len(self.pairs) - 1
            self.pair_i = (self.pair_i + 1) % len(self.pairs)
            if was_last:
                self.cycles += 1
            if self.stop_requested and was_last:
                self.phase = "IDLE"
                return "stopped"
            if self.max_cycles and self.cycles >= self.max_cycles and was_last:
                self.phase = "IDLE"
                return "stopped"
            self._step_t0 = float(now)
            self._air_s = 0.0
            self._enter("LIFT", now)
            return "cycle_done" if was_last else None

        return None

    def _enter(self, phase, now):
        self.phase = phase
        self._phase_t0 = float(now)


# ===========================================================================
# Reporting -- the cadence the robot chose
# ===========================================================================

class TrotReport:
    """Per-step scoring.  The deliverable is the PERIOD and its spread."""

    def __init__(self):
        self.rows = []
        self._peak_lean = 0.0
        self._peak_drift = 0.0
        self._peak_agree = 0.0

    def observe(self, lean_deg, drift_mm, agree_deg):
        """Called every sweep while the gait runs."""
        if math.isfinite(lean_deg):
            self._peak_lean = max(self._peak_lean, abs(lean_deg))
        if math.isfinite(drift_mm):
            self._peak_drift = max(self._peak_drift, abs(drift_mm))
        if math.isfinite(agree_deg):
            self._peak_agree = max(self._peak_agree, agree_deg)

    def add_step(self, pair, total_s, air_s, switch_worst_m):
        self.rows.append({"pair": "+".join(pair), "total_s": float(total_s),
                          "air_s": float(air_s),
                          "swing_mm": float(switch_worst_m) * 1e3,
                          "lean_deg": self._peak_lean})
        self._peak_lean = 0.0
        return (f"[trot] step {len(self.rows):3d} {'+'.join(pair):5s} "
                f"{total_s:5.2f}s (air {air_s:4.2f}s)  "
                f"swing {switch_worst_m*1e3:4.1f}mm  "
                f"rock {self.rows[-1]['lean_deg']:4.2f}deg")

    def summary(self):
        if not self.rows:
            return ["[trot] no steps taken"]
        out = [f"[trot] {len(self.rows)} steps, timed by the FK switch "
               "(the ROBOT chose these):"]
        for pair in sorted({r["pair"] for r in self.rows}):
            t = np.array([r["total_s"] for r in self.rows if r["pair"] == pair])
            a = np.array([r["air_s"] for r in self.rows if r["pair"] == pair])
            out.append(f"  {pair:5s} n={len(t):3d}  period {t.mean():5.2f} "
                       f"+/-{t.std():4.2f}s  air {a.mean():4.2f}s  "
                       f"(min {t.min():4.2f}, max {t.max():4.2f})")
        allt = np.array([r["total_s"] for r in self.rows])
        out.append(f"  cadence {1.0/allt.mean():4.2f} steps/s; period spread "
                   f"{allt.std()/allt.mean()*100:4.1f}% of the mean "
                   "-- this is how repeatable the FK switch is")
        out.append(f"  peak rock {max(r['lean_deg'] for r in self.rows):4.2f} deg, "
                   f"peak |EKF-FK| {self._peak_drift:4.1f} mm, "
                   f"peak |EKF-AHRS| {self._peak_agree:4.2f} deg")
        return out


# ===========================================================================
# Hardware run
# ===========================================================================

def run(port, stand_height, target_height, crouch_max_speed_dps, args):
    base.validate_hardware_config()
    print("[init] building per-leg height tables (IK, once) ...", flush=True)
    t0 = time.perf_counter()
    # The foot x/y offset is the NEGATIVE of the body shift: sliding every
    # foot back slides the trunk forward over the same footprint.  The sign is
    # flipped here, once, so everything else speaks in body terms.
    xy_off = (-args.com_shift_x, -args.com_shift_y)
    tables = s2.build_tables(stand_height,
                             clamp_m=args.clamp + args.level_clamp,
                             xy_offset=xy_off)
    if any(v != 0.0 for v in xy_off):
        print(f"[init] body trimmed {args.com_shift_x*1e3:+.1f} mm x, "
              f"{args.com_shift_y*1e3:+.1f} mm y over the footprint "
              "(feet moved the other way)")
    print(f"[init] tables built in {time.perf_counter()-t0:.2f}s; "
          + "  ".join(f"{leg}[{tables[leg].z_min*1e3:.0f},"
                      f"{tables[leg].z_max*1e3:.0f}]mm" for leg in LEGS))
    auth = min(-tables[leg].z_min - stand_height for leg in LEGS)
    # The stance push spends the SAME extension authority the height loop and
    # the leveling clamp already draw on, and it is the one that runs out
    # first: it is feedforward, so it takes its millimetres whether or not the
    # others need theirs.  Budget all three before the run, not during it.
    need = 0.013 + args.level_clamp + args.push
    print(f"[init] extension authority below the stand: {auth*1e3:.0f} mm; "
          f"budget = 13 sag + {args.level_clamp*1e3:.0f} leveling + "
          f"{args.push*1e3:.0f} push = {need*1e3:.0f} mm")
    if auth < need:
        print(f"[init]   !! SHORT by {(need-auth)*1e3:.0f} mm -- the stance "
              "push will bottom out against the leg's reach and the feet will")
        print("[init]      not clear.  Stand LOWER (--stand-height 0.17 buys "
              "about 20 mm) or reduce --push.")

    if target_height is None:
        target_height = (stand_height + FOOT_RADIUS_M
                         - IMU_BELOW_TRUNK_ORIGIN_M)
    unwrap = [base.CalibratedEncoderUnwrap() for _ in MOTOR_IDS]
    key = base.KeyPoller()
    gate = base.SafetyGate(tau_cap=base.STAGED_TAU_MAX)
    stats = s1.AttitudeStats()
    report = s1.StageReport()
    hctrl = s2.HeightController(gain_per_s=args.height_gain, clamp_m=args.clamp)
    lvl = LevelingLoop(anchors_xy={leg: tables[leg].xy for leg in LEGS},
                       gain_per_s=args.level_gain, clamp_m=args.level_clamp)
    latch = SetpointLatch(
        math.radians(args.setpoint_roll) if args.setpoint_roll is not None else None,
        math.radians(args.setpoint_pitch) if args.setpoint_pitch is not None else None)
    switch = TrotFKSwitch(tol_m=args.switch_tol, timeout_s=args.switch_timeout)
    clearance = SwingClearance(clear_m=args.contact_clear)
    balance = StanceLoadBalance(anchors_xy={leg: tables[leg].xy for leg in LEGS})
    gait = TrotGait(lift_m=args.lift, push_m=args.push, overlap_s=args.overlap,
                    switch=switch, max_cycles=args.cycles)
    trot_report = TrotReport()
    agree_veto_rad = math.radians(args.agree_veto)
    lean_abort_rad = math.radians(args.lean_abort)
    tilt_stop_rad = math.radians(args.tilt_stop)

    print("=" * 78)
    print("DOG5 TROT IN PLACE  (stage 3: diagonal pairs, switched by FK)")
    print("  control laws imported from stand_ekf_level_hw; this runner adds "
          "the gait only")
    print(f"  gait:     {args.lift*1e3:.0f} mm lift + {args.push*1e3:.0f} mm "
          f"stance push at {TROT_SLEW_M_S*1e3:.0f} mm/s, "
          f"air dwell {TROT_AIR_DWELL_S*1e3:.0f} ms, "
          + (f"4-foot overlap {args.overlap*1e3:.0f} ms"
             if args.overlap > 0 else "NO overlap -- TRUE TROT, diagonal support")
          + (f", stop after {args.cycles} cycles" if args.cycles else ""))
    print(f"  switch:   FK return within {args.switch_tol*1e3:.1f} mm of the "
          f"pre-swing baseline, held {TROT_SWITCH_HOLD_S*1e3:.0f} ms; "
          f"timeout {args.switch_timeout:.1f}s -> abort")
    print(f"  leveling: gain {args.level_gain:.2f}/s, clamp "
          f"+/-{args.level_clamp*1e3:.0f} mm/foot, AHRS veto {args.agree_veto:.1f} deg"
          + ("  setpoint FIXED "
             f"({args.setpoint_roll:+.2f},{args.setpoint_pitch:+.2f}) deg"
             if latch.fixed else f"  setpoint auto-latched over {LATCH_S:.0f}s"))
    print("            (it freezes itself below 3 planted feet, so it runs "
          "only in the overlap window)")
    print(f"  height:   gain {args.height_gain:.2f}/s, clamp "
          f"+/-{args.clamp*1e3:.0f} mm, target {target_height*1e3:.0f} mm at "
          "the TRUNK BOTTOM/IMU")
    print(f"  safety:   lean-abort {args.lean_abort:.1f} deg (stops the gait), "
          f"tilt-stop {args.tilt_stop:.0f} deg (stops the run)")
    print("  NO torque commanded; drivers hold position.  Lifting a diagonal "
          "ROCKS the trunk --")
    print("  commanded lift height IS rocking amplitude.  Start small.")
    print("  Keys: ENTER stand, T start/stop the gait, P park, X stop.")
    if args.fake_contacts:
        print("  !! --fake-contacts: the EKF is told all 4 feet stay planted "
              "through every swing (A/B).")
    if args.no_limit_check:
        print("  !! --no-limit-check: measured-pose limits OFF (preflight "
              "refusal + runtime estop).")
    print("=" * 78)

    enc_log = []
    stop_reason = None
    worker = shared = None
    log_t0 = None
    init_secs = 0.0
    z0_fk_imu = None
    z0_report = None
    try:
        from imu_ekf_feed import ImuEkfFeed
        with motorbus.MotorBus(MOTOR_IDS, dirs=base.MOTOR_DIRECTIONS) as mb, \
                ImuEkfFeed(port) as feed:
            if not mb.arm(rate_hz=CONTROL_HZ):
                raise RuntimeError("arm failed (bus / power / terminators)")
            if not feed.wait_for_raw(timeout=3.0):
                raise RuntimeError("no raw IMU (0x40) packets -- enable DETA10 raw mode")

            if args.no_limit_check:
                _saved_limits = base.soft_limits
                base.soft_limits = lambda: (np.full(N_JOINTS, -np.inf),
                                            np.full(N_JOINTS, np.inf))
                try:
                    start_q = base._zero_torque_preflight(mb, key, unwrap)
                finally:
                    base.soft_limits = _saved_limits
            else:
                start_q = base._zero_torque_preflight(mb, key, unwrap)
            if start_q is None:
                print("[abort] preflight not confirmed")
                return

            _lo, _hi = base.soft_limits()
            _out = np.flatnonzero((start_q < _lo) | (start_q > _hi))
            if len(_out):
                print("[preflight] measured pose OUTSIDE the soft limits:")
                for j in _out:
                    over = (start_q[j] - _hi[j] if start_q[j] > _hi[j]
                            else start_q[j] - _lo[j])
                    print(f"    {base.JOINT_LABELS[j]:9s} "
                          f"{math.degrees(start_q[j]):+8.1f} deg  "
                          f"({math.degrees(over):+.1f} deg past the limit)")
            else:
                print("[preflight] measured pose is inside the soft limits.")

            now = time.perf_counter()
            gate.start(now, start_q)
            shared = EkfShared(start_q)
            worker = threading.Thread(
                target=ekf_worker, args=(shared, feed),
                kwargs=dict(quiet_stages=QUIET_STAGES, control_hz=EKF_WORKER_HZ),
                daemon=True)
            worker.start()

            seq = LevelStandSequence(tables, stand_height)
            miss_monitor = base.CanMissMonitor(mb)
            slot = mb.slot(CONTROL_HZ)
            deadline = time.perf_counter() + slot
            status_period = 1.0 / base.FAULT_STATUS_HZ
            next_fault_status = np.array(
                [status_period + i * status_period / N_JOINTS
                 for i in range(N_JOINTS)])
            last_recover = {mid: -base.RECOVER_PERIOD_S for mid in MOTOR_IDS}
            start = now
            cmd_deg = recorded.POSITION_TARGET_DEG.copy()
            index = 0
            last_print = 0.0
            tx_fail = 0
            bias_printed = False
            quiet_t0 = None
            last_h_reason = None
            last_l_reason = None
            prev_now = now

            while True:
                mb.poll()
                joint_index = index % N_JOINTS
                if joint_index == 0:
                    now = time.perf_counter()
                    dt_sweep = now - prev_now
                    prev_now = now
                    now_mono = time.monotonic()
                    q, qd = base._joint_state(mb, unwrap)

                    pressed = key.get()
                    if pressed in ("x", "X"):
                        stop_reason = "operator X"
                        break

                    # ---- AHRS (leveling's independent watchdog) ----
                    ahrs_rp = None
                    r_a = p_a = float("nan")
                    att = feed.attitude()
                    if att is not None:
                        r_a = math.radians(att.roll_deg)
                        p_a = math.radians(att.pitch_deg)
                        ahrs_rp = (r_a, p_a)

                    # ---- gait, then the MEASURED contact schedule ----
                    # Order matters: the clearance is measured from the pose
                    # the encoders are reporting NOW, against a reference
                    # taken while all four feet were down, and the result --
                    # not the commanded lift -- is what the EKF is told.
                    out_now = shared.out
                    C_now = (out_now["C"]
                             if (shared.est_ready and out_now is not None)
                             else None)
                    swing = gait.swing()
                    if not swing:
                        clearance.note_stance(q, C_now, now)
                    air = clearance.airborne(q, C_now, swing)
                    gait_event = gait.update(now, q, air)
                    planted_d = gait.planted(air)
                    planted_a = np.array([planted_d[leg] for leg in LEGS])
                    n_planted = int(planted_a.sum())

                    # ---- leveling loop ----
                    # NOTE: no extra freeze is wired in for the swing phase.
                    # LevelingLoop.update already refuses to integrate below 3
                    # planted feet, so a diagonal-support phase holds its
                    # offsets by construction -- one mechanism, not two.
                    roll_e, pitch_e, l_active, l_reason = level_inputs(
                        shared, ahrs_rp, now_mono, agree_veto_rad)
                    l_engaged = l_active and seq.loop_engaged and latch.ready
                    if not l_active and gait.running:
                        msg = gait.abort(l_reason or "leveling lost")
                        if msg:
                            print(f"[trot] {msg}")
                    lvl.update(now, roll_e, pitch_e, latch.sp_roll,
                               latch.sp_pitch, planted_d, l_engaged,
                               l_reason if not l_active else "not holding")
                    if seq.loop_engaged and latch.ready and l_reason \
                            and l_reason != last_l_reason:
                        print(f"[level] FROZEN: {l_reason}")
                    last_l_reason = l_reason

                    err_roll = err_pitch = float("nan")
                    lean_deg = float("nan")
                    if latch.ready and math.isfinite(roll_e):
                        err_roll = roll_e - latch.sp_roll
                        err_pitch = pitch_e - latch.sp_pitch
                        lean_deg = math.degrees(max(abs(err_roll), abs(err_pitch)))
                        if gait.running and max(abs(err_roll),
                                                abs(err_pitch)) > lean_abort_rad:
                            msg = gait.abort(f"lean {lean_deg:.1f}deg")
                            if msg:
                                print(f"[trot] {msg}")

                    # ---- tilt run-stop (absolute) ----
                    if math.isfinite(roll_e) and \
                            max(abs(roll_e), abs(pitch_e)) > tilt_stop_rad:
                        stop_reason = (f"tilt-stop: |attitude| "
                                       f"{math.degrees(max(abs(roll_e), abs(pitch_e))):.1f}deg")
                        break

                    # ---- height loop (planted-aware FK veto) ----
                    h_ekf, h_fk, h_active, h_reason = height_inputs(
                        shared, q, z0_fk_imu, now_mono, planted_a,
                        xcheck_m=args.xcheck)
                    h_engaged = h_active and seq.loop_engaged
                    offset = hctrl.update(now, h_ekf, target_height, h_engaged,
                                          h_reason if not h_active else "not holding")
                    if seq.loop_engaged and h_reason and h_reason != last_h_reason:
                        print(f"[height] FROZEN: {h_reason} (holding "
                              f"{offset*1e3:+.1f} mm)")
                    last_h_reason = h_reason

                    # ---- compose per-leg targets ----
                    gz = gait.foot_offsets()
                    seq.extra_z = {leg: lvl.offsets[leg] + gz[leg] for leg in LEGS}
                    seq.planted_mask = planted_a

                    park_req = pressed in ("p", "P")
                    if park_req and gait.running:
                        print("[park] refused: stop the gait first (press T)")
                        park_req = False
                    q_cmd, contacts, event = seq.update(
                        now, q, qd,
                        enter_pressed=pressed in ("\r", "\n"),
                        park_pressed=park_req,
                        offset=offset)
                    cmd_deg = np.rad2deg(q_cmd)
                    if seq.fault:
                        stop_reason = seq.fault
                        break

                    # Load balance needs the COMMANDED pose, so it runs here
                    # rather than beside the clearance estimate above.  Only
                    # while all four are down and nothing is ramping: a leg
                    # chasing a moving target has tracking error on top of
                    # sag, and that is not load.
                    if not swing and seq.stage == "HOLD4":
                        if (balance.note_stance(q, q_cmd, now)
                                and not gait.running
                                and balance.worth_printing()):
                            for ln in balance.report():
                                print(ln)

                    # ---- gait keys (after seq.update so the stage is fresh) --
                    if pressed in ("t", "T"):
                        if seq.stage != "HOLD4":
                            print("[trot] refused: only in HOLD4")
                        elif gait.running:
                            print(f"[trot] {gait.stop()}")
                        elif not l_engaged:
                            print("[trot] refused: leveling loop not engaged "
                                  f"({l_reason or 'setpoint not latched yet'})")
                        else:
                            print(f"[trot] {gait.start(q, now)}")

                    # ---- gait events ----
                    if gait_event == "step_done" and gait.last_step:
                        pair, total_s, air_s = gait.last_step
                        print(trot_report.add_step(pair, total_s, air_s,
                                                   switch.worst_m))
                    elif gait_event == "stopped":
                        print(f"[trot] gait stopped after {gait.cycles} "
                              f"cycles, {gait.steps} steps; all feet down.")
                    elif gait_event and gait_event.startswith("aborted:"):
                        print(f"[trot] gait ABORTED ({gait_event[8:]}); "
                              "all feet down.")

                    shared.q = q
                    shared.qd = qd
                    shared.stage = seq.stage
                    # --fake-contacts LIES to the EKF: the motion and zFK stay
                    # honest, only the filter's contact belief changes.
                    shared.contacts = (np.ones(4, dtype=bool)
                                       if args.fake_contacts else contacts)

                    if event == "crouch_settled":
                        quiet_t0 = now
                        print("[stage] CROUCH settled; EKF initialising. "
                              "ENTER to STAND.")
                    elif event == "stand_started":
                        quiet_t0 = None
                        hctrl.reset(0.0)
                        lvl.reset()
                        latch.reset()
                        gait.reset()
                        clearance.reset()
                        balance.reset()
                        seq.clear_extra()
                        if seq.n_stands == 0 and log_t0 is not None:
                            init_secs = (now - start) - log_t0
                        print("[stage] STAND: open-loop ramp crouch -> stand.")
                    elif event == "stand_complete":
                        quiet_t0 = now
                        stats.begin_visit()
                        print(f"[stage] HOLD4 (stand #{seq.n_stands}): height "
                              "loop ENGAGED; leveling engages once the "
                              "setpoint latches, then T starts the gait.")
                    elif event == "park_started":
                        quiet_t0 = None
                        line = stats.end_visit()
                        if line:
                            print("[attitude] this stand scored:")
                            print(line)
                        print("[stage] PARK: unwinding offsets on the way down.")
                    elif event == "park_complete":
                        quiet_t0 = now
                        hctrl.reset(0.0)
                        lvl.reset()
                        latch.reset()
                        gait.reset()
                        clearance.reset()
                        balance.reset()
                        seq.clear_extra()
                        print("[stage] PARKED: holding the recorded crouch. "
                              "ENTER stands again, X stops.")

                    if not bias_printed and shared.est_ready:
                        print(f"[ekf] init: {shared.bias_str}")
                        bias_printed = True
                    if log_t0 is None and shared.est_ready \
                            and seq.stage == "WAIT_CROUCH":
                        shared.log_enabled = True
                        log_t0 = now - start

                    # ---- safety ----
                    temps = base._temperatures(mb)
                    misses = miss_monitor.update(mb)
                    errors = mb.errors()
                    if stop_reason is None:
                        stop_reason = gate.estop_reason(
                            q, qd, temps, misses, errors, now,
                            enforce_position_limits=not args.no_limit_check)
                    if stop_reason is None:
                        stop_reason = gate.overspeed_reason(qd, q, now)
                    if stop_reason:
                        break
                    latched_err = [mid for mid, e in errors.items()
                                   if e and (e & 0x80)]
                    recover = [mid for mid in latched_err
                               if (now - start) - last_recover[mid]
                               >= base.RECOVER_PERIOD_S]
                    if recover:
                        base._recover_input_lost(mb, recover, now - start,
                                                 last_recover, next_fault_status)

                    # ---- observe / log / latch ----
                    out = shared.out
                    re_ = pe = float("nan")
                    drift_mm = float("nan")
                    if shared.est_ready and out is not None:
                        re_, pe = _rp(out["C"])
                        z_fk_imu = fk_floor_height_planted(
                            q, out["C"], planted_a, ref="imu")
                        if z0_fk_imu is None:
                            z0_fk_imu = z_fk_imu
                        if z0_report is None:
                            z0_report = z_fk_imu
                        report.set_origin(z_fk_imu)
                        drift_mm = (float(out["r"][2])
                                    - (z_fk_imu - z0_report)) * 1e3
                        if quiet_t0 is not None and now - quiet_t0 >= SETTLE_S:
                            report.add(seq.stage, z_fk_imu,
                                       float(out["r"][2]), re_, pe)
                            if seq.stage == "HOLD4":
                                stats.add(re_, pe, r_a, p_a)
                                # the setpoint must be the RESTING attitude:
                                # never latch while the gait is rocking it
                                if not latch.ready and l_active \
                                        and not gait.running:
                                    if latch.add(now, re_, pe):
                                        print(f"[level] setpoint latched: roll "
                                              f"{math.degrees(latch.sp_roll):+.2f} "
                                              f"pitch {math.degrees(latch.sp_pitch):+.2f} "
                                              "deg -- leveling ENGAGED, "
                                              "T starts the gait.")
                    agree_deg = float("nan")
                    if math.isfinite(re_) and not math.isnan(r_a):
                        agree_deg = math.degrees(max(abs(re_ - r_a),
                                                     abs(pe - p_a)))
                    if gait.running:
                        trot_report.observe(lean_deg, drift_mm, agree_deg)

                    if log_t0 is not None:
                        enc_log.append((now_mono, *q, *contacts.astype(int),
                                        r_a, p_a))

                    if now - last_print >= 1.0 / base.STATUS_HZ:
                        last_print = now
                        if not (shared.est_ready and out is not None):
                            print(f"[hw] {seq.stage:11s} waiting for EKF  "
                                  f"Tmax={int(np.max(temps))}C", flush=True)
                        elif gait.running:
                            # focused while stepping: phase, who is up, and
                            # how far the FK switch still has to go
                            serr = (switch.error_m(q) * 1e3 if switch.armed
                                    else float("nan"))
                            # `clear` is the number the first runs exist to
                            # produce: MEASURED height of the swinging feet
                            # above the stance plane.  If it never passes
                            # --contact-clear, the feet are not leaving the
                            # floor and this is a rock, not a step.
                            # PER LEG, not a single number: the 2026-08-14
                            # failure was one foot of the pair clearing and
                            # the other not, which a min() or a mean() hides.
                            # A split between these two IS the body leaning
                            # onto the low one -- see StanceLoadBalance.
                            cl = "/".join(
                                f"{leg}{clearance.last[leg]*1e3:+5.1f}"
                                for leg in (swing or gait.pair))
                            print(f"[trot] {gait.phase:7s} "
                                  f"lift={gait.cur*1e3:4.1f}mm "
                                  f"clear={cl}mm "
                                  f"{n_planted} planted  "
                                  f"fk={serr:5.2f}mm  "
                                  f"rock={lean_deg:4.2f}deg  "
                                  f"zEKF-zFK={(h_ekf-h_fk)*1e3:+5.1f}mm  "
                                  f"cyc={gait.cycles}", flush=True)
                        else:
                            lflag = ("LVL" if l_engaged else
                                     ("latching" if seq.loop_engaged
                                      and not latch.ready else
                                      ("HOLD" if seq.loop_engaged else "----")))
                            e_r = math.degrees(err_roll) \
                                if math.isfinite(err_roll) else float("nan")
                            e_p = math.degrees(err_pitch) \
                                if math.isfinite(err_pitch) else float("nan")
                            warn = ""
                            if latched_err:
                                warn += f" latched={len(latched_err)}"
                            if tx_fail:
                                warn += f" txfail={tx_fail}"
                            print(f"[hw] {seq.stage:11s} [{lflag:8s}] "
                                  f"zEKF={h_ekf*1e3:7.1f} zFK={h_fk*1e3:7.1f} "
                                  f"diff={(h_ekf-h_fk)*1e3:+5.1f}mm  "
                                  f"tilt={e_r:+5.2f}/{e_p:+5.2f}deg  "
                                  f"Tmax={int(np.max(temps))}C{warn}",
                                  flush=True)

                mid = MOTOR_IDS[joint_index]
                if (time.perf_counter() - start) >= next_fault_status[joint_index]:
                    mb.status1_req(mid)
                    next_fault_status[joint_index] += status_period
                elif not mb.position(mid, float(cmd_deg[joint_index]),
                                     crouch_max_speed_dps):
                    tx_fail += 1
                index += 1
                overrun = mb.pace(deadline)
                deadline += slot
                if overrun and overrun > 2.0 * slot:
                    deadline = time.perf_counter() + slot

            base._soft_stop(mb)
    except KeyboardInterrupt:
        stop_reason = stop_reason or "KeyboardInterrupt"
    finally:
        if shared is not None:
            shared.run = False
        if worker is not None:
            worker.join(timeout=1.0)
        try:
            key.close()
        except Exception:
            pass
        if args.raw_log and shared is not None:
            imu = np.array(shared.imu_log, dtype=float)
            enc = np.array(enc_log, dtype=float)
            if len(imu) and len(enc):
                np.savez(args.raw_log,
                         imu_t=imu[:, 0], imu_f=imu[:, 1:4], imu_w=imu[:, 4:7],
                         enc_t=enc[:, 0], enc_alpha=enc[:, 1:13],
                         enc_contacts=enc[:, 13:17], ahrs_rp=enc[:, 17:19],
                         init_secs=max(init_secs, 0.5))
                print(f"[raw] wrote {args.raw_log} "
                      f"({len(imu)} IMU, {len(enc)} enc frames)")
    stats.end_visit()
    for line in stats.summary():
        print(line)
    for line in report.summary():
        print(line)
    for line in trot_report.summary():
        print(line)
    for line in balance.report():
        print(line)
    print(f"[height] final offset {hctrl.offset*1e3:+.1f} mm"
          + (" (SATURATED)" if hctrl.saturated else ""))
    print("[level] final offsets "
          + "  ".join(f"{leg}{lvl.offsets[leg]*1e3:+.1f}" for leg in LEGS)
          + " mm" + (" (SATURATED)" if lvl.saturated else ""))
    print(f"[stop] {stop_reason}")


# ===========================================================================
# offline self-test -> test_trot_fk_switch.py
# ===========================================================================

def self_test(stand_height):
    """Delegate to the suite in self-test/.  Lazy: a hardware run never loads it."""
    _sd = s1.selftest_dir()
    if _sd not in sys.path:
        sys.path.insert(0, _sd)
    from test_trot_fk_switch import self_test as gates
    return gates(stand_height)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--stand-height", type=float, default=TROT_STAND_HEIGHT_M,
                    help="LOWER than the rest of the track (0.19) on purpose: "
                         "the stance push needs extension authority below the "
                         "stand pose, and 0.19 does not have enough")
    ap.add_argument("--target-height", type=float, default=None,
                    help="height-loop target, metres from the floor to the "
                         "TRUNK BOTTOM / IMU BOARD")
    ap.add_argument("--crouch-max-speed-dps", type=float,
                    default=recorded.DEFAULT_CROUCH_MAX_MOTOR_DPS)
    ap.add_argument("--height-gain", type=float, default=s2.HEIGHT_GAIN_PER_S)
    ap.add_argument("--clamp", type=float, default=s2.HEIGHT_CLAMP_M)
    ap.add_argument("--xcheck", type=float, default=s2.HEIGHT_XCHECK_M)
    ap.add_argument("--level-gain", type=float, default=LEVEL_GAIN_PER_S)
    ap.add_argument("--level-clamp", type=float, default=LEVEL_CLAMP_M)
    ap.add_argument("--agree-veto", type=float, default=AGREE_VETO_DEG)
    ap.add_argument("--setpoint-roll", type=float, default=None)
    ap.add_argument("--setpoint-pitch", type=float, default=None)
    ap.add_argument("--lift", type=float, default=TROT_LIFT_M,
                    help="commanded lift per swinging foot (m).  On stiff "
                         "position legs this is ROCKING AMPLITUDE, not "
                         "ground clearance -- raise it a millimetre at a time")
    ap.add_argument("--com-shift-x", type=float, default=TROT_COM_SHIFT_X,
                    help="slide the BODY this far forward (+x) over the "
                         "footprint, metres.  Use it to put the CoM on the "
                         "support diagonals so a swinging pair does not tip "
                         "the body onto one of its own feet.  The run prints "
                         "a measured suggestion; direction is reliable, "
                         "magnitude is a starting point to bisect from")
    ap.add_argument("--com-shift-y", type=float, default=TROT_COM_SHIFT_Y,
                    help="slide the BODY this far to the left (+y), metres")
    ap.add_argument("--push", type=float, default=TROT_PUSH_M,
                    help="how far the STANCE pair extends while the other "
                         "diagonal swings (m).  Without it the trunk sinks by "
                         "about what the lift asked for and the feet never "
                         "leave the floor.  clearance ~ lift + push - 26 mm")
    ap.add_argument("--overlap", type=float, default=TROT_OVERLAP_S,
                    help="seconds with all four feet down between steps. "
                         "0 = no 4-foot window = a true trot (diagonal "
                         "support only, roll observable to the EKF alone)")
    ap.add_argument("--cycles", type=int, default=0,
                    help="stop the gait after this many full FL+RR/FR+RL "
                         "cycles (0 = run until T or P)")
    ap.add_argument("--contact-clear", type=float, default=TROT_CONTACT_CLEAR_M,
                    help="MEASURED clearance above the stance plane beyond "
                         "which a foot is reported airborne to the EKF (m). "
                         "Not the commanded lift -- see SwingClearance")
    ap.add_argument("--switch-tol", type=float, default=TROT_SWITCH_TOL_M,
                    help="FK return tolerance against the pre-swing baseline (m)")
    ap.add_argument("--switch-timeout", type=float, default=TROT_SWITCH_TIMEOUT_S,
                    help="no FK confirmation within this -> abort the gait (s)")
    ap.add_argument("--lean-abort", type=float, default=TROT_LEAN_ABORT_DEG,
                    help="attitude error that stops the gait (deg)")
    ap.add_argument("--tilt-stop", type=float, default=TILT_STOP_DEG,
                    help="absolute attitude that soft-stops the run (deg)")
    ap.add_argument("--fake-contacts", action="store_true",
                    help="A/B: tell the EKF all four feet stay planted through "
                         "every swing.  Motion and zFK unchanged; only the "
                         "filter's contact input is wrong.")
    ap.add_argument("--no-limit-check", action="store_true",
                    help="accept ANY measured joint pose (see README's "
                         "FR_knee/id12 note)")
    ap.add_argument("--raw-log", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if (args.setpoint_roll is None) != (args.setpoint_pitch is None):
        ap.error("--setpoint-roll and --setpoint-pitch go together")
    if args.self_test:
        sys.exit(self_test(args.stand_height))
    run(args.port, args.stand_height, args.target_height,
        args.crouch_max_speed_dps, args)


if __name__ == "__main__":
    main()
