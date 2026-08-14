#!/usr/bin/env python3
"""Experiment A: leveling stand closed on the RAW AHRS -- NO EKF anywhere.

WHY THIS EXISTS
    The 2026-08-12 stand_ekf_level_hw run froze leveling for its entire HOLD4:
    the EKF's attitude split 4-5 deg from the AHRS during the STAND ramp
    (false all-planted contacts dragged the filter) and the --agree-veto
    correctly refused to close a loop on it.  This script removes the EKF from
    the experiment entirely: the leveling loop is fed the DETA10's OWN fused
    attitude (packet 0x41, computed on the sensor; the Pi only frame-transforms
    NED->FLU and subtracts the flat-calibrated mount offsets).  AHRS zero IS
    level by calibration, so the default setpoint is (0, 0).

    If leveling works here and not in the EKF script, the leveling law and the
    legs are fine and the EKF's contact handling is the problem -- which is
    exactly what Experiment B (stand_ekf_schedcontact_hw.py) then fixes.

WHAT IS DELIBERATELY ABSENT (one variable at a time)
    * no EKF worker, no EkfShared, no [ekf] anything
    * no height loop -- the stand height is OPEN LOOP (the validated stage-1
      pose); sag stays, ~13 mm, and that is fine: this run tests ATTITUDE only
    * no setpoint latch -- the premise is that AHRS 0 = level
    * no EKF-vs-AHRS veto (nothing to compare); safety is the stale-AHRS
      freeze, the per-foot clamp/slew, the tilt-stop, and base's estop

WHAT IS DELIBERATELY PRESENT: A P TERM  (the EKF script has none)
    The EKF script's law is a pure integrator, because a P term there
    multiplies estimator noise onto stiff position-mode legs.  This loop is
    closed on the AHRS instead, whose noise floor is under the deadband, so
    the objection does not apply and the integrator's 4 s tail is just cost.
    `PiLevelingLoop` adds P -- and moves ki and slew with it, because the
    naive change makes the robot SLOWER:

        closed-loop pole = ki / (1 + kp)      (unity plant gain, by geometry)

    P shrinks the error the integrator sees by (1+kp) and slows the integrator
    by the same (1+kp).  Bolt kp=1 onto ki=0.25/s and the head looks brisk
    while the tail takes twice as long.  So: kp 1.0, ki 1.0/s, slew 10 mm/s.
    Self-test gate [2] measures all three tunings and prints the times.

    Above ~0.3 deg the loop is pinned to the SLEW anyway, so --level-slew is
    the real speed knob; the gains decide how fast the target gets ahead of
    the feet, the slew decides how fast the feet may chase it.

HOW TO RUN
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    sudo chrt -f 50 $V stand_ahrs_level_hw.py --level-gain 0 --level-kp 0
                                                   # A/B baseline (open loop)
    sudo chrt -f 50 $V stand_ahrs_level_hw.py      # leveling live, PI
    # the EKF script's exact law, for the like-for-like A/B:
    sudo chrt -f 50 $V stand_ahrs_level_hw.py --level-kp 0 --level-gain 0.25 \
                                              --level-slew 0.004

    Keys: ENTER stand, P park, X stop.
    Success = rp:imu collapses to (0,0) within the deadband in HOLD4 and the
    lvl offsets absorb the tilt.  Shim test: put 10 mm under one side's feet
    while it holds -- roll must return to ~0 and lvl must wind ~+/-5 mm.

STATUS LINE
    [hw] STAGE[flag] rp:imu=roll/pitch fk=roll/pitch err=... lvl=FL/FR/RL/RR
    imu = AHRS attitude (the feedback).  fk = plane-through-the-feet from
    encoders (display only -- it reads ~0 in position mode).  flag: LVL live,
    HOLD frozen (reason printed once), ---- outside HOLD4.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

sys.setswitchinterval(0.0005)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import stand_ekf_height_hw as s2                         # noqa: E402  tables
import stand_ekf_level_hw as lv                          # noqa: E402  law+seq

import motorbus                                          # noqa: E402
import stand_dog5_hw as base                             # noqa: E402
import stand_dog5_recorded_hw as recorded                # noqa: E402
from imu_dog import ImuDog, DEFAULT_PORT                 # noqa: E402

# Every tunable this script has is in stand_params.py.  This experiment's
# gains are the AHRS_* block there -- named apart from the EKF script's
# LEVEL_* on purpose: they are a DIFFERENT tuning of the same law, and while
# they shared a name the one that won depended on which module you were in.
from stand_params import (STAND_HEIGHT_DEFAULT,          # noqa: E402
                          TEMP_NOTICE_C, TILT_STOP_DEG, SETTLE_S,
                          AHRS_KP, AHRS_GAIN_PER_S, AHRS_SLEW_M_S,
                          AHRS_STALE_S, GAP_BUCKETS)

MOTOR_IDS = base.MOTOR_IDS
N_JOINTS = base.N_JOINTS
CONTROL_HZ = base.CONTROL_HZ
LEGS = base.LEGS


def _shrink(e, band):
    """Soft deadband: the first `band` of error costs nothing, the rest passes.

    Continuous at +/-band, unlike the hard deadband the integrator uses.  A P
    term on a HARD deadband steps its output by kp*band every time the error
    crosses the edge, which is a chatter generator; this just slides the curve.
    """
    if e > band:
        return e - band
    if e < -band:
        return e + band
    return 0.0


class PiLevelingLoop(lv.LevelingLoop):
    """`lv.LevelingLoop` plus a PROPORTIONAL term -- AHRS-only, so speed wins.

    WHY THE EKF SCRIPT HAS NO P TERM
        There the feedback is an EKF estimate and a P term multiplies its noise
        straight onto the foot targets, in stiff position mode, with no
        compliance anywhere to absorb it.  The integrator was the whole point:
        it cancels a DC error (sag, mount tilt, floor slope) while low-passing
        the estimate.

    WHY THIS ONE DOES
        The feedback here is the DETA10's own fused attitude (packet 0x41) --
        a sensor-side filter that no contact assumption can drag.  Its noise
        floor sits well under the 0.2 deg deadband, so the reason to refuse a
        P term is gone, and what is left is the integrator's cost: at ki =
        0.25/s a tilt takes tau = 4 s to wind in, and the loop spends the
        first second barely moving.

    THE LAW (per planted leg i, anchors (x_i, y_i), attitude errors phi/theta)

        e_i    = -x_i*theta + y_i*phi         # foot z that cancels the tilt
        d_i    = e_i - mean_planted(e)        # zero-mean: never fights height
        I_i   += clip(ki * d_i(hard db) * dt, +/-slew*dt)
        P_i    = kp * d_i(soft db)
        out_i += clip(I_i + P_i - out_i, +/-slew*dt)        # rate-limited TOTAL

    P IS FOR THE HEAD, I IS FOR THE TAIL -- KEEP BOTH.  The plant gain from
    offset to attitude is UNITY by construction (commanding off_i = e_i
    cancels the tilt exactly), so proportional-only feedback parks at
        err_ss = disturbance / (1 + kp)
    -- kp = 1 still leaves HALF the tilt, kp = 9 leaves a tenth, and nothing
    short of infinite gain leaves none.  Only the integrator reaches the
    deadband.  Gate [2] measures this residual and holds the design to it.

    AND P SLOWS THE TAIL, WHICH IS WHY ki HAD TO MOVE TOO.  With a static
    unity plant the closed loop is a single pole at
        1/tau = ki / (1 + kp)
    so bolting kp = 1 onto the EKF script's ki = 0.25/s would have DOUBLED the
    settling time while making the first second look brisk.  P shrinks the
    error the integrator sees by (1+kp) and slows it by the same factor: the
    two cancel, and to a fixed tolerance the naive PI is *worse* than pure I.
    Hence AHRS_GAIN_PER_S = 1.0 here.  Gate [2] measures the settling times
    of I-only, naive-PI and this tuning side by side so the claim is checked
    and not asserted.

    THE SLEW LIMIT NOW BOUNDS THE TOTAL, not just the integrator, and it is
    what keeps a P term safe on stiff position-mode legs: an AHRS step (or
    re-engaging after a freeze) moves the TARGET instantly, but the commanded
    foot z still ramps at `slew`.  The consequence for tuning is blunt --
    above ~0.3 deg the loop is pinned to `slew` no matter what kp and ki are,
    so **--level-slew is the speed knob** and the gains only decide how
    quickly the target gets out in front of the feet.

    Deadbands differ on purpose: the integrator keeps the parent's HARD
    deadband, so `--level-kp 0` reproduces `lv.LevelingLoop` bit for bit and
    the A/B against the EKF script survives; P uses the soft one.

    Anti-windup: I is clamped to `clamp - |P_i|`, so the pair can never demand
    more than the legs' reach and I never winds into authority P is using.
    """

    def __init__(self, anchors_xy, kp=AHRS_KP, gain_per_s=AHRS_GAIN_PER_S,
                 slew_m_s=AHRS_SLEW_M_S, **kw):
        super().__init__(anchors_xy, gain_per_s=gain_per_s,
                         slew_m_s=slew_m_s, **kw)
        self.kp = float(kp)
        self.i_offsets = {leg: 0.0 for leg in LEGS}
        self.p_offsets = {leg: 0.0 for leg in LEGS}

    def reset(self):
        super().reset()
        for leg in LEGS:
            self.i_offsets[leg] = 0.0
            self.p_offsets[leg] = 0.0

    @property
    def tau_s(self):
        """Predicted closed-loop time constant, (1+kp)/ki seconds."""
        if self.gain <= 0.0:
            return float("inf")
        return (1.0 + self.kp) / self.gain

    def _hold(self, reason):
        """Freeze: hold the commanded offsets, fold P back into the integrator.

        Absorbing P is what makes the freeze a true HOLD.  Left in `p_offsets`
        it would be a lie -- the next active sweep recomputes P from a fresh
        error, and the difference would have to come from somewhere.  Folded
        in, the held output IS the integrator state; re-engagement resumes
        from it and the rate limit ramps the new P in.
        """
        self.frozen_reason = reason
        for leg in LEGS:
            self.i_offsets[leg] = self.offsets[leg]
            self.p_offsets[leg] = 0.0

    def _zero_mean(self, legs_on, theta, phi):
        """Per-foot geometric error with the common mode removed."""
        e = {leg: -self.anchors[leg][0] * theta + self.anchors[leg][1] * phi
             for leg in legs_on}
        m = sum(e.values()) / len(legs_on)
        return {leg: v - m for leg, v in e.items()}

    def update(self, now, roll, pitch, sp_roll, sp_pitch, planted, active,
               reason=""):
        if not active or roll is None or pitch is None \
                or not (math.isfinite(roll) and math.isfinite(pitch)):
            self._last = float(now)
            self._hold(reason or "inactive")
            return
        if self._last is None:
            self._last = float(now)
            self.frozen_reason = ""
            return
        dt = float(np.clip(now - self._last, 0.0, 0.1))
        self._last = float(now)
        self.frozen_reason = ""
        if dt <= 0.0 or (self.gain == 0.0 and self.kp == 0.0):
            return

        if self._lpf is None:
            self._lpf = [float(roll), float(pitch)]
        else:
            tau_f = 1.0 / (2.0 * math.pi * self.lpf_fc_hz)
            a = dt / (tau_f + dt)
            self._lpf[0] += a * (roll - self._lpf[0])
            self._lpf[1] += a * (pitch - self._lpf[1])
        e_roll = self._lpf[0] - float(sp_roll)
        e_pitch = self._lpf[1] - float(sp_pitch)

        legs_on = [leg for leg in LEGS if planted[leg]]
        if len(legs_on) < 3:
            self._hold("fewer than 3 planted feet")
            return

        # integrator: HARD deadband, exactly the parent's law
        phi_i = e_roll if abs(e_roll) >= self.deadband_rad else 0.0
        theta_i = e_pitch if abs(e_pitch) >= self.deadband_rad else 0.0
        # proportional: SOFT deadband, continuous across the edge
        phi_p = _shrink(e_roll, self.deadband_rad)
        theta_p = _shrink(e_pitch, self.deadband_rad)
        d_i = self._zero_mean(legs_on, theta_i, phi_i)
        d_p = self._zero_mean(legs_on, theta_p, phi_p)

        max_step = self.slew * dt
        for leg in legs_on:
            p = self.kp * d_p[leg]
            i_lim = max(0.0, self.clamp - abs(p))       # anti-windup
            step = float(np.clip(self.gain * d_i[leg] * dt,
                                 -max_step, max_step))
            self.i_offsets[leg] = float(
                np.clip(self.i_offsets[leg] + step, -i_lim, i_lim))
            self.p_offsets[leg] = p
            target = float(np.clip(self.i_offsets[leg] + p,
                                   -self.clamp, self.clamp))
            self.offsets[leg] += float(np.clip(target - self.offsets[leg],
                                               -max_step, max_step))


class AhrsSource:
    """The one attitude source of this experiment: raw AHRS + staleness.

    Returns (roll_rad, pitch_rad, active, reason).  `active` is False until
    the first packet and whenever the stream stalls; the leveling loop then
    FREEZES (holds its offsets) exactly like the EKF script's vetoes.
    """

    def __init__(self, imu, stale_s=AHRS_STALE_S):
        self.imu = imu
        self.stale_s = float(stale_s)

    def read(self):
        att = self.imu.sample()
        if att is None:
            return float("nan"), float("nan"), False, "no AHRS yet"
        if self.imu.is_stale(self.stale_s):
            return (math.radians(att.roll_deg), math.radians(att.pitch_deg),
                    False, f"AHRS stale >{self.stale_s*1e3:.0f}ms")
        return (math.radians(att.roll_deg), math.radians(att.pitch_deg),
                True, None)


class HoldStats:
    """Mean/std/p2p of the AHRS attitude over the settled HOLD4 samples."""

    def __init__(self):
        self.rows = []

    def add(self, roll, pitch):
        if math.isfinite(roll):
            self.rows.append((roll, pitch))

    def summary(self):
        if not self.rows:
            return ["[attitude] no settled HOLD4 samples"]
        a = np.array(self.rows)
        lines = [f"[attitude] {len(a)} settled HOLD4 samples (AHRS):"]
        for j, name in ((0, "roll "), (1, "pitch")):
            col = np.degrees(a[:, j])
            lines.append(f"  {name} mean={col.mean():+6.2f} "
                         f"std={col.std():5.2f} p2p={np.ptp(col):5.2f} deg")
        return lines


class LoopTiming:
    """How long the once-per-sweep control block takes vs one CAN slot.

    WHY THIS EXISTS
        Everything in the `joint_index == 0` branch runs BEFORE that slot's
        position command goes out, so a block that outlasts its slot delays
        the FIRST motor of the round-robin -- and the pacing reset that
        follows a big overrun (`deadline = now + slot`) skips the next few
        outright.  MOTOR_IDS starts [7, 8, 9, ...], which is exactly the set
        that keeps latching input-lost on this rig.

        The driver watchdog is 50 ms, so what matters is not one slow sweep
        but how many CONSECUTIVE sweeps a motor goes unaddressed.  At
        250 Hz/motor that budget is 12.5 sweeps; at 150 Hz only 7.5 -- which
        is why lowering the rate made the latching worse, not better.
    """

    def __init__(self, slot):
        self.slot = float(slot)
        self.n = 0
        self.over = 0
        self.worst = 0.0
        self.total = 0.0
        self.resets = 0

    def add(self, dt):
        self.n += 1
        self.total += dt
        if dt > self.worst:
            self.worst = dt
        if dt > self.slot:
            self.over += 1

    def summary(self):
        if not self.n:
            return ["[timing] no control-block samples"]
        return [
            f"[timing] control block over {self.n} sweeps: "
            f"mean {self.total / self.n * 1e6:.0f} us, "
            f"worst {self.worst * 1e6:.0f} us, slot {self.slot * 1e6:.0f} us",
            f"[timing] outlasted its slot {self.over} times "
            f"({100.0 * self.over / self.n:.2f}%); pacing deadline reset "
            f"{self.resets} times"
            + ("  <-- delays MOTOR_IDS[0..2] = the latching set"
               if self.resets or self.over else "  (block fits: look elsewhere)"),
        ]




def gap_hist_report(gap_over, gap_n, watchdog_s):
    """How OFTEN each motor's command gap crossed each threshold.

    WHY THIS EXISTS
        `max_gap` alone hid the fault for a day: it read 13-15 ms and the
        watchdog was assumed to be 50 ms everywhere, so "the host is on
        time" looked proven.  It is not -- CAN 1/7/8/9 are configured with a
        10 ms input-signal timeout, and 13.7 ms clears that by 37%.  What
        predicts a latch is not the maximum but the COUNT of excursions past
        each motor's own threshold, so bucket them and let the number speak.
    """
    lines = [f"[gaps] command-gap excursions per motor "
             f"(this motor's watchdog: {watchdog_s * 1e3:.0f} ms):",
             "   CAN " + "".join(f"  >{b*1e3:.0f}ms" for b in GAP_BUCKETS)
             + "     sends"]
    for mid in MOTOR_IDS:
        lines.append(f"   {mid:3d} "
                     + "".join(f"{gap_over[mid][k]:8d}"
                               for k in range(len(GAP_BUCKETS)))
                     + f"{gap_n[mid]:10d}")
    return lines


def gap_report(mb, watchdog_s=0.050):
    """Worst gap between consecutive commands to each motor.

    THIS is the number the driver's input-timeout actually watches: a latch
    means that motor went `watchdog_s` unaddressed.  The bus maintains it on
    every send, so unlike LoopTiming it also catches stalls in the USB write
    and the pacing -- not just the bracketed control block.  If the control
    block were the whole story, `blk worst` and `gap worst` would agree; a
    gap far larger than the worst block means the stall is somewhere this
    script does not bracket.
    """
    rows = sorted(((mid, mb.rec(mid).max_gap, mb.rec(mid).missed)
                   for mid in MOTOR_IDS), key=lambda r: -r[1])
    lines = [f"[gap] worst command gap per motor "
             f"(driver watchdog {watchdog_s * 1e3:.0f} ms):"]
    for mid, gap, missed in rows:
        lines.append(f"   CAN {mid:2d}  max_gap {gap * 1e3:7.1f} ms  "
                     f"missed {missed}"
                     + ("   ** exceeded the watchdog"
                        if gap >= watchdog_s else ""))
    return lines


def blackbox_dump(bb_age, bb_t, bb_count, now, latched):
    """Per-motor worst reply silence in the ring buffer, printed at first latch.

    THE QUESTION THIS ANSWERS
        At the moment a driver reports input-lost, had it ALSO stopped
        replying beforehand?  The 0x80 flag arrives in a status reply, so at
        detection the motor is talking again -- the informative part is the
        silence HISTORY leading up to it:

        * latched motors show a contiguous silent window (hundreds of ms)
          shortly before detection -> the node went dark in BOTH directions
          -> it was off the bus (logic reset / brown-out / bus-off), which no
          host-side counter can see.
        * latched motors kept replying at the normal ~4 ms cadence right
          through the event -> they heard nothing yet said plenty -> RX-only
          deafness (common-mode / signal margin), a very different fault.

        The ring holds the last ~10 s of per-motor reply age, sampled once
        per control block (250 Hz), so the window and its timing relative to
        detection are both visible.
    """
    n = min(bb_count, bb_age.shape[0])
    if n == 0:
        return ["[blackbox] no samples before the first latch"]
    ages = bb_age[:n]
    ts = bb_t[:n]
    lines = [f"[blackbox] first latch: worst reply-silence per motor over "
             f"the prior {now - float(np.min(ts)):.1f} s "
             f"(normal cadence ~4 ms; * = latched now):"]
    order = np.argsort(-np.nanmax(ages, axis=0))
    for i in order:
        col = ages[:, i]
        if not np.any(np.isfinite(col)):
            lines.append(f"   CAN {MOTOR_IDS[i]:2d}  no replies at all")
            continue
        k = int(np.nanargmax(col))
        lines.append(
            f"   CAN {MOTOR_IDS[i]:2d}  worst {float(col[k]) * 1e3:7.1f} ms"
            f"  ending {now - float(ts[k]):5.1f} s before detection"
            + ("   *" if MOTOR_IDS[i] in latched else ""))
    return lines


def load_report(tmax, iqmax, iqsum, iqn, latched_ever):
    """Peak driver temperature and peak |iq| per motor, next to who latched.

    WHY THIS EXISTS
        The soak (zero torque, no load) never latches; HOLD4 latches every
        time, ~1-2 min in.  "Always, delayed, only under load" is the
        signature of something that ACCUMULATES with current, and the two
        accumulating quantities the drivers already report are temperature
        (0x9A, 5 Hz) and q-axis current (every position reply, 250 Hz).

        Board temperature alone stayed under the 60 C notice threshold in
        every run, so if the trigger is thermal it is the winding or an I2t
        limit inside the driver rather than the sensor the status line reads.
        Peak iq is what distinguishes those: if the latching set is also the
        current-carrying set, the load path is the lead; if iq is flat across
        all twelve, load is a coincidence and the cause is elsewhere.
    """
    lines = ["[load] peak temp / peak |iq| / MEAN |iq| per motor "
             "(stall protection integrates the SUSTAINED current, so the "
             "mean is the column that matters; * = latched this run):"]
    mean = {mid: (iqsum[mid] / iqn[mid] if iqn[mid] else None)
            for mid in MOTOR_IDS}
    rows = sorted(MOTOR_IDS, key=lambda m: -(mean[m] or 0))
    for mid in rows:
        m = mean[mid]
        lines.append(f"   CAN {mid:2d}  peak "
                     f"{tmax[mid] if tmax[mid] is not None else '--':>4} C"
                     f"  peak |iq| "
                     f"{iqmax[mid] if iqmax[mid] is not None else '--':>6}"
                     f"  mean |iq| {f'{m:6.0f}' if m is not None else '    --'}"
                     + ("   *" if mid in latched_ever else ""))
    return lines


def volt_report(vmin):
    """Lowest supply voltage each driver reported (0x9A telemetry, 5 Hz).

    An input-lost latch is what a driver reports after ANY gap in its command
    stream -- including one it caused itself by browning out and rebooting.
    That reading is the difference between "the host stopped talking" and
    "the motor stopped listening", and the two need opposite fixes.  A sag on
    exactly the latching set (and not the others) points at their shared
    power branch, not at the CAN bus or the loop.
    """
    rows = sorted(((mid, v) for mid, v in vmin.items()
                   if v is not None and math.isfinite(v)), key=lambda r: r[1])
    if not rows:
        return ["[volt] no supply telemetry captured"]
    lo = rows[0][1]
    return ["[volt] lowest supply seen per driver:"] + [
        f"   CAN {mid:2d}  min {v:5.2f} V"
        + ("   ** lowest on the bus" if v <= lo + 0.05 else "")
        for mid, v in rows]


# ===========================================================================
# Hardware run
# ===========================================================================

def auto_clamp(tables, stand_height, margin_m=0.002):
    """Leveling authority = the LEGS' reach, not an arbitrary number.

    Retraction (foot z toward the crouch) has ~150 mm of table above the
    stand, so the binding limit is EXTENSION: how far below the stand the
    tables actually reach before the leg runs out (z ~ -221 mm on this
    robot).  The clamp is that headroom minus a small margin, so a saturated
    offset means "the leg is physically out", never "a config said stop".
    """
    ext = min(-tables[leg].z_min - stand_height for leg in LEGS)
    return max(0.0, ext - margin_m)


def leveling_banner(args, xy, stand_height):
    """The tuning the operator is about to run, in the units they will watch.

    There are TWO speed limits and the smaller one is what actually shows up:

        gains -> tau = (1+kp)/ki, the exponential the linear loop would follow
        slew  -> a flat ceiling on foot speed, hence on deg/s of recovery

    Above the crossover the slew wins and the gains are irrelevant, so both are
    printed, converted into the deg and deg/s the status line reports.  Gate
    [5] renders this for every tuning corner (P-only, I-only, open loop) --
    a banner that lies about the tuning is worse than no banner.
    """
    lines = [f"  leveling: PI, kp {args.level_kp:.2f} (immediate) + ki "
             f"{args.level_gain:.2f}/s (residual), clamp "
             f"+/-{args.level_clamp*1e3:.0f} mm/foot (= the legs' reach); "
             f"height OPEN LOOP (stand {stand_height*1e3:.0f} mm)"]
    if args.level_gain <= 0.0 and args.level_kp <= 0.0:
        return lines
    tau = ((1.0 + args.level_kp) / args.level_gain
           if args.level_gain > 0.0 else float("inf"))
    roll_dps = math.degrees(args.level_slew / abs(xy["FL"][1]))
    pitch_dps = math.degrees(args.level_slew / abs(xy["FL"][0]))
    lines.append(
        "  response: tau = (1+kp)/ki = "
        + (f"{tau:.1f}s" if math.isfinite(tau)
           else "inf -- P ONLY, the residual never reaches the deadband")
        + f"; slew {args.level_slew*1e3:.0f} mm/s "
          f"= {pitch_dps:.1f} deg/s pitch, {roll_dps:.1f} deg/s roll "
          "<- the real cap")
    if args.level_kp > 0.0:
        lines.append(
            f"    kp alone parks at err/(1+kp) = "
            f"{100.0/(1.0+args.level_kp):.0f}% of any disturbance"
            + ("; nothing clears it -- ki is 0" if args.level_gain <= 0.0 else
               f"; ki clears the rest.  ki is {args.level_gain:.2f} BECAUSE kp "
               f"divides it: at the EKF script's 0.25 the tail would be "
               f"{(1.0+args.level_kp)/0.25:.0f}s, worse than no P at all"))
    db_mm = math.radians(args.level_deadband) * abs(xy["FL"][1]) * 1e3
    lines.append(f"    deadband {args.level_deadband:.2f} deg = {db_mm:.1f} mm "
                 "of foot travel -- the residual it settles to, not a knob")
    return lines


def run(port, stand_height, crouch_max_speed_dps, args):
    base.validate_hardware_config()
    print("[init] building per-leg height tables (IK, once) ...", flush=True)
    t0 = time.perf_counter()
    # ask for far more span than the legs have -- the table march stops at
    # the physical reach / soft limits, and THAT becomes the clamp
    tables = s2.build_tables(stand_height,
                             clamp_m=args.level_clamp or 0.080)
    print(f"[init] tables built in {time.perf_counter()-t0:.2f}s; "
          + "  ".join(f"{leg}[{tables[leg].z_min*1e3:.0f},"
                      f"{tables[leg].z_max*1e3:.0f}]mm" for leg in LEGS))
    if args.level_clamp is None:
        args.level_clamp = auto_clamp(tables, stand_height)
        print(f"[init] leveling clamp = physical reach: "
              f"+/-{args.level_clamp*1e3:.0f} mm/foot")

    unwrap = [base.CalibratedEncoderUnwrap() for _ in MOTOR_IDS]
    key = base.KeyPoller()
    gate = base.SafetyGate(tau_cap=base.STAGED_TAU_MAX)
    lvl = PiLevelingLoop(anchors_xy={leg: tables[leg].xy for leg in LEGS},
                         kp=args.level_kp,
                         gain_per_s=args.level_gain, clamp_m=args.level_clamp,
                         slew_m_s=args.level_slew,
                         deadband_deg=args.level_deadband)
    sp_roll = math.radians(args.setpoint_roll)
    sp_pitch = math.radians(args.setpoint_pitch)
    tilt_stop_rad = math.radians(args.tilt_stop)
    planted_d = {leg: True for leg in LEGS}
    stats = HoldStats()

    print("=" * 78)
    print("DOG5 AHRS-ONLY LEVELING STAND  (experiment A: no EKF anywhere)")
    print(f"  feedback = raw DETA10 AHRS (0x41); setpoint "
          f"({args.setpoint_roll:+.2f},{args.setpoint_pitch:+.2f}) deg; "
          f"stale-freeze {AHRS_STALE_S*1e3:.0f} ms")
    _xy = {leg: tables[leg].xy for leg in LEGS}
    _track = abs(_xy["FL"][1] - _xy["FR"][1])
    _wb = abs(_xy["FL"][0] - _xy["RL"][0])
    for line in leveling_banner(args, _xy, stand_height):
        print(line)
    print(f"  authority: pitch +/-"
          f"{math.degrees(math.atan2(2*args.level_clamp, _wb)):.1f} deg, "
          f"roll +/-"
          f"{math.degrees(math.atan2(2*args.level_clamp, _track)):.1f} deg "
          "-- errors beyond that SATURATE (status line shows SAT)")
    print(f"  tilt-stop {args.tilt_stop:.0f} deg.  CAN {args.control_hz:.0f} "
          f"Hz/motor ({1e3/args.control_hz:.0f} ms between commands, 50 ms "
          "driver watchdog)")
    print(f"  fault poll {args.fault_status_hz:.0f} Hz/motor -> latch "
          f"detection (limp window) <= {1e3/args.fault_status_hz:.0f} ms; "
          "[err16] prints raw status bytes at every recovery")
    print("  NO torque commanded; drivers hold position. Contacts irrelevant: "
          "no estimator.")
    print("  Keys: ENTER stand, P park, X stop.")
    if args.level_gain == 0.0:
        print("  !! gain 0 -- OPEN LOOP A/B: watch rp:imu, nothing will move")
    print("=" * 78)

    stop_reason = None
    timing = None
    gap_lines = []
    gap_last = {mid: None for mid in MOTOR_IDS}
    gap_over = {mid: [0] * len(GAP_BUCKETS) for mid in MOTOR_IDS}
    gap_n = {mid: 0 for mid in MOTOR_IDS}
    vmin = {mid: None for mid in MOTOR_IDS}
    tmax = {mid: None for mid in MOTOR_IDS}
    iqmax = {mid: None for mid in MOTOR_IDS}
    iqsum = {mid: 0.0 for mid in MOTOR_IDS}
    iqn = {mid: 0 for mid in MOTOR_IDS}
    latched_ever = set()
    # reply-silence blackbox: ~10 s of per-motor reply age at the control
    # block's 250 Hz, dumped once at the first input-lost latch
    bb_age = np.full((2500, N_JOINTS), np.nan)
    bb_t = np.zeros(2500)
    bb_i = 0
    bb_count = 0
    bb_dumped = False
    bb_lines = []          # kept so the exit summary re-prints the dump:
    #                        13 lines inline among the [hw] stream are easy
    #                        to scroll past, and the forensics must survive
    bb_path = "/tmp/ahrs_level_blackbox.npz"
    imu = ImuDog(port)
    try:
        with motorbus.MotorBus(MOTOR_IDS, dirs=base.MOTOR_DIRECTIONS) as mb:
            imu.start()
            if not mb.arm(rate_hz=args.control_hz):
                raise RuntimeError("arm failed (bus / power / terminators)")
            if not imu.wait_for_data(timeout=3.0):
                raise RuntimeError("no AHRS (0x41) packets from the DETA10")
            ahrs = AhrsSource(imu, stale_s=args.stale)

            start_q = base._zero_torque_preflight(mb, key, unwrap)
            if start_q is None:
                print("[abort] preflight not confirmed")
                return

            if args.freeze_gc:
                # A generational sweep is the classic source of a lone
                # multi-millisecond stall in an otherwise 250 us loop: it
                # fires on allocation count, so it lands at unpredictable
                # sweeps and drags every motor behind it.  freeze() moves
                # everything allocated so far (tables, numpy buffers) out of
                # the generations entirely, then disable() stops the rest.
                import gc
                gc.collect()
                gc.freeze()
                gc.disable()
                print("[gc] frozen and disabled for the control loop")

            now = time.perf_counter()
            gate.start(now, start_q)
            seq = lv.LevelStandSequence(tables, stand_height)
            miss_monitor = base.CanMissMonitor(mb)
            slot = mb.slot(args.control_hz)
            timing = LoopTiming(slot)
            # --poll-rotate: permute WHICH slot serves WHICH motor, changing
            # nothing else.  The latching set {7,8,9,1} sits at send positions
            # {0,1,2,9}; if the latch is caused by the polling structure (the
            # control block runs inside position 0's slot, ahead of its send)
            # it must FOLLOW the positions to whichever motors now hold them.
            # If it stays with CAN 7,8,9,1 wherever they are sent, the cause
            # is the physical units / bus topology, not the schedule.
            send_order = [(i + args.poll_rotate) % N_JOINTS
                          for i in range(N_JOINTS)]
            if args.poll_rotate:
                print(f"[poll] send order rotated by {args.poll_rotate}: "
                      + " ".join(str(MOTOR_IDS[j]) for j in send_order))
                print("[poll] positions {0,1,2,9} now serve CAN "
                      + "/".join(str(MOTOR_IDS[send_order[p]])
                                 for p in (0, 1, 2, 9))
                      + "  (was 7/8/9/1: if the latch follows the schedule, "
                        "THESE latch next)")
            deadline = time.perf_counter() + slot
            # Fault polling at 20 Hz/motor (base default is 5): the damage in
            # a latch episode is the LIMP WINDOW between the driver cutting
            # output and the recovery ladder restoring it.  Detection is
            # bounded by this poll period, so 5 -> 20 Hz shrinks the window
            # from <=0.2 s to <=0.05 s -- the FL_abd drift that reached the
            # estop margin (13 deg in ~0.4 s) scales down with it.  Cost: a
            # status request replaces that motor's position command in 8% of
            # its slots (20 of 250), one 8 ms command gap each -- nothing
            # against the 50 ms watchdog.
            status_period = 1.0 / args.fault_status_hz
            next_fault_status = np.array(
                [status_period + i * status_period / N_JOINTS
                 for i in range(N_JOINTS)])
            last_recover = {mid: -base.RECOVER_PERIOD_S for mid in MOTOR_IDS}
            start = now
            cmd_deg = recorded.POSITION_TARGET_DEG.copy()
            index = 0
            last_print = 0.0
            tx_fail = 0
            quiet_t0 = None
            last_reason = None

            while True:
                mb.poll()
                joint_index = index % N_JOINTS
                if joint_index == 0:
                    now = time.perf_counter()
                    q, qd = base._joint_state(mb, unwrap)

                    for _i, _mid in enumerate(MOTOR_IDS):
                        _lrt = mb.rec(_mid).last_reply_t
                        bb_age[bb_i, _i] = (now - _lrt
                                            if _lrt is not None else np.nan)
                    bb_t[bb_i] = now
                    bb_i = (bb_i + 1) % bb_age.shape[0]
                    bb_count += 1

                    pressed = key.get()
                    if pressed in ("x", "X"):
                        stop_reason = "operator X"
                        break

                    # ---- the ONE feedback path: AHRS -> leveling ----
                    # engage only once HOLD4 has SETTLED: the ramp's last
                    # transient is not an attitude error to integrate.  The
                    # STAND ramp itself always runs with zero offsets.
                    roll_a, pitch_a, active, reason = ahrs.read()
                    settled = (seq.loop_engaged and seq.stage_t0 is not None
                               and now - seq.stage_t0 >= args.engage_delay)
                    engaged = active and settled
                    if seq.loop_engaged and not settled:
                        reason = reason or (f"settling "
                                            f"{args.engage_delay - (now - seq.stage_t0):.1f}s")
                    lvl.update(now, roll_a, pitch_a, sp_roll, sp_pitch,
                               planted_d, engaged,
                               reason if not (active and settled)
                               else "not holding")
                    if seq.loop_engaged and reason and reason != last_reason:
                        print(f"[level] FROZEN: {reason}")
                    last_reason = reason

                    # ---- tilt run-stop (absolute, on the same AHRS) ----
                    if math.isfinite(roll_a) and \
                            max(abs(roll_a), abs(pitch_a)) > tilt_stop_rad:
                        stop_reason = (f"tilt-stop: |attitude| "
                                       f"{math.degrees(max(abs(roll_a), abs(pitch_a))):.1f}deg")
                        break

                    seq.extra_z = dict(lvl.offsets)
                    q_cmd, _contacts, event = seq.update(
                        now, q, qd,
                        enter_pressed=pressed in ("\r", "\n"),
                        park_pressed=pressed in ("p", "P"),
                        offset=0.0)                  # height stays OPEN LOOP
                    cmd_deg = np.rad2deg(q_cmd)
                    if seq.fault:
                        stop_reason = seq.fault
                        break

                    if event == "crouch_settled":
                        quiet_t0 = now
                        print("[stage] CROUCH settled. ENTER to STAND "
                              "(no EKF, nothing to initialise).")
                    elif event == "stand_started":
                        quiet_t0 = None
                        lvl.reset()
                        seq.clear_extra()
                        print("[stage] STAND: open-loop ramp crouch -> stand.")
                    elif event == "stand_complete":
                        quiet_t0 = now
                        print(f"[stage] HOLD4 (stand #{seq.n_stands}): "
                              f"settling {args.engage_delay:.1f}s, then "
                              "leveling engages on raw AHRS, setpoint "
                              f"({args.setpoint_roll:+.2f},"
                              f"{args.setpoint_pitch:+.2f}) deg. P parks.")
                    elif event == "park_started":
                        quiet_t0 = None
                        print("[stage] PARK: unwinding leveling "
                              + "/".join(f"{lvl.offsets[l]*1e3:+.1f}"
                                         for l in LEGS)
                              + " mm on the way down.")
                    elif event == "park_complete":
                        quiet_t0 = now
                        lvl.reset()
                        seq.clear_extra()
                        print("[stage] PARKED. ENTER stands again, X stops.")

                    # ---- safety ----
                    for _mid in MOTOR_IDS:
                        _r = mb.rec(_mid)
                        if _r.voltage is not None and (
                                vmin[_mid] is None or _r.voltage < vmin[_mid]):
                            vmin[_mid] = float(_r.voltage)
                        if _r.temp is not None and (
                                tmax[_mid] is None or _r.temp > tmax[_mid]):
                            tmax[_mid] = int(_r.temp)
                        if _r.iq is not None:
                            if (iqmax[_mid] is None
                                    or abs(_r.iq) > iqmax[_mid]):
                                iqmax[_mid] = abs(int(_r.iq))
                            iqsum[_mid] += abs(int(_r.iq))
                            iqn[_mid] += 1
                    temps = base._temperatures(mb)
                    misses = miss_monitor.update(mb)
                    errors = mb.errors()
                    if stop_reason is None:
                        stop_reason = gate.estop_reason(
                            q, qd, temps, misses, errors, now,
                            enforce_position_limits=True)
                    if stop_reason is None:
                        stop_reason = gate.overspeed_reason(qd, q, now)
                    if stop_reason:
                        break
                    latched_err = [mid for mid, e in errors.items()
                                   if e and (e & 0x80)]
                    latched_ever.update(latched_err)
                    if latched_err and not bb_dumped:
                        bb_dumped = True
                        bb_lines = blackbox_dump(bb_age, bb_t, bb_count,
                                                 now, set(latched_err))
                        # Raw status1 bytes for ALL twelve at the moment of
                        # first latch (LK MG5010E-i10; no interpretation).
                        # The healthy motors are the baseline: a data6 value
                        # that appears ONLY on the latched ones is a
                        # companion flag to the 0x80, not normal state.
                        bb_lines.append("[err16] status1 bytes at first "
                                        "latch (d6/d7, * = latched):")
                        for _m in MOTOR_IDS:
                            bb_lines.append(
                                f"   CAN {_m:2d}  "
                                f"d6=0x{(mb.rec(_m).state or 0):02X}  "
                                f"d7=0x{(mb.rec(_m).error or 0):02X}"
                                + ("   *" if _m in latched_err else ""))
                        try:
                            np.savez(bb_path, bb_age=bb_age, bb_t=bb_t,
                                     bb_count=bb_count, t_detect=now,
                                     latched=sorted(latched_err),
                                     motor_ids=list(MOTOR_IDS))
                            bb_lines.append(f"[blackbox] raw ring saved to "
                                            f"{bb_path}")
                        except OSError as e:
                            bb_lines.append(f"[blackbox] ring save failed: "
                                            f"{e}")
                        for line in bb_lines:
                            print(line, flush=True)
                    recover = [mid for mid in latched_err
                               if (now - start) - last_recover[mid]
                               >= base.RECOVER_PERIOD_S]
                    if recover:
                        # raw status1 bytes BEFORE recovery wipes rec.error:
                        # data6 alongside the input-lost data7, every episode
                        print("[err16] " + "  ".join(
                            f"CAN {m} d6=0x{(mb.rec(m).state or 0):02X}"
                            f"/d7=0x{(mb.rec(m).error or 0):02X}"
                            for m in sorted(recover)), flush=True)
                        base._recover_input_lost(mb, recover, now - start,
                                                 last_recover,
                                                 next_fault_status)
                        # _recover_input_lost reschedules the next status
                        # poll with base's 5 Hz period; re-clamp to ours so
                        # the post-recovery verdict also arrives at 20 Hz
                        for m in recover:
                            next_fault_status[MOTOR_IDS.index(m)] = (
                                now - start + status_period)

                    # ---- observe ----
                    if (seq.stage == "HOLD4" and quiet_t0 is not None
                            and now - quiet_t0 >= SETTLE_S):
                        stats.add(roll_a, pitch_a)

                    if now - last_print >= 1.0 / base.STATUS_HZ:
                        last_print = now

                        def _deg(a):
                            return (f"{math.degrees(a):+6.2f}"
                                    if math.isfinite(a) else "    --")

                        r_fk, p_fk = lv.fk_attitude(q)
                        flag = ("LVL" if engaged else
                                ("HOLD" if seq.loop_engaged else "----"))
                        err_r = (roll_a - sp_roll
                                 if math.isfinite(roll_a) else float("nan"))
                        err_p = (pitch_a - sp_pitch
                                 if math.isfinite(pitch_a) else float("nan"))
                        lvl_s = ("  lvl=" + "/".join(
                            f"{lvl.offsets[l]*1e3:+4.1f}" for l in LEGS)
                            + (" SAT" if lvl.saturated else "")
                            if (args.level_gain != 0.0
                                or args.level_kp != 0.0) else "")
                        t_max = int(np.max(temps))
                        hot = f"  {t_max}C" if t_max >= TEMP_NOTICE_C else ""
                        warn = ""
                        if latched_err:
                            warn += f" latched={len(latched_err)}"
                        if tx_fail:
                            warn += f" txfail={tx_fail}"
                        if timing.over or timing.resets:
                            warn += (f" blk={timing.worst*1e6:.0f}us"
                                     f"/{timing.over}ovr/{timing.resets}rst")
                        print(f"[hw] {seq.stage:10s}[{flag:4s}] "
                              f"rp:imu={_deg(roll_a)}/{_deg(pitch_a)} "
                              f"fk={_deg(r_fk)}/{_deg(p_fk)} "
                              f"err={_deg(err_r)}/{_deg(err_p)}deg"
                              f"{lvl_s}{hot}{warn}", flush=True)

                    # the block is over: everything above ran INSIDE slot 0,
                    # ahead of CAN 7's position command
                    timing.add(time.perf_counter() - now)

                j = send_order[joint_index]
                mid = MOTOR_IDS[j]
                _tsend = time.perf_counter()
                if gap_last[mid] is not None:
                    _g = _tsend - gap_last[mid]
                    gap_n[mid] += 1
                    for _k, _b in enumerate(GAP_BUCKETS):
                        if _g > _b:
                            gap_over[mid][_k] += 1
                gap_last[mid] = _tsend
                if (time.perf_counter() - start) >= next_fault_status[j]:
                    mb.status1_req(mid)
                    next_fault_status[j] += status_period
                # ALWAYS speed-capped (0xA4).  A 2026-08-13 A/B tried 0xA3
                # (no cap) during HOLD4 to vary the driver's command path:
                # the legs sag under load, so a standing position error is
                # permanent, and without the cap every motor chases it at
                # full speed -- CAN 5 browned out to 6.4 V and tripped its
                # low-voltage protection before HOLD4 even settled.  The cap
                # is load-bearing safety here, not a tuning knob.
                elif not mb.position(mid, float(cmd_deg[j]),
                                     crouch_max_speed_dps):
                    tx_fail += 1
                index += 1
                overrun = mb.pace(deadline)
                deadline += slot
                if overrun and overrun > 2.0 * slot:
                    timing.resets += 1
                    deadline = time.perf_counter() + slot

            gap_lines = gap_report(mb)
            base._soft_stop(mb)
    except KeyboardInterrupt:
        stop_reason = stop_reason or "KeyboardInterrupt"
    finally:
        try:
            imu.stop()
        except Exception:
            pass
        try:
            key.close()
        except Exception:
            pass
    for line in stats.summary():
        print(line)
    if timing is not None:
        for line in timing.summary():
            print(line)
    for line in gap_lines:
        print(line)
    for line in gap_hist_report(gap_over, gap_n, args.watchdog_ms * 1e-3):
        print(line)
    for line in volt_report(vmin):
        print(line)
    for line in load_report(tmax, iqmax, iqsum, iqn, latched_ever):
        print(line)
    if bb_lines:
        for line in bb_lines:            # re-print: the inline copy scrolls
            print(line)
    else:
        print("[blackbox] no input-lost latch this run -- nothing recorded")
    print("[level] final offsets "
          + "  ".join(f"{leg}{lvl.offsets[leg]*1e3:+.1f}" for leg in LEGS)
          + " mm" + (" (SATURATED)" if lvl.saturated else ""))
    # The split is the tuning evidence: a large P at the end means the loop
    # stopped on a LIVE error (P only exists while the error does), so the
    # integrator never finished -- raise ki.  A near-zero P with a large I is
    # the healthy end state: the tilt is absorbed and held by the integrator.
    print("[level]   split P " + "/".join(f"{lvl.p_offsets[l]*1e3:+.1f}"
                                          for l in LEGS)
          + "  I " + "/".join(f"{lvl.i_offsets[l]*1e3:+.1f}" for l in LEGS)
          + f" mm  (kp {lvl.kp:.2f}, ki {lvl.gain:.2f}/s, tau {lvl.tau_s:.1f}s)")
    print(f"[stop] {stop_reason}")
# ===========================================================================
# offline self-test -> test_stand_ahrs_level.py
# ===========================================================================
# The gates moved out of this file.  They were 283 lines of simulation that no
# hardware run ever executes, and several of them shared a plane plant, a fake
# EKF and a PASS/FAIL harness with the other runners -- all of which now live
# in selftest_common.py.  `--self-test` still runs exactly those gates.
#
#     $V self-test/test_stand_ahrs_level.py     # the same thing, run directly


def self_test(stand_height):
    """Delegate to the suite in self-test/.  Lazy: a hardware run never loads it."""
    _sd = s2.s1.selftest_dir()
    if _sd not in sys.path:
        sys.path.insert(0, _sd)
    from test_stand_ahrs_level import self_test as gates
    return gates(stand_height)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--stand-height", type=float, default=STAND_HEIGHT_DEFAULT)
    ap.add_argument("--crouch-max-speed-dps", type=float,
                    default=recorded.DEFAULT_CROUCH_MAX_MOTOR_DPS)
    ap.add_argument("--level-kp", type=float, default=AHRS_KP,
                    help="leveling PROPORTIONAL gain (dimensionless): the "
                         "fraction of the geometric correction commanded at "
                         "once.  1.0 = the whole of it.  P alone parks at "
                         "err/(1+kp), so it never replaces the integrator -- "
                         "it only stops the loop crawling at the start.  "
                         "0 = the EKF script's pure-integrator law")
    ap.add_argument("--level-gain", type=float, default=AHRS_GAIN_PER_S,
                    help="leveling INTEGRAL gain ki (1/s).  The closed-loop "
                         "pole is ki/(1+kp), so this must rise WITH --level-kp "
                         "or the tail gets slower; 1.0 with kp=1 matches the "
                         "EKF script's 0.25 with kp=0 and then beats it.  "
                         "0 with --level-kp 0 = open-loop A/B")
    ap.add_argument("--level-clamp", type=float, default=None,
                    help="max |offset| per foot (m); default: the legs' full "
                         "physical reach below the stand (no artificial cap)")
    ap.add_argument("--level-slew", type=float, default=AHRS_SLEW_M_S,
                    help="max foot speed the loop may command (m/s), applied "
                         "to the P+I TOTAL.  This is the real speed limit: "
                         "above ~0.3 deg of error the loop is pinned to it and "
                         "the gains do not matter, so raise THIS to recover "
                         "faster (10 mm/s = 1.7 deg/s pitch, 5.1 deg/s roll; "
                         "the STAND ramp already moves these feet at ~33)")
    ap.add_argument("--level-deadband", type=float,
                    default=lv.LEVEL_DEADBAND_DEG,
                    help="deg of attitude error the loop ignores; this is the "
                         "residual it settles to, not a speed knob")
    ap.add_argument("--setpoint-roll", type=float, default=0.0,
                    help="deg; AHRS zero is flat-calibrated, so 0 = level")
    ap.add_argument("--setpoint-pitch", type=float, default=0.0)
    ap.add_argument("--stale", type=float, default=AHRS_STALE_S,
                    help="AHRS age that freezes leveling (s)")
    ap.add_argument("--engage-delay", type=float, default=SETTLE_S,
                    help="seconds to hold still in HOLD4 before the loop "
                         "engages (the stand transient is not an error)")
    ap.add_argument("--poll-rotate", type=int, default=0,
                    help="rotate the per-slot send order by N positions "
                         "(0-11).  Decisive test for 'the latch is caused by "
                         "the polling': the latching set sits at send "
                         "positions {0,1,2,9}; rotating moves other motors "
                         "into those positions.  Latch follows the positions "
                         "-> schedule is the cause; stays with CAN 7/8/9/1 "
                         "-> the units/topology are")
    ap.add_argument("--freeze-gc", action="store_true",
                    help="gc.freeze()+disable() around the control loop; A/B "
                         "this against the [timing] worst-block figure to see "
                         "whether the multi-ms stalls are collections")
    ap.add_argument("--tilt-stop", type=float, default=TILT_STOP_DEG)
    ap.add_argument("--control-hz", type=float, default=CONTROL_HZ,
                    help="per-motor command rate (Hz); 50 ms driver watchdog")
    ap.add_argument("--watchdog-ms", type=float, default=10.0,
                    help="the drivers' input-signal timeout, for the margin "
                         "report.  CAN 1/7/8/9 on this rig are configured at "
                         "10 ms while the other eight are far longer -- which "
                         "is why only those four ever latch: a 13.7 ms "
                         "command gap clears 10 ms but not 50")
    ap.add_argument("--fault-status-hz", type=float, default=20.0,
                    help="per-motor error-flag poll rate (Hz); bounds the "
                         "limp window between a driver latching input-lost "
                         "and the recovery ladder (base default was 5)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if not 20.0 <= args.control_hz <= 300.0:
        ap.error("--control-hz outside [20, 300] (bus ceiling is 300)")
    if args.self_test:
        sys.exit(self_test(args.stand_height))
    run(args.port, args.stand_height, args.crouch_max_speed_dps, args)


if __name__ == "__main__":
    main()
