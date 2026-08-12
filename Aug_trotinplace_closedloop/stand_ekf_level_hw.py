#!/usr/bin/env python3
"""Stage 2b: a stand that holds its ATTITUDE, closed on the EKF.

WHAT IT DOES
    The EKF's roll/pitch drives per-foot differential z offsets, so the trunk
    holds the attitude it rests at instead of sagging off it.  The stage-2
    height loop runs underneath.  Position mode throughout: no software
    torque, the EKF only moves the targets the 0xA4 loops are given.
    All four feet stay planted -- the leg-lift experiment is a separate
    runner, lift_ekf_contact_hw.py.

HOW TO RUN
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    cd ~/Documents/can_motor_control/dog_stand_compliance_control/Aug_trotinplace_closedloop

    $V stand_ekf_level_hw.py --self-test        # offline gates, no hardware

    # 1. open-loop A/B -- reproduces stage 2 at this stand height
    sudo chrt -f 50 $V stand_ekf_level_hw.py --level-gain 0
    # 2. leveling live at half authority
    sudo chrt -f 50 $V stand_ekf_level_hw.py --level-clamp 0.006
    # 3. full run
    sudo chrt -f 50 $V stand_ekf_level_hw.py

    Keys: ENTER stand, P park, X stop.
    Support the robot at the zero-torque prompt, then hold it still through
    WAIT_CROUCH -- the EKF's gyro-bias init runs there and nowhere else.

WHEN FK IS ENOUGH -- AND WHEN IT IS NOT
    Commanding all four feet to the same z IS FK leveling, and it is exactly
    what --level-gain 0 leaves running.  On a flat level floor, with true
    encoder zeros and equal sag, it holds the trunk level and this whole
    script buys nothing.  That case is real; do not pay for the EKF there.

    FK CAN measure attitude, just not against gravity.  `fk_attitude` fits a
    plane through the measured feet and reads the trunk's tilt relative to
    it -- IMU-free, and printed as rp:fk= on the status line.  It is not
    trivially zero: the COMMANDED stand puts all four feet at one z, so any
    reading is the legs differing from the command, which under load is sag.

    What FK cannot know is how that foot plane sits relative to gravity.  So
    on a sloped floor FK still says "level" and means "level with the floor".
    That is the whole difference, and rp:d= (EKF minus FK) is the size of it:

        EKF - FK  =  floor slope  +  IMU mount tilt  +  encoder-zero error

    Three unknowns in one number, which is why the ABSOLUTE value proves
    little -- but the constants do not move while the robot stands, so a
    CHANGE in rp:d= is pure floor slope.  That is the measurement:

        shim a known height under one side, and the change in rp:d= must
        equal atan(shim / track).  Nothing else in the run has to be known.

    Self-tested in [7]: EKF-FK recovers a 2-3 deg floor slope to 0.001 deg
    regardless of what the trunk itself is doing.

HOW TO PROVE IT DOES ANYTHING
    A successful run LOOKS LIKE NOTHING HAPPENING, by construction: the
    setpoint is latched from the resting attitude, so the steady-state error
    is ~0 and the leveling offsets sit near 0.  On a flat floor with four feet
    down there is nothing to correct, and `lvl=` stays at zeros.  Worse, the
    loops only run in HOLD4 -- in CROUCH/WAIT_CROUCH/PARKED the flag reads
    ---- and every number on the line is just the robot sitting there.

    To see it work, change the FLOOR, not the robot:

      1. Flat floor.  ENTER to stand, wait for "[level] setpoint latched".
         The setpoint now means "trunk parallel to this flat floor" -- the IMU
         mount tilt is inside it and cancels, so no one has to measure it.
      2. Slide a ~10 mm shim under ONE foot (or two on one side) while it
         holds.  Watch `err=` spike, `lvl=` wind out to absorb it, `err=`
         come back inside the 0.2 deg deadband.  A spirit level on the trunk
         says it stayed put while the floor moved.
      3. The A/B that makes it a result: repeat with --level-gain 0.  Same
         shim, no leveling -- the trunk now tilts with the floor and stays
         tilted.  That difference is the whole claim of this stage.

    NOTE what is NOT being compared: FK has no opinion about attitude.  Legs
    measure height relative to the feet; only the IMU knows where gravity is,
    and `fk_floor_height` is *given* the EKF's C.  So "EKF vs FK" is a HEIGHT
    cross-check (the `diff=` column, and the point of lift_ekf_contact_hw.py).
    The attitude cross-check is EKF vs AHRS, and it is the --agree-veto.

WHAT TO WATCH
    [hw] HOLD4     [LVL     ] z= 172.3 fk= 171.8 d=+0.5  z_corr= +9.2mm
                              roll= +0.14 pitch= -1.81  err= +0.02/ -0.01deg
                              lvl=-3.1/+2.9/-2.8/+3.0

    z, fk   floor-to-TRUNK-BOTTOM (the IMU board) height, from the EKF and
            from FK -- what a ruler reads, and what --target-height means.
            The abd axis is 38 mm higher; the banner prints it once.
    d       z minus fk.  Past --xcheck the height loop freezes (FK is truth).
    z_corr  what the HEIGHT loop has wound in; converges on the sag.
    rp:     roll/pitch, degrees.  No yaw -- gravity does not constrain it.
      ekf     EKF attitude, referenced to GRAVITY.
      imu     the DETA10's OWN AHRS attitude, straight off the board, also
              referenced to gravity.  It is the leveling watchdog
              (--agree-veto freezes the loop when ekf and imu split), but
              note it is NOT independent evidence about the EKF: same
              accelerometer, same gyro, different filter.  Agreement checks
              the filter maths, not the sensor.
      fk      plane-through-the-feet attitude from encoders alone, referenced
              to the FLOOR.  No IMU anywhere in it.
      d       ekf minus fk = floor slope + mount tilt + zero errors.  Its
              CHANGE across a shim is pure floor slope (see WHEN FK IS
              ENOUGH); its absolute value mixes three unknowns.
    err     attitude minus the setpoint.  Default setpoint is (0,0) --
            ABSOLUTE level, valid because the IMU offset was calibrated with
            the robot lying flat.  --setpoint-latch restores the old
            hold-what-it-rests-at behaviour.  Hidden under --level-gain 0.
    lvl     the LEVELING loop's per-foot output, FL/FR/RL/RR mm.  Hidden
            under --level-gain 0 (it can only be zeros).  If it stays at
            zeros with the gain on, leveling has found nothing to do -- see
            HOW TO PROVE IT.
    flag    ---- idle | latching | LVL live | HOLD frozen (reason printed).
            "latching" now also waits for the height loop to settle, so the
            setpoint is the RESTED attitude and not a point on the sag ramp.
            Leveling holds whatever it latches -- latch a transient and the
            loop will faithfully keep the robot at it, which looks exactly
            like leveling making the attitude worse than --level-gain 0.
    Temperature appears only above 60 C; the estop handles 80 C.
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

import motorbus                                          # noqa: E402
import stand_dog5_hw as base                             # noqa: E402
import stand_dog5_recorded_hw as recorded                # noqa: E402
import dog5_kinematics                                   # noqa: E402
from ekf_runtime import EkfShared, ekf_worker, _rp       # noqa: E402
from imu_dog import DEFAULT_PORT                         # noqa: E402

MOTOR_IDS = base.MOTOR_IDS
N_JOINTS = base.N_JOINTS
CONTROL_HZ = base.CONTROL_HZ
LEGS = base.LEGS
Q_CROUCH = recorded.Q_RECORDED_CROUCH

FOOT_RADIUS_M = s1.FOOT_RADIUS_M
IMU_BELOW_TRUNK_ORIGIN_M = s1.IMU_BELOW_TRUNK_ORIGIN_M

T_STAND = s2.T_STAND
EKF_WORKER_HZ = s2.EKF_WORKER_HZ
QUIET_STAGES = s2.QUIET_STAGES
SETTLE_S = s2.SETTLE_S

# Stage-2 stood at the recorded default (0.20 m), but the legs' reach caps the
# commanded foot z at ~-221 mm -- at 0.20 that leaves ~21 mm of extension
# authority, less than height sag (~13 mm) + a leveling clamp (12 mm).  This
# script therefore defaults LOWER: 0.19 m keeps ~31 mm of extension authority
# and drops the CoM for the lift demo (the crawl track's reach-ceiling number).
STAND_HEIGHT_DEFAULT = 0.19

# ---- leveling loop tuning (constants = stand_hier's proven LevelingTrim) ---
LEVEL_GAIN_PER_S = 0.25      # integral gain on the geometric per-foot error
LEVEL_CLAMP_M = 0.012        # per-foot authority (~2.0 deg pitch, ~6 deg roll)
LEVEL_SLEW_M_S = 0.004       # caps the response to an EKF attitude step
LEVEL_DEADBAND_DEG = 0.2     # stage-1 noise floor is well under this
LEVEL_LPF_FC_HZ = 10.0
LATCH_S = 2.0                # setpoint = mean attitude over this window
HEIGHT_SETTLE_S = 1.0        # ... which may only start once the height loop
                             # has held its target this long (see
                             # HeightSettleGate: no latching mid-sag-ramp)
AGREE_VETO_DEG = 3.0         # |EKF-AHRS| beyond this freezes leveling
LEVEL_STALE_S = s2.HEIGHT_STALE_S

TILT_STOP_DEG = 12.0         # absolute attitude that soft-stops the run
TEMP_NOTICE_C = 60           # below this the status line stays quiet about
                             # temperature (base.TEMP_ESTOP=80 already stops)


# ===========================================================================
# Planted-aware FK height
# ===========================================================================

def fk_floor_height_planted(q, C=None, planted=None, ref="imu"):
    """`s1.fk_floor_height`, but averaged over the PLANTED feet only.

    The all-four average assumes four feet on one floor; a lifted foot would
    bias it by lift/4.  With `planted=None` (or all True) this is exactly the
    stage-1/2 function.
    """
    q = np.asarray(q, dtype=float)
    zs = []
    for i, leg in enumerate(LEGS):
        if planted is not None and not planted[i]:
            continue
        s = dog5_kinematics.foot_position(leg, q[3 * i:3 * i + 3])
        zs.append((C.T @ s)[2] if C is not None else s[2])
    if not zs:
        return float("nan")
    h_hip = -float(np.mean(zs)) + FOOT_RADIUS_M
    if ref == "hip":
        return h_hip
    cz = float(C[2, 2]) if C is not None else 1.0
    return h_hip - IMU_BELOW_TRUNK_ORIGIN_M * cz


def fk_attitude_from_feet(feet):
    """Trunk roll/pitch relative to the plane through `feet` (body frame).

    Feet on one flat floor satisfy `u . s_i = const`, where `u` is the floor
    normal expressed in BODY coordinates.  So a least-squares plane through
    the measured foot positions, `z = d + b*x + c*y`, hands back that normal
    as `u ~ (-b, -c, 1)` -- and `u` is exactly what `_rp` reads an attitude
    out of (it is `C @ e_z`, the up direction in body coordinates).  The
    result is therefore in the same convention as the EKF's roll/pitch and
    the two can be subtracted directly.

    Needs 3+ non-collinear planted feet; returns NaN otherwise.
    """
    feet = np.asarray(feet, dtype=float)
    if len(feet) < 3:
        return float("nan"), float("nan")
    A = np.column_stack([np.ones(len(feet)), feet[:, 0], feet[:, 1]])
    if np.linalg.matrix_rank(A, tol=1e-9) < 3:          # collinear anchors
        return float("nan"), float("nan")
    (_d, b, c), *_ = np.linalg.lstsq(A, feet[:, 2], rcond=None)
    u = np.array([-b, -c, 1.0])
    u /= np.linalg.norm(u)
    return (math.atan2(u[1], u[2]),
            math.atan2(-u[0], math.hypot(u[1], u[2])))


def fk_attitude(q, planted=None):
    """`fk_attitude_from_feet` on the MEASURED encoders' foot positions.

    WHAT IT IS: the trunk's attitude relative to the plane its feet stand on,
    computed from joint angles alone -- no IMU anywhere.  Comparable, number
    for number, with the EKF's roll/pitch.

    WHAT IT IS NOT: an attitude relative to gravity.  It cannot be: nothing
    in the legs knows which way is down.  It equals the true attitude only
    when the foot plane happens to be horizontal.

    So `EKF - FK` is the floor's slope relative to gravity, plus whatever
    constant errors the two estimators carry (IMU mount tilt, encoder zeros).
    On a fixed floor those constants do not move, which is what makes the
    shim test quantitative: tilt the floor by a known angle and the CHANGE in
    (EKF - FK) must equal it, with every unknown constant cancelling.

    Note this is computed from MEASURED angles, so it is not trivially zero:
    the commanded pose has all four feet at one z (a horizontal plane by
    construction), but the legs sag unequally under load, and that shows up
    here.
    """
    q = np.asarray(q, dtype=float)
    feet = [dog5_kinematics.foot_position(leg, q[3 * i:3 * i + 3])
            for i, leg in enumerate(LEGS)
            if planted is None or planted[i]]
    return fk_attitude_from_feet(feet)


def height_inputs(shared, q, z0_fk_imu, now_mono, planted,
                  xcheck_m=s2.HEIGHT_XCHECK_M):
    """Stage-2 `height_inputs` with the FK veto computed from planted feet.

    Heights are floor-to-TRUNK-BOTTOM (the IMU board), same as stage 2: that
    is the point `r` tracks and the only one a ruler reaches.  The abd axis is
    IMU_BELOW_TRUNK_ORIGIN_M higher and is where the leg tables work.
    """
    if shared is None or not shared.est_ready or shared.out is None:
        return float("nan"), float("nan"), False, "EKF not ready"
    if z0_fk_imu is None:
        return float("nan"), float("nan"), False, "no height origin"
    if now_mono - shared.tau_stamp > s2.HEIGHT_STALE_S:
        return float("nan"), float("nan"), False, "EKF stale"
    out = shared.out
    if not out.get("healthy", False):
        return float("nan"), float("nan"), False, "EKF unhealthy"
    C = out["C"]
    h_ekf = z0_fk_imu + float(out["r"][2])
    h_fk = fk_floor_height_planted(q, C, planted, ref="imu")
    if abs(h_ekf - h_fk) > xcheck_m:
        return h_ekf, h_fk, False, f"EKF-FK disagree {abs(h_ekf-h_fk)*1e3:.0f}mm"
    return h_ekf, h_fk, True, None


# ===========================================================================
# Leveling loop (LevelingTrim law + setpoint + planted-set awareness)
# ===========================================================================

class LevelingLoop:
    """EKF attitude error -> per-foot z offsets, zero-mean over planted feet.

    Law and constants are `stand_hier_hw.LevelingTrim`'s (proven on hardware
    in the inplace stand), with the error taken against a setpoint and the
    zero-mean/update restricted to the planted set:

        e_i = -x_i * (pitch - sp_pitch) + y_i * (roll - sp_roll)
        off_i += clip(gain * (e_i - mean_planted(e)) * dt, +/-slew*dt)

    Nose down beyond the setpoint (+pitch error) -> front feet more negative z
    -> front legs extend -> nose rises.  Right side down (+roll error) ->
    right legs extend.  Zero-mean over the planted set so leveling never
    fights the height loop; a lifted leg's offset is FROZEN (it has no floor
    to push on, and winding it would jump the foot on touchdown).

    Inactive updates freeze (hold, never unwind) -- same discipline as the
    stage-2 height loop.
    """

    def __init__(self, anchors_xy, gain_per_s=LEVEL_GAIN_PER_S,
                 clamp_m=LEVEL_CLAMP_M, slew_m_s=LEVEL_SLEW_M_S,
                 deadband_deg=LEVEL_DEADBAND_DEG, lpf_fc_hz=LEVEL_LPF_FC_HZ):
        self.anchors = {leg: np.asarray(anchors_xy[leg][:2], dtype=float)
                        for leg in LEGS}
        self.gain = float(gain_per_s)
        self.clamp = float(clamp_m)
        self.slew = float(slew_m_s)
        self.deadband_rad = math.radians(float(deadband_deg))
        self.lpf_fc_hz = float(lpf_fc_hz)
        self.offsets = {leg: 0.0 for leg in LEGS}
        self.frozen_reason = "init"
        self._lpf = None
        self._last = None

    def reset(self):
        for leg in LEGS:
            self.offsets[leg] = 0.0
        self._lpf = None
        self._last = None
        self.frozen_reason = "reset"

    @property
    def saturated(self):
        return any(abs(v) >= self.clamp - 1e-9 for v in self.offsets.values())

    def update(self, now, roll, pitch, sp_roll, sp_pitch, planted, active,
               reason=""):
        if not active or roll is None or pitch is None:
            self._last = float(now)
            self.frozen_reason = reason or "inactive"
            return
        if self._last is None:
            self._last = float(now)
            self.frozen_reason = ""
            return
        dt = float(np.clip(now - self._last, 0.0, 0.1))
        self._last = float(now)
        self.frozen_reason = ""
        if dt <= 0.0 or self.gain == 0.0:
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
        phi = e_roll if abs(e_roll) >= self.deadband_rad else 0.0
        theta = e_pitch if abs(e_pitch) >= self.deadband_rad else 0.0
        if phi == 0.0 and theta == 0.0:
            return
        legs_on = [leg for leg in LEGS if planted[leg]]
        if len(legs_on) < 3:
            self.frozen_reason = "fewer than 3 planted feet"
            return
        e = {leg: -self.anchors[leg][0] * theta + self.anchors[leg][1] * phi
             for leg in legs_on}
        mean_e = sum(e.values()) / len(legs_on)
        max_step = self.slew * dt
        for leg in legs_on:
            step = self.gain * (e[leg] - mean_e) * dt
            step = float(np.clip(step, -max_step, max_step))
            self.offsets[leg] = float(
                np.clip(self.offsets[leg] + step, -self.clamp, self.clamp))


def level_inputs(shared, ahrs_rp, now_mono, agree_veto_rad):
    """(roll, pitch, active, reason) for LevelingLoop.update.

    The AHRS is the independent watchdog: the leveling loop is closed on the
    EKF's attitude, so a large EKF-vs-AHRS split means one of them is lying
    and the loop must stop.  No AHRS at all -> no cross-check -> no loop.
    """
    if shared is None or not shared.est_ready or shared.out is None:
        return float("nan"), float("nan"), False, "EKF not ready"
    if now_mono - shared.tau_stamp > LEVEL_STALE_S:
        return float("nan"), float("nan"), False, "EKF stale"
    out = shared.out
    if not out.get("healthy", False):
        return float("nan"), float("nan"), False, "EKF unhealthy"
    roll, pitch = _rp(out["C"])
    if ahrs_rp is None:
        return roll, pitch, False, "no AHRS attitude"
    d_r = abs(roll - ahrs_rp[0])
    d_p = abs(pitch - ahrs_rp[1])
    if max(d_r, d_p) > agree_veto_rad:
        return roll, pitch, False, (f"EKF-AHRS disagree "
                                    f"{math.degrees(max(d_r, d_p)):.1f}deg")
    return roll, pitch, True, None


class HeightSettleGate:
    """True once the HEIGHT loop has stopped moving the trunk.

    The leveling setpoint is meant to be the attitude the trunk RESTS at, so
    it must not be sampled while the height loop is still winding in the sag.
    That ramp is ~13 mm at HEIGHT_SLEW_M_S -- seconds long -- and it tilts the
    trunk as it goes, because the front and rear legs do not sag equally.
    Latch mid-ramp and the setpoint is a transient the robot was only passing
    through; the leveling loop then holds it there for the rest of the stand,
    which reads as leveling making the attitude WORSE than the open-loop run.

    A height loop that is not running (--height-gain 0, or frozen by a veto)
    can never satisfy this test, so it counts as settled immediately --
    otherwise leveling could never engage in the open-loop-height A/B.
    """

    def __init__(self, hold_s=HEIGHT_SETTLE_S, band_m=s2.HEIGHT_DEADBAND_M):
        self.hold_s = float(hold_s)
        self.band = float(band_m)
        self.since = None

    def reset(self):
        self.since = None

    def update(self, now, height_loop_running, err_m):
        """Feed the height loop's state; returns True when it has been settled
        for `hold_s`."""
        if not height_loop_running or not math.isfinite(err_m):
            if self.since is None:
                self.since = float(now)
        elif abs(err_m) <= self.band:
            if self.since is None:
                self.since = float(now)
        else:
            self.since = None
        return self.since is not None and now - self.since >= self.hold_s


class SetpointLatch:
    """Setpoint = mean EKF attitude over LATCH_S once the stand has settled.

    Stage 1's result: the resting attitude is the setpoint, not zero -- it
    carries the IMU mount tilt.  A fixed CLI setpoint skips the latch.
    """

    def __init__(self, fixed_roll=None, fixed_pitch=None):
        self.fixed = (fixed_roll is not None and fixed_pitch is not None)
        self.sp_roll = fixed_roll if self.fixed else float("nan")
        self.sp_pitch = fixed_pitch if self.fixed else float("nan")
        self._acc = None

    def reset(self):
        if not self.fixed:
            self.sp_roll = self.sp_pitch = float("nan")
            self._acc = None

    @property
    def ready(self):
        return self.fixed or not math.isnan(self.sp_roll)

    def add(self, now, roll, pitch):
        """Feed settled HOLD4 attitude samples; returns True when it latches."""
        if self.ready:
            return False
        if self._acc is None:
            self._acc = [now, [], []]
        self._acc[1].append(float(roll))
        self._acc[2].append(float(pitch))
        if now - self._acc[0] >= LATCH_S:
            self.sp_roll = float(np.mean(self._acc[1]))
            self.sp_pitch = float(np.mean(self._acc[2]))
            return True
        return False


# ===========================================================================
# Stage machine -- stage 2's, with per-leg extra z and an honest contact mask
# ===========================================================================

class LevelStandSequence(s2.HeightStandSequence):
    """HeightStandSequence + per-leg z shifts (leveling + lift) in HOLD4.

    `extra_z[leg]` = leveling offset + lift, set by the run loop each sweep.
    PARK captures `foot_z(now)` -- extra included -- so parking still ramps
    from the pose actually held (P is only accepted with all feet down).
    """

    def __init__(self, tables, stand_height, t_stand=T_STAND):
        super().__init__(tables, stand_height, t_stand)
        self.extra_z = {leg: 0.0 for leg in LEGS}
        self.planted_mask = np.ones(4, dtype=bool)

    @property
    def contacts(self):
        if self.stage == "HOLD4":
            return self.planted_mask.copy()
        return np.ones(4, dtype=bool)

    def foot_z(self, now):
        z = super().foot_z(now)
        if self.stage == "HOLD4":
            z = {leg: z[leg] + self.extra_z[leg] for leg in LEGS}
        return z

    def clear_extra(self):
        self.extra_z = {leg: 0.0 for leg in LEGS}
        self.planted_mask = np.ones(4, dtype=bool)


# ===========================================================================
# Hardware run
# ===========================================================================

def run(port, stand_height, target_height, crouch_max_speed_dps, args):
    base.validate_hardware_config()
    print("[init] building per-leg height tables (IK, once) ...", flush=True)
    t0 = time.perf_counter()
    tables = s2.build_tables(stand_height,
                             clamp_m=args.clamp + args.level_clamp)
    print(f"[init] tables built in {time.perf_counter()-t0:.2f}s; "
          + "  ".join(f"{leg}[{tables[leg].z_min*1e3:.0f},"
                      f"{tables[leg].z_max*1e3:.0f}]mm" for leg in LEGS))
    # extension authority = how far below the stand the tables actually reach
    # (the legs' physical reach caps this well before the requested span)
    auth = min(-tables[leg].z_min - stand_height for leg in LEGS)
    need = 0.015 + args.level_clamp        # ~sag + leveling clamp
    print(f"[init] extension authority below the stand: {auth*1e3:.0f} mm"
          + ("" if auth >= need else
             f"  !! less than sag+leveling ({need*1e3:.0f} mm) -- "
             "consider a lower --stand-height"))

    if target_height is None:
        # trunk-bottom floor height of the open-loop stand pose (see stage 2)
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
    settle_gate = HeightSettleGate()
    agree_veto_rad = math.radians(args.agree_veto)
    tilt_stop_rad = math.radians(args.tilt_stop)
    # this runner never lifts a foot: the planted set is constant
    planted_d = {leg: True for leg in LEGS}
    planted_a = np.ones(4, dtype=bool)

    print("=" * 78)
    print("DOG5 CLOSED-LOOP LEVELING STAND  (stage 2b: EKF attitude in the loop)")
    print(f"  leveling: gain {args.level_gain:.2f}/s, clamp "
          f"+/-{args.level_clamp*1e3:.0f} mm/foot, AHRS veto {args.agree_veto:.1f} deg"
          + ("  setpoint FIXED "
             f"({args.setpoint_roll:+.2f},{args.setpoint_pitch:+.2f}) deg"
             if latch.fixed else f"  setpoint auto-latched over {LATCH_S:.0f}s"))
    print(f"  height:   gain {args.height_gain:.2f}/s, clamp "
          f"+/-{args.clamp*1e3:.0f} mm, target {target_height*1e3:.0f} mm "
          f"at the TRUNK BOTTOM/IMU (= {(target_height+IMU_BELOW_TRUNK_ORIGIN_M)*1e3:.0f}"
          f" mm at the abd axis), "
          f"FK veto {args.xcheck*1e3:.0f} mm (planted feet only)")
    print(f"  tilt-stop {args.tilt_stop:.0f} deg.  All four feet stay planted "
          "(lift experiment: lift_ekf_contact_hw.py).")
    print(f"  CAN: {args.control_hz:.0f} Hz/motor "
          f"({1e3/args.control_hz:.0f} ms between commands, 50 ms driver "
          "watchdog)")
    print("  NO torque commanded; drivers hold position.")
    print("  Keys: ENTER stand, P park, X stop.")
    print("  [hw] STAGE[flag]  h: ekf / fk cross-check / d=ekf-fk / "
          "corr=height-loop correction   (mm)")
    print("       rp: roll/pitch (deg, no yaw).  ekf=vs GRAVITY   fk=plane "
          "through the feet, vs the FLOOR, no IMU")
    print("           d=ekf-fk = floor slope + mount tilt + zero errors; its "
          "CHANGE across a shim is pure slope")
    if args.level_gain != 0.0:
        print("       lvl = per-foot leveling z, FL/FR/RL/RR (mm) -- the "
              "leveling loop's output")
    print("       flag: ---- loops idle | latching = waiting for the setpoint "
          "| LVL live | HOLD frozen")
    print(f"       temperature is printed only above {TEMP_NOTICE_C}C "
          f"(estop at {base.TEMP_ESTOP}C)")
    if args.no_limit_check:
        print("  !! --no-limit-check: measured-pose limits OFF (preflight "
              "refusal + runtime estop).")
        print("     Commanded targets are still clamped inside the soft "
              "limits by the z tables.")
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
            if not mb.arm(rate_hz=args.control_hz):
                raise RuntimeError("arm failed (bus / power / terminators)")
            if not feed.wait_for_raw(timeout=3.0):
                raise RuntimeError("no raw IMU (0x40) packets -- enable DETA10 raw mode")

            if args.no_limit_check:
                # Both measured-pose limit tests live in `base`: the preflight
                # ENTER refusal reads soft_limits() live, the runtime estop
                # reads SafetyGate's copy (taken at construction, above).  The
                # patch covers the first; `enforce_position_limits=False`
                # below covers the second.  COMMANDED targets are NOT affected
                # -- the z tables are still built inside the soft limits, so
                # this relaxes what we accept from the encoders, never what we
                # ask the motors to do.
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

            # Report which joints the limits would have rejected, and by how
            # much.  This is the number that says whether an encoder zero has
            # moved: a leg that LOOKS normal but reads far out of range is a
            # calibration shift, not a pose problem -- and Q_CROUCH was
            # recorded in the OLD zero frame, so the commanded crouch for that
            # leg is then wrong too.
            _lo, _hi = base.soft_limits()      # restored by the finally above
            _out = np.flatnonzero((start_q < _lo) | (start_q > _hi))
            if len(_out):
                print("[preflight] measured pose OUTSIDE the soft limits:")
                for j in _out:
                    over = (start_q[j] - _hi[j] if start_q[j] > _hi[j]
                            else start_q[j] - _lo[j])
                    print(f"    {base.JOINT_LABELS[j]:9s} "
                          f"{math.degrees(start_q[j]):+8.1f} deg  "
                          f"({math.degrees(over):+.1f} deg past the limit)")
                print("    Compare each against its mirror joint on the other "
                      "side at the same physical pose;")
                print("    a large split means that motor's 0x19 zero moved "
                      "and Q_CROUCH no longer fits it.")
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
            slot = mb.slot(args.control_hz)
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
            h_settled = False
            waiting_printed = False

            while True:
                mb.poll()
                joint_index = index % N_JOINTS
                if joint_index == 0:
                    now = time.perf_counter()
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

                    # ---- leveling loop ----
                    roll_e, pitch_e, l_active, l_reason = level_inputs(
                        shared, ahrs_rp, now_mono, agree_veto_rad)
                    l_engaged = l_active and seq.loop_engaged and latch.ready
                    lvl.update(now, roll_e, pitch_e, latch.sp_roll,
                               latch.sp_pitch, planted_d, l_engaged,
                               l_reason if not l_active else "not holding")
                    if seq.loop_engaged and latch.ready and l_reason \
                            and l_reason != last_l_reason:
                        print(f"[level] FROZEN: {l_reason}")
                    last_l_reason = l_reason

                    err_roll = err_pitch = float("nan")
                    if latch.ready and math.isfinite(roll_e):
                        err_roll = roll_e - latch.sp_roll
                        err_pitch = pitch_e - latch.sp_pitch

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
                    # the leveling setpoint may not be sampled until this is
                    # True -- see HeightSettleGate
                    h_settled = settle_gate.update(
                        now, h_engaged and args.height_gain != 0.0,
                        target_height - h_ekf)

                    # ---- compose per-leg targets ----
                    seq.extra_z = dict(lvl.offsets)

                    q_cmd, contacts, event = seq.update(
                        now, q, qd,
                        enter_pressed=pressed in ("\r", "\n"),
                        park_pressed=pressed in ("p", "P"),
                        offset=offset)
                    cmd_deg = np.rad2deg(q_cmd)
                    if seq.fault:
                        stop_reason = seq.fault
                        break

                    shared.q = q
                    shared.qd = qd
                    shared.stage = seq.stage
                    shared.contacts = contacts

                    if event == "crouch_settled":
                        quiet_t0 = now
                        print("[stage] CROUCH settled; EKF initialising. "
                              "ENTER to STAND.")
                    elif event == "stand_started":
                        quiet_t0 = None
                        hctrl.reset(0.0)
                        lvl.reset()
                        latch.reset()
                        settle_gate.reset()
                        waiting_printed = False
                        seq.clear_extra()
                        if seq.n_stands == 0 and log_t0 is not None:
                            init_secs = (now - start) - log_t0
                        print("[stage] STAND: open-loop ramp crouch -> stand.")
                    elif event == "stand_complete":
                        quiet_t0 = now
                        stats.begin_visit()
                        print(f"[stage] HOLD4 (stand #{seq.n_stands}): height "
                              "loop ENGAGED; leveling engages once the "
                              "setpoint latches. P parks, X stops.")
                    elif event == "park_started":
                        quiet_t0 = None
                        line = stats.end_visit()
                        if line:
                            print("[attitude] this stand scored:")
                            print(line)
                        print(f"[stage] PARK: unwinding offsets "
                              f"(height {offset*1e3:+.1f} mm, leveling "
                              + "/".join(f"{lvl.offsets[l]*1e3:+.1f}" for l in LEGS)
                              + " mm) on the way down.")
                    elif event == "park_complete":
                        quiet_t0 = now
                        hctrl.reset(0.0)
                        lvl.reset()
                        latch.reset()
                        settle_gate.reset()
                        waiting_printed = False
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
                    # IMU-free attitude from the measured encoders, for the
                    # EKF-vs-FK comparison on the status line
                    r_fk, p_fk = fk_attitude(q, planted_a)
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
                                if not latch.ready and l_active and h_settled:
                                    if not waiting_printed:
                                        waiting_printed = True
                                    if latch.add(now, re_, pe):
                                        print(f"[level] setpoint latched: roll "
                                              f"{math.degrees(latch.sp_roll):+.2f} "
                                              f"pitch {math.degrees(latch.sp_pitch):+.2f} "
                                              f"deg (height settled at "
                                              f"{offset*1e3:+.1f} mm) -- "
                                              "leveling ENGAGED.")
                                elif not latch.ready and l_active \
                                        and not waiting_printed:
                                    waiting_printed = True
                                    print("[level] holding the setpoint until "
                                          "the height loop settles -- sampling "
                                          "attitude mid-sag-ramp would latch a "
                                          "transient.")

                    if log_t0 is not None:
                        enc_log.append((now_mono, *q, *contacts.astype(int),
                                        r_a, p_a))

                    if now - last_print >= 1.0 / base.STATUS_HZ:
                        last_print = now
                        if not (shared.est_ready and out is not None):
                            print(f"[hw] {seq.stage:11s} waiting for EKF  "
                                  f"Tmax={int(np.max(temps))}C", flush=True)
                        else:
                            lflag = ("LVL" if l_engaged else
                                     ("latching" if seq.loop_engaged
                                      and not latch.ready else
                                      ("HOLD" if seq.loop_engaged else "----")))
                            warn = ""
                            if latched_err:
                                warn += f" latched={len(latched_err)}"
                            if tx_fail:
                                warn += f" txfail={tx_fail}"

                            # Each column is one loop's state, so a glance says
                            # what is being commanded and what came back:
                            #   z/fk   trunk-bottom height, EKF vs the FK veto
                            #   hoff   what the HEIGHT loop has wound in
                            #   att    EKF attitude, absolute (deg)
                            #   err    att MINUS the latched setpoint -- only
                            #          meaningful once latched, so before that
                            #          the column is a dash, not a NaN
                            #   lvl    per-foot z the LEVELING loop is holding
                            #          (FL/FR/RL/RR, mm) -- the actual output
                            def _deg(a):
                                """One angle, or an aligned dash when it is not
                                defined yet (an unlatched setpoint is not a
                                NaN error, and must not print like one)."""
                                return (f"{math.degrees(a):+6.2f}"
                                        if math.isfinite(a) else "    --")

                            # the leveling output is a column of zeros when the
                            # loop cannot produce one -- don't print it then
                            lvl_s = ("  lvl=" + "/".join(
                                f"{lvl.offsets[l]*1e3:+4.1f}" for l in LEGS)
                                if args.level_gain != 0.0 else "")
                            err_s = (f"  err={_deg(err_roll)}/"
                                     f"{_deg(err_pitch)}"
                                     if args.level_gain != 0.0 else "")
                            # temperature is estop-gated (base.TEMP_ESTOP=80C);
                            # show it only once it is worth a glance
                            t_max = int(np.max(temps))
                            hot = f"  {t_max}C" if t_max >= TEMP_NOTICE_C else ""
                            # attitude gets the same ekf/fk/d treatment as
                            # height: FK is the independent, IMU-free estimate
                            # and d is the number the EKF has to justify
                            print(f"[hw] {seq.stage:10s}[{lflag:8s}] "
                                  f"h:ekf={h_ekf*1e3:6.1f} fk={h_fk*1e3:6.1f} "
                                  f"d={(h_ekf-h_fk)*1e3:+4.1f} "
                                  f"corr={offset*1e3:+5.1f}mm  "
                                  f"rp:ekf={_deg(re_)}/{_deg(pe)} "
                                  f"imu={_deg(r_a)}/{_deg(p_a)} "
                                  f"fk={_deg(r_fk)}/{_deg(p_fk)} "
                                  f"d={_deg(re_-r_fk)}/{_deg(pe-p_fk)}deg"
                                  f"{err_s}{lvl_s}{hot}{warn}",
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
    print(f"[height] final offset {hctrl.offset*1e3:+.1f} mm"
          + (" (SATURATED)" if hctrl.saturated else ""))
    print("[level] final offsets "
          + "  ".join(f"{leg}{lvl.offsets[leg]*1e3:+.1f}" for leg in LEGS)
          + " mm" + (" (SATURATED)" if lvl.saturated else ""))
    print(f"[stop] {stop_reason}")


# ===========================================================================
# offline self-test
# ===========================================================================

_FAIL = []


def _check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))
    if not ok:
        _FAIL.append(name)


def _C_from_rp(roll, pitch):
    """An I->B rotation whose _rp() is exactly (roll, pitch): Rodrigues
    rotation taking e_z to the gravity direction implied by (roll, pitch)."""
    g = np.array([-math.sin(pitch),
                  math.cos(pitch) * math.sin(roll),
                  math.cos(pitch) * math.cos(roll)])
    e = np.array([0.0, 0.0, 1.0])
    v = np.cross(e, g)
    c = float(np.dot(e, g))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


class _PlanePlant:
    """Rigid trunk on stiff position legs, feet on a flat floor.

    Given commanded foot z per leg (trunk frame) and the planted set, solves
    the small-angle pose (h, roll, pitch) that puts every planted foot on the
    floor:   0 = h + z_i + roll*y_i - pitch*x_i   (least squares).
    `mount` adds the IMU mount tilt the EKF sees on top of the physical
    trunk-vs-floor attitude; `disturb` adds an external physical tilt.
    """

    def __init__(self, anchors, mount=(0.0, 0.0)):
        self.anchors = anchors
        self.mount = mount
        self.disturb = (0.0, 0.0)

    def solve(self, foot_z, planted):
        rows, rhs = [], []
        for leg in LEGS:
            if not planted[leg]:
                continue
            x, y = self.anchors[leg]
            rows.append([1.0, y, -x])
            rhs.append(-foot_z[leg])
        sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
        h, roll, pitch = float(sol[0]), float(sol[1]), float(sol[2])
        roll += self.disturb[0]
        pitch += self.disturb[1]
        return h, roll, pitch

    def ekf_attitude(self, foot_z, planted):
        h, roll, pitch = self.solve(foot_z, planted)
        return roll + self.mount[0], pitch + self.mount[1]


def _test_convention():
    """The _rp convention self-check the plant relies on."""
    for r, p in [(0.03, -0.01), (-0.05, 0.02), (0.0, 0.04)]:
        C = _C_from_rp(r, p)
        rr, pp = _rp(C)
        _check(f"_C_from_rp round-trips ({math.degrees(r):+.1f},"
               f"{math.degrees(p):+.1f}) deg",
               abs(rr - r) < 1e-9 and abs(pp - p) < 1e-9)
        break  # one detailed line; assert the rest silently
    ok = True
    for r, p in [(-0.05, 0.02), (0.0, 0.04), (0.1, -0.08)]:
        C = _C_from_rp(r, p)
        rr, pp = _rp(C)
        ok &= abs(rr - r) < 1e-9 and abs(pp - p) < 1e-9
    _check("_C_from_rp round-trips across the range", ok)


def _test_leveling(anchors):
    dt = 1.0 / CONTROL_HZ
    all_on = {leg: True for leg in LEGS}

    # pure roll error converges to the setpoint on the plane plant
    plant = _PlanePlant(anchors, mount=(math.radians(1.5), math.radians(-0.5)))
    plant.disturb = (math.radians(1.2), math.radians(-0.8))
    lvl = LevelingLoop(anchors)
    sp_r, sp_p = plant.mount        # setpoint = resting attitude, no disturb
    base_z = {leg: -0.22 for leg in LEGS}
    for k in range(int(60 / dt)):
        fz = {leg: base_z[leg] + lvl.offsets[leg] for leg in LEGS}
        r, p = plant.ekf_attitude(fz, all_on)
        lvl.update(k * dt, r, p, sp_r, sp_p, all_on, True)
    fz = {leg: base_z[leg] + lvl.offsets[leg] for leg in LEGS}
    r, p = plant.ekf_attitude(fz, all_on)
    _check("leveling drives a 1.2/-0.8 deg disturbance to the setpoint",
           abs(r - sp_r) < math.radians(0.25)
           and abs(p - sp_p) < math.radians(0.25),
           f"residual {math.degrees(r-sp_r):+.3f}/"
           f"{math.degrees(p-sp_p):+.3f} deg")
    h0 = plant.solve(base_z, all_on)[0]
    h1 = plant.solve(fz, all_on)[0]
    _check("zero-mean leveling leaves the trunk height alone",
           abs(h1 - h0) < 5e-4, f"{(h1-h0)*1e3:+.2f} mm")

    # deadband: tiny errors do not wind
    d = LevelingLoop(anchors)
    for k in range(int(5 / dt)):
        d.update(k * dt, math.radians(0.1), math.radians(-0.1), 0.0, 0.0,
                 all_on, True)
    _check("errors inside the deadband do not wind the offsets",
           all(v == 0.0 for v in d.offsets.values()))

    # clamp + slew under a huge permanent error
    c = LevelingLoop(anchors)
    prev = {leg: 0.0 for leg in LEGS}
    worst_rate = 0.0
    for k in range(int(20 / dt)):
        c.update(k * dt, math.radians(30), 0.0, 0.0, 0.0, all_on, True)
        for leg in LEGS:
            worst_rate = max(worst_rate, abs(c.offsets[leg] - prev[leg]) / dt)
            prev[leg] = c.offsets[leg]
    _check("offsets respect the clamp",
           max(abs(v) for v in c.offsets.values()) <= LEVEL_CLAMP_M + 1e-9
           and c.saturated)
    _check("offsets respect the slew limit",
           worst_rate <= LEVEL_SLEW_M_S + 1e-9,
           f"{worst_rate*1e3:.2f} mm/s")

    # freeze holds, never unwinds
    f = LevelingLoop(anchors)
    for k in range(int(10 / dt)):
        f.update(k * dt, math.radians(3), 0.0, 0.0, 0.0, all_on, True)
    held = dict(f.offsets)
    _check("offsets wound in before the freeze",
           max(abs(v) for v in held.values()) > 1e-4)
    for k in range(int(5 / dt)):
        f.update(10 + k * dt, 0.0, 0.0, 0.0, 0.0, all_on, False, "EKF stale")
    _check("frozen leveling holds its offsets, does not unwind",
           all(abs(f.offsets[leg] - held[leg]) < 1e-12 for leg in LEGS)
           and f.frozen_reason == "EKF stale")

    # planted-set awareness: lifted leg's offset frozen, mean over planted.
    # Run CLOSED LOOP on the 3-leg plant (open-loop constant error would just
    # saturate the clamp, where zero-mean cannot hold by construction).
    p3 = {leg: leg != "FL" for leg in LEGS}
    plant3 = _PlanePlant(anchors)
    plant3.disturb = (math.radians(0.8), math.radians(-0.6))
    g = LevelingLoop(anchors)
    for k in range(int(60 / dt)):
        fz = {leg: base_z[leg] + g.offsets[leg] for leg in LEGS}
        r3, p3_att = plant3.ekf_attitude(fz, p3)
        g.update(k * dt, r3, p3_att, 0.0, 0.0, p3, True)
    _check("a lifted leg's offset stays frozen",
           g.offsets["FL"] == 0.0,
           "FL " + f"{g.offsets['FL']*1e3:.2f} mm")
    mean3 = np.mean([g.offsets[leg] for leg in LEGS if leg != "FL"])
    _check("converged offsets stay zero-mean over the planted set",
           abs(mean3) < 5e-4, f"{mean3*1e3:+.3f} mm")
    two = {leg: leg in ("FL", "FR") for leg in LEGS}
    h2 = LevelingLoop(anchors)
    h2.update(0.0, 0.0, 0.0, 0.0, 0.0, two, True)
    h2.update(0.1, math.radians(3), 0.0, 0.0, 0.0, two, True)
    _check("fewer than 3 planted feet freezes leveling",
           all(v == 0.0 for v in h2.offsets.values())
           and "planted" in h2.frozen_reason)


def _test_latch():
    latch = SetpointLatch()
    _check("latch starts un-ready", not latch.ready)
    t = 0.0
    fired = False
    while t < LATCH_S + 0.1:
        fired |= latch.add(t, 0.02 + 0.001 * math.sin(t * 40), -0.01)
        t += 1.0 / CONTROL_HZ
    _check("latch fires after the window and lands on the mean",
           fired and latch.ready and abs(latch.sp_roll - 0.02) < 5e-4
           and abs(latch.sp_pitch + 0.01) < 1e-6,
           f"sp=({math.degrees(latch.sp_roll):+.3f},"
           f"{math.degrees(latch.sp_pitch):+.3f}) deg")
    fixed = SetpointLatch(0.01, 0.02)
    _check("fixed CLI setpoint skips the latch",
           fixed.ready and fixed.sp_roll == 0.01)
    latch.reset()
    _check("reset un-latches (fresh setpoint per stand)", not latch.ready)


def _test_settle_gate():
    """The gate that keeps the setpoint off the sag ramp."""
    dt = 1.0 / CONTROL_HZ
    g = HeightSettleGate()
    # a height loop still ramping (13 mm of sag, 5 mm/s) must NOT open it
    t, err, opened_during_ramp = 0.0, 0.013, False
    while err > s2.HEIGHT_DEADBAND_M:
        opened_during_ramp |= g.update(t, True, err)
        err -= s2.HEIGHT_SLEW_M_S * dt
        t += dt
    _check("gate stays shut while the height loop is still ramping",
           not opened_during_ramp, f"ramp took {t:.1f}s")
    # ... and opens HEIGHT_SETTLE_S after it arrives, not instantly
    opened_at = None
    t_arrive = t
    while t < t_arrive + HEIGHT_SETTLE_S + 0.5:
        if g.update(t, True, 0.0002) and opened_at is None:
            opened_at = t
        t += dt
    _check("gate opens only after the loop holds its target",
           opened_at is not None
           and abs((opened_at - t_arrive) - HEIGHT_SETTLE_S) < 5 * dt,
           f"opened {opened_at-t_arrive:.2f}s after arrival, "
           f"want {HEIGHT_SETTLE_S:.1f}s")
    # a fresh excursion outside the band must re-arm it
    g.update(t, True, 0.020)
    _check("leaving the band re-arms the gate", not g.update(t + dt, True, 0.020))
    # --height-gain 0 / frozen: nothing to settle, so do not block leveling
    g0 = HeightSettleGate()
    t = 0.0
    opened = False
    while t < HEIGHT_SETTLE_S + 0.5:
        opened |= g0.update(t, False, 0.013)     # loop not running, big error
        t += dt
    _check("a height loop that is not running counts as settled", opened)


class _FakeShared:
    def __init__(self, roll=0.0, pitch=0.0, healthy=True, ready=True, rz=0.0):
        self.est_ready = ready
        C = _C_from_rp(roll, pitch)
        self.out = None if not ready else {
            "r": np.array([0.0, 0.0, rz]), "healthy": healthy, "C": C}
        self.tau_stamp = time.monotonic()


def _test_level_vetoes():
    now = time.monotonic()
    veto = math.radians(AGREE_VETO_DEG)
    r, p, active, why = level_inputs(_FakeShared(0.02, -0.01), (0.02, -0.01),
                                     now, veto)
    _check("healthy agreeing EKF is accepted", active
           and abs(r - 0.02) < 1e-9, str(why))
    _, _, active, why = level_inputs(_FakeShared(ready=False), (0, 0), now, veto)
    _check("un-initialised EKF is vetoed", not active)
    _, _, active, why = level_inputs(_FakeShared(healthy=False), (0, 0), now, veto)
    _check("unhealthy EKF is vetoed", not active and why == "EKF unhealthy")
    stale = _FakeShared()
    stale.tau_stamp = now - 10 * LEVEL_STALE_S
    _, _, active, why = level_inputs(stale, (0, 0), now, veto)
    _check("stale EKF is vetoed", not active and why == "EKF stale")
    _, _, active, why = level_inputs(_FakeShared(), None, now, veto)
    _check("missing AHRS is vetoed (no cross-check, no loop)",
           not active and why == "no AHRS attitude")
    _, _, active, why = level_inputs(_FakeShared(math.radians(5), 0.0),
                                     (0.0, 0.0), now, veto)
    _check("EKF-vs-AHRS divergence vetoes the loop",
           not active and "disagree" in (why or ""), str(why))
    _, _, active, _ = level_inputs(_FakeShared(math.radians(1), 0.0),
                                   (0.0, 0.0), now, veto)
    _check("small EKF-AHRS disagreement is tolerated", active)


def _test_sequence(tables, stand_height, anchors):
    dt = 1.0 / CONTROL_HZ
    seq = LevelStandSequence(tables, stand_height)
    t, q, qd = 0.0, Q_CROUCH.copy(), np.zeros(N_JOINTS)
    seen = []
    while t < recorded.CROUCH_TIMEOUT_S + 3 * T_STAND and seq.stage != "PARKED":
        seq.update(t, q, qd,
                   enter_pressed=seq.stage == "WAIT_CROUCH",
                   park_pressed=(seq.stage == "HOLD4"
                                 and t - seq.stage_t0 > 2.0),
                   offset=0.005)
        q = seq.q_cmd(t)
        if seq.stage not in seen:
            seen.append(seq.stage)
        t += dt
    _check("stage order incl. park",
           seen == ["CROUCH", "WAIT_CROUCH", "STAND", "HOLD4", "PARK",
                    "PARKED"], " -> ".join(seen))
    _check("PARKED lands on the recorded crouch",
           float(np.max(np.abs(seq.q_cmd(t) - Q_CROUCH))) < 1e-5)

    s = LevelStandSequence(tables, stand_height)
    s.stage, s.stage_t0 = "HOLD4", 0.0
    q0 = s.q_cmd(0.0)
    s.extra_z = {"FL": 0.004, "FR": -0.002, "RL": 0.0, "RR": -0.002}
    q1 = s.q_cmd(0.0)
    ok = True
    for i, leg in enumerate(LEGS):
        f0 = dog5_kinematics.foot_position(leg, q0[3 * i:3 * i + 3])
        f1 = dog5_kinematics.foot_position(leg, q1[3 * i:3 * i + 3])
        ok &= abs((f1[2] - f0[2]) - s.extra_z[leg]) < 1e-4
    _check("per-leg extra z moves each commanded foot by its own amount", ok)

    s.planted_mask = np.array([True, True, False, True])
    _check("contact mask reaches the sequence in HOLD4",
           list(s.contacts) == [True, True, False, True])
    s2_ = LevelStandSequence(tables, stand_height)
    s2_.stage, s2_.stage_t0 = "PARK", 0.0
    s2_._park_from = dict(s2_.z_crouch)
    s2_.planted_mask = np.array([True, True, False, True])
    _check("contacts forced ON outside HOLD4", bool(np.all(s2_.contacts)))

    # PARK ramps from the held pose extra included
    s3 = LevelStandSequence(tables, stand_height)
    s3.stage, s3.stage_t0 = "HOLD4", 0.0
    s3.offset = 0.008
    s3.extra_z = {leg: 0.003 for leg in LEGS}
    held = s3.q_cmd(0.0)
    s3.update(1.0, Q_CROUCH, np.zeros(N_JOINTS), park_pressed=True,
              offset=0.008)
    _check("PARK starts from the pose actually being held (extra included)",
           float(np.max(np.abs(s3.q_cmd(1.0) - held))) < 1e-9)


def _test_fk_planted(tables, stand_height):
    q = np.zeros(N_JOINTS)
    for i, leg in enumerate(LEGS):
        q[3 * i:3 * i + 3] = tables[leg].q_at(-stand_height)
    h4 = fk_floor_height_planted(q, None, None, ref="hip")
    _check("planted FK with all feet down matches the stage-1 function",
           abs(h4 - s1.fk_floor_height(q, None, ref="hip")) < 1e-12)
    # lift FL by 15 mm: the all-four average is biased by lift/4, planted isn't
    ql = q.copy()
    ql[0:3] = tables["FL"].q_at(-stand_height + 0.015)
    h_all = s1.fk_floor_height(ql, None, ref="hip")
    h_planted = fk_floor_height_planted(ql, None,
                                        np.array([False, True, True, True]),
                                        ref="hip")
    # raising a foot makes the trunk look LOWER to the all-four average
    _check("planted FK ignores the lifted foot",
           abs(h_planted - h4) < 1e-4 and abs((h4 - h_all) - 0.015 / 4) < 2e-4,
           f"all-four biased {(h_all-h4)*1e3:+.2f} mm, planted "
           f"{(h_planted-h4)*1e3:+.2f} mm")


def _test_fk_attitude(tables, stand_height, anchors):
    """The IMU-free attitude estimate and what its gap to the EKF means."""
    # 1. the plane fit must invert _rp exactly, at any angle
    worst = 0.0
    for r_deg, p_deg in [(0, 0), (1.5, -0.5), (-3, 2), (8, -6), (0, 12)]:
        r, p = math.radians(r_deg), math.radians(p_deg)
        u = _C_from_rp(r, p) @ np.array([0.0, 0.0, 1.0])
        b, c = -u[0] / u[2], -u[1] / u[2]
        feet = [(x, y, -stand_height + b * x + c * y)
                for x, y in (anchors[leg] for leg in LEGS)]
        rr, pp = fk_attitude_from_feet(feet)
        worst = max(worst, abs(rr - r), abs(pp - p))
    _check("plane-fit attitude inverts the EKF's _rp convention",
           worst < 1e-9, f"worst {math.degrees(worst)*1e6:.2f} udeg")

    # 2. THE claim the shim test rests on: EKF - FK is the floor slope, and
    #    the trunk's own attitude drops out of it (so mount tilt and encoder
    #    zeros, being constants, cancel in the CHANGE across a shim)
    for slope_deg, axis in [(2.0, "roll"), (3.0, "pitch")]:
        s = math.radians(slope_deg)
        n = (np.array([0.0, -math.sin(s), math.cos(s)]) if axis == "roll"
             else np.array([math.sin(s), 0.0, math.cos(s)]))
        for trunk in [(0.0, 0.0), (1.0, -0.5), (-2.0, 1.5)]:
            C = _C_from_rp(math.radians(trunk[0]), math.radians(trunk[1]))
            u = C @ n                       # floor normal in body coords
            b, c = -u[0] / u[2], -u[1] / u[2]
            feet = [(x, y, -stand_height + b * x + c * y)
                    for x, y in (anchors[leg] for leg in LEGS)]
            r_fk, p_fk = fk_attitude_from_feet(feet)
            r_ek, p_ek = _rp(C)
            got = (r_ek - r_fk) if axis == "roll" else (p_ek - p_fk)
            off = (p_ek - p_fk) if axis == "roll" else (r_ek - r_fk)
            ok = abs(got - s) < math.radians(0.01) \
                and abs(off) < math.radians(0.01)
            if not ok:
                break
        _check(f"EKF-FK attitude = the floor slope ({slope_deg:.0f} deg "
               f"{axis}), whatever the trunk does", ok,
               f"read {math.degrees(got):+.4f} deg")

    # 3. fewer than 3 planted feet cannot define a plane
    r, p = fk_attitude_from_feet([(0.2, 0.1, -0.19), (-0.2, -0.1, -0.19)])
    _check("two feet do not define an attitude", math.isnan(r))
    q = np.zeros(N_JOINTS)
    for i, leg in enumerate(LEGS):
        q[3 * i:3 * i + 3] = tables[leg].q_at(-stand_height)
    r3, p3 = fk_attitude(q, np.array([True, True, True, False]))
    _check("three planted feet still give an attitude", math.isfinite(r3))

    # 4. the commanded stand is a horizontal foot plane by construction, so
    #    FK attitude reads zero there -- it is only non-zero once the legs
    #    differ, which on hardware means sag
    r0, p0 = fk_attitude(q)
    _check("FK attitude of the commanded stand is level",
           abs(r0) < 1e-9 and abs(p0) < 1e-9,
           f"{math.degrees(r0):+.2e}/{math.degrees(p0):+.2e} deg")

    # 5. differential sag is what it actually measures: drop the two front
    #    feet 5 mm and the nose must read DOWN (+pitch, _rp convention)
    q_sag = q.copy()
    for i, leg in enumerate(LEGS):
        if anchors[leg][0] > 0:                    # front legs
            q_sag[3 * i:3 * i + 3] = tables[leg].q_at(-stand_height + 0.005)
    r_s, p_s = fk_attitude(q_sag)
    span = abs(anchors["FL"][0] - anchors["RL"][0])
    _check("5 mm of front-leg sag reads as nose-down pitch",
           p_s > 0 and abs(p_s - math.atan2(0.005, span)) < math.radians(0.05),
           f"{math.degrees(p_s):+.3f} deg over a {span*1e3:.0f} mm wheelbase")


def _test_closed_loop(tables, stand_height, anchors):
    """End to end on all four feet: stand, latch the resting attitude, take a
    tilt disturbance, and hold both attitude and height through it."""
    dt = 1.0 / CONTROL_HZ
    mount = (math.radians(1.5), math.radians(-0.5))
    plant = _PlanePlant(anchors, mount=mount)
    lvl = LevelingLoop(anchors)
    hctrl = s2.HeightController()
    latch = SetpointLatch()
    all_on = {leg: True for leg in LEGS}
    # the loop's frame: floor to trunk bottom.  _PlanePlant.solve returns the
    # trunk ORIGIN (abd axis) above the FOOT SITES, so converting costs a foot
    # radius up to the floor and an IMU lever back down -- do it once, here,
    # exactly as the geometry does it on hardware.
    PLANT_TO_BOTTOM = FOOT_RADIUS_M - IMU_BELOW_TRUNK_ORIGIN_M
    target_h = stand_height + PLANT_TO_BOTTOM
    SAG = 0.010
    base_z = {leg: -stand_height for leg in LEGS}

    t, events, max_err = 0.0, [], 0.0
    while t < 90.0:
        fz = {leg: base_z[leg] - hctrl.offset + lvl.offsets[leg] for leg in LEGS}
        h, roll, pitch = plant.solve(fz, all_on)
        h += PLANT_TO_BOTTOM
        h -= SAG
        roll += mount[0]
        pitch += mount[1]
        if not latch.ready and t > SETTLE_S:
            if latch.add(t, roll, pitch):
                events.append("latched")
        engaged = latch.ready
        lvl.update(t, roll, pitch, latch.sp_roll if engaged else 0.0,
                   latch.sp_pitch if engaged else 0.0, all_on, engaged)
        hctrl.update(t, h, target_h, True)
        # a 1.2/-0.9 deg disturbance arrives once the setpoint is latched
        if "latched" in events and "disturbed" not in events and t > 20.0:
            plant.disturb = (math.radians(1.2), math.radians(-0.9))
            events.append("disturbed")
        if "disturbed" in events:
            max_err = max(max_err, abs(roll - latch.sp_roll),
                          abs(pitch - latch.sp_pitch))
        t += dt

    fz = {leg: base_z[leg] - hctrl.offset + lvl.offsets[leg] for leg in LEGS}
    h, roll, pitch = plant.solve(fz, all_on)
    h += PLANT_TO_BOTTOM
    h -= SAG
    roll += mount[0]
    pitch += mount[1]
    _check("sequence ran: latch -> disturbance", events == ["latched", "disturbed"],
           str(events))
    _check("setpoint latched onto the resting (mount-tilted) attitude",
           abs(latch.sp_roll - mount[0]) < math.radians(0.05)
           and abs(latch.sp_pitch - mount[1]) < math.radians(0.05),
           f"sp=({math.degrees(latch.sp_roll):+.2f},"
           f"{math.degrees(latch.sp_pitch):+.2f}) deg")
    _check("disturbance transient stayed bounded",
           max_err < math.radians(2.0), f"max {math.degrees(max_err):.2f} deg")
    _check("leveling drove the disturbance back to the setpoint",
           abs(roll - latch.sp_roll) < math.radians(0.3)
           and abs(pitch - latch.sp_pitch) < math.radians(0.3),
           f"{math.degrees(roll-latch.sp_roll):+.3f}/"
           f"{math.degrees(pitch-latch.sp_pitch):+.3f} deg")
    _check("height loop held the trunk-bottom target through it",
           abs(h - target_h) <= s2.HEIGHT_DEADBAND_M + 1e-6,
           f"{(h-target_h)*1e3:+.2f} mm")
    # the offset must equal the sag -- not the clamp.  Before the frame fix
    # this test's target was a foot radius off, so it converged only because
    # the required 30 mm happened to equal the clamp exactly.
    _check("height offset recovered the sag, and is nowhere near the clamp",
           abs(hctrl.offset - SAG) <= s2.HEIGHT_DEADBAND_M + 1e-6
           and not hctrl.saturated,
           f"{hctrl.offset*1e3:.1f} mm vs {SAG*1e3:.0f} mm sag, clamp "
           f"{s2.HEIGHT_CLAMP_M*1e3:.0f} mm")


def _test_timing(tables, anchors):
    lvl = LevelingLoop(anchors)
    planted = {leg: True for leg in LEGS}
    t0 = time.perf_counter()
    N = 200
    for k in range(N):
        lvl.update(k * 0.004, 0.01, -0.01, 0.0, 0.0, planted, True)
        for leg in LEGS:
            tables[leg].q_at(-0.22 + lvl.offsets[leg])
    per = (time.perf_counter() - t0) / N
    _check("leveling + 4-leg lookup fit the CAN slot", per < 250e-6,
           f"{per*1e6:.0f} us per sweep")


def self_test(stand_height):
    print("stand_ekf_level_hw self-test (no hardware)")
    print("[1] rotation convention")
    _test_convention()
    print("[2] height tables (stage 2's, wider span)")
    t0 = time.perf_counter()
    tables = s2.build_tables(stand_height,
                             clamp_m=s2.HEIGHT_CLAMP_M + LEVEL_CLAMP_M)
    print(f"  (built in {time.perf_counter()-t0:.2f}s)")
    anchors = {leg: tables[leg].xy for leg in LEGS}
    # The legs' physical reach caps the downward span (~-221 mm) well before
    # the requested clamp sum, so assert the authority that matters: room for
    # the measured sag (~13 mm) + the leveling clamp below the stand, and the
    # full lift above it.
    span_ok = all(tables[leg].z_min <= -stand_height - (0.013 + LEVEL_CLAMP_M)
                  for leg in LEGS)
    _check("table span covers sag + leveling authority below the stand",
           span_ok,
           "  ".join(f"{leg}[{tables[leg].z_min*1e3:.0f},"
                     f"{tables[leg].z_max*1e3:.0f}]" for leg in LEGS))
    print("[3] leveling loop on the plane plant")
    _test_leveling(anchors)
    print("[4] setpoint latch")
    _test_latch()
    _test_settle_gate()
    print("[5] leveling vetoes (AHRS watchdog)")
    _test_level_vetoes()
    print("[6] stage machine with per-leg extra z")
    _test_sequence(tables, stand_height, anchors)
    print("[7] FK attitude (IMU-free) vs the EKF")
    _test_fk_attitude(tables, stand_height, anchors)
    print("[7b] planted-aware FK height")
    _test_fk_planted(tables, stand_height)
    print("[8] closed loop end to end, four feet")
    _test_closed_loop(tables, stand_height, anchors)
    print("[9] timing")
    _test_timing(tables, anchors)
    print("self-test " + ("FAIL: " + ", ".join(_FAIL) if _FAIL else "PASS"))
    return 1 if _FAIL else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--stand-height", type=float,
                    default=STAND_HEIGHT_DEFAULT,
                    help="open-loop pose height; default LOWER than stage 2's "
                         "to keep leg-extension authority for leveling")
    ap.add_argument("--target-height", type=float, default=None,
                    help="height-loop target, metres from the floor to the "
                         "TRUNK BOTTOM / IMU BOARD (NOT the abd axis, which is "
                         f"{IMU_BELOW_TRUNK_ORIGIN_M*1e3:.0f} mm higher)")
    ap.add_argument("--crouch-max-speed-dps", type=float,
                    default=recorded.DEFAULT_CROUCH_MAX_MOTOR_DPS)
    ap.add_argument("--height-gain", type=float, default=s2.HEIGHT_GAIN_PER_S)
    ap.add_argument("--clamp", type=float, default=s2.HEIGHT_CLAMP_M,
                    help="height-loop max |offset| (m)")
    ap.add_argument("--xcheck", type=float, default=s2.HEIGHT_XCHECK_M,
                    help="|EKF-FK| height veto (m, planted feet only)")
    ap.add_argument("--level-gain", type=float, default=LEVEL_GAIN_PER_S,
                    help="leveling integral gain (1/s); 0 disables leveling")
    ap.add_argument("--level-clamp", type=float, default=LEVEL_CLAMP_M,
                    help="leveling max |offset| per foot (m)")
    ap.add_argument("--agree-veto", type=float, default=AGREE_VETO_DEG,
                    help="|EKF-AHRS| attitude split that freezes leveling (deg)")
    # Default setpoint is ABSOLUTE LEVEL (0,0): the IMU offset was calibrated
    # with the robot lying flat, so IMU zero IS level -- mount tilt is inside
    # the calibration, not something to preserve.  --setpoint-latch restores
    # the old behaviour (hold whatever attitude the stand settles at).
    ap.add_argument("--setpoint-roll", type=float, default=0.0,
                    help="leveling setpoint roll (deg; default 0 = level)")
    ap.add_argument("--setpoint-pitch", type=float, default=0.0,
                    help="leveling setpoint pitch (deg; default 0 = level)")
    ap.add_argument("--setpoint-latch", action="store_true",
                    help="latch the setpoint from the settled HOLD4 attitude "
                         "instead of using the fixed values above")
    ap.add_argument("--control-hz", type=float, default=CONTROL_HZ,
                    help="per-motor command rate (Hz); lower = more CAN "
                         f"margin under the 50 ms driver watchdog "
                         f"(default {CONTROL_HZ:.0f})")
    ap.add_argument("--tilt-stop", type=float, default=TILT_STOP_DEG,
                    help="absolute attitude that soft-stops the run (deg)")
    ap.add_argument("--no-limit-check", action="store_true",
                    help="accept ANY measured joint pose: skips the preflight "
                         "soft-limit refusal and the runtime position estop. "
                         "Commanded targets are unaffected (the z tables are "
                         "still built inside the soft limits). Use when an "
                         "encoder zero makes a physically fine pose read out "
                         "of range -- see the FR_knee/id12 note in README.")
    ap.add_argument("--raw-log", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.setpoint_latch:
        args.setpoint_roll = args.setpoint_pitch = None    # auto-latch mode
    if not 20.0 <= args.control_hz <= 300.0:
        ap.error("--control-hz outside [20, 300] (bus ceiling is 300)")
    if args.self_test:
        sys.exit(self_test(args.stand_height))
    run(args.port, args.stand_height, args.target_height,
        args.crouch_max_speed_dps, args)


if __name__ == "__main__":
    main()
