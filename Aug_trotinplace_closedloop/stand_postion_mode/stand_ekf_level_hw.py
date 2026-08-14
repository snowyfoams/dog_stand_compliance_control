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

# Every tunable this script has is in stand_params.py -- the LEVEL_* block
# there is this loop's (gain, clamp, slew, deadband) and the stand height,
# latch window and AHRS veto sit beside it.
from stand_params import (FOOT_RADIUS_M,                 # noqa: E402
                          IMU_BELOW_TRUNK_ORIGIN_M,
                          T_STAND, EKF_WORKER_HZ, QUIET_STAGES, SETTLE_S,
                          STAND_HEIGHT_DEFAULT,
                          LEVEL_GAIN_PER_S, LEVEL_CLAMP_M, LEVEL_SLEW_M_S,
                          LEVEL_DEADBAND_DEG, LEVEL_LPF_FC_HZ, LEVEL_STALE_S,
                          LATCH_S, HEIGHT_SETTLE_S, AGREE_VETO_DEG,
                          TILT_STOP_DEG, TEMP_NOTICE_C)

MOTOR_IDS = base.MOTOR_IDS
N_JOINTS = base.N_JOINTS
CONTROL_HZ = base.CONTROL_HZ
LEGS = base.LEGS
Q_CROUCH = recorded.Q_RECORDED_CROUCH


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
# offline self-test -> test_stand_ekf_level.py
# ===========================================================================
# The gates moved out of this file.  They were 540 lines of simulation that no
# hardware run ever executes, and several of them shared a plane plant, a fake
# EKF and a PASS/FAIL harness with the other runners -- all of which now live
# in selftest_common.py.  `--self-test` still runs exactly those gates.
#
#     $V self-test/test_stand_ekf_level.py     # the same thing, run directly


def self_test(stand_height):
    """Delegate to the suite in self-test/.  Lazy: a hardware run never loads it."""
    _sd = s1.selftest_dir()
    if _sd not in sys.path:
        sys.path.insert(0, _sd)
    from test_stand_ekf_level import self_test as gates
    return gates(stand_height)


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
