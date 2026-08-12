#!/usr/bin/env python3
"""Stage 2b experiment: what the EKF does when one foot leaves the ground.

WHAT IT DOES
    Lifts one foot 20 mm while the imported loops hold the trunk on the other
    three, and prints the EKF's height beside forward kinematics at three
    moments: four feet down, three feet, re-planted.  Every control law is
    imported from stand_ekf_level_hw.py -- this runner adds only the lift,
    the contact schedule handed to the filter, and the measurement.

THE RESULT ALREADY IN HAND (hardware 2026-08-12)
    zEKF and zFK AGREE, and that is CORRECT.  Three planted feet determine
    trunk height completely, which is the same information FK uses, so a
    one-leg lift cannot separate them -- the filter's height is leg odometry,
    corrected every update (build_measurement puts each planted foot's
    correction straight into body position).  Agreement means healthy.
    Use --fake-contacts for the informative run: it tells the EKF all four
    feet stay planted while one is physically lifted, so zEKF walks about
    lift/4 (~5 mm) off zFK.  The gap between the two runs is the answer.

HOW TO RUN
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    cd ~/Documents/can_motor_control/dog_stand_compliance_control/Aug_trotinplace_closedloop

    $V lift_ekf_contact_hw.py --self-test       # offline gates, no hardware

    # 1. honest contacts -- expect zEKF and zFK to agree
    sudo chrt -f 50 $V lift_ekf_contact_hw.py
    # 2. the A/B -- expect zEKF to walk ~5 mm off zFK
    sudo chrt -f 50 $V lift_ekf_contact_hw.py --fake-contacts

    --lift 0.030        a bigger lift
    --raw-log lift.npz  also save raw IMU+encoder for state_estimator/hw_replay
    --no-limit-check    if the preflight refuses on a measured joint pose

    Keys: ENTER stand, then 1-4 lift/lower FL FR RL RR (HOLD4 only, once
    leveling has latched), P park (all feet down), X stop.
    Support the robot at the zero-torque prompt, then hold it still through
    WAIT_CROUCH -- the EKF's gyro-bias init runs there and nowhere else.

WHAT TO WATCH
    [lift] FL HELD UP at 20 mm, 3 planted
             zEKF(bottom) =  173.1 mm   zFK(bottom, 3 planted) =  171.9 mm ...
             vs all-4 stance: ...  disagreement +0.7 mm
    "disagreement" is the number this run exists to produce.
    All heights are floor-to-TRUNK-BOTTOM (the IMU board); the abd axis is
    38 mm higher and shows on the status line as abd=.
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

# ---- the leveling controller, imported whole -------------------------------
# This runner owns no control law.  Everything below is stand_ekf_level_hw's,
# aliased so the loop reads the same in both files.
fk_floor_height_planted = lv.fk_floor_height_planted
height_inputs = lv.height_inputs
LevelingLoop = lv.LevelingLoop
level_inputs = lv.level_inputs
SetpointLatch = lv.SetpointLatch
LevelStandSequence = lv.LevelStandSequence
LEVEL_GAIN_PER_S = lv.LEVEL_GAIN_PER_S
LEVEL_CLAMP_M = lv.LEVEL_CLAMP_M
LEVEL_SLEW_M_S = lv.LEVEL_SLEW_M_S
LATCH_S = lv.LATCH_S
AGREE_VETO_DEG = lv.AGREE_VETO_DEG

# ---- lift experiment -------------------------------------------------------
LIFT_M = 0.020               # default commanded foot lift (visible clearance)
LIFT_SLEW_M_S = 0.008        # lift/lower ramp rate (~2 s for 15 mm)
CONTACT_OFF_M = 0.003        # commanded lift beyond this -> contact False
LIFT_SETTLE_S = 1.0          # settle time at full lift before scoring zEKF/zFK
LEAN_ABORT_DEG = 6.0         # attitude error that auto-lowers the lift
TILT_STOP_DEG = 12.0         # absolute attitude that soft-stops the run

LIFT_KEYS = {"1": "FL", "2": "FR", "3": "RL", "4": "RR"}



# ===========================================================================
# Lift manager
# ===========================================================================

class LiftManager:
    """One foot's slew-limited lift, and the honest contact schedule.

    A lift is commanded (keys 1-4), ramped at LIFT_SLEW_M_S, and reported to
    the EKF as contact False once the commanded lift exceeds CONTACT_OFF_M --
    approximate at the edges (the foot may drag through the first/last mm),
    but the schedule error is bounded by the threshold and brief.  Only one
    leg at a time; a second request while active is refused.
    """

    def __init__(self, lift_m=LIFT_M, slew_m_s=LIFT_SLEW_M_S,
                 contact_off_m=CONTACT_OFF_M):
        self.lift_m = float(lift_m)
        self.slew = float(slew_m_s)
        self.contact_off = float(contact_off_m)
        self.leg = None
        self.cur = 0.0
        self.goal = 0.0
        self._last = None

    @property
    def active(self):
        return self.leg is not None

    @property
    def airborne(self):
        return self.leg is not None and self.cur > self.contact_off

    @property
    def at_goal(self):
        """Lifted to the commanded height and no longer moving."""
        return (self.leg is not None and self.goal > 0.0
                and abs(self.cur - self.goal) < 1e-6)

    def request(self, leg):
        """Toggle: start a lift, or lower the currently lifted leg."""
        if self.leg is None:
            self.leg, self.goal = leg, self.lift_m
            return f"lifting {leg} to {self.lift_m*1e3:.0f} mm"
        if leg == self.leg:
            self.goal = 0.0
            return f"lowering {self.leg}"
        return f"refused: {self.leg} is already lifted (press {self._key()} to lower)"

    def lower(self, reason=None):
        if self.leg is not None and self.goal != 0.0:
            self.goal = 0.0
            return f"auto-lowering {self.leg}" + (f": {reason}" if reason else "")
        return None

    def reset(self):
        self.leg, self.cur, self.goal, self._last = None, 0.0, 0.0, None

    def advance(self, now):
        if self._last is None:
            self._last = float(now)
            return
        dt = float(np.clip(now - self._last, 0.0, 0.1))
        self._last = float(now)
        if self.leg is None:
            return
        step = float(np.clip(self.goal - self.cur, -self.slew * dt,
                             self.slew * dt))
        self.cur += step
        if self.goal == 0.0 and abs(self.cur) < 1e-6:
            self.leg, self.cur = None, 0.0

    def lift_z(self):
        """Per-leg upward z shift (m, >= 0)."""
        return {leg: (self.cur if leg == self.leg else 0.0) for leg in LEGS}

    def planted(self):
        return {leg: not (leg == self.leg and self.cur > self.contact_off)
                for leg in LEGS}

    def _key(self):
        return next(k for k, v in LIFT_KEYS.items() if v == self.leg)


def height_compare_block(title, h_ekf, h_fk, n_planted, base=None):
    """The zEKF-vs-zFK comparison -- the point of the lift.

    Both numbers are the TRUNK-BOTTOM (IMU board) height above the floor, the
    same reference stage 1's `zFK` and stage 2's target use: `r_z` is the
    IMU's rise since init and is used untouched, `zFK` is
    `fk_floor_height_planted(ref="imu")`.  The abd axis is
    IMU_BELOW_TRUNK_ORIGIN_M higher and appears only as the `abd=` cross-check
    on the status line.

    EXPECT THEM TO AGREE while any foot is planted.  The leg measurement
    (`build_measurement`) puts its correction straight into body position
    (`H[:, ErrorIndex.R] = -C`), so with a valid anchored foothold the EKF's
    height is leg odometry, corrected every update.  Three planted feet
    determine the trunk height completely -- exactly the information FK uses.
    Agreement is the filter working, not the test failing; divergence here
    would mean the EKF had lost the plot.  To see the filter actually *using*
    contacts, run `--fake-contacts` and watch this number walk.
    """
    lines = [
        f"[lift] {title}",
        f"         zEKF(bottom) = {h_ekf*1e3:7.1f} mm    "
        f"zFK(bottom, {n_planted} planted) = {h_fk*1e3:7.1f} mm    "
        f"zEKF-zFK = {(h_ekf-h_fk)*1e3:+6.1f} mm",
    ]
    if base is not None and all(math.isfinite(v) for v in base):
        b_ekf, b_fk = base
        grew = ((h_ekf - h_fk) - (b_ekf - b_fk)) * 1e3
        lines.append(
            f"         vs all-4 stance:  zEKF {(h_ekf-b_ekf)*1e3:+6.1f} mm    "
            f"zFK {(h_fk-b_fk)*1e3:+6.1f} mm    "
            f"disagreement {grew:+.1f} mm")
    return lines


class LiftReport:
    """Per-lift scoring: what did the EKF do while the foot was in the air."""

    def __init__(self):
        self.rows = []           # finished lifts
        self._cur = None

    def start(self, leg, now, drift_mm):
        self._cur = {"leg": leg, "t0": now, "air_s": 0.0, "drift0": drift_mm,
                     "max_err_deg": 0.0, "max_agree_deg": 0.0,
                     "drift_end": drift_mm}

    def add(self, now, dt, airborne, err_roll, err_pitch, agree_deg, drift_mm):
        if self._cur is None:
            return
        if airborne:
            self._cur["air_s"] += dt
        err = max(abs(err_roll), abs(err_pitch))
        if math.isfinite(err):
            self._cur["max_err_deg"] = max(self._cur["max_err_deg"],
                                           math.degrees(err))
        if math.isfinite(agree_deg):
            self._cur["max_agree_deg"] = max(self._cur["max_agree_deg"], agree_deg)
        if math.isfinite(drift_mm):
            self._cur["drift_end"] = drift_mm

    def finish(self, now):
        if self._cur is None:
            return None
        c = self._cur
        self._cur = None
        c["total_s"] = now - c["t0"]
        self.rows.append(c)
        return (f"[lift] {c['leg']}: {c['total_s']:.1f}s total, "
                f"{c['air_s']:.1f}s airborne; max attitude err "
                f"{c['max_err_deg']:.2f}deg, max |EKF-AHRS| "
                f"{c['max_agree_deg']:.2f}deg, EKF-FK drift "
                f"{c['drift0']:+.1f} -> {c['drift_end']:+.1f} mm")

    def summary(self):
        if not self.rows:
            return ["[lift] no lifts performed"]
        lines = [f"[lift] {len(self.rows)} lift(s) scored "
                 "(EKF through a real contact change):"]
        for c in self.rows:
            lines.append(f"  {c['leg']}: air {c['air_s']:4.1f}s  "
                         f"max err {c['max_err_deg']:5.2f}deg  "
                         f"agree {c['max_agree_deg']:5.2f}deg  "
                         f"drift {c['drift0']:+5.1f} -> {c['drift_end']:+5.1f} mm")
        return lines



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
    lift = LiftManager(lift_m=args.lift)
    lift_report = LiftReport()
    agree_veto_rad = math.radians(args.agree_veto)
    lean_abort_rad = math.radians(args.lean_abort)
    tilt_stop_rad = math.radians(args.tilt_stop)

    print("=" * 78)
    print("DOG5 LEG-LIFT CONTACT EXPERIMENT  (zEKF vs zFK across a contact "
          "change)")
    print("  control laws imported from stand_ekf_level_hw; this runner adds "
          "the lift only")
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
    print(f"  lift:     {args.lift*1e3:.0f} mm at {LIFT_SLEW_M_S*1e3:.0f} mm/s; "
          f"lean-abort {args.lean_abort:.0f} deg, tilt-stop {args.tilt_stop:.0f} deg")
    print("  NO torque commanded; drivers hold position.")
    print("  Keys: ENTER stand, 1-4 lift/lower FL FR RL RR, P park, X stop.")
    if args.fake_contacts:
        print("  !! --fake-contacts: the EKF is told all 4 feet stay planted "
              "through a lift (A/B).")
        print("     zFK and the motion stay honest; expect zEKF to walk off "
              "by roughly lift/4.")
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
            if not mb.arm(rate_hz=CONTROL_HZ):
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
            lift_base = None          # (h_ekf, h_fk) captured on all 4 feet
            lift_held_printed = False
            lift_settle_t0 = None

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

                    # ---- lift bookkeeping (before the loops: contacts) ----
                    lift.advance(now)
                    planted_d = lift.planted()
                    planted_a = np.array([planted_d[leg] for leg in LEGS])

                    # ---- leveling loop ----
                    roll_e, pitch_e, l_active, l_reason = level_inputs(
                        shared, ahrs_rp, now_mono, agree_veto_rad)
                    l_engaged = l_active and seq.loop_engaged and latch.ready
                    if not l_active and lift.active:
                        msg = lift.lower(l_reason)
                        if msg:
                            print(f"[lift] {msg}")
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
                        if lift.active and max(abs(err_roll),
                                               abs(err_pitch)) > lean_abort_rad:
                            msg = lift.lower(
                                f"lean {math.degrees(max(abs(err_roll), abs(err_pitch))):.1f}deg")
                            if msg:
                                print(f"[lift] {msg}")

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
                    lz = lift.lift_z()
                    seq.extra_z = {leg: lvl.offsets[leg] + lz[leg] for leg in LEGS}
                    seq.planted_mask = planted_a

                    park_req = pressed in ("p", "P")
                    if park_req and lift.active:
                        print("[park] refused: lower the lifted leg first "
                              f"(press {lift._key()})")
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

                    # ---- lift keys (after seq.update so stage is fresh) ----
                    if pressed in LIFT_KEYS and seq.stage == "HOLD4":
                        if not l_engaged:
                            print("[lift] refused: leveling loop not engaged "
                                  f"({l_reason or 'setpoint not latched yet'})")
                        else:
                            was_active = lift.active
                            print(f"[lift] {lift.request(LIFT_KEYS[pressed])}")
                            if not was_active and lift.active:
                                drift0 = float("nan")
                                if z0_report is not None and math.isfinite(h_fk):
                                    drift0 = (float(shared.out["r"][2])
                                              - (fk_floor_height_planted(
                                                  q, shared.out["C"], planted_a,
                                                  ref="imu") - z0_report)) * 1e3
                                lift_report.start(lift.leg, now, drift0)
                                # baseline taken with all four feet still down
                                lift_base = (h_ekf, h_fk)
                                lift_held_printed = False
                                lift_settle_t0 = None
                                for ln in height_compare_block(
                                        f"{lift.leg} baseline, all 4 planted",
                                        h_ekf, h_fk, 4):
                                    print(ln)
                    elif pressed in LIFT_KEYS:
                        print("[lift] refused: only in HOLD4")

                    shared.q = q
                    shared.qd = qd
                    shared.stage = seq.stage
                    # --fake-contacts LIES to the EKF: the physical motion and
                    # zFK stay honest, only the filter's contact belief changes.
                    # With a foot up but still flagged planted, its leg
                    # measurement insists a foot that actually rose has not
                    # moved, so the correction goes into body z (H[:, R] = -C)
                    # and zEKF walks away from zFK by roughly lift/4.  That
                    # divergence is the proof the schedule is doing work.
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
                        lift.reset()
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
                        lift.reset()
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
                                if not latch.ready and l_active \
                                        and not lift.active:
                                    if latch.add(now, re_, pe):
                                        print(f"[level] setpoint latched: roll "
                                              f"{math.degrees(latch.sp_roll):+.2f} "
                                              f"pitch {math.degrees(latch.sp_pitch):+.2f} "
                                              "deg -- leveling ENGAGED, "
                                              "1-4 lift a leg.")
                    agree_deg = float("nan")
                    if math.isfinite(re_) and not math.isnan(r_a):
                        agree_deg = math.degrees(max(abs(re_ - r_a),
                                                     abs(pe - p_a)))
                    lift_report.add(now, dt_sweep, lift.airborne,
                                    err_roll, err_pitch, agree_deg, drift_mm)
                    # the deliverable: zEKF vs zFK with the foot up and the
                    # pose settled, then again once it has re-planted
                    if lift.at_goal and not lift_held_printed:
                        if lift_settle_t0 is None:
                            lift_settle_t0 = now
                        elif now - lift_settle_t0 >= LIFT_SETTLE_S:
                            for ln in height_compare_block(
                                    f"{lift.leg} HELD UP at "
                                    f"{lift.cur*1e3:.0f} mm, 3 planted",
                                    h_ekf, h_fk, 3, lift_base):
                                print(ln)
                            lift_held_printed = True
                    elif not lift.at_goal:
                        lift_settle_t0 = None
                    if not lift.active and lift_report._cur is not None:
                        line = lift_report.finish(now)
                        if line:
                            print(line)
                        for ln in height_compare_block(
                                "re-planted, all 4 down again",
                                h_ekf, h_fk, 4, lift_base):
                            print(ln)
                        lift_base = None

                    if log_t0 is not None:
                        enc_log.append((now_mono, *q, *contacts.astype(int),
                                        r_a, p_a))

                    if now - last_print >= 1.0 / base.STATUS_HZ:
                        last_print = now
                        if not (shared.est_ready and out is not None):
                            print(f"[hw] {seq.stage:11s} waiting for EKF  "
                                  f"Tmax={int(np.max(temps))}C", flush=True)
                        elif lift.active:
                            # focused while a foot is up: the two heights first
                            print(f"[lift] {lift.leg} {lift.cur*1e3:4.1f}mm "
                                  f"{'AIR ' if lift.airborne else 'down'}  "
                                  f"zEKF={h_ekf*1e3:7.1f} zFK={h_fk*1e3:7.1f} "
                                  f"diff={(h_ekf-h_fk)*1e3:+5.1f}mm  "
                                  f"tilt={math.degrees(err_roll):+5.2f}/"
                                  f"{math.degrees(err_pitch):+5.2f}deg",
                                  flush=True)
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
                            # zEKF/zFK are TRUNK-BOTTOM heights; abd= is the
                            # same pose at the FK trunk origin.
                            print(f"[hw] {seq.stage:11s} [{lflag:8s}] "
                                  f"zEKF={h_ekf*1e3:7.1f} zFK={h_fk*1e3:7.1f} "
                                  f"abd={(h_ekf+IMU_BELOW_TRUNK_ORIGIN_M)*1e3:7.1f} "
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
    for line in lift_report.summary():
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

# the plane plant, rotation helper and fake EKF all belong to the controller's
# test harness -- reuse them rather than keeping a second copy in sync
_C_from_rp = lv._C_from_rp
_PlanePlant = lv._PlanePlant
_FakeShared = lv._FakeShared


def _check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))
    if not ok:
        _FAIL.append(name)


def _test_lift():
    dt = 1.0 / CONTROL_HZ
    lm = LiftManager(lift_m=0.015)
    _check("lift starts inactive, all planted",
           not lm.active and all(lm.planted().values()))
    msg = lm.request("FL")
    _check("lift request starts a ramp", lm.active and "lifting FL" in msg)
    t, worst_rate, prev = 0.0, 0.0, 0.0
    flipped_at = None
    while t < 5.0:
        lm.advance(t)
        worst_rate = max(worst_rate, abs(lm.cur - prev) / dt)
        prev = lm.cur
        if flipped_at is None and not lm.planted()["FL"]:
            flipped_at = lm.cur
        t += dt
    _check("ramp reaches the commanded lift", abs(lm.cur - 0.015) < 1e-9)
    _check("ramp respects the lift slew", worst_rate <= LIFT_SLEW_M_S + 1e-9,
           f"{worst_rate*1e3:.2f} mm/s")
    _check("contact flips OFF just past the threshold",
           flipped_at is not None
           and CONTACT_OFF_M < flipped_at < CONTACT_OFF_M + 2 * LIFT_SLEW_M_S * dt,
           f"at {flipped_at*1e3:.2f} mm")
    _check("only the lifted leg is airborne",
           lm.planted() == {"FL": False, "FR": True, "RL": True, "RR": True})
    msg = lm.request("RR")
    _check("a second leg is refused while one is lifted",
           "refused" in msg and lm.leg == "FL")
    msg = lm.request("FL")
    _check("the same key lowers", "lowering" in msg and lm.goal == 0.0)
    while t < 10.0:
        lm.advance(t)
        t += dt
    _check("lowered leg re-plants and clears",
           not lm.active and all(lm.planted().values()))
    lm2 = LiftManager()
    lm2.request("RL")
    for k in range(100):
        lm2.advance(k * dt)
    msg = lm2.lower("EKF stale")
    _check("auto-lower reports its reason",
           msg is not None and "EKF stale" in msg and lm2.goal == 0.0)


def _test_closed_loop_3leg(tables, stand_height, anchors):
    """End to end: stand, latch, lift FL under a load-shift disturbance;
    leveling + height hold on the 3 planted legs; lower; re-plant."""
    dt = 1.0 / CONTROL_HZ
    mount = (math.radians(1.5), math.radians(-0.5))
    plant = _PlanePlant(anchors, mount=mount)
    lvl = LevelingLoop(anchors)
    hctrl = s2.HeightController()
    latch = SetpointLatch()
    lift = LiftManager(lift_m=0.015)
    # _PlanePlant.solve returns the abd axis above the FOOT SITES; the loop
    # works in floor-to-trunk-bottom, so convert once (see stage 2b).
    PLANT_TO_BOTTOM = FOOT_RADIUS_M - IMU_BELOW_TRUNK_ORIGIN_M
    target_h = stand_height + PLANT_TO_BOTTOM
    SAG = 0.010
    base_z = {leg: -stand_height for leg in LEGS}

    def measure(fz, planted):
        h, roll, pitch = plant.solve(fz, planted)
        return h + PLANT_TO_BOTTOM - SAG, roll + mount[0], pitch + mount[1]

    t = 0.0
    events = []
    max_err_during_lift = 0.0
    settled_err = float("nan")
    while t < 90.0:
        lift.advance(t)
        planted = lift.planted()
        fz = {leg: base_z[leg] - hctrl.offset + lvl.offsets[leg]
              + (lift.cur if leg == lift.leg else 0.0) for leg in LEGS}
        h, r, p = measure(fz, planted)
        # setpoint latch on the settled stand
        if not latch.ready and t > SETTLE_S:
            if latch.add(t, r, p):
                events.append("latched")
        engaged = latch.ready
        lvl.update(t, r, p, latch.sp_roll if engaged else 0.0,
                   latch.sp_pitch if engaged else 0.0, planted, engaged)
        hctrl.update(t, h, target_h, True)
        if engaged and lift.active:
            max_err_during_lift = max(
                max_err_during_lift,
                abs(r - latch.sp_roll), abs(p - latch.sp_pitch))
            if lift.goal != 0.0:        # settled reading just before lowering
                settled_err = max(abs(r - latch.sp_roll),
                                  abs(p - latch.sp_pitch))
        # script: lift FL at t=10 with a 1.5 deg load-shift disturbance,
        # lower at t=55, disturbance gone once planted again
        if "latched" in events and lift.leg is None and t < 40.0 \
                and "lifted" not in events:
            lift.request("FL")
            plant.disturb = (math.radians(1.0), math.radians(-1.1))
            events.append("lifted")
        if "lifted" in events and t > 55.0 and lift.goal != 0.0:
            lift.request("FL")          # toggle -> lower
        if "lifted" in events and not lift.active \
                and "replanted" not in events:
            plant.disturb = (0.0, 0.0)
            events.append("replanted")
        t += dt

    fz = {leg: base_z[leg] - hctrl.offset + lvl.offsets[leg] for leg in LEGS}
    h, r, p = measure(fz, {leg: True for leg in LEGS})
    _check("sequence ran: latch -> lift -> lower -> re-plant",
           events == ["latched", "lifted", "replanted"], str(events))
    # the step disturbance transits through in full before the integrator
    # winds it out -- the claim is bounded transient + settled recovery
    _check("3-leg transient stays bounded (no runaway)",
           max_err_during_lift < math.radians(2.5),
           f"max err {math.degrees(max_err_during_lift):.2f} deg")
    _check("leveling settles the 3-leg stance back to the setpoint",
           settled_err < math.radians(0.35),
           f"settled err {math.degrees(settled_err):.2f} deg")
    _check("attitude back at the setpoint after re-plant",
           abs(r - latch.sp_roll) < math.radians(0.3)
           and abs(p - latch.sp_pitch) < math.radians(0.3),
           f"{math.degrees(r-latch.sp_roll):+.3f}/"
           f"{math.degrees(p-latch.sp_pitch):+.3f} deg")
    _check("height loop recovered the sag through it all",
           abs(h - target_h) <= s2.HEIGHT_DEADBAND_M + 1e-6,
           f"{(h-target_h)*1e3:+.2f} mm")
    _check("leveling stayed inside its clamp (no saturation)",
           not lvl.saturated,
           f"max |off| {max(abs(v) for v in lvl.offsets.values())*1e3:.1f} mm")


def _test_compare_block():
    """The printed comparison is the deliverable -- check what it asserts."""
    base = (0.2103, 0.2098)
    lines = height_compare_block("x", 0.2111, 0.2099, 3, base)
    _check("compare block reports both heights and their gap",
           "zEKF(bottom)" in lines[1] and "zFK(bottom, 3 planted)" in lines[1]
           and "+1.2 mm" in lines[1], lines[1].strip())
    _check("compare block reports the change vs the all-4 baseline",
           "+0.7 mm" in lines[2], lines[2].strip())
    _check("compare block omits the baseline row when there is none",
           len(height_compare_block("x", 0.21, 0.21, 4)) == 2)
    nan = float("nan")
    _check("a non-finite baseline does not crash the block",
           len(height_compare_block("x", 0.21, 0.21, 3, (nan, nan))) == 2)


def _test_fake_contacts_semantics():
    """--fake-contacts must change ONLY what the EKF is told."""
    lm = LiftManager(lift_m=0.020)
    lm.request("FL")
    for k in range(int(4.0 * CONTROL_HZ)):
        lm.advance(k / CONTROL_HZ)
    honest = np.array([lm.planted()[leg] for leg in LEGS])
    faked = np.ones(4, dtype=bool)
    _check("honest schedule marks the lifted leg airborne",
           not honest[LEGS.index("FL")] and honest.sum() == 3)
    _check("faked schedule claims all four planted", bool(faked.all()))
    # zFK must follow the HONEST set either way -- that is what keeps the
    # comparison a real measurement rather than a self-fulfilling one
    q = np.zeros(N_JOINTS)
    tables = _shared_tables()
    for i, leg in enumerate(LEGS):
        q[3*i:3*i+3] = tables[leg].q_at(-0.19)
    q[0:3] = tables["FL"].q_at(-0.19 + lm.cur)
    h_honest = fk_floor_height_planted(q, None, honest, ref="hip")
    h_faked = fk_floor_height_planted(q, None, faked, ref="hip")
    _check("zFK on the honest set ignores the raised foot",
           abs(h_honest - fk_floor_height_planted(
               np.concatenate([tables[l].q_at(-0.19) for l in LEGS]),
               None, honest, ref="hip")) < 1e-9)
    _check("zFK would be biased by lift/4 if it used the faked set",
           abs((h_honest - h_faked) - lm.cur / 4) < 2e-4,
           f"{(h_honest-h_faked)*1e3:+.2f} mm vs {lm.cur/4*1e3:.2f} mm")


_TABLES = None


def _shared_tables(stand_height=STAND_HEIGHT_DEFAULT):
    global _TABLES
    if _TABLES is None:
        _TABLES = s2.build_tables(stand_height,
                                  clamp_m=s2.HEIGHT_CLAMP_M + LEVEL_CLAMP_M)
    return _TABLES


def self_test(stand_height):
    print("lift_ekf_contact_hw self-test (no hardware)")
    print("  (control laws are stand_ekf_level_hw's and are gated there)")
    print("[1] height tables -- room for the lift above the stand")
    t0 = time.perf_counter()
    tables = _shared_tables(stand_height)
    print(f"  (built in {time.perf_counter()-t0:.2f}s)")
    anchors = {leg: tables[leg].xy for leg in LEGS}
    _check("table span covers the full lift above the stand",
           all(tables[leg].z_max >= -stand_height + LIFT_M for leg in LEGS),
           "  ".join(f"{leg}[{tables[leg].z_min*1e3:.0f},"
                     f"{tables[leg].z_max*1e3:.0f}]" for leg in LEGS))
    print("[2] lift manager (ramp, contact threshold, one leg at a time)")
    _test_lift()
    print("[3] the zEKF/zFK comparison block")
    _test_compare_block()
    print("[4] --fake-contacts changes only what the EKF is told")
    _test_fake_contacts_semantics()
    print("[5] closed loop end to end, 3-leg stance")
    _test_closed_loop_3leg(tables, stand_height, anchors)
    print("[6] timing (leveling + lift + 4-leg lookup)")
    _test_timing(tables, anchors)
    print("self-test " + ("FAIL: " + ", ".join(_FAIL) if _FAIL else "PASS"))
    return 1 if _FAIL else 0


def _test_timing(tables, anchors):
    lvl = LevelingLoop(anchors)
    planted = {leg: True for leg in LEGS}
    lm = LiftManager()
    t0 = time.perf_counter()
    N = 200
    for k in range(N):
        lm.advance(k * 0.004)
        lvl.update(k * 0.004, 0.01, -0.01, 0.0, 0.0, planted, True)
        lz = lm.lift_z()
        for leg in LEGS:
            tables[leg].q_at(-0.19 + lvl.offsets[leg] + lz[leg])
    per = (time.perf_counter() - t0) / N
    _check("leveling + lift + 4-leg lookup fit the CAN slot", per < 250e-6,
           f"{per*1e6:.0f} us per sweep")


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
    ap.add_argument("--setpoint-roll", type=float, default=None,
                    help="fixed leveling setpoint roll (deg; default: latch)")
    ap.add_argument("--setpoint-pitch", type=float, default=None,
                    help="fixed leveling setpoint pitch (deg; default: latch)")
    ap.add_argument("--lift", type=float, default=LIFT_M,
                    help="lift height for the 1-4 keys (m)")
    ap.add_argument("--lean-abort", type=float, default=LEAN_ABORT_DEG,
                    help="attitude error that auto-lowers a lift (deg)")
    ap.add_argument("--tilt-stop", type=float, default=TILT_STOP_DEG,
                    help="absolute attitude that soft-stops the run (deg)")
    ap.add_argument("--fake-contacts", action="store_true",
                    help="A/B: tell the EKF all four feet stay planted even "
                         "while one is lifted. Physical motion and zFK are "
                         "unchanged; only the filter's contact input is wrong. "
                         "zEKF should then diverge from zFK by ~lift/4 -- run "
                         "this against a normal run to see what the contact "
                         "schedule buys.")
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
    if (args.setpoint_roll is None) != (args.setpoint_pitch is None):
        ap.error("--setpoint-roll and --setpoint-pitch go together")
    if args.self_test:
        sys.exit(self_test(args.stand_height))
    run(args.port, args.stand_height, args.target_height,
        args.crouch_max_speed_dps, args)


if __name__ == "__main__":
    main()
