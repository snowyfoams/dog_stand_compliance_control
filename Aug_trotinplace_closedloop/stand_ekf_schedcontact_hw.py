#!/usr/bin/env python3
"""Experiment B: stand_ekf_level_hw + an HONEST contact schedule (no tau).

WHY THIS EXISTS
    The recorded crouch leaves the REAR feet off the floor, but the level
    script tells the EKF all four are planted.  So `initialise()` anchors two
    footholds in mid-air, and when those legs extend and touch during STAND
    the filter reads the "anchored" feet moving as body motion: on 2026-08-12
    hardware the EKF's pitch was dragged ~4 deg away from the AHRS during the
    ramp (0.02 deg split at init -> 4.9 deg at HOLD4 entry, growing BEFORE the
    first CAN dropout), the --agree-veto then froze leveling for the whole
    stand, and no correction was ever commanded.

    This experiment feeds the estimator the contact truth it already knows
    how to use (`handle_transitions` re-anchors on rising edges, swing feet
    get Qp_swing and drop out of the measurement) -- from a fixed GAIT
    SCHEDULE, deliberately no torque sensing:

        CROUCH/WAIT_CROUCH   only --crouch-planted feet (default FL,FR);
                             the EKF initialises anchoring ONLY those
        STAND                the rest flip ON at --touch-frac of the ramp
        HOLD4                all four
        PARK                 the schedule runs backwards: the late feet flip
                             OFF at (1 - touch-frac) of the descent
        PARKED               back to the crouch set

    SUCCESS = |EKF-AHRS| stays inside --agree-veto through the whole ramp,
    the flag reaches LVL, and leveling actually runs.  Compare the same run
    on stand_ekf_level_hw (all-ones contacts): that one froze.

DIFF FROM THE BASELINE (stand_ekf_level_hw)
    run() is a copy of the baseline's with the [SCHED] blocks -- diff the two
    files to review.  Everything else (height loop, leveling law, setpoint,
    vetoes, safety, prints) is imported unchanged from the baseline.

HOW TO RUN
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V stand_ekf_schedcontact_hw.py --self-test
    sudo chrt -f 50 $V stand_ekf_schedcontact_hw.py
    # if the rear feet audibly touch earlier/later in the ramp, adjust:
    sudo chrt -f 50 $V stand_ekf_schedcontact_hw.py --touch-frac 0.2
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
from ekf_runtime import EkfShared, ekf_worker, _rp       # noqa: E402
from imu_dog import DEFAULT_PORT                         # noqa: E402

MOTOR_IDS = base.MOTOR_IDS
N_JOINTS = base.N_JOINTS
CONTROL_HZ = base.CONTROL_HZ
LEGS = base.LEGS

FOOT_RADIUS_M = s1.FOOT_RADIUS_M
IMU_BELOW_TRUNK_ORIGIN_M = s1.IMU_BELOW_TRUNK_ORIGIN_M
T_STAND = lv.T_STAND
EKF_WORKER_HZ = lv.EKF_WORKER_HZ
QUIET_STAGES = lv.QUIET_STAGES
SETTLE_S = lv.SETTLE_S
TEMP_NOTICE_C = lv.TEMP_NOTICE_C

TOUCH_FRAC_DEFAULT = 0.3     # rear feet flip ON this far into the STAND ramp
CROUCH_PLANTED_DEFAULT = ("FL", "FR")


class ContactSchedule:
    """Stage-driven contact mask -- the gait-schedule-only contact truth.

    No sensing: the late feet flip ON at a fixed fraction of the STAND ramp
    and OFF at the mirrored fraction of the PARK ramp.  If the fraction is
    wrong the touchdown re-anchor happens early/late by a fraction of the
    ramp; tune --touch-frac by ear (the feet are audible) or from the raw log.
    """

    def __init__(self, crouch_planted=CROUCH_PLANTED_DEFAULT,
                 touch_frac=TOUCH_FRAC_DEFAULT):
        unknown = [l for l in crouch_planted if l not in LEGS]
        if unknown:
            raise ValueError(f"unknown legs {unknown}; legs are {LEGS}")
        if len(crouch_planted) < 2:
            raise ValueError("need at least 2 crouch-planted feet "
                             "(height must stay observable)")
        if not 0.0 <= touch_frac <= 1.0:
            raise ValueError("touch-frac must be in [0, 1]")
        self.crouch_mask = np.array([l in crouch_planted for l in LEGS])
        self.touch_frac = float(touch_frac)

    def mask(self, stage, progress):
        """Boolean mask (FL, FR, RL, RR order = LEGS) for stage/progress."""
        if stage == "STAND":
            return (np.ones(4, bool) if progress >= self.touch_frac
                    else self.crouch_mask.copy())
        if stage == "HOLD4":
            return np.ones(4, bool)
        if stage == "PARK":
            return (self.crouch_mask.copy()
                    if progress >= 1.0 - self.touch_frac
                    else np.ones(4, bool))
        return self.crouch_mask.copy()    # CROUCH / WAIT_CROUCH / PARKED


def _progress(seq, now):
    """0..1 through the current STAND/PARK ramp; 0 in holding stages."""
    if seq.stage in ("STAND", "PARK") and seq.stage_t0 is not None:
        return float(np.clip((now - seq.stage_t0) / seq.t_stand, 0.0, 1.0))
    return 0.0


# ===========================================================================
# Hardware run -- stand_ekf_level_hw.run() with the [SCHED] blocks
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
    auth = min(-tables[leg].z_min - stand_height for leg in LEGS)
    need = 0.015 + args.level_clamp
    print(f"[init] extension authority below the stand: {auth*1e3:.0f} mm"
          + ("" if auth >= need else
             f"  !! less than sag+leveling ({need*1e3:.0f} mm) -- "
             "consider a lower --stand-height"))

    if target_height is None:
        target_height = (stand_height + FOOT_RADIUS_M
                         - IMU_BELOW_TRUNK_ORIGIN_M)
    unwrap = [base.CalibratedEncoderUnwrap() for _ in MOTOR_IDS]
    key = base.KeyPoller()
    gate = base.SafetyGate(tau_cap=base.STAGED_TAU_MAX)
    stats = s1.AttitudeStats()
    report = s1.StageReport()
    hctrl = s2.HeightController(gain_per_s=args.height_gain, clamp_m=args.clamp)
    lvl = lv.LevelingLoop(anchors_xy={leg: tables[leg].xy for leg in LEGS},
                          gain_per_s=args.level_gain,
                          clamp_m=args.level_clamp)
    latch = lv.SetpointLatch(
        math.radians(args.setpoint_roll) if args.setpoint_roll is not None else None,
        math.radians(args.setpoint_pitch) if args.setpoint_pitch is not None else None)
    settle_gate = lv.HeightSettleGate()
    agree_veto_rad = math.radians(args.agree_veto)
    tilt_stop_rad = math.radians(args.tilt_stop)
    # [SCHED] the planted set is NOT constant here -- it follows the schedule
    sched = ContactSchedule(crouch_planted=args.crouch_planted,
                            touch_frac=args.touch_frac)
    mask = sched.crouch_mask.copy()
    planted_a = mask.copy()
    planted_d = {leg: bool(mask[i]) for i, leg in enumerate(LEGS)}

    print("=" * 78)
    print("DOG5 LEVELING STAND + HONEST CONTACT SCHEDULE  (experiment B)")
    print(f"  contacts: crouch = {','.join(args.crouch_planted)} only; "
          f"the rest touch at {args.touch_frac:.0%} of the STAND ramp "
          f"(schedule, no sensing)")
    print(f"  leveling: gain {args.level_gain:.2f}/s, clamp "
          f"+/-{args.level_clamp*1e3:.0f} mm/foot, AHRS veto {args.agree_veto:.1f} deg"
          + ("  setpoint FIXED "
             f"({args.setpoint_roll:+.2f},{args.setpoint_pitch:+.2f}) deg"
             if latch.fixed else f"  setpoint auto-latched over {lv.LATCH_S:.0f}s"))
    print(f"  height:   gain {args.height_gain:.2f}/s, clamp "
          f"+/-{args.clamp*1e3:.0f} mm, target {target_height*1e3:.0f} mm "
          f"at the TRUNK BOTTOM/IMU, FK veto {args.xcheck*1e3:.0f} mm "
          "(planted feet only)")
    print(f"  tilt-stop {args.tilt_stop:.0f} deg.  CAN {args.control_hz:.0f} "
          f"Hz/motor ({1e3/args.control_hz:.0f} ms between commands, 50 ms "
          "driver watchdog)")
    print("  NO torque commanded; drivers hold position.")
    print("  Keys: ENTER stand, P park, X stop.")
    print("  Status line as in stand_ekf_level_hw, plus ct=planted mask "
          "(FL FR RL RR, 1=down, from the schedule).")
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
                raise RuntimeError("no raw IMU (0x40) packets -- enable "
                                   "DETA10 raw mode")

            start_q = base._zero_torque_preflight(mb, key, unwrap)
            if start_q is None:
                print("[abort] preflight not confirmed")
                return

            now = time.perf_counter()
            gate.start(now, start_q)
            shared = EkfShared(start_q)
            # [SCHED] the EKF must NEVER see the airborne crouch feet as
            # planted -- set the honest mask before the worker exists, so
            # initialise() anchors only the true footholds
            shared.contacts = mask.copy()
            worker = threading.Thread(
                target=ekf_worker, args=(shared, feed),
                kwargs=dict(quiet_stages=QUIET_STAGES,
                            control_hz=EKF_WORKER_HZ),
                daemon=True)
            worker.start()

            seq = lv.LevelStandSequence(tables, stand_height)
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

                    # [SCHED] contact mask for THIS sweep, edges printed
                    new_mask = sched.mask(seq.stage, _progress(seq, now))
                    if not np.array_equal(new_mask, mask):
                        for i, leg in enumerate(LEGS):
                            if new_mask[i] and not mask[i]:
                                print(f"[contact] {leg} DOWN (schedule, "
                                      f"{_progress(seq, now):.0%} of ramp) -- "
                                      "EKF re-anchors this foothold")
                            elif mask[i] and not new_mask[i]:
                                print(f"[contact] {leg} UP (schedule)")
                        mask = new_mask
                        planted_a = mask.copy()
                        planted_d = {leg: bool(mask[i])
                                     for i, leg in enumerate(LEGS)}

                    # ---- AHRS (leveling's independent watchdog) ----
                    ahrs_rp = None
                    r_a = p_a = float("nan")
                    att = feed.attitude()
                    if att is not None:
                        r_a = math.radians(att.roll_deg)
                        p_a = math.radians(att.pitch_deg)
                        ahrs_rp = (r_a, p_a)

                    # ---- leveling loop ----
                    roll_e, pitch_e, l_active, l_reason = lv.level_inputs(
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

                    # ---- height loop ([SCHED]: veto on the honest mask) ----
                    h_ekf, h_fk, h_active, h_reason = lv.height_inputs(
                        shared, q, z0_fk_imu, now_mono, planted_a,
                        xcheck_m=args.xcheck)
                    h_engaged = h_active and seq.loop_engaged
                    offset = hctrl.update(now, h_ekf, target_height, h_engaged,
                                          h_reason if not h_active else "not holding")
                    if seq.loop_engaged and h_reason and h_reason != last_h_reason:
                        print(f"[height] FROZEN: {h_reason} (holding "
                              f"{offset*1e3:+.1f} mm)")
                    last_h_reason = h_reason
                    h_settled = settle_gate.update(
                        now, h_engaged and args.height_gain != 0.0,
                        target_height - h_ekf)

                    # ---- compose per-leg targets ----
                    seq.extra_z = dict(lvl.offsets)

                    q_cmd, _seq_contacts, event = seq.update(
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
                    # [SCHED] the estimator gets the schedule's truth, not the
                    # sequence's all-ones
                    shared.contacts = mask.copy()

                    if event == "crouch_settled":
                        quiet_t0 = now
                        print("[stage] CROUCH settled; EKF initialising on "
                              f"{','.join(l for i, l in enumerate(LEGS) if mask[i])}"
                              " only. ENTER to STAND.")
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
                        print("[stage] STAND: open-loop ramp; late feet touch "
                              f"at {args.touch_frac:.0%}.")
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
                            enforce_position_limits=True)
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
                                                 last_recover,
                                                 next_fault_status)

                    # ---- observe / log / latch ----
                    out = shared.out
                    re_ = pe = float("nan")
                    r_fk, p_fk = lv.fk_attitude(q, planted_a)
                    if shared.est_ready and out is not None:
                        re_, pe = _rp(out["C"])
                        z_fk_imu = lv.fk_floor_height_planted(
                            q, out["C"], planted_a, ref="imu")
                        if z0_fk_imu is None:
                            z0_fk_imu = z_fk_imu
                        if z0_report is None:
                            z0_report = z_fk_imu
                        report.set_origin(z_fk_imu)
                        if quiet_t0 is not None and now - quiet_t0 >= SETTLE_S:
                            report.add(seq.stage, z_fk_imu,
                                       float(out["r"][2]), re_, pe)
                            if seq.stage == "HOLD4":
                                stats.add(re_, pe, r_a, p_a)
                                if not latch.ready and l_active and h_settled:
                                    if latch.add(now, re_, pe):
                                        print(f"[level] setpoint latched: roll "
                                              f"{math.degrees(latch.sp_roll):+.2f} "
                                              f"pitch {math.degrees(latch.sp_pitch):+.2f} "
                                              "deg -- leveling ENGAGED.")
                                elif not latch.ready and l_active \
                                        and not waiting_printed:
                                    waiting_printed = True
                                    print("[level] holding the setpoint until "
                                          "the height loop settles.")

                    if log_t0 is not None:
                        # [SCHED] the raw log records the honest mask
                        enc_log.append((now_mono, *q, *mask.astype(int),
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

                            def _deg(a):
                                return (f"{math.degrees(a):+6.2f}"
                                        if math.isfinite(a) else "    --")

                            lvl_s = ("  lvl=" + "/".join(
                                f"{lvl.offsets[l]*1e3:+4.1f}" for l in LEGS)
                                if args.level_gain != 0.0 else "")
                            err_s = (f"  err={_deg(err_roll)}/"
                                     f"{_deg(err_pitch)}"
                                     if args.level_gain != 0.0 else "")
                            t_max = int(np.max(temps))
                            hot = (f"  {t_max}C"
                                   if t_max >= TEMP_NOTICE_C else "")
                            # [SCHED] ct= the honest mask, 1=planted
                            ct = "".join("1" if m else "0" for m in mask)
                            print(f"[hw] {seq.stage:10s}[{lflag:8s}] "
                                  f"ct={ct} "
                                  f"h:ekf={h_ekf*1e3:6.1f} fk={h_fk*1e3:6.1f} "
                                  f"d={(h_ekf-h_fk)*1e3:+4.1f} "
                                  f"corr={offset*1e3:+5.1f}mm  "
                                  f"rp:ekf={_deg(re_)}/{_deg(pe)} "
                                  f"imu={_deg(r_a)}/{_deg(p_a)} "
                                  f"fk={_deg(r_fk)}/{_deg(p_fk)}deg"
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


def _test_schedule():
    s = ContactSchedule(("FL", "FR"), touch_frac=0.3)
    m = s.mask("CROUCH", 0.0)
    _check("crouch mask plants only FL,FR",
           list(m) == [True, True, False, False], str(m.astype(int)))
    _check("WAIT_CROUCH (EKF init) uses the same honest mask",
           np.array_equal(s.mask("WAIT_CROUCH", 0.0), m))
    _check("early STAND keeps the rear feet up",
           np.array_equal(s.mask("STAND", 0.29), m))
    _check("STAND flips the rear feet ON at touch-frac",
           bool(np.all(s.mask("STAND", 0.30))))
    _check("HOLD4 is all planted", bool(np.all(s.mask("HOLD4", 0.0))))
    _check("early PARK keeps all planted",
           bool(np.all(s.mask("PARK", 0.69))))
    _check("late PARK lifts the rear feet (mirror of the touchdown)",
           np.array_equal(s.mask("PARK", 0.71), m))
    _check("PARKED returns to the crouch set",
           np.array_equal(s.mask("PARKED", 0.0), m))

    # exactly ONE rising edge per stand: walk a full stage sequence
    edges = 0
    prev = s.mask("WAIT_CROUCH", 0.0)
    for stage, steps in (("STAND", 50), ("HOLD4", 20), ("PARK", 50),
                         ("PARKED", 5)):
        for k in range(steps):
            cur = s.mask(stage, (k + 1) / steps)
            if np.any(cur & ~prev):
                edges += 1
            prev = cur
    _check("exactly one touchdown edge per stand cycle", edges == 1,
           f"{edges} rising edges")

    try:
        ContactSchedule(("FL",))
        _check("fewer than 2 planted feet is refused", False)
    except ValueError:
        _check("fewer than 2 planted feet is refused", True)
    try:
        ContactSchedule(("FL", "XX"))
        _check("unknown leg names are refused", False)
    except ValueError:
        _check("unknown leg names are refused", True)


def _test_masked_height(stand_height):
    """The height veto must average only the schedule's planted feet."""
    tables = s2.build_tables(stand_height, clamp_m=lv.LEVEL_CLAMP_M)
    q = np.zeros(N_JOINTS)
    for i, leg in enumerate(LEGS):
        q[3 * i:3 * i + 3] = tables[leg].q_at(-stand_height)
    # rear feet 20 mm short of the floor (the crouch situation, exaggerated)
    for i, leg in enumerate(LEGS):
        if leg in ("RL", "RR"):
            q[3 * i:3 * i + 3] = tables[leg].q_at(-stand_height + 0.020)
    mask = ContactSchedule(("FL", "FR")).crouch_mask
    h_masked = lv.fk_floor_height_planted(q, None, mask, ref="imu")
    h_all = lv.fk_floor_height_planted(q, None, None, ref="imu")
    # a foot raised TOWARD the trunk pulls the all-four average DOWN by
    # lift * n_air / 4; the masked height must not move
    _check("masked FK height ignores the airborne rear feet",
           abs((h_masked - h_all) - 0.020 / 2) < 1e-4,
           f"all-four biased {(h_all-h_masked)*1e3:+.1f} mm (20 mm x 2/4 legs)")
    r_fk, p_fk = lv.fk_attitude(q, mask)
    _check("2-foot plane fit refuses an attitude (dashes, not garbage)",
           math.isnan(r_fk) and math.isnan(p_fk))
    return tables


def _test_sequence_wiring(tables, stand_height):
    """The mask follows the stage machine through a full stand cycle."""
    sched = ContactSchedule(("FL", "FR"), touch_frac=0.3)
    seq = lv.LevelStandSequence(tables, stand_height)
    dt = 1.0 / CONTROL_HZ
    t, q, qd = 0.0, recorded.Q_RECORDED_CROUCH.copy(), np.zeros(N_JOINTS)
    seen = {}
    while t < recorded.CROUCH_TIMEOUT_S + 3 * T_STAND and seq.stage != "PARKED":
        cmd, _, _ = seq.update(
            t, q, qd,
            enter_pressed=seq.stage == "WAIT_CROUCH",
            park_pressed=(seq.stage == "HOLD4" and t - seq.stage_t0 > 2.0),
            offset=0.0)
        q = cmd
        m = sched.mask(seq.stage, _progress(seq, t))
        seen.setdefault(seq.stage, []).append(int(m.sum()))
        t += dt
    _check("WAIT_CROUCH never reports 4 planted",
           max(seen.get("WAIT_CROUCH", [0])) == 2, str(set(seen["WAIT_CROUCH"])))
    _check("STAND transitions 2 -> 4 planted",
           seen["STAND"][0] == 2 and seen["STAND"][-1] == 4)
    _check("HOLD4 is 4 planted throughout",
           set(seen["HOLD4"]) == {4})
    _check("PARK transitions 4 -> 2 planted",
           seen["PARK"][0] == 4 and seen["PARK"][-1] == 2)


def self_test(stand_height):
    print("stand_ekf_schedcontact_hw self-test (no hardware)")
    print("[1] contact schedule")
    _test_schedule()
    print("[2] schedule-masked FK height / attitude")
    tables = _test_masked_height(stand_height)
    print("[3] schedule follows the stage machine")
    _test_sequence_wiring(tables, stand_height)
    print("self-test " + ("FAIL: " + ", ".join(_FAIL) if _FAIL else "PASS"))
    return 1 if _FAIL else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--stand-height", type=float,
                    default=lv.STAND_HEIGHT_DEFAULT)
    ap.add_argument("--target-height", type=float, default=None,
                    help="metres, floor to TRUNK BOTTOM / IMU board")
    ap.add_argument("--crouch-max-speed-dps", type=float,
                    default=recorded.DEFAULT_CROUCH_MAX_MOTOR_DPS)
    ap.add_argument("--height-gain", type=float, default=s2.HEIGHT_GAIN_PER_S)
    ap.add_argument("--clamp", type=float, default=s2.HEIGHT_CLAMP_M)
    ap.add_argument("--xcheck", type=float, default=s2.HEIGHT_XCHECK_M)
    ap.add_argument("--level-gain", type=float, default=lv.LEVEL_GAIN_PER_S)
    ap.add_argument("--level-clamp", type=float, default=lv.LEVEL_CLAMP_M)
    ap.add_argument("--agree-veto", type=float, default=lv.AGREE_VETO_DEG,
                    help="|EKF-AHRS| that freezes leveling (deg) -- the "
                         "number this experiment exists to keep small")
    ap.add_argument("--setpoint-roll", type=float, default=0.0)
    ap.add_argument("--setpoint-pitch", type=float, default=0.0)
    ap.add_argument("--setpoint-latch", action="store_true")
    ap.add_argument("--tilt-stop", type=float, default=lv.TILT_STOP_DEG)
    ap.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    # [SCHED] the two knobs of this experiment
    ap.add_argument("--crouch-planted", default=",".join(CROUCH_PLANTED_DEFAULT),
                    help="comma list of feet ON THE FLOOR at the crouch "
                         "(default FL,FR -- the rear feet hang)")
    ap.add_argument("--touch-frac", type=float, default=TOUCH_FRAC_DEFAULT,
                    help="fraction of the STAND ramp where the hanging feet "
                         "touch down (schedule only, no sensing)")
    ap.add_argument("--raw-log", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    args.crouch_planted = tuple(s.strip().upper()
                                for s in args.crouch_planted.split(",") if s)
    if args.setpoint_latch:
        args.setpoint_roll = args.setpoint_pitch = None
    if not 20.0 <= args.control_hz <= 300.0:
        ap.error("--control-hz outside [20, 300] (bus ceiling is 300)")
    if args.self_test:
        sys.exit(self_test(args.stand_height))
    run(args.port, args.stand_height, args.target_height,
        args.crouch_max_speed_dps, args)


if __name__ == "__main__":
    main()
