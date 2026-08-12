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

HOW TO RUN
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V stand_ahrs_level_hw.py --self-test          # offline, no hardware
    sudo chrt -f 50 $V stand_ahrs_level_hw.py --level-gain 0   # A/B baseline
    sudo chrt -f 50 $V stand_ahrs_level_hw.py                  # leveling live

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

MOTOR_IDS = base.MOTOR_IDS
N_JOINTS = base.N_JOINTS
CONTROL_HZ = base.CONTROL_HZ
LEGS = base.LEGS

STAND_HEIGHT_DEFAULT = lv.STAND_HEIGHT_DEFAULT
TEMP_NOTICE_C = lv.TEMP_NOTICE_C
TILT_STOP_DEG = lv.TILT_STOP_DEG
AHRS_STALE_S = 0.2           # no fresh 0x41 packet this long -> freeze
SETTLE_S = lv.SETTLE_S


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
    lvl = lv.LevelingLoop(anchors_xy={leg: tables[leg].xy for leg in LEGS},
                          gain_per_s=args.level_gain, clamp_m=args.level_clamp)
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
    print(f"  leveling: gain {args.level_gain:.2f}/s, clamp "
          f"+/-{args.level_clamp*1e3:.0f} mm/foot (= the legs' reach); "
          f"height OPEN LOOP (stand {stand_height*1e3:.0f} mm)")
    print(f"  authority: pitch +/-"
          f"{math.degrees(math.atan2(2*args.level_clamp, _wb)):.1f} deg, "
          f"roll +/-"
          f"{math.degrees(math.atan2(2*args.level_clamp, _track)):.1f} deg "
          "-- errors beyond that SATURATE (status line shows SAT)")
    print(f"  tilt-stop {args.tilt_stop:.0f} deg.  CAN {args.control_hz:.0f} "
          f"Hz/motor ({1e3/args.control_hz:.0f} ms between commands, 50 ms "
          "driver watchdog)")
    print("  NO torque commanded; drivers hold position. Contacts irrelevant: "
          "no estimator.")
    print("  Keys: ENTER stand, P park, X stop.")
    if args.level_gain == 0.0:
        print("  !! gain 0 -- OPEN LOOP A/B: watch rp:imu, nothing will move")
    print("=" * 78)

    stop_reason = None
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

            now = time.perf_counter()
            gate.start(now, start_q)
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
            quiet_t0 = None
            last_reason = None

            while True:
                mb.poll()
                joint_index = index % N_JOINTS
                if joint_index == 0:
                    now = time.perf_counter()
                    q, qd = base._joint_state(mb, unwrap)

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
                            if args.level_gain != 0.0 else "")
                        t_max = int(np.max(temps))
                        hot = f"  {t_max}C" if t_max >= TEMP_NOTICE_C else ""
                        warn = ""
                        if latched_err:
                            warn += f" latched={len(latched_err)}"
                        if tx_fail:
                            warn += f" txfail={tx_fail}"
                        print(f"[hw] {seq.stage:10s}[{flag:4s}] "
                              f"rp:imu={_deg(roll_a)}/{_deg(pitch_a)} "
                              f"fk={_deg(r_fk)}/{_deg(p_fk)} "
                              f"err={_deg(err_r)}/{_deg(err_p)}deg"
                              f"{lvl_s}{hot}{warn}", flush=True)

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


class _FakeAhrs:
    """AhrsSource stand-in: scripted attitude + staleness."""

    def __init__(self):
        self.roll = 0.0
        self.pitch = 0.0
        self.stale = False
        self.none = False

    def read(self):
        if self.none:
            return float("nan"), float("nan"), False, "no AHRS yet"
        if self.stale:
            return self.roll, self.pitch, False, "AHRS stale"
        return self.roll, self.pitch, True, None


def _test_closed_loop(tables, stand_height, anchors):
    """Plane plant + AHRS feedback + setpoint (0,0) -> physical level.

    The AHRS is modelled as TRUTH (mount calibrated out, this experiment's
    premise), so driving it to zero must level the physical trunk exactly.
    """
    dt = 1.0 / CONTROL_HZ
    plant = lv._PlanePlant(anchors)              # mount = (0,0): AHRS = truth
    plant.disturb = (math.radians(2.0), math.radians(-1.5))
    loop = lv.LevelingLoop(anchors)
    all_on = {leg: True for leg in LEGS}
    base_z = {leg: -stand_height for leg in LEGS}
    t = 0.0
    while t < 60.0:
        fz = {leg: base_z[leg] + loop.offsets[leg] for leg in LEGS}
        roll, pitch = plant.ekf_attitude(fz, all_on)
        loop.update(t, roll, pitch, 0.0, 0.0, all_on, True)
        t += dt
    fz = {leg: base_z[leg] + loop.offsets[leg] for leg in LEGS}
    roll, pitch = plant.ekf_attitude(fz, all_on)
    _check("AHRS feedback levels a 2.0/-1.5 deg disturbance",
           abs(roll) < math.radians(0.3) and abs(pitch) < math.radians(0.3),
           f"residual {math.degrees(roll):+.3f}/{math.degrees(pitch):+.3f} deg")
    _check("offsets stayed inside the clamp", not loop.saturated,
           f"max {max(abs(v) for v in loop.offsets.values())*1e3:.1f} mm")

    # stale AHRS must freeze, not unwind
    held = dict(loop.offsets)
    src = _FakeAhrs()
    src.stale = True
    for k in range(int(2 / dt)):
        r, p, active, why = src.read()
        loop.update(t + k * dt, r, p, 0.0, 0.0, all_on,
                    active and True, why)
    _check("stale AHRS freezes the offsets (no unwind)",
           all(abs(loop.offsets[l] - held[l]) < 1e-12 for l in LEGS)
           and loop.frozen_reason == "AHRS stale")


def _test_sequence(tables, stand_height):
    seq = lv.LevelStandSequence(tables, stand_height)
    dt = 1.0 / CONTROL_HZ
    t, q, qd = 0.0, recorded.Q_RECORDED_CROUCH.copy(), np.zeros(N_JOINTS)
    seen = []
    while t < recorded.CROUCH_TIMEOUT_S + 3 * lv.T_STAND \
            and seq.stage != "PARKED":
        seq.extra_z = {leg: 0.005 for leg in LEGS}
        cmd, _, _ = seq.update(
            t, q, qd,
            enter_pressed=seq.stage == "WAIT_CROUCH",
            park_pressed=(seq.stage == "HOLD4" and t - seq.stage_t0 > 2.0),
            offset=0.0)
        q = cmd
        if seq.stage not in seen:
            seen.append(seq.stage)
        t += dt
    _check("stage order incl. park",
           seen == ["CROUCH", "WAIT_CROUCH", "STAND", "HOLD4", "PARK",
                    "PARKED"], " -> ".join(seen))
    _check("PARKED lands on the recorded crouch despite leveling extra",
           float(np.max(np.abs(seq.q_cmd(t) - recorded.Q_RECORDED_CROUCH)))
           < 1e-5)


def _test_timing(tables, anchors):
    loop = lv.LevelingLoop(anchors)
    planted = {leg: True for leg in LEGS}
    t0 = time.perf_counter()
    N = 200
    for k in range(N):
        loop.update(k * 0.004, 0.01, -0.01, 0.0, 0.0, planted, True)
        for leg in LEGS:
            tables[leg].q_at(-0.19 + loop.offsets[leg])
    per = (time.perf_counter() - t0) / N
    _check("leveling + 4-leg lookup fit the CAN slot", per < 250e-6,
           f"{per*1e6:.0f} us per sweep")


def self_test(stand_height):
    print("stand_ahrs_level_hw self-test (no hardware)")
    print("[1] tables + physical-reach clamp")
    t0 = time.perf_counter()
    tables = s2.build_tables(stand_height, clamp_m=0.080)
    anchors = {leg: tables[leg].xy for leg in LEGS}
    print(f"  (built in {time.perf_counter()-t0:.2f}s)")
    clamp = auto_clamp(tables, stand_height)
    ext = min(-tables[leg].z_min - stand_height for leg in LEGS)
    _check("auto clamp = extension headroom minus the margin",
           abs(clamp - (ext - 0.002)) < 1e-9,
           f"{clamp*1e3:.0f} mm of {ext*1e3:.0f} mm reach")
    _check("the 80 mm table request was capped by the legs, not granted",
           ext < 0.079, f"physical reach ceiling {ext*1e3:.0f} mm below stand")
    wb = abs(anchors["FL"][0] - anchors["RL"][0])
    _check("pitch authority at this stand exceeds the old 12 mm clamp's 2.0deg",
           math.degrees(math.atan2(2 * clamp, wb)) > 2.5,
           f"{math.degrees(math.atan2(2*clamp, wb)):.1f} deg")
    print("[2] AHRS-fed closed loop on the plane plant")
    _test_closed_loop(tables, stand_height, anchors)
    print("[3] stage machine")
    _test_sequence(tables, stand_height)
    print("[4] timing")
    _test_timing(tables, anchors)
    print("self-test " + ("FAIL: " + ", ".join(_FAIL) if _FAIL else "PASS"))
    return 1 if _FAIL else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--stand-height", type=float, default=STAND_HEIGHT_DEFAULT)
    ap.add_argument("--crouch-max-speed-dps", type=float,
                    default=recorded.DEFAULT_CROUCH_MAX_MOTOR_DPS)
    ap.add_argument("--level-gain", type=float, default=lv.LEVEL_GAIN_PER_S,
                    help="leveling integral gain (1/s); 0 = open-loop A/B")
    ap.add_argument("--level-clamp", type=float, default=None,
                    help="max |offset| per foot (m); default: the legs' full "
                         "physical reach below the stand (no artificial cap)")
    ap.add_argument("--setpoint-roll", type=float, default=0.0,
                    help="deg; AHRS zero is flat-calibrated, so 0 = level")
    ap.add_argument("--setpoint-pitch", type=float, default=0.0)
    ap.add_argument("--stale", type=float, default=AHRS_STALE_S,
                    help="AHRS age that freezes leveling (s)")
    ap.add_argument("--engage-delay", type=float, default=SETTLE_S,
                    help="seconds to hold still in HOLD4 before the loop "
                         "engages (the stand transient is not an error)")
    ap.add_argument("--tilt-stop", type=float, default=TILT_STOP_DEG)
    ap.add_argument("--control-hz", type=float, default=CONTROL_HZ,
                    help="per-motor command rate (Hz); 50 ms driver watchdog")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if not 20.0 <= args.control_hz <= 300.0:
        ap.error("--control-hz outside [20, 300] (bus ceiling is 300)")
    if args.self_test:
        sys.exit(self_test(args.stand_height))
    run(args.port, args.stand_height, args.crouch_max_speed_dps, args)


if __name__ == "__main__":
    main()
