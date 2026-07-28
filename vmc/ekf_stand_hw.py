#!/usr/bin/env python3
"""Position-mode 4-leg STAND with the state estimator running READ-ONLY.

Purpose: the main task is validating the EKF on hardware. A rigid *position*
stand (motor position loops hold each joint stiffly -- statically stable, no
balance control needed) gets the dog standing robustly while the EKF runs in a
worker thread and only observes/logs. NO torque is ever commanded by us.

Flow (references stand_dog5_inplace_hw / stand3_hold_hw's position stand):
    ZERO-TORQUE check
      -> CROUCH        native 0xA4 position to the recorded crouch
      -> WAIT_CROUCH   hold; EKF initialises on the static crouch; ENTER stands
      -> STAND         joint-interp crouch -> precomputed stand pose (feet rise
                       ~straight up), streamed as position; contacts=OFF (feet
                       drag a little, so the EKF dead-reckons through the rise)
      -> HOLD4         hold the stand pose; contacts=ON re-anchor -> clean
                       static 4-leg stand for EKF validation. Hand-rock here to
                       exercise attitude/bias. X stops.

Q_STAND is precomputed with incremental IK (stays on the correct branch, unlike
a one-shot solve). The stand streams a coordinated joint interpolation, so the
CAN loop needs no per-sweep IK and stays light (the EKF is off-thread).

Run (project venv; supported robot, feet on the ground):
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    sudo chrt -f 50 $V ekf_stand_hw.py --raw-log stand.npz
Then replay for the gate-C8 report:
    python3 ../state_estimator/hw_replay.py stand.npz --static
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import threading
import time

import numpy as np

sys.setswitchinterval(0.0005)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DESC = os.path.join(os.path.dirname(_HERE), "dog5_description")
_EST = os.path.join(os.path.dirname(_HERE), "state_estimator")
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _DESC, _EST, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import motorbus                                          # noqa: E402
import stand_dog5_hw as base                             # noqa: E402 (low-level lib)
import stand_dog5_recorded_hw as recorded                # noqa: E402 (recorded crouch)
import dog5_kinematics                                   # noqa: E402
from dog5_state_estimator import DOG5StateEstimator, quat_to_C  # noqa: E402
from imu_ekf_feed import ImuEkfFeed                      # noqa: E402
from imu_dog import DEFAULT_PORT                         # noqa: E402

MOTOR_IDS = base.MOTOR_IDS
N_JOINTS = base.N_JOINTS
CONTROL_HZ = base.CONTROL_HZ
LEGS = base.LEGS
Q_CROUCH = recorded.Q_RECORDED_CROUCH
POSITION_TARGET_DEG = recorded.POSITION_TARGET_DEG       # crouch, controller-order deg

T_STAND = 5.0                                            # crouch->stand ramp (s)
CONTROL_UPDATE_HZ = 100.0
N_INIT_SAMPLES = 30
DT_CLAMP = (1.0e-4, 0.05)


def _smoothstep(u):
    u = float(np.clip(u, 0.0, 1.0))
    return 3 * u * u - 2 * u * u * u


def compute_q_stand(stand_height):
    """Stand joint pose: feet at the crouch xy, lowered to -stand_height, via
    INCREMENTAL warm-started IK (stays on the crouch branch)."""
    q_stand = Q_CROUCH.copy()
    for i, leg in enumerate(LEGS):
        sl = slice(3 * i, 3 * i + 3)
        q = Q_CROUCH[sl].copy()
        f0 = dog5_kinematics.foot_position(leg, q)
        for n in range(1, 31):
            z = f0[2] + (n / 30.0) * (-stand_height - f0[2])
            tgt = np.array([f0[0], f0[1], z])
            for _ in range(60):
                e = tgt - dog5_kinematics.foot_position(leg, q)
                if np.linalg.norm(e) < 1e-10:
                    break
                J = dog5_kinematics.foot_jacobian(leg, q)
                q = q + 0.5 * J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e)
        q_stand[sl] = q
    lo, hi = base.soft_limits()
    if np.any(q_stand < lo) or np.any(q_stand > hi):
        raise RuntimeError("computed stand pose exceeds soft joint limits")
    return q_stand


class _Shared:
    def __init__(self, start_q):
        self.q = start_q.copy()
        self.qd = np.zeros(N_JOINTS)
        self.stage = "CROUCH"
        self.contacts = np.ones(4, dtype=bool)
        self.out = None
        self.bias_str = ""
        self.est_ready = False
        self.run = True
        self.imu_log = []     # (t, fx,fy,fz, wx,wy,wz)
        self.tau_stamp = 0.0


def _worker(shared, feed):
    """Read-only EKF: drain IMU, predict/update, publish outputs, log IMU."""
    est = DOG5StateEstimator()
    init_f, init_w = [], []
    last_imu_t = None
    w_last = np.zeros(3)
    control_dt = 1.0 / CONTROL_UPDATE_HZ
    while shared.run:
        t = time.perf_counter()
        stage = shared.stage
        if not shared.est_ready:
            for f, w, tm in feed.drain():
                init_f.append(f)
                init_w.append(w)
                shared.imu_log.append((tm, *f, *w))
            if stage in ("WAIT_CROUCH", "STAND", "HOLD4") and len(init_f) >= N_INIT_SAMPLES:
                est.initialise(np.array(init_f), np.array(init_w),
                               shared.q.reshape(4, 3), shared.contacts)
                last_imu_t = time.monotonic()
                shared.bias_str = (f"bw={np.round(est.state.bw,5)} "
                                   f"bf={np.round(est.state.bf,4)}")
                shared.est_ready = True
            else:
                time.sleep(0.01)
            continue
        q = shared.q
        contacts = shared.contacts
        samples = feed.drain()
        if samples:
            fmean = np.mean([s[0] for s in samples], axis=0)
            wmean = np.mean([s[1] for s in samples], axis=0)
            tm = samples[-1][2]
            dt = min(max(tm - last_imu_t, DT_CLAMP[0]), DT_CLAMP[1])
            est.predict(fmean, wmean, dt, contacts)
            last_imu_t, w_last = tm, wmean
            for f, w, ts in samples:
                shared.imu_log.append((ts, *f, *w))
        est.update(q.reshape(4, 3), contacts)
        out = est.outputs(last_w_meas=w_last)
        out["C"] = quat_to_C(out["q"])
        shared.out = out
        shared.tau_stamp = time.monotonic()
        rem = control_dt - (time.perf_counter() - t)
        if rem > 0:
            time.sleep(rem)


def run(port, stand_height, crouch_max_speed_dps, log_path=None, raw_log_path=None):
    base.validate_hardware_config()
    q_stand = compute_q_stand(stand_height)
    stand_deg = np.rad2deg(q_stand)                      # controller-order deg

    unwrap = [base.CalibratedEncoderUnwrap() for _ in MOTOR_IDS]
    key = base.KeyPoller()
    gate = base.SafetyGate(tau_cap=base.STAGED_TAU_MAX)  # used only for estop tests

    print("=" * 74)
    print("DOG5 POSITION STAND + read-only EKF (references stand3_hold_hw).")
    print(f"  stand height = {stand_height:.3f} m ; crouch->stand streamed over {T_STAND:.0f}s")
    print("  NO torque commanded by us; motors hold position. Feet on the ground.")
    print("=" * 74)

    log_file = None
    writer = None
    if log_path:
        log_file = open(log_path, "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["t", "stage", "rz", "vx", "vy", "vz", "roll_est", "pitch_est",
                         "roll_ahrs", "pitch_ahrs", "healthy"])
    enc_log = []       # (t_mono, alpha12, contacts4, roll_ahrs, pitch_ahrs)

    stop_reason = None
    armed = False
    worker = None
    shared = None
    try:
        with motorbus.MotorBus(MOTOR_IDS, dirs=base.MOTOR_DIRECTIONS) as mb, \
                ImuEkfFeed(port) as feed:
            if not mb.arm(rate_hz=CONTROL_HZ):
                raise RuntimeError("arm failed (bus/power/terminators)")
            armed = True
            if not feed.wait_for_raw(timeout=3.0):
                raise RuntimeError("no raw IMU (0x40) packets -- enable DETA10 raw mode")

            start_q = base._zero_torque_preflight(mb, key, unwrap)
            if start_q is None:
                print("[abort] preflight not confirmed")
                return

            now = time.perf_counter()
            gate.start(now, start_q)
            shared = _Shared(start_q)
            worker = threading.Thread(target=_worker, args=(shared, feed), daemon=True)
            worker.start()

            stage = "CROUCH"
            stage_t0 = now
            crouch_settle_since = None
            miss_monitor = base.CanMissMonitor(mb)
            slot = mb.slot(CONTROL_HZ)
            deadline = time.perf_counter() + slot
            status_period = 1.0 / base.FAULT_STATUS_HZ
            next_fault_status = np.array(
                [status_period + i * status_period / N_JOINTS for i in range(N_JOINTS)])
            last_recover = {mid: -base.RECOVER_PERIOD_S for mid in MOTOR_IDS}
            start = now
            cmd_deg = POSITION_TARGET_DEG.copy()
            index = 0
            last_print = 0.0
            tx_fail = 0
            bias_printed = False

            while True:
                mb.poll()
                joint_index = index % N_JOINTS
                if joint_index == 0:
                    now = time.perf_counter()
                    q, qd = base._joint_state(mb, unwrap)
                    shared.q = q
                    shared.qd = qd
                    shared.stage = stage
                    # contacts: planted except during the moving STAND ramp
                    shared.contacts = np.zeros(4, bool) if stage == "STAND" else np.ones(4, bool)

                    if not bias_printed and shared.est_ready:
                        print(f"[ekf] init: {shared.bias_str}")
                        bias_printed = True

                    # ---- stage machine (all POSITION mode) ----
                    if stage == "CROUCH":
                        cmd_deg = POSITION_TARGET_DEG
                        pose_err = float(np.max(np.abs(q - Q_CROUCH)))
                        settled = (pose_err <= recorded.RECORDED_POSE_TOL
                                   and float(np.max(np.abs(qd))) <= recorded.RECORDED_QD_TOL)
                        if not settled:
                            crouch_settle_since = None
                        elif crouch_settle_since is None:
                            crouch_settle_since = now
                        elif now - crouch_settle_since >= recorded.CROUCH_SETTLE_S:
                            stage, stage_t0 = "WAIT_CROUCH", now
                            print("[stage] CROUCH settled; EKF initialising. ENTER to STAND.")
                        elif now - stage_t0 > recorded.CROUCH_TIMEOUT_S:
                            stop_reason = "CROUCH timeout"
                    elif stage == "WAIT_CROUCH":
                        cmd_deg = POSITION_TARGET_DEG
                    elif stage == "STAND":
                        s = _smoothstep((now - stage_t0) / T_STAND)
                        cmd_deg = np.rad2deg(Q_CROUCH + s * (q_stand - Q_CROUCH))
                        if now - stage_t0 >= T_STAND:
                            stage, stage_t0 = "HOLD4", now
                            print("[stage] STAND complete -> HOLD4 (static 4-leg stand). "
                                  "Hand-rock to test the EKF. X stops.")
                    else:  # HOLD4
                        cmd_deg = stand_deg

                    # ---- safety (position mode: limits/temp/comms/overspeed) ----
                    temps = base._temperatures(mb)
                    misses = miss_monitor.update(mb)
                    errors = mb.errors()
                    if stop_reason is None:
                        stop_reason = gate.estop_reason(q, qd, temps, misses, errors, now,
                                                        enforce_position_limits=True)
                    if stop_reason is None:
                        stop_reason = gate.overspeed_reason(qd, q, now)
                    latched = [mid for mid, e in errors.items() if e and (e & 0x80)]

                    pressed = key.get()
                    if pressed in ("x", "X"):
                        stop_reason = "operator X"
                    elif pressed in ("\r", "\n") and stage == "WAIT_CROUCH":
                        stage, stage_t0 = "STAND", now
                        print("[stage] STAND: streaming crouch -> stand (feet ~straight up).")
                    if stop_reason:
                        break

                    recover = [mid for mid in latched
                               if (now - start) - last_recover[mid] >= base.RECOVER_PERIOD_S]
                    if recover:
                        base._recover_input_lost(mb, recover, now - start,
                                                 last_recover, next_fault_status)

                    # ---- logging ----
                    out = shared.out
                    r_a = p_a = float("nan")
                    att = feed.attitude()
                    if att is not None:
                        r_a, p_a = math.radians(att.roll_deg), math.radians(att.pitch_deg)
                    enc_log.append((time.monotonic(), *q, *shared.contacts.astype(int), r_a, p_a))
                    if writer and shared.est_ready and out is not None:
                        re_, pe = _rp(out["C"])
                        writer.writerow([f"{now-start:.4f}", stage, f"{out['r'][2]:.5f}",
                                         *[f"{x:.5f}" for x in out["v"]], f"{re_:.5f}",
                                         f"{pe:.5f}", f"{r_a:.5f}", f"{p_a:.5f}",
                                         int(out["healthy"])])

                    if now - last_print >= 1.0 / base.STATUS_HZ:
                        last_print = now
                        extra = ""
                        if shared.est_ready and out is not None:
                            re_, pe = _rp(out["C"])
                            extra = (f" EKF z={out['r'][2]*1e3:+5.0f}mm "
                                     f"roll={math.degrees(re_):+5.1f}(ahrs{math.degrees(r_a):+5.1f}) "
                                     f"pitch={math.degrees(pe):+5.1f}(ahrs{math.degrees(p_a):+5.1f}) "
                                     f"|v|={np.linalg.norm(out['v'])*1e3:4.0f}mm/s "
                                     f"H={'OK' if out['healthy'] else '!!'}")
                        print(f"[hw] {stage:11s} max|qd|={np.max(np.abs(qd)):.2f} "
                              f"Tmax={int(np.max(temps))}C latched={len(latched)} "
                              f"txfail={tx_fail}{extra}", flush=True)

                # ---- per-slot POSITION command dispatch ----
                mid = MOTOR_IDS[joint_index]
                if (time.perf_counter() - start) >= next_fault_status[joint_index]:
                    mb.status1_req(mid)
                    next_fault_status[joint_index] += status_period
                else:
                    if not mb.position(mid, float(cmd_deg[joint_index]), crouch_max_speed_dps):
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
        if log_file:
            log_file.close()
            print(f"[log] wrote {log_path}")
        if raw_log_path and shared is not None:
            imu = np.array(shared.imu_log, dtype=float)
            enc = np.array(enc_log, dtype=float)
            if len(imu) and len(enc):
                np.savez(raw_log_path,
                         imu_t=imu[:, 0], imu_f=imu[:, 1:4], imu_w=imu[:, 4:7],
                         enc_t=enc[:, 0], enc_alpha=enc[:, 1:13],
                         enc_contacts=enc[:, 13:17], ahrs_rp=enc[:, 17:19],
                         init_secs=1.0)
                print(f"[raw] wrote {raw_log_path} ({len(imu)} IMU, {len(enc)} enc frames)")
    print(f"[stop] {stop_reason}")


def _rp(C):
    g = C @ np.array([0.0, 0.0, 1.0])
    return math.atan2(g[1], g[2]), math.atan2(-g[0], math.hypot(g[1], g[2]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--stand-height", type=float, default=recorded.DEFAULT_STAND_HEIGHT)
    ap.add_argument("--crouch-max-speed-dps", type=float,
                    default=recorded.DEFAULT_CROUCH_MAX_MOTOR_DPS)
    ap.add_argument("--log", default=None, help="CSV of EKF outputs")
    ap.add_argument("--raw-log", default=None, help="NPZ raw IMU+enc for hw_replay (gate C8)")
    args = ap.parse_args()
    run(args.port, args.stand_height, args.crouch_max_speed_dps, args.log, args.raw_log)


if __name__ == "__main__":
    main()
