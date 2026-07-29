"""Reusable read-only EKF worker for hardware runners.

Transplant of the hardware-validated `_Shared`/`_worker` pattern from
`vmc/ekf_stand_hw.py` (2026-07-28 gate-C8 stand run) so gait runners can host
the estimator without copying it.  The control loop owns the CAN sweep and
writes `shared.q/qd/stage/contacts`; the worker owns the estimator and writes
`shared.out/bias_str/est_ready/extra/imu_log`.  Single-writer-per-field, no
locks -- same discipline as the validated stand.

Differences from the ekf_stand_hw original (deliberate):
  * init samples are accumulated ONLY while `stage` is in `quiet_stages`
    (the original also collected during the moving crouch, polluting the
    gyro-bias init);
  * `shared.imu_log` grows only while `shared.log_enabled` (so a raw log
    starts at a clean static prefix -- hw_replay's init assumes one);
  * optional `verbose` publishes per-leg innovation/foothold diagnostics
    into `shared.extra` (same peek pattern as hw_replay.py; the repeated
    handle_transitions inside update() is idempotent).

The caller starts the thread and MUST set `shared.run = False` + join on
shutdown.  Memory note: `imu_log` holds ~100-200 rows/s of 7-float tuples
(~40-80 MB/half hour) -- fine for a session, chunk to arrays if runs ever
exceed ~1 h.
"""
from __future__ import annotations

import math
import time

import numpy as np

from dog5_state_estimator import DOG5StateEstimator, quat_to_C, LEGS

DEFAULT_CONTROL_UPDATE_HZ = 100.0
DEFAULT_N_INIT_SAMPLES = 30
DEFAULT_DT_CLAMP = (1.0e-4, 0.05)


class EkfShared:
    """Mailbox between a CAN control loop and the EKF worker thread."""

    def __init__(self, start_q):
        self.q = np.asarray(start_q, dtype=float).copy()
        self.qd = np.zeros(len(self.q))
        self.stage = "CROUCH"
        self.contacts = np.ones(4, dtype=bool)
        self.out = None               # latest est.outputs() (+ "C"), worker-written
        self.bias_str = ""
        self.est_ready = False
        self.run = True
        self.imu_log = []             # (t, fx,fy,fz, wx,wy,wz); gated by log_enabled
        self.log_enabled = False
        self.extra = None             # verbose per-leg diagnostics dict
        self.tau_stamp = 0.0
        # Runner-published display state (stage, step, blk_ms, ahrs ...).
        # REBIND a fresh dict, never mutate in place -- readers (ekf_web's
        # sampler) take it without a lock and must never see a partial write.
        self.status = {}


def ekf_worker(shared, feed, *, quiet_stages,
               control_hz=DEFAULT_CONTROL_UPDATE_HZ,
               n_init=DEFAULT_N_INIT_SAMPLES,
               dt_clamp=DEFAULT_DT_CLAMP,
               verbose=False):
    """Read-only EKF loop: drain IMU, predict/update, publish outputs."""
    est = DOG5StateEstimator()
    init_f, init_w = [], []
    last_imu_t = None
    w_last = np.zeros(3)
    control_dt = 1.0 / control_hz
    while shared.run:
        t = time.perf_counter()
        stage = shared.stage
        if not shared.est_ready:
            samples = feed.drain()
            if stage in quiet_stages:
                for f, w, tm in samples:
                    init_f.append(f)
                    init_w.append(w)
                    if shared.log_enabled:
                        shared.imu_log.append((tm, *f, *w))
            if stage in quiet_stages and len(init_f) >= n_init:
                est.initialise(np.array(init_f), np.array(init_w),
                               shared.q.reshape(4, 3), shared.contacts)
                last_imu_t = time.monotonic()
                shared.bias_str = (f"bw={np.round(est.state.bw, 5)} "
                                   f"bf={np.round(est.state.bf, 4)}")
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
            dt = min(max(tm - last_imu_t, dt_clamp[0]), dt_clamp[1])
            est.predict(fmean, wmean, dt, contacts)
            last_imu_t, w_last = tm, wmean
            if shared.log_enabled:
                for f, w, ts in samples:
                    shared.imu_log.append((ts, *f, *w))
        alpha = q.reshape(4, 3)
        if verbose:
            # peek the innovation about to be applied (hw_replay pattern;
            # update() re-runs handle_transitions idempotently)
            est.handle_transitions(contacts, alpha)
            legs, y, H, R = est.build_measurement(alpha, contacts)
            extra = {"foot": {LEGS[i]: est.state.p(i).copy() for i in range(4)}}
            if len(legs):
                S = H @ est.state.P @ H.T + R
                e = np.abs(y) / np.sqrt(np.clip(np.diag(S), 1e-12, None))
                extra["innov"] = {LEGS[li]: float(np.max(e[3 * j:3 * j + 3]))
                                  for j, li in enumerate(legs)}
            else:
                extra["innov"] = {}
            shared.extra = extra
        est.update(alpha, contacts)
        out = est.outputs(last_w_meas=w_last)
        out["C"] = quat_to_C(out["q"])
        out["bf"] = est.state.bf.copy()
        out["bw"] = est.state.bw.copy()
        shared.out = out
        shared.tau_stamp = time.monotonic()
        rem = control_dt - (time.perf_counter() - t)
        if rem > 0:
            time.sleep(rem)


def _rp(C):
    """Roll, pitch (rad) from the I->B rotation matrix (gravity direction)."""
    g = C @ np.array([0.0, 0.0, 1.0])
    return math.atan2(g[1], g[2]), math.atan2(-g[0], math.hypot(g[1], g[2]))


def _rpy(C):
    """Roll, pitch, yaw (rad) from the I->B rotation.  Yaw is proprioceptively
    unobservable -- it drifts with the integrated gyro; display only."""
    r, p = _rp(C)
    yaw = math.atan2(C[0, 1], C[0, 0])   # R_IB = C.T; atan2(R[1,0], R[0,0])
    return r, p, yaw
