#!/usr/bin/env python3
"""Offline tests for the gait EKF pipeline (no hardware, numpy only).

1. Synthetic end-to-end: a static prefix + `dog5_sim.scenario_trot` converted
   into the exact NPZ schema `walk1_hw.py --raw-log` writes, then run through
   `hw_replay.load/replay/report(gait=True)` -- the same analyzer the real
   walk log will get.  Asserts the gait report PASSES and the touchdown
   (contact rising-edge) bookkeeping counts every replant.
2. Schema round-trip: the writer's key set matches what `replay()` consumes.
3. Headless `ekf_runtime.ekf_worker` against a fake feed + scripted stages:
   init only fires in `quiet_stages`, `imu_log` respects `log_enabled`,
   outputs publish, and `run = False` terminates promptly.

Run:  python3 test_hw_replay_gait.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hw_replay                                     # noqa: E402
from dog5_sim import SimConfig, scenario_static, scenario_trot  # noqa: E402
from ekf_runtime import EkfShared, ekf_worker        # noqa: E402

ENC_DECIM = 20        # 400 Hz sim -> 20 Hz encoder frames (hardware-like)


def build_trot_npz(path):
    """Static prefix + trot, in walk1_hw's --raw-log NPZ schema."""
    stat = scenario_static(SimConfig(duration=1.5, seed=1))
    trot = scenario_trot(SimConfig(duration=4.8, seed=2))
    t0 = stat.t[-1] + (stat.t[1] - stat.t[0])

    # continuous concatenation: both segments are at rest at Q_STANCE at the
    # boundary (static ends motionless; the trot ramps from zero velocity)
    imu_t = np.concatenate([stat.t, trot.t + t0])
    imu_f = np.concatenate([stat.f_meas, trot.f_meas])
    imu_w = np.concatenate([stat.w_meas, trot.w_meas])
    alpha = np.concatenate([stat.alpha, trot.alpha])          # (K,4,3)
    contacts = np.concatenate([stat.contacts, trot.contacts])  # (K,4)
    q_truth = np.concatenate([stat.q, trot.q])

    # encoder frames: subsample to ~20 Hz; AHRS reference from truth attitude
    sel = np.arange(0, len(imu_t), ENC_DECIM)
    ahrs = np.array([hw_replay._roll_pitch(q_truth[k]) for k in sel])
    np.savez(path,
             imu_t=imu_t, imu_f=imu_f, imu_w=imu_w,
             enc_t=imu_t[sel], enc_alpha=alpha[sel].reshape(len(sel), 12),
             enc_contacts=contacts[sel].astype(float),
             ahrs_rp=ahrs, init_secs=1.0)
    return path


def test_schema_round_trip(npz_path):
    consumed = {"imu_t", "imu_f", "imu_w", "enc_t", "enc_alpha",
                "enc_contacts", "ahrs_rp", "init_secs"}
    written = set(hw_replay.load(npz_path).keys())
    assert consumed == written, (consumed ^ written)
    print("[PASS] NPZ schema round-trip (writer keys == replay keys)")


def test_trot_replay(npz_path):
    data = hw_replay.load(npz_path)
    res = hw_replay.replay(data)
    ok = hw_replay.report(res, gait=True)
    assert ok, "gait report FAILED on the synthetic trot log"

    # every replant in the (post-init) encoder stream must be a counted event
    enc_t, enc_c = data["enc_t"], data["enc_contacts"].astype(bool)
    init_end = enc_t[0] + float(data["init_secs"])
    live = enc_t > init_end
    rising = enc_c[1:] & ~enc_c[:-1]
    expected = int(np.sum(rising[live[1:]]))
    assert expected > 0, "trot log produced no touchdowns -- test is vacuous"
    assert len(res["touchdowns"]) == expected, \
        (len(res["touchdowns"]), expected)
    print(f"[PASS] synthetic trot: gait report PASS, "
          f"{expected} touchdowns all captured")


class FakeFeed:
    """Duck-typed ImuEkfFeed serving level-and-still FLU samples."""

    def __init__(self):
        self.lock = threading.Lock()
        self.pending = []

    def push(self, n):
        t = time.monotonic()
        with self.lock:
            for k in range(n):
                self.pending.append((np.array([0.0, 0.0, 9.81]),
                                     np.zeros(3), t + 1e-3 * k))

    def drain(self):
        with self.lock:
            items = self.pending
            self.pending = []
        return items

    def attitude(self):
        return None


def test_worker_lifecycle():
    stance_alpha = np.array([[0.0, 0.6, -1.2]] * 4).reshape(12)
    shared = EkfShared(stance_alpha)
    feed = FakeFeed()
    worker = threading.Thread(
        target=ekf_worker, args=(shared, feed),
        kwargs=dict(quiet_stages=("WAIT_CROUCH",), control_hz=200.0),
        daemon=True)
    worker.start()

    # samples during a NON-quiet stage must not initialise (nor be counted)
    shared.stage = "CROUCH"
    feed.push(60)
    time.sleep(0.15)
    assert not shared.est_ready, "init fired outside quiet_stages"

    # quiet stage + enough samples -> init; log still gated OFF
    shared.stage = "WAIT_CROUCH"
    feed.push(60)
    deadline = time.time() + 2.0
    while not shared.est_ready and time.time() < deadline:
        time.sleep(0.01)
    assert shared.est_ready, "worker did not initialise in quiet stage"
    assert shared.bias_str
    assert len(shared.imu_log) == 0, "imu_log grew while log_enabled=False"

    # steady state publishes outputs; log grows once enabled
    shared.log_enabled = True
    feed.push(20)
    deadline = time.time() + 2.0
    while shared.out is None and time.time() < deadline:
        time.sleep(0.01)
    assert shared.out is not None and "C" in shared.out
    assert shared.out["healthy"]
    deadline = time.time() + 1.0
    while len(shared.imu_log) < 20 and time.time() < deadline:
        time.sleep(0.01)
    assert len(shared.imu_log) >= 20, "enabled imu_log did not grow"

    shared.run = False
    worker.join(timeout=1.0)
    assert not worker.is_alive(), "worker did not stop on run=False"
    print("[PASS] worker lifecycle: quiet-stage init, log gating, publish, stop")


def main():
    with tempfile.TemporaryDirectory() as td:
        npz = build_trot_npz(os.path.join(td, "trot.npz"))
        test_schema_round_trip(npz)
        test_trot_replay(npz)
    test_worker_lifecycle()
    print("test_hw_replay_gait PASS")


if __name__ == "__main__":
    main()
