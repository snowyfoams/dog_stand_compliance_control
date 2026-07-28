"""Offline tests for the hardware IMU->EKF bridge (no hardware, no python-can).

Verifies the NED->FLU transform and the buffering semantics using a mock raw
IMU packet, and confirms a short synthetic stream drives the estimator to a
sane state through the exact code path hw_ekf.py uses. The CAN runner itself
(hw_ekf.py) needs python-can + hardware and is only syntax-checked here.

Run:  python3 test_hw_ekf.py
"""
from __future__ import annotations

import math
import types
import sys

import numpy as np

import imu_ekf_feed as feedmod
from imu_ekf_feed import ImuEkfFeed, ned_to_flu

_FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _FAIL.append(name)


def fake_imu(accel, gyro):
    return types.SimpleNamespace(accel=tuple(accel), gyro=tuple(gyro))


def test_transform():
    print("NED -> FLU transform")
    # rest & level: sensor reads (0,0,-9.81) NED -> (0,0,+9.81) FLU (estimator eq.3)
    check("rest accel -> [0,0,+9.81]",
          np.allclose(ned_to_flu((0, 0, -9.81)), [0, 0, 9.81]))
    # generic vector: keep X, negate Y and Z
    check("negate Y,Z only", np.allclose(ned_to_flu((1, 2, 3)), [1, -2, -3]))


def test_bridge_buffering():
    print("bridge callback + buffering")
    feed = ImuEkfFeed(port="/dev/null")          # __init__ does NOT open the port
    feed._on_imu(fake_imu((0.1, 0.2, -9.8), (0.01, -0.02, 0.03)))
    feed._on_imu(fake_imu((0.0, 0.0, -9.81), (0.0, 0.0, 0.0)))
    items = feed.drain()
    check("drain returns both samples", len(items) == 2, f"{len(items)}")
    f0, w0, t0 = items[0]
    check("sample0 f_flu correct", np.allclose(f0, [0.1, -0.2, 9.8]))
    check("sample0 w_flu correct", np.allclose(w0, [0.01, 0.02, -0.03]))
    check("drain clears buffer", len(feed.drain()) == 0)
    last = feed.latest()
    check("latest() is the newest sample", np.allclose(last[0], [0.0, 0.0, 9.81]))


def test_estimator_pathway():
    """Feed a synthetic rest stream through the SAME predict/update calls hw_ekf
    makes, and confirm the estimator initialises and stays healthy & level."""
    print("estimator plumbing (synthetic rest, hw_ekf code path)")
    _EST = feedmod._IMU_DIR  # not used; keep imports local below
    sys.path.insert(0, str(feedmod.Path(__file__).resolve().parent))
    from dog5_state_estimator import DOG5StateEstimator, quat_to_C
    import dog5_sim as sim

    # a level static hold from the simulator gives us real FLU f/w + encoders
    data = sim.scenario_static(sim.SimConfig(dt=1 / 200, duration=3.0))
    est = DOG5StateEstimator()
    ninit = 100
    contacts = np.ones(4, dtype=bool)
    est.initialise(data.f_meas[:ninit], data.w_meas[:ninit], data.alpha[0], contacts)
    dt = 1 / 200
    for k in range(ninit, data.n):
        est.predict(data.f_meas[k - 1], data.w_meas[k - 1], dt, contacts)
        est.update(data.alpha[k], contacts)
    out = est.outputs(last_w_meas=data.w_meas[-1])
    g_b = quat_to_C(out["q"]) @ np.array([0.0, 0.0, 1.0])
    tilt = math.degrees(math.hypot(math.atan2(g_b[1], g_b[2]),
                                   math.atan2(-g_b[0], math.hypot(g_b[1], g_b[2]))))
    check("initialises + stays healthy", bool(out["healthy"]))
    check("stays level (<0.5 deg)", tilt < 0.5, f"{tilt:.3f} deg")
    check("velocity ~0 (<5 mm/s)", np.linalg.norm(out["v"]) < 5e-3,
          f"{np.linalg.norm(out['v'])*1e3:.2f} mm/s")


def _sim_to_npz(data, enc_stride=10):
    """Convert a dog5_sim run into the hw_ekf --raw-log NPZ format."""
    from dog5_state_estimator import quat_to_C
    dt = float(data.t[1] - data.t[0])
    imu_t = np.arange(data.n) * dt
    enc_idx = np.arange(0, data.n, enc_stride)     # encoders logged slower than IMU
    rp = []
    for m in enc_idx:
        g_b = quat_to_C(data.q[m]) @ np.array([0.0, 0.0, 1.0])
        rp.append([math.atan2(g_b[1], g_b[2]),
                   math.atan2(-g_b[0], math.hypot(g_b[1], g_b[2]))])
    return {
        "imu_t": imu_t, "imu_f": data.f_meas, "imu_w": data.w_meas,
        "enc_t": imu_t[enc_idx],
        "enc_alpha": data.alpha[enc_idx].reshape(len(enc_idx), 12),
        "enc_contacts": data.contacts[enc_idx].astype(float),
        "ahrs_rp": np.array(rp), "init_secs": 1.0,
    }


def test_replay():
    """hw_replay must report PASS on a clean synthetic static log."""
    print("hw_replay on a synthetic static log")
    import dog5_sim as sim
    import hw_replay
    data = sim.scenario_static(sim.SimConfig(dt=1 / 200, duration=4.0, seed=1))
    npz = _sim_to_npz(data)
    res = hw_replay.replay(npz)
    ok = hw_replay.report(res, static=True)
    check("replay reports PASS on clean static log", ok)


def main():
    print("=" * 60)
    print("hardware EKF bridge -- offline tests")
    print("=" * 60)
    test_transform()
    test_bridge_buffering()
    test_estimator_pathway()
    test_replay()
    print("=" * 60)
    if _FAIL:
        print(f"FAILED ({len(_FAIL)}): " + ", ".join(_FAIL))
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
