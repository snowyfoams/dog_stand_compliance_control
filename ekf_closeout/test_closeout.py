#!/usr/bin/env python3
"""Offline tests for the EKF close-out tooling.  No robot, no CAN.

    python3 test_closeout.py            # health fix + replay gates + drift math
    python3 test_closeout.py --logs     # also replay the real hardware logs
                                        # (slow: ~2 min per walk log)

The command-driver tests live in `walk_cmd_hw.py --self-test` because they
import walk1_hw -> motorbus -> python-can, which only exists in the project
venv:

    /home/robot01/Documents/can_motor_control/.venv/bin/python \\
        walk_cmd_hw.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EST = os.path.join(os.path.dirname(_HERE), "state_estimator")
for _p in (_HERE, _EST):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import replay_full                                        # noqa: E402
from estimator_health import (HealthFixEstimator, health_terms,  # noqa: E402
                              failing_term, MIN_EIG_REL_TOL, SIGMA_V_LIMIT)
from dog5_state_estimator import DOG5StateEstimator, ErrorIndex  # noqa: E402

REPO = os.path.dirname(os.path.dirname(_HERE))
WALK_LOG = os.path.join(REPO, "walk_0729_1748.npz")
STAND_LOG = os.path.join(os.path.dirname(_HERE), "vmc", "stand.npz")

PASSED = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    PASSED.append(bool(cond))
    if not cond:
        raise AssertionError(name + (f": {detail}" if detail else ""))


class _FakeState:
    def __init__(self, P):
        self.P = P
        self.x = np.zeros(28)


def _fake_out(P, sigma_v=0.004, min_eig=None):
    sigma = np.zeros(27)
    sigma[ErrorIndex.V] = sigma_v
    if min_eig is None:
        min_eig = float(np.min(np.linalg.eigvalsh(0.5 * (P + P.T))))
    return {"sigma": sigma, "min_eig_P": float(min_eig)}


# ---------------------------------------------------------------- health fix
# Numbers observed in walk_0729_1748.npz at t = 71.46 s (mid-swing, 3 feet
# planted, sigma_v 4.1 mm/s): eigvalsh jitter on a covariance whose swing
# foothold block has grown to ~7e8 m^2.
OBS_MAX_DIAG_P = 7.0e8
OBS_MIN_EIG = -3.68e-8


def test_health_fix():
    print("== health flag: relative min_eig tolerance ==")
    n = 27
    P = np.eye(n) * OBS_MAX_DIAG_P        # max diag matches the observed frame

    healthy, detail = health_terms(_FakeState(P),
                                   _fake_out(P, min_eig=OBS_MIN_EIG))
    stock_ok = OBS_MIN_EIG > -1.0e-9
    check("observed swing frame accepted by the relative tolerance", healthy,
          f"min_eig {detail['min_eig']:.2e} vs tol {detail['min_eig_tol']:.2e}")
    check("the stock absolute -1e-9 tolerance rejects that same frame",
          not stock_ok, "which is the bug this fixes")
    check("tolerance scales with P", detail["min_eig_tol"] < -1.0e-9,
          f"{detail['min_eig_tol']:.2e}")

    # A genuinely indefinite covariance must still be rejected: a real
    # negative direction is orders of magnitude larger than eigvalsh jitter.
    healthy, detail = health_terms(
        _FakeState(P), _fake_out(P, min_eig=-1.0e-3 * OBS_MAX_DIAG_P))
    check("genuinely indefinite P still rejected", not healthy,
          f"term={failing_term(detail)}, min_eig {detail['min_eig']:.2e}")

    # ... and so is anything above the jitter scale by a wide margin.
    healthy, _ = health_terms(_FakeState(P), _fake_out(P, min_eig=-1.0))
    check("min_eig = -1.0 rejected", not healthy)

    # The sigma_v term is untouched.
    healthy, detail = health_terms(_FakeState(P),
                                   _fake_out(P, sigma_v=0.2, min_eig=1.0))
    check("sigma_v term unchanged (0.2 m/s rejected)", not healthy,
          f"term={failing_term(detail)}, limit {SIGMA_V_LIMIT}")
    healthy, _ = health_terms(_FakeState(P),
                              _fake_out(P, sigma_v=0.149, min_eig=1.0))
    check("sigma_v just under the limit accepted", healthy)

    # Small, well-conditioned P: the relative tolerance must not become looser
    # than the stock one (the scale floors at 1.0).
    small = np.eye(n) * 1.0e-6
    _, detail = health_terms(_FakeState(small), _fake_out(small))
    check("small P: tolerance floors at the absolute scale",
          abs(detail["min_eig_tol"]) <= MIN_EIG_REL_TOL,
          f"{detail['min_eig_tol']:.2e}")


# ----------------------------------------------------------- report machinery
def _synth_res(n=1000, contacts_off=(100, 300), drift=(0.040, -0.003),
               z0=0.10, z1=0.175):
    """A synthetic replay result: static prefix, contacts-off rise, walk.

    Shaped like a real run: the dog stands still at HOLD4 for a beat before
    the first step and after the last, so the median windows at each end of
    the displacement measurement sit on stationary data.
    """
    t = np.arange(n) * 0.05
    n_c = np.full(n, 4)
    n_c[contacts_off[0]:contacts_off[1]] = 0
    i0 = contacts_off[1]
    r = np.zeros((n, 3))
    r[:, 2] = z0
    r[i0:, 2] = z1
    # rise dead-reckoning: a big bogus slide that must be excluded
    r[contacts_off[0]:i0, 0] = np.linspace(0, 0.5, i0 - contacts_off[0])
    r[i0:, 0] = 0.5
    # re-anchor settle transient (2 s), then a HOLD4 dwell, then the walk,
    # then a final HOLD4 dwell
    settle = min(i0 + 40, n)
    r[i0:settle, 0] = np.linspace(0.5, 0.45, settle - i0)
    dwell = min(settle + 60, n)          # 3 s standing before the first step
    r[settle:dwell, 0] = 0.45
    walk_end = max(dwell, n - 40)        # last 2 s standing again
    r[dwell:walk_end, 0] = 0.45 + np.linspace(0, drift[0], walk_end - dwell)
    r[dwell:walk_end, 1] = np.linspace(0, drift[1], walk_end - dwell)
    r[walk_end:, 0] = 0.45 + drift[0]
    r[walk_end:, 1] = drift[1]
    return {
        "t": t, "r": r, "z": r[:, 2], "v": np.zeros((n, 3)),
        "n_contacts": n_c, "healthy": np.ones(n, dtype=bool),
        "fail_term": np.array(["healthy"] * n),
        "sigma_v": np.zeros(n), "max_diag_P": np.ones(n),
        "min_eig": np.zeros(n), "roll_e": np.zeros(n), "pitch_e": np.zeros(n),
        "roll_a": np.zeros(n), "pitch_a": np.zeros(n),
        "bf": np.zeros((n, 3)), "bw": np.zeros((n, 3)),
        "innov": {"t": [], "y": [], "e": [], "legs": []}, "touchdowns": [],
    }


def test_report_machinery():
    print("== report machinery ==")
    res = _synth_res()

    i0 = replay_full.health_window_start(res)
    check("health window starts at the re-anchor", i0 == 300, f"i0={i0}")

    res_static = _synth_res(contacts_off=(0, 0))
    res_static["n_contacts"][:] = 4
    check("pure static log excludes nothing",
          replay_full.health_window_start(res_static) == 0)

    d, (t_start, t_end, rise) = replay_full.walk_displacement(res)
    check("walk displacement excludes the rise + settle transient",
          np.allclose(d, [0.040, -0.003], atol=1.5e-3),
          f"({d[0] * 1e3:+.1f}, {d[1] * 1e3:+.1f}) mm, rise "
          f"{np.linalg.norm(rise) * 1e3:.0f} mm excluded")
    check("displacement window starts after the settle", t_start >= 15.0 + 3.0,
          f"t_start {t_start:.1f}s")

    dz = replay_full.ekf_stand_dz(res)
    check("stand dz recovered from the contacts-off rise",
          abs(dz - 0.075) < 1.0e-3, f"{dz * 1e3:.1f} mm")
    check("stand dz is None without a rise",
          replay_full.ekf_stand_dz(res_static) is None)

    # gate arithmetic: 10 % of distance, floored at 20 mm
    for dist_mm, err_mm, want in ((160, 15, True), (160, 21, False),
                                  (50, 19, True), (800, 70, True),
                                  (800, 90, False)):
        lim = max(replay_full.DRIFT_FLOOR_MM,
                  replay_full.DRIFT_FRAC * dist_mm)
        check(f"drift gate: {err_mm} mm error on {dist_mm} mm "
              f"-> {'pass' if want else 'fail'}", (err_mm <= lim) == want)


def test_sidecar():
    print("== command sidecar round-trip ==")
    with tempfile.TemporaryDirectory() as d:
        npz = os.path.join(d, "walk_A1.npz")
        side = os.path.join(d, "walk_A1.cmd.json")
        cmd = {"goto_m": [0.16, 0.0], "heading_deg": 0.0, "n_cycles": 4,
               "step_m": 0.040, "quantized_dxy_m": [0.16, 0.0],
               "stand_height_m": 0.175, "time_scale": 1.0}
        with open(side, "w") as fh:
            json.dump(cmd, fh)
        got = replay_full.load_command(npz)
        check("sidecar loads next to the npz", got is not None
              and got["n_cycles"] == 4)
        check("missing sidecar is not an error",
              replay_full.load_command(os.path.join(d, "other.npz")) is None)


def test_replay_estimator_equivalence():
    print("== replay uses the same filter math as hw_replay ==")
    # Same class hierarchy, and only the health flag differs.
    check("HealthFixEstimator is a DOG5StateEstimator",
          issubclass(HealthFixEstimator, DOG5StateEstimator))
    base_only = set(dir(DOG5StateEstimator)) - set(dir(HealthFixEstimator))
    check("no estimator methods dropped", not base_only, str(base_only))
    overridden = [n for n in ("predict", "update", "handle_transitions",
                              "build_measurement", "initialise")
                  if getattr(HealthFixEstimator, n)
                  is not getattr(DOG5StateEstimator, n)]
    check("filter math untouched (only outputs overridden)", not overridden,
          str(overridden))


# ------------------------------------------------------------- hardware logs
def test_hardware_logs():
    print("== hardware log regression (slow) ==")
    from hw_replay import load
    for path, kind in ((WALK_LOG, "gait"), (STAND_LOG, "static")):
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        res = replay_full.replay(load(path))
        i0 = replay_full.health_window_start(res)
        frac = float(np.mean(res["healthy"][i0:]))
        check(f"{os.path.basename(path)}: healthy after re-anchor >= 99%",
              frac > 0.99, f"{frac * 100:.1f}%")
        stock_bad = int(np.sum(res["min_eig"] <= -1.0e-9))
        print(f"    (frames the stock absolute tolerance would reject: "
              f"{stock_bad})")
        if kind == "gait":
            d, _ = replay_full.walk_displacement(res)
            check("walk_0729_1748: EKF walked ~40 mm forward (1 cycle)",
                  abs(d[0] - 0.040) < 0.010 and abs(d[1]) < 0.010,
                  f"({d[0] * 1e3:+.1f}, {d[1] * 1e3:+.1f}) mm")
        else:
            tail = slice(int(0.3 * len(res["t"])), None)
            vmag = np.linalg.norm(res["v"][tail], axis=1)
            check("stand.npz: 95th-pct |v| passes where max fails",
                  np.percentile(vmag, 95) * 1e3 < 20.0
                  and np.max(vmag) * 1e3 > 20.0,
                  f"95th {np.percentile(vmag, 95) * 1e3:.1f}, "
                  f"max {np.max(vmag) * 1e3:.1f} mm/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", action="store_true",
                    help="also replay the real hardware logs (slow)")
    args = ap.parse_args()

    test_health_fix()
    test_report_machinery()
    test_sidecar()
    test_replay_estimator_equivalence()
    if args.logs:
        test_hardware_logs()
    else:
        print("== hardware log regression skipped (--logs to run) ==")

    print(f"\n{len(PASSED)} checks, all PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
