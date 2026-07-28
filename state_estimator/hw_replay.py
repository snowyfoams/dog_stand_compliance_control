#!/usr/bin/env python3
"""Offline replay + innovation analysis of a hardware EKF log (gate C8).

Reads an NPZ written by `hw_ekf.py --raw-log`, re-runs the estimator on the
recorded IMU + encoder streams, and reports the health of the fit: estimator
attitude vs the IMU-AHRS reference, innovation zero-mean & 3-sigma containment,
per-leg persistent offsets, and bias convergence. Pure numpy — runs under the
system python3 (no CAN, no hardware, no mujoco).

Usage:
    python3 hw_replay.py rest.npz
    python3 hw_replay.py rest.npz --static      # also check |v|~0, level

Diagnostic reading (plan.md 6.5):
    one leg persistently offset  -> that leg's kinematic zero / link length
    all legs offset together     -> base-frame / IMU extrinsic
    innovation grows with speed  -> IMU/encoder timestamp skew
    spikes at contact edges      -> contact-flag timing
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from dog5_state_estimator import DOG5StateEstimator, quat_to_C, ErrorIndex, LEGS

DT_CLAMP = (1.0e-4, 0.05)


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def _roll_pitch(q):
    g_b = quat_to_C(q) @ np.array([0.0, 0.0, 1.0])
    return math.atan2(g_b[1], g_b[2]), math.atan2(-g_b[0], math.hypot(g_b[1], g_b[2]))


def replay(data):
    """Re-run the EKF on logged inputs; return a results dict of time series."""
    imu_t, imu_f, imu_w = data["imu_t"], data["imu_f"], data["imu_w"]
    enc_t, enc_alpha = data["enc_t"], data["enc_alpha"]
    enc_c, ahrs_rp = data["enc_contacts"], data["ahrs_rp"]
    init_secs = float(data["init_secs"]) if "init_secs" in data else 1.0

    est = DOG5StateEstimator()

    # init from the static prefix of the IMU stream + first encoder pose
    t0 = imu_t[0]
    init_mask = imu_t < t0 + init_secs
    contacts0 = enc_c[0].astype(bool)
    est.initialise(imu_f[init_mask], imu_w[init_mask],
                   enc_alpha[0].reshape(4, 3), contacts0)

    res = {k: [] for k in ("t", "z", "v", "roll_e", "pitch_e", "roll_a", "pitch_a",
                           "bf", "bw", "healthy", "min_eig")}
    innov = {"t": [], "y": [], "e": [], "legs": []}   # per encoder update

    init_end = t0 + init_secs
    last_t = init_end
    ii = int(np.searchsorted(imu_t, init_end))        # first IMU sample after init

    for m in range(len(enc_t)):
        te = enc_t[m]
        if te <= init_end:
            continue
        # predict every IMU sample up to this encoder frame
        contacts = enc_c[m].astype(bool)
        while ii < len(imu_t) and imu_t[ii] <= te:
            dt = min(max(imu_t[ii] - last_t, DT_CLAMP[0]), DT_CLAMP[1])
            est.predict(imu_f[ii], imu_w[ii], dt, contacts)
            last_t = imu_t[ii]
            ii += 1
        alpha = enc_alpha[m].reshape(4, 3)

        # peek the innovation the filter is about to apply (post re-anchor)
        est.handle_transitions(contacts, alpha)
        legs, y, H, R = est.build_measurement(alpha, contacts)
        if len(legs):
            S = H @ est.state.P @ H.T + R
            e = y / np.sqrt(np.clip(np.diag(S), 1e-12, None))   # normalised
            innov["t"].append(te)
            innov["y"].append(y.copy())
            innov["e"].append(e.copy())
            innov["legs"].append(list(legs))
        est.update(alpha, contacts)

        out = est.outputs(last_w_meas=imu_w[min(ii, len(imu_w)) - 1])
        re_, pe = _roll_pitch(out["q"])
        res["t"].append(te)
        res["z"].append(float(out["r"][2]))
        res["v"].append(out["v"].copy())
        res["roll_e"].append(re_)
        res["pitch_e"].append(pe)
        res["roll_a"].append(float(ahrs_rp[m, 0]))
        res["pitch_a"].append(float(ahrs_rp[m, 1]))
        res["bf"].append(est.state.bf.copy())
        res["bw"].append(est.state.bw.copy())
        res["healthy"].append(bool(out["healthy"]))
        res["min_eig"].append(float(out["min_eig_P"]))

    for k in ("t", "z", "roll_e", "pitch_e", "roll_a", "pitch_a", "min_eig"):
        res[k] = np.array(res[k])
    for k in ("v", "bf", "bw"):
        res[k] = np.array(res[k])
    res["healthy"] = np.array(res["healthy"])
    res["innov"] = innov
    return res


def report(res, static=False):
    """Print a pass/fail summary. Returns True if all checks pass."""
    ok = True
    n = len(res["t"])
    print(f"replayed {n} encoder updates, {len(res['innov']['t'])} with contact")
    if n == 0:
        print("  no data")
        return False

    def line(name, passed, detail=""):
        nonlocal ok
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    # attitude vs AHRS reference (roll/pitch trusted)
    tail = slice(int(0.3 * n), None)   # skip the convergence transient
    droll = np.degrees(np.abs(res["roll_e"] - res["roll_a"]))[tail]
    dpitch = np.degrees(np.abs(res["pitch_e"] - res["pitch_a"]))[tail]
    line("EKF roll matches AHRS (<1.5 deg mean)", np.nanmean(droll) < 1.5,
         f"mean {np.nanmean(droll):.2f} max {np.nanmax(droll):.2f} deg")
    line("EKF pitch matches AHRS (<1.5 deg mean)", np.nanmean(dpitch) < 1.5,
         f"mean {np.nanmean(dpitch):.2f} max {np.nanmax(dpitch):.2f} deg")

    # innovation: zero-mean & 3-sigma containment
    if res["innov"]["e"]:
        e_all = np.concatenate(res["innov"]["e"])
        contain = float(np.mean(np.abs(e_all) < 3.0))
        line("innovation zero-mean (|mean normalised| < 0.5)",
             abs(float(np.mean(e_all))) < 0.5, f"{np.mean(e_all):+.3f}")
        line("innovation 3-sigma containment >= 95%", contain >= 0.95,
             f"{contain*100:.1f}%")

        # per-leg persistent offset (mean raw innovation magnitude, mm)
        per_leg = {l: [] for l in LEGS}
        for y, legs in zip(res["innov"]["y"], res["innov"]["legs"]):
            for j, leg_i in enumerate(legs):     # build_measurement returns indices
                per_leg[LEGS[leg_i]].append(y[3 * j:3 * j + 3])
        offs = {l: (np.mean(np.array(v), axis=0) if v else np.zeros(3))
                for l, v in per_leg.items()}
        worst = max(LEGS, key=lambda l: np.linalg.norm(offs[l]))
        worst_mm = float(np.linalg.norm(offs[worst]) * 1e3)
        print("    per-leg mean innovation (mm): " +
              "  ".join(f"{l}={np.linalg.norm(offs[l])*1e3:.1f}" for l in LEGS))
        line("no persistent per-leg offset (< 8 mm)", worst_mm < 8.0,
             f"worst {worst} {worst_mm:.1f} mm")

    # bias convergence (settled over the last 20%)
    bf, bw = res["bf"], res["bw"]
    bf_settle = float(np.max(np.std(bf[int(0.8 * n):], axis=0)))
    bw_settle = float(np.max(np.std(bw[int(0.8 * n):], axis=0)))
    line("gyro bias settled", bw_settle < 5e-4, f"std {bw_settle:.2e} rad/s")
    line("accel bias settled", bf_settle < 5e-3, f"std {bf_settle:.2e} m/s^2")
    print(f"    final bias  bf={np.round(bf[-1],4)}  bw={np.round(bw[-1],5)}")

    # health
    line("healthy throughout (>=99%)", float(np.mean(res["healthy"])) > 0.99,
         f"{np.mean(res['healthy'])*100:.1f}%")

    if static:
        vmax = float(np.max(np.linalg.norm(res["v"][tail], axis=1)))
        zdrift = float(np.max(np.abs(res["z"][tail] - np.mean(res["z"][tail]))))
        line("static: |v| < 20 mm/s", vmax < 0.02, f"{vmax*1e3:.1f} mm/s")
        line("static: z drift < 10 mm", zdrift < 0.01, f"{zdrift*1e3:.1f} mm")

    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("--static", action="store_true",
                    help="also check |v|~0 and level (dog held still)")
    args = ap.parse_args()
    print("=" * 66)
    print(f"EKF hardware replay -- {args.npz}")
    print("=" * 66)
    res = replay(load(args.npz))
    ok = report(res, static=args.static)
    print("=" * 66)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
