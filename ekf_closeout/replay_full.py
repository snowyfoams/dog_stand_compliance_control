#!/usr/bin/env python3
"""Full-state EKF replay + close-out report (offline, numpy only).

`state_estimator/hw_replay.py` keeps only the z component of position and
gates health with the stock (absolute) min_eig tolerance.  This replay stores
the FULL base position and applies the corrected gates, so the EKF's x/y and
z accuracy can be checked against tape measurements -- the no-mocap
substitute for ground truth.

What it adds over hw_replay:
  * full r (x, y, z) per frame, plus sigma_v / min_eig / max diag P
  * health gate evaluated on the post-re-anchor window only (the contacts-off
    STAND rise dead-reckons by design -- gating it is meaningless)
  * static |v| gated on the 95th percentile (max catches deliberate
    hand-rocking; see CONTROL_ROADMAP.md)
  * commanded displacement from the walk_cmd_hw sidecar (<log>.cmd.json)
  * --measured "X,Y"  : tape-measured actual body displacement (mm)
  * --measured-dz MM  : tape-measured crouch->stand height change (mm)
  * --health-detail   : per-cluster diagnosis of every unhealthy frame

The legacy hw_replay report still runs (attitude / innovation / per-leg
offset / bias / touchdowns are unchanged and remain the continuity record);
its health line is evaluated on the corrected window.

Usage:
    python3 replay_full.py ../../walk_0729_1748.npz --gait --health-detail
    python3 replay_full.py walk_A1.npz --gait --measured "158,4"
    python3 replay_full.py h_175.npz --measured-dz 62
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EST = os.path.join(os.path.dirname(_HERE), "state_estimator")
for _p in (_HERE, _EST):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hw_replay                                    # noqa: E402
from hw_replay import LEGS, DT_CLAMP, load, _roll_pitch   # noqa: E402
from estimator_health import HealthFixEstimator, failing_term  # noqa: E402

# Gates
DRIFT_FLOOR_MM = 20.0        # absolute floor for short walks
DRIFT_FRAC = 0.10            # 10 % of distance travelled (Bloesch RSS 2013)
DZ_TOL_MM = 10.0             # height cross-check tolerance
STATIC_V_LIMIT_MMS = 20.0    # on the 95th percentile
STATIC_Z_DRIFT_MM = 10.0


def replay(data, est=None):
    """Re-run the EKF, keeping the full state.

    Mirrors hw_replay.replay (same init, same two-rate loop, same innovation
    peek) so the shared gates stay comparable; the difference is the stored
    fields and the HealthFixEstimator default.
    """
    imu_t, imu_f, imu_w = data["imu_t"], data["imu_f"], data["imu_w"]
    enc_t, enc_alpha = data["enc_t"], data["enc_alpha"]
    enc_c, ahrs_rp = data["enc_contacts"], data["ahrs_rp"]
    init_secs = float(data["init_secs"]) if "init_secs" in data else 1.0

    est = est if est is not None else HealthFixEstimator()

    t0 = imu_t[0]
    init_mask = imu_t < t0 + init_secs
    contacts0 = enc_c[0].astype(bool)
    est.initialise(imu_f[init_mask], imu_w[init_mask],
                   enc_alpha[0].reshape(4, 3), contacts0)

    keys = ("t", "z", "v", "roll_e", "pitch_e", "roll_a", "pitch_a",
            "bf", "bw", "healthy", "min_eig",
            "r", "sigma_v", "max_diag_P", "n_contacts", "fail_term")
    res = {k: [] for k in keys}
    innov = {"t": [], "y": [], "e": [], "legs": []}
    touchdowns = []

    init_end = t0 + init_secs
    last_t = init_end
    ii = int(np.searchsorted(imu_t, init_end))
    prev_c = contacts0.copy()

    for m in range(len(enc_t)):
        te = enc_t[m]
        if te <= init_end:
            prev_c = enc_c[m].astype(bool)
            continue
        contacts = enc_c[m].astype(bool)
        while ii < len(imu_t) and imu_t[ii] <= te:
            dt = min(max(imu_t[ii] - last_t, DT_CLAMP[0]), DT_CLAMP[1])
            est.predict(imu_f[ii], imu_w[ii], dt, contacts)
            last_t = imu_t[ii]
            ii += 1
        alpha = enc_alpha[m].reshape(4, 3)

        rising = np.flatnonzero(contacts & ~prev_c)
        if len(rising):
            pre = {i: est.state.p(i).copy() for i in rising}
            r_pre, v_pre = est.state.r.copy(), est.state.v.copy()

        est.handle_transitions(contacts, alpha)
        legs, y, H, R = est.build_measurement(alpha, contacts)
        if len(legs):
            S = H @ est.state.P @ H.T + R
            e = y / np.sqrt(np.clip(np.diag(S), 1e-12, None))
            innov["t"].append(te)
            innov["y"].append(y.copy())
            innov["e"].append(e.copy())
            innov["legs"].append(list(legs))
        est.update(alpha, contacts)

        if len(rising):
            for i in rising:
                touchdowns.append((
                    te, int(i),
                    float(np.linalg.norm(est.state.p(i) - pre[i])),
                    float(est.state.r[2] - r_pre[2]),
                    float(np.linalg.norm(est.state.v - v_pre)),
                ))
        prev_c = contacts

        out = est.outputs(last_w_meas=imu_w[min(ii, len(imu_w)) - 1])
        re_, pe = _roll_pitch(out["q"])
        res["t"].append(te)
        res["r"].append(out["r"].copy())
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
        res["n_contacts"].append(int(contacts.sum()))
        detail = out.get("health_detail")
        res["sigma_v"].append(detail["sigma_v"] if detail else np.nan)
        res["max_diag_P"].append(detail["max_diag_P"] if detail else np.nan)
        res["fail_term"].append(failing_term(detail) if detail else "?")

    for k in ("t", "z", "roll_e", "pitch_e", "roll_a", "pitch_a", "min_eig",
              "sigma_v", "max_diag_P", "n_contacts"):
        res[k] = np.array(res[k])
    for k in ("v", "bf", "bw", "r"):
        res[k] = np.array(res[k])
    res["healthy"] = np.array(res["healthy"])
    res["fail_term"] = np.array(res["fail_term"])
    res["innov"] = innov
    res["touchdowns"] = touchdowns
    return res


def health_window_start(res):
    """First frame of the post-re-anchor window.

    The gait logs open with a contacts-off STAND rise: the EKF dead-reckons
    on purpose there, so sigma_v grows and health is not meaningful.  The
    window starts at the first all-4-contact frame AFTER the first all-off
    frame; if the log never drops all contacts (a pure static log), it starts
    at 0 and nothing is excluded.
    """
    n_c = res["n_contacts"]
    off = np.flatnonzero(n_c == 0)
    if len(off) == 0:
        return 0
    after = np.flatnonzero(n_c[off[-1]:] == 4)
    if len(after) == 0:
        return len(n_c)          # never re-anchored: nothing to gate
    return int(off[-1] + after[0])


def load_command(npz_path):
    """Read the walk_cmd_hw sidecar (<log>.cmd.json) if it sits next to the log."""
    side = os.path.splitext(npz_path)[0] + ".cmd.json"
    if not os.path.exists(side):
        return None
    try:
        with open(side) as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"  [warn] sidecar {side} unreadable: {exc}")
        return None


def _clusters(t, mask, gap_s=0.5):
    """Contiguous runs of True in `mask`, split on gaps > gap_s."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    splits = np.flatnonzero(np.diff(t[idx]) > gap_s)
    return np.split(idx, splits + 1)


def closeout_report(res, cmd=None, measured=None, measured_dz=None,
                    static=False, health_detail=False):
    """The corrected/new gates.  Returns (all_passed, healthy_masked)."""
    ok = True
    t = res["t"] - res["t"][0]
    n = len(t)

    def line(name, passed, detail=""):
        nonlocal ok
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail else ""))

    print("---- close-out gates ----")

    # -- health on the post-re-anchor window --------------------------------
    i0 = health_window_start(res)
    healthy_masked = res["healthy"].copy()
    if i0 > 0:
        healthy_masked[:i0] = True      # excluded from the legacy gate too
        print(f"    health window: excluding t < {t[i0]:.2f}s "
              f"({i0} frames, contacts-off rise -- EKF dead-reckons by design)")
    frac = float(np.mean(res["healthy"][i0:])) if i0 < n else 1.0
    line("healthy after re-anchor (>=99%)", frac > 0.99, f"{frac * 100:.1f}%")

    if health_detail:
        bad = ~res["healthy"]
        print(f"    unhealthy frames: {int(bad.sum())}/{n} "
              f"({int((~res['healthy'][i0:]).sum())} inside the gated window)")
        for cl in _clusters(t, bad):
            terms = ", ".join(sorted(set(res["fail_term"][cl])))
            print(f"      {t[cl[0]]:7.2f}-{t[cl[-1]]:7.2f}s  {len(cl):4d} frames"
                  f"  term={terms}  n_contacts={int(np.min(res['n_contacts'][cl]))}"
                  f"-{int(np.max(res['n_contacts'][cl]))}"
                  f"  sigma_v<={1e3 * np.nanmax(res['sigma_v'][cl]):.0f} mm/s"
                  f"  maxdiagP<={np.nanmax(res['max_diag_P'][cl]):.1e}")

    # -- static hygiene ------------------------------------------------------
    if static:
        tail = slice(int(0.3 * n), None)
        vmag = np.linalg.norm(res["v"][tail], axis=1)
        v95 = float(np.percentile(vmag, 95))
        line(f"static: 95th-pct |v| < {STATIC_V_LIMIT_MMS:.0f} mm/s",
             v95 * 1e3 < STATIC_V_LIMIT_MMS,
             f"{v95 * 1e3:.1f} mm/s (max {np.max(vmag) * 1e3:.1f}, "
             f"median {np.median(vmag) * 1e3:.1f})")
        zdrift = float(np.max(np.abs(res["z"][tail] - np.mean(res["z"][tail]))))
        line(f"static: z drift < {STATIC_Z_DRIFT_MM:.0f} mm",
             zdrift * 1e3 < STATIC_Z_DRIFT_MM, f"{zdrift * 1e3:.1f} mm")

    # -- horizontal displacement --------------------------------------------
    d_ekf, w = walk_displacement(res)
    print(f"    displacement window: t {w[0]:.1f}s -> {w[1]:.1f}s "
          f"(settled after the re-anchor; the contacts-off rise before it "
          f"dead-reckons {np.linalg.norm(w[2]) * 1e3:.0f} mm and is excluded)")
    print(f"    EKF xy displacement: ({d_ekf[0] * 1e3:+.1f}, "
          f"{d_ekf[1] * 1e3:+.1f}) mm  |d| {np.linalg.norm(d_ekf) * 1e3:.1f} mm")
    if cmd is not None:
        c = np.asarray(cmd["quantized_dxy_m"], float)
        e = d_ekf - c
        pct = (100.0 * np.linalg.norm(e) / np.linalg.norm(c)
               if np.linalg.norm(c) > 1e-6 else float("nan"))
        print(f"    commanded xy:        ({c[0] * 1e3:+.1f}, {c[1] * 1e3:+.1f}) mm"
              f"  heading {cmd['heading_deg']:+.1f} deg, {cmd['n_cycles']} cycles")
        print(f"    EKF - commanded:     ({e[0] * 1e3:+.1f}, {e[1] * 1e3:+.1f}) mm"
              f"  |e| {np.linalg.norm(e) * 1e3:.1f} mm ({pct:.1f}% of commanded)"
              "  [info: commanded != truth, the robot slips]")
    if measured is not None:
        mm = np.asarray(measured, float) / 1e3        # mm -> m
        e = d_ekf - mm
        dist = float(np.linalg.norm(mm))
        emag = float(np.linalg.norm(e))
        lim = max(DRIFT_FLOOR_MM / 1e3, DRIFT_FRAC * dist)
        print(f"    tape-measured xy:    ({mm[0] * 1e3:+.1f}, {mm[1] * 1e3:+.1f})"
              f" mm  |d| {dist * 1e3:.1f} mm")
        line(f"EKF vs tape <= max({DRIFT_FLOOR_MM:.0f} mm, "
             f"{DRIFT_FRAC * 100:.0f}% of distance)", emag <= lim,
             f"error ({e[0] * 1e3:+.1f}, {e[1] * 1e3:+.1f}) mm, |e| "
             f"{emag * 1e3:.1f} mm = {100 * emag / max(dist, 1e-9):.1f}% "
             f"of {dist * 1e3:.0f} mm; limit {lim * 1e3:.0f} mm")

    # -- height cross-check --------------------------------------------------
    if measured_dz is not None:
        dz = ekf_stand_dz(res)
        if dz is None:
            line("height cross-check", False,
                 "no contacts-off rise found in this log (need a stand)")
        else:
            err = dz - measured_dz / 1e3
            line(f"EKF stand dz vs tape (<= {DZ_TOL_MM:.0f} mm)",
                 abs(err) * 1e3 <= DZ_TOL_MM,
                 f"EKF {dz * 1e3:+.1f} mm vs tape {measured_dz:+.1f} mm, "
                 f"error {err * 1e3:+.1f} mm")

    return ok, healthy_masked


SETTLE_S = 3.0          # re-anchor transient (measured: settles in 2-3 s)
MEDIAN_WIN_S = 1.0      # averaging window at each end of the walk


def walk_displacement(res):
    """EKF xy displacement over the WALK, excluding the stand rise.

    Two effects make frame 0 the wrong origin:
      * during the contacts-off STAND rise the filter dead-reckons on the
        IMU alone and slides 150-550 mm (unobservable by construction);
      * the re-anchor at the end of the rise pulls the estimate back over a
        2-3 s transient.
    So the origin is a median over a 1 s window starting SETTLE_S after the
    re-anchor, and the end is a median over the last 1 s.  On a walk log this
    is exactly the interval the operator can also tape-measure: standing at
    HOLD4 before the first step, to standing at HOLD4 after the last.

    Returns (displacement_xy, (t_start, t_end, rise_drift_xy)).
    """
    t = res["t"] - res["t"][0]
    i0 = health_window_start(res)
    i0 = min(i0, len(t) - 1)
    t_start = min(t[i0] + SETTLE_S, t[-1] - MEDIAN_WIN_S) if len(t) > 1 else t[0]
    start_m = (t >= t_start) & (t <= t_start + MEDIAN_WIN_S)
    end_m = t >= t[-1] - MEDIAN_WIN_S
    if not start_m.any():
        start_m = np.zeros_like(t, dtype=bool)
        start_m[i0] = True
    origin = np.median(res["r"][start_m][:, :2], axis=0)
    end = np.median(res["r"][end_m][:, :2], axis=0)
    rise = res["r"][i0][:2] - res["r"][0][:2]
    return end - origin, (float(t_start), float(t[-1]), rise)


def ekf_stand_dz(res):
    """EKF height change across the stand rise.

    The rise is the contacts-off window (walk1 drops all contacts during
    STAND).  Take the median z over the static crouch prefix before it and
    over the settled all-contact window after it -- medians reject the
    transient without needing stage labels in the log.
    """
    n_c = res["n_contacts"]
    off = np.flatnonzero(n_c == 0)
    if len(off) == 0:
        return None
    i_pre, i_post = int(off[0]), int(off[-1])
    pre = res["z"][:i_pre]
    post = res["z"][i_post:]
    if len(pre) < 5 or len(post) < 5:
        return None
    # settle margin: skip the first 20 % after the rise (foot re-anchor)
    post = post[int(0.2 * len(post)):]
    return float(np.median(post) - np.median(pre))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--static", action="store_true",
                    help="dog held still: gate 95th-pct |v| and z drift")
    ap.add_argument("--gait", action="store_true",
                    help="walking log: relaxed legacy thresholds + touchdowns")
    ap.add_argument("--measured", type=str, default=None,
                    help='tape-measured actual body displacement, mm: "X,Y" '
                         "(X along the commanded heading, Y lateral)")
    ap.add_argument("--measured-dz", type=float, default=None,
                    help="tape-measured crouch->stand height change, mm")
    ap.add_argument("--health-detail", action="store_true",
                    help="dump every unhealthy cluster with its failing term")
    args = ap.parse_args()

    measured = None
    if args.measured:
        parts = [p for p in args.measured.replace(",", " ").split() if p]
        if len(parts) != 2:
            print('error: --measured wants "X,Y" in mm', file=sys.stderr)
            return 2
        measured = [float(p) for p in parts]

    print("=" * 66)
    print(f"EKF close-out replay -- {args.npz}")
    print("=" * 66)
    data = load(args.npz)
    res = replay(data)
    cmd = load_command(args.npz)
    if cmd is None and (args.measured or args.gait):
        print("    [info] no <log>.cmd.json sidecar (run walk_cmd_hw.py to "
              "get one); commanded displacement unavailable")

    ok_new, healthy_masked = closeout_report(
        res, cmd=cmd, measured=measured, measured_dz=args.measured_dz,
        static=args.static, health_detail=args.health_detail)

    # Legacy gates for continuity (attitude / innovation / per-leg offset /
    # bias / touchdowns).  Health is fed the windowed flags; the static |v|
    # and z gates are superseded above, so they are not re-run here.
    print("---- standard hw_replay gates ----")
    res_legacy = dict(res)
    res_legacy["healthy"] = healthy_masked
    ok_legacy = hw_replay.report(res_legacy, static=False, gait=args.gait)

    ok = ok_new and ok_legacy
    print("=" * 66)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
