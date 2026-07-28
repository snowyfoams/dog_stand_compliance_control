"""Headless acceptance gates for the DOG5 VMC + estimator MuJoCo loop.

Run:  .venv/bin/python test_vmc.py
Exits non-zero if any gate fails.

  V1  STAND        trunk holds height & stays level on four feet
  V2  ESTIMATOR    estimator tracks MuJoCo ground truth online (z, v, roll/pitch)
  V3  DIAGONAL     completes the diagonal step sequence without falling; feet lift
  V4  ABLATION     a trunk push is handled better with the estimator live than
                   with it frozen  (the estimator is load-bearing, not decorative)
"""
from __future__ import annotations

import sys

import numpy as np

import vmc_mujoco as H

_FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _FAIL.append(name)
    return ok


def test_stand_estimator_step():
    """V1+V2+V3 all come from a single nominal stand->hold->step run."""
    print("V1/V2/V3 — stand, estimator tracking, diagonal stepping")
    m = H.run()
    hold = m["hold_mask"]

    # V1 STAND: on four feet the trunk neither sags nor tilts.
    z_sag = float(np.max(np.abs(m["z_true"][hold])))
    tilt_hold = float(np.max(m["tilt_true_deg"][hold]))
    check("V1 stand: trunk height holds (<1 cm sag)", z_sag < 0.01, f"{z_sag*1e3:.1f} mm")
    check("V1 stand: trunk stays level (<2 deg)", tilt_hold < 2.0, f"{tilt_hold:.2f} deg")

    # V2 ESTIMATOR-IN-THE-LOOP: online tracking vs MuJoCo truth (C7 thresholds).
    check("V2 est: z error < 3 cm", m["z_err"].max() < 0.03, f"{m['z_err'].max()*1e3:.1f} mm")
    check("V2 est: |v| error < 0.05 m/s", m["v_err"].max() < 0.05,
          f"{m['v_err'].max()*1e3:.1f} mm/s")
    check("V2 est: roll/pitch error < 2 deg", m["rp_err_deg"].max() < 2.0,
          f"{m['rp_err_deg'].max():.2f} deg")
    check("V2 est: healthy throughout", m["healthy_frac"] > 0.99, f"{m['healthy_frac']:.3f}")

    # V3 DIAGONAL STEP: does not fall, feet actually lift, tilt stays bounded.
    check("V3 step: did not fall", not m["fell"], f"final z={m['final_z']:.3f}")
    check("V3 step: feet lifted (swing samples > 0)", m["n_swing"] > 100, f"{m['n_swing']}")
    check("V3 step: peak tilt bounded (<10 deg)", m["tilt_true_deg"].max() < 10.0,
          f"{m['tilt_true_deg'].max():.2f} deg")


def _recovery_stats(m, t0, t1):
    """Peak & RMS trunk tilt and height sag over the post-push recovery window."""
    tv = m["tv"]
    mask = (tv >= t0) & (tv < t1)
    tilt = m["tilt_true_deg"][mask]
    sag = np.abs(m["z_true"][mask])
    return float(tilt.max()), float(np.sqrt(np.mean(tilt ** 2))), float(sag.max())


def test_ablation():
    """V4 — estimator feedback demonstrably helps reject a disturbance.

    A clean lateral shove on the trunk during the four-foot hold. With the
    estimator live the VMC observes the tilt/velocity and rejects it; with the
    estimate frozen it is effectively blind and the body tilts and sags.
    """
    print("V4 — ablation: estimator live vs frozen under a trunk push")
    push = {"t0": 1.5, "dur": 0.15, "force": np.array([20.0, 0.0, 0.0])}
    live = H.run(push=push, mode_after_hold="STAND")       # stay on 4 feet to isolate the push
    frozen = H.run(push=push, mode_after_hold="STAND", freeze_estimator=True)

    lp, lr, lsag = _recovery_stats(live, 1.5, 3.0)
    fp, fr, fsag = _recovery_stats(frozen, 1.5, 3.0)
    print(f"    recovery (post-push):  live peak={lp:.2f} rms={lr:.2f} sag={lsag*1e3:.0f}mm | "
          f"frozen peak={fp:.2f} rms={fr:.2f} sag={fsag*1e3:.0f}mm")
    check("V4: live run survives the push", not live["fell"], f"final z={live['final_z']:.3f}")
    check("V4: live stays near-level (<3 deg)", lp < 3.0, f"{lp:.2f} deg")
    check("V4: estimator feedback beats frozen (tilt)", lp < fp - 3.0,
          f"live {lp:.2f} << frozen {fp:.2f} deg")
    check("V4: estimator feedback beats frozen (height sag)", lsag < fsag - 0.01,
          f"live {lsag*1e3:.0f}mm << frozen {fsag*1e3:.0f}mm")


def main():
    print("=" * 60)
    print("DOG5 VMC + estimator — MuJoCo acceptance gates")
    print("=" * 60)
    test_stand_estimator_step()
    test_ablation()
    print("=" * 60)
    if _FAIL:
        print(f"FAILED ({len(_FAIL)}): " + ", ".join(_FAIL))
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
