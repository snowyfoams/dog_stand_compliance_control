#!/usr/bin/env python3
"""Turn a torque-stand / trot .npz log into something a spreadsheet opens.

    $V tools_npz_to_csv.py t1.npz t2.npz t3.npz t5.npz
    $V tools_npz_to_csv.py *.npz --outdir csv

Each log becomes TWO files, because the npz holds two different kinds of
thing and putting them in one table is what makes a log unreadable:

    <name>.csv        one row per 4 ms sweep.  The (n,12) joint arrays and the
                      (n,4) per-leg arrays are FLATTENED INTO NAMED COLUMNS --
                      tau_cmd_FL_knee_Nm, fz_RR_N -- so a column header says
                      which leg and which joint without a lookup table.
    <name>_meta.csv   the run-level scalars: the gains the run was launched
                      with, the mass, the setpoints, the per-joint glitch
                      counts.  These are ONE value for the whole run, so as
                      columns beside 4000 rows they would just be repeated
                      noise, and as a header comment pandas would choke.

UNITS ARE IN THE COLUMN NAME, always.  The logs are SI (rad, m, N, Nm) but
every analysis anyone has actually run on them was in degrees, and a roll
column that does not say which one it is has already cost one wrong reading.
So roll/pitch ship BOTH: roll_rad and roll_deg, from the same samples.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

_R = "/home/robot01/Documents/can_motor_control"
for _p in (os.path.join(_R, "dog_stand_compliance_control", "dog5_description"), _R):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import stand_dog5_hw as _base
    JOINTS = list(_base.JOINT_LABELS)
except Exception:                                            # noqa: BLE001
    # the labels are a naming convention, not hardware -- a missing CAN stack
    # must not stop a log being read
    JOINTS = [f"{leg}_{j}" for leg in ("FL", "FR", "RL", "RR")
              for j in ("abd", "pitch", "knee")]

LEGS = ("FL", "FR", "RL", "RR")
XYZ = ("x", "y", "z")
WRENCH = ("fx_N", "fy_N", "fz_N", "mx_Nm", "my_Nm", "mz_Nm")

# per-sample fields: npz key -> (column prefix, unit suffix, sub-labels)
VEC = {
    "q":        ("q",        "rad",   JOINTS),
    "q_ref":    ("q_ref",    "rad",   JOINTS),
    "qd":       ("qd",       "rad_s", JOINTS),
    "qd_drv":   ("qd_drv",   "rad_s", JOINTS),
    "tau_cmd":  ("tau_cmd",  "Nm",    JOINTS),
    "tau_meas": ("tau_meas", "Nm",    JOINTS),
    "fz":       ("fz",       "N",     LEGS),
    "planted":  ("planted",  "",      LEGS),
    "w":        ("w",        "rad_s", XYZ),
    "v":        ("v",        "m_s",   XYZ),
}
SCALAR = {
    "t": "t_s", "z": "z_m", "z_hip": "z_hip_m", "load": "load_N",
    "stage": "stage",
}
# logged in rad; every reading of them has been in deg, so emit both
ANGLE = ("roll", "pitch", "roll_fk", "pitch_fk", "roll_raw", "pitch_raw")


def columns(d):
    """(header, list-of-1-D-arrays) in a fixed, readable order."""
    head, cols = [], []

    def put(name, arr):
        head.append(name)
        cols.append(arr)

    put("t_s", d["t"])
    put("t_rel_s", d["t"] - d["t"][0])          # the axis you actually plot
    if "stage" in d.files:
        put("stage", d["stage"].astype(str))
    for k in ("z", "z_hip", "load"):
        if k in d.files:
            put(SCALAR[k], d[k])
    for k in ANGLE:
        if k in d.files:
            put(f"{k}_rad", d[k])
            put(f"{k}_deg", np.degrees(d[k]))
    for k, (pre, unit, labels) in VEC.items():
        if k not in d.files:
            continue
        a = np.atleast_2d(d[k])
        if a.shape[1] != len(labels):
            continue
        for i, lab in enumerate(labels):
            put(f"{pre}_{lab}" + (f"_{unit}" if unit else ""), a[:, i])
    if "W" in d.files:
        W = np.atleast_2d(d["W"])
        for i, lab in enumerate(WRENCH[:W.shape[1]]):
            put(f"W_{lab}", W[:, i])
    if "f_foot" in d.files:
        # (n, 4, 3) BODY-frame force per foot, post-clamp.  Twelve columns and
        # worth every one: W_* above is what the PD law ASKED for, and these
        # are what the feet were given after distribute()'s clamps.  G f is the
        # achieved wrench, and its Mz is the only record of the yaw couple.
        a = np.asarray(d["f_foot"], dtype=float).reshape(len(d["t"]), 4, 3)
        for i, lab in enumerate(LEGS):
            for k, ax in enumerate(XYZ):
                put(f"f_{lab}_{ax}_N", a[:, i, k])
    if "step_xy" in d.files:
        # (n, 4, 2) and so too deep for VEC, which is a table of 2-D fields.
        # EMITTED IN MILLIMETRES: the reach clamp at the stand height is about
        # 10 mm, so every honest value in this column is single-digit, and a
        # metres column reads as a screenful of zeros next to v in m/s.
        s = np.asarray(d["step_xy"], dtype=float).reshape(len(d["t"]), -1)
        for i, lab in enumerate(LEGS):
            put(f"step_{lab}_x_mm", s[:, 2 * i] * 1e3)
            put(f"step_{lab}_y_mm", s[:, 2 * i + 1] * 1e3)
    return head, cols


def meta_rows(d, n):
    rows = []
    for k in d.files:
        a = d[k]
        if a.ndim == 0:
            rows.append((k, "", float(a)))
        elif a.ndim == 1 and a.shape[0] == len(JOINTS) and a.shape[0] != n:
            for i, lab in enumerate(JOINTS):          # glitches
                rows.append((k, lab, float(a[i])))
    rows.append(("n_samples", "", float(n)))
    dt = float(np.median(np.diff(d["t"]))) if n > 1 else float("nan")
    rows.append(("dt_median_s", "", dt))
    rows.append(("duration_s", "", float(d["t"][-1] - d["t"][0])))
    if "stage" in d.files:
        st = d["stage"].astype(str)
        for s in ("WAIT", "RISE", "HOLD", "TROT"):
            rows.append(("stage_duration_s", s, float((st == s).sum() * dt)))
    return rows


def convert(path, outdir):
    d = np.load(path, allow_pickle=True)
    n = len(d["t"])
    stem = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(outdir, exist_ok=True)

    head, cols = columns(d)
    # t_s IS AN ABSOLUTE perf_counter READING -- ~1e4 s, so %.6g quantises it
    # to 0.1 s and the 4 ms sweep vanishes.  It gets full precision; every
    # other column is a physical quantity where 6 figures is already more than
    # the sensor has.
    prec = ["%.12g" if h == "t_s" else "%.6g" for h in head]
    out = os.path.join(outdir, stem + ".csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for r in range(n):
            w.writerow(["" if c.dtype.kind != "U" and not np.isfinite(c[r])
                        else (c[r] if c.dtype.kind == "U" else f % c[r])
                        for c, f in zip(cols, prec)])

    mout = os.path.join(outdir, stem + "_meta.csv")
    with open(mout, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "label", "value"])
        for k, lab, v in meta_rows(d, n):
            w.writerow([k, lab, f"{v:.6g}"])

    print(f"{path:22s} -> {out}  ({n} rows x {len(head)} cols)  + {os.path.basename(mout)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--outdir", default="csv")
    a = ap.parse_args()
    missing = [p for p in a.logs if not os.path.exists(p)]
    for p in missing:
        print(f"[skip] {p}: not on disk", file=sys.stderr)
    for p in a.logs:
        if p not in missing:
            convert(p, a.outdir)


if __name__ == "__main__":
    main()
