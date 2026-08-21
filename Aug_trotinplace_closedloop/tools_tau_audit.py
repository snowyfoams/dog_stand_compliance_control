#!/usr/bin/env python3
"""How much of the control law actually reached the motors, per joint.

    $V tools_tau_audit.py run_20260821_153000.npz
    $V tools_tau_audit.py *.npz --stage TROT

ONE TORQUE, THREE STATIONS, TWO DIFFERENT FAILURES.  Every sweep the npz
carries the same 12-vector at three points in its life:

    tau_des    what the control law ASKED for (feedforward + impedance,
               before the gate).  Logged since 2026-08-21; absent before.
    tau_cmd    what TorqueGate let through: the arming ramp, the --tau-max
               cap, the hard 9 Nm clip and the slew limit, in that order.
    tau_meas   what the driver reports actually producing (iq scaled).

    des vs cmd   THE GATE'S BITE.  A law that is right but 60% clipped
                 behaves exactly like a law that is wrong, and nothing on
                 the status line separates them.  This is the number that
                 says whether a bad run needs a gain change or a cap raise.
    cmd vs meas  THE DRIVERS' TRACKING.  Slack here is below the model --
                 torque constant, current loop, CAN timing -- and no gain
                 or cap on the Pi can change it.

WHAT "CLIPPED" IS SPLIT INTO, because the fixes differ:
    cap    |tau_cmd| pinned at the ceiling (>= 99% of tau_max).  The law
           wants more torque than the run allows: raise --tau-max (staged!)
           or lower the gains asking for it.
    slew   the sweep-to-sweep step pinned at TAU_SLEW_NM_S * dt.  The law
           wants torque to CHANGE faster than allowed: the handover ramp,
           CONTACT_RAMP, or a gain acting on a noisy signal.
    ramp   inside the first TORQUE_RAMP_S after arming, where the cap is
           still rising from 0.  Expected at every arm; only a finding if
           it shows up later, which this tool's windows would reveal.

Reads like tools_npz_to_csv: no venv, no robot, no CAN stack needed.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

JOINTS = [f"{leg}_{j}" for leg in ("FL", "FR", "RL", "RR")
          for j in ("abd", "pitch", "knee")]

# Stages that command torque, in run order.  WAIT/CROUCH/PARK hold no torque
# and would only dilute every statistic with rows of zeros.
TORQUE_STAGES = ("RISE", "HOLD", "TROT")


def _rms(a, axis=0):
    return np.sqrt(np.mean(np.square(a), axis=axis))


def audit_one(d, stages):
    """Per-stage, per-joint audit of one loaded npz.  Returns report lines."""
    out = []
    t = np.asarray(d["t"], dtype=float)
    stage = d["stage"].astype(str)
    cmd = np.asarray(d["tau_cmd"], dtype=float)
    meas = np.asarray(d["tau_meas"], dtype=float)
    des = np.asarray(d["tau_des"], dtype=float) if "tau_des" in d.files \
        else None
    if des is None:
        out.append("  NOTE: no tau_des in this log (written before "
                   "2026-08-21) -- gate-bite half skipped, driver half only")
    tau_max = float(d["tau_max"]) if "tau_max" in d.files else float("nan")
    slew = (float(d["cfg_TAU_SLEW_NM_S"]) if "cfg_TAU_SLEW_NM_S" in d.files
            else 60.0)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else float("nan")
    out.append(f"  cap {tau_max:.2f} Nm, slew {slew:.0f} Nm/s "
               f"({slew*dt:.3f} Nm per {dt*1e3:.1f} ms sweep)")

    for st in stages:
        m = stage == st
        n = int(m.sum())
        if n == 0:
            continue
        out.append(f"  -- {st}: {n} sweeps ({n*dt:.1f} s) " + "-" * 38)
        c, me = cmd[m], meas[m]
        de = des[m] if des is not None else None

        if de is not None:
            gap = de - c
            # a real clip, not float dust: the smallest deliberate torque in
            # this stack is the 0.01-level impedance term, so 1 mNm separates
            # "the gate acted" from "the gate copied".
            clipped = np.abs(gap) > 1e-3
            at_cap = clipped & (np.abs(c) >= 0.99 * tau_max)
            frac = clipped.mean(axis=0)
            worst = int(np.argmax(frac))
            out.append(f"     gate bite: clipped {clipped.mean()*100:5.1f}% "
                       f"of joint-sweeps ({at_cap.mean()*100:.1f}% at the "
                       f"cap), worst {JOINTS[worst]} {frac[worst]*100:.1f}%")
            out.append(f"     |des-cmd|: rms {_rms(gap).max():.3f} Nm "
                       f"(worst joint), max {np.abs(gap).max():.2f} Nm; "
                       f"|des| max {np.abs(de).max():.2f} Nm vs cap "
                       f"{tau_max:.2f}")
        trk = c - me
        w = int(np.argmax(_rms(trk)))
        out.append(f"     driver:    |cmd-meas| rms {_rms(trk)[w]:.3f} Nm "
                   f"(worst {JOINTS[w]}), max {np.abs(trk).max():.2f} Nm")

        if de is not None:
            per = np.stack([_rms(de), _rms(c), _rms(me), _rms(de - c),
                            clipped.mean(axis=0) * 100.0], axis=1)
            out.append("     joint        rms_des  rms_cmd  rms_meas "
                       "rms_bite  clip%")
            for j, lab in enumerate(JOINTS):
                out.append(f"     {lab:<9s}" + "".join(
                    f"{per[j, k]:9.3f}" for k in range(4))
                    + f"{per[j, 4]:7.1f}")
    return out


def self_test():
    """The clip bookkeeping on a synthetic log, where the answer is known."""
    n = 1000
    des = np.zeros((n, 12))
    des[:, 0] = 2.0                       # FL_abd asks 2.0 against a 1.0 cap
    des[:, 1] = 0.5                       # FL_pitch inside the cap
    cmd = np.clip(des, -1.0, 1.0)
    d = {"t": np.arange(n) * 0.004, "stage": np.array(["TROT"] * n),
         "tau_des": des, "tau_cmd": cmd, "tau_meas": cmd * 0.9,
         "tau_max": np.asarray(1.0), "cfg_TAU_SLEW_NM_S": np.asarray(60.0)}

    class _D(dict):
        files = list(d)
    lines = audit_one(_D(d), ("TROT",))
    text = "\n".join(lines)
    ok = ("clipped   8.3% of joint-sweeps" in text      # 1 of 12 joints
          and "worst FL_abd 100.0%" in text
          and "max 1.00 Nm" in text)                    # the 2.0 - 1.0 clip
    print(text)
    print("self-test", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--stage", choices=TORQUE_STAGES, default=None,
                    help="audit one stage only (default: each that ran)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.logs:
        ap.error("give at least one npz (or --self-test)")
    stages = (a.stage,) if a.stage else TORQUE_STAGES
    for p in a.logs:
        if not os.path.exists(p):
            print(f"[skip] {p}: not on disk", file=sys.stderr)
            continue
        d = np.load(p, allow_pickle=True)
        print(f"{p}:")
        for line in audit_one(d, stages):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
