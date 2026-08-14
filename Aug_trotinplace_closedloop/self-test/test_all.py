#!/usr/bin/env python3
"""Run every offline self-test in this directory and summarise the verdicts.

Each suite runs in its OWN PROCESS.  That costs a few table builds -- the
suites cannot share `selftest_common.tables_for`'s cache across a process
boundary -- and buys the thing worth more: a suite cannot be passed a module
another suite has already poked at, so a green run here means each script is
green when run alone, which is how they are actually used.

Run:  $V test_all.py            # everything, ~2-4 min (table builds dominate)
      $V test_all.py level ahrs # only the suites whose name contains these
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

SUITES = [
    ("params   the single source of truth", "test_stand_params.py"),
    ("stage 1  read-only EKF", "test_stand_ekf_verify.py"),
    ("stage 2  closed-loop height", "test_stand_ekf_height.py"),
    ("stage 2b attitude hold", "test_stand_ekf_level.py"),
    ("exp. A   AHRS-only leveling", "test_stand_ahrs_level.py"),
    ("exp. B   scheduled contacts", "test_stand_ekf_schedcontact.py"),
    ("lift     one foot off the floor", "test_lift_ekf_contact.py"),
    ("stage 3  FK-switched trot in place", "test_trot_fk_switch.py"),
    # ---- torque track (torque_mode_control/) --------------------------------
    ("T params the torque source of truth", "test_torque_params.py"),
    ("T0       exact quasi-static model", "test_dog5_statics.py"),
    ("T0       AHRS + leg-odometry body state", "test_body_state.py"),
    ("T1       impedance floor + stance law", "test_stance_law.py"),
    ("T2       torque runners: seq, brake, load", "test_torque_runners.py"),
]


def run_suite(script, echo):
    """Run one suite; return (rc, n_pass, n_fail, seconds)."""
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, os.path.join(_HERE, script)],
                          capture_output=True, text=True)
    dt = time.perf_counter() - t0
    out = proc.stdout + proc.stderr
    if echo:
        print(out, end="" if out.endswith("\n") else "\n")
    n_pass = out.count("[PASS]")
    n_fail = out.count("[FAIL]")
    return proc.returncode, n_pass, n_fail, dt, out


def main():
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")]
    echo = "-q" not in sys.argv[1:]
    suites = [(t, s) for t, s in SUITES
              if not wanted or any(w in s for w in wanted)]
    if not suites:
        print(f"no suite matches {wanted}; known: "
              + ", ".join(s for _, s in SUITES))
        return 2

    rows, rc_total = [], 0
    for title, script in suites:
        print(f"\n{'='*78}\n=== {title}  ({script})\n{'='*78}")
        rc, n_pass, n_fail, dt, out = run_suite(script, echo)
        rc_total |= rc
        rows.append((script, rc, n_pass, n_fail, dt))
        if rc and not echo:
            print(out, end="")

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    total_pass = total_fail = 0
    for script, rc, n_pass, n_fail, dt in rows:
        total_pass += n_pass
        total_fail += n_fail
        print(f"  [{'PASS' if rc == 0 else 'FAIL'}] {script:34s} "
              f"{n_pass:3d} gates  {dt:5.1f}s"
              + (f"  {n_fail} FAILED" if n_fail else ""))
    print(f"  {total_pass} gates across {len(rows)} suites, "
          f"{total_fail} failed")
    return 1 if rc_total else 0


if __name__ == "__main__":
    sys.exit(main())
