#!/usr/bin/env python3
"""Shared harness and fixtures for the Aug_trotinplace_closedloop self-tests.

WHY THIS EXISTS
    Every runner grew its own copy of `_FAIL` + `_check` -- six of them, two
    of which had drifted into different line wrapping.  Worse, the plane
    plant, the rotation helper and the fake EKF lived inside
    `stand_ekf_level_hw.py` -- a HARDWARE script -- because that is where they
    were first needed, and two other runners then reached in to alias them
    out (`_PlanePlant = lv._PlanePlant`).  A control script could not be
    edited without first checking what its tests had borrowed from it.

    Everything test-only now lives here.  The hardware scripts carry hardware
    code, the `test_*.py` scripts carry gates, and the things both need are
    imported from one place.

WHAT IS HERE
    harness    check / report / raises           -- the PASS/FAIL protocol
    fixtures   PlanePlant, C_from_rp, FakeShared, FakeAhrs, FakeFeed
    helpers    tables_for (cached!), anchors_of, all_planted, flat_base_z,
               RateWatch, walk_stages, time_sweep
    geometry   PLANT_TO_BOTTOM and the frame note that goes with it

    Nothing in here talks to CAN, the IMU or the EKF worker.  It imports the
    stage-1 and stage-2 modules only for the constants and the table builder
    they own.

WHO USES IT
    test_stand_ekf_verify.py        test_stand_ekf_level.py
    test_stand_ekf_height.py        test_stand_ekf_schedcontact.py
    test_stand_ahrs_level.py        test_lift_ekf_contact.py
    test_all.py runs the lot.
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNERS = os.path.join(os.path.dirname(_HERE), "stand_postion_mode")
# ^ the hardware scripts live in stand_postion_mode/, beside this directory
for _p in (_HERE, _RUNNERS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# stage 1 owns the path bootstrap and the frame constants; stage 2 owns the
# leg tables.  Nothing else from the hardware scripts belongs in here.
import stand_ekf_verify_hw as s1                         # noqa: E402
import stand_ekf_height_hw as s2                         # noqa: E402
import stand_dog5_recorded_hw as recorded                # noqa: E402
# tunables come from the ONE place the runners get them from, not from
# whichever runner happened to re-export them
from stand_params import (FOOT_RADIUS_M,                 # noqa: E402
                          IMU_BELOW_TRUNK_ORIGIN_M, T_STAND)

LEGS = s1.LEGS
N_JOINTS = s1.N_JOINTS
CONTROL_HZ = s1.CONTROL_HZ
Q_CROUCH = s1.Q_CROUCH

DT = 1.0 / CONTROL_HZ            # one control sweep; every gate steps by this
SLOT_BUDGET_S = 250e-6           # the CAN slot a per-sweep block must fit in

# `PlanePlant.solve` returns the trunk ORIGIN (abd axis) above the FOOT SITES.
# The height loop works in floor-to-trunk-bottom, so converting costs a foot
# radius up to the floor and an IMU lever back down.  Both closed-loop gates
# (stage 2b and the lift) used to carry this line and its comment separately.
PLANT_TO_BOTTOM = FOOT_RADIUS_M - IMU_BELOW_TRUNK_ORIGIN_M


# ===========================================================================
# harness
# ===========================================================================

_FAIL = []


def check(name, ok, detail=""):
    """Report one gate.  Returns the verdict so callers can chain on it."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))
    if not ok:
        _FAIL.append(name)
    return bool(ok)


def raises(fn):
    """True if `fn()` throws -- for gates that assert an input is refused."""
    try:
        fn()
    except Exception:
        return True
    return False


def report():
    """Print the verdict line, clear the failures, return the exit code.

    Clearing matters: `test_all.py` may run several suites in one process,
    and a suite must not inherit the previous one's failures.
    """
    print("self-test " + ("FAIL: " + ", ".join(_FAIL) if _FAIL else "PASS"))
    rc = 1 if _FAIL else 0
    _FAIL.clear()
    return rc


def failures():
    """The gates that have failed so far (live view, for test_all)."""
    return list(_FAIL)


# ===========================================================================
# fixtures -- the fake world the gates run against
# ===========================================================================

def C_from_rp(roll, pitch):
    """An I->B rotation whose _rp() is exactly (roll, pitch): Rodrigues
    rotation taking e_z to the gravity direction implied by (roll, pitch)."""
    g = np.array([-math.sin(pitch),
                  math.cos(pitch) * math.sin(roll),
                  math.cos(pitch) * math.cos(roll)])
    e = np.array([0.0, 0.0, 1.0])
    v = np.cross(e, g)
    c = float(np.dot(e, g))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


class PlanePlant:
    """Rigid trunk on stiff position legs, feet on a flat floor.

    Given commanded foot z per leg (trunk frame) and the planted set, solves
    the small-angle pose (h, roll, pitch) that puts every planted foot on the
    floor:   0 = h + z_i + roll*y_i - pitch*x_i   (least squares).
    `mount` adds the IMU mount tilt the EKF sees on top of the physical
    trunk-vs-floor attitude; `disturb` adds an external physical tilt.
    """

    def __init__(self, anchors, mount=(0.0, 0.0)):
        self.anchors = anchors
        self.mount = mount
        self.disturb = (0.0, 0.0)

    def solve(self, foot_z, planted):
        rows, rhs = [], []
        for leg in LEGS:
            if not planted[leg]:
                continue
            x, y = self.anchors[leg]
            rows.append([1.0, y, -x])
            rhs.append(-foot_z[leg])
        sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
        h, roll, pitch = float(sol[0]), float(sol[1]), float(sol[2])
        roll += self.disturb[0]
        pitch += self.disturb[1]
        return h, roll, pitch

    def ekf_attitude(self, foot_z, planted):
        h, roll, pitch = self.solve(foot_z, planted)
        return roll + self.mount[0], pitch + self.mount[1]


class FakeShared:
    """Duck-typed `EkfShared` for the veto gates -- one class, both dialects.

    The height tests wanted a vertical position and a level attitude; the
    leveling tests wanted an attitude and no position.  The two copies of this
    class therefore disagreed on what the FIRST POSITIONAL argument meant
    (`rz` in one, `roll` in the other), which is exactly the kind of trap that
    survives a copy-paste.  Everything is keyword-only here, so a call site
    says which it means.
    """

    def __init__(self, *, rz=0.0, roll=0.0, pitch=0.0, healthy=True,
                 ready=True, C=None):
        self.est_ready = ready
        if C is None:
            C = C_from_rp(roll, pitch)      # (0,0) -> exactly eye(3)
        self.out = None if not ready else {
            "r": np.array([0.0, 0.0, rz]), "healthy": healthy, "C": C}
        self.tau_stamp = time.monotonic()


class FakeAhrs:
    """`AhrsSource` stand-in: scripted attitude + staleness."""

    def __init__(self):
        self.roll = 0.0
        self.pitch = 0.0
        self.stale = False
        self.none = False

    def read(self):
        if self.none:
            return float("nan"), float("nan"), False, "no AHRS yet"
        if self.stale:
            return self.roll, self.pitch, False, "AHRS stale"
        return self.roll, self.pitch, True, None


class FakeFeed:
    """Duck-typed `ImuEkfFeed` serving level-and-still FLU samples."""

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
            items, self.pending = self.pending, []
        return items

    def attitude(self):
        return None


# ===========================================================================
# helpers
# ===========================================================================

_TABLE_CACHE = {}


def tables_for(stand_height, clamp_m=None, announce=True):
    """Per-leg z->q tables, built once per (stand_height, clamp) and reused.

    A build is ~9 s of incremental IK.  Every runner paid it at least once and
    `lift_ekf_contact_hw` carried its own `_shared_tables` module global to
    avoid paying it twice -- this is that global, shared by everyone, so a
    suite that needs the same span twice gets it for free.
    """
    key = (round(float(stand_height), 9),
           None if clamp_m is None else round(float(clamp_m), 9))
    if key not in _TABLE_CACHE:
        t0 = time.perf_counter()
        kw = {} if clamp_m is None else {"clamp_m": clamp_m}
        _TABLE_CACHE[key] = s2.build_tables(stand_height, **kw)
        if announce:
            print(f"  (tables built in {time.perf_counter()-t0:.2f}s)")
    return _TABLE_CACHE[key]


def anchors_of(tables):
    """Foot x/y per leg -- the lever arms every attitude gate needs."""
    return {leg: tables[leg].xy for leg in LEGS}


def all_planted(value=True):
    return {leg: value for leg in LEGS}


def planted_except(*airborne):
    """Planted dict with the named legs lifted."""
    return {leg: leg not in airborne for leg in LEGS}


def flat_base_z(stand_height):
    """The commanded stand: every foot at one z, a horizontal plane."""
    return {leg: -stand_height for leg in LEGS}


class RateWatch:
    """Worst per-leg |d offset|/dt seen -- the slew-limit gate, four scripts over.

    Feed it the offsets dict once per sweep; `worst` is what the gate asserts
    against the loop's slew limit.
    """

    def __init__(self, offsets=None):
        self.prev = dict(offsets) if offsets else {leg: 0.0 for leg in LEGS}
        self.worst = 0.0

    def add(self, offsets, dt=DT):
        for leg in LEGS:
            self.worst = max(self.worst,
                             abs(offsets[leg] - self.prev[leg]) / dt)
        self.prev = dict(offsets)
        return self.worst


def walk_stages(seq, offset=0.0, q_from="cmd", before_tick=None,
                after_tick=None, park_after=2.0, until="PARKED", dt=DT):
    """Drive a stand sequence CROUCH -> ... -> `until` against an ideal plant.

    ENTER is pressed the moment WAIT_CROUCH is reached and P `park_after`
    seconds into HOLD4 -- the full cycle every runner's stage gate walks.
    Returns (stage order, final t, last cmd, last contacts).

    `q_from` picks what is fed back as the measured pose: "cmd" (the returned
    command, i.e. a perfectly stiff servo) or "q_cmd" (re-read from the
    sequence).  `before_tick(seq, t)` runs before each update -- that is where
    a caller sets `extra_z`; `after_tick(seq, t, cmd, contacts)` runs after,
    which is where a caller samples a contact mask.
    """
    q, qd = Q_CROUCH.copy(), np.zeros(N_JOINTS)
    t, seen = 0.0, []
    cmd = contacts = None
    t_max = recorded.CROUCH_TIMEOUT_S + 3 * T_STAND
    while t < t_max and seq.stage != until:
        if before_tick is not None:
            before_tick(seq, t)
        cmd, contacts, _ = seq.update(
            t, q, qd,
            enter_pressed=seq.stage == "WAIT_CROUCH",
            park_pressed=(seq.stage == "HOLD4"
                          and t - seq.stage_t0 > park_after),
            offset=offset)
        q = cmd if q_from == "cmd" else seq.q_cmd(t)
        if seq.stage not in seen:
            seen.append(seq.stage)
        if after_tick is not None:
            after_tick(seq, t, cmd, contacts)
        t += dt
    return seen, t, cmd, contacts


STAGE_ORDER = ["CROUCH", "WAIT_CROUCH", "STAND", "HOLD4", "PARK", "PARKED"]


def time_sweep(name, body, n=200, budget_s=SLOT_BUDGET_S):
    """Run `body(k)` n times and gate the MEAN sweep against the CAN slot.

    Everything in the `joint_index == 0` branch runs before that slot's
    command goes out, so a block that outlasts its slot delays the first motor
    of the round-robin.  250 us is the budget; the measured figure is printed
    either way because it is the number worth watching over time.
    """
    t0 = time.perf_counter()
    for k in range(n):
        body(k)
    per = (time.perf_counter() - t0) / n
    return check(name, per < budget_s, f"{per*1e6:.0f} us per sweep")


def stand_height_arg(parser, default):
    """The one CLI flag every self-test script takes."""
    parser.add_argument("--stand-height", type=float, default=default,
                        help="pose height the gates are built around (m)")
    return parser
