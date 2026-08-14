#!/usr/bin/env python3
"""Offline gates for stand_ekf_verify_hw (stage 1: the EKF, read-only).

WHAT IS UNDER TEST
    The stand pose, the stage machine and its park/re-stand cycle, the EKF
    worker's threading and init gating, the contact schedules, the FK height
    reference every later stage measures against, and the two reporters
    (StageReport, AttitudeStats) that ARE this stage's deliverable.

    Stage 1 puts nothing in the loop, so there is no controller here to gate:
    what has to be right is the measurement and the bookkeeping.

Run:  $V test_stand_ekf_verify.py
"""
from __future__ import annotations

import argparse
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

import selftest_common as tc                             # noqa: E402
from selftest_common import check, report                # noqa: E402

import stand_dog5_hw as base                             # noqa: E402
import stand_dog5_recorded_hw as recorded                # noqa: E402
import dog5_kinematics                                   # noqa: E402
import stand_ekf_verify_hw as s1                         # noqa: E402
from stand_ekf_verify_hw import (StandSequence,          # noqa: E402
                                 StageReport, AttitudeStats,
                                 compute_q_stand, fk_floor_height,
                                 POSITION_TARGET_DEG, QUIET_STAGES)
from ekf_runtime import EkfShared, ekf_worker, _rp       # noqa: E402

LEGS = tc.LEGS
N_JOINTS = tc.N_JOINTS
DT = tc.DT
Q_CROUCH = tc.Q_CROUCH
T_STAND = s1.T_STAND
FOOT_RADIUS_M = tc.FOOT_RADIUS_M
IMU_BELOW_TRUNK_ORIGIN_M = tc.IMU_BELOW_TRUNK_ORIGIN_M


def test_q_stand(stand_height):
    q_stand = compute_q_stand(stand_height)
    lo, hi = base.soft_limits()
    check("stand pose inside soft limits",
          bool(np.all(q_stand >= lo) and np.all(q_stand <= hi)))
    dxy = dz = 0.0
    for i, leg in enumerate(LEGS):
        sl = slice(3 * i, 3 * i + 3)
        f_c = dog5_kinematics.foot_position(leg, Q_CROUCH[sl])
        f_s = dog5_kinematics.foot_position(leg, q_stand[sl])
        dxy = max(dxy, float(np.linalg.norm(f_s[:2] - f_c[:2])))
        dz = max(dz, abs(f_s[2] + stand_height))
    check("feet keep their crouch x/y", dxy < 1e-6, f"max {dxy*1e3:.4f} mm")
    check("feet reach -stand_height", dz < 1e-6, f"max err {dz*1e3:.4f} mm")
    return q_stand


def test_sequence(q_stand):
    """Drive the stage machine with an ideal position plant (stiff servo).

    Not `selftest_common.walk_stages`: this walk stops at HOLD4 rather than
    parking, starts the plant well away from the crouch, and reads contacts
    per stage -- three differences that would turn the shared driver into a
    flag farm.  The four runners that DO share a walk use it.
    """
    # exercise the OFF schedule here: it is the one with stage-varying contacts
    seq = StandSequence(q_stand, contacts_during_ramp=False)
    t = 0.0
    q = Q_CROUCH + 0.5                       # start well away from the crouch
    qd = np.zeros(N_JOINTS)
    seen, contacts_by_stage, z_trunk = [], {}, []
    entered_stand = False
    while t < recorded.CROUCH_TIMEOUT_S + T_STAND + 2.0 and seq.stage != "HOLD4":
        enter = seq.stage == "WAIT_CROUCH"   # operator presses ENTER immediately
        cmd, contacts, event = seq.update(t, q, qd, enter_pressed=enter)
        if seq.stage not in seen:
            seen.append(seq.stage)
        contacts_by_stage.setdefault(seq.stage, []).append(contacts.copy())
        q = np.deg2rad(cmd)                  # ideal tracking
        if seq.stage == "STAND":
            entered_stand = True
            z_trunk.append(-dog5_kinematics.foot_position(LEGS[0], q[0:3])[2])
        t += DT
    cmd, contacts, _ = seq.update(t, q, qd)
    contacts_by_stage.setdefault(seq.stage, []).append(contacts.copy())

    check("stage order CROUCH->WAIT_CROUCH->STAND->HOLD4",
          seen == ["CROUCH", "WAIT_CROUCH", "STAND", "HOLD4"]
          and seq.stage == "HOLD4",
          " -> ".join(seen))
    check("HOLD4 commands the stand pose",
          bool(np.allclose(cmd, np.rad2deg(q_stand))))
    check("contacts OFF only during STAND",
          all((not np.any(np.array(v))) if s == "STAND"
              else bool(np.all(np.array(v)))
              for s, v in contacts_by_stage.items()),
          " ".join(f"{s}={'off' if not np.any(np.array(v)) else 'on'}"
                   for s, v in contacts_by_stage.items()))
    check("trunk rises monotonically through STAND",
          entered_stand and bool(np.all(np.diff(z_trunk) >= -1e-9)),
          f"{z_trunk[0]*1e3:.0f} -> {z_trunk[-1]*1e3:.0f} mm")

    # a plant that never settles must trip the CROUCH timeout, not hang
    stuck = StandSequence(q_stand, contacts_during_ramp=False)
    t = 0.0
    while t < recorded.CROUCH_TIMEOUT_S + 1.0 and stuck.fault is None:
        stuck.update(t, Q_CROUCH + 5.0, np.zeros(N_JOINTS))
        t += DT
    check("stuck crouch trips the timeout", stuck.fault == "CROUCH timeout",
          str(stuck.fault))
    return seq, q, t


def test_park(seq, q, t, q_stand):
    """From the HOLD4 left by test_sequence: park, then stand again."""
    qd = np.zeros(N_JOINTS)

    # P is accepted only from HOLD4 -- prove it is ignored mid-rise
    mid_rise = StandSequence(q_stand, contacts_during_ramp=False)
    mid_rise.stage, mid_rise.stage_t0 = "STAND", 0.0
    mid_rise.update(0.5 * T_STAND, q, qd, park_pressed=True)
    check("P ignored outside HOLD4", mid_rise.stage == "STAND", mid_rise.stage)

    _, _, event = seq.update(t, q, qd, park_pressed=True)
    check("P in HOLD4 starts PARK",
          seq.stage == "PARK" and event == "park_started", seq.stage)

    z_trunk, park_contacts = [], []
    t_park0 = t
    while t - t_park0 < T_STAND + 2 * DT and seq.stage == "PARK":
        cmd, contacts, _ = seq.update(t, q, qd)
        q = np.deg2rad(cmd)
        if seq.stage == "PARK":     # the tick that finishes the ramp is PARKED
            park_contacts.append(contacts.copy())
        z_trunk.append(-dog5_kinematics.foot_position(LEGS[0], q[0:3])[2])
        t += DT
    check("PARK completes into PARKED", seq.stage == "PARKED", seq.stage)
    check("trunk descends monotonically through PARK",
          bool(np.all(np.diff(z_trunk) <= 1e-9)),
          f"{z_trunk[0]*1e3:.0f} -> {z_trunk[-1]*1e3:.0f} mm")
    check("contacts OFF through PARK", not np.any(np.array(park_contacts)))

    cmd, contacts, _ = seq.update(t, q, qd)
    check("PARKED commands the crouch pose",
          bool(np.allclose(cmd, POSITION_TARGET_DEG)),
          f"max err {np.max(np.abs(cmd - POSITION_TARGET_DEG)):.4f} deg")
    check("contacts ON in PARKED", bool(np.all(contacts)))

    # ENTER from PARKED must rise again, and the rise must be counted
    _, _, event = seq.update(t, q, qd, enter_pressed=True)
    check("ENTER in PARKED stands again",
          seq.stage == "STAND" and event == "stand_started", seq.stage)
    t_re0 = t
    while t - t_re0 < T_STAND + 2 * DT and seq.stage == "STAND":
        cmd, _, _ = seq.update(t, q, qd)
        q = np.deg2rad(cmd)
        t += DT
    check("re-stand reaches HOLD4 at the stand pose",
          seq.stage == "HOLD4"
          and np.allclose(cmd, np.rad2deg(q_stand), atol=1e-6),
          f"{seq.stage}, n_stands={seq.n_stands}")
    check("both rises counted", seq.n_stands == 2, str(seq.n_stands))


def test_worker_wiring(q_stand):
    """The EKF worker must init only in a quiet stage and report level."""
    shared = EkfShared(q_stand)
    feed = tc.FakeFeed()
    worker = threading.Thread(target=ekf_worker, args=(shared, feed),
                              kwargs=dict(quiet_stages=QUIET_STAGES,
                                          control_hz=200.0),
                              daemon=True)
    worker.start()
    try:
        shared.stage = "CROUCH"               # NOT quiet: must not initialise
        feed.push(60)
        time.sleep(0.15)
        check("no EKF init outside the quiet stage", not shared.est_ready)

        shared.stage = "WAIT_CROUCH"
        feed.push(60)
        t_end = time.time() + 2.0
        while not shared.est_ready and time.time() < t_end:
            time.sleep(0.01)
        check("EKF initialises in WAIT_CROUCH", shared.est_ready,
              shared.bias_str)
        check("imu_log gated off before log_enabled", len(shared.imu_log) == 0)

        shared.log_enabled = True
        feed.push(40)
        t_end = time.time() + 2.0
        while shared.out is None and time.time() < t_end:
            time.sleep(0.01)
        ok = shared.out is not None and "C" in shared.out
        check("worker publishes outputs", ok)
        if ok:
            r, p = _rp(shared.out["C"])
            check("level-and-still reads level",
                  max(abs(r), abs(p)) < math.radians(1.0),
                  f"roll={math.degrees(r):+.2f} pitch={math.degrees(p):+.2f} deg")
            check("EKF reports healthy", bool(shared.out["healthy"]))
        # the worker may publish outputs before it drains the new samples
        feed.push(40)
        t_end = time.time() + 2.0
        while not shared.imu_log and time.time() < t_end:
            time.sleep(0.01)
        check("imu_log grows once enabled", len(shared.imu_log) > 0)
    finally:
        shared.run = False
        worker.join(timeout=1.0)


def test_contact_schedules(q_stand):
    """The default schedule must keep contacts ON everywhere, and that must
    mean the estimator never sees a rising edge (nothing is re-anchored)."""
    check("contacts stay ON through the ramps BY DEFAULT",
          StandSequence(q_stand).contacts_during_ramp is True)
    for during_ramp in (False, True):
        seq = StandSequence(q_stand, contacts_during_ramp=during_ramp)
        q, qd = Q_CROUCH.copy(), np.zeros(N_JOINTS)
        seen = []                                   # (stage, contacts)
        t = 0.0
        while t < recorded.CROUCH_TIMEOUT_S + 3 * T_STAND \
                and seq.stage != "PARKED":
            enter = seq.stage in ("WAIT_CROUCH",)
            park = seq.stage == "HOLD4"
            cmd, contacts, _ = seq.update(t, q, qd, enter_pressed=enter,
                                          park_pressed=park)
            q = np.deg2rad(cmd)
            seen.append((seq.stage, contacts.copy()))
            t += DT
        ramp = [c for s, c in seen if s in StandSequence.MOVING_STAGES]
        assert ramp, "test never entered a ramp"
        label = "ON" if during_ramp else "OFF"
        check(f"ramp contacts {label} as configured",
              bool(np.all(np.array(ramp))) == during_ramp)

        # a rising edge is what re-anchors a foothold; ON must produce none
        arr = np.array([c for _, c in seen])
        rising = int(np.sum(arr[1:] & ~arr[:-1]))
        check(f"contacts {label}: {'no' if during_ramp else 'some'} re-anchoring",
              (rising == 0) == during_ramp, f"{rising} rising edges")


def test_height(q_stand, stand_height):
    """FK height must be absolute and drift-free; drift must isolate the EKF."""
    h_hip = fk_floor_height(q_stand, ref="hip")
    check("FK hip-axis height = commanded + foot radius",
          abs(h_hip - (stand_height + FOOT_RADIUS_M)) < 1e-9,
          f"{h_hip*1e3:.1f} mm vs {(stand_height+FOOT_RADIUS_M)*1e3:.1f} mm")

    # the default reference is the IMU -- the point the EKF's r actually tracks
    h_crouch = fk_floor_height(Q_CROUCH)
    h_stand = fk_floor_height(q_stand)
    check("default reference is the IMU, 38 mm below the hip axis",
          abs((h_hip - h_stand) - IMU_BELOW_TRUNK_ORIGIN_M) < 1e-12,
          f"hip {h_hip*1e3:.0f} mm, IMU {h_stand*1e3:.0f} mm")
    check("an unknown reference is rejected",
          tc.raises(lambda: fk_floor_height(q_stand, ref="trunk_bottom")))
    check("FK crouch height is below the stand height",
          h_crouch < h_stand, f"{h_crouch*1e3:.0f} -> {h_stand*1e3:.0f} mm")

    # a level attitude must not change the answer; a tilt must
    eye = np.eye(3)
    check("identity attitude matches the level shortcut",
          abs(fk_floor_height(q_stand, eye) - h_stand) < 1e-12)
    tilt = math.radians(10.0)
    C_tilt = np.array([[math.cos(tilt), 0.0, -math.sin(tilt)],
                       [0.0, 1.0, 0.0],
                       [math.sin(tilt), 0.0, math.cos(tilt)]])
    check("a tilted trunk changes the FK height",
          abs(fk_floor_height(q_stand, C_tilt) - h_stand) > 1e-6,
          f"{fk_floor_height(q_stand, C_tilt)*1e3:.1f} mm at 10 deg pitch")
    # the IMU lever shortens the vertical offset as the trunk tilts
    lever = (fk_floor_height(q_stand, C_tilt, ref="hip")
             - fk_floor_height(q_stand, C_tilt))
    check("IMU vertical offset shrinks with tilt (cos, not constant)",
          lever < IMU_BELOW_TRUNK_ORIGIN_M - 1e-9,
          f"{lever*1e3:.3f} mm at 10 deg vs "
          f"{IMU_BELOW_TRUNK_ORIGIN_M*1e3:.1f} flat")

    # ...but the DRIFT number must not care: a constant cancels in the delta
    d_imu = ((h_stand - h_crouch))
    d_hip = (fk_floor_height(q_stand, ref="hip")
             - fk_floor_height(Q_CROUCH, ref="hip"))
    check("reference choice does not move the measured rise",
          abs(d_imu - d_hip) < 1e-12,
          f"IMU {d_imu*1e3:.3f} mm vs hip {d_hip*1e3:.3f} mm")

    # drift bookkeeping: a PERFECT EKF rising by the true amount drifts 0
    ht = StageReport()
    check("no origin -> add() is a no-op",
          ht.add("WAIT_CROUCH", h_crouch, 0.0) is None)
    ht.set_origin(h_crouch)
    ht.set_origin(999.0)                       # later origins must be ignored
    check("height origin is latched at init", ht.z0_fk == h_crouch)

    # the crouch baseline row: same pose the origin was latched at -> drift ~0
    d = ht.add("WAIT_CROUCH", h_crouch, 0.0, math.radians(0.3),
               math.radians(-0.2))
    check("crouch baseline drifts zero by construction", abs(d) < 1e-12,
          f"{d*1e3:.3f} mm")
    d = ht.add("HOLD4", h_stand, h_stand - h_crouch,
               math.radians(0.3), math.radians(-0.2))
    check("a perfect EKF shows zero drift", abs(d) < 1e-12, f"{d*1e3:.3f} mm")
    # the reported failure: parked back at the crouch, EKF still says -133 mm
    d = ht.add("PARKED", h_crouch, -0.133, math.radians(0.3),
               math.radians(-0.2))
    check("parked-with-stale-z reproduces the observed drift",
          abs(d + 0.133) < 1e-12, f"{d*1e3:.0f} mm")

    txt = "\n".join(ht.summary())
    check("summary reports every holding stage",
          all(s in txt for s in ("WAIT_CROUCH", "HOLD4", "PARKED")))
    check("stage rows are chronological, crouch first",
          txt.index("WAIT_CROUCH ") < txt.index("HOLD4") < txt.index("PARKED"))
    check("summary carries attitude per stage",
          "+0.30" in txt and "-0.20" in txt)
    check("summary confirms the legs returned to the crouch",
          "+0 mm from the crouch baseline" in txt,
          next((l.strip() for l in txt.splitlines() if "legs say" in l),
               "missing"))
    check("summary explains a large PARKED drift", "dead-reckoning" in txt)
    check("summary states what EKF z=0 means", "the crouch" in txt)

    # a bad FK reference / origin latch must be called out, not blamed on drift
    bad = StageReport()
    bad.set_origin(h_crouch)
    bad.add("WAIT_CROUCH", h_crouch + 0.030, 0.0)
    check("a non-zero crouch baseline is flagged as a reference error",
          "NOT dead reckoning" in "\n".join(bad.summary()))


def test_stats():
    st = AttitudeStats()
    check("empty stats do not crash", len(st.summary()) == 1)
    st.begin_visit()
    for _ in range(10):
        st.add(math.radians(1.0), math.radians(-2.0),
               math.radians(1.5), math.radians(-2.5))
    line = st.end_visit()
    check("a HOLD4 visit closes with its own line",
          line is not None and "+1.00" in line, (line or "").strip())
    txt = "\n".join(st.summary())
    check("stats report the resting means",
          "+1.00" in txt and "-2.00" in txt, txt.splitlines()[1].strip())
    check("stats report the AHRS disagreement", "0.50" in txt)

    # a second stand at a different attitude must show up as REPEAT spread
    st.begin_visit()
    for _ in range(10):
        st.add(math.radians(1.4), math.radians(-2.0), float("nan"),
               float("nan"))
    st.end_visit()
    txt = "\n".join(st.summary())
    check("park/stand cycles report repeatability spread",
          "REPEAT" in txt and "0.40" in txt,
          next((l.strip() for l in txt.splitlines() if "REPEAT" in l),
               "missing"))


def self_test(stand_height=recorded.DEFAULT_STAND_HEIGHT):
    print("stand_ekf_verify_hw self-test (no hardware)")
    print("[1] stand pose")
    q_stand = test_q_stand(stand_height)
    print("[2] stage machine")
    seq, q, t = test_sequence(q_stand)
    print("[3] park / re-stand")
    test_park(seq, q, t, q_stand)
    print("[4] EKF worker wiring")
    test_worker_wiring(q_stand)
    print("[5] contact schedules")
    test_contact_schedules(q_stand)
    print("[6] FK height vs EKF z")
    test_height(q_stand, stand_height)
    print("[7] attitude statistics")
    test_stats()
    return report()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    tc.stand_height_arg(ap, recorded.DEFAULT_STAND_HEIGHT)
    sys.exit(self_test(ap.parse_args().stand_height))


if __name__ == "__main__":
    main()
