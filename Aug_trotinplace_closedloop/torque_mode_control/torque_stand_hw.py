#!/usr/bin/env python3
"""Stages T2-T4 -- the torque stand.  The robot holds itself up with force.

    ZERO-TORQUE -> CROUCH (native 0xA4) -> WAIT_CROUCH -> RISE (torque)
                                                              |
                                        PARKED <- PARK <-  HOLD (torque)

WHAT THIS ANSWERS
    In position mode the drivers' own loops hold the pose, so once the robot
    settles nothing the upper layer computes has any observable effect.  You
    cannot tell a working feedback loop from an ignored one.  Here the trunk
    is held up by commanded forces, so the loop is the only thing keeping it
    up -- push it and the recovery IS the feedback.

    The A/B that makes it a measurement rather than a demo:

        torque_stand_hw.py --raw-log live.npz
        torque_stand_hw.py --kp-att 0 --kd-att 0 --raw-log ablate.npz

    Push the trunk the same way in each.  Live: attitude recovers and the
    per-foot load redistributes.  Ablated: it stays pushed and the load does
    not move.  The difference between those two logs is the answer.

THE TWO CORRECTIONS THIS RUNNER CARRIES (both are A/B-able)
    --no-leg-gravity   reproduces the 2026-07-30 stance law, which omitted
                       the legs' own link weights.  That is 63% of the
                       abduction torque, in the opposite direction.
    --mass             defaults to 5.8151 kg from dog5.xml.  The value live
                       in stand_dog5_hw.py is 5.3, which is 8.9% low.
    Running with both wrong should reproduce the original failure; that is
    what stage T2 measures, with --unload-leg.

SAFETY, IN THE ORDER IT APPLIES
    joint impedance (250 Hz, in-sweep)  no joint is ever an open integrator
    TorqueSafetyGate                    ramp, cap, slew, limits, overspeed
    RunawayBrake                        last resort, constants corrected
    LIMP                                SPACE, or automatic -- see below

RUN -- in this order, each stage signed off before the next
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    cd /home/robot01/Documents/can_motor_control/dog_stand_compliance_control/Aug_trotinplace_closedloop

    # 0. offline first.  Nothing here is worth arming a motor for if these
    #    are red -- and T0/T1 must have passed on hardware before this file
    #    is run at all (tau_calib_hw.py, then torque_hold_hw.py).
    $V self-test/test_all.py

    # T2  the force law at the CROUCH only, robot supported, low cap.
    #     Three runs, back to back, comparing the `load=` field:
    sudo HOME=$HOME chrt -f 50 $V torque_mode_control/torque_stand_hw.py \
        --tau-max 1.5 --no-leg-gravity --mass 5.3      # the 2026-07-30 law
    sudo HOME=$HOME chrt -f 50 $V torque_mode_control/torque_stand_hw.py \
        --tau-max 1.5                                   # corrected
    #     then the deliberate fault -- one foot raised while the stance law
    #     still believes it planted.  Hand on SPACE.
    sudo HOME=$HOME chrt -f 50 $V torque_mode_control/torque_stand_hw.py \
        --tau-max 1.5 --unload-leg RR --raw-log t2.npz

    # T3  the rise under torque, tethered, hand on SPACE.
    sudo HOME=$HOME chrt -f 50 $V torque_mode_control/torque_stand_hw.py \
        --tau-max 3.0 --kp-z 800 --raw-log t3.npz

    # T4  the A/B that answers the original question.  Push the trunk the
    #     SAME way in each run; the difference between the two logs is the
    #     feedback loop, made observable.
    sudo HOME=$HOME chrt -f 50 $V torque_mode_control/torque_stand_hw.py \
        --raw-log t4_live.npz
    sudo HOME=$HOME chrt -f 50 $V torque_mode_control/torque_stand_hw.py \
        --kp-att 0 --kd-att 0 --raw-log t4_ablate.npz

    First run of any stage: start at --tau-max 1.0 and leave --force-frac at
    1.  Do NOT start at --force-frac 0 with the feet on the floor -- it
    removes the only term holding the trunk up, the legs fold until the
    impedance error carries 57 N, and the runaway brake latches on the way
    down (measured 2026-08-14: seven joints, mostly rear, inside 2 s).
    Isolate the plumbing with tau_calib_hw --hang instead, which does it with
    the weight off the legs where it is actually safe.

WHAT TO WATCH ON THE STATUS LINE
    load=   sum of per-foot support force from MEASURED iq, inverted through
            J^-T.  It must read ~57 N.  This is the end-to-end proof the
            force loop is real; if it is wrong, nothing else means anything.
    trk=    |tau_meas| / |tau_cmd|.  Under ~80% the current loop is not
            delivering what the grasp map asked for -- T0 should have caught it.
    |dq|=   impedance error.  Small in nominal stance (the force law is doing
            the work); large means the impedance is carrying load the force
            law thinks is on the ground.
    brk=    latched joints.  Should be 0.  Anything else is the last-resort
            layer firing, which means the impedance floor did not hold.

Keys:  ENTER = advance stage / re-engage from limp
       SPACE = LIMP (torque off, loop and keep-alives keep running)
       P     = park (from HOLD)
       X     = soft stop and exit
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.setswitchinterval(0.0005)

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUG = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_AUG)
_REPO = os.path.dirname(_ROOT)
_DESC = os.path.join(_ROOT, "dog5_description")
_POS = os.path.join(_AUG, "stand_postion_mode")
_IMU = os.path.join(_REPO, "IMU_sensor")
for _p in (_HERE, _DESC, _POS, _REPO, _IMU):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _add_fdilink_root():
    """Put the `fdilink_imu` package on the path WITHOUT trusting $HOME.

    imu_dog.py resolves it as Path.home()/Documents/IMU_sensor, which breaks
    under sudo ($HOME becomes /root) -- the failure recorded in 7.28review.md.
    RT priority (`chrt -f 50`) wants root, so resolve repo-relative first and
    fall back to the invoking user's home.  Same logic as
    stand_ekf_verify_hw._add_fdilink_root; duplicated rather than imported
    because importing that module would drag the whole EKF stage-1 runner
    into a file whose entire point is that it has no EKF.
    """
    candidates = [os.path.join(os.path.dirname(_REPO), "IMU_sensor"),
                  os.path.join(_REPO, "IMU_sensor"),
                  os.path.join(os.path.expanduser("~"), "Documents",
                               "IMU_sensor")]
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user:
        candidates.append(os.path.join("/home", sudo_user, "Documents",
                                       "IMU_sensor"))
    for root in candidates:
        if os.path.isdir(os.path.join(root, "fdilink_imu")):
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    return None


if _add_fdilink_root() is None:
    print("[warn] fdilink_imu package not found; the IMU import will fail",
          file=sys.stderr)

import motorbus                                            # noqa: E402
import stand_dog5_hw as base                               # noqa: E402
import stand_dog5_recorded_hw as recorded                  # noqa: E402
import crawl_dog5_hw as crawl                              # noqa: E402
import body_state_ahrs as bs                               # noqa: E402
import dog5_statics as st                                  # noqa: E402
import stance_law as law                                   # noqa: E402
import torque_params as P                                  # noqa: E402
import torque_worker as tw                                 # noqa: E402
from imu_dog import ImuDog, DEFAULT_PORT                    # noqa: E402

MOTOR_IDS = base.MOTOR_IDS
N_JOINTS = base.N_JOINTS
CONTROL_HZ = base.CONTROL_HZ
LEGS = base.LEGS
Q_CROUCH = recorded.Q_RECORDED_CROUCH
POSITION_TARGET_DEG = recorded.POSITION_TARGET_DEG


def _smoothstep(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


class StandSequence:
    """CROUCH -> WAIT_CROUCH -> RISE -> HOLD -> PARK -> PARKED.

    The height reference is ABSOLUTE and starts at the MEASURED crouch
    height, never at the stand height.  That is not a nicety: body_wrench's
    Fz is kp_z*(z_des - z), so starting a 0.19 m reference while the robot
    sits at the 0.04 m crouch demands ~176 N on a 57 N robot -- three times
    its weight, on the first sweep, before the ramp has done anything.
    test_stance_law.test_zdes_must_track_the_measured_height gates this.
    """

    STAGES = ("CROUCH", "WAIT_CROUCH", "RISE", "HOLD", "PARK", "PARKED")

    def __init__(self, now, stand_height, t_rise=P.T_RISE):
        self.stage = "CROUCH"
        self.t0 = now
        self.stand_height = float(stand_height)
        self.t_rise = float(t_rise)
        self.z_crouch = None          # measured, filled in at WAIT_CROUCH
        self.settle_t = None

    def crouch_settled(self, now, qd, q):
        """The native position crouch is done when the joints stop."""
        if float(np.max(np.abs(qd))) > 0.25:
            self.settle_t = None
            return False
        if self.settle_t is None:
            self.settle_t = now
        return (now - self.settle_t) >= P.T_CROUCH_SETTLE_S

    def enter_wait(self, now, z_measured):
        self.stage = "WAIT_CROUCH"
        self.t0 = now
        self.z_crouch = float(z_measured)

    def start_rise(self, now):
        if self.stage != "WAIT_CROUCH" or self.z_crouch is None:
            return False
        self.stage = "RISE"
        self.t0 = now
        return True

    def park(self, now):
        if self.stage != "HOLD":
            return False
        self.stage = "PARK"
        self.t0 = now
        return True

    def update(self, now):
        if self.stage == "RISE" and now - self.t0 >= self.t_rise:
            self.stage = "HOLD"
            self.t0 = now
        elif self.stage == "PARK" and now - self.t0 >= self.t_rise:
            self.stage = "PARKED"
            self.t0 = now

    def z_des(self, now):
        """Absolute floor-to-trunk-origin height reference (m)."""
        if self.z_crouch is None:
            return None
        if self.stage in ("CROUCH", "WAIT_CROUCH"):
            return self.z_crouch
        if self.stage == "RISE":
            u = _smoothstep((now - self.t0) / self.t_rise)
            return self.z_crouch + u * (self.stand_height - self.z_crouch)
        if self.stage == "HOLD":
            return self.stand_height
        if self.stage == "PARK":
            u = _smoothstep((now - self.t0) / self.t_rise)
            return self.stand_height + u * (self.z_crouch - self.stand_height)
        return self.z_crouch

    @property
    def torque_active(self):
        return self.stage in ("RISE", "HOLD", "PARK")


class RunawayBrake:
    """vmc_stand_hw.RunawayBrake's logic, with BRAKE_PERIOD_S corrected.

    The zero-incremental-gain argument is still sound and worth keeping: the
    brake LATCHES rather than blending, so d(tau)/d(qd) = 0 inside each state
    and it adds no feedback gain at any sample rate.  What changed is that it
    is no longer the ONLY option -- the premise that continuous feedback was
    unaffordable rested on the 12x rate error, so the joint impedance now
    catches this fault long before the brake sees it.  This is the backstop
    for the cases impedance cannot fix: a wrong encoder zero, a latched
    driver, a model blow-up.

    As shipped the brake was a no-op: BRAKE_HOLD_NM came out at 0.128 Nm from
    the 48 ms period, less than the 0.48 Nm an abduction joint needs just to
    hold its own leg up.  At the true 4 ms the same formula gives 1.54 Nm.
    """

    def __init__(self):
        self.latched = np.zeros(N_JOINTS, dtype=bool)
        self.latch_t = np.zeros(N_JOINTS)
        self.release_t = np.full(N_JOINTS, -1.0)
        self.trips = np.zeros(N_JOINTS, dtype=int)

    def apply(self, tau, qd, now, cap=None):
        """`cap` is the operator's --tau-max.  The brake is applied AFTER
        SafetyGate (deliberately: it must be neither ramped nor slewed), so
        without this it is the one path that can exceed the cap.

        That was harmless while BRAKE_HOLD_NM was the 48 ms value of 0.128 Nm
        -- it was under every cap because it was under everything, which is
        also why it never stopped anything.  Correcting it to 1.54 Nm made it
        effective AND made it able to override a conservative --tau-max 1.0:
        hardware 2026-08-14 printed |tau|=1.54 on a 1.0 cap.  A safety layer
        must never command more than the operator allowed, so clamp -- and
        accept that at a low cap the brake is correspondingly weaker, which
        is the honest trade rather than a silent override.
        """
        tau = np.asarray(tau, dtype=float).copy()
        qd = np.asarray(qd, dtype=float)
        fast = np.abs(qd) > P.BRAKE_LATCH_RAD_S
        newly = fast & ~self.latched
        self.latched |= fast
        self.latch_t = np.where(newly, now, self.latch_t)
        self.trips += newly.astype(int)

        slow = np.abs(qd) < P.BRAKE_RELEASE_RAD_S
        dwelt = (now - self.latch_t) > P.BRAKE_MIN_LATCH_S
        releasing = self.latched & slow & dwelt
        self.release_t = np.where(releasing, now, self.release_t)
        self.latched &= ~releasing

        hold_nm = P.BRAKE_HOLD_NM if cap is None else min(P.BRAKE_HOLD_NM,
                                                          float(cap))
        hold = -np.sign(qd) * hold_nm
        # drive is restored on a TIME ramp, not a qd ramp, so recovery adds
        # no velocity feedback and cannot destabilise anything
        ramp = np.clip((now - self.release_t) / P.BRAKE_RECOVER_S, 0.0, 1.0)
        ramp = np.where(self.release_t < 0.0, 1.0, ramp)
        return np.where(self.latched, hold, tau * ramp), int(self.latched.sum())

    def stop_reason(self, now):
        over = np.where(self.trips > P.BRAKE_MAX_TRIPS)[0]
        if over.size:
            return (f"joint {over.tolist()} latched "
                    f">{P.BRAKE_MAX_TRIPS}x -- that leg is not bearing load")
        stuck = self.latched & ((now - self.latch_t) > P.BRAKE_LATCH_TIMEOUT_S)
        if stuck.any():
            return f"joint {np.where(stuck)[0].tolist()} latched and not slowing"
        return None

    def summary(self):
        if not self.trips.any():
            return "  brake: never latched"
        return "  brake trips per joint: " + ", ".join(
            f"{i}:{n}" for i, n in enumerate(self.trips) if n)


def bus_report(mb):
    """Per-motor link health at exit.  Named, not just counted.

    `missed` is cumulative: it counts sends made while the previous request
    to that motor was still unanswered.  A healthy motor at 250 Hz should sit
    at ~0 -- the reply comes back inside one 333 us slot and is drained long
    before that motor's next turn 4 ms later.  A few percent is jitter; tens
    of percent is one motor not answering, and it will show up as a
    MISS_ESTOP trip the moment anything checks.

    This is printed for EVERY run, pass or fail, because a miss rate that is
    merely high is invisible until it is fatal.
    """
    rows = []
    for mid in MOTOR_IDS:
        rec = mb.rec(mid)
        sent = max(rec.sent, 1)
        rows.append((mid, rec.sent, rec.missed, 100.0 * rec.missed / sent,
                     rec.max_gap or 0.0, rec.send_fail))
    worst = max(r[3] for r in rows)
    out = ["  CAN link (missed = sent while the previous reply was pending):",
           f"    {'id':>3s} {'joint':10s} {'sent':>7s} {'missed':>7s} "
           f"{'rate':>6s} {'max gap':>8s} {'txfail':>7s}"]
    names = {mid: f"{leg}_{j}" for leg, ids in
             zip(LEGS, [MOTOR_IDS[i:i + 3] for i in range(0, 12, 3)])
             for mid, j in zip(ids, ("abd", "pitch", "knee"))}
    for mid, sent, missed, rate, gap, txf in rows:
        flag = "  <--" if rate > 5.0 else ""
        out.append(f"    {mid:3d} {names.get(mid, '?'):10s} {sent:7d} "
                   f"{missed:7d} {rate:5.1f}% {gap*1e3:7.1f}ms {txf:7d}{flag}")
    if worst > 5.0:
        out.append(f"    worst {worst:.0f}% -- that motor is not answering "
                   f"reliably.  Check it with the bus-only soak before "
                   f"blaming the controller:")
        out.append("      $V stand_postion_mode/../can_soak_hw.py  # if present")
        out.append("      sudo chrt -f 50 $V ../../t8_lost_probe.py")
    return "\n".join(out)


class LoadWatch:
    """Sum of the per-foot support forces from MEASURED torque.

    The single most valuable number in the project: it is measured iq,
    inverted through J^-T with each leg's own weight removed, and it must add
    up to the robot.  If it does not, the grasp map is fantasy regardless of
    what the attitude looks like.  It also validates torque_gain in situ.
    """

    def __init__(self):
        self.last = np.full(4, np.nan)
        self.samples = []

    def update(self, q, tau_meas):
        support, _ = crawl.foot_load_map(q, tau_meas)
        self.last = np.array([support[leg] for leg in LEGS])
        tot = float(np.nansum(self.last))
        self.samples.append(tot)
        return tot

    def out_of_bounds(self):
        if not self.samples:
            return False
        tot = self.samples[-1]
        if not np.isfinite(tot):
            return False
        return abs(tot - P.WEIGHT_N) / P.WEIGHT_N > P.LOAD_SUM_TOL_FRAC

    def report(self):
        if not self.samples:
            return "  load: no samples"
        a = np.array([s for s in self.samples if np.isfinite(s)])
        if not a.size:
            return "  load: all samples non-finite (singular legs?)"
        return (f"  foot load sum: mean {a.mean():.1f} N, "
                f"min {a.min():.1f}, max {a.max():.1f}, "
                f"against {P.WEIGHT_N:.1f} N")


def run(args):
    base.validate_hardware_config()
    mass = args.mass
    print(f"[t2] mass {mass:.4f} kg ({P.WEIGHT_N:.1f} N)"
          + ("" if abs(mass - st.total_mass()) < 1e-3
             else "  <-- NOT the model mass, this is an A/B run"))
    print(f"[t2] leg gravity in stance: {args.leg_gravity}"
          + ("" if args.leg_gravity
             else "  <-- reproducing the 2026-07-30 law"))
    print(f"[t2] force_frac {args.force_frac}, impedance kp={args.kp} "
          f"kd={args.kd}, tau cap {args.tau_max} Nm")
    if args.force_frac < 0.2:
        # Hardware 2026-08-14: --force-frac 0 was described as a safe
        # "plumbing test".  It is not, with the robot's weight ON its legs.
        # force_frac scales the DISTRIBUTED GRF -- i.e. the only term that
        # holds the TRUNK up.  The leg-gravity feedforward carries each leg's
        # own links and nothing more, so at force_frac 0 the trunk is
        # supported solely by the impedance's positional error: the legs fold
        # until kp*dq balances 57 N.  At kp=15 that needs ~0.09 rad of sag per
        # joint, and on the way there the joints accelerate enough to latch
        # the runaway brake -- which is exactly what happened (7 joints, mostly
        # rear, within 2 s of RISE).
        sag = P.WEIGHT_N * 0.12 / max(args.kp, 1e-6)   # ~0.12 m lever
        print(f"[t2] WARNING: force_frac {args.force_frac} removes the term "
              f"that holds the TRUNK up.")
        print(f"[t2]   With weight on the legs the trunk sags until kp*dq "
              f"carries it: ~{sag*1e3:.0f} mrad/joint at kp={args.kp:.0f}.")
        print(f"[t2]   This is only a plumbing test with the robot ON A STAND, "
              f"feet in the air.")
        print(f"[t2]   Feet on the floor: use --force-frac 1 (the leg-gravity "
              f"and mass corrections are the A/B), or raise --kp.")
    print(f"[t2] sweep {P.SWEEP_S*1e3:.0f} ms; every joint commanded and read "
          f"at {CONTROL_HZ:.0f} Hz")

    gains = law.default_gains()
    if args.kp_att is not None:
        gains.kp_roll = gains.kp_pitch = args.kp_att
    if args.kd_att is not None:
        gains.kd_roll = gains.kd_pitch = args.kd_att
    if args.kp_z is not None:
        gains.kp_z = args.kp_z
    if gains.kp_roll == 0.0 and gains.kd_roll == 0.0:
        print("[t2] ATTITUDE GAINS ZEROED -- this is the ablation half of "
              "the A/B; the trunk will not push back")

    cfg = law.default_cfg(args.stand_height, args.tau_max)
    imp = law.JointImpedance(kp=args.kp, kd=args.kd)
    unload_i = LEGS.index(args.unload_leg) if args.unload_leg else None
    if unload_i is not None:
        print(f"[t2] FAULT INJECTION: {args.unload_leg} foot will be raised "
              f"{args.unload_mm} mm in HOLD while the stance law still counts "
              f"it planted.  Watch max|qd| on {args.unload_leg}_abd.")
    brake = RunawayBrake()
    loads = LoadWatch()

    key = base.KeyPoller()
    imu = ImuDog(port=args.port)
    shared = None
    worker = None
    log = []
    bus = ""

    try:
        imu.start()
        with motorbus.MotorBus(MOTOR_IDS, dirs=base.MOTOR_DIRECTIONS) as mb:
            unwrap = [base.CalibratedEncoderUnwrap() for _ in MOTOR_IDS]
            mb.arm(rate_hz=CONTROL_HZ)
            if not imu.wait_for_data(3.0):
                raise RuntimeError("no AHRS data")
            base._zero_torque_preflight(mb, key, unwrap)

            q, qd = base._joint_state(mb, unwrap)
            shared = tw.TorqueShared(q)
            shared.force_frac = args.force_frac
            shared.leg_gravity = args.leg_gravity
            worker = tw.start(shared, imu, gains, cfg, mass)

            now = time.perf_counter()
            seq = StandSequence(now, args.stand_height)
            gate = law.TorqueSafetyGate(args.tau_max)
            miss = base.CanMissMonitor(mb)
            armed = False
            limp = False
            q_ref = q.copy()
            tau_cmd = np.zeros(N_JOINTS)
            stop_reason = None
            last_print = 0.0

            slot = mb.slot(CONTROL_HZ)
            deadline = time.perf_counter() + slot
            index = 0
            print("[t2] CROUCH: native position to the recorded pose")

            while stop_reason is None:
                mb.poll()
                j = index % N_JOINTS
                if j == 0:
                    now = time.perf_counter()
                    q, qd = base._joint_state(mb, unwrap)
                    tau_meas = np.array([mb.torques_nm()[m] for m in MOTOR_IDS])
                    shared.q, shared.qd = q, qd
                    shared.stage = seq.stage
                    seq.update(now)

                    # EVERY sweep, in every stage.  CanMissMonitor DIFFERENCES
                    # a cumulative counter, so skipping calls does not skip
                    # the misses -- it piles them into the next call and the
                    # "consecutive" streak becomes "everything since I last
                    # looked".  Polling it only while torque was active turned
                    # a slow intermittent miss rate into a single 1447-long
                    # streak the instant RISE armed, and tripped MISS_ESTOP
                    # (20) on the first torque sweep.
                    miss_streaks = miss.update(mb)

                    pressed = key.get()
                    if pressed in ("x", "X"):
                        stop_reason = "operator X"
                        break
                    if pressed == P.KEY_LIMP and not limp:
                        limp = True
                        print("[t2] LIMP")
                    elif pressed in ("\r", "\n"):
                        if limp:
                            limp = False
                            gate.start(now, q)
                            q_ref = q.copy()
                            print("[t2] re-engaged")
                        elif seq.stage == "WAIT_CROUCH" and seq.start_rise(now):
                            gate.start(now, q)
                            armed = True
                            print(f"[t2] RISE: {seq.z_crouch*1e3:.0f} -> "
                                  f"{args.stand_height*1e3:.0f} mm over "
                                  f"{seq.t_rise:.0f} s")
                    elif pressed in ("p", "P") and seq.park(now):
                        print("[t2] PARK")

                    # ---- stage machine -------------------------------------
                    if seq.stage == "CROUCH":
                        if seq.crouch_settled(now, qd, q):
                            state = shared.state
                            z = (bs.fk_trunk_height(q, np.eye(3),
                                                    np.ones(4, dtype=bool))
                                 if state is None else
                                 bs.fk_trunk_height(q, np.eye(3),
                                                    np.ones(4, dtype=bool)))
                            seq.enter_wait(now, z)
                            q_ref = q.copy()
                            print(f"[t2] WAIT_CROUCH at {z*1e3:.0f} mm.  "
                                  f"ENTER to rise under torque, X to abort")

                    shared.z_des = seq.z_des(now)

                    # ---- latch: in torque mode this is a collapsing leg ----
                    errors = mb.errors()
                    latched = [m for m in MOTOR_IDS if (errors[m] or 0) & 0x80]
                    if latched and seq.torque_active and P.LATCH_LIMPS_ROBOT \
                            and not limp:
                        limp = True
                        print(f"[t2] LIMP: input-lost latch on {latched}")

                    # ---- torque ------------------------------------------
                    if seq.torque_active and armed and not limp:
                        stale = tw.stale_s(shared)
                        if stale > P.WORKER_STALE_S:
                            limp = True
                            print(f"[t2] LIMP: worker stale {stale*1e3:.0f} ms")
                        elif not shared.active:
                            # the worker is publishing leg-gravity only; still
                            # safe to command, the impedance holds the pose
                            pass
                        # ---- deliberate fault injection (stage T2) --------
                        # Raise ONE foot while the stance law still believes
                        # it is planted.  That is precisely the 2026-07-30
                        # failure mode: the wrench keeps assigning that leg a
                        # ground reaction that is not there.  Uncorrected it
                        # predicts qd_abd -> tau/kd = 1.87 rad/s; with the
                        # leg-gravity fix and the impedance floor it should
                        # stay under 0.3.  The lift goes into the IMPEDANCE
                        # REFERENCE, not the contact mask, so the force law
                        # is misinformed exactly as it was then.
                        ref = q_ref
                        if unload_i is not None and seq.stage == "HOLD":
                            ref = q_ref.copy()
                            sl = slice(3 * unload_i, 3 * unload_i + 3)
                            fr = st.leg_frames(LEGS[unload_i], q[sl])
                            J = st.foot_jacobian_from(fr[0], fr[1], fr[2])
                            try:
                                ref[sl] = q_ref[sl] + np.linalg.solve(
                                    J, np.array([0.0, 0.0, args.unload_mm
                                                 * 1e-3]))
                            except np.linalg.LinAlgError:
                                pass
                        raw = shared.tau_ff + imp.tau(q, qd, ref)
                        tau_cmd = gate.apply(raw, q, now)
                        tau_cmd, n_latched = brake.apply(tau_cmd, qd, now,
                                                         cap=args.tau_max)

                        reason = brake.stop_reason(now)
                        if reason:
                            stop_reason = reason
                            break
                        reason = gate.estop_reason(
                            q, qd, base._temperatures(mb), miss_streaks,
                            mb.errors(), now,
                            enforce_position_limits=not args.no_limit_check)
                        if reason:
                            stop_reason = reason
                            break
                        state = shared.state
                        if state is not None and abs(
                                np.degrees(state.roll)) > P.TILT_STOP_DEG:
                            stop_reason = "tilt stop"
                            break
                    elif limp:
                        tau_cmd = np.zeros(N_JOINTS)
                    else:
                        tau_cmd = np.zeros(N_JOINTS)

                    # q_ref tracks the sequence while torque is off, so
                    # re-engaging never steps the impedance
                    if not seq.torque_active:
                        q_ref = q.copy()

                    tot = loads.update(q, tau_meas)
                    if seq.stage == "HOLD" and loads.out_of_bounds() \
                            and not limp:
                        limp = True
                        print(f"[t2] LIMP: foot load sum {tot:.1f} N vs "
                              f"{P.WEIGHT_N:.1f} N expected")

                    if args.raw_log:
                        log.append((now, q.copy(), qd.copy(), tau_cmd.copy(),
                                    tau_meas.copy(), loads.last.copy(),
                                    seq.stage))

                    if now - last_print >= 1.0 / base.STATUS_HZ:
                        s = shared.state
                        att = s.status() if s is not None else "no state"
                        # tracking is only meaningful once we are commanding
                        # something; dividing by ~0 in the position stages
                        # printed 12618909%
                        cmd_sum = float(np.sum(np.abs(tau_cmd)))
                        trk = (float(np.sum(np.abs(tau_meas))) / cmd_sum
                               if cmd_sum > 0.05 else float("nan"))
                        trk_s = "  --" if not np.isfinite(trk) \
                            else f"{trk*100:3.0f}%"
                        zd = seq.z_des(now)
                        print(f"[t2] {'LIMP' if limp else seq.stage:11s} "
                              f"z*={0.0 if zd is None else zd*1e3:5.0f}mm  "
                              f"{att}  |tau|="
                              f"{float(np.max(np.abs(tau_cmd))):4.2f} "
                              f"trk={trk_s}  load={tot:5.1f}N  "
                              f"|dq|={imp.worst_dq()*1e3:4.1f}mrad "
                              f"brk={int(brake.latched.sum())}", flush=True)
                        last_print = now

                mid = MOTOR_IDS[j]
                if seq.stage == "CROUCH":
                    mb.position(mid, POSITION_TARGET_DEG[j],
                                args.crouch_max_speed_dps)
                else:
                    mb.torque(mid, float(tau_cmd[j]))
                index += 1
                mb.pace(deadline)
                deadline += slot

            print(f"[t2] stopped: {stop_reason}")
            # capture before the bus closes -- this is the report that tells
            # you whether a MISS_ESTOP was the controller's fault or a motor's
            bus = bus_report(mb)
            base._soft_stop(mb)
    except KeyboardInterrupt:
        print("\n[t2] aborted")
    finally:
        if shared is not None:
            shared.run = False
        if worker is not None:
            worker.join(timeout=1.0)
        try:
            imu.stop()
        except Exception:
            pass
        key.close()

    print()
    print("=" * 78)
    print("T2/T3/T4 torque stand")
    print("=" * 78)
    print(loads.report())
    print(brake.summary())
    if bus:
        print(bus)
    if args.raw_log and log:
        np.savez_compressed(
            args.raw_log,
            t=np.array([r[0] for r in log]),
            q=np.array([r[1] for r in log]),
            qd=np.array([r[2] for r in log]),
            tau_cmd=np.array([r[3] for r in log]),
            tau_meas=np.array([r[4] for r in log]),
            foot_load=np.array([r[5] for r in log]),
            stage=np.array([r[6] for r in log]),
            mass=mass, leg_gravity=args.leg_gravity,
            kp_att=gains.kp_roll, kd_att=gains.kd_roll)
        print(f"  raw log -> {args.raw_log} ({len(log)} sweeps)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--stand-height", type=float,
                    default=P.STAND_HEIGHT_DEFAULT)
    ap.add_argument("--tau-max", type=float, default=P.TAU_START_MAX)
    ap.add_argument("--kp", type=float, default=P.KP_JOINT_IMP)
    ap.add_argument("--kd", type=float, default=P.KD_JOINT_IMP)
    ap.add_argument("--force-frac", type=float, default=P.FORCE_FRAC_DEFAULT,
                    help="0 = pure joint impedance through the torque path, "
                         "1 = full VMC force")
    ap.add_argument("--mass", type=float, default=P.DOG5_MASS_KG)
    ap.add_argument("--no-leg-gravity", dest="leg_gravity",
                    action="store_false",
                    help="reproduce the 2026-07-30 stance law (A/B only)")
    ap.add_argument("--kp-att", type=float, default=None,
                    help="attitude stiffness; 0 with --kd-att 0 is the "
                         "ablation half of the disturbance A/B")
    ap.add_argument("--kd-att", type=float, default=None)
    ap.add_argument("--kp-z", type=float, default=None)
    ap.add_argument("--unload-leg", choices=list(LEGS), default=None,
                    help="stage T2 fault injection: raise this foot in HOLD "
                         "while the stance law still believes it planted")
    ap.add_argument("--unload-mm", type=float, default=15.0)
    ap.add_argument("--crouch-max-speed-dps", type=float, default=100.0)
    ap.add_argument("--raw-log", default=None)
    ap.add_argument("--no-limit-check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.path.insert(0, os.path.join(_AUG, "self-test"))
        import test_torque_runners                        # noqa: PLC0415
        return test_torque_runners.self_test()
    if not 0.0 < args.tau_max <= P.TAU_STAGED_MAX:
        ap.error(f"--tau-max must be in (0, {P.TAU_STAGED_MAX}]")
    if not 0.0 <= args.force_frac <= 1.0:
        ap.error("--force-frac must be in [0, 1]")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
