#!/usr/bin/env python3
"""Stage T0 -- is a commanded newton-metre a real newton-metre?

Nothing downstream means anything until this passes.  The grasp map computes
forces, `tau = -J^T f` turns them into joint torques, and `mb.torque()` turns
those into an iq command through a single constant (config.torque_gain =
206.04 iq-LSB per output N*m).  If that constant, the gearbox friction or the
mass model is off, the robot is balancing on arithmetic.
CONTROL_ROADMAP.md:174-177 names this the phase's big unknown.

NO BALANCING HAPPENS HERE.  The robot sits on a stand with its feet in the
air (--hang) or stands on the floor unloaded (--floor).

WHY NOT "READ THE CURRENT AT ZERO TORQUE"
    The obvious version of this test -- command zero torque, let the legs
    hang, and check the measured iq equals leg_gravity_torque -- does not
    work, and it is worth writing down why so nobody re-invents it.  At
    iq = 0 the driver's current loop REGULATES current to zero: the motor
    produces no torque, the leg falls limp, and the measured current reads
    ~0 no matter what the leg weighs.  Measured torque only tells you
    something when the motor is being asked to hold something up.

WHAT --hang ACTUALLY MEASURES
    Hold the leg statically in the air under joint impedance plus a SCALED
    gravity feedforward:

        tau_cmd = s * tau_model(q) + kp (q_ref - q) - kd qd

    A stationary leg in the air is in equilibrium, so by Newton the total
    commanded torque IS the true gravity torque:

        s * tau_model + kp * dq = tau_true        (qd = 0 at rest)
        =>   dq = (tau_true - s * tau_model) / kp

    Sweep s and find the crossing.  The s at which dq goes to zero is
    tau_true / tau_model -- a direct, unit-free measurement of the product
    (iq->torque scale) x (mass-model accuracy), needing no test weights and
    no fixture.

    Gearbox stiction shows up honestly as a BAND of s over which the leg does
    not move at all rather than a single crossing.  The band's centre is the
    scale; its width is the friction.  Both are printed, because a wide band
    is the thing that will limit force control later.

WHAT --floor MEASURES
    Feet on the floor, robot holding its own weight, still no balancing.
    `crawl_dog5_hw.foot_load_map` inverts measured torque through J^-T to a
    per-foot support force.  The sum must be the robot: 57.0 N.  That single
    number validates torque_gain, the direction map, the Jacobian and the
    mass all at once, and it is the number the status line carries for every
    run after this one.

    Note it is taken at a STAND pose, not the crouch: foot_load_map's own
    docstring warns that a folded crouch has sigma_min ~0.018, which turns a
    few 0.01 N*m of torque noise into ~1 N of phantom force per leg.

RUN -- this is the FIRST torque-mode script to run on the robot, ever
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    cd /home/robot01/Documents/can_motor_control/dog_stand_compliance_control/Aug_trotinplace_closedloop

    $V self-test/test_torque_runners.py       # offline gates first

    # 1. ROBOT ON A STAND, FEET HANGING FREE.  Arrange the legs by hand
    #    during the zero-torque preflight, then ENTER.
    $V torque_mode_control/tau_calib_hw.py --hang

    # 2. FEET ON THE FLOOR, at a stand-like pose (NOT a folded crouch --
    #    foot_load_map goes near-singular there and invents ~1 N per leg).
    $V torque_mode_control/tau_calib_hw.py --floor --tau-max 2.0

    No `sudo chrt` needed: nothing here balances, so a late sweep costs a
    watchdog latch at worst, not a fall.

WHAT TO WATCH
    s*      per joint, (true gravity torque) / (model gravity torque).
            ~1.00 means torque_gain = 206.04 and the link masses are both
            right.  A CONSISTENT s* != 1 across joints is a torque-scale
            error; ONE bad joint is that motor.  Abduction should show
            ~0.47 Nm of model torque at the crouch and the knee ~0.002, so
            judge on abduction and hip pitch -- the knee holds a 38 g shin
            and is dominated by friction either way.
    band    a LOWER BOUND on the stiction deadband.  Wide bands are what
            will limit force control later, so record them.
    sum     (--floor) must be 57 +/- 5 N.  Low means commanded torque is not
            reaching the joint; high usually means a near-singular pose.

    A joint reading the WRONG SIGN is a direction-map error, not a scale
    error -- stop and check dog5_hardware_map.  A joint reading ~0 while
    visibly loaded has a dead current loop.

GATE T0 -- if this fails, everything downstream is fiction.  Stop here.
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
for _p in (_HERE, _DESC, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import motorbus                                            # noqa: E402
import stand_dog5_hw as base                               # noqa: E402
import crawl_dog5_hw as crawl                              # noqa: E402
import dog5_statics as st                                  # noqa: E402
import stance_law as law                                   # noqa: E402
import torque_params as P                                  # noqa: E402

MOTOR_IDS = base.MOTOR_IDS
N_JOINTS = base.N_JOINTS
CONTROL_HZ = base.CONTROL_HZ
LEGS = base.LEGS

JOINT_NAMES = [f"{leg}_{j}" for leg in LEGS
               for j in ("abd", "pitch", "knee")]


class ScaleSweep:
    """Dwell at each gravity scale, then record the settled impedance error.

    The settle gate is on JOINT SPEED, not a timer: stiction means the leg
    may not move at all at some scales, and a timer would record a pose that
    was still drifting at others.
    """

    def __init__(self, scales, dwell_s=2.0, settle_qd=0.05):
        self.scales = list(scales)
        self.dwell_s = float(dwell_s)
        self.settle_qd = float(settle_qd)
        self.index = 0
        self.t0 = None
        self.records = []          # (scale, dq(12,), tau_cmd(12,), tau_meas(12,))

    @property
    def scale(self):
        return self.scales[min(self.index, len(self.scales) - 1)]

    @property
    def done(self):
        return self.index >= len(self.scales)

    def update(self, now, dq, qd, tau_cmd, tau_meas):
        """Returns a message when a scale completes, else ''."""
        if self.done:
            return ""
        if self.t0 is None:
            self.t0 = now
            return f"scale {self.scale:.2f} ..."
        if now - self.t0 < self.dwell_s:
            return ""
        if float(np.max(np.abs(qd))) > self.settle_qd:
            return ""                      # still moving; keep waiting
        s = self.scale
        self.records.append((s, np.array(dq), np.array(tau_cmd),
                             np.array(tau_meas)))
        self.index += 1
        self.t0 = None
        return (f"scale {s:.2f} settled: max|dq| = "
                f"{float(np.max(np.abs(dq)))*1e3:.1f} mrad")


def _crossing(scales, errors):
    """Linear interpolation of the zero crossing of dq(s), plus the flat band.

    dq(s) is a straight line in exact statics (dq = (tau_true - s*tau_model)/kp),
    so a linear fit is the right estimator and its residual is the nonlinearity
    -- friction, mostly.
    """
    scales = np.asarray(scales, float)
    errors = np.asarray(errors, float)
    if scales.size < 2 or np.ptp(errors) < 1e-9:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(scales, errors, 1)
    if abs(slope) < 1e-12:
        return float("nan"), float("nan")
    s_star = -intercept / slope
    resid = float(np.max(np.abs(errors - (slope * scales + intercept))))
    band = resid / abs(slope)          # residual expressed back in scale units
    return float(s_star), band


def report_hang(sweep, q_ref, kp):
    print()
    print("=" * 78)
    print("T0 --hang : gravity-scale sweep (feet in the air)")
    print("=" * 78)
    if not sweep.records:
        print("  no settled scales recorded")
        return 1

    scales = [r[0] for r in sweep.records]
    print(f"  {'joint':10s} {'tau_model':>10s} {'s*':>7s} {'band':>7s} "
          f"{'tau_true':>9s} {'tau_meas/cmd':>13s}")
    tau_model = law.gravity_stack(q_ref)
    bad = []
    for j, name in enumerate(JOINT_NAMES):
        errs = [float(r[1][j]) for r in sweep.records]
        s_star, band = _crossing(scales, errs)
        track = [abs(r[3][j]) / max(abs(r[2][j]), 1e-6) for r in sweep.records]
        trk = float(np.median(track))
        tau_true = s_star * tau_model[j] if np.isfinite(s_star) else float("nan")
        print(f"  {name:10s} {tau_model[j]:+10.3f} {s_star:7.3f} {band:7.3f} "
              f"{tau_true:+9.3f} {trk*100:12.0f}%")
        # only judge joints whose gravity torque is big enough to measure;
        # the knee holds a 38 g shin and is dominated by friction
        if abs(tau_model[j]) > 0.05:
            if not (0.7 < s_star < 1.3):
                bad.append(f"{name} s*={s_star:.2f}")
            elif trk < P.TAU_TRACK_MIN_FRAC:
                bad.append(f"{name} tracking {trk*100:.0f}%")

    print()
    print("  s* is (true gravity torque) / (model gravity torque).  1.00 means")
    print("  torque_gain = 206.04 and the link masses are both right.  A")
    print("  consistent s* != 1 across joints is a torque-scale error; one bad")
    print("  joint is that motor.  `band` is the linear fit's residual in the")
    print("  same units -- a LOWER BOUND on the stiction deadband (it reads")
    print("  about half the true width), so treat it as 'at least this much")
    print("  friction', not a measurement of it.")
    print()
    if bad:
        print(f"  GATE T0 FAIL: {', '.join(bad)}")
        return 1
    print("  GATE T0 PASS (hang)")
    return 0


def report_floor(loads, q):
    print()
    print("=" * 78)
    print("T0 --floor : does the measured torque add up to the robot?")
    print("=" * 78)
    if not loads:
        print("  no samples")
        return 1
    arr = np.array(loads)                      # (n, 4)
    mean = np.nanmean(arr, axis=0)
    total = float(np.nansum(mean))
    for i, leg in enumerate(LEGS):
        print(f"  {leg}  {mean[i]:7.2f} N")
    print(f"  {'sum':4s} {total:7.2f} N   against a "
          f"{P.WEIGHT_N:.2f} N robot ({P.DOG5_MASS_KG} kg)")
    err = abs(total - P.WEIGHT_N) / P.WEIGHT_N
    print()
    if not np.isfinite(total) or err > 0.10:
        print(f"  GATE T0 FAIL: sum is {err*100:.0f}% out (limit 10%)")
        print("  A LOW sum means commanded torque is not reaching the joint")
        print("  (torque_gain too high, or friction eating it).  A HIGH sum")
        print("  usually means the pose is near-singular -- check the leg")
        print("  Jacobians, and take this at a stand pose, not a folded crouch.")
        return 1
    print(f"  GATE T0 PASS (floor): {err*100:.1f}% error")
    return 0


def run(args):
    base.validate_hardware_config()
    print(f"[t0] mass {st.total_mass():.4f} kg, weight {P.WEIGHT_N:.2f} N, "
          f"per foot {P.PER_FOOT_GRF_N:.2f} N")
    print(f"[t0] sweep {P.SWEEP_S*1e3:.0f} ms, slot {P.SLOT_S*1e6:.0f} us, "
          f"every joint commanded and read at {CONTROL_HZ:.0f} Hz")
    print(f"[t0] impedance kp={args.kp} kd={args.kd}, tau cap {args.tau_max} Nm")

    imp = law.JointImpedance(kp=args.kp, kd=args.kd)
    sweep = ScaleSweep(np.arange(args.scale_lo, args.scale_hi + 1e-9,
                                 args.scale_step), dwell_s=args.dwell)

    key = base.KeyPoller()
    rc = 1
    try:
        with motorbus.MotorBus(MOTOR_IDS, dirs=base.MOTOR_DIRECTIONS) as mb:
            unwrap = [base.CalibratedEncoderUnwrap() for _ in MOTOR_IDS]
            mb.arm(rate_hz=CONTROL_HZ)
            q_ref = base._zero_torque_preflight(mb, key, unwrap)
            what = "sweeping the gravity scale" if args.hang else \
                "reading foot load"
            print(f"[t0] holding the measured pose; {what}")

            gate = law.TorqueSafetyGate(args.tau_max)
            now = time.perf_counter()
            gate.start(now, q_ref)
            miss = base.CanMissMonitor(mb)

            slot = mb.slot(CONTROL_HZ)
            deadline = time.perf_counter() + slot
            index = 0
            tau_cmd = np.zeros(N_JOINTS)
            loads = []
            last_print = 0.0
            stop_reason = None
            t_end = time.perf_counter() + args.seconds

            while stop_reason is None:
                mb.poll()
                j = index % N_JOINTS
                if j == 0:
                    now = time.perf_counter()
                    q, qd = base._joint_state(mb, unwrap)
                    tau_meas = np.array([mb.torques_nm()[m] for m in MOTOR_IDS])
                    # every sweep, never as an estop argument: the monitor
                    # differences a cumulative counter, so a poll skipped by a
                    # branch hands its whole backlog to the next call
                    miss_streaks = miss.update(mb)

                    pressed = key.get()
                    if pressed in ("x", "X"):
                        stop_reason = "operator X"
                        break

                    tau_model = law.gravity_stack(q)
                    scale = sweep.scale if args.hang else 1.0
                    raw = scale * tau_model + imp.tau(q, qd, q_ref)
                    tau_cmd = gate.apply(raw, q, now)

                    reason = gate.estop_reason(
                        q, qd, base._temperatures(mb), miss_streaks,
                        mb.errors(), now,
                        enforce_position_limits=not args.no_limit_check)
                    if reason:
                        stop_reason = reason
                        break

                    if args.hang:
                        msg = sweep.update(now, imp.dq, qd, tau_cmd, tau_meas)
                        if msg:
                            print(f"[t0] {msg}")
                        if sweep.done:
                            stop_reason = "sweep complete"
                            break
                    else:
                        support, _ = crawl.foot_load_map(q, tau_meas)
                        loads.append([support[leg] for leg in LEGS])

                    if now - last_print >= 1.0 / base.STATUS_HZ:
                        trk = (float(np.sum(np.abs(tau_meas)))
                               / max(float(np.sum(np.abs(tau_cmd))), 1e-6))
                        extra = (f"s={sweep.scale:.2f}" if args.hang else
                                 f"load={np.nansum(loads[-1]):5.1f}N")
                        print(f"[t0] |dq|={imp.worst_dq()*1e3:5.1f}mrad  "
                              f"|tau|={float(np.max(np.abs(tau_cmd))):4.2f}Nm  "
                              f"trk={trk*100:3.0f}%  {extra}", flush=True)
                        last_print = now

                    if time.perf_counter() > t_end:
                        stop_reason = "time limit"

                mid = MOTOR_IDS[j]
                if not mb.torque(mid, float(tau_cmd[j])):
                    pass
                index += 1
                mb.pace(deadline)
                deadline += slot

            print(f"[t0] stopped: {stop_reason}")
            base._soft_stop(mb)
            rc = (report_hang(sweep, q_ref, args.kp) if args.hang
                  else report_floor(loads, q_ref))
    except KeyboardInterrupt:
        print("\n[t0] aborted")
    finally:
        key.close()
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--hang", action="store_true",
                      help="feet in the air: sweep the gravity scale")
    mode.add_argument("--floor", action="store_true",
                      help="feet on the floor: sum the per-foot support force")
    ap.add_argument("--tau-max", type=float, default=P.TAU_START_MAX)
    ap.add_argument("--kp", type=float, default=P.KP_JOINT_IMP)
    ap.add_argument("--kd", type=float, default=P.KD_JOINT_IMP)
    ap.add_argument("--scale-lo", type=float, default=0.7)
    ap.add_argument("--scale-hi", type=float, default=1.3)
    ap.add_argument("--scale-step", type=float, default=0.1)
    ap.add_argument("--dwell", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--no-limit-check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.path.insert(0, os.path.join(_AUG, "self-test"))
        import test_torque_runners                          # noqa: PLC0415
        return test_torque_runners.self_test()
    if not 0.0 < args.tau_max <= P.TAU_STAGED_MAX:
        ap.error(f"--tau-max must be in (0, {P.TAU_STAGED_MAX}]")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
