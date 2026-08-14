#!/usr/bin/env python3
"""Stage T1 -- can software torque hold a pose at this loop rate?

THE QUESTION THIS STAGE EXISTS TO SETTLE
    On 2026-07-30 the torque-mode track was abandoned.  vmc/stand_hier_hw.py:3-6:

        "software torque at the 20.8 Hz per-joint CAN rate cannot stabilise
         the legs; native position mode is proven on this rig"

    That rate is wrong by a factor of twelve.  motorbus.slot() is
    1/(rate_hz * n_motors), so a slot is 333 us, a full 12-motor sweep is
    4 ms, and every joint is commanded AND replies at 250 Hz -- the 0xA1
    torque reply carries encoder, speed and iq in the same frame, so
    commanding a joint is sampling it.  At 48 ms no joint-space PD is stable
    at any useful gain; at 4 ms the bound is kp ~40-80, kd ~4.4.

    So this runner does the simplest possible thing that would have been
    impossible if the verdict were true: hold a pose under joint impedance,
    in torque mode, and see whether it holds.

        tau = kp (q_ref - q) - kd qd + leg_gravity(q)

    No wrench.  No grasp map.  No estimator.  No IMU.  Nothing that could
    fail for an interesting reason.  If this holds, the 2026-07-30 verdict is
    reversed by measurement, and stage T2 can put the force law on top with
    the impedance still underneath it.  If it oscillates at a fixed
    amplitude, the joint inertias in torque_params (J_KNEE, J_ABD) are wrong
    -- NOT the loop rate -- and the bounds need re-deriving before anything
    else happens.

WHY leg_gravity IS IN HERE AND NOT LEFT TO THE IMPEDANCE
    Without it the impedance has to develop the holding torque itself, so it
    sits at a permanent offset dq = tau_gravity/kp -- 31 mrad on abduction at
    kp = 15.  That is a bias, not an instability, but it would be read as
    "the hold is poor" when the gains are fine.  With the feedforward, dq is
    the MODEL ERROR, which is the quantity worth watching.

WHAT TO WATCH
    |dq|   settles below IMP_DQ_NOTICE_RAD and STOPS MOVING.  Drifting means
           the feedforward is wrong; oscillating at fixed amplitude means the
           gains are past the discrete bound -- abort and re-derive.
    trk    |tau_meas| / |tau_cmd|.  Below 80% the current loop is not
           delivering, and T0 should have caught it.
    blk    the in-sweep control block, against the 333 us slot.
    gap    per-motor command gap.  Anything over 8 ms is a step toward the
           10 ms watchdog on CAN 1/7/8/9.

RUN -- after tau_calib_hw has passed, before any force law exists
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    cd /home/robot01/Documents/can_motor_control/dog_stand_compliance_control/Aug_trotinplace_closedloop

    $V self-test/test_torque_runners.py       # offline gates first
    $V self-test/test_stance_law.py           # the stability prediction

    # ROBOT SUPPORTED, feet on the floor, TETHERED.  Climb the gain:
    sudo HOME=$HOME chrt -f 50 $V torque_mode_control/torque_hold_hw.py --kp 5  --kd 0.6
    sudo HOME=$HOME chrt -f 50 $V torque_mode_control/torque_hold_hw.py --kp 15 --kd 0.6
    sudo HOME=$HOME chrt -f 50 $V torque_mode_control/torque_hold_hw.py --kp 30 --kd 0.6

    # the independent check on both the gain and the gravity model: with the
    # feedforward off, |dq| should settle at exactly tau_gravity / kp
    sudo HOME=$HOME chrt -f 50 $V torque_mode_control/torque_hold_hw.py \
        --kp 15 --no-leg-gravity

    `sudo chrt -f 50` matters here -- this is the first script that holds the
    robot up with software torque, so a late sweep is a real gap.

Keys:  SPACE = limp (torque off, loop and keep-alives still running)
       ENTER = re-engage from limp
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
for _p in (_HERE, _DESC, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import motorbus                                            # noqa: E402
import stand_dog5_hw as base                               # noqa: E402
import crawl_dog5_hw as crawl                              # noqa: E402
import stance_law as law                                   # noqa: E402
import torque_params as P                                  # noqa: E402

MOTOR_IDS = base.MOTOR_IDS
N_JOINTS = base.N_JOINTS
CONTROL_HZ = base.CONTROL_HZ
LEGS = base.LEGS


class HoldStats:
    """Settled |dq| statistics, and the drift/oscillation discriminator.

    A hold can fail two ways that look the same on a status line: a slow
    drift (feedforward wrong) and a fixed-amplitude oscillation (gains past
    the discrete bound).  They need different fixes, so score them apart:
    drift is |mean(second half) - mean(first half)|; oscillation is the
    peak-to-peak that remains after removing that trend.
    """

    def __init__(self, settle_s=2.0):
        self.settle_s = float(settle_s)
        self.t0 = None
        self.samples = []

    def add(self, now, dq):
        if self.t0 is None:
            self.t0 = now
        if now - self.t0 >= self.settle_s:
            self.samples.append(float(np.max(np.abs(dq))))

    def report(self):
        if len(self.samples) < 20:
            return "  [hold] too few settled samples to score"
        a = np.array(self.samples)
        half = len(a) // 2
        drift = abs(float(a[half:].mean() - a[:half].mean()))
        p2p = float(np.ptp(a))
        lines = [
            f"  n={len(a)}  mean |dq| {a.mean()*1e3:5.1f} mrad   "
            f"p2p {p2p*1e3:5.1f}   drift {drift*1e3:+5.1f}",
        ]
        if a.mean() > P.IMP_DQ_NOTICE_RAD:
            lines.append("  the hold is loose: raise kp, or the gravity "
                         "feedforward is wrong (check T0's s*)")
        if drift > 0.3 * max(p2p, 1e-9) and drift > 2e-3:
            lines.append("  DRIFTING -- a steady creep is a feedforward "
                         "error, not a gain problem")
        elif p2p > 0.5 * max(a.mean(), 1e-9) and p2p > 5e-3:
            lines.append("  OSCILLATING at fixed amplitude -- this is the "
                         "discrete stability bound.  J_KNEE/J_ABD in "
                         "torque_params are wrong, NOT the loop rate.")
        else:
            lines.append("  steady: torque mode holds a pose at 250 Hz")
        return "\n".join(lines)


class GapWatch:
    """Per-motor command gap, bucketed.  The watchdog is 10 ms on CAN 1/7/8/9
    and a torque-mode latch collapses that leg, so this is not diagnostics --
    it is the reason a run is or is not trustworthy."""

    def __init__(self):
        self.last = {}
        self.worst = {mid: 0.0 for mid in MOTOR_IDS}
        self.counts = {mid: [0] * (len(P.GAP_BUCKETS) + 1) for mid in MOTOR_IDS}

    def mark(self, mid, now):
        prev = self.last.get(mid)
        self.last[mid] = now
        if prev is None:
            return
        gap = now - prev
        self.worst[mid] = max(self.worst[mid], gap)
        for k, edge in enumerate(P.GAP_BUCKETS):
            if gap < edge:
                self.counts[mid][k] += 1
                return
        self.counts[mid][-1] += 1

    def worst_overall(self):
        return max(self.worst.values()) if self.worst else 0.0

    def report(self):
        bad = [(mid, g) for mid, g in self.worst.items() if g > P.GAP_ABORT_S]
        out = [f"  worst command gap {self.worst_overall()*1e3:.1f} ms "
               f"(abort threshold {P.GAP_ABORT_S*1e3:.0f} ms, watchdog "
               f"{P.WATCHDOG_S*1e3:.0f} ms)"]
        if bad:
            out.append("  over threshold: " + ", ".join(
                f"CAN{mid} {g*1e3:.1f}ms" for mid, g in sorted(bad)))
        return "\n".join(out)


class BlockTimer:
    def __init__(self):
        self.worst = 0.0
        self.total = 0.0
        self.n = 0

    def add(self, dt):
        self.worst = max(self.worst, dt)
        self.total += dt
        self.n += 1

    def report(self):
        mean = self.total / max(self.n, 1)
        return (f"  in-sweep block: mean {mean*1e6:.0f} us, worst "
                f"{self.worst*1e6:.0f} us, against a {P.SLOT_S*1e6:.0f} us slot")


def run(args):
    base.validate_hardware_config()
    kd_bound = 2.0 * P.J_MIN / P.SWEEP_S
    print(f"[t1] sweep {P.SWEEP_S*1e3:.0f} ms -> every joint at "
          f"{CONTROL_HZ:.0f} Hz.  Sampled-damper bound 2J/dt = "
          f"{kd_bound:.1f} Nms/rad")
    print(f"[t1] (at the 48 ms this track was abandoned on it would be "
          f"{2*P.J_MIN/(P.N_JOINTS/CONTROL_HZ):.2f} -- see torque_params)")
    wn = (args.kp / P.J_KNEE) ** 0.5
    zeta = args.kd / (2.0 * (args.kp * P.J_KNEE) ** 0.5)
    print(f"[t1] kp={args.kp} kd={args.kd} -> omega_n {wn:.0f} rad/s, "
          f"omega_n*dt {wn*P.SWEEP_S:.3f}, zeta {zeta:.2f}")
    print(f"[t1] torque cap {args.tau_max} Nm, ramp {P.TORQUE_RAMP_S} s, "
          f"slew {P.TAU_SLEW_NM_S} Nm/s")

    imp = law.JointImpedance(kp=args.kp, kd=args.kd)
    stats = HoldStats(settle_s=args.settle)
    gaps = GapWatch()
    block = BlockTimer()

    key = base.KeyPoller()
    try:
        with motorbus.MotorBus(MOTOR_IDS, dirs=base.MOTOR_DIRECTIONS) as mb:
            unwrap = [base.CalibratedEncoderUnwrap() for _ in MOTOR_IDS]
            mb.arm(rate_hz=CONTROL_HZ)
            q_ref = base._zero_torque_preflight(mb, key, unwrap)
            print("[t1] holding the measured pose under torque.  "
                  "SPACE = limp, ENTER = re-engage, X = stop")

            gate = law.TorqueSafetyGate(args.tau_max)
            now = time.perf_counter()
            gate.start(now, q_ref)
            miss = base.CanMissMonitor(mb)

            slot = mb.slot(CONTROL_HZ)
            deadline = time.perf_counter() + slot
            index = 0
            tau_cmd = np.zeros(N_JOINTS)
            limp = False
            stop_reason = None
            last_print = 0.0
            t_end = time.perf_counter() + args.seconds

            while stop_reason is None:
                mb.poll()
                j = index % N_JOINTS
                if j == 0:
                    t_block = time.perf_counter()
                    now = t_block
                    q, qd = base._joint_state(mb, unwrap)
                    tau_meas = np.array([mb.torques_nm()[m] for m in MOTOR_IDS])
                    # every sweep, LIMP included: CanMissMonitor differences a
                    # cumulative counter, so a stalled poll does not skip the
                    # misses, it piles them into the next call and the streak
                    # becomes "everything since I last looked"
                    miss_streaks = miss.update(mb)

                    pressed = key.get()
                    if pressed in ("x", "X"):
                        stop_reason = "operator X"
                        break
                    if pressed == P.KEY_LIMP and not limp:
                        limp = True
                        print("[t1] LIMP -- torque zero, loop still running")
                    elif pressed in ("\r", "\n") and limp:
                        limp = False
                        gate.start(now, q)
                        q_ref = q.copy()
                        print("[t1] re-engaged at the current pose")

                    # a stance-leg latch means that leg has STOPPED producing
                    # torque and is collapsing -- limp everything, do not try
                    # to recover in place the way position mode does
                    errors = mb.errors()
                    latched = [m for m in MOTOR_IDS if (errors[m] or 0) & 0x80]
                    if latched and not limp and P.LATCH_LIMPS_ROBOT:
                        limp = True
                        print(f"[t1] LIMP: input-lost latch on {latched} -- in "
                              f"torque mode that leg stops pushing")

                    if limp:
                        tau_cmd = np.zeros(N_JOINTS)
                        imp.dq = np.zeros(N_JOINTS)
                    else:
                        raw = imp.tau(q, qd, q_ref)
                        if args.leg_gravity:
                            raw = raw + law.gravity_stack(q)
                        tau_cmd = gate.apply(raw, q, now)
                        stats.add(now, imp.dq)

                        reason = gate.estop_reason(
                            q, qd, base._temperatures(mb), miss_streaks,
                            mb.errors(), now,
                            enforce_position_limits=not args.no_limit_check)
                        if reason:
                            stop_reason = reason
                            break

                    if now - last_print >= 1.0 / base.STATUS_HZ:
                        trk = (float(np.sum(np.abs(tau_meas)))
                               / max(float(np.sum(np.abs(tau_cmd))), 1e-6))
                        support, _ = crawl.foot_load_map(q, tau_meas)
                        tot = float(np.nansum([support[l] for l in LEGS]))
                        print(f"[t1] {'LIMP ' if limp else 'HOLD '}"
                              f"|dq|={imp.worst_dq()*1e3:5.1f}mrad  "
                              f"|tau|={float(np.max(np.abs(tau_cmd))):4.2f}Nm "
                              f"trk={trk*100:3.0f}%  load={tot:5.1f}/"
                              f"{P.WEIGHT_N:.0f}N  "
                              f"blk={block.worst*1e6:3.0f}us  "
                              f"gap={gaps.worst_overall()*1e3:4.1f}ms",
                              flush=True)
                        last_print = now

                    if time.perf_counter() > t_end:
                        stop_reason = "time limit"
                    block.add(time.perf_counter() - t_block)

                mid = MOTOR_IDS[j]
                gaps.mark(mid, time.perf_counter())
                mb.torque(mid, float(tau_cmd[j]))
                index += 1
                mb.pace(deadline)
                deadline += slot

            print(f"[t1] stopped: {stop_reason}")
            base._soft_stop(mb)
    except KeyboardInterrupt:
        print("\n[t1] aborted")
    finally:
        key.close()

    print()
    print("=" * 78)
    print(f"T1 hold  kp={args.kp} kd={args.kd}")
    print("=" * 78)
    print(stats.report())
    print(block.report())
    print(gaps.report())
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kp", type=float, default=P.KP_JOINT_IMP)
    ap.add_argument("--kd", type=float, default=P.KD_JOINT_IMP)
    ap.add_argument("--tau-max", type=float, default=P.TAU_START_MAX)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--no-leg-gravity", dest="leg_gravity",
                    action="store_false",
                    help="drop the gravity feedforward; |dq| then shows the "
                         "gravity torque divided by kp, which is a useful "
                         "independent check of both")
    ap.add_argument("--no-limit-check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.path.insert(0, os.path.join(_AUG, "self-test"))
        import test_torque_runners                         # noqa: PLC0415
        return test_torque_runners.self_test()
    if not 0.0 < args.tau_max <= P.TAU_STAGED_MAX:
        ap.error(f"--tau-max must be in (0, {P.TAU_STAGED_MAX}]")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
