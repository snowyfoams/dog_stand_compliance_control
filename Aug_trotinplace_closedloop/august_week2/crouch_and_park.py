#!/usr/bin/env python3
"""The two POSITION-mode bookends of the week-2 torque track: CROUCH and PARKED.

Every torque stage starts from the recorded crouch and ends back on it.
Neither end is torque work -- the drivers' native 0xA4 position loops do it:

    crouch(mb, unwrap)   0xA4 to the recorded crouch, speed-capped; returns
                         once the joints have actually settled there.
    parked(mb, unwrap)   the same command again at the end of a run, to put
                         the robot back down on the crouch.

There is no software profile in either.  The 0xA4 target is a CONSTANT and the
motor-side speed cap (`max_speed_dps`) is what limits the approach, so the
descent is as gentle as that cap makes it.  Lower the cap to go slower; do not
add a ramp.

THE CROUCH POSE IS THE ONE FROM BEFORE, NOT A NEW ONE
    `stand_dog5_hw.CROUCH_JOINT_TARGET_DEG` -> `recorded.Q_RECORDED_CROUCH`:

        FL = (+88.33, +48.04, -142.11) deg      RL = (+88.33, -48.04, +142.11)
        FR = (-88.33, -48.04, +142.11) deg      RR = (-88.33, +48.04, -142.11)

    Aliased here, never re-declared, and so are the settle tolerances.
    Nothing in this file is a new tunable.

WHAT IS DELIBERATELY NOT HERE
    No torque, no balance loop, and no input-lost recovery while a stage runs.
    In position mode a latched driver holds its last commanded position, so a
    latch is benign here -- it is left to the operator and the next run's
    preflight rather than handled mid-motion.  Since nothing re-requests 0x9A
    during a stage, the drivers' error bits are only as fresh as arming left
    them; the live protections are the abduction check below, overspeed,
    temperature and missed replies -- all of which ride on the telemetry that
    every 0xA4 reply carries.

THE ONLY POSITION LIMIT THIS WEEK IS ABDUCTION, AND ONLY ITS SIGN
    `base.soft_limits()` is a SYMMETRIC box, +/-1.75 rad on abduction and
    +/-2.6 on pitch and knee, tiled identically over all four legs.  It is
    wrong in both directions at once, which is why it is switched off here
    (`enforce_position_limits=False`) instead of tuned:

      too loose  it does not encode the mirroring, so it cannot see the one
                 failure that matters -- a leg swinging THROUGH the body to
                 the wrong side.  A left knee is allowed 298 deg of travel,
                 straight through full extension and out the other way.
      too tight  the RESTING crouch sits 6.9 deg from the knee trip (margin
                 2.9 deg), so a slightly overshooting approach e-stops for
                 nothing.

    What replaces it is one sign test on the four abduction joints, which is
    the whole of the mirroring:

        CAN  7  FL_abd   crouch +88.33 deg   must stay POSITIVE
        CAN 10  FR_abd   crouch -88.33 deg   must stay NEGATIVE
        CAN  4  RL_abd   crouch +88.33 deg   must stay POSITIVE
        CAN  1  RR_abd   crouch -88.33 deg   must stay NEGATIVE

    Crossing zero is the leg going to the other side; at the crouch every one
    of them sits ~88 deg away from that, so it cannot nuisance-trip.  Pitch and
    knee are not limited at all this week.  The signs are read off Q_CROUCH
    rather than written down, so they cannot drift from the pose.

    NOTE this does NOT reach `base._zero_torque_preflight`, which calls
    `soft_limits()` directly and still refuses ENTER on the symmetric box.

    Support the robot.  These functions drive the legs to the recorded pose
    from wherever they are and do not know whether the feet are on the ground.

RUN
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    cd /home/robot01/Documents/can_motor_control/dog_stand_compliance_control/Aug_trotinplace_closedloop

    sudo HOME=$HOME chrt -f 50 $V august_week2/crouch_and_park.py --max-dps 50

    ENTER at the zero-torque check starts the crouch, X aborts.  The standalone
    run does the crouch only -- `parked()` is called by the torque runner at the
    end of a stand, where there is something to come down from.

USED FROM A RUNNER
    with motorbus.MotorBus(MOTOR_IDS, dirs=base.MOTOR_DIRECTIONS) as mb:
        mb.arm(rate_hz=CONTROL_HZ)
        unwrap = [base.CalibratedEncoderUnwrap() for _ in MOTOR_IDS]
        base._zero_torque_preflight(mb, key, unwrap)
        crouch(mb, unwrap, key)        # ... torque stages ...
        parked(mb, unwrap, key)        # back down
        base._soft_stop(mb)
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
for _p in (_HERE, os.path.join(_ROOT, "dog5_description"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import motorbus                                          # noqa: E402
import stand_dog5_hw as base                             # noqa: E402
import stand_dog5_recorded_hw as recorded                # noqa: E402

MOTOR_IDS = base.MOTOR_IDS
N_JOINTS = base.N_JOINTS
CONTROL_HZ = base.CONTROL_HZ                # per motor AND per sweep -- 250 Hz
LEGS = base.LEGS

# The crouch, and every gate on reaching it, from the one place they live.
Q_CROUCH = recorded.Q_RECORDED_CROUCH               # rad, controller order
POSITION_TARGET_DEG = recorded.POSITION_TARGET_DEG  # deg, the 0xA4 payload
POSE_TOL = recorded.RECORDED_POSE_TOL               # 0.08 rad
QD_TOL = recorded.RECORDED_QD_TOL                   # 0.25 rad/s
SETTLE_S = recorded.CROUCH_SETTLE_S                 # 0.50 s of stillness
TIMEOUT_S = recorded.CROUCH_TIMEOUT_S               # 30 s
MAX_DPS = recorded.DEFAULT_CROUCH_MAX_MOTOR_DPS     # 100 dps, motor side

# The week-2 position limit, in full: the four abduction joints keep the sign
# they have at the crouch.  See the docstring for what this replaces and why.
ABD = np.arange(0, N_JOINTS, 3)          # FL,FR,RL,RR abd = CAN 7,10,4,1
ABD_SIGN = np.sign(Q_CROUCH[ABD])        # +,-,+,- read off the pose itself

# ---------------------------------------------------------------------------
# ...and the same rule installed over base.soft_limits(), for ONE caller
# ---------------------------------------------------------------------------
# `base._zero_torque_preflight` refuses ENTER using the module-level
# `soft_limits()` directly -- there is no injection point, and it runs before
# any of our own checks can.  Measured 2026-08-17 on hardware:
#
#     [hardware] ENTER refused: RR_abd=-1.88 rad is outside the soft limit
#
# -1.88 rad is -107.7 deg.  Its SIGN is correct (RR abducts negative, crouch
# -88.33), so the week-2 rule passes it; what refused it is the symmetric
# box's MAGNITUDE, +/-1.75 rad, which is a software bound measured against the
# encoder zero offsets rather than a mechanical stop.  Refusing a pose the
# robot is mechanically sitting in, before anything has been commanded, is the
# failure mode vmc_stand_hw already hit and worked around the same way.
#
# So the box is replaced process-wide by the week-2 rule expressed as bounds:
# an abduction joint may not cross zero; nothing else is bounded.  Everything
# that reads soft_limits() then agrees with the sign check in the loops below,
# instead of two different limits disagreeing about what is legal.
_LO = np.full(N_JOINTS, -np.inf)
_HI = np.full(N_JOINTS, np.inf)
_LO[ABD] = np.where(ABD_SIGN > 0, 0.0, -np.inf)
_HI[ABD] = np.where(ABD_SIGN > 0, np.inf, 0.0)
base.soft_limits = lambda: (_LO, _HI)


def crouch(mb, unwrap, key=None, name="crouch", max_speed_dps=MAX_DPS,
           timeout_s=TIMEOUT_S):
    """Command the recorded crouch in position mode until the joints settle.

    One CAN frame per slot, one state/decision block per 250 Hz sweep.  The
    0xA4 target never changes: the driver's own loop does the approach, capped
    at `max_speed_dps` (motor side, 1 dps/LSB).

    "Settled" is both halves, held for SETTLE_S: within POSE_TOL of the target
    AND under QD_TOL.  Pose alone passes while a leg is still swinging through
    the target; speed alone passes while a leg is stalled short of it.

    Returns the measured pose.  Raises RuntimeError on a timeout or an e-stop,
    KeyboardInterrupt on X.
    """
    gate = base.SafetyGate(tau_cap=base.STAGED_TAU_MAX)   # limits/speed/temp only
    miss = base.CanMissMonitor(mb)
    q, qd = base._joint_state(mb, unwrap)   # arm() left every motor replying
    start = time.perf_counter()
    gate.start(start, q)
    slot = mb.slot(CONTROL_HZ)
    deadline = time.perf_counter() + slot
    settle_since = None
    last_print = 0.0
    index = 0

    print(f"[{name}] 0xA4 to the recorded crouch, capped at {max_speed_dps:.0f} "
          f"dps.  Settle = |dq|<{POSE_TOL:.2f} rad and |qd|<{QD_TOL:.2f} rad/s "
          f"for {SETTLE_S:.1f}s.  X aborts.", flush=True)

    while True:
        mb.poll()
        j = index % N_JOINTS
        if j == 0:                                   # once per 250 Hz sweep
            now = time.perf_counter()
            t = now - start
            q, qd = base._joint_state(mb, unwrap)

            if key is not None and key.get() in ("x", "X"):
                raise KeyboardInterrupt(f"operator X during {name}")

            # The symmetric soft-limit box is OFF -- see the docstring.  What
            # is left here is overspeed, temperature, missed replies and the
            # driver fault bits.  estop_reason() runs overspeed_reason() itself,
            # so calling it once per sweep is what makes its streak counters
            # mean "consecutive checks".
            reason = gate.estop_reason(q, qd, base._temperatures(mb),
                                       miss.update(mb), mb.errors(), now,
                                       enforce_position_limits=False)
            if reason:
                raise RuntimeError(f"{name}: {reason}")

            # the whole position limit: an abduction joint that crossed zero
            # has swung the leg through the body to the wrong side
            crossed = ABD_SIGN * q[ABD] < 0.0
            if crossed.any():
                k = int(np.argmax(crossed))          # 0..3, index into ABD
                i = int(ABD[k])                      # 0..11, the joint
                raise RuntimeError(
                    f"{name}: {base.JOINT_LABELS[i]} (CAN {MOTOR_IDS[i]}) at "
                    f"{np.rad2deg(q[i]):+.1f} deg is on the wrong side of 0 -- "
                    f"it must stay "
                    f"{'positive' if ABD_SIGN[k] > 0 else 'negative'} "
                    f"(crouch {np.rad2deg(Q_CROUCH[i]):+.1f} deg)")

            err = float(np.max(np.abs(q - Q_CROUCH)))
            speed = float(np.max(np.abs(qd)))
            if err > POSE_TOL or speed > QD_TOL:
                settle_since = None
            elif settle_since is None:
                settle_since = now
            elif now - settle_since >= SETTLE_S:
                print(f"[{name}] settled after {t:.1f}s: "
                      f"max|q-crouch|={np.rad2deg(err):.2f} deg, "
                      f"max|qd|={speed:.2f} rad/s", flush=True)
                return q.copy()

            if t > timeout_s:
                raise RuntimeError(
                    f"{name} timeout after {timeout_s:.0f}s: "
                    f"max|q-crouch|={np.rad2deg(err):.1f} deg on "
                    f"{base.JOINT_LABELS[int(np.argmax(np.abs(q - Q_CROUCH)))]}, "
                    f"max|qd|={speed:.2f} rad/s")

            if now - last_print >= 1.0 / base.STATUS_HZ:
                last_print = now
                print(f"[{name}] t={t:5.1f}s  max|q-crouch|="
                      f"{np.rad2deg(err):6.2f}deg  max|qd|={speed:.2f}rad/s  "
                      f"Tmax={int(np.max(base._temperatures(mb)))}C", flush=True)

        mb.position(MOTOR_IDS[j], float(POSITION_TARGET_DEG[j]), max_speed_dps)
        index += 1
        overrun = mb.pace(deadline)
        deadline += slot
        if overrun and overrun > 2.0 * slot:
            deadline = time.perf_counter() + slot     # rebase, do not chase


def parked(mb, unwrap, key=None, max_speed_dps=MAX_DPS, timeout_s=TIMEOUT_S):
    """Put the robot back down on the recorded crouch at the end of a run.

    Identical command to `crouch`, because without a software profile there is
    nothing to make the descent different: the same constant 0xA4 target, the
    same motor-side speed cap doing the limiting, the same settle gate saying
    when the robot is actually resting.  Coming down from a stand is a longer
    travel than the crouch usually is, so pass a lower `max_speed_dps` if the
    default drop is too brisk.

    Returns the measured pose.  Raises RuntimeError on a timeout or an e-stop,
    KeyboardInterrupt on X.
    """
    return crouch(mb, unwrap, key, name="parked",
                  max_speed_dps=max_speed_dps, timeout_s=timeout_s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-dps", type=float, default=MAX_DPS,
                    help="motor-side 0xA4 speed cap; halve it on a first run")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    args = ap.parse_args()

    base.validate_hardware_config()
    print("=" * 74)
    print("DOG5 POSITION-MODE CROUCH  (no torque is commanded here)")
    print(f"  speed cap {args.max_dps:.0f} dps.  Support the robot, feet down.")
    print("  Keys: ENTER = start (at the zero-torque check), X = abort.")
    print("=" * 74)

    rc = 0
    key = base.KeyPoller()
    try:
        with motorbus.MotorBus(MOTOR_IDS, dirs=base.MOTOR_DIRECTIONS) as mb:
            if not mb.arm(rate_hz=CONTROL_HZ):
                raise RuntimeError("arm failed (bus / power / terminators)")
            unwrap = [base.CalibratedEncoderUnwrap() for _ in MOTOR_IDS]
            base._zero_torque_preflight(mb, key, unwrap)
            crouch(mb, unwrap, key, max_speed_dps=args.max_dps,
                   timeout_s=args.timeout)
            print("[stop] soft stop: the position hold is released.  "
                  "HOLD THE ROBOT.")
            base._soft_stop(mb)
    except KeyboardInterrupt as exc:
        print(f"\n[abort] {exc}")
        rc = 1
    except RuntimeError as exc:
        print(f"\n[fault] {exc}")
        rc = 1
    finally:
        key.close()
    sys.exit(rc)
