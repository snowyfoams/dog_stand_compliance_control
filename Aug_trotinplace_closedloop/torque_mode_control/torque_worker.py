#!/usr/bin/env python3
"""The 100 Hz half of the torque stand: body state, wrench, GRF, feedforward.

WHY IT IS A THREAD AND NOT PART OF THE SWEEP
    Measured on this Pi, per 4-leg call: the wrench + grasp map + corrected
    stance law is ~1.5 ms.  A CAN slot is 333 us and the driver watchdog on
    CAN 1/7/8/9 is 10 ms, so this cannot run in the sweep -- not "should
    not", cannot.  The sweep keeps commanding every motor at 250 Hz while
    this thread recomputes the feedforward at 100 Hz, so the watchdog is fed
    regardless of what the controller is doing.

WHAT THE SPLIT BUYS, AND WHY IT IS NOT vmc_stand_hw's
    vmc_stand_hw._worker put the EKF *and* the whole torque law on one 100 Hz
    thread and had the CAN loop stream the result, so the commanded torque
    could be 10 ms stale -- including its velocity feedback.  Here the thread
    publishes only the FEEDFORWARD (tau_ff) and the reference (q_ref); the
    velocity-dependent part, which is what actually stabilises a joint, is
    evaluated in the sweep on telemetry at most one 4 ms sweep old.

    That is the whole payoff of the corrected loop rate, and it is spent on
    the one term that needed it.

PUBLISHING DISCIPLINE
    Every field is replaced by whole-object assignment (an atomic reference
    swap under the GIL), never mutated in place, so the CAN thread can never
    read a half-written vector.  Same contract as ekf_runtime.EkfShared.

RUN
    Not a runner: it is the thread torque_stand_hw starts, and it has no main.
    It touches no bus -- the CAN loop owns every frame -- so the only thing
    that can go wrong here is publishing something stale or a fiction, which
    is what the staleness and refusal paths above exist to prevent.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    cd /home/robot01/Documents/can_motor_control/dog_stand_compliance_control/Aug_trotinplace_closedloop

    $V self-test/test_body_state.py        # the refusals it depends on
    $V self-test/test_stance_law.py        # the law it calls

    Started for real by:
    sudo HOME=$HOME chrt -f 50 $V torque_mode_control/torque_stand_hw.py

    Watch `tau_age` / the LIMP line on that runner's status output: if this
    thread stops publishing for WORKER_STALE_S the CAN loop limps the robot
    rather than streaming the last wrench at a world that has stopped
    updating.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import body_state_ahrs as bs                               # noqa: E402
import stance_law as law                                   # noqa: E402
import torque_params as P                                  # noqa: E402

N_JOINTS = P.N_JOINTS


class TorqueShared:
    """Mailbox between the CAN sweep and the stance worker.

    Written by the CAN thread:  q, qd, stage, planted, z_des, q_ref_cmd
    Written by the worker:      tau_ff, tau_stamp, active, reason, diag, state
    """

    def __init__(self, q0):
        q0 = np.asarray(q0, dtype=float)
        # -- from the CAN loop
        self.q = q0.copy()
        self.qd = np.zeros(N_JOINTS)
        self.stage = "INIT"
        self.planted = np.ones(4, dtype=bool)
        self.z_des = None                 # set once the crouch height is known
        self.q_ref_cmd = q0.copy()        # the pose the sequence wants held
        # -- from the worker
        self.tau_ff = np.zeros(N_JOINTS)
        self.tau_stamp = 0.0
        self.active = False
        self.reason = "starting"
        self.diag = {}
        self.state = None
        # -- control
        self.run = True
        self.force_frac = P.FORCE_FRAC_DEFAULT
        self.leg_gravity = P.STANCE_LEG_GRAVITY


def stance_worker(shared, ahrs, gains, cfg, mass, *,
                  control_hz=P.CONTROL_UPDATE_HZ, verbose=False):
    """Own the AHRS and the force law; publish tau_ff.  Runs off the CAN loop.

    On any refusal from `BodyState` -- stale AHRS, too few planted feet -- the
    feedforward drops to the leg-gravity term ALONE rather than freezing at
    its last value.  That is deliberate and it is the opposite of what the
    position track does:

        position mode  freeze the offset.  A stale estimate leaves the robot
                       standing where it is, because the drivers hold their
                       last target.
        torque mode    a frozen wrench keeps pushing on a world model that
                       has stopped updating.  Falling back to "each leg holds
                       its own weight" is the honest command for a controller
                       that has lost the trunk, and the joint impedance in
                       the sweep still holds the pose.
    """
    state = bs.BodyState(ahrs)
    shared.state = state
    period = 1.0 / control_hz
    next_t = time.perf_counter()

    while shared.run:
        now = time.perf_counter()
        if now < next_t:
            time.sleep(min(next_t - now, period))
            continue
        next_t = now + period

        q = shared.q                     # atomic reads of whole objects
        qd = shared.qd
        planted = shared.planted
        out, active, reason = state.read(now, q, qd, planted)

        if not active:
            shared.active = False
            shared.reason = reason
            # honest fallback: hold each leg up, command no body wrench
            shared.tau_ff = law.gravity_stack(q) if shared.leg_gravity \
                else np.zeros(N_JOINTS)
            shared.tau_stamp = time.monotonic()
            if verbose:
                print(f"[worker] inactive: {reason}")
            continue

        z_des = shared.z_des
        if z_des is None:
            z_des = float(out["r"][2])   # never demand a height we have not
                                         # been told to go to -- see the gate
                                         # in test_stance_law
        sched = {
            "contacts": planted,
            "z_des": z_des,
            "v_cmd_world": np.zeros(3),
            "yawrate_cmd": 0.0,
        }
        tau_ff, diag = law.compute_stance(
            q, qd, out, sched, gains, cfg, mass,
            leg_gravity=shared.leg_gravity, force_frac=shared.force_frac)

        shared.diag = diag
        shared.tau_ff = tau_ff
        shared.tau_stamp = time.monotonic()
        shared.active = True
        shared.reason = ""


def start(shared, ahrs, gains, cfg, mass, **kw):
    t = threading.Thread(target=stance_worker,
                         args=(shared, ahrs, gains, cfg, mass),
                         kwargs=kw, daemon=True)
    t.start()
    return t


def stale_s(shared):
    """Seconds since the worker last published, for the CAN loop's watchdog."""
    return time.monotonic() - shared.tau_stamp
