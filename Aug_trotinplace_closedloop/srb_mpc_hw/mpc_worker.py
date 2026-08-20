#!/usr/bin/env python3
"""The off-thread solver, and the mailbox it publishes through.

WHY IT IS A THREAD AND NOT PART OF THE SWEEP -- "cannot", not "should not"
    A CAN slot is 333 us, a sweep is 4 ms, and the driver's input-lost
    watchdog on CAN 1/7/8/9 fires at 10 ms.  The condensed QP is a 96-variable
    dense problem: one ~1 million multiply-add Hessian, one 96x96
    factorisation and 60 ADMM iterations.  On the development box, driven from
    this thread against a live model block, that is 3.1 ms mean and 6.8 ms
    worst against a 25 ms budget; on the Pi it will be several times that, and
    it is MEASURED IN THE RUN -- the census below, printed by the exit
    report -- rather than guessed here.

    Sub-sampling it into the sweep the way the stance law is sub-sampled does
    not help, because the cost lands in ONE sweep however rarely it runs, and
    that sweep's frames are late by the whole solve.  CONTROL_ROADMAP says the
    same thing as a standing constraint: "keep the EKF/MPC off-thread pattern
    and the CAN loop dumb".

    torque_mode_control/torque_worker.py is the pattern, publishing discipline
    included, and this file is deliberately the same shape.

WHAT IS ON EACH SIDE OF THE LINE
    the sweep       every CAN frame, the joint impedance, the gate, the
                    e-stops, and at MODEL_EVERY the estimator, the contact
                    measurement, the swing arcs and the J^T map.  ALL of the
                    velocity feedback that stabilises a joint, on telemetry at
                    most 4 ms old.
    this thread     the plan: 12 forces per knot over 8 knots.  Its first TWO
                    knots are published and the CAN loop interpolates between
                    them by wall clock, so the force it applies stays aligned
                    with the contact set even between solves.  It is a
                    feedforward either way.

    That split is the whole payoff of the corrected loop rate, spent on the one
    term that needed it.  A stale plan is not a stale controller.

PUBLISHING DISCIPLINE
    Every field is replaced by WHOLE-OBJECT ASSIGNMENT -- an atomic reference
    swap under the GIL -- and never mutated in place, so the CAN thread can
    never read a half-written vector.  Same contract as
    torque_worker.TorqueShared and ekf_runtime.EkfShared.

    The CONTACT SCHEDULE is published BY the CAN thread rather than read from
    the gait object, for the same reason: the gait latches early touchdowns
    from the sweep, and a worker walking that object mid-latch would plan
    against a contact set that never existed.

THE GIL, STATED HONESTLY
    numpy releases the GIL for the BLAS calls, which is where this thread's
    time goes; the per-leg Python around them does not.  The runner sets
    sys.setswitchinterval(0.0005) so a CAN slot never waits a full 5 ms switch
    interval on it.  This is the same trade the 100 Hz stance worker already
    makes, at a lower rate and with more of the time inside BLAS.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from . import mpc_config as C
from . import mpc_controller as ctl
from .convex_mpc import ConvexMPC


class MpcPlan:
    """One published plan, as a single object so it cannot be read torn.

    THE THREE FIELDS BELONG TOGETHER OR NONE OF THEM MEAN ANYTHING.  A caller
    interpolating the plan needs the two knots AND the time the first one
    starts at; publishing them as three attributes would let the CAN thread
    read a new force against an old timestamp -- a quarter of a gait cycle of
    phase error, arriving silently.  One atomic reference swap, one consistent
    plan.
    """

    __slots__ = ("f0", "f1", "t0", "dt", "stamp")

    def __init__(self, f0, f1, t0, dt, stamp):
        self.f0 = f0                  # knot 0 force, world axes (4,3)
        self.f1 = f1                  # knot 1, for the interpolation
        self.t0 = float(t0)           # the time knot 0 was planned FOR
        self.dt = float(dt)           # one knot
        self.stamp = float(stamp)     # time.perf_counter() at publish

    @staticmethod
    def zero(dt=C.MPC_DT):
        return MpcPlan(np.zeros((C.N_LEGS, 3)), np.zeros((C.N_LEGS, 3)),
                       0.0, dt, 0.0)


class MpcShared:
    """The mailbox between the CAN sweep and the solver.

    Written by the CAN thread:  q, state, weight_sched, z_des, v_cmd, wz_cmd,
                                f_applied, armed, replan, run
    Written by the worker:      plan, stamp, active, reason, info, stats
    """

    def __init__(self, q0):
        q0 = np.asarray(q0, dtype=float)
        # -- from the CAN loop
        self.q = q0.copy()
        self.state = None                 # the estimator's dict, or None
        self.weight_sched = np.ones((C.N_HORIZON, C.N_LEGS))
        self.z_des = None                 # floor -> TRUNK BOTTOM (m)
        self.v_cmd = np.zeros(3)
        self.wz_cmd = 0.0
        self.f_applied = np.zeros((C.N_LEGS, 3))
        self.t_sched = 0.0                # the time weight_sched knot 0 is for
        self.armed = False                # solve only inside a torque stage
        self.replan = False               # contact set changed: solve NOW
        self.run = True
        # -- from the worker
        self.plan = MpcPlan.zero()
        self.stamp = 0.0
        self.active = False
        self.reason = "starting"
        self.info = {}
        # -- solve-time census.  CONTROL_ROADMAP calls this the phase's main
        #    risk, so it is a measurement the exit report prints, not a hope.
        self.n_solves = 0
        self.t_min = float("inf")
        self.t_max = 0.0
        self.t_sum = 0.0
        self.iters_max = 0
        self.n_late = 0                   # solves that overran their period


def mpc_worker(shared, mpc=None, control_hz=C.MPC_HZ, verbose=False):
    """Own the MPC; publish its first two knots.  Runs off the CAN loop.

    On any refusal -- no estimator state, not armed -- it publishes ZERO force
    and active=False rather than holding the last plan.  That is deliberate and
    it is the opposite of what a position-mode track does:

        position mode  freeze the offset.  A stale estimate leaves the robot
                       standing, because the drivers hold their last target.
        torque mode    a frozen plan keeps pushing on a world model that has
                       stopped updating -- and this plan carries a CONTACT
                       SCHEDULE, so it keeps pushing with feet that may since
                       have left the ground.

    The runner's own staleness watchdog (mpc_config.MPC_STALE_S) is what turns
    that into a LIMP; this thread's job is only never to publish a fiction.
    """
    if mpc is None:
        mpc = ConvexMPC()
    period = 1.0 / float(control_hz)
    next_t = time.perf_counter()

    while shared.run:
        now = time.perf_counter()
        if now < next_t and not shared.replan:
            # short sleep, not a full period: a contact-set change has to be
            # able to interrupt the wait, because every latched touchdown
            # invalidates the constraint set the current plan was built on
            time.sleep(min(next_t - now, 0.001))
            continue
        shared.replan = False
        next_t = now + period

        state = shared.state                  # atomic reads of whole objects
        q = shared.q
        armed = shared.armed
        z_des = shared.z_des
        if state is None or not armed or z_des is None:
            shared.active = False
            shared.reason = ("not armed" if not armed else
                             "no estimator state" if state is None else
                             "no height reference")
            shared.plan = MpcPlan.zero()
            shared.stamp = time.perf_counter()
            continue

        t0 = time.perf_counter()
        t_sched = shared.t_sched
        C_ib = state["C"]
        kine = ctl.LegKinematics(q, C_ib)
        x0 = ctl.mpc_state(state, kine, C_ib)
        x_ref = ctl.mpc_reference(ctl.com_height_ref(z_des, kine, C_ib),
                                  shared.v_cmd, shared.wz_cmd)
        r_feet = ctl.foot_lever_arms(kine, C_ib)
        try:
            f0, f1, info = mpc.solve(x0, x_ref, shared.weight_sched, r_feet,
                                     yaw=0.0, f_applied=shared.f_applied)
        except np.linalg.LinAlgError as exc:
            # The Cholesky is the positive-definiteness CHECK; if it fails the
            # problem is not the one this solver was written for, and the
            # honest answer is no plan rather than an unchecked inverse.
            mpc.reset()
            shared.active = False
            shared.reason = f"QP factorisation failed: {exc}"
            shared.plan = MpcPlan.zero()
            shared.stamp = time.perf_counter()
            continue
        dt = time.perf_counter() - t0

        shared.plan = MpcPlan(f0, f1, t_sched, mpc.dt, time.perf_counter())
        shared.info = info
        shared.stamp = time.perf_counter()
        shared.active = True
        shared.reason = ""
        shared.n_solves += 1
        shared.t_sum += dt
        shared.t_min = min(shared.t_min, dt)
        shared.t_max = max(shared.t_max, dt)
        shared.iters_max = max(shared.iters_max, int(info["iters"]))
        if dt > period:
            shared.n_late += 1
        if verbose:
            print(f"[mpc] {dt*1e3:.2f} ms  {info['iters']} it  "
                  f"fz {info['fz_total']:.1f} N")


def start(shared, mpc=None, **kw):
    t = threading.Thread(target=mpc_worker, args=(shared, mpc),
                         kwargs=kw, daemon=True)
    t.start()
    return t


def stale_s(shared):
    """Seconds since the worker last published, for the CAN loop's watchdog.

    perf_counter and not monotonic, because the CAN loop's `now` is
    perf_counter and an age is only meaningful against one clock.
    """
    return time.perf_counter() - shared.stamp


def census(shared):
    """One line of solve-time truth for the exit report."""
    if shared.n_solves == 0:
        return "no MPC solves ran"
    return (f"{shared.n_solves} solves: "
            f"{shared.t_min*1e3:.2f} / {shared.t_sum/shared.n_solves*1e3:.2f} "
            f"/ {shared.t_max*1e3:.2f} ms (min/mean/max) at "
            f"{C.MPC_HZ:.0f} Hz = {1e3/C.MPC_HZ:.0f} ms budget, "
            f"{shared.n_late} over budget, worst {shared.iters_max} ADMM "
            f"iterations of {C.QP_ITERS}")
