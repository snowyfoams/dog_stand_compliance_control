#!/usr/bin/env python3
"""The Bloesch EKF, offered to the MPC loop as a second opinion on velocity.

WHAT THIS IS, AND WHAT IT IS NOT
    mpc_trot_hw's feedback is august_week2/feedback_estimator: AHRS attitude,
    FK height, and the algebraic leg-odometry velocity read off planted feet.
    Two of those three are already drift-free and floor-referenced, and this
    file does NOT touch them.  What the leg odometry cannot do is stated at
    its own boundary: a slipping foot reads as real trunk velocity, and below
    MIN_PLANTED there is no velocity at all -- which is the line a gait with a
    flight phase crosses every cycle.

    The state_estimator EKF (Bloesch et al., RSS 2013; 27 error states) fuses
    the raw IMU with the same encoders and carries velocity THROUGH contact
    changes instead of re-deriving it from them.  This file runs that EKF
    beside the loop and hands back one thing: the EKF's trunk velocity, in the
    frame the loop already works in.  Attitude stays the AHRS's and height
    stays FK's, because replacing a measured, falsifiable signal with a
    filtered one needs a reason, and velocity is the only signal here that has
    one.

NOTHING IN THE FILTER STACK IS NEW
    EkfShared and ekf_worker are state_estimator/ekf_runtime's, unmodified --
    the transplant of the pattern the gate-C8 stand and the D1 trot ran on
    hardware.  ImuEkfFeed wraps the SAME ImuDog the runner already owns, so
    the AHRS keeps flowing to feedback_estimator while the raw 0x40 stream
    feeds the filter; one serial port, two readers, no contention.  This file
    adds only the mailbox plumbing and the frame change, which is why it is
    short.

THE FRAME CHANGE, EXACTLY
    The EKF's v is in its inertial frame I, whose heading is wherever the
    robot pointed at initialisation, and drifts with the integrated gyro.
    The loop's world is YAW-FREE by construction (feedback_estimator builds C
    from roll/pitch alone; the MPC re-anchors yaw to zero every solve).  The
    map between them that involves no Euler angles and no small-angle claim:

        v_loop = C_loop^T (C_ekf v_I)

    C_ekf v_I is the trunk velocity in the BODY frame -- exact, whatever the
    yaw error -- and C_loop^T re-expresses body coordinates in the loop's
    world.  Roll/pitch agree between the two frames to the filters' own
    accuracy, so the only thing this composition discards is the EKF's
    unobservable yaw, which is the thing that had to go.

THE TWO GATES, AND WHO CHECKS THEM
    healthy   the estimator's own M11 verdict (sigma_v < 0.15, finite, P PSD).
    fresh     the worker published within EKF_STALE_S.  Same argument as
              MPC_STALE_S one layer up: in torque mode a stale estimate drives
              real force, so age is a fault, not a detail.
    sample() reports both and invents nothing; the CALLER falls back to the
    leg odometry, which is on the same tick anyway.  There is no state in
    which this file's output replaces a measurement with a guess.

RUN
    A library for mpc_trot_hw --ekf; opens no bus, commands nothing.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUG = os.path.dirname(_HERE)
_SE = os.path.join(os.path.dirname(_AUG), "state_estimator")
for _p in (_AUG, _SE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ekf_runtime import EkfShared, ekf_worker, _rp          # noqa: E402

# stand_params.EKF_WORKER_HZ, the rate every hardware EKF run has used.
EKF_WORKER_HZ = 100.0

# The one stage this runner is genuinely still in: position-held after the
# crouch, waiting for ENTER.  The stand runners' lesson (ekf_runtime header):
# init samples collected during MOTION pollute the gyro-bias estimate, so the
# worker refuses to initialise anywhere else.  ~30 raw samples at the DETA10's
# rate is well under a second of WAIT.
EKF_QUIET_STAGES = ("WAIT",)

# Five worker periods.  The worker runs at 100 Hz and the model block reads at
# 83 Hz, so one missed publish is invisible and five is a stall, not jitter.
EKF_STALE_S = 0.05


class EkfFeedback:
    """The EKF worker, its mailbox, and the two answers the loop wants.

    Single-writer discipline, same as MpcShared: the CAN thread calls
    publish() and reads sample(); the worker thread owns everything else.
    """

    def __init__(self, feed, start_q):
        self.shared = EkfShared(np.asarray(start_q, dtype=float))
        self.shared.stage = "WAIT"
        self._announced = False
        self._worker = threading.Thread(
            target=ekf_worker, args=(self.shared, feed),
            kwargs=dict(quiet_stages=EKF_QUIET_STAGES,
                        control_hz=EKF_WORKER_HZ),
            daemon=True)
        self._worker.start()

    # -- CAN-thread side --------------------------------------------------
    def publish(self, q, qd, stage, contacts):
        """Every sweep: what the filter is allowed to know.  Whole-object
        rebinds, never in-place -- the worker must not see a half-write."""
        sh = self.shared
        sh.q = np.asarray(q, dtype=float).copy()
        sh.qd = np.asarray(qd, dtype=float).copy()
        sh.stage = stage
        sh.contacts = np.asarray(contacts, dtype=bool).copy()

    def sample(self):
        """Latest output plus the two gates; None until initialised.

        Announces the initialisation ONCE (the bias estimate is the number an
        operator can sanity-check against hw_ekf's rest runs)."""
        sh = self.shared
        out = sh.out
        if not sh.est_ready or out is None:
            return None
        if not self._announced:
            self._announced = True
            print(f"[mpc] EKF initialised: {sh.bias_str}")
        roll, pitch = _rp(out["C"])
        return {
            "v": np.asarray(out["v"], dtype=float),
            "C": np.asarray(out["C"], dtype=float),
            "roll": roll,
            "pitch": pitch,
            "healthy": bool(out["healthy"]),
            "fresh": (time.monotonic() - sh.tau_stamp) < EKF_STALE_S,
        }

    @staticmethod
    def v_loop(sample, C_loop):
        """The EKF velocity in the loop's yaw-free world -- see the header."""
        return C_loop.T @ (sample["C"] @ sample["v"])

    def stop(self):
        self.shared.run = False
        self._worker.join(timeout=1.0)
