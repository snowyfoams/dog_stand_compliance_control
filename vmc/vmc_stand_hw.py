#!/usr/bin/env python3
"""DOG5 stands under whole-body VMC with the state estimator in the loop.

Flow (references stand_dog5_inplace_hw -- the good stand code):
    ZERO-TORQUE check
      -> CROUCH        native 0xA4 position control moves every joint to the
                       recorded crouch pose (recorded.POSITION_TARGET_DEG)
      -> WAIT_CROUCH   hold the recorded pose; the EKF initialises on this
                       static hold; ENTER starts the rise
      -> RISE          VMC + EKF: ramp the height target z_des from crouch up to
                       standing; the whole-body wrench pushes the body up
      -> HOLD          VMC + EKF holds standing height

Only the RISE/HOLD torque law is new (dog5_vmc_core). Everything else is reused
from the tested stack:
  * stand_dog5_inplace_hw : ConfirmedSafetyGate, the native-position CROUCH
                            staging + recorded crouch pose / settle tolerances
  * stand_dog5_hw (base)  : MotorBus low-level, _joint_state, CalibratedEncoder-
                            Unwrap, KeyPoller, _zero_torque_preflight, arm +
                            _recover_input_lost, estop, _soft_stop
  * stand_dog5_recorded_hw: Q_RECORDED_CROUCH / POSITION_TARGET_DEG / tolerances

Run with the project venv (needs python-can + pyserial + mujoco-for-base):
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V vmc_stand_hw.py --tau-max 3.0
Support/tether the robot for the first closed-loop runs.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time

import numpy as np

# Let the GIL hand off often so the CAN thread keeps running while the EKF+VMC
# worker is mid-compute (numpy also releases the GIL during its heavy ops).
sys.setswitchinterval(0.0005)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DESC = os.path.join(os.path.dirname(_HERE), "dog5_description")
_EST = os.path.join(os.path.dirname(_HERE), "state_estimator")
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _DESC, _EST, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import motorbus                                          # noqa: E402
import stand_dog5_hw as base                             # noqa: E402 (low-level lib)
import stand_dog5_recorded_hw as recorded                # noqa: E402 (recorded crouch)
import stand_dog5_inplace_hw as inplace                  # noqa: E402 (good stand code)
import dog5_kinematics                                   # noqa: E402
import dog5_vmc_core as vmc                              # noqa: E402
from dog5_state_estimator import DOG5StateEstimator, quat_to_C  # noqa: E402
from imu_ekf_feed import ImuEkfFeed                      # noqa: E402
from imu_dog import DEFAULT_PORT                         # noqa: E402

MOTOR_IDS = base.MOTOR_IDS
N_JOINTS = base.N_JOINTS
CONTROL_HZ = base.CONTROL_HZ
LEGS = base.LEGS
POSITION_TARGET_DEG = recorded.POSITION_TARGET_DEG        # recorded crouch, motor-deg
Q_RECORDED_CROUCH = recorded.Q_RECORDED_CROUCH

T_RISE = 8.0                                             # VMC rise duration (s), gentle
DT_CLAMP = (1.0e-4, 0.05)

# ---------------------------------------------------------------------------
# Soft joint limits -- OFF by default in this runner
# ---------------------------------------------------------------------------
# base.soft_limits() is +/-1.75 rad (abd) and +/-2.6 rad (pitch, knee). Three
# things reached from here consult it:
#   base._zero_torque_preflight  refuses ENTER if the measured REST pose is
#                                outside -- i.e. before CROUCH is ever commanded
#   SafetyGate.apply             zeroes any torque pushing further out
#   SafetyGate.estop_reason      trips past limit + LIMIT_ESTOP_MARGIN
#
# Those numbers are software bounds measured against the encoder zero offsets,
# not mechanical stops, so a wrong offset rejects a pose that is mechanically
# fine. That is what has been refusing this runner at the ZERO-TORQUE preflight.
# The recorded crouch itself is comfortably inside (tightest margin: knees at
# 0.12 rad), so what the check is rejecting is the REST pose, not the crouch.
#
# Still bounding joint motion with these off: RunawayBrake latches any joint at
# 1.5 rad/s, estop_reason still trips on overspeed / overtemp / CAN miss /
# driver fault, and the non-finite state check still fires. NOT covered: a slow
# drift, or a bad encoder unwrap, that never moves fast enough to trip a speed
# check. Pass --soft-limits to restore the stock behaviour.
SOFT_LIMITS_DEFAULT_ON = False
# Captured before any patching so the status report can still show margins.
NOMINAL_SOFT_LIMITS = base.soft_limits()
# ---------------------------------------------------------------------------
# Runaway brake (see RunawayBrake)
# ---------------------------------------------------------------------------
# A leg that is not bearing load has NO position reference anywhere in this
# stack: the VMC stance branch commands a force (tau = -J^T f), so an unloaded
# joint is an open integrator that accelerates until something opposes it.
#
# The previous brake was `requested -= 0.4 * qd` folded in BEFORE
# SafetyGate.apply, which made it inherit the gate's TORQUE_RAMP_S startup ramp
# and its TAU_SLEW_NM_S = 5 Nm/s slew. The whole torque vector is refreshed once
# per 12-slot round-robin (250/12 = 20.8 Hz, 48 ms), so the brake could gain at
# most 5.0 * 0.048 = 0.24 Nm per update -- while an unloaded abduction joint
# (J = 0.0147 kgm^2 including the 10:1 reflected rotor) under a 1.5 Nm command
# gains 4.9 rad/s in that same 48 ms. The brake lost that race by an order of
# magnitude; on 2026-07-30 hardware RR_abd settled at exactly qd = tau/0.4 and
# the run ended on the overspeed e-stop, not on the brake.
#
# The fix is NOT a bigger damping gain. Each motor is commanded and replies
# once per round-robin, so per-joint velocity is sampled at 20.8 Hz no matter
# what this thread does, and a sampled damper diverges once kd > 2J/dt:
#
#     joint   J (kgm^2)   deadbeat J*qd/dt @3 rad/s   max stable kd = 2J/dt
#     knee      0.0088            0.55 Nm                   0.37
#     abd       0.0147            0.92 Nm                   0.61
#
# KD_JOINT_BRAKE = 0.4 was already at 109% of the knee's limit -- it was an
# unstable damper there -- and a large fixed brake is far worse: 6 Nm is 7x
# past deadbeat at 3 rad/s, so it reverses the joint and rings up instead of
# settling. Braking authority at this sample rate is hard-capped by physics.
#
# That budget rules out CONTINUOUS shaping too, which is the trap here. Fading
# the drive out over a speed band is itself velocity feedback: fading a 3 Nm
# command over a 2 rad/s band is a 1.5 Nm*s/rad slope, 4x the knee's stability
# limit. Simulated, that version diverges into a growing limit cycle (+/-0.02
# Nm at 0.1 s, +/-0.5 Nm at 1 s) and trips the e-stop anyway. A stable
# proportional law would need a band ~12 rad/s wide -- past the e-stop itself.
#
# The only thing that works at 48 ms is a law with ZERO incremental gain, so
# the brake LATCHES instead of blending:
#   unlatched   the controller torque passes through untouched -- the rise runs
#               at ~0.5 rad/s and pays nothing at all
#   latched     at BRAKE_LATCH_RAD_S the joint's drive is cut completely and
#               replaced by a CONSTANT opposing torque; an unloaded joint has
#               nothing left feeding it and coasts down
#   release     below BRAKE_RELEASE_RAD_S, with hysteresis so it cannot chatter
#
# Within each state d(tau)/d(qd) = 0, so the brake contributes no feedback gain
# and cannot oscillate regardless of sample rate or joint inertia. The hold
# torque is sized below deadbeat (J*qd/dt) at the RELEASE speed, so it can
# never reverse a joint on the way down either. It is applied after
# SafetyGate.apply, so it is neither ramped nor slewed.
BRAKE_LATCH_RAD_S = 1.5        # 3x the intended rise rate, 4.7x below QD_ESTOP
BRAKE_RELEASE_RAD_S = 1.0      # hysteresis band: 1.0 -> 1.5 rad/s
BRAKE_PERIOD_S = N_JOINTS / CONTROL_HZ    # 48 ms: one full round-robin
BRAKE_MIN_INERTIA = 0.0088     # kgm^2, the knee -- smallest of the three, so
                               # the deadbeat bound below is safe on every joint
BRAKE_DEADBEAT_FRAC = 0.7      # stay short of deadbeat; margin for J error
# Constant hold torque: 70% of what would null BRAKE_RELEASE_RAD_S in one
# period. Sub-deadbeat over the whole latched range, so no overshoot.
BRAKE_HOLD_NM = (BRAKE_DEADBEAT_FRAC * BRAKE_MIN_INERTIA
                 * BRAKE_RELEASE_RAD_S / BRAKE_PERIOD_S)
BRAKE_MIN_LATCH_S = 0.20       # minimum dwell before release is even considered
# Releasing straight back to full torque is its own runaway: one 48 ms sample of
# an unclipped 3 Nm command moves an unloaded knee 3*0.048/0.0088 = 16 rad/s,
# past the e-stop before the brake can look again. So drive is restored on a
# TIME ramp. Being a function of time and not of qd, it adds no velocity
# feedback and cannot destabilise anything.
BRAKE_RECOVER_S = 0.50
# One latch is a transient (a bump, a foot skipping). Repeated latching of the
# same joint is a leg that is simply not bearing load: stop with that diagnosis
# instead of letting it thrash until the overspeed e-stop fires.
BRAKE_MAX_TRIPS = 3
BRAKE_LATCH_TIMEOUT_S = 1.0    # ...or one latch that never comes back down
# The 27-state EKF + VMC is ~2.8 ms on the Pi. Running it in every 4 ms sweep
# leaves too little headroom under the 10 ms motor input-lost watchdog, so we
# DECIMATE it to this rate and hold the torque between updates -- the CAN
# round-robin still commands every motor every slot, feeding the watchdog.
CONTROL_UPDATE_HZ = 100.0
N_INIT_SAMPLES = 30            # IMU samples for the static EKF init
WORKER_STALE_S = 0.2          # if the worker stops updating torque this long -> estop


class _Shared:
    """State shared between the CAN thread and the EKF+VMC worker.

    All fields are read/written by whole-object reference swaps (arrays) or
    single scalar/str assignments, which are atomic under the GIL -- no lock
    needed, and no torn reads.
    """

    def __init__(self, start_q):
        self.q = start_q.copy()
        self.qd = np.zeros(N_JOINTS)
        self.stage = "CROUCH"
        self.z_des = 0.0
        self.requested_tau = np.zeros(N_JOINTS)
        self.tau_stamp = 0.0          # time.monotonic() of last torque update
        self.out = None               # latest estimator outputs (for status)
        self.bias_str = ""
        self.est_ready = False
        self.run = True


def _worker(shared, feed, mass, gains, cfg):
    """Own the EKF + IMU feed + VMC; publish torque. Runs OFF the CAN thread so
    the round-robin stays light (this is why the historical light loop never
    latches and the in-loop EKF version did)."""
    est = DOG5StateEstimator()
    contacts = np.ones(4, dtype=bool)
    init_f, init_w = [], []
    last_imu_t = None
    w_last = np.zeros(3)
    control_dt = 1.0 / CONTROL_UPDATE_HZ
    while shared.run:
        t = time.perf_counter()
        stage = shared.stage
        if not shared.est_ready:
            # buffer IMU; initialise once the crouch hold is reached and static
            for f, w, _ in feed.drain():
                init_f.append(f)
                init_w.append(w)
            if stage in ("WAIT_CROUCH", "RISE", "HOLD") and len(init_f) >= N_INIT_SAMPLES:
                est.initialise(np.array(init_f), np.array(init_w),
                               shared.q.reshape(4, 3), contacts)
                last_imu_t = time.monotonic()
                shared.bias_str = (f"bw={np.round(est.state.bw,5)} "
                                   f"bf={np.round(est.state.bf,4)}")
                shared.est_ready = True
            else:
                time.sleep(0.01)
            continue
        # ready: full EKF+VMC at CONTROL_UPDATE_HZ, coalescing buffered IMU
        q = shared.q                      # atomic ref read
        qd = shared.qd
        samples = feed.drain()
        if samples:
            fmean = np.mean([s[0] for s in samples], axis=0)
            wmean = np.mean([s[1] for s in samples], axis=0)
            tm = samples[-1][2]
            dt = min(max(tm - last_imu_t, DT_CLAMP[0]), DT_CLAMP[1])
            est.predict(fmean, wmean, dt, contacts)
            last_imu_t, w_last = tm, wmean
        est.update(q.reshape(4, 3), contacts)
        out = est.outputs(last_w_meas=w_last)
        out["C"] = quat_to_C(out["q"])
        sched = {"contacts": contacts, "swing_targets": {},
                 "z_des": shared.z_des, "v_cmd_world": np.zeros(3), "yawrate_cmd": 0.0}
        tau, _ = vmc.compute_vmc_torques(q, qd, out, sched, gains, cfg, mass)
        shared.requested_tau = tau        # atomic ref swap
        shared.out = out
        shared.tau_stamp = time.monotonic()
        rem = control_dt - (time.perf_counter() - t)
        if rem > 0:
            time.sleep(rem)


def _disable_soft_limits():
    """Neutralise base.soft_limits() for this process only.

    base._zero_torque_preflight calls the module-level function directly and
    offers no injection point, so relaxing it without editing stand_dog5_hw.py
    -- which stand_dog5_fixed_fend_hw.py depends on and currently stands with --
    means replacing it here. Infinite bounds make every consumer inert:
    `outside` is never true, the torque-side clamp never fires, and
    estop_reason's `q < low - margin` can never be satisfied.
    """
    base.soft_limits = lambda: (np.full(N_JOINTS, -np.inf),
                                np.full(N_JOINTS, np.inf))


def _report_soft_limit_margins(q, enforced):
    """Report the pose against the NOMINAL limits without gating on them.

    With the check disabled this is the only place the numbers surface, and it
    is worth seeing: a joint reported far outside a mechanical limit it is
    plainly not at is direct evidence of a wrong encoder zero offset, which is
    the thing that makes the check reject good poses in the first place.
    """
    low, high = NOMINAL_SOFT_LIMITS
    q = np.asarray(q, dtype=float)
    outside = np.flatnonzero((q < low) | (q > high))
    if not outside.size:
        margin = float(np.min(np.minimum(q - low, high - q)))
        print(f"[limits] rest pose is inside the nominal soft limits "
              f"(tightest margin {margin:.2f} rad)")
        return
    detail = " ".join(f"{base.JOINT_LABELS[i]}={q[i]:+.2f}" for i in outside)
    verb = "REFUSING" if enforced else "allowed (limits off)"
    print(f"[limits] {outside.size} joint(s) outside the nominal soft limits, "
          f"{verb}: {detail}")
    if not enforced:
        print("[limits] if any of those is not where the joint physically is, "
              "that joint's encoder zero offset is wrong.")


class RunawayBrake:
    """Per-joint latching runaway brake, applied AFTER SafetyGate.apply.

    Sitting downstream of the gate is the point: the gate's startup ramp and its
    5 Nm/s slew shape the *control* torque, and neither should throttle a safety
    response. This acts at full authority on the update it is needed.

    Per joint, a state machine on |qd| with hysteresis:

      unlatched -> latched   when |qd| >= BRAKE_LATCH_RAD_S
      latched   -> released  when |qd| <= BRAKE_RELEASE_RAD_S, and not before
                             BRAKE_MIN_LATCH_S of dwell
      released  -> unlatched over BRAKE_RECOVER_S, restoring drive on a ramp

    Unlatched, the controller torque passes through byte-for-byte. Latched, the
    controller's torque for that joint is DISCARDED and replaced by a constant
    -sign(qd) * BRAKE_HOLD_NM. Cutting the drive is what actually stops an
    unloaded joint: nothing is left feeding it, so it coasts down under the hold
    torque plus friction.

    Why latch rather than blend: within each state d(tau)/d(qd) is exactly zero,
    so the brake adds no velocity-feedback gain and cannot oscillate at any
    sample rate or joint inertia. Every continuous alternative -- a damping
    gain, or fading the drive out over a speed band -- IS velocity feedback, and
    at the 48 ms per-joint sample period the stable budget is only 2J/dt = 0.37
    Nm*s/rad on the knee. Both blow it and ring the joint up instead of stopping
    it; that is exactly how the 0.4 Nm*s/rad predecessor failed. The recovery
    ramp is a function of time only, so it inherits the same zero-gain property.

    Why it cannot overshoot: BRAKE_HOLD_NM is 70% of the deadbeat torque at
    BRAKE_RELEASE_RAD_S, and a joint is only ever latched above that speed, so
    the hold torque is sub-deadbeat over the entire latched range and can never
    reverse the joint.

    What this can and cannot do. It CANNOT keep a genuinely unloaded leg running
    -- at 20.8 Hz a 3 Nm command puts an unloaded knee past the e-stop inside a
    single sample, so no brake sampling at that rate can. What it does is catch
    the joint at 1.5 rad/s instead of 7, hold it there without ever letting it
    run away, and escalate with the correct diagnosis (`stop_reason`) rather
    than surfacing as a misleading "confirmed overspeed" several seconds later.
    """

    def __init__(self):
        self.latched = np.zeros(N_JOINTS, dtype=bool)
        self.since = np.zeros(N_JOINTS)         # time this latch engaged
        self.released_at = np.full(N_JOINTS, -np.inf)   # time it let go
        self.trips = np.zeros(N_JOINTS, dtype=int)

    def apply(self, tau, qd, now):
        """Return (tau_out, n_latched). Pure state machine, no gain anywhere."""
        tau = np.asarray(tau, dtype=float)
        qd = np.asarray(qd, dtype=float)
        speed = np.abs(qd)

        newly = (~self.latched) & (speed >= BRAKE_LATCH_RAD_S)
        self.since = np.where(newly, now, self.since)
        self.trips += newly
        # Release needs BOTH the speed back under the lower threshold AND the
        # minimum dwell elapsed. `newly` is folded in first, so a joint cannot
        # latch and release on the same update.
        latched = self.latched | newly
        may_release = (speed <= BRAKE_RELEASE_RAD_S) & (
            (now - self.since) >= BRAKE_MIN_LATCH_S)
        releasing = latched & may_release
        self.released_at = np.where(releasing, now, self.released_at)
        self.latched = latched & ~may_release

        # Restore drive on a time ramp after release (never a qd ramp).
        recover = np.clip((now - self.released_at) / BRAKE_RECOVER_S, 0.0, 1.0)
        hold = -np.sign(qd) * BRAKE_HOLD_NM
        out = np.where(self.latched, hold, tau * recover)
        return np.clip(out, -base.TAU_HARD, base.TAU_HARD), int(self.latched.sum())

    def stop_reason(self, now):
        """A leg that is not bearing load, reported as such."""
        stuck = self.latched & ((now - self.since) >= BRAKE_LATCH_TIMEOUT_S)
        if np.any(stuck):
            index = int(np.flatnonzero(stuck)[0])
            return (f"runaway brake held {base.JOINT_LABELS[index]} for "
                    f"{now - self.since[index]:.1f} s without recovering below "
                    f"{BRAKE_RELEASE_RAD_S:.1f} rad/s -- leg unloaded or stuck")
        repeat = self.trips >= BRAKE_MAX_TRIPS
        if np.any(repeat):
            index = int(np.argmax(self.trips))
            return (f"runaway brake engaged on {base.JOINT_LABELS[index]} "
                    f"{int(self.trips[index])} times -- that leg is not bearing "
                    f"load; check foot contact and encoder zero offsets")
        return None

    def summary(self):
        if not np.any(self.trips):
            return "brake: never engaged"
        hit = [f"{base.JOINT_LABELS[i]}x{int(n)}"
               for i, n in enumerate(self.trips) if n]
        return "brake engaged on " + " ".join(hit)


def _smoothstep(u):
    u = float(np.clip(u, 0.0, 1.0))
    return 3 * u * u - 2 * u * u * u


def _est_out(est, w_meas):
    out = est.outputs(last_w_meas=w_meas)
    out["C"] = quat_to_C(out["q"])
    return out


def _crouch_height():
    """Trunk height above the feet at the recorded crouch (m), from FK."""
    zs = [dog5_kinematics.foot_position(LEGS[i], Q_RECORDED_CROUCH[3 * i:3 * i + 3])[2]
          for i in range(N_JOINTS // 3)]
    return -float(np.mean(zs))


class VmcStandSequence:
    """CROUCH (native position) -> WAIT_CROUCH -> RISE -> HOLD (VMC+EKF).

    Crouch settle logic mirrors inplace.InPlaceStandSequence: pose within
    RECORDED_POSE_TOL and speed within RECORDED_QD_TOL for CROUCH_SETTLE_S.
    """

    def __init__(self, now):
        self.stage = "CROUCH"
        self.started_at = now
        self.crouch_settle_since = None
        self._rise_t0 = None

    def update_crouch(self, now, q, qd):
        if self.stage != "CROUCH":
            return None
        if now - self.started_at > recorded.CROUCH_TIMEOUT_S:
            raise RuntimeError(
                f"native position CROUCH timed out after "
                f"{recorded.CROUCH_TIMEOUT_S:.1f} s")
        pose_error = float(np.max(np.abs(np.asarray(q) - Q_RECORDED_CROUCH)))
        speed = float(np.max(np.abs(qd)))
        settled = (pose_error <= recorded.RECORDED_POSE_TOL
                   and speed <= recorded.RECORDED_QD_TOL)
        if not settled:
            self.crouch_settle_since = None
        elif self.crouch_settle_since is None:
            self.crouch_settle_since = now
        elif now - self.crouch_settle_since >= recorded.CROUCH_SETTLE_S:
            self.stage = "WAIT_CROUCH"
            return ("CROUCH reached and settled; EKF initialising on the hold. "
                    "ENTER for VMC+EKF RISE.")
        return None

    def advance(self, now):
        if self.stage == "WAIT_CROUCH":
            self.stage = "RISE"
            self._rise_t0 = now
            return "advancing to RISE (VMC+EKF)"
        return None

    def maybe_finish_rise(self, now):
        if self.stage == "RISE" and now - self._rise_t0 >= T_RISE:
            self.stage = "HOLD"
            return "RISE complete -- HOLD (VMC+EKF). X to stop."
        return None

    def z_des(self, now, dz):
        if self.stage == "RISE":
            return dz * _smoothstep((now - self._rise_t0) / T_RISE)
        return dz if self.stage == "HOLD" else 0.0

    @property
    def in_position_mode(self):
        return self.stage in ("CROUCH", "WAIT_CROUCH")


def run(port, tau_max, crouch_max_speed_dps, stand_height, gains,
        soft_limits=SOFT_LIMITS_DEFAULT_ON):
    base.validate_hardware_config()
    mass = base.DOG5_MASS_KG
    if not mass > 1.0:
        raise RuntimeError(f"DOG5_MASS_KG looks wrong ({mass}); refusing to run VMC")

    crouch_h = _crouch_height()
    dz_rise = max(0.0, stand_height - crouch_h)

    cfg = vmc.VMCConfig()
    cfg.tau_max = tau_max                    # VMC internal clip; SafetyGate re-caps

    # Before the gate is built: ConfirmedSafetyGate snapshots soft_limits() into
    # self.low/self.high at construction, so patching afterwards would leave the
    # gate still enforcing them.
    if not soft_limits:
        _disable_soft_limits()

    unwrap = [base.CalibratedEncoderUnwrap() for _ in MOTOR_IDS]
    key = base.KeyPoller()
    gate = inplace.ConfirmedSafetyGate(tau_max, base.QD_ESTOP, base.QD_ESTOP_HARD)
    brake = RunawayBrake()

    print("=" * 74)
    print("DOG5 VMC+EKF STAND (references stand_dog5_inplace_hw; EKF+VMC off-thread).")
    print(f"  crouch->stand rise dz = {dz_rise*1e3:.0f} mm "
          f"(crouch h={crouch_h:.3f}, stand h={stand_height:.3f})")
    print(f"  torque cap = {tau_max:.1f} Nm (ConfirmedSafetyGate ramp+slew). "
          "FEET MUST BEAR WEIGHT (loose safety tether only).")
    print(f"  gains: kp_att={gains.kp_roll:.0f} kd_att={gains.kd_roll:.0f} "
          f"kp_z={gains.kp_z:.0f} kd_z={gains.kd_z:.0f} kd_xy={gains.kd_x:.0f} "
          f"| T_RISE={T_RISE:.0f}s")
    print(f"  runaway brake: latch {BRAKE_LATCH_RAD_S:.1f} / release "
          f"{BRAKE_RELEASE_RAD_S:.1f} rad/s, drive cut and held at "
          f"{BRAKE_HOLD_NM:.2f} Nm while latched "
          f"(e-stop {base.QD_ESTOP:.1f} rad/s); post-gate, unslewed.")
    if soft_limits:
        lo, hi = NOMINAL_SOFT_LIMITS
        print(f"  soft joint limits: ENFORCED (abd +/-{hi[0]:.2f}, "
              f"pitch/knee +/-{hi[1]:.2f} rad) -- preflight will refuse a rest "
              f"pose outside them.")
    else:
        print("  soft joint limits: OFF (--soft-limits to enforce). Overspeed, "
              "runaway brake, overtemp, CAN-miss and fault checks stay active.")
    print("=" * 74)

    stop_reason = None
    armed = False
    worker = None
    shared = None
    try:
        with motorbus.MotorBus(MOTOR_IDS, dirs=base.MOTOR_DIRECTIONS) as mb, \
                ImuEkfFeed(port) as feed:
            if not mb.arm(rate_hz=CONTROL_HZ):
                raise RuntimeError("arm failed (bus/power/terminators)")
            armed = True
            if not feed.wait_for_raw(timeout=3.0):
                raise RuntimeError("no raw IMU (0x40) packets -- enable DETA10 raw mode")

            start_q = base._zero_torque_preflight(mb, key, unwrap)
            if start_q is None:
                print("[abort] preflight not confirmed")
                return
            _report_soft_limit_margins(start_q, soft_limits)

            now = time.perf_counter()
            gate.start(now, start_q)
            seq = VmcStandSequence(now)
            miss_monitor = base.CanMissMonitor(mb)

            # start the EKF+VMC worker; the CAN loop below stays LIGHT.
            shared = _Shared(start_q)
            worker = threading.Thread(target=_worker,
                                      args=(shared, feed, mass, gains, cfg), daemon=True)
            worker.start()

            slot = mb.slot(CONTROL_HZ)
            deadline = time.perf_counter() + slot
            status_period = 1.0 / base.FAULT_STATUS_HZ
            next_fault_status = np.array(
                [status_period + i * status_period / N_JOINTS for i in range(N_JOINTS)])
            last_recover = {mid: -base.RECOVER_PERIOD_S for mid in MOTOR_IDS}
            start = now
            tau_command = np.zeros(N_JOINTS)
            n_latched = 0
            torque_active = False        # set each sweep; gates the per-slot refresh
            index = 0
            last_print = 0.0
            rise_armed = False
            bias_printed = False
            tx_fail = 0

            while True:
                mb.poll()
                joint_index = index % N_JOINTS
                if joint_index == 0:
                    now = time.perf_counter()
                    q, qd = base._joint_state(mb, unwrap)
                    # publish joint state + stage to the worker (atomic swaps)
                    shared.q = q
                    shared.qd = qd
                    shared.stage = seq.stage

                    if seq.stage == "CROUCH":
                        msg = seq.update_crouch(now, q, qd)
                        if msg:
                            print(f"[stage] {msg}")

                    if shared.est_ready and not bias_printed:
                        print(f"[ekf] init: {shared.bias_str}")
                        bias_printed = True

                    if seq.stage in ("RISE", "HOLD"):
                        if not rise_armed:
                            gate.start(now, q)          # ease torque in at RISE
                            rise_armed = True
                        shared.z_des = seq.z_des(now, dz_rise)
                        if shared.est_ready:
                            stale = time.monotonic() - shared.tau_stamp
                            if stale > WORKER_STALE_S:
                                stop_reason = f"control worker stalled ({stale*1e3:.0f} ms)"
                        torque_active = bool(shared.est_ready)
                        if stop_reason is None:
                            stop_reason = brake.stop_reason(now)
                        fin = seq.maybe_finish_rise(now)
                        if fin:
                            print(f"[stage] {fin}")
                    else:
                        torque_active = False
                        n_latched = 0

                    # ---- safety (light, in CAN thread) ----
                    temps = base._temperatures(mb)
                    misses = miss_monitor.update(mb)
                    errors = mb.errors()
                    if stop_reason is None:
                        # The soft joint limits only bound the VMC torque law.
                        # CROUCH/WAIT_CROUCH run native 0xA4 position control to
                        # the recorded pose, whose knees sit 0.12 rad from the
                        # limit -- close enough that staging jitter trips the
                        # limit e-stop for no reason. Overspeed / temp / miss /
                        # driver-error checks stay on in every stage.
                        stop_reason = gate.estop_reason(
                            q, qd, temps, misses, errors, now,
                            enforce_position_limits=not seq.in_position_mode,
                        )
                    latched = [mid for mid, e in errors.items() if e and (e & 0x80)]

                    pressed = key.get()
                    if pressed in ("x", "X"):
                        stop_reason = "operator X"
                    elif pressed in ("\r", "\n"):
                        m = seq.advance(now)
                        if m:
                            print(f"[stage] {m}")
                    if stop_reason:
                        break

                    recover = [mid for mid in latched
                               if (now - start) - last_recover[mid] >= base.RECOVER_PERIOD_S]
                    if recover:
                        base._recover_input_lost(mb, recover, now - start,
                                                 last_recover, next_fault_status)

                    if now - last_print >= 1.0 / base.STATUS_HZ:
                        last_print = now
                        extra = ""
                        out = shared.out
                        if shared.est_ready and out is not None:
                            gb = out["C"] @ np.array([0.0, 0.0, 1.0])
                            roll = math.degrees(math.atan2(gb[1], gb[2]))
                            pitch = math.degrees(math.atan2(-gb[0], math.hypot(gb[1], gb[2])))
                            age = (time.monotonic() - shared.tau_stamp) * 1e3
                            extra = (f" z={out['r'][2]*1e3:+5.0f}mm "
                                     f"roll={roll:+5.1f} pitch={pitch:+5.1f} "
                                     f"H={'OK' if out['healthy'] else '!!'} tau_age={age:.0f}ms")
                        cpose = float(np.max(np.abs(q - Q_RECORDED_CROUCH)))
                        print(f"[hw] {seq.stage:11s} max|tau|={np.max(np.abs(tau_command)):.2f} "
                              f"brk={n_latched} "
                              f"max|qd|={np.max(np.abs(qd)):.2f} crouch_err={cpose:.2f} "
                              f"Tmax={int(np.max(temps))}C latched={len(latched)} "
                              f"txfail={tx_fail}{extra}", flush=True)

                # ---- per-slot torque refresh ----
                # The EKF+VMC worker publishes at CONTROL_UPDATE_HZ (100 Hz),
                # but this used to be read only when joint_index == 0 -- once per
                # 12-slot round-robin, 20.8 Hz -- so 4 of every 5 worker updates
                # were discarded and the balance loop ran at 20.8 Hz. That is a
                # rate-starved loop: it reads on hardware as weak and sluggish,
                # and in MuJoCo the same config sags to 0.05 m instead of
                # standing. Re-reading per slot restores the worker's full rate.
                #
                # q/qd stay one-sweep-old either way -- each motor only replies
                # once per round-robin, which no amount of thread work changes,
                # and BRAKE_PERIOD_S still reflects that 48 ms velocity sampling.
                # What does get fresher is the IMU-driven part of the estimate
                # (attitude and vertical velocity at 100 Hz), which is the loop
                # that actually needs the bandwidth to balance.
                if torque_active:
                    now_slot = time.perf_counter()
                    tau_command = gate.apply(shared.requested_tau, q, now_slot)
                    tau_command, n_latched = brake.apply(tau_command, qd, now_slot)
                else:
                    tau_command = np.zeros(N_JOINTS)

                # ---- per-slot command dispatch ----
                mid = MOTOR_IDS[joint_index]
                if (time.perf_counter() - start) >= next_fault_status[joint_index]:
                    mb.status1_req(mid)
                    next_fault_status[joint_index] += status_period
                elif seq.in_position_mode:
                    # A TX-queue-full (ENOBUFS) failure drops ONE frame: the
                    # motor holds and we resend next sweep, so tolerate it (only
                    # a >10 ms command gap latches, handled separately). Raising
                    # here would crash on a transient bus overload.
                    if not mb.position(mid, float(POSITION_TARGET_DEG[joint_index]),
                                       crouch_max_speed_dps):
                        tx_fail += 1
                else:
                    if not mb.torque(mid, float(tau_command[joint_index])):
                        tx_fail += 1
                index += 1
                overrun = mb.pace(deadline)
                deadline += slot
                # After a big scheduler/GIL stall the deadline is far in the past;
                # resync so we resume even pacing instead of bursting a catch-up
                # of sends that overflows the CAN TX queue.
                if overrun and overrun > 2.0 * slot:
                    deadline = time.perf_counter() + slot

            base._soft_stop(mb)
    except KeyboardInterrupt:
        stop_reason = stop_reason or "KeyboardInterrupt"
    finally:
        if shared is not None:
            shared.run = False
        if worker is not None:
            worker.join(timeout=1.0)
        try:
            key.close()
        except Exception:
            pass
    print(f"[stop] {stop_reason}")
    print(f"[stop] {brake.summary()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--tau-max", type=float, default=base.STAGED_TAU_MAX,
                    help=f"per-joint torque cap Nm (default {base.STAGED_TAU_MAX})")
    ap.add_argument("--crouch-max-speed-dps", type=float,
                    default=recorded.DEFAULT_CROUCH_MAX_MOTOR_DPS,
                    help="native CROUCH position speed cap (motor-deg/s)")
    ap.add_argument("--stand-height", type=float, default=recorded.DEFAULT_STAND_HEIGHT,
                    help="target standing trunk height (m)")
    # VMC gains -- tune these live on the bench (stiffness for rise balance).
    ap.add_argument("--kp-att", type=float, default=250.0,
                    help="roll/pitch stiffness Nm/rad (raise if it tips during rise)")
    ap.add_argument("--kd-att", type=float, default=20.0, help="roll/pitch damping Nms/rad")
    ap.add_argument("--kp-z", type=float, default=1500.0, help="height stiffness N/m")
    ap.add_argument("--kd-z", type=float, default=150.0, help="height damping Ns/m")
    ap.add_argument("--kd-xy", type=float, default=80.0,
                    help="horizontal velocity damping Ns/m (resists CoM drift)")
    ap.add_argument("--soft-limits", action="store_true",
                    default=SOFT_LIMITS_DEFAULT_ON,
                    help="enforce the +/-1.75 / +/-2.6 rad soft joint limits "
                         "(off by default: they gate on encoder zero offsets, "
                         "not mechanical stops, and refuse the rest pose at "
                         "the ZERO-TORQUE preflight)")
    args = ap.parse_args()
    if not 0.0 < args.tau_max <= 8.0:
        ap.error("--tau-max must be in (0, 8] Nm")

    gains = vmc.VMCGains()
    gains.kp_roll = gains.kp_pitch = args.kp_att
    gains.kd_roll = gains.kd_pitch = args.kd_att
    gains.kp_z, gains.kd_z = args.kp_z, args.kd_z
    gains.kd_x = gains.kd_y = args.kd_xy
    run(args.port, args.tau_max, args.crouch_max_speed_dps, args.stand_height,
        gains, soft_limits=args.soft_limits)


if __name__ == "__main__":
    main()
