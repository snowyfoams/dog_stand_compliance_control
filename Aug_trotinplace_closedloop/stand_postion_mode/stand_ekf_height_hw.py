#!/usr/bin/env python3
"""Stage 2: CLOSED-LOOP standing height on EKF feedback, position-mode motors.

Stage 1 (`stand_ekf_verify_hw.py`) proved the EKF read-only.  This script puts
it in the loop: the EKF's height drives a common foot-z offset so the trunk
holds a COMMANDED height instead of whatever height the legs happen to sag to.

Still NO software torque anywhere.  The drivers' native 0xA4 position loops do
all the joint-level work; the EKF only ever moves the *targets* we hand them.

    high level (per 250 Hz sweep)
        EKF height  ->  height error  ->  integral controller  ->  common
        foot-z offset (clamped + slew-limited)
    mid level (per sweep, ~free)
        per-leg precomputed z -> joint-angle TABLE, linearly interpolated.
        No IK in the CAN sweep: stand_hier_hw measured 4 legs of convergent IK
        at 5-7 ms, far past the 333 us slot.  x/y are pinned, so one table per
        leg is a complete description of the poses this controller can ask for.
    low level
        0xA4 native position loops, motor-side speed caps

HEIGHT REFERENCE -- every height this script targets, measures or prints is
the floor-to-TRUNK-BOTTOM distance, i.e. to the IMU board.  That is the point
the EKF's `r` actually tracks and the only trunk point a ruler reaches.  The
kinematics are natural at the abd-axis plane (the FK trunk origin, dog5.xml
puts the four hip bodies at z = 0), IMU_BELOW_TRUNK_ORIGIN_M higher; the leg
tables work there and the status line shows it as `abd=` for cross-checking,
but no target or error is ever expressed in it.

Why an INTEGRATOR and nothing else: the plant is a position servo, so the gap
between commanded and achieved height is a near-constant sag under load (stage
1 measured 13 mm).  An integrator drives that to zero without needing to know
it, and unlike a proportional term it adds no gain to EKF noise.

Flow (all POSITION mode, operator-gated):
    ZERO-TORQUE -> CROUCH -> WAIT_CROUCH (EKF inits) -> STAND (open-loop ramp)
      -> HOLD4    closed loop ENGAGES here, holding --target-height
      -> PARK     (P) ramps from the CURRENT held pose -- offset included --
                  back to the crouch, so PARKED always lands on the recorded
                  crouch no matter what the loop had wound in
      -> PARKED   ENTER stands again.  X stops.

Contacts are held ON throughout, per the stage-1 hardware result: the feet
never leave the floor here, and dropping them cost the whole height estimate.

SAFETY -- the loop is closed on the EKF, so FK is the independent watchdog.
Every sweep compares the EKF height against the drift-free FK height; on
disagreement beyond --xcheck the integrator FREEZES (it never unwinds, so the
robot holds its pose rather than diving).  Same freeze on a stale or unhealthy
EKF.  The offset is clamped and slew-limited, and the z tables are built inside
the soft joint limits, so no reachable command can exceed them.

Run:
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V stand_ekf_height_hw.py --self-test              # no hardware needed
    sudo chrt -f 50 $V stand_ekf_height_hw.py --log h.csv --raw-log h.npz
    # open-loop A/B (should reproduce stage 1 exactly):
    sudo chrt -f 50 $V stand_ekf_height_hw.py --height-gain 0
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import threading
import time

import numpy as np

sys.setswitchinterval(0.0005)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# stage 1 owns the path bootstrap, the FK height reference and the stage report
import stand_ekf_verify_hw as s1                        # noqa: E402

import motorbus                                          # noqa: E402
import stand_dog5_hw as base                             # noqa: E402
import stand_dog5_recorded_hw as recorded                # noqa: E402
import dog5_kinematics                                   # noqa: E402
from ekf_runtime import EkfShared, ekf_worker, _rp       # noqa: E402
from imu_dog import DEFAULT_PORT                         # noqa: E402

# Every tunable this script has is in stand_params.py -- the HEIGHT_* block
# there is this loop's, and the slew/clamp/deadband are the ones to reach for.
from stand_params import (FOOT_RADIUS_M,                 # noqa: E402
                          IMU_BELOW_TRUNK_ORIGIN_M,
                          T_STAND, EKF_WORKER_HZ, QUIET_STAGES, SETTLE_S,
                          HEIGHT_GAIN_PER_S, HEIGHT_CLAMP_M, HEIGHT_SLEW_M_S,
                          HEIGHT_DEADBAND_M, HEIGHT_STALE_S, HEIGHT_XCHECK_M,
                          TABLE_STEP_M, TABLE_MARGIN_M)

MOTOR_IDS = base.MOTOR_IDS
N_JOINTS = base.N_JOINTS
CONTROL_HZ = base.CONTROL_HZ
LEGS = base.LEGS
Q_CROUCH = recorded.Q_RECORDED_CROUCH
POSITION_TARGET_DEG = recorded.POSITION_TARGET_DEG

# stage 1 owns the FK height reference and the ramp shape (see the function
# map in stand_params' docstring); these are aliases, not second copies
fk_floor_height = s1.fk_floor_height
_smoothstep = s1._smoothstep


# ===========================================================================
# Per-leg height tables -- IK out of the control loop
# ===========================================================================

def _ik_leg(leg, q_seed, target, iters=60):
    """Damped least-squares IK for one leg, warm-started from `q_seed`."""
    q = np.asarray(q_seed, dtype=float).copy()
    for _ in range(iters):
        e = target - dog5_kinematics.foot_position(leg, q)
        if np.linalg.norm(e) < 1e-11:
            break
        J = dog5_kinematics.foot_jacobian(leg, q)
        q = q + 0.5 * J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e)
    return q, float(np.linalg.norm(target - dog5_kinematics.foot_position(leg, q)))


class LegHeightTable:
    """Joint angles for one leg as a function of commanded foot z.

    x/y stay pinned at the crouch FK values (the stage-1 stand geometry), so
    foot z is the only free variable and a 1-D table is exact.  Built by
    marching outward from the crouch pose with warm-started incremental IK --
    the same technique `compute_q_stand` uses to stay on the crouch branch
    instead of flipping the knee.

    Runtime cost is a linear interpolation, which is what keeps IK out of the
    250 Hz sweep entirely.
    """

    def __init__(self, leg, q_ref, z_lo, z_hi, step=TABLE_STEP_M,
                 xy_offset=(0.0, 0.0)):
        """`xy_offset` moves this leg's pinned foot x/y in the TRUNK frame.

        Zero for every stage before the trot, which is why x/y being "pinned
        at the crouch values" held.  Stage 3 needs it: shifting all four feet
        by the same amount slides the TRUNK the opposite way, which is how the
        CoM is brought onto the support diagonal.  Doing it here rather than
        at run time keeps IK out of the CAN sweep -- the offset is a constant
        for a run, so it costs one table build and nothing per sweep.
        """
        self.leg = leg
        lo_lim, hi_lim = base.soft_limits()
        i = LEGS.index(leg)
        sl = slice(3 * i, 3 * i + 3)
        self.lo_lim, self.hi_lim = lo_lim[sl], hi_lim[sl]

        q_ref = np.asarray(q_ref, dtype=float)
        f_ref = dog5_kinematics.foot_position(leg, q_ref)
        self.xy = f_ref[:2] + np.asarray(xy_offset, dtype=float)
        z_ref = float(f_ref[2])
        if not (z_lo <= z_ref <= z_hi):
            raise ValueError(f"{leg}: reference z {z_ref:.4f} outside "
                             f"[{z_lo:.4f}, {z_hi:.4f}]")

        n_up = int(math.ceil((z_hi - z_ref) / step))
        n_dn = int(math.ceil((z_ref - z_lo) / step))
        zs = [z_ref + k * step for k in range(-n_dn, n_up + 1)]
        qs = [None] * len(zs)
        i_ref = n_dn
        # The march warm-starts from the table's midpoint, so that entry has to
        # sit at the SHIFTED x/y -- keeping the unshifted q_ref there would
        # leave one row of the table on the old anchor and bend every
        # interpolation near the stand pose.
        if np.any(self.xy != f_ref[:2]):
            q_ref, err = _ik_leg(leg, q_ref,
                                 np.array([self.xy[0], self.xy[1], z_ref]))
            if err > 1e-6:
                raise RuntimeError(
                    f"{leg}: foot x/y offset {xy_offset} is unreachable at the "
                    f"reference pose (residual {err*1e3:.2f} mm)")
        qs[i_ref] = q_ref.copy()

        # march outward in both directions, each step warm-started by the last
        for direction in (+1, -1):
            q = q_ref.copy()
            k = i_ref
            while 0 <= k + direction < len(zs):
                k += direction
                tgt = np.array([self.xy[0], self.xy[1], zs[k]])
                q, err = _ik_leg(leg, q, tgt)
                if err > 1e-6 or np.any(q < self.lo_lim) or np.any(q > self.hi_lim):
                    break            # unreachable / past a joint limit: stop here
                qs[k] = q.copy()

        keep = [k for k, v in enumerate(qs) if v is not None]
        if len(keep) < 2:
            raise RuntimeError(f"{leg}: height table degenerate")
        k0, k1 = keep[0], keep[-1]
        self.z = np.array(zs[k0:k1 + 1])
        self.q = np.array([qs[k] for k in range(k0, k1 + 1)])
        self.z_min, self.z_max = float(self.z[0]), float(self.z[-1])

    def q_at(self, z):
        """Joint angles for commanded foot z (clipped to the built range)."""
        z = float(np.clip(z, self.z_min, self.z_max))
        return np.array([np.interp(z, self.z, self.q[:, j]) for j in range(3)])

    def clip(self, z):
        return float(np.clip(z, self.z_min, self.z_max))


def build_tables(stand_height, clamp_m=HEIGHT_CLAMP_M, margin=TABLE_MARGIN_M,
                 xy_offset=(0.0, 0.0)):
    """One table per leg, spanning crouch .. (stand + clamp + margin).

    `xy_offset` shifts every foot's pinned x/y by the same amount in the trunk
    frame, which slides the trunk the OPPOSITE way over the same footprint.
    Default (0, 0) is every stage before the trot, unchanged.
    """
    tables = {}
    for i, leg in enumerate(LEGS):
        sl = slice(3 * i, 3 * i + 3)
        z_crouch = float(dog5_kinematics.foot_position(leg, Q_CROUCH[sl])[2])
        z_hi = z_crouch + margin
        z_lo = -(stand_height + clamp_m + margin)
        tables[leg] = LegHeightTable(leg, Q_CROUCH[sl], z_lo, z_hi,
                                     xy_offset=xy_offset)
    return tables


def crouch_foot_z():
    return {leg: float(dog5_kinematics.foot_position(
        leg, Q_CROUCH[3 * i:3 * i + 3])[2]) for i, leg in enumerate(LEGS)}


# ===========================================================================
# The height loop
# ===========================================================================

class HeightController:
    """Integral height loop -> a common foot-z offset.

    `offset` is a TRUNK-height correction in metres: positive raises the trunk,
    which means the commanded foot z goes *more negative* (feet pushed further
    below the body).  The caller applies `foot_z = base_z - offset`.

    Frozen state is sticky per call: when `active` is False the integrator
    simply stops.  It never unwinds on its own, so losing the EKF leaves the
    robot holding its current pose rather than collapsing to the open-loop one.
    """

    def __init__(self, gain_per_s=HEIGHT_GAIN_PER_S, clamp_m=HEIGHT_CLAMP_M,
                 slew_m_s=HEIGHT_SLEW_M_S, deadband_m=HEIGHT_DEADBAND_M):
        self.gain = float(gain_per_s)
        self.clamp = float(clamp_m)
        self.slew = float(slew_m_s)
        self.deadband = float(deadband_m)
        self.offset = 0.0
        self.last_t = None
        self.frozen_reason = None
        self.saturated = False

    def reset(self, offset=0.0):
        self.offset = float(offset)
        self.last_t = None
        self.frozen_reason = None
        self.saturated = False

    def update(self, now, h_meas, h_target, active, reason=None):
        """Advance the integrator.  Returns the offset to apply (metres)."""
        dt = 0.0 if self.last_t is None else max(0.0, now - self.last_t)
        self.last_t = now
        self.frozen_reason = None if active else (reason or "inactive")
        if not active or dt <= 0.0 or self.gain == 0.0:
            return self.offset

        err = float(h_target) - float(h_meas)
        if abs(err) <= self.deadband:
            return self.offset

        step = self.gain * err * dt
        step = float(np.clip(step, -self.slew * dt, self.slew * dt))
        new = float(np.clip(self.offset + step, -self.clamp, self.clamp))
        self.saturated = abs(new) >= self.clamp - 1e-12
        self.offset = new
        return self.offset


def height_inputs(shared, q, z0_fk_imu, now_mono, xcheck_m=HEIGHT_XCHECK_M):
    """(h_meas, h_fk, active, reason) for HeightController.update.

    Heights are of the IMU BOARD / TRUNK BOTTOM above the floor -- the point
    the EKF physically tracks and the only trunk point a ruler can reach.  The
    abd (hip) axis is IMU_BELOW_TRUNK_ORIGIN_M higher; it is the FK trunk
    origin, so it is what the kinematics are natural in, but nothing outside
    the leg tables needs it.  Reporting at the IMU keeps `r_z` untouched:

        h_meas = z0_fk_imu + r_z          (r_z IS the IMU's rise since init)

    FK is computed independently from the same encoders and used only to VETO,
    never as the feedback.  It cannot drift, so a large disagreement means the
    EKF has, and the loop must stop trusting it.
    """
    if shared is None or not shared.est_ready or shared.out is None:
        return float("nan"), float("nan"), False, "EKF not ready"
    if z0_fk_imu is None:
        return float("nan"), float("nan"), False, "no height origin"
    if now_mono - shared.tau_stamp > HEIGHT_STALE_S:
        return float("nan"), float("nan"), False, "EKF stale"
    out = shared.out
    if not out.get("healthy", False):
        return float("nan"), float("nan"), False, "EKF unhealthy"

    C = out["C"]
    h_ekf = z0_fk_imu + float(out["r"][2])
    h_fk = fk_floor_height(q, C, ref="imu")
    if abs(h_ekf - h_fk) > xcheck_m:
        return h_ekf, h_fk, False, f"EKF-FK disagree {abs(h_ekf-h_fk)*1e3:.0f}mm"
    return h_ekf, h_fk, True, None


# ===========================================================================
# Stage machine
# ===========================================================================

class HeightStandSequence:
    """CROUCH -> WAIT_CROUCH -> STAND -> HOLD4(closed loop) -> PARK -> PARKED.

    Commands are FOOT Z per leg; the caller turns them into joint angles with
    the height tables.  The loop's offset is applied only in HOLD4.  PARK ramps
    from whatever pose was actually being held -- offset included -- so PARKED
    lands on the recorded crouch regardless of what the integrator wound in.
    """

    MOVING_STAGES = ("STAND", "PARK")

    def __init__(self, tables, stand_height, t_stand=T_STAND):
        self.tables = tables
        self.stand_height = float(stand_height)
        self.t_stand = float(t_stand)
        self.z_crouch = crouch_foot_z()
        self.z_stand = {leg: -self.stand_height for leg in LEGS}
        self.stage = "CROUCH"
        self.stage_t0 = None
        self.fault = None
        self.n_stands = 0
        self.offset = 0.0            # applied trunk-height correction (m)
        self._settle_since = None
        self._park_from = None       # foot z per leg captured at park start

    @property
    def contacts(self):
        """ON throughout: the feet never leave the floor in this script, and
        stage 1 measured that dropping them costs the entire height estimate."""
        return np.ones(4, bool)

    @property
    def loop_engaged(self):
        return self.stage == "HOLD4"

    def foot_z(self, now):
        """Commanded foot z per leg for the current stage."""
        if self.stage in ("CROUCH", "WAIT_CROUCH"):
            return dict(self.z_crouch)
        if self.stage == "STAND":
            s = _smoothstep((now - self.stage_t0) / self.t_stand)
            return {leg: self.z_crouch[leg]
                    + s * (self.z_stand[leg] - self.z_crouch[leg]) for leg in LEGS}
        if self.stage == "HOLD4":
            # positive offset raises the trunk -> push the feet further down
            return {leg: self.z_stand[leg] - self.offset for leg in LEGS}
        if self.stage == "PARK":
            s = _smoothstep((now - self.stage_t0) / self.t_stand)
            return {leg: self._park_from[leg]
                    + s * (self.z_crouch[leg] - self._park_from[leg])
                    for leg in LEGS}
        return dict(self.z_crouch)                      # PARKED

    def q_cmd(self, now):
        z = self.foot_z(now)
        q = np.zeros(N_JOINTS)
        for i, leg in enumerate(LEGS):
            q[3 * i:3 * i + 3] = self.tables[leg].q_at(z[leg])
        return q

    def update(self, now, q, qd, enter_pressed=False, park_pressed=False,
               offset=0.0):
        if self.stage_t0 is None:
            self.stage_t0 = now
        event = None
        self.offset = float(offset) if self.stage == "HOLD4" else self.offset

        if self.stage == "CROUCH":
            settled = (float(np.max(np.abs(q - Q_CROUCH))) <= recorded.RECORDED_POSE_TOL
                       and float(np.max(np.abs(qd))) <= recorded.RECORDED_QD_TOL)
            if not settled:
                self._settle_since = None
                if now - self.stage_t0 > recorded.CROUCH_TIMEOUT_S:
                    self.fault = "CROUCH timeout"
            elif self._settle_since is None:
                self._settle_since = now
            elif now - self._settle_since >= recorded.CROUCH_SETTLE_S:
                self.stage, self.stage_t0 = "WAIT_CROUCH", now
                event = "crouch_settled"
        elif self.stage == "WAIT_CROUCH":
            if enter_pressed:
                self.stage, self.stage_t0 = "STAND", now
                self.offset = 0.0
                event = "stand_started"
        elif self.stage == "STAND":
            if now - self.stage_t0 >= self.t_stand:
                self.stage, self.stage_t0 = "HOLD4", now
                self.n_stands += 1
                event = "stand_complete"
        elif self.stage == "HOLD4":
            if park_pressed:
                self._park_from = self.foot_z(now)      # freeze the held pose
                self.stage, self.stage_t0 = "PARK", now
                event = "park_started"
        elif self.stage == "PARK":
            if now - self.stage_t0 >= self.t_stand:
                self.stage, self.stage_t0 = "PARKED", now
                self.offset = 0.0
                event = "park_complete"
        else:                                            # PARKED
            if enter_pressed:
                self.stage, self.stage_t0 = "STAND", now
                self.offset = 0.0
                event = "stand_started"

        return self.q_cmd(now), self.contacts, event


# ===========================================================================
# Hardware run
# ===========================================================================

def run(port, stand_height, target_height, crouch_max_speed_dps, gain,
        clamp, xcheck, log_path=None, raw_log_path=None):
    base.validate_hardware_config()
    print("[init] building per-leg height tables (IK, once) ...", flush=True)
    t0 = time.perf_counter()
    tables = build_tables(stand_height, clamp_m=clamp)
    span = {leg: (tables[leg].z_min, tables[leg].z_max) for leg in LEGS}
    print(f"[init] tables built in {time.perf_counter()-t0:.2f}s; "
          + "  ".join(f"{leg}[{span[leg][0]*1e3:.0f},{span[leg][1]*1e3:.0f}]mm"
                      for leg in LEGS))

    if target_height is None:
        # IMU-board floor height of the open-loop stand pose.  Feet sit
        # stand_height below the abd axis, the floor is one foot radius below
        # the foot site, and the board is IMU_BELOW under the abd axis.
        target_height = (stand_height + FOOT_RADIUS_M
                         - IMU_BELOW_TRUNK_ORIGIN_M)
    unwrap = [base.CalibratedEncoderUnwrap() for _ in MOTOR_IDS]
    key = base.KeyPoller()
    gate = base.SafetyGate(tau_cap=base.STAGED_TAU_MAX)
    stats = s1.AttitudeStats()
    report = s1.StageReport()
    ctrl = HeightController(gain_per_s=gain, clamp_m=clamp)

    print("=" * 78)
    print("DOG5 CLOSED-LOOP HEIGHT STAND  (stage 2: EKF height in the loop)")
    print(f"  target = {target_height*1e3:.0f} mm above floor at the TRUNK "
          f"BOTTOM / IMU BOARD -- the point the ruler and the EKF both see")
    print(f"  open-loop pose = {(stand_height+FOOT_RADIUS_M-IMU_BELOW_TRUNK_ORIGIN_M)*1e3:.0f}"
          f" mm there, = {(stand_height+FOOT_RADIUS_M)*1e3:.0f} mm at the abd "
          f"axis ({IMU_BELOW_TRUNK_ORIGIN_M*1e3:.0f} mm higher)")
    print(f"  integral gain {gain:.2f}/s, clamp +/-{clamp*1e3:.0f} mm, "
          f"slew {HEIGHT_SLEW_M_S*1e3:.0f} mm/s, EKF-FK veto {xcheck*1e3:.0f} mm")
    print("  NO torque commanded; drivers hold position. Contacts ON throughout.")
    print("  Keys: ENTER = stand, P = park (from HOLD4), X = stop.")
    if gain == 0.0:
        print("  !! gain 0 -- OPEN LOOP, this reproduces stage 1")
    print("=" * 78)

    log_file = writer = None
    if log_path:
        log_file = open(log_path, "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["t", "stage", "h_target", "h_ekf", "h_fk", "offset",
                         "active", "reason", "rz", "roll_est", "pitch_est",
                         "roll_ahrs", "pitch_ahrs", "healthy"])
    enc_log = []

    stop_reason = None
    worker = shared = None
    log_t0 = None
    init_secs = 0.0
    z0_fk_imu = None
    try:
        from imu_ekf_feed import ImuEkfFeed
        with motorbus.MotorBus(MOTOR_IDS, dirs=base.MOTOR_DIRECTIONS) as mb, \
                ImuEkfFeed(port) as feed:
            if not mb.arm(rate_hz=CONTROL_HZ):
                raise RuntimeError("arm failed (bus / power / terminators)")
            if not feed.wait_for_raw(timeout=3.0):
                raise RuntimeError("no raw IMU (0x40) packets -- enable DETA10 raw mode")

            start_q = base._zero_torque_preflight(mb, key, unwrap)
            if start_q is None:
                print("[abort] preflight not confirmed")
                return

            now = time.perf_counter()
            gate.start(now, start_q)
            shared = EkfShared(start_q)
            worker = threading.Thread(
                target=ekf_worker, args=(shared, feed),
                kwargs=dict(quiet_stages=QUIET_STAGES, control_hz=EKF_WORKER_HZ),
                daemon=True)
            worker.start()

            seq = HeightStandSequence(tables, stand_height)
            miss_monitor = base.CanMissMonitor(mb)
            slot = mb.slot(CONTROL_HZ)
            deadline = time.perf_counter() + slot
            status_period = 1.0 / base.FAULT_STATUS_HZ
            next_fault_status = np.array(
                [status_period + i * status_period / N_JOINTS for i in range(N_JOINTS)])
            last_recover = {mid: -base.RECOVER_PERIOD_S for mid in MOTOR_IDS}
            start = now
            cmd_deg = POSITION_TARGET_DEG.copy()
            index = 0
            last_print = 0.0
            tx_fail = 0
            bias_printed = False
            quiet_t0 = None
            last_reason = None

            while True:
                mb.poll()
                joint_index = index % N_JOINTS
                if joint_index == 0:
                    now = time.perf_counter()
                    now_mono = time.monotonic()
                    q, qd = base._joint_state(mb, unwrap)

                    pressed = key.get()
                    if pressed in ("x", "X"):
                        stop_reason = "operator X"
                        break

                    # ---- the height loop ----
                    h_ekf, h_fk, active, reason = height_inputs(
                        shared, q, z0_fk_imu, now_mono, xcheck_m=xcheck)
                    engaged = active and seq.loop_engaged
                    offset = ctrl.update(now, h_ekf, target_height, engaged,
                                         reason if not active else "not holding")
                    if seq.loop_engaged and reason and reason != last_reason:
                        print(f"[loop] FROZEN: {reason} (holding offset "
                              f"{offset*1e3:+.1f} mm)")
                    last_reason = reason

                    q_cmd, contacts, event = seq.update(
                        now, q, qd,
                        enter_pressed=pressed in ("\r", "\n"),
                        park_pressed=pressed in ("p", "P"),
                        offset=offset)
                    cmd_deg = np.rad2deg(q_cmd)
                    if seq.fault:
                        stop_reason = seq.fault
                        break

                    shared.q = q
                    shared.qd = qd
                    shared.stage = seq.stage
                    shared.contacts = contacts

                    if event == "crouch_settled":
                        quiet_t0 = now
                        print("[stage] CROUCH settled; EKF initialising. ENTER to STAND.")
                    elif event == "stand_started":
                        quiet_t0 = None
                        ctrl.reset(0.0)
                        if seq.n_stands == 0 and log_t0 is not None:
                            init_secs = (now - start) - log_t0
                        print("[stage] STAND: open-loop ramp crouch -> stand.")
                    elif event == "stand_complete":
                        quiet_t0 = now
                        stats.begin_visit()
                        print(f"[stage] HOLD4 (stand #{seq.n_stands}): height loop "
                              f"ENGAGED, target {target_height*1e3:.0f} mm. "
                              "P parks, X stops.")
                    elif event == "park_started":
                        quiet_t0 = None
                        line = stats.end_visit()
                        if line:
                            print("[attitude] this stand scored:")
                            print(line)
                        print(f"[stage] PARK: unwinding {offset*1e3:+.1f} mm of "
                              "offset on the way down to the crouch.")
                    elif event == "park_complete":
                        quiet_t0 = now
                        ctrl.reset(0.0)
                        print("[stage] PARKED: holding the recorded crouch. "
                              "ENTER stands again, X stops.")

                    if not bias_printed and shared.est_ready:
                        print(f"[ekf] init: {shared.bias_str}")
                        bias_printed = True
                    if log_t0 is None and shared.est_ready and seq.stage == "WAIT_CROUCH":
                        shared.log_enabled = True
                        log_t0 = now - start

                    # ---- safety ----
                    temps = base._temperatures(mb)
                    misses = miss_monitor.update(mb)
                    errors = mb.errors()
                    if stop_reason is None:
                        stop_reason = gate.estop_reason(q, qd, temps, misses, errors,
                                                        now, enforce_position_limits=True)
                    if stop_reason is None:
                        stop_reason = gate.overspeed_reason(qd, q, now)
                    if stop_reason:
                        break
                    latched = [mid for mid, e in errors.items() if e and (e & 0x80)]
                    recover = [mid for mid in latched
                               if (now - start) - last_recover[mid] >= base.RECOVER_PERIOD_S]
                    if recover:
                        base._recover_input_lost(mb, recover, now - start,
                                                 last_recover, next_fault_status)

                    # ---- observe / log ----
                    out = shared.out
                    r_a = p_a = float("nan")
                    att = feed.attitude()
                    if att is not None:
                        r_a, p_a = math.radians(att.roll_deg), math.radians(att.pitch_deg)
                    re_ = pe = float("nan")
                    if shared.est_ready and out is not None:
                        re_, pe = _rp(out["C"])
                        z_fk_imu = fk_floor_height(q, out["C"], ref="imu")
                        if z0_fk_imu is None:
                            z0_fk_imu = z_fk_imu       # latch at EKF init (crouch)
                        report.set_origin(z_fk_imu)
                        if quiet_t0 is not None and now - quiet_t0 >= SETTLE_S:
                            report.add(seq.stage, z_fk_imu, float(out["r"][2]), re_, pe)
                            if seq.stage == "HOLD4":
                                stats.add(re_, pe, r_a, p_a)
                    if log_t0 is not None:
                        enc_log.append((now_mono, *q, *contacts.astype(int), r_a, p_a))
                    if writer and shared.est_ready and out is not None:
                        writer.writerow([f"{now-start:.4f}", seq.stage,
                                         f"{target_height:.5f}", f"{h_ekf:.5f}",
                                         f"{h_fk:.5f}", f"{offset:.5f}",
                                         int(engaged), reason or "",
                                         f"{out['r'][2]:.5f}", f"{re_:.5f}",
                                         f"{pe:.5f}", f"{r_a:.5f}", f"{p_a:.5f}",
                                         int(out["healthy"])])

                    if now - last_print >= 1.0 / base.STATUS_HZ:
                        last_print = now
                        extra = ""
                        if shared.est_ready and out is not None:
                            flag = ("LOOP" if engaged else
                                    ("hold" if seq.loop_engaged else "----"))
                            # h/fk are TRUNK-BOTTOM (IMU board) heights, what a
                            # ruler reads.  abd is the same pose at the FK
                            # trunk origin, for comparison with the kinematics.
                            h_abd = (h_ekf + IMU_BELOW_TRUNK_ORIGIN_M
                                     * float(out["C"][2, 2]))
                            extra = (f" h={h_ekf*1e3:6.1f}(fk{h_fk*1e3:6.1f}"
                                     f"|abd{h_abd*1e3:6.1f}) "
                                     f"err={(target_height-h_ekf)*1e3:+5.1f} "
                                     f"off={offset*1e3:+5.1f}mm [{flag}] "
                                     f"roll={math.degrees(re_):+5.1f} "
                                     f"pitch={math.degrees(pe):+5.1f}")
                        print(f"[hw] {seq.stage:11s} max|qd|={np.max(np.abs(qd)):.2f} "
                              f"Tmax={int(np.max(temps))}C latched={len(latched)} "
                              f"txfail={tx_fail}{extra}", flush=True)

                mid = MOTOR_IDS[joint_index]
                if (time.perf_counter() - start) >= next_fault_status[joint_index]:
                    mb.status1_req(mid)
                    next_fault_status[joint_index] += status_period
                elif not mb.position(mid, float(cmd_deg[joint_index]), crouch_max_speed_dps):
                    tx_fail += 1
                index += 1
                overrun = mb.pace(deadline)
                deadline += slot
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
        if log_file:
            log_file.close()
            print(f"[log] wrote {log_path}")
        if raw_log_path and shared is not None:
            imu = np.array(shared.imu_log, dtype=float)
            enc = np.array(enc_log, dtype=float)
            if len(imu) and len(enc):
                np.savez(raw_log_path,
                         imu_t=imu[:, 0], imu_f=imu[:, 1:4], imu_w=imu[:, 4:7],
                         enc_t=enc[:, 0], enc_alpha=enc[:, 1:13],
                         enc_contacts=enc[:, 13:17], ahrs_rp=enc[:, 17:19],
                         init_secs=max(init_secs, 0.5))
                print(f"[raw] wrote {raw_log_path} "
                      f"({len(imu)} IMU, {len(enc)} enc frames)")
    stats.end_visit()
    for line in stats.summary():
        print(line)
    for line in report.summary():
        print(line)
    print(f"[loop] final offset {ctrl.offset*1e3:+.1f} mm"
          + (f" (SATURATED at the {clamp*1e3:.0f} mm clamp)" if ctrl.saturated else ""))
    print(f"[stop] {stop_reason}")
# ===========================================================================
# offline self-test -> test_stand_ekf_height.py
# ===========================================================================
# The gates moved out of this file.  They were 296 lines of simulation that no
# hardware run ever executes, and several of them shared a plane plant, a fake
# EKF and a PASS/FAIL harness with the other runners -- all of which now live
# in selftest_common.py.  `--self-test` still runs exactly those gates.
#
#     $V self-test/test_stand_ekf_height.py     # the same thing, run directly


def self_test(stand_height):
    """Delegate to the suite in self-test/.  Lazy: a hardware run never loads it."""
    _sd = s1.selftest_dir()
    if _sd not in sys.path:
        sys.path.insert(0, _sd)
    from test_stand_ekf_height import self_test as gates
    return gates(stand_height)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--stand-height", type=float, default=recorded.DEFAULT_STAND_HEIGHT,
                    help="open-loop pose height (foot site below trunk origin)")
    ap.add_argument("--target-height", type=float, default=None,
                    help="closed-loop target, metres from the floor to the "
                         "TRUNK BOTTOM / IMU BOARD (NOT the abd axis, which is "
                         f"{IMU_BELOW_TRUNK_ORIGIN_M*1e3:.0f} mm higher). "
                         "Default: the open-loop pose, i.e. remove the sag")
    ap.add_argument("--crouch-max-speed-dps", type=float,
                    default=recorded.DEFAULT_CROUCH_MAX_MOTOR_DPS)
    ap.add_argument("--height-gain", type=float, default=HEIGHT_GAIN_PER_S,
                    help="integral gain (1/s); 0 = open loop, reproduces stage 1")
    ap.add_argument("--clamp", type=float, default=HEIGHT_CLAMP_M,
                    help="max |offset| the loop may wind in (m)")
    ap.add_argument("--xcheck", type=float, default=HEIGHT_XCHECK_M,
                    help="|EKF-FK| height disagreement that freezes the loop (m)")
    ap.add_argument("--log", default=None)
    ap.add_argument("--raw-log", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test(args.stand_height))
    run(args.port, args.stand_height, args.target_height,
        args.crouch_max_speed_dps, args.height_gain, args.clamp, args.xcheck,
        args.log, args.raw_log)


if __name__ == "__main__":
    main()
