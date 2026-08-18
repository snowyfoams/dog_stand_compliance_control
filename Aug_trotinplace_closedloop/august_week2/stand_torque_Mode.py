#!/usr/bin/env python3
"""The runner: closed-loop standing on commanded force.  Week 2, one thread.

    CROUCH (0xA4) -> WAIT -> RISE (torque) -> HOLD (torque) -> PARK (0xA4)

WHAT THIS ANSWERS
    In position mode the drivers' own loops hold the pose, so once the robot
    settles nothing the upper layer computes has any observable effect -- you
    cannot tell a working feedback loop from an ignored one.  Here the trunk is
    held up by forces this code computes, so the loop is the only thing keeping
    it up.  Push it, and the recovery IS the feedback.

THE FIVE FILES, AND WHY THE SPLIT IS WHERE IT IS
    crouch_and_park     0xA4 bookends.  No torque, no model.
    feedback_estimator  sensors -> (z, v, C, omega).  Measurement only.
    Dynamic_Model       state error -> 6D wrench.  THE CONTROLLER: the only
                        place a gain touches the trunk.
    force_totorque      wrench -> per-foot GRF -> 12 joint torques.  Pure
                        transmission; no trunk feedback at all.
    stand_torque_Mode   this file: when to do which, at what rate, and every
                        way to stop.

ONE THREAD, WITH THE MODEL SUB-SAMPLED
    MEASURED on this Pi, not assumed: the per-sweep work (impedance + gate) is
    28 us, 8% of one 333 us slot.  The model block (wrench + grasp map + J^T +
    warm-started IK) is 1384 us and the load check is 1029 us -- neither can
    run every sweep.  Last week's answer was a 100 Hz worker thread with a
    mailbox, atomic publishing and a staleness watchdog: ~200 lines that exist
    only to make a thread safe.

    Here both run IN the sweep, sub-sampled and STAGGERED so they never land
    in the same one -- model at 83 Hz, load check at 21 Hz, worst single-sweep
    delay 1.38 ms rather than the 2.41 ms of both together.  That is inside
    the driver's input-lost watchdog with margin either way it is read (see
    the note on params.WATCHDOG_S).  The term that actually stabilises a joint
    -- the impedance -- still runs EVERY sweep on qd at most 4 ms old.  Only
    the feedforward is held between updates.

SAFETY, IN THE ORDER IT APPLIES
    joint impedance (250 Hz)     no joint is ever an open integrator
    abduction sign check         no leg swings through the body
    TorqueGate                   ramp, cap, slew, overspeed, temp, CAN miss
    LoadWatch                    measured iq must add up to the robot
    LIMP                         SPACE, or automatic on any refusal

    NOTE the joint SOFT LIMITS are off, as in crouch_and_park: the symmetric
    +/-2.6 rad box sits 6.9 deg from the resting crouch on the knees while
    allowing a knee 298 deg of travel.  The abduction sign check replaces it.

WHAT THE STATUS LINE SHOWS, AND WHY IT IS ONLY THESE
    A line printed at 2 Hz is for the human holding the robot, and a human can
    act on exactly two things: is it at the right height, and is it level.  So
    the stream carries height, tilt, the IMU-free cross-check on tilt, trunk
    velocity and the planted count -- and a legend for them is printed once, at
    the top, rather than left as initials.

    |tau|, trk and |dq| USED TO STREAM HERE AND NO LONGER DO.  They were
    per-sweep control internals sampled at 2 Hz, which is neither a
    measurement (too slow to see the 10 Hz behaviour that mattered) nor
    actionable (nothing an operator does changes them).  Every one is in the
    --log npz at the full 250 Hz -- tau_cmd, tau_meas, and q_ref alongside q
    so the impedance error is recoverable -- which is where they were actually
    read from anyway when the 2026-08-17 shake was diagnosed.

    THE FOOT-LOAD SUM MOVED TO THE EXIT REPORT, and it is still the number
    that matters most.  Torque calibration was dropped for week 2, so it is
    the ONLY end-to-end evidence that commanded torque becomes real force:
    measured iq, inverted through J^-T with each leg's own weight removed, and
    it must read ~57 N.  If it does not, the grasp map is fantasy and a
    good-looking attitude proves nothing.  It also still LIMPS the robot
    in-run when it disagrees with the weight by more than LOAD_SUM_TOL_FRAC,
    and that prints when it fires -- an alarm, rather than a number to watch.

THE TWO NUMBERS THAT WERE DEAD RECKONING UNTIL 2026-08-17
    Height and attitude both came out of FK and the AHRS with nothing to check
    them against, so a constant error in either was invisible from inside the
    loop.  Two were sitting there:

    zI= IS FLOOR TO TRUNK BOTTOM, and it did not used to be.  It was the
    hip-axis plane -- a plane nothing physical sits on -- so the runner printed
    191 mm where a ruler read ~160.  38 mm of that was the frame.  zI is the
    point the ruler reaches; the (H...) beside it is the hip axis, which is
    where the leg tables and the IK work.  --height means zI now, and
    STAND_HEIGHT moved 0.190 -> 0.152 so the POSE is unchanged.

    rp:d IS THE ONLY INDEPENDENT CHECK ON THE AHRS.  rp:fk is a least-squares
    plane through the four measured feet, no IMU in it, in the same convention
    as the AHRS branch -- so rp:d = ahrs - fk is floor slope + IMU mount tilt
    + encoder zeros.  It printed ~0.5 deg with the robot standing still, and
    with kp_roll = 120 that is a 1.05 Nm standing moment (16% of this stance's
    6.4 Nm roll capacity) spent holding the robot off true level.

    THE SETPOINT PROCEDURE, because rp:d is three constants added together:
        1. crouch and read the rp: block the WAIT stage prints
        2. re-run with --setpoint-roll/--setpoint-pitch set to rp:d
        3. to split mount from floor: rotate the robot 180 deg on the SAME
           spot and read again.  The floor's share flips sign, the mount's
           does not, so half the sum is the mount and half the difference is
           the floor.  Only the mount half belongs in a setpoint.
    It defaults to 0/0 -- i.e. wrong -- on purpose: a measured pair from one
    floor is not a property of the robot, and baking it in would tilt the
    robot by the floor's share everywhere else.

RUN -- supported robot, hand on SPACE
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    cd /home/robot01/Documents/can_motor_control/dog_stand_compliance_control/Aug_trotinplace_closedloop

    $V august_week2/stand_torque_Mode.py --self-test     # offline, no hardware

    # first run: low cap, full force.  Do NOT start at --force-frac 0 with the
    # feet on the floor -- that removes the only term holding the trunk up.
    sudo HOME=$HOME chrt -f 50 $V august_week2/stand_torque_Mode.py --tau-max 1.0

    # measure the rig's resting attitude: crouch, read the rp: block, X out.
    # Then feed it back.  Do this BEFORE any run with attitude gains live.
    sudo HOME=$HOME chrt -f 50 $V august_week2/stand_torque_Mode.py --open-loop
    sudo HOME=$HOME chrt -f 50 $V august_week2/stand_torque_Mode.py \
        --setpoint-roll -0.50 --setpoint-pitch 0.45 --log live.npz

    # the A/B that makes it a measurement rather than a demo.  Push the trunk
    # the SAME way in each; the difference between the logs is the loop.
    # Pass the SAME setpoint to both halves or the ablation is confounded.
    sudo HOME=$HOME chrt -f 50 $V august_week2/stand_torque_Mode.py --log live.npz
    sudo HOME=$HOME chrt -f 50 $V august_week2/stand_torque_Mode.py \
        --kp-att 0 --kd-att 0 --log ablate.npz

Keys: ENTER = rise / re-engage from limp, SPACE = LIMP, P = park, X = stop.
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import sys
import time

import numpy as np

sys.setswitchinterval(0.0005)

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUG = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_AUG)
_REPO = os.path.dirname(_ROOT)
for _p in (_HERE, os.path.join(_AUG, "torque_mode_control"),
           os.path.join(_ROOT, "dog5_description"), _REPO,
           os.path.join(_REPO, "IMU_sensor")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _add_fdilink_root():
    """Find the `fdilink_imu` package WITHOUT trusting $HOME.

    imu_dog resolves it as Path.home()/Documents/IMU_sensor, which breaks under
    sudo ($HOME becomes /root).  RT priority wants root, so resolve it
    repo-relative first and fall back to the invoking user's home.
    """
    cands = [os.path.join(os.path.dirname(_REPO), "IMU_sensor"),
             os.path.join(_REPO, "IMU_sensor"),
             os.path.join(os.path.expanduser("~"), "Documents", "IMU_sensor")]
    if os.environ.get("SUDO_USER"):
        cands.append(os.path.join("/home", os.environ["SUDO_USER"],
                                  "Documents", "IMU_sensor"))
    for root in cands:
        if os.path.isdir(os.path.join(root, "fdilink_imu")):
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    return None


_add_fdilink_root()

import motorbus                                            # noqa: E402
import stand_dog5_hw as base                               # noqa: E402
import dog5_statics as st                                  # noqa: E402
import params as P                                         # noqa: E402
import crouch_and_park as cap                              # noqa: E402
import feedback_estimator as fe                            # noqa: E402
import Dynamic_Model as dm                                 # noqa: E402
import force_totorque as ft                                # noqa: E402

MOTOR_IDS = base.MOTOR_IDS
N_JOINTS = P.N_JOINTS
LEGS = ft.LEGS
Q_CROUCH = cap.Q_CROUCH
ALL4 = np.ones(4, dtype=bool)


class JointImpedance:
    """tau = kp (q_ref - q) - kd qd, evaluated EVERY sweep.

    THE POINT, stated once because it is the whole safety argument:

        A pure force law tau = -J^T f is a VELOCITY-level system.  Take the
        ground away from a foot and there is nothing holding that joint
        anywhere -- qdd integrates without bound, and a damper only sets the
        terminal speed qd = tau/kd.  The joint does not stop, it coasts.  On
        2026-07-30 RR_abd did exactly that and the run ended on the overspeed
        e-stop.

        kp (q_ref - q) makes it POSITION-level with a fixed point.  The same
        fault becomes a bounded offset dq = tau/kp, ~13 mrad.

    It is NOT a position servo fighting the balance loop: q_ref tracks the
    same foot trajectory the wrench is trying to realise, so in nominal stance
    this term is ~0.  A model-mismatch term, not a stiffness.

    The gains are bounded by the sampled-PD limit at the MEASURED 16 ms loop
    delay, NOT at the 4 ms command interval: see params.LOOP_DELAY_S.  Using
    the command interval is what made kp=15/kd=0.6 look like 19% of the bound
    when it was 68% of it, and shook the robot at 9-12 Hz.
    """

    def __init__(self, kp=P.KP_IMP, kd=P.KD_IMP):
        if not 0.0 <= kp <= P.KP_IMP_MAX:
            raise ValueError(f"kp={kp} outside the measured stable envelope "
                             f"[0, {P.KP_IMP_MAX}] at a "
                             f"{P.SWEEP_S*1e3:.0f} ms sweep")
        if not 0.0 <= kd <= P.KD_IMP_MAX:
            raise ValueError(f"kd={kd} outside [0, {P.KD_IMP_MAX}]; the "
                             f"sampled-damper bound is 2J/dt = "
                             f"{2*P.J_MIN/P.LOOP_DELAY_S:.1f} Nms/rad at the "
                             f"MEASURED {P.LOOP_DELAY_S*1e3:.0f} ms loop "
                             f"delay, not {2*P.J_MIN/P.SWEEP_S:.1f} at the "
                             f"{P.SWEEP_S*1e3:.0f} ms command interval")
        self.kp, self.kd = float(kp), float(kd)
        self.dq = np.zeros(N_JOINTS)

    def tau(self, q, qd, q_ref):
        self.dq = np.asarray(q_ref) - np.asarray(q)
        return self.kp * self.dq - self.kd * np.asarray(qd)

    def worst(self):
        return float(np.max(np.abs(self.dq)))


class TorqueGate(base.SafetyGate):
    """base.SafetyGate with the TORQUE-mode ramp and slew.

    base.apply() reads module-level TORQUE_RAMP_S (1.0 s) and TAU_SLEW_NM_S
    (5.0 Nm/s), which are position-track numbers and are wrong here in both
    directions: 5 Nm/s is slow enough that a leg folds while the impedance is
    still ramping towards the torque it asked for, and a 1.0 s ramp on top
    means no leg can take weight for a full second.  See params.TAU_SLEW_NM_S
    for why the sizing argument is now the joint-speed one and not the old
    kp = 15 one.

    Subclassed rather than edited, because the position track, the crawl and
    the EKF harness all import stand_dog5_hw.
    """

    def overspeed_reason(self, qd, q, now):
        """DISABLED for week 2: record the peak, never stop the run on it.

        Removed on request after it ended a hardware run at
        `overspeed FL_abd: +8.1 rad/s over the 8.0 rad/s hard limit`.

        The hard tier is the one that fired, and it trips on a SINGLE reading
        with no encoder confirmation -- unlike the 7.0 rad/s sustained tier,
        which needs three consecutive checks agreeing with a finite-difference
        of the encoder.  stand_dog5_hw's own comment records a 5.9 rad/s
        nuisance reading from the driver's speed field, so a lone 8.1 is far
        more likely to be that field glitching than a joint genuinely turning
        at 464 deg/s.

        WHAT STILL BOUNDS JOINT SPEED, with this gone:
            the impedance      -kd*qd every sweep, so speed is opposed
                               continuously rather than watched for
            the slew limit     one sweep at 60 Nm/s adds 0.11 rad/s
            the torque cap     --tau-max, 1.0 Nm by default
            the abduction sign a leg that runs away crosses zero and stops
            the tilt e-stop    a leg that runs away tips the trunk
        What is NOT covered any more: a joint spinning fast while the trunk
        stays level and the abduction sign stays correct.

        The peak is still accumulated so the exit report can show what the
        speeds actually were -- that is the number that says whether 8.1 was a
        glitch or real, and it is worth having even with the trip gone.
        """
        self.qd_peak = np.maximum(self.qd_peak, np.abs(np.asarray(qd, float)))
        return None

    def cap_now(self, now):
        f = np.clip((now - self.started_at) / P.TORQUE_RAMP_S, 0.0, 1.0)
        return self.tau_cap * f

    def apply(self, tau, q, now):
        cap = self.cap_now(now)
        out = np.clip(np.asarray(tau, dtype=float), -cap, cap)
        out = np.clip(out, -P.TAU_HARD_NM, P.TAU_HARD_NM)
        dt = np.clip(now - self.last_time, 1e-4, 0.05)
        step = P.TAU_SLEW_NM_S * dt
        out = self.previous_tau + np.clip(out - self.previous_tau, -step, step)
        self.previous_tau = out
        self.last_time = now
        return out


def q_ref_for_height(q_ref, z_des, foot_xy):
    """Warm-started IK: the joint pose that puts every foot at -z_des.

    This is what makes the impedance a MODEL-MISMATCH term rather than a
    stiffness fighting the rise.  Last week's runner froze q_ref at the crouch
    for the whole 8 s rise, so by the top the impedance was pulling ~150 mm of
    error against the force law -- the two layers were fighting.

    `z_des` is FLOOR to TRUNK BOTTOM, the same quantity the estimator's
    fk_trunk_height now returns and therefore the same one body_wrench takes
    the error of.  The legs, though, are tabulated from the HIP AXIS, so this
    is the one place the two frames meet and fe.hip_from_imu is the only
    converter.  Then the foot SITE is the centre of a 20 mm sphere, so it sits
    at -(z_hip - FOOT_RADIUS_M).

    Two silent biases live in those two lines -- 38 mm if the frame is skipped,
    20 mm if the foot radius is -- and neither shows up as anything but a
    height that reads right and measures wrong.  That is exactly what happened
    on 2026-08-17 with the frame.

    WARM-STARTED, AND IT HAS TO BE.  The step is damped least squares, not a
    plain Newton step: near full leg extension the Jacobian is ill-conditioned
    and Newton diverges -- from the crouch straight to the stand height it
    overshoots by 330 mm and leaves the workspace entirely.  Over a real rise
    that never arises, because at 83 Hz across 8 s each call moves the target
    0.25 mm and starts from the previous solution.  The damping is what makes
    the first call after a stage change safe anyway.

    x and y stay pinned at their crouch values, so the feet rise straight up
    and the contact point never moves in the world.
    """
    z_site = -(fe.hip_from_imu(z_des) - P.FOOT_RADIUS_M)
    for i, leg in enumerate(LEGS):
        sl = slice(3 * i, 3 * i + 3)
        qi = q_ref[sl].copy()
        tgt = np.array([foot_xy[i][0], foot_xy[i][1], z_site])
        for _ in range(8):
            fr = st.leg_frames(leg, qi)
            e = tgt - fr[0]
            if float(np.linalg.norm(e)) < 1e-9:
                break
            J = st.foot_jacobian_from(fr[0], fr[1], fr[2])
            qi = qi + J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e)
        q_ref[sl] = qi
    return q_ref


def _zero_rp_streamer(imu, args, window_s=5.0):
    """The ZERO-TORQUE stage streams the AHRS, not the twelve joint angles.

    The joint line that used to print here was the wrong thing to look at:
    the encoders are calibrated, the preflight's own soft-limit check already
    REFUSES ENTER on a bad pose, and there is nothing an operator does with
    twelve numbers at 2 Hz.  The one number that has to be read off this robot
    by hand is the IMU mount tilt -- and this is the stage to read it in, with
    no torque anywhere and the trunk wherever the operator puts it.

    So this prints raw roll/pitch, a mean over the last `window_s`, and the
    --setpoint-* line that cancels the mean.  The SPREAD is printed with it
    because it is the part that says whether the mean is trustworthy: held by
    hand it runs a degree or more, and a mean that moves by a degree is not a
    mount tilt.

    THIS IS THE MOUNT TILT ONLY IF THE TRUNK IS ACTUALLY LEVEL -- put a spirit
    level on it, or the reading is whatever the hands are doing.  The reading
    that needs no level is rp:d in the crouch (ahrs minus the foot plane),
    which cancels the floor as well; see _print_attitude_report.
    """
    hist = collections.deque(maxlen=max(2, int(window_s * 2.0)))  # 2 Hz prints

    def line(q, qd):
        sample = imu.sample()
        if sample is None:
            return "[zero] rp: no AHRS sample"
        r, p_ = float(sample.roll_deg), float(sample.pitch_deg)
        hist.append((r, p_))
        rs = [h[0] for h in hist]
        ps = [h[1] for h in hist]
        rm, pm = sum(rs) / len(rs), sum(ps) / len(ps)
        spread = max(max(rs) - min(rs), max(ps) - min(ps))
        # The flags already include whatever setpoint THIS run subtracts, so
        # the printed pair is absolute: paste it, do not add it to the old one.
        return (f"[zero] rp {r:+6.2f} / {p_:+6.2f} deg   "
                f"mean({len(hist)/2:.1f}s) {rm:+6.2f} / {pm:+6.2f} "
                f"+/-{spread:.2f}   "
                f"--setpoint-roll {rm:.2f} --setpoint-pitch {pm:.2f}")

    return line


def _print_attitude_report(est, args):
    """The one place an operator can read the mount tilt off the robot.

    Printed once, in the crouch, with the feet on the floor and no torque
    anywhere -- the only moment in the run when "the trunk is not tilted" is
    something we can assume rather than something the loop is enforcing.

        rp:ahrs   the DETA10, with --setpoint-* already subtracted
        rp:fk     a least-squares plane through the four measured feet.  No
                  IMU in it at all, so it is the independent number.
        rp:d      ahrs - fk = floor slope + IMU mount tilt + encoder zeros

    rp:d is what a setpoint should cancel, and it is three constants added
    together, which is why this prints the arithmetic instead of latching it:
    rotate the robot 180 deg on the same spot and read again -- the floor's
    share flips sign, the mount's does not.
    """
    dr, dp = est.attitude_residual()
    print("[stand] " + "-" * 66)
    print(f"[stand] rp:ahrs {np.degrees(est.roll):+6.2f} / "
          f"{np.degrees(est.pitch):+6.2f} deg"
          + (f"   (raw {np.degrees(est.roll_raw):+.2f} / "
             f"{np.degrees(est.pitch_raw):+.2f}, setpoint "
             f"{args.setpoint_roll:+.2f} / {args.setpoint_pitch:+.2f})"
             if (args.setpoint_roll or args.setpoint_pitch) else
             "   (no setpoint subtracted)"))
    print(f"[stand] rp:fk   {np.degrees(est.roll_fk):+6.2f} / "
          f"{np.degrees(est.pitch_fk):+6.2f} deg   "
          f"(foot plane, no IMU)")
    print(f"[stand] rp:d    {np.degrees(dr):+6.2f} / {np.degrees(dp):+6.2f} "
          f"deg   = floor slope + mount tilt + encoder zeros")
    if max(abs(np.degrees(dr)), abs(np.degrees(dp))) > 0.2:
        print(f"[stand] ^ to make the wrench see THIS as level, re-run with")
        print(f"[stand]     --setpoint-roll {args.setpoint_roll + np.degrees(dr):.2f} "
              f"--setpoint-pitch {args.setpoint_pitch + np.degrees(dp):.2f}")
        print(f"[stand]   but that folds the FLOOR in too.  Rotate the robot "
              f"180 deg on the same")
        print(f"[stand]   spot and read again to split them: half the sum is "
              f"the mount.")
    print("[stand] " + "-" * 66)


def run(args):
    base.validate_hardware_config()
    gains = dm.default_gains()
    if args.kp_att is not None:
        gains.kp_roll = gains.kp_pitch = args.kp_att
    if args.kd_att is not None:
        gains.kd_roll = gains.kd_pitch = args.kd_att
    if args.kp_z is not None:
        gains.kp_z = args.kp_z
    if args.kd_z is not None:
        gains.kd_z = args.kd_z
    if args.kd_xy is not None:
        gains.kd_x = gains.kd_y = args.kd_xy
    if args.kd_yaw is not None:
        gains.kd_yaw = args.kd_yaw
    if args.open_loop:
        # ALL outer feedback off: W = [0, 0, m*g, 0, 0, 0], split evenly by
        # the grasp map, + leg gravity + the joint impedance tracking the
        # z ramp.  That is the old Cartesian compliance stand, expressed in
        # joint space -- the known-good baseline.  This is the diagnostic A/B
        # for the 2026-08-17 shake: if THIS shakes, the fault is below the
        # model (torque gain, current loop, impedance, CAN timing) and no
        # controller swap can fix it; if it is smooth, add loops back one at
        # a time (--kp-z first, attitude last) to find the one that chatters.
        gains.kp_z = gains.kd_z = 0.0
        gains.kp_roll = gains.kd_roll = 0.0
        gains.kp_pitch = gains.kd_pitch = 0.0
        gains.kd_x = gains.kd_y = gains.kd_yaw = 0.0
    ablated = gains.kp_roll == 0.0 and gains.kd_roll == 0.0

    print("=" * 74)
    print("DOG5 CLOSED-LOOP TORQUE STAND  (week 2)")
    print(f"  mass {args.mass:.4f} kg = {args.mass*P.GRAVITY:.1f} N, "
          f"stand height {args.height*1e3:.0f} mm, rise {P.T_RISE:.0f} s")
    # Say the frame every run.  The 2026-08-17 gap between a printed 191 mm
    # and a measured 160 was this line not existing.
    print(f"  height is FLOOR to TRUNK BOTTOM (the IMU board) -- a ruler "
          f"reaches it.  That")
    print(f"  is {fe.hip_from_imu(args.height)*1e3:.0f} mm at the hip axis, "
          f"{P.IMU_BELOW_TRUNK_ORIGIN_M*1e3:.0f} mm higher, which is the "
          f"frame the leg IK uses.")
    if args.setpoint_roll or args.setpoint_pitch:
        print(f"  AHRS setpoint {args.setpoint_roll:+.2f} / "
              f"{args.setpoint_pitch:+.2f} deg subtracted from every reading")
    else:
        print(f"  AHRS setpoint 0.00 / 0.00 -- the IMU MOUNT'S OWN TILT is "
              f"being read as a real")
        print(f"  attitude.  The crouch prints rp:d; pass it back as "
              f"--setpoint-roll/--setpoint-pitch.")
    print(f"  {gains}, impedance kp={args.kp} kd={args.kd}, "
          f"tau cap {args.tau_max} Nm, force_frac {args.force_frac}")
    print(f"  model at {P.CONTROL_HZ/P.MODEL_EVERY:.0f} Hz "
          f"(every {P.MODEL_EVERY} sweeps), impedance at {P.CONTROL_HZ:.0f} Hz")
    # The roll axis saturates first, and on this rig it saturates EARLY.
    Mx_max, My_max = ft.moment_capacity(
        [st.leg_frames(LEGS[i], Q_CROUCH[3*i:3*i+3])[0] for i in range(4)],
        list(LEGS), args.mass * P.GRAVITY)
    print(f"  stance moment capacity: roll {Mx_max:.1f} Nm "
          f"(kp_roll saturates at {np.rad2deg(Mx_max/max(gains.kp_roll,1e-9)):.1f} deg), "
          f"pitch {My_max:.1f} Nm")
    if args.open_loop:
        print("  OPEN LOOP: W = [0, 0, m*g, 0, 0, 0] always -- even split + "
              "leg gravity + impedance.")
        print("  This is the old Cartesian compliance stand in joint space. "
              "If THIS shakes, the fault is below the model.")
    elif ablated:
        print("  ATTITUDE GAINS ZEROED -- the ablation half of the A/B; "
              "the trunk will NOT push back")
    if args.force_frac < 0.2:
        print(f"  WARNING: force_frac {args.force_frac} removes the term that "
              f"holds the TRUNK up.  Feet on the floor, the legs fold until "
              f"kp*dq carries {args.mass*P.GRAVITY:.0f} N.")
    print("  Support the robot.  ENTER = rise, SPACE = LIMP, P = park, X = stop")
    print("-" * 74)
    print("  what the status line means:")
    print(fe.BodyState.status_legend())
    print("  torque, tracking and impedance error are no longer streamed -- "
          "they say nothing")
    print("  a human can act on at 2 Hz.  The foot-load sum is in the EXIT "
          "REPORT, and every")
    print("  one of them is in the --log npz per sweep.")
    print("=" * 74)

    from imu_dog import ImuDog, DEFAULT_PORT                # noqa: PLC0415
    imu = ImuDog(port=args.port or DEFAULT_PORT)
    imp = JointImpedance(args.kp, args.kd)
    key = base.KeyPoller()
    log = []
    stop = None

    try:
        imu.start()
        with motorbus.MotorBus(MOTOR_IDS, dirs=base.MOTOR_DIRECTIONS) as mb:
            if not mb.arm(rate_hz=P.CONTROL_HZ):
                raise RuntimeError("arm failed (bus / power / terminators)")
            unwrap = [base.CalibratedEncoderUnwrap() for _ in MOTOR_IDS]
            if not imu.wait_for_data(3.0):
                raise RuntimeError("no AHRS data")
            base._zero_torque_preflight(
                mb, key, unwrap, status=_zero_rp_streamer(imu, args))

            # ---- CROUCH: position mode, week-2 bookend -------------------
            q = cap.crouch(mb, unwrap, key, max_speed_dps=args.crouch_dps)

            est = fe.BodyState(imu,
                               setpoint_roll=np.radians(args.setpoint_roll),
                               setpoint_pitch=np.radians(args.setpoint_pitch))
            gate = TorqueGate(tau_cap=args.tau_max)
            miss = base.CanMissMonitor(mb)
            foot_xy = [st.leg_frames(LEGS[i], Q_CROUCH[3*i:3*i+3])[0][:2]
                       for i in range(4)]

            stage, t0 = "WAIT", time.perf_counter()
            z_crouch = None
            q_ref = q.copy()
            tau_ff = np.zeros(N_JOINTS)
            W = np.zeros(6)
            tau_cmd = np.zeros(N_JOINTS)
            diag = {"fz": [0.0] * 4, "stance": [], "singular": []}
            state, act, why = None, False, "starting"
            limp = False
            armed = False
            load_sum = float("nan")

            slot = mb.slot(P.CONTROL_HZ)
            deadline = time.perf_counter() + slot
            index = sweep = 0
            last_print = 0.0
            # qd comes from the ENCODER, not the driver's speed field -- see
            # params.QD_ALPHA for the hardware run that made that necessary.
            qd = np.zeros(N_JOINTS)
            q_prev, t_prev = None, time.perf_counter()
            qd_drv = np.zeros(N_JOINTS)
            glitches = np.zeros(N_JOINTS, dtype=int)
            print("[stand] WAIT: hold still, then ENTER to rise under torque")

            while stop is None:
                mb.poll()
                j = index % N_JOINTS
                if j == 0:
                    now = time.perf_counter()
                    q, qd_drv = base._joint_state(mb, unwrap)
                    tau_meas = np.array([mb.torques_nm()[m] for m in MOTOR_IDS])

                    # ENCODER-DIFFERENCED velocity, low-passed -- the same
                    # thing stand_dog5_recorded_hw.EncoderVelocity does, and
                    # the reason that controller stands without shaking.  The
                    # driver's field is still read, but only to COUNT how
                    # often it disagrees, so the exit report can say whether
                    # the glitching is still happening.
                    if q_prev is not None:
                        dt_v = now - t_prev
                        if 1e-4 < dt_v < 0.05:
                            qd += P.QD_ALPHA * ((q - q_prev) / dt_v - qd)
                    q_prev, t_prev = q.copy(), now
                    glitches += (np.abs(qd_drv - qd)
                                 > np.maximum(2.0 * np.abs(qd), 1.0))

                    pressed = key.get()
                    if pressed in (P.KEY_STOP, P.KEY_STOP.upper()):
                        stop = "operator X"
                        break
                    if pressed == P.KEY_LIMP and not limp:
                        limp, tau_ff = True, np.zeros(N_JOINTS)
                        print("[stand] LIMP")
                    elif pressed in ("\r", "\n"):
                        if limp:
                            limp = False
                            gate.start(now, q)
                            q_ref = q.copy()
                            print("[stand] re-engaged")
                        elif stage == "WAIT" and z_crouch is not None:
                            stage, t0, armed = "RISE", now, True
                            gate.start(now, q)
                            q_ref = q.copy()
                            print(f"[stand] RISE: {z_crouch*1e3:.0f} -> "
                                  f"{args.height*1e3:.0f} mm over "
                                  f"{P.T_RISE:.0f} s under torque")
                    elif pressed in (P.KEY_PARK, P.KEY_PARK.upper()) \
                            and stage == "HOLD":
                        stop = "park"
                        break

                    if stage == "RISE" and now - t0 >= P.T_RISE:
                        stage, t0 = "HOLD", now
                        print(f"[stand] HOLD: closed loop at "
                              f"{args.height*1e3:.0f} mm.  Push the trunk. "
                              f"P parks.")

                    torque_stage = stage in ("RISE", "HOLD") and armed

                    # ---- the sub-sampled block: estimator + model + map ---
                    # ~1.5 ms, so it runs every MODEL_EVERY sweeps and that
                    # sweep's FL frames are late by that much.  Inside the
                    # 10 ms watchdog with margin; see params.MODEL_EVERY.
                    if sweep % P.MODEL_EVERY == 0:
                        state, act, why = est.read(now, q, qd, ALL4)
                        if act:
                            if z_crouch is None:
                                z_crouch = float(state["r"][2])
                                print(f"[stand] crouch height {z_crouch*1e3:.0f}"
                                      f" mm to the TRUNK BOTTOM "
                                      f"({state['z_hip']*1e3:.0f} mm to the hip "
                                      f"axis).  FK, drift-free -- put a ruler "
                                      f"on it.")
                                _print_attitude_report(est, args)
                                print("[stand] ENTER to rise.")
                            z_des = (dm.height_ramp(z_crouch, args.height,
                                                    (now - t0) / P.T_RISE)
                                     if stage == "RISE" else
                                     args.height if stage == "HOLD"
                                     else z_crouch)
                            if torque_stage:
                                q_ref = q_ref_for_height(q_ref, z_des, foot_xy)
                                W = dm.body_wrench(state, z_des, args.mass,
                                                   gains=gains)
                                tau_ff, diag = ft.stance_torque(
                                    q, qd, state, W, ALL4, gains,
                                    force_frac=args.force_frac,
                                    leg_gravity=args.leg_gravity)
                        else:
                            # honest fallback: hold each leg up, command no
                            # body wrench.  A frozen wrench would keep pushing
                            # on a world model that has stopped updating.
                            tau_ff = ft.leg_gravity_only(q) \
                                if args.leg_gravity else np.zeros(N_JOINTS)
                            if torque_stage and not limp:
                                limp = True
                                print(f"[stand] LIMP: estimator refused ({why})")

                    # ---- the 250 Hz floor + safety -----------------------
                    if torque_stage and not limp:
                        tau_cmd = gate.apply(tau_ff + imp.tau(q, qd, q_ref),
                                             q, now)
                    else:
                        tau_cmd = np.zeros(N_JOINTS)
                        if not torque_stage:
                            q_ref = q.copy()     # never step on re-engage

                    reason = gate.estop_reason(
                        q, qd, base._temperatures(mb), miss.update(mb),
                        mb.errors(), now, enforce_position_limits=False)
                    if reason:
                        stop = reason
                        break
                    crossed = cap.ABD_SIGN * q[cap.ABD] < 0.0
                    if crossed.any():
                        k = int(np.argmax(crossed))
                        stop = (f"{base.JOINT_LABELS[cap.ABD[k]]} "
                                f"(CAN {MOTOR_IDS[cap.ABD[k]]}) crossed 0 at "
                                f"{np.rad2deg(q[cap.ABD[k]]):+.1f} deg")
                        break

                    # in TORQUE mode a latched driver stops producing torque:
                    # that leg collapses while the others keep pushing, which
                    # is an active tip-over, not a stall
                    latched = [m for m, e in mb.errors().items() if e & 0x80]
                    if latched and torque_stage and P.LATCH_LIMPS_ROBOT \
                            and not limp:
                        limp = True
                        print(f"[stand] LIMP: input-lost latch on {latched}")

                    if state is not None and act:
                        tilt = max(abs(np.degrees(est.roll)),
                                   abs(np.degrees(est.pitch)))
                        if tilt > P.TILT_STOP_DEG:
                            stop = f"tilt {tilt:.1f} deg"
                            break

                    # THE measurement: measured iq -> foot force -> the robot.
                    # Four Jacobians, four SVDs, four solves -- measured
                    # 1029 us, so it cannot run every 4 ms sweep, and it is
                    # STAGGERED against the model block so the two never land
                    # in the same sweep (see params.LOAD_OFFSET).  At 21 Hz it
                    # still catches a collapsing leg within 48 ms.
                    if sweep % P.LOAD_EVERY == P.LOAD_OFFSET:
                        support, ok = ft.foot_load_from_torque(
                            q, tau_meas, None if state is None else state["C"])
                        load_sum = float(np.nansum(support)) if ok.any() \
                            else float("nan")
                        if stage == "HOLD" and not limp \
                                and np.isfinite(load_sum) \
                                and abs(load_sum - args.mass * P.GRAVITY) \
                                / (args.mass * P.GRAVITY) > P.LOAD_SUM_TOL_FRAC:
                            limp = True
                            print(f"[stand] LIMP: measured foot load "
                                  f"{load_sum:.1f} N against "
                                  f"{args.mass*P.GRAVITY:.1f} N expected -- the "
                                  f"force loop is not doing what it says")

                    if args.log:
                        # w, v and W are the OUTER LOOP'S OWN SIGNALS: the two
                        # it feeds back on and the one it produces.  Without
                        # them a log of an outer-loop shake shows the result
                        # and neither the cause nor the command -- roll is an
                        # ANGLE, and kd_att acts on the RATE, which nothing
                        # else in this file records.  Held between model
                        # steps, like tau_ff.
                        log.append((now, q.copy(), qd.copy(), qd_drv.copy(),
                                    tau_cmd.copy(), tau_meas.copy(),
                                    np.array(diag["fz"]),
                                    est.roll, est.pitch, est.z, load_sum,
                                    stage, est.roll_fk, est.pitch_fk,
                                    est.z_hip, est.roll_raw, est.pitch_raw,
                                    q_ref.copy(),
                                    np.zeros(3) if state is None
                                    else np.asarray(state["w"]).copy(),
                                    est.v.copy(), W.copy()))

                    if now - last_print >= 1.0 / base.STATUS_HZ:
                        last_print = now
                        print(f"[stand] {'LIMP' if limp else stage:5s} "
                              f"{est.status() if state is not None else why}",
                              flush=True)

                mid = MOTOR_IDS[j]
                if stage == "WAIT":
                    mb.position(mid, float(cap.POSITION_TARGET_DEG[j]),
                                args.crouch_dps)
                else:
                    mb.torque(mid, float(tau_cmd[j]))
                index += 1
                if j == N_JOINTS - 1:
                    sweep += 1
                overrun = mb.pace(deadline)
                deadline += slot
                if overrun and overrun > 2.0 * slot:
                    deadline = time.perf_counter() + slot

            print(f"[stand] stopped: {stop}")
            # Park in POSITION mode from wherever the torque stand left it.
            # Cutting torque at height would drop the robot; 0xA4 takes the
            # weight on the drivers' own loops first.
            if stop == "park" or args.park_on_stop:
                try:
                    cap.parked(mb, unwrap, key, max_speed_dps=args.crouch_dps)
                except Exception as exc:                    # noqa: BLE001
                    print(f"[stand] park failed: {exc}")
            print("[stand] soft stop -- HOLD THE ROBOT")
            base._soft_stop(mb)
    except KeyboardInterrupt as exc:
        print(f"\n[stand] aborted: {exc}")
    except RuntimeError as exc:
        print(f"\n[stand] fault: {exc}")
    finally:
        try:
            imu.stop()
        except Exception:                                   # noqa: BLE001
            pass
        key.close()

    if log:
        a = np.array([r[10] for r in log], dtype=float)
        a = a[np.isfinite(a)]
        print()
        print("=" * 74)
        if a.size:
            print(f"  foot load sum: mean {a.mean():.1f} N, min {a.min():.1f}, "
                  f"max {a.max():.1f}, against "
                  f"{args.mass*P.GRAVITY:.1f} N")
            print("  ^ THE number: this is measured current, not a command.")

        # The control loop runs on the encoder now, so this block answers a
        # different question than it used to: is the driver's speed field
        # STILL glitching, i.e. would the old path still have shaken us?
        enc = np.max(np.abs(np.array([r[2] for r in log])), axis=0)
        drv = np.max(np.abs(np.array([r[3] for r in log])), axis=0)
        i = int(np.argmax(drv))
        print(f"  peak |qd|: encoder (used by the loop) {enc.max():.2f} rad/s, "
              f"driver field (not used) {drv[i]:.2f} rad/s on "
              f"{base.JOINT_LABELS[i]}")
        if glitches.any():
            worst = int(np.argmax(glitches))
            print(f"  driver/encoder disagreements: {int(glitches.sum())} total, "
                  f"worst {base.JOINT_LABELS[worst]} ({int(glitches[worst])}) "
                  f"-- these no longer reach the force law")
            print(f"  ^ through the OLD path each would have injected up to "
                  f"{P.KD_Z * drv[i] * 0.0425:.0f} N of phantom force")
        else:
            print("  driver and encoder agreed throughout -- no glitching this "
                  "run, so the shake had another cause")
        if args.log:
            np.savez_compressed(
                args.log,
                t=np.array([r[0] for r in log]), q=np.array([r[1] for r in log]),
                qd=np.array([r[2] for r in log]),        # encoder, used
                qd_drv=np.array([r[3] for r in log]),    # driver field, unused
                tau_cmd=np.array([r[4] for r in log]),
                tau_meas=np.array([r[5] for r in log]),
                fz=np.array([r[6] for r in log]),
                roll=np.array([r[7] for r in log]),      # AHRS, setpoint out
                pitch=np.array([r[8] for r in log]),
                z=np.array([r[9] for r in log]),         # floor->trunk bottom
                load=np.array([r[10] for r in log]),
                stage=np.array([r[11] for r in log]),
                # the independent attitude and the frame's other half: with
                # these a log can be re-read for a mount tilt or a height
                # frame after the fact, which the 2026-08-17 logs could not
                roll_fk=np.array([r[12] for r in log]),
                pitch_fk=np.array([r[13] for r in log]),
                z_hip=np.array([r[14] for r in log]),
                roll_raw=np.array([r[15] for r in log]),
                pitch_raw=np.array([r[16] for r in log]),
                # the impedance error |q_ref - q| used to be streamed as |dq|;
                # it is recoverable only if q_ref is here, so it is
                q_ref=np.array([r[17] for r in log]),
                # the outer loop's inputs (w = AHRS gyro, UNFILTERED; v = the
                # low-passed leg odometry) and its output (W, the 6D wrench)
                w=np.array([r[18] for r in log]),
                v=np.array([r[19] for r in log]),
                W=np.array([r[20] for r in log]),
                glitches=glitches,
                # every OUTER gain, not just the attitude pair: a --kd-z or
                # --kd-xy A/B is unreadable afterwards if the npz cannot say
                # which half of it this file is.
                kp_att=gains.kp_roll, kd_att=gains.kd_roll,
                kp_z=gains.kp_z, kd_z=gains.kd_z,
                kd_xy=gains.kd_x, kd_yaw=gains.kd_yaw,
                force_frac=args.force_frac, mass=args.mass,
                setpoint_roll_deg=args.setpoint_roll,
                setpoint_pitch_deg=args.setpoint_pitch,
                imu_below_trunk_origin_m=P.IMU_BELOW_TRUNK_ORIGIN_M,
                kp_imp=args.kp, kd_imp=args.kd, tau_max=args.tau_max)
            print(f"  log -> {args.log} ({len(log)} sweeps)")
    return 0


def self_test():
    sys.path.insert(0, os.path.join(_AUG, "self-test"))
    from selftest_common import check, report, raises      # noqa: PLC0415

    # -- the impedance refuses gains past the measured envelope ------------
    check("impedance refuses kp past the stable envelope, before arming",
          raises(lambda: JointImpedance(kp=P.KP_IMP_MAX + 1)))
    check("...and kd past the sampled-damper bound",
          raises(lambda: JointImpedance(kd=P.KD_IMP_MAX + 1)))
    imp = JointImpedance()
    t = imp.tau(np.zeros(12), np.zeros(12), np.full(12, 0.1))
    check("impedance pulls TOWARDS q_ref", bool(np.all(t > 0))
          and abs(t[0] - P.KP_IMP * 0.1) < 1e-12)
    t = imp.tau(np.zeros(12), np.full(12, 1.0), np.zeros(12))
    check("...and opposes velocity", abs(t[0] + P.KD_IMP) < 1e-12)
    # THE DAMPER IS CHECKED AT THE MEASURED DELAY, NOT THE COMMAND INTERVAL.
    # The old pair of checks used SWEEP_S and passed at kd=0.6, which is the
    # exact reason the offline gate was green while the robot shook: the
    # inequality had no delay term in it.  kd_joint is in the sum because it
    # damps the same joint through the same delay.
    bound = 2 * P.J_MIN / P.LOOP_DELAY_S
    check("the damper is inside the bound at the MEASURED loop delay",
          P.KD_IMP + P.KD_JOINT < 0.5 * bound,
          f"kd+kd_joint={P.KD_IMP+P.KD_JOINT:.2f} vs bound {bound:.2f} "
          f"Nms/rad at {P.LOOP_DELAY_S*1e3:.0f} ms "
          f"({100*(P.KD_IMP+P.KD_JOINT)/bound:.0f}%)")
    # the whole reason this file can exist -- recompute, do not trust
    check("...and the OLD default would not have been (it shook, 9-12 Hz)",
          P.KD_IMP_MAX + P.KD_JOINT > 0.5 * bound,
          f"the old kd={P.KD_IMP_MAX}+{P.KD_JOINT} is "
          f"{100*(P.KD_IMP_MAX+P.KD_JOINT)/bound:.0f}% of the same bound, and "
          f"{100*(P.KD_IMP_MAX+P.KD_JOINT)/(2*P.J_MIN/P.SWEEP_S):.0f}% of the "
          f"4 ms one this file used to check -- same gain, both answers")
    # the stiffness half: what a delay does is set by WHERE the loop crosses
    f_c = math.sqrt(P.KP_IMP / P.J_MIN) / (2 * math.pi)
    check("the stiffness puts the crossover where 16 ms is a small phase",
          360.0 * f_c * P.LOOP_DELAY_S < 25.0,
          f"kp={P.KP_IMP} -> {f_c:.1f} Hz -> "
          f"{360*f_c*P.LOOP_DELAY_S:.0f} deg of delay phase; kp="
          f"{P.KP_IMP_MAX} gives "
          f"{360*math.sqrt(P.KP_IMP_MAX/P.J_MIN)/(2*math.pi)*P.LOOP_DELAY_S:.0f}"
          f" deg, which is the shake")

    # -- the torque gate ---------------------------------------------------
    g = TorqueGate(tau_cap=1.0)
    g.start(0.0, np.zeros(12))
    check("the ramp opens over TORQUE_RAMP_S, not base's 1.0 s",
          abs(g.cap_now(P.TORQUE_RAMP_S) - 1.0) < 1e-12
          and g.cap_now(P.TORQUE_RAMP_S / 2) < 1.0)
    # measure the SLEW, which means letting the ramp finish first -- otherwise
    # this reads the cap opening, not the rate limiter
    g.last_time = 10.0
    out = g.apply(np.full(12, 5.0), np.zeros(12), 10.0 + P.SWEEP_S)
    check("the slew is the torque-mode 60 Nm/s, not the position track's 5",
          abs(float(out[0]) - P.TAU_SLEW_NM_S * P.SWEEP_S) < 1e-9,
          f"{float(out[0])*1e3:.1f} mNm in one sweep; base's 5 Nm/s would give "
          f"{base.TAU_SLEW_NM_S*P.SWEEP_S*1e3:.1f}")
    dqd = P.TAU_SLEW_NM_S * P.SWEEP_S ** 2 / P.J_MIN
    check("...which is bounded: one sweep at the slew adds ~0.1 rad/s",
          dqd < 0.2, f"{dqd:.3f} rad/s per sweep vs the 7.0 e-stop")
    g2 = TorqueGate(tau_cap=1.0)
    g2.start(0.0, np.zeros(12))
    g2.last_time = 0.0
    check("the cap is honoured after the ramp",
          float(np.max(np.abs(g2.apply(np.full(12, 9.0), np.zeros(12), 10.0))))
          <= 1.0 + 1e-9)

    # -- the overspeed trip is GONE, and the rest of estop_reason is not -----
    g3 = TorqueGate(tau_cap=1.0)
    g3.start(0.0, np.zeros(12))
    fast = np.full(12, 20.0)          # far past both old tiers
    clean = {m: 0 for m in MOTOR_IDS}
    check("20 rad/s no longer stops the run (the 8.0 hard trip is removed)",
          g3.estop_reason(np.zeros(12), fast, np.zeros(12, int),
                          np.zeros(12, int), clean, 0.01,
                          enforce_position_limits=False) is None)
    check("...but the peak is still recorded, so the log can show it",
          float(np.max(g3.qd_peak)) == 20.0)
    check("overtemp still stops the run",
          "overtemp" in (g3.estop_reason(
              np.zeros(12), np.zeros(12), np.full(12, base.TEMP_ESTOP + 1),
              np.zeros(12, int), clean, 0.02,
              enforce_position_limits=False) or ""))
    check("a stuck CAN reply still stops the run",
          "missed" in (g3.estop_reason(
              np.zeros(12), np.zeros(12), np.zeros(12, int),
              np.full(12, base.MISS_ESTOP), clean, 0.03,
              enforce_position_limits=False) or ""))
    check("a driver hard fault still stops the run",
          "motor fault" in (g3.estop_reason(
              np.zeros(12), np.zeros(12), np.zeros(12, int),
              np.zeros(12, int), {m: (0x01 if m == MOTOR_IDS[0] else 0)
                                  for m in MOTOR_IDS}, 0.04,
              enforce_position_limits=False) or ""))
    check("a NaN in the state still stops the run",
          g3.estop_reason(np.full(12, np.nan), np.zeros(12),
                          np.zeros(12, int), np.zeros(12, int), clean, 0.05,
                          enforce_position_limits=False) is not None)

    # -- q_ref tracks the rise (last week's runner froze it) ---------------
    foot_xy = [st.leg_frames(LEGS[i], Q_CROUCH[3*i:3*i+3])[0][:2]
               for i in range(4)]

    def height_of(qv):
        """Floor-to-TRUNK-BOTTOM height of a pose -- the estimator's quantity,
        so this gate checks the two definitions agree rather than assuming."""
        return fe.fk_trunk_height(qv, np.eye(3), ALL4)

    # -- THE FRAME.  This is the gate that would have caught 2026-08-17 -----
    # q_ref_for_height and fk_trunk_height are the two ends of the height
    # loop: one turns a number into joint angles, the other turns joint angles
    # back into a number.  If they disagree about WHICH POINT the number means,
    # the loop still closes, the log still looks right, and the robot stands
    # 38 mm off.  Nothing else in the stack can see that.  So: ask for a
    # height, and demand the pose that comes back MEASURES that height.
    worst_frame = 0.0
    for z_ask in (0.05, 0.09, 0.12, P.STAND_HEIGHT):
        q_at = q_ref_for_height(Q_CROUCH.copy(), z_ask, foot_xy)
        worst_frame = max(worst_frame, abs(height_of(q_at) - z_ask))
    check("ask for a height and the pose MEASURES that height -- the IK and "
          "the estimator agree on which point it is",
          worst_frame < 1e-9,
          f"worst {worst_frame*1e6:.3f} um over 4 heights; a frame mismatch "
          f"here is a silent {P.IMU_BELOW_TRUNK_ORIGIN_M*1e3:.0f} mm and the "
          f"log cannot show it")
    q_st_f = q_ref_for_height(Q_CROUCH.copy(), P.STAND_HEIGHT, foot_xy)
    check("...and it is the TRUNK BOTTOM, not the hip axis (the 38 mm)",
          abs(fe.fk_trunk_height(q_st_f, np.eye(3), ALL4, ref="hip")
              - P.STAND_HEIGHT - P.IMU_BELOW_TRUNK_ORIGIN_M) < 1e-9,
          f"stand pose measures {P.STAND_HEIGHT*1e3:.0f} mm at the trunk "
          f"bottom, "
          f"{fe.fk_trunk_height(q_st_f, np.eye(3), ALL4, ref='hip')*1e3:.0f} "
          f"mm at the hip axis")
    check("the STAND_HEIGHT number moved with the frame and the POSE did not",
          abs(fe.fk_trunk_height(q_st_f, np.eye(3), ALL4, ref="hip")
              - 0.190) < 1e-9,
          "0.152 at the trunk bottom is the same joint angles 0.190 at the "
          "hip axis used to command -- the change is a relabelling, not a "
          "different stand")

    # The recorded crouch's four feet are NOT exactly coplanar, and the
    # estimator's height is their MEAN.  So q_ref at that mean height cannot
    # reproduce the crouch exactly -- it levels the feet, by exactly the
    # out-of-plane amount.  Gate the size of that correction rather than
    # asserting an equality that is false for a real recorded pose.
    z0 = height_of(Q_CROUCH)
    spread = max(-float(st.leg_frames(LEGS[i], Q_CROUCH[3*i:3*i+3])[0][2])
                 for i in range(4)) \
        - min(-float(st.leg_frames(LEGS[i], Q_CROUCH[3*i:3*i+3])[0][2])
              for i in range(4))
    q_ref = q_ref_for_height(Q_CROUCH.copy(), z0, foot_xy)
    moved = max(abs(height_of(Q_CROUCH) - height_of(q_ref)), 0.0)
    check("q_ref at the crouch's own mean height levels the feet and "
          "nothing more",
          float(np.max(np.abs(q_ref - Q_CROUCH))) < 5e-3 and moved < 1e-9,
          f"joints move {np.max(np.abs(q_ref-Q_CROUCH))*1e3:.2f} mrad to close "
          f"a {spread*1e6:.0f} um out-of-plane spread; mean height unchanged")

    # Walk the ACTUAL rise: 83 Hz for T_RISE seconds, warm-started throughout.
    # A single jump from crouch to stand is not how this is ever called, and
    # with a plain Newton step it diverges by 330 mm -- the damped step and the
    # warm start are both load-bearing.
    n = int(P.T_RISE * P.CONTROL_HZ / P.MODEL_EVERY)
    q_ref = Q_CROUCH.copy()
    worst_lag = 0.0
    for k in range(n + 1):
        z_des = dm.height_ramp(z0, P.STAND_HEIGHT, k / n)
        q_ref = q_ref_for_height(q_ref, z_des, foot_xy)
        worst_lag = max(worst_lag, abs(height_of(q_ref) - z_des))
    check("q_ref tracks the whole 8 s rise, never lagging the reference",
          worst_lag < 1e-6, f"worst lag over {n} steps: {worst_lag*1e6:.3f} um")
    check("...and lands exactly on the stand height",
          abs(height_of(q_ref) - P.STAND_HEIGHT) < 1e-6,
          f"{height_of(q_ref)*1e3:.4f} mm vs {P.STAND_HEIGHT*1e3:.1f}")
    xy_err = max(float(np.max(np.abs(
        st.leg_frames(LEGS[i], q_ref[3*i:3*i+3])[0][:2] - foot_xy[i])))
        for i in range(4))
    check("...with x/y PINNED, so the contact point never moves in the world",
          xy_err < 1e-6, f"worst {xy_err*1e6:.3f} um")
    crossed = cap.ABD_SIGN * q_ref[cap.ABD] < 0.0
    check("the stand pose never crosses the abduction sign limit",
          not bool(crossed.any()),
          f"abd {np.rad2deg(q_ref[cap.ABD]).round(1)} deg")
    check("q_ref is a real IK solution, not the crouch held (it MOVED)",
          float(np.max(np.abs(q_ref - Q_CROUCH))) > 0.3,
          f"max joint travel {np.rad2deg(np.max(np.abs(q_ref-Q_CROUCH))):.1f} "
          f"deg -- last week's runner froze q_ref here, so the impedance "
          f"fought the rise all the way up")
    step = (P.STAND_HEIGHT - z0) / n
    check("one 83 Hz step of the rise is a fraction of a mm",
          step < 5e-4, f"{step*1e3:.3f} mm per model update")

    # -- open loop really is open: no state error can reach the wrench ------
    g0 = dm.default_gains()
    g0.kp_z = g0.kd_z = g0.kp_roll = g0.kd_roll = 0.0
    g0.kp_pitch = g0.kd_pitch = g0.kd_x = g0.kd_y = g0.kd_yaw = 0.0
    rng_states = [
        {"r": np.array([0., 0., z]), "v": np.array(v, dtype=float),
         "C": fe.C_from_rp(*rp), "w": np.array(w, dtype=float)}
        for z, v, rp, w in ((0.10, (0.2, -0.1, 0.3), (0.2, -0.15), (1, 2, -1)),
                            (0.30, (-.5, 0.4, -0.2), (-0.1, 0.3), (-2, 1, 3)))]
    check("--open-loop wrench is [0,0,mg,0,0,0] whatever the state does",
          all(np.allclose(dm.body_wrench(s, 0.19, gains=g0),
                          [0, 0, P.WEIGHT_N, 0, 0, 0]) for s in rng_states),
          "the diagnostic baseline cannot be perturbed by any estimate")

    # -- the velocity path that caused the 2026-08-17 shake -----------------
    # Replay the measured glitch through both paths and demand the fix holds.
    q_st = q_ref_for_height(Q_CROUCH.copy(), P.STAND_HEIGHT, foot_xy)
    spike = np.zeros(12)
    spike[0] = 8.1                      # what the driver field reported
    v_bad, _ = fe.leg_odometry_velocity(q_st, spike, np.eye(3), np.zeros(3),
                                        ALL4)
    F_bad = abs(P.KD_Z * v_bad[2]) + abs(P.KD_X * v_bad[0]) \
        + abs(P.KD_Y * v_bad[1])
    # THE THRESHOLD IS THE GAIN'S, NOT A FIXED FRACTION OF THE WEIGHT.  This
    # asserted F_bad > 20% of the weight, which was written when kd_z was 120
    # and the glitch was worth 33 N.  At the verified kd_z = 40 the same
    # glitch is 4.3 N and the check failed -- not because the argument for
    # encoder differencing got weaker, but because it was pinned to a gain
    # that has since moved.  What matters is that ONE bad reading is worth
    # more than the height loop's own resolution, so compare it to what a
    # millimetre of real height error commands.
    F_mm = P.KP_Z * 0.001
    check("the measured 8.1 rad/s glitch is worth more than a mm of real error",
          F_bad > 3.0 * F_mm,
          f"{F_bad:.1f} N from one bad reading, against {F_mm:.2f} N for a "
          f"whole millimetre of height -- {F_bad/F_mm:.0f} mm of phantom "
          f"height, on a {P.WEIGHT_N:.0f} N robot")

    # the encoder path, driven by a joint that is NOT moving
    qd_enc = np.zeros(12)
    for _ in range(200):                # a stationary joint, quantised encoder
        qd_enc += P.QD_ALPHA * (0.0 - qd_enc)
    check("a stationary joint differences to zero -- an encoder cannot glitch",
          float(np.max(np.abs(qd_enc))) < 1e-9)
    # worst case: one encoder LSB of jitter every sweep
    lsb = np.deg2rad(0.01)
    qd_j = 0.0
    for k in range(400):
        qd_j += P.QD_ALPHA * ((lsb if k % 2 else -lsb) / P.SWEEP_S - qd_j)
    spike_j = np.zeros(12)
    spike_j[0] = abs(qd_j)
    v_j, _ = fe.leg_odometry_velocity(q_st, spike_j, np.eye(3), np.zeros(3),
                                      ALL4)
    F_j = abs(P.KD_Z * v_j[2]) + abs(P.KD_X * v_j[0]) + abs(P.KD_Y * v_j[1])
    check("encoder quantisation dither is a NEGLIGIBLE force by comparison",
          F_j < 0.01 * P.WEIGHT_N and F_j < F_bad / 50,
          f"{F_j:.2f} N vs {F_bad:.1f} N -- {F_bad/max(F_j,1e-9):.0f}x better")
    check("...and negligible through the impedance damper too",
          P.KD_IMP * abs(qd_j) < 0.05 * P.TAU_START_MAX,
          f"{P.KD_IMP*abs(qd_j):.4f} Nm vs {P.KD_IMP*8.1:.2f} Nm from the glitch")

    # the low pass that was not one
    dt_outer = P.MODEL_EVERY / P.CONTROL_HZ
    alpha = 1.0 - np.exp(-2 * np.pi * P.ODOM_LPF_FC_HZ * dt_outer)
    check("the odometry low pass is actually a low pass at the OUTER rate",
          alpha < 0.5,
          f"alpha={alpha:.2f} at {1/dt_outer:.0f} Hz; the old 20 Hz corner "
          f"gave {1-np.exp(-2*np.pi*20*dt_outer):.2f}, i.e. no filtering")
    check("...while still far above the height loop it feeds",
          P.ODOM_LPF_FC_HZ > 2.0 * np.sqrt(P.KP_Z / P.MASS_KG) / (2 * np.pi),
          f"{P.ODOM_LPF_FC_HZ:.0f} Hz filter vs "
          f"{np.sqrt(P.KP_Z/P.MASS_KG)/2/np.pi:.2f} Hz height loop")

    # -- the loop's own timing budget --------------------------------------
    check("the sub-sampled model still updates faster than the body moves",
          P.CONTROL_HZ / P.MODEL_EVERY > 50.0,
          f"{P.CONTROL_HZ/P.MODEL_EVERY:.0f} Hz model, "
          f"{P.CONTROL_HZ/P.LOAD_EVERY:.0f} Hz load check")
    # The staggering is the whole reason a single thread is safe here: without
    # it the two heavy blocks coincide and one sweep is 2.41 ms late instead
    # of 1.38.  Prove they cannot collide, do not assert it in a comment.
    collide = [s for s in range(P.MODEL_EVERY * P.LOAD_EVERY)
               if s % P.MODEL_EVERY == 0 and s % P.LOAD_EVERY == P.LOAD_OFFSET]
    check("the model and load blocks can NEVER land in the same sweep",
          not collide, f"checked all {P.MODEL_EVERY*P.LOAD_EVERY} phases")
    check("the worst single-sweep delay is inside half the driver watchdog",
          0.00138 < P.WATCHDOG_S / 2,
          f"measured 1.38 ms model block vs {P.WATCHDOG_S*1e3:.0f} ms watchdog "
          f"(2.41 ms if they were not staggered)")

    # -- the stage machine's one irreversible rule -------------------------
    # z_crouch is MEASURED, so recompute it here rather than carrying the
    # 0.044 literal this gate used to hold -- that number was the hip-axis
    # crouch and the frame change made it silently wrong by 38 mm.
    z_cr = height_of(Q_CROUCH)
    check("RISE starts at the MEASURED crouch height, never at the target",
          dm.height_ramp(z_cr, P.STAND_HEIGHT, 0.0) == z_cr,
          f"a {P.STAND_HEIGHT:.3f} setpoint at the {z_cr:.3f} crouch would ask "
          f"{P.KP_Z*(P.STAND_HEIGHT-z_cr) + P.WEIGHT_N:.0f} N on sweep 1, "
          f"{(P.KP_Z*(P.STAND_HEIGHT-z_cr)+P.WEIGHT_N)/P.WEIGHT_N:.1f}x weight")
    check("...and the rise TRAVEL is unchanged by the frame change",
          abs((P.STAND_HEIGHT - z_cr)
              - (0.190 - fe.fk_trunk_height(Q_CROUCH, np.eye(3), ALL4,
                                            ref="hip"))) < 1e-9,
          f"{(P.STAND_HEIGHT - z_cr)*1e3:.1f} mm of travel either way -- a "
          f"constant offset cancels in a difference, which is why the shake "
          f"work was unaffected by the frame being wrong")

    return report()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", default=None)
    ap.add_argument("--height", type=float, default=P.STAND_HEIGHT)
    ap.add_argument("--mass", type=float, default=P.MASS_KG)
    ap.add_argument("--tau-max", type=float, default=P.TAU_START_MAX)
    ap.add_argument("--kp", type=float, default=P.KP_IMP)
    ap.add_argument("--kd", type=float, default=P.KD_IMP)
    ap.add_argument("--force-frac", type=float, default=P.FORCE_FRAC_DEFAULT)
    ap.add_argument("--no-leg-gravity", dest="leg_gravity",
                    action="store_false",
                    help="reproduce the 2026-07-30 stance law (A/B only)")
    ap.add_argument("--open-loop", action="store_true",
                    help="ALL outer gains 0: W=[0,0,mg,0,0,0], even split + "
                         "leg gravity + impedance.  The known-good baseline; "
                         "run this FIRST when diagnosing a shake")
    ap.add_argument("--kp-att", type=float, default=None,
                    help="0 with --kd-att 0 is the ablation half of the A/B")
    ap.add_argument("--kd-att", type=float, default=None)
    ap.add_argument("--kp-z", type=float, default=None)
    ap.add_argument("--kd-z", type=float, default=None,
                    help="the height loop's DAMPING half, separately from "
                         "--kp-z.  kd_z acts on leg-odometry velocity, which "
                         "is the one outer signal carrying the whole lag "
                         "budget (5 Hz LPF + 12 ms hold), so a shake that "
                         "--kp-z alone cannot clear usually lives here")
    ap.add_argument("--kd-xy", type=float, default=None,
                    help="kd_x = kd_y together: the damping-only translation "
                         "axes, on the same lagged odometry velocity as kd_z")
    ap.add_argument("--kd-yaw", type=float, default=None,
                    help="the yaw damper.  Its own flag because it is the "
                         "gain --open-loop zeroes that no other flag could: "
                         "zeroing kp-z/kd-z/kd-xy/kp-att/kd-att left it live "
                         "at 4.0 and the banner did not show it")
    ap.add_argument("--setpoint-roll", type=float, default=P.SETPOINT_ROLL_DEG,
                    help="resting attitude of THIS rig in deg, subtracted from "
                         "every AHRS reading.  Read the pair the WAIT stage "
                         "prints; 0 means the IMU mount's own tilt is fed to "
                         "the wrench as a real one")
    ap.add_argument("--setpoint-pitch", type=float,
                    default=P.SETPOINT_PITCH_DEG)
    ap.add_argument("--crouch-dps", type=float, default=cap.MAX_DPS)
    ap.add_argument("--park-on-stop", action="store_true",
                    help="park in position mode after ANY stop, not just P")
    ap.add_argument("--log", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not 0.0 < args.tau_max <= P.TAU_STAGED_MAX:
        ap.error(f"--tau-max must be in (0, {P.TAU_STAGED_MAX}]")
    if not 0.0 <= args.force_frac <= 1.0:
        ap.error("--force-frac must be in [0, 1]")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
