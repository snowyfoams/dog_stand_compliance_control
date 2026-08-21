#!/usr/bin/env python3
"""The runner: trot in place on a convex-MPC plan.  One CAN thread, one solver.

    CROUCH (0xA4) -> WAIT -> RISE (week-2 wrench) -> HOLD (MPC) -> TROT (MPC)
                                                                 -> PARK (0xA4)

THIS FILE IS dog5_trot/trot_hw.py WITH ONE LAYER REPLACED
    CROUCH, WAIT, RISE, the joint impedance, the torque gate, the e-stop set,
    the estimator, the load watch and PARK are that file, unchanged.  A stand
    that already works is not the place to try a new controller, and rewriting
    the part that works is how the last attempt ended up unable to rise at all.

    What is new is the answer to ONE question -- "how hard must each foot push
    right now?" -- and where that answer comes from:

        week 2   a virtual spring on the trunk (Dynamic_Model.body_wrench) split
                 across the planted feet by a minimum-norm grasp map
                 (force_totorque.distribute).  Instantaneous, and quasi-static:
                 it knows nothing about a foot that is about to lift.
        here     a convex MPC over a 0.4 s horizon whose CONSTRAINTS are the
                 contact schedule, so the load leaves a foot BEFORE the gait
                 lifts it rather than after.

    Everything downstream of that answer is week 2's too: -J^T f plus the leg's
    own gravity (dog5_statics.stance_torque's law), the 250 Hz joint impedance
    under it, and the ramp/cap/slew gate over it.

WHY THE RISE STAYS ON THE WRENCH, AND THE MPC STILL SOLVES THROUGH IT
    The rise is the one stage where the robot is closest to falling and closest
    to the operator, and week 2's is the version that has been up and down on
    this robot.  So it is not touched.

    But the MPC is ARMED from the first sweep of RISE, solving against the same
    ramping height reference and publishing plans that are simply not applied.
    That costs nothing -- the solver is on its own thread -- and it buys two
    things: the plan is already fresh and warm-started at the instant HOLD
    hands over to it, and the log carries what the MPC WOULD have commanded
    beside what the wrench did, over the exact same rise.  That is an A/B
    nobody has to stage.

WHAT MAKES THE HANDOVER SAFE, RATHER THAN HOPEFUL
    At the top of the rise both laws are asked for the same thing -- hold this
    height, level -- and both answer with about the weight, so the step at
    handover is small.  What BOUNDS it is not that argument, though: it is the
    gate's 60 Nm/s slew, which is already the thing standing between every
    other stage change and the joints.  --wrench-hold keeps week 2 in HOLD as
    well, so "does the MPC change anything" is one flag, and the two logs are
    otherwise identical.

WHAT A STALE PLAN MEANS
    In position mode a stale estimate leaves the robot standing, because the
    drivers hold their last target.  In TORQUE mode a frozen plan keeps pushing
    on a world model that has stopped updating -- and this plan carries a
    CONTACT SCHEDULE, so a frozen one keeps pushing with feet that have since
    left the ground.  Past mpc_config.MPC_STALE_S with no publish, this runner
    LIMPS.  The solver's own refusals (no estimator state, factorisation
    failure) publish zero force rather than a fiction, so they arrive here as
    the same thing.

THE FOUR NUMBERS TO WATCH, AND WHERE THEY ARE
    the banner       the MPC's EFFECTIVE GAINS, beside week 2's verified pairs.
                     A cost weight is not a gain; this is the form in which the
                     two can be compared, and it is printed before the bus is
                     even opened.
    the status line  height, tilt, the IMU-free cross-check on tilt, trunk
                     velocity, planted count -- and in TROT, how many feet are
                     in the air and how old the plan is.
    the exit report  the foot-load sum (measured iq through J^-T, which must
                     come to ~57 N and is the only end-to-end proof the force
                     loop is real) and the SOLVE-TIME CENSUS.
                     CONTROL_ROADMAP calls solve time this phase's main risk,
                     and NO NUMBER FOR IT IS QUOTED ANYWHERE IN THIS PACKAGE
                     for the threaded case, because on a general-purpose box it
                     does not reproduce -- the same solves came back 5.9 ms
                     mean in one run and 20.5 ms with a 95 ms tail in the next.
                     Read it off this report, on the Pi, with chrt.  If it
                     shows solves over budget, the plan ages past
                     MPC_STALE_S and the runner LIMPS; see mpc_worker for what
                     to try, in order.
    the --log npz    every one of the above at the full 250 Hz, plus the plan
                     itself, the contact weights it was solved for, and which
                     feet the loop believed were down.

RUN -- supported robot, hand on SPACE
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    cd /home/robot01/Documents/can_motor_control/dog_stand_compliance_control/Aug_trotinplace_closedloop

    # OPENBLAS_NUM_THREADS=1: the solver thread must not fight the 250 Hz CAN
    # loop for cores.  See mpc_config.N_HORIZON for the measurement behind it.
    E="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1"

    # first run: low cap, MPC in HOLD only, no trot.  Read the banner's gain
    # table and the exit report's solve-time census before going further.
    sudo HOME=$HOME $E chrt -f 50 $V srb_mpc_hw/mpc_trot_hw.py --tau-max 1.0

    # the ablation: week 2's wrench in HOLD too, everything else identical.
    sudo HOME=$HOME $E chrt -f 50 $V srb_mpc_hw/mpc_trot_hw.py \\
        --wrench-hold --log wrench.npz
    sudo HOME=$HOME $E chrt -f 50 $V srb_mpc_hw/mpc_trot_hw.py --log mpc.npz

    # trot, once HOLD has been stood and pushed.  T from HOLD lifts feet.
    sudo HOME=$HOME $E chrt -f 50 $V srb_mpc_hw/mpc_trot_hw.py \\
        --tau-max 3.0 --log trot.npz

    # the horizontal-damping A/B, the one gain in this stack no run has tested
    sudo HOME=$HOME $E chrt -f 50 $V srb_mpc_hw/mpc_trot_hw.py --w-vel 0

Keys: ENTER = rise / re-engage, T = trot, SPACE = LIMP, P = park, X = stop.
      W/S = forward/back, A/D = left/right, Q/E = yaw, Z = zero the command.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.setswitchinterval(0.0005)

_HERE = os.path.dirname(os.path.abspath(__file__))          # srb_mpc_hw/
_AUG = os.path.dirname(_HERE)                               # Aug_trotinplace_...
_WEEK2 = os.path.join(_AUG, "august_week2")
_ROOT = os.path.dirname(_AUG)
_REPO = os.path.dirname(_ROOT)
# _HERE IS DELIBERATELY NOT ON THE PATH.  The repo's top-level config.py is
# what motorbus.py imports, and any directory of ours in front of it breaks the
# CAN layer with `module 'config' has no attribute 'encoder_gain'`.  This
# package is reached AS A PACKAGE from _AUG, which is why its own constants
# file is called mpc_config and not config.
for _p in (_WEEK2, _AUG, os.path.join(_AUG, "torque_mode_control"),
           os.path.join(_ROOT, "dog5_description"), _REPO,
           os.path.join(_REPO, "IMU_sensor")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _dup in ("", ".", _HERE):
    while _dup in sys.path:
        sys.path.remove(_dup)


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
import params as P                                         # noqa: E402
import crouch_and_park as cap                              # noqa: E402
import dog5_statics as st                                  # noqa: E402
import feedback_estimator as fe                            # noqa: E402
import Dynamic_Model as dm                                 # noqa: E402
import force_totorque as ft                                # noqa: E402

# THE IMPEDANCE AND THE GATE ARE IMPORTED, NOT COPIED.  Both are safety code
# whose constants are argued for in params.py against MEASURED hardware
# behaviour -- the 16 ms loop delay for the impedance bound, the 60 Nm/s slew
# for the joint-speed budget.  A second copy here would be a second thing to
# keep in step with that argument, and dog5_trot/trot_hw.py is where they
# already live.
from dog5_trot.trot_hw import JointImpedance, TorqueGate    # noqa: E402
from dog5_trot.trot_hw import q_ref_for_height              # noqa: E402
from dog5_trot.trot_hw import _zero_rp_streamer             # noqa: E402
from dog5_trot.trot_hw import _print_attitude_report        # noqa: E402

from srb_mpc_hw import mpc_config as C                      # noqa: E402
from srb_mpc_hw import mpc_controller as ctl                # noqa: E402
from srb_mpc_hw import mpc_gait                             # noqa: E402
from srb_mpc_hw import mpc_worker                           # noqa: E402
from srb_mpc_hw.convex_mpc import ConvexMPC                 # noqa: E402

MOTOR_IDS = base.MOTOR_IDS
N_JOINTS = P.N_JOINTS
LEGS = ft.LEGS
Q_CROUCH = cap.Q_CROUCH
ALL4 = np.ones(4, dtype=bool)


# ===========================================================================
# the banner
# ===========================================================================
def _print_gain_table(mpc):
    """The MPC's cost weights, as GAINS, beside the pairs week 2 verified.

    Printed before the bus is opened, because it is the one check an operator
    can make on a weight edit without touching the robot: a cost weight is not
    a gain, but the closed loop has one, and this is the form in which the two
    controllers can be compared at all.  ~50 ms of solving, no hardware.
    """
    g = ctl.effective_gains(mpc)
    print("  effective gains (unit error in, wrench out, four feet, nominal "
          "stance):")
    print(f"      support at zero error   {g['support_N']:7.2f} N   "
          f"(the robot weighs {C.WEIGHT:.2f} N)")
    rows = (("kp_roll ", g["kp_roll"], P.KP_ROLL, "Nm/rad ", "s3c_att20.npz"),
            ("kp_pitch", g["kp_pitch"], P.KP_PITCH, "Nm/rad ", "s3c_att20.npz"),
            ("kp_z    ", g["kp_z"], P.KP_Z, "N/m    ", "s2_kpz300.npz"),
            ("kd_roll ", g["kd_roll"], P.KD_ROLL, "Nms/rad", "s3c_att20.npz"),
            ("kd_pitch", g["kd_pitch"], P.KD_PITCH, "Nms/rad", "s3c_att20.npz"),
            ("kd_z    ", g["kd_z"], P.KD_Z, "Ns/m   ", "s1_kdz40.npz"),
            ("kd_x    ", g["kd_x"], P.KD_X, "Ns/m   ", "never run above 0"),
            ("kp_x    ", g["kp_x"], 0.0, "N/m    ", "no absolute x exists"))
    print(f"      {'axis':9s} {'this MPC':>10s}  {'week 2':>8s}   units    "
          f"week 2's log")
    for name, mine, theirs, unit, note in rows:
        print(f"      {name}  {mine:10.2f}  {theirs:8.2f}   {unit}  {note}")
    print("      the dampings are HIGHER by design: an MPC prices a rate error")
    print("      as the position error it becomes one horizon later.  See")
    print("      mpc_config -- roll runs 1.04x critical, height 1.22x.")
    return g


# ===========================================================================
# the run
# ===========================================================================
def run(args):
    base.validate_hardware_config()
    gains = dm.default_gains()      # week 2's wrench: RISE, and --wrench-hold

    mpc = ConvexMPC(
        n_horizon=args.n_horizon, dt=C.GAIT_PERIOD / args.n_horizon,
        w_att=C.W_ATT * args.w_att, w_pos=C.W_POS * np.array([1.0, 1.0, args.w_z]),
        w_omega=C.W_OMEGA, w_vel=C.W_VEL * np.array([args.w_vel, args.w_vel, 1.0]),
        iters=args.qp_iters)

    print("=" * 74)
    print("DOG5 SRB CONVEX-MPC TROT IN PLACE  (the DOG4.6 srb_mpc sim, on hardware)")
    print(f"  mass {C.MASS:.4f} kg = {C.WEIGHT:.1f} N, stand height "
          f"{args.height*1e3:.0f} mm, rise {P.T_RISE:.0f} s")
    print(f"  height is FLOOR to TRUNK BOTTOM (the IMU board) -- a ruler "
          f"reaches it.  That")
    print(f"  is {fe.hip_from_imu(args.height)*1e3:.0f} mm at the hip axis, "
          f"{P.IMU_BELOW_TRUNK_ORIGIN_M*1e3:.0f} mm higher, which is the frame "
          f"the leg IK uses.")
    if args.setpoint_roll or args.setpoint_pitch:
        print(f"  AHRS setpoint {args.setpoint_roll:+.2f} / "
              f"{args.setpoint_pitch:+.2f} deg subtracted from every reading")
    else:
        print(f"  AHRS setpoint 0.00 / 0.00 -- the IMU MOUNT'S OWN TILT is "
              f"being read as a real")
        print(f"  attitude.  The crouch prints rp:d; pass it back as "
              f"--setpoint-roll/--setpoint-pitch.")
    print("-" * 74)
    print(f"  horizon {mpc.N} knots x {mpc.dt*1e3:.0f} ms = "
          f"{mpc.N*mpc.dt*1e3:.0f} ms = {mpc.N*mpc.dt/C.GAIT_PERIOD:.2f} gait "
          f"cycles;  solver {args.mpc_hz:.0f} Hz off-thread")
    print(f"  QP: {12*mpc.N} variables, {20*mpc.N} constraint rows, ADMM "
          f"budget {mpc.iters} iterations at rho {mpc.rho}")
    print(f"  friction mu {C.MU} (pyramid inscribed at {C.MU_AXIS:.3f}), "
          f"fz in [{C.FZ_MIN:.0f}, {C.FZ_MAX:.0f}] N per foot")
    _print_gain_table(mpc)
    print("-" * 74)
    print(f"  impedance kp={args.kp} kd={args.kd} at {P.CONTROL_HZ:.0f} Hz, "
          f"model block at {P.CONTROL_HZ/P.MODEL_EVERY:.0f} Hz, "
          f"tau cap {args.tau_max} Nm, force_frac {args.force_frac}")
    print(f"  trot: period {args.period:.2f} s duty {args.duty:.2f} "
          f"(stance {args.period*args.duty*1e3:.0f} ms, swing "
          f"{args.period*(1-args.duty)*1e3:.0f} ms), swing height "
          f"{args.swing_height*1e3:.0f} mm")
    # A multi-line expression INSIDE an f-string is 3.12+ only, and the venv
    # this ships to is not this box's interpreter.  Built outside, then printed.
    promo = ("ON (--promote)" if args.promote else
             "OFF -- the clock alone, as dog5_trot/trot_hw trots today")
    print(f"  measured-contact promotion: {promo}.")
    print("  The detector runs and its fz is logged either way; see "
          "mpc_gait for what")
    print("  it reads off a foot in the AIR, and the one measurement that "
          "earns --promote.")
    # The roll axis saturates first, and on this rig it saturates EARLY.
    Mx_max, My_max = ft.moment_capacity(
        [st.leg_frames(LEGS[i], Q_CROUCH[3*i:3*i+3])[0] for i in range(4)],
        list(LEGS), C.WEIGHT)
    print(f"  stance moment capacity: roll {Mx_max:.1f} Nm, pitch "
          f"{My_max:.1f} Nm.  The MPC's cone constraint is what stops it "
          f"asking past these;")
    print(f"  week 2's grasp map had to RESCALE instead (measured 122 N "
          f"commanded on a 57 N robot at 10 deg of roll).")
    if args.wrench_hold:
        print("  --wrench-hold: HOLD runs WEEK 2's wrench.  The MPC still "
              "solves and is still")
        print("  logged; it is simply not applied.  T IS REFUSED in this "
              "mode -- the trot's swing")
        print("  and contact layers are the MPC controller's, so a trot here "
              "would not be the")
        print("  ablation of anything.  This flag answers one question: does "
              "the MPC hold better?")
    if args.force_frac < 0.2:
        print(f"  WARNING: force_frac {args.force_frac} removes the term that "
              f"holds the TRUNK up.  Feet on the floor, the legs fold until "
              f"kp*dq carries {C.WEIGHT:.0f} N.")
    print("  Support the robot.  ENTER = rise, T = trot, SPACE = LIMP, "
          "P = park, X = stop")
    print("-" * 74)
    print("  what the status line means:")
    print(fe.BodyState.status_legend())
    print("  plan  age of the MPC plan being applied (ms) / its total "
          "commanded support (N)")
    print("=" * 74)

    from imu_dog import ImuDog, DEFAULT_PORT                # noqa: PLC0415
    imu = ImuDog(port=args.port or DEFAULT_PORT)
    imp = JointImpedance(args.kp, args.kd)
    key = base.KeyPoller()
    log = []
    stop = None
    shared = None
    worker = None

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

            # min_planted = 2, AND THAT IS THE ONE ESTIMATOR CHANGE A TROT
            # NEEDS.  params.MIN_PLANTED = 3 is a STAND's alarm: a stand that
            # has lifted a foot has already gone wrong.  A trot lifting two is
            # the trot working, and the leg odometry v = -C^T(omega x s_i +
            # J_i qd_i) is solved by ONE planted foot; more only average.
            # What is lost: fk_attitude needs 3+ feet for a plane and returns
            # NaN on two, so the IMU-free attitude cross-check is unavailable
            # DURING the trot.  It still reads in the crouch and in HOLD.
            est = fe.BodyState(imu,
                               setpoint_roll=np.radians(args.setpoint_roll),
                               setpoint_pitch=np.radians(args.setpoint_pitch),
                               min_planted=2)
            gate = TorqueGate(tau_cap=args.tau_max)
            miss = base.CanMissMonitor(mb)
            foot_xy = np.array([st.leg_frames(LEGS[i], Q_CROUCH[3*i:3*i+3])[0][:2]
                                for i in range(4)])

            stand_gait = mpc_gait.StandGait(args.period, args.duty)
            trot_gait = mpc_gait.ContactAwareGait(
                mpc_gait.TrotGait(args.period, args.duty, C.PHASE_OFFSET),
                enabled=args.promote)
            controller = ctl.MpcController(stand_gait,
                                           force_frac=args.force_frac)
            controller.swing.step_height = args.swing_height
            shared = mpc_worker.MpcShared(q)
            worker = mpc_worker.start(shared, mpc, control_hz=args.mpc_hz)

            planted = ALL4
            stage, t0 = "WAIT", time.perf_counter()
            z_crouch = None
            z_des = None
            q_ref = q.copy()
            tau_ff = np.zeros(N_JOINTS)
            W = np.zeros(6)
            tau_cmd = np.zeros(N_JOINTS)
            f_now = np.zeros((4, 3))
            w_now = np.ones(4)
            kine = None
            state, act, why = None, False, "starting"
            limp = False
            armed = False
            handed_over = False
            load_sum = float("nan")
            plan_age = 0.0

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
            print("[mpc] WAIT: hold still, then ENTER to rise under torque")

            while stop is None:
                mb.poll()
                j = index % N_JOINTS
                if j == 0:
                    now = time.perf_counter()
                    q, qd_drv = base._joint_state(mb, unwrap)
                    tau_meas = np.array([mb.torques_nm()[m] for m in MOTOR_IDS])

                    # ENCODER-DIFFERENCED velocity, low-passed.  The driver's
                    # own speed field is read only to COUNT how often it
                    # disagrees; on 2026-08-17 it reported 8.1 rad/s on a joint
                    # whose encoder had moved 0.31, and through kd_z that is
                    # 24 N of phantom force on a 57 N robot.
                    if q_prev is not None:
                        dt_v = now - t_prev
                        if 1e-4 < dt_v < 0.05:
                            qd += P.QD_ALPHA * ((q - q_prev) / dt_v - qd)
                    q_prev, t_prev = q.copy(), now
                    glitches += (np.abs(qd_drv - qd)
                                 > np.maximum(2.0 * np.abs(qd), 1.0))

                    # ---- keys ----------------------------------------------
                    pressed = key.get()
                    if pressed in (P.KEY_STOP, P.KEY_STOP.upper()):
                        stop = "operator X"
                        break
                    if pressed == P.KEY_LIMP and not limp:
                        limp, tau_ff = True, np.zeros(N_JOINTS)
                        print("[mpc] LIMP")
                    elif pressed in ("\r", "\n"):
                        if limp:
                            limp = False
                            gate.start(now, q)
                            q_ref = q.copy()
                            controller.q_ref = q.copy()
                            print("[mpc] re-engaged")
                        elif stage == "WAIT" and z_crouch is not None:
                            stage, t0, armed = "RISE", now, True
                            gate.start(now, q)
                            q_ref = q.copy()
                            # ARM THE SOLVER NOW, not at the handover: it then
                            # reaches HOLD warm-started and already tracking
                            # the same ramp, and the rise's log carries what
                            # the MPC would have commanded beside what the
                            # wrench did.  Its output is not applied yet.
                            shared.armed = True
                            print(f"[mpc] RISE: {z_crouch*1e3:.0f} -> "
                                  f"{args.height*1e3:.0f} mm over "
                                  f"{P.T_RISE:.0f} s on WEEK 2's wrench "
                                  f"(the MPC solves alongside)")
                    elif pressed in (P.KEY_PARK, P.KEY_PARK.upper()) \
                            and stage == "HOLD":
                        stop = "park"
                        break
                    elif pressed in ("t", "T") and stage == "HOLD" \
                            and not limp and handed_over:
                        stage, t0 = "TROT", now
                        controller.set_gait(trot_gait, now, state, kine)
                        print(f"[mpc] TROT: {trot_gait!r}.  Feet are leaving "
                              f"the ground.  SPACE limps, P parks.")
                    elif pressed and pressed in "wsadqezWSADQEZ":
                        v = controller.v_cmd.copy()
                        wz = controller.wz_cmd
                        step, wstep = 0.01, 0.05
                        k = pressed.lower()
                        if k == "w":
                            v[0] += step
                        elif k == "s":
                            v[0] -= step
                        elif k == "a":
                            v[1] += step
                        elif k == "d":
                            v[1] -= step
                        elif k == "q":
                            wz += wstep
                        elif k == "e":
                            wz -= wstep
                        else:
                            v[:] = 0.0
                            wz = 0.0
                        controller.command(v, wz)
                        print(f"[mpc] cmd v {controller.v_cmd[0]*1e3:+.0f},"
                              f"{controller.v_cmd[1]*1e3:+.0f} mm/s  "
                              f"wz {controller.wz_cmd:+.2f} rad/s  "
                              f"(cap {C.V_CMD_MAX*1e3:.0f} mm/s -- it is the "
                              f"leg reach, not a preference)")

                    # ---- stage machine -------------------------------------
                    if stage == "RISE" and now - t0 >= P.T_RISE:
                        stage, t0 = "HOLD", now
                        if state is not None and kine is not None \
                                and not args.wrench_hold:
                            controller.start(now, q, state, kine, foot_xy)
                            q_ref = controller.q_ref
                            handed_over = True
                            print(f"[mpc] HOLD: the MPC now holds the robot up "
                                  f"at {args.height*1e3:.0f} mm.  Push the "
                                  f"trunk, then T to trot.  P parks.")
                        else:
                            why_wrench = ("--wrench-hold" if args.wrench_hold
                                          else "no estimator state")
                            print(f"[mpc] HOLD: week 2's wrench "
                                  f"({why_wrench}).  T is refused; P parks.")

                    torque_stage = stage in ("RISE", "HOLD", "TROT") and armed
                    use_mpc = handed_over and stage in ("HOLD", "TROT")

                    # WHICH FEET THE ESTIMATOR IS TOLD TO TRUST.  In a trot that
                    # is two of them for most of the cycle and four through the
                    # handover; anywhere else it is all four.
                    planted = controller.gait.contact(now) if stage == "TROT" \
                        else ALL4

                    # ---- the sub-sampled model block -----------------------
                    if sweep % P.MODEL_EVERY == 0:
                        state, act, why = est.read(now, q, qd, planted)
                        if act:
                            if z_crouch is None:
                                z_crouch = float(state["r"][2])
                                print(f"[mpc] crouch height "
                                      f"{z_crouch*1e3:.0f} mm to the TRUNK "
                                      f"BOTTOM ({state['z_hip']*1e3:.0f} mm to "
                                      f"the hip axis).  FK, drift-free -- put "
                                      f"a ruler on it.")
                                _print_attitude_report(est, args)
                                print("[mpc] ENTER to rise.")
                            z_des = (dm.height_ramp(z_crouch, args.height,
                                                    (now - t0) / P.T_RISE)
                                     if stage == "RISE" else
                                     args.height if stage in ("HOLD", "TROT")
                                     else z_crouch)
                            kine = ctl.LegKinematics(q, state["C"])
                            if torque_stage:
                                if use_mpc:
                                    tau_ff, contact, replan = controller.update(
                                        now, q, qd, tau_meas, state, kine,
                                        shared.plan, z_des,
                                        swing_enabled=(stage == "TROT"))
                                    q_ref = controller.q_ref
                                    f_now = controller.last["f_applied"]
                                    w_now = controller.last["weight"]
                                    if replan:
                                        # the contact set is a CONSTRAINT of
                                        # the QP; a plan built for the old one
                                        # is wrong about which foot may push
                                        shared.replan = True
                                else:
                                    # week 2's wrench, unchanged: RISE always,
                                    # HOLD under --wrench-hold
                                    q_ref = q_ref_for_height(q_ref, z_des,
                                                             foot_xy)
                                    W = dm.body_wrench(state, z_des, C.MASS,
                                                       gains=gains)
                                    tau_ff, diag = ft.stance_torque(
                                        q, qd, state, W, ALL4, gains,
                                        force_frac=args.force_frac,
                                        leg_gravity=True)
                                    # The wrench's OWN per-foot forces, in
                                    # world axes, so the solver's smoothing
                                    # term is anchored to what the robot is
                                    # actually being pushed with rather than
                                    # to zero -- otherwise every plan through
                                    # the rise is biased low and the shadow
                                    # A/B compares against a handicap.
                                    f_now = np.array(
                                        [state["C"].T @ diag["forces"][lg]
                                         if lg in diag["forces"]
                                         else np.zeros(3) for lg in LEGS])
                                    w_now = np.ones(4)
                            # -- feed the solver, every model tick ------------
                            # The gait object is walked HERE, in this thread,
                            # and the result published: it latches early
                            # touchdowns from this loop, and a worker reading
                            # it mid-latch would plan against a contact set
                            # that never existed.
                            shared.q = q.copy()
                            shared.state = state
                            shared.z_des = z_des
                            shared.v_cmd = controller.v_cmd.copy()
                            shared.wz_cmd = controller.wz_cmd
                            shared.f_applied = f_now
                            # THE SCHEDULE AND THE TIME IT IS FOR, PUBLISHED
                            # TOGETHER.  The worker hands t_sched back with the
                            # plan, so the CAN loop can read the plan AT A TIME
                            # instead of holding its first knot for a whole
                            # replan period; see mpc_controller.plan_force.
                            t_sched = now + C.MPC_SCHEDULE_LEAD_S
                            shared.weight_sched = \
                                controller.gait.contact_weight_schedule(
                                    t_sched, mpc.N, mpc.dt)
                            shared.t_sched = t_sched
                        else:
                            # honest fallback: hold each leg up, command no
                            # body force.  A frozen plan would keep pushing on
                            # a world model that has stopped updating.
                            shared.state = None
                            tau_ff = (ctl.leg_gravity_only(kine)
                                      if kine is not None
                                      else np.zeros(N_JOINTS))
                            if torque_stage and not limp:
                                limp = True
                                print(f"[mpc] LIMP: estimator refused ({why})")

                        # -- is the plan we are applying still alive? --------
                        plan_age = mpc_worker.stale_s(shared)
                        if use_mpc and not limp and torque_stage \
                                and plan_age > C.MPC_STALE_S:
                            limp = True
                            print(f"[mpc] LIMP: no plan for "
                                  f"{plan_age*1e3:.0f} ms "
                                  f"({shared.reason or 'solver overrun'}) -- a "
                                  f"held plan keeps pushing with feet that may "
                                  f"have left the ground")

                    # ---- the 250 Hz floor + safety -------------------------
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
                        print(f"[mpc] LIMP: input-lost latch on {latched}")

                    if state is not None and act:
                        tilt = max(abs(np.degrees(est.roll)),
                                   abs(np.degrees(est.pitch)))
                        if tilt > P.TILT_STOP_DEG:
                            stop = f"tilt {tilt:.1f} deg"
                            break

                    # THE measurement: measured iq -> foot force -> the robot.
                    # Staggered against the model block so the two never land
                    # in the same sweep (params.LOAD_OFFSET).
                    if sweep % P.LOAD_EVERY == P.LOAD_OFFSET:
                        support, ok = ft.foot_load_from_torque(
                            q, tau_meas, None if state is None else state["C"])
                        load_sum = float(np.nansum(support)) if ok.any() \
                            else float("nan")
                        if stage == "HOLD" and not limp \
                                and np.isfinite(load_sum) \
                                and abs(load_sum - C.WEIGHT) / C.WEIGHT \
                                > P.LOAD_SUM_TOL_FRAC:
                            limp = True
                            print(f"[mpc] LIMP: measured foot load "
                                  f"{load_sum:.1f} N against {C.WEIGHT:.1f} N "
                                  f"expected -- the force loop is not doing "
                                  f"what it says")

                    if args.log:
                        log.append((now, q.copy(), qd.copy(), qd_drv.copy(),
                                    tau_cmd.copy(), tau_meas.copy(),
                                    est.roll, est.pitch, est.z, load_sum, stage,
                                    est.roll_fk, est.pitch_fk, est.z_hip,
                                    est.roll_raw, est.pitch_raw, q_ref.copy(),
                                    np.zeros(3) if state is None
                                    else np.asarray(state["w"]).copy(),
                                    est.v.copy(), W.copy(),
                                    np.asarray(planted, float).copy(),
                                    f_now.copy(), w_now.copy(),
                                    np.asarray(shared.plan.f0).copy(),
                                    plan_age, controller.v_cmd.copy(),
                                    controller.wz_cmd))

                    if now - last_print >= 1.0 / base.STATUS_HZ:
                        last_print = now
                        extra = ""
                        if use_mpc:
                            extra = (f"  plan {plan_age*1e3:3.0f}ms/"
                                     f"{f_now[:, 2].sum():4.1f}N")
                        if stage == "TROT":
                            extra += f"  air={int(np.sum(w_now <= 0.0))}"
                        print(f"[mpc] {'LIMP' if limp else stage:5s} "
                              f"{est.status() if state is not None else why}"
                              f"{extra}", flush=True)

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

            print(f"[mpc] stopped: {stop}")
            # Park in POSITION mode from wherever the torque stage left it.
            # Cutting torque at height would drop the robot; 0xA4 takes the
            # weight on the drivers' own loops first.
            if stop == "park" or args.park_on_stop:
                try:
                    cap.parked(mb, unwrap, key, max_speed_dps=args.crouch_dps)
                except Exception as exc:                    # noqa: BLE001
                    print(f"[mpc] park failed: {exc}")
            print("[mpc] soft stop -- HOLD THE ROBOT")
            base._soft_stop(mb)
    except KeyboardInterrupt as exc:
        print(f"\n[mpc] aborted: {exc}")
    except RuntimeError as exc:
        print(f"\n[mpc] fault: {exc}")
    finally:
        if shared is not None:
            shared.run = False
            if worker is not None:
                worker.join(timeout=1.0)
        try:
            imu.stop()
        except Exception:                                   # noqa: BLE001
            pass
        key.close()

    # ---- the exit report --------------------------------------------------
    print()
    print("=" * 74)
    if shared is not None:
        print(f"  MPC {mpc_worker.census(shared)}")
        print("  ^ CONTROL_ROADMAP calls this the phase's main risk.  If "
              "'over budget' is not")
        print("    ~0, lower --mpc-hz before anything else: a missed publish "
              "is a limp.")
    if log:
        a = np.array([r[9] for r in log], dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            print(f"  foot load sum: mean {a.mean():.1f} N, min {a.min():.1f}, "
                  f"max {a.max():.1f}, against {C.WEIGHT:.1f} N")
            print("  ^ THE number: measured current, not a command.  If this "
                  "does not add up to the")
            print("    robot, the plan is fantasy however good the attitude "
                  "looks.")
        enc = np.max(np.abs(np.array([r[2] for r in log])), axis=0)
        drv = np.max(np.abs(np.array([r[3] for r in log])), axis=0)
        i = int(np.argmax(drv))
        print(f"  peak |qd|: encoder (used by the loop) {enc.max():.2f} rad/s, "
              f"driver field (not used) {drv[i]:.2f} rad/s on "
              f"{base.JOINT_LABELS[i]}")
        if glitches.any():
            worst = int(np.argmax(glitches))
            print(f"  driver/encoder disagreements: {int(glitches.sum())} "
                  f"total, worst {base.JOINT_LABELS[worst]} "
                  f"({int(glitches[worst])}) -- these no longer reach the "
                  f"force law")
        else:
            print("  driver and encoder agreed throughout -- no glitching this "
                  "run")
        if args.log:
            np.savez_compressed(
                args.log,
                t=np.array([r[0] for r in log]), q=np.array([r[1] for r in log]),
                qd=np.array([r[2] for r in log]),
                qd_drv=np.array([r[3] for r in log]),
                tau_cmd=np.array([r[4] for r in log]),
                tau_meas=np.array([r[5] for r in log]),
                roll=np.array([r[6] for r in log]),
                pitch=np.array([r[7] for r in log]),
                z=np.array([r[8] for r in log]),
                load=np.array([r[9] for r in log]),
                stage=np.array([r[10] for r in log]),
                roll_fk=np.array([r[11] for r in log]),
                pitch_fk=np.array([r[12] for r in log]),
                z_hip=np.array([r[13] for r in log]),
                roll_raw=np.array([r[14] for r in log]),
                pitch_raw=np.array([r[15] for r in log]),
                q_ref=np.array([r[16] for r in log]),
                w=np.array([r[17] for r in log]),
                v=np.array([r[18] for r in log]),
                W=np.array([r[19] for r in log]),
                planted=np.array([r[20] for r in log]),
                # THE PLAN, AND WHAT WAS ACTUALLY APPLIED OF IT.  Without both,
                # a log cannot separate "the MPC asked for this" from "the
                # contact clamp allowed that", and every per-foot number is
                # ambiguous.
                f_applied=np.array([r[21] for r in log]),
                contact_weight=np.array([r[22] for r in log]),
                f_plan=np.array([r[23] for r in log]),
                plan_age=np.array([r[24] for r in log]),
                v_cmd=np.array([r[25] for r in log]),
                wz_cmd=np.array([r[26] for r in log]),
                # the horizon and the cost, so a log is readable a month later
                n_horizon=mpc.N, mpc_dt=mpc.dt, mpc_hz=args.mpc_hz,
                w_att=mpc.q_state[0:3], w_pos=mpc.q_state[3:6],
                w_omega=mpc.q_state[6:9], w_vel=mpc.q_state[9:12],
                w_force=mpc.w_force, w_smooth=mpc.w_smooth,
                qp_iters=mpc.iters, qp_rho=mpc.rho,
                mu=C.MU, fz_min=C.FZ_MIN, fz_max=C.FZ_MAX,
                period=args.period, duty=args.duty,
                swing_height=args.swing_height,
                wrench_hold=bool(args.wrench_hold),
                force_frac=args.force_frac, mass=C.MASS,
                setpoint_roll_deg=args.setpoint_roll,
                setpoint_pitch_deg=args.setpoint_pitch,
                imu_below_trunk_origin_m=P.IMU_BELOW_TRUNK_ORIGIN_M,
                kp_imp=args.kp, kd_imp=args.kd, tau_max=args.tau_max,
                glitches=glitches)
            print(f"  log -> {args.log} ({len(log)} sweeps)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", default=None)
    ap.add_argument("--height", type=float, default=C.STAND_HEIGHT,
                    help="floor to TRUNK BOTTOM (m)")
    ap.add_argument("--tau-max", type=float, default=C.TAU_MAX_DEFAULT)
    ap.add_argument("--kp", type=float, default=P.KP_IMP)
    ap.add_argument("--kd", type=float, default=P.KD_IMP)
    ap.add_argument("--force-frac", type=float, default=C.FORCE_FRAC_DEFAULT)
    ap.add_argument("--wrench-hold", action="store_true",
                    help="run WEEK 2's wrench in HOLD as well as RISE.  The "
                         "MPC still solves and is still logged -- this is the "
                         "ablation half of 'does the MPC change anything'")
    # -- the MPC's own knobs, each an A/B rather than an edit ---------------
    ap.add_argument("--mpc-hz", type=float, default=C.MPC_HZ,
                    help="solver rate.  LOWER THIS FIRST if the exit report "
                         "shows solves over budget")
    ap.add_argument("--n-horizon", type=int, default=C.N_HORIZON,
                    help="knots.  The knot LENGTH is derived from it so the "
                         "horizon stays one gait cycle; above ~8 the QP "
                         "crosses numpy's threaded-BLAS threshold (see "
                         "mpc_config.N_HORIZON)")
    ap.add_argument("--qp-iters", type=int, default=C.QP_ITERS)
    ap.add_argument("--w-att", type=float, default=1.0,
                    help="scale on the attitude weights.  kp_att moves as "
                         "roughly its square root; the banner prints what you "
                         "actually got")
    ap.add_argument("--w-z", type=float, default=1.0,
                    help="scale on the height weight, same square-root rule")
    ap.add_argument("--w-vel", type=float, default=1.0,
                    help="scale on the HORIZONTAL velocity weight.  0 is the "
                         "ablation: params.KD_X/KD_Y have never been run above "
                         "zero on this robot, and this stack is the first to "
                         "put a gain there")
    # -- the gait ------------------------------------------------------------
    ap.add_argument("--period", type=float, default=C.GAIT_PERIOD)
    ap.add_argument("--duty", type=float, default=C.DUTY,
                    help="stance fraction.  0.5 is the textbook trot and has "
                         "ZERO double support; the contact ramp then has "
                         "nowhere to hand the load over")
    ap.add_argument("--swing-height", type=float, default=C.SWING_HEIGHT)
    ap.add_argument("--promote", action="store_true",
                    help="act on the MEASURED contact: promote a late-swing "
                         "touchdown to stance.  OFF by default because the "
                         "detector reads back the swing controller's own "
                         "command (+9 N of phantom ground reaction off a foot "
                         "in the air, measured) and nobody has yet checked "
                         "what is left after the command is subtracted, with "
                         "this robot's legs actually swinging.  "
                         "mpc_gait.ContactAwareGait names the one measurement "
                         "that earns this flag; the detector's fz is in the "
                         "--log npz either way")
    # -- the rig -------------------------------------------------------------
    ap.add_argument("--setpoint-roll", type=float, default=P.SETPOINT_ROLL_DEG,
                    help="resting attitude of THIS rig in deg, subtracted from "
                         "every AHRS reading.  0 means the IMU mount's own tilt "
                         "is fed to the controller as a real one")
    ap.add_argument("--setpoint-pitch", type=float,
                    default=P.SETPOINT_PITCH_DEG)
    ap.add_argument("--crouch-dps", type=float, default=cap.MAX_DPS)
    ap.add_argument("--park-on-stop", action="store_true")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    if not 0.0 < args.tau_max <= C.TAU_STAGED_MAX:
        ap.error(f"--tau-max must be in (0, {C.TAU_STAGED_MAX}]")
    if not 0.0 <= args.force_frac <= 1.0:
        ap.error("--force-frac must be in [0, 1]")
    if not 2 <= args.n_horizon <= 20:
        ap.error("--n-horizon must be in [2, 20]")
    if not 1.0 <= args.mpc_hz <= P.CONTROL_HZ:
        ap.error(f"--mpc-hz must be in [1, {P.CONTROL_HZ:.0f}]")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
