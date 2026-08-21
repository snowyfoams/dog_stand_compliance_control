#!/usr/bin/env python3
"""Every constant the hardware SRB-MPC reads.  No logic, no sibling imports.

WHAT IS *NOT* HERE, AND WHY THAT IS THE POINT
    Mass, inertia, the hip and stance geometry, the gait period and the joint
    caps are NOT re-declared in this file.  They are imported from
    dog5_trot.config and august_week2.params, which are the controlled copies
    of dog5.xml and of the measured bus facts respectively.

    params.py exists because eight gains were once declared TWICE as
    independent literals and editing one silently left the other behind.  A
    second copy of MASS in an MPC file is that bug waiting again -- and worse
    here than anywhere, because the MPC's B matrix is 1/m and its moment rows
    are I^-1: a mass that drifts 5% from the one force_totorque uses makes the
    plan and the execution disagree about the same robot.

    So this file declares only what is genuinely NEW: the horizon, the cost,
    the solver budget, and the constants the swing and contact layers need.

THE HORIZON IS ONE GAIT CYCLE, AND IT IS DERIVED
    N_HORIZON * MPC_DT == dog5_trot.config.GAIT_PERIOD, exactly, because
    MPC_DT is computed from the other two.  A horizon shorter than a cycle
    cannot see the next touchdown, which is the whole reason a trot wants an
    MPC instead of the instantaneous grasp map; a horizon longer than a cycle
    spends QP on a schedule the next replan will have moved anyway.

    The simulation used N=10, dt=0.03 against a 0.42 s period -- 71% of a
    cycle -- and got away with it because a simulated foot never lands early.

THE COST WEIGHTS ARE SIZED FROM THE GAINS THIS ROBOT HAS STOOD ON
    This is the one place where copying the simulation would have been wrong.
    A cost weight is not a gain, but the closed loop has an EFFECTIVE gain and
    it can be read straight out of the solver: put a pure roll error into x0,
    solve, and sum r_i x f_i.  The runner prints those four numbers in its
    banner at every start, so a weight change is never committed without the
    gain it implies.

    The targets are august_week2's verified pairs -- kp_z 300 N/m, kd_z 40,
    kp_att 10 Nm/rad, kd_att 0.5 -- and NOT the simulator's, because:

        kp_att   this stance is +/-112 mm wide, so its roll capacity is
                 WEIGHT * 0.112 = 6.4 Nm and a 120 Nm/rad gain saturates it
                 at 3.1 deg.  The simulator's attitude weights sit in that
                 regime.  Hardware walked the gain DOWN to 10 across the
                 2026-08-18 ladder, and params.py names the log for each step.
        kp_z     800/120 has 7.4 deg of phase margin against the measured
                 20 ms outer delay.  300/40 has 27.

    The MEASURED effective gains at the weights below are in the table beside
    them.  Re-read them from the banner after any edit.

WHAT THIS TRACK RUNS THAT WEEK 2 NEVER DID
    W_VEL[0:2] is a live gain on horizontal velocity.  params.KD_X and KD_Y
    are 0.0 and their comment says "never run > 0" -- the leg-odometry
    velocity carries a 5 Hz filter and a 12 ms hold, and nobody had tested
    damping on it.  A trot needs it: without a velocity term the MPC has no
    reason to produce the tangential force that answers a push, and the
    Raibert placement is left to do all of it one step later.

    It is small, it is friction-limited by the QP's own pyramid, and
    --w-vel 0 is the ablation half of that A/B.  Run the ablation.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUG = os.path.dirname(_HERE)
for _p in (_AUG, os.path.join(_AUG, "august_week2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# THE TWO OWNERS OF EVERY NUMBER THIS FILE DOES NOT DECLARE.
#   tcfg  geometry, mass, inertia, gait, swing, joint caps -- from dog5.xml
#   P     bus timing, torque caps, friction, the estimator's constants
from dog5_trot import config as tcfg                       # noqa: E402
import params as P                                         # noqa: E402

LEGS = tcfg.LEGS                     # ("FL", "FR", "RL", "RR") -- HARDWARE order
N_LEGS = tcfg.N_LEGS
N_JOINTS = tcfg.N_JOINTS
JOINT_INDEX = tcfg.JOINT_INDEX
PHASE_OFFSET = tcfg.PHASE_OFFSET

# THE LEG ORDER IS NOT THE SIMULATOR'S, AND NOTHING CONVERTS BETWEEN THEM.
# srb_mpc/robot.py is [FR, FL, RR, RL]; this tree is ("FL", "FR", "RL", "RR")
# everywhere -- dog5_kinematics, force_totorque, the CAN motor map, and every
# log ever recorded on this robot.  So the port ADOPTS the hardware order
# outright rather than carrying a permutation that one file would eventually
# forget to apply.  The trot pairing survives it: the diagonals are (FL, RR)
# and (FR, RL) here, exactly as (FR, RL) and (FL, RR) there, and
# tcfg.PHASE_OFFSET is already written in this order.

# ===========================================================================
# mass, inertia and geometry -- re-exported, never re-declared
# ===========================================================================
MASS = tcfg.MASS                     # 5.8151 kg
WEIGHT = tcfg.WEIGHT                 # 57.05 N
GRAVITY = tcfg.GRAVITY
# Whole-robot tensor about the whole-robot CoM at the nominal stance, in the
# TRUNK frame.  This is precisely the SRB-MPC's I_body: the simulator computes
# the same composite tensor from the MJCF at its standing keyframe.
INERTIA_BODY = tcfg.INERTIA_BODY
COM_BODY = tcfg.COM_BODY             # nominal; the loop uses the live one
FOOT_STANCE_BODY = tcfg.FOOT_STANCE_BODY
FOOT_RADIUS = tcfg.FOOT_RADIUS
LEG_REACH = tcfg.LEG_REACH

# ===========================================================================
# the horizon
# ===========================================================================
N_HORIZON = 8                        # knots.  12 forces each -> 96 QP variables
GAIT_PERIOD = tcfg.GAIT_PERIOD       # 0.40 s
DUTY = tcfg.DUTY                     # 0.60
MPC_DT = GAIT_PERIOD / N_HORIZON     # 0.050 s -- DERIVED, see the header
MPC_HZ = 40.0                        # solves per second, off-thread.  Two per
                                     # knot, so the plan the robot is holding
                                     # is never more than half a knot old.
# CONTROL_ROADMAP Phase 5 budgets "12 vars x 10-step at 25-50 Hz on the Pi 5"
# and calls solve time the phase's main risk.  It is measured IN THE RUN --
# the worker publishes min/mean/max and the exit report prints them -- rather
# than assumed here.  --mpc-hz is the flag to lower if that report says so.
#
# WHY 8 KNOTS AND NOT THE SIMULATOR'S 10, WHICH IS A NUMPY FACT AND NOT A
# CONTROL ONE.  The decision variable is 12*N, and BLAS switches the 120x120
# inverse onto its threaded path somewhere just under 100x100.  Measured on
# the development box, same code, same solve:
#
#     N = 8   (96 vars)   inverse 0.08 ms mean, 0.14 max
#     N = 10 (120 vars)   inverse 2.21 ms mean, 27.3 max
#
# A 27 ms outlier is longer than a whole replan period, and it arrives as a
# missed publish rather than a slow one.  8 knots keeps every array under the
# cliff, and the horizon is STILL exactly one gait cycle because MPC_DT is
# derived -- 0.05 s knots instead of 0.04.  Run the worker with
# OPENBLAS_NUM_THREADS=1 anyway (the runner's RUN block does); on a 4-core Pi
# sharing threads with a 250 Hz CAN loop is a jitter source, not a speed-up.

# WHAT THE SCHEDULE IS EVALUATED AT, WHICH IS NOT `now` AND IS NOT `now` PLUS
# THE LATENCY EITHER.
# ---------------------------------------------------------------------------
# Knot 0's force is not an instant, it is a BOX: the QP holds it constant
# across [t_sched, t_sched + MPC_DT], so its centre of mass in time is
# t_sched + MPC_DT/2.  What the robot actually feels is another box: the CAN
# loop applies the published force from the publish until the next one and the
# joint feels it params.LOOP_DELAY_S later, so that box is centred at
# t_solve + LOOP_DELAY + (1/MPC_HZ)/2.  Line the two centres up:
#
#     lead = LOOP_DELAY + 1/(2 MPC_HZ) - MPC_DT/2
#          = 0.016 + 0.0125 - 0.025 = 3.5 ms
#
# THE MIDDLE TERM ALONE IS THE MISTAKE THIS FILE MADE FIRST, and it is worth
# writing down because it looked right: leading by "half a replan period plus
# the loop delay" (28.5 ms) forgets that the knot the plan is being compared
# against is 50 ms wide.  The plan then runs more than half a knot ahead of
# the contact set the applied force is CLAMPED against, and at every diagonal
# handover it hands the load to a foot the clamp still calls airborne.
# Measured, four gait cycles driven at the model rate:
#
#     lead 28.5 ms   applied support fell below 90% of the weight on 11 of
#                    133 ticks, worst 24 N on a 57 N robot
#     lead  3.5 ms   see the same run in the commit message
#
# The STATE is deliberately not propagated forward by any of this.  Propagating
# needs the model to be right about the next 20 ms, and if it were that right
# the delay would not matter; the schedule shift needs only the CLOCK, which
# is exact.
MPC_SCHEDULE_LEAD_S = max(P.LOOP_DELAY_S + 0.5 / MPC_HZ - 0.5 * MPC_DT, 0.0)

# ===========================================================================
# the cost
# ===========================================================================
# State order is the simulator's: [rpy(3), p(3), omega_world(3), v(3), g].
# The trailing gravity state is a constant and carries weight 0.
#
# THE EFFECTIVE GAINS AT THESE WEIGHTS, MEASURED, four feet down, at the
# nominal stance -- put a unit error in one state, solve, and read the wrench
# back out (mpc_controller.effective_gains does exactly that).  The runner
# prints this table in its banner at every start, so an edit here is never
# committed without the gains it implies.
#
#     axis      this file    week 2 verified       the log week 2 names
#     kp_roll      8.00        10                  s3c_att20.npz
#     kp_pitch    10.92        10                  s3c_att20.npz
#     kp_z       314.5        300                  s2_kpz300.npz
#     kd_roll      1.51         0.5                s3c_att20.npz
#     kd_pitch     7.70         0.5                s3c_att20.npz
#     kd_z       104.3         40                  s1_kdz40.npz
#     kd_x        18.2          0     (never run above 0)
#     kp_x         9.5          0     (there is no absolute x)
#
# and the support at zero error is 57.05 N against a 57.046 N robot, which is
# the gravity-referenced regulariser doing its job (see convex_mpc).
#
# THE STIFFNESSES MATCH AND THE DAMPINGS DO NOT, AND THAT IS THE HORIZON.
# ---------------------------------------------------------------------------
# A rate error is a position error one horizon later, so an MPC prices the two
# together and the ratio it picks is of the horizon's own order:
#
#     kd/kp     roll 0.19 s    pitch 0.71 s    height 0.33 s
#     week 2    att  0.05 s                    height 0.13 s
#
# A PD ladder has no such constraint -- it walks one gain at a time and stops
# where the robot stops complaining -- so the two disagree by design, not by
# accident.  What matters is that the MPC is the MORE damped controller, not
# the twitchier one:
#
#     roll    critical damping is 2 sqrt(kp Ixx) = 2 sqrt(8.00 x 0.0658)
#             = 1.45 Nms/rad.  This runs 1.51, i.e. 1.04x critical, against
#             week 2's 0.34x.
#     height  critical is 2 sqrt(kp m) = 2 sqrt(314.5 x 5.815) = 85.5 Ns/m.
#             This runs 104, i.e. 1.22x critical, against week 2's 0.47x.
#
# THE ONE TO WATCH IS kd_z, because params.py names it as the axis carrying
# the whole lag budget (a 5 Hz filter on leg odometry plus a 12 ms hold).  Two
# numbers, both of which say it is affordable:
#     phase    the loop crosses over near kd/m = 17.9 rad/s = 2.85 Hz, where
#              the measured 20 ms outer delay is 21 deg -- so the margin is set
#              by the second-order rolloff, not by the delay
#     noise    encoder-differenced qd is 0.026 rad/s filtered, which through
#              this kd_z is 0.3 N on a 57 N robot.  The 8.1 rad/s DRIVER speed
#              field that caused the 2026-08-17 shake is not read by anything
#              in this stack, exactly as in week 2.
# If a height chatter does appear, --w-z is the flag: kp_z moves as roughly the
# SQUARE ROOT of it (160 -> 246, 240 -> 315) while kd_z barely moves at all, so
# it is a stiffness knob and not a damping one.  The damping knob is the
# HORIZON, and shortening that shortens what the plan can see.
W_ATT = np.array([80.0, 9.0, 20.0])        # roll, pitch, yaw
W_POS = np.array([1.0, 1.0, 240.0])        # x, y, z
W_OMEGA = np.array([1.5, 1.5, 3.0])        # wx, wy, wz (world axes)
W_VEL = np.array([0.25, 0.25, 12.0])       # vx, vy, vz

# X AND Y ARE NOT AN ABSOLUTE POSITION, AND THEIR WEIGHTS SAY SO.
# Every solve re-anchors x = y = 0 (see mpc_controller.mpc_state), so W_POS[:2]
# penalises drift WITHIN one 0.4 s horizon and nothing longer.  That is the
# roadmap's "x/y enter only as horizon-relative references (drift-safe by
# construction)", and it is what replaces the simulation's station-keeping
# outer loop -- which needs an absolute position this robot does not have.
# They are small because the only thing observing them is the integral of a
# lagged leg-odometry velocity.  --w-vel 0 removes the horizontal damper
# entirely, which is the ablation half of the one A/B this stack adds.

W_FORCE = 2.0e-4             # regulariser on every force in the horizon; the
                             # simulator's value, doing the same job here --
                             # 12 forces against 6 wrench rows is a 6-wide
                             # null space at every knot.
W_SMOOTH = 2.0e-4            # on (f_0 - f_applied) ONLY.  New here, and it is
                             # a HARDWARE term: the horizon smooths the plan,
                             # but nothing smooths the STEP between one solve's
                             # first knot and the next's, and at 40 Hz that
                             # step lands on a slew-limited actuator.  Same
                             # role as balance_qp.W_SMOOTH, one knot wide.
#
# IT IS THE SAME SIZE AS W_FORCE AND IT MUST NOT BE MUCH LARGER, WHICH IS A
# MEASUREMENT AND NOT A PREFERENCE.  The term pulls the new plan towards the
# force currently applied -- and that force was itself pulled towards the one
# before it, with nothing anchoring the chain.  Above about W_FORCE the drag
# beats the gravity anchor and the whole sequence walks downhill.  Driven with
# 300 solves of realistic per-solve state noise on a live trot schedule:
#
#     W_SMOOTH   total support: min / mean / max (N), on a 57.05 N robot
#     0          45.8 / 57.0 / 68.7
#     2e-4       47.0 / 57.0 / 67.7        <- ships
#     1e-3       40.3 / 55.6 / 64.8
#     5e-3       21.1 / 45.4 / 62.3        <- 12 N light, on average, for ever
#
# The 5e-3 row is what this file shipped before the sweep was run, and it is
# exactly the failure mode the gravity-referenced regulariser was added to
# remove -- reintroduced by a different term.  A standing robot that is 12 N
# light does not fall over; it sags, and every gain above it reads wrong.

# ===========================================================================
# the friction cone and the force box
# ===========================================================================
MU = P.MU_FRICTION           # 0.6, the same clamp the stand uses
# THE PYRAMID IS INSCRIBED IN THE CIRCLE.  |fx| <= mu fz and |fy| <= mu fz
# permits a RESULTANT of sqrt(2) mu fz on the diagonals -- 41% past the
# friction the floor has, i.e. the QP would call a slipping plan feasible.
# Dividing by sqrt(2) puts the pyramid inside the circle.  Identical argument,
# and identical constant, to dog5_trot/balance_qp.py.
MU_AXIS = MU / np.sqrt(2.0)
FZ_MIN = P.FZ_MIN_N          # 1.0 N: a planted foot never unloads to nothing
FZ_MAX = tcfg.FZ_MAX         # 1.5 * WEIGHT: one diagonal pair carries it all,
                             # with room to answer a push

# ===========================================================================
# the solver -- dense ADMM, because the venv has numpy and nothing else
# ===========================================================================
# dog5_trot/balance_qp.py records the fact this depends on: no scipy, no osqp,
# no quadprog, no cvxpy, no qpsolvers.  The iteration is OSQP's, written out.
#
# THE COST IS SPENT WHERE THE STRUCTURE IS.  The constraint matrix is
# blockdiag(D) with the SAME 5x3 D at every foot of every knot, so C x, C^T y
# and C^T C are a reshape and a 3x3 -- and the whole per-iteration cost is one
# 120x120 matrix-vector product.  60 of those is 0.9 MFLOP, against the 4
# MFLOP of building the condensed Hessian once.
QP_ITERS = 60                # the BUDGET, not the usual cost: warm started,
                             # this problem converges in 5-10 (the runner logs
                             # the count).  60 is what a cold start after a
                             # contact-pattern change is allowed to spend.
# RHO IS 0.05 AND NOT balance_qp's 1.0, AND THE DIFFERENCE IS NOT COSMETIC.
# ADMM's rho balances the objective against the constraints, so it has to be
# scaled to the problem -- and this one is two orders of magnitude stiffer
# than the 12-variable force split, because the condensed Hessian carries the
# horizon's state weights (hundreds) where balance_qp carries only W_TASK
# (tens).  Measured, at the weights above, standing:
#
#     rho 1.00   60 iterations, still 0.55 N of support missing
#     rho 0.30   60 iterations, 0.21 N missing
#     rho 0.05    5 iterations, 0.03 N missing
#
# A mis-scaled rho does not announce itself: the solve returns, the forces
# look plausible, and the robot stands 1% light for ever.  If the state
# weights are ever moved by an order of magnitude, re-read those three rows
# before trusting the new gains.
QP_RHO = 0.05
QP_SIGMA = 1.0e-6
# THE CONVERGENCE TOLERANCE IS IN NEWTONS, because the constraint rows are.
# 1 mN is three orders below the 0.02 N the torque sensing can even see, and
# the first knot is clamped feasible on the way out regardless -- so this is
# an early exit, not a correctness gate.
QP_TOL_N = 1.0e-3

# ===========================================================================
# swing -- Raibert placement and the arc
# ===========================================================================
SWING_HEIGHT = tcfg.SWING_HEIGHT     # 0.04 m apex over the stance plane
RAIBERT_KV = tcfg.RAIBERT_KV         # 0.03 s, the velocity-ERROR term
HIP_OFFSET = tcfg.HIP_OFFSET         # the reach clamp measures from here
# The Cartesian impedance that TRACKS the arc.  Scalars, not 3x3 diagonals: a
# trot in place has no reason to be stiffer in one direction than another, and
# a diagonal matrix that is secretly one number invites the reader to think
# otherwise.  Same values dog5_trot runs.
KP_SWING = float(tcfg.KP_SWING[0, 0])        # 200 N/m
KD_SWING = float(tcfg.KD_SWING[0, 0])        # 8 Ns/m
# Joint damping added to a STANCE leg inside the force law, on top of the
# runner's 250 Hz impedance.  force_totorque.stance_torque carries the same
# term with the same value; it is a robustness term against the force law's
# velocity-level nature, not a controller.
KD_JOINT_STANCE = P.KD_JOINT                 # 0.15 Nms/rad
# A SECOND low pass on the placement velocity, on top of the estimator's 5 Hz.
# The simulation records why: the instantaneous CoM velocity oscillates at
# GAIT frequency, and feeding it raw into the touchdown target wobbles the
# footholds in resonance and sustains a rocking limit cycle (0.3 deg / 2 mm
# over 30 s once filtered).  Gait frequency here is 1/0.40 = 2.5 Hz, which the
# estimator's 5 Hz corner passes almost untouched -- so the filter the sim
# needed is still needed.  Parametrised by a TIME CONSTANT, so it means the
# same thing at any control rate.
RAIBERT_V_TAU = 0.10                 # s
# The reachable-set clamp, as a fraction of LEG_REACH.  A Raibert step at
# speed asks for a foot the leg cannot meet; clamping in the PLANNER makes
# that a shorter step, which the gait survives, instead of a saturated joint.
REACH_FRAC = 0.95
# What this stance height can actually travel at, derived in dog5_trot.config
# from the leg reach left over at the nominal stance: 0.043 m/s.  It is a
# TROT IN PLACE and the number says so; a velocity command past it is clamped
# by the runner rather than quietly asking for an unreachable foothold.
V_CMD_MAX = tcfg.MAX_FORWARD_V_AT_STAND_HEIGHT
WZ_CMD_MAX = 0.30                    # rad/s.  The abduction limit caps lateral
                                     # authority at ~2 cm (CONTROL_ROADMAP,
                                     # "Risks"), and yaw is produced entirely
                                     # by tangential friction at the feet.

# ===========================================================================
# the contact-aware layer
# ===========================================================================
# The clock schedule is UNCHANGED and stays the plan.  What this adds is the
# simulator's early-touchdown promotion, and it is not cosmetic there: a purely
# clock-driven trot pumps itself over within 12 s, because each early contact
# injects an unmodelled impulse at gait frequency while the MPC still treats
# that foot as force-free.
#
# ON HARDWARE THE MEASUREMENT IS THE MOTOR CURRENT, not a contact buffer.
# force_totorque.foot_load_from_torque already inverts measured iq through
# J^-T with each leg's own weight removed -- the same quantity that has to sum
# to 57 N in the exit report.  A foot carrying more than CONTACT_FZ_ON of it
# is standing on something.
# AND IT IS OFF UNTIL SOMEBODY MEASURES THE DETECTOR ON THIS ROBOT.
# The inversion that reads the foot load is the exact inverse of the map the
# swing controller commands through, so on a swinging leg it reads back that
# controller's own force: -12 N of downward swing command comes back as +9 N of
# "ground reaction" with the foot in the air, and the sign is positive exactly
# in the late-swing window promotion looks at.  mpc_gait subtracts the command,
# which removes the dominant term and leaves the current loop's ~24% tracking
# error -- better, and still not measured with legs in the air.
#
# A false promotion is worse than no promotion: the gait plants an airborne
# foot, the MPC allocates it force, and that share of the weight is pushed into
# nothing.  Off, the trot runs on the clock alone -- which is what
# dog5_trot/trot_hw already does on this robot.
# ContactAwareGait names the one measurement that earns --promote.
PROMOTE_ENABLED = False
PROMOTE_AFTER = 0.5          # only a LATE-swing contact promotes.  An early
                             # one is the foot still leaving the ground.
CONTACT_FZ_ON = 8.0          # N.  0.55 of an even four-way share (14.3 N),
                             # and 8x the FZ_MIN a planted foot may sit at, so
                             # a foot merely grazing does not latch.
CONTACT_FZ_OFF = 4.0         # N.  Hysteresis: one sample at the boundary must
                             # not chatter the contact set, because every
                             # change of it forces a replan.

# ===========================================================================
# rates, staleness and the handover
# ===========================================================================
CONTROL_HZ = P.CONTROL_HZ            # 250, the sweep -- impedance and gate
MODEL_EVERY = P.MODEL_EVERY          # 3, so estimator + torque map at 83 Hz
# The MPC is the ONE block that cannot be sub-sampled into a sweep at any
# ratio: a 120-variable condensed QP is milliseconds, the CAN slot is 333 us
# and the driver's input-lost watchdog is 10 ms.  CONTROL_ROADMAP says the
# same thing as a standing constraint -- "keep the EKF/MPC off-thread pattern
# and the CAN loop dumb" -- and torque_mode_control/torque_worker.py is the
# pattern this follows, publishing discipline included.
MPC_STALE_S = 3.0 / MPC_HZ           # 0.075 s.  Three missed solves.
# WHAT A STALE MPC MEANS, stated once.  In position mode a stale estimate
# leaves the robot standing, because the drivers hold their last target.  In
# TORQUE mode a frozen plan keeps pushing on a world model that has stopped
# updating -- and this plan carries a CONTACT SCHEDULE, so a frozen one keeps
# pushing with feet that have since left the ground.  The runner limps.

# ===========================================================================
# caps -- the same ones every torque runner on this robot has used
# ===========================================================================
TAU_MAX_DEFAULT = P.TAU_START_MAX    # 1.0 Nm.  First runs, supported robot.
TAU_STAGED_MAX = P.TAU_STAGED_MAX    # 3.0 Nm ceiling for --tau-max
FORCE_FRAC_DEFAULT = P.FORCE_FRAC_DEFAULT
STAND_HEIGHT = P.STAND_HEIGHT        # 0.152 m, floor to TRUNK BOTTOM
T_RISE = P.T_RISE                    # 8.0 s, and the rise is still week 2's
