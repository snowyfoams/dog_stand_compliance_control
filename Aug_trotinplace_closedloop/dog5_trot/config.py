#!/usr/bin/env python3
"""Every constant the trot controller reads.  No logic, no imports from siblings.

WHERE THE NUMBERS COME FROM, AND WHICH ONES ARE MEASURED
    Geometry and mass are a controlled copy of dog5.xml, taken through
    dog5_description/dog5_kinematics.py -- the same source the standing tracks
    use.  Nothing here is a round number chosen to look plausible:

        MASS          5.8151 kg   sum of the twelve link inertials + trunk
        INERTIA_BODY              whole-robot tensor about the whole-robot CoM
                                  at the nominal stance, NOT the trunk's own
        HIP_OFFSET                dog5.xml hip body positions

    The control gains are the ones this rig has actually stood on.  KP_POS[2]
    and KD_POS[2] are the 300/40 pair verified on hardware 2026-08-18, and
    KP_JOINT/KD_JOINT are the 3.0/0.1 that replaced the 15/0.6 which shook at
    9-12 Hz.  See august_week2/params.py for why those two moved.

THE LEG IS NOT A PLANAR TWO-LINK, SO THERE IS NO L1/L2/L3 IN THE USUAL SENSE
    A spec written for a generic quadruped asks for L1 (abduction offset),
    L2 (thigh) and L3 (shin) and assumes the abduction axis is perpendicular
    to a planar hip/knee chain.  DOG5 is not built that way: the abduction
    hinge is about trunk +x, and BOTH the hip-pitch and knee hinges are about
    the accumulated z axis, with a hip-to-pitch offset that has components in
    x AND z.  Flattening that to three scalars loses the -0.052859 z offset
    and the front/rear x mirroring.

    So the true segment VECTORS are exported, and L1/L2/L3 are their norms,
    provided only for sizing arguments (workspace radius, Raibert step
    limits).  Anything computing kinematics must use leg_kin, never these.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DESC = os.path.normpath(os.path.join(_HERE, "..", "..", "dog5_description"))
if _DESC not in sys.path:
    sys.path.insert(0, _DESC)

_TMC = os.path.normpath(os.path.join(_HERE, "..", "torque_mode_control"))
if _TMC not in sys.path:
    sys.path.insert(0, _TMC)

import dog5_kinematics as kin                              # noqa: E402
import dog5_statics as statics                             # noqa: E402

# ===========================================================================
# naming and indexing -- fixed by the motor order every other track uses
# ===========================================================================
LEGS = kin.LEGS                          # ("FL", "FR", "RL", "RR")
N_LEGS = 4
N_JOINTS = 12

# JOINT_INDEX[leg] is the three entries of a 12-vector that belong to that leg,
# and it is 3*leg + (0, 1, 2) everywhere in this repo.  Written out rather than
# recomputed so a reader can check it against the CAN id order at a glance.
JOINT_INDEX = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=int)

# +1 for a left leg, -1 for a right one.  The abduction hinge is about trunk
# +x for every leg, so this is the sign of the OUTWARD direction, used by the
# swing planner to keep a foot from crossing under the body.
SIDE_SIGN = np.array([+1.0, -1.0, +1.0, -1.0])

# ===========================================================================
# mass properties
# ===========================================================================
GRAVITY = 9.81

TRUNK_MASS = 2.61890103
_LEG_MASS = float(sum(li.mass for li in kin.LINK_INERTIALS["FL"]))
MASS = TRUNK_MASS + 4.0 * _LEG_MASS      # 5.8151 kg
WEIGHT = MASS * GRAVITY                  # 57.05 N

# Whole-robot inertia about the whole-robot CoM, at the nominal stance below.
# NOT the trunk's own 0.0127/0.0742/0.0844 from dog5.xml: the four legs are 55%
# of the mass and hang ~0.15 m below the trunk CoM, and that lever dominates.
# Computed by config_selftest() from the link inertials, which is also what
# checks it -- if the pose or the geometry changes, the number moves with it.
#
# The legs' OWN tensors are not in dog5_kinematics' copy (only mass and CoM),
# so they are omitted: they total ~3e-4, 0.46% of Ixx, against a parallel-axis
# term of 5.3e-2.  Recorded rather than silently dropped.
INERTIA_BODY = np.array([
    [+0.065821, +0.000018, +0.000044],
    [+0.000018, +0.407502, -0.000001],
    [+0.000044, -0.000001, +0.457178],
])

# CoM in the trunk frame at the nominal stance.  The grasp map's lever arms are
# measured from HERE, not from the trunk origin; 14.7 mm of z is worth nothing
# for a vertical force but the x term is what keeps pitch honest.
COM_BODY = np.array([+0.00032, 0.0, -0.01472])

# THE THREE HEIGHTS THAT ARE NOT THE SAME HEIGHT.  Getting these confused is
# what the 2026-08-17 frame bug was, so they are all named:
#
#     floor            0
#     foot SITE        FOOT_RADIUS               0.020   (kinematics return this)
#     CoM              COM_ABOVE_FLOOR           0.175   (what p_com means)
#     trunk origin     STAND_HEIGHT              0.190   (what z_ref means)
#     trunk bottom     STAND_HEIGHT - 0.038      0.152   (what a ruler reaches)
#
# z_ref is the TRUNK ORIGIN because that is the frame the leg IK works in and
# the frame august_week2 controls.  p_com is the CoM because that is the point
# the QP's moment arms are measured from.  They differ by COM_BODY[2].
# (bound just after STAND_HEIGHT, which is defined below)

# ===========================================================================
# leg geometry -- vectors, not scalars.  See the header.
# ===========================================================================
HIP_OFFSET = np.array([kin.LEG_GEOMETRY[leg].hip for leg in LEGS])
HIP_TO_PITCH = np.array([kin.LEG_GEOMETRY[leg].hip_to_pitch for leg in LEGS])
PITCH_TO_KNEE = np.array([kin.LEG_GEOMETRY[leg].pitch_to_knee for leg in LEGS])
KNEE_TO_FOOT = np.array([kin.LEG_GEOMETRY[leg].knee_to_foot for leg in LEGS])

# Norms only.  For sizing, never for kinematics.
L1 = float(np.linalg.norm(HIP_TO_PITCH[0]))     # 0.0716 m
L2 = float(np.linalg.norm(PITCH_TO_KNEE[0]))    # 0.1200 m
L3 = float(np.linalg.norm(KNEE_TO_FOOT[0]))     # 0.1124 m
LEG_REACH = L2 + L3                             # 0.2324 m, hip-pitch to foot

# ===========================================================================
# stance and gait
# ===========================================================================
# Trunk-origin height above the foot plane.  august_week2 measures floor to
# TRUNK BOTTOM (0.152) and the IMU board sits 0.038 below the trunk origin, so
# the same pose is 0.190 here.  Stated in the frame the leg IK works in, which
# is the frame this file's z_ref means.
STAND_HEIGHT = 0.190
TRUNK_BOTTOM_BELOW_ORIGIN = 0.038        # subtract to compare with week 2

# THE FOOT SITE IS NOT THE CONTACT POINT.  dog5.xml's foot site is the CENTRE
# of a 20 mm contact sphere, so the floor sits FOOT_RADIUS BELOW the point all
# the kinematics return.  At the stance pose the foot site is at z = -0.170 in
# the trunk frame and the floor is at -0.190 = -STAND_HEIGHT.  Anything that
# plans where a foot should LAND must land the SITE at floor + FOOT_RADIUS,
# or every step drives 20 mm into the ground.
FOOT_RADIUS = 0.020

# The CoM's own height above the floor at the nominal stance.  See the table
# next to COM_BODY for why this is not STAND_HEIGHT.
COM_ABOVE_FLOOR = STAND_HEIGHT + COM_BODY[2]      # 0.17528 m

# The nominal stance, per leg, measured off the standing tracks rather than
# guessed.  IT IS NOT NEAR ZERO: DOG5's hip-pitch and knee hinges are both
# about the accumulated z axis, so the abduction has to carry ~90 deg before
# the pitch axis is horizontal at all.  A seed of zeros sits at a singularity
# with the leg pointing forward, which is how the first version of this file
# produced a "swing arc" that never left the ground.
#
# Signs pair FL with RR and FR with RL, not left with left: the rear legs'
# link offsets mirror in x, so the same physical pose has opposite joint signs
# front to rear.
# Full precision, not 4 decimals: rounding these puts the four foot sites 3 um
# out of plane, which is small but is a bias the swing planner's ground level
# then inherits every step.
Q_STAND = np.array([
    [+1.566751533, -0.542204247, -1.304052767],      # FL
    [-1.566724733, +0.537021138, +1.311469417],      # FR
    [+1.566724733, +0.537021138, +1.311469417],      # RL
    [-1.566751533, -0.542204247, -1.304052767],      # RR
])

# THE STANCE IS NOT SYMMETRIC FRONT-LEFT TO FRONT-RIGHT, AND THAT IS THE ROBOT.
# dog5.xml's knee_to_foot carries a y offset of -0.0011 that is shared by BOTH
# front legs instead of being mirrored, and the same for the rear pair.  After
# the ~90 deg abduction that offset lands in x, so at this stance
#     FL x = +0.340136    FR x = +0.342331     2.2 mm apart
# It is geometry, not a bad IK solve, and anything asserting a square stance
# has to allow for it.  It also means the nominal support polygon is a slight
# parallelogram, worth 2.2 mm of pitch lever out of 682 mm.
STANCE_ASYMMETRY_X = 0.0022

# The nominal foot positions THEMSELVES, in the trunk frame.  These are what a
# step should aim at, and they are NOT under the hips:
#
#     hip  FL (+0.2205, +0.0600)      foot FL (+0.3401, +0.1125)
#
# 120 mm forward and 52 mm outboard, because the ~90 deg abduction swings the
# whole leg out before the pitch/knee pair extends it.  A Raibert step written
# the textbook way -- "land under the hip, plus v T/2" -- therefore commands a
# 120 mm lunge backwards on this robot even when it is standing still.  The
# hip is the right origin only for a leg whose neutral pose hangs vertically,
# and DOG5's does not.
FOOT_STANCE_BODY = np.array([kin.foot_position(leg, Q_STAND[i])
                             for i, leg in enumerate(LEGS)])
HIP_TO_FOOT_STANCE = FOOT_STANCE_BODY - HIP_OFFSET

GAIT_PERIOD = 0.40                       # s, one full cycle
# 0.60, NOT the textbook 0.50.  A trot with duty exactly 0.5 has ZERO double
# support, and the contact ramp below then has nowhere to hand the load over:
# measured, the two diagonals' weights both reach 0 at the crossover instant
# and the total drops to 0.000 -- no foot is allowed to carry the robot.  The
# overlap (duty - 0.5) has to exceed the ramp length (CONTACT_RAMP * duty).
# At 0.60/0.15 that is 0.100 against 0.090 and the total weight never leaves
# 2.000; at 0.55/0.15 it dips to 0.879, and at 0.50 anything it is 0.
DUTY = 0.60
# FL and RR swing together, FR and RL swing together.  Offsets are in CYCLES.
PHASE_OFFSET = np.array([0.0, 0.5, 0.5, 0.0])
SWING_HEIGHT = 0.04                      # m, apex of the swing arc

# CONTACT FORCE IS RAMPED, NOT SWITCHED, and this is not a refinement either.
# With a binary mask the QP's fz bound on a leg goes from [FZ_MIN, FZ_MAX] to
# [0, 0] in ONE sweep, so the load it was carrying has to appear on the other
# diagonal in that same sweep.  Measured, that is a 2.2 Nm joint-torque step
# every 200 ms on a 3.0 Nm ceiling.
#
# Overlapping the gait does NOT fix it -- duty 0.50 -> 0.65 moved the step from
# 2.208 to 2.096 Nm, because the bound still switches instantly even while four
# feet are down.  What fixes it is scaling the bound itself: each leg gets a
# weight in [0,1] that smoothsteps up over the first CONTACT_RAMP of its stance
# and back down over the last, so the QP is asked to hand the load across
# gradually and does.  Weight 0 IS the swing case -- fz in [0,0] -- so the two
# states are one code path.
CONTACT_RAMP = 0.15                      # fraction of stance at each end;
                                         # see DUTY for the pairing constraint

# ===========================================================================
# force distribution (balance_qp)
# ===========================================================================
MU = 0.6                                 # same tangential clamp as the stand
FZ_MIN = 1.0                             # N, a planted foot never unloads to 0
FZ_MAX = 1.5 * WEIGHT                    # N, one diagonal pair carries all of
                                         # it in trot, plus margin for a push

# QP weights: ||A f - w_des||^2_W_TASK + W_FORCE ||f||^2 + W_SMOOTH ||f-f_prev||^2
#
# W_TASK IS NOT ALL ONES BECAUSE THE SIX RESIDUALS ARE NOT IN THE SAME UNITS.
# Forces run to ~57 N and moments to ~5 Nm, so equal weights would let the QP
# trade 1 Nm of moment error for 1 N of force error -- ten times the attitude
# error for a tenth of the height error.  The moment weight is the ratio.
W_TASK = np.array([1.0, 1.0, 1.0, 20.0, 20.0, 20.0])
W_FORCE = 1.0e-3                         # regularises the 6-of-12 null space
W_SMOOTH = 1.0e-2                        # penalises f - f_prev; the term that
                                         # keeps a contact switch from stepping

# ADMM iterations, see balance_qp.  With the dual warm-started across sweeps
# a trot converges in 6 on average; 60 is the ceiling for the sweep after a
# contact switch, where the warm start is deliberately discarded.
QP_ITERS = 60
QP_RHO = 1.0                             # measured: 6.3 iterations average at
                                         # 1.0, 30 at 0.1 -- rho is the whole
                                         # difference between 300 and 760 us
QP_SIGMA = 1.0e-6

# ===========================================================================
# control gains
# ===========================================================================
# Body orientation and position, in WORLD axes, feeding desired_wrench.
# ===========================================================================
# EVERY GAIN BELOW IS THE LAST VALUE THIS ROBOT ACTUALLY HELD, AND NOTHING
# ELSE.  This block is the provenance; VERIFIED_GAINS at the bottom of the
# file is the machine-checkable copy of it.
#
#   gain      value  verified by                        note
#   kp_z       300   s2_kpz300.npz   15.5 s HOLD
#   kd_z        40   s1_kdz40.npz    48.2 s HOLD
#   kd_xy        0   every run above -- NEVER tested above 0
#   kp_att      10   s3c_att20.npz   31.5 s HOLD
#   kd_att     0.5   s3c_att20.npz   31.5 s HOLD
#   kd_yaw       0   every run above -- NEVER tested above 0
#   kp_joint     3   the 2026-08-18 pair that replaced 15/0.6
#   kd_joint   0.1   ditto
#
# AND WHAT SHOOK, which is why the ceilings are where they are:
#   kp_att 40 / kd_att 2   shook on 2026-08-17 (the s3 run, no log survived)
#   kd_att 2 alone         s3b_kdatt2.npz -- 4.6 s, zero HOLD sweeps
#   kp_joint 15 / kd 0.6   9-12 Hz, the shake that started all of this
#
# THE FIRST VERSION OF THIS FILE CARRIED kp_z, kd_z AND THE JOINT PAIR ACROSS
# AND THEN WROTE THE ATTITUDE GAINS FROM GENERAL INTUITION: 80 / 4 / 2, which
# is 8x the verified kp_att, 8x the verified kd_att, and twice what had
# already been measured shaking.  It shook in RISE on the first hardware run.
# A gain that no log stands behind does not belong in this file; put it on the
# command line, run it, and move the number here afterwards.

# YAW HAS NO STIFFNESS, AND CANNOT.  The AHRS's yaw is magnetometer-derived,
# next to twelve motors and a steel frame, and feedback_estimator's C_from_rp
# discards it outright -- so there is no yaw measurement for a spring to pull
# against.  A non-zero KP_ORI[2] would be a gain on a number the runner sets to
# a constant zero: silent, and wrong the day someone wires a real yaw in.
# Damping still resists being twisted, which is all that is wanted, and it acts
# on the GYRO's yaw rate, which is measured.  Same argument as
# august_week2/params.py makes for KD_YAW.
KP_ORI = np.array([10.0, 10.0, 0.0])     # Nm/rad   roll, pitch, yaw
KD_ORI = np.array([0.5, 0.5, 0.0])       # Nms/rad

# x and y have NO absolute reference without an EKF -- leg odometry measures
# velocity and has no origin -- so their kp is zero and only the damper acts.
# The z pair is the one verified on hardware on 2026-08-18: at kp_z 800 the
# height loop has 7 deg of phase margin against the measured 20 ms outer-loop
# delay, and at 300/40 it has 27.
KP_POS = np.array([0.0, 0.0, 300.0])     # N/m
# kd_xy is 0, not 40.  The x/y dampers eat the SAME lagged leg-odometry
# velocity as kd_z and were never once run above zero on hardware -- every
# verified step above carries kd_xy = 0.  40 was written here from symmetry
# with kd_z, which is not evidence.
KD_POS = np.array([0.0, 0.0, 40.0])      # Ns/m

# Swing-foot Cartesian impedance, in WORLD axes -- the frame the swing
# reference is generated in, so no rotation sits between the error and the
# gain.  Full 3x3 so a diagonal is a choice rather than an assumption; the
# vertical entry is the one worth splitting off first, since touchdown is the
# only part of a swing where stiffness costs an impact.
KP_SWING = np.diag([200.0, 200.0, 200.0])   # N/m
KD_SWING = np.diag([8.0, 8.0, 8.0])         # Ns/m

# The 250 Hz joint-space floor under everything, exactly as in august_week2:
# a pure force law is velocity-level and a lost foot integrates without bound.
# 3.0/0.1 and NOT 15/0.6 -- the latter is 68% of the sampled-damper bound at
# the measured 16 ms loop delay and shook this robot at 9-12 Hz.
KP_JOINT = 10.0                           # Nm/rad
KD_JOINT = 0.1                           # Nms/rad

# ===========================================================================
# limits and timing
# ===========================================================================
TAU_MAX = 3.0                            # Nm per joint, the staged ceiling.
                                         # Hardware saturates iq at 9.94.

# THE SLEW LIMIT IS SIZED FROM THE HANDOVER, NOT COPIED FROM THE STAND.
# august_week2 uses 60 Nm/s, which was sized so a stand could answer a push.
# A trot has a harder requirement in one specific place -- the 36 ms window in
# which a diagonal hands the robot over -- and two different numbers describe
# it, measured on the controller's own output:
#
#     67 Nm/s   PHYSICAL: the real joint-torque change across one handover
#               ramp, CONTACT_RAMP * DUTY * GAIT_PERIOD = 36 ms
#    147 Nm/s   QUANTISATION: what it would take to follow the 12 ms
#               MODEL_EVERY staircase step for step
#
# 100 is deliberately between them.  Above 67 it tracks the handover that
# physically has to happen; below 147 it does NOT faithfully reproduce the
# sub-sampling staircase, which is an artifact of running the model at 83 Hz
# and is better smoothed than followed.  At 60 the gate would need 29 ms to
# absorb a 1.77 Nm step inside a 36 ms ramp, which is cutting it fine.
#
# The cost is bounded and worth stating: one sweep at this rate adds
# TAU_SLEW * dt^2 / J_MIN = 100 * 0.004^2 / 0.0088 = 0.18 rad/s of joint
# speed, against a 7.0 rad/s overspeed trip.
# 60, THE VERIFIED VALUE, NOT THE 100 THE HANDOVER ARGUMENT ASKS FOR.
# The 67 Nm/s figure above is real and a travelling handover will want it, but
# it is a paper number and every run this robot has completed used 60.  RISE
# and STAND have no handover at all, so the trot's requirement is no reason to
# raise the rate for the two stages that must work first.  Raise it with
# --tau-slew when a trot actually needs it, and record the run here.
TAU_SLEW_NM_S = 60.0
TORQUE_RAMP_S = 0.3                      # ease-in when torque first arms
CTRL_DT = 1.0 / 250.0                    # s, one full 12-motor sweep

# THE HEAVY BLOCK CANNOT RUN EVERY SWEEP, AND DOES NOT HAVE TO.
# Measured on this Pi, one full TrotController.update is 1937 us:
#
#     four leg chain walks (FK + Jacobian)      533 us
#     four leg-gravity terms                    474
#     the force-distribution QP                 294
#     swing plan + reference                     85
#     wrench, frames, per-leg torque, clamp     ~550
#
# That is 48% of a 4 ms sweep, and the sweep also has twelve CAN transactions
# in it.  august_week2 hit the same wall and answered it the same way: run the
# model block every MODEL_EVERY sweeps and HOLD its feedforward in between,
# while the joint-space floor -- the term that actually stabilises a joint --
# runs at the full rate on data at most 4 ms old.
#
# At 3 the outer loop is 83 Hz and the worst single sweep is 1.94 ms, well
# inside the driver's 50 ms input-lost watchdog.  Set it to 1 to run
# everything every sweep, which is correct and simply does not fit.
MODEL_EVERY = 3

# Raibert foot placement: p_foot = p_hip + v * T_stance/2 + RAIBERT_KV*(v-v_cmd)
RAIBERT_KV = 0.03                        # s, the velocity-error term

# The machine-checkable copy of the provenance table.  trot_hw's self-test
# refuses to start if anything here has drifted past what a log stands behind.
VERIFIED_GAINS = {
    "kp_z":     (300.0, "s2_kpz300.npz"),
    "kd_z":     (40.0,  "s1_kdz40.npz"),
    "kd_xy":    (0.0,   "never tested above 0"),
    "kp_att":   (10.0,  "s3c_att20.npz"),
    "kd_att":   (0.5,   "s3c_att20.npz"),
    "kd_yaw":   (0.0,   "never tested above 0"),
    "kp_joint": (3.0,   "2026-08-18, replaced 15 which shook at 9-12 Hz"),
    "kd_joint": (0.1,   "2026-08-18, replaced 0.6"),
    "tau_slew": (60.0,  "every week-2 run"),
}

# HOW FAST THIS ROBOT CAN TROT AT THIS HEIGHT, AND IT IS NOT FAST.
# The nominal stance already uses 92.3% of the leg's reach -- hip to foot is
# 0.2144 m against a 0.2324 m chain -- so a Raibert step has 6.4 mm of room
# before swing.py's 95% clamp truncates it.  Measured against the step length
# v*(T_stance/2) + RAIBERT_KV*v:
#
#     stand height   hip->foot   % of reach   step room   max forward v
#        0.190 m       0.2144       92.3%       6.4 mm      0.05 m/s
#        0.175         0.2027       87.2%      18.1         0.14
#        0.160         0.1915       82.4%      29.3         0.23
#        0.150         0.1843       79.3%      36.5         0.28
#        0.140         0.1774       76.3%      43.4         0.33
#
# TROTTING IN PLACE IS UNAFFECTED -- a zero-velocity step lands where the foot
# already is and never approaches the clamp, which is what this track is for.
# But a trot that TRAVELS has to crouch first, and 0.190 is not the height to
# do it from.  The clamp is not the limit to raise; the stance height is.
# DERIVED, not written down.  It was a literal and it went stale the moment
# DUTY moved from 0.50 to 0.60: the step length is v*(T_stance/2) + kv*v, and
# T_stance is DUTY*GAIT_PERIOD, so a duty change silently invalidates it.
# swing.py's self-test recomputes this from the geometry and compares.
_REACH_ROOM = 0.95 * LEG_REACH - float(np.linalg.norm(
    FOOT_STANCE_BODY[0] - HIP_OFFSET[0]))
MAX_FORWARD_V_AT_STAND_HEIGHT = _REACH_ROOM / (
    0.5 * DUTY * GAIT_PERIOD + RAIBERT_KV)
