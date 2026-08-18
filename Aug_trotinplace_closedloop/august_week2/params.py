#!/usr/bin/env python3
"""Every number the week-2 torque stand has, in one place.  Pure literals.

No imports, so a plot script or a notebook can read these without dragging in
python-can, MuJoCo or the IMU.

WHY THIS FILE EXISTS AT ALL
    Before the July rewrite the gains were spread across six 600-1100 line
    runners and then aliased between them, and EIGHT of them were declared
    TWICE as independent literals -- so editing one copy silently left the
    other behind.  One file, one declaration, is the whole rule.

===========================================================================
READ THIS FIRST -- THE LOOP RATE.  A 12x error here cost this project a year
===========================================================================
    slot  = 1 / (CONTROL_HZ * N_JOINTS) = 1/(250*12) = 333 us
    sweep = 12 slots                                 = 4 ms
    per-joint command AND reply rate                 = 250 Hz
    driver input-lost watchdog (CAN 1/7/8/9)         = 10 ms

    EVERY motor is commanded and replies at 250 Hz.  The 0xA1 torque reply
    carries temp/iq/speed/encoder in the same frame, so commanding a joint IS
    sampling it.  The whole 12-vector refreshes at 250 Hz, not 250/12.

    `250/12 = 20.8 Hz, 48 ms` is WRONG and appears in three older files.  On
    2026-07-30 it was used to prove "software torque cannot stabilise the
    legs", and the entire torque track was abandoned on that basis.  At the
    true 4 ms the sampled-damper bound kd < 2J/dt is 4.4 Nms/rad on the knee,
    not 0.37 -- so a joint-space PD, which diverges at any useful gain at
    48 ms, is comfortably stable here.  That is what makes KP_IMP possible.

    Do not re-derive 20.8 Hz.
"""

# ===========================================================================
# loop timing -- facts about the bus, not tunables
# ===========================================================================
CONTROL_HZ = 250.0           # per motor AND per sweep (see header)
N_JOINTS = 12
SWEEP_S = 1.0 / CONTROL_HZ   # 4 ms -- the period the whole 12-vector refreshes
SLOT_S = SWEEP_S / N_JOINTS  # 333 us -- one CAN frame's share
WATCHDOG_S = 0.050           # driver input-lost timeout on CAN 1/7/8/9

# How often the heavy blocks run, in sweeps.  MEASURED on this Pi, not
# assumed (august_week2 bench, 300 calls each):
#
#     per sweep    impedance 4 us + gate 24 us            =   28 us
#     model block  body_wrench 9 + stance_torque 1109
#                  + warm-started IK 266                  = 1384 us
#     load block   foot_load_from_torque                  = 1029 us
#
# The per-sweep total is 8% of one 333 us slot, so the 250 Hz loop is
# untouched.  The two heavy blocks cannot run every sweep -- but they do not
# need a thread either, because each is well inside the 10 ms watchdog.  They
# run IN the sweep, and the sweep they land in is late by that much.
#
# THEY ARE STAGGERED SO THEY NEVER LAND IN THE SAME SWEEP.  Together they are
# 2.41 ms; apart, the worst any single sweep is delayed is 1.38 ms.  With
# LOAD_EVERY = 12 and LOAD_OFFSET = 1, the load block runs at sweeps 1, 13,
# 25, ... whose remainder mod 3 is always 1, so it can never coincide with the
# model block at 0, 3, 6, ...
#
#     MODEL_EVERY = 3   ->  model at 83 Hz, 1.38 ms late every 12 ms
#     LOAD_EVERY  = 12  ->  load  at 21 Hz, 1.03 ms late every 48 ms
#
# 21 Hz still catches a collapsing leg within 48 ms, long before it has moved
# far.  What this buys over last week: no worker thread, no mailbox, no atomic
# publishing, no staleness watchdog -- ~200 lines that existed only to make a
# thread safe.  What it costs: that jitter, on the FL frames (CAN 7/8/9).
#
# The feedback that actually stabilises a joint -- the impedance -- still runs
# EVERY sweep on qd at most 4 ms old.  Only the feedforward is held.
MODEL_EVERY = 3
LOAD_EVERY = 12
LOAD_OFFSET = 1              # keeps the two blocks in different sweeps

# ===========================================================================
# mass and statics -- facts, asserted against dog5.xml by dog5_statics
# ===========================================================================
# dog5.xml: trunk 2.61890103 + 4 x (hip 0.39152916 + thigh 0.36932519
#                                   + shin 0.0381955) = 5.8151004 kg.
# The legs are 55% of that, which is why leg-link gravity is NOT negligible
# in the stance torque -- see STANCE_LEG_GRAVITY below.
MASS_KG = 5.8151
GRAVITY = 9.81
# DERIVED, not written down.  Last week's file carried WEIGHT_N = 57.05 as its
# own literal, 0.0039 N away from MASS_KG * GRAVITY -- harmless in itself, but
# it is exactly the drift this file exists to prevent, and it makes "Fz equals
# the weight" untestable to better than the rounding.
WEIGHT_N = MASS_KG * GRAVITY          # 57.046 N
PER_FOOT_GRF_N = WEIGHT_N / 4.0       # 14.26 N -- an even four-way split
# (stand_dog5_hw.DOG5_MASS_KG = 5.3 is 8.9% LOW and must not be used here)

# Joint inertias INCLUDING the 10:1 reflected rotor (armature = 0.0085 kgm^2
# in dog5.xml, which dominates the links).  These set every stability bound
# below, so they are stated once and the bounds are derived from them.
J_KNEE = 0.0088
J_ABD = 0.0147
J_MIN = J_KNEE               # the binding one

# ===========================================================================
# torque caps and slew
# ===========================================================================
TAU_START_MAX = 1.0          # --tau-max default: first runs, supported robot
TAU_STAGED_MAX = 3.0         # the ceiling --tau-max may be raised to
TAU_HARD_NM = 9.0            # absolute; iq saturates at 2048/206.04 = 9.94 Nm

# base's 5.0 Nm/s is a POSITION-track number.  The justification here used to
# be "a kp=15 impedance at 0.1 rad wants 1.5 Nm and 5 Nm/s takes 0.3 s to get
# there" -- that argument DIED WITH kp = 15: at KP_IMP = 3 the same error
# wants 0.3 Nm, which 5 Nm/s reaches in 60 ms.  The number is kept at 60 on
# the second argument alone, which never depended on a gain.
#
# NOTE WHAT 60 COSTS.  A rate limiter does not prevent a limit cycle, it sets
# the AMPLITUDE: on the 2026-08-17 shake the describing function predicts
# 5.8 deg of joint motion at 60 Nm/s against 0.49 deg at 5, and 2.7-5.5 deg
# was measured.  With the impedance now stable that amplitude has nothing to
# set, which is why this stays -- but if a shake ever returns, 60 is what
# makes it visible rather than what causes it.  Size it by how much joint
# SPEED one sweep can add:
#     dqd = TAU_SLEW * SWEEP_S^2 / J = 60*0.004^2/0.0088 = 0.11 rad/s per sweep
# an order of magnitude under the 7.0 rad/s e-stop, and fast enough to answer
# a push inside 30 ms.
TAU_SLEW_NM_S = 60.0
TORQUE_RAMP_S = 0.3          # ease-in when torque arms; the window in which
                             # the legs have no authority.  base's 1.0 is
                             # tuned for a position-mode preflight.

# ===========================================================================
# joint impedance -- the 250 Hz floor under the force law
# ===========================================================================
# A pure force law tau = -J^T f is a VELOCITY-level system: take the ground
# away from a foot and qdd integrates without bound, while a damper only sets
# the terminal speed qd = tau/kd -- the joint coasts, it does not stop.  That
# is how RR_abd failed on 2026-07-30 (settled at 1.87 rad/s, overspeed e-stop).
# kp*(q_ref - q) makes it POSITION-level with a fixed point: the same fault
# becomes a bounded offset dq = tau/kp -- 67 mrad at the KP_IMP below, five
# times the 13 mrad this line used to quote at kp = 15.  Still bounded, which
# is the whole property; a softer floor is a wider offset, not a runaway.
#
# q_ref is the SAME pose the force law is trying to hold, so in nominal stance
# this term is ~0 and the force law does the work.  It is a MODEL-MISMATCH
# term, not a stiffness fighting the balance loop.

# THE BOUND IS SET BY THE LOOP DELAY, NOT THE COMMAND INTERVAL.
# ---------------------------------------------------------------------------
# This file used to derive every impedance bound from SWEEP_S alone, and that
# is the interval at which a command is SENT -- not the time from computing a
# torque to feeling it.  Cross-correlating tau_cmd against tau_meas in the
# 2026-08-17 logs gives +3 sweeps of pure delay on all twelve joints
# (correlation 0.86-0.96, actuator gain 0.76).  Three sweeps of transport plus
# the 4 ms command interval is 16 ms, 4x what the bounds assumed:
LOOP_DELAY_S = 0.016         # MEASURED, not budgeted.  Re-measure it if the
                             # bus rate, the driver firmware or CONTROL_HZ
                             # changes -- every number below is 2J/dt on it.
#
#     sampled-damper bound 2*J_MIN/dt      4.40 Nms/rad at 4 ms   (assumed)
#                                          1.10 Nms/rad at 16 ms  (real)
#
# At the old KD_IMP = 0.6 the damper plus kd_joint = 0.75 is 68% of the real
# bound rather than the 17% this file used to claim, and the kp = 15 loop
# crosses -180 deg at 9.1 Hz with |L| = 1.24.  That is unstable, and it is
# what shook the robot at 9-12 Hz on 2026-08-17 -- identically with
# --open-loop, which is what proved the fault was below the model.
#
# THE VALUES BELOW ARE THE ONES THAT STAND ON HARDWARE.  Verified 2026-08-18:
# --kp 3 --kd 0.1 holds open-loop steady where the old defaults shook.  The
# arithmetic agrees -- kp = 3 puts the joint loop at sqrt(kp/J)/2pi = 2.9 Hz,
# where 16 ms of delay is only 17 deg of phase, against 38 deg at the 6.6 Hz
# that kp = 15 gives.
KP_IMP = 3.0                 # Nm/rad.  2.9 Hz crossover; 17 deg of delay phase
KD_IMP = 0.1                 # Nms/rad.  With kd_joint, 23% of the 16 ms bound
#
# The CEILINGS ARE THE OLD DEFAULTS, deliberately.  They are the largest gains
# ever run on this hardware and they are KNOWN TO SHAKE, so the CLI may still
# reach them -- reproducing the 9-12 Hz limit cycle is a legitimate A/B -- and
# may not go past them.  40/1.5 was never a "measured stable envelope"; it was
# the 4 ms bound with a margin, and the margin was against the wrong number.
KP_IMP_MAX = 15.0            # = the old KP_IMP: A/B only, expect the shake
KD_IMP_MAX = 0.6             # = the old KD_IMP: 68% of the 16 ms bound
IMP_DQ_NOTICE_RAD = 0.1      # |q_ref - q| above this means the impedance is
                             # carrying load the force law thinks is grounded.
                             # WAS 0.02, which was 0.3 Nm at kp = 15; at kp = 3
                             # the same torque is 0.1 rad, so the threshold
                             # moved with the gain to keep meaning one thing

# ===========================================================================
# body wrench -- the virtual spring/damper on the trunk (Dynamic_Model)
# ===========================================================================
# Stiffness ONLY on {z, roll, pitch}.  x, y and yaw get damping only: with no
# EKF those come from leg odometry, and there is no absolute reference to be
# stiff to.  Height stiffness is below the sim default of 1200 because kp_z is
# what a slew-limited actuator cannot honour -- 1200 N/m at 10 mm of error
# asks for 12 N of extra support, and the gate can add 0.24 Nm per sweep.
# THESE ARE NOW THE VALUES THE ROBOT HELD, NOT THE VALUES IT SHOOK AT.
# ---------------------------------------------------------------------------
# The 2026-08-18 bring-up walked the outer loop up one gain at a time and every
# step is a log.  Those numbers only ever existed as command-line flags, so any
# run that did not type all six of them got the old defaults back -- which is
# how a runner built on this file shook again after the ladder had finished.
#
#   gain      was    now   verified by      why the old value failed
#   kp_z      800    300   s2_kpz300.npz    800/120 has 7.4 deg of phase margin
#   kd_z      120     40   s1_kdz40.npz     against the measured 20 ms outer
#                                           delay; 300/40 has 27
#   kp_roll   120     10   s3c_att20.npz    120 saturates this stance's 6.42 Nm
#   kp_pitch  120     10   s3c_att20.npz    roll capacity at 3.06 deg of tilt
#   kd_roll     8    0.5   s3c_att20.npz    kd_att 2 alone gave zero HOLD
#   kd_pitch    8    0.5   s3c_att20.npz    sweeps in 4.6 s (s3b_kdatt2.npz)
#   kd_x/y     60      0   never run > 0    same lagged odometry velocity as
#   kd_yaw      4      0   never run > 0    kd_z, and yaw authority is 100%
#                                           friction; neither was ever tested
#
# WHAT THE LOWER ATTITUDE GAINS COST, stated so it is not a surprise: at
# kp_roll = 10 a standing roll residual of ~1.2 deg does not get corrected --
# the loop commands 0.21 Nm of the 6.42 available.  That is the trade the
# hardware chose.  Raising it is a --kp-att experiment, not an edit here, until
# a run completes at the higher value and this table can name it.
KP_Z = 300.0                 # N/m
KD_Z = 40.0                  # Ns/m
KP_ROLL = 10.0               # Nm/rad
KD_ROLL = 0.5                # Nms/rad
KP_PITCH = 10.0
KD_PITCH = 0.5
KD_X = 0.0                   # Ns/m   damping-only axes, and never yet damped
KD_Y = 0.0
KD_YAW = 0.0                 # Nms/rad
KD_JOINT = 0.15              # Nms/rad, stance joint damping inside the law

# ===========================================================================
# force distribution (force_totorque)
# ===========================================================================
MU_FRICTION = 0.6            # tangential clamp per foot
FZ_MIN_N = 1.0               # min per-foot normal force while in contact
GRASP_LAMBDA = 1.0e-3        # damped-LS regulariser on the grasp map
MIN_JAC_SINGULAR = 5.0e-3    # leg-Jacobian singularity guard

# The bring-up dial: it scales the DISTRIBUTED GRF, i.e. the only term holding
# the TRUNK up.  1 = full force, impedance only as a floor.
#
# 0 IS NOT A SAFE "PLUMBING TEST" WITH WEIGHT ON THE LEGS.  It was described
# that way and hardware disagreed on 2026-08-14: leg gravity carries each leg's
# own links and nothing else, so at 0 the trunk is held up only by the
# impedance's positional error.  The legs fold until kp*dq balances 57 N --
# ~90 mrad per joint -- and seven joints latched within 2 s of RISE.
FORCE_FRAC_DEFAULT = 1.0

# Whether the stance branch adds each leg's own link-weight torque.
# Omitting it is only correct for MASSLESS legs; ours are 55% of the mass.
# Checked against MuJoCo floating-base inverse dynamics (qfrc_bias - Jc^T f at
# qacc = 0): -J^T f alone is off by 0.482 Nm; with leg gravity it matches to
# machine precision.  As a fraction of |J^T f|, opposite in sign:
#     abduction 63-65%,  hip pitch 29-33%,  knee 2%
STANCE_LEG_GRAVITY = True
# Config-dependent whole-body CoM for the grasp map's lever arms.  Worth
# ~0.02 Nm: r_z drops out of r x (0,0,fz) identically, and x/y swings only
# 0.3 mm against a 340 mm foot half-spacing.
CONFIG_DEPENDENT_COM = True

# ===========================================================================
# body state (feedback_estimator)
# ===========================================================================
# A planted foot is fixed in the world, so differentiating r + C^T s_i = const
# gives the trunk velocity directly:
#     v_world = -C^T (omega x s_i + J_i qd_i)
# averaged over the planted set, with qd read from the motors' own speed
# field.  No integration, no filter, nothing to drift.
#
# THIS STOPS BEING TRUE THE MOMENT A FOOT LIFTS.  Below MIN_PLANTED the trunk
# velocity is not determined and the estimator must refuse rather than publish
# a fiction -- in torque mode a plausible fiction drives real force.
MIN_PLANTED = 3
AHRS_STALE_S = 0.2           # no fresh packet this long -> refuse
FOOT_RADIUS_M = 0.020        # foot site is the CENTRE of the contact sphere

# WHICH POINT ON THE TRUNK "HEIGHT" MEANS -- and it is NOT the hip axis.
# ---------------------------------------------------------------------------
# dog5.xml puts all four hip bodies at z = 0, so FK's natural trunk origin is
# the HIP-AXIS plane.  Nothing physical is at that plane and a tape cannot
# reach it.  The IMU board sits on the trunk BOTTOM, 38 mm lower (hip axis
# 8.5 in, trunk bottom 7.0 in; tape-validated 2026-07-31, and the same
# constant stand_postion_mode/stand_params.py carries).
#
# Hardware 2026-08-17: this runner reported z = 191 mm while a ruler on the
# trunk bottom read ~160.  191 - 38 = 153, so the "31 mm error" was almost
# entirely the frame.  The position track moved to the trunk bottom in
# 3cdd17e ("the +38/-38 mm round trip is gone"); this track had not, and its
# STAND_HEIGHT comment still claimed the two defaults were comparable.
#
# From here on EVERY height the operator reads and the height loop controls is
# floor-to-TRUNK-BOTTOM.  The leg IK still works in the hip frame -- that is
# where the leg tables live -- and feedback_estimator.hip_from_imu is the ONE
# place that converts.
IMU_BELOW_TRUNK_ORIGIN_M = 0.038

# THE RESTING ATTITUDE OF THIS RIG, subtracted from every AHRS reading.
# ---------------------------------------------------------------------------
# The IMU is bolted to the trunk through a mechanical mount and that mount's
# tilt is in every sample.  feedback_estimator has always had the subtraction;
# stand_torque_Mode never passed it, so the setpoint was 0 and the mount tilt
# went straight into the loop.  Measured across five runs of 2026-08-17, in the
# WAIT stage (recorded crouch, on the floor, position mode, zero torque):
#
#     roll  -0.41 .. -0.89 deg     pitch  +0.39 .. +0.50 deg
#     within-run sigma 0.001-0.006 deg -- the AHRS is not noisy, it is BIASED
#
# With kp_roll = 120 a 0.5 deg offset is a 1.05 Nm standing roll moment, 16% of
# this stance's 6.4 Nm roll capacity, spent holding the robot off true level.
#
# THEY ARE NOW SPLIT, so the mount's share is a default rather than a flag.
# The five numbers above are (mount tilt + floor slope + encoder zeros + real
# lean) and no crouch reading can separate them.  On 2026-08-18 the trunk was
# set on the floor and LEVELLED against a spirit level -- with the trunk known
# level every other term is zero by construction, so what the AHRS still reads
# is the mount alone:
#
#     roll  -0.29 deg     pitch  +0.12 deg          <- mount tilt, measured
#
# That is 0.61 Nm of standing roll moment at kp_roll = 120, 9% of the 6.4 Nm
# roll capacity.  Small -- and the point of baking it in is not the 9%, it is
# that everything LEFT in a crouch reading is now a real, physical tilt the
# attitude loop is supposed to remove, instead of an unknown mixture.
#
# WHAT REMAINS IS NOT SMALL, AND IS NOT A SETPOINT'S JOB.  The 2026-08-17 HOLD
# logs (attitude gains zeroed, full force) sat at roll -0.99 and -2.04 deg in
# two runs of the same rig.  Take the mount out and that is 0.7-1.75 deg of
# real lean under load, varying run to run -- so it cannot be calibrated away,
# and a setpoint that hid it would be hiding the loop's actual job.  At
# kp_roll = 120 a 1.75 deg lean is 3.67 Nm, 57% of capacity, leaving 1.3 deg
# before the grasp map clamps; see KP_ROLL.
#
# The floor's own share is still unmeasured and still does not belong here:
# re-read in the crouch on each floor, and rp:d (ahrs minus the foot plane)
# is the part a setpoint must never absorb.
SETPOINT_ROLL_DEG = -0.29
SETPOINT_PITCH_DEG = 0.12

# WHERE qd COMES FROM.  Not the driver's speed field -- the ENCODER, finite
# differenced, exactly as stand_dog5_recorded_hw.EncoderVelocity does it.
# ---------------------------------------------------------------------------
# Hardware 2026-08-17: the trunk shook so badly the run never reached HOLD.
# The driver's speed field reported 8.1 rad/s on FL_abd while a finite
# difference of that same joint's encoder showed 0.31 -- a 26x glitch, not
# motion.  It reaches the force law through two paths, and both are large:
#
#   qd 8.1 rad/s -> leg odometry v = 344 mm/s -> kd_z*v = 24.3 N of phantom
#                   force, on a robot that weighs 57 N
#   qd 8.1 rad/s -> impedance -kd*qd = 4.86 Nm, saturating the 1.0 Nm cap
#                   with a torque that is entirely fictitious
#
# stand_dog5_hw already warns that this field produces nuisance readings (it
# records 5.9 rad/s), and the OLD Cartesian compliance controller -- the one
# that stands without shaking -- never reads it at all.  This track inherited
# the field from body_state_ahrs.py and inherited the shake with it.
#
# Encoder differencing is noisier in the quantisation sense (0.01 deg over a
# 4 ms sweep is 0.044 rad/s raw, 0.026 filtered) but it CANNOT glitch: the
# encoder is a position, and a position that has not moved differences to
# zero.  0.026 rad/s is 0.13 N through kd_z, against the 24.3 N above.
QD_ALPHA = 0.35              # same as EncoderVelocity's; ~17 Hz at 250 Hz

# Low pass on the leg-odometry VELOCITY, on top of the above.
# ---------------------------------------------------------------------------
# WAS 20 Hz, and at 20 Hz it was not a filter.  It is evaluated in the outer
# loop at CONTROL_HZ/MODEL_EVERY = 83 Hz, i.e. dt = 12 ms, so
#     alpha = 1 - exp(-2*pi*20*0.012) = 0.78
# -- it passed 78% of a spike in a single step, and the outer loop's Nyquist
# is only 42 Hz, so a 20 Hz corner is barely below it.  The number came from a
# file whose worker ran at 100 Hz and was never re-derived for this rate.
#
# At 5 Hz alpha is 0.31, and the height loop it feeds has a bandwidth of
# sqrt(kp_z/m)/2pi = 1.87 Hz -- so 5 Hz is still 2.7x above anything the
# controller can act on, and costs no authority.
ODOM_LPF_FC_HZ = 5.0

# ===========================================================================
# stage machine
# ===========================================================================
T_RISE = 8.0                 # crouch -> stand under torque (s).  Slower than
                             # the position track's 5.0: the force law has to
                             # hand load between legs as it goes.
# floor to TRUNK BOTTOM (m).  Was 0.19 in the HIP-AXIS frame, with a comment
# claiming it matched the position track's default "for a like-for-like A/B".
# It did not: after 3cdd17e the position track's target is at the trunk bottom,
# so the two 0.19s were 38 mm apart and that A/B was never like-for-like.
#
# The NUMBER moved with the frame and the POSE did not: 0.190 - 0.038 = 0.152
# commands exactly the same joint angles as before.  The position track's
# equivalent is stand_height 0.19 -> 0.19 + FOOT_RADIUS - IMU_BELOW = 0.172 at
# the trunk bottom, so this track still stands 20 mm lower; that gap is real
# and is now visible instead of hidden in a frame mismatch.
STAND_HEIGHT = 0.152

# ===========================================================================
# safety
# ===========================================================================
TILT_STOP_DEG = 12.0         # absolute attitude that limps the run
# |sum of measured foot load - WEIGHT_N| beyond this -> limp.  WITH TORQUE
# CALIBRATION DROPPED THIS IS THE ONLY END-TO-END PROOF THE FORCE LOOP IS
# REAL: it is measured iq, inverted through J^-T, and it must add up to the
# robot.  If it does not, nothing downstream means anything, and a good-looking
# attitude proves nothing.
LOAD_SUM_TOL_FRAC = 0.40
TAU_TRACK_MIN_FRAC = 0.80    # |tau_meas|/|tau_cmd| below this warns: the
                             # current loop is not delivering what was asked

# Latch response INVERTS between the two modes:
#   position (0xA4): a latched motor holds its last commanded position. Benign.
#   torque   (0xA1): a latched motor STOPS PRODUCING TORQUE.  That leg
#                    collapses while the other three keep pushing -- an active
#                    tip-over, not a stall.
# So in a torque stage a stance-leg latch limps the whole robot immediately.
LATCH_LIMPS_ROBOT = True

# LIMP is not the same as stopping.  Killing the process stops the keep-alive
# stream, which latches all twelve drivers with the robot in a heap -- in
# torque mode "exit" is a WORSE abort than "go limp".  SPACE zeroes torque but
# keeps the round-robin running, and ENTER resumes.
KEY_LIMP = " "
KEY_STOP = "x"
KEY_PARK = "p"
