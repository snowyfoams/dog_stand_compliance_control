#!/usr/bin/env python3
"""Every tunable in the torque-mode track, in one place.  Pure literals.

Same contract as `stand_postion_mode/stand_params.py`: no imports, so a test,
a plotting script or a notebook can read these without dragging in python-can,
MuJoCo or the IMU.  The one thing that cannot live here is the mass assertion
against the MJCF -- that needs an import, so it is in `dog5_statics.py`.

===========================================================================
READ THIS FIRST -- THE LOOP RATE.  A 12x error here cost this project a year
===========================================================================
    slot   = 1 / (CONTROL_HZ * N_JOINTS) = 1/(250*12) = 333 us
    sweep  = 12 slots                                 = 4 ms
    per-joint command AND reply rate                  = 250 Hz
    driver input-lost watchdog (CAN 1/7/8/9)          = 10 ms

    EVERY motor is commanded and replies at 250 Hz.  The 0xA1 torque reply
    carries temp/iq/speed/encoder in the same frame (motorbus.py:310-314), so
    commanding a joint IS sampling it.  The whole 12-vector refreshes at
    250 Hz, not 250/12.

    `250/12 = 20.8 Hz, 48 ms` appears in three places in this repo
    (vmc_stand_hw.py:105, stand_hier_hw.py:3-6, stand_dog5_hw.py:85) and is
    WRONG in all three.  It is what `slot` would be if a sweep took one full
    CONTROL_HZ period per motor instead of per robot.  On 2026-07-30 it was
    used to prove that "software torque cannot stabilise the legs" and the
    whole torque-mode track was abandoned on that basis.  At the true 4 ms:

        sampled-damper bound  kd < 2J/dt      knee 4.4, abd 7.35 Nms/rad
        (at the assumed 48 ms it was          knee 0.37, abd 0.61)

    so KD_JOINT_BRAKE = 0.4, declared "109% of the limit, unstable", actually
    had ~5x margin -- and a joint-space PD, which diverges at ANY useful gain
    at 48 ms, is comfortably stable at 4 ms up to kp ~40-80.  That is what
    makes the impedance underlay in `stance_law.py` possible at all.

    Do not re-derive 20.8 Hz.  test_torque_params.py gates on it.

TORQUE AT A GLANCE -- what may be commanded, and what stops it
    TAU_START_MAX           1.0 Nm    first-run cap; --tau-max default
    TAU_STAGED_MAX          3.0 Nm    staged ceiling (base.STAGED_TAU_MAX)
    TAU_HARD_NM             9.0 Nm    absolute (base.TAU_HARD); iq saturates
                                      at 2048/206.04 = 9.94 Nm, so 9.0 is the
                                      last value the current loop can honour
    TAU_SLEW_NM_S           60 Nm/s   torque-mode slew -- 12x the position
                                      track's 5.0, see the derivation below
    BRAKE_HOLD_NM          1.54 Nm    the latching brake's constant opposition

IMPEDANCE AT A GLANCE -- the floor under the force law
    KP_JOINT_IMP     15 Nm/rad    ~19% of the 4 ms stability bound
    KD_JOINT_IMP    0.6 Nms/rad   ~14% of the knee's 2J/dt = 4.4
    -> omega_n = sqrt(kp/J_knee) = 41 rad/s (6.6 Hz), zeta = 0.83
    -> omega_n * dt = 0.165, well inside the ~0.3 one-sample-delay limit

MASS AT A GLANCE
    DOG5_MASS_KG          5.8151 kg   trunk 2.6189 + 4 x 0.79905 of leg
    WEIGHT_N              57.05 N     what the four feet must sum to
    PER_FOOT_GRF_N        14.26 N     an even four-way split
    (stand_dog5_hw.DOG5_MASS_KG = 5.3 is 8.9% LOW and must not be used here)

RUN
    Nothing to run: pure literals, zero imports, no main.  That is the point
    -- a test, a notebook or a plotting script can read these without pulling
    in python-can, MuJoCo or the IMU.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    cd /home/robot01/Documents/can_motor_control/dog_stand_compliance_control/Aug_trotinplace_closedloop

    $V self-test/test_torque_params.py     # 38 gates, no hardware
      # These do not check VALUES (those are engineering choices).  They check
      # that the values still follow from their own justifications: the
      # stability bounds are RECOMPUTED from J and dt rather than copied, so a
      # gain and its reason cannot drift apart, and BRAKE_PERIOD_S is asserted
      # equal to 1/CONTROL_HZ with the 12x-error story in the failure message.

    To read a value without importing anything heavy:
    $V -c "import sys; sys.path.insert(0,'torque_mode_control'); \
           import torque_params as P; print(P.SWEEP_S, P.DOG5_MASS_KG)"
"""

# ===========================================================================
# loop timing -- see the header.  These are facts about the bus, not tunables
# ===========================================================================
CONTROL_HZ = 250.0           # per motor AND per sweep (see header)
N_JOINTS = 12
SWEEP_S = 1.0 / CONTROL_HZ   # 4 ms -- the period the whole 12-vector refreshes
SLOT_S = SWEEP_S / N_JOINTS  # 333 us -- one CAN frame's share
WATCHDOG_S = 0.010           # driver input-lost timeout on CAN 1/7/8/9
CONTROL_UPDATE_HZ = 100.0    # stance worker thread; the CAN loop holds the
                             # torque between its updates and keeps sweeping,
                             # so the watchdog is fed at 250 Hz regardless

# In-sweep compute budget.  The 250 Hz block runs BEFORE slot 0's frame goes
# out, so an overrun delays MOTOR_IDS[0..2] -- which is FL (7,8,9), part of
# the set with the 10 ms watchdog.  selftest_common gates at SLOT_BUDGET_S =
# 250 us; the impedance block should come in far under that.
INSWEEP_BUDGET_S = 60e-6

# ===========================================================================
# mass and statics -- facts, asserted against dog5.xml in dog5_statics.py
# ===========================================================================
# dog5.xml: trunk 2.61890103 + 4 x (hip 0.39152916 + thigh 0.36932519
#                                   + shin 0.0381955) = 5.8151004 kg.
# The legs are 55% of that, which is why leg-link gravity is NOT negligible
# in the stance torque -- see dog5_statics.leg_gravity_torque_tilted and the
# comment on STANCE_LEG_GRAVITY below.
DOG5_MASS_KG = 5.8151
GRAVITY_M_S2 = 9.81
WEIGHT_N = 57.05             # DOG5_MASS_KG * GRAVITY_M_S2
PER_FOOT_GRF_N = 14.26       # an even four-way split; the T0 gate's target

# Joint inertias INCLUDING the 10:1 reflected rotor (armature = 0.0085 kgm^2
# in dog5.xml, which dominates the links).  From vmc_stand_hw.py:116-118 and
# re-derived from the MJCF: knee = armature + shin about the knee axis
# (0.038 kg at 76 mm + its own 9.3e-5) = 0.0085 + 0.0003.
# These set every stability bound in this file, so T0 measures them rather
# than trusting them: if the T1 hold oscillates at fixed amplitude, THESE are
# wrong, not the loop rate.
J_KNEE = 0.0088
J_ABD = 0.0147
J_MIN = J_KNEE               # the binding one for every bound below

# ===========================================================================
# torque caps and the slew override
# ===========================================================================
TAU_START_MAX = 1.0          # --tau-max default: first runs, supported robot
TAU_STAGED_MAX = 3.0         # staged ceiling; matches base.STAGED_TAU_MAX
TAU_HARD_NM = 9.0            # absolute; matches base.TAU_HARD

# base.TAU_SLEW_NM_S = 5.0 is a POSITION-track number and it strangles a
# balance loop.  The gate's slew is wall-clock, so at 5 Nm/s a disturbance
# response is capped at 0.25 Nm per 50 ms: a kp=15 impedance at 0.1 rad of
# error wants 1.5 Nm and would take 0.3 s to reach it, by which time the leg
# has folded.  With base.TORQUE_RAMP_S = 1.0 on top, a leg cannot take weight
# for a full second after the gate starts.
#
# Size it in "how much joint speed can one sweep add" instead, which is the
# quantity that actually threatens anything:
#     dqd = TAU_SLEW * SWEEP_S^2 / J   ... per sweep, at the slew limit
#     60 Nm/s -> 0.24 Nm/sweep -> 0.24 * 0.004 / 0.0088 = 0.11 rad/s per sweep
# Bounded, an order of magnitude under QD_ESTOP = 7.0, and fast enough to
# answer a push inside 30 ms.
TAU_SLEW_NM_S = 60.0
TORQUE_RAMP_S = 0.3          # ease-in when a torque stage arms.  base's 1.0 is
                             # tuned for a position-mode preflight; here it is
                             # the window in which the legs have no authority.

# ===========================================================================
# joint impedance -- the floor under the force law (stance_law.JointImpedance)
# ===========================================================================
# WHY THIS EXISTS, and why it is not just "more damping":
#   A pure force law tau = -J^T f is a VELOCITY-level system.  With no ground
#   reaction under a foot, qdd integrates without bound and a damper only
#   sets the TERMINAL velocity qd = tau/kd -- the joint never stops, it
#   coasts.  That is exactly how RR_abd failed on 2026-07-30: it settled at
#   qd = tau/0.4 and ran into the overspeed e-stop.
#   Adding kp*(q_ref - q) makes it POSITION-level, with a fixed point.  The
#   runaway becomes a bounded offset dq = tau/kp -- 13 mrad at the corrected
#   abduction torque of 0.266 Nm and kp = 20.  There is nothing left to brake.
#
# WHERE q_ref COMES FROM matters as much as the gains: it is the IK of the
# SAME foot trajectory the wrench loop is trying to realise (the stage-2 z ->
# joint tables).  So in nominal stance this term is ~0 and the force law does
# the work -- true force control, visibly compliant.  It only bites when the
# force law's assumption "this foot is loaded" is false.  It is a
# MODEL-MISMATCH term, not a stiffness that fights the balance loop.
#
# NOT base.KP_JOINT_HW = 600 / KD_JOINT_HW = 90.  Those are past the sampled
# bound by a factor of 20 (kd*dt/J = 90*0.004/0.0088 = 41, against a limit of
# 2) and survive only because the 3 Nm cap and 5 Nm/s slew keep them
# saturated -- i.e. they are a bang-bang controller wearing PD clothing.
KP_JOINT_IMP = 15.0          # Nm/rad.  Bound at 4 ms is ~40-80; this is ~19%.
KD_JOINT_IMP = 0.6           # Nms/rad.  Bound is 2*J_MIN/SWEEP_S = 4.4; ~14%.
KP_IMP_MAX = 40.0            # refuse anything above this from the CLI --
KD_IMP_MAX = 1.5             # the measured stable envelope, not a preference
IMP_DQ_NOTICE_RAD = 0.02     # |q_ref - q| above this on the status line means
                             # the impedance is carrying load the force law
                             # thinks is on the ground

# ===========================================================================
# force law -- how much of the wrench is actually commanded
# ===========================================================================
# The bring-up dial: it scales the DISTRIBUTED GRF, which is the only term
# holding the TRUNK up.  1 = full VMC force, impedance only as a floor.
#
# 0 IS NOT A SAFE "PLUMBING TEST" WITH WEIGHT ON THE LEGS.  It was described
# that way and hardware disagreed on 2026-08-14: the leg-gravity feedforward
# carries each leg's own links and nothing else, so at force_frac 0 the trunk
# is supported only by the impedance's positional error.  The legs fold until
# kp*dq balances 57 N -- ~90 mrad per joint at kp=15 -- and the joints pick up
# enough speed on the way down to latch the runaway brake.  Seven joints
# latched, mostly rear, within 2 s of RISE.
#
# force_frac 0 is a valid isolation test ONLY with the robot on a stand and
# its feet in the air (which is what tau_calib_hw --hang already does better).
FORCE_FRAC_DEFAULT = 1.0
FORCE_FRAC_WARN_BELOW = 0.2   # the runner prints the sag estimate under this

# Whether the stance branch adds the leg's own link-weight torque.
# ---------------------------------------------------------------------------
# dog5_vmc_core.py:224-227 omits it, arguing the m*g in the wrench already
# routes total weight to the feet.  That conflates two different balances:
#   EXTERNAL (the wrench, correct):  sum f = M*g about the whole-body CoM.
#                                    Sets HOW BIG f is.
#   INTERNAL (omitted):              with the trunk as base, each joint
#                                    carries the foot GRF AND the weight of
#                                    every link distal to it.  Sets HOW f
#                                    MAPS TO tau.
# The omission is only correct for massless legs.  Ours are 55% of the mass.
# Checked against MuJoCo floating-base inverse dynamics (qfrc_bias - Jc^T f at
# qacc = 0): -J^T f alone is off by 0.482 Nm; -J^T f + leg gravity matches to
# machine precision.  As a fraction of |J^T f|, opposite in sign:
#     abduction 63-65%,  hip pitch 29-33%,  knee 2%
# --no-leg-gravity exists ONLY to reproduce the 2026-07-30 failure in the T2
# A/B.  It is not a tuning option.
STANCE_LEG_GRAVITY = True

# Config-dependent whole-body CoM for the grasp map's lever arms.  Third in
# priority behind leg gravity (0.48 Nm) and mass (0.09 Nm) -- worth ~0.02 Nm.
# It swings +24.6 mm (crouch) to -19.3 mm (stand) in z, but in grasp_map the
# lever enters as skew(r_w), and for a vertical force
# r x (0,0,fz) = (ry*fz, -rx*fz, 0) -- r_z drops out identically.  It only
# reaches the answer through TANGENTIAL forces (the kd_x/kd_y damping and the
# friction clamp).  x/y is 0.3 mm against a 340 mm foot half-spacing.
CONFIG_DEPENDENT_COM = True

# ===========================================================================
# VMC trunk gains -- the virtual spring/damper on the body
# ===========================================================================
# dog5_vmc_core.VMCGains defaults, tuned in MuJoCo where there was no slew
# limit, no ZOH and no one-sweep delay.  Height stiffness is dropped for the
# first hardware runs because kp_z is what a slew-limited actuator cannot
# honour: 1200 N/m at 10 mm of error asks for 12 N of extra support, and the
# gate can only add 0.24 Nm per sweep.
KP_Z = 800.0                 # N/m    (sim default 1200)
KD_Z = 120.0                 # Ns/m
KP_ROLL = 120.0              # Nm/rad
KD_ROLL = 8.0                # Nms/rad
KP_PITCH = 120.0
KD_PITCH = 8.0
KD_X = 60.0                  # Ns/m   damping ONLY on x, y, yaw: with no EKF
KD_Y = 60.0                  #        these come from leg odometry, and there
KD_YAW = 4.0                 #        is no absolute reference to be stiff to
KD_JOINT_VMC = 0.15          # Nms/rad, the core's own stance damping term

MU_FRICTION = 0.6            # tangential clamp in distribute_wrench
FZ_MIN_N = 1.0               # min per-foot normal force while in contact
GRASP_LAMBDA = 1.0e-3        # damped-LS regulariser on the grasp map
MIN_JAC_SINGULAR = 5.0e-3    # leg-Jacobian singularity guard

# ===========================================================================
# runaway brake -- last resort, below the impedance and above SafetyGate
# ===========================================================================
# Logic is vmc_stand_hw.RunawayBrake verbatim: it LATCHES rather than blending,
# so d(tau)/d(qd) = 0 within each state and it contributes no feedback gain at
# any sample rate.  That argument is still sound; it is simply no longer the
# ONLY option, because the premise that continuous feedback is unaffordable
# was the 12x error.
#
# The constants were all derived from BRAKE_PERIOD_S and are therefore all
# wrong as shipped.  vmc_stand_hw.py:148 has
#     BRAKE_PERIOD_S = N_JOINTS / CONTROL_HZ    -> 48 ms
# which is the error itself; the sweep is 1/CONTROL_HZ.  The knock-on:
#     BRAKE_HOLD_NM = 0.7 * J_MIN * BRAKE_RELEASE_RAD_S / BRAKE_PERIOD_S
#     at 48 ms -> 0.128 Nm ... less than an abduction joint's own leg weight
#                             (0.48 Nm), i.e. the shipped brake was a no-op
#     at  4 ms -> 1.54 Nm  ... actually able to stop a joint
BRAKE_PERIOD_S = SWEEP_S     # 4 ms.  NOT N_JOINTS / CONTROL_HZ.
BRAKE_LATCH_RAD_S = 2.0      # raised from 1.5: the impedance underlay should
                             # hold speeds far below this, so a latch now
                             # means something has genuinely gone wrong
BRAKE_RELEASE_RAD_S = 1.0    # hysteresis band 1.0 -> 2.0
BRAKE_DEADBEAT_FRAC = 0.7    # stay short of J*qd/dt; margin for J error
BRAKE_HOLD_NM = 1.54         # 0.7 * 0.0088 * 1.0 / 0.004
BRAKE_MIN_LATCH_S = 0.20     # minimum dwell before release is considered
BRAKE_RECOVER_S = 0.50       # time ramp back to full drive (a function of
                             # time, not qd, so it adds no velocity feedback)
BRAKE_MAX_TRIPS = 3          # repeated latching of one joint = a leg that is
                             # not bearing load; stop with that diagnosis
BRAKE_LATCH_TIMEOUT_S = 1.0  # ...or one latch that never comes back down

# ===========================================================================
# stage machine
# ===========================================================================
T_RISE = 8.0                 # crouch -> stand under torque (s).  Slower than
                             # the position track's T_STAND = 5.0: the force
                             # law has to hand load between legs as it goes.
T_CROUCH_SETTLE_S = 0.5      # native-position crouch must be still this long
WORKER_STALE_S = 0.2         # stance worker stopped publishing -> limp
STAND_HEIGHT_DEFAULT = 0.19  # floor to trunk bottom; same as the position
                             # track's default, for a like-for-like A/B

# ===========================================================================
# body state -- AHRS + leg odometry, no EKF (body_state_ahrs.py)
# ===========================================================================
# Stage 1 closes on the raw AHRS.  That is only sufficient BECAUSE all four
# feet stay planted: a planted foot is fixed in the world, so differentiating
# r + C^T s_i = const gives the trunk velocity directly,
#     v_world = -C^T (omega x s_i + J_i qd_i)
# averaged over the planted set, with qd read from the motors' own speed
# field.  No integration, no filter, nothing to drift.
#
# THIS STOPS BEING TRUE THE MOMENT A FOOT LIFTS.  With fewer than
# MIN_PLANTED_FOR_ODOM feet down the trunk velocity is no longer determined
# and the worker must freeze rather than publish a fiction.  Stage 3 (trot)
# is where the EKF becomes necessary again -- see the Aug README's
# "which estimate to trust for what".
MIN_PLANTED_FOR_ODOM = 3
AHRS_STALE_S = 0.2           # no fresh 0x41 packet this long -> limp
ODOM_LPF_FC_HZ = 20.0        # first-order low pass on the odometry velocity.
                             # The motors' speed field is quantised at 0.1
                             # dps; 12 of them summed is visible in kd_z.

# ===========================================================================
# safety -- torque mode specific
# ===========================================================================
TILT_STOP_DEG = 12.0         # absolute attitude that limps the run
LOAD_SUM_TOL_FRAC = 0.40     # |sum foot load - WEIGHT_N| beyond this -> limp.
                             # The single best end-to-end check that the force
                             # loop is real: it is measured iq, inverted
                             # through J^-T, and it must add up to the robot.
TAU_TRACK_MIN_FRAC = 0.80    # |tau_meas| / |tau_cmd| below this is a warning:
                             # the current loop is not delivering what the
                             # grasp map asked for, so nothing downstream of
                             # it means anything
GAP_ABORT_S = 0.008          # a command gap this long blocks RISE -> HOLD.
                             # Straddles the 10 ms watchdog with margin.
GAP_BUCKETS = (0.008, 0.010, 0.015, 0.020, 0.050)

# Latch response INVERTS between the two modes, and this is the single most
# important difference in the whole file:
#   position (0xA4): a latched motor holds its last commanded position.
#                    Benign.  base._recover_input_lost retries in place and
#                    that is the right call.
#   torque   (0xA1): a latched motor STOPS PRODUCING TORQUE.  That leg
#                    collapses while the other three keep pushing -- an active
#                    tip-over, not a stall.
# So in a torque stage a stance-leg latch limps the whole robot immediately;
# recovery-in-place is not attempted until everything is at zero torque.
LATCH_LIMPS_ROBOT = True

# LIMP is not the same as stopping.  Killing the process stops the keep-alive
# stream, which latches all twelve drivers with the robot in a heap -- in
# torque mode "exit" is a WORSE abort than "go limp".  SPACE zeroes torque but
# keeps the round-robin and the keep-alives running, and ENTER resumes.
KEY_LIMP = " "
KEY_STOP = "x"
KEY_PARK = "p"
