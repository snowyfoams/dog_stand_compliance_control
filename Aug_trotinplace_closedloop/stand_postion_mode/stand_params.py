#!/usr/bin/env python3
"""Every tunable in the Aug track, in one place.  Pure literals -- no imports.

WHY THIS EXISTS
    The numbers were spread across six 600-1100 line runners and then aliased
    between them, so finding one meant following a chain:

        stand_ahrs_level_hw.SETTLE_S -> lv.SETTLE_S -> s2.SETTLE_S -> 1.5

    Worse, several were declared TWICE as independent literals -- `T_STAND`
    (5.0), `SETTLE_S` (1.5), `EKF_WORKER_HZ` (100.0), `QUIET_STAGES`,
    `STAND_HEIGHT_DEFAULT` (0.19), `TILT_STOP_DEG` (12.0) and the two frame
    constants -- so editing stage 1's copy silently left stage 2's behind.
    Nothing caught that; the values simply drifted apart on the next edit.

    This file is now the ONLY place any of them is written down.  Nothing
    imports anything here, so it is safe to import from a test, a plotting
    script or a notebook without dragging in python-can or the IMU.

SLEW LIMITS AT A GLANCE -- the caps that decide how fast anything moves
    HEIGHT_SLEW_M_S     5 mm/s    common foot z, the trunk-height loop
    LEVEL_SLEW_M_S      4 mm/s    differential foot z, EKF leveling
    AHRS_SLEW_M_S      10 mm/s    differential foot z, AHRS-only leveling
    LIFT_SLEW_M_S       8 mm/s    the deliberate one-foot lift
    TROT_SLEW_M_S      20 mm/s    the gait's swing -- the fastest of them
    (for reference, the STAND ramp itself moves the feet at ~33 mm/s average:
     165 mm of travel in T_STAND = 5 s)

HEIGHTS AT A GLANCE -- all floor-to-TRUNK-BOTTOM unless it says otherwise
    STAND_HEIGHT_DEFAULT       0.19 m   stages 2b/A/lift (reach headroom)
    STAND_HEIGHT_STAGE12       0.20 m   stages 1/2 (recorded default)
    TROT_STAND_HEIGHT_M        0.17 m   stage 3 (reach for the stance push)
    FOOT_RADIUS_M             0.020 m   foot site -> floor
    IMU_BELOW_TRUNK_ORIGIN_M  0.038 m   hip axis -> IMU board / trunk bottom

GAINS AT A GLANCE
    HEIGHT_GAIN_PER_S   0.5 /s    integral, common z          tau ~2 s
    LEVEL_GAIN_PER_S   0.25 /s    integral, differential z    tau ~4 s
    AHRS_KP             1.0       proportional, experiment A
    AHRS_GAIN_PER_S     1.0 /s    integral, experiment A      tau = (1+kp)/ki

WHERE THE SHARED *FUNCTIONS* LIVE (they are defined once and aliased, so the
chain is intentional -- this is the map, not a duplicate)
    fk_floor_height          stand_ekf_verify_hw   (stage 1 owns the frame)
    fk_floor_height_planted  stand_ekf_level_hw    (planted-aware version)
    fk_attitude              stand_ekf_level_hw
    LevelingLoop, SetpointLatch, LevelStandSequence, level_inputs
                             stand_ekf_level_hw    (the control laws)
    HeightController, build_tables, height_inputs
                             stand_ekf_height_hw
    EkfShared, ekf_worker    ../state_estimator/ekf_runtime.py
    test doubles + harness   self-test/selftest_common.py
"""

# ===========================================================================
# rig geometry -- facts about the robot, not tunables
# ===========================================================================
# Tape-validated 2026-07-31 (see vmc/stand_hier_hw.py).
#   * dog5_kinematics.foot_position() returns the foot SITE = the centre of a
#     20 mm contact sphere, so the floor is FOOT_RADIUS_M below it.
#   * The FK trunk origin is the HIP-AXIS plane (dog5.xml puts all four hip
#     bodies at z = 0).
#   * The EKF's r tracks the IMU.  dog5.xml models it AT the trunk origin
#     (`<site name="imu" pos="0 0 0">`, line 43) -- true in sim, WRONG on
#     hardware, where the board sits on the trunk bottom, measured
#     IMU_BELOW_TRUNK_ORIGIN_M lower (hip axis 8.5 in, trunk bottom 7.0 in).
# So FK-to-hip-axis and the EKF's r are NOT the same point on the real robot;
# fk_floor_height(ref="imu") converts between them.
FOOT_RADIUS_M = 0.020
IMU_BELOW_TRUNK_ORIGIN_M = 0.038

# ===========================================================================
# stage machine -- timing shared by every runner
# ===========================================================================
T_STAND = 5.0                # crouch -> stand ramp (s); also the park ramp
EKF_WORKER_HZ = 100.0        # estimator thread rate
QUIET_STAGES = ("WAIT_CROUCH",)   # EKF init happens only while truly still
SETTLE_S = 1.5               # ignore this long after entering a holding stage
                             # (WAIT_CROUCH / HOLD4 / PARKED) before scoring it
MIN_INIT_QUIET_S = 1.0       # warn if the raw log's static prefix is shorter

# ===========================================================================
# stand pose
# ===========================================================================
# Stage-2 stood at the recorded default (0.20 m), but the legs' reach caps the
# commanded foot z at ~-221 mm -- at 0.20 that leaves ~21 mm of extension
# authority, less than height sag (~13 mm) + a leveling clamp (12 mm).  The
# later scripts therefore default LOWER: 0.19 m keeps ~31 mm of extension
# authority and drops the CoM for the lift demo.
STAND_HEIGHT_DEFAULT = 0.19  # stages 2b, A, lift
STAND_HEIGHT_STAGE12 = 0.20  # stages 1 and 2 -- must equal
                             # stand_dog5_recorded_hw.DEFAULT_STAND_HEIGHT,
                             # which is asserted in test_stand_params.py

# ===========================================================================
# height loop -- stage 2, common foot z (stand_ekf_height_hw)
# ===========================================================================
HEIGHT_GAIN_PER_S = 0.5      # integral gain: offset += K * err * dt  (~2 s tau)
HEIGHT_CLAMP_M = 0.030       # max |offset| the loop may wind in
HEIGHT_SLEW_M_S = 0.005      # max offset rate -- caps the response to a step
HEIGHT_DEADBAND_M = 0.001    # below this error the integrator holds still
HEIGHT_STALE_S = 0.2         # EKF worker stall -> freeze
HEIGHT_XCHECK_M = 0.030      # |EKF - FK| beyond this -> freeze (FK is truth)

# per-leg z -> joint-angle tables (no IK in the CAN sweep)
TABLE_STEP_M = 0.0005        # z resolution of the tables
TABLE_MARGIN_M = 0.010       # extra table range beyond the clamp

# ===========================================================================
# leveling loop -- stage 2b, differential foot z, closed on the EKF
# ===========================================================================
# Constants are stand_hier_hw.LevelingTrim's, proven on hardware.
LEVEL_GAIN_PER_S = 0.25      # integral gain on the geometric per-foot error
LEVEL_CLAMP_M = 0.012        # per-foot authority (~2.0 deg pitch, ~6 deg roll)
LEVEL_SLEW_M_S = 0.004       # caps the response to an EKF attitude step
LEVEL_DEADBAND_DEG = 0.2     # stage-1 noise floor is well under this
LEVEL_LPF_FC_HZ = 10.0
LEVEL_STALE_S = HEIGHT_STALE_S    # same worker, same staleness rule
LATCH_S = 2.0                # setpoint = mean attitude over this window
HEIGHT_SETTLE_S = 1.0        # ... which may only start once the height loop
                             # has held its target this long (see
                             # HeightSettleGate: no latching mid-sag-ramp)
AGREE_VETO_DEG = 3.0         # |EKF-AHRS| beyond this freezes leveling

# ===========================================================================
# AHRS-only leveling -- experiment A (stand_ahrs_level_hw)
# ===========================================================================
# NOT the EKF script's numbers, and deliberately named apart from them: this
# loop is closed on the DETA10's own fused attitude, whose noise floor is
# under the deadband, so it can afford a P term and is tuned for SPEED.
# All three had to move together -- the closed-loop pole is ki/(1+kp), so
# adding kp=1 to LEVEL_GAIN_PER_S=0.25 would have made the tail SLOWER.
AHRS_KP = 1.0                # proportional gain, dimensionless: the fraction
                             # of the geometric error commanded IMMEDIATELY.
                             # 1.0 = "ask for the whole correction at once";
                             # the rate limit, not kp, decides how fast it
                             # actually gets there.
AHRS_GAIN_PER_S = 1.0        # integral gain (1/s).  4x the EKF script's.
AHRS_SLEW_M_S = 0.010        # max foot speed -- the REAL speed limit here:
                             # above ~0.3 deg the loop is pinned to it.
                             # 10 mm/s = 1.7 deg/s pitch, 5.1 deg/s roll.
AHRS_STALE_S = 0.2           # no fresh 0x41 packet this long -> freeze

# ===========================================================================
# lift experiment -- one foot off the floor (lift_ekf_contact_hw)
# ===========================================================================
LIFT_M = 0.020               # default commanded foot lift (visible clearance)
LIFT_SLEW_M_S = 0.008        # lift/lower ramp rate (~2 s for 15 mm)
CONTACT_OFF_M = 0.003        # commanded lift beyond this -> contact False
LIFT_SETTLE_S = 1.0          # settle at full lift before scoring zEKF/zFK
LEAN_ABORT_DEG = 6.0         # attitude error that auto-lowers the lift
LIFT_KEYS = {"1": "FL", "2": "FR", "3": "RL", "4": "RR"}

# ===========================================================================
# scheduled contacts -- experiment B (stand_ekf_schedcontact_hw)
# ===========================================================================
TOUCH_FRAC_DEFAULT = 0.3     # rear feet flip ON this far into the STAND ramp
CROUCH_PLANTED_DEFAULT = ("FL", "FR")   # the feet actually down at the crouch

# ===========================================================================
# trot in place -- stage 3, FK-switched diagonal pairs (trot_fk_switch_hw)
# ===========================================================================
# The gait is TIMED BY THE ROBOT, not by a clock: a pair steps, and the next
# pair is released only once forward kinematics confirms the last one is back
# where it started.  So there is no period tunable here -- the numbers below
# bound each phase, and the measured period is an OUTPUT of the run.
TROT_PAIRS = (("FL", "RR"), ("FR", "RL"))   # the two diagonals, in order
TROT_STAND_HEIGHT_M = 0.17   # this gait stands LOWER than everything else in
                             # the track, and it is not a preference.  The legs
                             # bottom out at a commanded foot z of -221 mm, so
                             # the stand height sets how much EXTENSION
                             # AUTHORITY is left below the pose -- 31 mm at
                             # STAND_HEIGHT_DEFAULT (0.19), 51 mm here.  The
                             # gait needs 13 (sag the height loop winds out)
                             # + 12 (leveling clamp) + 15 (TROT_PUSH_M) =
                             # 40 mm, which simply does not fit at 0.19: the
                             # z table clips, the stance push silently comes
                             # up short, and the feet do not clear.  Standing
                             # lower is the remedy; shrinking the push is not,
                             # because the push is what makes stepping
                             # possible at all.
TROT_COM_SHIFT_X = 0.0       # slide the BODY over the footprint (m, trunk
TROT_COM_SHIFT_Y = 0.0       # frame, +x forward / +y left).  Zero until a run
                             # measures otherwise -- this is a per-robot number
                             # (battery and harness placement), not a physics
                             # constant, so there is no honest default.
                             #
                             # WHY IT EXISTS.  Hardware 2026-08-14: with FL+RR
                             # commanded up, RR cleared and FL did not, and the
                             # body leaned toward FL.  While a diagonal swings
                             # the robot stands on TWO feet, and rotation about
                             # the line through them is UNACTUATED -- both
                             # stance legs are ON that axis, so extending them
                             # (TROT_PUSH_M) moves the axis but cannot rotate
                             # the body about it.  Gravity decides.  With the
                             # CoM off to one side the body falls that way
                             # until the near swing foot lands and takes load,
                             # and lifting that foot higher only tips it
                             # further to meet it: a statics problem, not a
                             # kinematics one.
                             #
                             # The two diagonals cross at the centre of the
                             # foot polygon, so ONE constant trim serves both.
                             # StanceLoadBalance measures it from per-leg load
                             # sag and prints the suggestion.
TROT_LIFT_M = 0.005          # commanded lift per swinging foot.  SMALL on
                             # purpose, and DO NOT read it as ground
                             # clearance -- two separate effects eat it:
                             #   * the stance legs are stiff position holds,
                             #     so a lifted diagonal tips the trunk about
                             #     the stance diagonal instead of raising the
                             #     feet (the foot anchors are 214 mm off that
                             #     diagonal, so 5 mm = 1.3 deg of rock, 15 mm
                             #     = 4.0 deg = TROT_LEAN_ABORT_DEG);
                             #   * the swinging leg UNLOADS, and a leg that
                             #     sags 13 mm under load extends by about
                             #     that much when the load comes off.
                             # So the first hardware run is a rock in place,
                             # and the run's job is to find the lift at which
                             # the feet actually leave the floor.  Raise it a
                             # few mm at a time and watch the `clear=` field.
TROT_PUSH_M = 0.015          # how far the STANCE pair extends while the other
                             # diagonal swings.  Hardware 2026-08-14: without
                             # this the knees move and the feet do not leave
                             # the floor, because lifting a diagonal transfers
                             # its load to the other one, which compresses
                             # further and SINKS THE TRUNK by about as much as
                             # the lift asked for.  To first order, with 13 mm
                             # of sag per leg at a quarter of the weight:
                             #
                             #   clearance = lift + push - 2 * 13 mm
                             #     (one 13 mm for the swing legs extending as
                             #      they unload, one for the stance pair
                             #      compressing as its load doubles)
                             #
                             # so lift alone has to beat 26 mm, whose rocking
                             # bound already exceeds TROT_LEAN_ABORT_DEG --
                             # there is NO push-free lift that both clears and
                             # stays inside the abort.  The push is what makes
                             # a stepping gait reachable at all.
                             # It costs extension authority: see the startup
                             # banner, and stand LOWER (--stand-height 0.17
                             # gives 51 mm of reach below the stand, against
                             # 31 mm at 0.19) if the push does not fit.
                             # The height loop cannot do this job -- at
                             # HEIGHT_SLEW_M_S it moves 2.5 mm in a 0.5 s
                             # swing, and it is feedback, arriving after the
                             # trunk has already dropped.
TROT_SLEW_M_S = 0.020        # lift/lower rate.  2.5x LIFT_SLEW_M_S: the lift
                             # experiment wanted a slow, watchable ramp; a
                             # gait wants the foot up and down inside a step.
                             # 5 mm at 20 mm/s = 0.25 s each way.
TROT_AIR_DWELL_S = 0.10      # hold at full lift before coming back down
TROT_OVERLAP_S = 0.30        # ALL FOUR planted between steps.  This is what
                             # makes the first hardware version a step in
                             # place rather than a true trot: support never
                             # hands straight from one diagonal to the other,
                             # so there is always a 4-foot window in which the
                             # leveling loop is allowed to run (it freezes
                             # itself below 3 planted -- LevelingLoop.update).
                             # Drive this to 0 and you have a real trot.
TROT_CONTACT_CLEAR_M = 0.002 # MEASURED clearance above the stance plane
                             # beyond which a foot is reported contact False.
                             # Measured, not commanded, because the two are
                             # not the same thing here: stage 1 measured 13 mm
                             # of load sag, so a swinging leg EXTENDS by about
                             # that much as it unloads and a 5 mm commanded
                             # lift buys no ground clearance at all.  Deriving
                             # the contact flag from the command would tell
                             # the EKF a foot is airborne while it is still
                             # carrying weight -- the exact failure the OFF
                             # schedule exists to prevent.  See SwingClearance.
TROT_LOAD_STILL_M = 0.0002   # a commanded foot z that moves more than this
                             # inside a load-estimate window throws the window
                             # away.  StanceLoadBalance reads the servo's
                             # tracking error as a load proxy, which is only
                             # true against a STATIONARY target -- hardware
                             # 2026-08-14 produced "RL carries 100%, trim the
                             # body 342 mm" because it ran while the height
                             # loop was winding 13 mm at HEIGHT_SLEW_M_S and
                             # leveling was winding +/-12 mm differentially.
                             # 0.2 mm is under a table step and well under any
                             # real sag, so it admits a settled stance and
                             # nothing else.
TROT_STANCE_REF_S = 0.20     # how much of each 4-foot window is averaged into
                             # the stance reference the clearance is measured
                             # against (taken at the END of the window, once
                             # the last touchdown has settled)
TROT_SWITCH_TOL_M = 0.0010   # FK switch: the swinging foot is "back" once its
                             # MEASURED foot z is within this of the z it left
                             # from.  Measured-against-its-own-baseline, so
                             # the several-mm load sag cancels -- and because
                             # the sag only returns as the leg re-loads, this
                             # waits for load transfer, not just for the
                             # encoder to reach its target.
TROT_SWITCH_HOLD_S = 0.05    # ... and has stayed there this long (debounce)
TROT_SWITCH_TIMEOUT_S = 2.0  # no FK confirmation within this -> ABORT the
                             # gait with all feet down.  Without it a leg that
                             # stalls mid-swing hangs the state machine with a
                             # foot in the air and no one watching.
TROT_LEAN_ABORT_DEG = 4.0    # attitude error that stops the gait (all feet
                             # down).  Tighter than the lift experiment's
                             # LEAN_ABORT_DEG: that one had three feet down
                             # and could afford to lean.

# ===========================================================================
# safety -- applies to every runner
# ===========================================================================
TILT_STOP_DEG = 12.0         # absolute attitude that soft-stops the run
TEMP_NOTICE_C = 60           # below this the status line stays quiet about
                             # temperature (base.TEMP_ESTOP=80 already stops)

# ===========================================================================
# CAN timing diagnostics -- reporting only, nothing is controlled by these
# ===========================================================================
GAP_BUCKETS = (0.008, 0.010, 0.015, 0.020, 0.050)   # command-gap histogram
                             # edges (s).  The first two straddle the 10 ms
                             # watchdog that CAN 1/7/8/9 are configured with.
