"""srb_mpc_hw -- the DOG4.6 `srb_mpc` simulation, put on the DOG5 hardware.

    mpc_config      every constant, no logic
    srb_model       the single rigid body: A, B, and an EXACT discretisation
    convex_mpc      the condensed QP over the horizon, and the ADMM that solves it
    mpc_gait        the contact-aware layer: clock + MEASURED touchdown
    mpc_swing       Raibert placement, the swing arc, and the tracking torque
    mpc_controller  the model block -- state in, twelve torques out
    mpc_worker      the off-thread solver and its mailbox
    mpc_trot_hw     the runner: the only file that opens a bus

WHAT WAS PORTED, AND FROM WHERE
    The simulation is DOG4.6_description/srb_mpc: a MIT-style convex MPC on
    the single-rigid-body model (Di Carlo, Wensing, Katz, Bledt, Kim, IROS
    2018), a clock trot with a contact-aware promotion layer, and Raibert
    footstep placement tracked by a Cartesian PD through J^T.

    Six of its seven control ideas transfer unchanged and are here.  Three
    things could NOT transfer and each has its own note where it lives:

        MuJoCo            robot.py is 200 lines of mj_* index bookkeeping.
                          Every one of its outputs already exists on this
                          robot -- dog5_statics for the chain, the Jacobian
                          and leg gravity, feedback_estimator for z/v/C/omega
                          -- so robot.py has no port at all.  See mpc_controller.
        scipy + osqp      the Pi's venv has numpy and NOTHING else (the same
                          fact dog5_trot/balance_qp.py records).  expm() is
                          replaced by an EXACT finite series -- the SRB's A is
                          nilpotent, so there is no approximation in it -- and
                          OSQP by the same dense ADMM balance_qp uses.  See
                          srb_model and convex_mpc.
        world x, y, yaw   the simulator reads them off the floating base.  This
                          robot has no absolute horizontal position and its
                          only yaw is a magnetometer's, next to twelve motors
                          and a steel frame.  The MPC is therefore re-anchored
                          at EVERY solve: x/y/yaw enter as zero and the
                          reference integrates the command away from zero, so
                          the horizon carries the same position feedback the
                          simulator had and nothing outside it can drift.
                          See mpc_controller._x0 and mpc_config.

    The sim package also carries an ETH-lineage comparison stack (slq_mpc,
    qp_wbc, eth_controller) and a scripted jump.  Those are the dissertation's
    A/B against the convex stack, not the thing that flies on this robot, and
    the QP-WBC needs the full M/h/Jdot of a floating base that this tree has
    no dynamics for.  They are deliberately NOT ported.

WHAT THE MPC REPLACES ON THIS ROBOT, WHICH IS EXACTLY ONE THING
    august_week2 answers "how hard must the feet push?" with a virtual spring
    on the trunk (Dynamic_Model.body_wrench) split across the feet by a
    minimum-norm grasp map (force_totorque.distribute).  Both are INSTANTANEOUS
    and quasi-static: they know nothing about a foot that is about to lift.

    The MPC answers the same question over a 0.4 s horizon that DOES know --
    the contact schedule is a constraint in the QP, so the load is handed over
    before a foot leaves rather than after.  Everything on either side of that
    answer is week 2's, unchanged and unrewritten:

        crouch/park bookends            crouch_and_park
        z, v, C, omega                  feedback_estimator
        -J^T f + leg gravity            dog5_statics.stance_torque
        the 250 Hz joint impedance      the runner, as in stand_torque_Mode
        ramp / cap / slew / e-stops     TorqueGate, CanMissMonitor, the gate set
        the RISE itself                 still the week-2 wrench.  A stand-up
                                        that already works is not the place to
                                        try a new controller.

IMPORT IT AS A PACKAGE.  `config` is a very common module name and this repo
already has its own at the top level, which motorbus.py imports.  Putting this
directory on sys.path shadows it and breaks the CAN layer with an unrelated
AttributeError -- which is why the constants file here is called `mpc_config`
and not `config`, and why every module reaches its siblings through the
package.
"""
