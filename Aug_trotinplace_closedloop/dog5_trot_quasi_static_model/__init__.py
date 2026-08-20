"""dog5_trot_quasi_static_model -- a trot controller for DOG5, split six ways.

    config      every constant, no logic
    leg_kin     single-leg FK / Jacobian / IK
    gait        the contact clock
    balance_qp  virtual-model wrench + the force-distribution QP
    swing       foot placement, swing arcs, swing tracking
    controller  the orchestration, and the only file that reads all the others

IMPORT IT AS A PACKAGE.  `config` is a very common module name and this repo
already has its own at the top level, which motorbus.py imports.  Putting this
directory on sys.path shadows it and breaks the CAN layer with an unrelated
AttributeError.  Every module here therefore imports its siblings through the
package when it is imported as one, and only falls back to a bare path insert
when it is being run directly for its --self-test.
"""
