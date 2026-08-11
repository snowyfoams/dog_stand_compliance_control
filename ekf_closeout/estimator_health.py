#!/usr/bin/env python3
"""EKF health flag with a RELATIVE min-eigenvalue tolerance.

**This fix is now in-tree.**  It was developed here, out-of-tree, so the
close-out replay could report honestly without touching the filter the live
hardware runners were using.  It has since been promoted into
`dog5_state_estimator` itself -- `SIGMA_V_LIMIT`, `MIN_EIG_REL_TOL`,
`health_terms()` and `failing_term()` all live there now, and the stock
`DOG5StateEstimator.outputs()` applies them.

Why it was promoted: a trot gates on `healthy` several times a second, in the
middle of a swing, and cannot use a flag that reads false for numerical reasons
(CONTROL_ROADMAP.md Sec. 1b and Sec. 3).

This module is kept as a re-export so the close-out suite and `replay_full.py`
keep importing the names they always did.  `HealthFixEstimator` is now simply an
alias of the base class -- there is nothing left to subclass.

The original diagnosis, retained for the record: replaying
`walk_0729_1748.npz`, all 316 unhealthy frames failed the min_eig term and
*none* failed the sigma_v term.  `swing_p` inflated the airborne foothold block
until max(diag P) reached ~7e8 m^2, so `eigvalsh` jitter of order
eps*||P|| ~ 1e-7 swamped the fixed -1e-9 floor.  Both halves are fixed in-tree
now: the tolerance is relative, and `swing_p` was reduced 1e4 -> 1e2.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EST = os.path.join(os.path.dirname(_HERE), "state_estimator")
for _p in (_HERE, _EST):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dog5_state_estimator import (  # noqa: E402,F401
    DOG5StateEstimator,
    ErrorIndex,
    MIN_EIG_REL_TOL,
    SIGMA_V_LIMIT,
    failing_term,
    health_terms,
)

# Nothing to override any more -- the base class carries the fix.
HealthFixEstimator = DOG5StateEstimator

__all__ = ["HealthFixEstimator", "health_terms", "failing_term",
           "MIN_EIG_REL_TOL", "SIGMA_V_LIMIT"]
