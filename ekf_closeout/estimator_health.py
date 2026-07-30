#!/usr/bin/env python3
"""EKF health flag with a RELATIVE min-eigenvalue tolerance.

Why this exists
---------------
`DOG5StateEstimator.outputs()` (dog5_state_estimator.py:638) declares the
filter unhealthy when

    min_eig(P) <= -1.0e-9          # ABSOLUTE tolerance

Replaying `walk_0729_1748.npz` shows all 316 unhealthy frames fail on exactly
that term -- and none on the sigma_v term (sigma_v stays ~4 mm/s through the
3-contact swings).  The cause is numerical, not a filter fault:

    swing_p = 1e4 (dog5_state_estimator.py:268) inflates the airborne
    foothold block, so max(diag P) reaches ~7e8 m^2 by the end of an 8 s
    swing.  `eigvalsh` on a PSD matrix carries jitter of order eps*||P||
    ~ 2e-16 * 7e8 ~ 1e-7, which swamps the fixed -1e-9 floor.  The clusters
    collapse exactly at touchdown because `sigma_reset` shrinks the foothold
    block again.

The observed jitter is machine-eps *relative*, so the fix is to scale the
tolerance with the matrix norm.  1e-12 relative leaves ~4 orders of margin
over the observed jitter while still catching genuine indefiniteness (a real
negative eigenvalue in an EKF covariance is many orders larger).

This subclass keeps the base class untouched: live hardware runners still use
the stock flag; the close-out replay uses this one.  The in-tree fix is a
roadmap item (CONTROL_ROADMAP.md), not part of this read-only close-out.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EST = os.path.join(os.path.dirname(_HERE), "state_estimator")
for _p in (_HERE, _EST):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dog5_state_estimator import DOG5StateEstimator, ErrorIndex  # noqa: E402

# Matches the base class's sigma_v term (dog5_state_estimator.py:638) so the
# only behavioural difference is the min_eig tolerance.
SIGMA_V_LIMIT = 0.15        # m/s
MIN_EIG_REL_TOL = 1.0e-12   # times max(1, max diag P)


def health_terms(state, out):
    """Evaluate the three health terms; returns (healthy, detail dict).

    Shared by the estimator subclass and the --health-detail report so the
    printed diagnosis is literally the gate that was applied.
    """
    P = state.P
    max_diag = float(np.max(np.diag(P)))
    scale = max(1.0, max_diag)
    tol = -MIN_EIG_REL_TOL * scale
    sigma_v = float(np.max(out["sigma"][ErrorIndex.V]))
    min_eig = float(out["min_eig_P"])
    finite = bool(np.all(np.isfinite(state.x)) and np.all(np.isfinite(P)))
    detail = {
        "sigma_v": sigma_v,
        "min_eig": min_eig,
        "min_eig_tol": tol,
        "max_diag_P": max_diag,
        "finite": finite,
        "sigma_v_ok": sigma_v < SIGMA_V_LIMIT,
        "min_eig_ok": min_eig > tol,
    }
    healthy = bool(detail["sigma_v_ok"] and finite and detail["min_eig_ok"])
    return healthy, detail


def failing_term(detail):
    """Which term rejected this frame ('sigma_v' / 'min_eig' / 'nonfinite')."""
    if not detail["finite"]:
        return "nonfinite"
    if not detail["sigma_v_ok"]:
        return "sigma_v"
    if not detail["min_eig_ok"]:
        return "min_eig"
    return "healthy"


class HealthFixEstimator(DOG5StateEstimator):
    """DOG5StateEstimator with the relative min_eig health tolerance.

    Filter math is untouched -- only the reported `healthy` flag changes.
    `outputs()` additionally returns `max_diag_P` and `health_detail` so the
    replay can show exactly which term failed.
    """

    def outputs(self, last_w_meas=None):
        out = super().outputs(last_w_meas=last_w_meas)
        healthy, detail = health_terms(self.state, out)
        out["healthy"] = healthy
        out["health_detail"] = detail
        out["max_diag_P"] = detail["max_diag_P"]
        return out
