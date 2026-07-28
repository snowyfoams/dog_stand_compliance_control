"""Quasi-static gait scheduler for the DOG5 VMC.

Two modes:
  * STAND    — all four feet in contact, forever. No contact transitions, so the
               estimator never re-anchors a foothold (the safest first target).
  * DIAGONAL — the trot footfall {FL,RR} <-> {FR,RL}, executed *quasi-statically*
               with a generous double-support phase (mostly four feet down) and a
               brief, low swing of one diagonal pair at a time. step_len = 0 gives
               stepping in place.

The scheduler owns the contact flags. They are fed identically to the estimator
and to the VMC each tick, and the flag is arranged to *lag* the swing so a
foothold is never re-anchored on an airborne foot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

LEGS = ["FL", "FR", "RL", "RR"]
DIAG_A = ("FL", "RR")
DIAG_B = ("FR", "RL")


@dataclass
class GaitParams:
    # Defaults tuned for a gentle quasi-static diagonal step: the body rolls
    # while on a 2-foot diagonal (limited roll authority about the support
    # line), and that roll grows with the single-support duration, so a short
    # swing window inside a long double-support keeps peak tilt small (~2 deg).
    period: float = 2.0          # s per full cycle (both pairs step once)
    ds_frac: float = 0.6         # fraction of each half-cycle spent in double support
    lift: float = 0.025          # m, swing-foot raise (toward the body)
    step_len: float = 0.0        # m, forward foot travel per step (0 = in place)
    z_des: float = 0.0           # trunk height target in the ESTIMATOR frame
                                 # (0 = hold the height at estimator init)
    v_cmd_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yawrate_cmd: float = 0.0


class GaitScheduler:
    """Emit contacts, swing foot targets, and body targets vs time."""

    def __init__(self, nominal_feet_body, mode="STAND", params: GaitParams | None = None):
        # nominal_feet_body: {leg: (3,)} foot positions in the body frame captured
        # at the standing hand-off; swings lift from and return to these.
        self.nominal = {l: np.asarray(nominal_feet_body[l], dtype=float) for l in LEGS}
        self.mode = mode
        self.p = params or GaitParams()

    # ---- timing of the diagonal cycle ----
    # half-cycle = double-support (t_ds) then single pair swing (t_sw)
    def _half(self):
        half = self.p.period / 2.0
        t_ds = self.p.ds_frac * half
        t_sw = half - t_ds
        return t_ds, t_sw

    def _swing_state(self, t):
        """Return (swing_pair or None, u in [0,1]) for the diagonal schedule."""
        if self.mode != "DIAGONAL":
            return None, 0.0
        t_ds, t_sw = self._half()
        half = self.p.period / 2.0
        phase = t % self.p.period
        if phase < half:                      # first half: pair A may swing
            local = phase
            pair = DIAG_A
        else:
            local = phase - half              # second half: pair B may swing
            pair = DIAG_B
        if local < t_ds or t_sw <= 0.0:       # double support
            return None, 0.0
        u = (local - t_ds) / t_sw             # 0..1 across the swing
        return pair, min(max(u, 0.0), 1.0)

    def contacts(self, t):
        swing_pair, _ = self._swing_state(t)
        c = np.ones(len(LEGS), dtype=bool)
        if swing_pair is not None:
            for leg in swing_pair:
                c[LEGS.index(leg)] = False
        return c

    def stance_set(self, t):
        c = self.contacts(t)
        return [LEGS[i] for i in range(len(LEGS)) if c[i]]

    def swing_targets(self, t):
        """{leg: (p_des(3), v_des(3))} in the body frame for swinging legs."""
        swing_pair, u = self._swing_state(t)
        if swing_pair is None:
            return {}
        _, t_sw = self._half()
        out = {}
        for leg in swing_pair:
            p0 = self.nominal[leg]
            # z lift arc (sin), returns to nominal at u=1; optional forward xy.
            dz = self.p.lift * math.sin(math.pi * u)
            ddz_du = self.p.lift * math.pi * math.cos(math.pi * u)
            dx = self.p.step_len * u
            ddx_du = self.p.step_len
            p_des = p0 + np.array([dx, 0.0, dz])
            du_dt = 1.0 / t_sw if t_sw > 0 else 0.0
            v_des = np.array([ddx_du, 0.0, ddz_du]) * du_dt
            out[leg] = (p_des, v_des)
        return out

    def sched_state(self, t):
        """Bundle everything the VMC needs this tick."""
        return {
            "contacts": self.contacts(t),
            "swing_targets": self.swing_targets(t),
            "z_des": self.p.z_des,
            "v_cmd_world": self.p.v_cmd_world,
            "yawrate_cmd": self.p.yawrate_cmd,
        }
