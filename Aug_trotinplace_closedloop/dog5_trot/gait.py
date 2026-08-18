#!/usr/bin/env python3
"""The trot clock.  Time in, contact schedule out.  No robot state at all.

WHAT A GAIT IS HERE, AND WHAT IT IS NOT
    This file answers exactly one question: at time t, which feet are
    SUPPOSED to be on the ground, and how far through its swing is each foot
    that is not.  It is an open-loop clock.

    It does NOT know whether a foot is actually touching -- that is a
    measurement, and mixing it in here would make the schedule depend on the
    thing the schedule is supposed to drive.  Early or late touchdown is
    handled where it belongs, by SwingPlanner.touchdown() re-anchoring the
    arc, and by the QP's contact mask being the caller's to choose.

PHASE, AND WHY IT IS IN CYCLES
    phase(t) is in [0, 1) per leg: 0 at the start of stance, DUTY at the
    start of swing, wrapping at 1.  Offsets are in CYCLES for the same reason
    -- a trot is "half a cycle apart" regardless of the period, so writing
    0.5 keeps the pairing correct when the period is retuned.  Seconds would
    have to be edited twice.

    A trot pairs the DIAGONALS: FL with RR, FR with RL.  With the default
    offsets (0, 0.5, 0.5, 0) over the leg order (FL, FR, RL, RR) that is
    exactly what comes out, and the self-test asserts it rather than trusting
    the reader to check four numbers against four names.

RUN
    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V dog5_trot/gait.py --self-test
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# Sibling imports go through the PACKAGE when there is one.  Only a direct
# `python dog5_trot/<this>.py --self-test` falls back to a path insert, and
# that insert is what would shadow the repo's own top-level config.py -- see
# the package docstring.  Keeping it off the library path is the point.
if __package__:
    from . import config as cfg
else:
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import config as cfg



class TrotGait:
    """The contact schedule, as a pure function of t once reset() has been called."""

    def __init__(self, period: float = cfg.GAIT_PERIOD,
                 duty: float = cfg.DUTY,
                 offsets=cfg.PHASE_OFFSET):
        if not period > 0.0:
            raise ValueError(f"period must be positive, got {period}")
        if not 0.0 < duty < 1.0:
            raise ValueError(f"duty must be in (0,1), got {duty} -- 1.0 would "
                             f"never lift a foot and 0.0 would never plant one")
        offsets = np.asarray(offsets, dtype=float).reshape(cfg.N_LEGS)
        self.period = float(period)
        self.duty = float(duty)
        self.offsets = np.mod(offsets, 1.0)
        self._t0 = 0.0

    # -- the clock ---------------------------------------------------------
    def reset(self, t: float) -> None:
        """Make `t` the zero of the cycle: every leg's phase becomes its offset."""
        self._t0 = float(t)

    def phase(self, t: float) -> np.ndarray:
        """(4,) phase in [0,1).  0 starts stance, `duty` starts swing."""
        return np.mod((float(t) - self._t0) / self.period + self.offsets, 1.0)

    def contact(self, t: float) -> np.ndarray:
        """(4,) True where the schedule says the foot should be planted.

        STRICTLY LESS THAN duty, so the instant a leg reaches `duty` it is in
        swing.  The boundary has to belong to one side or the other, and
        giving it to swing means a foot is never commanded to push at the same
        instant the swing planner starts lifting it.
        """
        return self.phase(t) < self.duty

    def swing_phase(self, t: float) -> np.ndarray:
        """(4,) progress through swing in [0,1); exactly 0 for a stance leg.

        Stance legs return 0 rather than NaN or a negative number: the swing
        planner multiplies by this, and a 0 that means "not swinging" costs
        nothing, while a NaN would poison a whole 12-vector of torque.
        """
        ph = self.phase(t)
        sw = (ph - self.duty) / (1.0 - self.duty)
        return np.where(ph < self.duty, 0.0, sw)

    def contact_weight(self, t: float, ramp: float = cfg.CONTACT_RAMP) -> np.ndarray:
        """(4,) how much force each leg may carry, in [0,1].

        1 through mid-stance, smoothstepping up over the first `ramp` of
        stance and down over the last.  Exactly 0 for a swinging leg, so a
        caller can use `weight > 0` wherever it used to use `contact`.

        THE RAMP IS INSIDE STANCE, NOT STRADDLING THE LIFT.  A foot is already
        unloaded by the time the schedule lifts it, which is what makes the
        liftoff free; a ramp that continued into swing would ask the QP for
        force from a foot that is off the ground.  See config.CONTACT_RAMP for
        the 2.2 Nm step this removes.
        """
        ph = self.phase(t)
        s = np.clip(ph / self.duty, 0.0, 1.0)          # stance progress
        r = max(float(ramp), 1e-9)
        u = np.clip(np.minimum(s / r, (1.0 - s) / r), 0.0, 1.0)
        w = u * u * (3.0 - 2.0 * u)                    # smoothstep
        return np.where(ph < self.duty, w, 0.0)

    # -- durations ---------------------------------------------------------
    @property
    def stance_duration(self) -> float:
        return self.period * self.duty

    @property
    def swing_duration(self) -> float:
        return self.period * (1.0 - self.duty)

    def __repr__(self):
        return (f"TrotGait(period={self.period:.3f}s duty={self.duty:.2f} "
                f"stance={self.stance_duration*1e3:.0f}ms "
                f"swing={self.swing_duration*1e3:.0f}ms)")


# ===========================================================================
# self-test
# ===========================================================================
_PASS = [0, 0]


def check(label, ok, detail=""):
    _PASS[1] += 1
    _PASS[0] += bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))


def self_test():
    g = TrotGait()
    g.reset(0.0)
    print(f"  {g}")

    check("reset makes phase equal the offsets exactly",
          np.allclose(g.phase(0.0), cfg.PHASE_OFFSET))

    # -- the pairing is the DIAGONALS, checked by name --------------------
    names = list(cfg.LEGS)
    fl, fr, rl, rr = (names.index(n) for n in ("FL", "FR", "RL", "RR"))
    ts = np.linspace(0.0, 3 * g.period, 977)
    c = np.array([g.contact(t) for t in ts])
    check("FL and RR are always in the same state (one diagonal)",
          bool(np.all(c[:, fl] == c[:, rr])))
    check("FR and RL are always in the same state (the other diagonal)",
          bool(np.all(c[:, fr] == c[:, rl])))
    # AT DUTY > 0.5 THE DIAGONALS OVERLAP, AND THEY MUST.  The old assertion
    # here was "always opposite", which is only true at duty exactly 0.5 --
    # and duty 0.5 is the one value at which the contact ramp has nowhere to
    # hand the load over.  What has to hold is that support never drops below
    # a diagonal, not that it never rises above one.
    overlap = int(round((g.duty - 0.5) * 100))
    check("the diagonals overlap by (duty - 0.5) of the cycle",
          bool(np.any(c[:, fl] & c[:, fr])) if g.duty > 0.5
          else bool(np.all(c[:, fl] != c[:, fr])),
          f"duty {g.duty:.2f} -> {overlap}% double support, "
          f"{100*np.mean(c[:, fl] & c[:, fr]):.0f}% measured")
    check("at least two feet are planted at EVERY instant",
          bool(np.all(c.sum(axis=1) >= 2)),
          f"planted counts seen: {sorted(set(c.sum(axis=1).tolist()))}")
    check("...and never one leg alone or none, which a trot cannot survive",
          bool(np.all(c.sum(axis=1) != 1) and np.all(c.sum(axis=1) != 0)))

    # -- duty really is the fraction of time planted ----------------------
    frac = c.mean(axis=0)
    check("each leg is planted for DUTY of the cycle",
          np.allclose(frac, cfg.DUTY, atol=2e-3),
          f"measured {np.round(frac, 4)} vs duty {cfg.DUTY}")

    # -- swing_phase is a clean 0->1 ramp, and 0 while planted ------------
    sp = np.array([g.swing_phase(t) for t in ts])
    check("swing_phase is exactly 0 wherever the leg is in contact",
          bool(np.all(sp[c] == 0.0)))
    check("...and covers [0,1) while it is not",
          sp[~c].min() >= 0.0 and sp[~c].max() < 1.0,
          f"range [{sp[~c].min():.3f}, {sp[~c].max():.3f})")

    # a fine sweep through one swing must be monotonic -- a planner that
    # integrates this would otherwise step backwards mid-arc
    fine = np.linspace(g.duty * g.period + 1e-6, g.period - 1e-6, 5000)
    s_fl = np.array([g.swing_phase(t)[fl] for t in fine])
    check("swing_phase increases monotonically through one swing",
          bool(np.all(np.diff(s_fl) > 0)))

    # -- the contact weight ------------------------------------------------
    w = np.array([g.contact_weight(t) for t in ts])
    check("contact_weight is exactly 0 wherever the leg is swinging",
          bool(np.all(w[~c] == 0.0)))
    check("...and reaches a full 1.0 through mid-stance",
          abs(float(w[c].max()) - 1.0) < 1e-12)
    check("...and is never negative or over 1",
          bool(np.all((w >= 0.0) & (w <= 1.0))))
    fine2 = np.linspace(1e-6, g.duty * g.period - 1e-6, 4000)
    wf = np.array([g.contact_weight(t)[fl] for t in fine2])
    check("it rises from 0 at touchdown and returns to 0 at liftoff",
          wf[0] < 1e-6 and wf[-1] < 1e-6 and wf.max() > 0.999,
          f"start {wf[0]:.2e}, peak {wf.max():.4f}, end {wf[-1]:.2e}")
    check("...with no step anywhere, which is the whole point",
          float(np.max(np.abs(np.diff(wf)))) < 0.01,
          f"worst adjacent change {np.max(np.abs(np.diff(wf))):.5f} over "
          f"{len(wf)} samples of one stance")
    check("the two diagonals' weights always sum to something that can "
          "carry the robot",
          bool(np.all(w.sum(axis=1) > 0.5)),
          f"worst total weight {w.sum(axis=1).min():.3f} of 2.0")

    # -- durations add up --------------------------------------------------
    check("stance + swing = period",
          abs(g.stance_duration + g.swing_duration - g.period) < 1e-15,
          f"{g.stance_duration*1e3:.1f} + {g.swing_duration*1e3:.1f} ms")

    # -- the clock is periodic and reset actually moves it ----------------
    check("the schedule is periodic in `period`",
          np.allclose(g.phase(0.123), g.phase(0.123 + 7 * g.period)))
    g2 = TrotGait()
    g2.reset(5.0)
    check("reset(t) shifts the whole schedule, it does not scale it",
          np.allclose(g2.phase(5.0 + 0.1), g.phase(0.1)))

    # -- the constructor refuses schedules that cannot lift a foot --------
    for bad in (0.0, 1.0, 1.5, -0.1):
        try:
            TrotGait(duty=bad)
            ok = False
        except ValueError:
            ok = True
        if not ok:
            break
    check("duty outside (0,1) is refused at construction", ok,
          "0 never plants a foot, 1 never lifts one")

    print(f"self-test {'PASS' if _PASS[0] == _PASS[1] else 'FAIL'} "
          f"({_PASS[0]}/{_PASS[1]})")
    return 0 if _PASS[0] == _PASS[1] else 1


if __name__ == "__main__":
    sys.exit(self_test())
