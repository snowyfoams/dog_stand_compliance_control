# DOG5 Control Roadmap — Position → Torque (VMC) → Dynamics → MPC

**Date:** 30 July 2026
**Scope:** plan overview. Each phase gets its own detailed plan when it starts;
this document fixes the order, the gates, and what is already done.

---

## 0. Where we are today

| Layer | Status |
|---|---|
| Position-mode stand + crawl (`crawl_hw2.0/stand3_hold_hw.py`, `walk1_hw.py`) | **Validated on hardware** (multi-cycle auto crawl, 40 mm steps) |
| Proprioceptive EKF (`state_estimator/`) | Sim gates C1–C7 pass; **hardware full-state accuracy assessed 30 Jul — see §1** |
| VMC in sim (`vmc/dog5_vmc_core.py`, MuJoCo) | Gates V1–V4 pass (stand + quasi-static diagonal steps) |
| VMC on hardware (`vmc/vmc_stand_hw.py`) | **Written, not yet run closed-loop** — this is the Phase 2 entry point |
| Torque plumbing (iq command, zero-torque keepalive, SafetyGate) | Exists and exercised (compliance demos, zero-torque preflights) |

Loop constraints that shape everything below: 250 Hz CAN ceiling with a
10 ms motor watchdog, Raspberry Pi 5 compute, 10:1 gear (already folded into
config gains), abduction range is the standing lateral-shift bottleneck.

---

## 1. EKF full-state accuracy — hardware replay results (30 Jul 2026)

Replayed all three existing hardware logs through
`state_estimator/hw_replay.py` (gate C8 material): `vmc/stand.npz` (static
stand, hand-rocked) and `walk_0729_1748 / _1806.npz` (crawl with EKF
`--raw-log`).

**Per-state verdict:**

| State | Evidence | Verdict |
|---|---|---|
| Attitude roll/pitch | 0.27–1.03° mean vs AHRS across all 3 logs (limit 1.5–2.5°) | **Good.** Stand-log roll mean 1.03° is the worst — looks like a small constant roll offset, worth a level-check once |
| Velocity v | Static: median 1.4 mm/s, 95th pct 4.8 mm/s. Walk: touchdown velocity jumps ≤ 1.1 mm/s | **Good.** The nominal 25 mm/s "static FAIL" is the max statistic catching the deliberate hand-rocking (roll activity 3× baseline at the peak), not estimator error |
| Height z | Static drift 5.9 mm over 46 s; touchdown z jumps ≤ 0.3 mm | **Good** |
| Footholds | Per-leg mean innovation ≤ 0.2 mm; landing miss ≈ 40 mm ≈ one step (predicted = previous foothold, expected) | **Good** |
| Biases b_f, b_ω | Settle statically (gyro std 2.4e-5 rad/s); stable across walk | **Good** |
| x, y, yaw | Unobservable by design (paper Sec. IV-A) — **not yet quantified on hardware** | **Open** — see 1a |
| Health flag | 97.7–98.7 % on walk logs (< 99 % gate) | **Numerical artefact, diagnosed and fixed** — see 1b |

**1a. Remaining accuracy work (small, do before closing any loop):**

- [ ] **Quantify x/y drift rate.** Walk a known straight distance
      (tape-measure), compare EKF displacement. Paper expectation ≈ 10 % of
      distance. This sets how long MPC may trust integrated position.
      **Tooling built and offline-verified** — see `ekf_closeout/RUNBOOK.md`
      Session A (`walk_cmd_hw.py --goto`, `replay_full.py --measured`).
- [ ] **Verify z scale** (Session B): tape-measure crouch→stand Δh at three
      heights against the EKF Δz. Drift is known good; scale is not yet
      checked.
- [x] **Checker hygiene** — done in `ekf_closeout/replay_full.py`: the health
      gate now runs on the post-re-anchor window only (the contacts-off rise
      dead-reckons by design), and `--static` gates the 95th-pct |v| with max
      demoted to info. All five existing logs now report honestly.
- [ ] Optional / deferred: gate C6 (FEJ consistency) — sim-side, does not
      block hardware work.

**1b. The walk-log health failures were a numerical artefact, not σ_v.**
Replaying `walk_0729_1748.npz` frame by frame: all 316 unhealthy frames fail
the `min_eig > -1e-9` term in `dog5_state_estimator.py:638`; **zero** fail the
σ_v term (σ_v stays ≈ 4 mm/s through the 3-contact swings). The cause is
`swing_p = 1e4` inflating the airborne foothold block to max(diag P) ≈ 7e8 m²
over an 8 s swing, so `eigvalsh` jitter (~eps·‖P‖ ≈ 1e-7) swamps the *absolute*
tolerance. The clusters end exactly at touchdown because `sigma_reset` shrinks
that block again.

Fix: scale the tolerance with the matrix — `min_eig > -1e-12 · max(1, max diag P)`
— which leaves ~4 orders of margin over the observed jitter while still
rejecting genuine indefiniteness. Implemented as a subclass in
`ekf_closeout/estimator_health.py` (the base class is untouched; live runners
still carry the stock flag). **Roadmap item: apply it in-tree** in
`dog5_state_estimator.outputs()` before any controller gates on `healthy`, and
consider reducing `swing_p` (1e4 gives σ_foot ≈ 28 km over an 8 s swing;
1e1–1e2 is still "unknown" but keeps P conditioned).

The only genuine σ_v ramp is the contacts-off STAND rise, where it reaches
120–139 mm/s against the 150 mm/s threshold — expected dead-reckoning, now
excluded from the gate, but a reason not to lengthen the rise.

**Bottom line:** the EKF's observable full state (z, v, roll/pitch, footholds,
biases) is hardware-accurate today. Good enough to close a VMC loop on; x/y/yaw
must only ever be damped or MPC-referenced over short horizons.

One number already on record from the existing gait log: over the single
completed cycle in `walk_0729_1748.npz` the EKF reports **+40.0 mm forward,
−2.7 mm lateral** against a commanded 40 mm — measured between settled HOLD4
poses, excluding the dead-reckoned rise. Session A turns that into a
tape-verified drift rate over a longer walk.

---

## 2. Phase roadmap

### Phase 1 — EKF close-out (≤ half a day)
The checkboxes in §1a. Offline half is **done** (`ekf_closeout/`: health fix,
corrected gates, world-frame command driver, 28-check suite). Remaining:
the two hardware sessions in `ekf_closeout/RUNBOOK.md`. Gate: clean PASS on a
fresh walk log with a tape-measured x/y error inside max(20 mm, 10 % of
distance), plus a z-scale check within 10 mm at three heights.

### Phase 2 — Torque mode: VMC stand on hardware
Entry point exists: `vmc/vmc_stand_hw.py` (CROUCH via native position →
EKF init on static hold → RISE/HOLD under whole-body VMC torque).

1. Tethered/supported first runs, `--tau-max` starting low (≈ 3 N·m),
   SafetyGate + estop paths already wired.
2. Verify the sim-tuned VMC gains transfer; expect retune of Kp_z/Kd and the
   grasp-map damping for real motor torque fidelity (iq→τ constant, friction).
3. Add disturbance rejection test: push the trunk, confirm compliant recovery
   (this is what position mode fundamentally can't do).

**Gate T1:** unassisted VMC stand ≥ 60 s, recovers a hand push, no SafetyGate
trips, EKF healthy ≥ 99 % during stance.

### Phase 3 — Torque mode: quasi-static crawl under VMC
Replace the position-mode gait with torque throughout — same crawl schedule
(the `crawl_hw2.0` planner logic is reusable), new execution layer:

1. **Weight shift under VMC**: track the CoM-shift reference with the body
   wrench instead of position IK. Reuse the tau-feedback UNLOAD gate.
2. **Swing legs**: Cartesian PD + gravity compensation via J^T (already in
   `dog5_vmc_core` swing path, sim-proven).
3. **Contact handling**: schedule-driven contacts with the tau gate as
   confirmation (as today), feeding both the EKF and the grasp map.
4. Keep the IMU tip-over run-stop and lean-watch abort semantics from the
   position crawl.

**Gate T2:** multi-cycle torque-mode crawl matching the position gait's steps
(40 mm), with measured compliance on foot-strike (no rigid impacts).

### Phase 4 — Dynamics
Two meanings, in order:

1. **Model**: move from the quasi-static grasp map to full rigid-body
   dynamics — add trunk momentum terms (ḣ = ΣGRF + mg) so the wrench
   distribution stays correct during acceleration; swing-leg inertia
   compensation if tracking demands it. `dog5_description` already carries the
   model; validate mass/inertia against the MuJoCo build (remember the
   `DOG5_MASS_KG=0` incident — assert on load).
2. **Gait**: first a faster crawl, then **trot** (two-beat, flight-free) in
   MuJoCo with the same VMC core, then hardware. Trot removes the
   statically-stable safety net — this is where the EKF velocity and the
   Phase-1 σ_v margin actually get spent.

**Gate D1:** trot in MuJoCo on the hardware-fidelity model (EKF in loop, not
sim truth). **Gate D2:** hardware trot in place, tethered.

### Phase 5 — Convex MPC
Standard single-rigid-body convex MPC (Di Carlo et al., Cheetah 3 style) on
top of the Phase 3/4 stack — MPC plans GRFs over a horizon; the existing
J^T/VMC layer executes them, swing legs unchanged:

1. **Model**: SRB with yaw-rotated inertia; inputs = per-foot GRFs; friction
   pyramid + unilateral constraints; contact schedule from the gait.
2. **Solver budget**: QP ≈ 12 vars × 10-step horizon at 25–50 Hz on the Pi 5.
   Prototype with OSQP in the existing off-thread worker pattern (EKF worker
   already proves the CAN loop tolerates a compute thread). Measure solve
   time *first*; this is the phase's main risk.
3. **State in**: EKF v (body), roll/pitch, z, ω — all validated in §1. x/y/yaw
   enter only as horizon-relative references (drift-safe by construction).
4. **Foothold planning**: Raibert heuristic + capture-point correction.
5. Validate in MuJoCo against the VMC baseline (same tests, V-gates), then
   hardware: stand → crawl → trot, each behind its predecessor's gate.

**Gate M1:** MPC stand + crawl in MuJoCo ≥ VMC baseline tracking. **Gate M2:**
hardware MPC trot with velocity-command following.

---

## 3. Risks and standing constraints

- **Torque fidelity is the big unknown of Phase 2**: iq→τ linearity, gear
  friction, and backdrivability at the 10:1 stage were only exercised in the
  compliance demos. Budget a per-joint τ calibration check (hang known
  weights / measured-current stall) before trusting the grasp map.
- **10 ms watchdog** applies in torque mode too — the wrench stream must never
  gap; keep the EKF/MPC off-thread pattern and the CAN loop dumb.
- **The min_eig health fix (§1b) is not yet in-tree.** Any controller that
  gates on `healthy` mid-swing will see false unhealthy frames until
  `dog5_state_estimator.outputs()` carries the relative tolerance.
- **Abduction limit** caps lateral authority (~2 cm shift) — trot and MPC
  references must respect it; lateral velocity commands will saturate early.
- **Parallel sessions** edit this tree — recheck mtimes/self-tests before
  editing shared files (`crawl_hw2.0`, `vmc`).
- Original stand scripts are frozen references; new work goes in new runners.
