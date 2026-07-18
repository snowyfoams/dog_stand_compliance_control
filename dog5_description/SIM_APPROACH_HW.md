# Making the MuJoCo Sim Approach the DOG5 Hardware

Goal: run weekend experiments in MuJoCo whose results transfer to the real
robot with minimal surprise. The rule is simple:

> **The sim must run the hardware's code path, see only the hardware's
> signals, and command only what the hardware can command.**

The current hardware track (branch `dog5-live-mirror`) is **kinematic
position control**: joint-pose and IK targets streamed as native `0xA4`
absolute-position commands, encoder-only feedback, no dynamics model, no
torque control law. The sim mirrors exactly that track —
`stand_by_position_command.py` for staged pose moves and
`crawl_dog5_hw.py` for the shift/unload/lift logic.

## 1. Reuse the hardware code verbatim

Anything that is pure software runs unchanged in sim. Copied into this
directory from the live-mirror branch:

| Piece | Source (hw file) | Role in sim |
|---|---|---|
| `dog5_kinematics.py` | same file | FK + foot Jacobian, trunk frame, no MuJoCo import. `check_dog5_kinematics.py` PASSes vs MuJoCo (max err ~1e-10) |
| `support_triangle_margin()` | `crawl_dog5_hw.py` | signed CoM-projection margin from **encoder FK only** |
| `_ik_to_target()` | `crawl_dog5_hw.py` | damped-least-squares IK (100 it, λ=1e-5, step 0.35) |
| `EncoderVelocity` | `stand_by_position_command.py` | low-pass finite-difference joint speed (α=0.35, ready after 5 samples) |
| Stage gates | `stand_by_position_command.py` | settle = pose err ≤ 0.08 rad AND \|qd\| ≤ 0.25 rad/s held 0.5 s; dwell 0.5 s before next stage |
| Poses | `stand_by_position_command.py` | `RECORDED_CROUCH_DEG`, `STAND_POSE_DEG` — the exact poses already run on hardware |
| Safety numbers | `stand_dog5_hw.py` | soft limits ±(1.75, 2.6, 2.6) rad; `QD_ESTOP`=7 / `QD_ESTOP_HARD`=8 rad/s; measured-torque trip (feedback trip, not a commanded cap) |
| Shift/lift logic | `crawl_dog5_hw.py` | diagonal body shift `-d·sign(foot_xy)/√2`, liftoff refused until FK margin ≥ 15 mm, unload gate: swing pitch/knee measured \|τ\| ≤ 0.45 N·m before lift |

If a controller decision in sim cannot be expressed with these pieces, it
will not port — don't write it.

## 2. Emulate what is physical

Three hardware layers do not exist in MuJoCo and are emulated explicitly.

### 2.1 The `0xA4` native position servo (per motor)

The real motion is produced by the driver's internal position loop, not by
our code. Sim model, per joint, every physics step (500 Hz):

```
q_enc    = quantize(q)                                    # its own encoder
qd_est   = LPF(Δq_enc/dt)                                 # its own speed estimate
setpoint += clip(target - setpoint, ±(speed_dps/GEAR)·(π/180)·dt)   # speed-capped slew
τ = KP_SERVO·(setpoint - q_enc) - KD_SERVO·qd_est,  clipped to ±TAU_SERVO_MAX
```

The driver is the **only** control-side object that touches the plant, and
the only thing it reads is joint position — the physical encoder. It servos
and damps on its own quantized encoder and derived speed; MuJoCo's `qvel`
is never read. Everything downstream (controller, gates, trips) consumes
CAN telemetry only: `(q_enc, qd_est, τ)`.

- `speed_dps` is the command's **motor-side** speed field; joint side is
  `/GEAR` (10:1), matching "100 motor-deg/s ≈ 10 output-deg/s".
- `TAU_SERVO_MAX = 9 N·m` (= `TAU_HARD`, the driver's capability, *not* a
  commandable cap — `0xA4` has no torque field, exactly like hardware).
- `KP_SERVO/KD_SERVO` are **unknowns to identify**: on hardware, command a
  small `0xA4` step on one supported joint, log encoder position at
  ≥100 Hz, and fit rise time / overshoot / steady-state error under a known
  hanging load. Then set the sim servo to match. Until then the sim uses a
  stiff guess (120 N·m/rad, 2.5 N·m·s/rad) — treat absolute tracking-error
  numbers as provisional, trends as valid.

### 2.2 The CAN round-robin

The control process is not 500 Hz and not parallel. Mirror
`stand_by_position_command.py` exactly:

- Loop at `CONTROL_HZ = 250`; each tick services **one** motor
  (sends its position target).
- Encoders are read and the stage machine / gates / trips run **once per
  sweep of 12**, i.e. at ~20.8 Hz. This is the real rate at which the
  robot "knows" its own state. Any behavior that needs faster feedback than
  ~21 Hz will not transfer — this is the single most important fidelity
  constraint.

### 2.3 The encoder

- Output-side, multi-turn, calibrated: joint angle = calibrated motor
  output, no software zero, no gearbox division.
- Quantize sim `qpos` to 0.0005 rad before anything sees it
  (~0.03°, conservative for a 16-bit multi-turn encoder).
- Controller-side velocity only ever comes from `EncoderVelocity` on the
  quantized, 20.8 Hz samples; the "driver-reported" speed used by the hard
  overspeed trip is the servo emulation's own encoder-difference estimate.
  `data.qvel` is never read on the control side.

## 3. The encoder-only discipline (what the controller may NOT see)

The controller half of the sim is forbidden to read:

- trunk world pose / attitude (`data.qpos[0:7]`, `data.xmat`) — no IMU on
  the robot;
- contact forces, `data.qfrc_*`, MuJoCo Jacobians, `data.site_xpos`,
  `data.qvel` — no foot sensors, no dynamics oracle, no perfect velocity;
- true CoM (`data.subtree_com`).

All FK, IK, Jacobians, and support margins on the control side come from
the NumPy `dog5_kinematics` module — the same file the hardware runs.

Allowed inputs: quantized `q`, `EncoderVelocity` output, and per-joint
"measured torque" (the servo emulation's output, standing in for the
driver's torque telemetry used by `--position-torque-trip` and the crawl
unload gate).

The privileged quantities are still computed — but only as the **oracle**:
pass/fail metrics printed by the harness (true CoM margin, trunk tilt,
swing-foot contact force). Oracle values judge the run; they never steer it.

## 4. Physics gaps and how each is handled

| Gap | Handling |
|---|---|
| Friction coefficient unknown | sweep μ ∈ {0.5, 0.8, 1.0} in the robustness suite; the stand must not depend on high μ |
| Gearbox friction/stiction not modeled | measured-torque thresholds (2.0 trip, 0.45 unload) will read differently on hw; sim reports its torque numbers so thresholds can be rescaled, and gates use dwell times so stiction-delayed motion still settles |
| Backlash not modeled | keep ≥ 2× the 15 mm liftoff margin at the planned shift; margins that survive 0.01 m of foot-position error are backlash-proof |
| Servo gains guessed | identify per §2.1; meanwhile sweep KP_SERVO ±50% in robustness suite |
| Mass/CoM error (FK assumes CoM at body origin) | sweep total mass ±10% and trunk CoM offset ±10 mm; the crawl's unload gate (measure, don't assume) is the hw backstop |
| Floor not rigid/level in reality | foot-sphere contact already compliant; margin reserve covers small tilt |

## 5. Validation ladder (sim → hardware)

1. `check_dog5_kinematics.py` — FK/Jacobian vs MuJoCo: **PASS** (required
   before anything else).
2. `stand3_dog5.py --self-test` — offline: every planned IK waypoint
   reachable, inside soft limits, FK margin ≥ 15 mm at liftoff, static
   stance torques under the servo capability.
3. `stand3_dog5.py --headless` — full sequence in sim; pass criteria are
   oracle-based: swing foot truly airborne (contact force 0, clearance
   > 10 mm), trunk tilt < 5°, no trips, clean re-plant and recenter.
4. Robustness sweep (`--sweep`) — the §4 perturbations; require pass on
   all combinations before calling the idea proven.
5. Port: the sim stage machine drops into a
   `stand_by_position_command.py`-style runner — same targets, same gates,
   same trips — with `mb.position(mid, deg, dps)` replacing the servo
   emulation. Operator Enter replaces the sim's auto-advance at every
   stage boundary.

## 6. Transfer notes found in sim (measured, 2026-07-17)

All from `stand3_dog5.py` runs at hardware speed caps (100 motor-deg/s
stages, 250 motor-deg/s streamed phases). The stand-up stage is the
hardware method by default (`--start crouch`): the robot is placed in the
recorded crouch and CROUCH → STAND are the same two native position moves
as `stand_by_position_command.py`, including the crouch→sprawl feet drag —
verified to settle in ~8 s in sim, matching what already worked on the
robot. (`--start flat` keeps the sim-only LIE → ROLL → CROUCH escape from
the singular calibration pose.)

- The hardware stand pose (`STAND_POSE_DEG`) is a sprawl: feet at
  x ≈ ±0.42 m, height 0.133 m, support polygon only ±0.08–0.10 m wide in y.
- Lifting any foot **without** a shift leaves the CoM ≤ 9 mm from the
  support edge (outside it for rear legs) — the shift is mandatory, as the
  crawl already assumes.
- Diagonal shift 0.05 m before lifting FR gives a 51 mm FK margin
  (0.03 m → 34 mm; the crawl default 0.03 m is fine for its brief swings,
  the long static 3-leg hold uses 0.05 m).
- **Expected trip on hardware**: the loaded front hip pitch peaked at
  **4.27 N·m** measured (static prediction 4.25) — the default
  `--position-torque-trip 2.0` **will fire**. For 3-leg tests use ≈ 6 N·m
  (confirm against the measured value on the supported robot) or stand
  less sprawled before shifting. Heaviest corner: FL at ≈ 26 N of the
  57 N total when lifting FR.
- **Unload gate in position mode**: the crawl's 0.45 N·m trip assumes a
  planted foot with faded feedforward. With the position-mode pre-lift
  (10 mm) the airborne swing leg still measures ≈ 0.5 N·m of leg-gravity
  torque; a still-loaded leg measures ≥ 0.9 N·m. The gate threshold moves
  to 0.7 N·m to separate the two — re-measure both levels on hardware.
- Headless verdict (nominal): swing foot force 0.000 N for the full 10 s
  hold, clearance 42 mm, trunk tilt ≤ 0.43°, true CoM margin ≥ 46.6 mm,
  encoder FK margin ≥ 50.8 mm — and FK margin agreed with the true-CoM
  margin within ~4 mm, i.e. the CoM-at-origin assumption is good on this
  robot.
- Robustness sweep: **10/10 PASS** across μ ∈ {0.5, 0.8, 1.0}, mass ±10%,
  trunk CoM ±10 mm (y), servo KP ±50%, and the combined worst case.
  Weakest link: half-stiffness servo → tilt 1.0°, clearance 33 mm (from
  elastic sag, still passing). The idea does not depend on friction,
  exact mass, or exact servo gains.

## 7. Hardware-gap experiments (`hw_gap_experiments.py`, 2026-07-17)

Run after the first hardware attempt at CoM-shift leg-unloading failed
("body shifts and is stable, but the leg will not move"):

- **A — friction demand of the shift.** During SHIFT the feet ride the
  friction cone (tangential/normal up to 0.96 at μ = 1.0), i.e. they
  micro-slip even in sim — and it does not matter. Static stability
  depends only on the CoM-vs-feet geometry in the trunk frame, which is
  the same whether the body slides over the feet or the feet creep under
  the body. A CoM shift therefore "works and is stable" on any normal
  floor, exactly as observed on hardware.
- **B — swing while loaded** (unload gate bypassed, 30 mm horizontal
  move). The shift leaves ~8–9 N on the swing foot. Commanding a
  horizontal move at that point is resisted by μ·N of grip: at μ = 2.0
  (rubber) with no pre-lift the foot travels only 12.6 of 30 mm and
  shoves the body 4 mm sideways — the observed "leg won't shift"
  failure, reproduced. Pre-lift fixes it: 10 mm suffices at μ = 1.0,
  **15–20 mm at μ = 2.0**. Friction never resists the vertical pre-lift
  itself.
- **C — calibration zero error** (±2° on all 12 joints). Mis-zeroing
  scrambles the statically indeterminate 4-leg load sharing — one run
  measured FL = 4.8 N / FR = 25.1 N / RL = 22.4 N / RR = 5.1 N where the
  ideal is 14.3 N each. This is why "wait for the shift to unload the
  leg" can never be a reliable gate. The full 3-leg sequence still
  passes at every tested pre-lift (10/15/20 mm) because the pre-lift
  unloads kinematically regardless of the skewed distribution.

**Hardware procedure that follows:** shift (margin from encoder FK) →
pre-lift 15–20 mm → confirm clearance with the encoder-FK stance-plane
distance (and/or the torque drop) → only then command any horizontal
foot motion. Before the next session, print the 12 measured torques in
the 4-leg stand: a large left/right/diagonal asymmetry is the
mis-calibration signature from experiment C, and re-zeroing shrinks it.

## 8. Position-mode crawl (`crawl_dog5_sim.py`, 2026-07-17)

The full crawl of `crawl_dog5_hw.py` — every leg stepping, gait order
RR→FR→RL→FL, graceful aborts — running as native position commands under
the same §2 emulation.  Each step is the §7 procedure made into a phase
machine:

    SHIFT → gate: settle + encoder FK margin ≥ 15 mm
    PRELIFT 20 mm → gate: encoder-FK stance-plane rise ≥ 5 mm ABSOLUTE
                          AND swing pitch/knee |τ| ≤ 0.7 N·m
                          (timeout: raise the pre-lift +8 mm and retry,
                           up to 36 mm, before aborting)
    LIFT → SWING → LOWER   (horizontal motion only after the clear gate;
                            swing height = measured corner sag + 30 mm,
                            capped at the offline-validated 41 mm)
    TOUCHDOWN gate: foot back on the stance plane (±8 mm); the MEASURED
                    anchor is committed, as the crawl does
    LOAD → RECENTER → gate: settle → next leg

`--swing-test` (step 0) crawls in place on the sprawl stand — the first
hardware test; walk mode runs a REPOSE cycle then walks.  Verdicts are
oracle-based as in §3.  Measured transfer notes:

- **Corner sag eats the pre-lift.**  Unloading a corner tilts the trunk
  1.4–1.6° toward it, which physically lowers that corner ~12 mm (~19 mm
  at half servo stiffness).  A 15 mm pre-lift left only 2.9 mm of true
  clearance and the clear gate (correctly) refused liftoff.  Three
  consequences, all applied: the default pre-lift is 20 mm (§7's upper
  value); the clear-gate threshold is an **absolute** plane rise (5 mm),
  not a fraction of the commanded pre-lift; and a clear-gate timeout
  raises the pre-lift (+8 mm, to 36 mm max) and retries before aborting,
  because the rise deficit measures exactly how much more is needed.  On
  hardware, expect the stance-plane rise readout to be ≈ (pre-lift −
  12 mm); do not interpret that as "the leg won't lift".
- **Swing height must come from the measurement, not a constant.**  The
  swing commands (measured corner sag + 30 mm) instead of a fixed 25 mm:
  at μ = 2.0 the gripping feet wind up laterally during SHIFT and the
  release tilts the trunk a further ~3.4° mid-swing, which cost a
  fixed-height swing its whole clearance (first repose swing grazed the
  floor at −0.6 mm; the first swing off the fully wound-up sprawl loses
  up to ~19 mm).  The adaptive height is encoder-only (the clear gate
  already measured the sag) and stays inside the offline-validated
  41 mm envelope.
- **The unload torque levels transfer.**  Airborne swing pitch/knee
  measured 0.36–0.64 N·m across all 12 steps of the nominal runs — the
  §6 gate at 0.7 N·m separates airborne from loaded exactly as planned.
- **Walking from the sprawl is IK-infeasible.**  The front-leg touchdown
  (step + diagonal shift) is unreachable from the ±0.42 m sprawl.  Walk
  mode therefore begins with a REPOSE cycle: each leg's first swing
  re-places its foot on a tucked rectangle x ±0.34 m, y ±0.10 m (same
  per-leg height), using the same shift/pre-lift/swing machinery, then
  the crawl walks from there.  This is §6's "stand less sprawled" made
  concrete.  Two geometry constraints found by the validators:
  - repose in **diagonal** order (RR, FL, RL, FR): the
    lateral gait order leaves a 2 mm support margin once both same-side
    legs are tucked; diagonal order keeps every mixed triangle ≥ 20 mm;
  - walk-stance y stays at 0.10 m — at y = 0.112 the 0.04 shift pushes
    the rear abduction past the ±1.75 rad soft limit during the repose
    swing (−100.3° at the limit).
- **Shift 0.04 m** (not the crawl's 0.03): rear-leg sprawl steps plan
  only ~16 mm of margin at 0.03 — one CoM error away from the 15 mm
  gate — and a −10 mm CoM offset left a 9.8 mm true margin at 0.035.
  0.04 plans ≥ 24.6 mm and keeps ≥ 12 mm true margin through the whole
  §4 sweep.
- Static mg/3 stance torque on the tucked rectangle is ~3.20 N·m (the
  sprawl 3-leg hold needs 4.3).  Peak measured torque across the crawl
  is 4.50 N·m (loaded pitch during sprawl phases) — for the crawl too,
  `--position-torque-trip` ≈ 6 N·m, as §6 concluded for the 3-leg test.
- Touchdowns anchored 0.2–2.5 mm from planned (the §7-A micro-slip and
  the measured-anchor rule absorb the rest).
- Nominal verdict (walk = repose + 1 gait cycle): all 8 steps complete,
  every swing truly airborne (contact force 0.000 N), swing clearance
  ≈ 28 mm (adaptive heights), trunk tilt ≤ 1.40°, true CoM margin
  ≥ 18.8 mm, encoder FK margin ≥ 24.0 mm, forward progress +33.8 mm vs
  +30 planned, stance-foot creep ≤ 8.5 mm per step.  `--swing-test`
  (4 in-place steps on the sprawl) passes the same checks.
- Robustness sweep: **12/12 PASS** across μ ∈ {0.5, 0.8, 1.0, 2.0},
  mass ±10%, trunk CoM ±10 mm (y), servo KP ±50%, calibration zero error
  ±2° on all 12 joints, and the combined worst case (μ 0.5, mass +10%,
  CoM −10 mm).  Weakest links: μ = 2.0 → tilt 3.5°, true margin 14 mm
  (wind-up); KP −50% → tilt 2.8°, with the clear-gate retry raising the
  pre-lift automatically (a fixed 20 mm pre-lift had aborted: rise
  1.3 mm).  Every case walks all 8 steps; no aborts anywhere.

**Port note:** the stage machine drops into the same
`stand_by_position_command.py`-style runner as §5; operator ENTER
replaces auto-advance at every phase boundary, and the first hardware
runs are `--swing-test --observe-phases` with the robot supported, as
`crawl_dog5_hw.py` already prescribes.
