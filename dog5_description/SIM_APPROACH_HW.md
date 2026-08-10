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

### 2.2 The CAN budget

The control process is not 500 Hz and not parallel. Mirror
`stand_by_position_command.py` exactly:

- `CONTROL_HZ = 250` is the rate **every** motor is refreshed at, not a
  budget divided across the drivers. One sweep commands all 12 and takes
  4 ms; each motor's slot inside it is ~333 µs.
- All 12 drivers share one 1 Mbit/s bus. A `0xA4` command and its reply are
  one 8-byte frame each, ~130 bits with worst-case stuffing, so a
  command+reply pair costs ~260 µs and a full sweep ~3.1 ms. That puts the
  ceiling at **~320 Hz per motor**; 250 Hz is **78% bus load**.
- Encoders are read and the stage machine / gates / trips run **once per
  sweep**, i.e. at 250 Hz — the same rate as the commands.
- **The binding constraint is compute, not bandwidth or feedback rate.**
  Every motor must hear from the host inside 10 ms or its driver latches
  *input lost*, so any host-side block longer than ~2.5 sweeps trips it.
  Budget host work against the 4 ms sweep, not against the bus.

> **Corrected 2026-08-10.** This section previously said each tick services
> one motor, making the state rate 250/12 = 20.8 Hz, and called that "the
> single most important fidelity constraint". That was wrong: on hardware
> every motor runs at 250 Hz — at 20.8 Hz per motor the 10 ms watchdog would
> latch on every sweep and the robot could not stand. The 20.8 Hz figure was
> an artifact of the simulator's slot loop, not a property of the robot, and
> it under-ran the real feedback rate by 12×.

### 2.3 The encoder

- Output-side, multi-turn, calibrated: joint angle = calibrated motor
  output, no software zero, no gearbox division.
- Quantize sim `qpos` to 0.0005 rad before anything sees it
  (~0.03°, conservative for a 16-bit multi-turn encoder).
- Controller-side velocity only ever comes from `EncoderVelocity` on the
  quantized, 250 Hz samples; the "driver-reported" speed used by the hard
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
- **Overspeed hard trip (8 rad/s) — hardware prints are telemetry, not
  motion** (measured 2026-07-18). True joint-speed peaks over the entire
  repertoire (privileged `qvel` probe, judge-only): **0.879 rad/s**
  (FR_abd, CROUCH load take-up transient); every 3-leg/crawl phase stays
  ≤ 0.56 rad/s. That is ~9× under `QD_ESTOP_HARD = 8.0` — a genuine 8 rad/s
  joint runaway cannot occur under these position commands. But
  0.879 rad/s joint = **504 motor-deg/s**: if the status-reply speed field
  is motor-side (like the `0xA4` command speed field the code already
  divides by `GEAR`; `_joint_state` converts the reply with `deg2rad()`
  and **no ÷10**), the hardware reads that same benign transient as
  8.8 "rad/s" → instant hard-tier trip, no encoder confirmation. The
  −5.9 rad/s RR_abd "nuisance reading" already baked into the gate's
  self-test = 338 dps = 0.59 rad/s joint — exactly the transient band the
  sim measures. Actions: (1) bench-verify the reply-speed units with one
  slow `0xA4` move (log `mb.speeds_dps()` vs encoder FD; ratio ≈ 10 →
  motor-side → divide by `GEAR` in `_joint_state`); (2) give the hard tier
  the same encoder confirmation (or a 2-sample streak) as the 7.0 tier —
  the extra sweep costs 4 ms, in which a real 8 rad/s runaway travels
  0.03 rad, negligible against the 2.6 rad soft range; (3) stick-slip snap
  release of a pinned foot can cause *real* millisecond spikes — the §7
  pre-lift fix removes that source too. The sim cannot reproduce these
  trips: its reported speed is the servo's own filtered encoder FD
  (clean, joint-side by construction); the driver speed-field noise
  channel is not modelled.

## 7. Hardware-gap experiments (`hw_gap_experiments.py`, 2026-07-17)

Run after the first hardware attempt at CoM-shift leg-unloading failed
("body shifts and is stable, but the leg will not move"):

- **A — friction demand of the shift.** During SHIFT the feet ride the
  friction cone (tangential/normal up to 0.86 at μ = 1.0), i.e. they
  micro-slip even in sim — and it does not matter. Static stability
  depends only on the CoM-vs-feet geometry in the trunk frame, which is
  the same whether the body slides over the feet or the feet creep under
  the body. A CoM shift therefore "works and is stable" on any normal
  floor, exactly as observed on hardware.
- **B — swing while loaded** (unload gate bypassed, 30 mm horizontal
  move). The shift leaves ~8–9 N on the swing foot. Commanding a
  horizontal move at that point is resisted by μ·N of grip: at μ = 2.0
  (rubber) with no pre-lift the foot travels only 13.0 of 30 mm and
  shoves the body 4.4 mm sideways — the observed "leg won't shift"
  failure, reproduced. Pre-lift fixes it: 10 mm suffices at μ = 1.0,
  **15–20 mm at μ = 2.0**. Friction never resists the vertical pre-lift
  itself.
- **C — calibration zero error** (±2° on all 12 joints). Mis-zeroing
  scrambles the statically indeterminate 4-leg load sharing — one run
  measured FL = 5.2 N / FR = 25.2 N / RL = 25.2 N / RR = 4.2 N where the
  ideal is 14.3 N each. This is why "wait for the shift to unload the
  leg" can never be a reliable gate. The full 3-leg sequence still
  passes at every tested pre-lift (10/15/20 mm) because the pre-lift
  unloads kinematically regardless of the skewed distribution.

  *(All three re-run 2026-08-10 at the corrected 250 Hz per-motor rate;
  every conclusion and the 15–20 mm pre-lift recommendation held. Only
  the quoted figures moved slightly — A's peak ratio 0.96 → 0.86, B's
  worst-case travel 12.6 → 13.0 mm, and C's per-leg loads — because the
  tighter loop tracks the streamed IK a little better.)*

**Hardware procedure that follows:** shift (margin from encoder FK) →
pre-lift 15–20 mm → confirm clearance with the encoder-FK stance-plane
distance (and/or the torque drop) → only then command any horizontal
foot motion. Before the next session, print the 12 measured torques in
the 4-leg stand: a large left/right/diagonal asymmetry is the
mis-calibration signature from experiment C, and re-zeroing shrinks it.

## 8. Kinematic crawl (`walk_dog5.py`, 2026-07-17)

The step primitive chained into the hardware crawl gait
(RR → FR → RL → FL), still position commands + encoder-FK gates only:

- The sprawl stand's front legs are near full reach — stepping forward
  from it is IK-infeasible (14 mm residual). Fix: one **GATHER cycle**
  first, re-anchoring each foot 50 mm inboard with the same
  shift/pre-lift/swing machinery (never dragging a loaded foot), then
  30 mm forward steps with the body advancing step/4 per leg slot.
- Full run (2 gait cycles, hw speed caps): PASS — 67 mm traveled
  (60 commanded; micro-slip nets slightly over), swing feet 0 N and
  ≥ 13 mm clearance, tilt ≤ 2.6°, liftoff gates ≥ 19 mm, no aborts,
  peak torque RL_pitch 4.32 N·m (same ≈ 6 N·m trip recommendation).
- Rear-swing margins are the thin ones (~19–26 mm vs ~45 mm for front
  swings) — on hardware, watch the SHIFT_GATE margin printout on RR/RL
  steps first.
