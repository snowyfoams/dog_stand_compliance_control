# crawl_hw2.0 — Work Review (2026-07-23)

One day of work: from "the torque-mode gait cannot unload a rear leg" to a
DOG5 that **walks autonomously** — 3+ gait cycles, 120+ mm, 40 mm steps, no
operator input, no aborts.

---

## 1. What was realized

| Result | Status |
|---|---|
| 3-leg stand hold, swing foot lifted and held | **Hardware-passed, all four legs** |
| One slow gait cycle (4 steps, RR→FL→RL→FR) | **Hardware-passed** |
| Multi-cycle autonomous walk (`--auto --cycles N`) | **Hardware-passed** (3+ cycles, 120+ mm) |
| Step length 30 → 40 mm | Validated + hardware-passed |
| Yaw (heading) bookkeeping: initial vs final on stop | Added |

Origin: the sim branch `dog5-crawl-sim` proved this control shape under
emulated hardware constraints (`stand3_dog5.py`, `crawl_dog5_sim.py`,
`SIM_APPROACH_HW.md`); this folder is the port back onto the real CAN bus
per its §5/§8 recipe.

## 2. How it was realized

### Control mode — kinematic position control

No torque law.  Motion is produced by each driver's **native 0xA4
multi-turn position servo** (joint-angle target + motor-side speed cap per
CAN command).  Our code plans *where the feet should be*; the drivers do
the actual control.  Safety is feedback trips, not commanded caps:
measured-torque trip 6.0 N·m, encoder-speed trip 1.0 rad/s, soft limits
±(1.75, 2.6, 2.6) rad, overspeed/temp/CAN-miss e-stops, and the IMU
(25° tip-over run-stop, 6° lean-watch step abort).

### Control loop

```
250 Hz per motor x 12 motors  ->  one CAN slot every 333 us
  each slot:  command ONE motor (0xA4 target + speed cap)
  every 3rd slot:  refine ONE leg of IK (<= 2 damped-Newton iters, warm)
  every 12th slot (~20.8 Hz):  the "sweep" -- read all encoders/torques,
      run the stage machine, gates, trips, IMU checks, update foot targets
```

- **Encoder-only state**: quantized joint angles, `EncoderVelocity`
  low-pass, driver torque telemetry.  All FK/IK/Jacobians from the NumPy
  `dog5_kinematics` module.  No dynamics model.
- **Hard real-time constraint**: any motor silent > 10 ms latches
  "input lost".  Solving 4 legs of IK in one sweep block (5–7 ms on the
  Pi) tripped this; the fix is the per-slot IK spread above.  Measured
  after: 1.7–2.3 ms compute/cycle, worst gap ~6.3 ms, `blk=` ≤ 0.9 ms in
  the real walk, IK tracking lag ≤ 1 mm.

### The step machine (each step, all gates encoder-verified)

```
SHIFT  (body 32 mm diagonal off the swing corner)
  -> gate: settle + stance-triangle FK margin >= 15 mm
UNLOAD (pre-lift 20 mm -- unload KINEMATICALLY, never wait for a load handoff)
  -> CLEAR GATE: ABSOLUTE stance-plane rise >= 4 mm AND swing |tau| <= 0.7 N*m
     (timeout: +5 mm pre-lift retry, to 30 mm, then graceful abort)
LIFT (+14 mm) -> SWING (foot +40 mm fwd) -> LOWER
  -> TOUCHDOWN gate (plane ±8 mm) -> commit the MEASURED anchor
LOAD -> RECENTER (body -> neutral advanced +10 mm)
```

### Control mode of every stage

Everything after READ streams **0xA4 native position commands** every CAN
slot; what differs per stage is *where the targets come from*, the
motor-side speed cap, and *what advances the stage*.

| Stage | Targets commanded | Speed cap (motor-side) | Advances on |
|---|---|---|---|
| READ (preflight) | **zero-torque keepalive** (0xA1, iq=0) — motors powered, no motion | — | operator ENTER (pose still + healthy) |
| CROUCH | fixed joint pose `Q_CROUCH` (joint-space, no IK) | 100 dps | settle gate (err ≤ 4.6°, \|qd\| ≤ 0.25, 0.5 s) |
| WAIT_CROUCH | hold `Q_CROUCH` | 100 dps | operator ENTER |
| STAND | **streamed IK**: feet fixed at crouch xy, body height crouch → 0.175 m over 8 s (smoothstep) | 100 dps | stream end → STAND_SETTLE settle gate |
| HOLD4 | hold last IK targets | 250 dps | ENTER / auto (next step), P parks |
| SHIFT | streamed IK: body translates 32 mm diagonal, all 4 feet anchored | 250 dps | stream end → SHIFT_SETTLE |
| SHIFT_SETTLE | hold targets | 250 dps | settle + **FK margin ≥ 15 mm** gate |
| UNLOAD | streamed IK: swing foot z + pre-lift (20 mm), other 3 anchored | 250 dps | stream end → CLEAR_GATE |
| CLEAR_GATE | hold targets (refiner keeps polishing) | 250 dps | **plane rise ≥ 4 mm AND swing \|τ\| ≤ 0.7** (timeout → re-stream UNLOAD at +5 mm) |
| LIFT | streamed IK: swing z + 14 mm | 250 dps | timer (stream end) |
| SWING | streamed IK: swing anchor x + 40 mm at height | 250 dps | timer |
| LOWER | streamed IK: swing z → stance level | 250 dps | stream end → TOUCHDOWN |
| TOUCHDOWN | hold targets | 250 dps | settle + **measured foot on stance plane ± 8 mm** → commit measured anchor |
| LOAD | hold targets (dwell 1 s — foot re-takes load) | 250 dps | timer |
| RECENTER | streamed IK: body → neutral advanced +10 mm | 250 dps | stream end → HOLD4 |
| PARK | fixed joint pose `Q_CROUCH` | 100 dps | settle gate → PARKED |
| PARKED | hold `Q_CROUCH` | 100 dps | X only |

Notes:
- "Streamed IK" = the sweep (20.8 Hz) moves the *foot targets*; the
  per-slot refiner turns them into joint targets (≤ 2 Newton iters/leg);
  every slot re-sends the current joint target to one motor.
- "Hold" stages keep commanding the same targets — the drivers' internal
  servos are the actual hold controllers; the refiner converges the IK
  residual to < 0.1 mm while gates measure.
- Only two speed caps exist: 100 motor-dps (≈10 joint-°/s) for the big
  joint-space moves, 250 motor-dps for all streamed/hold phases where the
  targets themselves move slowly and the cap is just a runaway limiter.

Ground-frame anchor bookkeeping: feet live where they *actually landed*
(measured FK + body position), so planning error cannot accumulate.
Aborts are graceful — an airborne failure lowers the foot where it is,
commits it, recenters, and waits; the run never e-stops for a gate.

### Mechanics — what the geometry allows

- **Stance: IN-PLACE** (feet at crouch xy, body height H = 0.175 m).  The
  sim's sprawl stand needs an 8 cm outward foot slide; real floor grip
  stalls a knee (46° short, no torque — driver current-limits quietly).
  Rising vertically over the feet eliminates sliding entirely and made
  **all four** swing legs IK-feasible (the sprawl blocked the rear two).
- **The abduction soft limit (±1.75 rad) is the binding constraint
  everywhere**: it caps the shift at 32 mm, the swing-foot raise at
  ~44 mm, and killed every "just shift more" idea.
- **Step 40 mm**: the diagonal shift's backward component subtracts from
  front-leg reach, so front steps shift only 15 mm back (lateral stays
  22.6 mm).  45 mm is a hard leg-length wall.
- H = 0.175 (up from 0.17) frees abduction travel for the higher raise.

### Measured on hardware (the multi-cycle walk log)

| Quantity | Value |
|---|---|
| Clear gate | first try every step; rise +19.7–19.9 mm at 20 mm pre-lift |
| → real corner sag | **~0.2 mm** (sim guessed ~12 mm — real servos are far stiffer) |
| Touchdown on plane | +0.2 … +1.3 mm |
| Anchor advance | exactly +40 mm per cycle, every foot |
| Stance margins | 20.5–30.9 mm (gate 15) |
| Airborne swing torque | 0.2–0.5 N·m (loaded ≥ 0.9 — the 0.7 gate separates cleanly) |
| Peak joint torque | 2.0 N·m (trip 6.0) |
| Control compute | `blk` ≤ 0.9 ms (budget ~6) |

---

## 3. Current problem: unexpected roll

### Symptom

During every FL step's LIFT/SWING the trunk rolls to **−7.2° peak**
(left-side-down, slight nose-down); RR steps reach ~−5.5° around
touchdown/load.  It stays under the 6° lean-watch *delta* and nothing
fails — but 7° of uncommanded body roll is large, and it is the first
thing that will bite when the walk speeds up or the floor is uneven.

### Why (analysis)

1. **The encoders cannot see it.**  The clear gate measures ~0.2 mm of
   sag via encoder FK while the IMU shows 7° of real body roll
   (≈ 14 mm of differential corner height at half-track 0.1125 m).  The
   joints are tracking their targets almost perfectly — the deflection
   lives in what sits *outside* the encoders: gear backlash under load
   reversal, rubber-foot compression, and frame flex.  This is the same
   lesson as the torque-mode era (encoder FK overstates stability), now
   quantified in position mode.
2. **Position mode has no posture correction.**  The torque-mode stand
   had the support trim + the IMU leveling loop (`--imu-level`); the
   position track currently commands rigid-geometry targets and lets the
   unmodeled compliance land where it lands.
3. **FL is worst by design**: its step uses the reduced 15 mm back-shift
   (the price of 40 mm steps), so the front-left corner region is the
   least unloaded when its foot leaves — more load redistribution onto
   compliant elements, more roll.

### IMU calibration (do this before any trimmed run)

The leveling trim drives the body toward *whatever the IMU calls level* —
so the mounting-offset zero must be fresh and taken on the test floor.
Offsets drift ~0.3° between floor spots; re-zero at the start of a
hardware session or whenever the robot moves to a different surface:

```bash
cd ~/Documents/can_motor_control/IMU_sensor
python3 imu_frame_test.py
#   1. dog in CROUCH on the test floor, motors OFF, hands off
#   2. press  z   -- averages 1 s of roll/pitch -> mounting offsets
#   3. press  s   -- saves to imu_calib.json
#   4. press  q   -- quit; live line should now read ~ +0.00/+0.00
```

Every tool (stand, crawl, walk) auto-loads `imu_calib.json` on startup and
echoes the offsets in its `[imu] streaming ... offsets` line — check that
line matches what you just saved.  Never zero on a tilted surface: the
trim would then faithfully hold that tilt.  (Full background:
`IMU_sensor/review.md` — frame conventions, the two calibration layers,
and why yaw gets no offset.)

### Fix plan (full design — ready to implement)

**Step 1 — Confirm the FL mechanism (one hardware run, zero code).**

```bash
cd ~/Documents/can_motor_control
.venv/bin/python dog_stand_compliance_control/crawl_hw2.0/walk1_hw.py \
    --step 0.030 --auto --cycles 1 --time-scale 0.8
```

Record the peak `rp=` roll on the FL and RR LIFT/SWING lines and compare
against the 40 mm baseline.  **Correction:** `FRONT_BACK_SHIFT_M` applies
to front steps regardless of `--step`, so this A/B varies **step length
only** (shift held constant) — a cleaner experiment than originally
described.

**Step 1 RESULTS (2026-07-23, hardware):**

| | `--step 0.030` | `--step 0.040` |
|---|---|---|
| 2-leg diagonal episodes | **none observed** | **yes** — on FL lift (and RR lift) the body teeters onto a diagonal pair |
| Yaw veer per cycle | +1.5° (left) | +1.8° (left) |

Interpretation:

- The 2-leg teeter (= the −7.2° roll) **scales with step length**: the
  longer swing carries leg mass further forward, lasts longer, and leaves
  the feet more spread mid-cycle — enough to push the *real* CoM (which
  the encoder-FK margin cannot see) onto the stance-diagonal edge.  At
  30 mm the walk stays honestly 3-legged.
- The yaw veer is systematic and left in both runs, roughly scaling with
  step — consistent with asymmetric stance-foot micro-slip during the
  body streams, plus pivot-slip during the diagonal teeters (why 40 mm
  veers more).  ~1.6°/cycle compounds to ~16° in ten cycles: a steering
  correction (differential per-side step length driven by the yaw
  reading) is now a motivated future feature, after the trim.
- Decision: proceed to step 2 unchanged — the trim directly counters the
  diagonal teeter (it extends the unloading corner's leg, pushing load
  back onto the third foot).  Until the trim exists, **30 mm is the
  honest default for unattended runs**; 40 mm is fine supervised and is
  the stress case the trimmed walk must pass.

**Step 2 — `--imu-level` for the walk (the IMU graduates from judging to
steering).**  Port the hardware-proven stand integrator with a
*phase-aware lifecycle* — this is the only genuinely new design element:

- **Law (unchanged from the stand):** filtered roll/pitch (LPF 10 Hz),
  deadband 0.2°, per-leg error `dz = −x·pitch + y·roll` using each foot's
  current trunk-frame xy, integrated at Ki 0.25/s, increments zero-mean
  across the participating legs.  **Clamp ±6 mm** (tighter than the
  stand's ±12 — these feet bear load mid-gait).
- **Applied as z-offsets on the PLANTED feet's targets** (added on top of
  the geometric targets each sweep).  The airborne swing foot never
  receives trim — its pre-lift/lift clearance stays exactly as commanded.
- **Lifecycle by stage** (the safety core):

  | Stages | Trim behavior | Why |
  |---|---|---|
  | HOLD4, SHIFT, SHIFT_SETTLE, LOAD, RECENTER | integrate, 4 legs, zero-mean | normal leveling on full support |
  | LIFT, SWING | **integrate, 3 stance legs**, zero-mean over 3 | this is where FL's −7° peaks — the trim fights it live |
  | WAIT_UNLOAD, UNLOAD, CLEAR_GATE, WAIT_SWING | **freeze** | the plane-rise measurement is running; trim moving stance z would corrupt `rise` |
  | LOWER, TOUCHDOWN | **freeze** | the touchdown-plane measurement is running |
  | CROUCH, STAND, PARK, PARKED | reset to zero | no valid attitude reference yet / shutting down |
  | IMU stale > 50 ms | freeze + warn | same as the stand |

- **Swing-foot handoff rule:** entering UNLOAD zeroes the swing leg's own
  stored trim (so a stale offset is never buried under the pre-lift);
  after TOUCHDOWN + LOAD it rejoins integration from zero.
- **Interaction guarantees:** frozen during every measurement window, so
  the clear gate and touchdown gate see exactly the geometry they saw in
  the proven no-trim walk; clamp ±6 mm keeps any single leg inside the
  margins measured on hardware (20.5 mm worst).
- **Expected effect:** at 7° roll the geometric error is ~14 mm — the
  integrator (rate-limited, clamped) removes ~5–6 mm within one swing and
  holds a standing correction across steps, roughly halving the peak
  roll; the 4-foot phases return the body to level between steps.
- Opt-in flag `--imu-level` (hard-error if the IMU is absent), so the
  hardware-proven no-trim walk stays reachable verbatim.
- **Offline self-test additions:** trim sign check (left-down roll →
  left stance feet extend), zero-mean, clamp, freeze-during-gate stages,
  swing exclusion, stale freeze, and a full dry-run cycle with a
  synthetic constant tilt asserting the gates still pass untouched.
- **Run commands (after implementation):**

  ```bash
  # offline first, as always:
  .venv/bin/python dog_stand_compliance_control/crawl_hw2.0/walk1_hw.py --self-test

  # first trimmed hardware run -- one cycle, ENTER-gated, watch lvl offsets:
  .venv/bin/python dog_stand_compliance_control/crawl_hw2.0/walk1_hw.py \
      --imu-level --cycles 1 --time-scale 0.8

  # acceptance run (the criteria below):
  .venv/bin/python dog_stand_compliance_control/crawl_hw2.0/walk1_hw.py \
      --imu-level --auto --cycles 3 --time-scale 0.8
  ```

**Step 3 — Tighten the lean watch (after step 2 is proven).**
Add `--lean-abort-deg` (default 6.0).  Once a trimmed walk shows peak
roll ≈ 3–4°, drop the default toward 4° so the watch guards against real
tipping rather than routine elastic roll.  Also re-run the yaw
initial→final report over several cycles — if the trim changes the net
veer, that feeds the future steering feature.

```bash
# verify the tightened watch does not false-trigger on a trimmed walk:
.venv/bin/python dog_stand_compliance_control/crawl_hw2.0/walk1_hw.py \
    --imu-level --auto --cycles 3 --time-scale 0.8 --lean-abort-deg 4.0
# then press X at the end and read the [imu] yaw net-rotation line
```

**Acceptance criteria** (all from one `--auto --cycles 3 --imu-level`
run): peak |roll| during FL steps ≤ 4°; every clear gate still first-try
at 20 mm pre-lift; touchdowns still within ±2 mm; no trim-caused aborts;
`lvl` offsets visibly non-zero and bounded ≤ 6 mm.

Not planned: modeling the compliance (sim §2.1 stiffness identification)
— worth doing only if the trim proves insufficient.
