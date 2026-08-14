# Aug — closed-loop trot in place

Track goal: **trot in place with the EKF closed in the loop.** Built in stages,
each one hardware-signed-off before the next starts.

| Stage | Script | Motors | EKF role | Status |
|---|---|---|---|---|
| 1. Verify | `stand_ekf_verify_hw.py` | position (0xA4) | **read-only** | **hardware PASS 2026-08-11** |
| 2. Height | `stand_ekf_height_hw.py` | position (0xA4) | **feedback → common foot-z** | hardware run 2026-08-11 (`h.npz`: loop live all HOLD4, off settled +0.5 mm, err −0.3 mm, no freezes) — sign-off pending |
| 2b. Level | `stand_ekf_level_hw.py` | position (0xA4) | **feedback → differential foot-z** | self-test PASS, hardware pending |
| 2b-x. Contact | `lift_ekf_contact_hw.py` | position (0xA4) | one-leg lift; measures zEKF vs zFK | self-test PASS; **hw 2026-08-12: they agree (expected — see below)** |
| 3. Trot | `trot_fk_switch_hw.py` | position (0xA4) | **feedback → contact schedule**; gait timed by FK | self-test PASS (64 gates); hw 2026-08-14: **feet clear at 12 mm lift** with the stance push; tips onto FL past 15 mm |

No software torque is commanded anywhere in this track. The drivers' native
position loops do the joint-level work; the EKF only ever shapes *targets*.

## Where the tunables live — `stand_params.py`

**Every number you would want to change is in `stand_params.py`, and nowhere
else.** It is pure literals with no imports, so a test, a plotting script or a
notebook can read it without dragging in python-can or the IMU.

Its docstring opens with at-a-glance tables for the ones that get hunted for
most — every **slew** limit side by side, every **height**, every **gain** —
followed by blocks grouped by concern:

| Block | What it tunes |
|---|---|
| rig geometry | `FOOT_RADIUS_M`, `IMU_BELOW_TRUNK_ORIGIN_M` — the frame every stage measures against |
| stage machine | `T_STAND`, `SETTLE_S`, `EKF_WORKER_HZ`, `QUIET_STAGES` |
| stand pose | `STAND_HEIGHT_DEFAULT` (0.19), `STAND_HEIGHT_STAGE12` (0.20), `TROT_STAND_HEIGHT_M` (0.17) |
| height loop | `HEIGHT_GAIN_PER_S`, `HEIGHT_CLAMP_M`, `HEIGHT_SLEW_M_S`, `HEIGHT_DEADBAND_M`, `HEIGHT_XCHECK_M`, table build |
| leveling loop | `LEVEL_GAIN_PER_S`, `LEVEL_CLAMP_M`, `LEVEL_SLEW_M_S`, `LEVEL_DEADBAND_DEG`, `LATCH_S`, `AGREE_VETO_DEG` |
| AHRS leveling | `AHRS_KP`, `AHRS_GAIN_PER_S`, `AHRS_SLEW_M_S`, `AHRS_STALE_S` |
| lift | `LIFT_M`, `LIFT_SLEW_M_S`, `CONTACT_OFF_M`, `LEAN_ABORT_DEG` |
| scheduled contacts | `TOUCH_FRAC_DEFAULT`, `CROUCH_PLANTED_DEFAULT` |
| trot in place | `TROT_STAND_HEIGHT_M`, `TROT_LIFT_M`, **`TROT_PUSH_M`**, **`TROT_COM_SHIFT_X/Y`**, `TROT_SLEW_M_S`, `TROT_OVERLAP_S`, `TROT_CONTACT_CLEAR_M`, `TROT_SWITCH_TOL_M`, `TROT_LEAN_ABORT_DEG` |
| safety | `TILT_STOP_DEG`, `TEMP_NOTICE_C` |

It also carries a map of **where the shared functions live** (`fk_floor_height`
is stage 1's, `LevelingLoop` is stage 2b's, and so on) so a hunt for one of
those ends in the same file.

Before this, the numbers were spread across six 600–1100 line runners and then
aliased between them — `stand_ahrs_level_hw.SETTLE_S → lv.SETTLE_S →
s2.SETTLE_S → 1.5`. Worse, eight were declared **twice** as independent
literals (`T_STAND`, `SETTLE_S`, `EKF_WORKER_HZ`, `QUIET_STAGES`,
`STAND_HEIGHT_DEFAULT`, `TILT_STOP_DEG`, and both frame constants), so editing
one copy silently left the other behind. `LEVEL_SLEW_M_S` meant 4 mm/s in the
EKF script and 10 mm/s in the AHRS one — the AHRS gains are now named `AHRS_*`
precisely so they cannot shadow.

`self-test/test_stand_params.py` keeps this true: it parses every runner and
**fails if any of them re-declares a tunable**, imports them all and fails if a
value has drifted, and checks the relations between the numbers (clamp vs slew,
veto vs authority, lift vs contact threshold, experiment A's τ vs the EKF
script's).

## Self-tests

Every runner has an offline gate suite — no CAN bus, no IMU, no hardware. They
live in **`self-test/`**, one script per module, not inside the runners:

| Suite | Covers | Gates |
|---|---|---|
| `test_stand_params.py` | no re-declared or drifted tunables; relations between them | 21 |
| `test_stand_ekf_verify.py` | stand pose, stage machine, EKF worker wiring, contact schedules, FK height, the two reporters | 55 |
| `test_stand_ekf_height.py` | leg tables, integral height loop, EKF vetoes, stage machine, sagging-plant closed loop | 40 |
| `test_stand_ekf_level.py` | rotation convention, leveling law, setpoint latch + settle gate, AHRS vetoes, FK attitude, 4-foot closed loop | 50 |
| `test_stand_ahrs_level.py` | the P+I law, its tuning claims (measured, not asserted), the startup banner | 18 |
| `test_stand_ekf_schedcontact.py` | contact schedule, masked FK height/attitude, schedule vs stage machine | 17 |
| `test_lift_ekf_contact.py` | lift manager, zEKF/zFK compare block, `--fake-contacts`, 3-leg closed loop | 26 |
| `test_trot_fk_switch.py` | FK return switch, swing state machine, stance push + reach budget, body trim + load balance, measured clearance, rocking geometry | 81 |

```bash
$V self-test/test_all.py               # all 291 gates, one process per suite
$V self-test/test_all.py level ahrs    # only the suites matching these names
$V self-test/test_stand_ekf_level.py   # one suite directly
$V stand_ekf_level_hw.py --self-test   # still works; delegates to the suite
```

`self-test/selftest_common.py` holds what the suites share: the PASS/FAIL
harness (`check`/`report`), the `PlanePlant` rigid-trunk simulator and
`C_from_rp`, the `FakeShared`/`FakeAhrs`/`FakeFeed` doubles, a **cached** table
builder (a build is ~9 s of IK), and the `walk_stages` / `time_sweep` drivers
that four suites had a copy of each. The plant and the fake EKF used to live
inside `stand_ekf_level_hw.py` — a hardware script — with two other runners
reaching in to alias them out; that is what this module ends.

## Stage 1 — `stand_ekf_verify_hw.py`

A rigid position stand with the estimator observing and logging. It commands no
torque and closes no loop, so it cannot be destabilised by a bad EKF — which is
the point: it is the measurement that tells us whether the EKF is good enough to
close a loop on.

```
ZERO-TORQUE -> CROUCH -> WAIT_CROUCH (EKF inits here) -> STAND -> HOLD4
                              ^                                     |
                              |                             P parks |
                          ENTER stands again                        v
                           PARKED  <-----------------------------  PARK
```

`WAIT_CROUCH` is the only *quiet stage*: the estimator's gyro-bias init runs
there and nowhere else, so a moving crouch cannot pollute it.

**Contact schedule — contacts stay ON through the ramps (default).** The feet
never leave the floor in this script: both ramps are a pure vertical
rise/descent with each foot's x/y *pinned* to its crouch value in the trunk
frame, so the contact point is stationary in the world. Two schedules were
possible, and **hardware decided between them on 2026-08-11**:

| | contacts ON (default) | `--dead-reckon-ramps` (old default) |
|---|---|---|
| model | physically honest for this motion | knowingly wrong, deliberately deaf |
| height during ramp | observable end to end | unobservable — open-loop double integration |
| re-anchoring | none — `handle_transitions` never fires, init landmarks stay valid | rising edge at each ramp end freezes the accumulated error in |
| **measured drift** | **none** | +11 mm over the rise, tens of mm after a park |
| failure mode | genuine foot drag enters the filter as body motion | z drifts invisibly (attitude still looks fine) |

The old default came from `vmc/ekf_stand_hw.py` ("feet drag a little, so the EKF
dead-reckons through the rise") — the conservative choice, since it cannot be
poisoned by drag if it isn't listening. But the drag it guards against turned
out not to exist at a level that matters, while the cost of not listening was
the entire height estimate. `--dead-reckon-ramps` is kept only to reproduce
older runs.

> **Do not carry "contacts always ON" into stage 3.** It is correct *here* only
> because the feet never leave the floor. A gait that lifts feet must feed the
> real schedule, or the filter gets told a swinging foot is planted — which is
> the exact failure the OFF schedule was invented to avoid.

`PARK` is the `STAND` ramp run backwards over the same `T_STAND`, so lowering is
exactly as gentle as rising and needs no second profile. It is accepted only
from `HOLD4` — parking mid-rise would reverse an unsettled pose. From `PARKED`,
`ENTER` stands again, and each `HOLD4` visit is scored separately, so a
park/stand cycle measures how **repeatable** the resting attitude is. That
matters for stage 2: a leveling setpoint is only meaningful if the robot returns
to the same attitude each time it stands.

### What it measures

Attitude is scored over `HOLD4`, after a 1.5 s settle, and printed at exit:

- **LEVEL** — resting roll/pitch mean, std, peak-to-peak. The mean is the
  **setpoint stage 2 must use**, not zero: the EKF's attitude is gravity-
  referenced through the IMU's mechanical mount, so the mount tilt appears
  here raw. (The AHRS path has its own calibrated offsets in `imu_dog`; the EKF
  path does not.) The std/p2p is the noise floor a leveling loop must live
  above.
- **AGREE** — `|EKF − AHRS|` mean and max, the independent cross-check that the
  EKF attitude is sane at all.
- **REPEAT** — printed once you have parked and stood more than once: the
  spread of the per-stand resting means. Large spread means the stand pose
  itself is not repeatable, and no leveling setpoint will hold.

### What EKF z means (and why PARKED is not 0)

`DOG5StateEstimator.initialise()` sets `st.r = 0`. **The inertial origin is
wherever the trunk was when the EKF initialised** — i.e. the crouch, in
`WAIT_CROUCH`. So:

- the **axes** are gravity-aligned, built from the measured gravity direction,
  so z really is "world vertical" in *direction*;
- the **origin** is not the floor. `z = 0` means "at the crouch height", not
  "on the ground". Expecting `z ≈ 0` back at `PARKED` is therefore correct —
  that is the same pose the origin was set at.

More importantly, z is only ever observable **relative to the currently
anchored footholds**. The leg measurement says "this planted foot has not
moved"; it says nothing about absolute height. And on every contact rising edge
`handle_transitions()` re-plants the landmark at `r + Cᵀ·s_i` — *wherever the
filter currently believes the foot is*. That accepts the current position error
as truth and freezes it in permanently.

Under `--dead-reckon-ramps` both ramps run with contacts **OFF**, so for those
5 s there is no measurement at all and z is pure accelerometer
double-integration. A ~1 cm/s² accelerometer bias residual is enough for
>100 mm over a ramp. The error is then locked in when the feet re-anchor. That
is where a `PARKED` offset comes from, and it is invisible in roll/pitch, which
stay good because attitude is directly observable from gravity. The default
schedule removes both mechanisms — see the contact-schedule table above — and
measured drift goes away entirely.

So the script prints an independent, drift-free reference next to it:

| column | meaning |
|---|---|
| `zFK` | **IMU-board** height above the floor, from encoders + attitude. FK only, no integration, so it cannot drift. |
| `zEKF` | the EKF's `r_z` — displacement from the init pose. |
| `drift` | `zEKF − (zFK − zFK_at_init)`. How far the EKF's integrated rise disagrees with what the legs say. This is the error number. |

#### Which point on the trunk

Three different reference points get called "height" here, and mixing them is
how the phantom-100 mm scare happened:

- **foot site** — `dog5_kinematics.foot_position()` returns the *centre* of a
  20 mm contact sphere, so the floor is `FOOT_RADIUS_M` below it.
- **hip axis** — the FK trunk origin. `dog5.xml` puts all four hip bodies at
  `z = 0`. This is what a tape measures to the hip pivot.
- **IMU board** — what the EKF's `r` actually tracks, and therefore what `zFK`
  reports by default (`fk_floor_height(..., ref="imu")`; pass `ref="hip"` for
  the tape-comparable number).

`dog5.xml:43` models the IMU **at** the trunk origin (`<site name="imu"
pos="0 0 0">`) — correct in sim, wrong on hardware, where the board sits on the
trunk bottom, measured 38 mm lower. The offset is a *body-frame* constant, so
its vertical component is `38 mm · C[2,2]`, shrinking slightly as the trunk
tilts (0.6 mm at 10°) rather than being subtracted flat.

Note what this does and does not change: it shifts the **absolute** `zFK` by
−38 mm so it is finally the same point `zEKF` refers to, but it leaves `drift`
untouched, because a constant cancels in `zFK − zFK_at_init`. A self-test gate
asserts exactly that (`reference choice does not move the measured rise`). So
this fixes what the number *means*, not the +11 mm.

At exit these are printed as one row per **holding** stage, chronologically:

```
[stages] EKF z=0 is the IMU at EKF init: FK floor height 6 mm (the crouch)
         zFK = IMU board above floor; the hip axis (what a tape measures) is +38 mm
  stage          n     zFK      zEKF   drift mean/max      roll   pitch
  WAIT_CROUCH   176      6mm      +0mm     +0/    -4mm   -0.91   -0.12 deg
  HOLD4        5478    169mm    +174mm    +11/   +15mm   -1.99   -0.08 deg
  PARKED       4206     14mm     +55mm    +47/   +51mm   -1.61   -0.17 deg
```

How to read it:

- **`WAIT_CROUCH` is the baseline.** It is the pose the origin was latched at,
  so its drift is ~0 *by construction*. It is printed rather than assumed
  because a non-zero value there would mean the FK reference or the origin
  latch is wrong — a different bug entirely from dead reckoning, and the report
  says so explicitly when it happens.
- **`HOLD4` localises the loss.** If drift is already large while standing, the
  *rise* ramp lost the height and the park ramp merely added to it. In the
  example above, +11 mm was gained going up and the rest coming down.
- **`PARKED` should return to the baseline.** `zFK` does (same pose); whatever
  `zEKF` still reads is accumulated error.
- The per-stage `roll`/`pitch` columns show whether the resting attitude is
  pose-dependent — crouch vs stand differing would matter to stage 2's setpoint.

### Run

```bash
V=/home/robot01/Documents/can_motor_control/.venv/bin/python

$V self-test/test_stand_ekf_verify.py                        # 55 offline gates, no hardware

# sudo is safe here -- see the $HOME note below
sudo chrt -f 50 $V stand_ekf_verify_hw.py \
    --log verify.csv --raw-log verify.npz

$V ../state_estimator/hw_replay.py verify.npz --static   # gate-C8 report
```

Keys: `ENTER` stands (from `WAIT_CROUCH` or `PARKED`), `P` parks (from `HOLD4`),
`X` stops at any time. Hold still in `WAIT_CROUCH` for at least a second — the
script warns if the raw log's static prefix is too short for `hw_replay` to
initialise well.

### Which estimate to trust for what

Stage 1 settled this, and stage 2/3 should follow it:

| quantity | use | why |
|---|---|---|
| **roll / pitch** | **EKF** | directly observable from gravity; drift-free and the thing leveling closes on |
| **absolute height, all four feet down** | **FK (`zFK`)** | no integration at all, so it cannot drift. Measured more accurate than `zEKF` while holding. |
| **height while feet are lifting** | **EKF** | FK height needs a contact assumption; mid-swing there isn't one |
| **velocity** | **EKF** | FK gives no velocity without differentiating noisy encoders |

The EKF is not redundant — it owns attitude and velocity outright. It simply
should not be asked for a height that the legs already know better.

### Pass criteria before stage 2

- `AGREE` max under ~2° on both axes (EKF and AHRS tell the same story)
- `LEVEL` p2p small and non-drifting over a ≥30 s hold (a slow ramp means bias
  is still converging — stage 2 would chase it)
- `hw_replay --static` reports PASS
- `REPEAT` spread small over 2–3 park/stand cycles (the stand pose returns to
  the same attitude, so a fixed setpoint is meaningful)
- Record the `LEVEL` means; they become stage 2's leveling setpoint

## Stage 2 — `stand_ekf_height_hw.py`

The EKF goes **into the loop**: its height drives a common foot-z offset so the
trunk holds a *commanded* height instead of whatever height the legs sag to.
Still no software torque — the loop only moves the targets the 0xA4 position
loops are given.

```
ZERO-TORQUE -> CROUCH -> WAIT_CROUCH -> STAND (open loop) -> HOLD4 (LOOP ON)
                  ^                                            |  P
              ENTER |                                          v
                 PARKED  <-------------------------------- PARK
```

### Why an integrator, and nothing else

The plant is a position servo, so the gap between commanded and achieved height
is a near-constant sag under load — stage 1 measured **13 mm** (commanded 220,
achieved 207). An integrator drives that to zero without needing to know it, and
unlike a proportional term it adds no gain to EKF noise. The self-test runs the
loop against a sagging plant and confirms it recovers the sag exactly.

It deliberately stops inside a 1 mm **deadband**, so the steady-state height
error is ≤1 mm rather than zero. That is a design choice, not slop: without it
EKF noise would make the pose chatter continuously.

### No IK in the CAN sweep

`stand_hier_hw` measured 4 legs of convergent IK at 5–7 ms — far past the 333 µs
slot. Since x/y are pinned, foot z is the only free variable, so each leg gets a
**precomputed z → joint-angle table** (0.5 mm grid, built with the same
warm-started incremental IK that keeps the stand on the crouch branch). Runtime
cost is a linear interpolation: measured **63 µs** for all four legs.

Table construction takes ~9 s at startup. It runs *before* the bus is armed, so
it cannot starve the 10 ms motor watchdog.

### Safety — FK is the watchdog

The loop is closed on the EKF, so the drift-free FK height becomes the
independent veto. Every sweep compares them; the integrator **freezes** on:

| condition | why |
|---|---|
| `EKF-FK disagree > --xcheck` (30 mm) | FK cannot drift, so a gap means the EKF has |
| EKF stale (>200 ms) | worker stalled |
| EKF unhealthy | filter's own covariance check |
| not in `HOLD4` | never wind up during a ramp |

Freezing **holds** the offset — it never unwinds. Losing the EKF leaves the
robot standing where it is rather than diving to the open-loop pose. The offset
is also clamped (±30 mm) and slew-limited (5 mm/s), and the tables are built
inside the soft joint limits, so no reachable command can exceed them.

`PARK` ramps from the pose **actually being held**, offset included, so `PARKED`
lands on the recorded crouch no matter what the loop wound in.

### Run

```bash
$V self-test/test_stand_ekf_height.py                           # 40 gates, no hardware

sudo chrt -f 50 $V stand_ekf_height_hw.py --log h.csv --raw-log h.npz

# open-loop A/B -- should reproduce stage 1 exactly
sudo chrt -f 50 $V stand_ekf_height_hw.py --height-gain 0
```

Watch the `off=` field on the status line: it should wind from 0 to ≈ the sag
(~13 mm) over a few seconds and then sit still, with `err=` collapsing toward 0
and the flag reading `LOOP`. If it reads `hold`, the loop is frozen and the
reason is printed once.

**First run:** start with `--height-gain 0` to confirm the tables reproduce the
stage-1 stand, then `--clamp 0.005` to bound the authority while you watch the
sign, before running the default.

## Stage 2b — `stand_ekf_level_hw.py`

**Two files, one job each.** The controller and the experiment are separate
runners so each reads on its own:

| file | does | keys |
|---|---|---|
| `stand_ekf_level_hw.py` | **the controller** — leveling + height, all four feet, always | ENTER, P, X |
| `lift_ekf_contact_hw.py` | **the experiment** — lifts one foot, measures zEKF vs zFK | + `1`–`4` |

The experiment imports the controller (`import stand_ekf_level_hw as lv`) and
aliases its laws, so there is exactly one copy of `LevelingLoop`,
`SetpointLatch`, `height_inputs` and `LevelStandSequence` — the same layering
stage 2 uses on stage 1. Control gates live in the controller's self-test; the
experiment's self-test covers only the lift, the comparison block and the
`--fake-contacts` semantics.

The loop closes on the quantity **only the EKF has while standing**:
gravity-referenced roll/pitch.  Attitude error against a latched setpoint
drives per-foot differential z offsets (the proven
`stand_hier_hw.LevelingTrim` law, zero-meaned over the *planted* set), on top
of the stage-2 height loop.  Still no software torque.

## Stage 2b experiment — `lift_ekf_contact_hw.py`

Keys `1`–`4` (FL FR RL RR) ramp one foot **20 mm** off the floor while the
imported loops hold the trunk on the other three legs — the first time this
track feeds the EKF a **real contact change**:

- the lifted leg's contact flag goes False → its leg odometry drops out of
  the filter and its foothold covariance inflates (`swing_p`);
- touchdown re-anchors the foothold — the crawl/trot-relevant event.

Each lift prints the **zEKF vs zFK comparison** at three moments — baseline on
all four feet, held up on three, and after re-planting. Both are absolute
hip-axis heights above the floor, so they compare directly; `zFK` runs on the
**planted** feet only and cannot drift:

```
[lift] FL HELD UP at 20 mm, 3 planted
         zEKF =   211.1 mm    zFK(3 planted) =   209.9 mm    zEKF-zFK =   +1.2 mm
         vs all-4 stance:  zEKF   +0.8 mm    zFK   +0.1 mm    disagreement +0.7 mm
```

> **Expect zEKF and zFK to AGREE, and read that as success.** Measured on
> hardware 2026-08-12: they track each other through a single-leg lift. That is
> the filter working, not the test failing. `build_measurement` puts each
> planted foot's correction straight into body position
> (`H[:, ErrorIndex.R] = -C`), so with any anchored foothold the EKF's height
> *is* leg odometry, corrected every update — and three planted feet determine
> trunk height completely, which is the same information FK uses. A single-leg
> lift leaves height fully observable, so it cannot separate the two.
>
> This is the same point that opened the track: a statically-supported pose is
> knowable from encoders alone. Lifting one leg does not change that. Height
> only becomes EKF-only when **no** foot is planted (a flight phase) or during
> the sub-second contact transition; roll becomes EKF-only when the support
> polygon degenerates to a **diagonal pair** — which is trot, i.e. stage 3.

Both labels are **hip-axis** height. Note this differs from stage 1, where
`zFK` means the *IMU board* (38 mm lower). `height_inputs` reconciles them by
lifting the EKF's IMU-tracked `r_z` to the hip with
`+ IMU_BELOW_TRUNK_ORIGIN_M * C[2,2]`, so the comparison is hip-vs-hip and the
38 mm constant cancels.

**To see the filter actually use contacts, run the A/B:**

```bash
sudo chrt -f 50 $V stand_ekf_level_hw.py --fake-contacts --no-limit-check
```

This tells the EKF all four feet stay planted while one is physically lifted.
The motion and `zFK` are unchanged — only the filter's contact input is wrong.
The lifted leg's measurement then insists a foot that actually rose has not
moved, the correction lands in body z, and **zEKF walks away from zFK by
roughly lift/4 (≈5 mm at a 20 mm lift)**. The gap between the normal run and
this one is what the contact schedule is worth.

While a foot is up the status line shows the same two heights at `STATUS_HZ`,
and a per-lift scorecard (airborne time, max attitude error, max |EKF−AHRS|)
prints at touchdown and in the exit summary.

Design points, mirroring stage 2's safety shape:

| loop | feedback | independent watchdog | on veto |
|---|---|---|---|
| height (kept from stage 2) | EKF z | FK height from **planted feet only** | freeze (hold, never unwind) |
| leveling (new) | EKF roll/pitch | **AHRS attitude** (`--agree-veto` 3°) | freeze + auto-lower the lift |

- **Setpoint is latched, not zero** (stage-1 result: the resting attitude
  carries the IMU mount tilt): mean EKF attitude over 2 s once each stand
  settles, or `--setpoint-roll/--setpoint-pitch` to force it.
- Lift auto-lowers on any leveling veto and on `--lean-abort` (6°, the
  crawl's lean-watch convention); `--tilt-stop` (12°) soft-stops the run.
  `P` is refused while a leg is up.
- **Default stand height is 0.19 m, lower than stage 2's 0.20** — the legs'
  reach caps commanded foot z at ≈ −221 mm, and 0.20 leaves less extension
  authority than sag + leveling clamp need.  Startup prints the measured
  authority and warns if short.
- Requires the in-tree `min_eig` relative-tolerance health fix (applied to
  `dog5_state_estimator.outputs()` 2026-08-12, per CONTROL_ROADMAP §1b) —
  with the stock absolute tolerance the long-swing covariance inflation
  would false-trip `healthy`, freezing leveling mid-lift.

### When the preflight refuses on a soft limit

`[hardware] ENTER refused: FR_knee=… rad is outside the soft limit.`

The soft limits are **±100° abduction, ±149° pitch/knee** — near the full
mechanical range, so this is not a tuning limit that a normal stand runs into.
Two enforcement points read the **measured** pose:
`_zero_torque_preflight` (the ENTER refusal) and `SafetyGate.estop_reason`
(runtime) — bypassing only the first estops on the first sweep.

`--no-limit-check` turns off **both**. It relaxes what we accept *from the
encoders*; it does **not** relax what we command — the z tables are still
built inside the soft limits, so no reachable target can exceed them.

```bash
sudo chrt -f 50 $V stand_ekf_level_hw.py --no-limit-check
```

**Before trusting a run made with it**, note what the refusal usually means.
The motors are limp during preflight, so a genuinely folded leg is fixed by
hand. If the leg looks *normal* and still reads far out of range, that motor's
hardware zero (0x19) has moved — `setzero_one.py` in the repo root defaults to
`MOTOR_ID = 12`, which is **FR_knee**, and 0x19 only takes effect after a
power cycle. That matters beyond the limit check: `Q_RECORDED_CROUCH` was
recorded in the *old* zero frame, so the commanded crouch for that leg is then
wrong too, and no limit flag fixes it.

The script now prints the measured pose against the limits at preflight
(`[preflight] measured pose …`) with each offender's overshoot in degrees, so
compare a joint against its mirror on the other side at the same physical
pose — a large split is the calibration shift.

Motor ↔ joint map for reference: FR = **id10 abd, id11 pitch, id12 knee**.

### Run

```bash
$V self-test/test_stand_ekf_level.py       # 50 controller gates, no hardware
$V self-test/test_lift_ekf_contact.py      # 26 experiment gates, no hardware
```

Staged hardware session — the controller first, on its own:

1. `stand_ekf_level_hw.py --level-gain 0` — reproduces stage 2 at the lower
   stand height (open-loop A/B).
2. `stand_ekf_level_hw.py --level-clamp 0.006` — leveling live at half
   authority; watch `tilt=` collapse toward 0 and `lvl=` wind in a few mm.
3. `stand_ekf_level_hw.py --raw-log lvl.npz` — full leveling
   run. This is the stage-2b sign-off.

Then the experiment, once leveling is trusted:

4. `lift_ekf_contact_hw.py --raw-log lift.npz` — after
   `[level] setpoint latched…`, press `1` (FL). Read the three comparison
   blocks; expect agreement. Press `1` again to lower before parking.
5. `lift_ekf_contact_hw.py --fake-contacts` — the same lift with the EKF
   misinformed. Expect zEKF to walk ~5 mm off zFK. **The difference between
   runs 4 and 5 is the actual result of this stage.**

Add `--no-limit-check` to any of these if the preflight refuses on a measured
pose (see above).

## Stage 3 — `trot_fk_switch_hw.py`

Trot in place by alternating the diagonals FL+RR and FR+RL, **with no gait
clock**. Each pair lifts, comes back, and the next pair is released only once
forward kinematics confirms the last one returned. The cadence is an *output*
of the run.

### What FK can and cannot time

FK cannot see touchdown — in position mode the encoder tracks its reference
whether or not the foot met the floor. And the geometric substitute ("the swing
foot crossed the stance plane") is unavailable exactly here: a diagonal support
pair is two points, and `fk_attitude` needs three non-collinear ones.

What FK *can* judge is **return**. `TrotFKSwitch` samples each swinging leg's
measured foot z when the swing starts and calls the pair back once it is within
`TROT_SWITCH_TOL_M` (1 mm) of that baseline for `TROT_SWITCH_HOLD_S`.

The baseline is the whole trick. Commanded and measured foot z differ by ~13 mm
of load sag, so a measured-vs-*commanded* test can never come inside 1 mm. Taken
against the leg's own pre-swing baseline the sag cancels — and because the sag
only reappears as the leg re-loads, the test waits for **load transfer**, not
for the encoder to reach a number. `TROT_SWITCH_TIMEOUT_S` is the other half:
a leg that never confirms aborts the gait instead of hanging it with a foot up.

### The contact schedule is measured, not commanded

Every earlier runner derives contacts from what it asked for. That breaks here,
because **a commanded lift is not a lifted foot**: a swinging leg unloads, and a
leg that sags 13 mm under load extends by about that much when the weight comes
off. A command-derived flag would tell the filter that a loaded foot is airborne
— the exact failure the OFF schedule exists to prevent.

`SwingClearance` measures it. The stance feet *are* the floor, so rotating each
foot into world-vertical with the EKF's `C` and subtracting the stance mean
gives height above the floor plane with the trunk's own position cancelling
out. Only **attitude** is needed, which is the EKF's strongest, AHRS-cross-
checked output — and it is the one term FK alone could never supply, since the
trunk's rotation about a two-foot support line is invisible to the legs. With
no attitude or no reference captured yet, everything reads planted (the
schedule stages 1/2/2b validated).

### Lifting is not enough — the stance pair has to push

**Hardware 2026-08-14: commanded FL+RR up, the knees moved, and the feet stayed
on the floor.** Raising `--lift` does not fix it. A lift moves the feet relative
to the *trunk*, but clearance is relative to the *floor*, and lifting a diagonal
moves the trunk too:

- the swinging legs **unload** and extend by ~13 mm (stage 1's measured sag), so
  the feet reach back down toward the floor;
- the stance pair's load **doubles**, so it compresses another ~13 mm and the
  **trunk sinks**, carrying the swinging feet down with it.

Both terms subtract, and each is about the size of the lift:

```
clearance ≈ lift + push − 2 × 13 mm
```

With no push the lift alone must beat 26 mm, whose rocking bound atan(26/214 mm)
= 6.9° is already past `--lean-abort`. **There is no push-free lift that both
clears the floor and stays inside the abort** — the push is not an optimisation,
it is what makes stepping reachable. So the stance pair extends by `--push` for
exactly as long as the other pair is up, ramped off the same variable as the
lift so the two cannot fall out of step. The height loop cannot stand in: at
`HEIGHT_SLEW_M_S` it moves 2.5 mm in a 0.5 s swing, and being feedback it
arrives after the trunk has already dropped.

**Measured on hardware 2026-08-14** at `--push 0.015`, reading `clear=` off the
ramp of a `--lift 0.020` run as it swept each height:

| lift | `clear` FL / RR | rock | what happened |
|---|---|---|---|
| 11 mm | +12.5 / +12.0 mm | 0.09° | both feet off, body still |
| 12 mm | +13.4 / +14.4 mm | 0.26° | both feet off, body still |
| 14 mm | +14.9 / +15.0 mm | 0.42° | both feet off, body still |
| 16 mm | +7.4 / **+41.9** mm | 5.69° | tipping — not a lift |
| 20 mm | — | 4–5° | `--lean-abort` stops the gait |

**The sag arithmetic is pessimistic** — 11 mm of command bought ~12 mm of real
clearance, not the −7 mm it predicts. The 13 mm figure is stage 1's *trunk* sag
across four legs; the per-leg unload extension is smaller and the push covers the
rest. Trust `clear=`, not the formula.

The number that matters is the **cliff between 14 and 16 mm**. Below it both feet
clear together and the trunk barely moves. Above it FL's clearance collapses while
RR's more than doubles — RR at +42 mm on a 16 mm command is not lift, it is the
trunk rotating about the FR–RL line and dropping FL back onto the floor. That is
the CoM problem below, and until the body is trimmed it caps the usable lift:
**run at 12 mm.**

### One foot lifts and the other doesn't — that's the CoM, not the lift

**Hardware 2026-08-14, with the stance push in: FL+RR commanded up, RR left the
floor, FL did not, and the body leaned toward FL.**

While a diagonal swings, the robot stands on two points, and rotation of the body
about the line through them is an **unactuated degree of freedom**. Both stance
legs lie *on* that axis, so extending them — together or differentially — moves
the axis but cannot rotate the body about it. Gravity alone decides. With the CoM
off to one side the body falls that way until the near swing foot touches down
and becomes a third support; that foot then carries load and cannot lift, and
commanding it higher only tips the body further to meet it. **A statics problem —
no lift, push or gain fixes it.** (Same fact the stage-2b note reaches from the
other side: roll about a diagonal is invisible to FK because the legs don't
control it.)

The remedy is to put the CoM **on** the support diagonal. The two diagonals cross
at the centre of the foot polygon, so **one constant trim serves both swings** —
there is no per-step weight shift to schedule. `--com-shift-x/y` slides the body
over the footprint; the feet move the opposite way, folded into the z tables at
build time (`build_tables(xy_offset=…)`), so the CAN sweep still costs one
interpolation. This is the first thing in the track to unpin foot x/y.

`StanceLoadBalance` measures which trim to use, **from encoders alone**. A
driver's position loop holds against load with finite stiffness, so each leg's
foot sits above its commanded z in proportion to what it carries:

```
sag_i = z_measured_i − z_commanded_i        (both from FK, no IMU)
load share = sag_i / Σ sag                  CoM = Σ share_i · anchor_i
```

```
[com] load share from stance sag:  FL 31.2%  FR 21.4%  RL 27.9%  RR 19.5%
[com] CoM offset from the foot-polygon centre: x  +9.8 mm  y  +6.1 mm
[com] -> try --com-shift-x -0.0098 --com-shift-y -0.0061
```

**The command must be still, and that is not a detail.** The 2026-08-14 run
reported `RL 100%` and a 342 mm trim — a full leg anchor, physically impossible.
`sag` is only a load proxy against a *stationary* target; once the height loop is
winding 13 mm at 5 mm/s and leveling is winding ±12 mm differentially, most of
`z_measured − z_commanded` is velocity lag on a moving reference, and the estimate
reads the integrators instead of the weight. Only the first readings, before
`[LVL]` engaged, were real. The estimator now throws away any window in which a
commanded foot z moves more than `TROT_LOAD_STILL_M`, refuses outright if a leg
comes back negative (a leg cannot carry less than nothing), and prints once
rather than at `STATUS_HZ`.

**Believe the direction, bisect the magnitude.** Backlash, per-leg stiffness
spread and encoder-zero error all land in `sag` and none of them are load. The
honest test is whether both feet of a diagonal clear — which the status line now
reports **per leg**, since a `min()` over the pair is exactly what hid this.

### Why stage 3 stands at 0.17, not 0.19

The push spends **extension authority** — how far below the stand pose the legs
can still reach — and so do the sag the height loop winds out and the leveling
clamp. The legs bottom out at a commanded foot z of −221 mm, so the stand height
sets the budget:

| stand | authority | 13 sag + 12 leveling + 15 push = 40 mm |
|---|---|---|
| 0.19 (rest of the track) | 31 mm | **9 mm short** |
| 0.17 (`TROT_STAND_HEIGHT_M`) | 51 mm | fits |

A push that does not fit fails *silently* — the z table clips, the push comes up
short, and the feet quietly do not clear, which looks exactly like having no push
at all. Standing lower is the remedy; shrinking the push is not. The startup
banner prints the budget and says if it is short.

### Safety shape

`TROT_OVERLAP_S` (300 ms) keeps all four feet down between steps, so support
never hands straight from one diagonal to the other. That window is also where
the leveling loop runs and where the clearance reference is taken — no extra
freeze is wired in for the swing, because `LevelingLoop.update` already refuses
to integrate below 3 planted feet. `--overlap 0` removes the window and gives a
true trot; that is stage 3 proper, not where to start.

`T` starts and stops the gait (a stop always finishes the current step, feet
down); a lean, a leveling veto or an FK timeout aborts it at the slew limit; `P`
is refused mid-gait.

### Run

```bash
$V self-test/test_trot_fk_switch.py        # 54 gates, no hardware
```

All of these stand at 0.17 by default (see above).

1. `trot_fk_switch_hw.py --lift 0.012 --cycles 2` — the measured working
   point. Expect `clear=` around +12 mm on **both** feet, `2 planted`, and
   `rock` under half a degree.
2. `trot_fk_switch_hw.py --cycles 2` — the 5 mm default, for contrast: the
   feet stay down and the state machine still runs.
3. If one foot of a pair clears and the other does not, read the `[com]` lines
   and re-run with `--com-shift-x/y`. Bisect until both legs of *both*
   diagonals show positive `clear=`; that trim is a per-robot constant worth
   recording.
4. `trot_fk_switch_hw.py --lift 0.020 --push 0.0 --cycles 2` — the A/B for the
   push: reproduces the 2026-08-14 failure deliberately. Knees move, `clear=`
   stays negative, `air` stays 0. **The difference between 2 and 4 is what the
   stance push is worth.**
5. `trot_fk_switch_hw.py --lift 0.020 --raw-log trot.npz` — the stage-3 run.
6. `trot_fk_switch_hw.py --lift 0.020 --fake-contacts` — the EKF told all four
   feet stay planted. **The difference between 5 and 6 is what the measured
   contact schedule is worth.**

## Notes

- `MotorBus.slot(rate_hz)` is `1/(rate_hz · n_motors)`, so `rate_hz` is **per
  motor**: at `CONTROL_HZ = 250` every motor is commanded at 250 Hz *and* a full
  12-motor sweep completes at 250 Hz (3000 frames/s on the bus). The
  `joint_index == 0` block therefore runs at 250 Hz, not 250/12. Older comments
  elsewhere in the repo say ~20.8 Hz; they are wrong.
- **`sudo` and `$HOME`.** `IMU_sensor/imu_dog.py` finds the `fdilink_imu`
  package via `Path.home()/Documents/IMU_sensor`, which breaks under `sudo`
  (`$HOME` becomes `/root`) — the failure recorded in `7.28review.md`. Since RT
  priority (`chrt -f 50`) wants root, this script resolves that package
  **repo-relative** first and only falls back to `$HOME`, so `sudo chrt` works.
  Other runners in this tree still have the original problem; run those without
  `sudo`, or with `sudo HOME=$HOME chrt …`.
- The EKF worker is the shared `state_estimator/ekf_runtime.py`
  (`EkfShared` + `ekf_worker`), not a private copy.
- Prior art: `vmc/ekf_stand_hw.py` (this script's ancestor, read-only EKF),
  `vmc/stand_hier_hw.py` (stage 2's `LevelingTrim` already exists there).
