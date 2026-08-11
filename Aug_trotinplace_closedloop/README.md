# Aug — closed-loop trot in place

Track goal: **trot in place with the EKF closed in the loop.** Built in stages,
each one hardware-signed-off before the next starts.

| Stage | Script | Motors | EKF role | Status |
|---|---|---|---|---|
| 1. Verify | `stand_ekf_verify_hw.py` | position (0xA4) | **read-only** | **hardware PASS 2026-08-11** |
| 2. Height | `stand_ekf_height_hw.py` | position (0xA4) | **feedback → common foot-z** | self-test PASS, hardware pending |
| 2b. Level | *(next)* | position (0xA4) | feedback → differential foot-z | not started |
| 3. Trot | *(later)* | position (0xA4) | feedback → attitude + timing | not started |

No software torque is commanded anywhere in this track. The drivers' native
position loops do the joint-level work; the EKF only ever shapes *targets*.

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

$V stand_ekf_verify_hw.py --self-test              # 56 offline gates, no hardware

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
$V stand_ekf_height_hw.py --self-test                 # 33 gates, no hardware

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
