# DOG5 Stand-Up via Cartesian Compliance Control

A 12-DOF quadruped (DOG5, exported from Fusion 360 to MJCF) stands up from its
flat calibration pose using joint-space staging plus Cartesian compliance
control. The controller can run in MuJoCo or directly on the 12-motor CAN
hardware.

![DOG5 stand-up demo](dog5_description/dog5_stand.gif)

## Why staging is required

At the calibration pose (all 12 joints = 0, legs flat fore-aft) the leg
Jacobian is rank-1: hip abduction, hip pitch and knee all move the foot purely
sideways, so a Cartesian force has **no vertical authority**. The controller
therefore runs four phases:

| Phase | Time | Controller | What happens |
|-------|------|-----------|--------------|
| 1 ROLL  | 0–1.2 s | joint PD | each `hip_abd` rolls to ±90° so legs point outboard and the knee plane becomes vertical (prevents left/right legs crossing the midline) |
| 2 FOLD  | 1.2–2.8 s | joint PD | pitch + knee fold in the vertical leg plane to a crouch, feet under hips |
| 3 STAND | 2.8–5.3 s | Cartesian compliance | foot targets in the trunk frame ramp from crouch (0.10 m) to standing height (0.20 m): `τ = Jᵀ(Kp·e + Kd·ė + F_ff)` with `F_ff = mg/4` per foot |
| 4 HOLD  | 5.3 s– | Cartesian compliance | fixed targets — the stand is soft; drag the trunk in the viewer and it yields and recovers |

Result: trunk settles at 0.221 m (target 0.20 m hip-to-foot + 0.02 m foot
radius) with zero steady-state droop, level attitude, and survives an
8 N × 0.3 s lateral shove.

## Files

- `dog5_description/dog5.xml` — MJCF model (mesh visuals, sphere feet + thigh-pad collision, motor armature)
- `dog5_description/stand_dog5.py` — stand-up controller; run without args for the interactive viewer, `--headless` for a metrics-only test
- `dog5_description/stand_dog5_hw.py` — hardware-only runner; real encoders in, real CAN torque out (no HIL or simulated plant)
- `dog5_description/stand_dog5_recorded_hw.py` — current pose to the measured crouch, then operator-gated Cartesian compliance stand
- `dog5_description/stand_dog5_fl_rr_extend_hw.py` — stand in place, then smoothly extend and hold the FL/RR diagonal foot endpoints
- `dog5_description/dog5_pose_monitor.py` — zero-torque live position monitor; captures a manually arranged pose and can hold it with conservative joint PD
- `dog5_description/dog5_hardware_map.py` — confirmed DOG5 CAN-to-joint map shared by the verifier and hardware runner
- `dog5_description/dog5_zero_calibrate.py` — one-time all-joint motor-zero writer (`0x19`, zero torque, confirmed)
- `dog5_description/dog5_zero_check.py` — read-only verification of the current motor-zero calibration
- `dog5_description/view_dog5.py` — open the model in the passive viewer
- `dog5_description/make_gif.py` — regenerate the demo GIF (2× slow motion)

## Run

```bash
python -m pip install mujoco pillow numpy
python dog5_description/stand_dog5.py
```

## Run on hardware

The hardware runner uses the same waypoints and targets as `stand_dog5.py`,
with slower hardware-only motion times (REST 2 s, ROLL 4 s, FOLD 5 s, STAND
6 s) and independently checked NumPy FK/Jacobians. Its confirmed physical
configuration maps `FL=7,8,9`, `FR=10,11,12`, `RL=4,5,6`, and `RR=1,2,3`.
Confirmed directions are `-1` for CAN 1, 3, 4, 6, 9, and 12 and `+1` for the
other motors. Joint state is calculated directly from the
calibrated motor output with no software zero and no gearbox division.

Start with the robot supported and a low torque cap:

```bash
../.venv/bin/python dog5_description/stand_dog5_hw.py --self-test
../.venv/bin/python dog5_description/stand_dog5_hw.py --tau-max 1.0
```

The program verifies the existing driver zero; it never captures another
offset. The first Enter starts REST, a smooth move from the measured pose to
all calibrated joint zeros. REST then holds zero and requires another Enter
before ROLL. ROLL and FOLD likewise hold their completed targets and require
Enter before the next stage; Cartesian STAND starts only from the settled
crouch. Pose error, velocity, motor faults, temperatures, CAN misses, joint
limits, a startup torque ramp, a torque slew limit, and a 3 N*m maximum
staged-test cap gate the sequence. See
[`DOG5_STAND_RUNBOOK.md`](../DOG5_STAND_RUNBOOK.md) for the complete procedure.

## Capture and hold a manually arranged pose

If the designed standing target is not suitable, use the pose monitor with the
robot mechanically supported. It commands zero torque while you arrange the
legs, displays each leg's `[abd, pitch, knee]` angles in degrees and its
encoder-derived foot `(x, y, z)` and hip-to-foot height, and saves only after
the encoders have been still for 0.75 s:

```bash
../.venv/bin/python dog5_description/dog5_pose_monitor.py --self-test
../.venv/bin/python dog5_description/dog5_pose_monitor.py --tau-max 1.0
```

Press `C` to capture the stationary pose to
`dog5_description/dog5_stand_pose.json`. Inspect the displayed values, then
press `H` to ramp into a low-torque joint-space hold of that exact pose. Press
`R` to return to zero torque or `X` to stop. A saved pose can be reused with:

```bash
../.venv/bin/python dog5_description/dog5_pose_monitor.py \
  --load dog5_description/dog5_stand_pose.json --tau-max 1.0
```

The hold refuses to engage unless all joints are stationary, within their
software limits, and within 8 degrees of the target. The saved JSON includes
the CAN IDs and directions, so it is rejected if the hardware map changes.
This encoder-only hold does not measure trunk attitude or foot contact; keep a
support/tether in place and do not treat it as a free-standing balance
controller.

## Stand from the measured crouch

`stand_dog5_recorded_hw.py` uses the confirmed model-positive joint angles from
the hardware observation:

```text
FL = (+90.74, +47.59, -131.02) deg
FR = (-90.39, -51.92, +138.39) deg
RL = (+90.23, -44.45, +128.40) deg
RR = (-93.95, +47.06, -125.12) deg
```

Its sequence is
`CURRENT -> CROUCH -> ENTER -> STAND -> WAIT_STAND -> HOLD`. Partial runs end
in `HOLD_PARTIAL` instead. The initial
Cartesian target is the exact FK position of the recorded crouch, so switching
from a crouch that has passed the pose gate to Cartesian compliance introduces
no nominal position or feedforward step. The target then moves over eight
seconds to the under-body stance and a 0.20 m hip-to-foot height. `WAIT_STAND`
continues commanding the final target until Cartesian error is at most 0.025 m
and encoder-derived joint speed is at most 0.30 rad/s for 0.5 seconds; the timer
alone no longer declares `HOLD`.

Run the offline check, then test only 25% of the complete path while the robot
is supported:

```bash
../.venv/bin/python dog5_description/stand_dog5_recorded_hw.py --self-test
../.venv/bin/python dog5_description/stand_dog5_recorded_hw.py \
  --tau-max 1.0 --travel-scale 0.25 \
  --crouch-max-speed-dps 100 \
  --crouch-torque-trip 2.0 --crouch-speed-trip 1.0
```

The first Enter accepts the measured current pose and sends the recorded
crouch through the motor's native `0xA4` absolute-position command. The command
holds that target until every joint is within 0.08 rad and encoder speed is
below 0.25 rad/s for 0.5 seconds (with a 30-second timeout). Inspect the held
crouch, then press Enter again to start Cartesian motion. Press `X` at any time
to stop. After reviewing the telemetry, increase `--travel-scale` through
`0.50`, `0.75`, and `1.0`; increase the Cartesian `--tau-max` only if the robot
is not tracking and remains securely supported. The offline check verifies the
entire Cartesian path is reachable on a continuous branch without crossing the
software joint limits.

The zero-torque read and CURRENT-to-CROUCH stages may begin outside the normal
soft position limits because the fixed native-position target is the validated,
in-limit recorded crouch. `--crouch-max-speed-dps` limits motor-side speed
(with the configured 10:1 reduction, `100` motor-deg/s is approximately
`10` output-deg/s). `--crouch-speed-trip` stops on excessive encoder-derived
joint speed and `--crouch-torque-trip` stops on excessive measured joint
torque. The `0xA4` protocol has no commanded torque-cap field, so this crouch
torque protection is a feedback-based emergency trip, not an active torque
clamp. The recorded-pose gate must pass before Cartesian STAND, where the normal
position-limit e-stop and the true `--tau-max` torque command cap are enabled.

`--travel-scale 0.25` is intentionally only a partial motion: it ends around
0.063–0.071 m encoder-derived leg height and will not make the dog fully stand.
It reports `HOLD_PARTIAL`, not `HOLD`. Use the partial run to inspect signs and
tracking, then progress toward `--travel-scale 1.0` for the complete 0.20 m
target.

## Extend FL and RR after standing

`stand_dog5_fl_rr_extend_hw.py` uses the stand-in-place controller, waits for
the full standing target to settle in `HOLD`, and requires another Enter before
moving the FL and RR targets farther downward along trunk Z. FR and RL keep
their normal endpoints. The default diagonal extension is 20 mm over 3 s and
then remains in `EXTENDED_HOLD`:

```bash
../.venv/bin/python dog5_description/stand_dog5_fl_rr_extend_hw.py --self-test
../.venv/bin/python dog5_description/stand_dog5_fl_rr_extend_hw.py \
  --tau-max 3.0 --extension-mm 20
```

Keep the robot mechanically supported or tethered. This encoder-only motion
cannot measure trunk attitude or verify foot contact, and the unequal diagonal
targets intentionally change the load distribution.

## Modelling notes (hard-won)

- **Collision**: leg meshes are visual-only (`contype=0`); MuJoCo collides
  convex hulls, which caused phantom self-collisions during folding. Contact
  runs on foot spheres + thigh pads + trunk hull, bitmasked
  (`contype=2 conaffinity=1`) to collide with the floor only.
- **Armature**: knee joints chattered at ±25 rad/s because the 38 g shin gives
  a sampled-damping stability bound of only kd ≈ 2I/Δt ≈ 0.3 N·m·s/rad at
  500 Hz. Adding the real reflected rotor inertia
  (850 g·cm² × 10² gear ratio = 0.0085 kg·m²) fixes it; model damping is
  integrated implicitly by `implicitfast`.
