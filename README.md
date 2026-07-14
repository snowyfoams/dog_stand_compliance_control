# DOG5 Stand-Up via Cartesian Compliance Control

A 12-DOF quadruped (DOG5, exported from Fusion 360 to MJCF) stands up from its
flat calibration pose using joint-space staging plus Cartesian compliance
control, simulated in MuJoCo.

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
- `dog5_description/view_dog5.py` — open the model in the passive viewer
- `dog5_description/make_gif.py` — regenerate the demo GIF (2× slow motion)

## Run

```bash
python -m pip install mujoco pillow numpy
python dog5_description/stand_dog5.py
```

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
