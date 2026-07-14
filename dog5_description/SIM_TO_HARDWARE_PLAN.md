# DOG5 Cartesian Stand: Simulation-to-Hardware Plan

## Goal

Move the standing controller in `stand_dog5.py` to the physical DOG5 using the
12 motor encoders as the primary leg-state measurement. The two main software
gaps are:

1. Replace MuJoCo kinematics with independently tested NumPy forward
   kinematics (FK) and Jacobians.
2. Define and estimate `h` correctly from encoder angles, while recognizing
   that encoders cannot measure absolute body height or gravity direction.

The first hardware goal is a slow, supported stand. Trotting is out of scope
until standing, orientation estimation, contact handling, and safety are
reliable.

## Current Controller

The simulated stand has four stages:

1. **ROLL, 0.0-1.2 s:** joint PD moves hip abduction to +90 degrees for FL/RL
   and -90 degrees for FR/RR. Hip pitch and knee remain at zero.
2. **FOLD, 1.2-2.8 s:** joint PD moves all legs to the configured crouch angles.
3. **STAND, 2.8-5.3 s:** Cartesian compliance ramps the desired hip/trunk-to-foot
   distance from `H_CROUCH = 0.10 m` to `H_STAND = 0.20 m`.
4. **HOLD, after 5.3 s:** Cartesian compliance holds the final foot targets.

Stages 3 and 4 use

```text
F = Kp (p_des - p_foot) + Kd (v_des - v_foot) + F_feedforward
tau = J^T F - joint_damping
```

and clamp each commanded torque to `TAU_MAX`.

## Edits Made During This Work

### Joint visualization

`view_dog5.py` now:

- enables body XYZ frames, which coincide with the joint origins in this MJCF;
- enables MuJoCo joint-axis markers and joint-name labels;
- prints every joint's world-frame anchor and axis;
- uses the passive viewer so visualization options can be set before display.

### NumPy geometric Jacobian check

`stand_dog5.py` now contains `leg_jacobian_numpy()`. For each hinge joint it
forms one translational Jacobian column:

```text
J_i = axis_i x (foot_position - joint_anchor_i)
```

The `--compare-jacobians` mode advances the simulation to a bent-leg pose and
compares this matrix with `mujoco.mj_jacSite`. The largest observed difference
was `5.551e-17`, which is floating-point roundoff.

Important limitation: this check still uses `data.xaxis`, `data.xanchor`, and
`data.site_xpos`. MuJoCo therefore performs the FK, while NumPy only performs
the final cross products. This is a useful Jacobian-formula check, but it is
not yet an independent hardware kinematics implementation.

## Meaning of `h`

The controller defines a foot target in the trunk frame:

```python
p_des = np.array([foot_x, foot_y, -h])
```

Therefore:

```text
h_desired  = -p_des[z]
h_measured = -p_foot_trunk[z]
```

`h` is a relative trunk/hip-to-foot vertical distance along the trunk's local
Z axis. It is not automatically the absolute trunk height above the floor.

Encoder angles are sufficient to calculate `p_foot_trunk` using FK. They are
not sufficient to determine whether the foot touches the floor, whether it is
slipping, or how the trunk is oriented relative to gravity. If the complete
robot is lifted without changing its joint angles, encoder-derived `h` does
not change.

## Required NumPy Kinematics

Implement a hardware-independent module, for example `dog5_kinematics.py`, that
does not read MuJoCo runtime transforms. Its inputs should be calibrated joint
angles and fixed geometry copied from one controlled source of truth.

For each leg, FK must calculate:

```text
encoder counts
  -> calibrated q = [q_abd, q_hip, q_knee]
  -> joint rotations and translated joint origins
  -> p_foot_trunk
  -> h_encoder = -p_foot_trunk[z]
```

The same intermediate FK results produce the Jacobian:

```text
J = [a1 x (pfoot-p1), a2 x (pfoot-p2), a3 x (pfoot-p3)]
```

Here `p1`, `p2`, and `p3` are joint origins in the trunk frame, and `a1`, `a2`,
and `a3` are their rotated hinge axes in the trunk frame.

Do not derive control coordinates from mesh quaternions. Mesh transforms only
place visual geometry. The kinematic chain comes from body positions, body
orientations, joint positions, and joint axes in the MJCF.

## Encoder Calibration Contract

For every motor, record:

```text
q_model = direction * (encoder_count - zero_count) * radians_per_count
```

The calibration table must include:

| Field | Meaning |
|---|---|
| `zero_count` | Encoder reading at the defined MJCF zero pose |
| `direction` | `+1` or `-1`, mapping hardware motion to model-positive rotation |
| `radians_per_count` | Encoder scale including gearbox ratio |
| `q_min`, `q_max` | Safe software joint limits |
| `tau_limit` | Safe motor-specific torque/current limit |

Determine `direction` by moving one joint a small amount with the robot
supported and verifying that FK predicts the observed link motion. Do not infer
motor direction only from FL/FR or RL/RR symmetry.

## Validation Plan

### 1. Offline FK validation

- Generate many safe random joint configurations.
- Compare NumPy `p_foot_trunk` against MuJoCo foot-site positions expressed in
  the trunk frame.
- Test all four legs, zero pose, crouch pose, joint limits, and asymmetric poses.
- Acceptance target: maximum position error below `1e-6 m` in software tests.

### 2. Offline Jacobian validation

- Compare the independent NumPy Jacobian with `mj_jacSite` in the trunk frame.
- Also validate using finite differences:

```text
J[:, i] ~= (FK(q + epsilon e_i) - FK(q - epsilon e_i)) / (2 epsilon)
```

- Check `v_foot = J @ dq` against finite-difference foot velocity.
- Acceptance target: analytic and finite-difference errors consistent with the
  selected `epsilon`, with no leg-specific sign failures.

### 3. Hardware kinematics validation without Cartesian torque

- Suspend or securely support the robot.
- Read encoders and display `q`, `p_foot_trunk`, and `h_encoder` at low rate.
- Manually move one joint at a time and confirm the predicted foot direction.
- Measure several joint-to-foot distances physically and compare with FK.
- Do not enable Cartesian torque during this step.

### 4. Joint-space stages on hardware

- Run ROLL and FOLD separately at reduced speed, torque/current, and travel.
- Begin with one leg while the robot is supported.
- Add joint-limit, encoder-validity, communication-timeout, and emergency-stop
  checks before testing all legs.
- Replace simulation-only `data.qfrc_bias` with an explicitly designed hardware
  feedforward term or initially omit it and use conservative gains.

### 5. Cartesian control on a supported robot

- Start with low Cartesian stiffness and damping.
- Command millimetre-scale changes in one Cartesian direction.
- Confirm that `J.T @ F` produces the expected motor directions.
- Monitor encoder limits, current/torque, loop timing, and Jacobian conditioning.
- Reject or damp commands near singular configurations.

### 6. Controlled standing test

- Use a gantry, tether, or frame that prevents a fall.
- Begin near the crouch pose rather than relying on the full floor stand-up
  sequence.
- Ramp `h_desired` slowly from the measured initial `h_encoder`.
- Limit `h_desired`, Cartesian error, force, torque/current, and rate of change.
- Stop on invalid encoders, stale communication, excessive tilt/current, joint
  limit approach, or control-loop overrun.

## Sensor Decision

Encoder-only control can regulate each foot relative to the trunk and can be
used for cautious, supported leg-extension experiments. For free standing, add
an IMU as the minimum next sensor. It provides gravity direction and trunk
angular velocity, allowing the controller to distinguish local trunk Z from
true vertical and to react to body roll and pitch.

Foot-force sensors are helpful but not mandatory for the first supported stand.
Without them, contact must be assumed or inferred from motor current and leg
behavior, which is less reliable. Dynamic trotting should wait until the system
has reliable orientation estimation and contact handling.

## Recommended Implementation Order

1. Create independent NumPy FK for all four legs.
2. Derive the NumPy Jacobian from the FK transforms.
3. Add automated MuJoCo and finite-difference tests.
4. Complete the 12-joint encoder calibration table.
5. Verify FK and motor signs on supported hardware.
6. Port and test joint-space ROLL/FOLD with strict safety limits.
7. Test low-gain Cartesian motion on supported hardware.
8. Add an IMU and body-orientation safety logic.
9. Attempt a tethered stand and tune gradually from logs.

## Immediate Next Deliverable

The next code change should be `dog5_kinematics.py` plus tests. It should expose
an interface independent of MuJoCo, such as:

```python
p_foot = foot_position(leg, q)
J = foot_jacobian(leg, q)
h = -p_foot[2]
```

Only after these functions match MuJoCo and finite differences for all legs
should the controller replace `mj_jacSite` and `data.site_xpos` with the NumPy
hardware path.
