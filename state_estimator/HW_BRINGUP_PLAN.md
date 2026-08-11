# EKF Hardware Bring-Up — IMU → Encoder → EKF (read-only)

**Goal.** Validate the proprioceptive state estimator on the real DOG5, one
sensor at a time, **without ever closing a control loop.** The MuJoCo VMC build
(`vmc/`, all gates passing) stays the reference model; here we only prove the
*inputs* and the *filter* are correct on hardware. Closing the loop (VMC on
hardware) is explicitly out of scope until this plan's final gate passes.

**Order and why.** Garbage in → garbage out. We validate the IMU feed, then the
encoder feed, then the two fused in the EKF — because an EKF fault is
un-diagnosable until each input is independently trusted. This is plan gate C8
(hardware log replay) plus the two input bridges it depends on.

**Reference conventions** (from `state_estimator/plan.md`, `dog5_sim.py`):
specific force body-FLU `f ≈ [0,0,+9.81]` at rest & level; gyro rad/s; joint
angles `alpha` (4,3) order `[FL,FR,RL,RR]×[hip_abd,hip_pitch,knee]`, radians;
contacts length-4 bool, produced outside the filter.

**Safety posture for every phase here.** Motors in **zero-torque keepalive**
only (`mb.keepalive` / `mb.arm` then command iq=0) — the robot never actuates.
Dog on a stand / cradle or lying flat; feet manipulated by hand. No SafetyGate
control path is exercised. The one moving test (Phase D3 excitation) is done by
**hand-rocking** the body, motors still passive.

---

## Phase 0 — Prerequisites & audit (0.5 h, no robot motion)

- [ ] Confirm hardware map: CAN id ↔ (leg,joint) and signs come from
      `dog5_description/dog5_hardware_map.py` (confirmed: FL=7,8,9 / FR=10,11,12
      / RL=4,5,6 / RR=1,2,3). **Do NOT trust `hil_map.json`** (placeholder,
      mismatched). Verify `hw_jointmap.HwMap` is sourced from the confirmed map
      or override it.
- [ ] Confirm the DETA10 is configured to stream the **raw IMU packet (0x40)**,
      not only AHRS. `imu_dog.py` currently registers only `on_ahrs`; the raw
      accelerometer/gyro live on `ImuDog._imu` (a `DETA10`) via `on_imu` /
      `latest_imu()`. If `latest_imu()` returns `None`, the raw stream is off —
      re-enable raw mode in the DETA10 config before proceeding.
- [ ] Run the **live** harness with the project venv (`.venv/bin/python`) — it
      has `python-can` + `pyserial`; the system `python3` has neither and errors
      at `import motorbus`. (Offline replay/analysis, numpy-only, runs under
      system `python3` fine — no mujoco needed for hardware.)

---

## Phase A — IMU feed validation (standalone)

**A1. Raw-IMU bridge** — new `state_estimator/imu_ekf_feed.py`.
Wrap `ImuDog._imu`: register `on_imu` (or poll `latest_imu()`), return
`(f_flu, w_flu, t_monotonic, age)`. Apply **NED→FLU: negate Y and Z** of both
`.accel` (specific force) and `.gyro`. Timestamp with a single `time.monotonic()`
at read (shared clock with the encoder reader — see Phase D).
- Unit test (offline, synthetic `IMUData`): NED `[0,0,-9.81]` → FLU `[0,0,+9.81]`;
  a known NED rate maps with the right sign flips.

**A2. Convention & rate check (hardware gate C1).** Dog level & still, motors
passive. Log 30 s of `f_flu, w_flu`.
- [ ] `f_flu ≈ [0, 0, +9.81]` (±0.3). If Z is negative → the NED→FLU flip is
      wrong or the site is upside down.
- [ ] Tip the body **nose-down** by hand → `f_flu.x` goes **negative**
      (gravity tips into −x); confirm sign matches `dog5_sim` eq. 3 expectation.
- [ ] Rotate slowly about each body axis by hand → `w_flu` sign matches (roll
      about +x > 0 when right side drops, etc.); cross-check against
      `ImuDog.sample()` rates.
- [ ] Record actual raw-IMU `rate_hz` (need ≥ 200 Hz for the EKF predict; if the
      raw packet is slower than AHRS, note it — it caps predict rate).

**A3. Noise & bias characterisation (feeds EKF params).**
- [ ] Reuse/extend `IMU_sensor/imu_noise_log.py` (currently AHRS; add a raw-accel
      variant) to log **motors-OFF** then **motors-ON static hold** (IMU is a
      separate USB device — no CAN conflict; run a passive keepalive in another
      terminal for the motors-on case).
- [ ] From the logs compute per-axis `σ_f, σ_ω` (white-noise density) and the
      **motors-on/off ratio** → set `EstimatorParams.sigma_f, sigma_w`. Compute
      static gyro-bias mean and accel magnitude residual; check run-to-run
      repeatability. (No accel/gyro bias calibration exists yet — the EKF
      estimates gyro bias at init and accel bias under excitation, D3.)

**Pass A:** `f_flu`/`w_flu` correct sign & magnitude at rest and under hand
motion; rate ≥ 200 Hz (or documented); `σ_f, σ_ω` measured and set.

---

## Phase B — Encoder feed validation (standalone)

**B1. Joint read path.** Reuse `stand_dog5_hw._joint_state(mb, unwrap)` →
`(q_rad(12), qd(12))` with `CalibratedEncoderUnwrap` (multi-turn unwrap from the
mod-36°-at-output single-turn encoder; the 10:1 gear is already in the gains).
Reshape to `alpha(4,3)` in controller order.
- [ ] Direction check: hand-move each joint a known direction → confirm the sign
      of `q` matches the controller frame (`dog5_hardware_map` direction). A
      flipped sign here silently corrupts FK.

**B2. Zero calibration.** Place the dog in the flat/calibration pose,
`unwrap.reset_zero(raw)`. Confirm `_joint_state` returns `q ≈ 0` at that pose.
This is the joint-zero the EKF's kinematics depend on — the last calibration date
should be recent (re-check after any mechanical change; memory flags suspected
zero offsets).

**B3. FK four-foot closure (hardware gate C2).** Stand the dog on a **flat
floor** (feet on ground, body roughly level), read all 12 encoders → FK all four
feet with `dog5_kinematics.foot_position`. The four contact points must lie in
one horizontal plane within a **few mm**. Residual tilt here becomes a direct
roll/pitch bias in the EKF, so fix it now (via zeros / measured link lengths)
before trusting the filter. (Mirrors the sim M2 closure test.)

**B4. Telemetry rate & unwrap robustness.** Confirm whole-body encoder refresh
(250 Hz: `MotorBus.slot()` is `1/(rate_hz·n_motors)`, so `rate_hz` is *per
motor* — every motor is visited at 250 Hz and a full 12-motor sweep also
completes at 250 Hz, 3000 frames/s on the bus) and that the multi-turn unwrap
has no jumps across the working joint range. This 250 Hz is the EKF **update**
rate (predict still runs at IMU rate).

**Pass B:** signs correct; zeros set; four-foot closure < few mm; encoder→alpha
stream clean at 250 Hz.

---

## Phase C — Contact input (trivial for validation)

For all bring-up tests the contact state is **known, not sensed**: static tests
use `contacts = [1,1,1,1]` (all feet down). The hardware torque-derived contact
is verified-unreliable, so we do **not** use it here. (A schedule-driven contact
comes later, only when a gait actually lifts feet.)

---

## Phase D — EKF on hardware, read-only (the main event = gate C8)

**D1. Synchronised logger** — new `state_estimator/hw_logger.py`.
One process opens the IMU bridge (A1) and the `MotorBus` reader (B1), keeps the
motors in **passive keepalive**, and logs to CSV/NPZ at each stream's native
rate with a **single monotonic clock** on every row:
`t, ax,ay,az, wx,wy,wz` (IMU rows) and `t, alpha[12], contacts[4]` (encoder
rows). Timestamp skew between the two streams is the #1 EKF failure mode — one
clock, stamped at read, is non-negotiable. Also log `ImuDog.sample()` roll/pitch
as an **independent attitude reference** for cross-checking the EKF.

**D2. Offline replay harness** — new `state_estimator/hw_replay.py`.
Read a log, drive `DOG5StateEstimator`: `initialise` on the first static window,
then `predict` per IMU row, `update` per encoder row (contacts from the log).
Emit per-tick `outputs()` + innovations + `healthy`. This is pure software —
**buildable and unit-testable now against a `dog5_sim`-generated log** before any
hardware exists.

**D3. Test sequence (record, then replay):**

| Step | Robot state | Expect on replay |
|---|---|---|
| D3a Static | standing still, all contact, 30 s | bias converges; EKF roll/pitch match the IMU-AHRS reference (±0.5°); v ≈ 0; z ≈ 0; innovations zero-mean & small; healthy=true |
| D3b Excitation | **hand-rock** body ±5–10° roll/pitch, feet planted, 20 s | accel bias becomes observable & converges; attitude tracks AHRS through the motion (mirrors sim `scenario_excite`) |
| D3c Manual leg lift | lift/replace one foot by hand, contacts edited in the log at the lift | no state jump at the transition; that foothold re-anchors cleanly (M6) |

**D4. Read the innovations (plan §6.5 diagnostic table):**
- zero-mean, ≥95 % inside 3σ → healthy.
- one leg persistently offset → that leg's kinematic zero/link length (Phase B).
- all legs offset together → base-frame / IMU extrinsic (Phase A).
- spikes at contact edits → contact timing.
- offset ∝ body speed → IMU/encoder timestamp skew (fix D1 clocking).

**D5. Online read-only EKF (real-time budget check).** Run the EKF **live** in a
loop process (in the `joint_index==0` slot alongside a passive keepalive), but
only **log** its outputs — never to control. Confirm the 27-state step fits the
slot without breaching the 10 ms motor watchdog; if it crowds the slot, spread
predict/update/output across slots like the IK (`IK_SLOT_STRIDE`). This de-risks
the eventual closed loop.

**Pass D / gate C8:** innovations zero-mean, ≥95 % in 3σ, no persistent per-leg
offset; bias converges; EKF roll/pitch match the AHRS reference; σ_yaw
non-decreasing over minutes; live step within the real-time budget.

---

## Phase E — Sign-off

When Pass A + B + D hold, the IMU, encoders, and EKF are hardware-validated and
the estimator is trustworthy as a **read-only** state source. Only then is it a
candidate to feed a controller — that (porting the `vmc/` loop to hardware with
SafetyGate/arm/recover) is a separate, later effort, gated on this one.

---

## What can be built NOW (no hardware needed)

These are pure software and unit-testable against `dog5_sim` today, so hardware
time is spent only on measurement, not coding:
1. `imu_ekf_feed.py` — raw-IMU bridge + NED→FLU, with a synthetic-`IMUData` test.
2. `hw_logger.py` — schema + single-clock logging (mock sources in a test).
3. `hw_replay.py` — offline replay; validate it reproduces `test_estimator.py`
   results when fed a `dog5_sim` log (i.e. replay == in-memory run).
4. raw-accel variant of `imu_noise_log.py`.

Building 1–3 now means Phase A/B/D are *measurement* sessions, not debugging
sessions.
