# crawl_hw2.0 — position-mode crawl, second hardware generation

Hardware ports of the sim-proven position-command track from the
`dog5-crawl-sim` branch (`stand3_dog5.py`, `crawl_dog5_sim.py`,
`SIM_APPROACH_HW.md`).  The sim ran the hardware's code path under emulated
0xA4 servos, 20.8 Hz encoder-only state, and quantized encoders; everything
here ports its stage machines back onto the real CAN bus with
`mb.position()` replacing the servo emulation and operator ENTER replacing
auto-advance — the §5/§8 port recipe.

## Contents

| File | Role |
|---|---|
| `stand3_hold_hw.py` | 3-leg stand hold: in-place stand → diagonal shift → pre-lift + clear gate → lift → HOLD3 → lower → recenter.  **All four legs passed on hardware 2026-07-23.** |
| `walk1_hw.py` | One slow gait cycle (4 steps, RR→FL→RL→FR): each step = the proven stand3 ladder + SWING (+30 mm) + TOUCHDOWN (measured-anchor commit) + forward RECENTER (+7.5 mm/step).  Offline-validated (worst margin 23.6 mm); hardware untested. |

## Sim findings baked in (SIM_APPROACH_HW.md §6-§8)

- shift **0.05 m** for the long static hold (~51 mm planned FK margin)
- `--torque-trip 6.0` (loaded front hip pitch measured 4.27 N·m static,
  4.50 N·m peak — the old 2.0 default WILL fire)
- pre-lift starts at **20 mm**; the clear gate demands an **absolute**
  stance-plane rise ≥ 5 mm AND swing pitch/knee |τ| ≤ **0.7 N·m**
  (position-mode airborne leg still reads ~0.5 N·m of leg gravity);
  a gate timeout raises the pre-lift **+8 mm** and retries, to **36 mm** max
- expect the plane-rise readout ≈ (pre-lift − 12 mm): corner unload tilts
  the trunk 1.4–1.6° and physically lowers that corner ~12 mm — that is
  sag, not "the leg won't lift"
- the 12 measured torques are printed at HOLD4: a large left/right or
  diagonal asymmetry is the §7-C mis-calibration signature (re-zero shrinks
  it); mis-zeroing scrambles 4-leg load sharing, which is why the pre-lift
  unloads kinematically instead of waiting for a load handoff

## Beyond the sim: the IMU is the hardware oracle

The sim judged runs with privileged MuJoCo reads (trunk tilt, contact
forces) that "never steer".  On hardware the trunk DETA10 plays that role:
roll/pitch on the status line (the sim's tilt oracle, for free), the 25°
tip-over run-stop, and a 6° lean watch during the 3-leg phases that
gracefully lowers the leg instead of holding a falling stand.  Control
stays encoder-only — the IMU judges and aborts, it never steers targets.

## Stance: IN-PLACE, not the sprawl (hardware lesson 2026-07-23)

The first hardware attempt used the sim's sprawl stand pose and **failed to
stand**: crouch → sprawl requires the feet to slide ~8 cm outward, and on
the real high-grip floor a foot stuck — FR_knee stalled 45.9° short of
target with zero speed and no torque trip (the driver current-limits below
6 N·m).  This is the sim's own experiment B (grippy feet won't slide)
biting the stand-up itself.

The fix: the body now **rises vertically over the crouch feet** (the
stand-in-place convention every proven torque-mode stand uses) — nothing
slides.  Bonus: **all four swing legs are IK-feasible** from this stance
(the sprawl blocked RL/RR on the rear abduction limit).  Retuned envelope,
offline-validated per leg: H = 0.175 m (the extra height frees abduction
travel), shift 32 mm (planned margins FL 25.9 / FR 27.2 / RL 29.4 /
RR 30.9 mm), pre-lift ladder 20 → 30 mm, lift 14 mm (44 mm total commanded
raise), clear-gate rise ≥ 4 mm absolute.

First hardware pass 2026-07-23: FR lifted and held on three legs (small
clearance at the original 36 mm envelope — hence the raise to 44 mm).

## Run

```bash
cd ~/Documents/can_motor_control
.venv/bin/python dog_stand_compliance_control/crawl_hw2.0/stand3_hold_hw.py --self-test
.venv/bin/python dog_stand_compliance_control/crawl_hw2.0/stand3_hold_hw.py   # hardware
```

Robot mechanically supported for first runs.  ENTER advances at every
phase boundary; X stops; P (at HOLD4) parks back to the crouch.

## walk1 + read-only EKF (full-state stream)

`walk1_hw.py` now hosts the DOG5 state estimator (read-only, worker thread,
same pattern as the validated `vmc/ekf_stand_hw.py` stand run).  The gait
stage machine feeds it a contact schedule (swing foot airborne from UNLOAD
through TOUCHDOWN; rising edge at the measured-anchor commit), and a 2 Hz
`[ekf]` terminal line streams the full state: z, body velocity, roll/pitch/yaw
vs AHRS, bias norms, contact bits, health.

```bash
cd ~/Documents/can_motor_control
V=.venv/bin/python
$V dog_stand_compliance_control/crawl_hw2.0/walk1_hw.py --self-test   # first
$V dog_stand_compliance_control/crawl_hw2.0/walk1_hw.py \
    --raw-log walk_$(date +%m%d_%H%M).npz                             # hardware
python3 dog_stand_compliance_control/state_estimator/hw_replay.py \
    walk_*.npz --gait                                                 # analyse
```

No sudo.  In WAIT_CROUCH, **wait for `[ekf] init: bw=... bf=...` before
ENTER** (it initialises on the static crouch).  During the walk watch:
`H=OK` throughout; `c=` flips exactly one leg 0 at UNLOAD and back 1 at
LOAD; `vb` returns to ~0 in every HOLD4; EKF roll/pitch within ~2 deg of
`ahrs`; `blk` staying < ~5 ms (10 ms motor-watchdog budget).  Keys
unchanged (ENTER/X/P; tip-over + lean aborts use the same AHRS, now via
the EKF's IMU feed).  `--no-ekf` keeps posture checks but skips the
estimator; `--ekf-verbose` adds per-leg footholds + innovations.

### Live browser dashboard (`--web`)

The 2 Hz `[ekf]` text line is hard to read while the dog moves, so the
estimator state can be streamed to a browser instead — scrolling strip charts
for z, body velocity, roll/pitch (EKF vs AHRS), yaw, and a contact timeline:

```bash
$V dog_stand_compliance_control/crawl_hw2.0/walk1_hw.py \
    --raw-log walk_$(date +%m%d_%H%M).npz --web
# then on your laptop:  http://192.168.137.2:8080/   (the run prints the URLs)
```

Stdlib `http.server` only — nothing to install, no X forwarding.  A sampler
thread reads the same `EkfShared` mailbox at 20 Hz and the page polls for new
samples; the CAN control thread only rebinds a small status dict per sweep, so
a stalled or closed browser cannot affect the walk.  `--web 9000` picks a port.

The same page replays a recorded log after the run (serves once the replay
finishes — tens of seconds for a long log):

```bash
python3 dog_stand_compliance_control/state_estimator/ekf_web.py walk_*.npz
```
