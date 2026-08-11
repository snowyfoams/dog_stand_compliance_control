# Gate D1 — closed-loop in-place trot in MuJoCo, EKF in the loop

**Date:** 11 August 2026
**Status:** **PASS.** `vmc/test_trot.py` — 15 offline scheduler gates + D1-1…D1-8,
all pass. Envelope sweep 14/14 upright.

> `CONTROL_ROADMAP.md` §2 Phase 4 — **Gate D1:** trot in MuJoCo on the
> hardware-fidelity model (EKF in loop, not sim truth).

This closes D1. It does **not** touch hardware; §6 is the D2 bring-up plan.

---

## 1. What was built

| File | What it is |
|---|---|
| `vmc/dog5_trot.py` | trot schedule + **closed-loop foot placement**; emits the same dict `compute_vmc_torques` already consumes, so the VMC core is unmodified |
| `vmc/trot_mujoco.py` | closed MuJoCo → EKF → VMC → torque loop, decimated to hardware rates |
| `vmc/test_trot.py` | the D1 gates and the envelope sweep |
| `state_estimator/dog5_state_estimator.py` | health fix promoted in-tree; `swing_p` 1e4 → 1e2 |
| `ekf_closeout/estimator_health.py` | reduced to a re-export; `HealthFixEstimator` is now an alias |

`vmc/dog5_gait.py` and `vmc/dog5_vmc_core.py` are **untouched**. V1–V4 still pass.

The loop is: MuJoCo IMU + encoders + schedule contacts → EKF `predict`/`update` →
`outputs()` → trot schedule (contacts + EKF-placed swing targets) → VMC body
wrench → grasp-map distribution → `τ = −Jᵀf` → `data.ctrl`. MuJoCo ground truth
is read only by `Oracle`, which grades and never steers.

---

## 2. Headline result

25 cycles, period 0.40 s, `ds_frac` 0.34, control 250 Hz, EKF 100 Hz. Statistics
over the settled window (first 2 cycles excluded as hand-off transient).

| Quantity | Measured | Gate |
|---|---|---|
| Fell | **no** (final trunk z 0.213 m) | — |
| Per-leg swing clearance | FL 20.1 · FR 18.9 · RL 19.1 · RR 20.3 mm | ≥ 10 mm |
| Contact force at swing apex | **0.00 N on all four** | ≤ 1 N |
| In-place drift | **3.9 mm net** over 23 cycles (0.17 mm/cycle) | < 30 mm |
| Peak trunk tilt (truth) | **0.18°** (first third 0.12 → last third 0.08) | < 8°, not diverging |
| EKF z error | 12.9 mm max / 11.2 mm mean | < 30 mm |
| EKF \|v\| error | 26.9 mm/s max / 4.0 mm/s mean | < 50 mm/s |
| EKF roll/pitch error | **0.09°** max | < 2° |
| EKF healthy fraction | **1.0000** | > 0.99 |
| Peak joint torque | **2.40 N·m** | < 9 N·m capability; also under the 6.0 N·m trip |
| CoM offset from support line | 0.29 mm mean / 0.76 mm max | — |

**The Stage 8/9 failure mode is gone.** `twostand_dog5.py --sweep` scores 0/9 —
one diagonal leg reaches clean clearance and the other never leaves the ground,
because a static 2-foot diagonal has no moment authority about its support line.
Here all four legs clear ~19–20 mm at 0.00 N, every cycle, for 25 cycles. The
support pattern is the same; what changed is that it is transient and
alternating, and the double-support window restores full rank in between.

---

## 3. The finding that mattered: torque saturation, not tipping

The plan predicted the gait would be governed by the tip about the diagonal,
θ(T_ss) ≈ ½·(m·g·d/I)·T_ss², and that shortening single support would therefore
be the lever. **The data does not support that**, and the sweep says so directly:
across nine (period, ds_frac) cells the measured `tilt/T_ss²` ranges from 10 to
225 deg/s² instead of collapsing to a constant.

What actually governs is **whether the wrench the VMC asks for fits inside the
per-joint torque clamp.** At `ds_frac ≤ 0.26`, peak torque pins at exactly
`tau_max = 8.00 N·m`, and once saturated the controller stops delivering the
wrench it computed — so tilt, drift and velocity error all degrade together:

| period | ds_frac | tilt | drift | peak τ | |
|---|---|---|---|---|---|
| 0.40 | 0.18 | 4.64° | 3.04 mm/cyc | **8.00** | saturated |
| 0.40 | 0.26 | 4.44° | 3.37 mm/cyc | 6.12 | |
| 0.40 | **0.34** | **0.18°** | **0.17 mm/cyc** | **2.40** | |
| 0.40 | 0.42 | 0.09° | 0.06 mm/cyc | 2.39 | |

A 25× reduction in tilt and a 3.3× reduction in peak torque, from one parameter.
Raising `tau_max` is not the fix — 8 N·m is already past the 6.0 N·m hardware
torque trip, so the saturated operating point was never portable to the robot.

The initial 15–20% double-support target was therefore wrong for this machine,
and the default ships at **`ds_frac = 0.34`** (per-leg duty factor 0.67). It is
still a flight-free two-beat diagonal gait — the footfall is unchanged — just
with more overlap than first assumed.

Period is bracketed from the other side: below ~0.35 s the same lift must be
flown in less time, so foot accelerations and torque climb again (period 0.30 →
tilt 2.20°, τ 4.02) *even though* single support is shorter. The flat optimum is
period 0.40–0.45 with `ds_frac` 0.30–0.46.

---

## 4. Two other things worth recording

**The crawl's contact rule is wrong for a trot, and inverting it was worth
42 mm/s.** The crawl biases contact flags toward *planted*
(`walk1_hw.py`, `EKF_SWING_Z_EPS_M`) because `stand_hier_hw.py:384-406` records
that calling an inertially-stationary foot airborne dead-reckoned the base
through an 8 s zero-contact window for ±70–80 mm of z error. A trot inverts both
halves: there are always two other feet down so the filter never dead-reckons,
while the swing is fast enough that calling a moving foot "planted" asserts a
point travelling at 0.1–0.4 m/s is fixed in inertial space. Ported naively, that
cost **68.8 mm/s** of EKF velocity error against a 50 mm/s gate. The trot flag is
airborne for its whole swing and re-plants on *evidence* (estimated foot height),
not on the clock.

Relatedly, the swing arc is a **raised cosine, not `sin(πu)`**. A sine arc puts
its maximum vertical foot speed exactly at liftoff and touchdown — 0.42 m/s for
this lift and swing time. The raised cosine has zero derivative at both ends, and
the horizontal placement offset is blended with a smoothstep for the same reason,
so the foot neither jumps at liftoff nor scuffs at touchdown.

**The EKF rate is *not* a constraint — I initially concluded it was, and that
was wrong.** Measured in the saturated regime, 100 Hz gave 59 mm/s and 250 Hz
gave 44 mm/s, which looked like the hardware worker's 100 Hz was the binding
limit. Re-measured at the unsaturated operating point:

| EKF rate | \|v\| err max | | control rate | \|v\| err max |
|---|---|---|---|---|
| 50 Hz | 58.2 mm/s | | 125 Hz | 26.7 mm/s |
| **100 Hz** | **26.9 mm/s** | | **250 Hz** | **26.9 mm/s** |
| 250 Hz | 23.0 mm/s | | 500 Hz | 26.4 mm/s |
| 500 Hz | 25.6 mm/s | | | |

The existing 100 Hz worker is sufficient with margin (50 Hz is not), and the trot
is not control-rate limited at 250 Hz. Integrating every buffered IMU sample
instead of collapsing the batch to its mean — flagged as a concern for dynamic
gaits — makes **no** measurable difference (26.8 vs 26.9 mm/s); it is the
measurement update rate that matters, not the propagation. `--ekf-per-sample`
keeps the A/B available.

---

## 5. Envelope sweep — 14/14 upright, EKF healthy 1.0000 in every case

```
 period    ds   T_ss       bias  fell   tilt   drift  minclr    tau  sat
   0.30  0.18  0.123     (0, 0) False   0.59     2.8    24.7   3.59
   0.30  0.34  0.099     (0, 0) False   2.20    23.1    20.5   4.02
   0.30  0.46  0.081     (0, 0) False   0.09     1.4    15.8   2.23
   0.40  0.18  0.164     (0, 0) False   4.64    31.6    16.2   8.00  SAT
   0.40  0.34  0.132     (0, 0) False   0.18     1.8    18.9   2.40
   0.40  0.46  0.108     (0, 0) False   0.16     1.8    17.8   2.32
   0.50  0.18  0.205     (0, 0) False   5.37    32.8    14.7   7.39
   0.50  0.34  0.165     (0, 0) False   0.79     6.7    17.0   2.54
   0.50  0.46  0.135     (0, 0) False   0.78     7.7    18.5   2.69
   0.40  0.34  0.132     (5, 0) False   0.66    12.6    19.1   2.68
   0.40  0.34  0.132    (-5, 0) False   0.53    10.0    19.0   2.55
   0.40  0.34  0.132     (0, 5) False   2.03    49.8    19.8   2.67
   0.40  0.34  0.132    (0, -5) False   2.02    49.5    19.3   2.59
   0.40  0.34  0.132     (8, 8) False   2.43    48.8    18.6   3.59
```

Every leg clears 10 mm in all 14 cases and nothing falls, including with a
saturated controller — saturation costs accuracy here, not stability.

**Lateral CoM bias is ~4× worse than fore-aft**: ±5 mm in y gives 2.0° tilt and
~50 mm drift, against 0.5–0.7° and 10–13 mm for ±5 mm in x. That is geometry —
the feet span ±0.36 m in x but only ±0.11 m in y, so a lateral offset is a much
larger fraction of the half-width and moves the CoM off the diagonal far more
effectively. It is also the axis where the abduction limit caps corrective
authority at ~2 cm. **Lateral CoM calibration is the thing to get right before
D2.**

One irregularity, not smoothed over: period 0.30 / `ds_frac` 0.34 (tilt 2.20°,
drift 23.1 mm) is worse than *both* its neighbours at the same period (0.18 →
0.59°, 0.46 → 0.09°). Non-monotonic and unexplained; the default is far from
this cell, but it means the envelope is not perfectly smooth and a hardware
parameter change should be re-swept rather than interpolated.

---

## 6. Ablations — is the estimator actually load-bearing?

**Yes, decisively.** Freezing the estimate (VMC blind to the body state it is
regulating):

| | drift max | tilt max |
|---|---|---|
| EKF live | 1.8 mm | 0.18° |
| EKF frozen | **422.8 mm** | **92.52°** |

**Closed-loop foot placement: real, consistent, but second-order.** This is the
new control law in `dog5_trot.py`, and honesty requires reporting that it is not
carrying the run. With a centred CoM and no disturbance there is nothing for it
to reject — 1.8 mm with it vs 1.9 mm without, i.e. noise. Against actual
disturbances it helps consistently, in the right direction, by a modest margin:

| disturbance | placement ON | OFF | benefit |
|---|---|---|---|
| none | 1.8 mm | 1.9 mm | none (noise) |
| push 25/12 N, 0.2 s | 97.6 mm | 106.4 mm | 8% |
| push 40 N, 0.25 s | 100.9 mm | 103.3 mm | 2% |
| **CoM bias +8/+8 mm** | **53.6 mm** | **66.1 mm** | **19%** |

The body-wrench velocity damping (`kd_x`/`kd_y` = 60 N·s/m in `VMCGains`) is
doing most of the regulation; foot placement adds to it. `k_place` is genuinely
optimal at the shipped 0.12 — under CoM bias, 0.00 → 61.5 mm, **0.12 → 53.6**,
0.25 → 55.0, 0.40 → 66.9, 0.60 → 101.4 mm, i.e. higher gains over-correct and
make both tilt and torque worse. D1-8 grades placement against the CoM-bias case,
because grading it on the undisturbed run would be self-congratulation.

---

## 7. D2 — hardware bring-up plan (NOT executed)

Gate D2 is "hardware trot in place, tethered". D1 does not authorise it. Two
things must happen first, and one of them is a correction to the record.

### 7.1 The reason torque mode was abandoned does not survive the rate fix

`vmc/stand_hier_hw.py:5-6` states the torque route is dead: *"software torque at
the 20.8 Hz per-joint CAN rate cannot stabilise the legs; native position mode is
proven on this rig."* The supporting arithmetic is `vmc_stand_hw.py:96-167`:
`BRAKE_PERIOD_S = N_JOINTS / CONTROL_HZ` = 48 ms, giving a sampled-damping bound
`kd < 2J/dt` of 0.37 N·m·s/rad at the knee.

That period is wrong by 12×. `stand_hier_hw.py:9-10` and `:886-888` show the loop
paces `mb.slot(250) = 1/(250·12) = 333 µs` per slot — a full 12-motor sweep every
**4 ms**, i.e. every motor at 250 Hz. This is the identical error `main` fixed in
commit c97f1f1 ("Run every motor at 250 Hz: fix a 12× CAN control-rate error"),
and the same two independent checks apply: the bus budget (12 × 250 Hz × ~260 µs
= 78% of 1 Mbit/s) and the 10 ms driver watchdog, which at 48 ms/motor would
latch *input lost* every sweep and the robot could not have stood at all.

At the true rate the bound is 4.4 N·m·s/rad at the knee and 7.3 at abduction;
even decimating the torque update to the worker's 100 Hz it is 1.76, so
`KD_JOINT_BRAKE = 0.4` sat at ~23% of the limit, not the 109% recorded.

This does **not** prove torque VMC works on hardware — the 30 Jul run also failed
on the compliant rise from a splayed crouch, and torque fidelity (iq→τ linearity,
gear friction, 10:1 backdrivability) is still uncalibrated. It does mean the
physics argument that closed the door should be re-derived before the door is
treated as closed. **Action: fix `BRAKE_PERIOD_S` and re-derive `RunawayBrake`.**

### 7.2 Prerequisites, in order

1. **Phase 2 / Gate T1 first.** VMC has still never run closed-loop on hardware.
   An unassisted VMC stand ≥ 60 s that recovers a hand push is the precursor; a
   trot is not the place to discover the torque path.
2. **Per-joint τ calibration** (`CONTROL_ROADMAP.md` §3) — the grasp map is only
   as good as iq→τ. D1's peak demand is 2.40 N·m, comfortably under the 6.0 N·m
   trip, which is the single most encouraging number here for D2.
3. **Lateral CoM calibration.** §5 shows y-bias is the dangerous axis and the
   abduction limit caps correction at ~2 cm. Measure it; do not assume it.
4. **EKF close-out Sessions A/B** (`ekf_closeout/RUNBOOK.md`) remain open.

### 7.3 Run plan

Tethered, on the `crawl_hw2.0`/`stand_hier_hw` runner skeleton (stage machine as
data, `validate_configuration()` before arming, zero-torque preflight, SafetyGate
+ estop, IMU tip-over run-stop, `abort_freeze` on lean). Start at `ds_frac` 0.42
(more margin than the 0.34 default), `--tau-max` low, and hold 3 cycles before
extending. Bail on the first saturated sweep — §3 is the reason.

---

## 8. Reproduce

```bash
python vmc/test_trot.py --self-test   # 15 scheduler gates, no MuJoCo needed
python vmc/test_trot.py               # D1-1 .. D1-8
python vmc/test_trot.py --sweep       # the §5 envelope
python vmc/trot_mujoco.py             # watch it
python vmc/trot_mujoco.py --headless --ekf-hz 50   # A/B any rate

python state_estimator/test_estimator.py   # unaffected by the health/swing_p change
python ekf_closeout/test_closeout.py       # 24 checks
python vmc/test_vmc.py                     # V1-V4 still pass
```
