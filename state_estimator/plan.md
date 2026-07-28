# DOG5.0 Proprioceptive State Estimator — EKF Implementation Plan

**Version:** 1.0 — 28 July 2026
**Method source:** M. Bloesch, M. Hutter, M. A. Hoepflinger, S. Leutenegger, C. Gehring, C. D. Remy, R. Siegwart, *State Estimation for Legged Robots — Consistent Fusion of Leg Kinematics and IMU*, Robotics: Science and Systems VIII, 2013, pp. 17–24. Full text: <https://infoscience.epfl.ch/record/181040>
**Reference code:** `dog5_state_estimator.py` (M0, M1, M3–M11 implemented and tested), `test_estimator.py` (all tests passing).
All equation numbers in this document refer to the paper. Every equation the implementation needs is transcribed in full in **Appendix B**, so no access to the PDF is required.

---

## 1. Objective and scope

Estimate the base state of DOG5.0 using only proprioceptive sensing:

| Quantity | Symbol | Observability (paper, Sec. IV-A) |
|---|---|---|
| Base position | r ∈ ℝ³ | x, y **unobservable** (drifts); height z observable through contacts |
| Base velocity | v ∈ ℝ³ | **Fully observable** with ≥ 1 foot in contact |
| Attitude | q (quaternion) | Roll, pitch **observable**; yaw **unobservable** (drifts) |
| Foothold positions | p₁…p₄ ∈ ℝ³ each | Estimated as landmarks (SLAM view) |
| Accelerometer bias | b_f ∈ ℝ³ | Observable for non-degenerate motion (Table I) |
| Gyroscope bias | b_ω ∈ ℝ³ | Observable for non-degenerate motion (Table I) |

**Sensors:** one body-mounted IMU (raw specific force and angular rate, body frame, already extrinsically aligned) and 12 joint encoders. A per-foot binary contact flag is a switching **input** to the filter, produced outside it.

**In scope:** the algorithm, unit tests, integration tests on simulated data, and offline replay of hardware logs.
**Out of scope (later phases):** contact-detector tuning, closing the control loop, VMC or MPC integration.

The drift in x, y, and yaw is a structural property proven by the paper's observability analysis. It is expected behaviour, not a defect. In the paper's hardware test, position drift reached roughly 10 % of distance travelled.

---

## 2. Fixed conventions (project-wide, do not change mid-build)

| Item | Convention |
|---|---|
| Frames | I = inertial (gravity-aligned, origin at startup). B = body (IMU frame after your verified extrinsic). |
| Quaternion | `[x, y, z, w]`, represents rotation I → B. |
| Product | JPL convention: `C(a ⊗ b) = C(a) · C(b)`. This makes eqs. (32) and (53) valid as written. |
| Rotation matrix | `C(q)` maps I-coordinates of a vector to B-coordinates. |
| Attitude error | Body-frame rotation vector δφ: `q = ζ(δφ) ⊗ q̂` (eqs. 11–12). Small angle: `C(ζ(δφ)) ≈ I − δφ×`. |
| Gravity | `g = [0, 0, −9.81]` in I. Measurement model `f = C(a − g)` (eq. 3): a level robot at rest reads `f ≈ [0, 0, +9.81]` in B. |
| Units | m, m/s, m/s², rad, rad/s throughout. Convert deg/s and g-units in the driver, never inside the filter. |
| Dimensions | Nominal state: 28. Error state: 27. The quaternion has 4 components but 3 degrees of freedom, so P is 27 × 27. |

**Error-state index map** (used by M1, M5, M7, M8, M11):

| Block | Slice |
|---|---|
| δr | 0:3 |
| δv | 3:6 |
| δφ | 6:9 |
| δp₁ … δp₄ | 9:21 (3 per leg) |
| δb_f | 21:24 |
| δb_ω | 24:27 |

---

## 3. System-level input, output, and data flow

**Inputs per tick**

| Signal | Dim | Rate | Source |
|---|---|---|---|
| f̃ specific force, B frame | 3 | IMU rate (≥ 200 Hz) | IMU |
| ω̃ angular rate, B frame | 3 | IMU rate | IMU |
| α̃ joint angles | 12 | encoder rate | CAN |
| c contact flags | 4 | control rate | gait schedule + force check |
| Δt | 1 | — | timestamps |

**Outputs per tick**

| Signal | Use |
|---|---|
| r, v, q, ω̂ | controller feedback |
| C·v (body-frame velocity) | what MPC/VMC usually consumes |
| σ = √diag(P) | health monitoring and gating |

**Two-rate structure**

```
IMU (fast) ──► [M4 propagate x] ──► [M5 propagate P]
                                          │
encoders + contacts (slower) ──► [M6 transitions]
                                 [M7 y, H, R] ◄───┘
                                 [M9 gate]
                                 [M8 update x, P] ──► [M11 outputs]
```

Prediction (M4, M5) runs every IMU sample. The update chain (M6–M9, M8) runs every encoder sample. If no foot is in contact, the update is skipped and the filter dead-reckons.

---

## 4. Module specifications

Each module lists: Function, Input, Output, Method (with paper equations), and its Test. Where the implementation deviates from the paper, the deviation is stated explicitly.

---

### M0 — Rotation math library

**Function.** Pure math kernel called by every other module. No hardware, no state.

**Contents**

| Function | Input → Output | Used in |
|---|---|---|
| `skew(v)` | ℝ³ → 3×3, the (·)^× operator | F (M5), H (M7) |
| `ζ(v)` | rotation vector → quaternion, eq. (13) | eqs. (32), (53) |
| `ζ⁻¹(q)` | quaternion → rotation vector | reading attitude error |
| `quat_mul(a,b)` | quaternion product, JPL | eqs. (32), (53) |
| `quat_to_C(q)` | quaternion → rotation matrix | everywhere |
| `normalize(q)` | unit norm, scalar part ≥ 0 | after every propagation |
| `Γₙ(ω, Δt)` | series of eq. (28); Γ₀ = exp(Δt·ω×), eq. (29) | F and Q (M5) |

**Method.** Direct implementation of eqs. (13), (28), (29). The continuous quaternion rate map Ω of eqs. (16), (20) is never used directly; the discrete form `q⁻ = ζ(Δt ω̂) ⊗ q` (eq. 32) replaces it.

**Test.** Four algebraic identities, all already passing in `test_estimator.py`:
1. `C(a ⊗ b) == C(a) · C(b)`
2. `ζ⁻¹(ζ(v)) == v`
3. `C(ζ(δ)) ≈ I − skew(δ)` for small δ
4. `Γ₀(ω, Δt) == expm(Δt · skew(ω))`

**Status: done.** Your only task is the 10-minute convention check against the real driver (Sec. 5, Step 0).

---

### M1 — State container

**Function.** Hold the nominal state x, the covariance P, and the index slices in one place.

**Input.** Number of legs N.
**Output.** x (28), P (27 × 27), named slices per the index map above.

**Method.** State vector eq. (8): `x = [r, v, q, p₁…p_N, b_f, b_ω]`. Error state eq. (10). Covariance definition eq. (9): `P = Cov(δx)`. The quaternion never appears in P; its error is the 3-vector δφ through eqs. (11)–(13).

**Test.** Slice bookkeeping: write a marker value into each block through its slice, read it back, confirm no overlap and full coverage of 0:27.

---

### M2 — Leg kinematics (the only DOG5-specific module)

**Function.** Forward kinematics and its analytic Jacobian for each leg.

**Input.** Leg index i, joint angles α_i ∈ ℝ³.
**Output.** s_i ∈ ℝ³ — foot contact point in the **body** frame; J_i = ∂s_i/∂α_i ∈ ℝ³ˣ³.

**Method.** Eq. (2): `s_i = lkin_i(α) + n_s,i`. The Jacobian is eq. (26) and is **required**, not optional: it maps encoder noise into Cartesian measurement noise via eq. (25) in M7. A leg near a kinematic singularity is then automatically trusted less.

**Requirements.** Measured link lengths of the assembled robot, not CAD values. Calibrated joint zero offsets. Port the structure from the DOG4.6 pipeline; replace the constants.

**Test.**
1. Finite-difference check of J_i (tolerance 1e−5). Pattern is in `test_estimator.py`.
2. Four-foot closure: body level on a flat floor, FK of all four legs must place the four contact points in one horizontal plane within a few millimetres. Residual tilt here becomes a direct roll/pitch bias in the filter.
3. If an IK exists, FK∘IK round trip.

---

### M3 — Initialisation

**Function.** Build x₀ and P₀ from a static hold.

**Input.** 5–10 s of static f̃, ω̃ samples (motors energised), α̃, contact flags.
**Output.** x₀, P₀.

**Method.** Not prescribed by the paper; standard static alignment, consistent with Sec. IV-A (yaw and position are unobservable, so they are simply defined to be zero):

```
b_ω = mean(ω̃)                                  # static gyro mean is the bias
ĝ_B = mean(f̃) / |mean(f̃)|                      # measured "up" in B
b_f = mean(f̃) − 9.81 · ĝ_B                     # magnitude residual
q₀  = rotation taking inertial +z onto ĝ_B
r₀  = 0,  v₀ = 0,  yaw = 0
p_i = r₀ + C(q₀)ᵀ · s_i                         # footholds from FK
```

P₀ diagonal: tight on r (defined, not measured), moderate on v, ~1° on attitude, moderate on footholds, datasheet-level on biases.

**Test.** Noiseless level input ⇒ roll = pitch = 0 exactly, footholds on z = 0.

---

### M4 — Nominal propagation (prediction of the state)

**Function.** Dead-reckon the nominal state forward one IMU step.

**Input.** x, f̃, ω̃, Δt.
**Output.** Predicted x⁻.

**Method.** Bias correction eqs. (36)–(37); gravity removal by inverting eq. (3); discrete propagation eqs. (30)–(35) under zero-order hold:

```
f̂  = f̃ − b_f                        (36)
ω̂  = ω̃ − b_ω                        (37)
a_I = Cᵀ f̂ + g                       from (3)
r⁻ = r + Δt·v + ½Δt²·a_I             (30)
v⁻ = v + Δt·a_I                      (31)
q⁻ = ζ(Δt·ω̂) ⊗ q, then normalize    (32)
p_i⁻ = p_i                            (33)  footholds stationary
b_f⁻ = b_f,  b_ω⁻ = b_ω               (34)–(35)
```

Note the leverage of attitude error here: 1° of attitude error leaves `9.81·sin(1°) ≈ 0.17 m/s²` of unremoved gravity, which integrates into 0.17 m/s of velocity error per second. This is why the correction step exists.

**Test.** Zero-noise, zero-bias integration of a known trajectory for 10 s: drift must be at the level of discretisation error only.

---

### M5 — Covariance propagation (prediction of the uncertainty)

**Function.** `P⁻ = F P Fᵀ + Q` (eq. 45). The most important and least intuitive module.

**Input.** P, C, f̂, ω̂, Δt, contact flags.
**Output.** F (27×27), Q (27×27), P⁻.

**Method.** Continuous error dynamics eqs. (38)–(43), discretised to F per eq. (44) using Γ₀, Γ₁; Q from Van Loan-style discretisation (the matrix printed after eq. 44) using Γ₂, Γ₃. Non-trivial blocks of F:

| Block | Value | Physical meaning |
|---|---|---|
| ∂δr/∂δv | Δt·I | position accumulates velocity error |
| ∂δr/∂δφ | −½Δt²·Cᵀ f̂× | tilt mis-cancels gravity → position |
| ∂δv/∂δφ | −Δt·Cᵀ f̂× | tilt mis-cancels gravity → velocity |
| ∂δv/∂δb_f | −Δt·Cᵀ | accel bias integrates into velocity |
| ∂δφ/∂δφ | Γ₀ᵀ | attitude error rotates with the body |
| ∂δφ/∂δb_ω | −Γ₁ᵀ | gyro bias integrates into attitude |

**These off-diagonal blocks are the entire mechanism of the filter.** The leg measurement (M7) observes only a position-type quantity; velocity and biases have zero rows in H. They are corrected exclusively through the correlations that F builds inside P. If F is wrong, the filter runs without crashing and simply never converges on v, b_f, b_ω.

Foothold process noise (eqs. 17, 21): per-axis diagonal in the **body** frame (tangential slip ≠ normal), rotated into I as `Δt · Cᵀ Q_p C`. A swinging foot receives a very large value (10⁴), which is the paper's Sec. III-B mechanism for releasing a lifted foot from its old landmark.

**Test.** Numerical verification of F: perturb one error-state component, propagate the nominal state with and without the perturbation, and confirm the difference matches `F · δx` to first order. Do this column by column. This test finds errors that visual inspection of the matrix will not. Also confirm P⁻ stays symmetric (symmetrise each step: `P ← ½(P + Pᵀ)`).

---

### M6 — Contact transition handling

**Function.** Manage the foothold "map" at liftoff and touchdown.

**Input.** Contact flags now and at the previous tick, α̃, current x.
**Output.** Re-anchored footholds; reset covariance blocks; recorded FEJ anchors for M10.

**Method.** The paper realises intermittent contact purely through the infinite-process-noise trick (Sec. III-B). The implementation uses the equivalent explicit form, which is numerically cleaner and produces a loggable event. On a rising edge (touchdown) of leg i:

```
p_i = r + Cᵀ · s̃_i            # plant a new landmark where the leg says it is
P[p_i-rows, :] = 0             # forget all old correlations
P[:, p_i-cols] = 0
P[p_i, p_i]   = σ_reset² · I   # fresh, moderate uncertainty
p*_i = p_i                     # FEJ anchor, eq. (76), consumed by M10
```

Falling edge (liftoff): nothing to do here; M5's large Q handles the swing phase.

**Test.** In simulation, lift a foot and replace it elsewhere: the body states r, v, q must not jump at either transition.

---

### M7 — Measurement assembly

**Function.** Build the innovation y, the Jacobian H, and the noise R for the m legs currently in contact.

**Input.** α̃, contact flags, predicted x⁻.
**Output.** y ∈ ℝ³ᵐ, H ∈ ℝ³ᵐˣ²⁷, R ∈ ℝ³ᵐˣ³ᵐ (block diagonal).

**Method.** Transformed measurement eqs. (22)–(24); measurement model eq. (27); innovation eq. (46); linearised error eq. (47); noise eq. (25); stacking eq. (48). Per contact leg i:

```
s̃_i = lkin_i(α̃_i)                        # M2
ŝ_i = C (p_i − r)                         # predicted appearance of the foot
y_i = s̃_i − ŝ_i                           (46)

H_i = [ −C | 0 | (C(p_i − r))× | 0 … +C … 0 | 0 | 0 ]     (47)
        δr   δv       δφ            δp_i        δb_f δb_ω

R_i = R_s + J_i R_α J_iᵀ                  (25)
```

Two structural facts to internalise:
- The δv and bias columns of H are **zero**. Velocity and biases are corrected only through P's correlations (see M5).
- The `(C(p_i − r))×` term is what makes roll and pitch observable: a wrong tilt rotates all predicted foot positions by the wrong amount, and the four feet disagree in a pattern unique to the tilt.

If m = 0 (flight phase), skip the update entirely.

**Test.** Numerical H: finite-difference the predicted measurement with respect to each error-state block and compare against the analytic H, column by column.

---

### M8 — Update

**Function.** Fuse y into x and P.

**Input.** P⁻, H, R, y.
**Output.** Corrected x⁺, P⁺.

**Method.** Eqs. (49)–(53), with one numerical improvement over the paper:

```
S  = H P⁻ Hᵀ + R                          (49)  innovation covariance
K  = P⁻ Hᵀ S⁻¹                            (50)  Kalman gain
Δx = K y                                  (51)
P⁺ = (I − KH) P⁻ (I − KH)ᵀ + K R Kᵀ       Joseph form of (52)
```

Injection into the nominal state: additive for every block **except attitude**, which is multiplicative per eq. (53):

```
q⁺ = ζ(Δφ) ⊗ q⁻, then normalize
```

The Joseph form costs one extra matrix product and keeps P symmetric positive semi-definite over long runs, which the plain `(I − KH)P⁻` of eq. (52) does not guarantee in floating point.

**Test.** P symmetric and PSD at every step (`min eig > −1e−9`). Then run gate C5 (Sec. 7) — the decisive checkpoint of the whole build.

---

### M9 — Outlier gating (addition; not in the paper)

**Function.** Reject a leg whose innovation is statistically impossible. This is the algorithmic safety net for a wrong contact flag — the input the paper itself identified as fault-prone on hardware.

**Input.** y, S.
**Output.** Reduced measurement set (or none).

**Method.** Per-leg Mahalanobis distance:

```
d²_i = y_iᵀ (S_ii)⁻¹ y_i
drop leg i if d²_i > 9.0        # χ², 3 DOF, ≈ 97 %
```

Recompute S after dropping. If all legs are dropped, skip the update.

**Rule during bring-up:** keep this **disabled** until gate C5 passes. An active gate can silently discard correct measurements and mask a real bug in M5 or M7.

**Test.** In simulation, force one false "in contact" flag during swing: the gate must drop that leg and the state error must stay bounded.

---

### M10 — Observability constraint (First-Estimates Jacobian)

**Function.** Keep the linearised filter's unobservable subspace identical to the true one (x, y position and yaw), preventing the spurious confidence described in Sec. IV-B, which otherwise makes the filter overconfident and eventually inconsistent.

**Input.** Foothold anchors p*_i recorded by M6 at touchdown.
**Output.** Modified H (M7 uses p*_i in the lever-arm term).

**Method.** The paper imposes constraint (70), `M U = 0`, through linearisation choices (75)–(76) plus auxiliary IMU terms (77)–(79) that make the prediction constraints (71)–(73) hold. The reference implementation applies the **reduced form**: only the foothold lever arm in H is frozen at its first-available (touchdown) value, eq. (76); r and q in the Jacobian use the current a priori estimate, eq. (75); the innovation and the nominal state always use current values. The full auxiliary-term machinery (77)–(79) is a documented later refinement, to be added only if long-duration runs show yaw covariance shrinking.

**Bias-estimation switch (Table I).** Observability rank drops when ω ≈ 0. A quasi-static crawl sits on that singularity. Procedure: estimate biases during a deliberate excitation manoeuvre at startup (±10° body roll/pitch, feet planted), then call `set_bias_estimation(False)` before the gait starts. The paper's authors built the same runtime switch for the same reason.

**Test.** Multi-minute simulated run: σ_yaw must grow monotonically. If it shrinks, the constraint is not working. (Add this module **last**; the filter works without it on short runs, and its presence complicates debugging of M5–M8.)

---

### M11 — Output interface and health monitor

**Function.** Expose exactly what a controller consumes, plus self-diagnosis.

**Input.** x, P.
**Output.** r, v, q, ω̂ (bias-corrected rate); body-frame velocity C·v; σ = √diag(P); a boolean health flag.

**Method.** Health rule, first version:

```
healthy = all σ_v < 0.15 m/s  AND  all(P finite)  AND  min eig(P) > −1e−9
```

On `not healthy`, the controller falls back to a safe stance instead of acting on the estimate. Build this now, even though it is only consumed when the loop closes — retrofitting safety after the first crash is the expensive order.

**Test.** Feed deliberately corrupted contact flags in simulation and confirm the flag trips before the velocity error exceeds the fallback threshold.

---

### M12 — Simulated data generator

**Function.** Ground-truth scenarios with known noise and known biases. This is the only environment in which "correct" is defined exactly, so it is built **before** the filter, not after.

**Input.** A truth trajectory (r, v, a, q, ω over time) and noise/bias settings.
**Output.** Synthetic f̃, ω̃, α̃ streams and contact flags.

**Scenarios**

| # | Scenario | What it isolates |
|---|---|---|
| a | Static hold | conventions, initialisation, bias estimation |
| b | Pure yaw rotation | gyro path, Γ matrices, yaw unobservability |
| c | Forward translation, feet planted | velocity observability through P correlations |
| d | Trot: diagonal pairs, planted stance feet, swept swing feet | contact transitions, foothold resets, full pipeline |

**Critical construction detail.** Stance feet must be genuinely stationary in the inertial frame; encoder angles are synthesised by IK from the (moving) body to the (fixed) foot. Synthesising a foot that slides along with the body violates the filter's core assumption and produces large, entirely fake velocity errors. This exact bug occurred during reference-code development and cost the most debugging time of any single issue — the estimator was right and the test data were wrong.

**Test (of the generator itself).** With zero noise and zero bias, the filter must reproduce the truth to numerical precision. Any residual error is then unambiguously an algorithm bug.

---

## 5. Build plan (step by step)

Dependency order:

```
M0 ─► M2 ─► M12 ─► M1 ─► M3 ─► M4 ─► M5 ─► M6 ─► M7 ─► M8 ─► M9, M10, M11
```

| Step | Work | Effort | Gate |
|---|---|---|---|
| 0 | Convention check of M0 against the real driver: `f ≈ [0, 0, +9.81]` at rest and level; rate in rad/s; measured sample rate. Fix signs/units **in the driver**. | 0.5 h | C1 |
| 1 | M2: port FK from DOG4.6, insert measured link lengths and calibrated zeros, write the analytic Jacobian. | 0.5–1 day | C2 |
| 2 | M12: scenarios (a)–(d), with the planted-foot construction. | 0.5 day | generator noiseless-exact check |
| 3 | M1 + M3. | 0.5 day | C3a |
| 4 | M4. | 0.5 day | C3 |
| 5 | M5. Budget the most time here; the numerical-F test is non-negotiable. | 1 day | C4 |
| 6 | M6 + M7, with the numerical-H test. | 0.5 day | — |
| 7 | M8. Run scenario (a) noiseless, then noisy. | 0.5 day | **C5 — the decisive gate** |
| 8 | M9, M10, M11. Enable the gate only after C5. | 0.5 day | C6 |
| 9 | Scenario (d) full trot simulation. | 0.5 day | C7 |
| 10 | Hardware log replay (when logs exist). | — | C8 |

**Working rules**

1. Test every module against M12 immediately after writing it. Never write two untested modules in a row.
2. When a test fails, the fault is in the newest module until proven otherwise.
3. Keep M9 (gating) off until C5 passes; keep M10 (FEJ) out until C5 passes.
4. Commit after every passing gate, so any regression bisects in minutes.

---

## 6. Test plan

### 6.1 Unit level (per module)

- M0: the four algebraic identities (already passing).
- M2: finite-difference Jacobian; four-foot closure; FK∘IK round trip.
- M5: numerical F, column by column.
- M7: numerical H, column by column.
- M8: P symmetric and PSD every step.

Finite differencing is the workhorse: every analytic Jacobian in the filter has a numerical twin, and disagreement between them localises the bug to one function.

### 6.2 Integration ladder (against M12)

Run in this order; each level isolates one new mechanism:

1. **Static, noiseless, zero bias** → every error ≈ 0 to machine precision. Catches conventions, indexing, sign errors.
2. **Static, noiseless, with bias** → bias states converge to the injected truth; attitude settles. Catches the bias columns of F.
3. **Static, noisy, with bias** → errors settle inside the 3σ hull. Catches noise bookkeeping (Q, R).
4. **Rotate** (scenario b) → attitude tracks; yaw σ grows. Catches Γ handling.
5. **Translate** (scenario c) → velocity converges although nothing measures it. Catches the P-correlation mechanism end to end.
6. **Trot** (scenario d) → transitions clean, no jumps at touchdown, bounded errors. Catches M6.

### 6.3 Statistical consistency

- **3σ containment:** estimate errors remain inside the ±3σ hull ≥ 99 % of samples (the paper's own validation criterion, Sec. V and Figs. 3–5).
- **Innovation whiteness:** per leg and per axis, y is zero-mean and roughly uncorrelated in time. Persistent structure in y is the primary diagnostic channel (see Sec. 8).
- **FEJ check:** σ_yaw monotonically non-decreasing over minutes.

### 6.4 Numerical health and timing

- No NaN/Inf anywhere in x or P; `min eig(P) > −1e−9`; symmetry enforced each step.
- Per-cycle time (predict + update, 27 states) target < 1 ms. Reference implementation measures ≈ 0.8 ms in pure NumPy, comfortable for a 400 Hz loop.

### 6.5 Hardware log replay (gate C8)

Replay recorded IMU + encoder + contact logs through the filter offline and read the innovations:

- Zero-mean, inside 3σ → healthy.
- One leg persistently offset → that leg's kinematic calibration (M2 constants).
- All legs offset together → base-frame definition or IMU extrinsic.
- Spikes at gait transitions → contact flag timing.
- Offset proportional to body speed → sensor timing skew (outside the algorithm; fix timestamps).

---

## 7. Checkpoints and acceptance criteria

| Gate | After | Pass criteria |
|---|---|---|
| C1 | Step 0 | Four M0 identities pass in your environment; driver outputs `+9.81` convention, rad/s. |
| C2 | M2 | Jacobian FD error < 1e−5; four-foot closure coplanar within a few mm. |
| C3a | M3 | Noiseless level init: roll = pitch = 0; footholds at z = 0. |
| C3 | M4 | 10 s noiseless dead reckoning: drift at discretisation level only. |
| C4 | M5 | Numerical-F match, every column, first order. |
| **C5** | **M8** | **Static noiseless: all errors < 1e−9 after transient. Static noisy (motors-on noise levels): |v| error < 5 mm/s, roll/pitch < 0.3°, z < 5 mm, steady state.** |
| C6 | M10 | σ_yaw non-decreasing over a multi-minute run. |
| C7 | trot sim | 6 s trot: |v| error < 0.05 m/s, roll/pitch < 2°, z < 3 cm. (Reference run: 2 mm/s, 0.32°, 3.6 mm.) |
| C8 | log replay | Innovations zero-mean, ≥ 95 % of samples inside 3σ, no leg with a persistent offset. |

C5 is the point of no ambiguity: if it fails, do not proceed; the fault is in M0–M8 and is findable with the numerical tests above.

---

## 8. Debugging guide — failure signatures

| Symptom | Most likely cause | Where to look |
|---|---|---|
| Looks right for ~1 s, then slow divergence | Quaternion convention (product order, C transpose) | M0 identities |
| Velocity never converges; attitude fine | Missing/incorrect off-diagonal blocks of F | M5 numerical-F test |
| Bias states drift to nonsense | Bias columns of F or Q signs | M5 |
| One leg: constant innovation offset | Link length or joint zero of that leg | M2 constants |
| All legs: common innovation offset | Base-frame origin or IMU extrinsic | frame definitions |
| State jumps at every touchdown | Anchor/reset logic, or flags arriving late | M6, contact timing |
| Innovation grows with body speed | Timing skew between IMU and encoders | timestamps (not the filter) |
| σ_yaw shrinking over minutes | Observability constraint absent/wrong | M10 |
| P loses symmetry / goes indefinite | Plain (I−KH)P form, no symmetrisation | M8 (Joseph form) |
| Fine noiseless, diverges with noise | Q or R too small (overconfidence) | motors-on noise scaling |
| Update rejects everything | Gate enabled too early, or S ill-conditioned | disable M9, check R |

---

## 9. Parameters (initial values and provenance)

| Parameter | Start value | Provenance |
|---|---|---|
| σ_f, σ_ω | datasheet noise density × motors-on/off ratio from static logs | Module 2 of the hardware plan |
| σ_bf, σ_bω | datasheet bias instability | Allan/datasheet |
| σ_α (encoder) | 0.002 rad | resolution + observed jitter |
| σ_s (kinematic model error) | 0.010 m | dominant tuning knob; reduce as calibration improves |
| σ_p,xy / σ_p,z (slip, in contact) | 1 mm/√s / 0.5 mm/√s | tune upward if the filter fights the contact flags |
| Swing-foot Q | 10⁴ | paper Sec. III-B ("very large value") |
| σ_reset (touchdown) | 0.05 m | fresh-landmark uncertainty |
| Mahalanobis gate | 9.0 (χ², 3 DOF) | ≈ 97 %; **off** until C5 |
| P₀ attitude | (1°)² | static-alignment quality |

Tuning order after C8: σ_s first, then σ_p, IMU terms last. Do not use IMU covariances to compensate for a kinematics or contact problem.

---

## 10. References

1. M. Bloesch, M. Hutter, M. A. Hoepflinger, S. Leutenegger, C. Gehring, C. D. Remy, R. Siegwart, *State Estimation for Legged Robots — Consistent Fusion of Leg Kinematics and IMU*, Robotics: Science and Systems VIII, 2013, pp. 17–24. <https://infoscience.epfl.ch/record/181040>
2. G. P. Huang, A. I. Mourikis, S. I. Roumeliotis, *Observability-based Rules for Designing Consistent EKF SLAM Estimators*, International Journal of Robotics Research, 29:502–528, 2010. (FEJ / OC-EKF; paper ref. [10].)
3. R. Hermann, A. Krener, *Nonlinear Controllability and Observability*, IEEE Trans. Automatic Control, 22(5):728–740, 1977. (Observability analysis; paper ref. [8].)
4. C. Van Loan, *Computing Integrals Involving the Matrix Exponential*, IEEE Trans. Automatic Control, 23(3):395–404, 1978. (Discretisation of F, Q; paper ref. [18].)
5. N. El-Sheimy, H. Hou, X. Niu, *Analysis and Modeling of Inertial Sensors Using Allan Variance*, IEEE Trans. Instrumentation and Measurement, 57(1):140–149, 2008. (IMU noise identification; paper ref. [5].)
6. M. Camurri et al., *Probabilistic Contact Estimation and Impact Detection for State Estimation of Quadruped Robots*, IEEE Robotics and Automation Letters, 2(2):1023–1030, 2017. (Upgrade path for contact detection.)
7. R. Hartley, M. Ghaffari Jadidi, J. Grizzle, R. M. Eustice, *Contact-Aided Invariant Extended Kalman Filtering for Legged Robot State Estimation*, RSS 2018. Code: <https://github.com/UMich-BipedLab/Contact-Aided-Invariant-EKF>. (Modern InEKF alternative with better convergence properties; a possible later migration, same inputs and outputs.)
8. M. Bloesch, *LSE — Legged State Estimation Library* (original C++ code of the reference paper): <https://github.com/bloesch/LSE>
9. GTSAM, *The Manifold Kalman Filter Hierarchy, Part 2: Legged State Estimation* (2026 tutorial with reference implementations): <https://gtsam.org/2026/03/17/legged-state-estimation-part2.html>
10. mayataka, *legged_state_estimator* (C++ InEKF library with quadruped MPC example): <https://github.com/mayataka/legged_state_estimator>

---

## Appendix A — audit the real DOG5 codebase first

This appendix is about **your actual DOG5 repository**, not about any example code shown earlier in this conversation. Before following the build plan in Sec. 5 as a schedule, check what already exists in your own codebase — do not assume a module is absent just because it was not discussed here, and do not assume something old still works. Do this audit before writing anything new.

For each row, check your repository and fill in the last column yourself.

| Module | What to look for in the real DOG5 repo | Real-repo status |
|---|---|---|
| M0 rotation math | any existing quaternion/rotation utility (e.g. ported from DOG4.6 sim) | unknown — check |
| M1 state container | none expected yet | unknown — check |
| M2 leg kinematics | your DOG4.6 FK/IK code; **is it using measured link lengths or CAD values?** | unknown — check |
| M3 initialisation | none expected yet | unknown — check |
| M4/M5 prediction | none expected yet | unknown — check |
| M6 contact transitions | none expected yet | unknown — check |
| M7/M8 update | none expected yet | unknown — check |
| IMU driver | raw-mode configuration, extrinsic transform — you said this is done and passed | **done** (your report) |
| Joint zero calibration | last calibration date/method; has anything moved since (e.g. after the observed body-roll issue)? | unknown — check |
| Contact/force estimation | your torque-derived foot force function — what state is it in? | unknown — check |
| Logger | timestamped multi-channel logging capability | unknown — check |
| CAN bandwidth | measured achievable polling rate at 12 joints | unknown — check |

The build plan in Sec. 5 assumes all of the "unknown — check" rows start from zero. If any already exist in your repo, that step in Sec. 5 shrinks to a verification task instead of a build task — but that can only be decided after you look. Nothing about the real codebase's state is known until this audit is done.

---

## Appendix B — Complete equation reference (paper transcription)

**Purpose.** The implementing agent (Claude Code) cannot open the paper PDF. This appendix transcribes every equation the implementation needs, with the paper's own numbering, so this file is self-sufficient. Equations (54)–(66) — the robocentric transformation and the observability-analysis derivation — are analysis-only and deliberately omitted; no line of code evaluates them (see B.12).

### B.0 Notation

```
I, B          inertial frame / body frame
C = C(q)      3×3 rotation matrix mapping I-coordinates into B-coordinates
q             quaternion [x, y, z, w] representing the rotation I → B
g             gravity in I:  [0, 0, −9.81]ᵀ
r, v          base position / velocity, expressed in I
p_i           foothold position of leg i, expressed in I     (i = 1 … N, N = 4)
b_f, b_ω      accelerometer / gyroscope bias, expressed in B
f, ω          true specific force / angular rate, in B
α             joint angle vector (12);  α_i = the 3 joints of leg i
s_i           foot contact point of leg i, expressed in B
Δt            IMU sample period
(·)^×         skew-symmetric matrix:  a^× b = a × b
⊗             quaternion product (convention fixed in B.11)
x̃  /  x̂       measured  /  estimated quantity
x⁻ /  x⁺      a priori  /  a posteriori estimate
x*            linearisation point (Sec. IV-B only)
```

### B.1 Sensor models — eqs. (1)–(7)  [noise inputs of M5, M7]

```
(1)  α̃ = α + n_α                      n_α  ~ N(0, R_α)    discrete
(2)  s_i = lkin_i(α) + n_s,i          n_s,i ~ N(0, R_s)    discrete, body frame
(3)  f = C (a − g)                    a = absolute acceleration in I
(4)  f̃ = f + b_f + w_f                w_f  : white noise, covariance Q_f
(5)  ḃ_f = w_bf                       w_bf : white noise, covariance Q_bf
(6)  ω̃ = ω + b_ω + w_ω                w_ω  : white noise, covariance Q_ω
(7)  ḃ_ω = w_bω                       w_bω : white noise, covariance Q_bω
```

All covariance parameters are **isotropic diagonal** (σ²·I). This assumption is load-bearing: it is what lets rotation matrices commute past the Q blocks in B.7.

Consequence of (3) to verify on hardware: level and at rest, a = 0 ⇒ f = −Cg ≈ [0, 0, +9.81] in B.

### B.2 State definition and attitude error — eqs. (8)–(13)  [M1]

```
(8)   x  := [ r, v, q, p_1 … p_N, b_f, b_ω ]             nominal, dim 16+3N = 28
(9)   P  := Cov(δx)
(10)  δx := [ δr, δv, δφ, δp_1 … δp_N, δb_f, δb_ω ]      error,   dim 15+3N = 27
(11)  q = δq ⊗ q̂                                          error applied by LEFT multiplication
(12)  δq = ζ(δφ)
(13)  ζ(v) = [ sin(‖v‖/2) · v/‖v‖ ,  cos(‖v‖/2) ]        vector part first, scalar last
```

Small-angle identity relied on by (39) and (47):  `C(ζ(δφ)) ≈ I − δφ^×`.

ζ⁻¹ (needed for (79) and for reading attitude corrections): for q = [q_v, q_w],
`δφ = 2·atan2(‖q_v‖, q_w) · q_v/‖q_v‖`, wrapped to (−π, π].

### B.3 Continuous prediction model — eqs. (14)–(21)  [background of M4/M5]

```
(14)  ṙ = v
(15)  v̇ = a = Cᵀ (f̃ − b_f − w_f) + g
(16)  q̇ = ½ Ω(ω̃ − b_ω − w_ω) q
(17)  ṗ_i = Cᵀ w_p,i                                      foothold noise defined in B
(18)  ḃ_f = w_bf
(19)  ḃ_ω = w_bω

(20)  Ω(ω) = [    0     ω_z   −ω_y    ω_x
                −ω_z     0     ω_x    ω_y
                 ω_y   −ω_x     0     ω_z
                −ω_x   −ω_y   −ω_z     0   ]
```

(20) is never called by the discrete implementation — eq. (32) replaces it.

```
(21)  Q_p,i = diag( w_p,i,x , w_p,i,y , w_p,i,z )         body frame, per axis
```

Contact switching rule (Sec. III-B): while foot i has no ground contact, set Q_p,i to a very large value, so its foothold estimate is free to relocate; when contact is regained, the old foothold is effectively dropped. (The reference code implements the equivalent explicit re-anchor + covariance reset — see M6.)

### B.4 Measurement model — eqs. (22)–(27)  [M7]

```
(22)  s̃_i := lkin_i(α̃)
(23)        ≈ lkin_i(α) + J_lkin,i · n_α
(24)        ≈ s_i + n_i ,          n_i := −n_s,i + J_lkin,i · n_α
(25)  R_i = R_s + J_lkin,i · R_α · J_lkin,iᵀ
(26)  J_lkin,i := ∂ lkin_i(α) / ∂ α_i                     3×3, analytic (from M2)
(27)  s̃_i = C (p_i − r) + n_i                             THE measurement equation
```

### B.5 Discretisation helpers — eqs. (28)–(29)  [M0]

```
(28)  Γ_n := Σ_{j≥0}  Δt^{j+n} / (j+n)!  ·  (ω^×)^j
(29)  Γ_0 = Σ_{j≥0} (Δt·ω^×)^j / j!  =  exp( Δt·ω^× )
```

Γ₀ and Γ₁ enter F; Γ₂ and Γ₃ enter Q. The paper notes a closed form similar to Rodrigues' formula; the reference code evaluates the truncated series (12 terms), which is exact to machine precision for Δt·‖ω‖ ≲ 0.05 — always true at ≥ 200 Hz for realistic body rates.

### B.6 Discrete prediction, zero-order hold — eqs. (30)–(37)  [M4]

```
(36)  f̂_k = f̃_k − b̂⁺_{f,k}
(37)  ω̂_k = ω̃_k − b̂⁺_{ω,k}

(30)  r̂⁻_{k+1} = r̂⁺_k + Δt·v̂⁺_k + ½Δt²·( Ĉ⁺ᵀ_k·f̂_k + g )
(31)  v̂⁻_{k+1} = v̂⁺_k + Δt·( Ĉ⁺ᵀ_k·f̂_k + g )
(32)  q̂⁻_{k+1} = ζ(Δt·ω̂_k) ⊗ q̂⁺_k                        normalise afterwards
(33)  p̂⁻_{i,k+1} = p̂⁺_{i,k}
(34)  b̂⁻_{f,k+1} = b̂⁺_{f,k}
(35)  b̂⁻_{ω,k+1} = b̂⁺_{ω,k}
```

### B.7 Error dynamics, F and Q — eqs. (38)–(45)  [M5]

Continuous error dynamics (all higher-order terms dropped):

```
(38)  δṙ  = δv
(39)  δv̇  = −Cᵀ·f^×·δφ − Cᵀ·δb_f − Cᵀ·w_f
(40)  δφ̇  = −ω^×·δφ − δb_ω − w_ω
(41)  δṗ_i = Cᵀ·w_p,i
(42)  δḃ_f = w_bf
(43)  δḃ_ω = w_bω
```

Discrete error transition F_k, eq. (44). Block order [δr, δv, δφ, δp(3N), δb_f, δb_ω]; the foothold rows/columns are identity blocks and are shown collapsed:

```
         δr     δv         δφ                        δp     δb_f                δb_ω
δr    [  I    Δt·I    −½Δt²·Ĉ⁺ᵀ_k·f̂_k^×             0    −½Δt²·Ĉ⁺ᵀ_k           0       ]
δv    [  0     I      −Δt·Ĉ⁺ᵀ_k·f̂_k^×               0    −Δt·Ĉ⁺ᵀ_k             0       ]
δφ    [  0     0       Γ̂₀,ₖᵀ                        0     0                   −Γ̂₁,ₖᵀ   ]
δp    [  0     0       0                            I     0                    0       ]
δb_f  [  0     0       0                            0     I                    0       ]
δb_ω  [  0     0       0                            0     0                    I       ]
```

Discrete process noise Q_k (the matrix printed after eq. 44; symmetric — nonzero blocks only):

```
Q[δr , δr ]   =  Δt³/3 · Q_f  +  Δt⁵/20 · Q_bf
Q[δr , δv ]   =  Δt²/2 · Q_f  +  Δt⁴/8  · Q_bf
Q[δr , δb_f]  = −Δt³/6 · Ĉ⁺ᵀ_k · Q_bf
Q[δv , δv ]   =  Δt    · Q_f  +  Δt³/3  · Q_bf
Q[δv , δb_f]  = −Δt²/2 · Ĉ⁺ᵀ_k · Q_bf
Q[δφ , δφ ]   =  Δt · Q_ω  +  ( Γ̂₃,ₖ + Γ̂₃,ₖᵀ ) · Q_bω
Q[δφ , δb_ω]  = −Γ̂₂,ₖᵀ · Q_bω
Q[δp_i, δp_i] =  Δt · Ĉ⁺ᵀ_k · Q_p,i · Ĉ⁺_k              per foot; Q_p,i huge off contact
Q[δb_f, δb_f] =  Δt · Q_bf
Q[δb_ω, δb_ω] =  Δt · Q_bω
```

Mirror blocks are transposes. The paper prints the lower-left mirrors with the factors in the other order (e.g. −Δt³/6·Q_bf·Ĉ⁺_k); with isotropic Q parameters (B.1) the two orderings are identical.

```
(45)  P⁻_{k+1} = F_k · P⁺_k · F_kᵀ + Q_k
```

### B.8 Update — eqs. (46)–(53)  [M7, M8]

```
(46)  y_k = [ s̃_{1,k} − Ĉ⁻_k·( p̂⁻_{1,k} − r̂⁻_k )
              ⋮
              s̃_{m,k} − Ĉ⁻_k·( p̂⁻_{m,k} − r̂⁻_k ) ]      stack ONLY the m legs in contact
```

Linearised measurement error (47), which fixes the H blocks:

```
(47)  s_{i,k} − Ĉ⁻_k·( p̂⁻_{i,k} − r̂⁻_k )
        ≈  −Ĉ⁻_k·δr⁻_k  +  Ĉ⁻_k·δp⁻_{i,k}  +  [ Ĉ⁻_k·( p⁻_{i,k} − r⁻_k ) ]^× · δφ⁻_k
```

H_k row block for contact leg i (the matrix printed after eq. 47), zeros in every other foothold's columns:

```
         δr        δv     δφ                                  δp_i       δb_f   δb_ω
       [ −Ĉ⁻_k     0     [ Ĉ⁻_k·( p̂⁻_{i,k} − r̂⁻_k ) ]^×      +Ĉ⁻_k      0      0   ]
```

The δv and bias columns are structurally zero: the measurement corrects them only through the correlations that F builds inside P.

```
(48)  R_k = blockdiag( R_{1,k}, …, R_{m,k} )               each R_i from (25)
(49)  S_k = H_k·P⁻_k·H_kᵀ + R_k
(50)  K_k = P⁻_k·H_kᵀ·S_k⁻¹
(51)  Δx_k = K_k·y_k
(52)  P⁺_k = ( I − K_k·H_k )·P⁻_k
(53)  q̂⁺_k = ζ(Δφ_k) ⊗ q̂⁻_k
```

Implementation notes: (52) is replaced by the algebraically equivalent Joseph form `(I−KH)·P⁻·(I−KH)ᵀ + K·R·Kᵀ` for numerical robustness. All state blocks except attitude are corrected additively, `x⁺ = x⁻ + Δx`; attitude uses (53) with the Δφ slice of Δx.

### B.9 Observability constraint (OC-EKF / FEJ) — eqs. (67)–(79)  [M10]

```
(67)  x_{k+1} = F_k·x_k + w_lin,k                          linearised system
(68)  y_k    = H_k·x_k + n_lin,k
(69)  M = [ H_k ;  H_{k+1}·F_k ;  H_{k+2}·F_{k+1}·F_k ;  … ]   local observability matrix
(70)  M·U = 0          U spans the TRUE unobservable subspace:
                       3D translation of I  +  rotation about the gravity axis
```

Constraints the linearisation points must satisfy (71)–(74), and the choices that satisfy them (75)–(79):

```
(71)  r*_{k+1} = r*_k + Δt·v*_k + ½Δt²·( C*ᵀ_k·f*_{k,1} + g )
(72)  v*_{k+1} = v*_k + Δt·( C*ᵀ_k·f*_{k,2} + g )
(73)  q*_{k+1} = ζ(ω*_k) ⊗ q*_k
(74)  p*_{i,k+1} = p*_{i,k}

(75)  r*_k = r⁻_k ,   v*_k = v⁻_k ,   q*_k = q⁻_k          first-available estimates
(76)  p*_{i,k} = p⁻_{i, l_i}       l_i = timestep of foot i's most recent touchdown

(77)  f*_{k,1} = C*_k · [ ( r*_{k+1} − r*_k − Δt·v*_k ) / (½Δt²)  −  g ]
(78)  f*_{k,2} = C*_k · [ ( v*_{k+1} − v*_k ) / Δt  −  g ]
(79)  ω*_k = ζ⁻¹( q*_{k+1} ⊗ (q*_k)⁻¹ )
```

(77)–(79) are (71)–(73) solved for the measurement terms, so the prediction constraints hold exactly at the chosen linearisation points; two acceleration terms exist because the position-based and velocity-based reconstructions of f can differ.

**Reduced form implemented in the reference code:** only (76) is applied — the foothold lever arm inside H is frozen at its touchdown value. (75) holds automatically because the update linearises at the a priori state. (77)–(79) are omitted; add them only if a long-duration run shows σ_yaw decreasing (gate C6 failing).

Side note, eq. (80): rank loss recurs whenever ω ≡ 0, since then Cᵀ_{k+2} − Cᵀ_k = 0. This is the formal basis for freezing bias estimation during quasi-static gaits.

### B.10 Table I — observability rank-loss scenarios  [bias-switch logic]

Columns: angular rate ω, specific force f, velocity v, foot vectors s₁…s_N. `*` means any value.

| ω | f | v | s₁ … s_N | Rank loss |
|---|---|---|---|---|
| ω·Cg ≠ 0 | * | * | not co-linear | 0 |
| ω·Cg ≠ 0 | det O₃ ≠ 0 | * | ≥ 1 contact | 0 |
| ω·Cg = 0 | * | * | ≥ 1 contact | ≥ 1 |
| 0 | * | * | ≥ 1 contact | ≥ 2 |
| 0 | * | * | not co-linear | 2 |
| 0 | 0 | * | s₁ = … = s_N | 3 |
| 0 | −½·Cg | * | s₁ = … = s_N | 5 |

Practical reading for DOG5: a quasi-static crawl has ω ≈ 0, which sits on the rows with rank loss ≥ 2 — the IMU biases become weakly separable from the gravity direction and the velocity. Hence: estimate biases only during the startup excitation manoeuvre, then `set_bias_estimation(False)` before the gait starts.

### B.11 Quaternion convention (fixes what the paper leaves implicit)

The paper never writes out the ⊗ product. The implementation fixes it as the JPL/Shuster product, defined by the property

```
C( a ⊗ b ) = C(a) · C(b)
```

with q mapping I → B and increments composed by **left** multiplication — exactly as (11), (32), and (53) are written. This is already a unit test (M0, identity 1). If a third-party quaternion library instead satisfies `C(a ⊗ b) = C(b)·C(a)` (Hamilton convention), swap the operand order wherever (11), (32), (53) are applied. Never mix the two conventions in one codebase.

### B.12 What is deliberately NOT transcribed

- **Eqs. (54)–(66):** the robocentric coordinate change z, the transformed dynamics, the row-echelon observability matrix O with its O₁–O₃ sub-blocks, and the unobservable-subspace basis U in original coordinates. They prove the observability claims summarised in Sec. 1 of this plan; no line of the implementation evaluates them.
- **The worked-out matrix M of Sec. IV-B** (the block matrix with `#` placeholder entries): a verification device in the paper, not part of the filter.
- **Initialisation:** the paper specifies none. Use M3 of this plan.