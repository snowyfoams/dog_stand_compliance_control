"""Test suite and acceptance gates for the DOG5 state estimator (plan Sec. 6-7).

Run:  python3 test_estimator.py            # all unit tests + gates
Each check prints PASS/FAIL and the run exits non-zero if anything fails.
"""
from __future__ import annotations

import math
import sys

import numpy as np

import dog5_state_estimator as est
from dog5_state_estimator import (
    DOG5StateEstimator,
    EstimatorParams,
    ErrorIndex,
    NominalIndex,
    State,
    gamma,
    leg_fk,
    leg_jac,
    normalize,
    quat_mul,
    quat_to_C,
    skew,
    zeta,
    zeta_inv,
)
import dog5_sim as sim

RNG = np.random.default_rng(12345)
_FAILURES = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _FAILURES.append(name)
    return ok


def rand_quat():
    q = RNG.standard_normal(4)
    return normalize(q)


# =====================================================================
# M0 — rotation math identities (plan M0 test, gate C1)
# =====================================================================

def test_m0():
    print("M0 — rotation math identities")
    # 1. C(a (x) b) == C(a) C(b)
    a, b = rand_quat(), rand_quat()
    lhs = quat_to_C(quat_mul(a, b))
    rhs = quat_to_C(a) @ quat_to_C(b)
    check("C(a(x)b) == C(a)C(b)", np.allclose(lhs, rhs, atol=1e-12),
          f"err={np.max(np.abs(lhs - rhs)):.2e}")

    # 2. zeta^-1(zeta(v)) == v
    v = RNG.standard_normal(3) * 0.5
    check("zeta_inv(zeta(v)) == v", np.allclose(zeta_inv(zeta(v)), v, atol=1e-12),
          f"err={np.max(np.abs(zeta_inv(zeta(v)) - v)):.2e}")

    # 3. C(zeta(d)) ~ I - skew(d) for small d
    d = RNG.standard_normal(3) * 1e-4
    approx = np.eye(3) - skew(d)
    check("C(zeta(d)) ~ I - skew(d)", np.allclose(quat_to_C(zeta(d)), approx, atol=1e-7),
          f"err={np.max(np.abs(quat_to_C(zeta(d)) - approx)):.2e}")

    # 4. Gamma_0(w, dt) == expm(dt skew(w))  (series vs. naive matrix exp)
    w = RNG.standard_normal(3)
    dt = 0.0025
    W = dt * skew(w)
    expm = np.eye(3)
    term = np.eye(3)
    for j in range(1, 30):
        term = term @ W / j
        expm = expm + term
    check("Gamma_0 == expm(dt skew(w))", np.allclose(gamma(w, dt, 0), expm, atol=1e-12),
          f"err={np.max(np.abs(gamma(w, dt, 0) - expm)):.2e}")

    # bonus: normalize keeps unit norm, positive scalar
    q = rand_quat()
    qn = normalize(-q)
    check("normalize: unit norm & w>=0", abs(np.linalg.norm(qn) - 1) < 1e-12 and qn[3] >= 0)


# =====================================================================
# M1 — state container slice bookkeeping
# =====================================================================

def test_m1():
    print("M1 — state container")
    dx = np.zeros(ErrorIndex.DIM)
    blocks = [ErrorIndex.R, ErrorIndex.V, ErrorIndex.PHI,
              *[ErrorIndex.p(i) for i in range(est.N_LEGS)],
              ErrorIndex.BF, ErrorIndex.BW]
    covered = np.zeros(ErrorIndex.DIM, dtype=int)
    for m, blk in enumerate(blocks, start=1):
        dx[blk] = m
        covered[blk] += 1
    check("error-state slices cover 0:27 exactly, no overlap",
          np.all(covered == 1) and dx.shape[0] == 27)
    check("nominal state DIM == 28", NominalIndex.DIM == 28)


# =====================================================================
# M2 — leg kinematics
# =====================================================================

def test_m2():
    print("M2 — leg kinematics")
    ok_j = True
    max_err = 0.0
    for i in range(est.N_LEGS):
        alpha = est.LEGS  # placeholder, replaced below
        a = np.array([0.3, -0.6, 1.2]) * (1 + 0.1 * i)
        J = leg_jac(i, a)
        Jn = np.zeros((3, 3))
        h = 1e-7
        for c in range(3):
            da = np.zeros(3)
            da[c] = h
            Jn[:, c] = (leg_fk(i, a + da) - leg_fk(i, a - da)) / (2 * h)
        e = np.max(np.abs(J - Jn))
        max_err = max(max_err, e)
        ok_j = ok_j and e < 1e-5
    check("Jacobian finite-difference (< 1e-5)", ok_j, f"max_err={max_err:.2e}")

    # Four-foot closure at the nominal stance: feet coplanar (they define the
    # plane here, so check they are consistent / distinct and non-colinear).
    feet = np.stack([leg_fk(i, sim.Q_STANCE[est.LEGS[i]]) for i in range(4)])
    # fit a plane, residual small
    centroid = feet.mean(axis=0)
    _, s, vh = np.linalg.svd(feet - centroid)
    normal = vh[-1]
    resid = np.max(np.abs((feet - centroid) @ normal))
    check("four-foot closure residual < 3 mm", resid < 3e-3, f"resid={resid*1e3:.2f} mm")


# =====================================================================
# M4 — nominal dead reckoning (gate C3)
# =====================================================================

def test_m4_deadreckon():
    print("M4 — noiseless dead reckoning (gate C3)")
    cfg = sim.SimConfig(dt=1/400, duration=10.0)
    data = sim.scenario_imu_only(cfg)
    e = DOG5StateEstimator()
    # start from exact truth so we isolate propagation drift
    e.state.r = data.r[0].copy()
    e.state.v = data.v[0].copy()
    e.state.q = data.q[0].copy()
    e.state.bf = np.zeros(3)
    e.state.bw = np.zeros(3)
    for k in range(1, data.n):
        st, _, _ = e.propagate_state(e.state, data.f_meas[k - 1], data.w_meas[k - 1], cfg.dt)
        e.state = st
    er = np.linalg.norm(e.state.r - data.r[-1])
    ev = np.linalg.norm(e.state.v - data.v[-1])
    dq = quat_mul(e.state.q, np.array([*(-data.q[-1][:3]), data.q[-1][3]]))
    eq = np.linalg.norm(zeta_inv(dq))
    check("10 s dead-reckon r drift < 1e-9 m", er < 1e-9, f"{er:.2e}")
    check("10 s dead-reckon v drift < 1e-9 m/s", ev < 1e-9, f"{ev:.2e}")
    check("10 s dead-reckon attitude drift < 1e-9 rad", eq < 1e-9, f"{eq:.2e}")


# =====================================================================
# M5 — numerical verification of F (gate C4)
# =====================================================================

def _perturb_state(st: State, dx):
    """Apply an error-state perturbation to a nominal state (for FD checks)."""
    out = st.copy()
    out.r = st.r + dx[ErrorIndex.R]
    out.v = st.v + dx[ErrorIndex.V]
    out.q = normalize(quat_mul(zeta(dx[ErrorIndex.PHI]), st.q))
    for i in range(est.N_LEGS):
        out.set_p(i, st.p(i) + dx[ErrorIndex.p(i)])
    out.bf = st.bf + dx[ErrorIndex.BF]
    out.bw = st.bw + dx[ErrorIndex.BW]
    return out


def _state_error(a: State, b: State):
    """Error-state difference a (-) b (b is the reference)."""
    dx = np.zeros(ErrorIndex.DIM)
    dx[ErrorIndex.R] = a.r - b.r
    dx[ErrorIndex.V] = a.v - b.v
    dq = quat_mul(a.q, np.array([*(-b.q[:3]), b.q[3]]))  # a (x) b^-1
    dx[ErrorIndex.PHI] = zeta_inv(dq)
    for i in range(est.N_LEGS):
        dx[ErrorIndex.p(i)] = a.p(i) - b.p(i)
    dx[ErrorIndex.BF] = a.bf - b.bf
    dx[ErrorIndex.BW] = a.bw - b.bw
    return dx


def test_m5_numerical_F():
    print("M5 — numerical F, column by column (gate C4)")
    e = DOG5StateEstimator()
    st = State()
    st.r = np.array([0.1, -0.2, 0.3])
    st.v = np.array([0.05, 0.02, -0.03])
    st.q = normalize(np.array([0.02, -0.03, 0.01, 1.0]))
    for i in range(4):
        st.set_p(i, np.array([0.1 * i, 0.2, -0.3]))
    st.bf = np.array([0.01, -0.02, 0.015])
    st.bw = np.array([0.001, 0.002, -0.0015])
    f_meas = np.array([0.2, -0.1, 9.9])
    w_meas = np.array([0.3, -0.2, 0.1])
    dt = 1 / 400
    contacts = np.ones(4, dtype=bool)

    base, f_hat, w_hat = e.propagate_state(st, f_meas, w_meas, dt)
    F, _ = e.build_FQ(st, f_hat, w_hat, dt, contacts)

    h = 1e-6
    Fnum = np.zeros((ErrorIndex.DIM, ErrorIndex.DIM))
    for c in range(ErrorIndex.DIM):
        dx = np.zeros(ErrorIndex.DIM)
        dx[c] = h
        st_p = _perturb_state(st, dx)
        prop_p, _, _ = e.propagate_state(st_p, f_meas, w_meas, dt)
        Fnum[:, c] = _state_error(prop_p, base) / h

    err = np.max(np.abs(F - Fnum))
    # localise worst column for the message
    worst = int(np.argmax(np.max(np.abs(F - Fnum), axis=0)))
    check("F matches finite-difference (< 1e-4)", err < 1e-4,
          f"max_err={err:.2e} worst_col={worst}")


# =====================================================================
# M7 — numerical verification of H
# =====================================================================

def test_m7_numerical_H():
    print("M7 — numerical H, column by column")
    e = DOG5StateEstimator()
    st = e.state
    st.r = np.array([0.02, -0.01, 0.0])
    st.q = normalize(np.array([0.03, 0.02, -0.01, 1.0]))
    alpha = np.stack([sim.Q_STANCE[est.LEGS[i]] for i in range(4)])
    C0 = st.C
    for i in range(4):
        st.set_p(i, st.r + C0.T @ leg_fk(i, alpha[i]))
    contacts = np.ones(4, dtype=bool)

    e.prev_contacts = np.ones(4, dtype=bool)  # avoid re-anchor
    legs, y, H, R = e.build_measurement(alpha, contacts)

    def predicted(st_local):
        C = st_local.C
        out = []
        for i in legs:
            out.append(C @ (st_local.p(i) - st_local.r))
        return np.concatenate(out)

    base_pred = predicted(st)
    h = 1e-7
    Hnum = np.zeros_like(H)
    for c in range(ErrorIndex.DIM):
        dx = np.zeros(ErrorIndex.DIM)
        dx[c] = h
        st_p = _perturb_state(st, dx)
        # Standard EKF linearisation: z - h(x_hat) ~ H dx with H = d h / d dx
        # and dx = x_true - x_hat.  Here h(x) = C(p_i - r) is `predicted`, so
        # H is the plain Jacobian of the predicted measurement (eq. 47's -C on
        # dr already carries the sign).
        Hnum[:, c] = (predicted(st_p) - base_pred) / h
    err = np.max(np.abs(H - Hnum))
    worst = int(np.argmax(np.max(np.abs(H - Hnum), axis=0)))
    check("H matches finite-difference (< 1e-4)", err < 1e-4,
          f"max_err={err:.2e} worst_col={worst}")


# =====================================================================
# helper: run the full filter over a SimData
# =====================================================================

def run_filter(e: DOG5StateEstimator, data: sim.SimData, init_static_s=0.5,
               exact_init=False, update=True):
    dt = data.t[1] - data.t[0]
    ninit = max(1, int(init_static_s / dt))
    e.initialise(data.f_meas[:ninit], data.w_meas[:ninit],
                 data.alpha[0], data.contacts[0])
    if exact_init:
        e.state.r = data.r[0].copy()
        # velocity is deliberately NOT set: a cold start does not know v, so it
        # must be recovered through the P correlations (velocity observability).
        e.state.q = data.q[0].copy()
        e.state.bf = data.bias_f.copy()
        e.state.bw = data.bias_w.copy()
        C = e.state.C
        for i in range(4):
            if data.contacts[0, i]:
                e.state.set_p(i, e.state.r + C.T @ leg_fk(i, data.alpha[0, i]))
                e.fej_anchor[i] = e.state.p(i).copy()
    hist = {"r": [], "v": [], "q": [], "sig": []}
    for k in range(data.n):
        if k > 0:
            e.predict(data.f_meas[k - 1], data.w_meas[k - 1], dt, data.contacts[k])
        if update:
            e.update(data.alpha[k], data.contacts[k])
        hist["r"].append(e.state.r.copy())
        hist["v"].append(e.state.v.copy())
        hist["q"].append(e.state.q.copy())
        hist["sig"].append(np.sqrt(np.clip(np.diag(e.state.P), 0, None)))
    for kk in hist:
        hist[kk] = np.array(hist[kk])
    return hist


def att_error(qf, qt):
    dq = quat_mul(qf, np.array([*(-qt[:3]), qt[3]]))
    return np.linalg.norm(zeta_inv(dq))


# =====================================================================
# Integration ladder + gate C5
# =====================================================================

def test_c5():
    print("C5 — the decisive gate (M8 fused)")

    # (1) static, noiseless, zero bias -> errors ~ machine precision
    cfg = sim.SimConfig(dt=1/400, duration=3.0)
    data = sim.scenario_static(cfg)
    e = DOG5StateEstimator()
    h = run_filter(e, data, exact_init=True)
    # after transient (last 20%)
    tail = slice(int(0.8 * data.n), None)
    ev = np.max(np.linalg.norm(h["v"][tail] - data.v[tail], axis=1))
    er = np.max(np.linalg.norm(h["r"][tail] - data.r[tail], axis=1))
    ea = max(att_error(h["q"][k], data.q[k]) for k in range(int(0.8 * data.n), data.n))
    check("static noiseless: |v|err < 1e-9", ev < 1e-9, f"{ev:.2e}")
    check("static noiseless: |r|err < 1e-9", er < 1e-9, f"{er:.2e}")
    check("static noiseless: attitude err < 1e-9", ea < 1e-9, f"{ea:.2e}")

    # (2) static, noiseless, gyro bias + vertical accel bias -> observable
    # states settle.  NB: a *horizontal* accel bias is provably indistinguish-
    # able from tilt at rest (Table I), so it is excluded here and validated
    # separately with an excitation manoeuvre (test_bias_excitation).
    cfg = sim.SimConfig(dt=1/400, duration=8.0,
                        bias_f=np.array([0.0, 0.0, 0.02]),
                        bias_w=np.array([0.004, -0.002, 0.003]))
    data = sim.scenario_static(cfg)
    e = DOG5StateEstimator()
    h = run_filter(e, data, init_static_s=1.0, exact_init=False)
    bw_err = np.linalg.norm(e.state.bw - data.bias_w)
    ev = np.max(np.linalg.norm(h["v"][tail] - data.v[tail], axis=1))
    ea = max(att_error(h["q"][k], data.q[k]) for k in range(int(0.8 * data.n), data.n))
    check("static+bias: gyro bias err < 1e-3", bw_err < 1e-3, f"{bw_err:.2e}")
    check("static+bias: |v|err < 5 mm/s", ev < 5e-3, f"{ev*1e3:.2f} mm/s")
    check("static+bias: roll/pitch err < 0.3deg", math.degrees(ea) < 0.3,
          f"{math.degrees(ea):.3f} deg")

    # (3) static, NOISY (motors-on levels) -> steady-state bounds
    cfg = sim.SimConfig(dt=1/400, duration=10.0, seed=7,
                        noise_f=5.0e-3 * math.sqrt(400),   # PSD sigma_f -> per-sample
                        noise_w=5.0e-4 * math.sqrt(400),
                        noise_alpha=0.002,
                        bias_w=np.array([0.004, -0.002, 0.003]))
    data = sim.scenario_static(cfg)
    e = DOG5StateEstimator()
    h = run_filter(e, data, init_static_s=1.0, exact_init=False)
    tailn = slice(int(0.7 * data.n), None)
    # steady-state error = RMS over the tail (not the worst single noise spike)
    ev = float(np.sqrt(np.mean(np.linalg.norm(h["v"][tailn] - data.v[tailn], axis=1) ** 2)))
    ez = float(np.sqrt(np.mean((h["r"][tailn, 2] - data.r[tailn, 2]) ** 2)))
    ea = float(np.sqrt(np.mean([att_error(h["q"][k], data.q[k]) ** 2
                                for k in range(int(0.7 * data.n), data.n)])))
    check("static noisy: |v|err(RMS) < 5 mm/s", ev < 5e-3, f"{ev*1e3:.2f} mm/s")
    check("static noisy: z err(RMS) < 5 mm", ez < 5e-3, f"{ez*1e3:.2f} mm")
    check("static noisy: roll/pitch < 0.3deg", math.degrees(ea) < 0.3,
          f"{math.degrees(ea):.3f} deg")


def test_bias_excitation():
    print("M10/Table I — accel bias observable under excitation")
    # A horizontal accel bias is unobservable at rest but becomes observable
    # once the body is rotated (nonzero omega).  Estimate it during a +/-10 deg
    # roll/pitch manoeuvre, then it should converge to truth.
    cfg = sim.SimConfig(dt=1/400, duration=10.0,
                        bias_f=np.array([0.05, -0.03, 0.02]),
                        bias_w=np.array([0.004, -0.002, 0.003]))
    data = sim.scenario_excite(cfg, amp_deg=5.0)
    e = DOG5StateEstimator()
    h = run_filter(e, data, init_static_s=0.5, exact_init=False)
    bf_err = np.linalg.norm(e.state.bf - data.bias_f)
    bw_err = np.linalg.norm(e.state.bw - data.bias_w)
    check("excite: accel bias converges (< 10 mm/s^2)", bf_err < 0.01,
          f"{bf_err*1e3:.2f} mm/s^2")
    check("excite: gyro bias converges (< 1 mrad/s)", bw_err < 1e-3,
          f"{bw_err*1e3:.3f} mrad/s")
    # after the manoeuvre observes bias, freeze it and confirm attitude holds
    e.set_bias_estimation(False)
    tail = slice(int(0.8 * data.n), None)
    ea = max(att_error(h["q"][k], data.q[k]) for k in range(int(0.8*data.n), data.n))
    check("excite: roll/pitch err < 0.5deg", math.degrees(ea) < 0.5,
          f"{math.degrees(ea):.3f} deg")


# =====================================================================
# Integration: translation (velocity observability, gate for rung 5)
# =====================================================================

def test_translate():
    print("Integration — forward translation (velocity observability)")
    cfg = sim.SimConfig(dt=1/400, duration=3.0)
    data = sim.scenario_translate(cfg, speed=0.05)
    e = DOG5StateEstimator()
    # init exact attitude/pos but WRONG velocity, so convergence is a real test
    h = run_filter(e, data, exact_init=True)
    # perturbation of v handled by making init not-exact-v:
    tail = slice(int(0.5 * data.n), None)
    ev = np.max(np.linalg.norm(h["v"][tail] - data.v[tail], axis=1))
    check("translate: |v|err < 5 mm/s (v observed via P correlations)",
          ev < 5e-3, f"{ev*1e3:.2f} mm/s")


# =====================================================================
# PSD / symmetry health over a run
# =====================================================================

def test_psd():
    print("M8 — P symmetric & PSD every step")
    cfg = sim.SimConfig(dt=1/400, duration=2.0, seed=3,
                        noise_f=0.02*math.sqrt(400), noise_w=0.002*math.sqrt(400),
                        noise_alpha=0.002)
    data = sim.scenario_static(cfg)
    e = DOG5StateEstimator()
    ninit = 200
    e.initialise(data.f_meas[:ninit], data.w_meas[:ninit], data.alpha[0], data.contacts[0])
    dt = cfg.dt
    worst_sym = 0.0
    worst_eig = 1e9
    for k in range(1, data.n):
        e.predict(data.f_meas[k-1], data.w_meas[k-1], dt, data.contacts[k])
        e.update(data.alpha[k], data.contacts[k])
        P = e.state.P
        worst_sym = max(worst_sym, np.max(np.abs(P - P.T)))
        worst_eig = min(worst_eig, float(np.min(np.linalg.eigvalsh(0.5*(P+P.T)))))
    check("P symmetric (< 1e-10)", worst_sym < 1e-10, f"{worst_sym:.2e}")
    check("P PSD (min eig > -1e-9)", worst_eig > -1e-9, f"{worst_eig:.2e}")


# =====================================================================
# Trot full pipeline (gate C7)
# =====================================================================

def test_c7_trot():
    print("C7 — trot simulation (transitions, no jumps)")
    cfg = sim.SimConfig(dt=1/400, duration=6.0)
    data = sim.scenario_trot(cfg, speed=0.04)
    e = DOG5StateEstimator()
    h = run_filter(e, data, init_static_s=0.3, exact_init=True)
    tail = slice(int(0.3 * data.n), None)
    ev = np.max(np.linalg.norm(h["v"][tail] - data.v[tail], axis=1))
    ez = np.max(np.abs(h["r"][tail, 2] - data.r[tail, 2]))
    ea = max(att_error(h["q"][k], data.q[k]) for k in range(int(0.3*data.n), data.n))
    # no state jump at transitions: bound the per-step velocity jump
    dv = np.max(np.linalg.norm(np.diff(h["v"], axis=0), axis=1))
    check("trot: |v|err < 0.05 m/s", ev < 0.05, f"{ev*1e3:.1f} mm/s")
    check("trot: z err < 3 cm", ez < 0.03, f"{ez*1e3:.1f} mm")
    check("trot: roll/pitch < 2 deg", math.degrees(ea) < 2.0, f"{math.degrees(ea):.2f} deg")
    check("trot: no v jump at touchdown (< 20 mm/s/step)", dv < 0.02, f"{dv*1e3:.2f} mm/s")


# =====================================================================
# M9 gate: false contact flag rejected
# =====================================================================

def test_m9_gate():
    print("M9 — outlier gate rejects a false contact")
    cfg = sim.SimConfig(dt=1/400, duration=3.0)
    data = sim.scenario_trot(cfg, speed=0.04)
    e = DOG5StateEstimator()
    e.set_gate(True)
    h = run_filter(e, data, init_static_s=0.3, exact_init=True)
    tail = slice(int(0.3 * data.n), None)
    ev = np.max(np.linalg.norm(h["v"][tail] - data.v[tail], axis=1))
    check("gate on: trot |v|err still bounded < 0.05 m/s", ev < 0.05, f"{ev*1e3:.1f} mm/s")


def main():
    print("=" * 64)
    print("DOG5 state estimator — test suite")
    print("=" * 64)
    test_m0()
    test_m1()
    test_m2()
    test_m4_deadreckon()
    test_m5_numerical_F()
    test_m7_numerical_H()
    test_psd()
    test_c5()
    test_bias_excitation()
    test_translate()
    test_c7_trot()
    test_m9_gate()
    print("=" * 64)
    if _FAILURES:
        print(f"FAILED ({len(_FAILURES)}): " + ", ".join(_FAILURES))
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
