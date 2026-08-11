"""Acceptance gates for the DOG5 closed-loop in-place trot — CONTROL_ROADMAP D1.

    Gate D1: trot in MuJoCo on the hardware-fidelity model
             (EKF in loop, not sim truth).

Run:
    python test_trot.py --self-test   # scheduler logic only, no MuJoCo needed
    python test_trot.py               # D1-1 .. D1-8, exits non-zero on failure
    python test_trot.py --sweep       # period x ds_frac x CoM-bias envelope

  D1-1  SURVIVES     completes the trot without falling
  D1-2  BOTH DIAGONALS  every leg gets genuinely airborne -- the Stage 8/9
                     failure mode (one leg lifts, the other never leaves the
                     ground) must be caught here
  D1-3  IN PLACE     net horizontal drift bounded; this is what the EKF-driven
                     foot placement buys and nothing else in the stack provides
  D1-4  TILT BOUNDED peak tilt bounded AND not growing cycle-over-cycle
  D1-5  EKF ACCURATE online tracking vs MuJoCo truth at the C7 thresholds
  D1-6  EKF HEALTHY  healthy through the trot (needs the in-tree health fix)
  D1-7  TORQUE       peak |tau| reported against the hardware trip
  D1-8  ABLATION     the estimator is load-bearing: frozen does materially worse
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

_FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _FAIL.append(name)
    return ok


# =====================================================================
# Offline gates — scheduler logic only, no MuJoCo, no physics
# =====================================================================

def test_scheduler():
    """The trot schedule and the foot-placement law, checked in isolation.

    These are the properties the whole design rests on, and they are cheap and
    deterministic, so they run before anything expensive.
    """
    import dog5_trot
    from dog5_gait import LEGS

    print("S — trot schedule and closed-loop foot placement (offline)")
    nominal = {"FL": np.array([0.360, 0.114, -0.175]),
               "FR": np.array([0.350, -0.113, -0.175]),
               "RL": np.array([-0.368, 0.113, -0.175]),
               "RR": np.array([-0.373, -0.115, -0.175])}
    p = dog5_trot.TrotParams()
    s = dog5_trot.TrotScheduler(nominal, p)

    ts = np.linspace(0.0, p.period, 400, endpoint=False)
    C = np.array([s.contacts(t) for t in ts])

    # Flight-free: a trot swings one diagonal at a time, so two feet are always
    # down. If this ever fails the gait has an aerial phase and every gate below
    # is measuring something else.
    check("S1 flight-free (>= 2 feet down at all times)",
          int(C.sum(axis=1).min()) >= 2, f"min {int(C.sum(axis=1).min())} feet")

    air = ~C
    i = {l: LEGS.index(l) for l in LEGS}
    check("S2 diagonal pairs swing together (FL+RR, FR+RL)",
          np.array_equal(air[:, i["FL"]], air[:, i["RR"]])
          and np.array_equal(air[:, i["FR"]], air[:, i["RL"]]))
    check("S3 diagonals never airborne simultaneously",
          not bool(np.any(air[:, i["FL"]] & air[:, i["FR"]])))

    duty = air[:, i["FL"]].mean()
    check("S4 duty factor is a trot (30-45% airborne)", 0.30 < duty < 0.45,
          f"{duty*100:.1f}% airborne")

    # The swing foot must leave and land with ZERO vertical speed. A sin(pi*u)
    # arc peaks its vertical speed at exactly liftoff and touchdown (0.42 m/s
    # for this lift and swing time), which slams the foot down and corrupts the
    # contact flag at the worst moment -- measured as 68.8 mm/s of EKF velocity
    # error before this was changed to a raised cosine.
    vz0 = abs(s.swing_targets(1e-9 + p.t_ds)["FL"][1][2])
    vz1 = abs(s.swing_targets(p.t_ds + p.t_sw * (1 - 1e-9))["FL"][1][2])
    check("S5 swing leaves and lands with ~zero vertical speed",
          vz0 < 0.01 and vz1 < 0.01,
          f"liftoff {vz0*1e3:.1f} mm/s, touchdown {vz1*1e3:.1f} mm/s")
    vxy = float(np.linalg.norm(
        s.swing_targets(p.t_ds + p.t_sw * (1 - 1e-9))["FL"][1][:2]))
    check("S5 swing lands with ~zero horizontal speed (no scuff)", vxy < 0.01,
          f"{vxy*1e3:.1f} mm/s")

    lift = max(s.swing_targets(t).get("FL", (nominal["FL"], 0))[0][2]
               - nominal["FL"][2] for t in ts)
    check("S6 swing arc reaches the commanded lift", abs(lift - p.lift) < 1e-4,
          f"{lift*1e3:.1f} mm")
    # Airborne through the swing: the flag is biased airborne and re-plants on
    # evidence, so a moving foot is never reported as a fixed inertial point.
    u_air = [u for t in ts
             for pair, u, _ in [s._swing_state(t)]
             if pair is not None and not s.contacts(t)[i["FL"]]]
    check("S6 flag is airborne across the swing", bool(u_air)
          and min(u_air) < 0.05 and max(u_air) > 0.95,
          f"airborne for u in [{min(u_air):.3f}, {max(u_air):.3f}]")

    # Foot placement: sign, magnitude, clamp. Sign is the one the whole design
    # rests on -- drifting forward must place the foot FORWARD to brake.
    def settled_dx(vx, vy=0.0):
        sc = dog5_trot.TrotScheduler(nominal, dog5_trot.TrotParams())
        e = {"v_body": np.array([vx, vy, 0.0]), "r": np.zeros(3), "C": np.eye(3)}
        for _ in range(100):
            sc._placement_offset(e)
        tgt = sc.swing_targets(p.t_ds + 0.99 * p.t_sw, e)["FL"][0]
        return tgt[0] - nominal["FL"][0], tgt[1] - nominal["FL"][1]

    dx0, _ = settled_dx(0.0)
    check("S7 zero velocity -> no placement offset", abs(dx0) < 1e-9,
          f"{dx0*1e3:.3f} mm")
    dxf, _ = settled_dx(0.10)
    check("S8 SIGN: forward drift places the foot forward (brakes)", dxf > 0.005,
          f"+0.10 m/s -> {dxf*1e3:+.1f} mm")
    dxb, _ = settled_dx(-0.10)
    check("S9 SIGN: backward drift places the foot back", dxb < -0.005,
          f"-0.10 m/s -> {dxb*1e3:+.1f} mm")
    _, dyl = settled_dx(0.0, 0.10)
    check("S10 lateral drift places the foot laterally", dyl > 0.005,
          f"vy +0.10 m/s -> {dyl*1e3:+.1f} mm")
    dxc, _ = settled_dx(5.0)
    check("S11 placement is clamped (abduction authority ~2 cm)",
          abs(dxc) <= p.place_clamp + 1e-9, f"{dxc*1e3:.1f} mm "
          f"<= {p.place_clamp*1e3:.0f} mm")

    # Early-touchdown detection must stay disarmed until the contact plane has
    # been latched during a double-support window.
    s2 = dog5_trot.TrotScheduler(nominal, dog5_trot.TrotParams())
    fp = [nominal[l] for l in LEGS]
    e = {"v_body": np.zeros(3), "r": np.zeros(3), "C": np.eye(3)}
    h = s2._foot_height(e, fp, "FL")
    check("S12 foot-height test disarmed before the plane is latched",
          not np.isfinite(h), f"{h}")
    s2.contacts(0.0, e, fp)          # double support -> latches the plane
    h = s2._foot_height(e, fp, "FL")
    check("S13 plane latches in double support, feet then read ~0",
          np.isfinite(h) and abs(h) < 1e-6, f"{h*1e3:.4f} mm")

    # HOLD mode must be a genuine four-foot hold.
    s3 = dog5_trot.TrotScheduler(nominal, dog5_trot.TrotParams(), mode="HOLD")
    check("S14 HOLD mode keeps all four feet planted",
          all(s3.contacts(t).all() for t in ts) and not s3.swing_targets(0.1))

    # The output contract the VMC core consumes must not drift from GaitScheduler.
    import dog5_gait
    g = dog5_gait.GaitScheduler(nominal, "DIAGONAL")
    check("S15 sched_state keys match GaitScheduler (VMC core unchanged)",
          set(s.sched_state(0.1, e, fp)) == set(g.sched_state(0.1)))


# =====================================================================
# D1 gates — full closed loop in MuJoCo
# =====================================================================

def test_d1(cycles=25):
    import trot_mujoco as H

    print(f"\nD1 — closed-loop in-place trot, {cycles} cycles, EKF in the loop")
    m = H.run(n_cycles=cycles, quiet=False)
    H._print_metrics(m)
    print()

    check("D1-1 survives: did not fall", not m["fell"],
          f"final z={m['final_z']:.3f}")
    if not m.get("n_trot"):
        check("D1-1 reached the trot phase", False, "no trot samples")
        return m

    # D1-2 is the Stage 8/9 discriminator. A run where one diagonal leg lifts
    # clean and the other stays loaded on the ground passes every other gate.
    worst_leg = min(m["clear_max"], key=lambda k: m["clear_max"][k])
    worst_clear = m["clear_max"][worst_leg]
    check("D1-2 every leg gets airborne (>= 10 mm clearance)",
          worst_clear >= 0.010,
          f"worst {worst_leg} {worst_clear*1e3:.1f} mm")
    worst_f = max(m["force_at_apex"].values())
    check("D1-2 every leg unloaded at swing apex (<= 1 N)", worst_f <= 1.0,
          f"worst {max(m['force_at_apex'], key=lambda k: m['force_at_apex'][k])}"
          f" {worst_f:.2f} N")

    check("D1-3 stays in place (net drift < 30 mm)", m["drift_net"] < 0.030,
          f"{m['drift_net']*1e3:.1f} mm net, "
          f"{m['drift_max']*1e3:.1f} mm max excursion")

    check("D1-4 peak tilt bounded (< 8 deg)", m["tilt_true_deg"].max() < 8.0,
          f"{m['tilt_true_deg'].max():.2f} deg")
    check("D1-4 tilt not diverging (last third <= first third + 2 deg)",
          m["tilt_last"] <= m["tilt_first"] + 2.0,
          f"first {m['tilt_first']:.2f} -> last {m['tilt_last']:.2f} deg")

    check("D1-5 est: z error < 3 cm", m["z_err"].max() < 0.03,
          f"{m['z_err'].max()*1e3:.1f} mm")
    check("D1-5 est: |v| error < 0.05 m/s", m["v_err"].max() < 0.05,
          f"{m['v_err'].max()*1e3:.1f} mm/s")
    check("D1-5 est: roll/pitch error < 2 deg", m["rp_err_deg"].max() < 2.0,
          f"{m['rp_err_deg'].max():.2f} deg")

    check("D1-6 estimator healthy > 99%", m["healthy_frac"] > 0.99,
          f"{m['healthy_frac']:.4f}")

    # Not a pass/fail on the hardware trip -- this is sim -- but it must be
    # reported, because it is what decides whether D2 is even attemptable.
    over = m["tau_max"] > 6.0
    check("D1-7 peak |tau| within the 9 Nm driver capability",
          m["tau_max"] < 9.0,
          f"{m['tau_max']:.2f} Nm" + ("  WARNING: over the 6.0 Nm hardware trip"
                                      if over else "  (under the 6.0 Nm trip)"))
    return m


def test_ablation(cycles=12):
    """D1-8 — the estimator must be load-bearing, not decorative.

    Mirrors V4 in test_vmc.py. With the estimate frozen the VMC is blind to the
    body state it is regulating and the foot placement has no velocity to act
    on, so a trot should degrade badly.
    """
    import trot_mujoco as H
    print(f"\nD1-8 — ablation: estimator live vs frozen ({cycles} cycles)")
    live = H.run(n_cycles=cycles)
    frozen = H.run(n_cycles=cycles, freeze_estimator=True)

    def stat(m):
        if not m.get("n_trot"):
            return dict(fell=m["fell"], drift=float("inf"), tilt=float("inf"))
        return dict(fell=m["fell"], drift=m["drift_max"],
                    tilt=float(m["tilt_true_deg"].max()))
    l, f = stat(live), stat(frozen)
    print(f"    live:   fell={l['fell']}  drift_max={l['drift']*1e3:.1f} mm  "
          f"tilt_max={l['tilt']:.2f} deg")
    print(f"    frozen: fell={f['fell']}  drift_max={f['drift']*1e3:.1f} mm  "
          f"tilt_max={f['tilt']:.2f} deg")
    check("D1-8 live run survives", not l["fell"])
    check("D1-8 estimator feedback beats frozen",
          (f["fell"] and not l["fell"]) or l["drift"] < f["drift"] * 0.6
          or l["tilt"] < f["tilt"] - 2.0,
          f"drift {l['drift']*1e3:.1f} vs {f['drift']*1e3:.1f} mm, "
          f"tilt {l['tilt']:.2f} vs {f['tilt']:.2f} deg")

    # Foot placement is a DISTURBANCE-REJECTION mechanism, so it must be graded
    # against a disturbance.  With a centred CoM and no push there is nothing for
    # it to reject: measured 1.8 mm drift with it and 1.9 mm without, which is
    # noise, and gating on that would be self-congratulation rather than a test.
    # A CoM bias is the honest case -- it is also the one that actually threatens
    # the hardware, where the CoM is not centred to 0.3 mm the way the symmetric
    # model is.
    bias = (0.008, 0.008)
    pl = stat(H.run(n_cycles=cycles, com_bias=bias))
    npl = stat(H.run(n_cycles=cycles, com_bias=bias, placement=False))
    print(f"    CoM bias {bias[0]*1e3:.0f}/{bias[1]*1e3:.0f} mm: "
          f"placement ON drift_max={pl['drift']*1e3:.1f} mm | "
          f"OFF drift_max={npl['drift']*1e3:.1f} mm")
    check("D1-8 closed-loop foot placement rejects a CoM bias",
          pl["drift"] < npl["drift"] * 0.95,
          f"{pl['drift']*1e3:.1f} mm with vs {npl['drift']*1e3:.1f} mm without "
          f"({(1 - pl['drift']/max(npl['drift'], 1e-9))*100:.0f}% better)")


def run_sweep(cycles=15):
    """Map the feasible envelope and the tip-vs-T_ss^2 relationship."""
    import trot_mujoco as H
    import dog5_trot

    print("\nSWEEP — period x ds_frac, then CoM bias at the default timing")
    rows = []
    print(f"{'period':>7} {'ds':>5} {'T_ss':>6} {'bias':>10} {'fell':>5} "
          f"{'tilt':>6} {'drift':>7} {'minclr':>7} {'tau':>6} {'sat':>4} "
          f"{'health':>7}")
    # ds_frac 0.18 is included deliberately: it is BELOW the torque-saturation
    # boundary, so the sweep shows the failure mode rather than only the
    # comfortable region.
    cases = [(pp, dd, (0.0, 0.0)) for pp in (0.30, 0.40, 0.50)
             for dd in (0.18, 0.34, 0.46)]
    cases += [(0.40, 0.34, b) for b in ((0.005, 0.0), (-0.005, 0.0),
                                        (0.0, 0.005), (0.0, -0.005),
                                        (0.008, 0.008))]
    for per, ds, bias in cases:
        p = dog5_trot.TrotParams()
        p.period, p.ds_frac = per, ds
        m = H.run(n_cycles=cycles, params=p, com_bias=bias)
        if m.get("n_trot"):
            tilt = m["tilt_true_deg"].max()
            drift = m["drift_max"] * 1e3
            minclr = min(m["clear_max"].values()) * 1e3
            health = m["healthy_frac"]
            tau = m["tau_max"]
        else:
            tilt = drift = minclr = tau = float("nan")
            health = 0.0
        sat = "SAT" if tau > 7.9 else ""
        rows.append((per, ds, p.t_sw, bias, m["fell"], tilt, drift, minclr,
                     health, tau, m.get("com_line_d_mean")))
        print(f"{per:7.2f} {ds:5.2f} {p.t_sw:6.3f} "
              f"{str(tuple(round(b*1e3) for b in bias)):>10} "
              f"{str(m['fell']):>5} {tilt:6.2f} {drift:7.1f} {minclr:7.1f} "
              f"{tau:6.2f} {sat:>4} {health:7.4f}")

    ok = [r for r in rows if not r[4] and r[7] >= 10.0]
    print(f"\n  {len(ok)}/{len(rows)} cases upright with every leg clearing "
          f"10 mm")
    # The tip relation: tilt should scale ~T_ss^2 at fixed CoM offset.
    base = [r for r in rows if r[3] == (0.0, 0.0) and not r[4]]
    if len(base) >= 2:
        print("  tilt vs single-support time (zero CoM bias):")
        for r in sorted(base, key=lambda r: r[2]):
            print(f"    T_ss={r[2]:.3f}s  tilt={r[5]:.2f} deg  "
                  f"tilt/T_ss^2={r[5]/r[2]**2:8.1f} deg/s^2")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="scheduler gates only; no MuJoCo required")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--cycles", type=int, default=25)
    args = ap.parse_args()

    print("=" * 64)
    print("DOG5 closed-loop in-place trot — Gate D1 acceptance")
    print("=" * 64)
    test_scheduler()
    if args.self_test:
        print("=" * 64)
        if _FAIL:
            print(f"FAILED ({len(_FAIL)}): " + ", ".join(_FAIL))
            return 1
        print("SCHEDULER GATES PASS (run without --self-test for D1)")
        return 0

    test_d1(cycles=args.cycles)
    test_ablation()
    if args.sweep:
        run_sweep()

    print("=" * 64)
    if _FAIL:
        print(f"FAILED ({len(_FAIL)}): " + ", ".join(_FAIL))
        return 1
    print("ALL D1 GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
