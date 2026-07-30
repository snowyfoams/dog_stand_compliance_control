#!/usr/bin/env python3
"""MuJoCo gate for stand_hier_hw: rise + the MANDATORY leveling-sign check.

Drives the REAL runner classes (StandHierPlan / StandHierSequence /
LevelingTrim imported from stand_hier_hw) against a 0xA4 plant model:
per joint, a reference slews toward q_cmd at the output-side speed cap,
then a stiff PD + gravity bias stands in for the driver's kHz loop.

Scenarios (each prints PASS/FAIL; exit 1 on any FAIL):
    rise         gain 0, sweep at 20.8 Hz AND 250 Hz: reaches HOLD at
                 height, level, feet do not slide
    level-sign   SHIM test, the exact analogue of the hardware runbook:
                 enlarge two foot spheres (+10 mm front / +8 mm right) so
                 that side stands on a shim.  gain 0 -> baseline tilt
                 (~0.8-2 deg); gain 0.25 -> tilt must fall below 30% of
                 baseline with the right offset signs (shimmed side
                 SHORTENS); gain -0.25 (deliberate wrong sign) -> must
                 visibly diverge/saturate, proving the gate can catch a
                 sign error.  (A pure trunk moment is useless here: a
                 stiff position-servo robot tilts < 0.05 deg under any
                 sane moment -- measured 0.01 deg at 1.5 N*m.)
    level-ekf    level-sign's front-shim case with the EKF as attitude
                 source (contacts all-False during the rise drag, per the
                 validated schedule); also |EKF - truth| < 1 deg

    .venv/bin/python stand_hier_sim.py            # all scenarios
    .venv/bin/python stand_hier_sim.py rise level-sign
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import mujoco

_HERE = Path(__file__).resolve().parent
_DESC = _HERE.parent / "dog5_description"
_EST = _HERE.parent / "state_estimator"
for _p in (str(_HERE), str(_DESC), str(_EST)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stand_hier_hw as H                                # noqa: E402
from vmc_mujoco import VMCSim                            # noqa: E402
from dog5_state_estimator import DOG5StateEstimator, quat_to_C  # noqa: E402
from ekf_runtime import _rp                              # noqa: E402

XML = str(_DESC / "dog5.xml")
LEGS = H.LEGS
DT = 0.002                       # dog5.xml timestep
KP_POS = 150.0                   # 0xA4 stand-in: stiff PD on a slewed ref
KD_POS = 3.0
T_HOLD_S = 15.0                  # hold window for the leveling scenarios


class PositionPlant:
    """Driver-side 0xA4 model: q_ref slews to q_cmd at the speed cap, a
    stiff PD tracks q_ref.  Gravity bias included (the driver's integral
    action holds load without droop)."""

    def __init__(self, sim, q0):
        self.sim = sim
        self.q_ref = np.asarray(q0, dtype=float).copy()

    def step_ctrl(self, q_cmd, cap_motor_dps):
        cap = math.radians(cap_motor_dps / 10.0) * DT   # output rad per step
        delta = np.clip(np.asarray(q_cmd) - self.q_ref, -cap, cap)
        self.q_ref += delta
        for i, leg in enumerate(LEGS):
            sl = slice(3 * i, 3 * i + 3)
            qa, da = self.sim.qadr[leg], self.sim.dadr[leg]
            self.sim.d.ctrl[self.sim.actadr[leg]] = (
                KP_POS * (self.q_ref[sl] - self.sim.d.qpos[qa])
                + KD_POS * (-self.sim.d.qvel[da])
                + self.sim.d.qfrc_bias[da])


class SimEkf:
    """Minimal EKF host for the sim: init on WAIT_CROUCH statics, predict
    every step, update at the sweep cadence (matches the hardware split)."""

    def __init__(self):
        self.est = DOG5StateEstimator()
        self.init_f, self.init_w = [], []
        self.ready = False
        self.out = None

    def step(self, sim, stage, contacts):
        f, w = sim.imu()
        if not self.ready:
            if stage == "WAIT_CROUCH":
                self.init_f.append(f)
                self.init_w.append(w)
                if len(self.init_f) >= 30:
                    self.est.initialise(np.array(self.init_f),
                                        np.array(self.init_w),
                                        sim.alpha44(), np.ones(4, bool))
                    self.ready = True
            return
        self.est.predict(f, w, DT, contacts)

    def update(self, sim, contacts):
        if not self.ready:
            return
        self.est.update(sim.alpha44(), contacts)
        out = self.est.outputs()
        out["C"] = quat_to_C(out["q"])
        self.out = out


def run_case(gain, att="truth", shim=None, sweep_dt=0.048,
             stand_height=0.20, t_rise=8.0, level_clamp=H.LEVEL_CLAMP_M):
    """Run one full CROUCH->...->HOLD case.  `shim` = (legs, dz_m) enlarges
    those feet's sphere radii by dz -- that side stands dz higher, the
    exact analogue of the hardware runbook's shim-under-feet check."""
    model = mujoco.MjModel.from_xml_path(XML)
    foot_gid = {leg: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                       f"foot_{leg}") for leg in LEGS}
    if shim is not None:
        for leg in shim[0]:
            gid = foot_gid[leg]
            model.geom_size[gid][0] += shim[1]
            # collision pruning uses PRECOMPILED bounds -- update every one,
            # or the grown sphere is silently culled and the "shim" vanishes
            # (verified: rbound alone is not enough, geom_aabb also prunes)
            model.geom_rbound[gid] += shim[1]
            model.geom_aabb[gid][3:6] = model.geom_size[gid][0]
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    sim = VMCSim(model, data)

    plan = H.StandHierPlan(stand_height)
    crouch_h = -float(np.mean([plan.crouch_foot[leg][2] for leg in LEGS]))
    for i, leg in enumerate(LEGS):
        data.qpos[sim.qadr[leg]] = H.Q_RECORDED_CROUCH[3 * i:3 * i + 3]
    data.qpos[0:3] = [0.0, 0.0, crouch_h + 0.02 + 0.005]  # onto the feet
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    trim = H.LevelingTrim(plan.anchors_xy, gain_per_s=abs(gain),
                          clamp_m=level_clamp)
    if gain < 0:                       # deliberate wrong-sign variant
        for leg in LEGS:
            trim.anchors[leg] = -trim.anchors[leg]
    seq = H.StandHierSequence(0.0, plan, trim, t_rise=t_rise)
    plant = PositionPlant(sim, H.Q_RECORDED_CROUCH)
    ekf = SimEkf() if att == "ekf" else None
    caps = {"crouch": 100.0, "rise": 100.0, "stream": 250.0}

    hold_t0 = None
    hold_foot_xy = None
    metrics = {"tilt_deg": [], "z": [], "drift_m": 0.0, "off": None,
               "ekf_err_deg": 0.0, "stages": [seq.stage], "fault": None}
    next_sweep = 0.0
    t_end = 60.0
    enter_sent = False

    while data.time < t_end:
        t = data.time
        stage = seq.stage
        contacts = seq.contacts
        if ekf is not None:
            ekf.step(sim, stage, contacts)

        if t >= next_sweep:
            next_sweep += sweep_dt
            q, qd = sim.joint_state()

            # leveling attitude source
            if seq.stage in ("PARK", "PARK_SETTLE", "PARKED"):
                trim.decay(t)
            else:
                roll = pitch = None
                active = False
                if abs(gain) > 0 and seq.stage in ("HOLD",) \
                        and not seq.aborted:
                    if att == "truth":
                        roll, pitch = sim.truth_rp()
                        active = True
                    elif ekf is not None and ekf.ready \
                            and ekf.out is not None \
                            and ekf.out.get("healthy", False):
                        roll, pitch = _rp(ekf.out["C"])
                        active = True
                trim.update(t, roll, pitch, active)

            event = seq.sweep(t, q, qd, True)
            if event:
                metrics["stages"].append(seq.stage)
            if ekf is not None:
                ekf.update(sim, seq.contacts)
            for k in range(4):
                seq.refine_leg(k)
            if seq.fault and metrics["fault"] is None:
                metrics["fault"] = seq.fault

            # auto-operator: ENTER once WAIT_CROUCH has dwelled
            if (not enter_sent and seq.stage == "WAIT_CROUCH"
                    and seq.wait_since is not None
                    and t - seq.wait_since >= 0.6
                    and (ekf is None or ekf.ready)):
                ok, _ = seq.request_next(t, healthy=True, ekf_ok=True)
                enter_sent = ok

            if seq.stage == "HOLD" and hold_t0 is None:
                hold_t0 = t
                hold_foot_xy = {leg: data.geom_xpos[foot_gid[leg]][:2].copy()
                                for leg in LEGS}

        plant.step_ctrl(seq.q_cmd, seq.speed_cap_dps(caps))
        mujoco.mj_step(model, data)

        if hold_t0 is not None:
            r, p = sim.truth_rp()
            metrics["tilt_deg"].append(
                (t - hold_t0, math.degrees(r), math.degrees(p)))
            metrics["z"].append(sim.trunk_z())
            drift = max(float(np.linalg.norm(
                data.geom_xpos[foot_gid[leg]][:2] - hold_foot_xy[leg]))
                for leg in LEGS)
            metrics["drift_m"] = max(metrics["drift_m"], drift)
            if ekf is not None and ekf.out is not None:
                er, ep = _rp(ekf.out["C"])
                metrics["ekf_err_deg"] = max(
                    metrics["ekf_err_deg"],
                    abs(math.degrees(er) - math.degrees(r)),
                    abs(math.degrees(ep) - math.degrees(p)))
            if t - hold_t0 >= T_HOLD_S:
                break

    metrics["off"] = dict(trim.offsets)
    metrics["crouch_h"] = crouch_h
    metrics["reached_hold"] = hold_t0 is not None
    return metrics


def _tail_tilt(m, axis_idx, window_s=3.0):
    """Mean |tilt| (deg) about one axis over the last `window_s` of HOLD."""
    rows = [row for row in m["tilt_deg"]
            if row[0] >= (m["tilt_deg"][-1][0] - window_s)]
    return float(np.mean([abs(row[1 + axis_idx]) for row in rows]))


FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}"
          f"{'  ' + detail if detail else ''}")
    if not ok:
        FAIL.append(name)


def scenario_rise():
    print("[A] rise, gain 0 (sweep 20.8 Hz and 250 Hz)")
    for sweep_dt in (0.048, 0.004):
        m = run_case(gain=0.0, sweep_dt=sweep_dt)
        held_z = float(np.mean(m["z"][-500:])) if m["z"] else 0.0
        tilt = max(max(abs(r), abs(p)) for _, r, p in m["tilt_deg"]) \
            if m["tilt_deg"] else 99.0
        check(f"sweep_dt={sweep_dt}: reaches HOLD", m["reached_hold"],
              "->".join(m["stages"]))
        check(f"sweep_dt={sweep_dt}: held z within 10 mm of target",
              abs(held_z - (0.20 + 0.02)) < 0.010,
              f"z={held_z:.3f} (target 0.220 incl. foot radius)")
        check(f"sweep_dt={sweep_dt}: tilt < 2 deg", tilt < 2.0,
              f"{tilt:.2f} deg")
        check(f"sweep_dt={sweep_dt}: feet do not slide (< 10 mm)",
              m["drift_m"] < 0.010, f"{m['drift_m'] * 1e3:.1f} mm")
        check(f"sweep_dt={sweep_dt}: no guard fault", m["fault"] is None,
              str(m["fault"]))


FRONT_SHIM = (("FL", "FR"), 0.010)     # +10 mm under the front feet
RIGHT_SHIM = (("FR", "RR"), 0.008)     # +8 mm under the right feet


def scenario_level_sign():
    print("[B] level-sign: shim under two feet in HOLD (truth attitude)")
    for shim, axis, axis_name, in ((FRONT_SHIM, 1, "front+10mm (nose up)"),
                                   (RIGHT_SHIM, 0, "right+8mm (right up)")):
        base_m = run_case(gain=0.0, shim=shim)
        base_tilt = _tail_tilt(base_m, axis)
        lvl_m = run_case(gain=H.LEVEL_GAIN_PER_S, shim=shim)
        lvl_tilt = _tail_tilt(lvl_m, axis)
        check(f"{axis_name}: baseline tilts ({base_tilt:.2f} deg)",
              base_tilt > 0.4)
        check(f"{axis_name}: leveling cuts tilt to < 30% of baseline",
              lvl_tilt < 0.30 * base_tilt,
              f"{base_tilt:.2f} -> {lvl_tilt:.2f} deg")
        off = lvl_m["off"]
        mean_off = sum(off.values()) / 4.0
        check(f"{axis_name}: offsets zero-mean", abs(mean_off) < 1e-4,
              f"mean {mean_off * 1e3:+.2f} mm")
        if axis == 1:      # front on shims -> shorten front (positive z off)
            check("pitch: front offsets positive (shorten), rear negative",
                  off["FL"] > 0 and off["FR"] > 0
                  and off["RL"] < 0 and off["RR"] < 0,
                  " ".join(f"{leg}{off[leg] * 1e3:+.1f}" for leg in LEGS))
        else:              # right on shims -> shorten right (y<0) legs
            check("roll: right offsets positive (shorten), left negative",
                  off["FR"] > 0 and off["RR"] > 0
                  and off["FL"] < 0 and off["RL"] < 0,
                  " ".join(f"{leg}{off[leg] * 1e3:+.1f}" for leg in LEGS))

    # deliberate wrong sign must be DETECTABLE
    base_tilt = _tail_tilt(run_case(gain=0.0, shim=FRONT_SHIM), 1)
    wrong = run_case(gain=-H.LEVEL_GAIN_PER_S, shim=FRONT_SHIM)
    wrong_tilt = _tail_tilt(wrong, 1)
    sat = max(abs(v) for v in wrong["off"].values()) >= 0.9 * H.LEVEL_CLAMP_M
    check("wrong-sign gain visibly diverges or saturates the clamp",
          wrong_tilt > 1.2 * base_tilt or sat,
          f"tilt {base_tilt:.2f} -> {wrong_tilt:.2f} deg, "
          f"clamp_sat={sat}")


def scenario_level_ekf():
    print("[C] level-ekf: front shim, EKF attitude in the loop")
    m = run_case(gain=H.LEVEL_GAIN_PER_S, att="ekf", shim=FRONT_SHIM)
    base_m = run_case(gain=0.0, att="ekf", shim=FRONT_SHIM)
    tilt = _tail_tilt(m, 1)
    base_tilt = _tail_tilt(base_m, 1)
    check("reaches HOLD with the EKF initialised", m["reached_hold"],
          "->".join(m["stages"]))
    check("leveling on EKF attitude cuts tilt to < 30% of baseline",
          tilt < 0.30 * base_tilt, f"{base_tilt:.2f} -> {tilt:.2f} deg")
    check("|EKF - truth| roll/pitch < 1 deg during HOLD",
          m["ekf_err_deg"] < 1.0, f"{m['ekf_err_deg']:.2f} deg")


def main():
    names = sys.argv[1:] or ["rise", "level-sign", "level-ekf"]
    for name in names:
        {"rise": scenario_rise,
         "level-sign": scenario_level_sign,
         "level-ekf": scenario_level_ekf}[name]()
    print()
    print("FAILURES:", FAIL if FAIL else "none")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
