"""Closed-loop MuJoCo harness: DOG5 stands and diagonal-steps under whole-body
VMC, with the proprioceptive state estimator in the feedback path.

Pipeline each physics tick (500 Hz):
    MuJoCo sensors -> (specific force, gyro, joint angles, schedule contacts)
        -> estimator.predict + update  ->  outputs() (r, v, q, w_hat)
        -> VMC body wrench -> foot-force distribution -> joint torques
        -> data.ctrl -> mj_step

Sequence: stand-up (reuses stand_dog5.Dog5Stand) -> estimator init on a static
hold -> VMC hold (all four feet) -> diagonal stepping -> hold.

Run (mujoco lives in the project venv):
    .venv/bin/python vmc_mujoco.py            # interactive viewer
    .venv/bin/python vmc_mujoco.py --headless # metrics + snapshots, no window
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import mujoco

_DESC = Path(__file__).resolve().parent.parent / "dog5_description"
_EST = Path(__file__).resolve().parent.parent / "state_estimator"
for _p in (_DESC, _EST):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dog5_vmc_core as vmc          # noqa: E402
import dog5_gait                      # noqa: E402
from dog5_state_estimator import DOG5StateEstimator, quat_to_C  # noqa: E402
import stand_dog5                     # noqa: E402 (stand-up maneuver)

XML = str(_DESC / "dog5.xml")
LEGS = vmc.LEGS
DT = 0.002                            # matches dog5.xml timestep

# phase durations (s)
T_STANDUP = stand_dog5.T_STAND + 1.0  # let Dog5Stand reach and settle standing
T_INIT = 0.5                          # static window to initialise the estimator
T_HOLD = 3.0                          # VMC hold (all four feet)
GAIT_PERIOD = 2.0
N_STEP_CYCLES = 4
T_HOLD_END = 1.5
FALL_Z = 0.10                         # trunk height below this = fallen


def wxyz_to_R(quat):
    """MuJoCo [w,x,y,z] (body->world) -> 3x3 rotation matrix."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
    return R.reshape(3, 3)


class VMCSim:
    """Thin MuJoCo accessor: joint state, IMU, ground truth, torque write."""

    def __init__(self, model, data):
        self.m, self.d = model, data
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.qadr, self.dadr, self.actadr = {}, {}, {}
        for leg in LEGS:
            names = [f"hip_abd_{leg}", f"hip_pitch_{leg}", f"knee_{leg}"]
            jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
            self.qadr[leg] = np.array([model.jnt_qposadr[j] for j in jids])
            self.dadr[leg] = np.array([model.jnt_dofadr[j] for j in jids])
            self.actadr[leg] = np.array(
                [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_motor")
                 for n in names])
        self._sadr = {}
        for name in ("imu_acc", "imu_gyro", "imu_quat", "trunk_pos", "trunk_linvel"):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self._sadr[name] = (model.sensor_adr[sid], model.sensor_dim[sid])
        self.mass = float(model.body_mass.sum())

    def sensor(self, name):
        adr, dim = self._sadr[name]
        return self.d.sensordata[adr:adr + dim].copy()

    def joint_state(self):
        q = np.concatenate([self.d.qpos[self.qadr[leg]] for leg in LEGS])
        qd = np.concatenate([self.d.qvel[self.dadr[leg]] for leg in LEGS])
        return q, qd

    def alpha44(self):
        return np.stack([self.d.qpos[self.qadr[leg]] for leg in LEGS], axis=0)

    def imu(self):
        return self.sensor("imu_acc"), self.sensor("imu_gyro")

    def truth_rp(self):
        C = wxyz_to_R(self.sensor("imu_quat")).T     # I->B
        return vmc.attitude_error_rp(C)

    def trunk_z(self):
        return float(self.sensor("trunk_pos")[2])

    def write_torque(self, tau12):
        for i, leg in enumerate(LEGS):
            self.d.ctrl[self.actadr[leg]] = tau12[3 * i:3 * i + 3]


def _est_out(est, w_meas):
    out = est.outputs(last_w_meas=w_meas)     # w_hat = w_meas - gyro bias
    out["C"] = quat_to_C(out["q"])
    return out


class VMCPipeline:
    """The full stand -> hold -> step state machine, one `step()` per tick.

    Shared by the headless and viewer drivers so the control path is identical.
    """

    def __init__(self, model, data, gains=None, cfg=None,
                 mode_after_hold="DIAGONAL", gait_params=None):
        self.sim = VMCSim(model, data)
        self.stand = stand_dog5.Dog5Stand(model)
        self.est = DOG5StateEstimator()
        self.gains = gains or vmc.VMCGains()
        self.cfg = cfg or vmc.VMCConfig()
        self.mode_after_hold = mode_after_hold
        self.gait_params = gait_params or dog5_gait.GaitParams(period=GAIT_PERIOD)
        self.total_t = (T_STANDUP + T_INIT + T_HOLD
                        + N_STEP_CYCLES * self.gait_params.period + T_HOLD_END)
        # runtime state
        self.scheduler = None
        self._init_acc, self._init_gyro = [], []
        self._started = False
        self._vmc_t0 = 0.0
        self._sched_t0 = 0.0
        self._z_true0 = None
        self._frozen = None

    def step(self, freeze_estimator=False, push=None):
        """Advance the controller (writes data.ctrl). Returns a record dict once
        the VMC phase is active, else None. Does NOT call mj_step."""
        d = self.sim.d
        t = d.time
        if t < T_STANDUP:
            self.stand.control(d)
            return None
        if t < T_STANDUP + T_INIT:
            self.stand.control(d)
            a, g = self.sim.imu()
            self._init_acc.append(a)
            self._init_gyro.append(g)
            return None

        if not self._started:
            self.est.initialise(np.array(self._init_acc), np.array(self._init_gyro),
                                self.sim.alpha44(), np.ones(4, dtype=bool))
            nominal = {leg: vmc.dog5_kinematics.foot_position(leg, self.sim.alpha44()[i])
                       for i, leg in enumerate(LEGS)}
            self.scheduler = dog5_gait.GaitScheduler(nominal, "STAND", self.gait_params)
            self._z_true0 = self.sim.trunk_z()
            self._vmc_t0 = t
            self._started = True

        tv = t - self._vmc_t0
        if self.scheduler.mode == "STAND" and tv >= T_HOLD:
            self.scheduler.mode = self.mode_after_hold
            self._sched_t0 = tv
        gt = tv - self._sched_t0

        q, qd = self.sim.joint_state()
        f_meas, w_meas = self.sim.imu()
        contacts = self.scheduler.contacts(gt)
        self.est.predict(f_meas, w_meas, DT, contacts)
        self.est.update(self.sim.alpha44(), contacts)

        if freeze_estimator:
            if self._frozen is None:
                self._frozen = _est_out(self.est, w_meas)
            out = self._frozen
        else:
            out = _est_out(self.est, w_meas)

        sched = self.scheduler.sched_state(gt)
        tau, diag = vmc.compute_vmc_torques(q, qd, out, sched, self.gains, self.cfg,
                                            self.sim.mass)
        self.sim.write_torque(tau)

        if push is not None and push["t0"] <= tv < push["t0"] + push["dur"]:
            d.xfrc_applied[self.sim.trunk, :3] = push["force"]
        else:
            d.xfrc_applied[self.sim.trunk, :3] = 0.0

        z_true = self.sim.trunk_z() - self._z_true0
        return {
            "tv": tv, "gt": gt, "mode": self.scheduler.mode,
            "z_est": float(out["r"][2]), "z_true": z_true,
            "v_est": out["v"].copy(), "v_true": self.sim.sensor("trunk_linvel").copy(),
            "rp_est": vmc.attitude_error_rp(out["C"]), "rp_true": self.sim.truth_rp(),
            "healthy": bool(out["healthy"]), "fell": self.sim.trunk_z() < FALL_Z,
            "stance": diag["stance"], "contacts": contacts.copy(),
        }


def run(snapshot_dir=None, push=None, freeze_estimator=False,
        mode_after_hold="DIAGONAL", gait_params=None):
    """Headless run; returns metrics + recorded history over the VMC phase."""
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    pipe = VMCPipeline(model, data, mode_after_hold=mode_after_hold,
                       gait_params=gait_params)

    snaps = list(np.linspace(0, pipe.total_t, 8)) if snapshot_dir else []
    renderer, cam = None, None
    if snapshot_dir:
        Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 1.4, 130, -15

    hist = []
    fell = False
    while data.time < pipe.total_t:
        rec = pipe.step(freeze_estimator=freeze_estimator, push=push)
        if rec is not None:
            hist.append(rec)
            fell = fell or rec["fell"]
        mujoco.mj_step(model, data)
        if renderer and snaps and data.time >= snaps[0]:
            cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.12]
            renderer.update_scene(data, camera=cam)
            import PIL.Image
            PIL.Image.fromarray(renderer.render()).save(
                Path(snapshot_dir) / f"vmc_t{snaps[0]:.1f}.png")
            snaps.pop(0)

    return _summarise(hist, fell, pipe.sim.trunk_z())


def _summarise(hist, fell, final_z):
    m = {"fell": fell, "final_z": final_z, "n": len(hist)}
    if not hist:
        return m
    z_est = np.array([r["z_est"] for r in hist])
    z_true = np.array([r["z_true"] for r in hist])
    v_est = np.array([r["v_est"] for r in hist])
    v_true = np.array([r["v_true"] for r in hist])
    rp_est = np.array([r["rp_est"] for r in hist])
    rp_true = np.array([r["rp_true"] for r in hist])
    tv = np.array([r["tv"] for r in hist])
    m["z_err"] = np.abs(z_est - z_true)
    m["v_err"] = np.linalg.norm(v_est - v_true, axis=1)
    m["rp_err_deg"] = np.degrees(np.abs(rp_est - rp_true))
    m["tilt_true_deg"] = np.degrees(np.linalg.norm(rp_true, axis=1))
    m["healthy_frac"] = float(np.mean([r["healthy"] for r in hist]))
    m["tv"] = tv
    m["hold_mask"] = tv < T_HOLD          # all-four-feet hold window
    m["z_true"] = z_true
    m["n_swing"] = int(sum(1 for r in hist if len(r["stance"]) < 4))
    m["hist"] = hist
    return m


def _print_metrics(m):
    print(f"fell={m['fell']}  final trunk z={m['final_z']:.3f}  samples={m['n']}")
    if m["n"] == 0:
        return
    print(f"  est vs truth: z_err max/mean = {m['z_err'].max()*1e3:.1f}/"
          f"{m['z_err'].mean()*1e3:.1f} mm")
    print(f"                |v|_err max/mean = {m['v_err'].max()*1e3:.1f}/"
          f"{m['v_err'].mean()*1e3:.1f} mm/s")
    print(f"                roll/pitch_err max = {m['rp_err_deg'].max():.2f} deg")
    print(f"  body tilt (truth) max = {m['tilt_true_deg'].max():.2f} deg")
    print(f"  estimator healthy fraction = {m['healthy_frac']:.3f}")


def _live_loop(model):
    import mujoco.viewer
    import time
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    pipe = VMCPipeline(model, data)
    print("Viewer: stand -> VMC hold -> diagonal step (estimator in the loop)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.perf_counter()
        while viewer.is_running() and data.time < pipe.total_t:
            pipe.step()
            mujoco.mj_step(model, data)
            viewer.sync()
            lag = data.time - (time.perf_counter() - t0)
            if lag > 0:
                time.sleep(lag)
            elif lag < -0.5:
                t0 = time.perf_counter() - data.time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--snapshots", type=str, default=None)
    args = ap.parse_args()
    if args.headless or args.snapshots:
        _print_metrics(run(snapshot_dir=args.snapshots))
    else:
        _live_loop(mujoco.MjModel.from_xml_path(XML))


if __name__ == "__main__":
    main()
