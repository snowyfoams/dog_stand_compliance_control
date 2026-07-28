"""Sim validation: VMC+EKF RISE from the RECORDED hardware crouch, then hold.

This mirrors vmc_stand_hw.py exactly: start from the recorded crouch pose
(stand_dog5_recorded_hw.Q_RECORDED_CROUCH -- feet splayed to x=+/-0.34, trunk
only ~24 mm above the feet), init the EKF on a static hold of that pose, then
have VMC+EKF ramp z_des up to standing height (~176 mm rise) and hold.

The recorded crouch is much lower/more-splayed than stand_dog5's Q_CROUCH, so
this is the pose that actually matters for hardware. A sign/gain error or a
reachability problem in the rise shows here, not on the robot.

    .venv/bin/python vmc_rise_sim.py
"""
from __future__ import annotations

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
import dog5_kinematics               # noqa: E402
import stand_dog5_recorded_hw as recorded   # noqa: E402
from dog5_state_estimator import DOG5StateEstimator, quat_to_C  # noqa: E402
from vmc_mujoco import VMCSim, _est_out      # noqa: E402

XML = str(_DESC / "dog5.xml")
LEGS = vmc.LEGS
DT = 0.002

Q_CROUCH = recorded.Q_RECORDED_CROUCH        # controller order, radians
STAND_HEIGHT = recorded.DEFAULT_STAND_HEIGHT
T_SETTLE = 1.5                                # settle on feet + EKF init
T_RISE = 5.0                                  # gentle rise over the 176 mm
T_HOLD = 3.0
KP_CROUCH = 12.0                              # joint-PD hold of the crouch
KD_CROUCH = 0.5


def _smoothstep(u):
    u = float(np.clip(u, 0.0, 1.0))
    return 3 * u * u - 2 * u * u * u


def _crouch_height():
    zs = [dog5_kinematics.foot_position(LEGS[i], Q_CROUCH[3 * i:3 * i + 3])[2]
          for i in range(4)]
    return -float(np.mean(zs))


def run(record=True):
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    sim = VMCSim(model, data)

    # place the robot in the recorded crouch, trunk just above the feet
    crouch_h = _crouch_height()
    for i, leg in enumerate(LEGS):
        data.qpos[sim.qadr[leg]] = Q_CROUCH[3 * i:3 * i + 3]
    data.qpos[2] = crouch_h + 0.03            # drop a little onto the feet
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    est = DOG5StateEstimator()
    gains, cfg = vmc.VMCGains(), vmc.VMCConfig()
    dz_rise = max(0.0, STAND_HEIGHT - crouch_h)

    init_a, init_g, started = [], [], False
    t_vmc0 = None
    z_true0 = None
    hist = []
    total = T_SETTLE + T_RISE + T_HOLD
    q_crouch = {leg: Q_CROUCH[3 * i:3 * i + 3] for i, leg in enumerate(LEGS)}

    while data.time < total:
        t = data.time
        if t < T_SETTLE:
            # joint-PD hold the recorded crouch (stands in for native position
            # control); gather IMU for EKF init.
            for leg in LEGS:
                qa, da = sim.qadr[leg], sim.dadr[leg]
                q, dq = data.qpos[qa], data.qvel[da]
                data.ctrl[sim.actadr[leg]] = (KP_CROUCH * (q_crouch[leg] - q)
                                              + KD_CROUCH * (-dq)
                                              + data.qfrc_bias[da])
            a, g = sim.imu()
            init_a.append(a)
            init_g.append(g)
        else:
            if not started:
                est.initialise(np.array(init_a), np.array(init_g),
                               sim.alpha44(), np.ones(4, dtype=bool))
                t_vmc0 = t
                z_true0 = sim.trunk_z()
                started = True
            tv = t - t_vmc0
            z_des = dz_rise * _smoothstep(tv / T_RISE)
            q, qd = sim.joint_state()
            f_meas, w_meas = sim.imu()
            c = np.ones(4, dtype=bool)
            est.predict(f_meas, w_meas, DT, c)
            est.update(sim.alpha44(), c)
            out = _est_out(est, w_meas)
            sched = {"contacts": c, "swing_targets": {}, "z_des": z_des,
                     "v_cmd_world": np.zeros(3), "yawrate_cmd": 0.0}
            tau, diag = vmc.compute_vmc_torques(q, qd, out, sched, gains, cfg, sim.mass)
            sim.write_torque(tau)
            if record:
                rp = sim.truth_rp()
                hist.append({
                    "tv": tv, "z_des": z_des, "trunk_z": sim.trunk_z(),
                    "z_est": float(out["r"][2]), "z_true": sim.trunk_z() - z_true0,
                    "tilt": float(np.degrees(np.hypot(*rp))),
                    "max_tau": float(np.max(np.abs(tau))),
                    "healthy": bool(out["healthy"]),
                })
        mujoco.mj_step(model, data)

    return hist, sim.trunk_z(), crouch_h, dz_rise


def main():
    hist, final_z, crouch_h, dz = run()
    fell = final_z < 0.14
    tilt_max = max(h["tilt"] for h in hist)
    zerr_max = max(abs(h["z_est"] - h["z_true"]) for h in hist)
    tau_max = max(h["max_tau"] for h in hist)
    tail = [h for h in hist if h["tv"] > T_RISE]
    z_reached = float(np.mean([h["trunk_z"] for h in tail])) if tail else 0.0
    healthy = float(np.mean([h["healthy"] for h in hist]))
    print(f"recorded crouch h = {crouch_h:.3f} m  -> rise dz = {dz*1e3:.0f} mm")
    print(f"final trunk z = {final_z:.3f}   held standing (post-rise) = {z_reached:.3f}")
    print(f"peak tilt = {tilt_max:.2f} deg   fell = {fell}")
    print(f"peak joint tau = {tau_max:.2f} Nm   EKF z_err max = {zerr_max*1e3:.1f} mm")
    print(f"healthy fraction = {healthy:.3f}")
    ok = (not fell) and z_reached > 0.17 and tilt_max < 10.0 and healthy > 0.95
    print("RESULT:", "PASS -- VMC+EKF rises from the recorded crouch and holds"
          if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
