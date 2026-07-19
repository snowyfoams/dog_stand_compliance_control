r"""Cartesian compliance stand-up controller for DOG5.

Starts from the calibration pose (all 12 joints = 0, legs flat fore-aft,
trunk resting on the ground) and stands up in four phases:

  1. ROLL  joint-space PD rolls every hip_abd to +/-90 deg so the legs point
           outboard and the knee plane is vertical - keeps left/right legs
           from crossing the midline during the fold. (Joint-space is needed
           anyway to escape the calibration pose, which is rank-1 singular:
           a Cartesian force has no vertical authority there.)
  2. FOLD  joint-space PD folds pitch+knee in the (now vertical) leg plane
           down to the crouch configuration, feet under the hips.
  3. STAND Cartesian compliance: per-leg foot targets in the trunk frame
           ramp from crouch height to standing height;
           tau = J^T (Kp e + Kd edot + F_ff).
  4. HOLD  same compliance law, fixed targets. The stand is soft:
           Ctrl+right-drag the trunk in the viewer and it yields and recovers.

Run with the project venv:
    D:\mujoco\.venv\Scripts\python.exe stand_dog5.py            # interactive viewer
    D:\mujoco\.venv\Scripts\python.exe stand_dog5.py --headless # 7 s test, prints metrics
    D:\mujoco\.venv\Scripts\python.exe stand_dog5.py --compare-jacobians
"""
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

XML = str(Path(__file__).parent / "dog5.xml")

LEGS = ["FL", "FR", "RL", "RR"]

# ---------------- tuning ----------------
T_ROLL = 1.2          # s, hip_abd rolls to +/-90 deg (legs point outboard)
T_FOLD = 2.8          # s, end of pitch/knee fold to crouch
T_STAND = 5.3         # s, end of Cartesian stand ramp
H_CROUCH = 0.10       # m, hip-to-foot height at end of fold
H_STAND = 0.20        # m, hip-to-foot height when standing
STANCE_OUT = 0.04     # m, feet this much outboard of the hips (wider = better roll stability)

KP_JOINT = 12.0       # Nm/rad   roll/fold-phase PD
KD_JOINT = 0.5        # Nms/rad

KP_CART = np.diag([600.0, 600.0, 600.0])   # N/m    Cartesian stiffness
KD_CART = np.diag([30.0, 30.0, 30.0])      # Ns/m   Cartesian damping
KD_JOINT_STAND = 0.15                      # Nms/rad extra joint damping in stand
TAU_MAX = 8.0                              # Nm torque clamp

HALF_PI = np.pi / 2
# legs rolled outboard, still extended fore-aft (knee plane now vertical):
Q_ROLL = {
    "FL": np.array([HALF_PI, 0.0, 0.0]),
    "FR": np.array([-HALF_PI, 0.0, 0.0]),
    "RL": np.array([HALF_PI, 0.0, 0.0]),
    "RR": np.array([-HALF_PI, 0.0, 0.0]),
}
# crouch configuration from damped-least-squares IK (feet under hips, h = H_CROUCH):
Q_CROUCH = {
    "FL": np.array([1.219, -0.796, -2.328]),
    "FR": np.array([-1.219, 0.796, 2.345]),
    "RL": np.array([1.219, 0.796, 2.345]),
    "RR": np.array([-1.219, -0.796, -2.328]),
}
# (time, joint targets) waypoints for the joint-space phases:
WAYPOINTS = [
    (0.0, {leg: np.zeros(3) for leg in LEGS}),
    (T_ROLL, Q_ROLL),
    (T_FOLD, Q_CROUCH),
]


def leg_jacobian_numpy(data, foot_site_id, joint_ids):
    """Return a leg's 3x3 translational Jacobian using NumPy only.

    For a hinge joint, the linear velocity of a point is

        v = axis x (point - joint_anchor) * joint_rate.

    MuJoCo has already performed forward kinematics, so ``xaxis``, ``xanchor``
    and ``site_xpos`` are expressed in the world frame.  Each resulting cross
    product is one Jacobian column.
    """
    foot = data.site_xpos[foot_site_id]
    return np.column_stack([
        np.cross(data.xaxis[jid], foot - data.xanchor[jid])
        for jid in joint_ids
    ])


class Dog5Stand:
    def __init__(self, model):
        self.m = model
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.site = {}
        self.jids = {}
        self.qadr = {}
        self.dadr = {}
        self.actadr = {}
        for leg in LEGS:
            self.site[leg] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"foot_{leg}")
            joints = [f"hip_abd_{leg}", f"hip_pitch_{leg}", f"knee_{leg}"]
            jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints]
            self.jids[leg] = np.array(jids)
            self.qadr[leg] = np.array([model.jnt_qposadr[j] for j in jids])
            self.dadr[leg] = np.array([model.jnt_dofadr[j] for j in jids])
            self.actadr[leg] = np.array(
                [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_motor") for j in joints]
            )
        hips = {leg: model.body_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"hip_{leg}")]
                for leg in LEGS}
        # foot targets in trunk frame: under the hips, slightly outboard
        self.foot_xy = {leg: np.array([hips[leg][0], hips[leg][1] + np.sign(hips[leg][1]) * STANCE_OUT])
                        for leg in LEGS}
        total_mass = model.body_mass.sum()
        self.f_support = total_mass * 9.81 / 4.0   # static share per foot
        self._jacp = np.zeros((3, model.nv))

    def compare_jacobians(self, data):
        """Print NumPy and MuJoCo leg Jacobians and return the largest error."""
        largest_error = 0.0
        for leg in LEGS:
            J_numpy = leg_jacobian_numpy(data, self.site[leg], self.jids[leg])

            self._jacp.fill(0.0)
            mujoco.mj_jacSite(self.m, data, self._jacp, None, self.site[leg])
            J_mujoco = self._jacp[:, self.dadr[leg]]

            error = J_numpy - J_mujoco
            max_error = np.max(np.abs(error))
            largest_error = max(largest_error, max_error)
            print(f"\n{leg} NumPy Jacobian [m/rad]:\n{J_numpy}")
            print(f"{leg} MuJoCo Jacobian [m/rad]:\n{J_mujoco}")
            print(f"{leg} difference (NumPy - MuJoCo), max |error|={max_error:.3e}:\n{error}")
        return largest_error

    @staticmethod
    def _cubic(t, T):
        """Smoothstep position s in [0,1] and rate ds/dt over duration T."""
        u = np.clip(t / T, 0.0, 1.0)
        return 3 * u**2 - 2 * u**3, (6 * u - 6 * u**2) / T

    def control(self, data):
        """Compute torques for the current state; writes data.ctrl."""
        t = data.time
        Rt = data.xmat[self.trunk].reshape(3, 3)

        for leg in LEGS:
            qa, da = self.qadr[leg], self.dadr[leg]
            q, dq = data.qpos[qa], data.qvel[da]

            if t < T_FOLD:
                # ---- phase 1/2: joint-space waypoint tracking (roll, then fold) ----
                for (t0, w0), (t1, w1) in zip(WAYPOINTS, WAYPOINTS[1:]):
                    if t < t1 or t1 == WAYPOINTS[-1][0]:
                        break
                s, ds = self._cubic(t - t0, t1 - t0)
                q_des = w0[leg] + s * (w1[leg] - w0[leg])
                dq_des = ds * (w1[leg] - w0[leg])
                tau = (KP_JOINT * (q_des - q) + KD_JOINT * (dq_des - dq)
                       + data.qfrc_bias[da])
            else:
                # ---- phase 2/3: Cartesian compliance ----
                s, ds = self._cubic(t - T_FOLD, T_STAND - T_FOLD)
                h = H_CROUCH + s * (H_STAND - H_CROUCH)
                p_des = np.array([*self.foot_xy[leg], -h])
                v_des = np.array([0.0, 0.0, -ds * (H_STAND - H_CROUCH)])

                # foot position/velocity relative to trunk, in trunk frame
                p_foot = Rt.T @ (data.site_xpos[self.site[leg]] - data.xpos[self.trunk])
                mujoco.mj_jacSite(self.m, data, self._jacp, None, self.site[leg])
                J = Rt.T @ self._jacp[:, da]          # 3x3, leg joints only
                v_foot = J @ dq

                F = KP_CART @ (p_des - p_foot) + KD_CART @ (v_des - v_foot)
                F[2] -= self.f_support                # feed-forward: push down to carry weight
                tau = J.T @ F - KD_JOINT_STAND * dq

            data.ctrl[self.actadr[leg]] = np.clip(tau, -TAU_MAX, TAU_MAX)


def main():
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    ctrl = Dog5Stand(model)

    if "--compare-jacobians" in sys.argv:
        # Move away from the initial straight-leg singularity so the comparison
        # contains general, nonzero Jacobian entries.
        while data.time < 3.5:
            ctrl.control(data)
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        error = ctrl.compare_jacobians(data)
        print(f"\nLargest error across all legs: {error:.3e}")
        return

    if "--headless" in sys.argv:
        import PIL.Image
        outdir = Path(sys.argv[sys.argv.index("--headless") + 1]) if len(sys.argv) > 2 else None
        renderer = mujoco.Renderer(model, height=480, width=640) if outdir else None
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 1.2, 135, -15
        snaps = [0.0, 1.2, 1.8, 2.4, 2.8, 5.3, 7.5]
        while data.time < 7.5:
            ctrl.control(data)
            mujoco.mj_step(model, data)
            if abs(data.time % 0.5) < model.opt.timestep:
                rpy = np.degrees(np.arcsin(np.clip(data.xmat[ctrl.trunk].reshape(3, 3)[2, :2], -1, 1)))
                print(f"t={data.time:5.2f}  trunk z={data.qpos[2]:.3f}  "
                      f"tilt(x,y)={rpy[0]:6.1f},{rpy[1]:6.1f} deg  "
                      f"|qvel|={np.linalg.norm(data.qvel):.3f}")
            if renderer and snaps and data.time >= snaps[0]:
                cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.12]
                renderer.update_scene(data, camera=cam)
                PIL.Image.fromarray(renderer.render()).save(outdir / f"stand_t{snaps[0]:.1f}.png")
                snaps.pop(0)
        print(f"FINAL: trunk z={data.qpos[2]:.3f} (target ~{H_STAND})")
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            t0 = time.perf_counter()
            while viewer.is_running():
                ctrl.control(data)
                mujoco.mj_step(model, data)
                viewer.sync()
                # real-time pacing
                lag = data.time - (time.perf_counter() - t0)
                if lag > 0:
                    time.sleep(lag)
                elif lag < -0.5:            # viewer was paused or reset
                    t0 = time.perf_counter() - data.time


if __name__ == "__main__":
    main()
