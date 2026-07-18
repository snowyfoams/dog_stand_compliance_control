#!/usr/bin/env python3
"""Compare independent DOG5 NumPy FK/Jacobians with MuJoCo and finite differences."""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from dog5_kinematics import LEGS, foot_jacobian, foot_position


XML = Path(__file__).with_name("dog5.xml")
JOINT_PREFIXES = ("hip_abd", "hip_pitch", "knee")

POSITION_TOL = 1.0e-10
JACOBIAN_TOL = 1.0e-10
FINITE_DIFFERENCE_TOL = 1.0e-8
FD_STEP = 1.0e-7

# Important controller poses, followed by randomized poses inside conservative
# software joint limits.
Q_CROUCH = {
    "FL": np.array([1.219, -0.796, -2.328]),
    "FR": np.array([-1.219, 0.796, 2.345]),
    "RL": np.array([1.219, 0.796, 2.345]),
    "RR": np.array([-1.219, -0.796, -2.328]),
}
Q_LOW = np.array([-1.70, -2.50, -2.50])
Q_HIGH = np.array([1.70, 2.50, 2.50])


class MujocoReference:
    """Small MuJoCo oracle used only by this offline comparison program."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)
        self.trunk = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk"
        )
        self.site = {}
        self.qadr = {}
        self.dadr = {}
        for leg in LEGS:
            self.site[leg] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, f"foot_{leg}"
            )
            joint_ids = [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{leg}"
                )
                for prefix in JOINT_PREFIXES
            ]
            self.qadr[leg] = np.array(
                [self.model.jnt_qposadr[jid] for jid in joint_ids]
            )
            self.dadr[leg] = np.array(
                [self.model.jnt_dofadr[jid] for jid in joint_ids]
            )
        self.jacp = np.zeros((3, self.model.nv))

    def evaluate(self, leg: str, q) -> tuple[np.ndarray, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr[leg]] = q
        mujoco.mj_forward(self.model, self.data)

        rotation_trunk = self.data.xmat[self.trunk].reshape(3, 3)
        position = rotation_trunk.T @ (
            self.data.site_xpos[self.site[leg]] - self.data.xpos[self.trunk]
        )
        mujoco.mj_jacSite(
            self.model, self.data, self.jacp, None, self.site[leg]
        )
        jacobian = rotation_trunk.T @ self.jacp[:, self.dadr[leg]]
        return position, jacobian


def finite_difference_jacobian(leg: str, q) -> np.ndarray:
    """Central-difference derivative of the independent NumPy FK."""
    q = np.asarray(q, dtype=float)
    jacobian = np.empty((3, 3))
    for column in range(3):
        step = np.zeros(3)
        step[column] = FD_STEP
        jacobian[:, column] = (
            foot_position(leg, q + step) - foot_position(leg, q - step)
        ) / (2.0 * FD_STEP)
    return jacobian


def comparison_poses(leg: str, samples: int, rng) -> list[np.ndarray]:
    fixed = [
        np.zeros(3),
        Q_CROUCH[leg],
        np.array([0.45, -0.70, 1.10]),
        np.array([-0.65, 1.25, -1.55]),
        Q_LOW,
        Q_HIGH,
    ]
    random_poses = rng.uniform(Q_LOW, Q_HIGH, size=(samples, 3))
    return fixed + list(random_poses)


def compare(samples: int, seed: int) -> bool:
    reference = MujocoReference()
    rng = np.random.default_rng(seed)
    passed = True

    print("DOG5 NumPy kinematics comparison (all values in trunk frame)")
    print(f"random poses per leg: {samples}; seed: {seed}; FD step: {FD_STEP:g}")
    print(" leg   poses      max |FK-MJ|       max |J-MJ|       max |J-FD|")
    print(" ----  -----  ---------------  ---------------  ---------------")

    overall = np.zeros(3)
    for leg in LEGS:
        maxima = np.zeros(3)
        poses = comparison_poses(leg, samples, rng)
        for q in poses:
            position_numpy = foot_position(leg, q)
            jacobian_numpy = foot_jacobian(leg, q)
            position_mujoco, jacobian_mujoco = reference.evaluate(leg, q)
            jacobian_fd = finite_difference_jacobian(leg, q)
            maxima = np.maximum(
                maxima,
                (
                    np.max(np.abs(position_numpy - position_mujoco)),
                    np.max(np.abs(jacobian_numpy - jacobian_mujoco)),
                    np.max(np.abs(jacobian_numpy - jacobian_fd)),
                ),
            )
        overall = np.maximum(overall, maxima)
        print(
            f" {leg:>4}  {len(poses):5d}  {maxima[0]:15.3e}  "
            f"{maxima[1]:15.3e}  {maxima[2]:15.3e}"
        )

    passed = bool(
        overall[0] <= POSITION_TOL
        and overall[1] <= JACOBIAN_TOL
        and overall[2] <= FINITE_DIFFERENCE_TOL
    )
    print(" ----  -----  ---------------  ---------------  ---------------")
    print(
        f" limits       {POSITION_TOL:15.3e}  {JACOBIAN_TOL:15.3e}  "
        f"{FINITE_DIFFERENCE_TOL:15.3e}"
    )
    print("PASS" if passed else "FAIL")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=int,
        default=250,
        help="random poses per leg, in addition to six fixed poses (default: 250)",
    )
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("--samples must be non-negative")
    return 0 if compare(args.samples, args.seed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
