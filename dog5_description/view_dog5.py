r"""Open dog5.xml with joint axes, coordinate frames, and names visible.

Run with the project venv:
    D:\mujoco\.venv\Scripts\python.exe view_dog5.py
"""
import time
from pathlib import Path

import mujoco
import mujoco.viewer

XML = Path(__file__).parent / "dog5.xml"

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)

# Start from the "home" keyframe (standing height, all joints at zero).
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# Print numerical joint coordinates in the world frame.  xanchor is the joint
# origin and xaxis is the hinge axis after forward kinematics.
print("Joint coordinates in world frame (metres):")
print(f"{'joint':<18} {'anchor [x y z]':<32} axis [x y z]")
for jid in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    anchor = " ".join(f"{v: .4f}" for v in data.xanchor[jid])
    axis = " ".join(f"{v: .4f}" for v in data.xaxis[jid])
    print(f"{name:<18} [{anchor}]       [{axis}]")

# Passive mode lets us configure visualization before the first frame.
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Colored XYZ triads at body origins.  Every articulated body in dog5.xml
    # has its joint at local pos="0 0 0", so these are also the joint frames.
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    # MuJoCo's dedicated joint-axis markers and joint-name labels.
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    viewer.opt.label = mujoco.mjtLabel.mjLABEL_JOINT

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)
