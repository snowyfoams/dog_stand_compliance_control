"""Open dog5.xml in the interactive MuJoCo viewer.

Run with the project venv:
    D:\mujoco\.venv\Scripts\python.exe view_dog5.py
"""
from pathlib import Path

import mujoco
import mujoco.viewer

XML = Path(__file__).parent / "dog5.xml"

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)

# Start from the "home" keyframe (standing height, all joints at zero).
mujoco.mj_resetDataKeyframe(model, data, 0)

# Full interactive viewer: it runs the physics itself.
# Space = pause/resume, Backspace = reset, Ctrl+right-drag = apply force,
# Control panel on the right = manual motor torques.
mujoco.viewer.launch(model, data)
