r"""Render the DOG5 stand-up demo to an animated GIF (2x slow motion).

Captures one frame every 20 ms of simulated time and plays them back at
25 fps, i.e. everything appears at half speed. Phase + time are stamped
in the corner so viewers can follow the sequence.

    D:\mujoco\.venv\Scripts\python.exe make_gif.py        ->  dog5_stand.gif
"""
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import stand_dog5 as S

OUT = Path(__file__).parent / "dog5_stand.gif"

SIM_END = 6.0        # s of simulation to record (converged well before this)
CAPTURE_DT = 0.025   # s of sim time between frames
CAPTURE_DT_HOLD = 0.1  # slower capture once holding (nothing moves)
PLAY_MS = 50         # ms each frame is shown (20 fps) -> 2x slow motion
W, H = 480, 360

model = mujoco.MjModel.from_xml_path(S.XML)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
ctrl = S.Dog5Stand(model)

renderer = mujoco.Renderer(model, height=H, width=W)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 0.12]
cam.distance, cam.azimuth, cam.elevation = 1.15, 135, -18

try:
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 17)
except OSError:
    font = ImageFont.load_default()


def phase_label(t):
    if t < S.T_ROLL:
        return "1/4  ROLL  (hips out 90 deg)"
    if t < S.T_FOLD:
        return "2/4  FOLD  (legs under body)"
    if t < S.T_STAND:
        return "3/4  STAND  (Cartesian compliance)"
    return "4/4  HOLD  (compliant stand)"


frames = []
next_capture = 0.0
while data.time < SIM_END:
    if data.time >= next_capture:
        renderer.update_scene(data, camera=cam)
        img = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(img)
        text = f"t = {data.time:4.2f} s   {phase_label(data.time)}   [0.5x speed]"
        draw.text((11, 9), text, font=font, fill=(0, 0, 0))
        draw.text((10, 8), text, font=font, fill=(255, 255, 255))
        frames.append(img.quantize(colors=128, dither=Image.Dither.NONE))
        next_capture += CAPTURE_DT if data.time < S.T_STAND + 0.3 else CAPTURE_DT_HOLD
    ctrl.control(data)
    mujoco.mj_step(model, data)

durations = [PLAY_MS] * len(frames)
durations[-1] = 1500          # pause on the final standing pose before looping
frames[0].save(OUT, save_all=True, append_images=frames[1:],
               duration=durations, loop=0, optimize=True)
print(f"saved {OUT}  ({len(frames)} frames, {OUT.stat().st_size/1e6:.1f} MB)")
