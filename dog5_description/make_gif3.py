r"""Render the DOG5 three-leg stand to an animated GIF.

The stand-up stages (LIE..HOLD4, ~33 s) play fast; the interesting part
(SHIFT -> UNLOAD -> LIFT -> HOLD3 -> LOWER -> RECENTER) plays near real
time.  Stage + time are stamped in the corner.

    D:\mujoco\.venv\Scripts\python.exe make_gif3.py   ->  dog5_stand3.gif
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import stand3_dog5 as S

OUT = Path(__file__).parent / "dog5_stand3.gif"

PLAY_MS = 60          # ms per frame
W, H = 480, 360
FAST_DT = 0.60        # capture interval during stand-up stages
SLOW_DT = 0.12        # capture interval during the 3-leg sequence
HOLD3_DT = 0.50       # sparse during the static hold (nothing moves)
FAST_STAGES = {"LIE", "ROLL", "CROUCH", "STAND", "HOLD4"}

STAGE_TEXT = {
    "LIE": "stand-up (fast-forward)", "ROLL": "stand-up (fast-forward)",
    "CROUCH": "stand-up (fast-forward)", "STAND": "stand-up (fast-forward)",
    "HOLD4": "4-leg stand",
    "SHIFT": "SHIFT  body away from FR", "SHIFT_SETTLE": "SHIFT settle + margin gate",
    "UNLOAD": "UNLOAD  pre-lift 10 mm", "UNLOAD_GATE": "UNLOAD gate (torque check)",
    "LIFT": "LIFT  foot +40 mm", "HOLD3": "HOLD3  standing on 3 legs",
    "LOWER": "LOWER  foot down", "RELOAD": "RELOAD",
    "RECENTER": "RECENTER  body back", "DONE": "DONE  4-leg stand again",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--swing-leg", default=S.DEFAULT_SWING_LEG, choices=S.LEGS)
    parser.add_argument("--shift", type=float, default=S.DEFAULT_SHIFT_M)
    parser.add_argument("--lift", type=float, default=S.DEFAULT_LIFT_M)
    parser.add_argument("--torque-trip", type=float, default=S.DEFAULT_TORQUE_TRIP_NM)
    parser.add_argument("--unload-trip", type=float, default=S.DEFAULT_UNLOAD_TAU_TRIP_NM)
    parser.add_argument("--crouch-dps", type=float, default=S.DEFAULT_CROUCH_DPS)
    parser.add_argument("--stand-dps", type=float, default=S.DEFAULT_STAND_DPS)
    parser.add_argument("--stream-dps", type=float, default=S.DEFAULT_STREAM_DPS)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    args = parser.parse_args()

    model, data = S.build_sim()
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 1.35, 50, -16

    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    frames = []
    state = {"next": 0.0, "hold3_since": None}

    def capture(now, stage, _oracle=None):
        if stage in FAST_STAGES:
            interval = FAST_DT
        elif stage == "HOLD3":
            interval = HOLD3_DT
        else:
            interval = SLOW_DT
        if now < state["next"]:
            return
        state["next"] = now + interval
        cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.10]
        renderer.update_scene(data, camera=cam)
        img = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(img)
        text = f"t={now:5.1f}s  {STAGE_TEXT.get(stage, stage)}"
        draw.text((11, 9), text, font=font, fill=(0, 0, 0))
        draw.text((10, 8), text, font=font, fill=(255, 255, 255))
        frames.append(img.quantize(colors=128, dither=Image.Dither.NONE))

    controller, oracle = S.run(model, data, args, frame_hook=capture, quiet=True)
    passed = S.report(controller, oracle, quiet=True)
    print(f"sequence {'PASS' if passed else 'FAIL'}; {len(frames)} frames")

    durations = [PLAY_MS] * len(frames)
    durations[-1] = 1500
    frames[0].save(OUT, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"saved {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
