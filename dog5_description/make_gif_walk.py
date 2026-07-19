r"""Render the DOG5 kinematic crawl to an animated GIF (~4x speed).

    D:\mujoco\.venv\Scripts\python.exe make_gif_walk.py  ->  dog5_walk.gif
"""
import argparse
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw, ImageFont

import stand3_dog5 as S
import walk_dog5 as W

OUT = Path(__file__).parent / "dog5_walk.gif"

PLAY_MS = 60
W_PX, H_PX = 480, 360
FAST_DT = 0.6         # capture interval during stand-up
WALK_DT = 0.25        # capture interval while walking
FAST_STAGES = {"CROUCH", "STAND", "HOLD4"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=float, default=W.DEFAULT_STEP_M)
    parser.add_argument("--shift", type=float, default=W.DEFAULT_SHIFT_M)
    parser.add_argument("--prelift", type=float, default=W.DEFAULT_PRELIFT_M)
    parser.add_argument("--swing-height", type=float,
                        default=W.DEFAULT_SWING_H_M)
    parser.add_argument("--cycles", type=int, default=W.DEFAULT_CYCLES)
    parser.add_argument("--torque-trip", type=float,
                        default=S.DEFAULT_TORQUE_TRIP_NM)
    parser.add_argument("--unload-trip", type=float,
                        default=S.DEFAULT_UNLOAD_TAU_TRIP_NM)
    parser.add_argument("--crouch-dps", type=float, default=S.DEFAULT_CROUCH_DPS)
    parser.add_argument("--stand-dps", type=float, default=S.DEFAULT_STAND_DPS)
    parser.add_argument("--stream-dps", type=float, default=S.DEFAULT_STREAM_DPS)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    args = parser.parse_args()

    model, data = S.build_sim(start="crouch")
    renderer = mujoco.Renderer(model, height=H_PX, width=W_PX)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 1.4, 50, -16

    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    frames = []
    state = {"next": 0.0}

    def capture(now, stage, _oracle=None):
        interval = FAST_DT if stage in FAST_STAGES else WALK_DT
        if now < state["next"]:
            return
        state["next"] = now + interval
        cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.10]
        renderer.update_scene(data, camera=cam)
        img = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(img)
        label = "stand-up (fast-forward)" if stage in FAST_STAGES else stage
        text = f"t={now:5.1f}s  {label}"
        draw.text((11, 9), text, font=font, fill=(0, 0, 0))
        draw.text((10, 8), text, font=font, fill=(255, 255, 255))
        frames.append(img.quantize(colors=128, dither=Image.Dither.NONE))

    controller, oracle = W.run(model, data, args, frame_hook=capture,
                               quiet=True)
    passed = W.report(controller, oracle, quiet=True)
    print(f"sequence {'PASS' if passed else 'FAIL'}; {len(frames)} frames")

    durations = [PLAY_MS] * len(frames)
    durations[-1] = 1500
    frames[0].save(OUT, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"saved {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
