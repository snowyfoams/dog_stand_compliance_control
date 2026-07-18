r"""Render the DOG5 position-mode crawl to an animated GIF.

Stand-up plays fast; the crawl steps play near real time with the current
step/phase stamped in the corner.

    D:\mujoco\.venv\Scripts\python.exe make_gif_crawl.py                # walk
    D:\mujoco\.venv\Scripts\python.exe make_gif_crawl.py --swing-test
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import crawl_dog5_sim as C
import stand3_dog5 as S

PLAY_MS = 120         # ms per frame (2x real time at SLOW_DT)
W, H = 480, 360
FAST_DT = 0.60        # capture interval during stand-up stages
SLOW_DT = 0.24        # capture interval during the crawl
FAST_STAGES = {"LIE", "ROLL", "CROUCH", "STAND", "HOLD4"}


def main():
    parser = C.build_parser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    args.zero_error = None
    out = Path(__file__).parent / (
        args.out or ("dog5_crawl_swing.gif" if args.swing_test
                     else "dog5_crawl.gif"))

    model, data = S.build_sim(kp_scale=args.kp_scale, start=args.start)
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 1.35, 50, -16

    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    frames = []
    state = {"next": 0.0}

    def capture(now, controller, _oracle):
        interval = FAST_DT if controller.stage in FAST_STAGES else SLOW_DT
        if now < state["next"]:
            return
        state["next"] = now + interval
        cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.10]
        renderer.update_scene(data, camera=cam)
        img = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(img)
        if controller.stage == "CRAWL" and controller.plan.step_ctx:
            ctx = controller.plan.step_ctx
            step = min(controller.plan.step_index + 1,
                       controller.plan.total_steps)
            text = (f"t={now:5.1f}s  step {step}/"
                    f"{controller.plan.total_steps} [{ctx['kind']}] "
                    f"{ctx['swing']} {controller.phase}")
        elif controller.stage in FAST_STAGES:
            text = f"t={now:5.1f}s  stand-up (fast-forward)"
        else:
            text = f"t={now:5.1f}s  {controller.stage}"
        draw.text((11, 9), text, font=font, fill=(0, 0, 0))
        draw.text((10, 8), text, font=font, fill=(255, 255, 255))
        frames.append(img.quantize(colors=96, dither=Image.Dither.NONE))

    controller, oracle = C.run(model, data, args, frame_hook=capture,
                               quiet=True)
    passed = C.report(controller, oracle, quiet=True)
    print(f"sequence {'PASS' if passed else 'FAIL'}; {len(frames)} frames")

    durations = [PLAY_MS] * len(frames)
    durations[-1] = 1500
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"saved {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
