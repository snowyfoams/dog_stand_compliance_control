r"""Render the hardware crawl video to an animated GIF for inline README embedding.

GitHub does not play mp4/MOV inline in a README -- only images. This
re-encodes the full-resolution source video (``result_hw/crawl.mp4``) down to
a small, looping GIF the same way ``dog5_stand_hw.gif`` was produced from the
Stage 4 stand-up video, so the hardware crawl result is visible directly on
the repo page; the source mp4 stays linked below it for full quality.

    D:\mujoco\.venv\Scripts\python.exe make_gif_crawl_hw.py   ->  dog5_crawl_hw.gif
"""
from pathlib import Path

import imageio
from PIL import Image

SRC = Path(__file__).parent.parent / "result_hw" / "crawl.mp4"
OUT = Path(__file__).parent / "dog5_crawl_hw.gif"

TARGET_FPS = 7.5     # matches dog5_stand_hw.gif's sampling rate
OUT_WIDTH = 400       # matches dog5_stand_hw.gif's width

reader = imageio.get_reader(SRC)
meta = reader.get_meta_data()
src_fps = meta["fps"]
stride = max(1, round(src_fps / TARGET_FPS))

frames = []
for i, frame in enumerate(reader):
    if i % stride:
        continue
    img = Image.fromarray(frame)
    h = round(img.height * OUT_WIDTH / img.width)
    img = img.resize((OUT_WIDTH, h), Image.LANCZOS)
    frames.append(img.quantize(colors=256, method=Image.Quantize.MEDIANCUT))
reader.close()

duration_ms = round(1000 * stride / src_fps)
durations = [duration_ms] * len(frames)
durations[-1] = 1500          # pause on the last frame before looping
frames[0].save(OUT, save_all=True, append_images=frames[1:],
               duration=durations, loop=0, optimize=True)
print(f"saved {OUT}  ({len(frames)} frames, {OUT.stat().st_size / 1e6:.1f} MB)")
