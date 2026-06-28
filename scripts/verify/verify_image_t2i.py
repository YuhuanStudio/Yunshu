"""Text-to-image generation gate (real Z-Image engine, discriminative).

The LoRA/weighting/ControlNet gates exercise diffusion features; this is the
baseline t2i correctness check: the prompt actually drives the pixels. Generate
a strongly-red scene and a strongly-green scene and assert each image is a valid
non-blank image AND its dominant color matches the prompt (red image is redder
than the green image, and vice-versa). Proves generation isn't returning noise
or a prompt-independent constant.

  - both images are valid + non-trivial (correct size, real variance)
  - the "red" prompt yields a more red-dominant image than the "green" prompt
  - the "green" prompt yields a more green-dominant image than the "red" prompt

Run: PYTHONPATH=. uv run python scripts/verify_image_t2i.py
"""
from __future__ import annotations

import asyncio
import io
import os
import sys

MODEL = os.environ.get("YUNSHU_IMAGE_MODEL", "./models/Z-Image-Turbo-MLX-4bit")


async def main() -> int:
    if not os.path.isdir(MODEL):
        print(f"SKIP: image model not available ({MODEL})")
        return 0

    import numpy as np
    from PIL import Image

    from yunshu_engine.image_engine import ImageGenEngine

    def arr(b):
        return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.float32)

    eng = ImageGenEngine(MODEL)
    await eng.start()
    checks: dict[str, bool] = {}
    detail: list[str] = []
    try:
        common = dict(width=512, height=512, num_inference_steps=8, seed=7)
        red = arr(await eng.generate_image(
            prompt="a solid bright red apple filling the frame, vivid scarlet red", **common))
        green = arr(await eng.generate_image(
            prompt="a lush deep green forest of leaves, vivid emerald green", **common))
    finally:
        await eng.stop()

    def channels(im):
        return im[..., 0].mean(), im[..., 1].mean(), im[..., 2].mean()

    rr, rg, rb = channels(red)
    gr, gg, gb = channels(green)
    checks["both images valid + non-blank"] = (
        red.shape == (512, 512, 3) and green.shape == (512, 512, 3)
        and red.var() > 100 and green.var() > 100)
    # red prompt's red-dominance (R-G) exceeds green prompt's
    checks["red prompt → more red than green prompt"] = (rr - rg) > (gr - gg)
    # green prompt's green-dominance (G-R) exceeds red prompt's
    checks["green prompt → more green than red prompt"] = (gg - gr) > (rg - rr)
    detail.append(f"red img  RGB=({rr:.0f},{rg:.0f},{rb:.0f}) R-G={rr - rg:+.0f}")
    detail.append(f"green img RGB=({gr:.0f},{gg:.0f},{gb:.0f}) G-R={gg - gr:+.0f}")

    for k, v in checks.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    for line in detail:
        print(f"     {line}")
    ok = all(checks.values())
    print(f"RESULT: {sum(checks.values())}/{len(checks)} passed")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
