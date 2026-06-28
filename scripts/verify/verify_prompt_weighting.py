"""Prompt-weighting gate — A1111/ComfyUI `(word:weight)` emphasis.

Verifies the image engine parses and APPLIES emphasis weighting: emphasizing vs
de-emphasizing a color must (a) change the image meaningfully and (b) move it in
the RIGHT direction (more weight on "red" → objectively redder). Z-Image RMSNorms
the caption, so this uses direction-based weighting (Wave 613ai); a regression to
magnitude-scaling would silently make weighting a no-op and fail this gate.

Run: PYTHONPATH=. uv run python scripts/verify_prompt_weighting.py
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

    def redness(a):  # R minus the other channels → how red the image is
        return float((a[:, :, 0] - (a[:, :, 1] + a[:, :, 2]) / 2).mean())

    eng = ImageGenEngine(MODEL)
    await eng.start()
    P = dict(width=512, height=512, num_inference_steps=8, seed=11)
    hi = arr(await eng.generate_image(prompt="a woman in a room, (red:1.7) dress", **P))
    lo = arr(await eng.generate_image(prompt="a woman in a room, (red:0.2) dress", **P))
    await eng.stop()

    diff = float(np.abs(hi - lo).mean())
    r_hi, r_lo = redness(hi), redness(lo)
    checks = {
        "changed (>10/255)": diff > 10.0,
        "right direction (emphasized redder)": r_hi > r_lo + 3.0,
    }
    for k, v in checks.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    print(f"RESULT: diff={diff:.1f}/255 redness hi={r_hi:.1f} lo={r_lo:.1f}")
    ok = all(checks.values())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
