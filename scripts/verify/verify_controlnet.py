"""Z-Image ControlNet structural-following gate.

Discriminative test (a similar control+prompt pair can pass by luck — this one
can't): make a control map from the edges of a TALL BOTTLE, then generate with the
prompt "a red apple". If ControlNet imposes structure, the output follows the bottle
silhouette (tall), and its edges correlate with the control map MORE than a free
"red apple" (round) does. Guards the fix (refiner-hint injection + pre-
noise_refiner ordering) — a regression makes control a no-op and fails this.

Run: PYTHONPATH=. uv run python scripts/verify_controlnet.py
"""
from __future__ import annotations

import asyncio
import io
import os
import sys

MODEL = os.environ.get("YUNSHU_IMAGE_MODEL", "./models/Z-Image-Turbo-MLX-4bit")


def _has_controlnet() -> bool:
    return os.path.isdir("models") and any(
        p.endswith(".safetensors") and "controlnet" in p.lower() for p in os.listdir("models"))


async def main() -> int:
    if not os.path.isdir(MODEL) or not _has_controlnet():
        print("SKIP: image model or ControlNet weights not available")
        return 0
    import numpy as np
    from PIL import Image, ImageFilter

    from yunshu_engine.image_engine import ImageGenEngine

    def edges_gray(b):
        return np.asarray(Image.open(io.BytesIO(b)).convert("L").filter(ImageFilter.FIND_EDGES),
                          dtype=np.float32)

    def edge_bytes(b):
        g = Image.open(io.BytesIO(b)).convert("L").filter(ImageFilter.FIND_EDGES)
        buf = io.BytesIO(); Image.merge("RGB", (g, g, g)).save(buf, "PNG"); return buf.getvalue()

    def corr(a, c):
        a, c = a.ravel() - a.mean(), c.ravel() - c.mean()
        d = (np.linalg.norm(a) * np.linalg.norm(c)) or 1.0
        return float((a @ c) / d)

    eng = ImageGenEngine(MODEL)
    await eng.start()
    P = dict(width=512, height=512, num_inference_steps=8, seed=5)
    bottle = await eng.generate_image(prompt="a tall slim glass bottle standing upright, centered", **P)
    ctrl_map = edge_bytes(bottle)
    ctrl_gray = edges_gray(ctrl_map)
    controlled = await eng.generate_controlled_image("a red apple", ctrl_map,
                                                     control_scale=1.0, **P)
    free = await eng.generate_image(prompt="a red apple", **P)
    await eng.stop()

    c_ctrl = corr(edges_gray(controlled), ctrl_gray)
    c_free = corr(edges_gray(free), ctrl_gray)
    checks = {
        "follows control (corr_controlled > corr_free + 0.05)": c_ctrl > c_free + 0.05,
        "control has real effect (corr_controlled > 0.15)": c_ctrl > 0.15,
    }
    for k, v in checks.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    print(f"RESULT: edge-corr controlled={c_ctrl:.3f} vs free={c_free:.3f}")
    ok = all(checks.values())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
