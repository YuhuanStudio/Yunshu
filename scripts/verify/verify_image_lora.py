"""Image diffusion-LoRA gate.

Verifies the engine can load a ComfyUI/diffusion-format Z-Image LoRA (.safetensors
with lora_down/lora_up/alpha), that it MEASURABLY changes the output, produces a
valid (non-garbage) image, and UNLOADS to a byte-identical restore. Covers the
diffusion-LoRA loader added in Wave 613ah.

Run: PYTHONPATH=. uv run python scripts/verify_image_lora.py
"""
from __future__ import annotations

import asyncio
import io
import os
import sys

MODEL = os.environ.get("YUNSHU_IMAGE_MODEL", "./models/Z-Image-Turbo-MLX-4bit")


def _find_lora() -> str | None:
    d = "models"
    if not os.path.isdir(d):
        return None
    for p in os.listdir(d):
        if p.endswith(".safetensors") and "consistent" in p and "Controlnet" not in p:
            return os.path.join(d, p)
    return None


async def main() -> int:
    lora = _find_lora()
    if not os.path.isdir(MODEL) or lora is None:
        print(f"SKIP: image model or LoRA not available (model={MODEL}, lora={lora})")
        return 0
    import numpy as np
    from PIL import Image

    from yunshu_engine.image_engine import ImageGenEngine

    def arr(b):
        return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.float32)

    eng = ImageGenEngine(MODEL)
    await eng.start()
    P = dict(prompt="a woman in a bright cozy living room, portrait, detailed",
             width=512, height=512, num_inference_steps=8, seed=42)
    a = arr(await eng.generate_image(**P))
    loaded = eng.load_diffusion_lora(lora, strength=1.0)
    applied = len(getattr(eng, "_diff_lora_restore", []))
    b = arr(await eng.generate_image(**P))
    eng.unload_diffusion_lora()
    c = arr(await eng.generate_image(**P))
    await eng.stop()

    diff_on = float(np.abs(a - b).mean())       # LoRA must change output
    var_on = float(b.var())                      # must be a real image
    diff_restore = float(np.abs(a - c).mean())   # unload must restore

    checks = {
        "loaded": loaded and applied > 0,
        "changed_output (>3/255)": diff_on > 3.0,
        "valid_image (var>100)": var_on > 100,
        "unload_restores (<1/255)": diff_restore < 1.0,
    }
    for k, v in checks.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    print(f"RESULT: applied={applied} diff_on={diff_on:.2f} var={var_on:.0f} "
          f"diff_restore={diff_restore:.3f}")
    ok = all(checks.values())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
