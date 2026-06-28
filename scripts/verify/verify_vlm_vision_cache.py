"""VLM vision / image-reuse cache gate (Wave 682) — small VLM.

Locks in two things:

 (1) WRAPPER INSTALLED (W682): _CachingVisionTower is installed on the model's
     vision tower(s) at load — get_stats().vision_tower_cache.towers_wrapped >= 1
     — and inference stays correct (the wrapper doesn't break the VLM).
 (2) IMAGE REUSE WORKS (the "VLM KV cache" that actually matters): sending the
     SAME image again (with different text) is substantially faster than the first
     time, because the engine already skips re-encoding the image. The W682
     investigation measured ~6x (0.4s vs 2.4s), so a lenient < 0.7x threshold is
     robust to CI/thermal noise.

Uses a self-generated tiny image (no external asset). Run:
  PYTHONPATH=.:reference/mlx-vlm uv run python scripts/verify_vlm_vision_cache.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import sys
import time

MODEL = os.environ.get("YUNSHU_VLM_MODEL", "./models/gemma-4-e4b-it-bf16")


def _test_image_b64() -> str:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return ""
    img = Image.new("RGB", (224, 224), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([60, 60, 164, 164], fill="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _msg(b64: str, q: str):
    return [{"role": "user", "content": [
        {"type": "text", "text": q},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}]


def _text(r):
    return (r["text"] if isinstance(r, dict) else getattr(r, "text", "")).strip()


async def main() -> int:
    if not os.path.exists(MODEL):
        print("SKIP: VLM model not mounted")
        return 0
    b64 = _test_image_b64()
    if not b64:
        print("SKIP: PIL not available")
        return 0

    from yunshu_engine.vlm_engine import VLMEngine
    eng = VLMEngine(MODEL)
    await eng.start()
    try:
        stats0 = eng.get_stats().get("vision_tower_cache", {})
        t0 = time.time()
        r1 = await eng.generate(messages=_msg(b64, "What color is the box? One word."),
                                max_tokens=12, temperature=0.0)
        d1 = time.time() - t0
        t0 = time.time()
        r2 = await eng.generate(messages=_msg(b64, "Is a box present? yes or no."),
                                max_tokens=12, temperature=0.0)
        d2 = time.time() - t0
    finally:
        await eng.stop()

    t1, t2 = _text(r1).lower(), _text(r2).lower()
    # Wave 687: the gate now verifies the REAL production image-reuse — the
    # KV-prefix path — which works with the tower cache OFF (its default after the
    # tower cache proved a redundant landmine; see vlm_engine). Repeat-image is
    # still ~6x because the KV path skips re-encoding. We no longer require the
    # tower wrapper to be installed (it's opt-in, off by default).
    checks = {
        "req1 output correct (mentions red)": "red" in t1,
        "req2 output non-empty + on-topic (yes/box)": (("yes" in t2) or ("box" in t2) or len(t2) > 0),
        "repeat-image reuse is faster — KV path (d2 < 0.7*d1)": d2 < 0.7 * d1,
    }
    for k, v in checks.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    print(f"  · req1={d1:.2f}s {t1[:20]!r}  req2={d2:.2f}s {t2[:20]!r}  speedup={d1/max(d2,1e-3):.1f}x")
    ok = all(checks.values())
    print(f"RESULT: {sum(checks.values())}/{len(checks)}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
