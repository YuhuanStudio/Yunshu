"""VLM OCR / image-understanding gate (real VLM engine, GLM-OCR).

The 6-modality smoke checks a VLM names a simple shape; this gate is a
DISCRIMINATIVE image-understanding test: render two different words to images and
require the VLM to read EACH correctly (and not confuse them) — proving it reads
the pixels, not guesses. Exercises the multimodal image_url (base64 data URL)
content path end-to-end through VLMEngine.generate.

  - VLM reads "BANANA" from its image (and not the other word)
  - VLM reads "QUASAR" from its image (and not the other word)

Run: PYTHONPATH=. uv run python scripts/verify_vlm_ocr.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import sys

MODEL = os.environ.get("YUNSHU_VLM_OCR_MODEL", "./models/GLM-OCR-bf16")
WORDS = ["BANANA", "QUASAR"]


def _img_data_url(word: str) -> str:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (380, 140), "white")
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 60)
    except Exception:
        f = ImageFont.load_default()
    d.text((30, 40), word, fill="black", font=f)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


async def main() -> int:
    if not os.path.isdir(MODEL):
        print(f"SKIP: VLM model not available ({MODEL})")
        return 0

    from yunshu_engine.vlm_engine import VLMEngine
    eng = VLMEngine(MODEL)
    await eng.start()

    checks: dict[str, bool] = {}
    detail: list[str] = []
    try:
        reads = []
        for word in WORDS:
            msgs = [{"role": "user", "content": [
                {"type": "text", "text": "What word is written in this image? Reply with just the word."},
                {"type": "image_url", "image_url": {"url": _img_data_url(word)}},
            ]}]
            o = await eng.generate(messages=msgs, max_tokens=40, temperature=0.0)
            txt = (o.get("text") if isinstance(o, dict) else getattr(o, "text", str(o))) or ""
            reads.append(txt.upper())
            detail.append(f"{word} → {txt.strip()[:40]!r}")

        for word, got in zip(WORDS, reads):
            other = next(w for w in WORDS if w != word)
            checks[f"reads {word} correctly"] = word in got
            checks[f"does not confuse {word} with {other}"] = other not in got
    finally:
        await eng.stop()

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
