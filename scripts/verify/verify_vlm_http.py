"""VLM image-chat HTTP route gate (in-process ASGI, model-manager injection).

verify_vlm_ocr.py tests VLMEngine directly. This gate covers the full multimodal
chat HTTP route: an OpenAI image_url (base64 data URL) content message →
/v1/chat/completions → model-manager VLM resolution → image preprocessing →
VLM inference → response. Like the image route, the VLM path resolves through the
model manager, so a pre-loaded VLMEngine is injected as a manager entry.

  - HTTP 200; assistant content non-empty
  - the VLM reads the word rendered in the supplied image (BANANA)
  - a text-only follow-up on the same route still works (no image required)

Run: PYTHONPATH=. uv run python scripts/verify_vlm_http.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import sys

MODEL_PATH = os.environ.get("YUNSHU_VLM_OCR_MODEL", "./models/GLM-OCR-bf16")
MODEL_ID = "GLM-OCR-bf16"


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
    if not os.path.isdir(MODEL_PATH):
        print(f"SKIP: VLM model not available ({MODEL_PATH})")
        return 0

    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    os.environ["YUNSHU_DRAIN_TIMEOUT"] = "0"

    import httpx

    from yunshu_engine.model_manager import ModelType
    from yunshu_engine.vlm_engine import VLMEngine
    from yunshu_gateway.engine import get_model_manager, init_model_manager
    from yunshu_gateway.main import create_app

    app = create_app()
    mgr = get_model_manager() or init_model_manager()
    eng = VLMEngine(MODEL_PATH)
    await eng.start()
    mgr.register_model(MODEL_ID, MODEL_PATH, model_type=ModelType.VLM)
    entry = mgr.get_entry(MODEL_ID)
    entry.engine = eng
    entry.is_loaded = True

    checks: dict[str, bool] = {}
    detail: list[str] = []
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=180) as client:
            r = await client.post("/v1/chat/completions", json={
                "model": MODEL_ID, "max_tokens": 40, "temperature": 0.0,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "What word is written in this image? Reply with just the word."},
                    {"type": "image_url", "image_url": {"url": _img_data_url("BANANA")}},
                ]}],
            })
            checks["VLM chat HTTP: 200"] = r.status_code == 200
            if r.status_code != 200:
                detail.append(f"status={r.status_code} body={r.text[:200]}")
            else:
                txt = (r.json()["choices"][0]["message"]["content"] or "")
                checks["assistant content non-empty"] = bool(txt.strip())
                checks["VLM reads the image word (BANANA)"] = "banana" in txt.lower()
                detail.append(f"image answer={txt[:50]!r}")

            # text-only on the same route still works
            r2 = await client.post("/v1/chat/completions", json={
                "model": MODEL_ID, "max_tokens": 16, "temperature": 0.0,
                "messages": [{"role": "user", "content": "Reply with the single word: hello"}]})
            checks["text-only request on VLM route works"] = (
                r2.status_code == 200 and bool((r2.json()["choices"][0]["message"]["content"] or "").strip()))
    finally:
        entry.is_loaded = False
        entry.engine = None
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
