"""Image-generation HTTP route gate (in-process ASGI, model-manager injection).

verify_image_t2i.py tests the engine directly. This gate covers the /v1/images/
generations HTTP ROUTE end-to-end: request parsing, model-manager resolution,
the b64_json response envelope, and size handling. Because the image route
resolves through the model manager (not the global set_engine), this populates a
manager entry with a pre-loaded ImageGenEngine.

  - POST /v1/images/generations → HTTP 200 with a `data` list
  - data[0].b64_json decodes to a valid, non-blank PNG of the requested size
  - an invalid response_format is rejected 4xx (request validation)

Run: PYTHONPATH=. uv run python scripts/verify_image_http.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import sys

MODEL_PATH = os.environ.get("YUNSHU_IMAGE_MODEL", "./models/Z-Image-Turbo-MLX-4bit")
MODEL_ID = "Z-Image-Turbo-MLX-4bit"


async def main() -> int:
    if not os.path.isdir(MODEL_PATH):
        print(f"SKIP: image model not available ({MODEL_PATH})")
        return 0

    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    os.environ["YUNSHU_DRAIN_TIMEOUT"] = "0"

    import httpx
    import numpy as np
    from PIL import Image

    from yunshu_engine.image_engine import ImageGenEngine
    from yunshu_engine.model_manager import ModelType
    from yunshu_gateway.engine import get_model_manager, init_model_manager
    from yunshu_gateway.main import create_app

    app = create_app()
    # ASGITransport doesn't run the app lifespan, so init the manager manually.
    mgr = get_model_manager() or init_model_manager()
    eng = ImageGenEngine(MODEL_PATH)
    await eng.start()

    # Inject the pre-loaded engine as a manager entry so the route resolves it.
    mgr.register_model(MODEL_ID, MODEL_PATH, model_type=ModelType.IMAGE_GEN)
    entry = mgr.get_entry(MODEL_ID)
    entry.engine = eng
    entry.is_loaded = True

    checks: dict[str, bool] = {}
    detail: list[str] = []
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=180) as client:
            r = await client.post("/v1/images/generations", json={
                "model": MODEL_ID, "prompt": "a vivid red apple on a white table",
                "size": "512x512", "num_inference_steps": 8, "response_format": "b64_json",
            })
            checks["/v1/images/generations: HTTP 200"] = r.status_code == 200
            if r.status_code != 200:
                detail.append(f"status={r.status_code} body={r.text[:200]}")
            else:
                data = r.json().get("data") or []
                checks["response has data[0].b64_json"] = bool(data) and bool(data[0].get("b64_json"))
                if data and data[0].get("b64_json"):
                    png = base64.b64decode(data[0]["b64_json"])
                    im = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32)
                    checks["decodes to valid 512x512 non-blank PNG"] = (
                        im.shape == (512, 512, 3) and im.var() > 100)
                    detail.append(f"img shape={im.shape} var={im.var():.0f}")

            # invalid response_format → 4xx
            rb = await client.post("/v1/images/generations", json={
                "model": MODEL_ID, "prompt": "x", "response_format": "bogus"})
            checks["invalid response_format → 4xx"] = 400 <= rb.status_code < 500
            detail.append(f"bad response_format status={rb.status_code}")
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
