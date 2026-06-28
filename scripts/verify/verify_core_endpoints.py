"""Core OpenAI endpoints gate: /v1/models + /v1/completions (in-process ASGI).

Two standard endpoints no other gate covers: the model-listing endpoint and the
legacy text-completion endpoint (distinct from chat-completions).

  - GET /v1/models: object=="list", data non-empty, each entry has an id and
    object=="model"; the loaded model appears
  - POST /v1/completions: object=="text_completion", choices[0].text non-empty,
    finish_reason present, usage with prompt/completion tokens

Run: PYTHONPATH=. uv run python scripts/verify_core_endpoints.py
"""
from __future__ import annotations

import asyncio
import os
import sys

MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen3.5-0.8B-MLX-bf16")


async def main() -> int:
    if not os.path.exists(MODEL):
        print("SKIP: model not mounted")
        return 0

    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    os.environ["YUNSHU_DRAIN_TIMEOUT"] = "0"

    import httpx

    from yunshu_engine.batched_engine import BatchedEngine
    from yunshu_gateway.engine import set_engine
    from yunshu_gateway.main import create_app

    eng = BatchedEngine(model_name=MODEL)
    await eng.start()
    set_engine(eng)

    checks: dict[str, bool] = {}
    detail: list[str] = []
    try:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120) as client:
            # ── GET /v1/models ───────────────────────────────────────────────
            r = await client.get("/v1/models")
            ok200 = r.status_code == 200
            checks["/v1/models: HTTP 200"] = ok200
            if ok200:
                d = r.json()
                data = d.get("data") or []
                checks["/v1/models: object==list, non-empty"] = d.get("object") == "list" and len(data) >= 1
                checks["/v1/models: entries have id + object==model"] = all(
                    e.get("id") and e.get("object") == "model" for e in data)
                detail.append(f"models: n={len(data)} ids={[e.get('id') for e in data][:3]}")

            # ── POST /v1/completions (legacy) ────────────────────────────────
            r = await client.post("/v1/completions", json={
                "model": MODEL, "prompt": "The capital of France is",
                "max_tokens": 8, "temperature": 0.0})
            ok200 = r.status_code == 200
            checks["/v1/completions: HTTP 200"] = ok200
            if ok200:
                d = r.json()
                ch = (d.get("choices") or [{}])[0]
                usage = d.get("usage") or {}
                checks["/v1/completions: object==text_completion"] = d.get("object") == "text_completion"
                checks["/v1/completions: choices[0].text non-empty"] = bool((ch.get("text") or "").strip())
                checks["/v1/completions: finish_reason + usage"] = (
                    ch.get("finish_reason") is not None
                    and usage.get("prompt_tokens", 0) > 0 and usage.get("completion_tokens", 0) > 0)
                detail.append(f"completion text={ (ch.get('text') or '')[:45]!r} fr={ch.get('finish_reason')} usage={usage}")
            else:
                detail.append(f"completions status={r.status_code} body={r.text[:150]}")
    finally:
        set_engine(None)
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
