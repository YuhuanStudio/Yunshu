"""JSON mode (response_format: json_object) gate (in-process ASGI, real engine).

response_format={"type":"json_object"} must constrain the output to a single
valid JSON value, distinct from json_schema (free-form JSON, no schema). Covered
end-to-end via the gateway.

  - HTTP 200; the assistant content parses as JSON (an object)
  - no leading/trailing prose around the JSON
  - a plain request (no response_format) is unaffected (control: still answers)

Run: PYTHONPATH=. uv run python scripts/verify_json_mode.py
"""
from __future__ import annotations

import asyncio
import json
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
            r = await client.post("/v1/chat/completions", json={
                "model": MODEL,
                "messages": [{"role": "user", "content":
                              "Give me a JSON object describing a person with a name and an age."}],
                "response_format": {"type": "json_object"},
                "temperature": 0.0, "max_tokens": 80,
            })
            checks["json_object: HTTP 200"] = r.status_code == 200
            if r.status_code != 200:
                detail.append(f"status={r.status_code} body={r.text[:200]}")
            else:
                content = (r.json()["choices"][0]["message"]["content"] or "").strip()
                parsed = None
                try:
                    parsed = json.loads(content)
                except Exception as e:
                    detail.append(f"JSON parse failed: {e}")
                checks["json_object: output parses as JSON"] = parsed is not None
                checks["json_object: result is an object"] = isinstance(parsed, dict)
                checks["json_object: no prose wrapper (starts { ends })"] = (
                    content.startswith("{") and content.endswith("}"))
                detail.append(f"content={content[:70]!r}")

            # control: plain request still works (no response_format)
            r2 = await client.post("/v1/chat/completions", json={
                "model": MODEL, "messages": [{"role": "user", "content": "Say hello."}],
                "temperature": 0.0, "max_tokens": 16})
            checks["control: plain request unaffected"] = (
                r2.status_code == 200 and bool((r2.json()["choices"][0]["message"]["content"] or "").strip()))
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
