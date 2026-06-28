"""Anthropic tool_use E2E gate (in-process ASGI, real engine).

Anthropic represents tool calls differently from OpenAI: a `tool_use` content
block ({type, id, name, input:{}}) plus stop_reason=="tool_use". This gate
covers that distinct assembly over a real HTTP round-trip. Deterministic at
temperature 0.

  - HTTP 200; stop_reason == "tool_use"
  - content carries a tool_use block with a non-empty id, a name, and an input
    that is a dict (parsed JSON, not a string)
  - the correct tool is selected and arguments are right (get_weather, city~Tokyo)

Run: PYTHONPATH=. uv run python scripts/verify_anthropic_tools.py
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
            body = {
                "model": MODEL,
                "max_tokens": 80,
                "temperature": 0.0,
                "tools": [{
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }],
                "messages": [{"role": "user", "content": "What's the weather in Tokyo right now? Use the tool."}],
            }
            r = await client.post("/v1/messages", json=body)
            checks["HTTP 200"] = r.status_code == 200
            if r.status_code != 200:
                detail.append(f"status={r.status_code} body={r.text[:200]}")
            else:
                d = r.json()
                content = d.get("content") or []
                tu = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
                checks["stop_reason == tool_use"] = d.get("stop_reason") == "tool_use"
                checks["tool_use block present"] = bool(tu)
                if tu:
                    b = tu[0]
                    inp = b.get("input")
                    checks["tool_use block well-formed (id/name/input dict)"] = (
                        bool(b.get("id")) and bool(b.get("name")) and isinstance(inp, dict))
                    checks["correct tool + args (get_weather, city~Tokyo)"] = (
                        b.get("name") == "get_weather"
                        and isinstance(inp, dict)
                        and "tokyo" in str(inp.get("city", "")).lower())
                    detail.append(f"tool_use name={b.get('name')!r} input={inp} id={b.get('id')!r}")
                detail.append(f"stop_reason={d.get('stop_reason')} n_blocks={len(content)}")
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
