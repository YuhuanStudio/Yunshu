"""Streaming tool_calls delta-assembly gate (in-process ASGI, real engine).

verify_tool_calls_e2e.py covers the NON-streaming tool path. Streaming tool_calls
have their own contract (OpenAI: the first tool_call delta carries id+type+
function.name; later deltas carry argument fragments; the client concatenates
them). This gate streams a tool query and reassembles the call. Deterministic at
temp 0.

  - the stream emits delta.tool_calls fragments
  - reassembled: function name == get_weather, arguments concatenate to JSON
    parsing to {city ~ Tokyo}
  - the terminal chunk's finish_reason == "tool_calls"
  - stream ends with [DONE]

Run: PYTHONPATH=. uv run python scripts/verify_streaming_tool_calls.py
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
            body = {
                "model": MODEL, "stream": True, "temperature": 0.0, "max_tokens": 80,
                "tools": [{"type": "function", "function": {
                    "name": "get_weather", "description": "Get current weather for a city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
                "messages": [{"role": "user", "content": "What's the weather in Tokyo right now? Use the tool."}],
            }
            name = ""
            args = ""
            saw_tc = False
            final_fr = None
            done = False
            async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        done = True
                        continue
                    try:
                        obj = json.loads(payload)
                    except Exception:
                        continue
                    for c in obj.get("choices", []):
                        delta = c.get("delta") or {}
                        for tc in delta.get("tool_calls") or []:
                            saw_tc = True
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                name = fn["name"]
                            if fn.get("arguments"):
                                args += fn["arguments"]
                        if c.get("finish_reason"):
                            final_fr = c["finish_reason"]

            parsed = None
            try:
                parsed = json.loads(args) if args else None
            except Exception:
                parsed = None

            checks["stream emits delta.tool_calls"] = saw_tc
            checks["reassembled function name == get_weather"] = name == "get_weather"
            checks["reassembled arguments parse to {city~Tokyo}"] = (
                isinstance(parsed, dict) and "tokyo" in str(parsed.get("city", "")).lower())
            checks["terminal finish_reason == tool_calls"] = final_fr == "tool_calls"
            checks["stream ends with [DONE]"] = done
            detail.append(f"name={name!r} args={args!r} fr={final_fr} done={done}")
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
