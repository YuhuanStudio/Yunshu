"""Tool-calling end-to-end gateway gate (in-process ASGI, real engine).

verify_tool_calls.py gates the PARSER in isolation (no GPU). This gate covers the
full request→response assembly the parser feeds into: tool system-prompt
injection, model emission, parse, and OpenAI tool_calls envelope. Deterministic
at temperature 0 (probed: a clear weather query emits the call 3/3 times).

  - finish_reason == "tool_calls"
  - message.tool_calls present; each has a non-empty id, type=="function", and a
    function with a name and an arguments STRING that parses to JSON
  - the assembled call is correct: get_weather(city ~ "Tokyo")
  - content is null/empty when tool_calls are present (OpenAI contract)
  - tool SELECTION: given two tools, the weather query picks get_weather (not
    the distractor)

Run: PYTHONPATH=. uv run python scripts/verify_tool_calls_e2e.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen3.5-0.8B-MLX-bf16")


def _tool(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


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

    tools = [
        _tool("get_weather", "Get current weather for a city",
              {"city": {"type": "string"}}, ["city"]),
        _tool("send_email", "Send an email to a recipient",
              {"to": {"type": "string"}, "body": {"type": "string"}}, ["to", "body"]),
    ]

    checks: dict[str, bool] = {}
    detail: list[str] = []
    try:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120) as client:
            r = await client.post("/v1/chat/completions", json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "What's the weather in Tokyo right now? Use the tool."}],
                "tools": tools, "tool_choice": "auto",
                "temperature": 0.0, "max_tokens": 80,
            })
            checks["HTTP 200"] = r.status_code == 200
            if r.status_code != 200:
                detail.append(f"status={r.status_code} body={r.text[:200]}")
            else:
                d = r.json()
                ch = d["choices"][0]
                msg = ch.get("message") or {}
                tcs = msg.get("tool_calls") or []
                checks["finish_reason == tool_calls"] = ch.get("finish_reason") == "tool_calls"
                checks["tool_calls present"] = len(tcs) >= 1
                if tcs:
                    tc = tcs[0]
                    fn = tc.get("function") or {}
                    args_str = fn.get("arguments")
                    parsed = None
                    try:
                        parsed = json.loads(args_str) if isinstance(args_str, str) else None
                    except Exception:
                        parsed = None
                    checks["call envelope (id/type/function name)"] = (
                        bool(tc.get("id")) and tc.get("type") == "function" and bool(fn.get("name")))
                    checks["arguments is a JSON string"] = isinstance(args_str, str) and parsed is not None
                    checks["correct tool selected (get_weather, city~Tokyo)"] = (
                        fn.get("name") == "get_weather"
                        and isinstance(parsed, dict)
                        and "tokyo" in str(parsed.get("city", "")).lower())
                    detail.append(f"call={fn.get('name')}({args_str!r}) id={tc.get('id')!r}")
                # content must be null/empty when tool_calls present
                checks["content empty when tool_calls present"] = not (msg.get("content") or "").strip()
                detail.append(f"fr={ch.get('finish_reason')} n_calls={len(tcs)} content={ (msg.get('content') or '')[:30]!r}")
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
