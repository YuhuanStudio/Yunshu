"""Anthropic /v1/messages protocol gate (in-process ASGI, real engine).

A protocol surface entirely distinct from the OpenAI one. Through an in-process
ASGI round-trip with a real engine, asserts:

  Non-streaming envelope:
  - HTTP 200; top-level type=="message", role=="assistant"
  - content is a list of typed blocks with a non-empty {type:"text", text}
  - stop_reason ∈ {end_turn, max_tokens, stop_sequence}
  - usage carries input_tokens > 0 and output_tokens > 0
  - a `system` prompt is accepted (no error)

  Streaming event protocol (distinct from OpenAI SSE deltas):
  - emits message_start … content_block_delta(s) … message_stop
  - the concatenated text_delta content is non-empty

Run: PYTHONPATH=. uv run python scripts/verify_anthropic_messages.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen3.5-0.8B-MLX-bf16")
_VALID_STOP = {"end_turn", "max_tokens", "stop_sequence"}


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
            # ── non-streaming envelope ───────────────────────────────────────
            body = {
                "model": MODEL,
                "max_tokens": 40,
                "system": "You are a concise assistant. Answer in one short sentence.",
                "messages": [{"role": "user", "content": "What color is a clear daytime sky?"}],
                "temperature": 0.0,
            }
            r = await client.post("/v1/messages", json=body)
            checks["msg: HTTP 200 (system accepted)"] = r.status_code == 200
            if r.status_code != 200:
                detail.append(f"status={r.status_code} body={r.text[:200]}")
            else:
                d = r.json()
                content = d.get("content")
                text_blocks = [b for b in (content or []) if isinstance(b, dict) and b.get("type") == "text"]
                usage = d.get("usage") or {}
                checks["msg: type==message, role==assistant"] = (
                    d.get("type") == "message" and d.get("role") == "assistant")
                checks["msg: content has non-empty text block"] = (
                    bool(text_blocks) and bool((text_blocks[0].get("text") or "").strip()))
                checks["msg: stop_reason valid"] = d.get("stop_reason") in _VALID_STOP
                checks["msg: usage input/output tokens > 0"] = (
                    usage.get("input_tokens", 0) > 0 and usage.get("output_tokens", 0) > 0)
                detail.append(f"stop={d.get('stop_reason')} usage={usage} "
                              f"text={(text_blocks[0].get('text') if text_blocks else '')[:50]!r}")

            # ── streaming event protocol ─────────────────────────────────────
            events: list[str] = []
            stream_text = ""
            async with client.stream("POST", "/v1/messages",
                                     json=dict(body, stream=True)) as resp:
                etype = None
                async for line in resp.aiter_lines():
                    if line.startswith("event: "):
                        etype = line[7:].strip()
                        events.append(etype)
                    elif line.startswith("data: "):
                        try:
                            obj = json.loads(line[6:])
                        except Exception:
                            continue
                        delta = obj.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            stream_text += delta.get("text", "")
            checks["stream: has message_start"] = "message_start" in events
            checks["stream: has content_block_delta"] = "content_block_delta" in events
            checks["stream: has message_stop"] = "message_stop" in events
            checks["stream: text_delta content non-empty"] = bool(stream_text.strip())
            detail.append(f"events={events[:8]}{'…' if len(events) > 8 else ''} "
                          f"stream_text={stream_text[:50]!r}")
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
