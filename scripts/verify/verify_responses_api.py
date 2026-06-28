"""Responses API /v1/responses protocol gate (in-process ASGI, real engine).

A third protocol surface (OpenAI Responses), distinct from chat-completions and
Anthropic. Through an in-process ASGI round-trip with a real engine, asserts:

  Non-streaming envelope:
  - HTTP 200; object=="response", status ∈ {completed, incomplete}
  - output is a non-empty list containing a message item (type=="message",
    role=="assistant") whose content carries a non-empty output_text block
  - usage has input_tokens > 0 and output_tokens > 0
  - `instructions` (the Responses system field) is accepted

  Streaming lifecycle (distinct event protocol):
  - emits response.created … response.output_text.delta(s) … a terminal event
    (response.completed, or response.incomplete when max_output_tokens truncates
    — both are spec-correct; which one fires depends on the output length)
  - the concatenated output_text deltas are non-empty

Run: PYTHONPATH=. uv run python scripts/verify_responses_api.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen3.5-0.8B-MLX-bf16")


def _find_output_text(output):
    """Extract the assistant message's output_text from a Responses output list."""
    for item in output or []:
        if item.get("type") == "message" and item.get("role") == "assistant":
            for c in item.get("content") or []:
                if c.get("type") == "output_text":
                    return c.get("text") or ""
    return None


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
                "instructions": "You are concise. Answer in one short sentence.",
                "input": "What is the largest planet in our solar system?",
                "max_output_tokens": 40,
                "temperature": 0.0,
            }
            # ── non-streaming envelope ───────────────────────────────────────
            r = await client.post("/v1/responses", json=body)
            checks["resp: HTTP 200 (instructions accepted)"] = r.status_code == 200
            if r.status_code != 200:
                detail.append(f"status={r.status_code} body={r.text[:200]}")
            else:
                d = r.json()
                txt = _find_output_text(d.get("output"))
                usage = d.get("usage") or {}
                checks["resp: object==response, status valid"] = (
                    d.get("object") == "response" and d.get("status") in ("completed", "incomplete"))
                checks["resp: output has assistant output_text"] = bool(txt and txt.strip())
                checks["resp: usage input/output tokens > 0"] = (
                    usage.get("input_tokens", 0) > 0 and usage.get("output_tokens", 0) > 0)
                detail.append(f"status={d.get('status')} usage={usage} text={(txt or '')[:50]!r}")

            # ── streaming lifecycle ──────────────────────────────────────────
            events: list[str] = []
            stream_text = ""
            async with client.stream("POST", "/v1/responses", json=dict(body, stream=True)) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("event: "):
                        events.append(line[7:].strip())
                    elif line.startswith("data: "):
                        try:
                            obj = json.loads(line[6:])
                        except Exception:
                            continue
                        if obj.get("type") == "response.output_text.delta":
                            stream_text += obj.get("delta", "")
            terminal = {"response.completed", "response.incomplete"} & set(events)
            checks["stream: response.created"] = "response.created" in events
            checks["stream: response.output_text.delta"] = "response.output_text.delta" in events
            checks["stream: terminal event (completed|incomplete)"] = bool(terminal)
            checks["stream: output_text delta non-empty"] = bool(stream_text.strip())
            detail.append(f"events={events[:6]}{'…' if len(events) > 6 else ''} terminal={sorted(terminal)} "
                          f"stream_text={stream_text[:40]!r}")
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
