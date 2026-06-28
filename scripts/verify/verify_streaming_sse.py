"""Streaming SSE chat-completions protocol gate (in-process ASGI, real engine).

The streaming path is a large protocol surface that's easy to get subtly wrong
(chunk envelope, role-first delta, terminal finish_reason, [DONE], usage). No
other gate covers the ASSEMBLED streamed output. Through an in-process ASGI
round-trip with a real engine, this asserts:

  - chunks are `data: {...}` SSE with object == "chat.completion.chunk"
  - the first delta carries role == "assistant"
  - exactly one chunk carries a finish_reason (the terminal content chunk)
  - the stream ends with `data: [DONE]`
  - stream_options.include_usage yields a usage object with prompt/completion
  - CORRECTNESS: at temperature 0, the concatenated streamed deltas EQUAL the
    non-streamed completion text (streaming must not change the output)

Run: PYTHONPATH=. uv run python scripts/verify_streaming_sse.py
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
            msgs = [{"role": "user", "content": "Name three primary colors in one short line."}]
            base = {"model": MODEL, "messages": msgs, "max_tokens": 40,
                    "temperature": 0.0, "enable_thinking": False}

            # ── streamed ────────────────────────────────────────────────────
            sse_lines: list[str] = []
            async with client.stream("POST", "/v1/chat/completions",
                                     json=dict(base, stream=True,
                                               stream_options={"include_usage": True})) as resp:
                checks["stream: HTTP 200"] = resp.status_code == 200
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        sse_lines.append(line[6:])

            done = sse_lines and sse_lines[-1].strip() == "[DONE]"
            checks["stream: ends with [DONE]"] = bool(done)
            payloads = [json.loads(x) for x in sse_lines if x.strip() and x.strip() != "[DONE]"]

            envelope_ok = all(p.get("object") == "chat.completion.chunk" for p in payloads if p.get("choices"))
            checks["stream: chunk object == chat.completion.chunk"] = bool(payloads) and envelope_ok

            # first delta with content/role
            first_role = None
            assembled = ""
            fr_count = 0
            usage_obj = None
            for p in payloads:
                if p.get("usage"):
                    usage_obj = p["usage"]
                for c in p.get("choices", []):
                    delta = c.get("delta") or {}
                    if first_role is None and delta.get("role"):
                        first_role = delta["role"]
                    if delta.get("content"):
                        assembled += delta["content"]
                    if c.get("finish_reason"):
                        fr_count += 1
            checks["stream: first delta role == assistant"] = first_role == "assistant"
            checks["stream: exactly one finish_reason chunk"] = fr_count == 1
            checks["stream: include_usage yields usage"] = (
                isinstance(usage_obj, dict) and usage_obj.get("completion_tokens", 0) > 0)
            checks["stream: assembled text non-empty"] = bool(assembled.strip())
            detail.append(f"chunks={len(payloads)} fr_count={fr_count} usage={usage_obj} role={first_role}")

            # ── non-streamed comparison (correctness at temp 0) ──────────────
            r2 = await client.post("/v1/chat/completions", json=dict(base, stream=False))
            ns_text = (r2.json()["choices"][0]["message"]["content"]) if r2.status_code == 200 else None
            checks["stream: assembled == non-stream text (temp0)"] = (
                ns_text is not None and assembled == ns_text)
            detail.append(f"streamed={assembled[:55]!r}")
            detail.append(f"nonstrm ={(ns_text or '')[:55]!r}")
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
