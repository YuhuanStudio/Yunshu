"""Context-window overflow gate (in-process ASGI, real engine).

A prompt that exceeds the model's max context window must be rejected with a
clean 400 (not a 500 or a crash). This gate confirms the guard fires with an
informative message, and that a normal prompt is unaffected.

  - a prompt exceeding max_ctx → HTTP 400 with a "too long" / context message
  - a normal short prompt → HTTP 200 (control: the guard is not over-eager)

The under-window-but-over-prefill-budget case (the Wave 634 prefill guard) is
covered separately + cheaply by verify_prefill_guard.py (across all protocols).

Run: PYTHONPATH=. uv run python scripts/verify_context_overflow.py
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
    from yunshu_gateway.streaming import get_max_context_window

    eng = BatchedEngine(model_name=MODEL)
    await eng.start()
    set_engine(eng)

    max_ctx = get_max_context_window(MODEL, eng) or 32768

    checks: dict[str, bool] = {}
    detail: list[str] = []
    try:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120) as client:
            # over the window: ~1.2x max_ctx tokens ("word " ~= 1 token each)
            over = "word " * int(max_ctx * 1.2)
            r = await client.post("/v1/chat/completions", json={
                "model": MODEL, "messages": [{"role": "user", "content": over}], "max_tokens": 4})
            checks["over-context → HTTP 400 (not 5xx/crash)"] = r.status_code == 400
            msg = ""
            try:
                msg = str((r.json().get("error") or {}).get("message") or r.json())
            except Exception:
                msg = r.text
            checks["400 message mentions length/context"] = any(
                k in msg.lower() for k in ("too long", "context", "exceed", "max"))
            detail.append(f"max_ctx={max_ctx} over_status={r.status_code} msg={msg[:90]!r}")

            # control: a normal prompt is fine
            r2 = await client.post("/v1/chat/completions", json={
                "model": MODEL, "messages": [{"role": "user", "content": "Say hi."}],
                "max_tokens": 8, "temperature": 0.0})
            checks["normal prompt → HTTP 200 (guard not over-eager)"] = r2.status_code == 200
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
