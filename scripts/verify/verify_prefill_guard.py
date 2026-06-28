"""Optional prefill-cap coverage across protocols (in-process ASGI).

YUNSHU_MAX_PREFILL_TOKENS is an OPT-IN operator cap (OFF by default — prompts up
to the context window prefill fine, just slowly; a 200k prefill peaks ~12GB/167s,
there is no OOM). When set, it lets a multi-tenant/latency-sensitive operator
stop one huge prompt from monopolizing the single MLX executor. This gate
confirms the cap is honored on ALL three generation entry points — OpenAI chat,
Anthropic /v1/messages, Responses /v1/responses (the latter two previously had no
prompt-size validation at all).

It sets YUNSHU_MAX_PREFILL_TOKENS to a tiny value and sends a small prompt that
exceeds it — the cap rejects (413) before any prefill, so the test is fast.

  - over-cap prompt → clean 413 on chat, Anthropic, and Responses
  - a prompt under the cap → 200 on each (guard not over-eager)

Run: PYTHONPATH=. uv run python scripts/verify_prefill_guard.py
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
    os.environ["YUNSHU_MAX_PREFILL_TOKENS"] = "12"  # tiny cap → small prompt trips it, no OOM

    import httpx

    from yunshu_engine.batched_engine import BatchedEngine
    from yunshu_gateway.engine import set_engine
    from yunshu_gateway.main import create_app

    eng = BatchedEngine(model_name=MODEL)
    await eng.start()
    set_engine(eng)

    BIG = "Tell me about the entire history of computing in great detail, please."  # >12 tokens
    SMALL = "Hi"  # <12 tokens

    checks: dict[str, bool] = {}
    detail: list[str] = []
    try:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=60) as client:
            async def chat(text):
                return await client.post("/v1/chat/completions", json={
                    "model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": 8})

            async def anthropic(text):
                return await client.post("/v1/messages", json={
                    "model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": text}]})

            async def responses(text):
                return await client.post("/v1/responses", json={
                    "model": MODEL, "input": text, "max_output_tokens": 8})

            for name, fn in (("chat", chat), ("Anthropic", anthropic), ("Responses", responses)):
                over = await fn(BIG)
                under = await fn(SMALL)
                checks[f"{name}: over-cap → 413"] = over.status_code == 413
                checks[f"{name}: under-cap → 200 (not over-eager)"] = under.status_code == 200
                detail.append(f"{name}: over={over.status_code} under={under.status_code}")
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
