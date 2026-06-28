"""Anthropic count_tokens endpoint gate (in-process ASGI, real engine).

/v1/messages/count_tokens returns {"input_tokens": N} without generating. This
gate checks the endpoint exists, returns a positive count, and that the count
agrees with the input_tokens a real /v1/messages call reports for the same input
(the count must be a faithful prefill estimate, not a constant).

  - HTTP 200; input_tokens > 0
  - count matches the actual /v1/messages usage.input_tokens (exactly or ±2)
  - a longer prompt yields a strictly larger count (monotonic, not constant)

Run: PYTHONPATH=. uv run python scripts/verify_count_tokens.py
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
            short = {"model": MODEL, "messages": [{"role": "user", "content": "Hello there."}]}
            long = {"model": MODEL, "messages": [{"role": "user", "content":
                    "Hello there. " + "Please summarize the history of computing in great detail. " * 6}]}

            rc = await client.post("/v1/messages/count_tokens", json=short)
            checks["count_tokens: HTTP 200"] = rc.status_code == 200
            if rc.status_code != 200:
                detail.append(f"status={rc.status_code} body={rc.text[:200]}")
            else:
                n_short = rc.json().get("input_tokens")
                checks["count_tokens: input_tokens > 0"] = isinstance(n_short, int) and n_short > 0

                # actual message call usage for the SAME short input
                rm = await client.post("/v1/messages", json=dict(short, max_tokens=4))
                actual = (rm.json().get("usage") or {}).get("input_tokens") if rm.status_code == 200 else None
                checks["count_tokens: matches real usage (±2)"] = (
                    isinstance(actual, int) and abs(actual - n_short) <= 2)

                # longer prompt → strictly larger count (not a constant)
                rl = await client.post("/v1/messages/count_tokens", json=long)
                n_long = rl.json().get("input_tokens") if rl.status_code == 200 else None
                checks["count_tokens: monotonic (longer > shorter)"] = (
                    isinstance(n_long, int) and n_long > n_short)
                detail.append(f"short={n_short} actual={actual} long={n_long}")
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
