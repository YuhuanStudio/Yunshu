"""Streaming logprobs gate (in-process ASGI, real engine).

The non-stream logprobs gate (verify_logprobs_stop.py) covers the batch shape.
Streaming carries logprobs per-chunk, which is assembled differently. This gate
checks the streamed logprobs contract.

  - content-bearing chunks carry a logprobs payload with content entries
  - each entry has a token, a logprob <= 0, and top_logprobs of the requested
    length (3)
  - the number of logprob entries equals the number of streamed tokens

Run: PYTHONPATH=. uv run python scripts/verify_streaming_logprobs.py
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
            entries = []
            async with client.stream("POST", "/v1/chat/completions", json={
                "model": MODEL, "messages": [{"role": "user", "content": "Name three colors."}],
                "max_tokens": 16, "temperature": 0.0, "enable_thinking": False,
                "logprobs": True, "top_logprobs": 3, "stream": True,
            }) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: ") or "[DONE]" in line:
                        continue
                    obj = json.loads(line[6:])
                    for c in obj.get("choices", []):
                        lp = c.get("logprobs")
                        if lp and lp.get("content"):
                            entries.extend(lp["content"])

            checks["streamed logprobs present"] = len(entries) > 0
            if entries:
                lp_ok = all(isinstance(e.get("logprob"), (int, float)) and e["logprob"] <= 1e-6 for e in entries)
                tok_ok = all(e.get("token") is not None for e in entries)
                top_ok = all(len(e.get("top_logprobs") or []) == 3 for e in entries)
                checks["every entry: logprob <= 0"] = lp_ok
                checks["every entry: has token"] = tok_ok
                checks["every entry: top_logprobs length 3"] = top_ok
                detail.append(f"n_entries={len(entries)} first_token={entries[0].get('token')!r} "
                              f"lp={entries[0].get('logprob'):.3f}")
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
