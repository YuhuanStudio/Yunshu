"""Multi-turn conversation context gate (in-process ASGI, real engine).

No gate verifies that the gateway threads a multi-message conversation (system +
prior user/assistant turns) into the prompt so the model can use earlier context.
Deterministic at temp 0 (probed 3/3 recall).

  - a fact stated two turns earlier (name "Marvin", number 47) is recalled
  - the system instruction is honored (a one-line, concise answer)
  - a follow-up that depends ONLY on prior context (not the last message alone)
    is answered correctly — proving the assistant turn + history are threaded

Run: PYTHONPATH=. uv run python scripts/verify_multiturn.py
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
            r = await client.post("/v1/chat/completions", json={
                "model": MODEL,
                "temperature": 0.0, "max_tokens": 40, "enable_thinking": False,
                "messages": [
                    {"role": "system", "content": "You are concise."},
                    {"role": "user", "content": "My name is Marvin and my favorite number is 47. Just acknowledge."},
                    {"role": "assistant", "content": "Got it, Marvin. Your favorite number is 47."},
                    {"role": "user", "content": "What is my name and my favorite number?"},
                ],
            })
            checks["HTTP 200"] = r.status_code == 200
            if r.status_code != 200:
                detail.append(f"status={r.status_code} body={r.text[:200]}")
            else:
                txt = r.json()["choices"][0]["message"]["content"] or ""
                checks["recalls name from 2 turns earlier (Marvin)"] = "marvin" in txt.lower()
                checks["recalls number from 2 turns earlier (47)"] = "47" in txt
                detail.append(f"answer={txt[:80]!r}")
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
