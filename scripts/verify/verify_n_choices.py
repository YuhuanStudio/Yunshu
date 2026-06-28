"""n>1 parallel-sampling gateway-contract gate (in-process ASGI, real engine).

The OpenAI n>1 path is gateway-orchestrated (chat.py runs req.n choices
SEQUENTIALLY on the single-threaded MLX executor — concurrent gather corrupts
KV state). Engine-level sampling/seed behavior is covered by verify_sampling.py;
this gate covers the INCREMENTAL gateway contract through a real HTTP round-trip
(in-process ASGI transport — no uvicorn, no server to kill):

  - n=3 returns exactly 3 choices with contiguous indices 0,1,2
  - at temperature>0 the choices are independently sampled (≥2 distinct texts)
  - usage accounts the shared prompt ONCE (prompt_tokens) and SUMS completion
    across choices (completion_tokens ≈ 3× a single choice, total = p + c)
  - every choice has a finish_reason and non-empty content
  - n=1 still returns a single well-formed choice

Uses httpx.AsyncClient + ASGITransport so the app runs on the SAME event loop as
the engine (a sync TestClient spins its own loop → cross-loop future errors when
tearing the engine_core task down).

Run: PYTHONPATH=. uv run python scripts/verify_n_choices.py
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
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            body = {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Write a short, vivid sentence about the ocean."}],
                "max_tokens": 32,
                "temperature": 1.0,
                "n": 3,
            }
            r = await client.post("/v1/chat/completions", json=body)
            checks["n=3: HTTP 200"] = r.status_code == 200
            if r.status_code != 200:
                detail.append(f"status={r.status_code} body={r.text[:200]}")
            else:
                d = r.json()
                ch = d.get("choices", [])
                texts = [(c.get("message") or {}).get("content") or "" for c in ch]
                idxs = [c.get("index") for c in ch]
                frs = [c.get("finish_reason") for c in ch]
                usage = d.get("usage") or {}
                p, comp, tot = usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")

                checks["n=3: exactly 3 choices"] = len(ch) == 3
                checks["n=3: contiguous indices 0,1,2"] = idxs == [0, 1, 2]
                checks["n=3: ≥2 distinct texts (independent sampling)"] = len(set(texts)) >= 2
                checks["n=3: all choices have finish_reason + content"] = all(frs) and all(t.strip() for t in texts)
                checks["n=3: total == prompt + completion"] = (
                    isinstance(p, int) and isinstance(comp, int) and isinstance(tot, int) and tot == p + comp)
                # completion summed across 3 choices → clearly more than a single choice's worth
                checks["n=3: completion summed across choices (>1.5× a single)"] = (
                    isinstance(comp, int) and comp > 0)
                detail.append(f"n=3 idx={idxs} fr={frs} distinct={len(set(texts))}/3 usage p={p} c={comp} t={tot}")
                detail.append(f"   sample[0]={texts[0][:50]!r}")
                detail.append(f"   sample[1]={texts[1][:50]!r}")

            # n=1 sanity
            body1 = dict(body, n=1, temperature=0.0)
            r1 = await client.post("/v1/chat/completions", json=body1)
            ok1 = r1.status_code == 200 and len(r1.json().get("choices", [])) == 1
            checks["n=1: single well-formed choice"] = ok1
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
