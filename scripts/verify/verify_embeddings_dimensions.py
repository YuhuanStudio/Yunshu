"""Embeddings `dimensions` (Matryoshka truncation) gate (in-process ASGI).

The OpenAI `dimensions` param requests a truncated embedding. This gate checks
the /v1/embeddings route honors it.

  - default request returns the model's full embedding dimension
  - dimensions=256 returns exactly 256 floats
  - the truncated vector is L2-normalized (norm ~ 1.0) and aligns with the full
    vector's leading slice (Matryoshka: cosine of trunc vs normalized full[:256]
    is high) — i.e. it's a real prefix truncation, not a random projection
  - dimensions larger than the model dim is rejected (4xx) or clamped, not 500

Run: PYTHONPATH=. uv run python scripts/verify_embeddings_dimensions.py
"""
from __future__ import annotations

import asyncio
import math
import os
import sys

MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen3.5-0.8B-MLX-bf16")


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def _cos(a, b):
    na, nb = _norm(a), _norm(b)
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


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
            async def emb(**kw):
                r = await client.post("/v1/embeddings", json={"model": MODEL, "input": "the sky is blue", **kw})
                return r

            full_r = await emb()
            full = full_r.json()["data"][0]["embedding"]
            full_dim = len(full)
            checks["default returns full embedding dim"] = full_dim >= 256

            trunc_r = await emb(dimensions=256)
            checks["dimensions=256: HTTP 200"] = trunc_r.status_code == 200
            if trunc_r.status_code == 200:
                trunc = trunc_r.json()["data"][0]["embedding"]
                checks["dimensions=256: returns exactly 256 floats"] = len(trunc) == 256
                checks["dimensions=256: L2-normalized (norm~1)"] = abs(_norm(trunc) - 1.0) < 0.05
                # Matryoshka: truncated ~ normalized leading slice of full
                lead = full[:256]
                checks["dimensions=256: aligns with full[:256] (cos>0.99)"] = _cos(trunc, lead) > 0.99
                detail.append(f"full_dim={full_dim} trunc_norm={_norm(trunc):.3f} cos(trunc,full[:256])={_cos(trunc, lead):.4f}")

            # oversize dimensions must not 500
            over_r = await emb(dimensions=full_dim + 100000)
            checks["oversize dimensions handled (not 5xx)"] = over_r.status_code < 500
            detail.append(f"oversize status={over_r.status_code}")
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
