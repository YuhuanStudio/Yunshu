"""Scoring-router HTTP endpoints gate (in-process ASGI, real engine).

verify_embeddings_scoring.py exercises the engine primitives (embed/pool) + the
cosine logic directly. This gate covers the actual HTTP ROUTES — request models,
auth, response envelopes — for /v1/embeddings, /v1/score, /v1/rerank,
/v1/classify, which the engine gate does not touch.

  - /v1/embeddings: object=="list", data has one embedding vector per input,
    each a non-empty list of floats; base64 encoding_format also works
  - /v1/score: returns a numeric similarity score
  - /v1/rerank: results sorted by relevance_score, on-topic doc ranked first
  - /v1/classify: returns label scores, the correct sentiment label on top

Run: PYTHONPATH=. uv run python scripts/verify_scoring_endpoints.py
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
            # ── /v1/embeddings ───────────────────────────────────────────────
            r = await client.post("/v1/embeddings", json={
                "model": MODEL, "input": ["hello world", "goodbye world"]})
            ok200 = r.status_code == 200
            checks["/v1/embeddings: HTTP 200"] = ok200
            if ok200:
                d = r.json()
                data = d.get("data") or []
                vecs_ok = (d.get("object") == "list" and len(data) == 2
                           and all(isinstance(x.get("embedding"), list) and x["embedding"] for x in data))
                checks["/v1/embeddings: 2 float vectors"] = vecs_ok
                detail.append(f"embeddings: n={len(data)} dim={len(data[0]['embedding']) if data else 0}")
                # base64 format
                rb = await client.post("/v1/embeddings", json={
                    "model": MODEL, "input": "hi", "encoding_format": "base64"})
                b64_ok = rb.status_code == 200 and isinstance(
                    (rb.json().get("data") or [{}])[0].get("embedding"), str)
                checks["/v1/embeddings: base64 format"] = b64_ok

            # ── /v1/score ────────────────────────────────────────────────────
            r = await client.post("/v1/score", json={
                "model": MODEL, "text_1": "a cat on a mat", "text_2": "a feline on a rug"})
            ok200 = r.status_code == 200
            checks["/v1/score: HTTP 200 + numeric score"] = ok200 and isinstance(
                (r.json().get("data") or [{}])[0].get("score"), (int, float))
            if ok200:
                detail.append(f"score={ (r.json().get('data') or [{}])[0].get('score')}")

            # ── /v1/rerank ───────────────────────────────────────────────────
            r = await client.post("/v1/rerank", json={
                "model": MODEL, "query": "What is the capital of France?",
                "documents": ["Paris is the capital of France.",
                              "Photosynthesis happens in plant leaves."]})
            ok200 = r.status_code == 200
            checks["/v1/rerank: HTTP 200"] = ok200
            if ok200:
                res = r.json().get("results") or []
                # sorted descending + top result is the on-topic (index 0) doc
                scores = [x.get("relevance_score") for x in res]
                checks["/v1/rerank: sorted desc + on-topic first"] = (
                    bool(res) and scores == sorted(scores, reverse=True) and res[0].get("index") == 0)
                detail.append(f"rerank top_index={res[0].get('index') if res else None} scores={[round(s,3) for s in scores]}")

            # ── /v1/classify ─────────────────────────────────────────────────
            r = await client.post("/v1/classify", json={
                "model": MODEL, "input": "I absolutely loved this, fantastic!",
                "labels": ["positive sentiment", "negative sentiment"]})
            ok200 = r.status_code == 200
            checks["/v1/classify: HTTP 200"] = ok200
            if ok200:
                res = r.json().get("results") or []
                checks["/v1/classify: correct label on top"] = (
                    bool(res) and res[0].get("label") == "positive sentiment")
                detail.append(f"classify top={res[0].get('label')!r} score={round(res[0].get('score',0),3) if res else None}")
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
