"""Embeddings + scoring-endpoint semantic gate (direct engine, no gateway).

The /v1/embeddings, /v1/pooling, /v1/score, /v1/rerank, /v1/classify endpoints
all build on engine.embed() / engine.pool() (see routers/scoring.py). This gate
exercises those two engine primitives and the cosine logic the routers use, and
asserts they actually capture SEMANTICS — not just return vectors of the right
shape:

  - embed: a semantically-similar pair scores HIGHER cosine than a dissimilar
    pair (relative ordering — a causal-LM mean-pool has a high cosine floor, so
    we test discrimination, not an absolute threshold).
  - rerank: the on-topic document out-ranks the off-topic one for a query.
  - classify: cosine-to-label picks the correct sentiment label.
  - pool: MEAN / CLS / LAST produce DISTINCT vectors of the right dim.

Run: PYTHONPATH=. uv run python scripts/verify_embeddings_scoring.py
"""
from __future__ import annotations

import asyncio
import math
import os
import sys

MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen3.5-0.8B-MLX-bf16")


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


async def main() -> int:
    if not os.path.exists(MODEL):
        print("SKIP: model not mounted")
        return 0
    from yunshu_engine.batched_engine import BatchedEngine
    eng = BatchedEngine(model_name=MODEL)
    await eng.start()

    checks: dict[str, bool] = {}
    detail: list[str] = []
    try:
        # ── embed: similar pair vs dissimilar pair ──────────────────────────
        texts = [
            "The cat sat on the warm windowsill in the sun.",   # 0
            "A feline rested by the sunny window.",             # 1  ~ 0
            "Quarterly tax filings are due at the end of June.",  # 2  ≠ 0/1
        ]
        embs = await asyncio.get_running_loop().run_in_executor(
            None, eng.embed, texts)
        sim = _cos(embs[0], embs[1])      # near-paraphrase
        dis = _cos(embs[0], embs[2])      # unrelated
        checks["embed: paraphrase cos > unrelated cos"] = sim > dis + 1e-3
        detail.append(f"embed sim={sim:.4f} dis={dis:.4f} (Δ={sim - dis:+.4f}) dim={len(embs[0])}")

        # ── rerank: on-topic doc out-ranks off-topic ────────────────────────
        query = "What is the capital city of France?"
        docs = [
            "Paris is the capital and most populous city of France.",  # on-topic
            "Photosynthesis converts sunlight into chemical energy.",  # off-topic
        ]
        q_emb = (await asyncio.get_running_loop().run_in_executor(
            None, eng.embed, [query]))[0]
        d_embs = await asyncio.get_running_loop().run_in_executor(
            None, eng.embed, docs)
        r_on, r_off = _cos(q_emb, d_embs[0]), _cos(q_emb, d_embs[1])
        checks["rerank: on-topic doc ranks above off-topic"] = r_on > r_off
        detail.append(f"rerank on={r_on:.4f} off={r_off:.4f}")

        # ── classify: pick correct sentiment label ──────────────────────────
        inp = "Absolutely loved it — best purchase I've made all year!"
        labels = ["a positive, happy review", "a negative, angry complaint"]
        i_emb = (await asyncio.get_running_loop().run_in_executor(
            None, eng.embed, [inp]))[0]
        l_embs = await asyncio.get_running_loop().run_in_executor(
            None, eng.embed, labels)
        s_pos, s_neg = _cos(i_emb, l_embs[0]), _cos(i_emb, l_embs[1])
        checks["classify: positive label wins for positive text"] = s_pos > s_neg
        detail.append(f"classify pos={s_pos:.4f} neg={s_neg:.4f}")

        # ── pool: MEAN / CLS / LAST distinct + right dim ────────────────────
        t = ["The quick brown fox jumps over the lazy dog."]
        pm = (await asyncio.get_running_loop().run_in_executor(None, eng.pool, t, "MEAN"))[0]
        pc = (await asyncio.get_running_loop().run_in_executor(None, eng.pool, t, "CLS"))[0]
        pl = (await asyncio.get_running_loop().run_in_executor(None, eng.pool, t, "LAST"))[0]
        same_dim = len(pm) == len(pc) == len(pl) == len(embs[0])
        distinct = _cos(pm, pc) < 0.999 and _cos(pm, pl) < 0.999 and _cos(pc, pl) < 0.999
        checks["pool: MEAN/CLS/LAST distinct + same dim"] = same_dim and distinct
        detail.append(f"pool dim={len(pm)} cos(M,C)={_cos(pm, pc):.3f} cos(M,L)={_cos(pm, pl):.3f}")
    finally:
        await eng.stop()

    for k, v in checks.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    for d in detail:
        print(f"     {d}")
    ok = all(checks.values())
    print(f"RESULT: {sum(checks.values())}/{len(checks)} passed")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
