"""GPU-queued real HTTP head/reranker parity. CPU dry-run covered by unit tests."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

PAIRS = [
    ("What is the capital of China?", "The capital of China is Beijing."),
    (
        "What is the capital of China?",
        "Gravity attracts two bodies towards each other.",
    ),
    ("What is the capital of China?", "Paris is the capital of France."),
    ("What is the capital of China?", "北京是中國的首都。"),
]
TEXTS = [
    "I loved this movie!",
    "This was an awful boring film.",
    "A surprising and enjoyable story.",
]


def compare(got, ref, tolerance=0.003):
    if len(got) != len(ref) or not got:
        raise ValueError("Missing scores or row count mismatch")
    flat_g = [v for row in got for v in row] if isinstance(got[0], list) else got
    flat_r = [v for row in ref for v in row] if isinstance(ref[0], list) else ref
    if len(flat_g) != len(flat_r) or not all(math.isfinite(v) for v in flat_g + flat_r):
        raise ValueError("Invalid score shape or non-finite scores")
    gap = max(abs(a - b) for a, b in zip(flat_g, flat_r, strict=True))

    def rank(row):
        return sorted(range(len(row)), key=lambda i: row[i], reverse=True)

    same = (
        all(rank(a) == rank(b) for a, b in zip(got, ref, strict=True))
        if isinstance(got[0], list)
        else rank(got) == rank(ref)
    )
    return {
        "max_abs_error": gap,
        "same_ranking": same,
        "passed": gap <= tolerance and same,
        "tolerance": tolerance,
    }


async def run(model_dir, reference_path):
    import httpx

    from yunshu_engine.scoring_engine import TextScoringEngine
    from yunshu_gateway.engine import set_engine
    from yunshu_gateway.main import create_app

    ref = json.loads(Path(reference_path).read_text())
    if ref.get("complete") is not True:
        raise ValueError("Incomplete oracle")
    engine = TextScoringEngine(model_dir)
    await engine.start()
    set_engine(engine)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            if engine.is_reranker:
                body = {
                    "model": model_dir,
                    "text_1": [a for a, b in PAIRS],
                    "text_2": [b for a, b in PAIRS],
                }
                r = await client.post("/v1/score", json=body)
                r.raise_for_status()
                got = [x["score"] for x in r.json()["data"]]
                verdict = compare(got, ref["scores"])
                if not verdict["passed"]:
                    return {**verdict, "scores": got}
                rr = await client.post(
                    "/v1/rerank",
                    json={
                        "model": model_dir,
                        "query": PAIRS[0][0],
                        "documents": [b for a, b in PAIRS],
                        "top_n": 2,
                    },
                )
                rr.raise_for_status()
                results = rr.json()["results"]
                order = sorted(range(len(got)), key=lambda i: got[i], reverse=True)[:2]
                if [x["index"] for x in results] != order or any(
                    abs(x["relevance_score"] - got[x["index"]]) > 1e-6 for x in results
                ):
                    raise ValueError("rerank/score inconsistency")
                # Scalar-to-list broadcast must preserve one score per document.
                rb = await client.post(
                    "/v1/score", json={**body, "text_1": PAIRS[0][0]}
                )
                rb.raise_for_status()
                if [x["score"] for x in rb.json()["data"]] != got:
                    raise ValueError("Broadcast scores differ")
            else:
                r = await client.post(
                    "/v1/classify", json={"model": model_dir, "input": TEXTS}
                )
                r.raise_for_status()
                got = [x["probs"] for x in r.json()["data"]]
                verdict = compare(got, ref["scores"])
            return {**verdict, "scores": got, "kind": engine.kind, "model": model_dir}
    finally:
        set_engine(None)
        await engine.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        result = compare([0.8, 0.2], [0.8, 0.2])
    else:
        result = asyncio.run(run(a.model, a.reference))
    with open(a.out, "w") as f:
        f.write(json.dumps({**result, "complete": True}) + "\n")
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
