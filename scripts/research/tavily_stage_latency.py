"""Per-stage /tavily/search latency (Server-Timing: serp / fetch / ir / generate).

--fixture (default): in-process ASGI, fake provider and pages; measures the engine's own
overhead (CPU, no network). --live: real built-in metasearch and real page fetches
(needs internet; manual measurement only, never run by unit tests). CPU only: no model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

DEPTHS = ("ultra-fast", "fast", "basic", "advanced")


def parse_timing(header: str) -> dict[str, float]:
    out = {}
    for part in header.split(","):
        name, _, dur = part.strip().partition(";dur=")
        if name and dur:
            out[name] = float(dur)
    return out


def summarize(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    names = sorted({k for s in samples for k in s})
    out = {}
    for name in names:
        vals = sorted(s[name] for s in samples if name in s)
        out[name] = {
            "p50_ms": round(statistics.median(vals), 2),
            "max_ms": round(vals[-1], 2),
            "n": len(vals),
        }
    return out


async def run(args) -> list[dict]:
    import httpx

    if args.live:
        from fastapi import FastAPI

        from yunshu_gateway.routers.tavily import router

        app = FastAPI()
        app.include_router(router)
    else:
        from tavily_fixture import create_fixture

        app = create_fixture()
    rows = []
    queries = args.queries or ["python asyncio tutorial", "paris weather today"]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://local",
        headers={"Authorization": "Bearer tvly-local"},
        timeout=60,
    ) as client:
        for di, depth in enumerate(DEPTHS):
            if depth not in args.depths:
                continue
            samples, totals = [], []
            for i in range(args.reps):
                # Distinct query per (depth, rep): search and page caches must not be shared.
                q = queries[(di * args.reps + i) % len(queries)]
                t0 = time.perf_counter()
                r = await client.post(
                    "/tavily/search",
                    json={"query": q, "search_depth": depth, "max_results": 5},
                )
                total = (time.perf_counter() - t0) * 1000
                if r.status_code != 200:
                    raise SystemExit(f"{depth}: HTTP {r.status_code} {r.text[:200]}")
                samples.append(parse_timing(r.headers.get("server-timing", "")))
                totals.append(total)
            rows.append(
                {
                    "depth": depth,
                    "reps": args.reps,
                    "total_p50_ms": round(statistics.median(totals), 2),
                    "stages": summarize(samples),
                    "live": args.live,
                }
            )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live", action="store_true")
    p.add_argument("--depths", nargs="*", default=list(DEPTHS))
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--queries", nargs="*")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = asyncio.run(run(args))
    rows.append({"complete": True})
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
