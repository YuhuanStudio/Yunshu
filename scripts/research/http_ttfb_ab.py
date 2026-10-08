"""CPU-only HTTP-layer latency probe: time to first SSE byte of a streaming chat request.

The real gateway app (all middleware) runs in-process over httpx ASGITransport with a
BatchedEngine subclass whose generate_stream yields the first token immediately, so the
number is framework + gateway overhead only.  Run it once per environment (for example an
old and a new fastapi lock) and compare with ``compare``; arms must alternate in batches.

    python scripts/research/http_ttfb_ab.py run --n 50 --out a.json
    python scripts/research/http_ttfb_ab.py compare old.json new.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarize(xs):
    return {
        "n": len(xs),
        "median_ms": statistics.median(xs),
        "p90_ms": pct(xs, 0.9),
        "mean_ms": statistics.fmean(xs),
    }


async def measure(n, warmup=10):
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    import fastapi
    import httpx
    import starlette

    from yunshu_engine.batched_engine import BatchedEngine, GenerationOutput
    from yunshu_gateway.engine import set_engine
    from yunshu_gateway.main import create_app

    class Fast(BatchedEngine):
        async def stream_chat(self, *a, **k):
            yield GenerationOutput(
                new_text="hi", text="hi", prompt_tokens=3, completion_tokens=1
            )
            yield GenerationOutput(
                new_text="!",
                text="hi!",
                prompt_tokens=3,
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
            )

    engine = Fast()
    engine._model = object()
    engine._loaded = True
    engine.model_name = "test-model"
    engine._running = True
    set_engine(engine)
    app = create_app()
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "max_tokens": 8,
    }
    times = []
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        for i in range(warmup + n):
            t0 = time.perf_counter()
            async with c.stream("POST", "/v1/chat/completions", json=body) as r:
                if r.status_code != 200:
                    raise SystemExit(f"status {r.status_code}")
                first = None
                async for chunk in r.aiter_raw():
                    if chunk and first is None:
                        first = time.perf_counter() - t0
                        if b"data:" not in chunk:
                            raise SystemExit(f"first bytes are not SSE: {chunk[:80]!r}")
                    # drain fully so every request completes
            if first is None:
                raise SystemExit("no SSE bytes received")
            if i >= warmup:
                times.append(first * 1000)
    set_engine(None)
    return {
        "fastapi": fastapi.__version__,
        "starlette": starlette.__version__,
        "times_ms": times,
    }


def compare(paths):
    arms = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        arms.setdefault(
            f"fastapi {d['fastapi']}/starlette {d['starlette']}", []
        ).extend(d["times_ms"])
    rows = {k: summarize(v) for k, v in arms.items()}
    for k, v in rows.items():
        print(
            f"{k}: n={v['n']} median={v['median_ms']:.3f} ms "
            f"p90={v['p90_ms']:.3f} ms mean={v['mean_ms']:.3f} ms"
        )
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--n", type=int, default=50)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        res = asyncio.run(measure(a.n))
        Path(a.out).write_text(json.dumps(res))
        print(json.dumps(summarize(res["times_ms"])))
    else:
        compare(a.files)


if __name__ == "__main__":
    main()
