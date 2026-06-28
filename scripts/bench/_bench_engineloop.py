"""Engine-loop (radix) cold-vs-reuse micro-bench. Emits @@RESULTEL@@ <json>.

Same cold/reuse protocol as the other benches so numbers are comparable.
Run via scripts/bench_all.py (sequentially, never concurrently).
"""
import asyncio
import json
import logging
import os
import time

logging.basicConfig(level=logging.ERROR)
os.environ.setdefault("YUNSHU_ENGINE_LOOP", "1")
os.environ.setdefault("YUNSHU_KV_OFFLOAD", "1")
os.environ.setdefault("YUNSHU_KV_OFFLOAD_THRESHOLD", "0.02")
MODEL = os.environ["YBENCH_MODEL"]
QUERY = "List the first 6 even numbers, comma separated."


def _doc(t):
    return f"Reference document {t}. " + (
        "Photosynthesis converts sunlight into chemical energy stored in glucose. " * 110)


async def main():
    from yunshu_engine.batched_engine import BatchedEngine
    e = BatchedEngine(model_name=MODEL)
    await e.start()
    P = _doc("PRIMARY")

    async def chat(s, u, mt):
        t = time.perf_counter()
        o = await e.chat(messages=[{"role": "system", "content": s},
                                   {"role": "user", "content": u}],
                         max_tokens=mt, temperature=0.0, enable_thinking=False)
        return time.perf_counter() - t, o

    await chat(_doc("W"), "hi", 2)
    colds = []
    for i in range(3):
        d, _ = await chat(_doc(f"C{i}"), QUERY, 1)
        colds.append(d)
    cold = min(colds)
    _, rref = await chat(_doc("R"), QUERY, 16)
    ref = rref.text.strip()
    await chat(P, "Summarize briefly.", 8)
    rs = []
    out = None
    for _ in range(3):
        d, out = await chat(P, QUERY, 1)
        rs.append(d)
    reuse = min(rs)
    _, ot = await chat(P, QUERY, 16)
    res = {
        "model": os.path.basename(MODEL.rstrip("/")),
        "cold_ms": round(cold * 1000, 1),
        "reuse_ms": round(reuse * 1000, 1),
        "speedup": round(cold / reuse, 2) if reuse else 0,
        "cached": int(getattr(out, "cached_tokens", 0)),
        "lossless": ot.text.strip() == ref,
    }
    print("@@RESULTEL@@ " + json.dumps(res))
    import contextlib
    with contextlib.suppress(Exception):
        await e.stop()


if __name__ == "__main__":
    asyncio.run(main())
