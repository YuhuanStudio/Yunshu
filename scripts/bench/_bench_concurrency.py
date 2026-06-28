"""Concurrent-throughput bench (continuous batching). Emits @@CONC@@ <json>.
Fires N concurrent distinct requests, measures wall time + aggregate decode tok/s
+ mean TTFT-ish. Yunshu: set YUNSHU_ENGINE_LOOP=1 for the batching path.
"""
import asyncio
import json
import logging
import os
import time

logging.basicConfig(level=logging.ERROR)
MODEL=os.environ["YBENCH_MODEL"]; N=int(os.environ.get("CONC_N","8")); MT=int(os.environ.get("CONC_MT","64"))
def prompt(i): return f"Topic {i}: " + ("Explain how photosynthesis works in detail. "* 40) + f" (variant {i})"
async def main():
    from yunshu_engine.batched_engine import BatchedEngine
    e=BatchedEngine(model_name=MODEL); await e.start()
    async def one(i):
        t=time.perf_counter()
        o=await e.chat(messages=[{"role":"user","content":prompt(i)}],max_tokens=MT,temperature=0.0,enable_thinking=False)
        return time.perf_counter()-t, getattr(o,"completion_tokens",0)
    # warmup
    await one(999)
    t0=time.perf_counter()
    res=await asyncio.gather(*[one(i) for i in range(N)])
    wall=time.perf_counter()-t0
    toks=sum(r[1] for r in res)
    res={"model":os.path.basename(MODEL.rstrip("/")),"N":N,"wall_s":round(wall,2),
         "total_tok":toks,"agg_tps":round(toks/wall,1) if wall else 0,
         "loop":os.environ.get("YUNSHU_ENGINE_LOOP","0")}
    print("@@CONC@@ "+json.dumps(res))
    import contextlib
    with contextlib.suppress(Exception): await e.stop()
asyncio.run(main())
