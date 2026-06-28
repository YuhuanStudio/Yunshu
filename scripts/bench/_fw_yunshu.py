import asyncio
import json
import logging
import os
import time

logging.basicConfig(level=logging.ERROR)
MODEL=os.environ["YBENCH_MODEL"]; MT=64
P="Topic: "+("Explain how photosynthesis works in detail. "*40)
def pr(i): return f"[{i}] "+P
async def main():
    from yunshu_engine.batched_engine import BatchedEngine
    e=BatchedEngine(model_name=MODEL); await e.start()
    async def chat(i,mt):
        t=time.perf_counter()
        o=await e.chat(messages=[{"role":"user","content":pr(i)}],max_tokens=mt,temperature=0.0,enable_thinking=False)
        return time.perf_counter()-t, getattr(o,"completion_tokens",0)
    await chat(999,2)
    # single: TTFT (mt=1 min3) + decode tps (mt=64)
    t1=min([(await chat(0,1))[0] for _ in range(3)])
    t64,ct=await chat(0,MT)
    dec=(ct-1)/(t64-t1) if t64>t1 and ct>1 else 0
    out={"fw":os.environ["FW"],"model":os.path.basename(MODEL.rstrip("/")),"ttft_ms":round(t1*1000,1),"decode_tps":round(dec,1),"batch":{}}
    # batched N=8,16,32
    for N in [8,16,32]:
        t0=time.perf_counter()
        res=await asyncio.gather(*[chat(i,MT) for i in range(N)])
        wall=time.perf_counter()-t0; toks=sum(r[1] for r in res)
        out["batch"][str(N)]=round(toks/wall,1) if wall else 0
    print("@@FW@@ "+json.dumps(out),flush=True)
    import contextlib
    with contextlib.suppress(Exception): await e.stop()
asyncio.run(main())
