import asyncio
import json
import logging
import os
import time

logging.basicConfig(level=logging.ERROR)
MODEL=os.environ["OMLX_MODEL"]; N=int(os.environ.get("CONC_N","8")); MT=int(os.environ.get("CONC_MT","64"))
def prompt(i): return f"Topic {i}: " + ("Explain how photosynthesis works in detail. "* 40) + f" (variant {i})"
async def main():
    name=os.path.basename(MODEL.rstrip("/"))
    try:
        from omlx.engine.batched import BatchedEngine
        e=BatchedEngine(model_name=MODEL); await e.start()
    except Exception as ex:
        print("@@CONC@@ "+json.dumps({"model":name,"status":f"LOAD FAIL: {str(ex)[:60]}"})); return
    async def one(i):
        t=time.perf_counter()
        o=await e.chat(messages=[{"role":"user","content":prompt(i)}],max_tokens=MT,temperature=0.0)
        return time.perf_counter()-t, getattr(o,"completion_tokens",0)
    await one(999)
    t0=time.perf_counter()
    res=await asyncio.gather(*[one(i) for i in range(N)])
    wall=time.perf_counter()-t0
    toks=sum(r[1] for r in res)
    out={"model":name,"N":N,"wall_s":round(wall,2),"total_tok":toks,"agg_tps":round(toks/wall,1) if wall else 0,"engine":"omlx"}
    print("@@CONC@@ "+json.dumps(out))
    import contextlib
    with contextlib.suppress(Exception): await e.stop()
asyncio.run(main())
