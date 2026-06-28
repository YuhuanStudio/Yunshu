import asyncio
import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.ERROR)
sys.path.insert(0, "reference/vllm-mlx")
MODEL=os.environ["YBENCH_MODEL"]; MT=64
P="Topic: "+("Explain how photosynthesis works in detail. "*40)
async def main():
    name=os.path.basename(MODEL.rstrip("/"))
    try:
        from vllm_mlx.engine.simple import SimpleEngine
        e=SimpleEngine(model_name=MODEL); e._is_mllm=False; await e.start()
        tok=e.tokenizer
    except Exception as ex:
        traceback.print_exc()
        print("@@FW@@ "+json.dumps({"fw":"vllm-mlx","model":name,"status":f"LOAD FAIL: {str(ex)[:50]}"})); return
    def prompt(i):
        return tok.apply_chat_template([{"role":"user","content":f"[{i}] "+P}],add_generation_prompt=True,tokenize=False)
    async def gen(i,mt):
        t=time.perf_counter()
        r=await e.generate(prompt(i),max_tokens=mt,temperature=0.0)
        return time.perf_counter()-t, getattr(r,"completion_tokens",0)
    try:
        await gen(999,2)
        t1=min([(await gen(0,1))[0] for _ in range(3)])
        t64,ct=await gen(0,MT); dec=(ct-1)/(t64-t1) if t64>t1 and ct>1 else 0
        out={"fw":"vllm-mlx","model":name,"ttft_ms":round(t1*1000,1),"decode_tps":round(dec,1),"batch":{}}
        for N in [8,16,32]:
            t0=time.perf_counter(); res=await asyncio.gather(*[gen(i,MT) for i in range(N)]); wall=time.perf_counter()-t0
            out["batch"][str(N)]=round(sum(r[1] for r in res)/wall,1) if wall else 0
    except Exception as ex:
        traceback.print_exc()
        out={"fw":"vllm-mlx","model":name,"status":f"RUN FAIL: {str(ex)[:50]}"}
    print("@@FW@@ "+json.dumps(out),flush=True)
    with contextlib.suppress(Exception): await e.stop()
asyncio.run(main())
