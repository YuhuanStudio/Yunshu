import asyncio, json, logging, os, time
logging.basicConfig(level=logging.ERROR)
MODEL=os.environ["OMLX_MODEL"]; QUERY="List the first 6 even numbers, comma separated."
def doc(t): return f"Reference document {t}. "+("Photosynthesis converts sunlight into chemical energy stored in glucose. "*110)
async def main():
    name=os.path.basename(MODEL.rstrip("/"))
    try:
        from omlx.engine.batched import BatchedEngine
        from omlx.scheduler import SchedulerConfig
        sc=SchedulerConfig(); sc.paged_ssd_cache_dir=os.environ["OMLX_SSD"]; sc.paged_cache_block_size=int(os.environ.get("OMLX_BLK","128"))
        e=BatchedEngine(model_name=MODEL, scheduler_config=sc); await e.start()
    except Exception as ex:
        import traceback; traceback.print_exc()
        print("@@RESULTOMLX@@ "+json.dumps({"model":name,"status":f"LOAD FAILED: {type(ex).__name__}: {str(ex)[:90]}"})); return
    async def chat(s,u,mt):
        t=time.perf_counter()
        o=await e.chat(messages=[{"role":"system","content":s},{"role":"user","content":u}],max_tokens=mt,temperature=0.0)
        return time.perf_counter()-t,o
    try:
        await chat(doc("W"),"hi",2)
        colds=[]
        for i in range(3): d,_=await chat(doc(f"C{i}"),QUERY,1); colds.append(d)
        cold=min(colds)
        _,rref=await chat(doc("R"),QUERY,16); ref=rref.text.strip()
        await chat(P:=doc("PRIMARY"),"Summarize briefly.",8)
        rs=[]
        for _ in range(3): d,_o=await chat(P,QUERY,1); rs.append(d)
        reuse=min(rs)
        _,ot=await chat(P,QUERY,16)
        res={"model":name,"cold_ms":round(cold*1000,1),"reuse_ms":round(reuse*1000,1),
             "speedup":round(cold/reuse,2) if reuse else 0,"lossless":ot.text.strip()==ref,"status":"ok"}
    except Exception as ex:
        import traceback; traceback.print_exc()
        res={"model":name,"status":f"RUN FAILED: {type(ex).__name__}: {str(ex)[:90]}"}
    print("@@RESULTOMLX@@ "+json.dumps(res))
    import contextlib
    with contextlib.suppress(Exception): await e.stop()
asyncio.run(main())
