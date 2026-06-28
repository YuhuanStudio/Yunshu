import json
import logging
import os
import time

logging.basicConfig(level=logging.ERROR)
from mlx_lm import batch_generate, load, stream_generate

MODEL=os.environ["YBENCH_MODEL"]; MT=64
def _load(path):
    # Mirror Yunshu's loader: some checkpoints (e.g. gemma-4 e4b) ship extra
    # KV-shared weights the model class omits → "Received N parameters not in
    # model". Those are safe to drop; retry strict=False ONLY for that case so
    # raw mlx-lm gets a fair batched number instead of a bogus NO RESULT.
    try:
        return load(path)
    except ValueError as e:
        if "not in model" not in str(e):
            raise
        from pathlib import Path
        from mlx_lm.utils import load_model, load_tokenizer
        mp = Path(path)
        if not mp.exists():
            from mlx_lm.utils import hf_repo_to_path
            mp = hf_repo_to_path(path)
        ret = load_model(mp, strict=False)
        return (ret[0] if isinstance(ret, tuple) else ret), load_tokenizer(mp)
model,tok=_load(MODEL)
P="Topic: "+("Explain how photosynthesis works in detail. "*40)
def toks(i): return tok.apply_chat_template([{"role":"user","content":f"[{i}] "+P}],add_generation_prompt=True)
for _ in stream_generate(model,tok,toks(0),max_tokens=2): pass
# single
ttfts=[]
for _ in range(3):
    t0=time.perf_counter(); first=None; n=0; tl=t0
    for _r in stream_generate(model,tok,toks(0),max_tokens=MT):
        now=time.perf_counter(); n+=1
        if first is None: first=now-t0
        tl=now
    ttfts.append(first)
t1=min(ttfts); dec=(n-1)/(tl-t0-t1) if (tl-t0)>t1 and n>1 else 0
out={"fw":"mlx-lm","model":os.path.basename(MODEL.rstrip("/")),"ttft_ms":round(t1*1000,1),"decode_tps":round(dec,1),"batch":{}}
for N in [8,16,32]:
    assert N<=32
    prompts=[toks(i) for i in range(N)]
    t0=time.perf_counter(); r=batch_generate(model,tok,prompts,max_tokens=MT,verbose=False); wall=time.perf_counter()-t0
    g=getattr(r,"generation_tokens",None); total=g if isinstance(g,int) else N*MT
    out["batch"][str(N)]=round(total/wall,1) if wall else 0
print("@@FW@@ "+json.dumps(out),flush=True)
