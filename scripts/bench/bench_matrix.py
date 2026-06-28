"""FULL combinatorial matrix: (framework+technique config) x model x setting.
Each cell = TTFT, decode tok/s, throughput@N=8/16/32 (N hard-capped at 32, safe).
All SEQUENTIAL. oMLX via OMLX_PYTHON native venv. One giant table out.

Run: PYTHONPATH=. OMLX_PYTHON=.venvs/omlx/bin/python uv run python scripts/bench_matrix.py
"""
import json
import os
import re
import subprocess
import sys

REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS=[("Q3.5-0.8B","./models/Qwen3.5-0.8B-MLX-bf16","hybrid"),
        ("Q3.5-2B","./models/Qwen3.5-2B-MLX-bf16","hybrid"),
        ("Q2.5-3B","./models/Qwen2.5-3B-Instruct-bf16","std"),
        ("gemma4","./models/gemma-4-e4b-it-bf16","std"),
        ("Q3.5-9B4b","./models/Qwen3.5-9B-MLX-4bit","hybrid")]
# (label, script, extra-env, model-env-var, python)
SELF=sys.executable
OMLXPY=os.environ.get("OMLX_PYTHON",SELF)
CONFIGS=[
 ("yunshu-fast",        "_fw_yunshu.py", {"FW":"yunshu-fast"}, "YBENCH_MODEL", SELF, "."),
 ("yunshu-loop",        "_fw_yunshu.py", {"FW":"yunshu-loop","YUNSHU_ENGINE_LOOP":"1"}, "YBENCH_MODEL", SELF, "."),
 ("yunshu-loop+kvq4",   "_fw_yunshu.py", {"FW":"y","YUNSHU_ENGINE_LOOP":"1","YUNSHU_KV_QUANT_BITS":"4"}, "YBENCH_MODEL", SELF, "."),
 ("yunshu-loop+hybpfx", "_fw_yunshu.py", {"FW":"y","YUNSHU_ENGINE_LOOP":"1","YUNSHU_HYBRID_PREFIX":"1"}, "YBENCH_MODEL", SELF, "."),
 ("mlx-lm",             "_fw_mlxlm.py",  {}, "YBENCH_MODEL", SELF, "."),
 ("vllm-mlx",           "_fw_vllmmlx.py",{}, "YBENCH_MODEL", SELF, "."),
 ("oMLX",               "_fw_omlx.py",   {}, "OMLX_MODEL", OMLXPY, "./reference/omlx"),
]
def run(cfg, model_path):
    label,script,env_extra,mvar,py,pp=cfg
    env=dict(os.environ, PYTHONPATH=pp, **env_extra); env[mvar]=model_path
    try:
        p=subprocess.run([py,f"scripts/{script}"],env=env,cwd=REPO,capture_output=True,text=True,timeout=1500)
    except subprocess.TimeoutExpired: return {"status":"TIMEOUT"}
    m=re.search(r"@@FW@@ (\{.*\})", p.stdout+p.stderr)
    if m:
        try: return json.loads(m.group(1))
        except Exception: return {"status":"PARSE"}
    return {"status":"NORESULT"}
def main():
    grid={}
    for mlabel,mpath,_arch in MODELS:
        if not os.path.exists(os.path.join(REPO,mpath)): continue
        for cfg in CONFIGS:
            label=cfg[0]
            print(f"  cell: {mlabel} x {label} ...",flush=True)
            grid[(mlabel,label)]=run(cfg,mpath)
        json.dump({f"{k[0]}|{k[1]}":v for k,v in grid.items()}, open("/tmp/matrix.json","w"), indent=2)
    # render
    def c(r,k):
        if not r or ("status" in r and k not in r and "ttft_ms" not in r): return (r or {}).get("status","-")[:6]
        if k=="ttft": return f"{r.get('ttft_ms','-')}"
        if k=="dec": return f"{r.get('dec','')or r.get('decode_tps','-')}"
        return f"{r.get('batch',{}).get(k,'-')}"
    print("\n"+"="*120)
    print("FULL MATRIX: config x model — cells = TTFT(ms) / decode(t/s) / tp@8 / tp@16 / tp@32")
    print("="*120)
    cfgs=[c[0] for c in CONFIGS]; ms=[m[0] for m in MODELS]
    for metric,key in [("TTFT ms","ttft"),("decode t/s","dec"),("throughput@8","8"),("throughput@16","16"),("throughput@32","32")]:
        print(f"\n--- {metric} ---")
        print("config".ljust(20)+"".join(m.ljust(11) for m in ms))
        for cfg in cfgs:
            print(cfg.ljust(20)+"".join(str(c(grid.get((m,cfg),{}),key)).ljust(11) for m in ms))
    print("\nfull json -> /tmp/matrix.json")
main()
