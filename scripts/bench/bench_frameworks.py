"""IN-PROCESS (internal) throughput bench — every framework, across models,
single-request (TTFT, decode tok/s) + batched (N=8/16/32 aggregate). All SEQUENTIAL,
batch hard-capped at 32. oMLX runs in its native venv via OMLX_PYTHON.

(methodology): the PRIMARY cross-framework comparison is the EXTERNAL one
(scripts/bench/bench_serve.py — every framework's real OpenAI HTTP server; only real
servers are a fair production measure). This internal bench is the OTHER half of the
pair: every framework is measured both internally (here) and externally (serve) so the
report can compute each one's internal-vs-external parity (gateway efficiency →
surfaces middle-layer bugs per framework).

Run: PYTHONPATH=. OMLX_PYTHON=.venvs/omlx/bin/python uv run python scripts/bench/bench_frameworks.py
"""
import json, os, re, subprocess, sys
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS=["./models/Qwen3.5-0.8B-MLX-bf16","./models/Qwen3.5-2B-MLX-bf16",
        "./models/Qwen2.5-3B-Instruct-bf16","./models/gemma-4-e4b-it-bf16"]
# per-framework timeout was 1200s (20min). The real killer was that
# vllm-mlx (CUDA — does NOT run on Apple Silicon) and oMLX (needs a separate venv
# via OMLX_PYTHON) were tried for EVERY model and each hung to that 20-min
# timeout → ~80 min wasted on frameworks that can't run here, so --tier full never
# finished. Now: vllm-mlx/oMLX are OPT-IN (only run when actually available), and
# the timeout is a tight 300s (a single-model bench takes 1-3 min). The default
# (yunshu-fast, yunshu-loop, mlx-lm) completes in ~15-20 min.
_TIMEOUT = int(os.environ.get("YUNSHU_FW_TIMEOUT", "300"))
def run(cmd, env):
    try: p=subprocess.run(cmd,env=env,cwd=REPO,capture_output=True,text=True,timeout=_TIMEOUT)
    except subprocess.TimeoutExpired: return {"status":"TIMEOUT"}
    m=re.search(r"@@FW@@ (\{.*\})", p.stdout+p.stderr)
    if m:
        try: return json.loads(m.group(1))
        except Exception: return {"status":"PARSE ERR"}
    return {"status":"NO RESULT"}
def bench(model):
    # (methodology): this is the IN-PROCESS (internal) leg. Every framework
    # is measured BOTH internally (here) and externally (bench_serve.py, real HTTP).
    # The PRIMARY cross-framework comparison is the external one (only real servers
    # are a fair production measure); the internal numbers exist so the report can
    # compute each framework's INTERNAL-vs-EXTERNAL parity (gateway efficiency →
    # surfaces middle-layer bugs per framework, not just yunshu).
    name=os.path.basename(model.rstrip("/")); base=dict(os.environ,PYTHONPATH=".")
    rows=[]
    print(f"  [{name}] yunshu-fast...",flush=True)
    rows.append(run([sys.executable,"scripts/bench/_fw_yunshu.py"],dict(base,FW="yunshu-fast",YBENCH_MODEL=model)))
    print(f"  [{name}] yunshu-loop...",flush=True)
    rows.append(run([sys.executable,"scripts/bench/_fw_yunshu.py"],dict(base,FW="yunshu-loop",YUNSHU_ENGINE_LOOP="1",YBENCH_MODEL=model)))
    print(f"  [{name}] mlx-lm...",flush=True)
    rows.append(run([sys.executable,"scripts/bench/_fw_mlxlm.py"],dict(base,YBENCH_MODEL=model)))
    if os.path.isdir(os.path.join(REPO,"reference/vllm-mlx")):
        print(f"  [{name}] vllm-mlx...",flush=True)
        rows.append(run([sys.executable,"scripts/bench/_fw_vllmmlx.py"],dict(base,PYTHONPATH=".:./reference/vllm-mlx",YBENCH_MODEL=model)))
    op=os.environ.get("OMLX_PYTHON")
    if op and os.path.exists(op) and os.path.isdir(os.path.join(REPO,"reference/omlx")):
        print(f"  [{name}] oMLX...",flush=True)
        rows.append(run([op,"scripts/bench/_fw_omlx.py"],dict(base,PYTHONPATH="./reference/omlx",OMLX_MODEL=model)))
    return name,rows
def cell(r,k):
    if "status" in r and k not in r: return r["status"][:9]
    if k=="ttft": return f"{r.get('ttft_ms','-')}"
    if k=="dec": return f"{r.get('decode_tps','-')}"
    b=r.get("batch",{}); return f"{b.get(k,'-')}"
def main():
    allr=[]
    for m in MODELS:
        if not os.path.exists(os.path.join(REPO,m)): print("SKIP",m); continue
        print(f"\n>>> {m}",flush=True); allr.append(bench(m))
    W=[14,11,8,9,9,9]
    H=["framework","TTFT ms","dec t/s","batch8","batch16","batch32"]
    print("\n"+"="*70); print("FRAMEWORK COMPARISON (single-req + batched, N<=32, sequential)"); print("="*70)
    for name,rows in allr:
        print(f"\n### {name}")
        print(" ".join(h.ljust(w) for h,w in zip(H,W,strict=False)))
        for r in rows:
            fw=r.get("fw","?")
            print(" ".join(str(c).ljust(w) for c,w in zip([fw,cell(r,"ttft"),cell(r,"dec"),cell(r,"8"),cell(r,"16"),cell(r,"32")],W,strict=False)))
    with open("/tmp/bench_frameworks.json","w") as f: json.dump([{"model":n,"rows":r} for n,r in allr],f,indent=2)
    print("\nfull json -> /tmp/bench_frameworks.json")
main()
