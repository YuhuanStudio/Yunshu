"""Independent process memory benchmark — one framework per subprocess."""
import json
import subprocess
import sys
import time
import psutil

MODEL = "/Users/yuhuan/Documents/Yunshu/models/Qwen3.5-9B-MLX-bf16"

SCRIPTS = {
    "mlx-lm": f"""
import time, psutil
from mlx_lm.utils import load as load_model
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

def rss(): return psutil.Process().memory_info().rss / 1048576

rss0 = rss()
model, tokenizer = load_model("{MODEL}")
mx.synchronize()
mx.clear_cache()
rss_load = rss()

sampler = make_sampler(temp=0.0)
prompt = "Write a short essay. " * 5
ids = mx.array(tokenizer.encode(prompt))

# Warmup
for _ in generate_step(ids, model, max_tokens=16, sampler=sampler):
    pass
mx.synchronize()
mx.clear_cache()
rss_warm = rss()

# Gen
t0 = time.perf_counter()
tokens = []
for tok, _ in generate_step(ids, model, max_tokens=128, sampler=sampler):
    tokens.append(int(tok))
mx.synchronize()
mx.clear_cache()
dt = time.perf_counter() - t0
rss_gen = rss()

import json
print(json.dumps({{"rss_load": round(rss_load), "rss_warmup": round(rss_warm), "rss_gen": round(rss_gen), "tokens": len(tokens), "dt_s": round(dt, 2), "tok_s": round(len(tokens)/dt, 1)}}))
""",
    "yunshu": f"""
import asyncio, time, json, psutil
def rss(): return psutil.Process().memory_info().rss / 1048576

async def main():
    rss0 = rss()
    from yunshu_engine.batched_engine import BatchedEngine
    engine = BatchedEngine(model_name="{MODEL}")
    await engine.start()
    rss_load = rss()

    prompt = "Write a short essay. " * 5
    await engine.generate(prompt=prompt, max_tokens=16, temperature=0.0)
    rss_warm = rss()

    t0 = time.perf_counter()
    r = await engine.generate(prompt=prompt, max_tokens=128, temperature=0.0)
    dt = time.perf_counter() - t0
    rss_gen = rss()
    await engine.stop()
    print(json.dumps({{"rss_load": round(rss_load), "rss_warmup": round(rss_warm), "rss_gen": round(rss_gen), "tokens": r.completion_tokens, "dt_s": round(dt, 2), "tok_s": round(r.completion_tokens/dt, 1), "ttft_ms": r.ttft_ms}}))

asyncio.run(main())
""",
    "vllm-mlx": f"""
import asyncio, time, json, sys, psutil
sys.path.insert(0, "/Users/yuhuan/Documents/Yunshu/reference/vllm-mlx")
def rss(): return psutil.Process().memory_info().rss / 1048576

async def main():
    rss0 = rss()
    from vllm_mlx.engine.simple import SimpleEngine
    engine = SimpleEngine(model_name="{MODEL}")
    engine._is_mllm = False
    await engine.start()
    rss_load = rss()

    prompt = "Write a short essay. " * 5
    await engine.generate(prompt, max_tokens=16, temperature=0.0)
    rss_warm = rss()

    t0 = time.perf_counter()
    r = await engine.generate(prompt, max_tokens=128, temperature=0.0)
    dt = time.perf_counter() - t0
    rss_gen = rss()
    text = r.text if hasattr(r, 'text') else str(r)
    n_tok = getattr(r, 'completion_tokens', 0) or len(engine.tokenizer.encode(text))
    await engine.stop()
    print(json.dumps({{"rss_load": round(rss_load), "rss_warmup": round(rss_warm), "rss_gen": round(rss_gen), "tokens": n_tok, "dt_s": round(dt, 2), "tok_s": round(n_tok/dt, 1)}}))

asyncio.run(main())
""",
    "omlx": f"""
import asyncio, time, json, sys, psutil
sys.path.insert(0, "/Users/yuhuan/Documents/Yunshu/reference/omlx")
def rss(): return psutil.Process().memory_info().rss / 1048576

async def main():
    rss0 = rss()
    from omlx.engine import BatchedEngine
    engine = BatchedEngine(model_name="{MODEL}")
    await engine.start()
    rss_load = rss()

    prompt = "Write a short essay. " * 5
    await engine.generate(prompt, max_tokens=16, temperature=0.0)
    rss_warm = rss()

    t0 = time.perf_counter()
    r = await engine.generate(prompt, max_tokens=128, temperature=0.0)
    dt = time.perf_counter() - t0
    rss_gen = rss()
    text = r.text if hasattr(r, 'text') else str(r)
    n_tok = getattr(r, 'completion_tokens', 0) or len(engine.tokenizer.encode(text))
    await engine.stop()
    print(json.dumps({{"rss_load": round(rss_load), "rss_warmup": round(rss_warm), "rss_gen": round(rss_gen), "tokens": n_tok, "dt_s": round(dt, 2), "tok_s": round(n_tok/dt, 1)}}))

asyncio.run(main())
""",
}

results = {}
for name, script in SCRIPTS.items():
    print(f"  {name}...", flush=True)
    try:
        proc = subprocess.run(
            ["uv", "run", "python", "-c", script],
            capture_output=True, text=True, timeout=300,
            cwd="/Users/yuhuan/Documents/Yunshu",
        )
        output = proc.stdout.strip().split("\n")[-1]
        data = json.loads(output)
        results[name] = data
        print(f"    RSS: {data['rss_load']}MB (load) → {data['rss_warmup']}MB (warmup) → {data['rss_gen']}MB (gen) | {data['tok_s']} tok/s", flush=True)
    except Exception as e:
        print(f"    FAILED: {e}", flush=True)
        if proc:
            print(f"    stderr: {proc.stderr[-200:]}", flush=True)

print(f"\n{'='*80}")
print(f"  MEMORY BENCHMARK (independent processes, fair comparison)")
print(f"{'='*80}")
print(f"  {'Framework':<14} {'RSS load':>10} {'RSS warmup':>12} {'RSS gen':>10} {'tok/s':>10}")
print(f"  {'─'*14} {'─'*10} {'─'*12} {'─'*10} {'─'*10}")

best_mem = float('inf')
for name in ["mlx-lm", "yunshu", "vllm-mlx", "omlx"]:
    d = results.get(name, {})
    rss_gen = d.get("rss_gen", 0)
    if rss_gen and rss_gen < best_mem:
        best_mem = rss_gen

for name in ["mlx-lm", "yunshu", "vllm-mlx", "omlx"]:
    d = results.get(name, {})
    if not d:
        print(f"  {name:<14} {'FAILED':>10}")
        continue
    rss_l = d.get("rss_load", "?")
    rss_w = d.get("rss_warmup", "?")
    rss_g = d.get("rss_gen", 0)
    tps = d.get("tok_s", "?")
    mark = " ★" if rss_g and rss_g <= best_mem * 1.02 else ""
    print(f"  {name:<14} {rss_l:>8}MB {rss_w:>10}MB {rss_g:>8}MB{mark} {tps:>10}")

print(f"{'='*80}")
