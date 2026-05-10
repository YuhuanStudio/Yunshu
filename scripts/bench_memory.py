"""Independent process memory benchmark v2 — with warmup for Yunshu."""
import json, subprocess, sys, time

MODEL = "/Users/yuhuan/Documents/Yunshu/models/Qwen3.5-4B-MLX-bf16"

SCRIPTS = {
    "mlx-lm": f"""
import time, json, psutil
def rss(): return psutil.Process().memory_info().rss / 1048576

from mlx_lm.utils import load as load_model
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

model, tokenizer = load_model("{MODEL}")
mx.synchronize(); mx.clear_cache()
rss_load = rss()

sampler = make_sampler(temp=0.0)
ids = mx.array(tokenizer.encode("Hi"))
for _ in generate_step(ids, model, max_tokens=1, sampler=sampler):
    break
mx.synchronize(); mx.clear_cache()
rss_warm = rss()

prompt = "Write a short essay. " * 5
ids = mx.array(tokenizer.encode(prompt))
t0 = time.perf_counter()
tokens = []
for tok, _ in generate_step(ids, model, max_tokens=128, sampler=sampler):
    tokens.append(int(tok))
mx.synchronize(); mx.clear_cache()
dt = time.perf_counter() - t0
print(json.dumps({{"rss_load": round(rss_load), "rss_warmup": round(rss_warm), "rss_gen": round(rss()), "tokens": len(tokens), "tok_s": round(len(tokens)/dt, 1)}}))
""",
    "yunshu": f"""
import asyncio, time, json, psutil
def rss(): return psutil.Process().memory_info().rss / 1048576

async def main():
    from yunshu_engine.batched_engine import BatchedEngine
    engine = BatchedEngine(model_name="{MODEL}")
    await engine.start()
    rss_warm = rss()

    prompt = "Write a short essay. " * 5
    t0 = time.perf_counter()
    r = await engine.generate(prompt=prompt, max_tokens=128, temperature=0.0)
    dt = time.perf_counter() - t0
    print(json.dumps({{"rss_warmup": round(rss_warm), "rss_gen": round(rss()), "tokens": r.completion_tokens, "tok_s": round(r.completion_tokens/dt, 1), "ttft_ms": r.ttft_ms}}))
    await engine.stop()

asyncio.run(main())
""",
    "vllm-mlx": f"""
import asyncio, time, json, sys, psutil
sys.path.insert(0, "/Users/yuhuan/Documents/Yunshu/reference/vllm-mlx")
def rss(): return psutil.Process().memory_info().rss / 1048576

async def main():
    from vllm_mlx.engine.simple import SimpleEngine
    engine = SimpleEngine(model_name="{MODEL}")
    engine._is_mllm = False
    await engine.start()
    await engine.generate("Hi", max_tokens=1, temperature=0.0)
    rss_warm = rss()

    prompt = "Write a short essay. " * 5
    t0 = time.perf_counter()
    r = await engine.generate(prompt, max_tokens=128, temperature=0.0)
    dt = time.perf_counter() - t0
    n_tok = getattr(r, 'completion_tokens', 0) or len(engine.tokenizer.encode(getattr(r, 'text', '')))
    print(json.dumps({{"rss_warmup": round(rss_warm), "rss_gen": round(rss()), "tokens": n_tok, "tok_s": round(n_tok/dt, 1)}}))
    await engine.stop()

asyncio.run(main())
""",
    "omlx": f"""
import asyncio, time, json, sys, psutil
sys.path.insert(0, "/Users/yuhuan/Documents/Yunshu/reference/omlx")
def rss(): return psutil.Process().memory_info().rss / 1048576

async def main():
    from omlx.engine import BatchedEngine
    engine = BatchedEngine(model_name="{MODEL}")
    await engine.start()
    await engine.generate("Hi", max_tokens=1, temperature=0.0)
    rss_warm = rss()

    prompt = "Write a short essay. " * 5
    t0 = time.perf_counter()
    r = await engine.generate(prompt, max_tokens=128, temperature=0.0)
    dt = time.perf_counter() - t0
    n_tok = getattr(r, 'completion_tokens', 0) or len(engine.tokenizer.encode(getattr(r, 'text', '')))
    print(json.dumps({{"rss_warmup": round(rss_warm), "rss_gen": round(rss()), "tokens": n_tok, "tok_s": round(n_tok/dt, 1)}}))
    await engine.stop()

asyncio.run(main())
""",
}

print(f"{'='*75}")
print(f"  MEMORY BENCHMARK (independent processes, warmup for all)")
print(f"{'='*75}")

results = {}
for name, script in SCRIPTS.items():
    print(f"  {name}...", end=" ", flush=True)
    proc = subprocess.run(
        ["uv", "run", "python", "-c", script],
        capture_output=True, text=True, timeout=300,
        cwd="/Users/yuhuan/Documents/Yunshu",
    )
    try:
        output = proc.stdout.strip().split("\\n")[-1]
        data = json.loads(output)
        results[name] = data
        print(f"{data.get('rss_gen','?')}MB | {data.get('tok_s','?')} tok/s")
    except Exception as e:
        print(f"FAILED: {e}")
        if proc.stderr:
            print(f"  stderr: {proc.stderr[-200:]}")

print(f"\\n{'─'*75}")
print(f"  {'Framework':<14} {'RSS warmup':>12} {'RSS gen':>10} {'tok/s':>10}")
print(f"  {'─'*14} {'─'*12} {'─'*10} {'─'*10}")
for name in ["yunshu", "mlx-lm", "vllm-mlx", "omlx"]:
    d = results.get(name, {})
    if not d: 
        print(f"  {name:<14} FAILED")
        continue
    print(f"  {name:<14} {d.get('rss_warmup','?'):>10}MB {d.get('rss_gen','?'):>8}MB {d.get('tok_s','?'):>10}")
print(f"{'─'*75}")
