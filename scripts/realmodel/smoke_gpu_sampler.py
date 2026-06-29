"""real-model smoke: on-GPU temp>0 sampler (YUNSHU_GPU_SAMPLER=1) —
coherent output, no n>1 collapse, and faster (removes the per-token GPU sync).
Run: PYTHONPATH=. uv run python scripts/realmodel/smoke_gpu_sampler.py
"""
import asyncio, os, time
MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

async def _gen(gpu: bool, seed):
    os.environ["YUNSHU_GPU_SAMPLER"] = "1" if gpu else "0"
    from yunshu_engine.batched_engine import BatchedEngine
    eng = BatchedEngine(MODEL)
    await eng.start()
    msgs = [{"role": "user", "content": "Write one sentence about the ocean."}]
    await eng.chat(msgs, max_tokens=8, temperature=0.8, seed=seed)  # warmup
    t0 = time.perf_counter()
    r = await eng.chat(msgs, max_tokens=64, temperature=0.8, top_p=0.95, seed=seed)
    dt = time.perf_counter() - t0
    tps = r.completion_tokens / dt if dt else 0
    txt = r.text
    await eng.stop()
    return txt, round(tps, 1)

async def main():
    g_txt, g_tps = await _gen(True, 1)
    n_txt, n_tps = await _gen(False, 1)
    print(f"[GPU sampler]   {g_tps} tok/s :: {g_txt[:70]!r}")
    print(f"[numpy sampler] {n_tps} tok/s :: {n_txt[:70]!r}")
    assert g_txt.strip(), "GPU sampler produced empty output"
    # n>1 no-collapse: two different seeds must differ
    a, _ = await _gen(True, 11)
    b, _ = await _gen(True, 22)
    print(f"[GPU n>1 distinct] seed11!=seed22: {a != b}")
    assert a != b, "GPU sampler collapsed (different seeds gave identical text)"
    print(f"\nspeedup (GPU/numpy) = {g_tps/n_tps:.2f}x")
    print("GPU sampler smoke: PASS")

asyncio.run(main())
