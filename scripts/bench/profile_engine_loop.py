"""Profile the engine-loop continuous-batching hot path to find the reclaimable
orchestration overhead (the 1.5-2.5x gap vs raw mlx-lm static batch, wave603).

Runs N concurrent requests through the engine-loop, reports aggregate tok/s, GPU
compute-utilization (step vs idle), and a cProfile of the top time consumers.

Run: PYTHONPATH=. YBENCH_MODEL=./models/Qwen3.5-2B-MLX-bf16 \
       uv run python scripts/profile_engine_loop.py
"""
import asyncio
import cProfile
import io
import os
import pstats
import time

os.environ["YUNSHU_ENGINE_LOOP"] = "1"
os.environ.setdefault("YUNSHU_SSD_CACHE", "0")
MODEL = os.environ.get("YBENCH_MODEL", "./models/Qwen3.5-2B-MLX-bf16")
N = int(os.environ.get("PROF_N", "16"))
GEN = int(os.environ.get("PROF_GEN", "120"))


async def _one(engine, i):
    o = await engine.chat(
        messages=[{"role": "user", "content": f"Write a detailed paragraph about topic {i}."}],
        max_tokens=GEN, temperature=0.0, enable_thinking=False)
    return getattr(o, "completion_tokens", 0)


async def run(engine):
    t0 = time.perf_counter()
    counts = await asyncio.gather(*[_one(engine, i) for i in range(N)])
    wall = time.perf_counter() - t0
    return sum(counts), wall


async def main():
    from yunshu_engine.batched_engine import BatchedEngine
    engine = BatchedEngine(model_name=MODEL)
    await engine.start()
    # warmup
    await engine.chat(messages=[{"role": "user", "content": "hi"}], max_tokens=8, temperature=0.0)

    core = getattr(engine, "_engine_core", None)
    # reset util counters if possible
    if core is not None:
        for a in ("_total_step_time_ms", "_total_idle_time_ms"):
            if hasattr(core, a):
                setattr(core, a, 0.0)

    pr = cProfile.Profile()
    pr.enable()
    total_tok, wall = await run(engine)
    pr.disable()

    agg = total_tok / wall if wall else 0
    util = core.get_compute_utilization() if core is not None else -1
    step_ms = getattr(core, "_total_step_time_ms", 0) if core else 0
    idle_ms = getattr(core, "_total_idle_time_ms", 0) if core else 0
    print(f"\n== ENGINE-LOOP PROFILE: {os.path.basename(MODEL)} N={N} gen={GEN} ==")
    print(f"  aggregate: {agg:.1f} tok/s ({total_tok} tok in {wall:.2f}s)")
    print(f"  GPU compute-utilization: {util:.1f}%  (step {step_ms:.0f}ms / idle {idle_ms:.0f}ms)")
    print(f"  => ~{100-util:.0f}% of wall is NON-GPU orchestration (the reclaimable gap)\n")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    # Show the most relevant lines (engine/gateway/asyncio hot path)
    lines = s.getvalue().splitlines()
    print("== top cumulative (filtered to our code + asyncio/detok) ==")
    shown = 0
    for ln in lines:
        if any(k in ln for k in ("yunshu_engine", "detoken", "scheduler", "asyncio",
                                 "queue", "run_in_executor", "generate_step", "sample")):
            print(" ", ln.strip()[:160]); shown += 1
        if shown >= 22:
            break
    await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
