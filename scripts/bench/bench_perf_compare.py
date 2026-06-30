#!/usr/bin/env python3
"""Performance benchmark: Yunshu vs mlx-lm vs vllm-mlx vs omlx.
Metrics: throughput (tok/s), TTFT, prefill latency, memory.
"""
import asyncio
import gc
import sys
import time
from pathlib import Path

ROOT = Path(".")
REF_DIR = ROOT / "reference"
MODEL_PATH = str(ROOT / "models" / "Qwen3.5-4B-MLX-bf16")
LOG = ROOT / "bench" / "results" / "perf_compare.log"

sys.path.insert(0, str(REF_DIR / "mlx-lm"))
sys.path.insert(0, str(ROOT / "python"))


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
        f.flush()
    print(msg, flush=True)


def get_rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1048576
    except Exception:
        return 0.0


def cleanup():
    gc.collect()
    import mlx.core as mx
    mx.synchronize()
    mx.clear_cache()


PROMPT_SHORT = "What is the capital of France?"
PROMPT_MEDIUM = "Write a detailed essay about the history of computing. " * 5
PROMPT_LONG = "Write a detailed essay about the history of computing. " * 20


def bench_mlxlm():
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    log("  Loading model...")
    model, tokenizer = load(MODEL_PATH)
    sampler = make_sampler(temp=0.0)
    rss = get_rss_mb()
    log(f"  Model loaded. RSS: {rss:.0f} MB")

    results = {"framework": "mlx-lm", "rss_mb": rss}

    for label, prompt, max_tok in [
        ("short_prompt_short_gen", PROMPT_SHORT, 128),
        ("medium_prompt_medium_gen", PROMPT_MEDIUM, 256),
        ("long_prompt_long_gen", PROMPT_LONG, 512),
    ]:
        # Warmup
        for _ in stream_generate(model, tokenizer, prompt, max_tokens=32, sampler=sampler):
            pass

        # TTFT + throughput
        ttfts = []
        total_toks = []
        total_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            first_tok_time = None
            n_tok = 0
            for _resp in stream_generate(model, tokenizer, prompt, max_tokens=max_tok, sampler=sampler):
                if first_tok_time is None:
                    first_tok_time = time.perf_counter()
                n_tok += 1
            t_end = time.perf_counter()
            ttft = first_tok_time - t0
            gen_time = t_end - first_tok_time if first_tok_time else t_end - t0
            ttfts.append(ttft)
            total_toks.append(n_tok)
            total_times.append(gen_time)

        avg_ttft = sum(ttfts) / len(ttfts) * 1000  # ms
        avg_tps = sum(t / s for t, s in zip(total_toks, total_times, strict=False)) / len(total_toks)
        results[label] = {
            "ttft_ms": round(avg_ttft, 1),
            "tok_per_s": round(avg_tps, 1),
            "avg_tokens": round(sum(total_toks) / len(total_toks)),
        }
        log(f"    {label}: TTFT={avg_ttft:.1f}ms, {avg_tps:.1f} tok/s, avg {sum(total_toks)//len(total_toks)} tok")

    del model, tokenizer
    cleanup()
    return results


async def bench_yunshu():
    from yunshu_engine.batched_engine import BatchedEngine

    log("  Loading model...")
    engine = BatchedEngine(model_name=MODEL_PATH)
    await engine.start()
    rss = get_rss_mb()
    log(f"  Model loaded. RSS: {rss:.0f} MB")

    results = {"framework": "yunshu", "rss_mb": rss}

    for label, prompt, max_tok in [
        ("short_prompt_short_gen", PROMPT_SHORT, 128),
        ("medium_prompt_medium_gen", PROMPT_MEDIUM, 256),
        ("long_prompt_long_gen", PROMPT_LONG, 512),
    ]:
        # Warmup
        await engine.generate(prompt=prompt, max_tokens=32, temperature=0.0)

        # TTFT + throughput via streaming
        ttfts = []
        total_toks = []
        total_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            first_tok_time = None
            n_tok = 0
            async for _chunk in engine.stream_generate(prompt=prompt, max_tokens=max_tok, temperature=0.0):
                if first_tok_time is None:
                    first_tok_time = time.perf_counter()
                n_tok += 1
            t_end = time.perf_counter()
            ttft = first_tok_time - t0 if first_tok_time else t_end - t0
            gen_time = t_end - first_tok_time if first_tok_time else t_end - t0
            ttfts.append(ttft)
            total_toks.append(n_tok)
            total_times.append(gen_time)

        avg_ttft = sum(ttfts) / len(ttfts) * 1000
        avg_tps = sum(t / s for t, s in zip(total_toks, total_times, strict=False)) / len(total_toks)
        results[label] = {
            "ttft_ms": round(avg_ttft, 1),
            "tok_per_s": round(avg_tps, 1),
            "avg_tokens": round(sum(total_toks) / len(total_toks)),
        }
        log(f"    {label}: TTFT={avg_ttft:.1f}ms, {avg_tps:.1f} tok/s, avg {sum(total_toks)//len(total_toks)} tok")

    await engine.stop()
    cleanup()
    return results


async def bench_vllm_mlx():
    sys.path.insert(0, str(REF_DIR / "vllm-mlx"))
    from vllm_mlx.engine.simple import SimpleEngine

    log("  Loading model...")
    engine = SimpleEngine(model_name=MODEL_PATH)
    engine._is_mllm = False
    await engine.start()
    rss = get_rss_mb()
    log(f"  Model loaded. RSS: {rss:.0f} MB")

    results = {"framework": "vllm-mlx", "rss_mb": rss}

    for label, prompt, max_tok in [
        ("short_prompt_short_gen", PROMPT_SHORT, 128),
        ("medium_prompt_medium_gen", PROMPT_MEDIUM, 256),
        ("long_prompt_long_gen", PROMPT_LONG, 512),
    ]:
        # Warmup
        await engine.generate(prompt=prompt, max_tokens=32, temperature=0.0)

        # TTFT + throughput via streaming
        ttfts = []
        total_toks = []
        total_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            first_tok_time = None
            n_tok = 0
            async for _chunk in engine.stream_generate(prompt=prompt, max_tokens=max_tok, temperature=0.0):
                if first_tok_time is None:
                    first_tok_time = time.perf_counter()
                n_tok += 1
            t_end = time.perf_counter()
            ttft = first_tok_time - t0 if first_tok_time else t_end - t0
            gen_time = t_end - first_tok_time if first_tok_time else t_end - t0
            ttfts.append(ttft)
            total_toks.append(n_tok)
            total_times.append(gen_time)

        avg_ttft = sum(ttfts) / len(ttfts) * 1000
        avg_tps = sum(t / s for t, s in zip(total_toks, total_times, strict=False)) / len(total_toks)
        results[label] = {
            "ttft_ms": round(avg_ttft, 1),
            "tok_per_s": round(avg_tps, 1),
            "avg_tokens": round(sum(total_toks) / len(total_toks)),
        }
        log(f"    {label}: TTFT={avg_ttft:.1f}ms, {avg_tps:.1f} tok/s, avg {sum(total_toks)//len(total_toks)} tok")

    await engine.stop()
    cleanup()
    return results


async def bench_omlx():
    sys.path.insert(0, str(REF_DIR / "omlx"))
    from omlx.engine import BatchedEngine

    log("  Loading model...")
    engine = BatchedEngine(model_name=MODEL_PATH)
    await engine.start()
    rss = get_rss_mb()
    log(f"  Model loaded. RSS: {rss:.0f} MB")

    results = {"framework": "omlx", "rss_mb": rss}

    for label, prompt, max_tok in [
        ("short_prompt_short_gen", PROMPT_SHORT, 128),
        ("medium_prompt_medium_gen", PROMPT_MEDIUM, 256),
        ("long_prompt_long_gen", PROMPT_LONG, 512),
    ]:
        # Warmup
        await engine.generate(prompt=prompt, max_tokens=32, temperature=0.0)

        # TTFT + throughput via streaming
        ttfts = []
        total_toks = []
        total_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            first_tok_time = None
            n_tok = 0
            async for _chunk in engine.stream_generate(prompt=prompt, max_tokens=max_tok, temperature=0.0):
                if first_tok_time is None:
                    first_tok_time = time.perf_counter()
                n_tok += 1
            t_end = time.perf_counter()
            ttft = first_tok_time - t0 if first_tok_time else t_end - t0
            gen_time = t_end - first_tok_time if first_tok_time else t_end - t0
            ttfts.append(ttft)
            total_toks.append(n_tok)
            total_times.append(gen_time)

        avg_ttft = sum(ttfts) / len(ttfts) * 1000
        avg_tps = sum(t / s for t, s in zip(total_toks, total_times, strict=False)) / len(total_toks)
        results[label] = {
            "ttft_ms": round(avg_ttft, 1),
            "tok_per_s": round(avg_tps, 1),
            "avg_tokens": round(sum(total_toks) / len(total_toks)),
        }
        log(f"    {label}: TTFT={avg_ttft:.1f}ms, {avg_tps:.1f} tok/s, avg {sum(total_toks)//len(total_toks)} tok")

    await engine.stop()
    cleanup()
    return results


async def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")

    log("=" * 70)
    log("  Performance Benchmark: Yunshu vs mlx-lm vs vllm-mlx vs omlx")
    log("  Model: Qwen3.5-9B bf16 | temp=0.0")
    log("=" * 70)

    all_results = []

    # mlx-lm (sync)
    log("\n─" * 70)
    log("  MLX-LM (stream_generate)")
    log("─" * 70)
    all_results.append(bench_mlxlm())

    # Yunshu
    log("\n─" * 70)
    log("  Yunshu (BatchedEngine)")
    log("─" * 70)
    all_results.append(await bench_yunshu())

    # vllm-mlx
    log("\n─" * 70)
    log("  vllm-mlx (SimpleEngine)")
    log("─" * 70)
    all_results.append(await bench_vllm_mlx())

    # omlx
    log("\n─" * 70)
    log("  omlx (BatchedEngine)")
    log("─" * 70)
    all_results.append(await bench_omlx())

    # Summary
    log("\n" + "=" * 70)
    log("  SUMMARY")
    log("=" * 70)

    for scenario in ["short_prompt_short_gen", "medium_prompt_medium_gen", "long_prompt_long_gen"]:
        log(f"\n  ── {scenario} ──")
        log(f"  {'Framework':<12} {'TTFT':>10} {'tok/s':>10} {'Memory':>10}")
        log(f"  {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 10}")
        for r in all_results:
            s = r.get(scenario, {})
            ttft = f"{s.get('ttft_ms', 0):.1f}ms"
            tps = f"{s.get('tok_per_s', 0):.1f}"
            mem = f"{r.get('rss_mb', 0):.0f} MB"
            log(f"  {r['framework']:<12} {ttft:>10} {tps:>10} {mem:>10}")

    log("\nDONE")


if __name__ == "__main__":
    asyncio.run(main())
