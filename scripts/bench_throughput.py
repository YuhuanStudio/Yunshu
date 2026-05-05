"""Throughput Benchmark: 4-framework speed comparison (GPU utilization test).

Measures tok/s with 128-token generation, GPU should be fully utilized.
This tests ENGINE throughput, not correctness (use bench_mmlu.py for that).

Usage:
    PYTHONPATH=. uv run python scripts/bench_throughput.py
    PYTHONPATH=. uv run python scripts/bench_throughput.py --quick
    PYTHONPATH=. uv run python scripts/bench_throughput.py --framework mlx-lm
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import sys
import time
from pathlib import Path

import mlx.core as mx

REF_DIR = Path(__file__).resolve().parent.parent.parent / "reference"
MODEL = "models/Qwen3.5-9B-MLX-4bit"
PROMPT = "Write a detailed essay about the history of computing. " * 20


def P(msg):
    print(msg, flush=True)


def cleanup():
    gc.collect(); mx.synchronize(); mx.clear_cache()


def bench_mlx_lm(n_runs, max_tokens):
    from mlx_lm.utils import load
    from mlx_lm.generate import generate_step

    P("  Loading model...")
    model, tokenizer = load(MODEL)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT[:500]}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = mx.array(tokenizer.encode(text))

    # Warmup
    for _ in generate_step(ids, model, max_tokens=16):
        pass

    speeds = []
    ttfts = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        first = True
        tokens = []
        for tok, _ in generate_step(ids, model, max_tokens=max_tokens):
            if first:
                ttfts.append(time.perf_counter() - t0)
                first = False
            tokens.append(tok)
        elapsed = time.perf_counter() - t0
        speeds.append(len(tokens) / elapsed)
        if (i + 1) % 5 == 0:
            P(f"  Run {i+1}/{n_runs}: {len(tokens)} tok, {speeds[-1]:.1f} tok/s")

    del model, tokenizer; cleanup()
    return speeds, ttfts


def bench_yunshu(n_runs, max_tokens):
    from yunshu_engine.batched_engine import BatchedEngine

    async def _run():
        engine = BatchedEngine(model_name=MODEL)
        await engine.start()

        # Warmup
        await engine.generate(prompt=[{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.0)

        speeds = []
        ttfts = []
        for i in range(n_runs):
            t0 = time.perf_counter()
            r = await engine.generate(
                prompt=[{"role": "user", "content": PROMPT[:500]}],
                max_tokens=max_tokens, temperature=0.0, enable_thinking=False,
            )
            elapsed = time.perf_counter() - t0
            n_tok = r.completion_tokens if hasattr(r, 'completion_tokens') else 0
            speeds.append(n_tok / elapsed if elapsed > 0 else 0)
            # Approximate TTFT from total time minus generation time
            gen_time = n_tok / speeds[-1] if speeds[-1] > 0 else 0
            ttfts.append(elapsed - gen_time)
            if (i + 1) % 5 == 0:
                P(f"  Run {i+1}/{n_runs}: {n_tok} tok, {speeds[-1]:.1f} tok/s")

        await engine.stop()
        return speeds, ttfts

    r = asyncio.run(_run()); cleanup()
    return r


def bench_omlx(n_runs, max_tokens):
    sys.path.insert(0, str(REF_DIR / "omlx"))
    from omlx.models.llm import MLXLanguageModel

    P("  Loading model...")
    llm = MLXLanguageModel(MODEL)
    llm.load()

    # Warmup
    llm.generate("test", max_tokens=10, temperature=0.0)

    speeds = []
    ttfts = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        r = llm.generate(PROMPT[:500], max_tokens=max_tokens, temperature=0.0)
        elapsed = time.perf_counter() - t0
        n_tok = len(r.tokens) if hasattr(r, 'tokens') else 0
        speeds.append(n_tok / elapsed if elapsed > 0 else 0)
        gen_time = n_tok / speeds[-1] if speeds[-1] > 0 else 0
        ttfts.append(elapsed - gen_time)
        if (i + 1) % 5 == 0:
            P(f"  Run {i+1}/{n_runs}: {n_tok} tok, {speeds[-1]:.1f} tok/s")

    del llm; cleanup()
    return speeds, ttfts


def bench_vllm_mlx(n_runs, max_tokens):
    sys.path.insert(0, str(REF_DIR / "vllm-mlx"))
    from mlx_lm.utils import load
    from vllm_mlx.engine_core import EngineCore, EngineConfig
    from vllm_mlx.request import SamplingParams

    P("  Loading model...")
    model, tokenizer = load(MODEL)

    async def _run():
        core = EngineCore(model, tokenizer, EngineConfig())
        await core.start()

        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT[:500]}],
            tokenize=False, add_generation_prompt=True,
        )

        # Warmup
        sp = SamplingParams(max_tokens=16, temperature=0.0)
        await core.generate(prompt=text, sampling_params=sp)

        speeds = []
        ttfts = []
        for i in range(n_runs):
            sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
            t0 = time.perf_counter()
            try:
                out = await core.generate(prompt=text, sampling_params=sp)
                elapsed = time.perf_counter() - t0
                n_tok = out.completion_tokens if hasattr(out, 'completion_tokens') else max_tokens
                speeds.append(n_tok / elapsed if elapsed > 0 else 0)
                gen_time = n_tok / speeds[-1] if speeds[-1] > 0 else 0
                ttfts.append(elapsed - gen_time)
            except Exception as e:
                P(f"  Run {i+1}: ERROR {e}")
                continue
            if (i + 1) % 5 == 0:
                P(f"  Run {i+1}/{n_runs}: {n_tok} tok, {speeds[-1]:.1f} tok/s")

        await core.stop()
        return speeds, ttfts

    r = asyncio.run(_run())
    del model, tokenizer; cleanup()
    return r


RUNNERS = {
    "mlx-lm": bench_mlx_lm,
    "yunshu": bench_yunshu,
    "omlx": bench_omlx,
    "vllm-mlx": bench_vllm_mlx,
}


def main():
    parser = argparse.ArgumentParser(description="Throughput benchmark: 4-framework speed comparison")
    parser.add_argument("--quick", action="store_true", help="3 runs instead of 10")
    parser.add_argument("--framework", choices=["all", "mlx-lm", "yunshu", "omlx", "vllm-mlx"], default="all")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    if not Path(MODEL).exists():
        P(f"Model not found: {MODEL}")
        sys.exit(1)

    n_runs = 3 if args.quick else 10
    fws = list(RUNNERS.keys()) if args.framework == "all" else [args.framework]

    P(f"Model: {MODEL}")
    P(f"Runs: {n_runs}, max_tokens: {args.max_tokens}")
    P(f"Frameworks: {', '.join(fws)}")
    P(f"")

    all_results = {}

    for fw in fws:
        P(f"\n{'='*60}")
        P(f"  {fw.upper()} — Throughput Benchmark")
        P(f"{'='*60}")
        try:
            speeds, ttfts = RUNNERS[fw](n_runs, args.max_tokens)
            avg_speed = sum(speeds) / len(speeds) if speeds else 0
            avg_ttft = sum(ttfts) / len(ttfts) * 1000 if ttfts else 0
            all_results[fw] = {
                "avg_tok_s": avg_speed,
                "min_tok_s": min(speeds) if speeds else 0,
                "max_tok_s": max(speeds) if speeds else 0,
                "avg_ttft_ms": avg_ttft,
                "runs": len(speeds),
            }
            P(f"\n  RESULT: {avg_speed:.1f} tok/s (TTFT ≈ {avg_ttft:.0f}ms, n={len(speeds)})")
        except Exception as e:
            import traceback
            P(f"  FAILED: {e}")
            traceback.print_exc()

    # Summary
    if len(all_results) >= 2:
        P(f"\n{'='*60}")
        P(f"  THROUGHPUT COMPARISON (max_tokens={args.max_tokens})")
        P(f"{'='*60}")
        P(f"  {'Framework':<12} {'tok/s':>10} {'TTFT ms':>10} {'Runs':>6}")
        P(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*6}")
        for fw in fws:
            if fw in all_results:
                r = all_results[fw]
                P(f"  {fw:<12} {r['avg_tok_s']:>9.1f} {r['avg_ttft_ms']:>9.0f} {r['runs']:>6}")

        # Relative speed
        baseline = all_results.get("mlx-lm", {}).get("avg_tok_s", 0)
        if baseline > 0:
            P(f"\n  Relative to mlx-lm baseline ({baseline:.1f} tok/s):")
            for fw in fws:
                if fw in all_results:
                    r = all_results[fw]
                    ratio = r["avg_tok_s"] / baseline
                    P(f"    {fw:<12} {ratio:.2f}x")


if __name__ == "__main__":
    main()
