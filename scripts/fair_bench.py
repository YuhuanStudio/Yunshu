"""Fair apples-to-apples benchmark: Yunshu vs mlx-lm.

Target comparison (same model, same quantization):
  LLM:        Yunshu vs mlx-lm vanilla (same engine oMLX uses)

Measures THREE dimensions per comparison:
  1. Accuracy/quality — same prompt, compare outputs
  2. Speed — tok/s, TTFT, E2E latency
  3. Memory — peak active memory (Apple Silicon unified)

Usage:
    PYTHONPATH=. uv run python scripts/fair_bench.py --model models/Qwen3.5-9B-MLX-4bit
    PYTHONPATH=. uv run python scripts/fair_bench.py --model models/Qwen3.5-9B-MLX-4bit --quick
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

# Evaluation prompts with ground-truth checks
PROMPTS = [
    {"q": "What is 15 + 27?", "a": "42"},
    {"q": "What is the capital of Japan?", "a": "tokyo"},
    {"q": "What is the chemical formula for table salt?", "a": "nacl"},
    {"q": "Who wrote Romeo and Juliet?", "a": "shakespeare"},
    {"q": "What planet is closest to the Sun?", "a": "mercury"},
    {"q": "What is 100 - 37?", "a": "63"},
    {"q": "What is the largest ocean on Earth?", "a": "pacific"},
    {"q": "How many sides does a hexagon have?", "a": "6"},
    {"q": "What gas do plants absorb from the atmosphere?", "a": "carbon dioxide"},
    {"q": "What is the square root of 144?", "a": "12"},
    {"q": "In what year did World War II end?", "a": "1945"},
    {"q": "What is the boiling point of water in Celsius?", "a": "100"},
    {"q": "What element has the atomic number 1?", "a": "hydrogen"},
    {"q": "How many continents are there?", "a": "7"},
    {"q": "What is 8 * 7?", "a": "56"},
]

QUICK_PROMPTS = PROMPTS[:5]


def check_answer(output: str, expected: str) -> bool:
    return expected.lower() in output.lower()


def _cleanup():
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def bench_mlx_lm(model_path: str, prompts: list[dict]) -> dict:
    """Benchmark vanilla mlx-lm (baseline competitor for LLM)."""
    from mlx_lm.utils import load as mlx_load
    from mlx_lm.generate import generate_step

    print("  Loading mlx-lm model...")
    model, tokenizer = mlx_load(model_path)

    mem_after_load = mx.get_active_memory()

    results = {"name": "mlx-lm-vanilla", "runs": [], "accuracy": 0.0}
    correct = 0
    total_ttft = []
    total_gen_time = []
    total_tokens = []

    for i, prompt in enumerate(prompts):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["q"]}],
            tokenize=False, add_generation_prompt=True,
        )
        # 1D input — generate_step adds batch dim internally
        token_ids = mx.array(tokenizer.encode(text))

        t0 = time.perf_counter()
        first_token = True
        ttft = None
        tokens = []
        for token, _ in generate_step(token_ids, model, max_tokens=64):
            if first_token:
                ttft = time.perf_counter() - t0
                first_token = False
            tokens.append(token)

            if hasattr(tokenizer, 'eos_token_id') and token == tokenizer.eos_token_id:
                break
            if hasattr(tokenizer, 'eos_token_ids') and token in tokenizer.eos_token_ids:
                break

        elapsed = time.perf_counter() - t0
        output = tokenizer.decode(tokens, skip_special_tokens=True)

        try:
            mem_peak = mx.get_peak_memory()
            mx.reset_peak_memory()
        except Exception:
            mem_peak = 0

        is_correct = check_answer(output, prompt["a"])
        if is_correct:
            correct += 1

        gen_time = elapsed - (ttft or 0)
        total_ttft.append(ttft or 0)
        total_gen_time.append(gen_time)
        total_tokens.append(len(tokens))

        results["runs"].append({
            "prompt": prompt["q"],
            "expected": prompt["a"],
            "output": output[:100],
            "correct": is_correct,
            "ttft_ms": round((ttft or 0) * 1000, 1),
            "tokens": len(tokens),
            "gen_tok_s": round(len(tokens) / gen_time, 1) if gen_time > 0 else 0,
            "mem_peak_mb": round(mem_peak / 1024**2, 1),
        })

        if (i + 1) % 5 == 0:
            print(f"    mlx-lm: {i + 1}/{len(prompts)}")

    results["accuracy"] = correct / len(prompts)
    results["mem_after_load_mb"] = round(mem_after_load / 1024**2, 1)
    results["summary"] = {
        "ttft_p50_ms": round(statistics.median(total_ttft) * 1000, 1),
        "gen_tok_s": round(statistics.mean(
            [t / g for t, g in zip(total_tokens, total_gen_time) if g > 0]
        ), 1),
        "e2e_mean_ms": round(statistics.mean(total_ttft) * 1000 + statistics.mean(total_gen_time) * 1000, 1),
        "mem_peak_avg_mb": round(statistics.mean([r["mem_peak_mb"] for r in results["runs"]]), 1),
    }
    return results


def bench_yunshu(model_path: str, prompts: list[dict]) -> dict:
    """Benchmark Yunshu BatchedEngine.

    Must run start/generate/stop inside a single asyncio.run() so that
    EngineCore's _engine_loop task stays alive on the same event loop.
    """
    from yunshu_engine.batched_engine import BatchedEngine

    print("  Loading Yunshu engine...")

    async def _run():
        engine = BatchedEngine(model_name=model_path)
        await engine.start()

        mem_after_load = mx.get_active_memory()

        results = {"name": "yunshu-engine", "runs": [], "accuracy": 0.0}
        correct = 0
        total_gen_time = []
        total_tokens = []

        for i, prompt in enumerate(prompts):
            messages = [{"role": "user", "content": prompt["q"]}]

            t0 = time.perf_counter()
            result = await engine.generate(prompt=messages, max_tokens=64, temperature=0.0)
            elapsed = time.perf_counter() - t0

            try:
                mem_peak = mx.get_peak_memory()
                mx.reset_peak_memory()
            except Exception:
                mem_peak = 0

            output = result.text if hasattr(result, 'text') else str(result)
            output_tokens = result.completion_tokens if hasattr(result, 'completion_tokens') else 0

            is_correct = check_answer(output, prompt["a"])
            if is_correct:
                correct += 1

            total_gen_time.append(elapsed)
            total_tokens.append(output_tokens)

            results["runs"].append({
                "prompt": prompt["q"],
                "expected": prompt["a"],
                "output": output[:100],
                "correct": is_correct,
                "e2e_ms": round(elapsed * 1000, 1),
                "tokens": output_tokens,
                "gen_tok_s": round(output_tokens / elapsed, 1) if elapsed > 0 else 0,
                "mem_peak_mb": round(mem_peak / 1024**2, 1),
            })

            if (i + 1) % 5 == 0:
                print(f"    yunshu: {i + 1}/{len(prompts)}")

        await engine.stop()

        results["accuracy"] = correct / len(prompts)
        results["mem_after_load_mb"] = round(mem_after_load / 1024**2, 1)
        results["summary"] = {
            "e2e_mean_ms": round(statistics.mean(total_gen_time) * 1000, 1),
            "gen_tok_s": round(statistics.mean(
                [t / g for t, g in zip(total_tokens, total_gen_time) if g > 0]
            ), 1),
            "mem_peak_avg_mb": round(statistics.mean([r["mem_peak_mb"] for r in results["runs"]]), 1),
        }
        return results

    return asyncio.run(_run())


def main():
    parser = argparse.ArgumentParser(
        description="Fair benchmark: Yunshu vs mlx-lm"
    )
    parser.add_argument("--model", default="models/Qwen3.5-9B-MLX-4bit", help="Model path")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}")
        sys.exit(1)

    prompts = QUICK_PROMPTS if args.quick else PROMPTS
    print(f"Model: {args.model}")
    print(f"Prompts: {len(prompts)}")
    print()

    # Run mlx-lm baseline
    print("Phase 1: mlx-lm baseline")
    mlx_results = bench_mlx_lm(args.model, prompts)
    print(f"  Accuracy: {mlx_results['accuracy']:.0%}")
    print(f"  TTFT p50: {mlx_results['summary']['ttft_p50_ms']}ms")
    print(f"  Gen speed: {mlx_results['summary']['gen_tok_s']} tok/s")
    print()

    _cleanup()

    # Run Yunshu
    print("Phase 2: Yunshu engine")
    yunshu_results = bench_yunshu(args.model, prompts)
    print(f"  Accuracy: {yunshu_results['accuracy']:.0%}")
    print(f"  Gen speed: {yunshu_results['summary']['gen_tok_s']} tok/s")
    print()

    # Compare
    print("=" * 60)
    print("COMPARISON: Yunshu vs mlx-lm (oMLX baseline)")
    print("=" * 60)

    acc_mlx = mlx_results["accuracy"]
    acc_ys = yunshu_results["accuracy"]
    speed_mlx = mlx_results["summary"]["gen_tok_s"]
    speed_ys = yunshu_results["summary"]["gen_tok_s"]
    mem_mlx = mlx_results["summary"].get("mem_peak_avg_mb", 0)
    mem_ys = yunshu_results["summary"].get("mem_peak_avg_mb", 0)
    mem_load_mlx = mlx_results.get("mem_after_load_mb", 0)
    mem_load_ys = yunshu_results.get("mem_after_load_mb", 0)

    acc_diff = acc_ys - acc_mlx
    acc_verdict = "SAME" if abs(acc_diff) < 0.01 else ("DEGRADATION" if acc_diff < 0 else "IMPROVED")

    speed_ratio = speed_ys / speed_mlx if speed_mlx > 0 else 0
    speed_verdict = "SAME" if abs(speed_ratio - 1.0) < 0.05 else f"{speed_ratio:.2f}x"

    print(f"\n  ┌──────────┬───────────────┬───────────────┬──────────────┐")
    print(f"  │ Metric   │ mlx-lm/oMLX   │ Yunshu        │ Verdict      │")
    print(f"  ├──────────┼───────────────┼───────────────┼──────────────┤")
    print(f"  │ Accuracy │ {acc_mlx:>10.0%}    │ {acc_ys:>10.0%}    │ {acc_verdict:<12} │")
    print(f"  │ Speed    │ {speed_mlx:>8} t/s  │ {speed_ys:>8} t/s  │ {speed_verdict:<12} │")
    print(f"  │ Mem load │ {mem_load_mlx:>8} MB  │ {mem_load_ys:>8} MB  │ {'OK' if mem_load_ys <= mem_load_mlx * 1.1 else 'HIGH':<12} │")
    print(f"  │ Mem peak │ {mem_mlx:>8} MB  │ {mem_ys:>8} MB  │ {'OK' if mem_ys <= mem_mlx * 1.1 else 'HIGH':<12} │")
    print(f"  └──────────┴───────────────┴───────────────┴──────────────┘")

    # Per-prompt comparison
    print(f"\n  Per-prompt detail:")
    for i, (mlx_r, ys_r) in enumerate(zip(mlx_results["runs"], yunshu_results["runs"])):
        match = "ok" if mlx_r["correct"] == ys_r["correct"] else ("!" if not ys_r["correct"] else "+")
        print(f"    [{match}] Q{i}: mlx={'Y' if mlx_r['correct'] else 'N'} ys={'Y' if ys_r['correct'] else 'N'} "
              f"speed={ys_r.get('gen_tok_s', '?')} t/s "
              f"mlx_out='{mlx_r['output'][:40]}' ys_out='{ys_r['output'][:40]}'")

    # Final verdict
    print(f"\n  VERDICT:")
    if acc_ys >= acc_mlx and speed_ys >= speed_mlx * 0.95:
        print("  Yunshu matches or exceeds mlx-lm/oMLX on both quality and speed.")
    elif acc_ys < acc_mlx:
        print(f"  ACCURACY REGRESSION: Yunshu is {abs(acc_diff):.0%} below baseline.")
    elif speed_ys < speed_mlx * 0.9:
        print(f"  SPEED REGRESSION: Yunshu is {1/speed_ratio:.2f}x slower than baseline.")
    else:
        print("  Mixed results. See details above.")

    if args.json:
        print(json.dumps({
            "mlx_lm": mlx_results,
            "yunshu": yunshu_results,
        }, indent=2, default=str))


if __name__ == "__main__":
    main()
