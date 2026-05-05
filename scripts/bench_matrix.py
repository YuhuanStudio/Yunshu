"""Yunshu Comprehensive Benchmark Matrix.

Compares Yunshu against multiple competitor frameworks per modality:

  LLM:   mlx-lm vanilla (same engine oMLX uses internally)
  VLM:   Yunshu VLM engine (via mlx-vlm model classes)

Test scenarios per comparison:
  1. Accuracy — same prompts, same model, check output quality
  2. Speed — tok/s, TTFT, E2E latency
  3. Memory — peak active memory after load, peak during generation

All comparisons use the SAME model path and quantization level.

Usage:
    PYTHONPATH=. uv run python scripts/bench_matrix.py --modality llm --quick
    PYTHONPATH=. uv run python scripts/bench_matrix.py --modality llm --model models/Qwen3.5-9B-MLX-4bit
    PYTHONPATH=. uv run python scripts/bench_matrix.py --modality vlm --quick
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx

# ── Test Scenarios ──────────────────────────────────────────────────────────

LLM_ACCURACY_PROMPTS = [
    {"q": "What is 15 + 27?", "a": "42", "category": "math"},
    {"q": "What is the capital of Japan?", "a": "tokyo", "category": "knowledge"},
    {"q": "What is the chemical formula for table salt?", "a": "nacl", "category": "science"},
    {"q": "Who wrote Romeo and Juliet?", "a": "shakespeare", "category": "knowledge"},
    {"q": "What planet is closest to the Sun?", "a": "mercury", "category": "science"},
    {"q": "What is 100 - 37?", "a": "63", "category": "math"},
    {"q": "What is the largest ocean on Earth?", "a": "pacific", "category": "geography"},
    {"q": "How many sides does a hexagon have?", "a": "6", "category": "math"},
    {"q": "What gas do plants absorb?", "a": "carbon dioxide", "category": "science"},
    {"q": "What is the square root of 144?", "a": "12", "category": "math"},
    {"q": "In what year did WWII end?", "a": "1945", "category": "history"},
    {"q": "What is the boiling point of water in Celsius?", "a": "100", "category": "science"},
    {"q": "How many continents are there?", "a": "7", "category": "geography"},
    {"q": "What is 8 * 7?", "a": "56", "category": "math"},
    {"q": "What element has atomic number 1?", "a": "hydrogen", "category": "science"},
]


@dataclass
class BenchResult:
    framework: str
    modality: str
    scenario: str
    accuracy: float = 0.0
    correct: int = 0
    total: int = 0
    ttft_ms: float = 0.0
    e2e_ms: float = 0.0
    gen_tok_s: float = 0.0
    mem_peak_mb: float = 0.0
    mem_load_mb: float = 0.0
    details: list[dict] = field(default_factory=list)


def check_answer(output: str, expected: str) -> bool:
    return expected.lower() in output.lower()


def _cleanup():
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


# ── Framework Runners ───────────────────────────────────────────────────────

def run_mlx_lm(model_path: str, prompts: list[dict], max_tokens: int = 64) -> BenchResult:
    """Run mlx-lm vanilla (baseline for LLM comparison)."""
    from mlx_lm.utils import load as mlx_load
    from mlx_lm.generate import generate_step

    model, tokenizer = mlx_load(model_path)
    mem_load = mx.get_active_memory()

    result = BenchResult(framework="mlx-lm", modality="llm", scenario="accuracy")
    mem_peaks = []

    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["q"]}],
            tokenize=False, add_generation_prompt=True,
        )
        token_ids = mx.array(tokenizer.encode(text))

        t0 = time.perf_counter()
        ttft = None
        tokens = []
        first = True
        for token, _ in generate_step(token_ids, model, max_tokens=max_tokens):
            if first:
                ttft = time.perf_counter() - t0
                first = False
            tokens.append(token)
            if hasattr(tokenizer, 'eos_token_id') and token == tokenizer.eos_token_id:
                break

        elapsed = time.perf_counter() - t0
        output = tokenizer.decode(tokens, skip_special_tokens=True)

        try:
            mp = mx.get_peak_memory()
            mx.reset_peak_memory()
            mem_peaks.append(mp)
        except Exception:
            pass

        correct = check_answer(output, prompt["a"])
        if correct:
            result.correct += 1
        result.total += 1
        gen_time = elapsed - (ttft or 0)
        result.details.append({
            "prompt": prompt["q"], "expected": prompt["a"],
            "output": output[:100], "correct": correct,
            "ttft_ms": round((ttft or 0) * 1000, 1),
            "tokens": len(tokens),
            "gen_tok_s": round(len(tokens) / gen_time, 1) if gen_time > 0 else 0,
        })

    result.accuracy = result.correct / result.total if result.total > 0 else 0
    result.mem_load_mb = round(mem_load / 1024**2, 1)
    result.mem_peak_mb = round(statistics.mean(mem_peaks) / 1024**2, 1) if mem_peaks else 0
    if result.details:
        result.ttft_ms = round(statistics.median([d["ttft_ms"] for d in result.details]), 1)
        gen_speeds = [d["gen_tok_s"] for d in result.details if d["gen_tok_s"] > 0]
        result.gen_tok_s = round(statistics.mean(gen_speeds), 1) if gen_speeds else 0
    return result


def run_yunshu_llm(model_path: str, prompts: list[dict], max_tokens: int = 64) -> BenchResult:
    """Run Yunshu BatchedEngine — single async context for engine loop lifetime."""
    from yunshu_engine.batched_engine import BatchedEngine

    async def _run():
        engine = BatchedEngine(model_name=model_path)
        await engine.start()
        mem_load = mx.get_active_memory()

        result = BenchResult(framework="yunshu", modality="llm", scenario="accuracy")
        mem_peaks = []

        for prompt in prompts:
            messages = [{"role": "user", "content": prompt["q"]}]
            t0 = time.perf_counter()
            gen_result = await engine.generate(prompt=messages, max_tokens=max_tokens, temperature=0.0)
            elapsed = time.perf_counter() - t0

            output = gen_result.text if hasattr(gen_result, 'text') else str(gen_result)
            n_tokens = gen_result.completion_tokens if hasattr(gen_result, 'completion_tokens') else 0

            try:
                mp = mx.get_peak_memory()
                mx.reset_peak_memory()
                mem_peaks.append(mp)
            except Exception:
                pass

            correct = check_answer(output, prompt["a"])
            if correct:
                result.correct += 1
            result.total += 1
            result.details.append({
                "prompt": prompt["q"], "expected": prompt["a"],
                "output": output[:100], "correct": correct,
                "tokens": n_tokens,
                "gen_tok_s": round(n_tokens / elapsed, 1) if elapsed > 0 else 0,
            })

        await engine.stop()

        result.accuracy = result.correct / result.total if result.total > 0 else 0
        result.mem_load_mb = round(mem_load / 1024**2, 1)
        result.mem_peak_mb = round(statistics.mean(mem_peaks) / 1024**2, 1) if mem_peaks else 0
        if result.details:
            gen_speeds = [d["gen_tok_s"] for d in result.details if d["gen_tok_s"] > 0]
            result.gen_tok_s = round(statistics.mean(gen_speeds), 1) if gen_speeds else 0
        return result

    return asyncio.run(_run())


def run_yunshu_vlm(model_path: str, prompts: list[dict], max_tokens: int = 64) -> BenchResult:
    """Run Yunshu VLM engine — single async context."""
    from yunshu_engine.vlm_engine import VLMEngine

    async def _run():
        engine = VLMEngine(model_path)
        await engine.start()
        mem_load = mx.get_active_memory()

        result = BenchResult(framework="yunshu-vlm", modality="vlm", scenario="text_accuracy")

        for prompt in prompts:
            messages = [{"role": "user", "content": prompt["q"]}]
            t0 = time.perf_counter()
            gen_result = await engine.generate(messages=messages, max_tokens=max_tokens, temperature=0.0)
            elapsed = time.perf_counter() - t0
            output = gen_result.get("text", "")
            n_tokens = gen_result.get("elapsed", 0)

            correct = check_answer(output, prompt["a"])
            if correct:
                result.correct += 1
            result.total += 1
            result.details.append({
                "prompt": prompt["q"], "expected": prompt["a"],
                "output": output[:100], "correct": correct,
                "e2e_ms": round(elapsed * 1000, 1),
            })

        await engine.stop()

        result.accuracy = result.correct / result.total if result.total > 0 else 0
        result.mem_load_mb = round(mem_load / 1024**2, 1)
        if result.details:
            result.e2e_ms = round(statistics.median([d["e2e_ms"] for d in result.details]), 1)
        return result

    return asyncio.run(_run())


# ── Speed Benchmarks ────────────────────────────────────────────────────────

def bench_speed_mlx_lm(model_path: str, max_tokens: int = 128, n_runs: int = 5) -> BenchResult:
    """Speed benchmark: mlx-lm generate_step."""
    from mlx_lm.utils import load as mlx_load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = mlx_load(model_path)
    mem_load = mx.get_active_memory()

    result = BenchResult(framework="mlx-lm", modality="llm", scenario="speed")
    prompt = "Write a short essay about artificial intelligence."
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    sampler = make_sampler(temp=0.7)

    for _ in range(n_runs):
        token_ids = mx.array(tokenizer.encode(text))
        t0 = time.perf_counter()
        tokens = []
        for token, _ in generate_step(token_ids, model, max_tokens=max_tokens, sampler=sampler):
            tokens.append(token)
        elapsed = time.perf_counter() - t0
        result.details.append({
            "tokens": len(tokens),
            "e2e_ms": round(elapsed * 1000, 1),
            "gen_tok_s": round(len(tokens) / elapsed, 1),
        })

    result.mem_load_mb = round(mem_load / 1024**2, 1)
    speeds = [d["gen_tok_s"] for d in result.details]
    result.gen_tok_s = round(statistics.mean(speeds), 1)
    return result


def bench_speed_yunshu(model_path: str, max_tokens: int = 128, n_runs: int = 5) -> BenchResult:
    """Speed benchmark: Yunshu BatchedEngine — single async context."""
    from yunshu_engine.batched_engine import BatchedEngine

    async def _run():
        engine = BatchedEngine(model_name=model_path)
        await engine.start()
        mem_load = mx.get_active_memory()

        result = BenchResult(framework="yunshu", modality="llm", scenario="speed")
        messages = [{"role": "user", "content": "Write a short essay about artificial intelligence."}]

        for _ in range(n_runs):
            t0 = time.perf_counter()
            gen_result = await engine.generate(prompt=messages, max_tokens=max_tokens, temperature=0.7)
            elapsed = time.perf_counter() - t0
            n_tokens = gen_result.completion_tokens if hasattr(gen_result, 'completion_tokens') else 0
            result.details.append({
                "tokens": n_tokens,
                "e2e_ms": round(elapsed * 1000, 1),
                "gen_tok_s": round(n_tokens / elapsed, 1),
            })

        await engine.stop()

        result.mem_load_mb = round(mem_load / 1024**2, 1)
        speeds = [d["gen_tok_s"] for d in result.details]
        result.gen_tok_s = round(statistics.mean(speeds), 1)
        return result

    return asyncio.run(_run())


# ── Comparison Matrix ───────────────────────────────────────────────────────

def print_comparison_matrix(results: list[BenchResult]) -> None:
    print("\n" + "=" * 90)
    print("  YUNSHU COMPREHENSIVE BENCHMARK MATRIX")
    print("=" * 90)

    by_modality: dict[str, list[BenchResult]] = {}
    for r in results:
        by_modality.setdefault(r.modality, []).append(r)

    for modality, mod_results in sorted(by_modality.items()):
        print(f"\n  ┌─ {modality.upper()} ─────────────────────────────────────────────────────────────────┐")
        print(f"  │ {'Framework':<20} {'Accuracy':>10} {'tok/s':>8} {'TTFT ms':>10} {'E2E ms':>10} {'Peak MB':>10} │")
        print(f"  ├{'─' * 78}┤")

        for r in sorted(mod_results, key=lambda x: x.framework):
            acc = f"{r.accuracy:.0%}" if r.total > 0 else "N/A"
            tok = f"{r.gen_tok_s:.1f}" if r.gen_tok_s > 0 else "N/A"
            ttft = f"{r.ttft_ms:.1f}" if r.ttft_ms > 0 else "N/A"
            e2e = f"{r.e2e_ms:.1f}" if r.e2e_ms > 0 else "N/A"
            mem = f"{r.mem_peak_mb:.0f}" if r.mem_peak_mb > 0 else "N/A"
            print(f"  │ {r.framework:<20} {acc:>10} {tok:>8} {ttft:>10} {e2e:>10} {mem:>10} │")

        print(f"  └{'─' * 78}┘")

        if len(mod_results) >= 2:
            yunshu = next((r for r in mod_results if r.framework.startswith("yunshu")), None)
            baseline = next((r for r in mod_results if not r.framework.startswith("yunshu")), None)
            if yunshu and baseline:
                if yunshu.accuracy >= baseline.accuracy and yunshu.gen_tok_s >= baseline.gen_tok_s * 0.95:
                    verdict = "Yunshu matches/exceeds baseline"
                elif yunshu.accuracy < baseline.accuracy:
                    verdict = f"Accuracy regression: {baseline.accuracy - yunshu.accuracy:.0%}"
                else:
                    verdict = f"Speed: {yunshu.gen_tok_s / baseline.gen_tok_s:.2f}x baseline"
                print(f"  │ Verdict: {verdict}")


# ── Main ────────────────────────────────────────────────────────────────────

AVAILABLE_MODELS = {
    "llm": "models/Qwen3.5-9B-MLX-4bit",
    "vlm": "models/Qwen3-Omni-30B-A3B-Instruct-4bit",
}

FRAMEWORK_RUNNERS = {
    "llm": {
        "yunshu": run_yunshu_llm,
        "mlx-lm": run_mlx_lm,
    },
    "vlm": {
        "yunshu": run_yunshu_vlm,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Yunshu comprehensive benchmark matrix")
    parser.add_argument("--modality", choices=["llm", "vlm", "all"], default="llm")
    parser.add_argument("--model", help="Override model path")
    parser.add_argument("--quick", action="store_true", help="Fewer prompts")
    parser.add_argument("--competitors", nargs="+", help="Specific competitors to run")
    parser.add_argument("--speed", action="store_true", help="Run speed benchmarks instead of accuracy")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    modalities = list(FRAMEWORK_RUNNERS.keys()) if args.modality == "all" else [args.modality]
    prompts = LLM_ACCURACY_PROMPTS[:5] if args.quick else LLM_ACCURACY_PROMPTS
    all_results: list[BenchResult] = []

    for mod in modalities:
        model_path = args.model or AVAILABLE_MODELS.get(mod, "")
        if not model_path or not Path(model_path).exists():
            print(f"Skipping {mod}: model not found at {model_path}")
            continue

        if args.speed and mod == "llm":
            print(f"\n{'='*60}")
            print(f"  MODALITY: {mod} (SPEED BENCHMARK)")
            print(f"  Model: {model_path}")
            print(f"{'='*60}")

            for name, runner in [("yunshu", bench_speed_yunshu), ("mlx-lm", bench_speed_mlx_lm)]:
                if args.competitors and name not in args.competitors and name != "yunshu":
                    continue
                print(f"\n  Running {name} speed benchmark...")
                try:
                    result = runner(model_path)
                    all_results.append(result)
                    print(f"    Speed: {result.gen_tok_s} tok/s, Memory: {result.mem_load_mb} MB")
                except Exception as e:
                    print(f"    FAILED: {e}")
                _cleanup()
            continue

        runners = FRAMEWORK_RUNNERS.get(mod, {})
        if args.competitors:
            runners = {k: v for k, v in runners.items() if k in args.competitors or k == "yunshu"}

        print(f"\n{'='*60}")
        print(f"  MODALITY: {mod}")
        print(f"  Model: {model_path}")
        print(f"  Frameworks: {', '.join(runners.keys())}")
        print(f"{'='*60}")

        for name, runner in runners.items():
            print(f"\n  Running {name}...")
            try:
                result = runner(model_path, prompts)
                all_results.append(result)
                print(f"    Accuracy: {result.accuracy:.0%}, Speed: {result.gen_tok_s} tok/s, "
                      f"Memory: {result.mem_peak_mb} MB")
            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback
                traceback.print_exc()
            _cleanup()

    print_comparison_matrix(all_results)

    if args.json:
        print(json.dumps([{
            "framework": r.framework,
            "modality": r.modality,
            "accuracy": r.accuracy,
            "gen_tok_s": r.gen_tok_s,
            "ttft_ms": r.ttft_ms,
            "e2e_ms": r.e2e_ms,
            "mem_peak_mb": r.mem_peak_mb,
            "mem_load_mb": r.mem_load_mb,
        } for r in all_results], indent=2))


if __name__ == "__main__":
    main()
