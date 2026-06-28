"""Model quality comparison: Yunshu engine vs mlx-lm baseline.

Runs the same prompts through both engines and compares outputs to detect
accuracy degradation from our customizations (thinking budget, JSON constraints,
scheduler modifications, etc.).

This ensures speed/memory optimizations don't sacrifice model quality.

Usage:
    PYTHONPATH=. uv run python scripts/quality_comparison.py
    PYTHONPATH=. uv run python scripts/quality_comparison.py --model-path models/Qwen3-0.6B-FP16
    PYTHONPATH=. uv run python scripts/quality_comparison.py --quick  # Fewer prompts
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROMPTS = [
    # Factual / knowledge
    {"messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
     "check": lambda r: "paris" in r.lower()},
    {"messages": [{"role": "user", "content": "What is 2 + 3? Answer with just the number."}],
     "check": lambda r: "5" in r.strip()[:3]},
    {"messages": [{"role": "user", "content": "What is the chemical symbol for water?"}],
     "check": lambda r: "H2O" in r.upper() or "h2o" in r.lower()},
    # Reasoning
    {"messages": [{"role": "user", "content": "If a shirt costs $25 and is on sale for 20% off, what is the sale price?"}],
     "check": lambda r: "20" in r},
    {"messages": [{"role": "user", "content": "What comes next in the sequence: 2, 4, 6, 8, ?"}],
     "check": lambda r: "10" in r},
    # Multi-turn
    {"messages": [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What is my name?"},
    ], "check": lambda r: "alice" in r.lower()},
    # Instruction following
    {"messages": [{"role": "user", "content": "Write a haiku about coding. Follow the 5-7-5 syllable pattern exactly."}],
     "check": lambda r: len(r.strip().split("\n")) >= 2},
    # Format compliance
    {"messages": [{"role": "user", "content": "List 3 colors separated by commas."}],
     "check": lambda r: r.count(",") >= 2},
    # Longer generation
    {"messages": [{"role": "user", "content": "Explain what a variable is in programming in 2-3 sentences."}],
     "check": lambda r: len(r.split()) >= 10},
    {"messages": [{"role": "user", "content": "Translate 'hello world' to French."}],
     "check": lambda r: "bonjour" in r.lower() or "monde" in r.lower()},
]

QUICK_PROMPTS = PROMPTS[:5]


def run_mlx_lm_baseline(model_path: str, prompts: list[dict]) -> list[str]:
    """Run prompts through vanilla mlx-lm generate (no Yunshu modifications)."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.utils import load as mlx_load

    model, tokenizer = mlx_load(model_path)

    results = []
    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            prompt["messages"], tokenize=False, add_generation_prompt=True,
        ) or "\n".join(m["content"] for m in prompt["messages"])

        token_ids = mx.array(tokenizer.encode(text))
        tokens = []
        for token, _ in generate_step(token_ids, model, max_tokens=128):
            tokens.append(token)
            if token in _get_eos_ids(tokenizer):
                break

        output = tokenizer.decode(tokens, skip_special_tokens=True)
        results.append(output)
    return results


def run_yunshu_engine(model_path: str, prompts: list[dict]) -> list[str]:
    """Run prompts through Yunshu's BatchedEngine."""
    import asyncio

    from yunshu_engine.batched_engine import BatchedEngine

    async def _run():
        engine = BatchedEngine(model_path)
        await engine.start()

        results = []
        for prompt in prompts:
            result = await engine.generate(
                prompt=prompt["messages"],
                max_tokens=128,
                temperature=0.0,
            )
            results.append(result.text if hasattr(result, 'text') else str(result))
        await engine.stop()
        return results

    return asyncio.run(_run())


def _get_eos_ids(tokenizer) -> list[int]:
    eos_ids = set()
    for attr in ("eos_token_id", "eos_token_ids"):
        val = getattr(tokenizer, attr, None)
        if isinstance(val, (list, tuple)):
            eos_ids.update(val)
        elif isinstance(val, int):
            eos_ids.add(val)
    return list(eos_ids)


def compare_outputs(
    prompts: list[dict],
    baseline: list[str],
    engine: list[str],
) -> dict:
    """Compare outputs and produce a quality report."""
    report = {
        "total": len(prompts),
        "baseline_correct": 0,
        "engine_correct": 0,
        "both_correct": 0,
        "baseline_only": 0,
        "engine_only": 0,
        "neither_correct": 0,
        "details": [],
    }

    for i, (prompt, bl, eng) in enumerate(zip(prompts, baseline, engine, strict=False)):
        bl_pass = prompt["check"](bl)
        eng_pass = prompt["check"](eng)

        if bl_pass:
            report["baseline_correct"] += 1
        if eng_pass:
            report["engine_correct"] += 1
        if bl_pass and eng_pass:
            report["both_correct"] += 1
        elif bl_pass and not eng_pass:
            report["baseline_only"] += 1
        elif eng_pass and not bl_pass:
            report["engine_only"] += 1
        else:
            report["neither_correct"] += 1

        report["details"].append({
            "index": i,
            "baseline_pass": bl_pass,
            "engine_pass": eng_pass,
            "baseline_output": bl[:200],
            "engine_output": eng[:200],
        })

    report["baseline_accuracy"] = report["baseline_correct"] / report["total"]
    report["engine_accuracy"] = report["engine_correct"] / report["total"]
    return report


def main():
    parser = argparse.ArgumentParser(description="Yunshu model quality comparison")
    parser.add_argument("--model-path", default="models/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--quick", action="store_true", help="Fewer prompts")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()

    model_path = args.model_path
    prompts = QUICK_PROMPTS if args.quick else PROMPTS

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Download a model first: huggingface-cli download ...")
        sys.exit(1)

    print(f"Model: {model_path}")
    print(f"Prompts: {len(prompts)}")
    print()

    # Baseline: vanilla mlx-lm
    print("Running mlx-lm baseline...")
    t0 = time.time()
    baseline = run_mlx_lm_baseline(model_path, prompts)
    baseline_time = time.time() - t0
    print(f"  Done in {baseline_time:.1f}s")

    # Yunshu engine
    print("Running Yunshu engine...")
    t0 = time.time()
    engine = run_yunshu_engine(model_path, prompts)
    engine_time = time.time() - t0
    print(f"  Done in {engine_time:.1f}s")

    # Compare
    report = compare_outputs(prompts, baseline, engine)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    # Summary
    print()
    print("=" * 60)
    print("QUALITY COMPARISON REPORT")
    print("=" * 60)
    print(f"Baseline (mlx-lm) accuracy: {report['baseline_accuracy']:.0%}")
    print(f"Yunshu engine accuracy:     {report['engine_accuracy']:.0%}")
    print(f"Both correct:   {report['both_correct']}")
    print(f"Baseline only:  {report['baseline_only']}")
    print(f"Engine only:    {report['engine_only']}")
    print(f"Neither:        {report['neither_correct']}")
    print()

    if report["engine_accuracy"] >= report["baseline_accuracy"]:
        print("PASS: Yunshu engine matches or exceeds baseline quality.")
    else:
        diff = report["baseline_accuracy"] - report["engine_accuracy"]
        print(f"WARNING: Yunshu engine is {diff:.0%} below baseline quality.")
        print("Check details for regression patterns.")

    print()
    print("PERFORMANCE:")
    print(f"  Baseline: {baseline_time:.1f}s")
    print(f"  Engine:   {engine_time:.1f}s")
    if engine_time > 0:
        print(f"  Ratio:    {baseline_time / engine_time:.2f}x")

    print()
    print("DETAILS:")
    for d in report["details"]:
        status = "✓" if d["engine_pass"] else "✗"
        bl_status = "✓" if d["baseline_pass"] else "✗"
        print(f"  [{status}] Prompt {d['index']} (baseline: {bl_status})")
        print(f"       Engine:  {d['engine_output'][:80]}...")
        print(f"       Baseline: {d['baseline_output'][:80]}...")


if __name__ == "__main__":
    main()
