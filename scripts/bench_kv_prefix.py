"""Benchmark KV Prefix Cache — measure multi-turn TTFT speedup.

Scenarios:
1. Cold start (no cache) — baseline TTFT
2. Exact prefix hit (same system prompt) — should be ~0ms TTFT
3. Partial prefix hit (growing conversation) — proportional speedup
4. Cache under pressure (many concurrent sessions)

Usage:
    PYTHONPATH=. .venv/bin/python3 scripts/bench_kv_prefix.py
    PYTHONPATH=. .venv/bin/python3 scripts/bench_kv_prefix.py --model Qwen2.5-0.5B-Instruct-4bit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import mlx.core as mx


def main():
    parser = argparse.ArgumentParser(description="KV Prefix Cache Benchmark")
    parser.add_argument("--model", default="Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--turns", type=int, default=5)
    args = parser.parse_args()

    from mlx_lm.utils import load_model_and_tokenizer
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import make_prompt_cache
    from yunshu_engine.kv_prefix_cache import KVPrefixCache

    print(f"Loading {args.model}...")
    model_path = str(ROOT / "models" / args.model)
    if not Path(model_path).exists():
        # Try HuggingFace hub
        model_path = args.model

    model, tokenizer = load_model_and_tokenizer(model_path)
    print(f"Model loaded: {args.model}")

    # Multi-turn conversation simulation
    system_prompt = "You are a helpful assistant. Be concise."
    turns = [
        "What is the capital of France?",
        "What about Germany?",
        "And Japan?",
        "Tell me about Brazil.",
        "What about Australia?",
    ]
    turns = turns[:args.turns]

    sampler = make_sampler(temp=0.0)
    prefix_cache = KVPrefixCache(max_entries=32, min_prefix_length=16)

    results = {
        "cold": [],       # TTFT without cache
        "cached": [],     # TTFT with cache
        "speedup": [],    # ratio
    }

    conversation_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

    # ── Cold pass: no cache ──
    print("\n=== Cold Start (no cache) ===")
    full_prompt = conversation_text
    for i, user_msg in enumerate(turns):
        full_prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        ids = mx.array(tokenizer.encode(full_prompt))

        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        first = True
        ttft = 0.0
        tok_count = 0
        for token, _logits in generate_step(
            ids, model, max_tokens=args.max_tokens, sampler=sampler,
            prompt_cache=cache,
        ):
            if first:
                ttft = time.perf_counter() - t0
                first = False
            tok_count += 1
            if tok_count >= args.max_tokens:
                break
        mx.synchronize()

        # Append response to conversation
        response = tokenizer.decode([token])
        full_prompt += response + "<|im_end|>\n"

        results["cold"].append({
            "turn": i + 1,
            "prompt_tokens": len(ids),
            "ttft_ms": round(ttft * 1000, 1),
            "tok_count": tok_count,
        })
        print(f"  Turn {i+1}: {len(ids)} prompt tokens, TTFT={ttft*1000:.1f}ms, {tok_count} tokens")

    # ── Cached pass: with KV prefix cache ──
    print("\n=== Cached (with KV prefix cache) ===")
    full_prompt = conversation_text
    for i, user_msg in enumerate(turns):
        full_prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        ids = mx.array(tokenizer.encode(full_prompt))

        cached_kv, remaining, matched = prefix_cache.get(ids)
        if cached_kv is not None:
            cache = cached_kv
            ids_to_prefill = ids[matched:]
        else:
            cache = make_prompt_cache(model)
            ids_to_prefill = ids

        t0 = time.perf_counter()
        first = True
        ttft = 0.0
        tok_count = 0
        for token, _logits in generate_step(
            ids_to_prefill, model, max_tokens=args.max_tokens, sampler=sampler,
            prompt_cache=cache,
        ):
            if first:
                ttft = time.perf_counter() - t0
                first = False
            tok_count += 1
            if tok_count >= args.max_tokens:
                break
        mx.synchronize()

        # Store cache for next turn
        prefix_cache.add(ids, cache)

        response = tokenizer.decode([token])
        full_prompt += response + "<|im_end|>\n"

        cached_tokens = matched if cached_kv is not None else 0
        results["cached"].append({
            "turn": i + 1,
            "prompt_tokens": len(ids),
            "cached_tokens": cached_tokens,
            "remaining_tokens": remaining if cached_kv is not None else len(ids),
            "ttft_ms": round(ttft * 1000, 1),
            "tok_count": tok_count,
            "cache_hit": cached_kv is not None,
        })
        print(
            f"  Turn {i+1}: {len(ids)} prompt tokens, "
            f"cached={cached_tokens}/{len(ids)}, "
            f"TTFT={ttft*1000:.1f}ms, {tok_count} tokens"
        )

    # ── Summary ──
    print("\n=== Summary ===")
    print(f"{'Turn':<6} {'Cold TTFT':<12} {'Cached TTFT':<12} {'Speedup':<10} {'Cached Tokens':<15}")
    print("-" * 55)

    for i in range(len(turns)):
        cold_ttft = results["cold"][i]["ttft_ms"]
        cached_ttft = results["cached"][i]["ttft_ms"]
        cached_tokens = results["cached"][i]["cached_tokens"]
        prompt_tokens = results["cached"][i]["prompt_tokens"]
        speedup = cold_ttft / cached_ttft if cached_ttft > 0 else float("inf")

        results["speedup"].append(speedup)
        print(
            f"{i+1:<6} {cold_ttft:<12.1f} {cached_ttft:<12.1f} "
            f"{speedup:<10.2f}x {cached_tokens}/{prompt_tokens}"
        )

    avg_cold = sum(r["ttft_ms"] for r in results["cold"]) / len(results["cold"])
    avg_cached = sum(r["ttft_ms"] for r in results["cached"]) / len(results["cached"])
    avg_speedup = avg_cold / avg_cached if avg_cached > 0 else float("inf")

    print(f"\nAverage TTFT: cold={avg_cold:.1f}ms, cached={avg_cached:.1f}ms, speedup={avg_speedup:.2f}x")

    stats = prefix_cache.get_stats()
    print(f"Cache stats: {stats['entries']} entries, {stats['total_cached_tokens']} total tokens")

    # Save results
    output = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "turns": args.turns,
        "avg_cold_ttft_ms": round(avg_cold, 1),
        "avg_cached_ttft_ms": round(avg_cached, 1),
        "avg_speedup": round(avg_speedup, 2),
        "cache_stats": stats,
        "turns_detail": {
            "cold": results["cold"],
            "cached": results["cached"],
        },
    }
    out_path = ROOT / "bench" / "kv_prefix_cache_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
