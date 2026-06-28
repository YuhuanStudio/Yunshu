"""Benchmark MTP with n_confirmed=1 vs old restore+refeed approach.

Compares:
1. Baseline (mlx-lm generate_step, no MTP)
2. MTP with old restore+refeed (current approach, n_confirmed=False)
3. MTP with n_confirmed=1 (new approach, zero-cost reject)

Expected results (based on throughput math):
  Old: speedup = (1+p) / (1.15 + (1-p)*1.0) → negative for p < 0.735
  New: speedup = (1+p) / 1.15 → positive for p > 0.15

Usage:
    .venv/bin/python3 scripts/bench_n_confirmed.py
    .venv/bin/python3 scripts/bench_n_confirmed.py --model Qwen3.5-9B-MLX-4bit
    .venv/bin/python3 scripts/bench_n_confirmed.py --max-tokens 128
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))

import mlx.core as mx


def _greedy(logits: mx.array) -> int:
    return int(mx.argmax(logits).item())


def _get_eos_ids(tokenizer) -> set:
    eos_ids = set()
    if hasattr(tokenizer, 'eos_token_id'):
        eid = tokenizer.eos_token_id
        if isinstance(eid, (list, tuple)):
            eos_ids.update(eid)
        elif eid is not None:
            eos_ids.add(eid)
    return eos_ids


def bench_baseline(model, tokenizer, prompt, max_tokens):
    """Standard mlx-lm generate_step (no MTP)."""
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    ids = mx.array(tokenizer.encode(prompt))
    tokens = []
    t0 = time.perf_counter()
    for tok, _ in generate_step(ids, model, max_tokens=max_tokens, sampler=sampler):
        tokens.append(tok)
    mx.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "method": "baseline",
        "n": len(tokens),
        "time_s": round(elapsed, 3),
        "tok_s": round(len(tokens) / elapsed, 1) if elapsed > 0 else 0,
    }


def bench_mtp(model, tokenizer, prompt, max_tokens, use_n_confirmed: bool):
    """Run MTP with specified n_confirmed mode."""
    from yunshu_engine.mtp_decoder import MTPConfig, MTPDecoder

    config = MTPConfig(
        max_tokens=max_tokens,
        cooldown_on_reject=False,
        use_n_confirmed=use_n_confirmed,
    )
    decoder = MTPDecoder(model, tokenizer, config)

    t0 = time.perf_counter()
    tokens = decoder.generate(prompt, max_tokens=max_tokens)
    mx.synchronize()
    elapsed = time.perf_counter() - t0

    s = decoder.stats
    cycles = s.accepts + s.rejects
    label = "n_confirmed" if use_n_confirmed else "restore_refeed"
    return {
        "method": f"mtp_{label}",
        "n": len(tokens),
        "time_s": round(elapsed, 3),
        "tok_s": round(len(tokens) / elapsed, 1) if elapsed > 0 else 0,
        "acceptance": round(s.accepts / cycles, 3) if cycles > 0 else 0,
        "accepts": s.accepts,
        "rejects": s.rejects,
        "cycles": cycles,
    }


def main():
    parser = argparse.ArgumentParser(description="MTP n_confirmed Benchmark")
    parser.add_argument("--model", default="Qwen3.5-4B-MLX-bf16")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    from mlx_lm.utils import load_tokenizer

    from yunshu_engine.mtp_patch import load_model_with_mtp
    from yunshu_engine.n_confirmed_patch import apply_n_confirmed_patch

    model_path = ROOT / "models" / args.model
    print(f"Loading {args.model} with MTP + n_confirmed...")
    apply_n_confirmed_patch()
    model = load_model_with_mtp(str(model_path))
    tokenizer = load_tokenizer(model_path)
    print("Loaded.\n")

    prompts = [
        "The capital of France is",
        "In machine learning, gradient descent works by",
        "The key difference between TCP and UDP is that",
    ][:args.num_prompts]

    results = {}
    for prompt in prompts:
        label = prompt[:40] + "..."
        print(f"=== {label} ===")

        # Warmup
        for _ in range(args.warmup):
            bench_baseline(model, tokenizer, prompt, min(16, args.max_tokens))

        r_base = bench_baseline(model, tokenizer, prompt, args.max_tokens)
        print(f"  Baseline:       {r_base['tok_s']:>6.1f} tok/s ({r_base['n']} tok)")

        r_old = bench_mtp(model, tokenizer, prompt, args.max_tokens, use_n_confirmed=False)
        su_old = r_old['tok_s'] / r_base['tok_s'] if r_base['tok_s'] else 0
        print(f"  MTP (old):      {r_old['tok_s']:>6.1f} tok/s ({su_old:.2f}x, "
              f"accept={r_old['acceptance']:.1%}, "
              f"{r_old['accepts']}/{r_old['rejects']} a/r)")

        r_new = bench_mtp(model, tokenizer, prompt, args.max_tokens, use_n_confirmed=True)
        su_new = r_new['tok_s'] / r_base['tok_s'] if r_base['tok_s'] else 0
        print(f"  MTP (n_conf):   {r_new['tok_s']:>6.1f} tok/s ({su_new:.2f}x, "
              f"accept={r_new['acceptance']:.1%}, "
              f"{r_new['accepts']}/{r_new['rejects']} a/r)")

        # Token consistency check (greedy should produce same tokens)
        if r_old['n'] != r_new['n']:
            print(f"  ⚠ Token count mismatch: old={r_old['n']} new={r_new['n']}")

        results[label] = {
            "baseline": r_base,
            "mtp_restore_refeed": r_old,
            "mtp_n_confirmed": r_new,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    base_tps = [r["baseline"]["tok_s"] for r in results.values()]
    old_tps = [r["mtp_restore_refeed"]["tok_s"] for r in results.values()]
    new_tps = [r["mtp_n_confirmed"]["tok_s"] for r in results.values()]
    old_ar = [r["mtp_restore_refeed"]["acceptance"] for r in results.values()]
    new_ar = [r["mtp_n_confirmed"]["acceptance"] for r in results.values()]

    ab = sum(base_tps) / len(base_tps)
    a_old = sum(old_tps) / len(old_tps)
    a_new = sum(new_tps) / len(new_tps)
    ar_old = sum(old_ar) / len(old_ar)
    ar_new = sum(new_ar) / len(new_ar)

    print(f"{'Method':<24} {'tok/s':>8} {'Speedup':>8} {'Accept':>8}")
    print("-" * 52)
    print(f"{'Baseline':<24} {ab:>8.1f} {'1.00x':>8} {'N/A':>8}")
    print(f"{'MTP (restore+refeed)':<24} {a_old:>8.1f} {a_old/ab:>7.2f}x {ar_old:>7.1%}")
    print(f"{'MTP (n_confirmed=1)':<24} {a_new:>8.1f} {a_new/ab:>7.2f}x {ar_new:>7.1%}")

    # Theoretical prediction
    if ab > 0:
        print(f"\nTheoretical (p={ar_new:.1%}):")
        print(f"  Old: (1+{ar_new:.2f}) / (1.15 + (1-{ar_new:.2f})*1.0) = "
              f"{(1+ar_new) / (1.15 + (1-ar_new)*1.0):.2f}x")
        print(f"  New: (1+{ar_new:.2f}) / 1.15 = {(1+ar_new) / 1.15:.2f}x")

    out = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "summary": {
            "baseline_avg": round(ab, 1),
            "mtp_restore_refeed": {
                "avg_tps": round(a_old, 1),
                "speedup": round(a_old / ab, 2) if ab else 0,
                "acceptance": round(ar_old, 3),
            },
            "mtp_n_confirmed": {
                "avg_tps": round(a_new, 1),
                "speedup": round(a_new / ab, 2) if ab else 0,
                "acceptance": round(ar_new, 3),
            },
        },
        "by_prompt": results,
    }
    p = ROOT / "bench" / "n_confirmed_results.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {p}")


if __name__ == "__main__":
    main()
