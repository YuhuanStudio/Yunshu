"""Benchmark MTP always-advance with llama.cpp-inspired optimizations.

Compares:
1. Baseline (mlx-lm generate_step, no MTP)
2. MTP always-advance (current implementation)
3. MTP always-advance + cooldown (from llama.cpp PR #20700)

The cooldown optimization skips the MTP draft on the cycle after a rejection,
instead doing a single-token decode to get fresh logits. This prevents the
cascade of bad drafts that occurs when MTP logits come from the rejected
draft position (13% → 95% acceptance recovery per llama.cpp PR #20700).

Usage:
    .venv/bin/python3 scripts/bench_mtp_cooldown.py
    .venv/bin/python3 scripts/bench_mtp_cooldown.py --model Qwen3.5-9B-MLX-4bit
    .venv/bin/python3 scripts/bench_mtp_cooldown.py --max-tokens 128
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


def _snap(cache):
    return [
        (('arrays', list(c.cache)) if hasattr(c, 'cache') and isinstance(c.cache, list)
         else ('kv', c.offset) if hasattr(c, 'offset')
         else (None, None))
        for c in cache
    ]


def _restore(cache, snapshot):
    for i, (kind, state) in enumerate(snapshot):
        if kind == 'arrays':
            cache[i].cache = state
        elif kind == 'kv':
            cache[i].offset = state


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


def bench_mtp_no_cooldown(model, tokenizer, prompt, max_tokens):
    """MTP always-advance WITHOUT cooldown (current approach)."""
    from mlx_lm.models.cache import make_prompt_cache

    getattr(model, "language_model", model)
    ids = mx.array(tokenizer.encode(prompt)).reshape(1, -1)
    eos_ids = _get_eos_ids(tokenizer)
    cache = make_prompt_cache(model)

    out, hidden = model(ids, cache=cache, return_hidden=True)
    mx.synchronize()
    first = _greedy(out[0, -1, :])

    primary = first
    primary_h = hidden[:, -1:, :]
    generated = [first]
    accepts = rejects = 0

    t0 = time.perf_counter()
    while len(generated) < max_tokens:
        s = _snap(cache)
        mtp_out = model.mtp_forward(primary_h, mx.array([[primary]]), None)
        draft = _greedy(mtp_out[0, -1, :])

        verify_out, verify_h = model(
            mx.array([[primary, draft]]), cache=cache, return_hidden=True,
        )
        mx.synchronize()
        v0 = _greedy(verify_out[0, 0, :])
        v1 = _greedy(verify_out[0, 1, :])

        if v0 == draft:
            accepts += 1
            generated.append(draft)
            if draft in eos_ids or len(generated) >= max_tokens:
                break
            generated.append(v1)
            primary = v1
            primary_h = verify_h[:, -1:, :]
        else:
            rejects += 1
            _restore(cache, s)
            out2, hid2 = model(mx.array([[primary]]), cache=cache, return_hidden=True)
            mx.synchronize()
            correction = _greedy(out2[0, -1, :])
            generated.append(correction)
            if correction in eos_ids or len(generated) >= max_tokens:
                break
            primary = correction
            primary_h = hid2[:, -1:, :]

    elapsed = time.perf_counter() - t0
    cycles = accepts + rejects
    return {
        "method": "mtp_no_cooldown",
        "n": len(generated),
        "time_s": round(elapsed, 3),
        "tok_s": round(len(generated) / elapsed, 1) if elapsed > 0 else 0,
        "acceptance": round(accepts / cycles, 3) if cycles > 0 else 0,
        "accepts": accepts,
        "rejects": rejects,
        "cycles": cycles,
    }


def bench_mtp_cooldown(model, tokenizer, prompt, max_tokens):
    """MTP always-advance WITH cooldown (from llama.cpp PR #20700).

    After a rejection, skip the next MTP draft and do a single-token
    backbone decode to get fresh logits from the correct position.
    """
    from yunshu_engine.mtp_decoder import MTPConfig, MTPDecoder

    decoder = MTPDecoder(model, tokenizer, MTPConfig(
        max_tokens=max_tokens,
        cooldown_on_reject=True,
    ))

    t0 = time.perf_counter()
    tokens = decoder.generate(prompt, max_tokens=max_tokens)
    elapsed = time.perf_counter() - t0

    s = decoder.stats
    cycles = s.accepts + s.rejects
    return {
        "method": "mtp_cooldown",
        "n": len(tokens),
        "time_s": round(elapsed, 3),
        "tok_s": round(len(tokens) / elapsed, 1) if elapsed > 0 else 0,
        "acceptance": round(s.accepts / cycles, 3) if cycles > 0 else 0,
        "accepts": s.accepts,
        "rejects": s.rejects,
        "cooldowns": s.cooldowns,
        "cycles": cycles,
    }


def main():
    parser = argparse.ArgumentParser(description="MTP Cooldown Benchmark")
    parser.add_argument("--model", default="Qwen3.5-4B-MLX-bf16")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=3)
    args = parser.parse_args()

    from mlx_lm.utils import load_tokenizer

    from yunshu_engine.mtp_patch import load_model_with_mtp

    model_path = ROOT / "models" / args.model
    print(f"Loading {args.model} with MTP...")
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

        r_base = bench_baseline(model, tokenizer, prompt, args.max_tokens)
        print(f"  Baseline:      {r_base['tok_s']} tok/s ({r_base['n']} tok)")

        r_no_cd = bench_mtp_no_cooldown(model, tokenizer, prompt, args.max_tokens)
        su_no = r_no_cd['tok_s'] / r_base['tok_s'] if r_base['tok_s'] else 0
        print(f"  MTP (no cool): {r_no_cd['tok_s']} tok/s ({su_no:.2f}x, "
              f"accept={r_no_cd['acceptance']:.1%}, "
              f"{r_no_cd['accepts']}/{r_no_cd['rejects']} a/r)")

        r_cd = bench_mtp_cooldown(model, tokenizer, prompt, args.max_tokens)
        su_cd = r_cd['tok_s'] / r_base['tok_s'] if r_base['tok_s'] else 0
        print(f"  MTP (cool):    {r_cd['tok_s']} tok/s ({su_cd:.2f}x, "
              f"accept={r_cd['acceptance']:.1%}, "
              f"{r_cd['accepts']}/{r_cd['rejects']} a/r, "
              f"{r_cd['cooldowns']} cooldowns)")

        results[label] = {
            "baseline": r_base,
            "mtp_no_cooldown": r_no_cd,
            "mtp_cooldown": r_cd,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    base_tps = [r["baseline"]["tok_s"] for r in results.values()]
    no_cd_tps = [r["mtp_no_cooldown"]["tok_s"] for r in results.values()]
    cd_tps = [r["mtp_cooldown"]["tok_s"] for r in results.values()]
    no_cd_ar = [r["mtp_no_cooldown"]["acceptance"] for r in results.values()]
    cd_ar = [r["mtp_cooldown"]["acceptance"] for r in results.values()]

    ab = sum(base_tps) / len(base_tps)
    a_no = sum(no_cd_tps) / len(no_cd_tps)
    a_cd = sum(cd_tps) / len(cd_tps)
    ar_no = sum(no_cd_ar) / len(no_cd_ar)
    ar_cd = sum(cd_ar) / len(cd_ar)

    print(f"{'Method':<22} {'tok/s':>8} {'Speedup':>8} {'Accept':>8}")
    print("-" * 50)
    print(f"{'Baseline':<22} {ab:>8.1f} {'1.00x':>8} {'N/A':>8}")
    print(f"{'MTP (no cooldown)':<22} {a_no:>8.1f} {a_no/ab:>7.2f}x {ar_no:>7.1%}")
    print(f"{'MTP (cooldown)':<22} {a_cd:>8.1f} {a_cd/ab:>7.2f}x {ar_cd:>7.1%}")

    out = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "summary": {
            "baseline_avg": round(ab, 1),
            "mtp_no_cooldown": {
                "avg_tps": round(a_no, 1),
                "speedup": round(a_no / ab, 2),
                "acceptance": round(ar_no, 3),
            },
            "mtp_cooldown": {
                "avg_tps": round(a_cd, 1),
                "speedup": round(a_cd / ab, 2),
                "acceptance": round(ar_cd, 3),
            },
        },
        "by_prompt": results,
    }
    p = ROOT / "bench" / "mtp_cooldown_results.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {p}")


if __name__ == "__main__":
    main()
