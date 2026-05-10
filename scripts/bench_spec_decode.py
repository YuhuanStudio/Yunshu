"""Benchmark Speculative Decoding — draft/target with real models.

Supports same-model validation and cross-model spec decode.
Uses reference-based cache rollback (no deep copy, no full rebuild).

Usage:
    # Same model validation (expect ~100% acceptance)
    PYTHONPATH=python .venv/bin/python3 scripts/bench_spec_decode.py

    # Cross-model spec decode with different K values
    PYTHONPATH=python .venv/bin/python3 scripts/bench_spec_decode.py \\
        --draft Qwen3.5-0.8B-MLX-bf16 --target Qwen3.5-4B-MLX-bf16 -K 2 4 8

    # Baseline only
    PYTHONPATH=python .venv/bin/python3 scripts/bench_spec_decode.py --baseline-only
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


def _call(model, token_id: int, cache) -> mx.array:
    out = model(mx.array([[token_id]]), cache=cache)
    logits = out if not hasattr(out, 'logits') else out.logits
    return logits[0, -1, :]


def _greedy(logits: mx.array) -> int:
    return int(mx.argmax(logits).item())


def _snapshot_cache(cache: list) -> list:
    snap = []
    for c in cache:
        if hasattr(c, 'cache') and isinstance(getattr(c, 'cache', None), list):
            snap.append(('arrays', list(c.cache)))
        elif hasattr(c, 'offset'):
            snap.append(('kv', c.offset))
        else:
            snap.append((None, None))
    return snap


def _restore_cache(cache: list, snapshot: list) -> None:
    for i, (kind, state) in enumerate(snapshot):
        if kind == 'arrays':
            cache[i].cache = state
        elif kind == 'kv':
            cache[i].offset = state


def run_spec_decode(target_model, draft_model, tokenizer, prompt, K, max_tokens):
    """Run speculative decoding and return detailed timing."""
    prompt_ids = tokenizer.encode(prompt)
    prompt_t = mx.array(prompt_ids).reshape(1, -1)

    from mlx_lm.models.cache import make_prompt_cache
    target_cache = make_prompt_cache(target_model)
    draft_cache = make_prompt_cache(draft_model)

    # Prefill both
    t_out = target_model(prompt_t, cache=target_cache)
    t_logits = t_out if not hasattr(t_out, 'logits') else t_out.logits
    first = _greedy(t_logits[0, -1, :])
    draft_model(prompt_t, cache=draft_cache)

    eos_ids = set()
    if hasattr(tokenizer, 'eos_token_id'):
        eid = tokenizer.eos_token_id
        if isinstance(eid, (list, tuple)):
            eos_ids.update(eid)
        elif eid is not None:
            eos_ids.add(eid)

    generated = [first]
    total_draft = 0
    total_accepted = 0
    steps = 0
    # Timing breakdown
    t_draft = 0.0
    t_verify = 0.0
    t_rollback = 0.0

    t0 = time.perf_counter()

    while len(generated) < max_tokens:
        # Snapshot draft cache (reference-based, no copy)
        draft_snap = _snapshot_cache(draft_cache)

        # Draft K tokens
        td0 = time.perf_counter()
        last = generated[-1]
        draft_tokens = []
        d_logits = _call(draft_model, last, draft_cache)
        draft_tokens.append(_greedy(d_logits))
        for _ in range(K - 1):
            d_logits = _call(draft_model, draft_tokens[-1], draft_cache)
            draft_tokens.append(_greedy(d_logits))
        mx.synchronize()
        t_draft += time.perf_counter() - td0
        total_draft += K

        # Target verify one-by-one
        tv0 = time.perf_counter()
        last = generated[-1]
        t_logits = _call(target_model, last, target_cache)

        accepted = 0
        rejected = False
        for j in range(K):
            tc = _greedy(t_logits)
            if tc == draft_tokens[j]:
                accepted += 1
                generated.append(draft_tokens[j])
                if draft_tokens[j] in eos_ids:
                    rejected = True
                    break
                t_logits = _call(target_model, draft_tokens[j], target_cache)
            else:
                generated.append(tc)
                rejected = True
                break

        if not rejected:
            bonus = _greedy(t_logits)
            generated.append(bonus)
        mx.synchronize()
        t_verify += time.perf_counter() - tv0

        total_accepted += accepted
        steps += 1

        # Rollback + refeed
        if accepted < K:
            tr0 = time.perf_counter()
            _restore_cache(draft_cache, draft_snap)
            refeed = [generated[-(accepted + 1) - 1]]
            refeed += generated[-(accepted + 1):]
            for tok in refeed:
                draft_model(mx.array([[tok]]), cache=draft_cache)
            mx.synchronize()
            t_rollback += time.perf_counter() - tr0

        if any(t in eos_ids for t in generated):
            break

    total_time = time.perf_counter() - t0
    ar = total_accepted / total_draft if total_draft > 0 else 0
    return {
        "tokens": generated,
        "n": len(generated),
        "total_s": round(total_time, 3),
        "tps": round(len(generated) / total_time, 1) if total_time > 0 else 0,
        "ar": round(ar, 3),
        "steps": steps,
        "t_draft_s": round(t_draft, 3),
        "t_verify_s": round(t_verify, 3),
        "t_rollback_s": round(t_rollback, 3),
        "pct_draft": round(t_draft / total_time * 100, 1) if total_time > 0 else 0,
        "pct_verify": round(t_verify / total_time * 100, 1) if total_time > 0 else 0,
        "pct_rollback": round(t_rollback / total_time * 100, 1) if total_time > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding Benchmark")
    parser.add_argument("--draft", default=None, help="Draft model (smaller)")
    parser.add_argument("--target", default="Qwen3.5-4B-MLX-bf16", help="Target model")
    parser.add_argument("--model", default=None, help="Same model for both draft+target (validation)")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("-K", "--draft-lengths", type=int, nargs="+", default=[4],
                        help="Draft lengths to test (space-separated)")
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    if args.model:
        args.draft = args.model
        args.target = args.model
    elif not args.draft:
        args.draft = args.target

    from mlx_lm.utils import load_model, load_tokenizer
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    target_path = ROOT / "models" / args.target
    if not target_path.exists():
        target_path = Path(args.target)
    print(f"Loading target: {args.target}...")
    target_model, _ = load_model(target_path)
    tokenizer = load_tokenizer(target_path)
    print("Target loaded.")

    draft_model = None
    if not args.baseline_only:
        draft_path = ROOT / "models" / args.draft
        if not draft_path.exists():
            draft_path = Path(args.draft)
        print(f"Loading draft: {args.draft}...")
        draft_model, _ = load_model(draft_path)
        draft_tok = load_tokenizer(draft_path)
        print("Draft loaded.")

        if draft_tok.encode("Hello") != tokenizer.encode("Hello"):
            print("WARNING: Tokenizers are different! Results may be incorrect.")

    prompts = [
        "The capital of France is",
        "In machine learning, gradient descent works by",
        "The key difference between TCP and UDP is that",
    ][:args.num_prompts]

    sampler = make_sampler(temp=0.0)

    # ── Baseline ──
    print(f"\n=== Baseline ({args.target}, {args.max_tokens} tok) ===")
    baseline = []
    for i, prompt in enumerate(prompts):
        ids = mx.array(tokenizer.encode(prompt))
        tokens = []
        t0 = time.perf_counter()
        for tok, _ in generate_step(ids, target_model, max_tokens=args.max_tokens, sampler=sampler):
            tokens.append(tok)
        mx.synchronize()
        elapsed = time.perf_counter() - t0
        tps = len(tokens) / elapsed if elapsed > 0 else 0
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        baseline.append({"n": len(tokens), "s": round(elapsed, 3), "tps": round(tps, 1), "text": text[:100]})
        print(f"  [{i+1}] {len(tokens)} tok, {elapsed:.3f}s, {tps:.1f} tok/s")

    if args.baseline_only or draft_model is None:
        return

    ab = sum(r["tps"] for r in baseline) / len(baseline)

    # ── Speculative for each K ──
    all_results = {}
    for K in args.draft_lengths:
        print(f"\n=== Speculative ({args.draft}→{args.target}, K={K}) ===")
        spec = []
        for i, prompt in enumerate(prompts):
            result = run_spec_decode(target_model, draft_model, tokenizer, prompt, K, args.max_tokens)
            text = tokenizer.decode(result["tokens"], skip_special_tokens=True)
            b_text = baseline[i]["text"]
            match = text[:100] == b_text[:100]

            print(f"  [{i+1}] {result['n']} tok, {result['total_s']:.3f}s, {result['tps']:.1f} tok/s, "
                  f"accept={result['ar']:.1%}, {result['steps']} steps "
                  f"[draft={result['pct_draft']}% verify={result['pct_verify']}% rollback={result['pct_rollback']}%] "
                  f"{'MATCH' if match else 'DIFF'}")
            spec.append(result)

        asp = sum(r["tps"] for r in spec) / len(spec)
        aa = sum(r["ar"] for r in spec) / len(spec)
        su = asp / ab if ab > 0 else 0

        # Average timing breakdown
        avg_draft = sum(r["pct_draft"] for r in spec) / len(spec)
        avg_verify = sum(r["pct_verify"] for r in spec) / len(spec)
        avg_rollback = sum(r["pct_rollback"] for r in spec) / len(spec)

        print(f"  → Speedup: {su:.2f}x, Acceptance: {aa:.1%}")
        print(f"    Time split: draft={avg_draft:.1f}% verify={avg_verify:.1f}% rollback={avg_rollback:.1f}%")

        all_results[K] = {
            "speedup": round(su, 2),
            "acceptance": round(aa, 3),
            "avg_tps": round(asp, 1),
            "time_split": {"draft_pct": round(avg_draft, 1), "verify_pct": round(avg_verify, 1),
                           "rollback_pct": round(avg_rollback, 1)},
        }

    # ── Summary ──
    print("\n=== Summary ===")
    print(f"Baseline: {ab:.1f} tok/s ({args.target})")
    for K, res in all_results.items():
        print(f"  K={K}: {res['avg_tps']:.1f} tok/s ({res['speedup']:.2f}x, accept={res['acceptance']:.1%}, "
              f"draft={res['time_split']['draft_pct']}% verify={res['time_split']['verify_pct']}% "
              f"rollback={res['time_split']['rollback_pct']}%)")

    out = {
        "draft": args.draft, "target": args.target, "max_tokens": args.max_tokens,
        "baseline_avg": round(ab, 1), "results_by_K": all_results,
    }
    p = ROOT / "bench" / "spec_decode_results.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {p}")


if __name__ == "__main__":
    main()
