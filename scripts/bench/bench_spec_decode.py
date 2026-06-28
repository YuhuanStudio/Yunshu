"""Benchmark Speculative Decoding — MTP, cross-model, and baseline.

Supports three modes:
  1. --mtp           Pure MTP (same-model MTP head drafts, backbone verifies)
  2. --draft X       Cross-model spec decode (X→target)
  3. --baseline-only  Just baseline (no speculative decoding)

Usage:
    # Pure MTP on any Qwen3.5 model
    PYTHONPATH=python .venv/bin/python3 scripts/bench_spec_decode.py \
        --mtp --target Qwen3.5-4B-MLX-bf16 -K 2 4

    # Cross-model spec decode
    PYTHONPATH=python .venv/bin/python3 scripts/bench_spec_decode.py \
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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))

import mlx.core as mx


def _greedy(logits: mx.array) -> int:
    return int(mx.argmax(logits).item())


def _call(model, token_id: int, cache) -> mx.array:
    out = model(mx.array([[token_id]]), cache=cache)
    logits = out if not hasattr(out, 'logits') else out.logits
    return logits[0, -1, :]


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


# ── Cross-model spec decode ──


def run_cross_model(target_model, draft_model, tokenizer, prompt, K, max_tokens):
    """Cross-model speculative decoding (draft model → target verify)."""
    prompt_ids = tokenizer.encode(prompt)
    prompt_t = mx.array(prompt_ids).reshape(1, -1)

    from mlx_lm.models.cache import make_prompt_cache
    target_cache = make_prompt_cache(target_model)
    draft_cache = make_prompt_cache(draft_model)

    t_out = target_model(prompt_t, cache=target_cache)
    t_logits = t_out if not hasattr(t_out, 'logits') else t_out.logits
    first = _greedy(t_logits[0, -1, :])
    draft_model(prompt_t, cache=draft_cache)

    eos_ids = _get_eos_ids(tokenizer)
    generated = [first]
    total_draft = 0
    total_accepted = 0
    steps = 0
    t_draft = 0.0
    t_verify = 0.0
    t_rollback = 0.0

    t0 = time.perf_counter()

    while len(generated) < max_tokens:
        draft_snap = _snapshot_cache(draft_cache)

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

        tv0 = time.perf_counter()
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
        "mode": "cross_model",
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


# ── Pure MTP spec decode ──


def run_mtp(model, tokenizer, prompt, max_tokens):
    """Pure MTP — same model's MTP head drafts 1 token, backbone verifies.

    The MTP head predicts token t+2 from hidden state at t + embedding of t+1.
    Each step: backbone decodes 1 token → MTP drafts 1 token → backbone verifies.
    When MTP matches, we get 2 tokens per backbone step.
    """
    from mlx_lm.models.cache import make_prompt_cache


    inner = getattr(model, "language_model", model)
    if not hasattr(inner, "mtp"):
        raise RuntimeError("Model has no MTP head. Use load_model_with_mtp().")

    prompt_ids = tokenizer.encode(prompt)
    prompt_t = mx.array(prompt_ids).reshape(1, -1)

    cache = make_prompt_cache(model)

    out, hidden = model(prompt_t, cache=cache, return_hidden=True)
    first = _greedy(out[0, -1, :])

    eos_ids = _get_eos_ids(tokenizer)
    generated = [first]
    mtp_calls = 0
    mtp_matches = 0
    t_mtp = 0.0
    t_verify = 0.0

    t0 = time.perf_counter()

    while len(generated) < max_tokens:
        last_hidden = hidden[:, -1:, :]
        current_tok = generated[-1]
        next_ids = mx.array([[current_tok]])

        # MTP draft
        tm0 = time.perf_counter()
        mtp_cache = inner.make_mtp_cache()
        mtp_logits = model.mtp_forward(last_hidden, next_ids, mtp_cache)
        mx.synchronize()
        t_mtp += time.perf_counter() - tm0
        mtp_calls += 1
        mtp_tok = _greedy(mtp_logits[0, -1, :])

        # Backbone verify + advance
        tv0 = time.perf_counter()
        t_out, hidden = model(next_ids, cache=cache, return_hidden=True)
        backbone_tok = _greedy(t_out[0, -1, :])
        mx.synchronize()
        t_verify += time.perf_counter() - tv0

        if mtp_tok == backbone_tok:
            mtp_matches += 1
            generated.append(backbone_tok)
        else:
            generated.append(backbone_tok)

        if backbone_tok in eos_ids:
            break

    total_time = time.perf_counter() - t0
    ar = mtp_matches / mtp_calls if mtp_calls > 0 else 0
    return {
        "mode": "mtp",
        "tokens": generated,
        "n": len(generated),
        "total_s": round(total_time, 3),
        "tps": round(len(generated) / total_time, 1) if total_time > 0 else 0,
        "ar": round(ar, 3),
        "mtp_calls": mtp_calls,
        "t_mtp_s": round(t_mtp, 3),
        "t_verify_s": round(t_verify, 3),
        "pct_mtp": round(t_mtp / total_time * 100, 1) if total_time > 0 else 0,
        "pct_verify": round(t_verify / total_time * 100, 1) if total_time > 0 else 0,
    }


# ── Helpers ──


def _get_eos_ids(tokenizer) -> set:
    eos_ids = set()
    if hasattr(tokenizer, 'eos_token_id'):
        eid = tokenizer.eos_token_id
        if isinstance(eid, (list, tuple)):
            eos_ids.update(eid)
        elif eid is not None:
            eos_ids.add(eid)
    return eos_ids


# ── Main ──


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding Benchmark")
    parser.add_argument("--target", default="Qwen3.5-4B-MLX-bf16", help="Target model")
    parser.add_argument("--draft", default=None, help="Draft model for cross-model mode")
    parser.add_argument("--mtp", action="store_true", help="Use same-model MTP head")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("-K", "--draft-lengths", type=int, nargs="+", default=[4],
                        help="Draft lengths for cross-model mode (space-separated)")
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    from mlx_lm import load as mlx_load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load_model, load_tokenizer

    if args.mtp:
        from yunshu_engine.mtp_patch import load_model_with_mtp

    target_path = ROOT / "models" / args.target

    # Load model — local path takes precedence; otherwise HF lookup via mlx_lm.load()
    if args.mtp:
        if not target_path.exists():
            raise FileNotFoundError(
                f"MTP requires local model dir at {target_path}; HF fallback not supported for --mtp"
            )
        print(f"Loading target with MTP: {args.target}...")
        target_model = load_model_with_mtp(str(target_path))
        tokenizer = load_tokenizer(target_path)
    elif target_path.exists():
        print(f"Loading target: {args.target}...")
        target_model, _ = load_model(target_path)
        tokenizer = load_tokenizer(target_path)
    else:
        print(f"Loading target via HF: {args.target}...")
        target_model, tokenizer = mlx_load(args.target)
    print("Target loaded.")

    draft_model = None
    if not args.baseline_only and args.draft:
        draft_path = ROOT / "models" / args.draft
        if draft_path.exists():
            print(f"Loading draft: {args.draft}...")
            draft_model, _ = load_model(draft_path)
            draft_tok = load_tokenizer(draft_path)
        else:
            print(f"Loading draft via HF: {args.draft}...")
            draft_model, draft_tok = mlx_load(args.draft)
        print("Draft loaded.")
        if draft_tok.encode("Hello") != tokenizer.encode("Hello"):
            print("WARNING: Tokenizers are different!")

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

    if args.baseline_only:
        return

    ab = sum(r["tps"] for r in baseline) / len(baseline)
    all_results = {}

    # ── Pure MTP ──
    if args.mtp:
        print(f"\n=== Pure MTP ({args.target}, {args.max_tokens} tok) ===")
        mtp_results = []
        for i, prompt in enumerate(prompts):
            result = run_mtp(target_model, tokenizer, prompt, args.max_tokens)
            text = tokenizer.decode(result["tokens"], skip_special_tokens=True)
            b_text = baseline[i]["text"]
            match = text[:100] == b_text[:100]

            print(f"  [{i+1}] {result['n']} tok, {result['total_s']:.3f}s, {result['tps']:.1f} tok/s, "
                  f"accept={result['ar']:.1%}, {result['mtp_calls']} calls "
                  f"[mtp={result['pct_mtp']}% verify={result['pct_verify']}%] "
                  f"{'MATCH' if match else 'DIFF'}")
            mtp_results.append(result)

        am_tps = sum(r["tps"] for r in mtp_results) / len(mtp_results)
        am_ar = sum(r["ar"] for r in mtp_results) / len(mtp_results)
        su = am_tps / ab if ab > 0 else 0
        avg_mtp = sum(r["pct_mtp"] for r in mtp_results) / len(mtp_results)
        avg_verify = sum(r["pct_verify"] for r in mtp_results) / len(mtp_results)

        print(f"  → Speedup: {su:.2f}x, Acceptance: {am_ar:.1%}")
        print(f"    Time split: mtp={avg_mtp:.1f}% verify={avg_verify:.1f}%")

        all_results["mtp"] = {
            "speedup": round(su, 2),
            "acceptance": round(am_ar, 3),
            "avg_tps": round(am_tps, 1),
            "time_split": {"mtp_pct": round(avg_mtp, 1), "verify_pct": round(avg_verify, 1)},
        }

    # ── Cross-model ──
    if draft_model is not None:
        for K in args.draft_lengths:
            print(f"\n=== Cross-model ({args.draft}→{args.target}, K={K}) ===")
            spec = []
            for i, prompt in enumerate(prompts):
                result = run_cross_model(target_model, draft_model, tokenizer, prompt, K, args.max_tokens)
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
            avg_draft = sum(r["pct_draft"] for r in spec) / len(spec)
            avg_verify = sum(r["pct_verify"] for r in spec) / len(spec)
            avg_rollback = sum(r["pct_rollback"] for r in spec) / len(spec)

            print(f"  → Speedup: {su:.2f}x, Acceptance: {aa:.1%}")
            print(f"    Time split: draft={avg_draft:.1f}% verify={avg_verify:.1f}% rollback={avg_rollback:.1f}%")

            all_results[f"cross_K{K}"] = {
                "speedup": round(su, 2),
                "acceptance": round(aa, 3),
                "avg_tps": round(asp, 1),
                "time_split": {"draft_pct": round(avg_draft, 1), "verify_pct": round(avg_verify, 1),
                               "rollback_pct": round(avg_rollback, 1)},
            }

    # ── Summary ──
    print("\n=== Summary ===")
    print(f"Baseline: {ab:.1f} tok/s ({args.target})")
    for mode, res in all_results.items():
        if mode == "mtp":
            print(f"  MTP: {res['avg_tps']:.1f} tok/s ({res['speedup']:.2f}x, accept={res['acceptance']:.1%}, "
                  f"mtp={res['time_split']['mtp_pct']}% verify={res['time_split']['verify_pct']}%)")
        else:
            print(f"  {mode}: {res['avg_tps']:.1f} tok/s ({res['speedup']:.2f}x, accept={res['acceptance']:.1%}, "
                  f"draft={res['time_split']['draft_pct']}% verify={res['time_split']['verify_pct']}% "
                  f"rollback={res['time_split']['rollback_pct']}%)")

    out = {
        "target": args.target,
        "draft": args.draft,
        "mtp": args.mtp,
        "max_tokens": args.max_tokens,
        "baseline_avg": round(ab, 1),
        "results": all_results,
    }
    p = ROOT / "bench" / "spec_decode_results.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {p}")


if __name__ == "__main__":
    main()
