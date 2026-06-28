"""Cross-framework MTP benchmark — Yunshu vs mlx-lm baseline vs spec decode.

Compares:
1. mlx-lm baseline (generate_step, no MTP)
2. Yunshu MTP (same-model MTP draft + backbone verify)
3. Yunshu cross-model spec decode (0.8B→4B)

Outputs JSON results and formatted table.

Usage:
    .venv/bin/python3 scripts/bench_mtp_horizontal.py
    .venv/bin/python3 scripts/bench_mtp_horizontal.py --model Qwen3.5-4B-MLX-bf16
    .venv/bin/python3 scripts/bench_mtp_horizontal.py --max-tokens 128
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


def bench_baseline(model, tokenizer, prompt: str, max_tokens: int) -> dict:
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
        "n_tokens": len(tokens),
        "time_s": round(elapsed, 3),
        "tok_s": round(len(tokens) / elapsed, 1) if elapsed > 0 else 0,
    }


def bench_mtp(model, tokenizer, prompt: str, max_tokens: int, model_name: str) -> dict:
    """Yunshu MTP — same-model MTP head drafts 1 token, backbone verifies."""
    from mlx_lm.models.cache import make_prompt_cache

    inner = getattr(model, "language_model", model)
    if not hasattr(inner, "mtp"):
        return {"method": "mtp", "error": "no MTP module"}

    prompt_ids = tokenizer.encode(prompt)
    prompt_t = mx.array(prompt_ids).reshape(1, -1)
    cache = make_prompt_cache(model)

    out, hidden = model(prompt_t, cache=cache, return_hidden=True)
    first_tok = int(mx.argmax(out[0, -1, :]).item())
    generated = [first_tok]

    eos_ids = set()
    if hasattr(tokenizer, "eos_token_id"):
        eid = tokenizer.eos_token_id
        if isinstance(eid, (list, tuple)):
            eos_ids.update(eid)
        elif eid is not None:
            eos_ids.add(eid)

    total_mtp_calls = 0
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
        total_mtp_calls += 1

        mtp_tok = int(mx.argmax(mtp_logits[0, -1, :]).item())

        # Backbone verify
        tv0 = time.perf_counter()
        t_out, hidden = model(next_ids, cache=cache, return_hidden=True)
        backbone_tok = int(mx.argmax(t_out[0, -1, :]).item())
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
    ar = mtp_matches / total_mtp_calls if total_mtp_calls > 0 else 0
    return {
        "method": "mtp",
        "model": model_name,
        "n_tokens": len(generated),
        "time_s": round(total_time, 3),
        "tok_s": round(len(generated) / total_time, 1) if total_time > 0 else 0,
        "acceptance": round(ar, 3),
        "t_mtp_s": round(t_mtp, 3),
        "t_verify_s": round(t_verify, 3),
        "pct_mtp": round(t_mtp / total_time * 100, 1) if total_time > 0 else 0,
        "pct_verify": round(t_verify / total_time * 100, 1) if total_time > 0 else 0,
    }


def bench_cross_model_spec(target_model, draft_model, tokenizer, prompt: str,
                           max_tokens: int, K: int = 4) -> dict:
    """Cross-model speculative decoding (draft → target verify)."""
    from mlx_lm.models.cache import make_prompt_cache

    prompt_ids = tokenizer.encode(prompt)
    prompt_t = mx.array(prompt_ids).reshape(1, -1)

    target_cache = make_prompt_cache(target_model)
    draft_cache = make_prompt_cache(draft_model)

    target_model(prompt_t, cache=target_cache)
    draft_model(prompt_t, cache=draft_cache)

    def _call(m, tok, cache):
        out = m(mx.array([[tok]]), cache=cache)
        return out[0, -1, :] if out.ndim == 3 else out

    def _greedy(logits):
        return int(mx.argmax(logits).item())

    # Get first token from target
    _call(target_model, prompt_ids[-1], target_cache)
    # Actually we already did prefill, need to get last position logits
    out = target_model(prompt_t, cache=target_cache)
    first = _greedy(out[0, -1, :])
    # We prefilled twice for target... let's fix by using new caches
    target_cache = make_prompt_cache(target_model)
    draft_cache = make_prompt_cache(draft_model)

    t_out = target_model(prompt_t, cache=target_cache)
    first = _greedy(t_out[0, -1, :])
    draft_model(prompt_t, cache=draft_cache)

    eos_ids = set()
    if hasattr(tokenizer, "eos_token_id"):
        eid = tokenizer.eos_token_id
        if isinstance(eid, (list, tuple)):
            eos_ids.update(eid)
        elif eid is not None:
            eos_ids.add(eid)

    generated = [first]
    total_draft = 0
    total_accepted = 0
    steps = 0

    def _snapshot(cache):
        snap = []
        for c in cache:
            if hasattr(c, 'cache') and isinstance(getattr(c, 'cache', None), list):
                snap.append(('arrays', list(c.cache)))
            elif hasattr(c, 'offset'):
                snap.append(('kv', c.offset))
            else:
                snap.append((None, None))
        return snap

    def _restore(cache, snapshot):
        for i, (kind, state) in enumerate(snapshot):
            if kind == 'arrays':
                cache[i].cache = state
            elif kind == 'kv':
                cache[i].offset = state

    t0 = time.perf_counter()

    while len(generated) < max_tokens:
        draft_snap = _snapshot(draft_cache)
        last = generated[-1]

        # Draft K tokens
        draft_tokens = []
        d_logits = _call(draft_model, last, draft_cache)
        draft_tokens.append(_greedy(d_logits))
        for _ in range(K - 1):
            d_logits = _call(draft_model, draft_tokens[-1], draft_cache)
            draft_tokens.append(_greedy(d_logits))
        mx.synchronize()
        total_draft += K

        # Verify
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
            generated.append(_greedy(t_logits))
        mx.synchronize()

        total_accepted += accepted
        steps += 1

        if accepted < K:
            _restore(draft_cache, draft_snap)
            refeed = [generated[-(accepted + 1) - 1]] + generated[-(accepted + 1):]
            for tok in refeed:
                draft_model(mx.array([[tok]]), cache=draft_cache)

        if any(t in eos_ids for t in generated):
            break

    total_time = time.perf_counter() - t0
    ar = total_accepted / total_draft if total_draft > 0 else 0
    return {
        "method": f"cross_model_K{K}",
        "n_tokens": len(generated),
        "time_s": round(total_time, 3),
        "tok_s": round(len(generated) / total_time, 1) if total_time > 0 else 0,
        "acceptance": round(ar, 3),
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-framework MTP benchmark")
    parser.add_argument("--model", default="Qwen3.5-4B-MLX-bf16")
    parser.add_argument("--draft", default="Qwen3.5-0.8B-MLX-bf16")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("-K", type=int, default=4)
    parser.add_argument("--num-prompts", type=int, default=3)
    args = parser.parse_args()

    from mlx_lm.utils import load_model, load_tokenizer

    from yunshu_engine.mtp_patch import load_model_with_mtp

    model_path = ROOT / "models" / args.model
    draft_path = ROOT / "models" / args.draft

    print(f"Loading {args.model} with MTP...")
    model = load_model_with_mtp(str(model_path))
    tokenizer = load_tokenizer(model_path)

    print(f"Loading draft {args.draft}...")
    draft_model, _ = load_model(draft_path)
    print("Models loaded.\n")

    prompts = [
        "The capital of France is",
        "In machine learning, gradient descent works by",
        "The key difference between TCP and UDP is that",
    ][:args.num_prompts]

    all_results = {}

    for prompt in prompts:
        label = prompt[:40] + "..."
        print(f"\n=== Prompt: {label} ===")

        # 1. Baseline
        r_base = bench_baseline(model, tokenizer, prompt, args.max_tokens)
        print(f"  Baseline: {r_base['tok_s']} tok/s ({r_base['n_tokens']} tok, {r_base['time_s']}s)")

        # 2. MTP
        r_mtp = bench_mtp(model, tokenizer, prompt, args.max_tokens, args.model)
        if "error" not in r_mtp:
            su = r_mtp["tok_s"] / r_base["tok_s"] if r_base["tok_s"] > 0 else 0
            print(f"  MTP:      {r_mtp['tok_s']} tok/s ({su:.2f}x, accept={r_mtp['acceptance']:.1%}, "
                  f"mtp={r_mtp['pct_mtp']}% verify={r_mtp['pct_verify']}%)")
        else:
            print(f"  MTP:      SKIP ({r_mtp['error']})")

        # 3. Cross-model
        r_cross = bench_cross_model_spec(model, draft_model, tokenizer, prompt,
                                         args.max_tokens, K=args.K)
        su_cross = r_cross["tok_s"] / r_base["tok_s"] if r_base["tok_s"] > 0 else 0
        print(f"  Cross K{args.K}:  {r_cross['tok_s']} tok/s ({su_cross:.2f}x, "
              f"accept={r_cross['acceptance']:.1%}, {r_cross['steps']} steps)")

        all_results[label] = {
            "baseline": r_base,
            "mtp": r_mtp,
            f"cross_K{args.K}": r_cross,
        }

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    base_tps = []
    mtp_tps = []
    cross_tps = []
    mtp_ar = []
    cross_ar = []

    for label, results in all_results.items():
        base_tps.append(results["baseline"]["tok_s"])
        if "error" not in results["mtp"]:
            mtp_tps.append(results["mtp"]["tok_s"])
            mtp_ar.append(results["mtp"]["acceptance"])
        cross_key = f"cross_K{args.K}"
        cross_tps.append(results[cross_key]["tok_s"])
        cross_ar.append(results[cross_key]["acceptance"])

    avg_base = sum(base_tps) / len(base_tps) if base_tps else 0
    avg_mtp = sum(mtp_tps) / len(mtp_tps) if mtp_tps else 0
    avg_cross = sum(cross_tps) / len(cross_tps) if cross_tps else 0
    avg_mtp_ar = sum(mtp_ar) / len(mtp_ar) if mtp_ar else 0
    avg_cross_ar = sum(cross_ar) / len(cross_ar) if cross_ar else 0

    print(f"{'Method':<20} {'tok/s':>8} {'Speedup':>8} {'Accept':>8}")
    print("-" * 48)
    print(f"{'Baseline':<20} {avg_base:>8.1f} {'1.00x':>8} {'N/A':>8}")
    if mtp_tps:
        print(f"{'MTP (same-model)':<20} {avg_mtp:>8.1f} {avg_mtp/avg_base:>7.2f}x {avg_mtp_ar:>7.1%}")
    print(f"{f'Cross K{args.K}':<20} {avg_cross:>8.1f} {avg_cross/avg_base:>7.2f}x {avg_cross_ar:>7.1%}")

    # Save
    out_data = {
        "model": args.model,
        "draft": args.draft,
        "max_tokens": args.max_tokens,
        "K": args.K,
        "summary": {
            "baseline_avg_tps": round(avg_base, 1),
            "mtp_avg_tps": round(avg_mtp, 1) if mtp_tps else None,
            "mtp_avg_acceptance": round(avg_mtp_ar, 3) if mtp_ar else None,
            "cross_avg_tps": round(avg_cross, 1),
            "cross_avg_acceptance": round(avg_cross_ar, 3),
        },
        "by_prompt": all_results,
    }
    out_path = ROOT / "bench" / "mtp_horizontal_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
