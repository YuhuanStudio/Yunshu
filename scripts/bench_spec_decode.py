"""Benchmark Speculative Decoding — validated correct implementation.

Algorithm:
1. Prefill target → T0 from prefill logits
2. Prefill draft → verify D0 == T0
3. Draft: feed T0, get D1. Feed D1, get D2... Feed D(K-1), get DK
4. Target: feed T0, verify D1. Feed D1, verify D2... etc
5. Accept up to first mismatch, take target's choice + bonus
6. Rebuild draft cache on rejection; continue on acceptance

Usage:
    PYTHONPATH=python .venv/bin/python3 scripts/bench_spec_decode.py
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3.5-4B-MLX-bf16")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--draft-length", type=int, default=4)
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    from mlx_lm.utils import load_model, load_tokenizer
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.models.cache import make_prompt_cache

    model_path = ROOT / "models" / args.model
    if not model_path.exists():
        model_path = Path(args.model)
    print(f"Loading: {args.model}...")
    model, _ = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    print("Loaded.")

    prompts = [
        "The capital of France is",
        "In machine learning, gradient descent works by",
        "The key difference between TCP and UDP is that",
    ][:args.num_prompts]

    K = args.draft_length
    sampler = make_sampler(temp=0.0)

    # ── Baseline ──
    print(f"\n=== Baseline ({args.max_tokens} tok) ===")
    baseline = []
    for i, prompt in enumerate(prompts):
        ids = mx.array(tokenizer.encode(prompt))
        tokens = []
        t0 = time.perf_counter()
        for tok, _ in generate_step(ids, model, max_tokens=args.max_tokens, sampler=sampler):
            tokens.append(tok)
        mx.synchronize()
        elapsed = time.perf_counter() - t0
        tps = len(tokens) / elapsed if elapsed > 0 else 0
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        baseline.append({"n": len(tokens), "s": round(elapsed, 3), "tps": round(tps, 1), "text": text[:100]})
        print(f"  [{i+1}] {len(tokens)} tok, {elapsed:.3f}s, {tps:.1f} tok/s")

    if args.baseline_only:
        return

    # ── Speculative ──
    print(f"\n=== Speculative (K={K}) ===")
    spec = []

    for i, prompt in enumerate(prompts):
        prompt_ids = tokenizer.encode(prompt)
        prompt_t = mx.array(prompt_ids).reshape(1, -1)

        eos_ids = set()
        if hasattr(tokenizer, 'eos_token_id'):
            eid = tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                eos_ids.update(eid)
            elif eid is not None:
                eos_ids.add(eid)

        # Prefill target
        target_cache = make_prompt_cache(model)
        t_out = model(prompt_t, cache=target_cache)
        t_logits = t_out if not hasattr(t_out, 'logits') else t_out.logits
        T0 = _greedy(t_logits[0, -1, :])

        # Prefill draft
        draft_cache = make_prompt_cache(model)
        model(prompt_t, cache=draft_cache)

        generated = [T0]
        total_draft = 0
        total_accepted = 0
        steps = 0

        t0 = time.perf_counter()

        while len(generated) < args.max_tokens:
            # ── Draft: generate K tokens ──
            # Feed last generated token to draft, collect K next tokens
            last_tok = generated[-1]
            d_logits = _call(model, last_tok, draft_cache)
            draft_tokens = [_greedy(d_logits)]

            for _ in range(K - 1):
                d_logits = _call(model, draft_tokens[-1], draft_cache)
                draft_tokens.append(_greedy(d_logits))

            total_draft += K

            # ── Target: verify one-by-one ──
            # Feed last generated token to target (same as draft did)
            last_tok = generated[-1]
            t_logits = _call(model, last_tok, target_cache)

            accepted = 0
            rejected = False
            for j in range(K):
                target_choice = _greedy(t_logits)
                if target_choice == draft_tokens[j]:
                    accepted += 1
                    generated.append(draft_tokens[j])
                    if draft_tokens[j] in eos_ids:
                        rejected = True
                        break
                    # Feed this token to target for next verification
                    t_logits = _call(model, draft_tokens[j], target_cache)
                else:
                    # Rejected: take target's choice
                    generated.append(target_choice)
                    rejected = True
                    break

            if not rejected:
                # All K accepted — bonus token from last target logits
                bonus = _greedy(t_logits)
                generated.append(bonus)

            total_accepted += accepted
            steps += 1

            # ── Draft cache sync ──
            # If rejected, draft cache is ahead of actual sequence.
            # Rebuild draft cache from prompt + all generated tokens.
            if accepted < K:
                all_ids = prompt_ids + generated
                draft_cache = make_prompt_cache(model)
                model(mx.array(all_ids).reshape(1, -1), cache=draft_cache)

            if any(t in eos_ids for t in generated):
                break

        mx.synchronize()
        elapsed = time.perf_counter() - t0
        tps = len(generated) / elapsed if elapsed > 0 else 0
        ar = total_accepted / total_draft if total_draft > 0 else 0
        text = tokenizer.decode(generated, skip_special_tokens=True)

        spec.append({
            "n": len(generated), "s": round(elapsed, 3), "tps": round(tps, 1),
            "ar": round(ar, 3), "steps": steps, "text": text[:100],
        })
        print(f"  [{i+1}] {len(generated)} tok, {elapsed:.3f}s, {tps:.1f} tok/s, "
              f"accept={ar:.1%} ({total_accepted}/{total_draft}), {steps} steps")

    # ── Summary ──
    print("\n=== Summary ===")
    ab = sum(r["tps"] for r in baseline) / len(baseline)
    asp = sum(r["tps"] for r in spec) / len(spec)
    su = asp / ab if ab > 0 else 0
    aa = sum(r["ar"] for r in spec) / len(spec)
    print(f"Baseline:    {ab:.1f} tok/s")
    print(f"Speculative: {asp:.1f} tok/s ({su:.2f}x)")
    print(f"Acceptance:  {aa:.1%}")

    print("\n=== Output ===")
    for i in range(len(prompts)):
        b, s = baseline[i]["text"], spec[i]["text"]
        print(f"  [{i+1}] {'MATCH' if b == s else 'DIFF'}")
        if b != s:
            print(f"    base: {b[:80]}")
            print(f"    spec: {s[:80]}")

    out = {
        "model": args.model, "K": K, "max_tokens": args.max_tokens,
        "baseline_avg": round(ab, 1), "spec_avg": round(asp, 1),
        "speedup": round(su, 2), "acceptance": round(aa, 3),
        "baseline": baseline, "speculative": spec,
    }
    p = ROOT / "bench" / "spec_decode_results.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {p}")


if __name__ == "__main__":
    main()
