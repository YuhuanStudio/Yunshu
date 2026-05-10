"""Benchmark Speculative Decoding — draft/target with real models.

Supports same-model validation and cross-model spec decode.
Validated: 2B→4B spec decode produces identical output to baseline.

Usage:
    # Same model validation (expect ~100% acceptance)
    PYTHONPATH=python .venv/bin/python3 scripts/bench_spec_decode.py

    # Cross-model spec decode
    PYTHONPATH=python .venv/bin/python3 scripts/bench_spec_decode.py \\
        --draft Qwen3.5-2B-MLX-bf16 --target Qwen3.5-4B-MLX-bf16

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


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding Benchmark")
    parser.add_argument("--draft", default=None, help="Draft model (smaller)")
    parser.add_argument("--target", default="Qwen3.5-4B-MLX-bf16", help="Target model")
    parser.add_argument("--model", default=None, help="Same model for both draft+target (validation)")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--draft-length", type=int, default=4)
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    # If --model is given, use same model for both
    if args.model:
        args.draft = args.model
        args.target = args.model
    elif not args.draft:
        args.draft = args.target  # same model by default

    from mlx_lm.utils import load_model, load_tokenizer
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.models.cache import make_prompt_cache

    # Load target
    target_path = ROOT / "models" / args.target
    if not target_path.exists():
        target_path = Path(args.target)
    print(f"Loading target: {args.target}...")
    target_model, _ = load_model(target_path)
    tokenizer = load_tokenizer(target_path)
    print("Target loaded.")

    # Load draft (if different from target or explicitly requested)
    draft_model = None
    if not args.baseline_only:
        draft_path = ROOT / "models" / args.draft
        if not draft_path.exists():
            draft_path = Path(args.draft)
        if args.draft != args.target or True:
            print(f"Loading draft: {args.draft}...")
            draft_model, _ = load_model(draft_path)
            draft_tok = load_tokenizer(draft_path)
            print("Draft loaded.")

            # Verify tokenizer compatibility
            prompt_test = "Hello"
            if draft_tok.encode(prompt_test) != tokenizer.encode(prompt_test):
                print("WARNING: Tokenizers are different! Results may be incorrect.")

    prompts = [
        "The capital of France is",
        "In machine learning, gradient descent works by",
        "The key difference between TCP and UDP is that",
    ][:args.num_prompts]

    K = args.draft_length
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

    # ── Speculative ──
    print(f"\n=== Speculative ({args.draft}→{args.target}, K={K}) ===")
    spec = []

    for i, prompt in enumerate(prompts):
        prompt_ids = tokenizer.encode(prompt)
        prompt_t = mx.array(prompt_ids).reshape(1, -1)

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

        t0 = time.perf_counter()

        while len(generated) < args.max_tokens:
            # Draft K tokens
            last = generated[-1]
            draft_tokens = []
            d_logits = _call(draft_model, last, draft_cache)
            draft_tokens.append(_greedy(d_logits))
            for _ in range(K - 1):
                d_logits = _call(draft_model, draft_tokens[-1], draft_cache)
                draft_tokens.append(_greedy(d_logits))
            total_draft += K

            # Target verify
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

            total_accepted += accepted
            steps += 1

            if accepted < K:
                all_ids = prompt_ids + generated
                draft_cache = make_prompt_cache(draft_model)
                draft_model(mx.array(all_ids).reshape(1, -1), cache=draft_cache)

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
    print(f"Baseline:    {ab:.1f} tok/s ({args.target})")
    print(f"Speculative: {asp:.1f} tok/s ({args.draft}→{args.target}, {su:.2f}x)")
    print(f"Acceptance:  {aa:.1%}")

    print("\n=== Output ===")
    all_match = True
    for i in range(len(prompts)):
        b, s = baseline[i]["text"], spec[i]["text"]
        match = b == s
        if not match:
            all_match = False
        print(f"  [{i+1}] {'MATCH' if match else 'DIFF'}")
        if not match:
            print(f"    base: {b[:80]}")
            print(f"    spec: {s[:80]}")
    if all_match:
        print("\nAll outputs match baseline — spec decode produces identical results.")

    out = {
        "draft": args.draft, "target": args.target, "K": K, "max_tokens": args.max_tokens,
        "baseline_avg": round(ab, 1), "spec_avg": round(asp, 1),
        "speedup": round(su, 2), "acceptance": round(aa, 3),
        "output_match": all_match,
        "baseline": baseline, "speculative": spec,
    }
    p = ROOT / "bench" / "spec_decode_results.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {p}")


if __name__ == "__main__":
    main()
