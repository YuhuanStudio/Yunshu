"""Pure MTP benchmark — always-advance with snapshot/restore for hybrid models.

For Qwen3.5 (hybrid SSM + attention), cache snapshot/restore is needed on
reject because SSM state is irreversible after seeing the draft token.

Per-cycle flow:
  1. Snapshot cache state
  2. MTP forward → draft token D (mtp_cache=None, no persistent MTP cache)
  3. Verify: backbone forward [P, D] (2 tokens, cache advances 2)
  4. Accept (pos0 argmax == D): emit D + bonus(B), B = next primary
  5. Reject: restore cache snapshot, refeed P only → correction C = next primary

Accept cost: MTP + 2-token verify ≈ 1.15× backbone
Reject cost: MTP + 2-token verify + restore + 1-token refeed ≈ 2.15× backbone
Speedup = (1 + p) / (1.15 + 0.15×(1-p)) where p = acceptance rate

Usage:
    .venv/bin/python3 scripts/bench_mtp.py --model Qwen3.5-4B-MLX-bf16
    .venv/bin/python3 scripts/bench_mtp.py --model Qwen3.5-9B-MLX-4bit --max-tokens 128
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


def run_mtp(model, tokenizer, prompt, max_tokens):
    getattr(model, "language_model", model)
    ids = mx.array(tokenizer.encode(prompt)).reshape(1, -1)
    eos_ids = _get_eos_ids(tokenizer)

    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)

    out, hidden = model(ids, cache=cache, return_hidden=True)
    mx.synchronize()
    first = int(mx.argmax(out[0, -1, :]).item())

    primary = first
    primary_h = hidden[:, -1:, :]
    generated = [first]
    accepts = 0
    rejects = 0

    t0 = time.perf_counter()
    while len(generated) < max_tokens:
        s = _snap(cache)

        mtp_out = model.mtp_forward(primary_h, mx.array([[primary]]), None)
        draft = int(mx.argmax(mtp_out[0, -1, :]).item())

        verify_out, verify_h = model(
            mx.array([[primary, draft]]), cache=cache, return_hidden=True,
        )
        mx.synchronize()
        v0 = int(mx.argmax(verify_out[0, 0, :]).item())
        v1 = int(mx.argmax(verify_out[0, 1, :]).item())

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
            correction = int(mx.argmax(out2[0, -1, :]).item())
            mx.synchronize()
            generated.append(correction)
            if correction in eos_ids or len(generated) >= max_tokens:
                break
            primary = correction
            primary_h = hid2[:, -1:, :]

    elapsed = time.perf_counter() - t0
    cycles = accepts + rejects
    return {
        "n": len(generated),
        "total_s": round(elapsed, 3),
        "tps": round(len(generated) / elapsed, 1),
        "acceptance": round(accepts / cycles, 3) if cycles > 0 else 0,
        "cycles": cycles,
        "accepts": accepts,
        "rejects": rejects,
    }


def main():
    parser = argparse.ArgumentParser(description="Pure MTP Benchmark")
    parser.add_argument("--model", default="Qwen3.5-4B-MLX-bf16")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=3)
    args = parser.parse_args()

    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
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

    sampler = make_sampler(temp=0.0)

    # Baseline
    print(f"=== Baseline ({args.model}, {args.max_tokens} tok) ===")
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
        baseline.append({"n": len(tokens), "tps": round(tps, 1)})
        print(f"  [{i+1}] {len(tokens)} tok, {tps:.1f} tok/s")

    ab = sum(r["tps"] for r in baseline) / len(baseline)

    # MTP
    print(f"\n=== MTP always-advance ({args.model}, {args.max_tokens} tok) ===")
    mtp_results = []
    for i, prompt in enumerate(prompts):
        result = run_mtp(model, tokenizer, prompt, args.max_tokens)
        print(f"  [{i+1}] {result['n']} tok, {result['tps']:.1f} tok/s, "
              f"accept={result['acceptance']:.1%} ({result['accepts']}/{result['cycles']})")
        mtp_results.append(result)

    am = sum(r["tps"] for r in mtp_results) / len(mtp_results)
    ar = sum(r["acceptance"] for r in mtp_results) / len(mtp_results)
    su = am / ab if ab > 0 else 0

    print(f"\n{'='*50}")
    print(f"  Baseline: {ab:.1f} tok/s")
    print(f"  MTP:      {am:.1f} tok/s ({su:.2f}x, accept={ar:.1%})")

    out = {
        "model": args.model,
        "baseline_avg": round(ab, 1),
        "mtp": {"avg_tps": round(am, 1), "speedup": round(su, 2), "acceptance": round(ar, 3)},
    }
    p = ROOT / "bench" / "mtp_results.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {p}")


if __name__ == "__main__":
    main()
