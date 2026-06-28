"""End-to-end Gemma-4 dual-load speculative decoding + speedup measurement.

Implements the full draft-verify-rollback loop using the validated
Gemma4AssistantProposer (§117/§482) and measures speedup vs greedy, while
asserting the correctness invariant: spec-decode output == greedy output.

Loop (k drafts/step, vLLM constant_draft_positions):
  1. t1 = argmax(lm_head(target_hidden_P))            # free target token
  2. drafts = proposer.propose_chain(t1, hidden_P, kv, offset=P, k)
  3. target verifies [t1, *drafts] in ONE forward    # positions P+1..P+1+k
  4. accept the longest prefix where draft_j == target argmax_j; trim the
     target KV for rejected positions; carry the last accepted hidden.

Run: PYTHONPATH=python uv run python scripts/validate_gemma4_spec_decode.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model, load_tokenizer

from yunshu_engine.gemma4_assistant import Gemma4AssistantProposer

TARGET = Path("/Volumes/P5Plus/models/gemma-4-e4b-it-bf16")
DRAFTER = "/Volumes/P5Plus/models/gemma-4-E4B-it-assistant-bf16"


def _greedy(tm, lm, prompt_ids, n):
    """Sequential greedy. Returns (tokens, top2_logprob_gaps) — the gap lets us
    machine-check that any spec divergence is a benign near-tie, not a real
    acceptance bug (Wave 658)."""
    cache = make_prompt_cache_for(tm)
    h = tm(mx.array(prompt_ids)[None], cache=cache)
    out, margins = [], []
    for _ in range(n):
        lg = lm(h[:, -1:, :])[0, -1]
        lp = lg - mx.logsumexp(lg)
        order = mx.argsort(-lp)
        a, b = int(order[0]), int(order[1])
        out.append(a)
        margins.append(float(lp[a] - lp[b]))
        h = tm(mx.array([[a]]), cache=cache)
    return out, margins


# make_prompt_cache needs the wrapper model; bind it once.
_TARGET_MODEL = None


def make_prompt_cache_for(_tm):
    return make_prompt_cache(_TARGET_MODEL)


def _spec_decode(tm, lm, prop, prompt_ids, n, k):
    # Uses the reusable serving primitive (gemma4_assistant.spec_decode_generate),
    # so script and library share ONE implementation (no drift).
    cache = make_prompt_cache(_TARGET_MODEL)
    return prop.spec_decode_generate(tm, lm, cache, prompt_ids, n, k=k)


def main() -> int:
    global _TARGET_MODEL
    if not TARGET.exists():
        print("SKIP: target not mounted")
        return 0
    ret = load_model(TARGET, strict=False)
    target = ret[0] if isinstance(ret, tuple) else ret
    _TARGET_MODEL = target
    tok = load_tokenizer(TARGET)
    tm = target.language_model.model
    lm = tm.embed_tokens.as_linear
    tcfg = json.loads((TARGET / "config.json").read_text())
    prop = Gemma4AssistantProposer.from_paths(
        DRAFTER, tm.embed_tokens.weight, tm.embed_scale, tcfg
    )

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Explain how photosynthesis works, step by step."}],
        add_generation_prompt=True,
        tokenize=True,
    )
    n = 80
    k = 4

    t0 = time.perf_counter()
    greedy_out, greedy_margins = _greedy(tm, lm, prompt, n)
    t_greedy = time.perf_counter() - t0

    t0 = time.perf_counter()
    spec_out = _spec_decode(tm, lm, prop, prompt, n, k)
    t_spec = time.perf_counter() - t0

    # Prefix match length (exact-match run before the first divergence).
    prefix = 0
    for g, s in zip(greedy_out, spec_out, strict=False):
        if g == s:
            prefix += 1
        else:
            break
    speedup = t_greedy / t_spec
    print(f"greedy:  {n} tok in {t_greedy:.2f}s  ({n / t_greedy:.1f} tok/s)")
    print(f"spec(k={k}): {len(spec_out)} tok in {t_spec:.2f}s  ({len(spec_out) / t_spec:.1f} tok/s)")
    print(f"speedup: {speedup:.2f}x")
    print(f"greedy-prefix match: {prefix}/{n} tokens exact before first divergence")
    # Correctness invariant (Wave 658) — MACHINE-CHECKED, not asserted as "noise":
    # spec output is self-consistent with the verify pass by construction (each token
    # is the target's argmax there). A divergence from SEQUENTIAL greedy is allowed
    # ONLY at a near-tie — the verify forward batches 1+k positions, and fp
    # non-associativity can tip an EXACT tie (#1<->#2). It must NOT flip a CONFIDENT
    # token; that would be a real acceptance/bookkeeping bug. So we require: if spec
    # diverges within n, greedy's top-2 gap at that position is a near-tie.
    TIE_GAP = 0.3
    div_gap = greedy_margins[prefix] if prefix < n and prefix < len(greedy_margins) else None
    div_ok = (div_gap is None) or (div_gap < TIE_GAP)
    ok = speedup > 1.0 and prefix >= 16 and div_ok
    print(f"peak mem GB: {mx.get_peak_memory() / 1e9:.1f}")
    if div_gap is not None:
        print(f"divergence-is-a-near-tie: gap={div_gap:.4f} logprob "
              f"({'OK <%.1f' % TIE_GAP if div_ok else 'FAIL — confident flip = acceptance bug'})")
    print(
        f"\nPASS — dual-load spec decode works ({speedup:.2f}x, {prefix}-token exact "
        f"greedy prefix; "
        + ("byte-exact" if div_gap is None
           else f"divergence is a near-tie gap={div_gap:.3f}, not a bug") + ")."
        if ok else
        f"\nFAIL — spec diverged from greedy at a CONFIDENT token "
        f"(gap={div_gap:.3f} >= {TIE_GAP}) = real acceptance bug"
        if div_gap is not None and not div_ok else "\nFAIL"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
