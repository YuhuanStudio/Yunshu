"""Live Gemma-4 dual-load acceptance test (target + assistant drafter).

Measures the spec-decode acceptance rate: how often the drafter's argmax
matches the target's argmax, given the target's real hidden states and KV
cache (KV-sharing). Loads the 14.9GB target + 159MB drafter (~15GB total —
single dual-load, OOM-safe on M3 Max 36GB).

Method (single full-sequence forward, RoPE offset 0, causal):
  1. Greedy-generate a self-consistent continuation from each prompt (the
     realistic spec-decode scenario: the drafter proposes the target's own
     next tokens, not arbitrary text).
  2. Re-run the target over the full sequence -> hidden states (1,L,2560) +
     filled KV cache (24 caches, one per non-shared layer 0..23).
  3. target_next[t] = argmax(lm_head(hidden[t])).
  4. Feed the drafter: inputs_embeds = target_embed(ids[t]) (backbone-dim,
     unshifted), hidden_states = target hidden[t], per-layer KV shared from the
     target's last non-shared layer of matching type (sliding -> L22, full ->
     L23).
  5. draft_next[t] = argmax(draft_lm_head(draft_hidden[t])).
  6. acceptance = mean(draft_next[t] == target_next[t]) over the generated
     region (where the sequence is greedy-self-consistent).

Run: PYTHONPATH=python uv run python scripts/validate_gemma4_dual_load.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model, load_tokenizer

from yunshu_engine.gemma4_assistant import load_assistant_drafter

TARGET = Path("/Volumes/P5Plus/models/gemma-4-e4b-it-bf16")
DRAFTER = Path("/Volumes/P5Plus/models/gemma-4-E4B-it-assistant-bf16")

# Target non-shared layer of each type that the drafter shares KV from
# (last non-shared layer per type; verified from config: sliding=22, full=23).
TARGET_SLIDING_KV_LAYER = 22
TARGET_FULL_KV_LAYER = 23


def _kv_from_cache(cache, layer_idx, seq_len):
    c = cache[layer_idx]
    k, v = c.state  # RotatingKVCache.state -> (keys, values)
    # Trim to the valid prefix (seq_len <= sliding window so layout is linear).
    return k[:, :, :seq_len, :], v[:, :, :seq_len, :]


def main() -> int:
    if not TARGET.exists() or not DRAFTER.exists():
        print("SKIP: model drive not mounted")
        return 0

    t0 = time.time()
    ret = load_model(TARGET, strict=False)
    target = ret[0] if isinstance(ret, tuple) else ret
    tok = load_tokenizer(TARGET)
    predictor, embedder, cfg = load_assistant_drafter(DRAFTER)
    print(f"loaded target + drafter in {time.time() - t0:.1f}s")

    text_model = target.language_model.model  # Gemma4TextModel
    embed = text_model.embed_tokens
    embed_scale = text_model.embed_scale
    draft_lm = predictor.embed_tokens.weight  # (vocab, 256), tied lm_head

    prompts = [
        "Explain photosynthesis in simple terms.",
        "Write a short story about a robot learning to paint.",
        "What are the main causes of climate change?",
    ]
    n_gen = 48

    total_hits = 0
    total_positions = 0
    for prompt in prompts:
        # 1. Greedy-generate a self-consistent continuation.
        ids = tok.encode(prompt)
        cache = make_prompt_cache(target)
        h = text_model(mx.array(ids)[None], cache=cache)
        for _ in range(n_gen):
            nt = int(mx.argmax(embed.as_linear(h[:, -1:, :])[0, -1]))
            ids.append(nt)
            h = text_model(mx.array([[nt]]), cache=cache)

        # 2. Re-run the full sequence to get clean hidden + KV.
        full = mx.array(ids)[None]
        seq_len = full.shape[1]
        gen_start = seq_len - n_gen
        cache = make_prompt_cache(target)
        hidden = text_model(full, cache=cache)
        mx.eval(hidden)
        target_next = mx.argmax(embed.as_linear(hidden)[0], axis=-1)
        mx.eval(target_next)

        # 3. Drafter forward (inputs_embeds unshifted + target hidden + KV-share).
        inputs_embeds = (embed(full) * embed_scale)[0]
        ksl, vsl = _kv_from_cache(cache, TARGET_SLIDING_KV_LAYER, seq_len)
        kfu, vfu = _kv_from_cache(cache, TARGET_FULL_KV_LAYER, seq_len)
        kv_per_layer = [(ksl, vsl), (ksl, vsl), (ksl, vsl), (kfu, vfu)]
        draft_hidden, _ = predictor(inputs_embeds, hidden[0], kv_per_layer)
        draft_next = mx.argmax(draft_hidden @ draft_lm.T, axis=-1)
        mx.eval(draft_next)

        # 4. Acceptance over the greedy-consistent generated region.
        tn = target_next[gen_start:-1]
        dn = draft_next[gen_start:-1]
        hits = int(mx.sum(tn == dn).item())
        n = tn.shape[0]
        total_hits += hits
        total_positions += n
        print(f"  L={seq_len} (gen {n_gen}): acceptance {hits}/{n} = {hits / n:.1%}")

    rate = total_hits / total_positions if total_positions else 0.0
    print(f"\nOVERALL acceptance: {total_hits}/{total_positions} = {rate:.1%}")
    print(f"peak mem GB: {mx.get_peak_memory() / 1e9:.1f}")
    print("\nPASS — dual-load runs end-to-end on real weights." if rate > 0
          else "\nWARN — zero acceptance; check wiring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
