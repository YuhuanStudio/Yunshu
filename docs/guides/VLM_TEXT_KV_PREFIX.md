# VLM text-path 4-tier KV prefix cache

## Status: DEFAULT ON — `YUNSHU_VLM_KV_PREFIX=0` to disable

`VLMEngine` wires a 4-tier `KVPrefixCache` (HOT full-precision / WARM 4-bit-in-RAM
/ SSD int8-on-disk) into the text path — both `_generate_vlm_text` (non-streaming)
and `_stream_vlm_text` (streaming), under the single `YUNSHU_VLM_KV_PREFIX` flag.
It closes the gap that VLM models had **no** cross-request KV prefix reuse on the
text path (only per-image KV-state reuse + the token-id/template cache).

Verified **byte-lossless** on mRoPE full-attention VLMs and **auto-bypassed** for
backbones that can't reuse losslessly. This mirrors the LLM fast path's 4-tier KV
hierarchy, now applied to multimodal models' text path.

## What makes reuse lossless (the three hard-won fixes)

KV prefix reuse on this stack is lossless only when the cache is resumable AND
positions are handled correctly. nailed three independent issues:

1. **Store at the prompt boundary.** `add()` runs right after prefill (before the
   decode loop), capturing clean prompt-only KV. A post-generation `add()` would
   store a cache polluted by generated tokens — irrecoverable for sliding-window
   caches.
2. **Full-match re-prefill, not single-token refeed.** On an exact match the
   cache holds every prompt token; re-prefill the last block (≤128 tokens) as a
   multi-token call. A 1-token refeed computes attention in a different numerical
   context and flips the greedy argmax.
3. **mRoPE native rope-state priming (the unlock).** mRoPE backbones
   (GLM-OCR, Qwen-VL, Qwen3-Omni) track position OUTSIDE the KV cache and
   `clear_rope_state()` resets it to None each request — so on a reused prefix the
   model recomputes the suffix's positions from 0, producing wrong-context output
   (e.g. answering about the wrong entity). FIX (`_prime_mrope_reuse_state`): set
   the model's native `_rope_deltas = 0` (the text-only delta) before reuse, so
   the model's OWN position logic computes cache-offset-based positions for the
   suffix + every decode step. This is model-native, so it works for BOTH simple
   mRoPE (GLM-OCR) AND **interleaved mRoPE (Qwen3-Omni)** — supplying our own
   `position_ids` only worked for the simple variant. No-op on a cold full prefill
   (offset 0 → the model's get_rope_index overwrites it).

## Two-gate safety (`_text_prefix_reuse_safe`)

Reuse is enabled only when BOTH pass:

1. **Capability gate** (`model_backend.py`, shared classifier): the cache must be
   resumable. Bypassed for:
   - **Sliding-window** (gemma `RotatingKVCache`, max_size=512) — a >window prompt
     rotates the circular buffer and loses linear history. Detect the layer TYPE,
     NOT `can_trim_prompt_cache` (returns True for an empty rotating probe).
   - **Hybrid recurrent** (Qwen3.5/3.6 GatedDeltaNet ArraysCache) — not sliceable.
   `requires_explicit_positions` = mRoPE; mRoPE does NOT disqualify, it signals the
   caller must supply `position_ids` (the text path does).

2. **Empirical probe** (`_probe_text_reuse_lossless`, run ONCE at load on the
   executor thread, memoized). Runs the ACTUAL reuse path (boundary snapshot +
   suffix prefill + native rope-state priming) vs a full prefill on a ~300-token
   sequence and requires the **greedy 12-token output sequences to match** (the
   true definition of lossless for greedy serving — a single-logit `<1e-2` gate
   was too strict for 30B-MoE routing noise that never flips the greedy token).
   This is the robust guard: mRoPE is not monolithic, and config detection (which
   even MISSES Qwen3-Omni's rope nested under `thinker_config.text_config`) can't
   classify safety — so we VERIFY empirically.
   - GLM-OCR (simple mRoPE) AND Qwen3-Omni-30B (interleaved mRoPE): probe PASSES
     with native rope-state priming → **lossless reuse**.

## Verification matrix (greedy, temp=0, M3 Max, max_tokens=150)

| model | cache | mRoPE | gate that fires | result |
|-------|-------|-------|-----------------|--------|
| GLM-OCR (VLM) | KVCache | simple | both pass | **byte-lossless** — HOT 12.6× (full + partial + streaming) |
| Qwen3-Omni-30B (VLM) | KVCache | interleaved | both pass (rope-state primed) | **byte-lossless** — HOT 13.1× |
| gemma-4 (VLM) | RotatingKVCache (sw=512) | no | capability bypass | bypassed (output == cache-off) |
| Qwen3.5/3.6-VL | KVCache + ArraysCache | — | capability bypass (hybrid) | bypassed |
| Qwen2.5-3B (LLM) | KVCache | no | n/a | lossless via separate BatchedEngine path |

`scripts/verify_vlm_text_kv_prefix.py` asserts both contracts (lossless reuse on
reuse-safe models; output unchanged on bypassed models). The reuse mechanism is
also proven bit-identical in engine-free logit tests (0.0 Δ).

## Shared capability layer (Part 2/3)

`python/yunshu_engine/model_backend.py` (unit-tested in
`tests/unit/test_model_backend.py`):
- `classify_cache(layers) -> CacheClass(has_sliding_window, is_hybrid, .resumable)`
- `derive_capabilities(kind, layers, is_mrope) -> BackendCapabilities`
  (`.supports_kv_prefix_reuse`, `.requires_explicit_positions`, `.bypass_reason()`)
- `ModelBackend` Protocol. Both `VLMEngine` and `BatchedEngine` expose
  `backend_capabilities()` and use the SAME classifier — one source of truth for
  the reuse gate. Decode drivers stay separate (vision caches / mRoPE / mlx_vlm
  wrapper are backbone-specific); the engines are not merged.

## Follow-ups

1. **Sliding-window reuse (gemma) — DEFERRED with rationale.** Only safe for
   prompt ≤ window (512); but caching pays off for LONG shared prefixes (RAG /
   system prompts) which exceed the window and can't be losslessly resumed from a
   rotated buffer. So the win is confined to short prompts where caching is itself
   low-value (decode dominates). Enabling it would also need per-request length
   guards (the load-time probe is length-independent). Not worth the complexity →
   gemma stays bypassed.
2. **Hybrid VLM (Qwen3.5/3.6-VL) boundary-snapshot reuse — IMPLEMENTED + probe-gated
.** Ported the LLM fast path's `_capture_hybrid_prefix` to VLMEngine
   (`_capture_vlm_hybrid_prefix` + no_trim boundary snapshots + `_probe_hybrid_reuse`,
   `YUNSHU_VLM_HYBRID_PREFIX`). BUT the greedy probe finds it is NOT lossless on the
   only available hybrid VLM (Qwen3.5-2B via VLMEngine / mlx_vlm.qwen3_5): mlx_vlm's
   GatedDeltaNet prefill is **not split/chunk-invariant** (same class as the ~0.16
   logit non-invariance found on GLM-OCR), so the recurrent state at a boundary ≠
   continuous prefill → the probe diverges at token 1 → it safely BYPASSES (verified:
   reuse output == cache-off, no corruption). The LLM path's hybrid reuse works
   because mlx-lm's GatedDeltaNet IS split-invariant. The VLM code is in place and
   will engage automatically if a chunk-invariant hybrid VLM (or fixed mlx_vlm
   GatedDeltaNet) appears — the probe is the gate. Text Qwen3.5/3.6 already get
   lossless hybrid reuse via BatchedEngine (matrix F-HOT 6.2×).
3. Validate on more mRoPE VLMs (Qwen2.5-VL / Qwen3-VL) when available — same
   native-rope-state mechanism, expected lossless.

**SSD net-negative guard:** `YUNSHU_SSD_RESTORE_MIN_TOKENS` (default 0)
skips the disk restore when the candidate prefix is below N tokens — fast-prefill
models (e.g. GLM-OCR) where reading from SSD is slower than re-prefilling can set
this to e.g. 512. Default 0 preserves prior behaviour.
