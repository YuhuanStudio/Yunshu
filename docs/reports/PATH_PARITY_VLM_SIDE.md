# Path parity, VLM side (text-path fixes ported to mtp / dflash / batch / generic)

> Line numbers in this report refer to the tree when it was written (2026-10-26, before the later review-fix and
> adoption merges) and drift; search by symbol. The "Update after integration" section at the end is the final state.


The text fast path got a round of fixes (commits below). This records, for each one, what the
VLM serving paths did before, what was done, and what the unified path must keep. Paths follow
`python/yunshu_engine/path_matrix.json`: **mtp** (`mtp_lane.py`), **dflash** (`dflash_*.py`),
**batch** (stock shared batch on `vlm_batch_runner.py`), **generic** (other mlx-vlm families on
the same runner: Gemma, Qwen4, Omni thinker). `driver` was deleted in 03b03e7 and was skipped.
batch and generic share one runner and one set of processors, so they have one cell each only
where they differ.

Status words: `ok` = already equivalent, `fixed` = changed here, `gap` = left, with the reason.

| Fix | Text commit | Before (mtp / dflash / batch / generic) | Done (VLM file:line) | A unified path must keep |
|---|---|---|---|---|
| (a) dense penalty / bias / suppress / min_tokens | 4634f98 | Penalties are mlx-lm closures: a constant number of ops per token, no per-id loop, stateless, window = last 20 tokens (`vlm_batch_runner.py:304` `build_penalty_processors`). Not O(unique ids). Three real differences: (1) `TokenMaskProcessor._block` rebuilt `zeros + scatter` for suppress and EOS on every token (all four paths); (2) `logit_bias` and suppress/EOS ids outside `[0, vocab)` were not ignored: `-1` wrapped onto the LAST token; (3) mtp/dflash `apply_row_processors` turned the whole prompt context into a new `mx.array` every verify window. | fixed (1): masks built once per vocab width, `vlm_batch_runner.py:367` `_block`. fixed (2): `LogitBiasProcessor` (`vlm_batch_runner.py:271`) filters to in-range ids and runs the identical `at[].add`; `_block` filters too. fixed (3): `constrained_spec.py:334` `PENALTY_CONTEXT`, `apply_row_processors` slices the context tail (`:350`); `build_penalty_processors` now passes that constant as the context size so the two cannot drift. **gap:** the semantic difference below. | Processors must stay stateless-per-row or expose `pure_at` / `advance` so spec rows equal serial rows; `LogitBiasProcessor.context_free` and `spec_safe_processors` are what admit them to the spec lane. Out-of-range ids ignored. If the text path's dense vectors are adopted, give them a per-row rollback (speculative windows), otherwise spec on != off. |
| (b) top_logprobs without a full sort | f3f1d5c | mtp, dflash: `SpecRequest.reports` did `argsort` over `[rows, V]` per window. batch, generic: mlx-vlm's `BatchGenerator._step` does `argsort` over `[B, V]` every token inside upstream code (`mlx_vlm/generate/ar.py:1246`, `:2326`). | fixed for mtp + dflash: `text_logprobs.py` `top_logprobs_rows`, used at `constrained_spec.py` `SpecRequest.reports`. Tie order is the one the upstream sort produces (descending value, DESCENDING id), which is the opposite of the text path's ascending id; boundary ties, `-inf` tails and NaN rows redo the old sort. **gap:** batch and generic keep the upstream sort (upstream-owned code, see below). | One tie order for logprobs everywhere. Today the text path says ascending id and the VLM path descending id for equal values; the unified path must pick one and keep spec lanes equal to the serial path. |
| (c) per-element mx to python conversions | 9d9c7d8 | None found on the request path. The VLM path never used the inflight tracker; prompt ids are `.tolist()` once (`vlm_engine.py:3232`, `vlm_batch_runner.py:683`), later `int(t)` loops run over Python lists. The one mx-array case was (a)(3) above. | n/a, plus (a)(3). Related: `apc_manager.py:54` `_sequence_hash` (see (i)). | Convert prompt ids once; never iterate an `mx.array` per element. |
| (d) wired limit | 71c183e | The VLM path does not call `mx.set_wired_limit` at load. `idle_memory.py:178` holds the recommended wired limit during idle beats and restores it at the idle trim, which is a different mechanism. | Not changed on purpose: it changes memory behaviour and needs a Mac A/B (same-checkpoint, load time, resident set, swap). | If the unified path adopts the text load-time wired limit, do it once at load for both model kinds and reconcile it with the idle beat's save / restore of the previous limit (`_WIRED_BEFORE`), or the two will overwrite each other. |
| (e) automatic length-gated 8-bit KV not in token-keyed caches | 6cc09fb | The VLM path has no automatic length-gated quantisation. Its only lossy KV is the explicit `YUNSHU_KV_PRECISION=int8` (shared decode batch, Qwen3.5-family), and that precision is part of the APC namespace fingerprint (`vlm_engine.py:2053`), so a hit cannot mix precisions. | n/a, nothing to gate. | If an automatic gate is ever added to the VLM path, its KV must stay out of APC checkpoints, or the gate value must join the fingerprint. |
| (f) special-token neutralisation | 969b1e9 | Render paths in `vlm_engine.py` (`:3825`, `:4017`) were covered. Gaps: the MLX-MTP fallback renderer `mlxvlm_mtp.py` `_encode`, and every count / preview route, which rendered raw text so counts included real control tokens that generation never sees: `/tokenize`, `/apply-template`, the `max_tokens=0` probe in `chat.py`, Anthropic `count_tokens`, `responses._count_prompt_tokens`. | fixed: `mlxvlm_mtp.py:269`, `tokenize.py:108,319`, `chat.py:1952`, `anthropic.py:3946`, `responses.py:3970`. The Responses non-batched legacy-engine renders (`responses.py` ~2080, ~2600) were left alone: they serve the deprecated text `Engine`, not the VLM paths. | Every render, count and preview goes through `neutralize_special_text`; a count must equal the generated prompt length. Test: `tests/unit/test_special_token_count_parity.py`. |
| (g) llguidance `allowed_mask` fast path | 490898d | ok on all four. Runner `ConstraintProcessor._mask_row` tries `allowed_mask` first; the spec lanes build the bitmask straight from the matcher for llguidance (`constrained_spec.py` ~170) and use `allowed_mask` for the JSON-schema constraint. `RegexConstraint` and `ChoiceConstraint` have no `allowed_mask` and fall back to the id list on every path, text included. | none | Keep the mask-first order and the list fallback for constraints without one. |
| (h) llguidance prewarm | 0dd5bc2 | ok: `vlm_engine.py:639` calls `prewarm_for_engine` in model load, before `_finish_vlm_load`, so it covers all four. | none (verified) | Call prewarm in the shared load, not per model kind. |
| (i) block hashes byte-once | b9f5305 | The VLM prefix cache does not use `yunshu_kv.hash`: it hashes with `mlx_vlm.apc` (`hash((parent, tokens, extra))` per block, and `_sequence_hash` for whole-sequence keys). `_sequence_hash` built its int32 array from a per-token `int(t)` loop and our code added another `tuple(int(t) ...)` before it. | fixed: `apc_manager.py:54` local `_sequence_hash`, same digest as upstream (tested against it), one `np.asarray` instead of the loop; other input types fall back to upstream. The block-chain hash stays upstream's, because persisted SSD checkpoints are keyed by it. | One hash scheme per cache. The VLM APC and the text prefix cache hash differently on purpose (disk identity); a merge must not change either digest without a namespace bump. |
| (j) template / tokenize off the event loop | 157cb01 | Not moved off the event loop (157cb01 only moved the text path; not re-verified in code here). | Deliberately not moved (per the team's note: MLX-thread FIFO ordering). Recorded only. | If the unified path moves rendering to a worker, keep admission order (submit to the MLX thread in arrival order), not completion order of the renders. |

## Open differences (not fixed here)

- **Penalty semantics differ between the paths.** Text counts every generated token from the first (OpenAI semantics: frequency scales with count, presence is any occurrence). The VLM path uses mlx-lm's window of the last 20 tokens of prompt plus generation. A `presence_penalty` therefore penalises prompt tokens on the VLM path and forgets old generated tokens. Aligning them is a behaviour change and needs per-row state with rollback in the spec lanes (`ProcessorWindow.commit` / `pure_at`); it was not attempted as a "port". Decide which semantics the unified path has first.
- **batch and generic `top_logprobs`** still run the upstream full-vocab `argsort` for the whole batch each token whenever any row asks for them (`top_logprobs_k` is per batch, `GenerationBatch._step` upstream). Fixing it means replacing the inline block of an upstream method; `vlm_batch_runner.py` already wraps `GenerationBatch._step` (~line 2663), but only to add behaviour around it. Use `text_logprobs.top_logprobs_rows` if that wrapper is extended.
- `logit_bias` ids are now ignored when out of range on the VLM path (as on the text path), which is a deliberate behaviour change for requests that sent such ids.

## Tests added

- `tests/unit/test_vlm_token_mask_cache.py`: cached masks bit-equal to the old per-token rebuild (float32, bfloat16, min_tokens boundary, top-n-sigma); out-of-range ids ignored.
- `tests/unit/test_vlm_logit_bias_range.py`: out-of-range ids ignored (stock mlx-lm wraps `-1`, pinned); in-range logits bit-equal to stock mlx-lm; still spec-safe.
- `tests/unit/test_spec_window_processor_context.py`: window logits with the tail context equal full-history logits, float32 and bfloat16, contexts of 5 to 5000 tokens; unknown processors still get the full history.
- `tests/unit/test_vlm_top_logprobs_rows.py`: `top_logprobs_rows` and `SpecRequest.reports` equal the argsort order, with ties, `-inf` tails and NaN.
- `tests/unit/test_special_token_count_parity.py`: counts equal generation's prompt (failed before the change); chat and Anthropic probes have a source-level guard because they need a running server.
- `tests/unit/test_apc_sequence_hash_parity.py`: digest equal to `mlx_vlm.apc._sequence_hash`.

CPU limits: the mtp / dflash lanes themselves need Metal kernels, so the lane-level tests exercise `apply_row_processors`, `SpecRequest.reports` and `build_penalty_processors` directly; a Mac run of `test_constrained_spec` and the real-model script is still owed for the end-to-end spec on == off check.

## Update after integration (same day)

* The batch / generic `top_logprobs` gap is closed: `cc0b039` takes them from one device argpartition
  (`text_logprobs.top_logprobs_device`, hooked in the `GenerationBatch._step` / `PromptProcessingBatch.generate`
  wrappers) with the upstream order kept bit-for-bit; it falls back to the upstream sort if the uint64 probe fails.
  Needs a Metal A/B before claiming a speed win there.
* `path_matrix.json` now has `fast_top_logprobs`, `special_token_text`, `empty_prompt_refusal`,
  `finite_state_publish`, `grammar_mask_fast`, `generation_suppress_default` and `grammar_dead_end_isolation` rows with
  a cell on every path; the two `test_path_parity` failures noted above were fixed on the branch.
* An empty prompt on the VLM path now raises the shared `EmptyPromptError` (`cb075e1`), like the text paths.
* Still open: the penalty-window semantics and the logprob tie order (both listed as decisions in
  [`HANDOFF_REVIEW_2026-10-26.md`](HANDOFF_REVIEW_2026-10-26.md)).
