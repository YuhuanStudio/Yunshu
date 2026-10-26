# Path parity, text side: VLM fixes checked on every text path

> Line numbers in this report refer to the tree when it was written (2026-10-26, before the later review-fix and
> adoption merges) and drift; search by symbol. The "Update after integration" section at the end is the final state.


Scope: the VLM-path review fixes (`vlm_engine.py` / `vlm_batch_runner.py`) and the APC-lane fixes, checked and
ported on each text serving path, so the unification of the text fast path (`BatchedEngine`) and the VLM path
loses none of them. Paths:

| path | code |
|---|---|
| **text** | mlx-lm single-request fast path: `_generate_fast` (`engine_fast.py`), `_stream_generate_fast` (`engine_stream.py`) |
| **ngram** | n-gram speculative sub-path (`engine_ngram.py`; non-streaming route, greedy, `spec_decode` or `YUNSHU_NGRAM_DEFAULT`) |
| **spec** | draft-model speculative sub-path (`engine_speculative.py`, `YUNSHU_SPEC_UNVERIFIED=eagle`, non-streaming, greedy) |
| **loop** | EngineCore continuous batching (`engine_core.py`, `scheduler.py`; `YUNSHU_ENGINE_LOOP`, off by default, "not grown": bug fixes only) |

Cell legend: **ported** (fixed in this branch), **ok** (already correct, pinned by a test), **n/a** (path does not
have the code), **dead** (route is not reachable), **not grown** (loop optimisation, deliberately not added).
Evidence for every "ported" and "ok" cell is a test under `tests/unit/` named in the last column.

## Fix table

| # | fix | VLM commit | text before | text | ngram | spec | loop | what a unified path must keep | test |
|---|---|---|---|---|---|---|---|---|---|
| i | `generation_config.json` `suppress_tokens` is the default mask when a request sets none | d7a56fc | ignored (only a request's own list masked) | **ported**: `resolve_suppress_tokens` at the top of `generate()` / `stream_generate()` (`batched_engine.py:1393`, `:1931`; reader in new `generation_suppress.py`) | **ported**: n-gram does not implement the mask, so `_spec_route(plain_only=...)` (`batched_engine.py:1287`) sends a request with `suppress_tokens` / `min_tokens` / `ignore_eos` to the plain path (the route used to drop all three silently) | **ported**: same `plain_only` gate (the eagle route already excluded them) | **ported**: the resolved list reaches `add_request` / `SamplingParams.suppress_tokens` (scheduler already bans it) | one reader (`generation_suppress.default_suppress_ids`: valid int ids only, inside the vocab, cached per file version), request list wins, and any speculative lane must either apply the mask or step aside | `test_text_generation_config_suppress.py` |
| j | a zero-token prompt is refused with a clean error | 40aeb09 | silently replaced by one BOS/EOS token and generated from | **ported**: `require_non_empty_prompt` (`context_window.py:80`; `engine_fast.py:403`, `engine_stream.py:156`) raising `EmptyPromptError(ValueError)`; gateway maps it to 400 `invalid_request_error` / `code: empty_prompt` (`yunshu_gateway/main.py`) | **ported** (`engine_ngram.py:159`, `:1008`) | **ported** (`engine_speculative.py:217`, `:372`) | **ported** (`engine_core.py:1248`, replaced the BOS/EOS injection) | refuse before any state is built or queued, one error type with a 4xx mapping (the VLM raises a bare `ValueError`; unify on `EmptyPromptError`) | `test_text_empty_prompt.py` |
| k | EOS / end of generation inside an open `<think>` keeps the tail as reasoning | 8a3efe4 | streaming already classified the tail by `_in_thinking` (ok); the non-streaming parsers returned the whole reply as **content** when the prompt opened `<think>` and the reply never closed it | **ported** non-stream (`reasoning_parser.close_unterminated_reasoning`, `engine_fast.py:1648`); stream **ok** (pinned) | **ported** non-stream (`engine_ngram.py:856`); streaming n-gram route is **dead** | **ported** (`engine_speculative.py:688`) | **ok**: `_ReasoningTracker` is per token; EOS keeps the open state | the final tail inherits the state at the stop token; a reply with no close tag after a prompt-opened think is all reasoning; structured JSON replies are exempt | `test_text_eos_inside_think.py` |
| l | usage counts the tokens the consumer took; first-token event flagged once for TTFT | 001a092 | the stream billed held-back stop-prefix tokens (5 where the same reply non-streamed said 4); counts otherwise ride each queue tuple (no producer lead); TTFT is recorded once by the producer | **ported**: `_delivered_token_count` re-encodes the delivered text when a user stop string ended the reply (`engine_stream.py:1361`, helper `engine_policy.py:63`); TTFT: ok | non-stream already re-encodes on suffix stop; streaming route **dead** | n/a (no stop-string billing; counts `len(token_ids)`) | **ok**: `RequestOutputCollector` merges outputs and carries the scheduler's count; no consumer lead exists | usage = tokens whose text reached the client; the first event that has a token carries the TTFT once | `test_text_usage_counts.py` |
| m | the stream's terminal sentinel / error is never dropped by a full queue | b60bc98 | **ok**: `StreamBridge` + `make_stream_queue` (terminal slots outside the data capacity, overflow turns into one visible error) is already in place | **ok** (`engine_stream.py:367`) | **ok** (`engine_ngram.py:1135`) | n/a (no streaming) | **ok**: `RequestOutputCollector` is a single slot plus a sentinel flag, it cannot fill | keep terminal items out of band of the data capacity | existing `test_stream_bridge.py` |
| n | a checkpoint with a non-finite recurrent state is never published to a prefix cache | 5a46c83 | stored anything | **ported** at the door every store goes through: `KVPrefixCache.add` (`kv_prefix_cache.py:467`, `recurrent_state_finite` at `:60`) and `PromptCacheManager.store` (`prompt_cache.py:195`); the auto-quant call sites other agents own are untouched | **ported** (same `add`) | n/a (no cache stores) | **ported** (scheduler stores via the same `KVPrefixCache.add`) | one fused isfinite reduction and one host sync per store, recurrent (`ArraysCache`) layers only; dense KV is not inspected (cost) | `test_text_prefix_cache_nonfinite.py` |
| o | a grammar dead end fails only its own request | e7d0821 | **ok**: single request, the dead end ends as a terminal error output (stream chunk / `finish_reason=error`) and `_active_fast_path_count` returns to 0 | **ok** (pinned both ways) | **ok** (outer handler returns an error output) | **ok** | **ported**: `_DeadEndGuardSampler` per constrained row (`scheduler.py:730`, `_guard_row_sampler` `:4141`, finish with `finish_reason="error"` in `_process_responses` `:3654`); before, the exception left `BatchGenerator.next()` and `_fail_active_requests` failed every request | a dead end is a per-row fault that finishes that row with an error; nothing a row's processor raises may escape the shared step | `test_text_grammar_dead_end.py`, `test_loop_grammar_dead_end.py` |
| p | a client timeout is a deadline on the whole call | 0799979 | non-stream checked every 32 tokens (overshoot of seconds on a slow model); stream has a producer total deadline plus an inactivity gap | **ported**: checked every token (`engine_fast.py:1047`); stream **ok** | **ported**: checked every cycle (`engine_ngram.py:431`; a `len % 32` check was skipped by cycles that accept several tokens); streaming route **dead** | **ok**: `asyncio.wait_for` around the whole call | **ok**: `engine_core` request deadline | total deadline from request start, enforced per token, in addition to any inactivity gap; no default deadline (f138284) | `test_text_total_timeout.py` |
| q | orphan adoption / cancel after disconnect | cc7e7e1 | n/a: the text engine has no orphan adoption (`orphan_adoption.py` is imported by `vlm_engine.py` only; the gateway's `AutoSeed` is just a seed marker) | n/a (recorded, not built) | n/a | n/a | n/a | if adoption is built for the unified path, the cancel-after-disconnect rule must come with it | none |
| r | temp media cleanup | 9f9f1ec | n/a: text requests carry no media files | n/a | n/a | n/a | n/a | media cleanup stays with `prepare_media` | none |
| s | bounded prompt / template cache (tokens and characters, int32) | dc34956 | n/a: the text path has no prompt-token or template cache (it tokenizes per request). Existing caches are bounded: `PromptCacheManager` (entries and bytes), `ResponseCache` (entries, bytes, TTL), `TokenizerPrefixCache` (entries and bytes), `KVPrefixCache` (entries, memory pressure) | n/a | n/a | n/a | n/a | any cache of token ids added in the unified path needs the VLM bound (total tokens and chars, int32 `array`) | none |
| t | hash the request key once | 5808886 | the fast path serialized and hashed the whole rendered prompt twice (prompt-cache lookup and store) | **ported**: key memoised on the request (`engine_fast.py:830`) | n/a (no prompt cache) | n/a | n/a | one conversation hash per request, shared by lookup and store | `test_text_request_hashed_once.py` |
| u | ASGI / `ClosingStreamingResponse` | 82cd159 | gateway level, shared | skipped | skipped | skipped | skipped | n/a | n/a |
| v | APC e6e56ec: failed WARM promotion keeps the entry | e6e56ec | n/a: the text WARM tier quantizes in place (`_maybe_demote_to_warm`), there is no take-then-promote; a failed quantize leaves the full-precision list in place | n/a | n/a | n/a | n/a | n/a | none |
| w | APC 0d36e4c: whole-prompt memo keys | 0d36e4c | n/a: the text path has no divergence-point memo; `KVPrefixCache` is keyed by the full token hash and block chain, with a token-compare on collision | n/a | n/a | n/a | n/a | n/a | none |
| x | APC ae33517: transient load errors keep the file | ae33517 | `SSDKVCache.load_block` and `HybridSnapshotStore.load` deleted the checkpoint on any exception | **ported**: new shared `transient_errors.is_transient_load_error` used by `ssd_kv_cache.py:886` and `hybrid_ssd_snapshot.py:132,:382` (scan keeps the file too); corruption still deletes | **ported** (same stores) | n/a | **ported** (same stores) | corruption is the only reason to delete a checkpoint | `test_text_ssd_transient_load_errors.py` |

## Text-path fixes already on the branch, checked on the sub-paths

| fix | commit | ngram | spec | loop | action |
|---|---|---|---|---|---|
| no default 300 s deadline | f138284 | ported earlier | ported earlier | ok | none |
| wired limit sized from `tree_flatten(parameters())` | 71c183e | shared `_wired_limit_ctx` | shared | n/a | none |
| dense frequency / presence / logit_bias vectors | 4634f98 | **ported**: `engine_sampling._apply_spec_bonus_penalties` (the n-gram and `SpeculativeDecoder` bonus-token path) was the same per-id MLX loop, now dense (44 ms to 0.27 ms per call at about 500 unique tokens, bit-identical for float32 / bfloat16) | **ported** (same helper) | **not grown** (`scheduler._freq_pres_penalty` keeps the per-id loop) | `test_spec_bonus_penalties_dense.py` |
| `top_logprobs` argpartition | f3f1d5c | n/a (route excludes logprobs) | n/a (route excludes logprobs) | **not grown** | none |
| inflight register from the Python list | 9d9c7d8 | **ported**: both n-gram paths rebuilt the list with `[int(t) for t in mx.array]` (1.06 s per 32k tokens on CPU); now `list(input_ids)` (`engine_ngram.py:368`, `:1251`) | n/a | `find_prefix` receives a Python list already | `test_ngram_inflight_register.py` |
| auto length-gated 8-bit KV not stored in token-keyed caches | 6cc09fb | owned by another agent (`engine_ngram.py` `prefix_cache.add` sites) | n/a | not touched | none |
| special-token strings in message text stay text | 969b1e9 | via `_apply_chat_template` (ok) | via `_apply_chat_template` (ok) | **ported**: `EngineCore._messages_to_text` called the tokenizer directly and skipped `neutralize_special_text` (a user could close their turn and open a system one) (`engine_core.py:3483`) | `test_loop_control_token_injection.py` |
| `allowed_mask` for llguidance / JSON string states | 490898d | keeps `get_allowed_tokens` lists (`engine_ngram.py:266`, `:1096`; follow-up) | n/a (cross-model decoder drives the constraint through lists) | shares `ConstrainedSampler` | **ported** for `ConstrainedSampler` (text fast, stream, loop): `json_schema.apply_allowed_mask_from`; `test_text_allowed_mask.py` |

## Not done

* Orphan adoption on the text path (q): not built, by instruction.
* n-gram and speculative `get_allowed_tokens` list calls (490898d follow-up) and `top_logprobs` argpartition on the loop
  were not touched: loop is not grown, and the n-gram constrained loop is a separate edit.
* Dense KV finiteness (n): only recurrent state is checked, as on the VLM path, because the KV scan is memory-bound.
* The gateway returns the non-stream timeout as `finish_reason="length"` (`_INTERNAL_MAP["timeout"]`), the stream reports
  `timeout`; both reach the client as `length`.

## `path_matrix.json` cells for the lead to apply

`tests/unit/test_path_parity.py` currently fails two checks on the integration branch itself (neither comes from this
work):

1. `penalties_in_spec` / `text`: evidence `["python/yunshu_engine/engine_fast.py", "_freq_pres_penalty"]` no longer exists
   (4634f98 moved it). Replace with `["python/yunshu_engine/text_logit_processors.py", "class FrequencyPresencePenalty"]`.
   Optional added cell note: the n-gram / speculative bonus path is dense too,
   `["python/yunshu_engine/engine_sampling.py", "fpc = _mx.zeros"]`.
2. `YUNSHU_GRAMMAR_PREWARM` has no entry in `settings` (map it to `constrained_bitmask_engine`, owner of the grammar lane).

New features introduced by this work need no row (no new setting, no new optimisation module); if the lead wants them
tracked, suggested rows (text path, status `engaged`):

| feature id | path | status | evidence |
|---|---|---|---|
| `spec_bonus_dense_penalties` | text (n-gram / spec bonus) | engaged | `python/yunshu_engine/engine_sampling.py`, `fpc = _mx.zeros` |
| `allowed_mask_sampler` | text | engaged | `python/yunshu_engine/json_schema.py`, `def apply_allowed_mask_from` |
| `allowed_mask_sampler` | loop | engaged (shared class) | same |
| `row_isolated_dead_end` | loop | engaged | `python/yunshu_engine/scheduler.py`, `class _DeadEndGuardSampler` |
| `ssd_transient_errors` | text | engaged | `python/yunshu_engine/transient_errors.py`, `def is_transient_load_error` |

## Update after integration (same day)

* The "not grown" loop cells were closed at the user's request (every path gets every fix before the unification):
  `aed9970` moves the loop's penalties / logit_bias / suppress / ignore_eos / min_tokens to the shared
  `text_logit_processors` (bit-identical to the old closures) and gives the loop real `top_logprobs` entries through
  `text_logprobs.logprob_entry` (before, the raw vocab-wide logprobs array leaked into every output).
* The n-gram draft grammar filter now tests membership through the constraint's mask
  (`json_schema.allowed_membership`, `96e5815`) instead of `set(get_allowed_tokens(...))`. The draft-model speculative
  path still drives its cross-model constraint through lists.
* The `path_matrix.json` cells listed above were applied (plus the new feature rows), and `test_path_parity` is green.
