# Upstream and peer adoption, 2026-10-26

Outcome of going through every upstream/peer row the 2026-10-26 audit left MISSING (ids 09.\*, 10.\*, 13.\*: the
catch-up reviews of mlx / mlx-lm / mlx-vlm / mlx-audio, of the vendored and watched peers, and of the unwatched
peers). Each row was read at its upstream commit, checked against the code on this branch, and either adopted
(code + test, one commit per item), recorded as not applicable with the reason, or recorded as needing a Mac step.
Nothing here changes a pin: mlx stays 0.32.3, mlx-vlm 0.7.8, mlx-audio 0.5.9 (see "Held for the mlx 0.32.4 decision").

Status words: **adopted** `<commit>`, **not applicable** (reason), **needs Mac** (the step), **verified** (the code
already does it; the evidence is named), **handled elsewhere** (another lane owns it, listed at the end only so
the ids are accounted for).

## Commits

| commit | item |
|---|---|
| `79c3c78` | PIL images keep their channels-last layout in the Gemma 4 / EmbeddingGemma 2 image processors (mlx-vlm 8dd7746, 1e64ba4) |
| `44203ed` | a request whose filters keep one token (top_k 1, min_p 1) decodes greedily (splash 8d5e05b) |
| `0c1a9bb` | jump-forward loop prefills a long prompt in chunks and reads the last position only (mlx-swift-lm 3615fe4) |
| `862ba7e` | Claude Code `<total_tokens>` budget marker dropped wherever it sits in the instruction text (omlx 38f420d9) |
| `36a75f1` | tool-grammar cache key keeps the request's property order (mlx-swift-lm d02948b) |
| `bb3f948` | tests: Gemma nested tool-call arguments and long streamed call (mlx-swift-lm 0754e10, a1f575d) already handled |
| `c79c54b` | tests: promoted qmv bias sums equal omlx's 0.32.4 transcription (omlx 94363f89) |
| `edc1707` | text prefix cache: a lookup whose refeed trim fails is not counted as a hit (omlx 5628bb7e) |

Path coverage of the code changes (path_matrix.json cells): `pil_channels_last` engaged on mtp / dflash / batch /
generic (text, loop take no images); `one_token_greedy` engaged on all six; `budget_marker_strip` engaged on
mtp / dflash / batch / generic / text (loop applies no message adapter at all, na); the jump-forward chunking, the
grammar key and the hit counter live in code shared by the paths that use them (jump-forward and the text prefix
cache are text-path only, the grammar key is shared by the VLM and text engines). vendor.json gained two `patches`
rows (the Gemma 4 and EmbeddingGemma 2 `preprocess` bodies the image fix depends on).

## mlx / mlx-lm / mlx-vlm / mlx-audio (09.\*)

| id | upstream ref | status |
|---|---|---|
| 09.3 | re-derive nax_prefill.py / attn_prefill.py, relax the `== "0.32.3"` gates and doctor.py:578 | needs Mac, bump-coupled. Both kernels are derived from `quantized_nax.h` / `steel_attention_nax.h` at v0.32.3 and the gates fall back to stock on any other mlx. Step: with the 0.32.5 tag, diff those headers against the derived code, re-derive, run the same-checkpoint A/B for both kernels on the M5, then relax the gates, doctor.py and the pyproject pin together. |
| 09.4 | mlx 065ec468 sdpa_split verify-row path | needs Mac, bump-coupled. It changes the verify-vs-decode attention reduction order for qL<=8 with 33<=qL\*gqa<=128 (Qwen3.8-27B has gqa 6). Step: after the bump, run the spec on == off bit test (MTP and DFlash, 6..8 verify rows, long KV) on 0.32.5 before enabling. |
| 09.6 | mlx-lm qwen3_coder untyped-param JSON decode | handled elsewhere (done). |
| 09.9 | mlx-vlm 259802e `_release()` | handled elsewhere (listed in the brief). |
| 09.10 | `check_upstream.py --update-hashes` for the 2 changed symbols | not applicable until the mlx-vlm bump: against the installed 0.7.8 every `patches` row matches (`check_upstream.py --no-fetch` reports no changed or missing symbol; the two new rows are included). Run `--update-hashes` as the last step of the bump after porting `_target_verify_linears` and `_normalize_qk` (09.8). |
| 09.11a | mlx-vlm b567294 rotating-cache sinks / absolute positions | not applicable on 0.7.8. The change only differs for a cache with a Python-int `offset` and no `_idx` (a single `RotatingKVCache`) or `keep > 0`. Yunshu never passes `max_kv_size` to the VLM runner, so no rotating cache is created with `keep=4` (mlx-vlm `make_prompt_cache`); Gemma 4 builds its sliding layers with `keep=0` and b567294 does not touch the gemma4 modules; Qwen3-Omni / Qwen3-VL / Qwen3.5 use plain or `Batch*` caches where `_idx` was already preferred. For the bump: `cache_position_offset(c0)` reads `_offset` or `_idx` from a batched cache, and `RaggedKVCache` exposes `_idx` (ragged_kv.py), so nothing to port. |
| 09.11b | mlx-vlm 8dd7746, 1e64ba4 Gemma 4 / EmbeddingGemma 2 PIL channels-last | adopted `79c3c78` (reproduced on CPU: a 3-pixel-tall RGB image reached the tower as `(3, 4, 3)`). Gemma4Unified imports the helpers inside `preprocess()` and is not covered. |
| — | mlx-vlm 3fc4e5a d1 / d1-omni faster, lower memory | not applicable: the change is in `mlx_vlm/models/d1*`; Yunshu's d1-3B receipt rows load through `lfm2_vl`, not those modules. It arrives with the mlx-vlm bump. |
| 09.1, 09.8 | mlx 0.32.4 APC-identity blocker, mlx-vlm 0.7.10 port | already tracked (handoff Verify #7); unchanged here. |

## Vendored and watched peers (10.\*)

| id | upstream ref | status |
|---|---|---|
| 10.1c | omlx 94363f89 `float()` cast in moe_verify_gather | verified and pinned `c79c54b`: `stock_load_vector(_HEADER, True)` produces omlx's multi-term promoted sums for 4/5/6-bit in `load_vector` and `load_vector_safe` (the 8-bit single term is `float += half` either way); gathered kernels keep requiring N % 8 == 0 through `supported()`. |
| 10.2.1 | TensorFold 6588b5a1 / #1321 shared-prefix wait | verified, no change. The VLM runner builds every generator with `prefill_batch_size=1` (vlm_batch_runner.py:1017) and mlx-vlm looks a request's prefix up when its own prompt batch starts (`_apc_pick_for`), i.e. after the leader's prefill stored its checkpoints; the text fast path runs one request at a time on the single MLX thread. A follower therefore always resumes the leader's cut. |
| 10.2.2 | TensorFold #1262 / Rapid-MLX 421bf658 document-seam checkpoint | handled elsewhere. |
| 10.2.4 | TensorFold 993cebd3 simdgroup prompt attention before M5 | needs Mac (M1-M4). attn_prefill.py is M5-only; M1-M4 use stock MLX. Step: on an M3/M4 time cold prefill at 8k / 32k with a new simdgroup kernel against stock on the same checkpoint (their M3 Ultra figure: 176 -> 269 tok/s at 32k) and compare top-1 and logits; only then write a kernel (same-checkpoint A/B rule). |
| 10.2.5 | omlx sdpa256_attention (d98d9ffe, e0af2a26, 0eb8df1f, 03bc3dd7) | handled elsewhere (listed in the brief as M1-M4 sdpa256). |
| 10.2.6 | omlx e9ada8f3 / a3e2fcdd, Rapid-MLX d84585a7 MTP park / cohort cap | handled elsewhere (MTP at >= 4 rows verified). |
| 10.2.9 | splash f52d1ad constrained and free lanes in one command | verified, not adopted. A tool-call-only guide already shares the free batch (`invariant_guide`). A response_format or logprobs row takes the stock-serial arithmetic and APC namespace (`_serial_rows`, group key at vlm_batch_runner.py ~1190), so it decodes in its own group by design: merging would change the arithmetic the free rows see. needs Mac only if the 2-client decode loss of that split is worth measuring: 1 constrained + 1 free client, tok/s of each vs alone. |
| 10.2.10 | mlx2 b69c198 cancelled long prefill keeps progress | needs Mac. Yunshu cuts the prompt end, one interval boundary and the end of the system turn; a prefill cancelled before the interval boundary keeps nothing but the head boundary. Step: disconnect a 30k-token request at 50% of its prefill, retry it, read `cached_tokens` (and TTFT) of the retry; if it is ~0, design a cancel-time checkpoint at the last completed aligned chunk (lossless, but costs memory at cancel). |
| 10.2.11 | mlx-serve c24807e / 3de5370 shutdown flushes owed SSD slices | verified. Gateway shutdown calls `engine.stop()` (main.py lifespan), which waits up to 10 s for in-flight work and then `apc.close()`: the writer threads drain the FIFO write queue (capped by `APC_DISK_QUEUE_MAX_GB`, 1 GB by default) before exit, and uvicorn's graceful timeout cancels open streams, which cancels their prefill at the next chunk. No unbounded wait in practice. |
| 10.2.12 | Rapid-MLX 66b39f56 hybrid boundary retention | handled elsewhere. |
| 10.3.1 | omlx c4bb311e cache-store opt-out | handled elsewhere. |
| 10.3.2 | llama.cpp ead9f3205 logprobs with speculative decoding | handled elsewhere (verified done). |
| 10.3.4 | Rapid-MLX 3de1c6c4 Responses `max_tokens` alias | handled elsewhere (done). |
| 10.3.5 | `/v1/completions` `max_tokens=0` | handled elsewhere (decided: keep). |
| 10.3.6 | TensorFold #819 reasoning efforts in `/v1/models` | not applicable. Each item already carries the model card (`yunshu.reasoning`: `default_enabled`, `effort_levels`, `default_effort`, aliases) and the LM Studio flat `reasoning: true`; the OpenRouter object form would reuse the same key with a different type (model_card_formats.py:287). |
| 10.3.8 | omlx f0e3f6a8 bound thinking for Claude Code's auto-mode classifier | not applicable: a Messages request with no `thinking` field is thinking-off here (`test_anthropic_absent_thinking_is_disabled`), so the classifier is never given an unbounded thinking default. |
| 10.3.8 | omlx 38f420d9 strip budget markers | adopted `862ba7e`. The lone-message filter existed; the marker as a tail of the system text or as an extra system block was not stripped and made the prompt diverge on every request. |
| 10.4.3 | mlx2 fe3e538 / omlx #4526 padded batched prefill | not applicable: the VLM runner prefills one request at a time (`prefill_batch_size=1`), so no padded multi-row prefill exists; decode-time left padding touches only attention KV. The text loop's batched prefill is mlx-lm's BatchGenerator (loop is off by default and not grown), whose recurrent mask is mlx-lm's own. |
| 10.4.4, 10.4.5 | mlx2 1436775 row_exact K=64/128, b9db287 2 GiB offsets | handled elsewhere. |
| 10.4.6 | mlx2 3dc6f0f finish-path cleanup | handled elsewhere (done, `36d71b4`). |
| 10.4.8 | omlx 50b427f1 release the hit of a rejected request | not applicable here. Admission rejections (`reject_overlong_prompt`, `require_non_empty_prompt`, the gateway queue / memory refusals) happen before any APC lookup, so no hit exists to release; a restore refused after the lookup retracts its counters (`f5705ed`), an insert failure releases its request (`738242c`, `d76d6c1`); the text prefix cache hands out detached copies and holds no references. |
| 10.4.9, 10.4.10 | omlx 746447e2 forced think close, mlx2 01dcd75 grammar after reasoning | handled elsewhere. |
| 10.4.13a | omlx ee0c8c20 cast a bf16 DFlash drafter to an fp16 target | not applicable: the DFlash lane is bf16 by construction (the context ring and the empty-context placeholder are allocated bf16, dflash_ring.py:36-39, dflash_context.py:144, and the on-disk window refuses other dtypes, dflash_apc_window.py:147); an fp16 target is not on the lane, so there is no drafter/target dtype mix to resolve. |
| 10.4.13b | splash 8d5e05b top_k 1 / min_p 1 select greedily | adopted `44203ed`. |
| — | omlx 5628bb7e serving prefix hits counted after the reuse is accepted | adopted for the text cache `edc1707` (a failed refeed trim kept the hit counted). The VLM APC already counts accepted restores only (`ea82586`, `f5705ed`); `cached_tokens` is set from the accepted match. |
| — | mlx2 c250447 non-finite fp16 recurrent state | verified on both paths: `recurrent_state_finite` guards `store_exact_cache` (apc_manager.py) and `KVPrefixCache.add` (kv_prefix_cache.py), tests `test_apc_nonfinite_state.py`, `test_text_prefix_cache_nonfinite.py`; path_matrix `finite_state_publish` is engaged on all six paths. |
| 10.5.1 | omlx 2b1822bf / d9c294e8 generation-prompt tail in the same prefill step | not applicable: it fixes omlx's decode-fairness scheduler (the tail step waited for decode to repay a chunk). Yunshu has no debt-based fairness between prefill and decode; the BatchGenerator prefill is not split at the generation prompt. |
| 10.5.2 | Rapid-MLX 90b2540f, 65eec0da warm system+tools prefix, bounded `/prefill` | not adopted. Functionally covered: the end of the tool list and the end of the system turn are always checkpoints (prompt_caching.py), and a `max_tokens: 1` request warms them. A dedicated output-free route would add API surface and route budget; revisit if Yunmo needs it. |
| 10.5.2 | Rapid-MLX 75add145 quantized argmax float bias sum | not applicable: the argmax kernels take their header from mlx-vlm's `_target_verify_qlinear_header`, which `verify_select.install()` wraps with `stock_load_vector` when mlx >= 0.32.4 (and leaves stock on 0.32.3, which is what stock does). |
| 10.5.2 | Rapid-MLX 338d9e00 | not applicable: Rapid-MLX scheduler workspace seal and admission lock; Yunshu has neither (one prefill row, no workspace seal). |
| 10.5.2 | Rapid-MLX 897ffab0 persist text-only vision-lane prefix cache | not applicable: `YunshuAPCManager.close` writes the resident checkpoints (text-only included) and `SpillDiskStore` keeps them across restarts. |
| 10.5.3 | splash 1fec207 YaRN factor from max context | not adopted. A static YaRN factor changes attention scaling at every length, so it is not lossless; Yunshu advertises no context beyond the checkpoint's native window. needs Mac only if a beyond-native context is ever offered: A/B perplexity at 128k+ with and without the factor. |
| 10.5.4 | mlx2 9973c9b tree-size-invariant position law | needs Mac. Step: port `scripts/check_tree_size_invariance.py` against `dflash_tree` on Qwen3.8-27B (same teacher-forced stream verified through trees of 1 / 4 / 8 / 16 rows, shapes and mixes; every pair must give bitwise-equal fp32 logits per position). There is no tree-size test in tests/ or scripts/ today. |
| 10.5.4 | mlx2 7e1354d q4 drafter default | needs Mac (the commit gives no numbers): same-checkpoint A/B of `YUNSHU_DRAFT_BITS` 4 vs 8 on 27B, accept length and tok/s. |
| 10.5.4 | mlx2 1b574a2 wired weights resident between requests | verified, equivalent exists: `idle_memory.py` holds the wired limit and a 1 s heartbeat until `IDLE_TRIM_S` (30 s), measured 2026-10-22 (idle follow-up TTFT 0.23 s within 2 s, 0.44-0.66 s from 4 s on without it). |
| 10.5.5 | llama.cpp f174553e1, 61ad80bf9 SWA slot checkpoints | not applicable: llama-server restores a slot's checkpoint against its live context; Yunshu's checkpoints are keyed by the token hash of the exact prefix, so a checkpoint covering tokens the new prompt does not share is never found. The Gemma 4 ring metadata is validated on load (`check_loaded_cache`). |
| 10.5.5 | llama.cpp 098ce526b in-place aliasing | not applicable: `mx.fast.metal_kernel` outputs are fresh allocations and no Yunshu kernel binds an output over an input (grep of kernels/ for donate / in-place / atomic outputs). |

## Unwatched peers (13.\*)

| id | upstream ref | status |
|---|---|---|
| 13.A1 | vllm 0bfeb8d0e DFlash last prefix block | handled elsewhere. |
| 13.A2 | vllm d486219fc stop-trimmed spec tokens vs cache offset | not applicable: decode-time KV is never published to the prefix cache (checkpoints are cut during prefill; the next turn re-prefills the reply, `next_turn.py` says why), so a stop inside an accepted block cannot leave a cache offset that counts trimmed tokens. |
| 13.A3 | vllm 6540d5c8f / 4ac1f8d2a, ollama 1ce2b680 | handled elsewhere (grammar after reasoning). |
| 13.A5 | vllm f0abb2f82, 5be4266f5, d9fcde70f Anthropic error vocabulary, streaming errors, disabled thinking | not applicable: the Anthropic router already emits `api_error` / `overloaded_error` / `not_found_error` in the non-stream envelope and in `event: error` frames, and `thinking: {type: disabled}` maps to thinking off without a render probe (no vLLM renderer here). |
| 13.A8 | HF watch entries | needs a networked host: huggingface.co answers 403 through this sandbox's proxy. Re-run the HF checks for DeepSeek-V4-Flash-0731, Cloudflare/clef, Qwen3-Reranker-0.6B-4bit and the laya / embeddinggemma entries there. |
| 13.B1 | vllm-metal dafa052 split decode rows out of a mixed launch | not applicable: needs a unified prefill+decode kernel launch; the runner interleaves prefill chunks and decode steps as separate dispatches. |
| 13.B2 | vllm-metal ddca466 n-gram draft cap at state-block boundaries | not applicable: decode state is never checkpointed (see 13.A2), so no draft length has to align with a checkpoint boundary. |
| 13.B3 | vllm-metal b9a5167 GDN decay in FP32 | handled elsewhere. |
| 13.B4 | vllm-metal ca27320 `vm_stat` swap-outs as the startup pressure signal | needs Mac. Admission reads MLX active memory over the working set (admission.py:73). Step: on a Mac run a model past RAM and compare `Swapouts` against `YUNSHU_MEMORY_PRESSURE_REJECT`'s signal; add the swap term only if the ratio signal lags. |
| 13.B5 | vllm-metal 3b1bc46 wake after a failed KV release | not applicable: Yunshu has no sleep mode. |
| 13.C2 | ollama / mlx-lm / swift-lm 991266a clear the buffer cache per prefill chunk | needs Mac. mlx-lm's `generate_step` already clears per chunk (text path inherits it); mlx-vlm's batch generator clears at the end of the prompt batch, not per chunk, and Yunshu keeps MLX's pool on purpose (model_optimizations.py:1300 notes the compile-cache cost). Step: peak memory and TTFT of a 32k and 64k cold prefill with and without a per-chunk clear (same checkpoint), keep it only if peak drops without a TTFT loss. |
| 13.C4, 13.C5 | ollama b859a945 / 6137793a / 8d66f083, a5b659e6 | handled elsewhere. |
| 13.C6 | ollama cf6c9de6 residency refresh after GPU idle | verified, equivalent exists (`idle_memory.py`, see 10.5.4). Beyond `IDLE_TRIM_S` (30 s) the weights are released by design. needs Mac only to lengthen it: first-request TTFT at 30 / 60 / 120 s idle. |
| 13.C7 | ollama c1c53629 candidate logits in float32 for quantized heads | not applicable: Yunshu verifies draft rows with the same batch-invariant head as decode so that spec on == off bit for bit; scoring only candidates in fp32 would break that equality. |
| 13.C8 | ollama 1548f78c bound loads by system free memory | needs Mac (low priority). The model pool checks its `max_memory_bytes` budget before a load and the working-set pressure after it; a system-free-memory term needs macOS `vm_stat` semantics. Step: load a second model while another app holds memory and see whether the post-load check evicts in time. |
| 13.C10 | swift-lm c408452 GDN L2 eps | handled elsewhere (mlx-vlm b8bf04c port is bump-coupled, 09.8). |
| 13.C11 | swift-lm 12d935f rotating-KV rewind, 0ea4465 Qwen3-VL masks after rotation | not applicable. APC never rewinds a rotating cache (exact checkpoints validated by ring metadata); the only rotating-cache trim, the Gemma 4 assistant's rejected-tail trim, already stops speculating once the ring wraps (gemma4_assistant.py ~690). 0ea4465 needs a rotating cache under Qwen3-VL, which only `max_kv_size` creates and Yunshu never sets. |
| 13.C14 | swift-lm d02948b declared property order | adopted `36a75f1`: `grammar_key` sorted keys, so tool sets that differ only in property order shared one compiled grammar (llguidance emits properties in schema order; checked). |
| 13.C14 | swift-lm 0754e10 Gemma nested arguments | verified and pinned `bb3f948`: all five nested / marker-quoted shapes parse to typed arguments through the mlx-vlm gemma4 reader. |
| 13.C14 | swift-lm a1f575d quadratic end-tag rescan | verified, pinned `bb3f948`: a 40k-character call fed one character at a time takes ~0.3 s (`str.find` on the held buffer is C speed; the quadratic part of the Swift scanner was its frame scan). No window needed. |
| 13.C15 | swift-lm e9c8500 Qwen reasoning history, token-identical prefix | needs Mac / real template. Step: replay a 6-turn thinking transcript through the Qwen3.8 chat template with and without `reasoning_content` resent and assert each turn's prompt tokens are a prefix of the next turn's; then measure warm TTFT per turn. The tokenizer and template are not on this host. |
| 13.C16 | swift-lm 3615fe4 skip the unused vocabulary projection in prefill | adopted `0c1a9bb` for the one place that needed it (the opt-in jump-forward loop ran the whole prompt as one forward and sliced the last logits). The VLM path passes `logits_to_keep=1` (mlx-vlm ar.py) and mlx-lm's `generate_step` never evaluates prompt-position logits. Lazy vision-tower loading (6d84ef6) is a memory trade, not adopted. |

## Held for the mlx 0.32.4 decision

mlx-vlm >= 0.7.9 requires mlx >= 0.32.4 and mlx-audio >= 0.6.0 pins it exactly, while the unexplained
0.32.4 FP32 SSD APC-identity regression and the two derived NAX kernels are open (09.1, 09.3, 09.4, 09.8,
09.10). The rows above that depend on the bump (b567294 details, d1, the audio fixes) arrive with it; none was
ported ahead of the pin.

## Handled elsewhere, ids accounted for

Forced think close (omlx 746447e2), grammar after reasoning (01dcd75, 4ac1f8d2a, 1ce2b680), cleanup terminal
(3dc6f0f), logprobs on the spec lane, MTP >= 4 rows, qwen3_coder untyped parameters, `max_tokens: 0`, the
Responses `max_tokens` alias, system-note hoisting (5a0a15a7), cache-store opt-out (c4bb311e), the
document/question seam checkpoint (TensorFold #1262, Rapid-MLX 421bf658), Rapid-MLX 66b39f56, vllm 0bfeb8d0e,
ollama b859a945 / 6137793a / a5b659e6, mlx2 1436775 and b9db287, omlx sdpa256 for M1-M4, mlx-vlm b8bf04c /
63fc979 / 259802e, mlx-audio f37c9a9 / dbcc13c / 2cd66c6 / ebcdbfc, vllm-metal b9a5167, mlx-swift-lm c408452.

## vendor.json

The `watch` entries for the peers whose rows are all closed above (adopted, verified, not applicable, or recorded
with a Mac step) now carry `reviewed_commit` at the head the catch-up review read, `reviewed_date` 2026-10-26 and a
`review_reason` pointing here: omlx, TensorFold, llama.cpp, splash, Rapid-MLX, mlx2, mlx-serve, vllm, ollama,
mlx-swift-lm, vllm-metal, mlxcel and qwen38-mlx-swift. mlx, mlx-vlm and mlx-audio keep their 2026-10-15 review:
their bump-coupled rows are still open. The Hugging Face entries were not re-checked (13.A8).
