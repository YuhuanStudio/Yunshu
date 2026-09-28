<!-- Recorded 2026-09-28 at commit e26737b3, before the legacy VLM loop was deleted. Line numbers refer to that commit's python/yunshu_engine/vlm_engine.py. -->

# VLMEngine legacy generation path: implementation record

Snapshot of `python/yunshu_engine/vlm_engine.py` (6615 lines) taken before the legacy (non-runner) code is deleted. Line numbers are from this snapshot. The runner is `VLMBatchRunner` in `vlm_batch_runner.py`.

## 0. Critical scope fact

`_build_batch_runner` (L4258) returns `None` unless `config.model_type` is in `_RUNNER_MODEL_TYPES = ("qwen3_5", "qwen3_6", "qwen3_5_moe")` (L4226). So every other VLM family uses the legacy path today. That includes GLM-OCR, Qwen3-Omni (`qwen3_omni_moe`), Qwen2.5/3-VL (when routed to VLMEngine), Nemotron-Omni, Gemma-4 (when routed to VLMEngine) and the `SINGLE_IMAGE_ONLY_MODELS`. Deleting the legacy path without widening this gate removes generation for all of them.

## 1. Entry points and dispatch

**Runner construction (L1365-1379):** only when `_is_vlm` and `YUNSHU_VLM_RUNNER` is not `0/false/no`. Any exception sets `_batch_runner=None` and logs "using legacy text loop".

**`_runner_text_eligible` (L4408):** true when the runner exists and none of `_RUNNER_UNSUPPORTED_KWARGS = ("logits_processors", "lora_adapter")` (L4406) is truthy. Logprobs do not affect eligibility.

**`generate()` (L1588), inner `_generate_sync` (L1686), checked in order:**
1. Images, no audio, `_has_vision`, `max_tokens>0`, eligible → `prepare_images` then `_RunnerCall(_generate_vlm_runner_text)` (L1695-1737).
2. (Images and `_has_vision` and `_is_vlm`) or (audio and `_is_vlm`) → `_generate_vlm_vision` (L1738-1763). A `json_schema` here raises `ValueError` (L1745).
3. Media present but model is text-only → `RuntimeError` (L1765).
4. `_is_vlm` text: eligible → runner (L1779); otherwise `_generate_vlm_text` (L1812).
5. Not `_is_vlm` (loaded via mlx-lm, or mlx_vlm load failed at L1103-1130) → the `mlx_lm.generate_step` fallback (L1834-1950).

**`generate_stream()` (L2076), `_stream_sync` (L2225):** same order.
- Runner with images (L2239) → `_stream_vlm_vision` (L2283) → runner text (L2321) → `_stream_vlm_text` (L2364).
- Otherwise the `generate_step` fallback (L2392 to about L2745).

**Paths that land on legacy:**
- any non-runner family;
- any audio input (runner image path requires `not audio`);
- `lora_adapter` or `logits_processors` set;
- `YUNSHU_VLM_RUNNER=0`;
- a runner build failure;
- `max_tokens<=0`.

**LoRA and `logits_processors` are not actually implemented on legacy.** Neither kwarg is read anywhere in the legacy functions (grep finds `lora` only at L4406). Such requests go to legacy and are silently ignored there, which contradicts the docstring at L4409-4411.

**Return contract:** 6-tuple `(text, reasoning_tokens, completion_tokens, stop_hit, budget_hit, cached_tokens)`. Streaming pushes `RequestOutput` through `_ThreadSafeQueue` (L2185), which uses `call_soon_threadsafe`.

**Timeouts:**
- Non-streaming: `timeout_seconds` only (L1957).
- Streaming: inactivity timeout `timeout_seconds or timeout or 300` (L2771), which sets `cancel_event`.

## 2. Features implemented on legacy

**Per-model special cases**
- **Gemma-4 thinking off by default:** if `enable_thinking is None` and `"gemma-4"` is in `model_name` (L1640-1645). This is only in `generate()`; `generate_stream()` has no such default (quirk).
- **Gemma channel reasoning:** `_resolve_reasoning_channel_ids` (L6402) and `_decode_with_reasoning_channels` (L6435) turn `<|channel>thought…<channel|>` into `<think>…</think>`. Used by non-streaming text and fallback decode; the streaming paths don't use it.
- **`SINGLE_IMAGE_ONLY_MODELS` (L459):** glm_ocr, phi3_v, phi3.5_v, florence2, moondream1/2, minicpmv/2, llava_llama3, paligemma. `_extract_images` truncates to 1 image (L5842). `max_images` is threaded into `_build_vlm_messages` / `_apply_vlm_template_with_cache` so the placeholder count matches.
- **GLM / tool calls:** `_format_prompt` (L5533)
  - remaps roles `developer→system`, `function→tool`;
  - calls `message_adapter.adapt_messages`;
  - keeps `tool_calls`, `tool_call_id`, `name`, `reasoning_content`;
  - `_normalize_vlm_tool_calls` (L5449) converts string args to dicts, because GLM-4V templates raise on strings;
  - assistant-prefill → `continue_final_message`, with retries when the template rejects `continue_final_message` or `enable_thinking`.
- **`_tokenize_with_cache` (L5646):** double-BOS guard.
- **Nemotron-Omni:** processor has no chat template, so fall back to the tokenizer template (L5747-5769). `_audio_arg` reads sample rate from `processor.audio_sampling_rate` (L5936). The model_type remap lives in `mlx_vlm_patches.py`.
- **Audio input:** `_extract_audio` (L5942) handles `input_audio` base64 and `audio_url`. `_audio_arg` (L5912) preloads with `mlx_vlm.utils.load_audio` at the feature-extractor rate (default 16 kHz) and always returns a list (a bare ndarray trips `audio or None`). Audio goes to `mlx_vlm.generate/stream_generate(audio=…)`, which builds `input_features` / `feature_attention_mask` internally. The template gets `num_audios`; `_build_vlm_messages` emits `{"type":"audio"}`.
- **Video:** `_extract_video_frames` (L6207) accepts `data:video`, `file://`, bare path and `video_file`; http is rejected. It runs ffmpeg at 1 fps with at most 8 frames. Frames are appended to `image_paths`. Placeholders are deferred to the end of the message (L5415-5440) to keep the image order right. The runner's `prepare_images` gets these frames as images too, so this part is shared.

**Sampling and decoding**
- **`_build_noncached_sampler` (L74):** numpy sampler for temp>0, fixes the "⚠️ A" VLM determinism bug. Greedy uses mlx-lm `make_sampler`.
- **Seeds:** `mx.random.seed` inside `generation_stream` (L3527-3542). Vision seeding is best-effort only (L2928).
- **Penalties:** repetition, frequency, presence and logit_bias in Python loops, on the text paths only (L3720-3738, L5120-5140). The vision paths ignore them.
- **XTC:** accepted but **never applied** on any legacy path.
- **min_tokens, ignore_eos, suppress_tokens, top_n_sigma:** not on legacy (runner only, L4636-4639).
- **Logprobs:** not supported on legacy (warnings at L1676, L2109).
- **Grammar/JSON:** `JsonSchemaConstraint` or `ConstraintFactory` (regex, choice, cfg), text paths only (L3461-3495). Image/audio requests with a schema raise an error (non-streaming only; streaming vision silently ignores it).

**Stop handling**
- Single-token stop strings are added to `stop_ids`.
- Multi-token stops:
  - non-streaming text: suffix check at the end (L3796);
  - streaming: `StopHoldbackBuffer` plus `feed()+take_stopped()`;
  - vision: `str.find` truncation (L3020) or incremental holdback (L4046-4092).
- The non-streaming `generate_step` fallback ignores multi-token stop strings.

**Thinking budget**
- `reasoning_effort` maps to low=2048, medium=8192, high=32768. `_template_effort_extra` (L5504) instead passes the effort to templates that support it.
- Detection:
  - token-ID based when `<think` and `</think` are single tokens;
  - otherwise a text suffix scan;
  - vision paths use `_find_think_tag` (L6472).
- When the budget is hit:
  - non-streaming text/fallback appends `</think>`;
  - non-streaming vision truncates the thinking by chars-per-token after the fact (L3063-3081);
  - streaming vision emits `</think>` with `finish_reason="length"`;
  - streaming text uses `finish_reason="stop"` (inconsistent).

**mRoPE**
- Text paths call `clear_rope_state` at entry (L3438, L4779).
- `_prime_mrope_reuse_state` (L3242) sets `_rope_deltas=0` on prefix reuse.
- Vision calls `capture_rope_deltas` after generation (L3084), for logging only.

**Cancellation**
- Text loop checks every 16 steps (L3714).
- Streaming checks every step.
- `_prefill_vlm_ar_text` checks between chunks.
- Vision stream checks per yielded token; non-streaming vision cannot be cancelled.

**Other**
- `_then_clear` (L5474): `mx.clear_cache` after the job on large models (`_mx_large_model`, from `YUNSHU_VLM_LARGE_MODEL_GB`, default 10).
- prompt_tokens estimation via `_estimate_image_tokens` (L6505).
- Temp-file tracking per request.

## 3. Legacy-only caches

| Cache | What it holds | Key | Invalidation / bounds | Benefit (documented) |
|---|---|---|---|---|
| `_text_kv_prefix_cache` (`KVPrefixCache`, L917-943), text paths | Prompt-boundary KV snapshot (HOT/WARM 4-bit/SSD) | Token-id prefix, `min_prefix_length=32` | LRU `YUNSHU_PREFIX_MAX_ENTRIES`=128, `YUNSHU_PREFIX_HOT_LIMIT`=32; large models 24/8 (`YUNSHU_VLM_LARGE_PREFIX_MAX/HOT`, L1343); cleared in `stop()` L1513 | GLM-OCR HOT 12.59×, WARM 8.44×, SSD 1.15×; Qwen3-Omni-30B HOT 13.27×, WARM 12.35×, SSD 2.05× (KV_CACHE_MATRIX.md L21-22); doc says HOT 12.6× / 13.1× (VLM_TEXT_KV_PREFIX.md) |
| Reuse gates | `_text_prefix_reuse_safe` (L3154) = capability layer + `_probe_text_reuse_lossless` (L3172, greedy 12-step match, run at load L1387-1407 only when no runner) | — | Memoized per engine | Gemma (RotatingKVCache) and hybrid Qwen3.5 are bypassed |
| Hybrid prefix (`YUNSHU_VLM_HYBRID_PREFIX`=1, `_BLOCK`=128) | trim=0 boundary snapshots via `_capture_vlm_hybrid_prefix` (L3330); above 8192 tokens keeps only the latest boundary; `_no_trim_mode` | Token prefix at block multiples | Probe `_probe_hybrid_reuse` (L3274) | Doc: probe FAILS on Qwen3.5-2B because mlx_vlm GatedDeltaNet isn't chunk-invariant, so it is bypassed in practice |
| Full-match refeed | Re-prefill the last ≤128 tokens (non-hybrid) or cold-prefill (hybrid) (L3577-3589) | — | — | Needed for greedy parity |
| SSD tier (`YUNSHU_SSD_CACHE`, `_DIR` default `~/.cache/yunshu/kv-ssd-vlm`, `_MAX_GB`=10) | int8 KV on disk | Model basename + prefix | `note_prefill_tps` auto-gate (L3655); `YUNSHU_SSD_RESTORE_MIN_TOKENS` | Net-negative for fast-prefill GLM-OCR (0.93–0.96×) |
| `_kv_prefix_states` (L900), vision paths | mlx_vlm `PromptCacheState` passed as `prompt_cache_state` | `_compute_image_hash` (sha256 of path + bytes, 16 hex chars) | Max 32; evicts the first 8 when full; cleared in `stop()` | None measured. First sight stores an empty state; reuse depends on mlx_vlm (unclear) |
| `_CachingVisionTower` (L575), `YUNSHU_VLM_VISION_CACHE`=0, `_ENTRIES`=4 | Vision tower output | blake2b of pixel bytes + args | LRU | Default OFF. Comments (L1418-1425): redundant, `hits=0`; hangs the Qwen3-Omni-30B tower; had a wrong-image key collision (fixed) |
| `VisionFeatureCache` + `_MlxVlmVisionCacheAdapter` (L721), `YUNSHU_VISION_CACHE` (default on), `_DIR` | Vision features | (file-bytes hash, model_name) | — | **Inert:** mlx_vlm has no `vision_cache` kwarg (L2955-2960) |
| `EncoderCacheManager` (L883), `YUNSHU_ENCODER_CACHE_MAX`=64, `_TTL`=300 | `result.encoder_outputs` | `vlm-{image_hash}` | TTL/LRU | Writes only if the result has `encoder_outputs` (unclear if any does); never read |
| `_VLMTextPromptCache` (L475), `YUNSHU_VLM_TEXT_CACHE_MAX`=256 | Token ids / template text | blake2b(messages, thinking, audios, max_images, extra) | LRU | Shared: the runner also calls `_tokenize_with_cache` / `_apply_vlm_template_with_cache` |
| Spec prefill `YUNSHU_VLM_SPEC_PREFILL` (L970, L3549) | Nothing | — | — | **No-op placeholder**, logs only when >8192 tokens |
| `_prefill_vlm_ar_text` (L3373), `YUNSHU_VLM_AR_PREFILL_CHUNK_TOKENS`=0 | Chunked prefill (qwen3_5_text only, min 16) with `mx.eval` + `clear_cache` per chunk | — | — | Cancellable prefill; opt-in |

## 4. Env flags that exist only for the legacy path

- `YUNSHU_VLM_RUNNER` (switches to legacy)
- `YUNSHU_VLM_KV_PREFIX`
- `YUNSHU_PREFIX_MAX_ENTRIES`, `YUNSHU_PREFIX_HOT_LIMIT` (VLM use)
- `YUNSHU_VLM_LARGE_PREFIX_MAX`, `YUNSHU_VLM_LARGE_PREFIX_HOT`
- `YUNSHU_VLM_HYBRID_PREFIX`, `YUNSHU_VLM_HYBRID_PREFIX_BLOCK`
- `YUNSHU_SSD_CACHE`, `YUNSHU_SSD_CACHE_DIR`, `YUNSHU_SSD_CACHE_MAX_GB` (VLM instance), `YUNSHU_SSD_RESTORE_MIN_TOKENS`
- `YUNSHU_VLM_VISION_CACHE`, `YUNSHU_VLM_VISION_CACHE_ENTRIES`
- `YUNSHU_VISION_CACHE`, `YUNSHU_VISION_CACHE_DIR`
- `YUNSHU_ENCODER_CACHE_MAX`, `YUNSHU_ENCODER_CACHE_TTL`
- `YUNSHU_VLM_SPEC_PREFILL`
- `YUNSHU_VLM_AR_PREFILL_CHUNK_TOKENS`
- `YUNSHU_VLM_ASYNC` (mentioned at L990; `_async_core` is never used, so unclear/dead)

`YUNSHU_VLM_LARGE_MODEL_GB` is shared: it also drives the runner's `clear_on_idle`.

## 5. Known bugs and quirks noted in code, commits and docs

- **Omni turn-2 "!!!!" corruption:** lazy mRoPE `position_ids` aliased across `generate_step` calls. The fork added `mx.eval(position_ids)` (commit a1ff3c81). Per docs/research/runs/2026-09-28-omni/README.md and commit 8638e572 it does not reproduce on mlx 0.32.2 / mlx-vlm 0.7.3 (text and audio). Images and 20-turn sessions were not tested.
- **S=0 guard:** `mlx_vlm_patches._patch_qwen3_5_empty_chunk` (L81-108) returns `x` for empty chunks. It is hit by hybrid boundary resume with an empty suffix.
- **Empty-suffix `lm([])` crash** is avoided by the full-match refeed (L3574-3576).
- **Store before decode:** `add()` runs before decode because sliding-window caches get corrupted by a post-generation add (L3664-3674).
- **Single-token refeed** flips the greedy argmax; a block refeed is used instead.
- **Doc inconsistency:**
  - KV_CACHE_MATRIX.md L84 says Qwen3-Omni is auto-bypassed, while its table and VLM_TEXT_KV_PREFIX.md say it is lossless with rope priming.
  - The `__init__` comment (L910) still says the path supplies "explicit position_ids"; the code primes `_rope_deltas` instead.
- **Gemma-4 thinking default** is missing in `generate_stream`.
- **XTC** is accepted but ignored on legacy.
- **Vision paths:** ignore penalties, logit_bias and `stop_token_ids`.
- **Template extra:** non-streaming vision doesn't pass `template_extra`, but prompt counting does (L2013), so counts can mismatch.
- **Budget `finish_reason`** differs: text streaming "stop", vision streaming "length".
- **Inert components:** the vision_cache kwarg and the encoder cache.
- **Quantization:** a failed mlx_vlm load on a quant-shape error raises instead of falling back (L1103-1120).
- **`_wrap_mlx_vlm_for_mlx_lm` (L373)** patches `LanguageModelOutput` and `InputEmbeddingsFeatures` unpack issues for the `generate_step` fallback.

## 6. Porting checklist for the runner

**Essential (user-visible capability loss otherwise)**
1. Widen `_RUNNER_MODEL_TYPES` (or make it capability-based) and validate GLM-OCR, Qwen3-Omni, Qwen-VL, Gemma-4 and Nemotron-Omni in the runner matrix. Includes mRoPE `rope_deltas` per row (the runner already stores `job.rope_delta`, L449/L527) and the interleaved-mRoPE Omni thinker.
2. Audio input: extend `prepare_images` to call `prepare_inputs(audio=_audio_arg(...))`, forwarding `input_features` / `feature_attention_mask` and the `num_audios` template kwarg. The APC salt must include audio (currently `media={"audio": None}`, L254).
3. Keep `_build_vlm_messages` / `_apply_vlm_template_with_cache` semantics: SINGLE_IMAGE truncation, deferred video placeholders, and the Nemotron tokenizer-template fallback.
4. Gemma channel-token reasoning split in `_runner_events` (it only knows `<think>` IDs) and the Gemma-4 thinking-off default (apply it in both entry points).
5. Decide on LoRA: implement it or reject `lora_adapter` with a 400. Today it is silently ignored. Same for `logits_processors`.
6. Non-runner text-only fallback (`generate_step` for mlx_lm-loaded models): route elsewhere (BatchedEngine) or keep a minimal path.
7. Streaming inactivity timeout and cancel semantics; grammar/JSON on image requests (the runner may already support it via `ConstraintProcessor`; verify).

**Nice to have**
- Multi-token `<think>` text detection for tokenizers without single-token tags.
- Chunked, cancellable AR prefill.
- Large-model `clear_cache` hygiene (already covered by `clear_on_idle`).
- Legacy stats keys in `get_stats` (L6549-6590).

**Can be dropped**
- `_CachingVisionTower`, `VisionFeatureCache` adapter, encoder cache, `_kv_prefix_states`, spec prefill.
- The text KV prefix cache, hybrid probes and VLM SSD tier, once APC (plus the `YUNSHU_VLM_APC_DISK_DIR` disk tier) covers the new families. Re-measure GLM-OCR and Omni hit rates first: the legacy path's documented 12-13× HOT speedup is the benchmark to match.