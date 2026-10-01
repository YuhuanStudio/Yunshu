# API surface

Every public route, what it is checked against, and what was removed. The OpenAI and Anthropic
references are the specification; a route stays only if it matches one (or a de-facto local-server
API) and has a unit test plus a smoke test against a real model.

How each row was checked: **SDK** = the official `openai` / `anthropic` / `ollama` Python SDK against a
real model (Qwen3.5-0.8B via the VLM runner, Qwen2.5-3B-Instruct-4bit via the mlx-lm fast path,
Qwen3-Embedding-0.6B, Qwen3-ASR-1.7B, whisper-large-v3-mlx, Qwen3-TTS, Z-Image-Turbo, GLM-OCR).
**unit** = a test in `tests/unit` that needs no model. The smoke scripts ran through the GPU queue during the audit (about 80 checks per chat model); the checks that matter are also unit tests (`test_api_conformance.py`,
`test_ollama_api.py`, `test_tokenize.py`, `test_regex_constraint_linear.py`, `test_anthropic.py`).

`model` is advisory in single-model mode (`yunshu serve -m`): the loaded model answers under any name
and the response echoes the requested name, so existing clients need no reconfiguration. In
multi-model mode (`--models-dir`) an unknown model is a 404 `model_not_found`.

## OpenAI

| Route | Status | Notes |
|---|---|---|
| `POST /v1/chat/completions` | kept, fixed | SDK + unit. All params below verified. `stop` accepts a string or a list. `usage.prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens` are always present. Context overflow is 400 `context_length_exceeded`. Images on a text-only model are a 400. An embedding-only model (sentence-transformers export) answers a 400 that points to `/v1/embeddings`. |
| `POST /v1/completions` | kept | SDK + unit. `echo`, `logprobs` (int; also on the VLM runner, streamed and not), `n`, `stop`, `seed`, `stream_options.include_usage`, prompt as string / list / token ids. |
| `POST /v1/responses` | kept, fixed | SDK + unit. `text.format` (`json_schema`, `json_object`) now maps to constrained decoding (it was ignored). `instructions`, input items, `previous_response_id` / `store`, `function_call` and `function_call_output`, `text.format` json_schema, `reasoning`, streaming event types (created, in_progress, output_item, content_part, output_text delta/done, completed). Usage details always present. Server-side tools run inside the generation loop: `web_search` (`web_search_call` items, `url_citation` annotations, `filters.allowed_domains`, `user_location`) and `{type: "mcp"}` (`mcp_list_tools`, `mcp_call`, `mcp_approval_request` / `mcp_approval_response`, `allowed_tools`, `require_approval`), see [Server-side tools](#server-side-tools). `generate: false` (Codex's WebSocket prewarm) prefills the prompt and returns a chainable empty response; `include: ["reasoning.encrypted_content"]` returns reasoning items that come back as `reasoning_content`; `namespace` / `custom` tools, `developer` messages and unknown input item types are accepted; `input_file` / `input_image` file ids resolve from the local Files store. The Response echoes the request configuration (`instructions`, `temperature`, `top_p`, `max_output_tokens`, `tools`, `tool_choice`, `text`, `reasoning`, `truncation`, `store`, `service_tier`, `max_tool_calls`, ...) in the body, the stored copy and every `response.*` event. `truncation: "auto"` drops the oldest input items when the prompt overflows the context (`"disabled"`, the default, answers 400); `max_tool_calls` caps the function calls of a response; `include`, `prompt_cache_key`, `safety_identifier` are accepted and echoed only. `x_yunshu` is in `usage`. |
| `GET/DELETE /v1/responses/{id}`, `POST /v1/responses/{id}/cancel` | kept | unit + SDK. |
| `POST /v1/responses/input_tokens` | added | Counts the input tokens a request would use. |
| `POST /v1/embeddings` | kept | SDK + unit. `dimensions`, `encoding_format=base64`, token-id input, L2-normalised, empty input 400. Multimodal (`Qwen3-VL-Embedding`) as an extension. |
| `GET /v1/models`, `GET /v1/models/{id}` | kept, extended | SDK (openai + anthropic, real server). Works for every modality. Each item is the model's card in every dialect: OpenAI spec, Anthropic `ModelInfo`, OpenRouter, vLLM, LM Studio, plus the full card under `yunshu`. See [Model cards](#model-cards). |
| `POST /v1/audio/transcriptions`, `/v1/audio/translations` | kept | SDK. Translation needs a Whisper model (other ASR models answer 501 with the reason). |
| `POST /v1/audio/speech`, `/v1/audio/speech/stream` | kept | SDK. WAV out. |
| `GET /v1/audio/voices` | kept | Used by `yunshu voices`. |
| `POST /v1/images/generations` (+ `/stream`), `/v1/images/edits`, `/v1/images/variations` | kept | SDK for generations; edits and variations are unit-tested. |
| `POST /v1/ocr` | kept, extension | GLM-OCR. Not an OpenAI route; kept because `yunshu ocr` and the release gate use it. |
| `WS /v1/realtime` | kept | OpenAI Realtime, GA schema (what `client.realtime.connect()` speaks) or beta with `OpenAI-Beta: realtime=v1`. Checked with the official `openai` SDK (`scripts/dev/realtime_conformance.py`); differences listed in [TRANSPORTS.md](TRANSPORTS.md). |
| `WS /realtime` | kept | Legacy path, beta schema. |
| `POST /v1/realtime/calls` (WebRTC) | planned | Not implemented: needs `aiortc`; plan in [TRANSPORTS.md](TRANSPORTS.md). |
| `WS /v1/responses` | kept | OpenAI Responses WebSocket mode (`client.responses.connect()`): `response.create` in, raw `response.*` events out, `stream_id` lanes. |
| `WS /v1/stream` | kept, extension | Yunshu protocol: many chat.completions / completions / responses / messages requests on one socket, cancel / stop / max_tokens update by id, heartbeats, backpressure. Anthropic has no official WebSocket mode. |
| Unix socket (`yunshu serve --uds PATH`) | kept | Same app; `curl --unix-socket`, httpx `uds=`. |
| HTTP/2 (h2c) | not offered | uvicorn is HTTP/1.1 only; see [TRANSPORTS.md](TRANSPORTS.md). |
| `POST/GET /v1/files`, `GET/DELETE /v1/files/{id}`, `GET /v1/files/{id}/content` | added | Local file store (`YUNSHU_FILES_DIR`, 512 MB per file). OpenAI shape, or the Anthropic Files shape when the request carries `anthropic-version`. unit + OpenAI SDK. |
| `POST/GET /v1/batches`, `GET /v1/batches/{id}`, `POST /v1/batches/{id}/cancel` | added | JSONL input file, one background worker calling this server over loopback, resumes after a restart; output and error files. unit + OpenAI SDK. |
| `/v1/fine_tuning`, `/v1/moderations`, `/v1/assistants`, `/v1/vector_stores`, `/v1/uploads` | not applicable | Hosted-platform features with no local-engine meaning. |

### Chat completions: parameters and fields

| Item | Status |
|---|---|
| `messages` (system, developer, user, assistant, tool; content parts; `image_url`) | implemented (image needs a VLM; audio input on omni models) |
| `temperature`, `top_p`, `max_tokens`, `max_completion_tokens`, `n`, `seed`, `stop`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `user` | implemented |
| `logprobs`, `top_logprobs` (also streamed) | implemented |
| `tools`, `tool_choice` (`auto`, `none`, `required`, named function), `parallel_tool_calls`, tool result messages | implemented; streamed `tool_calls` deltas |
| `response_format` `json_object`, `json_schema` (strict) | implemented (constrained decoding) |
| `stream`, `stream_options.include_usage` | implemented; the usage chunk has empty `choices` |
| `reasoning_effort`, `enable_thinking` | implemented; `reasoning_content` carries the reasoning |
| `store`, `metadata`, `service_tier`, `modalities` (text), `prediction` | accepted and ignored (no meaning locally) |
| `audio` output modality | not applicable here (use `/v1/audio/speech` or Realtime) |
| Finish reasons `stop`, `length`, `tool_calls` | implemented |
| Errors `{error:{message,type,param,code}}`, 400 / 401 / 404 / 429 / 500 / 503 | implemented, also for streaming errors before the first chunk and for malformed JSON |

## Anthropic

| Route | Status | Notes |
|---|---|---|
| `POST /v1/messages` (and `/messages`) | kept, fixed | SDK + unit. `system` (string or blocks with `cache_control`), `tools`, `tool_choice` (`auto`, `any`, `tool`, `none`), `thinking` (`budget_tokens` must be < `max_tokens`), `stop_sequences` (`stop_reason: stop_sequence` and the matched string, streaming and not), `metadata`, `top_k`, `tool_use` / `tool_result` / `image` (base64; a 400 on a text-only model, like OpenAI `image_url`) / `document` (text source) blocks. Streaming: `message_start`, `content_block_start/delta/stop` (`text_delta`, `thinking_delta`, `input_json_delta`), `message_delta`, `message_stop`. Usage includes `cache_read_input_tokens` and `cache_creation_input_tokens`. |
| `POST /v1/messages/count_tokens` | kept | SDK + unit. Counts system, messages, tools, images. |
| `POST /v1/messages` with `web_search_20250305`, `web_fetch_20250910` | added | Server tools, run inside the generation loop: `server_tool_use` + `web_search_tool_result` / `web_fetch_tool_result` blocks, text with `citations` (`web_search_result_location`), `max_uses`, `allowed_domains` / `blocked_domains`, `user_location`, `usage.server_tool_use`, `pause_turn` at the iteration cap. Search needs a configured provider, otherwise `web_search_tool_result_error` `unavailable` with an `x_yunshu` hint. SDK (`messages`, `beta.messages`) + unit + Claude Code end to end. See [Server-side tools](#server-side-tools). |
| `POST /v1/messages` with `mcp_servers` (beta `mcp-client`) and `mcp_toolset` | added | The gateway connects to the named MCP servers (streamable HTTP or legacy SSE) and runs their tools: `mcp_tool_use` / `mcp_tool_result` blocks, `authorization_token`, `tool_configuration.allowed_tools`, per-tool enable. SDK + unit. |
| `thinking: {type: "adaptive"}`, `output_config.effort`, `context_management` | accepted | Claude Code sends all three (the first two used to be a 400 / ignored). `adaptive` leaves the model's template default, `output_config.effort` becomes the template's `reasoning_effort`; earlier `thinking` blocks return as `reasoning_content`. |
| `POST/GET /v1/messages/batches`, `GET .../{id}`, `.../{id}/results`, `.../{id}/cancel`, `DELETE .../{id}` | added | Message Batches over the same loopback worker as `/v1/batches`; results JSONL. unit. |
| `POST/GET /v1/files` (Anthropic Files, beta `files-api-2025-04-14`) | added | Same paths as OpenAI Files, dispatched on `anthropic-version`; `document` / `image` blocks with `source: {type: "file"}` are inlined before generation. |
| `GET /v1/models`, `GET /v1/models/{id}` | kept, extended | Same route as OpenAI. `ModelInfo` is complete: `display_name`, `created_at`, `max_input_tokens`, `max_tokens` and the `capabilities` object (`thinking`, `effort.{low,medium,high,xhigh,max}`, `image_input`, `structured_outputs`, ...). Verified with `anthropic.models.list/retrieve`. |
| Errors `{type:"error", error:{type,message}}` | implemented | `invalid_request_error`, `authentication_error`, `permission_error`, `not_found_error`, `request_too_large`, `rate_limit_error`, `api_error`, `overloaded_error`. |
| `x-api-key`, `anthropic-version`, `anthropic-beta` headers | accepted | `x-api-key` is honoured as the bearer token when `YUNSHU_AUTH_TOKEN` is set. |
| Code execution, computer use, memory and other hosted server tools | not applicable | Hosted services without a local meaning. |

## Server-side tools

The two APIs let the model call tools the *server* runs. Yunshu runs them inside the generation loop, off the
MLX thread: generate, the model calls a server tool, the tool runs (async, with timeouts), the result is
appended, generation continues. Every round re-renders the same conversation plus the new turn, so the prefix
cache serves the shared prefix and a continuation prefills only the new tokens (per-round
`input_tokens` / `cache_read_input_tokens` are in `x_yunshu.server_tools.round_usage`).

| Tool | Anthropic | OpenAI Responses |
|---|---|---|
| Web search | `web_search_20250305` | `web_search`, `web_search_preview` |
| Web fetch | `web_fetch_20250910` | not part of the API |
| MCP connector | `mcp_servers` + `mcp_toolset` | `{type: "mcp"}` |

Off unless configured (settings, see [AGENT_COMPAT.md](AGENT_COMPAT.md#server-side-tools)): a self-hosted SearXNG,
Brave, Tavily or Exa for search; `web_fetch` needs no provider and blocks private, loopback and link-local
addresses (also after redirects and DNS resolution). With no provider a search request gets the API's own error
shape and `x_yunshu.server_tools` carries the hint that says how to configure one; `GET /v1/models` shows the
state under `yunshu.server_tools`. The streamed blocks follow the specs; the differences that remain:
`page_age` is whatever the provider returns, `encrypted_content` is an opaque envelope of the result text (a
local server has nothing to hide from its own client), and citations come from the `[n]` markers the model writes
next to the results it used.

## Model cards

`/v1/models` returns more than an id. Every field is derived from the checkpoint files (`config.json`, the chat
template, `generation_config.json`, safetensors headers, `model_index.json`) or from the engine's own routing tables
(`spec_select` for speculative decoding); nothing is guessed from the name except the same embedding / reranker name
rule the engine's own detector uses. Code: `python/yunshu_engine/model_card.py` (card), `python/yunshu_gateway/model_card_formats.py`
(wire formats), `python/yunshu_gateway/model_cards.py` (registry lookup).

One item carries every dialect, since the field names do not collide:

| Dialect | Fields |
|---|---|
| OpenAI | `id`, `object`, `created`, `owned_by` |
| Anthropic | `type: "model"`, `display_name`, `created_at`, `max_input_tokens`, `max_tokens`, `capabilities` (object form; flat LM Studio / vLLM booleans `vision`, `trained_for_tool_use`, `tools`, `function_calling`, `reasoning`, `embedding` ride inside it) |
| OpenRouter | `name`, `canonical_slug`, `description`, `context_length`, `architecture.{modality,input_modalities,output_modalities,tokenizer}`, `pricing` (all `"0"`), `top_provider.{context_length,max_completion_tokens}`, `supported_parameters`, `default_parameters` |
| vLLM | `root`, `parent`, `max_model_len`, `task` (`generate`, `embed`, `score`, `transcription`, `speech`, `image_generation`, `ocr`) |
| LM Studio | `arch`, `quantization` (`4bit`, `mixed-4/5bit`), `state` (`loaded` / `not-loaded`), `max_context_length`, `publisher`, `compatibility_type: "mlx"`, `model_type` |
| Ollama (`/api/show`) | `capabilities`, `model_info`, `details` (see above) |
| Yunshu | everything else, nested under `yunshu` so strict parsers ignore it |

The `yunshu` block (the ModelCard):

| Field | Meaning |
|---|---|
| `kind` | `chat`, `vlm`, `omni`, `embedding`, `reranker`, `asr`, `tts`, `sts`, `image`, `ocr`, `video` |
| `family`, `architecture`, `parameters` | config `model_type`, `architectures[0]`, parameter count from the safetensors headers (quantized words unpacked at each layer's own bit width, scales skipped) |
| `quantization` | `bits`, `group_size`, `mode`, `layer_groups` (`{bits: layers}` for mixed-precision checkpoints), `skip_components` (diffusion) |
| `input_modalities`, `output_modalities` | `text`, `image`, `video`, `audio`, `embedding`, `score` |
| `context` | `length`, `native`, `effective`, `source`, `rope_scaling` (from `max_position_embeddings`; the engine adds no tighter cap) |
| `max_output_tokens` | `min(131072, context)`, the largest `max_tokens` the chat routes accept |
| `reasoning` | `supported`, `toggle` (`enable_thinking`), `default_enabled`, `effort_levels` and `default_effort` (parsed from the chat template: Qwen3.8 is `xhigh` / `medium` / `low`), `effort_aliases` (OpenAI `high` maps to `xhigh`, `minimal` to `low`), `budget_field`, `output_field` |
| `tools`, `structured_output`, `logprobs` | tool support and `tool_choice` values, `json_object` / `json_schema` / `regex` / `grammar` / `choice`, `max_top_logprobs` |
| `embeddings` | `dimensions`, `pooling`, `normalized` |
| `audio`, `image` | ASR languages, Whisper translation and window; TTS languages, voices, `voice_design`; diffusion pipeline and components |
| `speculative` | from `spec_select`: `method` (`mtp`, `dflash2`), `mtp_head`, `drafter`, `block_size`, `lossless` |
| `prefix_cache` | `supported`, `kind` (`apc`), `hybrid_checkpoints`, `media_keyed`; false for sliding-window models |
| `state`, `memory` | `loaded`, `loading`, `pinned`, `error`; `weights_bytes` and `estimated_bytes` |
| `api` | `endpoints`, `formats` (`chat_completions`, `responses`, `messages`, `completions`), `ollama` |
| `supported_parameters` | request fields the chat routes accept for this model (checked against `ChatCompletionRequest` by a test) |
| `generation_defaults` | `generation_config.json` sampling defaults |

Anonymous callers never see filesystem paths; authenticated callers also get `loaded`, `size_gb`, `stats` and `yunshu.path`.
`reasoning_effort` values outside the template's own list are mapped (`high` becomes `xhigh`) instead of failing the template.

**Yunxin.** Its `vllm` adapter reads `max_model_len`, `task`, `capabilities`; its `lmstudio` adapter reads `max_context_length`,
`type` / `capabilities` (`vision`, `trained_for_tool_use`, `reasoning`); its `openrouter` adapter reads `context_length`,
`architecture.*_modalities`, `top_provider.max_completion_tokens`. All are present, so a Yunshu server registered under any of
the three needs no custom code.

### Capability contract (`yunshu.contract`)

Every card carries a `contract` object, built by `yunshu_engine/capability_contract.py` from the checkpoint, and
`POST /v1/chat/completions` enforces the same object:

| Contract key | States |
|---|---|
| `tools` | `supported`, `mode` (`template` when the chat template renders tools, else `prompt` injection), `parallel`, `tool_choice` values |
| `structured_output` | `json_object`, `json_schema.engines` (`in-house`, plus `llguidance` for schemas outside the in-house subset), `regex`, `choice`, `grammar` (Lark via llguidance) |
| `logprobs`, `reasoning` | support, `max_top_logprobs`, effort levels |
| `media` | accepted input and produced output modalities beyond text |
| `speculative` | `mode` (`mtp`, `dflash2`, `none`) and `lossless` |
| `cache_tiers` | `ram` (`kv_prefix` or `apc`) and `ssd` when that tier is enabled |
| `context` | window and maximum output tokens |

A request that uses what the contract rules out returns 400 naming the field: `reasoning_effort` / `thinking_budget` on a model
without a reasoning mode, an `image` / `audio` / `video` content part the model does not accept, and tools, `response_format`,
guided decoding or `logprobs` on a model that does not generate text. A checkpoint whose config cannot be read has no contract
and is not gated.

## Ollama-compatible (`/api/*`)

A thin translation layer over the OpenAI routes (loopback), verified with the `ollama` Python SDK.
Streaming is NDJSON. Auth follows the app-wide token.

| Route | Status | Notes |
|---|---|---|
| `POST /api/chat` | implemented | `messages` with `images`, `tools` (arguments are objects), `tool` role, `format` (`"json"` or a schema; on a VLM thinking defaults to off when `format` is set, because the constraint masks the output from the first token), `think`, `options` (`temperature`, `top_p`, `top_k`, `min_p`, `seed`, `num_predict`, `stop`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`; `num_ctx` is accepted and ignored; `keep_alive` and `X-Request-Id` are forwarded), `total_duration`, `load_duration`, `prompt_eval_count/duration`, `eval_count/duration` (ns, from the engine stats) in the final chunk. |
| `POST /api/generate` | implemented | Verified on the VLM runner too (with `think` the reasoning arrives in `thinking`). Errors from the OpenAI routes pass through: images on a text-only model and chat on an embedding model are 400s. `prompt`, `system`, `images`, `format`, `options`; an empty prompt answers `done_reason: load`. `raw`, `suffix`, `template`, `context` are not supported. |
| `POST /api/embed`, `POST /api/embeddings` | implemented | |
| `GET /api/tags`, `GET /api/ps`, `POST /api/show`, `GET /api/version` | implemented | Built from the model card: `size` is the weight bytes, `details` has family / parameter size / quantization, `show` returns `capabilities` (`completion`, `tools`, `vision`, `thinking`, `embedding`) and `model_info` (`general.architecture`, `general.parameter_count`, `<arch>.context_length`, `<arch>.embedding_length`). `show` 404s for an unknown model. |
| `POST /api/pull`, `/api/push`, `/api/create`, `/api/copy`, `DELETE /api/delete` | not applicable | 501 with a message: models are managed with `yunshu pull` / `yunshu model`. |

## Tokenizer (vLLM schema)

| Route | Status | Notes |
|---|---|---|
| `POST /tokenize` and `/v1/tokenize` | kept, rewritten | Body: `prompt` (string or list) **or** `messages` (chat template; `add_generation_prompt`, `continue_final_message`, `tools`, `chat_template_kwargs`), plus `add_special_tokens`, `return_token_strs`, `model` (optional). Returns `count`, `max_model_len`, `tokens`, `token_strs`. `text` and `input` are accepted as aliases of `prompt`. |
| `POST /detokenize` and `/v1/detokenize` | kept, rewritten | `{tokens}` returns `{prompt}`. |
| `POST /v1/messages/count_tokens` | kept | Anthropic. |

## Retrieval (vLLM-style)

| Route | Status | Notes |
|---|---|---|
| `POST /v1/score`, `/v1/rerank`, `/v1/pooling`, `/v1/classify` | kept | Unit + SDK-less curl on an embedding model and a text LLM. On a model that cannot embed (hybrid architectures) they answer a 400 that says so, not a 500. |

## Other

| Route | Status | Notes |
|---|---|---|
| `GET /health`, `/health/live`, `/health/ready`, `GET /version` | kept | |
| `GET /metrics` | kept, moved | Prometheus. Was `/api/v1/gw/monitoring/prometheus` behind a deny-by-default check; now always mounted and covered by the app-wide token. |
| `GET /debug/*` (`engine`, `system`, `models`, `requests`, `kv-cache`, `spec-decode`, `memory-guard`, `ssd-cache`, `per-model`, `prometheus`, `all`) | kept, off by default | Mounted only with `YUNSHU_DEBUG_ROUTES=1`; they need the token (or `YUNSHU_AUTH_DISABLED`). `yunshu status` uses them when enabled. |
| `POST /v1/cancel`, `GET /v1/active-generations` | kept | Cancel an in-flight generation by request id. |
| `POST /v1/models/load`, `/v1/models/unload/{id}` | kept | Multi-model mode; token required. |
| `POST /v1/mcp`, `GET /v1/mcp/sse`, `/v1/mcp/tools`, `/v1/mcp/client/*` | kept | JSON-RPC; `tools/list` answers. |
| `POST /v1/omni/speech/stream` | kept | Qwen3-Omni Thinker to Talker over SSE. |

## Yunshu extensions

Additive and namespaced; the SDKs above ignore all of it. Design and rationale:
[API_EXTENSIONS.md](API_EXTENSIONS.md). Checked by `tests/unit/test_api_extensions.py` and
`scripts/realmodel/smoke_api_extensions.py` (OpenAI, Anthropic and Ollama SDKs against a real model).

| Route / field | Status | Notes |
|---|---|---|
| `X-Request-Id` request header, echoed on every response (errors too) | added | Accepted when 1-128 chars of `[A-Za-z0-9._:-]`, else generated (`req_...`). |
| `GET /v1/yunshu/status` | added | Version, state, uptime, models (loaded, keep-alive, expires), memory, active requests by phase, recent throughput, the last finished request (`last`: cache hit, TTFT, decode speed; `yunshu statusline` shows it in Claude Code). |
| `GET /v1/requests`, `GET /v1/requests/{id}` | added | Live phase (`queued` / `prefill` with tokens, %, ETA / `decode`) of in-flight requests; poll it for a non-streaming long prefill. |
| `DELETE /v1/requests/{id}` | added | Cancel by the client's `X-Request-Id` (or the completion id). `POST /v1/cancel {request_id}` accepts the same ids. |
| `POST /v1/yunshu/warmup` | added | Load the model, run a 1-token generation, optionally prefill `prompt` / `messages` into the prefix cache; takes `keep_alive`. |
| `keep_alive` on `/v1/chat/completions` and `/v1/completions` | added | Ollama semantics (`"5m"`, `300`, `-1`, `0`); multi-model mode frees the model after that idle time. Single-model mode never frees its model. |
| `: yunshu-progress {...}` SSE comment (streaming chat / completions) | added | Every `YUNSHU_PROGRESS_INTERVAL_S` (default 2 s) before the first token: phase, prompt / processed tokens, %, tokens/s, ETA, queue position. |
| `x_yunshu` object: chat / completions JSON body, streaming usage chunk (or `: yunshu-stats` comment without `include_usage`); inside `usage` on Messages and Responses | added | TTFT, queue wait, prefill / decode tokens/s, cached tokens, speculative mode and acceptance, llama.cpp-style `timings`. `null` when unknown. |
| `X-Yunshu-*` response headers | added | `Queue-Position`, `Queue-Est-Wait-Ms` (all streaming and non-streaming generation responses); `TTFT-Ms`, `Prefill-Tps`, `Decode-Tps`, `Cached-Tokens`, `Queue-Wait-Ms`, `Total-Ms`, `Spec`, `Spec-Acceptance` (non-streaming chat / completions). |
| `error.x_yunshu.hint` / `.request_id`; `(hint: ...)` appended to `error.message` (OpenAI-format errors) | added | The fix, e.g. context too long, 401, 429, model loading. |

## Removed

| Route | Why |
|---|---|
| `POST /v1/video/generations` | Backend missing (`mlx_video`); the route answered 503 on every real setup. |
| `POST /v1/batch`, `/v1/batch/*` | A custom format that is neither the OpenAI Batch API nor used by anything. |
| `/sleep`, `/wake-up`, `/v1/sleep`, `/v1/wake-up`, `/sleep/status` | Management surface; nothing needs a sleeping server. |
| `/v1/start_profile`, `/v1/stop_profile`, `/v1/profile/*` | Debug tooling; use Instruments or the bench scripts. |
| `/api/v1/bench/*` (roofline, latency, throughput, bfcl-eval, model, batch) | In-server benchmarks; the CLI (`yunshu bench`) and `scripts/` cover them. |
| `/api/v1/gw/monitoring/*` dashboards for dead subsystems (`radix-tree`, `auto-tuner`, `request-coalescer`, `token-scheduler`, `attention-eviction`, `inflight-prefix-sharing`, `health-dashboard`, `batch-size`, `ane-embeddings`, `memory-pressure`, `response-cache`, `thinking-segments`, `reasoning-tokens`, `prefill-progress`) | Reported state of subsystems that no longer exist or never served anything. |
| `/v1/cachedContents/*` and the `cached_content` field | Gemini-style handle API; the automatic prefix cache and Anthropic `cache_control` do the same job. |
| `/v1/images/inpaint`, `/controlnet`, `/depth-guided` | Not OpenAI routes; half-built. |
| `/v1/audio/speech-to-speech/*`, `/v1/audio/voice-pipeline` | Not OpenAI routes; Realtime covers voice. |
| `/v1/token_count` | Replaced by `/tokenize` (`count`) and Anthropic `count_tokens`. |
| `scripts/audit_closure.py` | A self-grading endpoint tracker. |

The engine code only those routes used (video engine and pipeline, ControlNet block engine, STS engine) is deleted too. Loading a
video or speech-to-speech checkpoint fails with a clear message; image generation, including the real
Z-Image ControlNet through `control_image`, is unchanged.

## CLI

| Command | Status | Notes |
|---|---|---|
| `serve`, `chat`, `pull`, `doctor`, `config` (`set`, `unset`, `path`), `service` (`install`, `uninstall`, `start`, `stop`, `restart`, `status`, `logs`), `model` (`list`, `info`, `load`, `unload`, `download`) | kept | Smoke-tested; `model load/unload` need the token on the server. |
| `launch claude` / `codex` / `opencode` / `pi` (`--dry-run`, `--effort`) | extended | Reads the model card and configures the agent with the real context window, output limit, reasoning levels and vision support; see [AGENT_COMPAT.md](AGENT_COMPAT.md#launching-an-agent). |
| `status`, `diagnose gpu`, `diagnose server`, `launch list` | kept | `diagnose gpu` and `bench roofline` printed thousands of TFLOPS because the lazy matmuls were never evaluated; fixed. |
| `complete`, `embed`, `tokenize`, `detokenize`, `rerank`, `score`, `classify`, `transcribe`, `speak`, `ocr`, `image`, `image-edit`, `image-variations`, `voices`, `cancel` | kept | Talk to a running server. |
| `bench roofline`, `latency`, `throughput`, `memory`, `inference`, `eval` | kept | |
| `image-inpaint`, `image-controlnet`, `image-depth`, `video`, `audio-enhance`, `audio-separate`, `audio-transform`, `voice-pipeline` | removed | Their routes are gone. |

## Known gaps

| Item | State |
|---|---|
| Unknown `model` in single-model mode | Served, not 404 (deliberate, see the top). |
| Tool calling on tiny models with thinking off | Qwen3.5-0.8B without thinking emits malformed `<tool_call>` markup; the 27B and thinking-on paths pass the release gate. |
