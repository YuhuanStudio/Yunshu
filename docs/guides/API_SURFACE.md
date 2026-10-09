# API surface

Every public route, what it is checked against, how it is verified, and what was removed. The OpenAI and Anthropic
references are the specification; a route stays only if it matches one (or a de-facto local-server
API). **The Verified column of each row is the evidence, and a row claims no more than it says.**

| Tag | Meaning |
|---|---|
| `real <date>` | A real `yunshu serve` process answered, on the M3 lane (`scripts/dev/m3sweep`). `routes` is `scripts/research/route_checks.py` (official `openai` / `anthropic` SDKs and their typed models wherever they cover the route; raw HTTP, `websockets` and a fake SearXNG / MCP backend elsewhere), `wire` is `scripts/research/m3sweep_jobs.py wire` (the SDK wire matrix). Models: Qwen3.5-0.8B-MLX-bf16 (VLM runner), Qwen2.5-3B-Instruct-4bit (mlx-lm path), `wire` also Qwen3.5-9B-MLX-4bit, and the two behind a token-protected `--models-dir` server. The M3 gives correctness evidence only, no timing. |
| `unit` | A test in `tests/unit` with a scripted engine, a fake or `TestClient`: **mock only**, no model ran. |
| `audit` | A real-model smoke run during the earlier audit (Qwen3-Embedding-0.6B, Qwen3-ASR-1.7B, whisper-large-v3-mlx, Qwen3-TTS, Z-Image-Turbo, GLM-OCR, Qwen3-Omni). No automated job repeats it: it carries no date and is not evidence for today's code. |

The served path of every modality is verified on its own checkpoint (Qwen3-TTS, Qwen3-ASR, GLM-OCR, Z-Image-Turbo, Qwen3-Embedding; allowlisted on the M3). A check that only sees an absent-capability or error answer (a chat model asked for speech) is an error-path check and never counts as coverage: `route_checks.py` declares `served=True/False` per check and the gate needs one served check per route. Exempt (reviewed list): `POST /v1/omni/speech/stream` (Qwen3-Omni, omnismall line), `POST /v1/audio/translations` (Whisper-only), and Ollama registry uploads (`/api/push`, documented 501). Realtime is served here by a text turn on a chat model; voice in / out belongs to the omnismall line.

**Route coverage gate.** `tests/unit/test_route_coverage.py` lists every route the gateway registers (websockets included) and fails when one
has no SERVED check in `scripts/research/route_checks.py` (or a reasoned entry in its `EXEMPT` table: reviewed by the lead; the gate counts only checks that declare `served=True`). `scripts/dev/m3sweep` (jobs
`routes`, `wire`, `agent`, `units`) fails when a registered route was not verified by a passing check in the same run. Add a route, add a check.

`model` is advisory in single-model mode (`yunshu serve -m`): the loaded model answers under any name
and the response echoes the requested name, so existing clients need no reconfiguration. In
multi-model mode (`--models-dir`) an unknown model is a 404 `model_not_found`.

## OpenAI

| Route | Status | Notes | Verified |
|---|---|---|---|
| `POST /v1/chat/completions` | kept, fixed | `stop` accepts a string or a list. `usage.prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens` are always present. Context overflow is 400 `context_length_exceeded`. Images on a text-only model are a 400. An embedding-only model (sentence-transformers export) answers a 400 that points to `/v1/embeddings`. | real 2026-10-06: `routes` + `wire` (SDK, typed) on 0.8B, 3B and (wire) 9B-4bit; image input on 0.8B, 400 on 3B; `input_audio` / `audio_url`, image + audio and `video_url` on gemma-4-e2b-it-4bit (M5, answers reflect the spoken word / the colour; stream and not). Parameters: see the table below |
| `GET /v1/chat/completions`, `GET/POST/DELETE /v1/chat/completions/{id}`, `GET /v1/chat/completions/{id}/messages` | added | Stored chat completions. `store: true` (with optional `metadata`, at most 16 string pairs) keeps the finished completion and its input messages in `YUNSHU_CHAT_COMPLETIONS_DIR` (default `~/.yunshu/chat_completions`, one JSON file each, the oldest evicted past `YUNSHU_CHAT_COMPLETIONS_MAX`, default 5000); a stream is stored once it ends with `[DONE]`, folded into one `chat.completion`. List takes `after` / `limit` (1-100) / `order` / `model` / `metadata[k]=v`; `messages` returns the request messages with ids and `content_parts`; `POST` updates `metadata`; `DELETE` returns `chat.completion.deleted`. Unknown id is 404. Requests without `store` are never kept. | unit: the real `openai` SDK against a TestClient (retrieve, update, delete, list filters / pagination / order, messages with image parts, stream folding, eviction, 400s). real 2026-10-08 (M5, 0.8B): `routes` check `chat_stored_completions` passed (gpuq job `1008-075611-00-apiplanned-routes-1`: store, retrieve equals the returned completion, stream stored equals streamed text, list by metadata, messages, update, delete, 404) |
| `POST /v1/completions` | kept | `echo`, `logprobs` (int; also on the VLM runner, streamed and not), `n`, `stop`, `seed`, `stream_options.include_usage`, prompt as string / list / token ids. | real 2026-10-06: `routes` + `wire` (SDK; basic, truncation, stop, usage). `echo`, `logprobs`, `n`, `seed`, token-id prompts: unit |
| `POST /v1/responses` | kept, fixed | `text.format` (`json_schema`, `json_object`) now maps to constrained decoding (it was ignored). `instructions`, input items, `previous_response_id` / `store`, `function_call` and `function_call_output`, `text.format` json_schema, `reasoning`, streaming event types (created, in_progress, output_item, content_part, output_text delta/done, completed). Usage details always present. Server-side tools run inside the generation loop: `web_search` (`web_search_call` items, `url_citation` annotations, `filters.allowed_domains`, `user_location`) and `{type: "mcp"}` (`mcp_list_tools`, `mcp_call`, `mcp_approval_request` / `mcp_approval_response`, `allowed_tools`, `require_approval`), see [Server-side tools](#server-side-tools). `generate: false` (Codex's WebSocket prewarm) prefills the prompt and returns a chainable empty response; `include: ["reasoning.encrypted_content"]` returns reasoning items that come back as `reasoning_content`; `namespace` tools are flattened; freeform `custom` tools map to a native input function and return `custom_tool_call` items / `response.custom_tool_call_input` events; custom text, regex and Lark formats are supported (see Agent-client additions below); unsupported combinations return a clear 400. `developer` messages and unknown input item types are accepted; `input_file` / `input_image` file ids resolve from the local Files store. The Response echoes the request configuration (`instructions`, `temperature`, `top_p`, `max_output_tokens`, `tools`, `tool_choice`, `text`, `reasoning`, `truncation`, `store`, `service_tier`, `max_tool_calls`, ...) in the body, the stored copy and every `response.*` event. `truncation: "auto"` drops the oldest input items when the prompt overflows the context (`"disabled"`, the default, answers 400); `max_tool_calls` caps the function calls of a response; `include`, `prompt_cache_key`, `safety_identifier` are accepted and echoed only. `x_yunshu` is in `usage`. | real 2026-10-06: `routes` + `wire` (create, stream, tools, `json_schema`, `previous_response_id`, conversation, background, input image, `web_search` and MCP against a fake backend, WS). `generate: false`, `include`, `truncation`, `max_tool_calls`, `namespace` / `custom` tools, `developer` messages: unit |
| `GET/DELETE /v1/responses/{id}`, `POST /v1/responses/{id}/cancel` | kept, fixed | Cancelling a running background response answers `status: "cancelled"` (it answered the `in_progress` snapshot) and a poll agrees; cancelling a finished response returns it unchanged. | real 2026-10-06: `routes` lifecycle (create `store` -> retrieve -> input_items -> chain -> background cancel -> delete -> 404) on 0.8B and 3B |
| `GET /v1/responses/{id}/input_items` | added | The stored response's own input as Responses items (`message` with `input_text` / `input_image`, `function_call`, `function_call_output`), `limit` 1-100, `order` (`desc` default), `after` / `before` cursors. `instructions` and system / developer messages are not listed (like the hosted API); history through `previous_response_id` belongs to the earlier responses. The route was missing (the SDK's `input_items.list` got a 404). | real 2026-10-06: `routes` (SDK `input_items.list`) |
| `POST /v1/responses/compact` | kept | Server-side compaction: returns a `compaction` item the next request takes as input. | real 2026-10-06: `routes` (SDK `responses.compact`, then the output used as input) |
| `POST /v1/responses/input_tokens` | added, fixed | The input tokens a request would use. It counted the raw text (no chat template, no tools, no history): 14 against a usage of 28. It now builds the prompt like generation (instructions, input items, conversation, compaction items, `previous_response_id` chain, function tools rendered natively when the template does it, injected otherwise) and counts the rendered template. Not counted: server-side tool definitions, the assistant prefill of a forced `tool_choice` on an engine without native tools, the compaction a `context_management` threshold would trigger. | real 2026-10-06: `routes`, equal to the real call's `usage.input_tokens` for plain, tools, `previous_response_id` and conversation requests (0.8B, 3B) |
| `POST/GET/DELETE /v1/conversations`, `GET /v1/conversations/{id}`, `POST /v1/conversations/{id}`, `POST/GET /v1/conversations/{id}/items`, `GET/DELETE /v1/conversations/{id}/items/{item_id}` | added | Local conversation store (`YUNSHU_CONVERSATIONS_DIR`); a Responses request with `conversation` prepends its items and appends the turn. | real 2026-10-06: `routes` (SDK create / retrieve / update / items create, list, retrieve, delete / delete, and a response appended to it) |
| `POST /v1/embeddings` | kept | `dimensions`, `encoding_format=base64`, token-id input, L2-normalised, empty input 400. Multimodal (`Qwen3-VL-Embedding`, EmbeddingGemma 2: text / image / audio / video) as an extension: object items, vLLM-style `messages`, `task`, `instruction` (docs/API.md). | real 2026-10-06: `routes` served on Qwen3-Embedding-0.6B (SDK typed): 1024 dimensions, unit length, similar > unrelated, a batch equals the singles, `dimensions=256`, base64, empty input 400. Multimodal (`Qwen3-VL-Embedding`): unit + audit. Also answers on chat models used as embedders (not a served-path check) |
| `GET /v1/models`, `GET /v1/models/{id}` | kept, extended | Works for every modality. Each item is the model's card in every dialect: OpenAI spec, Anthropic `ModelInfo`, OpenRouter, vLLM, LM Studio, plus the full card under `yunshu`. See [Model cards](#model-cards). | real 2026-10-06: `routes` (`openai` and `anthropic` SDKs, list + retrieve) on 0.8B, 3B and the multi-model server |
| `POST /v1/audio/transcriptions`, `/v1/audio/translations` | kept | Translation needs a Whisper model (other ASR models answer 501 with the reason). | real 2026-10-06: `routes` served on Qwen3-ASR-1.7B-bf16: the Qwen3-TTS output transcribed back and matched to the input text (json, `verbose_json`, text, srt, vtt), a non-audio upload is a 400. Translations: only Whisper-family models translate; Qwen3-ASR answers the documented 501 (error path). real 2026-10-06 whisper-large-v3-mlx (M5; the M3 run waits for the allowlist to reach the daemon): Chinese speech made by Qwen3-TTS ("今天天气很好，我们一起去公园散步吧。") came back as "The weather is very good today, let's go for a walk in the park." (json, SDK, `response_format=text`) |
| `POST /v1/audio/speech`, `/v1/audio/speech/stream` | kept | WAV out. | real 2026-10-06: `routes` served on Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16: valid 24 kHz WAV of plausible duration and non-silent (one-shot, SDK, and the SSE stream: header, audio chunks, done); an `mp3` request on a machine without ffmpeg answers a WAV labelled `audio/wav` with `X-Yunshu-Audio-Format-Fallback` |
| `GET /v1/audio/voices` | kept | Used by `yunshu voices`. | real 2026-10-06: `routes` on the TTS server |
| `POST /v1/images/generations` (+ `/stream`), `/v1/images/edits`, `/v1/images/variations` | kept | `edits` and `variations` take multipart/form-data like the OpenAI SDK sends (JSON with a base64 `image` also works); they accepted only JSON, so `client.images.edit(...)` was a 400. | real 2026-10-06: `routes` served on Z-Image-Turbo-MLX-4bit: generations (+ `/stream`), edits and variations through the SDK return a valid 256x256 PNG (2 steps), the same seed gives the same image, a bad size is a 400 |
| `POST /v1/ocr` | kept, extension | GLM-OCR. Not an OpenAI route; kept because `yunshu ocr` and the release gate use it. | real 2026-10-06: `routes` served on GLM-OCR-bf16: a rendered text image returns its text, a non-image upload is a 400; Qwen3.5-0.8B reads an image through the VLM fallback, Qwen2.5-3B answers 503 naming the fix |
| `WS /v1/realtime` | kept | OpenAI Realtime, GA schema (what `client.realtime.connect()` speaks) or beta with `OpenAI-Beta: realtime=v1`. Differences from the hosted API are listed in [TRANSPORTS.md](TRANSPORTS.md); `scripts/dev/realtime_conformance.py` is the SDK conformance script. | real 2026-10-06: `routes` raw-socket text turn (`session.created` -> `response.done` with usage) and a disconnect mid-response on 0.8B and 3B. Voice turn, real 2026-10-06 (M5): cascade on gemma-4-e2b-it-4bit + Qwen3-ASR-1.7B + Qwen3-TTS (spoken "The secret word is pineapple." in; `input_audio_transcription.completed` carries the same words; 150 KB of speech out) and native speech on Qwen3-Omni-30B-A3B-4bit. The SDK conformance script: unit (fake engine) |
| `WS /realtime` | kept | Legacy path, beta schema. | real 2026-10-06: same checks as `/v1/realtime` (`routes`); voice turn on the beta schema (`modalities`) on the same cascade (gemma-4-e2b-it-4bit + ASR + TTS) and on Qwen3-Omni-30B-A3B-4bit (M5) |
| `POST /v1/realtime/client_secrets`, `POST /v1/realtime/sessions`, `POST /v1/realtime/transcription_sessions` | added | Ephemeral keys for browser / device clients. The caller authenticates normally and gets an `ek_...` value (default 600 s, `expires_after.seconds` 10-7200 else 400) with the effective session (GA `realtime` or `transcription`, or the beta shapes). The `/v1/realtime` socket accepts the secret as its bearer token until it expires, starts with that session configuration and the secret's session id, and a secret can open several sockets. In memory only; a restart drops every secret. Transcription configuration is echoed but not acted on (the engine has no transcription-only session); a transcription secret cannot create model responses. Expiry prevents new connections, not already accepted sessions. | unit: the real `openai` SDK (`realtime.client_secrets.create`, `beta.realtime.sessions` / `transcription_sessions`) and a TestClient websocket with a static token set (secret accepted and applied, unknown or expired refused). real 2026-10-08 (M5, 0.8B): `routes` check `realtime_client_secrets` passed (same job: an `ek_` secret opens `/v1/realtime`, session id and instructions applied, a text turn ends in `response.done`, beta sessions, TTL 1 is 400) |
| `POST /v1/realtime/calls` (WebRTC) | optional extra | `yunshu[webrtc]`: SDP offer/answer, GA data channel, PCM/RTP audio. No public ICE relay. Missing extra: 503 and WebSocket alternative. See [TRANSPORTS.md](TRANSPORTS.md). | unit: SDK + two-peer audio tests; M5 respfeat probe pending |
| `WS /v1/responses` | kept | OpenAI Responses WebSocket mode (`client.responses.connect()`): `response.create` in, raw `response.*` events out, `stream_id` lanes. | real 2026-10-06: `routes` raw socket (two chained turns, malformed message, disconnect mid-generation) + SDK `client.responses.connect()` on 0.8B and 3B; unauthenticated upgrade refused (multi-model server with a token) |
| `WS /v1/stream` | kept, extension | Yunshu protocol: many chat.completions / completions / responses / messages requests on one socket, cancel / stop / max_tokens update by id, heartbeats, backpressure. Anthropic has no official WebSocket mode. | real 2026-10-06: `routes` (all four APIs on one socket, cancel by id, malformed message, disconnect mid-generation) on 0.8B and 3B |
| Unix socket (`yunshu serve --uds PATH`) | kept | Same app; `curl --unix-socket`, httpx `uds=`. | unit + audit |
| HTTP/2 (h2c) | not offered | uvicorn is HTTP/1.1 only; see [TRANSPORTS.md](TRANSPORTS.md). | not offered |
| `POST/GET /v1/files`, `GET/DELETE /v1/files/{id}`, `GET /v1/files/{id}/content` | added | Local file store (`YUNSHU_FILES_DIR`, 512 MB per file). OpenAI shape, or the Anthropic Files shape when the request carries `anthropic-version`. unit + OpenAI SDK. | real 2026-10-06: `routes` lifecycle (SDK upload, list, retrieve, content, delete, 404 after delete; Anthropic Files shape: upload, list, metadata, download of an uploaded file refused 403 like the hosted API, `document` block by file id) |
| `POST/GET /v1/batches`, `GET /v1/batches/{id}`, `POST /v1/batches/{id}/cancel` | added | JSONL input file, one background worker calling this server over loopback, resumes after a restart; output and error files. unit + OpenAI SDK. | real 2026-10-06: `routes` (SDK: file -> create -> poll -> output file rows -> cancel an in-flight batch -> cancelled; bad input file 4xx) |
| `/v1/fine_tuning`, `/v1/moderations`, `/v1/assistants`, `/v1/vector_stores`, `/v1/uploads` | not applicable | Hosted-platform features with no local-engine meaning. | not applicable |

### Chat completions: parameters and fields

| Item | Status | Verified |
|---|---|---|
| `messages` (system, developer, user, assistant, tool; content parts; `image_url`) | implemented (image needs a VLM; audio input on omni models) | real 2026-10-06: `wire` (system, user, assistant, tool messages) and `routes` (`image_url` on 0.8B; image block on gemma-4-e2b-it-4bit answers "blue"). Messages has no audio block (Anthropic API). `developer` role: unit |
| `temperature`, `top_p`, `max_tokens`, `max_completion_tokens`, `n`, `seed`, `stop`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `user` | implemented | real 2026-10-06: `max_tokens` / `max_completion_tokens` (`wire` truncation). The sampling fields, `n`, `seed`, `stop` lists beyond single characters, `logit_bias`, `user`: unit |
| `logprobs`, `top_logprobs` (also streamed) | implemented | unit (scripted engine); audit on the VLM runner |
| `tools`, `tool_choice` (`auto`, `none`, `required`, named function), `parallel_tool_calls`, tool result messages | implemented; streamed `tool_calls` deltas | real 2026-10-06: `wire` (`auto`, `none`, `required`, named, `parallel_tool_calls: false`; stream and not) on 0.8B, 3B, 9B-4bit |
| `response_format` `json_object`, `json_schema` (strict) | implemented (constrained decoding) | real 2026-10-06: `wire` (`json_schema`, stream and not; chat, Responses, Ollama). `json_object`: unit |
| `stream`, `stream_options.include_usage` | implemented; the usage chunk has empty `choices` | real 2026-10-06: `wire` + `routes` (usage-only chunk, usage equal to the non-stream call) |
| `reasoning_effort`, `enable_thinking` | implemented; `reasoning_content` carries the reasoning | real 2026-10-06: thinking on Qwen3.5 (`wire` usage `reasoning_tokens`). Effort levels and template mapping: unit |
| `store`, `metadata`, `service_tier`, `modalities` (text), `prediction` | accepted and ignored (no meaning locally) | unit |
| `audio` output modality | not applicable here (use `/v1/audio/speech` or Realtime) | not applicable |
| Finish reasons `stop`, `length`, `tool_calls` | implemented | real 2026-10-06: `wire` (`stop`, `length`, `tool_calls`; stop sequences on Messages) |
| Errors `{error:{message,type,param,code}}`, 400 / 401 / 404 / 429 / 500 / 503 | implemented, also for streaming errors before the first chunk and for malformed JSON | real 2026-10-06: `wire` error requests (all four dialects, status and shape) and every `routes` error answer; 401 / 404 / 400 seen. 429 / 500 / 503 shapes: unit |

## Anthropic

| Route | Status | Notes | Verified |
|---|---|---|---|
| `POST /v1/messages` (and `/messages`) | kept, fixed | `system` (string or blocks with `cache_control`), `tools`, `tool_choice` (`auto`, `any`, `tool`, `none`), `thinking` (`budget_tokens` must be < `max_tokens`), `stop_sequences` (`stop_reason: stop_sequence` and the matched string, streaming and not), `metadata`, `top_k`, `tool_use` / `tool_result` / `image` (base64; a 400 on a text-only model, like OpenAI `image_url`) / `document` (text source) blocks. Streaming: `message_start`, `content_block_start/delta/stop` (`text_delta`, `thinking_delta`, `input_json_delta`), `message_delta`, `message_stop`. Usage includes `cache_read_input_tokens` and `cache_creation_input_tokens`. | real 2026-10-06: `routes` + `wire` (SDK; stream and not, tools, `tool_choice`, stop sequences, image block on 0.8B, 400 on 3B, the `/messages` alias). `thinking.budget_tokens`, `cache_control` breakpoints, `document` text source, `top_k`, `metadata`: unit |
| `POST /v1/messages/count_tokens` | kept | Counts system, messages, tools, images. Tools are rendered the way generation renders them (natively when the chat template does it), so the count equals the call's usage (it counted an injected prompt generation no longer uses: 101 against 272). | real 2026-10-06: `routes` (SDK): equal to the real call's usage without and with tools; alias path; 400 without messages |
| `POST /v1/messages` with `web_search_20250305`, `web_fetch_20250910` | added | Server tools, run inside the generation loop: `server_tool_use` + `web_search_tool_result` / `web_fetch_tool_result` blocks, text with `citations` (`web_search_result_location`), `max_uses`, `allowed_domains` / `blocked_domains`, `user_location`, `usage.server_tool_use`, `pause_turn` at the iteration cap. Search defaults to best-effort DDG/Wikipedia; when disabled, `web_search_tool_result_error` `unavailable` with an `x_yunshu` hint. SDK (`messages`, `beta.messages`) + unit + Claude Code end to end. See [Server-side tools](#server-side-tools). | real 2026-10-06: `routes` against a fake SearXNG (forced `web_search`: `server_tool_use` + `web_search_tool_result`, `max_uses`, stream assembled by the SDK, `usage.server_tool_use`); no provider configured (the `unavailable` error blocks); `web_fetch` of a loopback page refused with `web_fetch_tool_result_error`. Citations, `allowed_domains`, `pause_turn`, a real search provider and a real page fetch: unit |
| `POST /v1/messages` with `mcp_servers` (beta `mcp-client`) and `mcp_toolset` | added | The gateway connects to the named MCP servers (streamable HTTP or legacy SSE) and runs their tools: `mcp_tool_use` / `mcp_tool_result` blocks, `authorization_token`, `tool_configuration.allowed_tools`, per-tool enable. | real 2026-10-06: `routes` against a fake MCP server (initialize, tools/list reach it; tool call when the model makes one). `authorization_token`, `allowed_tools`, per-tool enable: unit |
| `thinking: {type: "adaptive"}`, `output_config.effort`, `context_management` | accepted | Claude Code sends all three (the first two used to be a 400 / ignored). `adaptive` leaves the model's template default, `output_config.effort` becomes the template's `reasoning_effort`; earlier `thinking` blocks return as `reasoning_content`. | unit; Claude Code 2.1.285 / 2.1.291 sessions in `agentcompat` (see AGENT_COMPAT.md) |
| `POST/GET /v1/messages/batches`, `GET .../{id}`, `.../{id}/results`, `.../{id}/cancel`, `DELETE .../{id}` | added | Message Batches over the same loopback worker as `/v1/batches`; results JSONL. unit. | real 2026-10-06: `routes` (SDK: create -> poll -> results -> cancel an in-flight batch -> delete -> 404) |
| `POST/GET /v1/files` (Anthropic Files, beta `files-api-2025-04-14`) | added | Same paths as OpenAI Files, dispatched on `anthropic-version`; `document` / `image` blocks with `source: {type: "file"}` are inlined before generation. | real 2026-10-06: `routes` (SDK `beta.files`: upload, list, metadata, delete; a `document` block by file id is inlined) |
| `GET /v1/models`, `GET /v1/models/{id}` | kept, extended | Same route as OpenAI. `ModelInfo` is complete: `display_name`, `created_at`, `max_input_tokens`, `max_tokens` and the `capabilities` object (`thinking`, `effort.{low,medium,high,xhigh,max}`, `image_input`, `structured_outputs`, ...). Verified with `anthropic.models.list/retrieve`. | real 2026-10-06: `routes` (`openai` and `anthropic` SDKs, list + retrieve) on 0.8B, 3B and the multi-model server |
| Errors `{type:"error", error:{type,message}}` | implemented | `invalid_request_error`, `authentication_error`, `permission_error`, `not_found_error`, `request_too_large`, `rate_limit_error`, `api_error`, `overloaded_error`. | real 2026-10-06: `wire` Anthropic error requests and every Anthropic-dialect `routes` error (batches, files, count_tokens, image on a text model); all seen as `{type: error, error: {type, message}}` |
| `x-api-key`, `anthropic-version`, `anthropic-beta` headers | accepted | `x-api-key` is honoured as the bearer token when `YUNSHU_AUTH_TOKEN` is set. | real 2026-10-06: `routes` (token as `x-api-key` on the multi-model server) |
| Code execution, computer use, memory and other hosted server tools | not applicable | Hosted services without a local meaning. | not applicable |

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

Zero-config best-effort DDG/Wikipedia; queries leave the machine. Configured providers take precedence (settings, see [AGENT_COMPAT.md](AGENT_COMPAT.md#server-side-tools)): a self-hosted SearXNG,
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
| `kind` | `chat`, `vlm`, `omni`, `embedding`, `reranker`, `classifier`, `decision`, `asr`, `tts`, `sts`, `image`, `ocr`, `video` |
| `family`, `architecture`, `parameters` | config `model_type`, `architectures[0]`, parameter count from the safetensors headers (quantized words unpacked at each layer's own bit width, scales skipped) |
| `quantization` | `bits`, `group_size`, `mode`, `layer_groups` (`{bits: layers}` for mixed-precision checkpoints), `skip_components` (diffusion) |
| `input_modalities`, `output_modalities` | `text`, `image`, `video`, `audio`, `embedding`, `score`, `decision` |
| `context` | `length`, `native`, `effective`, `source`, `rope_scaling` (derived from the model config; trained scoring models also report a `serving_cap`) |
| `max_output_tokens` | `min(131072, context)` for `chat`, `vlm` and `omni` cards (131072 when context is unknown); this advertised card limit is separate from the route validation cap of 1,048,576 |
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

Anonymous callers never see filesystem paths; authenticated callers also get `loaded`, `size_gb` (binary GB, 1024^3, like macOS; exact `size_bytes` beside it), `stats` and `yunshu.path`.
`reasoning_effort` values outside the template's own list are mapped (`high` becomes `xhigh`) instead of failing the template.

**Yunxin.** Its `vllm` adapter reads `max_model_len`, `task`, `capabilities`; its `lmstudio` adapter reads `max_context_length`,
`type` / `capabilities` (`vision`, `trained_for_tool_use`, `reasoning`); its `openrouter` adapter reads `context_length`,
`architecture.*_modalities`, `top_provider.max_completion_tokens`. All are present, so a Yunshu server registered under any of
the three needs no custom code.

### Capability contract (`yunshu.contract`)

Every card carries a `contract` object, built by `yunshu_engine/capability_contract.py` from the checkpoint, and
every generation route enforces the same object (`/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/responses`;
`/api/chat` and `/api/generate` through the chat route they call), each answering in its own error shape (OpenAI `{"error": {...}}`,
Anthropic `{"type": "error", ...}`, Ollama `{"error": "..."}`):

| Contract key | States |
|---|---|
| `tools` | `supported`, `mode` (`template` when the chat template renders tools, else `prompt` injection), `parallel`, `tool_choice` values |
| `structured_output` | `json_object`, `json_schema.engines` (`in-house`, plus `llguidance` for schemas outside the in-house subset), `regex`, `choice`, `grammar` (Lark via llguidance) |
| `logprobs`, `reasoning` | support, `max_top_logprobs`, effort levels |
| `media` | accepted input and produced output modalities beyond text |
| `speculative` | `mode` (`mtp`, `dflash2`, `none`) and `lossless` |
| `cache_tiers` | `ram` (`kv_prefix` or `apc`) and `ssd` when that tier is enabled |
| `context` | window and maximum output tokens |

A request that uses what the contract rules out returns 400 naming the field: an `image` / `audio` / `video` content part the model does not accept
(also inside an Anthropic `tool_result` or a Responses `function_call_output`), and tools, `response_format`,
guided decoding or `logprobs` on a model that does not generate text. `reasoning_effort` / `thinking_budget` on a model without
a reasoning mode are accepted and have no effect (coding agents send an effort on every request; the card's `reasoning` field
says whether it applies). A checkpoint whose config cannot be read has no contract and is not gated.

## Ollama-compatible (`/api/*`)

A thin translation layer over the OpenAI routes (loopback). The `ollama` SDK is not a dependency: the real checks use raw HTTP and NDJSON.
Streaming is NDJSON. Auth follows the app-wide token.

| Route | Status | Notes | Verified |
|---|---|---|---|
| `POST /api/chat` | implemented | `messages` with `images`, `tools` (arguments are objects), `tool` role, `format` (`"json"` or a schema; on a VLM thinking defaults to off when `format` is set, because the constraint masks the output from the first token), `think`, `options` (`temperature`, `top_p`, `top_k`, `min_p`, `seed`, `num_predict`, `stop`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`; `num_ctx` is accepted and ignored; `keep_alive` and `X-Request-Id` are forwarded), `total_duration`, `load_duration`, `prompt_eval_count/duration`, `eval_count/duration` (ns, from the engine stats) in the final chunk. | real 2026-10-06: `routes` + `wire` (raw HTTP: Ollama has no SDK dependency here; non-stream, NDJSON stream, `images` on 0.8B). `think`, `format` schema (wire), `keep_alive`, the options list: unit |
| `POST /api/generate` | implemented | Verified on the VLM runner too (with `think` the reasoning arrives in `thinking`). Errors from the OpenAI routes pass through: images on a text-only model and chat on an embedding model are 400s. `prompt`, `system`, `images`, `format`, `options`; an empty prompt answers `done_reason: load`. `raw`, `suffix`, `template`, `context` are not supported. | real 2026-10-06: `routes` + `wire` (non-stream and NDJSON) |
| `POST /api/embed`, `POST /api/embeddings` | implemented | | real 2026-10-06: `routes` served on Qwen3-Embedding-0.6B (`/api/embed` equals `/v1/embeddings`, similar > unrelated; `/api/embeddings`). `show`, `tags`, `ps`, `version`: chat models |
| `GET /api/tags`, `GET /api/ps`, `POST /api/show`, `GET /api/version` | implemented | Built from the model card: `size` is the weight bytes, `details` has family / parameter size / quantization, `show` returns `capabilities` (`completion`, `tools`, `vision`, `thinking`, `embedding`) and `model_info` (`general.architecture`, `general.parameter_count`, `<arch>.context_length`, `<arch>.embedding_length`). `show` 404s for an unknown model. | real 2026-10-06: `routes` (tags, ps, version, `show` with `capabilities` and `details`) |
| `POST /api/pull`, `/api/create`, `/api/copy`, `DELETE /api/delete` | implemented for native models | Multi-model mode, administrative token; native HF safetensors pull, persistent names sharing weights, `create {model, from}`, deletion inside the configured model directory. Unsupported create options are 400. See the native management contract below. | real 2026-10-07: M3 `routes` 12-check API sweep (`3e365a31`, `agentapi-smoke-routes-3e365a31-1007`), existing-model pull / alias lifecycle / copied model generation; new network downloads: unit only |
| `POST /api/push` | not applicable | Ollama registry uploads are unsupported (501). | real 2026-10-06: `routes` error-path check, exempt from the served gate |

## Tokenizer (vLLM schema)

| Route | Status | Notes | Verified |
|---|---|---|---|
| `POST /tokenize` and `/v1/tokenize` | kept, rewritten | Body: `prompt` (string or list) **or** `messages` (chat template; `add_generation_prompt`, `continue_final_message`, `tools`, `chat_template_kwargs`), plus `add_special_tokens`, `return_token_strs`, `model` (optional). Returns `count`, `max_model_len`, `tokens`, `token_strs`. `text` and `input` are accepted as aliases of `prompt`. | real 2026-10-06: `routes` (both paths, round trip with detokenize; `messages` count equal to the chat usage's prompt tokens; 400 without input) |
| `POST /detokenize` and `/v1/detokenize` | kept, rewritten | `{tokens}` returns `{prompt}`. | real 2026-10-06: `routes` (round trip on text with CJK) |
| `POST /v1/messages/count_tokens` | kept | Anthropic. | real 2026-10-06: `routes` (see Anthropic) |

## Retrieval (vLLM-style)

| Route | Status | Notes | Verified |
|---|---|---|---|
| `POST /v1/score`, `/v1/rerank`, `/v1/pooling`, `/v1/classify` | kept | On a model that cannot embed (hybrid architectures, Qwen3.5) all four answer a 400 that says so (`classify` answered a 500). | real 2026-10-06: `routes` served on Qwen3-Embedding-0.6B (embedding-similarity scoring, not a trained reranker): pooling width, score similar > unrelated, rerank puts the relevant document first, classify picks the right label and sums to 1. On chat models: Qwen2.5-3B answers 200, Qwen3.5-0.8B the 400 "cannot be used as a text embedder" (error path) |

## Decisions (OpenAI Decisions API, TypeSafe System One)

| Route | Status | Notes | Verified |
|---|---|---|---|
| `POST /v1/decisions` | added | OpenAI's Decisions API (public beta 2026-10-06), wire format from the `openai` 3.26 types (`Decision`, `DecisionCreateParams`). `input` is a string or user messages with `input_text` and inline `input_image` (base64 data URLs only; at most 128). `questions` are `predicate`, `choice` (typed `value`: a string `"true"` and a boolean `true` collide, so they are a 400) and `score` (`levels`); the response lists `answers` in question order with probabilities, a `refusal` for a question whose logits are not finite (never a made-up value), and `usage` (`output_tokens` 0, nothing is generated). Stateless, non-streaming. Served by a decision checkpoint (`ModelType.DECISION`): Cloudflare Clef / Clef-flash in MLX format (backbone through mlx-vlm plus the joint schema head, one forward pass). A chat model, or a checkpoint whose head is not the Clef joint schema head, is not served: a 400 / a load error, never a plain LLM that drops the head. Images need the checkpoint's vision tower, else 400. An input over 16384 tokens (state + images + schema) is a 400, not truncated. `safety_identifier` is accepted (64 characters) and ignored. | unit: fake engine, request validation, error shape, and the real `openai` 3.26 client (`client.decisions.create` -> typed `Decision`); the MLX head against a torch transcription of the reference. real 2026-10-08 (M5, `scripts/research/decisions_verify.py`, gpuq job `1008-002058-00-decisions-verify-clef4-c`): `yunshu serve` on abenzerps/Clef-MLX 4-bit (detected as DECISION, loaded in 6 s), the `openai` 3.26 client: a rainy text gives rain 0.92 / dry 0.02, a complaint routes to `support` and scores 0.35 against 3.91 for a glowing review, a boolean choice stays boolean, reordering choices or repeating a request is bit-identical, base64 red and blue images are answered red and blue, `/v1/chat/completions` on the decision model is a 400; about 0.23 s for one question and 0.53 s for five on a 235-token input (informal, not a quiet-CPU timing) |
| `POST /v1/systemone` | added | TypeSafe Jev / System One wire on the same engine and one shared internal request: `{model, state (string, object or array), questions: {id: {type: noul / choice / score, instructions, criteria}}, images?}` returning `{model, answers: {id: ...}, usage: {input_tokens, output_tokens}}` with `noul`, `choice` + `confidence` + `probabilities` and `score` + `legend`, rounded to 4 places like the reference. Validation errors are 422 (as TypeSafe documents). | unit: as above. real 2026-10-08: same job, `/v1/systemone` routes the same complaint to `support` (noul 0.79, score 0.55) |

## Other

| Route | Status | Notes | Verified |
|---|---|---|---|
| `GET /health`, `/health/live`, `/health/ready`, `GET /version` | kept | | real 2026-10-06: `routes` (no token needed; `ready` true; the multi-model server with nothing loaded is ready, was 503) |
| `GET /metrics` | kept, moved | Prometheus. Was `/api/v1/gw/monitoring/prometheus` behind a deny-by-default check; now always mounted and covered by the app-wide token. | real 2026-10-06: `routes` (Prometheus text) |
| `GET /debug/*` (`engine`, `system`, `models`, `requests`, `kv-cache`, `spec-decode`, `memory-guard`, `ssd-cache`, `per-model`, `prometheus`, `all`) | kept, off by default | Mounted only with `YUNSHU_DEBUG_ROUTES=1`; they need the token (or `YUNSHU_AUTH_DISABLED`). `yunshu status` uses them when enabled. | unit (off by default) |
| `POST /v1/cancel`, `GET /v1/active-generations` | kept | Cancel an in-flight generation by request id. | real 2026-10-06: `routes` (a real stream cancelled by `X-Request-Id`, ends early, disappears from the list) |
| `POST /v1/models/load`, `/v1/models/unload/{id}` | kept | Multi-model mode; token required. | real 2026-10-06: `routes` on a token-protected `--models-dir` server (0.8B + 3B): load, chat, unload, unload again, unknown model 404, empty 400, reload by chat; refused without a token (401) |
| `POST /v1/mcp`, `GET /v1/mcp/sse`, `/v1/mcp/tools`, `/v1/mcp/client/*` | kept | JSON-RPC; `tools/list` answers. | real 2026-10-06: `routes` (initialize, tools/list, unknown method is a JSON-RPC error, notification, `GET /v1/mcp/tools` equals tools/list, SSE endpoint event, client status and tools) |
| `POST /v1/omni/speech/stream` | kept | Qwen3-Omni Thinker to Talker over SSE. | real 2026-10-06: Qwen3-Omni-30B-A3B-Instruct-4bit (M5, the only small-enough-to-test speech-out checkpoint Yunshu serves; 20 GB, never on the M3): text in, audio-in ("pineapple" answered) and image-in ("Red") each stream `text` + 24 kHz `audio` + `done` + `[DONE]`; unknown speaker is a 400 |

## Yunshu extensions

Additive and namespaced; the SDKs above ignore all of it. Design and rationale:
[API_EXTENSIONS.md](API_EXTENSIONS.md). Checked by `tests/unit/test_api_extensions.py` and
`scripts/realmodel/smoke_api_extensions.py` (an earlier real-model run, not part of `m3sweep`).

| Route / field | Status | Notes | Verified |
|---|---|---|---|
| `X-Request-Id` request header, echoed on every response (errors too) | added | Accepted when 1-128 chars of `[A-Za-z0-9._:-]`, else generated (`req_...`). | real 2026-10-06: used by the `routes` cancel check (the id addresses the live request) |
| `GET /v1/yunshu/status` | added | Version, state, uptime, models (loaded, keep-alive, expires), memory, active requests by phase, recent throughput, the last finished request (`last`: cache hit, TTFT, decode speed; `yunshu statusline` shows it in Claude Code). | real 2026-10-06: `routes` |
| `GET /v1/requests`, `GET /v1/requests/{id}` | added | Live phase (`queued` / `prefill` with tokens, %, ETA / `decode`) of in-flight requests; poll it for a non-streaming long prefill. | real 2026-10-06: `routes` (live request found by id while streaming, 404 afterwards) |
| `GET /v1/yunshu/requests/recent?limit=&model=&since=` | added | The finished-request ring (512), newest first: path, model, finish reason, queue wait, TTFT, token counts, prefill / decode speed, cache tier and reload time, speculative stats, cancelled, and `offsets_ms` (arrive, admit, first token, last token, done, in ms after arrival; null for a phase not reached) with `t0_wall`. Numbers and enums only, never prompt text. The same `t0_wall` / `offsets_ms` are in `x_yunshu` and the live progress object. | unit-tested; real-server check `console_data` added, not yet run |
| `GET /v1/yunshu/history?since=&step=` | added | Columnar samples (`t` plus one array per field: active / cache / footprint GB, pressure, active and queued requests, decode / prefill tok/s, TTFT p50 / p95, prompt / completion tok/s) from a preallocated in-memory ring, 12 h at 5 s (`YUNSHU_HISTORY_HOURS`, `YUNSHU_HISTORY_INTERVAL_S`; 0.46 MiB fixed). `step` averages into buckets. Not persisted; `enabled: false` when the sampler is off. | unit-tested; real-server check `console_data` added, not yet run |
| `GET /v1/yunshu/memory` | added | Unified-memory ledger: `owners[]` (weights per loaded model, prefix cache RAM and warm tier, MLX cache, live KV, residual `other`) with `bytes`, `reclaimable`, `estimated`, `source`; MLX active / cache / peak / recommended working set, process footprint, host pressure level / swap / wired limit, limits, free. Every figure from a counter; unknown is null (live KV is null until it is counted). | unit-tested; real-server check `console_data` added, not yet run |
| `GET /v1/yunshu/config?include=stable\|all` | added | Effective settings (`yunshu config` as JSON): value, default, source (cli / env / file / default), category, stability, type, choices, description (240 chars), warnings, experimental count / max. Secrets are `***`. Each row also has `applies` (`live` / `reload` / `restart`), `minimum` and `secret`. | unit-tested; real-server check `console_data` added, not yet run |
| `PATCH /v1/yunshu/config` | added | Body `{settings: {NAME: value or null}, confirm_experimental?, dry_run?}`. Validates type, choices and minimum (422 `errors` per name, nothing written on any error); experimental / internal names need `confirm_experimental` (400). Saves to the config file in use (else `~/.yunshu/config.toml`); `null` resets. Per setting: `status` = `applied` (live) / `needs_reload` / `needs_restart` / `overridden` (env or `--set` wins at runtime), plus `restart_required`, `restart` hint. Secrets are write-only (no value echoed). `YUNSHU_CONFIG` is read-only here. Needs the `admin` permission. | unit-tested; no real-server check (mutates the operator's config) |
| `GET /v1/yunshu/service` | added | launchd agent: `label`, `plist`, `installed`, `loaded`, `pid`, `state`, `last_exit_code`, `log`, `under_launchd` (this process is the agent's pid), `version`, `uptime_s`, CLI equivalents. `admin`. | unit-tested (launchctl mocked) |
| `POST /v1/yunshu/service/restart` | added | Body `{confirm: true}`. Only when this server runs under the `yunshu service` agent: answers 202, waits for in-flight requests up to `YUNSHU_DRAIN_TIMEOUT`, then `launchctl kickstart -k`. Otherwise 409 `not_under_launchd` with the manual command. `admin`. CLI: `yunshu service restart`. | unit-tested (launchctl mocked) |
| `GET /v1/yunshu/cors`, `PATCH /v1/yunshu/cors` | added | Allowed origins (`origins`, `wildcard`, `credentials`, `source`, `warnings`, `request_origin_allowed`). PATCH `{origins, allow_any_origin?}`: URL origins only (`scheme://host[:port]`, no path), at most 50, `null` resets; `*` needs `allow_any_origin` and is never combined with other origins or credentials, and warns loudly. Applies live (the CORS middleware re-reads `YUNSHU_CORS_ORIGINS`); an env value still wins. Bind address / LAN exposure stay read-only. `admin`. | unit-tested |
| `DELETE /v1/requests/{id}` | added | Cancel by the client's `X-Request-Id` (or the completion id). `POST /v1/cancel {request_id}` accepts the same ids. | real 2026-10-06: `routes` (cancels a live stream) |
| `POST /v1/yunshu/warmup` | added | Load the model, run a 1-token generation, optionally prefill `prompt` / `messages` into the prefix cache; takes `keep_alive`. | real 2026-10-06: `routes` |
| `GET /v1/yunshu/keys` | added | Stored API keys (admin scope): `id`, `name`, `prefix` (first characters, display only), `created`, `last_used`, `enabled`, `scopes` (`infer`, `admin`; admin implies infer), `expires` (epoch or null), `quotas` (`requests_per_day`, `tokens_per_day`, `max_concurrent`; null = unlimited) and the live rolling-24h `window` (requests, tokens, inflight). Never the secret or hash. | unit-tested; real-server check `api_keys` added, not yet run |
| `POST /v1/yunshu/keys` | added | Create a key from `{name, scopes?, quotas?, expires?}`. 201 with the key object plus `secret` (`ysk-...`), shown once; only its SHA-256 is stored in `~/.yunshu/keys.json` (0600). Any stored key turns auth on, like `YUNSHU_AUTH_TOKEN`, which stays a full admin key. | unit-tested; real-server check `api_keys` added, not yet run |
| `PATCH /v1/yunshu/keys/{key_id}` | added | Change `name`, `enabled`, `scopes`, `quotas` (merged), `expires`. 400 on unknown fields, 404 on an unknown id. | unit-tested; real-server check `api_keys` added, not yet run |
| `DELETE /v1/yunshu/keys/{key_id}` | added | Remove a key and its usage. | unit-tested; real-server check `api_keys` added, not yet run |
| `POST /v1/yunshu/keys/{key_id}/rotate` | added | New secret for the same id (returned once); the old one stops working at once. | unit-tested; real-server check `api_keys` added, not yet run |
| `GET /v1/yunshu/usage?key=&since=&group=day\|key\|total` | added | Per-key usage by UTC day: `requests`, `prompt_tokens`, `completion_tokens`, `cached_tokens`, `errors` (status >= 400). `since` is `YYYY-MM-DD` or `7d`. Tokens are counted once, from the finished-request record. Admin scope. | unit-tested; real-server check `api_keys` added, not yet run |
| `POST /v1/yunshu/downloads` | added | Body `{repo, revision?, allow_patterns?}` (`org/name`). Lists the repo's files first, refuses with 507 `{message, needed_bytes, free_bytes, path}` when the disk cannot hold the remainder (plus 5%, at least 256 MB), 502 when the hub cannot be read, then queues the download into `<models dir>/<org>/<name>` (the `yunshu pull` layout) and answers 202 with the job. One download runs at a time, the rest queue; the same request while one is active returns that job; a model already complete on disk answers a finished job with `already_present`. Resume: the hub's `.incomplete` files stay after a cancel or failure, so posting again continues. On success the folder is checked (weights complete) and registered with the model manager. `admin`. | unit-tested with a fake hub; no real-server check (downloads from the network) |
| `GET /v1/yunshu/downloads`, `GET /v1/yunshu/downloads/{id}` | added | Job: `id`, `repo`, `revision`, `state` (`queued` / `running` / `done` / `failed` / `cancelled`), `error`, `path`, `bytes_total` and `bytes_done` (real bytes from the hub's progress bars), `files_total`, `files_done`, `active_files`, `rate_bps` (last 10 s), `eta_s`, timestamps, `registered`, `already_present`. The list adds `active`, `free_bytes` and `models_dir` and is small (no per-file array, 50 finished jobs kept); `{id}` adds `files[]` (name, bytes). | unit-tested with a fake hub |
| `DELETE /v1/yunshu/downloads/{id}` | added | Cancel a queued or running download (the transfer stops at the next progress tick; partial files stay for resume). Returns the job. `admin`. | unit-tested with a fake hub |
| `POST /api/pull` (Ollama) | extended | Streams by default like Ollama: `pulling manifest`, then `{status: "pulling <id>", digest, total, completed}` lines (one aggregate layer, real bytes), then `success`; failures (also out of disk) are a final `{error}` line. `stream: false` blocks and answers `{status: "success"}`. Backed by the same registry, so the job is visible and cancellable under `/v1/yunshu/downloads`; a client disconnect does not cancel it. | unit-tested with a fake hub |
| `GET /v1/yunshu/models/local?refresh=` | added | Every model on disk, registered or not (models dir and Hugging Face cache): `id`, `path`, `source`, `size_bytes`, `model_type`, `kind`, `architecture`, `family`, `parameters`, `quantization {bits, group_size}`, `context_length`, `capabilities[]` (from the model card), `complete` and `complete_reason`, `registered_as`, `loaded`; plus `total_bytes`, `models_dir`, `free_bytes`. Directory sizes are cached for 15 s (`refresh=true` rescans). | unit-tested |
| `POST /v1/yunshu/models/{id}/reload` | added | Admin only. Unloads and loads the same model so settings that `PATCH /v1/yunshu/config` reports as `needs_reload` take effect (multi-model mode; a single-engine server answers 400 and restarts instead). Body `{force?: bool}`: refuses with 409 while the model has running requests or leases, unless `force` (running requests are torn down). Answers `{status: reloaded, model, was_loaded, forced, elapsed_s}`; 404 unknown model, 409 while a load/unload of it is in flight, 500 if the load after the unload fails (the model then stays unloaded). | unit-tested, real-server check |
| `GET /v1/yunshu/models/{id}/fit` | added | Dry run of the load-time memory check (multi-model mode): `weights_bytes`, `kv_reserve_bytes`, `needed_bytes`, `budget_bytes`, `used_bytes`, `free_bytes`, `would_evict[]` (LRU order, only models the load itself may evict), `verdict` = `fits` / `tight` (fits with under 5% of the budget spare, or only after evicting) / `wont_fit`, `reason`, `basis.estimated`. Nothing is mutated; the verdict agrees with `_ensure_memory_available`. | unit-tested against the real manager |
| `GET /v1/yunshu/cache/tiers?entries=` | added | Prefix-cache view per loaded VLM-runner model: `tiers[]` (`ram` / `warm` / `ssd`: `used_bytes`, `cap_bytes`, `entries`, `hits`; SSD adds `hit_bytes`, `read_bps`, `effective_read_bps`, `cost_rejected`, `path`), `lookups {hit, miss, by_tier}` and `entries[]` capped at 200 (`key` = 8-hex hash label, `tokens`, `bytes`, `tier`, `lru_rank` (0 = newest), `hits`, `last_hit_age_s`; never token ids or text). Hit counts are an accounting side table (4,096 boundaries, O(1) per lookup) and never feed a lookup. | unit-tested on the real cache manager |
| `POST /v1/yunshu/cache/tiers/clear` | added | Body `{tier?: ram\|warm\|ssd, model?}` (no body: every tier of every model). Drops the entries (RAM entries are dropped, not demoted), runs on the MLX thread, keeps the counters, returns `cleared[]` and `freed_bytes`. `admin`. | unit-tested on the real cache manager |
| `GET /v1/yunshu/logs?level=&since=&since_id=&q=&limit=` | added | Newest `limit` (max 2,000) of an in-memory ring of 2,000 server log records, oldest first: `records[] {id, t, level, logger, msg}`, `next_id` (cursor for `since_id`), `dropped`, `capacity`. Messages are redacted when emitted (credentials, tokens, echoed request payload fragments), cut at 2 KB, and exceptions show only their type. `admin`. | unit-tested |
| `GET /v1/yunshu/logs/stream?level=&q=&since_id=` | added | The same records as server-sent events (`id:` = record id), from the tail, with a keepalive comment every 15 s. `admin`. | unit-tested on the generator |
| `keep_alive` on `/v1/chat/completions` and `/v1/completions` | added | Ollama semantics (`"5m"`, `300`, `-1`, `0`); multi-model mode frees the model after that idle time. Single-model mode never frees its model. | unit |
| `: yunshu-progress {...}` SSE comment (streaming chat / completions) | added | Every `YUNSHU_PROGRESS_INTERVAL_S` (default 2 s) before the first token: phase, prompt / processed tokens, %, tokens/s, ETA, queue position. | unit |
| `x_yunshu` object: chat / completions JSON body, streaming usage chunk (or `: yunshu-stats` comment without `include_usage`); inside `usage` on Messages and Responses | added | TTFT, queue wait, prefill / decode tokens/s, cached tokens, speculative mode and acceptance, llama.cpp-style `timings`. `null` when unknown. | unit; `wire` reads usage only |
| `X-Yunshu-*` response headers | added | `Queue-Position`, `Queue-Est-Wait-Ms` (all streaming and non-streaming generation responses); `TTFT-Ms`, `Prefill-Tps`, `Decode-Tps`, `Cached-Tokens`, `Queue-Wait-Ms`, `Total-Ms`, `Spec`, `Spec-Acceptance` (non-streaming chat / completions). | unit |
| `error.x_yunshu.hint` / `.request_id`; `(hint: ...)` appended to `error.message` (OpenAI-format errors) | added | The fix, e.g. context too long, 401, 429, model loading. | real 2026-10-06: seen on every error answer of the `routes` checks (hint appended to the message) |
| `X-Yunshu-Deadline-Ms: N` request header (generation routes) | added | Wall time the client will wait, from arrival, queue wait included. Past it the generation is cancelled: `504` `deadline_exceeded` (`timeout_error` on Messages) before the stream starts, one terminal error event after. A malformed value is a 400. | unit |
| `429` `queue_full` / `503` `memory_pressure` with `Retry-After` (generation routes) | added | `YUNSHU_QUEUE_LIMIT` requests in flight, or memory nearly full while others run: refused at once in the route's dialect (`rate_limit_error` / `overloaded_error` on Messages) with `error.x_yunshu.queue_depth`, never queued without bound. | unit |
| `x_yunshu.cancelled` / `X-Yunshu-Cancelled: true` | added | The answer was cut short by `DELETE /v1/requests/{id}`, `POST /v1/cancel` or a deadline, so it is not mistaken for a finished one. | unit |
| `x_yunshu.context_policy` / `X-Yunshu-Context-Policy` | added | When the context-window manager (or Responses `truncation: "auto"`) removed turns: policy, tokens before / after, messages and roles removed, the budget. Streams carry it in the usage chunk / `: yunshu-stats` (headers are already sent). | unit |
| `x_yunshu.budget` | added | `max_tokens` (and `thinking_budget`) clamped to the room the prompt leaves in the context window: requested vs granted. A prompt that fills the window is a 400. | unit |

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

See [CLI guide](CLI.md) for first-run selection, model management, monitoring, JSON and shell completion.

| Command | Status | Notes |
|---|---|---|
| `serve`, `chat`, `pull`, `doctor`, `config` (`set`, `unset`, `path`), `service` (`install`, `uninstall`, `start`, `stop`, `restart`, `status`, `logs`, `rotate-logs`), `cache` (`status`, `gc`), `model` (`list`, `info`, `load`, `unload`, `download`) | kept | Smoke-tested; `model load/unload` need the token on the server. |
| `setup`, `models` (`list`, `pull`, `show`, `rm`), `top`, `completion` (`zsh`, `bash`, `fish`) | added | First-run guidance; global `--json` produces machine-readable results; `top` JSON is one snapshot. |
| `launch claude` / `codex` / `opencode` / `pi` (`--dry-run`, `--effort`) | extended | Reads the model card and configures the agent with the real context window, output limit, reasoning levels and vision support; see [AGENT_COMPAT.md](AGENT_COMPAT.md#launching-an-agent). |
| `status`, `diagnose gpu`, `diagnose server`, `diagnose bundle`, `launch list` | kept | `diagnose gpu` and `bench roofline` printed thousands of TFLOPS because the lazy matmuls were never evaluated; fixed. |
| `complete`, `embed`, `tokenize`, `detokenize`, `rerank`, `score`, `classify`, `transcribe`, `speak`, `ocr`, `image`, `image-edit`, `image-variations`, `voices`, `cancel` | kept | Talk to a running server. |
| `bench roofline`, `latency`, `throughput`, `memory`, `inference`, `eval` | kept | |
| `image-inpaint`, `image-controlnet`, `image-depth`, `video`, `audio-enhance`, `audio-separate`, `audio-transform`, `voice-pipeline` | removed | Their routes are gone. |

### Native model management and retrieval contracts

Ollama `pull`, `copy`, `create` and `delete` use the Ollama body and response shapes in
multi-model mode. `pull` accepts Hugging Face native safetensors repositories (or an
already registered model), with a final `status: success` in JSON or NDJSON. `copy`
and `create {model, from}` persist a symlink name without duplicating weights. `delete`
unlinks that name, or removes checkpoint data inside the configured models directory;
external checkpoint paths are refused. Delete copies before deleting their source.
`create` rejects custom templates, system/messages, parameters, adapters, blobs and
quantization; GGUF conversion and Ollama registry downloads/uploads are unsupported.
These operations use the same administrative authorization as `/v1/models/load`.
`ps` lists only loaded models even when `/v1/models` omits private state fields.
Verification: unit regressions and the `ollama_management` real-server check; a real
run is required before claiming a dated result.

Responses freeform custom tools preserve plain-text `input` outward and in chained history,
with SDK-typed custom delta/done events. The model sees one internal string parameter;
that transport detail is not exposed in the public tool definition. Custom CFG formats
and custom + server-side tool combinations are refused with 400 rather than silently dropped.
Verification: real 2026-10-07 M3 `responses_custom_tool` on 0.8B and 3B,
stream / non-stream + chained tool output (`d26ebaaf`, `agentapi-smoke-custom-routes-1007b`).

Forced tools are checked before streaming headers. A tool grammar that cannot compile
(unsupported marker tokenization or recursive references) returns 400 with
`Cannot guarantee forced tool_choice`; auto retains its existing fallback.
Realtime `?model=` uses the HTTP lazy loader, including aliases, and emits an error
and closes on an unknown model or load failure. It never silently selects a different
model. Verification: unit + `forced_tool_uncompilable` / `realtime_lazy_load` checks.

`score` accepts vLLM's `queries/documents`, `queries/items`, `data_1/data_2` and the
legacy `text_1/text_2` names. A cross-encoder scores each pair jointly and receives
`instruction`; `chat_template_kwargs.instruction` takes precedence. A bi-encoder
ignores scoring instructions, matching [vLLM's score-template contract](https://docs.vllm.ai/en/latest/models/pooling_models/scoring/).
Other score template kwargs are rejected explicitly. `classify` remains Yunshu's
label-similarity extension (`input`, `labels`, temperature), rather than vLLM's
trained classification-head API (`input` or `messages`, no candidate labels,
`data[].probs/num_classes`). No trained classification head is implemented here;
clients must not treat its zero-shot scores as those probabilities.

## SDK coverage walk

`scripts/dev/api_coverage.py` reads the resource modules of the installed `openai` and `anthropic` SDKs (AST only) and lists every
endpoint they can request (575 on openai 3.26.0 / anthropic 1.11.0, websockets included). `tests/unit/test_api_coverage.py` fails when
one is neither served by the gateway nor declared in `scripts/dev/api_coverage_na.json` as `not_applicable` or `planned`, each with a
reason, and when a declaration matches nothing or still calls a served route planned/not applicable. An `implemented` declaration documents completed work but cannot hide a missing route. This replaces building the matrix from the routes
we already had (which is how `POST /v1/decisions` was missed). Current state: 75 implemented, 0 planned, the rest not applicable (OpenAI and Anthropic
skills, which mount into hosted sandboxes, SIP call control, organization and admin APIs, fine-tuning, Assistants/Threads, vector stores, hosted
agent platforms, video, containers, webhooks, ChatKit, Live).
Upgrading an SDK is the trigger: a new endpoint fails the test until someone decides.

## Known gaps

| Item | State |
|---|---|
| Unknown `model` in single-model mode | Served, not 404 (deliberate, see the top). |
| Tool calling on tiny models with thinking off | Qwen3.5-0.8B without thinking emits malformed `<tool_call>` markup; the 27B and thinking-on paths pass the release gate. |

## Wire contract

`tests/unit/test_wire_*.py` drive the routers with the official `openai` and `anthropic` SDKs (Ollama over raw NDJSON, since the
`ollama` SDK is not a dependency) against a **scripted engine (mock)**; `m3sweep` runs the same client adapters (`tests/unit/wire_clients.py`) against a real server (`wire` job), so one generation can be compared across dialects, stream against not.

| Guarantee | Where it is checked |
|---|---|
| Event order and field shapes: chat chunks (role first, one finish chunk, usage-only chunk, `[DONE]`), completions, Messages (`message_start` .. `message_stop`, thinking before text, `tool_use` blocks), Responses (`response.created` .. `response.completed` / `.incomplete`, ascending `sequence_number`, reasoning item before the message), Ollama NDJSON | `test_wire_stream_shapes.py` |
| Tools: single, parallel, `tool_choice` `required` / named / `none`, `parallel_tool_calls=false` / `disable_parallel_tool_use`, JSON schema reaching the engine, on chat, Messages, Responses and Ollama, stream and not | `test_wire_tools_matrix.py` |
| Usage: prompt / completion / reasoning / cached agree between stream and non-stream and across dialects for plain, cached, thinking, truncated, stop-string, zero-output and cut-off-while-thinking generations; details never exceed their totals; Anthropic `message_delta.usage` restates the prompt split (`input_tokens`, `cache_*`) and the cumulative output | `test_wire_usage_matrix.py`, `test_wire_usage_invariants.py` (shapes in `yunshu_gateway/usage_shapes.py`) |
| A failure after the stream headers is one terminal error event on the same response, never a second HTTP response and never a clean-looking finish: chat / completions `data: {"error": ...}` then `[DONE]`; Messages `event: error` (no `message_delta` / `message_stop`); Responses `response.failed`; Ollama `{"error": ...}` line; `/v1/stream` ends with `done(reason=error)`; Realtime sends `error` then `response.done` with `status: failed` and `status_details`. Cancel stops generation on `/v1/stream`, Responses WS and Realtime | `test_wire_errors.py` |
| The capability contract answers every dialect in its own error shape | `test_capability_contract_dialects.py` |

Known, accepted differences: with `tool_choice: none` a model that still emits tool-call markup has it returned as plain text when
streaming; non-streaming chat and Messages strip it. Responses delivers a function call whole (`output_item.added` carries the arguments,
no `function_call_arguments.delta`) and, like its non-stream form, keeps the empty message item before it. A non-stream answer's content
is whitespace-trimmed, a stream's concatenated deltas are not.

## Sampling contract

- **Seeded sampling is position-keyed.** On every VLM-runner path (speculative lane and shared
  batch, alone or mixed with other requests) the token drawn at generation index `g` is a function
  of `(logits, seed, g)` only, so the same `seed` gives the same stream whatever the admission
  path or concurrency, given batch-invariant logits. Unseeded requests draw a random seed.
  Exceptions that stay stateful: XTC (`xtc_probability > 0`), the text-only `mlx-lm` fast path,
  and the round driver's own sampler. Different paths are different (equally correct) random
  streams, not a distribution bias.
- **`top_p`** always keeps the most probable token; `top_p = 0` therefore means greedy over the
  filtered row.
- **`top_k`** at or above the vocabulary size means "no truncation" (no error); `0` disables it.
- **`n > 1`** choice `i` uses seed `seed + i` wrapped to signed 64 bits (choice 0 keeps `seed`),
  identically on chat, completions and responses.

Prompt-cache boundaries on the VLM runner are resolved against the full rendered
chat template. Anthropic `cache_control` on system text blocks, native tool
JSON definitions, message text, text documents and tool-result text creates an
exact hybrid checkpoint at that token endpoint. Top-level automatic
`cache_control` moves to the last eligible block. Up to four writes, `5m` and
`1h` inactivity TTLs are supported; a successful read refreshes TTL. RAM and
entry budgets can evict an endpoint earlier. A marker inside a BPE token keeps
that token in the uncached suffix. Diagnostic rendering must reproduce the
original prompt exactly. Image placeholder expansions are verified against the
processor's actual token IDs; an endpoint inside a media span is rejected.

`cache_creation_input_tokens` counts only successfully stored breakpoint tokens
beyond the restored prefix. `cache_read_input_tokens` counts the actual restored
prefix; those two fields plus `input_tokens` equal the rendered prompt length.
OpenAI chat/Responses `cached_tokens` is the actual restored token count, without
cloud billing rounding. OpenAI text content blocks carrying
`prompt_cache_breakpoint: {"mode": "explicit"}` use the same renderer mapping.
Automatic APC remains available without hints. Proprietary cloud model minimum
lengths, cloud billing rates and guaranteed 24-hour residency are not emulated.
The text fast path retains its older character-hint contract.

Explicit endpoints have a separate numerical cache identity. Their suffixes
finish the same absolute prefill spans as a cold request; restores from a
different earlier breakpoint plan are rejected. These endpoints are retained
within this process's bounded APC policy; a restart may require a new write.

### Text reranking and classification heads

Text `Qwen3-Reranker` checkpoints use the model-card Transformers prompt and the
last-position yes/no logits, with a sigmoid of the logit difference. Original
`BertForSequenceClassification`, `RobertaForSequenceClassification` and
`XLMRobertaForSequenceClassification` safetensors checkpoints use their trained
heads, including published quantized encoder/head weights through mlx-vlm.
Other head architectures return a load error; Jina v3 `JinaForRanking` is a
separate unsupported architecture. Encoder heads require the `vision` extra
(mlx-vlm); no mlx-embeddings source or reconstructed classifier is used. Model cards report the effective
serving window: Qwen3 scoring caps at 8192 tokens; RoBERTa position padding offsets
and the tokenizer window constrain encoder inputs. Busy scoring work blocks
non-forced model unload, including when its HTTP waiter has been cancelled.

`POST /v1/rerank` retains `query`, `documents`, `instruction`, `top_n`, and
`return_documents`, and the existing `results[{index,relevance_score,document?}]`
shape. Text cross-encoders require string inputs; image objects require a VL reranker.

`POST /v1/score` uses joint query/document scoring for single-label head models
(sigmoid) and Qwen3 rerankers (yes/no probability), and cosine similarity for
embedding models. vLLM's `queries` / `documents` names are accepted alongside the
existing `text_1` / `text_2`. Scalar/list and length-one broadcasting preserve one
result per pair. `use_activation: false` returns the trained raw logit (Qwen3:
yes-minus-no logit), while omission or `null` uses probability scores. `instruction`
is passed to the Qwen3 prompt. `dot` / `euclidean` remain embedding-only extensions.

For trained heads, `POST /v1/classify` accepts `input` as a string or list and no
`labels`. It returns `data[{object:"classification",index,probs}]` with checkpoint
`labels` in head order: softmax for single-label multiclass, sigmoid for one logit
or a checkpoint declaring `multi_label_classification`. A scalar input also returns
the existing sorted `results[{label,score,index}]` convenience field. Providing
candidate labels selects the existing embedding-based zero-shot mode and is rejected
on trained heads. `temperature` applies only to the embedding-based mode.

`yv ab --base BASE_SHA --cand CAND_SHA --suite rerank --label rerank-TOPIC --priority -1`
checks candidate HTTP scores against independent float32 CPU Transformers recipes
on four small original checkpoints (five pairs, including a long document, and
three classification inputs). This capability stage uses the original
checkpoint as its numerical oracle; the base commit is pinned and recorded, but
has no trained-head endpoint to compare against. Scores must differ by at most
0.003 and preserve every ranking. No speed claims are made by this stage.

The recipes follow the [Qwen model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
and [vLLM scoring semantics](https://docs.vllm.ai/en/latest/models/pooling_models/scoring/).

Verified 2026-10-07 on M5, code commit `08a91d28`, yv base `5269e9e5`:
Qwen3-Reranker-0.6B, BGE-reranker-base, MiniLM-L-6-v2 (five pairs, including the
5840-character document), and BERT-tiny SST2 (three inputs). The candidate used
`TextScoringEngine` through `instantiate_engine`; all rankings matched independent
CPU Transformers float32 inference. Maximum probability errors were respectively
0.000208504, 0.000011891, 0.000037973 and 0.000001683 (limit 0.003). Raw score
activation, broadcasting and the registered embed-route checks passed. Evidence:
`/Volumes/P5Plus/yunshu-build/verify/runs/rerank-tiny-heads-handoff-1007-08a91d280b74/verdict.json`.
This is numerical and API evidence, with no speed or retrieval-quality claim.

### Published retrieval checkpoints

EmbeddingGemma 2 published bf16/4bit loading, quantized BERT-family trained heads,
and Qwen3-VL retrieval wrapper contracts are described in
[PRIOR_ART_FORMATS.md](../PRIOR_ART_FORMATS.md). Jina v3 `JinaForRanking` remains
explicitly unsupported. Empty VL embeddings return `[]` without processor work.

### Evals

All 12 OpenAI Evals endpoints are served under `/v1/evals`: create/list/retrieve/update/delete evals; create/list/retrieve/cancel/delete runs; list/retrieve run output items. SDK 3.26 tests and the `evals` real-server route check cover them. See the [official Evals reference](https://developers.openai.com/api/reference/resources/evals/methods/create).

Runs accept `jsonl` and `completions` data sources, with inline `file_content`, Files API `file_id`, or `stored_completions` filtered by model, metadata and inclusive creation timestamps. `completions` can sample the local model using message templates or an item reference. Items are validated against the eval's JSON schema before scheduling. Stored completions read the exact atomic JSON storage format of apiplanned (`fc28350d`) without copying its module; merged stored-completion support supplies `store=true` ingestion.

Supported graders: `string_check` (`eq`, `ne`, substring `like`/case-insensitive `ilike`), `text_similarity`, `score_model`, and `label_model`. Similarity is model-free: token-frequency cosine, character SequenceMatcher fuzzy match, effective-order sentence BLEU without smoothing, GLEU, ROUGE n-gram F1 (1–5), ROUGE-L F1, and exact-token METEOR with fragmentation penalty on both candidate and reference alignments (no stemming/synonym corpus). Lexical metrics return 0 for empty token inputs or unavailable n-grams; character fuzzy match preserves its raw-string equality/whitespace behavior. Scores use a caller-supplied pass threshold; local model graders request schema-constrained JSON through `/v1/chat/completions`. SDK Evals text/image/audio content blocks are normalized to the ordinary chat wire format. Python graders and Responses sampling sources return 400 as unsupported.

State uses atomic JSON replacement under `YUNSHU_EVALS_DIR` (default `~/.yunshu/evals`). Source rows and grader definitions are snapshotted per run; credentials stay in memory. Pure lexical grading uses one dedicated CPU worker with cooperative cancellation, keeping metadata and cancel routes available during long comparisons. Cancellation interrupts CPU grading and the active normal request, preserving completed items. Deleted evals cascade to their runs; parent checks and progress writes are serialized with deletion, and recovery removes orphan children after an interrupted cascade. Interrupted runs become `failed` on server restart; they are not automatically replayed. Reports are available through output-item routes (`report_url` is empty; no hosted dashboard). Maximum 10,000 rows per run; list pages accept 1–100 items with cursor/order/status filtering.
### Single-operator console backend

All five routes below use the model-management admin authentication contract:
set `YUNSHU_AUTH_TOKEN` and present it as Bearer / `x-api-key`, or explicitly opt
into `YUNSHU_AUTH_DISABLED`. Unconfigured admin access is denied.

| Route | Contract |
|---|---|
| `GET /v1/yunshu/host` | CPU-only `pmset -g therm`, `pmset -g batt`, OS memory pressure and memory/swap bytes. Cached 15 s; failed probes return `state: unknown` and `reason`. No root required. |
| `POST /v1/yunshu/models/register` | `{model, path}`: validate local config/model type and complete safetensors shards; register without loading or copying. Accepts an HF cache snapshot directory, including shard symlinks into `blobs/`; index filenames must be relative and cannot contain `..`. Requires multi-model mode; registration lasts for this process. Duplicate id: 409; invalid checkpoint: 400. |
| `DELETE /v1/yunshu/models/register/{model_id}` | Remove only an unloaded, non-loading registration (409 otherwise). Does not delete checkpoint files or HF cache. |
| `POST /v1/yunshu/models/cancel` | `{model}`: mark an active load and/or HF pull for cooperative cancellation. 202-style `status: cancelling` response (HTTP 200); poll status until loading clears. MLX work already executing is safely drained, then stopped and discarded, never handed to waiters. HF partial files remain resumable. No active operation: 404. |
| `GET /v1/yunshu/requests/recent?limit=50` | Latest completed successful requests, newest first (limit 1–512), including `latency`. In-memory bounded metadata only; no prompt/response bodies. |

`x_yunshu.latency` and recent rows expose `milestones_ms` relative to gateway
receive, and `durations_ms` for model lease, gateway admission, engine queue,
template/tokenize (including media preparation on VLM), APC lookup/restore,
prefill, first decode and SSE first flush. All use the same monotonic
`perf_counter` clock. Missing or fused stages are `null`, not invented zeroes.
The first-flush boundary is completion of ASGI `send` for the first content event,
not network delivery at the client. Engine prefill completion is a host callback
boundary, not a GPU profiler measurement; upstream may fuse its last forward
with first-token sampling. Gateway admission precedes model acquisition; runner
admission follows templating, so these milestones describe actual execution order.

Real-server coverage is registered in `scripts/research/route_checks.py`;
`yv ab --base BASE_SHA --cand CAND_SHA --suite console --model PATH --label
consolefeat-TOPIC --priority -1` runs the small-model cancellation, registration,
host and SSE-latency probe on the pinned candidate.

### Host power and request energy

`GET /v1/yunshu/host` preserves consolefeat host fields and adds cached `telemetry`
(CPU/GPU/ANE/DRAM/package watts, active GPU MHz/ratio, die temperatures, explicit
unknown reasons), under the same console admin permission. `x_yunshu.energy` and
`GET /v1/yunshu/requests/recent` expose GPU+DRAM phase-window estimates. See
[TELEMETRY.md](TELEMETRY.md) for the exact schema, cache ages, and concurrency limits.

### Console round 10 backend data

All routes require the same admin authentication as model management. OpenAPI lists
the routes and query parameters. M5 Qwen3.5-0.8B-MLX-bf16 pilot (2026-10-08, `1159a343`) passed all six console checks via job `1008-142909-00-consolegaps-tiny-pilot-1008-console-cand-a1-44f5`: entries/events, schema enforcement, bundle/manifest, impact, clear and history after restart. A second M5 served check at `e9ddd468` also passed all six checks (`1008-145849-00-consolegaps-tiny-final-1008-console-cand-a1-5911`); the final worker report carries merge-gate evidence.

| Route | Behavior |
|---|---|
| `GET /v1/yunshu/spec-decode` | Process-lifetime VLM MTP/DFlash counters and drafted/accepted tokens per zero-based draft depth; tree siblings share the depth denominator. Prometheus `yunshu_spec_decode_*_total`, labelled by engine/mode/position, uses the same totals. |
| `GET /v1/yunshu/cache` | APC entries: model, namespace, tokens, logical/attributed physical bytes, tier, last hit and process hits. Up to 512 lifecycle events, reason and originating request ID when known. Read-side snapshots run on the existing MLX executor. |
| `POST /v1/yunshu/cache/clear` | Refuses if any loaded engine cannot be proven idle (409). Clears resident APC, including pending WARM encode results; keeps SSD files and WARM configuration. Scope is explicitly `resident_apc`. |
| `GET /v1/yunshu/requests/history?limit=50&before=CURSOR` | Newest-first rotated serve-log metadata; limit 1–512, opaque exclusive cursor. Disabled log returns `enabled:false`. Unknown old measurements remain null. `YUNSHU_SERVE_LOG_RETENTION_DAYS` filters aged records; MAX_MB/KEEP bound stored bytes. No prompt/output/token IDs. |
| `GET /v1/yunshu/bundle/manifest` | Exact top-level fields included by `yunshu_cli.bundle.build`, redaction/exclusion policy and line caps. |
| `GET /v1/yunshu/bundle` | JSON attachment from the CLI diagnostics builder; no upload. |
| `GET /v1/yunshu/models/impact?model=ID` | Non-forced unload rejects in-flight requests (does not wait or interrupt). Advisory pre-load budget/slot LRU eviction list; post-load pressure is rechecked separately. Multi-model mode required. |

Recent rows now carry `model`, `speculative` (rounds and per-depth counts), cache
provenance and `structured_output`. Response `x_yunshu.structured_output` reports
`requested`, `enforced`, `engine`, `grammar_backend` and a reason. Enforcement means
the actual constraint was installed; output JSON validation remains a client task,
and a response cut short by max_tokens can be incomplete. No constraint setup
failure is represented as enforced. Status request rows always include `model`
(null only before the requested model has been attributed).

Cache physical bytes mean attributed array/storage bytes, not allocator footprint.
SSD file physical bytes include headers; primary logical bytes are tensor payload size, lower-tier logical bytes are restored raw-file size. `device` names a configured lower tier; unknown metadata is null. Entry hit
counts are bounded process metadata and restart at zero. Events are bounded and
may omit earlier lifecycle changes; a null request ID denotes an unattributed
background/legacy operation. Persistent history cursors are exclusive; concurrent
rotation can omit a row that leaves retention while paging.

### Agent-client additions (2026-10-07)

Responses client tools `custom`, legacy `local_shell`, and client-executed `tool_search`
are adapted to the model's function template and returned as `custom_tool_call`,
`local_shell_call`, and `tool_search_call`. `call_id` survives manual history and
`previous_response_id`; legacy shell outputs may identify the call with `id`.
Custom input supports text, regex, and Lark formats. Forced custom input streams
incrementally through the constrained decoder. Auto mode preserves text streaming;
a selected grammar-bearing custom call adds a constrained generation, sharing the
request's output-token budget. Deferred schemas remain hidden until loaded by a
client `tool_search_output`. Hosted tool search and duplicate names across namespaces
are rejected explicitly.

Anthropic documents accept text, custom content, stored-file references, and bounded
PDF base64/HTTPS sources. Vision models receive PDF page images and the text layer;
image-only PDFs require a vision model. Citations use checked character, page, or
content-block ranges and round-trip as `citations_delta` events. Document requests with
citations enabled stream live: only a possible partial `[[cite:...]]` marker is
held, and validated markers emit `citations_delta` immediately. Invalid source ranges
emit an SSE error without a successful terminal event. Client tool
schemas cover versioned bash, text editor, and legacy computer tools; computer zoom
is opt-in, and text-editor `max_characters` is tool configuration. The newer
`computer_toolset_20260801` member protocol is not implemented.

Chat and text completions support `stream_options.continuous_usage_stats` together with `include_usage`.
HTTP(S) video fetches use DNS-pinned redirects, TLS verification, and
`YUNSHU_VLM_MAX_VIDEO_BYTES` (100 MiB by default). `POST /apply-template` renders the
loaded tokenizer's template; `GET /props` exposes minimal loaded-model properties.
Both have `/v1` aliases.

CPU regression evidence: `test_agent_client_compat.py`. Served probes are registered
as `agent-custom-tools`, `agent-shell-search`, `agent-documents-citations`,
`agent-anthropic-client-tools`, `agent-continuous-usage`, `agent-template-props`,
and `agent-http-video` in `route_checks_agent_compat.py`. `yv --suite client_compat`
uses the M3 lane; `client_compat_m5` uses the M5. Both run the 0.8B pilot before the
3B text model, validate the commit-pinned source tree, and fail closed on missing
or unsuccessful checks. Real-server evidence is pending for this addition.


### Responses computer and local voice enrollment

`tools: [{"type":"computer"}]` maps to a local function schema with nine action
variants. Responses expose `computer_call.actions` in order, and
`computer_call_output` screenshots (URL or local file ID) retain their `call_id` and
vision content. The client executes every action. The gateway does not execute a
computer action or invent model safety checks; `pending_safety_checks` is empty.
Legacy single `action` input calls can be replayed, while the new tool emits `actions`.

`POST /v1/audio/voice_consents` records multipart `name`, `language`, `recording`.
`POST /v1/audio/voices` accepts `name`, `consent` and `audio_sample` through the OpenAI
SDK. Consent IDs are local; OpenAI-hosted consent IDs cannot be resolved here.
Audio must decode locally, be at most 10 MiB and 60 seconds, with at most 10 MiB
of decoded PCM. Enrollment keeps the reference and metadata in the bounded Files
store (`YUNSHU_FILES_DIR`, quota and TTL apply). It records a submitted consent;
it does not authenticate a speaker's identity. `GET /v1/audio/voices` includes
custom records. Speech accepts `voice: {"id":"voice_..."}` only for TTS models with
an explicit `ref_audio` generate parameter. Other models return a clear 400.
Qwen3-TTS Base also requires `ref_text` (sample transcript): submit it at enrollment
with SDK `extra_body` or in the speech request. Its CustomVoice/VoiceDesign variants
ignore cloning references and are rejected. Prompt-designed Live voices are
unsupported. Enrollment requires the audio extra.

CPU SDK and streaming evidence: `tests/unit/test_respfeat.py`. Real-server checks
are `respfeat-computer`, `respfeat-citations`, `respfeat-voices`, `respfeat-webrtc`;
`yv --suite respfeat` runs one M5 Qwen3.5-0.8B job with a ten-minute budget.
Enrollment evidence does not claim a real TTS voice-cloning round trip.
## Memory units

All engine-returned `*_gb` fields use binary GiB: 1 GiB = 1024^3 bytes,
with exact integer `*_bytes` siblings. This includes status memory, model `size_gb` /
`size_bytes`, model-pool memory, hardware info and trace host statistics.
A 128 GB Mac reports `total_gb: 128.0`; earlier decimal values were about 7% higher.
Prometheus memory metrics remain in bytes. See [API extensions](API_EXTENSIONS.md#memory-units).

### API keys and quotas

A key is presented like the token (`Authorization: Bearer ysk-...` or `x-api-key`). Over a quota the answer is 429 in the route's own error dialect (OpenAI `rate_limit_error`, Anthropic `rate_limit_error`, JSON-RPC for `/v1/mcp`) with `Retry-After` and `code` `requests_per_day_exceeded`, `tokens_per_day_exceeded` or `max_concurrent_exceeded`. The day is a rolling 24 h window in 5-minute buckets; token usage is counted when a request finishes, so a token quota stops the next request after it is crossed. A disabled, expired or unknown key is 401; an `infer` key on an admin route is 403. The realtime WebSocket accepts keys (infer or admin scope) but does not count against quotas.
