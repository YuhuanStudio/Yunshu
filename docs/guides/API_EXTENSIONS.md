# API extensions

What Yunshu adds on top of the OpenAI / Anthropic / Ollama APIs, why, and how it stays out of the way.
The rule: every extension is additive and namespaced, so the official SDKs keep working unchanged.

- JSON: a top-level `x_yunshu` object (responses) or `error.x_yunshu` (errors). The OpenAI SDKs keep
  unknown fields in `model_extra`; the Anthropic and Ollama SDKs ignore them.
- SSE: comment lines (`: yunshu-progress {...}`). Every SSE parser skips comment lines, and the
  chat stream already sends `: keep-alive` comments the same way.
- Headers: `X-Request-Id` and `X-Yunshu-*`.

Route and field list: [API_SURFACE.md](API_SURFACE.md#yunshu-extensions). Tests:
`tests/unit/test_api_extensions.py` (fake engine, real `RunStats` / tracker / middleware) and
`scripts/realmodel/smoke_api_extensions.py` (OpenAI + Anthropic + Ollama SDKs against a real server,
Qwen3.5-0.8B and Qwen3.8-27B-oQ4e-mtp, 23/23 and 24/24 checks).

## What other servers expose, and what we took

| Server | What it exposes | Taken | Left out, and why |
|---|---|---|---|
| Ollama | Per-response `total_duration`, `load_duration`, `prompt_eval_count/duration`, `eval_count/duration`; `keep_alive` on requests; `/api/ps` with `expires_at` | Same measurements (as `x_yunshu.timings`, in ms rather than ns); `keep_alive` with Ollama's grammar (`"5m"`, `300`, `-1`, `0`) on chat and completions; `expires_in_s` per model in the status endpoint | Nanosecond fields: the Ollama layer (`/api/*`) keeps its own timings. Ollama's default of unloading after 5 minutes: a single-node engine that is slow to load should keep its model unless told otherwise (`YUNSHU_MODEL_TTL_SECONDS`). |
| llama.cpp `llama-server` | `timings` object (`prompt_n`, `prompt_ms`, `prompt_per_second`, `predicted_*`, `cache_n`) in every response; `return_progress` sends `prompt_progress` (total / cache / processed / time_ms) while prefilling; `/props`, `/slots` (per-slot state), `/health` | `timings` with the same key names inside `x_yunshu` (tools that read them find them); prefill progress with total / processed / cached, plus %, tokens/s and ETA; `/v1/requests` is the analogue of `/slots` | Progress as a JSON `data:` chunk (`return_progress`): a strict OpenAI client would parse it as a chat chunk with no `choices`. We send an SSE comment instead. `n_probs` is not exposed here. `/props` now provides minimal loaded-model properties; `/v1/models` and `/api/show` carry the model contract. |
| vLLM | `/metrics` (Prometheus), `/health`, `X-Request-Id` support (`--enable-request-id-headers`), `/tokenize`, `/detokenize`, abort-on-disconnect | Request ids on by default; `/metrics`, `/tokenize` already exist; disconnect aborts already exist | Continuous-batching scheduler stats (waiting / running per step): not a goal of a single-node engine. |
| LM Studio | `stats` block per response (`tokens_per_second`, `time_to_first_token`, `generation_time`, `stop_reason`); `/api/v0/*` REST with model state | The same numbers in `x_yunshu` (`ttft_ms`, `decode_tps`, ...) | A separate `/api/v0` namespace: the additive fields on the standard routes reach existing clients with no code change. |
| OpenRouter | `GET /generation?id=` (cost, tokens, latency by generation id); provider error metadata (`error.metadata.provider_name`, raw upstream error); `X-Request-Id`-style ids | `GET /v1/requests/{id}` by the id the client chose; `error.x_yunshu` next to the OpenAI error object, with the id and the fix | Cost accounting and a persistent generation log: no billing on a local engine. |
| Yunxin (the user's gateway, read-only) | Its provider adapters (`lmstudio`, `moonshot`, ...) read `usage.prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens`, force `stream_options.include_usage` (else 0-token billing), map `Retry-After`, and send `Connection: keep-alive` SSE | Nothing in those fields changed: usage keeps the spec shape; `x_yunshu` sits beside it. 429 / 503 keep `Retry-After`; error `type` / `code` are unchanged, so an error mapper keyed on them is unaffected. `X-Request-Id` lets a gateway correlate its log line with ours and cancel upstream by id when its own client disconnects | |

## What was built

### Request ids

Every request gets an id: the client's `X-Request-Id` when it is 1-128 characters of
`[A-Za-z0-9._:-]`, else a generated `req_...`. It is echoed on every response, 4xx / 5xx included,
appears in the server log line, in `error.x_yunshu.request_id` and in `x_yunshu.request_id`. One id
does everything:

```
curl -H 'X-Request-Id: job-7' .../v1/chat/completions ...        # start
curl .../v1/requests/job-7                                        # where is it?
curl -X DELETE .../v1/requests/job-7                              # cancel (POST /v1/cancel {"request_id":"job-7"} works too)
```

A cancel that arrives before the request reached the engine (tokenizing, decoding an image) is
remembered and applied the moment the generation registers.

### Prefill progress (the 131K-token wait)

A long prompt sends nothing for minutes. While a streaming chat / completions request is queued or
prefilling, the server writes an SSE comment every `YUNSHU_PROGRESS_INTERVAL_S` (default 2 s; 0 turns
it off) and stops at the first token. The comment doubles as the proxy keep-alive.

```
: yunshu-progress {"request_id":"smoke-long-1","elapsed_s":8.12,"phase":"prefill","prompt_tokens":93268,"cached_tokens":0,"processed_tokens":92160,"percent":98.8,"tokens_per_second":8125.3,"eta_s":0.1}
: yunshu-progress {"request_id":"job-8","elapsed_s":3.0,"phase":"queued","queue_position":1,"queue_est_wait_ms":41200.0}
```

`percent` counts only tokens that are actually computed: a prefix-cache hit is reported as
`cached_tokens` and is not part of the denominator. Progress comes from the engine's own prefill loop
(one step per 2048 tokens on the batch path, one span per round on the round driver), not from a
timer. Measured: 93K-token prompt on Qwen3.5-0.8B, 32 comments over 8 s; 11.6K tokens on the 27B,
48 comments.

Non-streaming requests cannot be interrupted midway (the status code and headers are already fixed
when the first byte goes out, so padding the body with whitespace would turn a later error into a
200), so they poll `GET /v1/requests/{id}` with the id they sent. Prefer `stream: true` for long
prompts.

### Overload, deadlines and cancel

Nothing waits without bound. A generation request (chat, completions, messages, responses) that
arrives while `YUNSHU_QUEUE_LIMIT` (default 64) requests are in flight gets `429` at once, with
`Retry-After` (the queue's estimated wait, 1-60 s) and `error.x_yunshu` = `{reason: "queue_full",
queue_depth, queue_limit, retry_after_s}`; while MLX memory is above `YUNSHU_MEMORY_PRESSURE_REJECT`
(default 0.95 of the Metal working set) *and other requests are running* it gets `503`
(`memory_pressure`). An idle server never refuses on memory: waiting would not help. Each dialect
gets its own error type (`rate_limit_error` / `overloaded_error` on Messages; `rate_limit_error` /
`server_error` with `code` on the OpenAI routes), and the OpenAI / Anthropic SDKs already retry both
statuses and honour `Retry-After`.

`X-Yunshu-Deadline-Ms: N` is the wall time the client will wait, counted from arrival (the queue
included). OpenAI has no such field and a body field would be rejected by strict clients, so it is a
header next to `X-Request-Id`. Past the deadline the generation is cancelled and the client gets
`504` `deadline_exceeded` (Messages: `timeout_error`); once a stream has started, exactly one
terminal error event (`data: {"error": ...}` + `[DONE]`, `event: error`) ends it. The existing
`timeout` field is different: it ends *generation* with a partial answer (`finish_reason: "length"`,
`stop_reason: "max_tokens"`, `status: "incomplete"`).

An answer cut short by `DELETE /v1/requests/{id}`, `POST /v1/cancel` or a deadline carries
`x_yunshu.cancelled: true` (and `X-Yunshu-Cancelled: true` when not streaming); a queued request is
cancelled the same way and never reaches the engine.

### Context policy and token budget

`x_yunshu.context_policy` appears when the prompt was shortened to fit: `policy` (the strategy, or
`truncation_auto` for the Responses API's own option), `tokens_before` / `tokens_after` /
`tokens_removed`, `messages_before` / `messages_after` / `messages_removed`, `removed_roles` (user
turns are counted under `user`), `budget_tokens` and `cannot_fit`. Non-streaming responses also carry
`X-Yunshu-Context-Policy: truncate_oldest; removed=3 messages (2 user turns); tokens=9000->4000`;
streams report it in the usage chunk (the headers are sent before the prompt is shortened). The
latest user turn and the system messages are never removed: when they alone do not fit, the request
is a `400` `context_length_exceeded`, not a silent cut.

`x_yunshu.budget` appears when the prompt fits but leaves less room than `max_tokens` asks for:
the prompt, the thinking and the answer share one window, so `max_tokens` is clamped to what is
left (`max_tokens_requested` vs `max_tokens_granted`) and `thinking_budget` (a part of those tokens)
to what was granted. A prompt that leaves no room at all is a `400`. Only a window the model itself
reports counts; the tokenizer's `model_max_length` never shrinks a request.

### Per-response stats

`x_yunshu` is in the JSON body of non-streaming chat / completions, in the streaming usage chunk
(`stream_options.include_usage`), and, when the client did not ask for usage, in a trailing
`: yunshu-stats {...}` comment before `[DONE]`. On the Anthropic Messages and OpenAI Responses
routes the same object sits inside `usage` (the `message` body and the `message_delta` event;
the Response body and the terminal `response.completed` / `.incomplete` event), where the SDK
models keep unknown fields (`usage.model_extra["x_yunshu"]`); the Ollama layer maps it onto Ollama's
own nanosecond fields (below). Real response, Qwen3.8-27B (MTP):

```json
"x_yunshu": {
  "request_id": "smoke-nonstream-1",
  "queue_wait_ms": 0.4, "ttft_ms": 309.5,
  "prompt_tokens": 58, "cached_tokens": 0, "prefill_ms": 287.5, "prefill_tps": 201.7,
  "completion_tokens": 24, "decode_ms": 541.3, "decode_tps": 42.5, "total_ms": 857.3,
  "speculative": {"mode": "mtp", "drafted": 20, "accepted": 15, "acceptance_rate": 0.75},
  "timings": {"cache_n": 0, "prompt_n": 58, "prompt_ms": 287.5, "prompt_per_second": 201.7,
              "predicted_n": 24, "predicted_ms": 541.3, "predicted_per_second": 42.5}
}
```

| Field | Meaning |
|---|---|
| `queue_wait_ms` | Time between reaching the engine and the start of its prefill. |
| `ttft_ms` | Request arrival to first generated token (what the client sees). |
| `prefill_tps` | Uncached prompt tokens / prefill time. |
| `decode_tps` | (completion tokens - 1) / time between first and last token. |
| `cached_tokens` | Prompt tokens served from the prefix cache. |
| `speculative` | `mode` (`mtp` / `dflash`) when a drafter served the request. `drafted` / `accepted` / `acceptance_rate` count the draft tokens of this request: the round driver counts them per row, the single-row lane (`mtp_lane`, the MTP / DFlash tree rounds, upstream's loops) bumps the drafter's lifetime counters and the runner takes the per-request difference. `null` when nothing was drafted. |

Fields are `null` when a path cannot measure them. The text-only mlx-lm fast path fills the same
engine-side clocks as the VLM runner (queue wait, prefill progress from `generate_step`'s callback,
TTFT, prefill and decode tokens per second).
Non-streaming responses also carry `X-Yunshu-Queue-Wait-Ms`, `-TTFT-Ms`, `-Prefill-Tps`,
`-Decode-Tps`, `-Cached-Tokens`, `-Total-Ms`, `-Spec`, `-Spec-Acceptance`, for clients that log
headers and never parse bodies.

### Queue visibility

Generation responses carry `X-Yunshu-Queue-Position` (requests that arrived earlier and are still in
flight; 0 = served at once) and `X-Yunshu-Queue-Est-Wait-Ms`. The estimate is a lower bound: the
prefill still owed by the requests ahead, from their measured prefill speed; it does not guess how
long they will decode. `GET /v1/requests` and the `queued` / `prefill` / `decode` counters in the
status endpoint show the same live.

### Status endpoint

`GET /v1/yunshu/status` (same access rule as inference: open on a default local server, token when
`YUNSHU_AUTH_TOKEN` is set). `/health`, `/health/live` and `/health/ready` keep their semantics and
stay minimal: they are public, and orchestrators key off their status codes.

```json
{"object":"yunshu.status","version":"0.1.2","state":"running","uptime_s":17.3,
 "models":[{"id":"Qwen3.8-27B-oQ4e-mtp","type":"VLMEngine","loaded":true,"pinned":true}],
 "memory":{"active_gb":16.2,"active_bytes":17394617344,"cache_gb":0.28,"cache_bytes":300000000,"peak_gb":16.9,"peak_bytes":18100000000,"total_gb":128.0,"total_bytes":137438953472,"pressure":0.127},
 "requests":{"active":1,"queued":0,"prefill":0,"decode":1,"items":[{"request_id":"job-7","phase":"decode","completion_tokens":112}]},
 "throughput":{"window_s":60,"requests":3,"prompt_tokens":11765,"completion_tokens":64,
               "live_decode_tps":41.8,"mean_prefill_tps":502.9,"mean_decode_tps":51.7}}
```

### keep_alive and warmup

`keep_alive` on `/v1/chat/completions` and `/v1/completions` (and on the warmup call) follows Ollama:
seconds or a duration string, negative = keep loaded, `0` = free the model once idle. It applies in
multi-model mode (`--models-dir`), where a sweeper (every 5 s, or the memory enforcer when a memory
limit is set) unloads a model idle past its keep-alive, or past `YUNSHU_MODEL_TTL_SECONDS` when the
request set none. A model with active requests is never unloaded. A single-model server never frees
its model, so `keep_alive` is accepted and has nothing to do there. The Ollama layer (`/api/*`)
forwards its `keep_alive` and the request's `X-Request-Id` (generated when absent, echoed on the
response) to the OpenAI route, so cancel-by-id and `GET /v1/requests/{id}` work for Ollama clients.
Its final message carries Ollama's own fields in nanoseconds from the engine stats:
`total_duration`, `load_duration` (0: the model is resident), `prompt_eval_count` /
`prompt_eval_duration` (prefill), `eval_count` / `eval_duration` (decode).

`POST /v1/yunshu/warmup {"model": ..., "prompt": "<your system prompt>", "keep_alive": "30m"}` loads
the model if needed (`load_ms`), runs a one-token generation through the normal path (kernel compile,
allocator warm-up, `warmup_ms`) and leaves `prompt` / `messages` in the prefix cache, so the first
real request is a warm one. The reply includes that generation's `x_yunshu`.

### Errors that say what to do

OpenAI-format errors (including middleware ones: auth, rate limit, body size) get the fix in two
places: appended to `error.message`, the only field many SDKs print, and structured in
`error.x_yunshu`:

```json
{"error":{"message":"body.max_tokens: Input should be greater than or equal to 0 (hint: the request body does not match the API schema; docs/guides/API_SURFACE.md lists every accepted field)",
          "type":"invalid_request_error","param":"max_tokens","code":"validation_error",
          "x_yunshu":{"hint":"the request body does not match the API schema; ...","request_id":"smoke-err-1"}}}
```

Hints exist for: context too long, out of memory, 401 (which key), unknown model (`/v1/models`,
`yunshu pull`), 413, 429 (`Retry-After`, `YUNSHU_RATE_LIMIT_RPM`), 503 loading / shutting down,
validation, and 500 (`yunshu doctor`, the request id). Anthropic-format errors are left exactly as
the Anthropic spec has them.

### Server-side tools

`x_yunshu.server_tools` on a Messages message (`message_delta` when streaming, top level otherwise) and on a
Responses object (`response.completed` / the JSON body) reports what the server-side loop did; the SDKs ignore it.

| Field | Meaning |
|---|---|
| `rounds` | Generate / run tool / continue rounds the request took. |
| `round_usage[]` | Per round: `input_tokens` (prefilled), `cache_read_input_tokens` / `cached_tokens` (served from the prefix cache), `output_tokens`. A continuation reads the shared prefix from the cache and prefills only the new turn. |
| `tools[]` | Per call: `tool`, `kind` (`web_search` / `web_fetch` / `mcp`), `ok`, `error_code`, `ms`, `query`, `provider`, and `hint` when the tool is not configured (for web search: how to set a provider up). |

`GET /v1/models` items carry `yunshu.server_tools` (`web_search.available` / `provider` / `setup`, `web_fetch`,
`mcp_connector`) so a client or `yunshu launch` knows what the server can run before it sends a request.

`generate: false` on `POST /v1/responses` (and on the WebSocket's `response.create`) is a prewarm: the prompt is
prefilled into the prefix cache, the response has no output and `x_yunshu.prewarm: true`.

## Known gaps

| Item | State |
|---|---|
| Non-streaming keep-alive whitespace | Rejected on purpose, see Prefill progress. |
| `load_duration` on the Ollama layer | Always 0: a model that has to be loaded first is reported by the model-loading status, not per response. |

## Memory units

All engine-returned `*_gb` fields use binary GiB: 1 GiB = 1024^3 bytes.
Each has an exact integer `*_bytes` sibling, including status memory
(`active_bytes`, `cache_bytes`, `peak_bytes`, `total_bytes`) and model `size_bytes`.
A 128 GB Mac reports `total_gb: 128.0`. Earlier decimal values were about 7% higher;
update consumers when upgrading. Prometheus memory metrics remain in bytes.
