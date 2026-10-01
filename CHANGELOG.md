# Changelog

All notable changes to Yunshu are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/). Versions go up in small steps: a patch
(0.1.1, 0.1.2, …) for fixes and improvements, a minor (0.2.0) only for a real capability jump.
Release steps: [RELEASING.md](RELEASING.md).

## [Unreleased]

## [0.1.2] - 2026-10-01

Agent compatibility, constrained decoding that says what it cannot do, a bounded SSD prefix cache,
and sampled requests that use speculative decoding.

### Added

- Native agent compatibility for Claude Code, Codex and opencode: server-side `web_search`,
  `web_fetch` and the MCP connector run inside the generation loop (Messages and Responses);
  Files, Batches and Conversations APIs (OpenAI and Anthropic shapes), Responses compaction and
  `generate=false` prewarm; reasoning items with `encrypted_content`; `YUNSHU_MODEL_ALIASES` so
  agent-style model names resolve to a served model; `yunshu launch` starts agents with the
  model's real window and effort levels. The agent x feature matrix with end-to-end evidence is in
  [AGENT_COMPAT.md](docs/guides/AGENT_COMPAT.md).
- `yunshu statusline`: live engine state (prefill progress, decode speed, last cache hit) for
  Claude Code's status line, read from `/v1/yunshu/status`.
- Capability contract per model: `/v1/models` states tools, structured-output engines, logprobs,
  media, speculative mode, cache tiers and context length.
- `yunshu doctor` checks dependency versions, extras, llguidance, API feature state,
  half-downloaded models, the cache disk budget and cache integrity, each with a fix.
  `yunshu cache status|gc` finds truncated, corrupt and old-format SSD cache entries and orphaned
  temp files and trims to the size cap. `yunshu diagnose` writes a local diagnostics bundle
  (version, redacted settings, doctor output, recent errors with trace ids; never prompts, never
  uploaded). `yunshu service rotate-logs` plus size / age log rotation with gzip archives and
  secrets redacted.
- Constrained decoding: CFG constraints and JSON Schemas outside the in-house subset (pattern,
  length, range, multipleOf, item counts, prefixItems, recursive refs, formats) are enforced with
  llguidance; Messages `output_config.format` maps to constrained decoding.
- Tool-call structural-tag grammar (llguidance) for Qwen3.x XML and Hermes JSON calls, as an
  experimental option (default off).
- Per-request cache provenance (`x_yunshu.cache`, `X-Yunshu-Cache-*`); `x_yunshu` inside usage on
  Messages and Responses; Ollama `keep_alive`, request id and eval / load / total durations; speculative
  drafted / accepted counters; prefill progress on the mlx-lm fast path.
- Realtime GA fields (`idle_timeout_ms`, `noise_reduction`, `output_audio_buffer.*`,
  `rate_limits.updated`), Responses request echo, `truncation=auto`, `max_tool_calls`.
- Paired downstream evaluation harness (GSM8K, MMLU-Pro, IFEval, needle, BFCL; McNemar and
  paired CI) and an agentic coding benchmark driving real agent CLIs against a local server.

### Changed

- Behavior you may notice:
  - Unsupported regex constructs and the JSON-Schema keywords `uniqueItems`, `not`, `if / then /
    else` and `contains` now answer 400 before generation instead of being approximated or ignored.
  - The context budget answers 400 `context_length_exceeded` when the system prompt and the latest
    user turn do not fit, instead of silently dropping them. Truncation is unit-based, keeps the
    system prompt and the latest user turn, and summaries are quoted user history, never a system
    message.
  - Capability-contract 400s: a chat request using tools, media parts or other features the
    model lacks gets an explicit error.
  - The APC SSD prefix cache is on by default (`~/.yunshu/cache/apc`, opt out with
    `YUNSHU_VLM_APC_DISK=0`) under one global disk budget per cache root (64 GiB cap, LRU across
    models), a free-space reserve (the larger of 10% and 20 GiB, rechecked per write), and stale
    namespace pruning. A write error pauses spilling with one warning instead of tracebacks.
  - The tool-call structural-tag grammar is experimental and off by default.
- Sampling contract: `top_p` always keeps the best token, `top_k` at or above the vocabulary is a
  no-op, the shared batch and the round driver use the position-keyed sampler, and per-choice
  seeds are derived the same way on every route (choice 0 keeps the caller's seed).
- Messages and Responses hand tools to templates that render them natively (Qwen3.x tool-call
  format); mid-conversation system messages stay in place for prefix reuse.
- Checkpoint fingerprints (weights, config, tokenizer, template, adapter, layout, format) are part
  of text SSD and APC cache keys; SSD loads are validated and fall back to a cold prefill on
  corruption.
- The speculative lane uses lane-linear projections by default (27B: MTP prose 1K +12%, DFlash2
  prose 1K +17%, parity unchanged); the packed path and `YUNSHU_LANE_LINEAR` are gone.
- The round driver remains experimental (single-request greedy decode 10-40% behind the default
  lane); its sampled drafting is behind `YUNSHU_ROUND_KEYED_DRAFT`.
- The mypy baseline gate is shared by `just lint` and CI; `llguidance` is an explicit dependency.

### Fixed

- Malformed Qwen3.x tool calls (missing `</function>`, broken JSON, tool-name tags) become
  `tool_use` blocks and call markup no longer leaks into text; unreadable calls are dropped.
- Family adapters keep media parts on the vision path (Gemma 4 and Mistral dropped images); a
  Claude Code turn with an image tool result no longer fails with "System message must be at the
  beginning"; an engine error after the stream starts is an error event on Messages, chat and
  Responses.
- Streaming and lifecycle: terminal events can no longer be dropped from a full queue, a failed
  runner submit drains every job, per-job output is bounded, a request cancelled before admit never
  prefills, reset no longer orphans an MLX worker, and a model lease prevents unload racing
  `get_engine`. TTS cancels between chunks.
- Constrained decoding: regex uses an exact Unicode DFA, a dead-end mask raises instead of
  releasing the constraint, EOS shape is normalized, reasoning tags inside tool JSON stay data,
  and a draft the tool mask forbids is never fed to the matcher.
- Stop / reasoning split keeps tag text inside JSON payloads; a streaming parser holds spaced tags.
- Responses no longer double-templates on VLMs; the Anthropic route keeps native tools with
  `tool_choice` / parallel; a 500 on Claude Code system reminders in image requests.
- Gateway: constant-time UTF-8 token compare, request alias survives duplicate ids, non-finite
  settings rejected, malformed `Content-Length` rejected, a counted body without `Content-Length`
  is forwarded (was empty), response cache keyed by engine generation, Conversations I/O off the
  event loop, file-store quota and crash-orphan cleanup.
- Round driver: greedy rows take the serial-equivalent argmax; an idle decode batch releases its
  slot buffers (b8 after a 32K request 34 to 21 GiB).

### Security

- One network policy (`netguard`) for MCP, `web_fetch` and media downloads: resolve once, classify
  mapped IPv6 / CGNAT / multicast, connect to the pinned IP, re-check every redirect hop, total
  deadline and byte budget; MCP JSON-RPC replies are validated and the legacy SSE endpoint must be
  same-origin.
- Diagnostics and rotated logs redact secrets; the docs list the real outbound connections.

### Performance

- Sampled requests use speculative decoding (position-keyed Gumbel sampling makes draft acceptance
  exact): sampled 27B agent traffic 20 to 69-93 tok/s, token distribution checked against the
  serial sampler.
- The APC SSD tier reloads bit-exact checkpoints 10-40x faster than re-prefill on 27B and survives
  restarts; a longer SSD prefix beats a short RAM hit on the text engine.
- Byte-bounded prefix cache with superseded checkpoints dropped and an end-of-system-turn checkpoint.
- Ported oMLX SiLU probe and `qmv_fast` layout rule (27B greedy output identical, decode unchanged).

## [0.1.1] - 2026-09-29

Yunshu is now positioned as a local LLM / VLM inference engine (decode speed, TTFT, prefix reuse,
API completeness), with Qwen3.8-27B as the first fully tuned model. Speech-to-speech and the other
modalities remain supported capabilities; the README, CLAUDE.md and AGENTS.md were restructured to
match.

### Added

- First-run commands: `yunshu doctor` checks the Mac (Apple Silicon / Rosetta, macOS, Python, MLX
  and Metal, memory, settings, models directory, the model and whether it fits, port) and prints a
  fix for each problem; `yunshu pull <org/name>` downloads into the models directory, refuses to
  download a model that is already on disk (models directory or Hugging Face cache) and resumes an
  interrupted download; `yunshu model list` also lists the Hugging Face cache; `yunshu --version`.
- Models from the Hugging Face cache are usable in place: `yunshu serve -m org/name` (or a name under
  the models directory) serves the local copy instead of downloading, and multi-model mode lists
  cached models next to the models directory (`YUNSHU_HF_CACHE_MODELS`, on by default).
- `yunshu config set KEY VALUE` / `unset` / `path`: settings saved in `~/.yunshu/config.toml`, read
  by every command and the service (e.g. `yunshu config set models_dir /Volumes/Models`).
- `yunshu service install|uninstall|start|stop|restart|status|logs`: a per-user launchd agent
  that starts Yunshu at login, restarts it after a crash, and logs to `~/Library/Logs/Yunshu`.
- Docs: [connecting clients](docs/guides/CLIENTS.md), [service](docs/guides/SERVICE.md),
  [troubleshooting](docs/guides/TROUBLESHOOTING.md), [benchmark sources](docs/BENCHMARKS.md),
  [RELEASING.md](RELEASING.md); a tag-triggered release workflow (PyPI trusted publishing behind
  an approval step) and a Homebrew formula for the `yuhuanstudio/tap` tap (`packaging/homebrew/`).
- One settings registry for every `YUNSHU_*` setting: a TOML config file (`--config`),
  `yunshu serve --set KEY=VALUE`, `yunshu config` (effective value and source of each), generated
  [docs/CONFIGURATION.md](docs/CONFIGURATION.md); a bad value stops startup, a misspelled name
  warns.
- Qwen3.5-family VLM runner (Qwen3.5 / 3.6 / 3.8) on `mlx-vlm`'s generator: prefix cache with
  exact hybrid checkpoints (text + image-pixel keys, optional SSD tier), MTP or DFlash speculative
  decode, streaming reasoning split, tools, JSON schema, stop sequences, logprobs and cancel.
  On Qwen3.8-27B (M5 Max) warm chat TTFT went from 2.5 s to 0.2 s, a repeated 8K prompt from 9 s to
  0.1 s, and decode from 30 to 57 tok/s.
- Exact verify kernels (GatedDeltaNet prework/replay, split SDPA, 5-bit streamed matmul) and
  batch-invariant decode kernels, partly vendored from oMLX (Apache-2.0).
- Per-row-length (ragged) KV cache for models with Qwen3.5-family attention, now the default: rows
  in the shared batch no longer read or copy the longest row's padding, and the single-request
  speculative lane runs decode and verify attention on one per-row kernel. Qwen3.8-27B (M5 Max):
  MMLU-Pro 300 at 8 in flight 249/300 in 28.3 min at 139 tok/s, peak 33.7 GiB (padded cache:
  250/300, 46.5 min, 88 tok/s, 45 GiB); single-request decode at 131K context 24.5 → 46.6 tok/s,
  at 32K 39.6 → 59.1; short contexts unchanged within noise; spec on == off at 1K / 32K / 131K.
- `YUNSHU_KV_PRECISION=bf16|int8` (default bf16): int8 K/V with one fp16 scale per 32 dims for the
  shared batch — about half the KV memory and bandwidth, lossy (MMLU-Pro 251/300, peak 28.1 GiB).
- Experimental own round driver for the Qwen3.5 family (`YUNSHU_ROUND_DRIVER`, off by default):
  multi-row MTP drafting with a cost-aware draft depth (TensorFold's allocation rule, MIT) and
  multi-prompt prefill; being measured.
- `YUNSHU_LOG_LEVEL`; `scripts/research/` benchmark matrix, realistic soak and MMLU-Pro soak.

- Automatic speculative path: a DFlash2 drafter that matches the served model (Qwen3.8-27B:
  `incoai/Qwen3.8-27B-DFlash2`, in the models directory or the Hugging Face cache) is used without
  a flag, with the cost-aware chain depth and 8-bit drafter weights. Qwen3.8-27B server, novel_en
  1K/8K/32K/131K 57.1 / 48.2 / 46.1 / 32.9 tok/s (MTP 47.5 at 1K), code corpus 82.0 / 89.1 / 70.5 /
  79.9, 34/34 checks, TTFT about 207 s at 131K; lossless (spec on == off). `YUNSHU_VLM_DRAFT=mtp`
  forces the MTP head, `=off` disables drafting, a path picks a drafter; `yunshu doctor` prints the
  path that will be used.
- `/health/ready` returns a `reason` when the model failed to load.
- `scripts/dev/robustness.py`: black-box probe (load failure, disconnect during prefill and
  streaming, limits, mixed concurrent features, 500-request memory, SIGTERM / SIGINT with requests
  in flight).

### Fixed

- A non-streaming request whose client disconnected kept the GPU busy to the end of its prefill or
  `max_tokens`: Starlette's `is_disconnected()` never fires behind `BaseHTTPMiddleware`. A
  pure-ASGI watcher now records the disconnect and the request's row is dropped at the next chunk
  (prefill of a 60K-token prompt: next request waited 4.0 s before, 0.8 s after, on a 0.8B model).
- JSON-schema, regex and grammar constraints no longer allow special tokens (`<|im_end|>`) inside
  a string: a model that chose one ended the response mid-string (Qwen3.5-0.8B stopped at
  `{"name": "Alice`).
- Ctrl-C / SIGTERM with streams in flight waited for them forever (a 27B stream takes minutes);
  connections are now cut after `YUNSHU_DRAIN_TIMEOUT` (default 30 s) and the server exits.

### Changed

- A model that fails to load (missing path, truncated weights, out of memory) no longer stops
  `yunshu serve`: the server stays up, `/health/ready` answers 503 with the reason and chat requests
  get a 503. `yunshu serve` still prints the problem and its fix first.
- Rate limiting is off by default (`YUNSHU_RATE_LIMIT_RPM=0`; the old 120 per minute rejected a
  local client's 500 sequential requests with 429). Set it above 0 to enable it.
- `yunshu serve` binds `127.0.0.1` by default (was `0.0.0.0`) and warns when it serves the network
  without `--auth-token`. It checks the model before loading and stops with a fix when the path
  does not exist, a download is incomplete, or the weights exceed memory.
- The default models directory is `~/.yunshu/models` (was the source tree's `models/`, which
  pointed inside the installed package for a wheel install). `YUNSHU_MODELS_DIR` still overrides.
- `/version`, the OpenAPI schema and the MCP `serverInfo` report the installed package version
  (they said `0.1.0-dev`).
- With `--auth-token`, the token is also accepted as `x-api-key` (what Anthropic SDKs send).
- The wheel and sdist include `THIRD_PARTY_NOTICES.md`; the sdist no longer carries benchmark data.
- Default decode for MTP models is lossless batch-invariant decode at MTP block 6: decode and verify
  matmuls of a drafting request share one row-invariant kernel (M5 tensor-unit packed where
  available), so greedy output with speculation on equals speculation off. Qwen3.8-27B (M5 Max,
  in-process) code / prose / JSON-like: 88.6 / 59.9 / 67.3 tok/s, vs 57–67 / 50–53 / 58–62 for the
  previous exact kernels at block 3; server matrix decode 57 → 80 tok/s, 33/33 checks. Requests that
  cannot draft (sampling, logits processors, logprobs) keep the stock kernels.
- Every `mlx-vlm` model is served by the VLM batch runner: concurrent requests share one
  continuous batch with per-row sampling settings, and a request that is alone uses speculative
  decoding (Qwen3.5 family). The older per-request VLM loop was deleted; how it worked is recorded
  in `docs/archive/legacy_vlm_loop/`.
- 5/6/8-bit projections run on TensorFold's integer-code tensor-unit matmul when packed (M5).
- Lossy optimizations are options with a lossless default. Text engine: KV cache quantization
  (`YUNSHU_KV_QUANT_BITS`, was `auto`) and the 4-bit warm prefix tier (`YUNSHU_PREFIX_HOT_LIMIT`)
  are off by default, and the SSD prefix cache stores KV and recurrent state bit-exact
  (`YUNSHU_SSD_CACHE_PRECISION=native`; `int8` stays available and old int8 files still load).
- `reasoning_effort` (top-level or in `chat_template_kwargs`) is passed to chat templates that
  support it (Qwen3.8: low / medium / xhigh) instead of being mapped to a thinking-token cap.
- Dependencies: MLX 0.32.2, transformers 5.17, upstream `mlx-vlm` 0.7.3 (the fork is gone).
- JSON-schema constraint: cached vocab split (in-string step 144 ms → 3 ms); structural whitespace
  limited to spaces and newlines.
- Research scripts and parity tests take model paths from arguments or the environment
  (`M`/`D`/`TF` for the validation scripts, optionally from a gitignored
  `scripts/research/local.env`; `YUNSHU_PARITY_MODEL` for the parity tests, which skip when unset).
- A model served from the Hugging Face cache is listed under its repo id (`org/name`) in
  `/v1/models`, not the snapshot path.

### Fixed

- Streaming thinking on Qwen3.8 was sent as content; it is now sent as reasoning.
- Gateway startup failure no longer crashes in shutdown on an unbound engine.
- VLM responses under concurrent load are returned as soon as each generation finishes; before, a
  post-request cache clear queued behind every other request held all responses until the last
  one finished. The non-streaming VLM path also no longer applies a 120 s default timeout that
  counted time spent waiting in the queue (a client-set `timeout` still applies).
- Tool calls from non-Qwen families (Gemma-4, GLM-4.7, Mistral, pythonic, DeepSeek, Kimi,
  MiniMax, …) were left in `content` with `tool_calls` empty. The tool-call format now comes from
  the model's chat template (upstream mlx-vlm/mlx-lm registry) instead of guessing from the
  request's model name, and every route (chat, Anthropic, Responses, Realtime; streaming and not)
  parses with that format's upstream parser. Arguments are typed with the request's tool schemas.
  Gemma-4 E4B capability checks: 7/34 → 33/34.
- DFlash2 speculative decode: the drafter's hidden-state capture covered the whole prompt instead
  of the drafter's 2047-position window (about 48 TFLOP extra at 131K), the draft block was capped
  at 4 (the drafter is trained for 8), and speculation on did not equal speculation off at longer
  contexts; it now uses the per-row prefill path and batch-invariant kernels like MTP.
- The SSD prefix cache's native-precision bf16 writer reinterpreted values instead of bits (it was
  never reached while every write went through int8).

### Removed

- The deprecated `Engine` class, multi-node / multi-tenant leftovers (distributed diffusion,
  connection pool, data-parallel endpoint, tenant auth naming), and unused modules
  (`vlm_async_engine`, `wan_vae`, `lid`, `optimizations`, `vision_encoding`).
- The older VLM MTP / APC side paths, superseded by the runner.
- The runner's upstream quantized-KV option (slower than bf16 at every measured length) and other
  measured-out flags (`YUNSHU_PACKED_5BIT`, `YUNSHU_VLM_KV_BITS`, `YUNSHU_VLM_INVARIANT`,
  `YUNSHU_MTP_FAST_VERIFY`, `YUNSHU_RAGGED_KV` — ragged is now always on for qwen3_5 attention).
- Yunshu's own tool-call parsers (`tool_call_parser.py`, `tool_call_parsers.py`, about 4,500 lines
  with their tests), superseded by the upstream parser registry; Yunshu keeps only the formats
  upstream lacks (its prompt-injected JSON format, DeepSeek, whole-message JSON).

## [0.1.0] - 2026-07-01

First public release — the full capability surface below, each endpoint smoke-verified
against a real model on Apple Silicon.

### Added

- **Native streaming speech-to-speech** (the flagship): Qwen3-Omni Thinker→Talker
  on Apple MLX via `mlx-vlm` — `POST /v1/omni/speech/stream` (SSE) and the
  OpenAI-Realtime `WS /v1/realtime` socket. Speech-in (raw audio, no ASR) and
  speech-out, ~1.2 s first-audio text-in / ~1.4 s speech-in (warm). Multi-turn
  conversation context preserved (bounded to protect TTFT). Tool-calling works on the
  voice path too — a tool turn emits `function_call` items and the spoken JSON is
  suppressed (the voice doesn't read the call aloud).
- **OpenAI-compatible API**: chat/completions (tool-calling, JSON-schema/grammar,
  streaming, logprobs), completions, responses, embeddings, audio
  (transcriptions/translations/speech), images (generation + edits/variations/
  inpaint/controlnet), tokenizer utilities, batch inference.
- **Tool-calling**: tools are rendered via the model's own chat template when it supports
  them natively (better adherence), and a forced `tool_choice` (required / named) is
  structurally enforced with an assistant prefill — consistently across
  `/chat/completions`, `/v1/responses`, and Anthropic `/v1/messages`.
- **Trained image ControlNet**: the `/images/controlnet` + `/depth-guided` routes run the
  real Z-Image Fun-Controlnet-Union when weights are present (canny/depth preprocessing),
  and inline `<lora:name:weight>` applies on every image route.
- **Anthropic-compatible** `/v1/messages` (+ `count_tokens`).
- **Multimodal retrieval**: `Qwen3-VL-Embedding` (text / image / cross-modal in one
  shared space) on `/v1/embeddings`, and `Qwen3-VL-Reranker` true cross-encoder on
  `/v1/rerank` (bi-encoder cosine fallback otherwise).
- **Vision/VLM, OCR, ASR, TTS, image & video generation, MCP** server/client.
- **Single-node decoding/optimization**: single-request fast path (`mlx-lm
  generate_step`), KV prefix + prompt caching, lossless n-gram speculative decode
  on greedy (default-on), per-request top-nσ sampler, constrained decoding
  (JSON-schema/regex/grammar), jump-forward, GPU sampler, in-memory MXFP4/NVFP4
  weight quant, automatic KV quant. The situational ones are opt-in (see
  [docs/CONFIGURATION.md](docs/CONFIGURATION.md)).
- Single-model mode serves the loaded model under any requested name
  (Ollama/LM-Studio behavior); multi-model auto-discovery via `YUNSHU_MODELS_DIR`.
- Docs: [API reference](docs/API.md), [configuration reference](docs/CONFIGURATION.md),
  runnable [examples/](examples/).

### Changed

- **Refocused** from an over-scoped "distributed inference platform" to an honest
  single-node omni engine. Removed: multi-node mesh / distributed paths
  (sharded-load, disaggregated prefill/decode), the multi-tenant control plane,
  and tiered-KV offload. All single-node decoding/optimization tech is kept.

### Notes

- macOS + Apple Silicon only (wraps MLX; no custom Metal kernels). Python 3.13+, `uv`.
- Apache-2.0 licensed.
