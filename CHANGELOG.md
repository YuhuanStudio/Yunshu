# Changelog

All notable changes to Yunshu are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/). Versions go up in small steps: a patch
(0.1.1, 0.1.2, …) for fixes and improvements, a minor (0.2.0) only for a real capability jump.
Release steps: [RELEASING.md](RELEASING.md).

## [Unreleased]

The 0.1.5 cycle adds local decision models, repeatable evaluations and broader coding-agent
and web-retrieval compatibility to the `yunshu` package. These changes are merged on main;
0.1.4 remains the published version.

### Highlights

- The web console is its own light process (`yunshu console`, default port 8100, never loads MLX).
  `yunshu serve` starts it next to the engine (`--no-console` to skip), `yunshu service install` runs it as its own
  launchd job, and `yunshu console --engine URL` watches a remote engine. It serves the console and docs, proxies
  the engine API (streams and WebSockets included) on one origin, and records a 1 s / 10 s / 1 min history for 30
  days plus a metadata-only request log in SQLite, with engine outages kept as gaps and events, so the console
  stays usable and keeps recording while the engine restarts or crashes. The engine keeps no history store and
  gains only a cursor on its finished-request read and its pid in `/v1/yunshu/status`.
  **Deprecation:** `/console/` on the engine now only redirects to the console process (or explains how to start
  it); the engine no longer serves the console files, and this pointer goes away in a later release.
  Settings: `YUNSHU_CONSOLE`, `YUNSHU_CONSOLE_PORT`, `YUNSHU_CONSOLE_HOST`, `YUNSHU_CONSOLE_ENGINE`,
  `YUNSHU_CONSOLE_ENGINE_TOKEN`, `YUNSHU_CONSOLE_POLL_S`, `YUNSHU_CONSOLE_HISTORY`,
  `YUNSHU_CONSOLE_RETENTION_DAYS`, `YUNSHU_CONSOLE_DB_MAX_MB`. [Console guide](docs/CONSOLE.md).

- Ask typed predicate, choice and score questions with the Decisions API on Clef MLX checkpoints,
  without generating a text answer. [Guide](docs/guides/DECISIONS.md).
- Store chat completions locally and use them in repeatable Evals runs, with cancellable sampling,
  lexical graders and local model graders. [Guide](docs/guides/EVALS.md).
- Coding clients can use custom grammar-bearing tools, local shell/tool-search calls, Anthropic
  documents/citations and continuous streamed usage. [Coverage and limits](docs/guides/API_SURFACE.md).
- Search, extract, crawl and research through a local Tavily-compatible API with provider health
  backoff and lexical ranking; generation runs only where requested. [Guide](docs/guides/TAVILY.md).

### Upgrade notes / breaking changes

- New APIs are available from a source build of main until 0.1.5 is released; upgrading the published
  0.1.4 package does not add them. No package version bump is part of this draft.
- Decisions require a supported Clef joint-schema checkpoint; ordinary chat checkpoints and unknown
  decision heads are rejected. OpenJev, Laya and D1 support is not included in this merged loader.
- `store: true` retains chat content locally. Configure `YUNSHU_CHAT_COMPLETIONS_DIR` and
  `YUNSHU_CHAT_COMPLETIONS_MAX` to choose its directory and retention cap; omitted/false does not store.
- Search sends queries to external providers and fetch sends URLs to destination sites. Use
  `YUNSHU_WEB_SEARCH_PROVIDER=none` and `YUNSHU_WEB_FETCH=0` to disable these server tools.

- Credentialed browser clients must use explicit `YUNSHU_CORS_ORIGINS`; any wildcard now disables
  CORS credentials. Configuration writes replace files atomically with owner-only permissions (0600).
- Every `*_gb` field the engine returns (`/v1/yunshu/status` memory and models, `/v1/models` `size_gb`, model-pool status, `memory_usage`, hardware info, trace host stats) is now binary: GB = 1024^3 bytes, the unit macOS, `yunshu doctor` and the `YUNSHU_*_GB` settings use. It was decimal (1e9) for engine memory and model sizes, so values drop by about 7% (a 128 GB Mac reads 128.0, not 137.4). Scripts that read `*_gb` see the new numbers; each field now has an exact integer `*_bytes` sibling (`active_bytes`, `cache_bytes`, `peak_bytes`, `total_bytes`, `size_bytes`, `current_bytes`, `max_bytes`, `max_memory_bytes`, `current_memory_bytes`, `total_memory_bytes`, `working_set_bytes`, `memory_available_bytes`). Prometheus metrics stay in bytes.

### Performance

| Machine | Model / mode | Metric / workload | Before → after | Recorded source |
|---|---|---|---|---|

No new decode or TTFT claim is made for this cycle here. Historical measurements remain in
[Benchmarks](docs/BENCHMARKS.md).

### Added

- Console backend: VLM speculative acceptance by draft depth, resident APC entries and bounded lifecycle events, metadata-only serve-log history with cursor pages, CLI diagnostics bundle download/manifest, actual structured-decoding enforcement reports, and advisory model unload/load impact.
- OpenAI Evals API: 12 CRUD/run/output-item endpoints, atomic local persistence, cancellable background runs through normal chat inference, JSONL/file/stored-completion sources, lexical similarity and local score/label graders.
- Add optional unprivileged Apple IOReport/HID host telemetry, request GPU+DRAM energy estimates, Prometheus gauges/counter, `yunshu top`, and tfbench/yv efficiency fields. Handle qualified macOS 27 CLPC counters and Max ANE/MTR sensor names. Off by default (opt in with `YUNSHU_TELEMETRY=on`): the 27B A/B found a small follow-up TTFT cost when on.
- Add authenticated console model load/download cancellation and validated local/HF snapshot registration without loading or copying weights.
- Expose cached CPU-only thermal, power, OS memory pressure and swap telemetry with explicit unknown reasons.
- Record per-request latency milestones in `x_yunshu` and expose them through recent request metadata.

- Client-executed Responses computer actions and screenshot round trips.
- Incremental Anthropic document citation streaming with checked source ranges.
- Optional WebRTC Realtime transport (`yunshu[webrtc]`) and local audio-sample voice enrollment for reference-audio TTS models.

- First-run model selection (`setup` / unconfigured `serve`), `models list/pull/show/rm`,
  live `top`, zsh/bash/fish completion and consistent global `--json` output with next steps.
  [CLI guide](docs/guides/CLI.md).

- `POST /v1/decisions` and `POST /v1/systemone`: text and inline-image typed decisions on Clef MLX
  models, with probabilities, refusals for non-finite head results and no text decoding.
- Stored chat completions: creation with `store: true` / `metadata`, list/retrieve/update/delete,
  input-message listing, pagination and filters; streams are stored after completion.
- Realtime ephemeral keys: `POST /v1/realtime/client_secrets` and beta session/transcription-session
  creation. Secrets authenticate the Realtime WebSocket and apply session configuration; transcription-only
  configuration is echoed but is not executed as a transcription-only engine session.
- Twelve Evals routes for definitions, runs, cancellation and output items, with atomic local persistence,
  inline/file/stored-completion sources and `string_check`, `text_similarity`, `score_model`, `label_model`.
- Tavily-compatible search/extract/crawl/map/research, feedback/logs/usage/provider health and native MCP;
  additional metasearch adapters and optional bounded Chromium rendering. [Limits](docs/guides/TAVILY.md).
- Agent-client adaptations for custom text/regex/Lark tools, legacy local shell, client tool search,
  Anthropic documents/citations, continuous usage stats, template rendering and model properties.
- Source-built YunUI console with engine status/resource charts, model operations, request inspection,
  cancellation and a streaming diagnostic playground. [Build and scope](docs/CONSOLE.md).
- `scripts/dev/release_check`: SHA-pinned release checklist, CI first and concurrent gate/M3/client checks;
  informational agentbench is submitted separately at priority -3 and collected later. CPU-only planning uses `--dry-run`.
- SDK endpoint coverage walker, prior-art discovery tool, local clean-checkout CI, and live private
  research-index generation. Public-doc checks enforce translated README structure and registered APIs/settings.

### Changed

- Refresh the dependency lock for the next cycle (including Anthropic 1.12 and FastAPI 0.143.0) and review upstream changes. MLX stays below 0.32.4 pending version-matched kernel validation.
- Documentation coverage: a capability overview table in all three READMEs (generated from `docs/feature_index.json`, checked by a unit test), the console guide rewritten page by page with screenshots, and new guides for [inference features](docs/guides/INFERENCE.md), [multimodal endpoints](docs/guides/MULTIMODAL.md) and [authentication, keys, settings and CORS](docs/guides/AUTH_AND_KEYS.md).
- gpuq admits declared short verification jobs between long cells without preemption, with a bounded
  time budget; `--gate` takes precedence over same-priority backlog. Foreign CPU contention is measured,
  parsed job records are cached, and non-quiet filler work can run while quiet work waits for CPU admission.
- CI and local CI include frontend type checks, tests and build plus Python lint/format/mypy and package checks.
- Local CI matches the release runner's Python 3.13 environment, short paths and inaccessible local data;
  tests avoid shared server-port collisions. [Contributor workflow](CONTRIBUTING.md).
- Modality qualification through yv includes published embedding/classifier loaders,
  prepared-input parity and bounded fixed-seed diffusion comparisons. Diffusion
  timing requires three same-device interleaved quiet pairs after a successful pilot.

### Fixed

- Untyped XML tool arguments containing JSON objects or arrays reach clients as containers; scalar text and declared string unions keep their existing types. Numeric or nested container text containing non-finite values stays literal instead of emitting invalid JSON.
- GPU guard blocks broad `pkill` commands that could terminate another worker or user process.

- EmbeddingGemma 2 loading retains every weight shard instead of keeping only the last shard.
- MCP notifications return an empty 204 body, preventing a dropped connection from a JSON `null` body.
- Server probes wait/retry on occupied test ports; interrupted short jobs no longer immediately pause
  behind the long job they were admitted between. Queue tests no longer leave isolated daemons behind.
- Public documentation now includes the new APIs and correct native Ollama model operations, supported
  decision heads and current release status.
- EmbeddingGemma 2 loader constructs published quantized layers through the pinned
  MIT mlx-vlm port, preferring the installed native loader when available.
- Quantized BERT-family sequence classifiers retain their trained head under strict
  loading. Jina v3 custom ranking heads are rejected rather than scored as Qwen yes/no.
- Qwen3-VL embedding handles empty inputs and keeps caller dictionaries unchanged.
  Format support and release shard impact are documented in `docs/PRIOR_ART_FORMATS.md`.

### Security

- Drop authorization/API-key/cookie headers when a download redirects to another origin; redact Realtime
  ephemeral secrets from logs. Transcription secrets cannot create model responses. Model downloads
  reject traversal-style repository IDs before disk access. Authenticated clients still share Files,
  stored completions and Evals data under one static token; this is not per-key isolation.
- HTTP video and optional rendered-page resources use checked fetch boundaries; unsupported redirects,
  private destinations and cross-origin rendered resources are rejected according to the fetch policy.
  See [API surface](docs/guides/API_SURFACE.md) and [Tavily limitations](docs/guides/TAVILY.md).

[Full changelog: v0.1.4…main](https://github.com/YuhuanStudio/Yunshu/compare/v0.1.4...main)

## [0.1.4] - 2026-10-08

Long conversations stay responsive and reusable, forced tool calls and Responses compaction behave the way agents
expect, EmbeddingGemma 2 adds image, audio and video embeddings, and two silent "random weights" loads are fixed.
Everything is in the `yunshu` package; the speed rows were measured on one M5 Max with Qwen3.8-27B.

### Highlights

- A short request no longer waits behind a long cold prefill: after each prefill step the scheduler decodes the other
  rows first, with a capped decode share for rows past their first tokens so a long reply cannot halve a prefill.
  Identity 0/36 mismatches, speed and memory neutral, long-context QA 30/30 in the long suite.
- Branch from the middle of a long session and hit the cache: the prefix cache keeps recurrent-state anchors, so a
  branch at 63.7K tokens of a 127K session answers in **0.62 s instead of 77.9 s** (a 161K session: 105 → 0.78 s).
  [Measurements](docs/reports/PERF_TREND.md).
- A forced `tool_choice` always yields a tool call on every API dialect (whitespace is bounded in the forced
  grammar, EOS is masked while reasoning, Hermes `$ref` is inlined); 40 runs per variant on two small models, 0 misses.
  Anthropic `any` / `tool` now turns thinking off unless the request enables it, as the Messages API does.
- Responses compaction reuses the prompt cache (about 99% hit on the history), folds histories larger than the
  window in up to 8 passes at safe tool-call boundaries (45K history under a 32K window works) and keeps streaming
  clients alive while it summarizes.
- New: EmbeddingGemma 2 embeds text, images, audio and video; served vectors match sentence-transformers fp32 at
  cosine ≥ 0.99985 over 33 cases.
- Fixed: Qwen3-Embedding checkpoints without the `model.` prefix loaded as random weights and served random
  vectors; they now load correctly or fail.

- Long replies at long context decode faster: the DFlash fast tree now stays on past 10K tokens and 256 generated
  tokens because a tree commit compacts the KV cache in place instead of copying it (64K: 16.0 → 2.4 ms per commit).
  Long suite: decode +7.7–21.9% at 32K–64K, +4.1% at 128K prose; identity 0/36 mismatches.

### Upgrade notes / breaking changes

- **`POST /v1/responses/compact`: `instructions` is now the conversation's system prompt** (as in `POST /responses`;
  Codex sends its base instructions here) and leads the summary request. It used to be handed to the summarizer as
  its own instruction. `tools` is now accepted and must be an array.
- Request text that contains an unpaired UTF-16 surrogate now answers HTTP 400 on every API (it used to fail inside
  the prompt hash or tokenizer, often after an agent cut tool output through an emoji).
- Weight loads no longer tolerate missing tensors. A checkpoint whose parameters would stay random now fails to
  load (Qwen3-Embedding, the MTP target and TTS loads). Re-download or fix a checkpoint that used to "load".
- `/v1/embeddings` answers 400 when image / audio / video inputs (`messages` or object items) go to a text-only
  embedder; `input` and `messages` together are rejected.
- Dependencies: mlx-vlm 0.7.6, mlx-audio 0.5.8, transformers 5.19, openai 3.26, diffusers 0.41, datasets 5.1.
  mlx is capped below 0.32.4 until the row-exact quantized-matvec copies follow mlx `f8aaf49d`. Run `yunshu doctor`
  after upgrading.
- Prometheus counters follow the `_total` convention: `yunshu_request_count` → `yunshu_http_requests_total`,
  `yunshu_inference_count` → `yunshu_inferences_total`, `yunshu_error_count` → `yunshu_errors_total`,
  `yunshu_mtp_total_cycles` → `yunshu_mtp_cycles_total`. Update dashboards and alerts.
- `--drain-timeout 0` now stops in-flight requests at once on shutdown (it used to wait forever).
- Requests that omit `max_tokens` now get up to `YUNSHU_DEFAULT_MAX_TOKENS` (default 32768, clamped by the
  context budget) instead of being cut at 512 tokens (2048 on Responses).
- Stricter validation (400 instead of a silent fallback): `tool_choice` naming an undeclared tool,
  `response_format: json_schema` without a schema, `stream_options` without `stream`, negative
  `prompt_logprobs`, conflicting or invalid `structured_outputs`.
- `YUNSHU_SPEC_TREE` is now a stable setting with default `auto` (see Performance); `off` keeps the chain verifier.
  Existing prefix-cache namespaces may be recomputed after the prefill and APC changes.

### Performance

M5 Max, Jundot/Qwen3.8-27B-oQ4e-mtp, separate experiments, not cumulative. Sources: [PERF_TREND](docs/reports/PERF_TREND.md).

| Machine | Model / mode | Metric / workload | Before → after | Recorded source |
|---|---|---|---|---|
| M5 Max | 27B | TTFT, branch at 63.7K of a 127K session | 77.9 → 0.62 s | PERF_TREND "apc2: 96K footprint-peak timeline" |
| M5 Max | 27B | TTFT, branch at 31.9K of that session | 34.2 → 0.47 s | same |
| M5 Max | 27B | TTFT, branch at 80.5K of a 161K session | 105 → 0.78 s | same |
| M5 Max | 27B | Follow-up TTFT, 127K linear | 1.41 → 0.92 s | same |
| M5 Max | 27B, DFlash | Decode, 1K code cold / APC warm (greedy, 64–256 token replies) | 109.7 → 120.5 / 107.0 → 112.9 tok/s | PERF_TREND "2026-10-04 wide6" |
| M5 Max | 27B, DFlash | Decode, 1K prose cold / APC warm | 54.4 → 62.8 / 50.0 → 59.4 tok/s | same |
| M5 Max | 27B, MTP | Cold TTFT, 8K / 32K prose or code | 8.55/8.43 → 7.77/7.79 s; 36.8/36.7 → 34.1/34.1 s (−6.9 … −9.2%) | merge ce5f4197, M5 Max and 27B shapes only |
| M5 Max | 27B | APC warm hit TTFT, 32K prose / code | 148 → 116 / 241 → 210 ms | PERF_TREND "native APC restore handles" |
| M5 Max | 27B | Follow-up TTFT, 8K code / 32K code | 500 → 494 / 703 → 681 ms | merge 02af4fc5 |
| M5 Max | 27B | Process footprint, 125K turn-1 peak / idle 35 s | 59.5–63.3 → 45.1–48.1 GiB / 50.8 → 29.0 GiB | PERF_TREND "server memory: peak and idle hold" |
| M5 Max | 27B, DFlash | Decode, 2048-token replies, 32K code / prose | 88.7 → 97.6 / 59.3 → 64.0 tok/s | PERF_TREND "2026-10-07 longgap" (yv long, 2 reps) |
| M5 Max | 27B, DFlash | Decode, 2048-token replies, 64K code / prose | 97.4 → 118.7 / 48.2 → 51.9 tok/s | same |
| M5 Max | 27B, DFlash | Decode, 2048-token replies, 128K prose | 40.1 → 41.6 tok/s | same (128K code 68.3 → 64.0 within ±10.8% noise; tfbench A/B 69.6 → 70.0) |
| M5 Max | 27B, DFlash | Verify round at 1K / 8K | 54.5 → 50.7 ms / 62.3 → 56.0 ms | merge 583edbeb |

Release gate, v0.1.3 → v0.1.4 (long suite, 27B with default settings, 2048-token replies; two full passes, ranges
cover both): decode at 32K / 64K / 128K +8.3–9.3% / +5.8–16.2% / +4.3–6.5%; cold TTFT −4.8% to −7.3%; warm TTFT
−0.9% to −37.5%; server peak memory 75.3 → 56.7–56.9 GiB; identity 0/36 mismatches and long-context QA 30/30 on both.

Regressions and limits: 8K code follow-up TTFT +1.7% (0.528 → 0.537 s) from the memory work; 32K turn-2 warm
TTFT is unchanged on the restore-handle change; the fast tree only applies to bounded 1K-class greedy requests
(32K +0.7–1.0%); one anchor holds about 0.28 GiB of recurrent state (about 3 anchors at 128K); on the M5 the apc2
long suite measured memory peak −0.9 GiB and idle +0.24 GiB. The dependency bump changed warm decode by −0.40% to
+0.84% (no claim).

### Added

- EmbeddingGemma 2: `/v1/embeddings` accepts object items `{text, image, audio, video}`, interleaving markers,
  `messages` in vLLM chat form, `task` / `instruction` prompts and Matryoshka `dimensions`; usage counts real tokens
  and undecodable media is a 400.
- `YUNSHU_SPEC_TREE=auto`: lossless fast DFlash tree for bounded short requests on certified M5 27B (see Performance);
  `YUNSHU_DRAFT_BITS` (default 8) exposes drafter bits (4 is token-identical).
- Every route the gateway registers has a real-server check (`scripts/dev/m3sweep`, `tests/unit/test_route_coverage.py`);
  `scripts/dev/agentbench` runs Claude Code, Codex and opencode on the 27B as one repeatable command.
- `GET /v1/responses/{id}/input_items`; Responses usage reports `cache_write_tokens`.
- `/debug/memory-census` names the objects holding MLX memory; the idle server returns freed GPU buffers after 30 s.
- Real rerankers and classifiers: Qwen3-Reranker and BGE-style cross-encoders on `/v1/rerank`, trained
  sequence-classification heads on `/v1/classify`, and `/v1/score` with vLLM semantics (`queries`/`documents`,
  `data_1`/`data_2`, `use_activation`, `instruction`). Scores match the Transformers recipe within 0.00021 on four
  checkpoints, same ranking.
- Ollama `/api/copy`, `/api/delete`, `/api/create` and `/api/pull` on a `--models-dir` server (names confined to that
  directory), freeform `custom` tools on Responses (Codex
  `apply_patch` shape), and Realtime sessions that load their model lazily.

### Changed

- Embedding text inputs may be up to 65536 characters (was 8192); the 2048-input cap is unchanged.
- Embeddings pool and normalise in float32 (bf16 vectors were not unit length).
- Auxiliary and uncached requests (such as titles) are scheduled after interactive turns on qualified models
  (main TTFT p90 8.60 → 7.62 s, six-turn completion −17%, identical tokens; other models stay FIFO).
- Follow-up prompts reuse the tokenizer prefix: only the new suffix is encoded (32K tokenize 17–25 ms → 2–3 ms).
- Dependencies refreshed as above.

### Fixed

- MCP notifications (JSON-RPC messages without an `id`) answer HTTP 204 with an empty body. They used to send
  `null`, uvicorn dropped the connection, and the client's next request on that connection failed.
- Speculative decoding off now produces the same tokens as on for the Qwen3.5 family (the batch-invariant kernels
  were only installed with a drafter; 27B 12/12 cells differed).
- A tool call inside unclosed reasoning reaches the client; `/v1/completions` non-stream reasoning is untagged like
  stream; prefill is excluded from `prompt_tokens` in one place.
- Messages on the text engine no longer silently fell back to prompt injection for a forced `tool_choice`; a flat
  forced function choice on Responses reaches the tool prompt.
- Server memory: finished requests release their KV at once (it sat in a reference cycle), the APC no longer holds
  two full copies while storing, and an APC peak of +11.7 GiB above 150K tokens is gone.
- `POST /v1/responses/input_tokens` and `POST /v1/messages/count_tokens` equal the real call's `usage`; cancelling a
  background response answers `cancelled`; `/v1/images/edits` and `/variations` accept multipart; OCR on a
  single-model server works (vision model) or answers 503 naming the fix (text model); `/v1/classify` on a model
  that cannot embed is 400; a `--models-dir` server with models registered is ready before the first load.
- Gemma 4 audio, Whisper translations and video (OpenCV, 400 instead of a silent drop); Ollama `show` card leak;
  Qwen3-Coder unclosed parameters; vLLM-style streaming tool-call cases.
- Metal event leak in the DFlash tree path.
- A forced `tool_choice` whose tool grammar cannot compile for the model (unsupported format, multi-token markers,
  recursive schema) answers 400 before streaming starts instead of decoding unconstrained.

### Security

- No security-policy change in this range.

[Full changelog: v0.1.3…v0.1.4](https://github.com/YuhuanStudio/Yunshu/compare/v0.1.3...v0.1.4)

## [0.1.3] - 2026-10-03

Faster structured responses, long prompts and follow-up turns on Apple Silicon, more
reliable coding-agent requests and prompt reuse, and correct batch-invariant decoding on
M1–M4 Macs.

### Highlights

- Get complete JSON responses faster: warm decode **23.4 → 111.4 tok/s** with
  identical output and logprobs in the recorded Qwen3.8-27B test. [Measurements](docs/BENCHMARKS.md#013-draft-measurements-2026-10-03).
- Start follow-up answers sooner: 32K code first-token latency **900 → 721 ms**;
  8K code **569 → 512 ms**. [Measurements](docs/BENCHMARKS.md#013-draft-measurements-2026-10-03).
- Speed up repetitive follow-up code: raising the verified prompt-copy cap from
  8 to 16 improved the measured 8K code turn from **106.1 → 129.2 tok/s**.
  [Measurements](docs/BENCHMARKS.md#013-draft-measurements-2026-10-03).
- Generate repetitive code faster with verified prompt copies: DFlash 32K code
  **72.74 → 82.56 tok/s**; prose results vary by context.
  [Measurements](docs/BENCHMARKS.md#013-draft-measurements-2026-10-03).
- Correct speculative decoding on M1–M4 Macs: output now matches plain decoding there
  too, with M5 speed unchanged (27B, 20 cells digest-equal, −0.4% to +3.0%).

### Upgrade notes / breaking changes

- Upgrade dependencies together: minimums now include `mlx-lm>=0.32.0`,
  `llguidance>=1.9.1` and `transformers>=5.18.0`. Run `yunshu doctor` after upgrading.
- JSON-schema and `json_object` requests now use llguidance by default.
  `YUNSHU_JSON_SCHEMA_ENGINE` selects the alternative engine; unsupported
  constraints remain explicit errors.
- Overlong text prompts now return an error instead of silently losing their
  beginning. Invalid grammars return HTTP 400 before generation.
- Cold-prefill arithmetic changed; existing cached prompts may be recomputed in
  a new cache namespace. Decode verification remains exact for qualified paths.
- `YUNSHU_SPEC_COPY_ROWS` now defaults to 16 on certified models (8 on narrower
  backends); use 8 to retain the previous cap or 0 to disable prompt copying.
  The measured prose tradeoff for the cap change was −0.2% to −1.0%.
- M1–M4 Macs: the batch-invariant decode kernels used by speculative decoding now
  compute correct results there (some output columns were wrong before, so speculative
  output could differ from plain decoding). No action needed; M5 kernels are unchanged.

### Performance

All rows use M5 Max and Jundot/Qwen3.8-27B-oQ4e-mtp. These are separate
same-checkpoint experiments, not cumulative gains or promises for every model.
October 3 results use three interleaved clean repetitions. Sources and limitations:
[BENCHMARKS](docs/BENCHMARKS.md#013-draft-measurements-2026-10-03).

| Machine | Model / mode | Metric / workload | Before → after | Recorded source |
|---|---|---|---|---|
| M5 Max | Qwen3.8-27B, MTP | Cold TTFT, 8K / 32K | 11.0 → 8.57 s / 47.5 → 38.3 s | `a4d71bc8`, Oct 2 prefill |
| M5 Max | Qwen3.8-27B, MTP | Follow-up TTFT, 8K / 32K code | 569 → 512 ms / 900 → 721 ms | `d624e52d`, Oct 3 singleton capacity |
| M5 Max | Qwen3.8-27B, DFlash | Complete JSON warm decode | 23.4 → 111.4 tok/s | `dad4641c`, Oct 3 complete-output run |
| M5 Max | Qwen3.8-27B, DFlash | Complete tool-call warm decode | 23.0 → 77.7 tok/s | Same complete-output run |
| M5 Max | Qwen3.8-27B, DFlash | Cold code decode, 8K / 32K | 74.50 → 82.19 / 72.74 → 82.56 tok/s | `78c03c94`, Oct 3 prompt-copy islands |

### Added

- Qualified JSON-schema, grammar and tool-call requests can use speculative
  decoding while preserving accepted output tokens and requested logprobs.
- Concurrent requests can share one initial prompt computation; Anthropic
  `cache_control` and OpenAI `cached_tokens` reflect the rendered prompt boundaries.
- Optional extra RAM and disk prompt-cache tiers, inspected by `yunshu doctor`:
  `YUNSHU_VLM_APC_WARM`, `YUNSHU_VLM_APC_WARM_SHARE`,
  `YUNSHU_VLM_APC_DISK_TIERS`, `YUNSHU_VLM_APC_DISK_ENCODING`.
  Compressed lossless RAM is off; int8/int4 remain lossy opt-ins.
- Optional local numbers-only serving logs (`YUNSHU_SERVE_LOG`) and per-request
  speculative `rounds` counters; logs contain no prompts, outputs or token IDs.

### Changed

- Long-prompt processing and cached follow-up turns do less allocation and copying;
  prompt checkpoint writes no longer delay the first token on eligible requests.
- MTP and greedy DFlash can verify copied prompt passages; experimental tree
  decoding remains off. No universal code or prose speedup is claimed.
- Dependency minimums match the refreshed lock; text-serving internals were
  reorganized without changing their supported import surface.

### Fixed

- Claude Code requests with mixed system/developer messages no longer fail with
  HTTP 500; streamed tool and reasoning responses retain all requested logprobs.
- Models stay loaded until responses and WebSockets finish, including cancellation;
  media context limits account for the processor's actual token costs.
- External draft generation follows the target sampling distribution and committed
  conversation prefix; experimental tree requests reach the selected mode.
- Growing conversations remove superseded disk checkpoints safely; transient I/O
  errors retain valid files and active readers are protected from deletion.
- Settings paths expand `~`, including defaults (`1fb15d20`); Realtime speech stops
  when the user interrupts, and malformed `web_fetch` input gives actionable errors.
- M1–M4 Macs: speculative decoding output equals plain decoding again; the invariant
  matmul and attention kernels are chosen once per GPU generation and checked with
  real data on GPUs they were not written for.
- Coding-agent tool loops keep earlier reasoning when the client does not send it back,
  accept empty tool arguments, keep tool-call argument types from the tool schema in
  history, and end GLM tool-result turns correctly.

- macOS virtual machines (Apple Paravirtual GPU, e.g. hosted CI runners): the server uses the stock
  runner there, without the prefix cache or speculative decoding, because that virtual GPU's
  arithmetic differs by prefill span; physical Apple GPUs are unaffected.

### Security

- No separate security-policy change in this range. Model leases and cache-reader
  protection strengthen request lifecycle safety; see [SECURITY.md](SECURITY.md).

[Full changelog: v0.1.2…v0.1.3](https://github.com/YuhuanStudio/Yunshu/compare/v0.1.2...v0.1.3)

## [0.1.2] - 2026-10-02

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
