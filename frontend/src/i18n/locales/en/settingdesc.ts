const settingdesc = {
  YUNSHU_MODEL:
    "Model path or Hugging Face id served in single-model mode; every requested model name maps to it.",
  YUNSHU_MULTI_MODEL:
    "Multi-model mode: discover models under `YUNSHU_MODELS_DIR` and load them on demand. Ignored when `YUNSHU_MODEL` is set.",
  YUNSHU_MODELS_DIR:
    "Directory of model folders for multi-model mode (folder name is the model id). Unset: `~/.yunshu/models`, which is also where `yunshu pull` downloads to.",
  YUNSHU_HF_CACHE_MODELS:
    "Multi-model mode: also offer models already in the Hugging Face cache (the models directory wins on name clashes). `yunshu serve -m org/name` uses a cached copy either way.",
  YUNSHU_MODEL_TTL_SECONDS:
    "Multi-model mode: unload a model idle for this many seconds. Unset: never.",
  YUNSHU_ALLOW_AUTO_LOAD:
    "Multi-model mode: let audio requests load a model that is not loaded yet (otherwise they are rejected).",
  YUNSHU_MAX_LORAS: "Maximum LoRA adapters kept loaded (text models).",
  YUNSHU_TRUST_REMOTE_CODE:
    "Allow Hugging Face tokenizers/processors that need `trust_remote_code`.",
  YUNSHU_WARM_PROMPTS:
    "Prompts prefilled at startup to warm the prefix cache: text or file paths separated by `||`.",
  YUNSHU_CONFIG:
    "TOML config file with `YUNSHU_*` settings (lower precedence than the environment).",
  YUNSHU_MAX_CONCURRENT:
    "Cap on concurrently admitted requests. Unset: adaptive, starting at 8.",
  YUNSHU_UNCACHED_SCHEDULING:
    "Order VLM prefill atoms by estimated uncached work and wait time, protect interactive prefills, and reserve measured decode time between atoms, without changing any token span. On by default only with qualified batch-invariant kernels; other backends keep FIFO unless enabled. Disable to restore upstream ordering.",
  YUNSHU_AUXILIARY_SCHEDULING:
    "Deprioritize captured opencode title requests: interactive VLM jobs run first, and auxiliary rows pause between GPU slices and never read or write the APC. On by default only with qualified batch-invariant kernels; other backends need explicit opt-in. In-flight GPU operations cannot be interrupted.",
  YUNSHU_QUEUE_LIMIT:
    "Generation requests in flight at once, running and waiting together; the next one is refused at once with 429, `Retry-After` and the queue depth in `error.x_yunshu`. 0 = no limit.",
  YUNSHU_MEMORY_PRESSURE_REJECT:
    "Share of the Metal working set above which a new generation request is refused with 503 and `Retry-After` while other requests are running, to avoid an OOM. An idle server never refuses. 0 = off.",
  YUNSHU_COMPLETION_BATCH_SIZE:
    "Text engine: maximum sequences decoded together.",
  YUNSHU_DEFAULT_MAX_TOKENS:
    "Completion length when a chat or responses request omits `max_tokens`; the context budget still clamps it.",
  YUNSHU_MAX_PREFILL_TOKENS:
    "Reject prompts longer than this many tokens (0: no limit beyond the model context).",
  YUNSHU_STARTUP_TIMEOUT:
    "Seconds to wait for the model to load before startup fails.",
  YUNSHU_DRAIN_TIMEOUT: "Seconds to wait for in-flight requests on shutdown.",
  YUNSHU_KEEP_ALIVE_TIMEOUT: "Seconds an idle HTTP connection stays open.",
  YUNSHU_UDS:
    "Serve on this Unix domain socket instead of a TCP port (same app; `curl --unix-socket`, httpx `uds=`).",
  YUNSHU_WS_MAX_INFLIGHT:
    "Text WebSocket (`/v1/stream`, wss `/v1/responses`): maximum concurrent requests per connection.",
  YUNSHU_WS_PING_INTERVAL:
    "Text WebSocket: seconds between server heartbeat pings (0 disables).",
  YUNSHU_WS_SEND_QUEUE:
    "Text WebSocket: outbound events buffered per connection before generation is paused (backpressure).",
  YUNSHU_MAX_REQUEST_SIZE: "Maximum request body size in bytes.",
  YUNSHU_PROGRESS_INTERVAL_S:
    "Streaming chat/completions: seconds between `: yunshu-progress` SSE comments (queue and prefill progress, ETA) before the first token; 0 turns them off. Strict SSE clients ignore comment lines.",
  YUNSHU_SLOW_REQUEST_THRESHOLD:
    "Log a warning for requests slower than this many seconds.",
  YUNSHU_CORS_ORIGINS:
    "Comma-separated allowed CORS origins (`*` for any); also checked against the Realtime WebSocket Origin header.",
  YUNSHU_RESPONSE_CACHE: "Cache identical non-streaming responses in memory.",
  YUNSHU_BATCH_MAX_ITEMS: "Batch API: maximum requests per batch.",
  YUNSHU_BATCH_TIMEOUT: "Batch API: default per-batch timeout in seconds.",
  YUNSHU_ALLOW_LOCAL_FILES:
    "Allow requests to reference any local file path (default: only under `YUNSHU_MEDIA_DIR`).",
  YUNSHU_MEDIA_DIR:
    "Directory that local media paths must live under. Unset: `$TMPDIR/yunshu_media`.",
  YUNSHU_FILES_DIR:
    "Directory of the local Files / Batch API store. Unset: `~/.yunshu/files`.",
  YUNSHU_FILES_MAX_BYTES:
    "Files API: maximum size of one uploaded file in bytes (default 512 MB).",
  YUNSHU_FILES_TTL_DAYS:
    "Files API: delete uploaded files after this many days. Unset: keep forever.",
  YUNSHU_FILES_MAX_TOTAL_BYTES:
    "Files API: total bytes the store may hold; an upload that would exceed it fails with 413 `storage_quota_exceeded` (expired files are reaped first). 0: unlimited.",
  YUNSHU_CONVERSATIONS_DIR:
    "Directory of the Conversations API store (JSON, one file per conversation). Unset: `~/.yunshu/conversations`.",
  YUNSHU_CHAT_COMPLETIONS_DIR:
    "Directory of stored chat completions (`store=true`; JSON, one file per completion). Unset: `~/.yunshu/chat_completions`.",
  YUNSHU_CHAT_COMPLETIONS_MAX:
    "Stored chat completions kept; the oldest are evicted past this count.",
  YUNSHU_CONVERSATION_MAX_ITEMS:
    "Conversations API: maximum number of items one conversation may hold.",
  YUNSHU_COMPACT_MAX_TOKENS:
    "Responses compaction: maximum tokens of the model-written summary.",
  YUNSHU_AUTH_TOKEN:
    "Bearer token. When set, every request except health, version and docs needs it; unset: inference is open and operational endpoints are denied.",
  YUNSHU_AUTH_DISABLED:
    "Disable auth entirely (operational endpoints open too). Local development only.",
  YUNSHU_DEBUG_ROUTES:
    "Mount the `/debug/*` diagnostic routes (engine, system, kv-cache, spec-decode, ...). They need the auth token or `YUNSHU_AUTH_DISABLED`. `/metrics` is always mounted.",
  YUNSHU_DEBUG_STREAM_CAPTURE:
    "Debugging aid: append one JSON line per VLM-runner generation to this file, with the generated token ids, the text pieces handed to the gateway and their joined text, so a delivered answer can be compared with what was generated. Off when unset.",
  YUNSHU_ACTOR_IDENTITY:
    "Identity recorded for authenticated requests in the audit log.",
  YUNSHU_RATE_LIMIT_RPM:
    "Per-client request rate limit in requests per minute; 0 (default) turns it off. A local single-user engine does not need it; set it when the server is exposed to other machines.",
  YUNSHU_TRUSTED_PROXIES:
    "Comma-separated proxy IPs whose `X-Forwarded-For` header is trusted.",
  YUNSHU_FOOTPRINT_SAMPLE_MS:
    "Sample this process’s `phys_footprint` every N ms on a background thread and export the peak as `yunshu_process_footprint_bytes` on `/metrics`. 0 (default) = off.",
  YUNSHU_MAX_MEMORY_GB:
    "Multi-model mode memory ceiling in GiB, e.g. `48` or `48GB`; `disabled` turns the enforcer off. Unset: 80% of unified memory.",
  YUNSHU_PREFILL_STEP_SIZE:
    "Text engine: prompt tokens per prefill forward pass; lower it to cap the prefill activation peak on small-memory machines.",
  YUNSHU_MEM_PRESSURE_THRESHOLD:
    "Text engine: evict prefix-cache entries above this memory use (percent, or a fraction <= 1).",
  YUNSHU_PREFIX_MAX_ENTRIES: "Text engine: number of prefix KV cache entries.",
  YUNSHU_PREFIX_HOT_LIMIT:
    "Text engine: keep only this many prefix KV entries at full precision and store older ones in 4-bit in RAM (lossy on reuse; memory vs quality). 0 = every entry full precision.",
  YUNSHU_SSD_CACHE: "Text engine: persist prefix KV to SSD.",
  YUNSHU_SSD_CACHE_DIR: "Text engine: SSD prefix-cache directory.",
  YUNSHU_SSD_CACHE_PRECISION:
    "Text engine: SSD prefix-cache precision: `native` (bit-exact KV and recurrent state; lossless) or `int8` (per-tensor int8, about half the disk bytes of bf16; lossy on reuse).",
  YUNSHU_SSD_CACHE_PREFILL_CEIL_TPS:
    "Text engine: skip an SSD prefix restore when the observed prefill speed exceeds this (tok/s), since re-prefilling is then as fast as reading the KV back.",
  YUNSHU_SSD_CACHE_MAX_GB:
    "Text engine: SSD prefix-cache size cap in GiB, one budget for the whole directory (all models together).",
  YUNSHU_CACHE_RESERVE_PCT:
    "SSD prefix caches (APC and text): free space, as a percentage of the volume, that no cache write may eat into. The reserve is the larger of this and `YUNSHU_CACHE_RESERVE_GB`, and also limits a cache root’s effective cap.",
  YUNSHU_CACHE_RESERVE_GB:
    "SSD prefix caches (APC and text): minimum free space in GiB left on the volume (see `YUNSHU_CACHE_RESERVE_PCT`). A write that would leave less is dropped and spilling pauses until space is back.",
  YUNSHU_CACHE_STALE_DAYS:
    "SSD prefix caches (APC and text): a checkpoint namespace unused for this many days, or whose checkpoint no longer exists or has changed, is removed before anything else is evicted (0 = never by age).",
  YUNSHU_KV_QUANT_BITS:
    "Text engine KV cache quantization (lossy; memory vs quality): `off` (lossless), `auto` (8-bit once the KV cache would exceed about 2 GiB), or 2/3/4/8 bits always.",
  YUNSHU_VLM_APC_MEMORY_GB:
    "VLM runner prefix cache (APC) RAM budget in GiB; 0 disables it. Unset: half of the memory left after weights and an OS/activation reserve, capped at a quarter of the machine and 32 GiB, and off when under 1 GiB would be left. A 27B checkpoint costs about 130 KiB per cached token.",
  YUNSHU_VLM_APC_DISK:
    "APC SSD tier: prefix checkpoints that RAM evicts (and, at shutdown, those still resident) are written to disk and read back instead of re-prefilling (bit-exact, lossless; a 27B checkpoint reloads about 20x faster than it prefills). Set 0 to keep the prefix cache in RAM only.",
  YUNSHU_VLM_APC_DISK_DIR:
    "Directory of the APC SSD tier. Unset: `~/.yunshu/cache/apc` (internal disk). Put it on a fast volume to keep the internal disk clean.",
  YUNSHU_KV_PRECISION:
    "KV cache precision of the Qwen3.5-family runner’s shared decode batch: `bf16` (lossless) or `int8` (about 0.53x the KV memory and read bandwidth for a small attention error; memory vs quality). A lone request and the speculative lane stay bf16.",
  YUNSHU_VLM_APC_DISK_GB:
    "Size cap of the APC SSD tier in GiB, one budget for the whole directory (least recently used files go first across namespaces; 0 = no configured cap, the free-space reserve still applies). Unset: a quarter of the volume, at most 64 GiB. A 27B checkpoint costs about 130 KiB per token, so 64 GiB holds about 500K tokens.",
  YUNSHU_PREFILL_MATMUL:
    "Qwen3.5-family VLM runner on M5-class GPUs: matmul for prefill chunks over 512 rows. `stock` runs MLX’s quantized matmul on the untiled weight (cold 27B prefill about 25% faster; a chunk’s bits depend on its row count); `lane` keeps the row-invariant lane kernel for every row count. Part of every APC key and SSD namespace.",
  YUNSHU_PREFILL_BUFFER_CACHE_GB:
    "VLM runner: GiB of MLX’s freed-buffer cache kept across prefill steps. Unset: 5% of physical RAM, at most 6 GiB. Keeps long-prefix restore buffers warm across requests; 0 = upstream clear after every chunk. Allocator only; output unchanged.",
  YUNSHU_PREFILL_GDN:
    "Qwen3.5-family VLM runner: GatedDeltaNet core for prefill chunks of 64 or more tokens. `chunked` uses MLX’s `mx.fast.gated_delta_update` (2.7x faster per layer; within bf16 rounding of `step`); `step` keeps mlx-vlm’s per-token kernel. Part of every APC key and SSD namespace.",
  YUNSHU_VLM_APC_DISK_TIERS:
    "Further APC storage tiers below the SSD directory, comma-separated `PATH[@GiB]` (external SSD, HDD, NAS). Volumes are profiled at startup and ordered by read speed; evicted checkpoints move down instead of being deleted, and a tier serves a hit only when restoring beats re-prefilling. Unmounted tiers are skipped. Lossless; unset: SSD only.",
  YUNSHU_VLM_APC_DISK_ENCODING:
    "How lower APC storage tiers (`YUNSHU_VLM_APC_DISK_TIERS`) hold a checkpoint: `raw` (copy of the SSD file), `zstd` (lossless byte-plane shuffle plus zstd), or `auto` (zstd only where it measurably raises effective read bandwidth, such as slow disks and network shares). Never lossy.",
  YUNSHU_VLM_APC_WARM:
    "APC WARM tier: what happens to a prefix checkpoint leaving the RAM (HOT) tier before it goes to SSD. `off`: straight to SSD; `lossless`: kept compressed in RAM (bit-exact, costs CPU); `int8` / `int4`: attention K/V kept as quantized codes (lossy; the SSD tier keeps exact states). The WARM tier takes `YUNSHU_VLM_APC_WARM_SHARE` of the APC RAM budget.",
  YUNSHU_VLM_APC_WARM_SHARE:
    "Share of the APC RAM budget (`YUNSHU_VLM_APC_MEMORY_GB`) taken by the WARM tier when `YUNSHU_VLM_APC_WARM` is on; the HOT tier keeps the rest. Total APC RAM does not grow.",
  YUNSHU_VLM_MAX_IMAGE_BYTES:
    "Largest image a request may reference by URL, in bytes.",
  YUNSHU_VLM_INSECURE_SSL:
    "Retry image downloads without TLS verification when verification fails.",
  YUNSHU_MTP:
    "Qwen3.5-family VLMs: draft with the checkpoint’s MTP head (batch-invariant; speculative on equals off).",
  YUNSHU_VLM_DRAFT:
    "Qwen3.5-family VLMs: speculative draft override. A DFlash drafter directory; `mtp` forces the checkpoint MTP head; `off` disables drafting. Unset: a matching DFlash2 drafter in the models dir or Hugging Face cache is used automatically, else the MTP head.",
  YUNSHU_MTP_BLOCK_SIZE:
    "Draft block size (for DFlash, the ceiling its acceptance-driven depth stays under). Unset: 6 for MTP, the drafter’s trained block for DFlash.",
  YUNSHU_SPEC_COPY_ROWS:
    "Qwen3.5-family single-request speculative lane: verify rows a prompt-copy round may use (copy drafts = rows - 1). A copy round proposes the continuation of the longest earlier occurrence of the current tail in the prompt plus generated text, and the same verify checks it, so output is unchanged. Agent, code-editing and multi-turn traffic that quotes its context commits several times more tokens per round. Default 16 rows, capped to the backend’s certified width; 0 turns copy rounds off.",
  YUNSHU_DRAFT_BITS:
    "Qwen3.5-family DFlash drafter weight bits: 8 (default), 4, or 0 to keep the shipped bf16. Drafts are verified by the target, so output is token-identical for every value; fewer bits cut the drafter’s bytes per round but can lower acceptance.",
  YUNSHU_SPEC_TREE:
    "Qwen3.5-family single-request speculative lane: `off` keeps trained chain plus copy; `tree` forces the draft-tree verifier; `auto` uses the certified M5 Q4 DFlash2 fast tree for bounded 1K-class greedy requests and chain plus copy elsewhere. Greedy tokens stay identical to plain decode.",
  YUNSHU_NGRAM_DEFAULT:
    "Text models: lossless n-gram speculation on greedy requests by default (per-request `spec_decode` also enables it). Wins on repetitive output.",
  YUNSHU_SPEC_PROPOSER:
    "Text models: speculative proposer family for n-gram speculation.",
  YUNSHU_GEMMA4_ASSISTANT:
    "Text Gemma-4 models: assistant drafter directory (KV-shared speculative drafter).",
  YUNSHU_GPU_SAMPLER:
    "Text models: on-GPU Gumbel-max sampling (no per-token GPU-to-CPU sync).",
  YUNSHU_JUMP_FORWARD:
    "Text models: emit grammar-forced structural tokens of JSON-schema output without a forward pass.",
  YUNSHU_JSON_SCHEMA_ENGINE:
    "Engine that masks JSON-schema and `json_object` constrained decoding: `llguidance` (about 0.25 ms median per-token mask on a 248K vocabulary; properties in schema order) or `inhouse` (Python state machine, about 3 ms median; any property order). A schema llguidance cannot compile falls back to the in-house engine when it supports it.",
  YUNSHU_GRAMMAR_BITMASK:
    "Constrained decoding with the xgrammar-style bitmask engine instead of the allowlist sampler.",
  YUNSHU_TOOL_GRAMMAR:
    "Tool-call constrained decoding (structural tags): free text until the tool-call start marker, then the call body is masked to this request’s exact call grammar (tool name, parameter keys, typed values, closing). A forced `tool_choice` is always constrained; this flag governs only `auto`. Off: auto tool calls are decoded unconstrained and repaired afterwards.",
  YUNSHU_QUANT_MODE:
    "Quantize weights in memory at load (lossy; memory vs quality): `mxfp4`, `nvfp4`, `mxfp8` or `affine` (empty keeps the checkpoint).",
  YUNSHU_QUANT_CONFIG:
    "Bits and group size for `affine` in-memory quantization: JSON (with `bits` and `group_size` keys) or `bits` / `bits,group`.",
  YUNSHU_ROUND_PREFILL_CHUNK:
    "Round driver: prompt tokens per prefill span. A decoding request only steps between prefill forwards, so smaller spans keep it running next to a long prompt, at about 20% lower prefill speed. Spans are fixed per prompt, so output is independent of what else is running; prompts prefilled with different sizes are self-consistent but not bit-identical to each other.",
  YUNSHU_ROUND_DRIVER:
    "Dense Qwen3.5-family VLMs: serve text requests with Yunshu’s round driver (packed forwards over every decoding row, alternating with batched prefill steps; MTP drafts for every row; position-keyed sampling; see `docs/guides/ROUND_DRIVER.md`). Off: upstream BatchGenerator plus the single-request speculative lane. Image prompts, int8 KV and MoE stay on the upstream path either way.",
  YUNSHU_MTP_ROW_EXACT:
    "Qwen3.5-family runner: oMLX row-exact verify (verify rows bit-identical to one-row decode) instead of batch-invariant kernels.",
  YUNSHU_ENGINE_LOOP:
    "Text models: EngineCore continuous-batching loop instead of the single-request fast path.",
  YUNSHU_OVERLAP:
    "Text engine loop: overlap CPU and GPU work (`cpu_gpu`) or split a batch into two overlapping halves (`two_batch`).",
  YUNSHU_SPEC_UNVERIFIED:
    "Text models: explicitly enable the unverified external mlx-lm draft experiment. `eagle` is the historical route name; an ordinary LM draft works, no trained EAGLE head needed. Only greedy, non-streaming requests with default sampling and penalties; schema, custom processors, stop strings, thinking budgets, LoRA and token-mask options keep the regular fast path.",
  YUNSHU_DRAFT_MODEL:
    "External mlx-lm draft checkpoint for `YUNSHU_SPEC_UNVERIFIED=eagle` (an ordinary LM draft, not a trained EAGLE head).",
  YUNSHU_REALTIME_OMNI:
    "Native Qwen3-Omni speech on the Realtime socket: `auto` when a speakable model is served, `on`, or `off` (ASR, LLM, TTS cascade).",
  YUNSHU_OMNI_MODEL:
    "Voice path model when it differs from the served model (an omni served model is reused automatically).",
  YUNSHU_OMNI_PRELOAD:
    "Warm the omni model at boot so the first voice request is not cold.",
  YUNSHU_OMNI_THINKER_MAX:
    "Maximum tokens the omni Thinker writes per voice turn (the spoken reply’s length cap).",
  YUNSHU_OMNI_PERSONA:
    "Realtime system persona when the request has none; empty string disables it. Unset: a built-in concise spoken style.",
  YUNSHU_REALTIME_SILENCE_MS:
    "Server VAD: pause before the model answers, in ms.",
  YUNSHU_REALTIME_BARGE_IN_MS:
    "Sustained speech needed to interrupt the model mid-reply, in ms.",
  YUNSHU_REALTIME_VAD_THRESHOLD: "Server VAD speech-detection threshold.",
  YUNSHU_REALTIME_PREFIX_PADDING_MS:
    "Server VAD: audio kept before detected speech, in ms.",
  YUNSHU_REALTIME_VAD: "Server VAD implementation: `energy` or `silero`.",
  YUNSHU_REALTIME_VAD_MODEL:
    "Silero VAD model id when `YUNSHU_REALTIME_VAD=silero`.",
  YUNSHU_REALTIME_MAX_INPUT_AUDIO_BYTES:
    "Realtime: largest buffered input audio, in bytes.",
  YUNSHU_REALTIME_MAX_CONVERSATION_ITEMS:
    "Realtime: conversation items kept per session.",
  YUNSHU_DIFFUSION_SCHEDULER:
    "Image generation sampler override (empty uses the pipeline’s own).",
  YUNSHU_ANE_EMBEDDINGS:
    "Compute embeddings on the Apple Neural Engine via CoreML when available.",
  YUNSHU_ANE_EMBEDDING_MODEL: "Embedding model used on the ANE path.",
  YUNSHU_MCP_CONFIG: "MCP client config file (JSON/YAML) listing tool servers.",
  YUNSHU_MCP_SERVERS:
    "MCP tool servers as a JSON array (alternative to `YUNSHU_MCP_CONFIG`).",
  YUNSHU_WEB_SEARCH_PROVIDER:
    "Search backend for the server-side `web_search` tool. `auto` picks the first configured of searxng, brave, tavily, exa; `none` disables. Unconfigured: requests get the API’s `unavailable` error with a hint.",
  YUNSHU_SEARXNG_URL:
    "Base URL of a self-hosted SearXNG instance (JSON output enabled), e.g. `http://127.0.0.1:8080`. The recommended privacy-friendly default.",
  YUNSHU_BRAVE_API_KEY: "Brave Search API key.",
  YUNSHU_TAVILY_API_KEY: "Tavily API key.",
  YUNSHU_EXA_API_KEY: "Exa API key.",
  YUNSHU_WEB_SEARCH_RESULTS: "Results returned per `web_search` call.",
  YUNSHU_WEB_FETCH:
    "Serve the server-side `web_fetch` tool (needs no provider). Off: `web_fetch` requests get an `unavailable` error.",
  YUNSHU_WEB_FETCH_ALLOW_PRIVATE:
    "Let `web_fetch` reach private, loopback and link-local addresses. Off (default) blocks them, including after redirects and DNS resolution (SSRF protection).",
  YUNSHU_WEB_FETCH_MAX_BYTES: "Largest response body `web_fetch` downloads.",
  YUNSHU_WEB_FETCH_TIMEOUT: "Seconds `web_fetch` waits for a page.",
  YUNSHU_WEB_FETCH_MAX_TEXT_CHARS:
    "Extracted page text handed to the model is cut to this many characters (a request’s `max_content_tokens` can lower it).",
  YUNSHU_MCP_CONNECTOR:
    "Serve the MCP connector: Anthropic `mcp_servers` and OpenAI Responses `mcp` tools are executed by this server, which connects to the named MCP servers over streamable HTTP / SSE.",
  YUNSHU_MCP_CONNECTOR_ALLOW_PRIVATE:
    "Let the MCP connector reach private and loopback MCP servers (local tool servers are the common case). Off: only public addresses.",
  YUNSHU_MCP_CONNECTOR_TIMEOUT:
    "Seconds an MCP connector call (initialize, `tools/list`, `tools/call`) may take, DNS included.",
  YUNSHU_MCP_CONNECTOR_MAX_BYTES:
    "Largest single reply (JSON body, or one SSE event) an MCP connector server may send, after decompression.",
  YUNSHU_SERVER_TOOL_MAX_ITERATIONS:
    "Most generate, run-tool, continue rounds one request may take.",
  YUNSHU_MODEL_ALIASES:
    "Multi-model mode: map model names agents ask for (`claude-sonnet-4-5`, `opus`, `gpt-5`) onto a served model, as a JSON object of pattern to served model id; patterns are exact names, `prefix*` or `*` (first match wins; a real model name always wins). Single-model mode answers to every name already.",
  YUNSHU_LOG_LEVEL:
    "Log level for Yunshu’s loggers (third-party loggers stay at WARNING).",
  YUNSHU_AUDIT_LOG_FILE: "Also write the audit log to this file.",
  YUNSHU_LOG_MAX_MB:
    "Service log (launchd): rotate the log file at this size in MiB; 0 turns size rotation off.",
  YUNSHU_LOG_ROTATE_HOURS:
    "Service log: also rotate when this many hours have passed since the last rotation; 0 turns time rotation off.",
  YUNSHU_LOG_KEEP:
    "Service log: number of rotated files kept (gzip, secrets redacted).",
  YUNSHU_LOG_RETENTION_DAYS:
    "Service log: delete rotated files older than this many days; 0 keeps them until `YUNSHU_LOG_KEEP` prunes them.",
  YUNSHU_SERVE_LOG:
    "Write one numbers-only JSON line per finished generation request (timings, token counts, speculative acceptance, cache tier, concurrency, arm, build) to a local, size-capped, rotated file. Never prompts, outputs or token ids. Off by default; nothing leaves the machine.",
  YUNSHU_SERVE_LOG_DIR: "Directory of the serve log. Unset: `~/.yunshu/logs`.",
  YUNSHU_SERVE_LOG_MAX_MB:
    "Serve log: rotate at this size in MiB; with `YUNSHU_SERVE_LOG_KEEP` the directory is capped at max * (keep + 1).",
  YUNSHU_SERVE_LOG_KEEP: "Serve log: number of rotated files kept.",
  YUNSHU_ARM:
    "Label recorded in the serve log for the configuration arm this server runs (for offline A/B analysis); it changes no behaviour.",
  YUNSHU_GATEWAY_URL: "Server URL used by the `yunshu` CLI client commands.",
  YUNSHU_HF_ENDPOINT:
    "Hugging Face Hub endpoint for `yunshu serve` (exported as `HF_ENDPOINT`).",
  YUNSHU_HISTORY_INTERVAL_S:
    "Console history: seconds between samples of the in-memory ring behind GET /v1/yunshu/history (throughput, request counts, memory, TTFT percentiles); 0 turns sampling off. The ring has a fixed size and never grows.",
  YUNSHU_HISTORY_HOURS:
    "Console history: hours the history ring keeps (capacity = hours × 3600 ÷ sample interval, allocated once; 12 h at 5 s is about 0.4 MiB).",
  YUNSHU_EVALS_DIR:
    "Folder of the local Evals API JSON store; unset: `~/.yunshu/evals`.",
  YUNSHU_VLM_MAX_VIDEO_BYTES:
    "Largest video a request may reference by URL, in bytes.",
  YUNSHU_WEB_SEARCH_PROVIDER_TIMEOUT:
    "Per-provider deadline of the combined web search, in seconds; a slow provider cannot hold up the whole query.",
  YUNSHU_WEB_SEARCH_HEALTH_FILE:
    "Small snapshot of provider health (no queries, results or credentials) read by `yunshu config`.",
  YUNSHU_WEB_MWMBL:
    "Adds Mwmbl's open small-web index to automatic web search (CC-BY-NC-SA 4.0 data, noncommercial); asking for provider=mwmbl also opts in.",
  YUNSHU_WEB_KEYLESS:
    "Allows keyless DuckDuckGo (best effort, may be blocked) and Wikipedia search; the query text and IP leave this machine.",
  YUNSHU_MOJEEK_API_KEY:
    "API key for Mojeek's independent index; once set it joins automatic web search.",
  YUNSHU_MARGINALIA_API_KEY:
    "Explicit opt-in to Marginalia small-web search; public API data is CC-BY-NC-SA 4.0 (noncommercial), commercial keys have their own terms, and there is no implicit public key.",
  YUNSHU_SERPER_API_KEY: "API key for Serper (Google search results).",
  YUNSHU_PERPLEXITY_API_KEY:
    "API key for Perplexity Search (raw results, not Sonar).",
  YUNSHU_WEB_RESEARCH:
    "Enriches search snippets with origin pages, untrusted excerpts and local ranking; a stable opt-in pending quality evaluation; fetched URLs leave this machine.",
  YUNSHU_WEB_RENDER:
    "Local Chromium fallback for pages that are only a JavaScript shell in advanced Tavily extract/crawl/map (needs the web-render extra and an installed Playwright Chromium; same-origin GET only, no cookies, never downloads a browser).",
  YUNSHU_WEB_RESEARCH_BUDGET:
    "Overall deadline for enriching search snippets, in seconds (at most 4).",
  YUNSHU_WEB_RESEARCH_PAGES:
    "Most origin pages used per enrichment (capped at 6).",
  YUNSHU_WEB_RESEARCH_MODEL:
    "Id of an already-loaded local embedding model; it never loads one, and without it ranking is BM25 only. Qwen3-Embedding-0.6B is recommended.",
  YUNSHU_TELEMETRY:
    "Unprivileged Apple power, GPU and temperature sampler (`on` or `off`); restart required.",
  YUNSHU_TELEMETRY_INTERVAL_S:
    "Host telemetry sampling interval in seconds (restart required).",
  YUNSHU_SERVE_LOG_RETENTION_DAYS:
    "History API metadata retention window in days; 0 disables age filtering.",
};
export default settingdesc;
