# Changelog

All notable changes to Yunshu are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/). Versions go up in small steps: a patch
(0.1.1, 0.1.2, …) for fixes and improvements, a minor (0.2.0) only for a real capability jump.
Release steps: [RELEASING.md](RELEASING.md).

## [Unreleased]

Proposed as **0.1.1** (not released yet).

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
- `yunshu service install|uninstall|start|stop|restart|status|logs`: a per-user launchd agent
  that starts Yunshu at login, restarts it after a crash, and logs to `~/Library/Logs/Yunshu`.
- Docs: [connecting clients](docs/guides/CLIENTS.md), [service](docs/guides/SERVICE.md),
  [troubleshooting](docs/guides/TROUBLESHOOTING.md), [benchmark sources](docs/BENCHMARKS.md),
  [RELEASING.md](RELEASING.md); a tag-triggered release workflow (PyPI trusted publishing behind
  an approval step) and a draft Homebrew formula (`packaging/homebrew/`).
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
  opt-in fast verify / batch-invariant kernels, partly vendored from oMLX (Apache-2.0).
- `YUNSHU_LOG_LEVEL`; `scripts/research/` benchmark matrix, realistic soak and MMLU-Pro soak.

### Changed

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
- Text engine: KV cache quantization and the 4-bit warm prefix tier are opt-in (both are lossy).
- `reasoning_effort` (top-level or in `chat_template_kwargs`) is passed to chat templates that
  support it (Qwen3.8: low / medium / xhigh) instead of being mapped to a thinking-token cap.
- Dependencies: MLX 0.32.2, transformers 5.17, upstream `mlx-vlm` 0.7.3 (the fork is gone).
- JSON-schema constraint: cached vocab split (in-string step 144 ms → 3 ms); structural whitespace
  limited to spaces and newlines.

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

### Removed

- The deprecated `Engine` class, multi-node / multi-tenant leftovers (distributed diffusion,
  connection pool, data-parallel endpoint, tenant auth naming), and unused modules
  (`vlm_async_engine`, `wan_vae`, `lid`, `optimizations`, `vision_encoding`).
- The older VLM MTP / APC side paths, superseded by the runner.
- The runner's upstream quantized-KV option (slower than bf16 at every measured length) and other
  measured-out flags (`YUNSHU_PACKED_5BIT`, `YUNSHU_VLM_KV_BITS`, `YUNSHU_VLM_INVARIANT`).

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
