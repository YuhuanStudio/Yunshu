# Changelog

All notable changes to Yunshu are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/). Pre-1.0 versioning: a minor bump
signals new capability, a patch bump signals fixes.

## [Unreleased]

The current `main` — the capability surface for the first public release.

### Added

- **Native streaming speech-to-speech** (the flagship): Qwen3-Omni Thinker→Talker
  on Apple MLX via `mlx-vlm` — `POST /v1/omni/speech/stream` (SSE) and the
  OpenAI-Realtime `WS /v1/realtime` socket. Speech-in (raw audio, no ASR) and
  speech-out, ~1.2 s first-audio text-in / ~1.4 s speech-in (warm). Multi-turn
  conversation context preserved (bounded to protect TTFT).
- **OpenAI-compatible API**: chat/completions (tool-calling, JSON-schema/grammar,
  streaming, logprobs), completions, responses, embeddings, audio
  (transcriptions/translations/speech), images (generation + edits/variations/
  inpaint/controlnet), tokenizer utilities, batch inference.
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
