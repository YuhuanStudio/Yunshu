# Changelog

All notable changes to Yunshu are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/). Pre-1.0 versioning: a minor bump
signals new capability, a patch bump signals fixes.

## [Unreleased]

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
