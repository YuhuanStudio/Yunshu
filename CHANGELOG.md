# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2025-05-05

### Added

- **5-modality inference** — LLM text, Vision-Language (VLM), Text-to-Speech (TTS), Speech Recognition (ASR), Image Generation
- **5 protocol support** — OpenAI Chat/Completions, Anthropic Messages, Model Context Protocol (MCP), Realtime WebSocket, Admin API
- **Continuous batching** — mlx-lm `BatchGenerator` with per-request samplers and sequence state machines
- **Speculative decoding** — EAGLE-3 draft model with auto-detection of 5 draft head types
- **4-tier KV cache** — Hot (GPU) → Warm (CPU + quantized) → Cool (SSD) → Cold (distributed)
- **Metal 3.1 kernels** — PagedAttention, SDPA, GEMV, SGMV (LoRA), KIVI quantization, MLA with configurable tile sizes
- **Multi-model serving** — LRU eviction, memory guard, 5-modality auto-detection
- **RBAC multi-tenant auth** — 3-tier (ADMIN/DEVELOPER/USER) with per-key rate limits
- **OpenAI-compatible API** — streaming SSE, tool calling, JSON mode, logprobs, `n>1`, reasoning mode
- **Per-request sampler** — temperature, top-p, top-k, frequency/presence penalty, logit bias, seed
- **Context window validation** — reject requests exceeding model context length
- **WebUI dashboard** — Next.js 16 management interface with SLO alerts
- **Python SDK** — drop-in OpenAI client replacement
- **CLI** — `yunshu` command-line interface via Typer
- **Unified benchmark** — 4-framework LLM comparison (mlx-lm, oMLX, vLLM-MLX, Yunshu) with MMLU accuracy validation
- **2,162 tests** — 95+ test files covering unit and integration scenarios
- **Prometheus metrics** — `/metrics` endpoint for monitoring
- **Health check** — `/health` endpoint for Kubernetes probes
- **Distributed ready** — mx.distributed compute mesh for multi-node inference

### Performance

- **44.4 tok/s** throughput on M3 Max with Qwen3.5-9B-4bit (1.13x mlx-lm baseline)
- **96% MMLU accuracy** — zero quality degradation vs baseline
- **Non-streaming fast path** — direct `generate_step` matching mlx-lm speed (~50 tok/s)

### Supported Models

- Qwen3.5-9B (LLM, 4-bit MLX)
- Qwen3-Omni-30B-A3B (VLM, 4-bit MLX)
- Qwen3-TTS-1.7B (TTS, BF16)
- Qwen3-ASR-1.7B (ASR, BF16)
- Z-Image-Turbo (Image Gen, 4-bit MLX)

[0.0.1]: https://github.com/YuhuanStudio/Yunshu/releases/tag/v0.0.1
