<div align="center">

# Yunshu

**Production-grade MLX inference platform for Apple Silicon.**

5 modalities. 5 protocols. 4-tier KV cache. Zero compromise.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-2482%20passing-brightgreen.svg)]()

[Getting Started](#getting-started) · [Features](#features) · [Benchmarks](#benchmarks) · [Architecture](#architecture) · [API Reference](#api-reference) · [Contributing](CONTRIBUTING.md)

</div>

---

Yunshu is a high-performance inference server that runs large language models, vision-language models, speech synthesis, speech recognition, and image generation — all locally on Apple Silicon using the MLX framework.

It exposes an **OpenAI-compatible API** so any existing client (curl, Python SDK, Swift app) works out of the box. Just change the `base_url` and you're done.

## Features

- **5 Modalities** — LLM text, Vision-Language (VLM), Text-to-Speech (TTS), Speech Recognition (ASR), Image Generation
- **5 Protocols** — OpenAI Chat/Completions, Anthropic Messages, Model Context Protocol (MCP), Realtime WebSocket, Admin API
- **Continuous Batching** — mlx-lm `BatchGenerator` with per-request samplers and sequence state machines
- **Speculative Decoding** — EAGLE-3 draft model + N-gram model-free proposer + MTP multi-token prediction
- **4-Tier KV Cache** — Hot (GPU) → Warm (CPU + quantized) → Cool (SSD) → Cold (distributed)
- **Metal Kernels** — PagedAttention, SDPA, GEMV, SGMV (LoRA), KIVI quantization — all with configurable tile sizes
- **Multi-Model Serving** — LRU eviction, memory guard, 5-modality auto-detection
- **RBAC Multi-Tenant** — 3-tier auth (ADMIN/DEVELOPER/USER) with per-key rate limits
- **WebUI Dashboard** — Next.js 16 management interface with SLO alerts
- **Distributed Ready** — mx.distributed compute mesh for multi-node inference

## Getting Started

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- [just](https://github.com/casey/just) command runner (`brew install just`)

### Install & Run

```bash
git clone https://github.com/YuhuanStudio/Yunshu.git
cd Yunshu
just setup          # install Python + WebUI dependencies

# Place a model in ./models/ (any HF-format MLX quantized model)
just dev            # start server on :8000
```

### First Request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-MLX-4bit",
    "messages": [{"role": "user", "content": "Explain quantum computing in one paragraph"}],
    "stream": true
  }'
```

### Python SDK (Drop-in OpenAI Replacement)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="any")

# Chat with streaming
for chunk in client.chat.completions.create(
    model="Qwen3.5-9B-MLX-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")

# Text-to-speech
response = client.audio.speech.create(
    model="Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
    voice="alloy",
    input="Hello, world!",
)
```

### Multi-Model Mode

```bash
YUNSHU_MULTI_MODEL=1 just dev-multi
# Auto-discovers all models in ./models/ — LLM, VLM, TTS, ASR, Image Gen
```

## Benchmarks

Verified on M3 Max with Qwen3.5-9B-4bit using the [unified benchmark](scripts/bench_unified.py):

| Framework | MMLU Accuracy | Throughput | vs Baseline |
|-----------|:------------:|:----------:|:-----------:|
| **Yunshu** | **96%** | **44.4 tok/s** | **1.13x** |
| mlx-lm | 96% | 39.3 tok/s | 1.00x (baseline) |
| oMLX | 96% | 41.8 tok/s | 1.06x |

Yunshu exceeds the mlx-lm baseline in throughput with **zero accuracy degradation**.

Run it yourself:

```bash
PYTHONPATH=. uv run python scripts/bench_unified.py --quick
```

## Architecture

Yunshu is a 5-layer stack, each layer independently replaceable:

```
┌─────────────────────────────────────────┐
│  L0  Client Layer                       │  Swift app / CLI / SDK
├─────────────────────────────────────────┤
│  L1  Gateway                            │  OpenAI · Anthropic · MCP · Realtime
│     FastAPI + SSE + WebSocket           │  14 routers, 6 middleware
├─────────────────────────────────────────┤
│  L2  Control Plane                      │  RBAC · Priority Queue · Admin API
│     Auth, scheduling, management        │
├─────────────────────────────────────────┤
│  L3  Compute Mesh                       │  mx.distributed · mDNS · Heartbeat
│     Ring/FC/Pipeline topologies         │
├─────────────────────────────────────────┤
│  L4  Inference Engine                   │  5-modal · Continuous batching
│     Speculative decode · JSON schema    │  EAGLE-3 · Sarathi hybrid prefill
├─────────────────────────────────────────┤
│  L5  KV Hierarchy                       │  Hot/Warm/Cool/Cold 4-tier cache
│     Radix attention · SSD store         │  Thinking segment compression
└─────────────────────────────────────────┘
         backed by Metal 3.1 GPU kernels
```

### Code Statistics

```
160+ Python files    38,000+ lines
6 Metal kernels      874 lines (Metal 3.1)
12 TypeScript files  3,823 lines (Next.js 16)
2,162 tests passing  95+ test files
```

## API Reference

Yunshu implements the OpenAI API spec, so any OpenAI client works without modification. Just change `base_url`.

### Key Endpoints

| Method | Path | Description |
|:------:|------|-------------|
| POST | `/v1/chat/completions` | Chat (streaming, tools, JSON mode, n>1, logprobs, thinking) |
| POST | `/v1/completions` | Text completions |
| POST | `/v1/embeddings` | Text embeddings |
| POST | `/v1/audio/speech` | Text-to-speech synthesis |
| POST | `/v1/audio/speech/stream` | Streaming TTS via SSE |
| POST | `/v1/audio/transcriptions` | Speech-to-text (ASR) |
| GET | `/v1/audio/voices` | List available TTS voices |
| POST | `/v1/images/generations` | Image generation |
| POST | `/v1/messages` | Anthropic Messages API |
| POST | `/v1/mcp` | Model Context Protocol |
| WS | `/v1/realtime` | Realtime audio (ASR + TTS) |
| GET | `/v1/models` | List loaded models |
| GET | `/health` | Health check (K8s probes) |
| GET | `/metrics` | Prometheus metrics |
| GET | `/admin/*` | Management API (RBAC, models, config) |

### Full OpenAI Feature Support

- Streaming (SSE with keepalive + disconnect guard)
- Tool calling (Hermes/Qwen/Llama XML/code-block formats)
- JSON mode (`json_object` and `json_schema`)
- `n > 1` parallel completions
- `logprobs` + `top_logprobs`
- `frequency_penalty` / `presence_penalty` / `logit_bias`
- `response_format` (JSON schema constrained generation)
- Reasoning mode (`enable_thinking` for Qwen3.5 and other reasoning models)
- Context window validation
- Per-request seed for reproducibility

## Configuration

Environment variables (no config files needed):

| Variable | Default | Description |
|----------|---------|-------------|
| `YUNSHU_MODEL` | — | Path to single model to load on startup |
| `YUNSHU_MULTI_MODEL` | — | Set `1` for multi-model auto-discovery |
| `YUNSHU_MODELS_DIR` | `./models` | Directory to scan for models |
| `YUNSHU_PORT` | `8000` | Server port |
| `YUNSHU_HOST` | `0.0.0.0` | Server bind address |
| `YUNSHU_MAX_MEMORY_GB` | 80% UMA | Max GPU memory (GB) |
| `YUNSHU_AUTH_TOKEN` | — | Bearer token for API authentication |
| `YUNSHU_CORS_ORIGINS` | `*` | Comma-separated CORS origins |
| `YUNSHU_RATE_LIMIT_RPM` | `0` | Requests per minute per client (0=unlimited) |
| `YUNSHU_LOG_LEVEL` | `INFO` | Logging level |

## Project Structure

```
yunshu/
├── python/
│   ├── yunshu_gateway/    # L1: FastAPI HTTP server (14 routers, 6 middleware)
│   ├── yunshu_control/    # L2: RBAC, scheduling, request queue
│   ├── yunshu_api/        # L2: Admin management API
│   ├── yunshu_mesh/       # L3: Compute mesh (discovery, heartbeat, collectives)
│   ├── yunshu_engine/     # L4: Inference engine (40+ modules)
│   ├── yunshu_kv/         # L5: KV cache hierarchy (12 modules)
│   ├── yunshu_sdk/        # Python SDK
│   └── yunshu_cli/        # CLI
├── metal/                  # Metal 3.1 compute shaders (6 kernels)
├── webui/                  # Next.js 16 dashboard
├── tests/                  # Test suite (2162 tests)
├── scripts/                # Benchmarks and utilities
├── docs/                   # Documentation
└── bench/                  # Benchmark results
```

## Development

```bash
just dev              # Start dev server with auto-reload
just test             # Run all tests (2162 tests)
just test-unit        # Unit tests only
just lint             # Lint with ruff + mypy
just format           # Auto-format
just bench-roofline   # Apple Silicon roofline benchmark
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.

## Supported Models

Any HuggingFace-format model that works with mlx-lm, mlx-vlm, or mlx-audio. Tested with:

| Model | Modality | Size | Format |
|-------|----------|------|--------|
| Qwen3.5-9B | LLM | 9B | 4-bit MLX |
| Qwen3-Omni-30B-A3B | VLM | 30B (A3B MoE) | 4-bit MLX |
| Qwen3-TTS-1.7B | TTS | 1.7B | BF16 |
| Qwen3-ASR-1.7B | ASR | 1.7B | BF16 |
| Z-Image-Turbo | Image Gen | — | 4-bit MLX |

## Roadmap

| Phase | Scope | Status |
|:-----:|-------|:------:|
| 0 | Platform validation | ✅ Done |
| 1 | MVP single-node OpenAI-compatible | ✅ Done |
| 2 | Distributed 4-node mesh | 🔄 In progress |
| 3 | 5-modality + Anthropic + MCP | ✅ Done |
| 4 | Speculative decoding + Realtime | ✅ Done |
| 5 | v1.0 release | 📋 Planned |

## License

[Apache License 2.0](LICENSE)

## Acknowledgments

Yunshu builds on the Apple MLX ecosystem:
- [MLX](https://github.com/ml-explore/mlx) — Apple's array framework for Apple Silicon
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — LLM inference on MLX
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — Vision-language models on MLX
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — Audio models on MLX
