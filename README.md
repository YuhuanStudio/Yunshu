<div align="center">

# Yunshu

**A local, single-node, multimodal (omni) inference engine for Apple Silicon.**

One OpenAI/Anthropic-compatible endpoint for text · vision · OCR · speech-in · speech-out · images — running entirely on your Mac via MLX.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

</div>

---

## What this is

Yunshu is the **local sensory body** for a digital being — it gives one process a unified, OpenAI-compatible
surface for every modality a being needs: a tool-calling LLM brain, eyes (VLM/OCR), ears (ASR), a voice
(TTS + a Realtime WebSocket), and optionally an imagination (image generation). It is built on Apple's MLX
stack (`mlx-lm`, `mlx-vlm`, `mlx-audio`) and runs fully on-device.

It exists primarily to serve **[Yunmo](../Yunmo)** (a local digital-being framework) so that Yunmo can finally
honour its own rule of running with no cloud dependency. It is also usable standalone as a drop-in local
OpenAI/Anthropic endpoint.

## What this is NOT (honest non-goals)

- **Not a distributed / multi-node system.** There is no working multi-Mac cluster; that thesis was abandoned.
- **Not chasing throughput.** Default serving is a single-request fast path. Continuous batching exists but is
  not the point — the consumer is one being, not a multi-tenant fleet.
- **No compute moat.** Yunshu wraps MLX; it has no custom kernels. Single-stream decode is at parity with
  `mlx-lm`. Its only real performance edge is **low TTFT / voice round-trip latency**, which is the one axis
  that matters for a real-time being — and the one it keeps optimizing.

If you need production multi-tenant serving or multi-node sharding on Apple Silicon, use
[oMLX](https://github.com/jundot/omlx), [vllm-mlx](https://github.com/waybarrios/vllm-mlx), or
[exo](https://github.com/exo-explore/exo). Yunshu is deliberately narrower.

## Capabilities

| Modality | Endpoint | Backend |
|----------|----------|---------|
| Text (tool-calling, JSON-schema constrained, streaming, logprobs) | `/v1/chat/completions`, `/v1/messages` | `mlx-lm` |
| Vision (VLM) | `/v1/chat/completions` (image content) | `mlx-vlm` |
| OCR | `/v1/chat/completions` (OCR models) | `mlx-vlm` |
| Speech-to-text (ASR) | `/v1/audio/transcriptions` | `mlx-audio` / whisper |
| Text-to-speech | `/v1/audio/speech` | `mlx-audio` |
| Realtime voice (bidirectional) | `WS /v1/realtime` | ASR + TTS pipeline |
| Image generation | `/v1/images/generations` | self-implemented diffusion |

Plus: a single-node KV prefix cache (+ optional SSD persistence and per-request KV quantization),
and MCP (server + client). The protocol surface is OpenAI- and Anthropic-compatible, so existing clients
work by changing `base_url`.

## Getting started

```bash
just setup                       # install Python deps via uv
YUNSHU_MODEL=/path/to/mlx-model just dev   # start the server on :8000
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
print(client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Hello!"}],
).choices[0].message.content)
```

## Architecture

```
  Client (Yunmo daemon, or any OpenAI/Anthropic SDK)
        │   OpenAI / Anthropic / MCP / Realtime-WS
  ┌─────┴───────────────────────────────────────────┐
  │  Gateway (FastAPI)   protocol surface + routers  │
  ├──────────────────────────────────────────────────┤
  │  Engine              modality dispatch + serving  │
  │   · LLM fast path (mlx-lm generate_step)          │
  │   · VLM / OCR (mlx-vlm)   · ASR / TTS (mlx-audio) │
  │   · image diffusion       · KV prefix cache       │
  └──────────────────────────────────────────────────┘
              runs on-device via Apple MLX
```

## Status

This repo is being **refocused** from an over-scoped "inference platform" down to a honest single-node omni
engine. Dead subsystems (multi-node mesh, speculative-decode strategies, tiered-KV offload, the multi-tenant
control plane) are being removed. See `CLAUDE.md` for the current working state. Run `just test` for the
test suite; benchmark trends live in `docs/reports/PERF_TREND.md`.

## Built on

- [MLX](https://github.com/ml-explore/mlx) · [mlx-lm](https://github.com/ml-explore/mlx-lm) ·
  [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) · [mlx-audio](https://github.com/Blaizzy/mlx-audio)

## License

[Apache License 2.0](LICENSE)
