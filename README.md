<div align="center">

# Yunshu

**Native streaming speech-to-speech on Apple Silicon — powered by Qwen3-Omni.**

One process. One endpoint. Speech in → speech out, first audio in ~1.3 s, fully on-device.
No cloud. No cascade. No ASR + LLM + TTS pipeline. The model speaks with its own voice.

[![PyPI](https://img.shields.io/pypi/v/yunshu.svg?label=PyPI)](https://pypi.org/project/yunshu/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml)

</div>

---

## Why this exists

Most local speech pipelines are **cascades**: ASR transcribes your speech → LLM generates text → TTS reads it aloud.
Each hop adds latency, loses prosody, and can't reason about tone or sound.

Qwen3-Omni's **Talker** architecture is different: one model ingests raw audio, reasons, and decodes speech tokens
directly — no intermediate text. Yunshu is built to expose that natively on Apple Silicon via `mlx-vlm`, with a
streaming SSE endpoint that starts emitting audio chunks within ~1 s of your utterance.

```
You (audio) ──► Qwen3-Omni Thinker (reason) ──► Talker (stream audio out) ──► You
                     one unified model, no pipeline hops
```

---

## Quickstart

> **Requires [uv](https://docs.astral.sh/uv/).** The `[omni]` extra pins `mlx-vlm` to a fork that
> carries a Qwen3-Omni multi-turn fix not yet in upstream. `pip` ignores the fork pin and silently
> installs broken upstream. Use `uv` — it respects the `[tool.uv.sources]` pin.

```bash
# 1. Install (uv required)
uv pip install "yunshu[omni]"      # native Qwen3-Omni voice (speech-in/out)
uv pip install "yunshu[all]"       # everything: text + vision + audio + omni + image + embeddings

# 2. Start the server
yunshu serve -m /path/to/Qwen3-Omni-30B-A3B-Instruct-4bit --port 8000
# Any 4-bit Qwen3-Omni variant from mlx-community works

# 3. Stream a spoken reply (Server-Sent Events: text deltas + base64 PCM16 @ 24kHz)
curl -N -X POST http://localhost:8000/v1/omni/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Say hello in one sentence.", "speaker": "Ethan"}'
# For speech-IN, add "audio_path": "question.wav" (the spoken turn);
# "text" then carries any system instruction.
```

The endpoint streams SSE events, not a WAV file. For a ready-made client that consumes the
stream and writes `omni_out.wav`, see [examples/quickstart.py](examples/quickstart.py) — which
also shows the text/vision/ASR/TTS endpoints.

For the **bidirectional voice-agent** path (speak in over a WebSocket, hear the model speak back),
see [examples/realtime_voice.py](examples/realtime_voice.py) — it runs one native speech-to-speech
turn against the OpenAI-Realtime `WS /v1/realtime` endpoint (set `YUNSHU_REALTIME_OMNI=1` on the
server).

> **Dev checkout**: `just setup` then `YUNSHU_MODEL=<model> just dev`.

---

## What this is

Yunshu is the **local sensory body** for a digital being — a single OpenAI-compatible process that handles
every modality: LLM brain, eyes (VLM/OCR), ears (ASR), voice (TTS + native Talker), and imagination (image gen).
Built on Apple's MLX stack, runs fully on-device.

It exists primarily to serve **[Yunmo](../Yunmo)** — a local digital-being framework — so Yunmo can run with
zero cloud dependency. It is also usable standalone as a drop-in local OpenAI/Anthropic endpoint.

## What this is NOT

- **Not a distributed system.** No multi-Mac cluster; that thesis was abandoned.
- **Not a throughput race.** Single-request fast path; one consumer (a being), not a multi-tenant fleet.
- **No custom Metal kernels.** Wraps MLX. Single-stream decode is at parity with `mlx-lm`. The one performance
  axis that matters here is **voice round-trip latency**.

If you need production multi-tenant serving or multi-node sharding, look at
[oMLX](https://github.com/jundot/omlx), [vllm-mlx](https://github.com/waybarrios/vllm-mlx), or
[exo](https://github.com/exo-explore/exo).

## Capabilities

| Modality | Endpoint | Backend | Extra |
|---|---|---|---|
| **Native speech-to-speech** (Qwen3-Omni, streaming; ~1.3s first-audio speech-in, ~1.1s text-in) | `POST /v1/omni/speech/stream` | `mlx-vlm` Thinker+Talker | `omni` |
| Text (tool-calling, JSON-schema, streaming, logprobs) | `/v1/chat/completions`, `/v1/messages` | `mlx-lm` | _(core)_ |
| Vision / OCR | `/v1/chat/completions` (image content) | `mlx-vlm` | `vision` |
| ASR | `/v1/audio/transcriptions` | `mlx-audio` / Whisper | `audio` |
| TTS | `/v1/audio/speech` | `mlx-audio` | `audio` |
| Realtime voice WS | `WS /v1/realtime` | ASR + TTS | `audio` |
| Image generation | `/v1/images/generations` | diffusion | `generation` |
| Embeddings | `/v1/embeddings` | `mlx-embeddings` | `embeddings` |

Also: single-node KV prefix cache (+ optional SSD persistence + per-request KV quant), MCP server/client,
Anthropic-compatible `/v1/messages` surface.

## Architecture

```
  Client (Yunmo daemon / any OpenAI-Anthropic SDK)
        │   OpenAI / Anthropic / MCP / Realtime-WS / SSE
  ┌─────┴──────────────────────────────────────────────┐
  │  Gateway (FastAPI)    routers + middleware           │
  ├────────────────────────────────────────────────────┤
  │  Engine               modality dispatch + serving   │
  │   · LLM fast path (mlx-lm generate_step)            │
  │   · VLM / OCR (mlx-vlm)  · ASR / TTS (mlx-audio)   │
  │   · OmniEngine (Qwen3-Omni Thinker→Talker)          │
  │   · image diffusion       · KV prefix cache          │
  └────────────────────────────────────────────────────┘
              runs on-device via Apple MLX
```

## Status

This repo is being **refocused** from an over-scoped "inference platform" down to an honest single-node omni
engine. Dead subsystems (multi-node mesh, speculative-decode strategies, tiered-KV offload, multi-tenant
control plane) are being removed. Run `just test` for the test suite; benchmark trends live in
`docs/reports/PERF_TREND.md`.

## Built on

- [MLX](https://github.com/ml-explore/mlx) · [mlx-lm](https://github.com/ml-explore/mlx-lm) ·
  [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) · [mlx-audio](https://github.com/Blaizzy/mlx-audio)

## License

Apache 2.0 — see [LICENSE](LICENSE).
