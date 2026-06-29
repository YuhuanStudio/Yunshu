<div align="center">

# Yunshu

**A fast, local, multimodal inference engine for Apple Silicon.**

OpenAI/Anthropic-compatible — text, vision, OCR, audio, images, and a Realtime voice socket — multi-model,
all on-device via MLX. Its standout: **native streaming speech-to-speech** (Qwen3-Omni Talker), first audio
in ~1.4 s — no cloud, no ASR + LLM + TTS cascade, the model speaks with its own voice.

[![PyPI](https://img.shields.io/pypi/v/yunshu.svg?label=PyPI)](https://pypi.org/project/yunshu/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml)

</div>

---

## Why this exists

Yunshu is a general drop-in local AI server — point any OpenAI/Anthropic client at it for text, vision, OCR,
audio, or images, on-device. What sets it apart from other local servers is **native streaming
speech-to-speech**.

Most local speech pipelines are **cascades**: ASR transcribes your speech → LLM generates text → TTS reads it aloud.
Each hop adds latency, loses prosody, and can't reason about tone or sound.

Qwen3-Omni's **Talker** architecture is different: one model ingests raw audio, reasons, and decodes speech tokens
directly — no intermediate text. Yunshu exposes that natively on Apple Silicon via `mlx-vlm`, with a
streaming SSE endpoint that starts emitting audio chunks within ~1 s of your utterance. (Other models run too —
any `mlx-lm`/`mlx-vlm`/`mlx-audio` model; non-omni models get the standard endpoints.)

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

### Point your existing OpenAI client at it

Text, vision, embeddings, and rerank all speak the standard API — no client changes:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")  # any key works

# Chat. In single-model mode the model name is a placeholder — the server serves
# whatever you loaded (like Ollama / LM Studio), so "local" is fine.
print(
    client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": "Explain MLX in one sentence."}],
    ).choices[0].message.content
)

# Embeddings — text, or multimodal (image / cross-modal) with a Qwen3-VL-Embedding model.
client.embeddings.create(model="local", input=["hello", "world"])
```

Runnable scripts in **[examples/](examples/)**: `quickstart.py` (every endpoint), `realtime_voice.py`
(WebSocket speech-to-speech), `multimodal_embeddings.py` (image / cross-modal retrieval + reranking).

> **Dev checkout**: `just setup` then `YUNSHU_MODEL=<model> just dev`.
> **All tunables**: see the [configuration reference](docs/CONFIGURATION.md).

---

## What this is

A single OpenAI/Anthropic-compatible process that serves **every modality on-device** — LLM (brain),
VLM/OCR (eyes), ASR (ears), TTS + native Talker (voice), embeddings/rerank (retrieval), and image
generation (imagination) — built on Apple's MLX stack. Point any OpenAI/Anthropic SDK at it.

It's a **general-purpose** local inference engine, usable standalone. One notable consumer is Yunmo, a
local digital-being framework that uses Yunshu as its sensory body — but that's an example of what it
can power, not the definition of what it is.

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
| **Native speech-to-speech** (Qwen3-Omni, streaming; ~1.4s first-audio speech-in, ~1.2s text-in, warm) | `POST /v1/omni/speech/stream` | `mlx-vlm` Thinker+Talker | `omni` |
| Text (tool-calling, JSON-schema, streaming, logprobs) | `/v1/chat/completions`, `/v1/messages` | `mlx-lm` | _(core)_ |
| Vision / OCR | `/v1/chat/completions` (image content) | `mlx-vlm` | `vision` |
| ASR | `/v1/audio/transcriptions` | `mlx-audio` / Whisper | `audio` |
| TTS | `/v1/audio/speech` | `mlx-audio` | `audio` |
| Realtime voice WS | `WS /v1/realtime` | ASR + TTS | `audio` |
| Image generation | `/v1/images/generations` | diffusion | `generation` |
| Embeddings (text + **multimodal**: image / cross-modal via Qwen3-VL-Embedding) | `/v1/embeddings` | `mlx-embeddings` | `embeddings` |
| Rerank (bi-encoder cosine, or **true cross-encoder** via Qwen3-VL-Reranker) | `/v1/rerank` | `mlx-embeddings` | `embeddings` |

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

**Refocused** from an over-scoped "inference platform" down to an honest single-node omni engine: the
multi-node mesh / distributed paths (sharded-load, disaggregated prefill/decode), the multi-tenant control
plane, and tiered-KV offload have been removed.

On the default serving path a request gets the **single-request fast path** (mlx-lm `generate_step`) with KV
prefix + prompt caching, automatic KV-quant (only when the cache would dominate bandwidth), constrained
decoding (JSON-schema / regex / grammar, when requested), per-request stop/reasoning state, and **n-gram
speculative decode on greedy requests** (lossless — the verifier accepts only the model's own argmax; opt out
with `YUNSHU_NGRAM_DEFAULT=0`) — all on by default. The heavier or more situational optimizations are
**opt-in**, not magic-on: alternative spec proposers (cross-model via `spec_decode=true` + a draft model;
Suffix Decoding via `YUNSHU_SPEC_PROPOSER=suffix`), the top-nσ sampler (per-request
`"top_n_sigma"` on `/v1/chat/completions`, or server-wide `YUNSHU_TOP_N_SIGMA`), jump-forward
(`YUNSHU_JUMP_FORWARD`), GPU sampler (`YUNSHU_GPU_SAMPLER`), in-memory MXFP4/NVFP4 weight quant
(`YUNSHU_QUANT_MODE`), and sparse spec-prefill (`YUNSHU_SPEC_PREFILL` + a draft model). We keep them current,
but we don't claim they're running when they aren't. Run `just test` for the suite; benchmark trends live in
`docs/reports/PERF_TREND.md`.

## Built on

- [MLX](https://github.com/ml-explore/mlx) · [mlx-lm](https://github.com/ml-explore/mlx-lm) ·
  [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) · [mlx-audio](https://github.com/Blaizzy/mlx-audio)

## License

Apache 2.0 — see [LICENSE](LICENSE).
