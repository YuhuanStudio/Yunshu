<div align="center">

# Yunshu

**A fast, local, multimodal inference engine for Apple Silicon.**

One process, OpenAI/Anthropic-compatible, all on-device via MLX: text, vision, OCR, audio, images,
embeddings, and a realtime voice socket. Its standout is **native streaming speech-to-speech** —
you talk, the model talks back in ~1.4 s, in its own voice, with no cloud and no
speech-to-text → LLM → text-to-speech cascade.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml)

**English** · [简体中文](./README.zh-CN.md) · [繁體中文](./README.zh-TW.md)

</div>

---

## Native speech-to-speech

Most local voice setups are **cascades**: speech-to-text transcribes you → an LLM writes a reply →
text-to-speech reads it aloud. Every hop adds latency and throws away prosody — the system never
hears your tone, and can't shape its own.

Qwen3-Omni's **Talker** architecture is one model: it ingests raw audio, reasons, and decodes speech
tokens directly — no text in the middle. Yunshu serves that natively on Apple Silicon via `mlx-vlm`,
streaming audio back within ~1 s of your utterance.

```
You (audio) ──► Qwen3-Omni Thinker (reason) ──► Talker (stream audio out) ──► You
                     one unified model, no pipeline hops
```

Everything else runs too: any `mlx-lm` / `mlx-vlm` / `mlx-audio` model gets the standard endpoints.

---

## Quickstart

> **Requires [uv](https://docs.astral.sh/uv/).** Not on PyPI yet — install from source.
> `uv sync` honors `[tool.uv.sources]`, so it pulls the `mlx-vlm` fork Yunshu needs for Qwen3-Omni
> (latest upstream + a Thinker early-exit optimization and omni fixes). A plain `pip install` would
> pull unpinned upstream `mlx-vlm` instead, so use `uv sync`.

```bash
# 1. Install from source
git clone https://github.com/YuhuanStudio/Yunshu.git
cd Yunshu
uv sync --extra omni        # native Qwen3-Omni voice (speech in/out)
# or: uv sync --all-extras  # everything: text + vision + audio + omni + image + embeddings

# 2. Serve a model. Any 4-bit Qwen3-Omni variant from mlx-community works — native voice
#    is on automatically (the same loaded model serves text and speech, no extra memory).
uv run yunshu serve -m /path/to/Qwen3-Omni-30B-A3B-Instruct-4bit --port 8000
```

### Talk to it

```bash
uv run --with sounddevice --with numpy --with websockets python examples/talk.py
```

[`examples/talk.py`](examples/talk.py) is a real spoken conversation: press Enter, speak, press Enter
again — the model answers out loud, and remembers the conversation.

Prefer not to wire up a mic? [`examples/quickstart.py`](examples/quickstart.py) streams a spoken
reply to a WAV file and shows the text endpoints — no audio hardware needed.

### Use it from any OpenAI client

Text, vision, embeddings, and rerank all speak the standard API — no client changes:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")  # any key works

# In single-model mode the model name is a placeholder — the server serves whatever
# you loaded (like Ollama / LM Studio), so "local" is fine.
print(
    client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": "Explain MLX in one sentence."}],
    ).choices[0].message.content
)
```

> **Dev checkout**: `just setup` then `YUNSHU_MODEL=<model> just dev`.
> **Docs**: [API reference](docs/API.md) · [configuration reference](docs/CONFIGURATION.md).

---

## Capabilities

| Modality | Endpoint | Backend | Extra |
|---|---|---|---|
| **Native speech-to-speech** (Qwen3-Omni, streaming; ~1.4s first-audio speech-in, ~1.2s text-in, warm) | `POST /v1/omni/speech/stream` | `mlx-vlm` Thinker+Talker | `omni` |
| Text (tool-calling, JSON-schema, streaming, logprobs) | `/v1/chat/completions`, `/v1/messages` | `mlx-lm` | _(core)_ |
| Vision / OCR | `/v1/chat/completions` (image content) | `mlx-vlm` | `vision` |
| ASR | `/v1/audio/transcriptions` | `mlx-audio` / Whisper | `audio` |
| TTS | `/v1/audio/speech` | `mlx-audio` | `audio` |
| Realtime voice WS | `WS /v1/realtime` | omni or ASR + TTS | `audio` |
| Image generation | `/v1/images/generations` | diffusion | `generation` |
| Video generation (Wan 2.x / LTX-2, text-to-video + image-to-video) | `/v1/video/generations` | `mlx-video` | `video` |
| Embeddings (text + **multimodal**: image / cross-modal via Qwen3-VL-Embedding) | `/v1/embeddings` | `mlx-lm` (text) / `mlx-embeddings` (multimodal) | `embeddings` |
| Rerank (bi-encoder cosine, or **true cross-encoder** via Qwen3-VL-Reranker) | `/v1/rerank` | `mlx-lm` (text) / `mlx-embeddings` (multimodal) | `embeddings` |

Also: single-node KV prefix cache (+ optional SSD persistence + per-request KV quant), MCP
server/client, and an Anthropic-compatible `/v1/messages` surface.

## Architecture

```
  Client (any OpenAI / Anthropic SDK)
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

## Performance

Yunshu optimizes for **low latency**, not throughput — it serves one request at a time on a fast
path, which is the right shape for a local, single-user server. A request goes through mlx-lm's
`generate_step` with KV prefix + prompt caching on by default; single-stream decode is at parity
with `mlx-lm`. Situational knobs — lossless n-gram speculative decode (wins on repetitive/agentic
output, slower on normal prose, so it's opt-in), alternative samplers, in-memory weight quant,
jump-forward — are opt-in, never silently on; see the [configuration reference](docs/CONFIGURATION.md).
Honest benchmark trends live in [docs/reports/PERF_TREND.md](docs/reports/PERF_TREND.md).

## Built on

[MLX](https://github.com/ml-explore/mlx) · [mlx-lm](https://github.com/ml-explore/mlx-lm) ·
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) · [mlx-audio](https://github.com/Blaizzy/mlx-audio)

## License

Apache 2.0 — see [LICENSE](LICENSE).
