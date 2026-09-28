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

> **Requires [uv](https://docs.astral.sh/uv/).** Not on PyPI yet — install from source with
> `uv sync`, which installs the exact versions in `uv.lock` (MLX 0.32, upstream `mlx-vlm` 0.7.3+).

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
  │   · VLM runner (Qwen3.5/3.6/3.8): prefix cache +    │
  │     MTP / DFlash speculative decode + verify kernels│
  │   · VLM / OCR (mlx-vlm)  · ASR / TTS (mlx-audio)   │
  │   · OmniEngine (Qwen3-Omni Thinker→Talker)          │
  │   · image diffusion       · KV prefix cache          │
  └────────────────────────────────────────────────────┘
              runs on-device via Apple MLX
```

## Performance

Yunshu serves one request at a time and optimizes for latency: first token (cold and cached),
decode speed, and prefix reuse. The first fully tuned model is **Qwen3.8-27B**. Qwen3.5-family
VLMs (Qwen3.5 / 3.6 / 3.8) run on a dedicated runner built on `mlx-vlm`'s generator:

- **Prefix cache (APC)** with exact hybrid-model checkpoints, keyed by image pixels as well as text,
  8 GiB RAM by default plus an optional SSD tier. Repeated or edited long prompts skip prefill.
- **Speculative decode** with the checkpoint's MTP head, or an external DFlash drafter
  (`YUNSHU_VLM_DRAFT`). By default every decode and verify matmul of a drafting request goes
  through one batch-invariant kernel, so greedy output with speculation on is token-identical to
  speculation off (the same guarantee Splash calls lossless). Non-exact fast verify is opt-in
  (`YUNSHU_MTP_FAST_VERIFY=1`).
- Streaming reasoning split, tool calls, JSON-schema constraints, stop sequences, logprobs and
  cancellation all work on this path.

Measured on an M5 Max (128 GB), Qwen3.8-27B, 2026-09-28. Same Jundot `oQ4e-mtp` checkpoint unless
noted; raw data and methods in
[docs/research/runs/2026-09-28-matrix](docs/research/runs/2026-09-28-matrix/README.md).

| Engine | Capability checks | Chat TTFT (warm) | 8K prompt: cold / repeat / edited tail | Decode tok/s |
|---|---|---|---|---|
| **Yunshu** (default: MTP block 6, batch-invariant) | 33/33 | 0.195 s | 8.40 / 0.112 / 0.259 s | 80 |
| **Yunshu** (DFlash2 + fast verify, opt-in) | 31/31 | 0.185 s | 8.71 / 0.112 / 0.239 s | 86 |
| mlx-vlm 0.7.3 server (APC) | 27/28 | 0.212 s | 8.60 / 0.108 / 0.265 s | 32 |
| oMLX.app 0.7 (MTP + cache) | 31/31 | 0.312 s | 8.60 / 0.361 / 0.376 s | 85 |
| Splash 1.1 (own quantized model + DFlash2) | 31/31 | 0.206 s | 7.88 / 0.131 / 7.88 s | 119 |

Where Yunshu stands: prefix reuse and TTFT are the best measured; prefill is at the hardware
ceiling for this checkpoint; default decode is on par with oMLX while keeping lossless output, and
**behind Splash**, which pairs DFlash2 with its own quantized model. Closing that gap is the
current work. (Yunshu's matrix has two more checks than the older runs: logprobs and the
streaming reasoning split.) Other, situational knobs (n-gram speculation, alternative samplers, KV quant,
jump-forward) are opt-in; see the [configuration reference](docs/CONFIGURATION.md). The long-run
benchmark log is [docs/reports/PERF_TREND.md](docs/reports/PERF_TREND.md).

## Built on

[MLX](https://github.com/ml-explore/mlx) · [mlx-lm](https://github.com/ml-explore/mlx-lm) ·
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) · [mlx-audio](https://github.com/Blaizzy/mlx-audio).
Some verify kernels are vendored from [oMLX](https://github.com/jundot/omlx) (Apache-2.0); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
