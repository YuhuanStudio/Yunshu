<div align="center">

# Yunshu

**A fast local LLM / VLM inference engine for Apple Silicon.**

One process, OpenAI- and Anthropic-compatible, running on-device via MLX. Built for low latency on
a single machine: fast first token, fast lossless decode, and prefix reuse that skips work you
already paid for. The first fully tuned model is **Qwen3.8-27B**.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml)

**English** · [简体中文](./README.zh-CN.md) · [繁體中文](./README.zh-TW.md)

</div>

---

## What it does differently

- **Lossless speculative decode.** MTP (the checkpoint's own head) or an external DFlash drafter.
  Every decode and verify matmul of a drafting request goes through one batch-invariant kernel, so
  greedy output with speculation on is token-identical to speculation off — the guarantee Splash
  calls lossless. Non-exact fast verify exists but is opt-in.
- **Prefix cache for hybrid models.** Qwen3.5-family models mix attention with recurrent
  GatedDeltaNet layers, which ordinary KV caches cannot slice. Yunshu keeps exact checkpoints,
  keyed by image pixels as well as text, 8 GiB in RAM by default plus an optional SSD tier.
  Repeated or edited long prompts skip prefill.
- **Verify kernels checked against output.** GatedDeltaNet, attention and 5-bit matmul verify
  kernels, partly vendored from oMLX, each adopted only after a same-checkpoint A/B.
- **The full API surface on the fast path.** Tool calls, JSON-schema constraints, stop sequences,
  logprobs, a streaming reasoning/content split, `reasoning_effort` passed to chat templates that
  support it (Qwen3.8), and cancellation on client disconnect.

## Quickstart

> **Requires [uv](https://docs.astral.sh/uv/).** Not on PyPI yet — install from source with
> `uv sync`, which installs the exact versions in `uv.lock` (MLX 0.32, upstream `mlx-vlm` 0.7.3+).

```bash
git clone https://github.com/YuhuanStudio/Yunshu.git
cd Yunshu
uv sync --extra vision       # LLM + VLM (Qwen3.5 / 3.6 / 3.8 need this)
# or: uv sync --all-extras   # every modality

uv run yunshu serve -m /path/to/Qwen3.8-27B-mlx --port 8000
```

Any OpenAI client works unchanged:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")  # any key works

# Single-model mode: the model name is a placeholder, the server serves what you loaded.
r = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Explain MLX in one sentence."}],
    extra_body={"reasoning_effort": "medium"},
)
print(r.choices[0].message.content)
```

> **Dev checkout**: `just setup` then `YUNSHU_MODEL=<model> just dev`.
> **Docs**: [API reference](docs/API.md) · [configuration reference](docs/CONFIGURATION.md).

## Performance

Measured on an M5 Max (128 GB), Qwen3.8-27B, 2026-09-28. Same Jundot `oQ4e-mtp` checkpoint unless
noted; raw data and methods in
[docs/research/runs/2026-09-28-matrix](docs/research/runs/2026-09-28-matrix/README.md).

| Engine | Capability checks | Chat TTFT (warm) | 8K prompt: cold / repeat / edited tail | Decode tok/s |
|---|---|---|---|---|
| **Yunshu** (default: MTP block 6, batch-invariant) | 34/34 | 0.194 s | 8.45 / 0.115 / 0.258 s | 80 |
| **Yunshu** (DFlash2 + fast verify, opt-in) | 31/31 | 0.185 s | 8.71 / 0.112 / 0.239 s | 86 |
| mlx-vlm 0.7.3 server (APC) | 27/28 | 0.212 s | 8.60 / 0.108 / 0.265 s | 32 |
| oMLX.app 0.7 (MTP + cache) | 31/31 | 0.312 s | 8.60 / 0.361 / 0.376 s | 85 |
| Splash 1.1 (own quantized model + DFlash2) | 31/31 | 0.206 s | 7.88 / 0.131 / 7.88 s | 119 |

Lossless decode by output type (same checkpoint, in-process, greedy, 384 tokens; tok/s):

| Decode | Code | Prose | JSON-like | Spec on == off |
|---|---|---|---|---|
| **Default**: batch-invariant + packed, MTP block 6 | 88.6 | 59.9 | 67.3 | yes on short prompts¹ |
| Previous default: exact verify kernels, MTP block 3 | 57–67 | 50–53 | 58–62 | yes on short prompts¹ |
| Non-exact fast verify (opt-in) | 83.8 | 59.9 | 66.7 | no |

¹ Matmuls are row-invariant, but verify attention uses a different MLX kernel than one-row decode,
so at longer contexts speculative output can differ from plain decode in rare tokens. A row-exact
attention path (synced from oMLX) is being validated to close this.

MMLU-Pro, 300 questions, 8 in flight, max 16384 tokens, `reasoning_effort=medium` (accuracy and a
long-run soak; same settings for every engine):

| Engine | Correct | Wall time | Aggregate tok/s | Peak footprint |
|---|---|---|---|---|
| **Yunshu** (shared batch, commit fbbb1378) | 250 / 300 | 46.5 min | 88 | 45 GiB (back to 17 at the end) |
| Splash 1.1 | 252 / 300 | 17.2 min | 223 | 67 GiB |
| oMLX.app | 229 / 300 (27 rejected by its prefill memory guard) | 29.2 min | 120 | 75 GiB |

Speed sweep (unique prompts, no cache hits; 128 generated tokens; tok/s unless noted):

| | Yunshu | oMLX | Splash |
|---|---|---|---|
| TTFT at 8K / 131K / 200K tokens | 8.4 / 209 / 394 s | 8.5 / 214 / 401 s | 7.9 / 207 / 390 s |
| Decode after 1K / 32K / 200K | 58 / 44 / 17 | 71 / 60 / 29 | 101 / 48 / 66 |
| 8 concurrent 1K prompts, aggregate | 61 | 53 | 70 |

Where Yunshu stands:
- Prefix reuse and warm TTFT are the best measured; cold prefill is at the hardware ceiling (all three
  engines within ~5%).
- Accuracy matches Splash; 0 errors in every long run.
- **Behind Splash** on long-context decode (it keeps KV in INT8) and on concurrent long outputs (the
  upstream batch cache pads every row to the longest one). Quantized KV and a ragged per-row KV cache
  are being validated to close both.
- A 60-minute mixed soak (chat, long documents, images, tools, JSON schema, thinking, disconnects)
  finished 699 requests with 0 server errors and no memory growth (footprint 17–26 GiB).

Yunshu's matrix has two more checks than the older runs: logprobs and the streaming reasoning split.
The long-run benchmark log is [docs/reports/PERF_TREND.md](docs/reports/PERF_TREND.md).

## Supported models

| Tier | Models | Path | What you get |
|---|---|---|---|
| 1 — tuned and measured | Qwen3.5 / 3.6 / 3.8 family (text + images) | VLM batch runner | prefix cache (RAM + SSD), MTP / DFlash lossless spec decode, all API features above |
| 2 — supported | any `mlx-lm` text model | single-request fast path (`generate_step`) | KV prefix cache, tools, JSON schema, logprobs; opt-in n-gram spec, KV quant |
| 2 — supported | any other `mlx-vlm` model (GLM, Qwen-VL, Gemma-4, Qwen3-Omni, Nemotron-Omni, …) | the same VLM batch runner | continuous batching, prefix cache (unless the model uses a sliding window), images / audio / video, all API features above; no speculative decode |

Only tier 1 was re-measured in the 2026-09-28 round; tier-2 VLMs moved onto the runner afterwards and still need their real-model smoke run.

## Other modalities

These ship in the same server behind optional extras. **None were re-verified in the 2026-09-28
round**, which covered LLM/VLM only.

| Modality | Endpoint | Backend | Extra |
|---|---|---|---|
| Native speech-to-speech (Qwen3-Omni Thinker→Talker, streaming) | `POST /v1/omni/speech/stream` | `mlx-vlm` | `omni` |
| Realtime voice | `WS /v1/realtime` | omni, or ASR → LLM → TTS | `audio` |
| ASR | `/v1/audio/transcriptions` | `mlx-audio` / Whisper | `audio` |
| TTS | `/v1/audio/speech` | `mlx-audio` | `audio` |
| Image generation | `/v1/images/generations` | diffusion | `generation` |
| Video generation (Wan 2.x / LTX-2) | `/v1/video/generations` | `mlx-video` | `video` |
| Embeddings / rerank (text + multimodal) | `/v1/embeddings`, `/v1/rerank` | `mlx-lm` / `mlx-embeddings` | `embeddings` |

For speech-to-speech, serve a Qwen3-Omni model (`uv sync --extra omni`) and try
[`examples/talk.py`](examples/talk.py) (microphone) or [`examples/quickstart.py`](examples/quickstart.py)
(writes a WAV, no audio hardware). Upstream `mlx-vlm` 0.7.3 was checked to keep multi-turn omni
output correct ([notes](docs/research/runs/2026-09-28-omni/README.md)); the server's Realtime path
was not.

Also: MCP server/client and an Anthropic-compatible `/v1/messages` surface.

## Architecture

```
  Client (any OpenAI / Anthropic SDK)
        │   OpenAI / Anthropic / MCP / Realtime-WS / SSE
  ┌─────┴───────────────────────────────────────────────┐
  │  Gateway (FastAPI)     routers + middleware           │
  ├─────────────────────────────────────────────────────┤
  │  Engine                                               │
  │   · VLM batch runner (every mlx-vlm model)            │
  │       continuous batching · prefix cache (RAM + SSD)  │
  │       Qwen3.5 family: MTP / DFlash + batch-invariant  │
  │   · LLM fast path (mlx-lm generate_step)              │
  │       KV prefix cache · constrained decoding          │
  │   · other modalities: omni, ASR/TTS, image, video,    │
  │     embeddings                                        │
  └─────────────────────────────────────────────────────┘
        one MLX thread · runs on-device via Apple MLX
```

## Serving model

All GPU work runs on one MLX thread. VLM (mlx-vlm) models serve concurrent requests in one
continuous batch, each row with its own sampling settings; a request that is alone uses speculative
decoding (Qwen3.5 family), and requests that arrive meanwhile join the shared batch without it.
Text-only mlx-lm models use the single-request fast path, so their concurrent requests run one at a
time. Every response returns as soon as its own generation finishes.

## Configuration

Environment variables and `yunshu serve` flags are in the
[configuration reference](docs/CONFIGURATION.md), including the runner knobs
(`YUNSHU_VLM_APC_*`, `YUNSHU_MTP*`, `YUNSHU_VLM_DRAFT`, `YUNSHU_VLM_INVARIANT`).

## Built on

[MLX](https://github.com/ml-explore/mlx) · [mlx-lm](https://github.com/ml-explore/mlx-lm) ·
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) · [mlx-audio](https://github.com/Blaizzy/mlx-audio).
Some verify kernels are vendored from [oMLX](https://github.com/jundot/omlx) (Apache-2.0); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
