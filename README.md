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
  calls lossless.
- **Per-row KV for concurrent requests.** Rows of the shared batch keep their own KV length, so a
  short request never reads a long one's padding (Qwen3.5 family; MMLU-Pro at 8 in flight 88 → 139
  tok/s, 131K-context decode 24 → 47 tok/s).
- **Lossy only when you ask.** Every default is lossless. Memory savers that change outputs (int8
  KV, KV quantization, 4-bit cached prefixes, int8 SSD cache) are settings you turn on.
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

Needs a Mac with Apple Silicon (macOS 14+) and [uv](https://docs.astral.sh/uv/).

```bash
# Install. The vision extra covers the Qwen3.5 / 3.6 / 3.8 family and every VLM.
uv tool install "yunshu[vision]"

yunshu doctor                                   # checks this Mac and prints fixes
yunshu pull mlx-community/Qwen3.5-9B-MLX-4bit   # downloads to ~/.yunshu/models/
yunshu serve -m mlx-community/Qwen3.5-9B-MLX-4bit
```

Other ways to install: `pipx install "yunshu[vision]"`, Homebrew
(`brew install yuhuanstudio/tap/yunshu`), or the latest `main`
(`uv tool install "yunshu[vision] @ git+https://github.com/YuhuanStudio/Yunshu"`).

`yunshu serve -m org/name` uses a model already in the models directory or the Hugging Face cache
and downloads only when neither has it. Models live in `~/.yunshu/models`; to keep them elsewhere,
run `yunshu config set models_dir /path/to/models` (saved in `~/.yunshu/config.toml`).

The server listens on `http://127.0.0.1:8000`. Any OpenAI client works unchanged:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")  # any key works

# Single-model mode: the model name is a placeholder, the server serves what you loaded.
r = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Explain MLX in one sentence."}],
    extra_body={"reasoning_effort": "medium"},
)
print(r.choices[0].message.content)
```

To run it in the background at login: `yunshu service install -m <model>`
([service guide](docs/guides/SERVICE.md)). Every command has `--help`. `yunshu model list` shows
local models, including the Hugging Face cache.

**From source** (development): clone the repository, run `uv sync --extra vision` (or
`--all-extras`), then `uv run yunshu serve -m <model>`. `uv.lock` pins the exact versions
(MLX 0.32, `mlx-vlm` 0.7.3+).

**No telemetry.** Yunshu sends nothing anywhere. The only outgoing connections are the model
downloads you ask for and MCP servers you configure.

**Docs:**
- [connecting clients](docs/guides/CLIENTS.md) (OpenAI / Anthropic SDKs, coding agents, Open WebUI)
- [troubleshooting](docs/guides/TROUBLESHOOTING.md)
- [API reference](docs/API.md)
- [configuration reference](docs/CONFIGURATION.md)

## Performance

Measured on an M5 Max (128 GB), Qwen3.8-27B, 2026-09-28/29. Same Jundot `oQ4e-mtp` checkpoint unless
noted. How each table was measured (scripts and methods) is in [docs/BENCHMARKS.md](docs/BENCHMARKS.md);
the raw run data stays with the maintainers.

| Engine | Capability checks | Chat TTFT (warm) | 8K prompt: cold / repeat / edited tail | Decode tok/s |
|---|---|---|---|---|
| **Yunshu 0.1.1** (default: MTP block 6, batch-invariant, ragged KV) | 34/34 | 0.192 s | 8.42 / 0.115 / 0.259 s | 73 |
| mlx-vlm 0.7.3 server (APC) | 27/28 | 0.212 s | 8.60 / 0.108 / 0.265 s | 32 |
| oMLX.app 0.7 (MTP + cache) | 31/31 | 0.312 s | 8.60 / 0.361 / 0.376 s | 85 |
| Splash 1.1 (own quantized model + DFlash2) | 31/31 | 0.206 s | 7.88 / 0.131 / 7.88 s | 119 |
| TensorFold 0.3.6.1 (MTP, parallel 8) | 23/34 | — | — | 28 |

Yunshu's matrix has more checks than the older runs (logprobs, the streaming reasoning split);
TensorFold fails the image, tool, JSON-schema and logprobs checks.

Lossless single-request decode by output type (same checkpoint, in-process, greedy, 384 tokens;
tok/s):

| Context | Code | Prose | JSON-like | Spec on == off |
|---|---|---|---|---|
| 1K | 82.1 | 57.8 | 69.0 | yes (tested)¹ |
| 32K | 75.4 | 51.2 | 64.2 | yes (tested)¹ |
| 131K | 59.7 | 43.8 | 46.0 | yes (tested)¹ |

¹ Speculative and plain greedy output matched token for token on every task at every context
above. Matmuls are row-invariant, and decode and verify attention run one per-row kernel whose
result for a token does not depend on how many tokens are verified with it.

MMLU-Pro, 300 questions, 8 in flight, max 16384 tokens, `reasoning_effort=medium` (accuracy and a
long-run soak; same settings for every engine):

| Engine | Correct | Wall time | Aggregate tok/s | Peak footprint |
|---|---|---|---|---|
| **Yunshu** (ragged KV) | 249 / 300 | 28.3 min | 139 | 33.7 GiB |
| Yunshu 0.1.0-era shared batch (padded KV) | 250 / 300 | 46.5 min | 88 | 45 GiB |
| TensorFold 0.3.6.1 (MTP, parallel 8) | 250 / 300 | 24.9 min | 159 | 35.1 GiB |
| Splash 1.1 | 252 / 300 | 17.2 min | 223 | 67 GiB |
| oMLX.app | 229 / 300 (27 rejected by its prefill memory guard) | 29.2 min | 120 | 75 GiB |

With `YUNSHU_KV_PRECISION=int8` (lossy, opt-in) Yunshu scored 251 / 300 at a 28.1 GiB peak (measured
on an earlier build of the ragged cache, 34.6 min).

Speed sweep (unique prompts, no cache hits; 128 generated tokens; tok/s unless noted):

| | Yunshu | oMLX | Splash | TensorFold (MTP) |
|---|---|---|---|---|
| TTFT at 8K / 131K tokens | 8.6 / 207 s | 8.5 / 214 s | 7.9 / 207 s | 9.8 / 293 s |
| Decode after 1K / 32K / 131K | 59² / 59 / 47 | 71 / 60 / 38 | 101 / 48 / 68 | 26 / 57 / 19 |
| 8 concurrent 1K prompts, aggregate | 64 | 53 | 70 | 65 |

² Single-request decode depends on how many drafted tokens are accepted, which varies with the
prompt; Yunshu's 1K figure is the mean of 8 runs (single runs ranged 40–70). The other cells are
single runs.

Where Yunshu stands:
- Prefix reuse and warm TTFT are the best measured; cold prefill is at the hardware ceiling (every
  engine within ~10%).
- Accuracy matches the others; 0 errors in every long run.
- **Behind Splash** on long-context decode (131K: 47 vs 68 tok/s) and on concurrent long outputs
  (MMLU-Pro: 139 vs 223 tok/s). TensorFold is also ahead there (159) because it drafts for every
  row; Yunshu drafts only for a request that is alone. Multi-row speculative decoding is in
  progress (`YUNSHU_ROUND_DRIVER`, experimental).
- A long prompt that arrives while others decode stalls their decode during its prefill; every
  engine measured does this.
- A 60-minute mixed soak on the 2026-09-28 build (chat, long documents, images, tools, JSON schema,
  thinking, disconnects) finished 699 requests with 0 server errors and no memory growth
  (footprint 17–26 GiB).

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
output correct (maintainer check, 2026-09-28); the server's Realtime path
was not.

Video generation needs `mlx-video` from git: the PyPI release (0.1.0) only has preprocessing, so
`yunshu[video]` from PyPI installs an incomplete backend and `yunshu[all]` leaves video out. To add it:
`uv tool install "yunshu[vision]" --with "mlx-video @ git+https://github.com/Blaizzy/mlx-video.git"`
(a source checkout gets it with `uv sync --extra video`).

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

Every setting is a `YUNSHU_*` name listed in the
[configuration reference](docs/CONFIGURATION.md) (generated from one registry). Set it as an
environment variable, in a TOML file (`yunshu serve --config yunshu.toml`), or with
`yunshu serve --set KEY=VALUE`; `yunshu config` shows each effective value and where it came
from. A bad value stops startup; a misspelled name gets a warning.

## Built on

[MLX](https://github.com/ml-explore/mlx) · [mlx-lm](https://github.com/ml-explore/mlx-lm) ·
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) · [mlx-audio](https://github.com/Blaizzy/mlx-audio).
Some kernels are vendored from [oMLX](https://github.com/jundot/omlx) (Apache-2.0) and
[TensorFold](https://github.com/ashhart/TensorFold) (MIT); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
