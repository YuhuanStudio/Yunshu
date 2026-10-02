<div align="center">

# Yunshu

**A fast, local, single-node LLM / VLM inference engine for Apple Silicon.**

One OpenAI/Anthropic-compatible process, on-device via MLX. Built for LLM/VLM decode
speed, cold and cached time to first token (TTFT), prefix reuse and complete inference
features. **Qwen3.8-27B is the first fully tuned model.** Speech, Realtime voice,
image generation, video input and embeddings are supported capabilities; LLM/VLM serving is
the focus. Yunmo is one consumer.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml)

**English** · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md)

</div>

Latest release: **v0.1.2 (2026-10-02)**; **v0.1.1 (2026-09-29)**. This README also
covers **unreleased main**; see [CHANGELOG](CHANGELOG.md).

## Inference features

- **Speculative decoding.** Matching DFlash2 drafter discovered automatically, otherwise
  the checkpoint's MTP head. Invariant decode/verify kernels support greedy parity within
  that path; sampled drafting uses position-keyed draws. Main adds invariant verify up to
  32 rows and prompt-copy drafting in the MTP lane (on by default,
  `YUNSHU_SPEC_COPY_ROWS=0` opts out). Tree drafting remains experimental and off.
  Plain runner vs invariant-lane output is not universally identical: see
  [accuracy evidence](docs/guides/ACCURACY.md) and [benchmark limits](docs/BENCHMARKS.md).
- **Prefix reuse.** Hybrid Qwen3.5-family checkpoints retain attention KV and recurrent
  state; text and media keys prevent mismatched reuse. RAM APC plus a default bounded SSD
  cache survive repeated turns and restarts. Main adds optional WARM RAM and further
  storage tiers with measured restore-cost selection. Lossless WARM stays off; int8/int4
  cache formats and KV quantization are explicit lossy options. See
  [cache tiers](docs/guides/KV_CACHE_MATRIX.md).
- **Complete request features.** Tools, JSON-schema constraints, stop, logprobs, separate
  reasoning/content streams, template-supported reasoning effort and cancellation.
  `/v1/models` advertises each model's capabilities; unsupported requests get explicit
  errors. Main defaults JSON constraints to llguidance; unsupported constructs are rejected.
- **Coding agents.** Claude Code, Codex and opencode use native Messages / Responses /
  Chat APIs, server-side search/fetch/MCP, Files, Batches and Conversations. `yunshu launch`
  supplies model limits; `yunshu statusline` exposes live engine state for Claude Code.
  [Compatibility evidence](docs/guides/AGENT_COMPAT.md) distinguishes verified features
  from client limits.
- **Local diagnostics.** `yunshu doctor`, `yunshu cache status`, `yunshu cache gc` and
  `yunshu diagnose`; diagnostics stay local and contain no prompts.

## Quickstart

Apple Silicon, macOS 14+, Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv tool install "yunshu[vision]"
yunshu doctor
yunshu pull mlx-community/Qwen3.5-9B-MLX-4bit
yunshu serve -m mlx-community/Qwen3.5-9B-MLX-4bit
```

The server listens on `http://127.0.0.1:8000`. Models are stored in `~/.yunshu/models`;
`serve -m org/name` also finds the Hugging Face cache and downloads only if needed.
`yunshu config set models_dir /path/to/models` changes the location.

### Qwen3.8-27B

```bash
yunshu pull Jundot/Qwen3.8-27B-oQ4e-mtp
yunshu pull incoai/Qwen3.8-27B-DFlash2
yunshu doctor -m Jundot/Qwen3.8-27B-oQ4e-mtp
yunshu serve -m Jundot/Qwen3.8-27B-oQ4e-mtp
```

With the matching drafter installed, the startup log should say `Speculative decoding:
dflash`; `doctor` reports the selected path. `YUNSHU_VLM_DRAFT=mtp` forces MTP,
`YUNSHU_VLM_DRAFT=off` disables drafting, and an absolute drafter path selects it explicitly.
Memory needs depend on weights, context, drafting and cache budgets; use the model check
in `doctor` on your Mac instead of treating a benchmark footprint as a minimum-RAM promise.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
r = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Explain MLX in one sentence."}],
    extra_body={"reasoning_effort": "medium"},
)
print(r.choices[0].message.content)
```

Any key works unless `--auth-token` was set. In single-model mode `local` is a placeholder;
in multi-model mode use an id from `/v1/models`.

For unreleased main / source development:

```bash
git clone https://github.com/YuhuanStudio/Yunshu.git
cd Yunshu
uv sync --extra vision
uv run yunshu serve -m Jundot/Qwen3.8-27B-oQ4e-mtp
```

`uv.lock` pins dependencies; main requires MLX 0.32.3+, mlx-lm 0.32.0+, mlx-vlm 0.7.4+.
`yunshu model list` lists local models. `yunshu service install -m <model>` installs the
login service ([service guide](docs/guides/SERVICE.md)); each command has `--help`.

**No telemetry.** Model downloads, configured MCP/search providers and request-triggered
web fetches can connect out. No usage analytics or crash reports are sent. Optional main
`YUNSHU_SERVE_LOG` writes local numbers only, without prompts, outputs or token ids.

## Performance

Every measurement below is sourced in [BENCHMARKS](docs/BENCHMARKS.md) or the
[dated PERF_TREND log](docs/reports/PERF_TREND.md). M5 Max / 128 GB, Qwen3.8-27B
Jundot oQ4e-mtp unless stated otherwise. Dates identify measurements, not release dates.

**2026-10-02 tfbench**, recorded in `4e1338c4`: greedy, 256 output tokens, three
independent server sessions per cell, medians. Yunshu was that morning's **unreleased
main (SHA not recorded), actually using MTP**; TensorFold used DFlash2. This predates
main's prefill / prompt-copy / wide-verify merges and is not a same-drafter A/B.

| Context / output | TensorFold 0.6.1 tok/s / cold TTFT s | Yunshu main (MTP) tok/s / cold TTFT s |
|---|---|---|
| 1K code | 140.5 / 1.2 | 69.3 / 1.4 |
| 1K prose | 72.4 / 1.2 | 52.9 / 1.4 |
| 8K code | 80.6 / 8.4 | 60.2 / 11.1 |
| 8K prose | 67.4 / 8.5 | 57.5 / 11.1 |
| 32K code | 91.2 / 39.4 | 58.7 / 47.2 |
| 32K prose | 55.1 / 39.4 | 46.3 / 47.7 |

TensorFold is ahead in all these cells. **Splash is ahead in the 2026-09-27 exploratory
TTFT check**: 3.221 / 0.130 s cold / repeat, versus Yunshu direct VLMEngine 3.511 /
3.512 s, for a 3,323-token prompt. Splash 1.1.0 used its own quantized model and INT8
KV; single runs, shared GPU, Yunshu SHA unknown. These are different weights and cache
conditions, not an engine-only comparison. No updated full comparison follows the merges.

**Later 2026-10-02 main measurements, unreleased:**

| Workload | Before | After | Commit / source |
|---|---|---|---|
| Cold TTFT, 8K | 11.0 s | 8.57 s | `a4d71bc8`, BENCHMARKS |
| Cold TTFT, 32K | 47.5 s | 38.3 s | `a4d71bc8`, BENCHMARKS |
| MTP prompt-copy, code turn 2, 32K | 63 tok/s | 100–138 tok/s | `c666be70` / `bb4895ca`, PERF_TREND |
| MTP prompt-copy, code turn 2, 8K | 62 tok/s | 86 tok/s | same |

Prefill figures are the dated merge record; public repeat counts / intervals are unavailable.
Prompt-copy greedy digests matched off/on; prose stayed within ±3% single-run noise.
Cold-prefill numerics can differ from earlier builds; APC namespaces include the prefill
settings. No universal decode speedup or hardware-ceiling claim is implied.

**Agentic coding, partial, measured 2026-10-02** (PERF_TREND / `2815311c`):

| Snapshot / agent | Pass / runs | Rate (Wilson 95%) | Wall median / p90 s | Decode median tok/s |
|---|---|---|---|---|
| prod `c4e2b244+` / opencode | 34 / 41 | 83% (69–91%) | 337 / 1200 | 22.2 |
| prod4 `d225f16c` / Claude Code | 14 / 15 | 93% (70–99%) | 93 / 309 | 60.8 |
| prod4 `d225f16c` / opencode | 9 / 10 | 90% (60–98%) | 112 / 137 | 56.9 |

Failures remain in the denominator, including five prod timeouts. Prod4 covers 10 of 20
tasks, with 1–3 repeats; snapshot `d225f16c` is included in v0.1.2. Different snapshots
and task mixes prevent claiming a paired speedup. Codex and TensorFold agentic results
are pending. Accuracy Tier 3 and APC replay results, including losses and incomplete
subsets, are in [BENCHMARKS](docs/BENCHMARKS.md).

## Models and serving paths

| Models | Serving path | Scope |
|---|---|---|
| Qwen3.5 / 3.6 / 3.8 family; Qwen3.8-27B tuned first | VLM batch runner | APC, MTP/DFlash on supported family models, per-row sampling |
| Other mlx-vlm models (Gemma, GLM, Qwen-VL, Omni, …) | Same VLM batch runner | model-dependent media and tools; APC where cache layout permits; no family-specific speculation |
| Text-only mlx-lm models | Single-request `generate_step` fast path | prefix cache, constraints, tools and logprobs where supported; concurrent requests serialize |

All GPU work runs on one MLX thread. The default VLM runner shares concurrent decode
rows; a lone supported request can draft. Multi-row drafting uses the experimental
`YUNSHU_ROUND_DRIVER`, off by default, with measured latency tradeoffs. Main's M01 split
organizes the text engine into modules without changing serving paths. See the model
card for actual capabilities, not just a model-family name.

## Other supported capabilities

| Capability | API / example | Extra |
|---|---|---|
| Native Qwen3-Omni speech-to-speech | `/v1/omni/speech/stream`, [talk.py](examples/talk.py) | `omni` |
| Realtime voice, ASR, TTS | `/v1/realtime`, `/v1/audio/transcriptions`, `/v1/audio/speech` | `audio` (native Omni needs `omni`) |
| Image generation | `/v1/images/generations` | `generation` |
| Embeddings / rerank | `/v1/embeddings`, `/v1/rerank` | `embeddings` |
| Text WebSocket / Responses WebSocket / Unix socket | `/v1/stream`, `/v1/responses`, `yunshu serve --uds PATH` | core |

These capabilities have separate model/backend requirements and are not covered by the
LLM/VLM performance tables. Video input is supported by applicable VLMs; video generation
has no engine backend or public HTTP route. WebRTC and HTTP/2 remain unimplemented.
[API surface](docs/guides/API_SURFACE.md) and [transports](docs/guides/TRANSPORTS.md)
list the checks and limits.

## Configuration and docs

Settings go through one registry: environment, TOML (`yunshu serve --config yunshu.toml`),
or `yunshu serve --set KEY=VALUE`; `yunshu config` shows effective values and sources.

- [Clients](docs/guides/CLIENTS.md)
- [Troubleshooting](docs/guides/TROUBLESHOOTING.md)
- [API reference](docs/API.md)
- [Configuration](docs/CONFIGURATION.md)
- [Documentation index](docs/README.md)

## Built on and license

[MLX](https://github.com/ml-explore/mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm),
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm), [mlx-audio](https://github.com/Blaizzy/mlx-audio).
Vendored kernels from [oMLX](https://github.com/jundot/omlx) and
[TensorFold](https://github.com/ashhart/TensorFold) have their notices in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md). Yunshu is Apache 2.0: [LICENSE](LICENSE).
