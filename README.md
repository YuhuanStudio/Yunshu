<div align="center">

# Yunshu

**A fast local LLM / VLM inference engine for Apple Silicon.**

One process, OpenAI- and Anthropic-compatible, running on-device via MLX. Built for low latency on
a single Mac: fast first token, fast lossless decode, and prefix reuse that skips work you already
paid for. The first fully tuned model is **Qwen3.8-27B**.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml)

**English** · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md)

</div>

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/console/en/overview-dark.webp">
  <img src="docs/images/console/en/overview-light.webp" alt="Yunshu console overview: live request phases, tok/s, prefix-cache hits">
</picture>


## Highlights

- **Lossless speculative decoding** — DFlash2 or MTP drafting with batch-invariant kernels: greedy
  output with speculation on is token-identical to speculation off.
- **Prefix cache for hybrid models** — exact checkpoints for attention + GatedDeltaNet models, in RAM
  and on SSD, surviving restarts; optional extra storage tiers.
- **The full API on the fast path** — tools, JSON schema, stop, logprobs, reasoning, cancellation,
  across OpenAI Chat / Responses, Anthropic Messages and Ollama.
- **Typed decisions** — `/v1/decisions` answers yes/no, choice and score questions with calibrated
  probabilities in one forward pass of a decision checkpoint, without generating text.
- **Native coding-agent support** — Claude Code, Codex and opencode work through their own APIs,
  including server-side web search / fetch and MCP.
- **Lossless by default** — anything that can change output is an explicit setting.
- **Local and private** — no usage analytics; diagnostics never contain prompts.
- **A real console** — a local web console with live request phases (queued, prefill with cache-hit
  segments, decode), request history, logs, models, cache, a playground, API keys with per-key quotas,
  settings (including a CORS editor) and host power / clock / temperature, plus a floating status
  island. [Console guide](docs/CONSOLE.md).
- **Web search with citations** — server-side `web_search` / `web_fetch` for Claude Code, Codex and the
  APIs, six providers (DuckDuckGo, Wikipedia, SearXNG, Tavily, Serper, Perplexity), a Tavily-compatible
  API and optional research enrichment. [Guide](docs/guides/WEB_SEARCH.md).
- **Evals and stored completions** — `store: true` keeps chat completions locally and the OpenAI Evals
  API grades them with local graders. [Guide](docs/guides/EVALS.md).
- **Host telemetry** — power, GPU clock, temperatures, thermal and memory pressure, and a per-request
  energy estimate, unprivileged and local. [Guide](docs/guides/TELEMETRY.md).
- **Multiple API keys** — scopes, expiry and daily quotas per key. [Guide](docs/guides/AUTH_AND_KEYS.md).

## Capabilities

Everything Yunshu serves, grouped. Each row links to the guide with an example and the limits; items marked partial or experimental say so. The same list is machine-readable in [`docs/feature_index.json`](docs/feature_index.json).

**Inference**

| Capability | What it does |
|---|---|
| [Lossless speculative decoding](docs/guides/INFERENCE.md) | DFlash2, MTP and prompt-copy drafting with batch-invariant kernels; greedy output equals speculation off |
| [Text-model speculation](docs/guides/INFERENCE.md) | n-gram / suffix and Gemma 4 assistant drafting for mlx-lm models (external-draft experiments stay experimental) |
| [Prefix cache (APC)](docs/guides/KV_CACHE_MATRIX.md) | exact hybrid checkpoints in RAM, SSD and optional storage tiers; image and audio keys |
| [Structured output](docs/guides/INFERENCE.md) | JSON schema, regex, choice and grammar enforced during decoding, including tool arguments |
| [Lossy memory options](docs/guides/INFERENCE.md) | int8 KV, compressed WARM tier and load-time quantization; off by default |
| [Large MoE loading (partial)](docs/guides/INFERENCE.md) | bits and group size derived from tensor shapes for community DeepSeek-V4-Flash packs; fit check before loading |
| [Round driver (experimental)](docs/guides/ROUND_DRIVER.md) | own speculative round driver for dense Qwen3.5-family models |
| [Model support](docs/guides/MODEL_SUPPORT.md) | what is tuned, generic and validated, per family |

**APIs**

| Capability | What it does |
|---|---|
| [OpenAI API](docs/guides/API_SURFACE.md) | Chat, Completions, Responses (HTTP and WebSocket), Embeddings, Models, Files, Batches, Conversations |
| [Anthropic API](docs/guides/API_SURFACE.md) | Messages, count_tokens, batches, thinking, tools, cache_control, documents and citations |
| [Ollama API](docs/guides/API_SURFACE.md) | /api/chat, /api/generate and native model management (pull, copy, delete, show) |
| [Responses extras](docs/guides/API_SURFACE.md) | client-executed computer tool, conversations, compaction, previous_response_id |
| [Stored chat completions](docs/guides/API_SURFACE.md) | store: true, then list, retrieve, update, delete and list messages |
| [Evals](docs/guides/EVALS.md) | OpenAI Evals API with string-check, similarity and local model graders |
| [Typed decisions](docs/guides/DECISIONS.md) | yes/no, choice and score answers with probabilities in one forward pass, no text generation |
| [Batches, Files and tokenizer](docs/guides/API_SURFACE.md) | OpenAI and Anthropic batches, Files API, tokenize / detokenize / apply-template / props |
| [Transports](docs/guides/TRANSPORTS.md) | SSE, WebSocket streams, Unix socket, HTTP/2, optional WebRTC |
| [Yunshu extensions](docs/guides/API_EXTENSIONS.md) | request ids, live phases, prefill progress, cancel, deadlines, x_yunshu stats, warmup, memory units |
| [Prompt-caching APIs](docs/guides/PROMPT_CACHING_APIS.md) | cache_control and cached-token accounting across the three dialects |

**Agents & tools**

| Capability | What it does |
|---|---|
| [Coding agents](docs/guides/AGENT_COMPAT.md) | Claude Code, Codex and opencode through their own APIs; `yunshu launch` |
| [Web search and fetch](docs/guides/WEB_SEARCH.md) | server-side web_search / web_fetch with citations; DuckDuckGo, Wikipedia, SearXNG, Tavily, Serper, Perplexity |
| [Tavily-compatible API](docs/guides/TAVILY.md) | search, extract, crawl, map, research and MCP, with provider health backoff |
| [MCP](docs/guides/API_SURFACE.md) | native MCP endpoint, and remote MCP servers named in a request |
| [Client setup](docs/guides/CLIENTS.md) | curl, OpenAI and Anthropic SDKs, Open WebUI |

**Multimodal**

| Capability | What it does |
|---|---|
| [Image, audio and video input](docs/guides/MULTIMODAL.md) | mlx-vlm models take media in Chat, Responses and Messages; media enters the cache key |
| [Speech-to-speech (partial)](docs/guides/MULTIMODAL.md) | Qwen3-Omni native speech; a repeated image does not yet hit the cache |
| [Realtime voice](docs/guides/MULTIMODAL.md) | WebSocket and optional WebRTC, server VAD, barge-in, ephemeral client secrets, local voices |
| [Speech to text, text to speech](docs/guides/MULTIMODAL.md) | transcriptions, translations, speech and voices |
| [OCR and image generation](docs/guides/MULTIMODAL.md) | GLM-OCR text extraction; image generations, edits and variations |
| [Embeddings, rerank, score, classify](docs/guides/MULTIMODAL.md) | text and multimodal embeddings, cross-encoder rerank, trained classification heads |

**Operations & console**

| Capability | What it does |
|---|---|
| [Web console](docs/CONSOLE.md) | live request phases, history, logs, models, downloads, cache, playground, keys, settings, status island |
| [API keys and quotas](docs/guides/AUTH_AND_KEYS.md) | several keys with scopes, expiry, per-key usage and daily quotas |
| [Settings and CORS editor](docs/guides/AUTH_AND_KEYS.md) | write engine settings (restart-required ones prompt) and edit CORS from the console or API |
| [Host telemetry](docs/guides/TELEMETRY.md) | power, GPU clock, temperatures, thermal and memory pressure, per-request energy estimate, Prometheus |
| [Command line](docs/guides/CLI.md) | setup, doctor, models list/pull/rm/show, top, service, completions and inference commands |
| [Login service](docs/guides/SERVICE.md) | launchd agent, restart and logs |
| [Configuration](docs/CONFIGURATION.md) | one settings registry: environment, TOML file or `--set` |
| [Security](docs/guides/AUTH_AND_KEYS.md) | auth token, CORS credentials rule, redirect header stripping, secret redaction, SSRF guard |
| [First run and troubleshooting](docs/guides/FIRST_RUN.md) | install, choose a model, readiness checks, fixes |
| [Benchmarks and accuracy](docs/BENCHMARKS.md) | where each number comes from, how to reproduce it, accuracy gates |

## Quickstart

For model selection, external storage, readiness checks and upgrades, follow the [first-run guide](docs/guides/FIRST_RUN.md).

Apple Silicon, macOS 14+, Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv tool install --python 3.13 "yunshu[vision]"      # or: brew install yuhuanstudio/tap/yunshu
yunshu doctor                                   # checks this Mac and says how to fix problems
yunshu pull mlx-community/Qwen3.5-9B-MLX-4bit
yunshu serve -m mlx-community/Qwen3.5-9B-MLX-4bit
```

The server listens on `http://127.0.0.1:8000`. Any OpenAI or Anthropic client works unchanged:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")  # any key works
r = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Explain MLX in one sentence."}],
)
print(r.choices[0].message.content)
```

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:8000", api_key="local")
msg = client.messages.create(
    model="local", max_tokens=512,
    messages=[{"role": "user", "content": "Explain MLX in one sentence."}],
)
print(msg.content[0].text)
```

Models live in `~/.yunshu/models` (`yunshu config set models_dir PATH` selects the directory for future downloads; it does not move existing weights);
`serve -m org/name` also finds the Hugging Face cache and downloads only if needed.
`yunshu service install -m <model>` runs the server at login.

### Local console

The local web console at `/console/` shows what the engine is doing and manages it, in English,
Traditional Chinese or Simplified Chinese. It ships inside the `yunshu` package (pip, uv tool and
Homebrew installs alike), so there is nothing to build:

```bash
yunshu serve -m <model>
open http://127.0.0.1:8000/console/
```

- **Overview** — live request phases, prefill and decode tok/s, prefix-cache hit segments inside the
  prefill bar, time to first token, Metal memory, host power, clock and temperature.
- **Requests** and **Logs** — latency distribution (cold and warm), speculative acceptance, request
  history, cancel; a redacted live log with search and download.
- **Diagnostics** — resource readouts, health checks, a Realtime probe and the support bundle.
- **Models**, **Downloads**, **Cache** — load, unload, warm up and fit-check models, pull from Hugging
  Face with progress, and inspect or clear the RAM / WARM / SSD prefix-cache tiers.
- **Playground** and **API access** — stream a chat with reasoning, compare, view the code; paste-ready
  setup for Claude Code, Codex, opencode, the SDKs and curl.
- **Keys** and **Settings** — API keys with quotas and usage, every engine setting (restart-required
  ones prompt), and a CORS editor.
- A floating **status island** follows you across pages, and the layout works on a phone.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/console/en/requests-dark.webp">
  <img src="docs/images/console/en/requests-light.webp" alt="Requests and latency distribution">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/console/en/playground-dark.webp">
  <img src="docs/images/console/en/playground-light.webp" alt="Playground streaming a reply">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/console/en/keys-dark.webp">
  <img src="docs/images/console/en/keys-light.webp" alt="API keys with quotas and usage">
</picture>

<img src="docs/images/console/en/mobile-overview-dark.webp" alt="Console on a phone" width="260">

Full page-by-page tour with screenshots: [Console guide](docs/CONSOLE.md).

### Qwen3.8-27B

```bash
yunshu pull Jundot/Qwen3.8-27B-oQ4e-mtp
yunshu pull incoai/Qwen3.8-27B-DFlash2          # optional drafter, picked up automatically
yunshu serve -m Jundot/Qwen3.8-27B-oQ4e-mtp
```

With the drafter installed the startup log says `Speculative decoding: dflash`; without it the
model's own MTP head drafts. `yunshu doctor -m <model>` reports the selected path and whether the
model fits this Mac's memory.

### From source

```bash
git clone https://github.com/YuhuanStudio/Yunshu.git && cd Yunshu
uv sync --extra vision
uv run yunshu serve -m <model>
```

To work on the console itself, run `pnpm install && pnpm dev` in `frontend/` (see the [console guide](docs/CONSOLE.md#developing-the-console)).

## How it works

```
 OpenAI / Anthropic / Ollama clients ──► FastAPI gateway (one process)
                                           │  request validation, tool/reasoning parsing,
                                           │  server-side tools (web search / fetch / MCP)
                                           ▼
                                  engine (one MLX thread)
          ┌────────────────────────────────┴───────────────────────────────┐
   VLM batch runner (every mlx-vlm model)                 text fast path (mlx-lm models)
   shared continuous batch, per-row sampling              single-request generate_step
   speculative lane: DFlash2 / MTP / prompt-copy
   prefix cache: RAM ─► SSD ─► optional storage tiers
```

All GPU work runs on a single MLX thread, so requests never fight over the GPU. Each response
returns as soon as its own generation ends.

### Speculative decoding

For Qwen3.5-family models a single request decodes in a speculative lane:

- **DFlash2** — a separate block drafter proposes several tokens per round; used automatically when
  a matching drafter is installed.
- **MTP** — the checkpoint's own multi-token-prediction head; the fallback when no drafter is present.
- **Prompt-copy drafting** — when the output starts repeating text from the prompt (code edits,
  quoting tool results, multi-turn agents), the lane proposes the continuation of that earlier text
  and verifies it in the same pass. It is on by default (`YUNSHU_SPEC_COPY_ROWS`, `0` turns it off).

Every decode and verify matmul goes through **batch-invariant kernels**: a token's arithmetic is the
same whether it is verified alone or among other rows. That is what makes greedy output identical
with speculation on and off, and sampled output exact under position-keyed sampling. Choose the
drafter with `YUNSHU_VLM_DRAFT` (`mtp`, `off`, or a drafter path).

### Prefix cache (APC)

Qwen3.5-family models mix attention layers with recurrent GatedDeltaNet layers. A recurrent state
cannot be cut back to an earlier token the way a KV cache can, so ordinary prefix caching does not
work. Yunshu stores **exact checkpoints** (KV plus recurrent state) at prefix boundaries, keyed by
the text tokens and by image pixels / audio features, so a cache hit produces exactly what a cold
prefill would.

| Tier | Where | Default |
|---|---|---|
| HOT | ready-to-use arrays in RAM | on, sized from free memory (`YUNSHU_VLM_APC_MEMORY_GB`) |
| WARM | compressed in RAM (lossless zstd, or lossy int8 / int4) | off (`YUNSHU_VLM_APC_WARM`) |
| SSD | `~/.yunshu/cache/apc`, one global disk budget with a free-space reserve, survives restarts | on (`YUNSHU_VLM_APC_DISK`, `_DIR`, `_GB`) |
| Storage tiers | external SSD, HDD, NAS (`YUNSHU_VLM_APC_DISK_TIERS`) | off; each volume's speed is measured and it is used only when restoring beats recomputing |

Repeated or edited long prompts, multi-turn chats and agent loops restore from the nearest
checkpoint instead of prefilling again. When a conversation grows, older checkpoints of the same
conversation are replaced rather than piling up. `yunshu cache status` and `yunshu cache gc` show
and clean the SSD caches.

### Structured output

JSON schema, JSON object, regex and grammar constraints are enforced during decoding (llguidance by
default), including tool-call arguments. Unsupported schema constructs are rejected with an error
instead of being silently ignored.

## API compatibility

| API | Routes |
|---|---|
| OpenAI | `/v1/chat/completions`, `/v1/completions`, `/v1/responses` (HTTP and WebSocket), `/v1/embeddings`, `/v1/models`, `/v1/audio/*`, `/v1/images/*`, `/v1/realtime`, `/v1/files`, `/v1/batches` |
| Anthropic | `/v1/messages` (thinking, tools, `cache_control`, server tools `web_search` / `web_fetch`, `mcp_servers`), `/v1/messages/count_tokens`, `/v1/messages/batches`, Files |
| Ollama | `/api/chat`, `/api/generate` and the model routes |
| Yunshu extensions | live request phases (`/v1/requests`), cancel by request id, warmup, deadlines, queue headers, prefill progress in streams, `/v1/yunshu/status` |

Parameters covered on chat: `tools` / `tool_choice` / `parallel_tool_calls`, `response_format`
(`json_object`, strict `json_schema`), `stop`, `logprobs` / `top_logprobs` (also streamed), `n`,
`seed`, penalties, `logit_bias`, `reasoning_effort` (reasoning returned separately), streaming with
usage, and cached-token counts in `usage`. Errors use each API's own error shape. Extensions are
namespaced (`x_yunshu`, `X-Yunshu-*`), so the official SDKs ignore them. The full matrix, with how
each row was verified, is in [API surface](docs/guides/API_SURFACE.md).

### Typed decisions

`POST /v1/decisions` (and the System One wire format, `/v1/systemone`) asks a decision checkpoint
predicate, choice and ordinal-score questions about text or images. One forward pass returns typed
answers with probabilities; no tokens are generated, and a non-finite result is a refusal, never a
made-up probability. Supported today: Cloudflare Clef / Clef-flash MLX checkpoints. Other decision-head
families (OpenJev, Laya, D1) are not loaded yet.

```python
decision = client.decisions.create(
    model="abenzerps/Clef-MLX",
    input="The customer asks for a refund after a broken delivery.",
    questions=[{"type": "choice", "name": "route", "instructions": "Choose the support team.",
                "choices": [{"value": "support"}, {"value": "sales"}]}],
)
print(decision.answers)  # typed value, confidence, every option's probability
```

Also new on main for 0.1.5: stored chat completions, Evals (`/v1/evals`) and Realtime client secrets.

[Decisions](docs/guides/DECISIONS.md), [Evals](docs/guides/EVALS.md), [web search](docs/guides/WEB_SEARCH.md) and [Tavily API](docs/guides/TAVILY.md)

## Coding agents

```bash
yunshu launch claude      # or: codex, opencode
```

`yunshu launch` writes the client configuration (base URL, model, context window and output limits,
reasoning effort) and starts the agent. For Claude Code it also installs a status line showing live
prefill progress, decode speed and cache hits.

- **Claude Code** — Messages API with streaming, thinking, `count_tokens` for `/context`, model
  discovery, and its WebSearch tool, which Yunshu runs server-side.
- **Codex** — Responses API with reasoning items, function calls, local compaction and `web_search`.
- **opencode** — Chat Completions with tools and usage.

Server-side web search defaults to best-effort DuckDuckGo HTML and Wikipedia; queries leave the machine.
`YUNSHU_WEB_SEARCH_PROVIDER=none` disables it. Auto metasearch runs configured keyed providers in parallel with the keyless sources;
SearXNG is optional. MCP servers named in a request are connected by the gateway. What each agent calls and how it was checked is in
[Agent compatibility](docs/guides/AGENT_COMPAT.md).

## Performance

Qwen3.8-27B (oQ4e) on an M5 Max, 128 GB, single request, greedy. Methods, raw results and the full
comparison tables are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| | Yunshu | TensorFold 0.6.1 |
|---|---|---|
| Cold TTFT, 8K prompt | 8.6 s | 8.5 s |
| Cold TTFT, 32K prompt | 38.3 s | 39.4 s |
| Repeated / edited long prompt | restores from the prefix cache instead of re-prefilling | — |
| Follow-up turn TTFT, 8K / 32K code | 512 / 721 ms | 505 / 670 ms |
| Decode, short code prompt | ~110 tok/s (DFlash2) | ~140 tok/s (DFlash2) |
| JSON-schema / tool-call output, warm | 111 / 78 tok/s (speculative decoding stays on) | — |

Snapshot of main on 2026-10-02/03 (unreleased after v0.1.4; the table's TensorFold column is the 2026-10-02 same-window run); a fresh same-window 0.1.4 comparison is pending.
TensorFold is still faster at single-request decode and at 32K follow-up turns; closing those gaps is the
main ongoing work. Structured output keeps speculative decoding (23 tok/s without it).
Speculation never changes Yunshu's greedy output. Accuracy against the stock MLX path is checked at
three levels (logit alignment, greedy divergence, paired downstream evals) in
[Accuracy](docs/guides/ACCURACY.md).

## Models

| Models | Serving path |
|---|---|
| Qwen3.5 / 3.6 / 3.8 family (Qwen3.8-27B tuned first) | VLM batch runner with prefix cache and MTP / DFlash2 speculation |
| Other mlx-vlm models (Gemma, GLM, Qwen-VL, Qwen-Omni, …) | Same runner; images, audio and video input where the model supports them; prefix cache where the cache layout allows |
| Text-only mlx-lm models | Single-request fast path with constraints, tools and logprobs |

`/v1/models` returns each model's card: context length, output limit, and which inputs and features
it actually supports.

## Other capabilities

| Capability | Endpoint | Extra |
|---|---|---|
| Qwen3-Omni speech-to-speech | `/v1/omni/speech/stream` ([example](examples/talk.py)) | `omni` |
| Realtime voice, ASR, TTS | `/v1/realtime`, `/v1/audio/transcriptions`, `/v1/audio/speech` | `audio` |
| OCR | `/v1/ocr` (GLM-OCR) | `vision` |
| Image generation and editing | `/v1/images/generations`, `/v1/images/edits` | `generation` |
| Embeddings, rerank, scoring | `/v1/embeddings`, `/v1/rerank`, `/v1/score` | `embeddings` |

## Command line

| Command | What it does |
|---|---|
| `yunshu doctor` | check this Mac, dependencies and a model; says how to fix problems |
| `yunshu pull` / `yunshu model` | download and manage models |
| `yunshu serve` / `yunshu service` | run the server, or install it as a login service |
| `yunshu launch` / `yunshu statusline` | start a coding agent wired to Yunshu; live engine status line |
| `yunshu chat`, `complete`, `embed`, `transcribe`, `speak`, `ocr`, `image` | use a running server from the terminal |
| `yunshu status`, `cancel` | server state, cancel an in-flight request |
| `yunshu top` | (0.1.5) live view of server health, memory, engine and models (`--json` for one snapshot) |
| `yunshu config` | effective settings and where each came from |
| `yunshu cache status` / `gc` | inspect and clean the SSD prefix caches |
| `yunshu bench`, `eval`, `diagnose` | benchmarks, accuracy evals, system diagnostics |

Every command has `--help`.

## Configuration

Every setting goes through one registry: environment variables, a TOML file
(`yunshu serve --config yunshu.toml`) or `--set KEY=VALUE`. `yunshu config` shows the effective
values and their sources. Common ones:

| Setting | Purpose |
|---|---|
| `YUNSHU_VLM_DRAFT` | drafter choice: `mtp`, `off`, or a drafter path |
| `YUNSHU_SPEC_COPY_ROWS` | prompt-copy drafting width (`0` = off) |
| `YUNSHU_VLM_APC_MEMORY_GB`, `YUNSHU_VLM_APC_DISK_GB`, `YUNSHU_VLM_APC_DISK_DIR` | prefix-cache RAM and SSD budgets and location |
| `YUNSHU_VLM_APC_DISK_TIERS` | extra storage tiers, e.g. `/Volumes/Ext/apc@200,/Volumes/NAS/apc` |
| `YUNSHU_VLM_APC_WARM`, `YUNSHU_KV_PRECISION` | lossy memory savers (off by default) |
| `YUNSHU_AUTH_TOKEN`, `YUNSHU_QUEUE_LIMIT` | API key and request queue limit |

All settings: [Configuration](docs/CONFIGURATION.md).

## Docs

- [Console](docs/CONSOLE.md) — every page, with screenshots; [CLI](docs/guides/CLI.md) — first run, models, launchd, JSON and shell completion
- [Inference features](docs/guides/INFERENCE.md), [Multimodal endpoints](docs/guides/MULTIMODAL.md), [Model support](docs/guides/MODEL_SUPPORT.md)
- [Clients](docs/guides/CLIENTS.md) — curl, OpenAI / Anthropic SDKs, Open WebUI, agents
- [API surface](docs/guides/API_SURFACE.md), [API extensions](docs/guides/API_EXTENSIONS.md), [API reference](docs/API.md), [Transports](docs/guides/TRANSPORTS.md)
- [Decisions](docs/guides/DECISIONS.md), [Evals](docs/guides/EVALS.md), [Web search](docs/guides/WEB_SEARCH.md), [Tavily API](docs/guides/TAVILY.md)
- [Agent compatibility](docs/guides/AGENT_COMPAT.md)
- [Authentication, keys, settings and CORS](docs/guides/AUTH_AND_KEYS.md), [Telemetry](docs/guides/TELEMETRY.md)
- [KV cache tiers](docs/guides/KV_CACHE_MATRIX.md) and [prompt-caching APIs](docs/guides/PROMPT_CACHING_APIS.md)
- [Benchmarks](docs/BENCHMARKS.md) and [Accuracy](docs/guides/ACCURACY.md)
- [First run](docs/guides/FIRST_RUN.md), [Service](docs/guides/SERVICE.md), [Troubleshooting](docs/guides/TROUBLESHOOTING.md), [Configuration](docs/CONFIGURATION.md), [Changelog](CHANGELOG.md)

## Privacy

No usage analytics or crash reports are sent. Host telemetry (power, thermal, memory) is sampled locally for the console and never leaves the machine ([guide](docs/guides/TELEMETRY.md)). Yunshu connects out only to download models, to
web-search / MCP providers you configure, and for web fetches a request asks for.

## Built on and license

[MLX](https://github.com/ml-explore/mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm),
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm), [mlx-audio](https://github.com/Blaizzy/mlx-audio).
Vendored kernels from [oMLX](https://github.com/jundot/omlx) and
[TensorFold](https://github.com/ashhart/TensorFold) keep their notices in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md). Yunshu is Apache 2.0: [LICENSE](LICENSE).
