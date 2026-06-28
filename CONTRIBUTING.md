# Contributing to Yunshu

Thank you for your interest in contributing to Yunshu! This guide covers the minimum you need to get a
working dev loop. Read the repo's `CLAUDE.md` for the full architectural picture (what's live, what's
dead, what's being refactored).

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4) — Yunshu does not run on x86/Linux/GPU.
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- [just](https://github.com/casey/just) command runner (`brew install just`)

Optional, only if you touch those surfaces:

- [pnpm](https://pnpm.io/) for the WebUI
- Xcode Command Line Tools for Metal shader compilation

## Setup

```bash
git clone https://github.com/YuhuanStudio/Yunshu.git
cd Yunshu
just setup
```

This runs `uv sync` (core + dev deps). The heavy modality backends are **opt-in extras**:

```bash
uv sync --extra vision        # mlx-vlm (VLM / OCR)
uv sync --extra audio         # mlx-audio (ASR / TTS / Realtime voice)
uv sync --extra generation    # diffusers + torch (image / video generation)
uv sync --extra embeddings    # mlx-embeddings (/v1/embeddings)
uv sync --all-extras          # everything — what most contributors want
```

## Development loop

```bash
just lint              # ruff check
just format            # ruff format
just test-unit         # unit tests only (fast — no models needed)
just test              # full suite (unit + integration)
just test-single tests/unit/test_foo.py   # one file
```

Run `just dev` to start the gateway on `:8000` against a local model. CI runs
`ruff check python/ tests/` + `pytest tests/unit -q` on every push — keep that green.

## Where things live

Yunshu is a flat monorepo (no `yunshu/` subdir):

| Layer | Directory | Responsibility |
|:-----:|-----------|----------------|
| L1 | `python/yunshu_gateway/` | FastAPI HTTP server, OpenAI/Anthropic/MCP/Realtime routers |
| L2 | `python/yunshu_control/` | Lightweight admin / usage accounting |
| L4 | `python/yunshu_engine/` | Inference engine (text + vision + audio + image) |
| L5 | `python/yunshu_kv/` | KV prefix cache |
| CLI | `python/yunshu_cli/` | `yunshu serve / chat / model / ...` |

> **Note:** `python/yunshu_engine/` is being actively refactored by the owner. If your change is
> engine-side, coordinate before opening a large PR. The dead `yunshu_mesh/` and `yunshu_api/` trees
> are not shipped and should not be edited.

Metal GPU kernels live in `metal/` (built via `just build-metal`). The Next.js dashboard is in `webui/`.

## Commit & PR conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(cli): add `yunshu model pull` subcommand
fix(gateway): correct streaming SSE keepalive interval
docs(readme): clarify non-goals
```

Open a PR against `main`. Include:

- **What** changed and **why**
- `just test-unit` output (counts)
- A benchmark comparison only if your change touches the hot path

## Code style

- **Python:** ruff-formatted, 88-char line, Python 3.13+. `just lint` is authoritative.
- **Metal:** C++14, `[[function_constant]]` for compile-time tile specialization.
- **TypeScript/React:** Next.js 16 App Router, strict mode.
- Comments explain the **why** (hidden constraint, subtle invariant, workaround), not the what.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE).
