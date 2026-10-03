# Contributing to Yunshu

Thank you for your interest in contributing to Yunshu! This guide covers the minimum you need to get a
working dev loop. Read [AGENTS.md](AGENTS.md) for the architectural scope and settings rules. By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Prerequisites

- macOS with Apple Silicon (M-series) — Yunshu requires macOS and a native arm64 Python.
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- [just](https://github.com/casey/just) command runner (`brew install just`)

## Setup

```bash
git clone https://github.com/YuhuanStudio/Yunshu.git
cd Yunshu
just setup
```

This runs `uv sync --all-extras --dev`. For a lighter
checkout, pick just the extras you need instead of `--all-extras`:

```bash
uv sync --extra omni          # mlx-vlm + torch — native Qwen3-Omni speech-to-speech (a supported capability)
uv sync --extra vision        # mlx-vlm (VLM / OCR)
uv sync --extra audio         # mlx-audio (ASR / TTS / Realtime voice cascade)
uv sync --extra generation    # diffusers + torch (image generation)
uv sync --extra embeddings    # mlx-embeddings (/v1/embeddings, /v1/rerank)
uv sync --all-extras          # everything — what most contributors want
```

## Development loop

```bash
just lint              # ruff check + mypy baseline gate
just format            # ruff format
just test-unit         # unit tests only (fast — no models needed)
just test              # unit suite (integration tests are opt-in, ignored by default)
just test-single tests/unit/test_foo.py   # one file
```

Run `just dev` to start the gateway on `:8000` against a local model. CI runs lint and package checks on pushes/PRs; the macOS unit suite runs on
release/manual triggers. Run the full unit suite locally before handoff. See
[the hardware validation plan](docs/guides/HARDWARE_VALIDATION.md) for planned coverage.

## Where things live

Yunshu is a flat monorepo (no `yunshu/` subdir):

| Layer | Directory | Responsibility |
|:-----:|-----------|----------------|
| L1 | `python/yunshu_gateway/` | FastAPI HTTP server, OpenAI/Anthropic/MCP/Realtime routers |
| L2 | `python/yunshu_control/` | Audit log and token counting |
| L4 | `python/yunshu_engine/` | Inference engine (text + vision + audio + image) |
| L5 | `python/yunshu_kv/` | KV prefix cache |
| CLI | `python/yunshu_cli/` | `yunshu serve / chat / model / ...` |

> **Note:** `python/yunshu_engine/` is being actively refactored by the owner. If your change is
> engine-side, coordinate before opening a large PR.

Yunshu builds on MLX and includes custom and vendored Metal kernels. They compile at runtime;
there is no separate kernel build step. Adopt a kernel only with a same-checkpoint end-to-end
A/B and output checks. Performance priorities are decode, cold/cached TTFT and prefix reuse.

## Commit & PR conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(cli): add `yunshu pull` subcommand
fix(gateway): correct streaming SSE keepalive interval
docs(readme): clarify non-goals
```

Open a PR against `main`. Include:

- **What** changed and **why**
- `just test-unit` output (counts)
- A benchmark comparison only if your change touches the hot path

## Code style

- **Python:** ruff-formatted, 88-char line, Python 3.13+. `just lint` is authoritative.
- Comments explain the **why** (hidden constraint, subtle invariant, workaround), not the what.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE).


## Review expectations

Keep changes focused and explain the user-visible before/after behavior. Fixes
need a regression test that fails before the fix. Document commands, results and
anything unverified; doc-only changes do not need model benchmarks. Disclose AI
assistance and review the result yourself, including sources and license notices.

A reviewer checks correctness, project scope, compatibility, failure/cancellation
behavior and evidence before recommending a merge. Maintainers resolve substantive
objections and record tradeoffs. Hot-path changes need same-checkpoint output
checks and end-to-end measurements; a faster isolated kernel alone is insufficient.

Public settings belong in the settings registry; regenerate CONFIGURATION.md.
List API/CLI/default changes and migration steps in Unreleased. Stable interfaces
follow the [deprecation policy](RELEASING.md#compatibility-and-deprecation-policy).
Large architecture changes use the [RFC process](docs/ROADMAP.md#lightweight-rfc-process).
Triage uses the [label vocabulary](.github/LABELS.md).
