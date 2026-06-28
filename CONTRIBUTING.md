# Contributing to Yunshu

Thank you for your interest in contributing to Yunshu! This guide covers everything you need to get started.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- [just](https://github.com/casey/just) command runner (`brew install just`)
- [pnpm](https://pnpm.io/) (for WebUI development)
- Xcode Command Line Tools (for Metal shader compilation)

## Setup

```bash
git clone https://github.com/YuhuanStudio/Yunshu.git
cd Yunshu
just setup
```

This installs Python dependencies (via `uv sync`) and WebUI dependencies (via `pnpm install`).

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feat/your-feature-name
```

Branch naming conventions:
- `feat/` — new features
- `fix/` — bug fixes
- `refactor/` — code refactoring
- `docs/` — documentation changes
- `test/` — test additions or fixes
- `bench/` — benchmark improvements

### 2. Make Changes

The project follows a 5-layer architecture (L0–L5):

| Layer | Directory | Responsibility |
|:-----:|-----------|----------------|
| L1 | `python/yunshu_gateway/` | FastAPI HTTP server, 14 routers, 6 middleware |
| L2 | `python/yunshu_control/`, `python/yunshu_api/` | RBAC, scheduling, admin API |
| L3 | `python/yunshu_mesh/` | Compute mesh, mx.distributed |
| L4 | `python/yunshu_engine/` | 5-modality inference engine |
| L5 | `python/yunshu_kv/` | 4-tier KV cache hierarchy |

Metal GPU kernels live in `metal/`, the Next.js dashboard in `webui/`.

### 3. Test

```bash
just test              # All tests (unit + integration)
just test-unit         # Unit tests only
just test-single tests/unit/test_foo.py  # Single file
```

All 2,162 tests must pass. Run the relevant subset for your changes — the full suite takes a few minutes.

### 4. Lint & Format

```bash
just format            # Auto-format with ruff
just lint              # Check with ruff + mypy
```

Fix any lint errors before submitting. We use `ruff` for formatting and linting, `mypy` for type checking.

### 5. Benchmark (if relevant)

If your change affects inference performance:

```bash
just bench-roofline    # Apple Silicon roofline
PYTHONPATH=. uv run python scripts/bench_unified.py --quick
```

The benchmark must show no accuracy regression and no throughput regression below 95% of baseline.

### 6. Commit

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(engine): add speculative decoding with EAGLE-3
fix(gateway): correct streaming SSE keepalive interval
refactor(kv): simplify boundary snapshot serialization
docs(readme): add benchmark results table
test(engine): add logprobs extraction unit tests
bench(metal): add GEMV tile size sweep
```

### 7. Push & PR

```bash
git push origin feat/your-feature-name
```

Open a pull request against `main`. Include:

- **What** changed and **why**
- Test results (`just test` output)
- Benchmark comparison (if performance-related)

## Code Style

- **Python**: ruff-formatted, 88-char line length, Python 3.13+
- **Metal**: C++14 style, `[[function_constant]]` for compile-time parameters
- **TypeScript/React**: Next.js 16 App Router, strict mode
- **No comments** unless the WHY is non-obvious (hidden constraint, subtle invariant, workaround)
- **No docstrings** unless the function is part of a public API

## Project Conventions

### Architecture Decisions
- Single-language Python stack (L1–L5) — no Rust, no gRPC FFI
- `uv` as sole Python package manager
- `just` for build orchestration
- `uvicorn` + FastAPI for HTTP serving
- mlx-lm `BatchGenerator` for continuous batching

### Engine API
- Per-request sampler + `SequenceStateMachine` for stop/eos/reasoning state
- Per-request detokenizer (never pool — `reset()` leaks byte buffers)
- Single Metal thread via `ThreadPoolExecutor(max_workers=1)` for all GPU work

### Testing
- Tests in `tests/unit/` and `tests/integration/`
- Async tests use `pytest-asyncio` with `asyncio_mode = "auto"`
- No mocking of MLX or mlx-lm internals — test real behavior

## Running the Server

```bash
# Single model
just dev

# Multi-model (auto-discovery)
YUNSHU_MULTI_MODEL=1 just dev-multi

# Specific model
just dev-model Qwen3.5-9B-MLX-4bit
```

Place model weights in `./models/` — any HuggingFace-format MLX quantized model works.

## Metal Kernel Development

Metal shaders are in `metal/` and compiled with `make`:

```bash
just build-metal       # Build all kernels
just metal-compile     # Clean rebuild
just metal-profile paged_attention  # Profile a specific kernel
```

Kernels use Metal 3.1 features and `[[function_constant]]` for compile-time tile size specialization.

## WebUI Development

The dashboard is a Next.js 16 app in `webui/`:

```bash
just dev-webui         # Start dev server with hot reload
just build-webui       # Production build
```

## Reporting Issues

- **Bugs**: Open a GitHub issue with reproduction steps, macOS version, and hardware (M1/M2/M3/M4).
- **Feature requests**: Open an issue describing the use case and expected behavior.
- **Performance**: Include benchmark output from `scripts/bench_unified.py`.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

## Questions?

Open a GitHub issue or start a discussion. We're happy to help you get oriented.
