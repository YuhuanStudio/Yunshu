# Yunshu command runner
# Usage: just <recipe>

set dotenv-load

default:
    @just --list

# ── Setup ──

# Python deps only — what you need to run/develop the server. The WebUI dashboard
# (Node/pnpm) is optional; run `just setup-webui` separately if you touch it.
setup:
    uv sync --all-extras --dev

# Optional: WebUI dashboard deps (needs Node + corepack/pnpm).
setup-webui:
    cd webui && corepack enable && pnpm install

# ── Build ──
# No build-metal: there are no hand-written Metal kernels (they benchmarked slower
# than mx.fast/mx.matmul on Apple Silicon and were removed). All compute is via MLX.

build: build-python build-webui

build-python:
    uv sync

build-webui:
    cd webui && pnpm build

# ── Test ──

test: test-unit

test-unit:
    uv run pytest tests/ -v

test-integration:
    uv run pytest tests/integration -v

test-single FILE:
    uv run pytest {{ FILE }} -v

# ── Lint ──

lint:
    uv run ruff check python/ tests/
    uv run mypy python/

format:
    uv run ruff format python/ tests/

# ── Run ──
# Set YUNSHU_MODEL to a local MLX model path (via .env or the env). For the omni
# use case, prefer multi-model mode so brain + VLM + ASR + TTS can co-reside.

dev:
    uv run uvicorn python.yunshu_gateway.main:app --host 0.0.0.0 --port 8000

# Multi-model mode (discovers all models in ./models/)
dev-multi:
    YUNSHU_MULTI_MODEL=1 YUNSHU_MODELS_DIR=./models uv run uvicorn python.yunshu_gateway.main:app --host 0.0.0.0 --port 8000

# Load a specific model
dev-model MODEL:
    YUNSHU_MODEL=./models/{{ MODEL }} uv run uvicorn python.yunshu_gateway.main:app --host 0.0.0.0 --port 8000

# CLI
cli *ARGS:
    uv run python -m yunshu_cli {{ ARGS }}

# WebUI dev server
dev-webui:
    cd webui && pnpm dev
