# Yunshu command runner
# Usage: just <recipe>

set dotenv-load

default:
    @just --list

# ── Setup ──

setup:
    uv sync --all-extras --dev

# ── Build ──
# No build step for kernels: the Metal kernels in python/yunshu_engine/kernels/ are
# compiled at runtime by mx.fast.metal_kernel.

build: build-python

build-python:
    uv sync


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
    uv run python scripts/dev/mypy_gate.py

format:
    uv run ruff format python/ tests/

# ── Run ──
# Set YUNSHU_MODEL to a local MLX model path (via .env or the env). For the omni
# use case, prefer multi-model mode so brain + VLM + ASR + TTS can co-reside.

dev:
    uv run uvicorn python.yunshu_gateway.main:app --host 127.0.0.1 --port 8000

# Multi-model mode (discovers all models in ./models/)
dev-multi:
    YUNSHU_MULTI_MODEL=1 YUNSHU_MODELS_DIR=./models uv run uvicorn python.yunshu_gateway.main:app --host 127.0.0.1 --port 8000

# Load a specific model
dev-model MODEL:
    YUNSHU_MODEL=./models/{{ MODEL }} uv run uvicorn python.yunshu_gateway.main:app --host 127.0.0.1 --port 8000

# CLI
cli *ARGS:
    uv run python -m yunshu_cli {{ ARGS }}

# Release artifacts: sdist + wheel in dist/, metadata checked (see RELEASING.md)
dist:
    rm -rf dist && uv build && uvx twine check dist/*

# What changed upstream for vendored kernels, watched repos and pinned packages
vendor-check *args:
    uv run python scripts/vendor/check_upstream.py {{args}}

# Private read-only live view of docs/research on http://0.0.0.0:3990 (content never enters git)
research-site:
    cd tools/research-site && pnpm install --prefer-offline && pnpm build && node server/index.ts
