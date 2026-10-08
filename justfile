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

# ── Docs site (site/: Next.js + fumadocs, static export) ──

# Regenerate the pages built from code: the OpenAPI/route dump and the configuration guide
docs-gen:
    uv run python site/scripts/dump_openapi.py
    uv run python site/scripts/gen_content.py

docs-install:
    cd site && pnpm install --frozen-lockfile

# Dev server on :3991 (LAN-visible)
docs-dev: docs-gen
    cd site && pnpm install --frozen-lockfile && pnpm exec next dev -H 0.0.0.0 -p 3991

# Static export to site/out, then typecheck, route test and link check
docs-build: docs-gen
    cd site && pnpm install --frozen-lockfile && pnpm run typecheck && pnpm exec next build && pnpm test && pnpm run check:links
