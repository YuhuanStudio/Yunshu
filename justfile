# Yunshu command runner
# Usage: just <recipe>

set dotenv-load

default:
    @just --list

# ── Setup ──

setup:
    uv sync --all-extras --dev
    cd webui && corepack enable && pnpm install

# ── Build ──

build: build-metal build-python build-webui

build-metal:
    cd metal && make all

build-python:
    uv sync

build-webui:
    cd webui && pnpm build

# ── Test ──

test: test-unit test-integration

test-unit:
    uv run pytest tests/unit -v

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

dev:
    YUNSHU_MODEL=./models/Qwen3.5-9B-MLX-4bit uv run uvicorn python.yunshu_gateway.main:app --host 0.0.0.0 --port 8000

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

# WebUI production build
build-webui:
    cd webui && pnpm build

# ── Benchmark ──

bench-roofline:
    uv run python bench/roofline/run.py

bench-kivi-metal:
    uv run python bench/kivi_metal/run.py

bench-all:
    uv run python bench/run_all.py

# ── Metal ──

metal-compile:
    cd metal && make clean && make all

metal-profile KERNEL:
    xcrun metal -O3 -gline-tables-only -c metal/{{ KERNEL }}.metal -o build/{{ KERNEL }}.air
