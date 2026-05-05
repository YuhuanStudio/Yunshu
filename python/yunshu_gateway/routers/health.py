"""Yunshu Health Check and Version Info router.

Provides Kubernetes-style health probes and version metadata:
- /health       — aggregate health status
- /health/ready — readiness probe (engine loaded, memory OK)
- /health/live  — liveness probe (always ok when process is alive)
- /version      — version, python, MLX availability
"""
from __future__ import annotations

import logging
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


# ── Response schemas ──


class HealthResponse(BaseModel):
    status: str
    timestamp: str


class ReadyCheckDetail(BaseModel):
    model_loaded: bool = False
    gpu_memory_ok: bool = True
    not_shutting_down: bool = True


class ReadyResponse(BaseModel):
    ready: bool
    checks: ReadyCheckDetail


class LiveResponse(BaseModel):
    alive: bool
    shutting_down: bool = False


class VersionResponse(BaseModel):
    version: str
    python: str
    mlx_available: bool
    platform: str


# ── Helpers ──


def _get_engine():
    """Lazily import engine accessor to avoid circular imports."""
    try:
        from ..engine import get_engine, get_model_manager

        return get_engine(), get_model_manager()
    except Exception:
        return None, None


def _is_model_loaded(engine: Any, manager: Any) -> bool:
    """Check if at least one model is loaded."""
    if manager is not None:
        for entry in manager.list_entries():
            if entry.is_loaded:
                return True
    if engine is not None and getattr(engine, "is_loaded", False):
        return True
    return False


def _check_gpu_memory() -> bool:
    """Check that GPU memory usage is below 95%."""
    try:
        import mlx.core as mx

        active = mx.get_active_memory()
        try:
            import subprocess

            r = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
            )
            total_uma = int(r.stdout.strip())
        except Exception:
            total_uma = 0

        if total_uma > 0:
            return (active / total_uma) < 0.95
        return True
    except Exception:
        return True


def _get_version() -> str:
    """Read version from package metadata."""
    try:
        from importlib.metadata import version

        return version("yunshu")
    except Exception:
        return "0.1.0.dev0"


def _is_mlx_available() -> bool:
    """Check if MLX is importable."""
    try:
        import mlx.core  # noqa: F401

        return True
    except Exception:
        return False


# ── Endpoints ──


@router.get("/health", response_model=HealthResponse)
async def health() -> dict:
    """Aggregate health status."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health/ready", response_model=ReadyResponse)
async def readiness() -> dict:
    """Readiness probe — is the server ready to accept traffic?

    Checks:
    - model_loaded: at least one model is loaded in the engine
    - gpu_memory_ok: GPU memory usage < 95%
    - not_shutting_down: server is not in graceful shutdown
    """
    engine, manager = _get_engine()

    checks = ReadyCheckDetail()
    checks.model_loaded = _is_model_loaded(engine, manager)
    checks.gpu_memory_ok = _check_gpu_memory()

    # Check shutdown state via main module
    try:
        import yunshu_gateway.main as _main

        checks.not_shutting_down = not getattr(_main, "_shutting_down", False)
    except Exception:
        checks.not_shutting_down = True

    ready = checks.model_loaded and checks.gpu_memory_ok and checks.not_shutting_down

    return {
        "ready": ready,
        "checks": checks.model_dump(),
    }


@router.get("/health/live", response_model=LiveResponse)
async def liveness() -> dict:
    """Liveness probe — always returns ok if the process is alive."""
    shutting_down = False
    try:
        import yunshu_gateway.main as _main

        shutting_down = getattr(_main, "_shutting_down", False)
    except Exception:
        pass

    return {
        "alive": True,
        "shutting_down": shutting_down,
    }


@router.get("/version", response_model=VersionResponse)
async def version_info() -> dict:
    """Version and runtime information."""
    return {
        "version": _get_version(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "mlx_available": _is_mlx_available(),
        "platform": platform.platform(),
    }
