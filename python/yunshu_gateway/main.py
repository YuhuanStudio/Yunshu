"""Yunshu L1 API Gateway — FastAPI app factory."""

import os
import asyncio
import logging
import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .engine import get_engine, get_model_manager, init_engine, init_model_manager

# Default model from environment or None (requires explicit load via API)
DEFAULT_MODEL = os.environ.get("YUNSHU_MODEL")
MODELS_DIR = os.environ.get(
    "YUNSHU_MODELS_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "models"),
)

# ProcessMemoryEnforcer instance (multi-model mode only)
_memory_enforcer = None

# Shutdown state
_shutting_down = False
_active_requests = 0
_drain_event: asyncio.Event | None = None


def _get_memory_limit_bytes() -> int:
    """Compute memory limit from environment or UMA size.

    Uses YUNSHU_MAX_MEMORY_GB env var, or defaults to 80% of UMA.
    """
    env_val = os.environ.get("YUNSHU_MAX_MEMORY_GB")
    if env_val:
        return int(float(env_val) * 1024**3)

    # Default: 80% of UMA (reserve for system + KV cache)
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
        )
        uma = int(result.stdout.strip())
        return int(uma * 0.8)
    except Exception:
        return 0  # unlimited


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown lifecycle."""
    global _memory_enforcer, _drain_event, _shutting_down

    _drain_event = asyncio.Event()
    _shutting_down = False

    if DEFAULT_MODEL:
        # Single-model mode: use BatchedEngine directly
        from yunshu_engine.batched_engine import BatchedEngine
        from .engine import set_engine

        engine = BatchedEngine(model_name=DEFAULT_MODEL)
        set_engine(engine)
        await engine.start()
    elif os.environ.get("YUNSHU_MULTI_MODEL"):
        # Multi-model mode: auto-discover and register models from models_dir
        max_bytes = _get_memory_limit_bytes()
        manager = init_model_manager(
            max_memory_bytes=max_bytes if max_bytes > 0 else None,
            models_dir=MODELS_DIR,
        )

        # Auto-discover models using model_discovery module (oMLX pattern)
        from pathlib import Path as _Path
        models_path = _Path(MODELS_DIR)
        if models_path.exists():
            try:
                from yunshu_engine.model_discovery import discover_models
                discovered = discover_models(models_path)
                for mid, info in discovered.items():
                    try:
                        manager.register_model(
                            model_id=mid,
                            model_path=info.model_path,
                            estimated_bytes=info.estimated_size,
                        )
                    except Exception:
                        pass
                if discovered:
                    import logging as _logging
                    _logging.getLogger(__name__).info(
                        "Auto-discovered %d models from %s",
                        len(discovered), models_path,
                    )
            except Exception as e:
                import logging as _logging
                _logging.getLogger(__name__).warning("Model discovery failed: %s", e)

        # Start ProcessMemoryEnforcer (oMLX pattern)
        if max_bytes > 0:
            from yunshu_engine.process_memory_enforcer import ProcessMemoryEnforcer

            ttl_seconds = os.environ.get("YUNSHU_MODEL_TTL_SECONDS")
            _memory_enforcer = ProcessMemoryEnforcer(
                model_manager=manager,
                max_bytes=max_bytes,
                poll_interval=float(os.environ.get("YUNSHU_MEMORY_POLL_INTERVAL", "2.0")),
                ttl_seconds=float(ttl_seconds) if ttl_seconds else None,
            )
            _memory_enforcer.start()

    yield

    # Graceful shutdown with request draining
    _shutting_down = True

    # Wait for active requests to drain (up to 30s)
    if _active_requests > 0 and _drain_event is not None:
        try:
            await asyncio.wait_for(_drain_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            pass

    if _memory_enforcer is not None:
        await _memory_enforcer.stop()

    engine = get_engine()
    if engine and engine.is_loaded:
        await engine.stop()

    manager = get_model_manager()
    if manager:
        for entry in manager.list_entries():
            if entry.is_loaded and entry.engine:
                await entry.engine.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Yunshu",
        version="0.1.0-dev",
        description="Production-grade MLX inference platform for Apple Silicon",
        lifespan=lifespan,
    )

    # CORS: configurable via YUNSHU_CORS_ORIGINS (comma-separated).
    # Defaults to ["*"] in dev, should be restricted in production.
    cors_origins_str = os.environ.get("YUNSHU_CORS_ORIGINS", "*")
    cors_origins = (
        cors_origins_str.split(",") if cors_origins_str != "*" else ["*"]
    )
    if cors_origins == ["*"]:
        logger.warning("CORS: allow_origins=['*'] — set YUNSHU_CORS_ORIGINS for production")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
    )

    # Gateway middleware (order: outermost first)
    from .middleware.metrics import MetricsMiddleware
    from .middleware.rate_limit import RateLimitMiddleware
    from .middleware.request_logging import RequestLoggingMiddleware
    from .middleware.tenant_auth import TenantAuthMiddleware

    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(TenantAuthMiddleware)

    # Startup warnings
    if os.environ.get("YUNSHU_AUTH_DISABLED", "").lower() in ("true", "1", "yes"):
        logger.warning(
            "AUTH IS DISABLED — all endpoints are publicly accessible. "
            "Set YUNSHU_AUTH_TOKEN or remove YUNSHU_AUTH_DISABLED for production."
        )
    if not os.environ.get("YUNSHU_AUTH_TOKEN") and not os.environ.get("YUNSHU_AUTH_DISABLED"):
        logger.warning(
            "No YUNSHU_AUTH_TOKEN set — requests require no authentication by default. "
            "Set YUNSHU_AUTH_TOKEN=<secret> to enable Bearer token auth."
        )

    # Import routers lazily to reduce startup memory
    from .routers import anthropic, audio, batch_inference, bench, chat, completions, embeddings, images, mcp, models, monitoring as gw_monitoring, realtime, tokenize

    # Routes — L1 Gateway
    app.include_router(chat.router, prefix="/v1")
    app.include_router(completions.router, prefix="/v1")
    app.include_router(embeddings.router, prefix="/v1")
    app.include_router(models.router, prefix="/v1")
    app.include_router(anthropic.router, prefix="/v1")
    app.include_router(audio.router, prefix="/v1")
    app.include_router(images.router, prefix="/v1")
    app.include_router(tokenize.router, prefix="/v1")
    app.include_router(batch_inference.router, prefix="/v1")
    app.include_router(mcp.router, prefix="/v1")
    app.include_router(realtime.router)
    app.include_router(bench.router)

    # Routes — L1 Gateway Monitoring (system, models, requests, prometheus)
    app.include_router(gw_monitoring.router, prefix="/api/v1")

    # Routes — L2 Control Plane (lazy: only loaded when yunshu_api is installed)
    # Avoids importing heavy admin/dashboard/mesh modules in single-model mode
    try:
        from yunshu_api.routers import admin, dashboard, mesh, monitoring
        app.include_router(admin.router, prefix="/api/v1")
        app.include_router(monitoring.router, prefix="/api/v1")
        app.include_router(mesh.router, prefix="/api/v1")
        app.include_router(dashboard.router, prefix="/api/v1")
    except ImportError:
        pass

    @app.get("/health")
    async def health() -> dict:
        from .engine import get_engine, get_model_manager
        engine = get_engine()
        manager = get_model_manager()
        result = {
            "status": "ok",
            "engine": engine.get_stats() if engine else {"loaded": False},
            "model_manager": manager.memory_usage if manager else None,
        }
        if _memory_enforcer is not None:
            result["memory_enforcer"] = _memory_enforcer.get_status()

        # Add server metrics summary
        try:
            from yunshu_engine.server_metrics import get_server_metrics
            result["metrics"] = get_server_metrics().get_snapshot()
        except Exception:
            pass

        return result

    @app.get("/health/ready")
    async def readiness() -> dict:
        """Readiness probe — is the server ready to accept traffic?"""
        engine = get_engine()
        manager = get_model_manager()

        checks = {}
        ready = True

        # Check if at least one model is loaded
        has_loaded_model = False
        if manager is not None:
            for entry in manager.list_entries():
                if entry.is_loaded:
                    has_loaded_model = True
                    break
        elif engine and engine.is_loaded:
            has_loaded_model = True

        checks["model_loaded"] = has_loaded_model
        if not has_loaded_model:
            ready = False

        # Check GPU memory available
        try:
            import mlx.core as mx
            active = mx.get_active_memory()
            total_uma = 0
            try:
                import subprocess
                r = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
                )
                total_uma = int(r.stdout.strip())
            except Exception:
                pass
            if total_uma > 0:
                mem_pct = active / total_uma
                checks["gpu_memory_ok"] = mem_pct < 0.95
                if mem_pct >= 0.95:
                    ready = False
            else:
                checks["gpu_memory_ok"] = True
        except Exception:
            checks["gpu_memory_ok"] = True

        # Check if shutting down
        checks["not_shutting_down"] = not _shutting_down
        if _shutting_down:
            ready = False

        return {
            "ready": ready,
            "checks": checks,
        }

    @app.get("/health/live")
    async def liveness() -> dict:
        """Liveness probe — is the server alive?"""
        return {"alive": True, "shutting_down": _shutting_down}

    @app.get("/version")
    async def version() -> dict:
        """Server version info."""
        return {
            "version": "0.1.0-dev",
            "service": "yunshu",
            "description": "Production-grade MLX inference platform for Apple Silicon",
        }

    # Anthropic SDK sends requests to /v1/messages without /v1 prefix
    app.include_router(anthropic.router)

    return app


app = create_app()
