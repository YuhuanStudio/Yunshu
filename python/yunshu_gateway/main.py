"""Yunshu L1 API Gateway — FastAPI app factory."""

import os
import asyncio
import logging
import time

from starlette.requests import Request
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .engine import get_engine, get_model_manager, init_model_manager

# Default model from environment or None (requires explicit load via API)
DEFAULT_MODEL = os.environ.get("YUNSHU_MODEL")
MODELS_DIR = os.environ.get(
    "YUNSHU_MODELS_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "models"),
)

# ProcessMemoryEnforcer instance (multi-model mode only)
_memory_enforcer = None

# Background tasks tracked for clean shutdown
_background_tasks: list[asyncio.Task] = []

# Startup timestamp for uptime tracking
_startup_time: float = 0.0

# Shutdown state machine (vLLM pattern: RUNNING → REQUESTED → SHUTTING_DOWN)
class ServerState:
    RUNNING = "running"
    REQUESTED = "shutdown_requested"
    SHUTTING_DOWN = "shutting_down"

_server_state = ServerState.RUNNING
_active_requests = 0
_drain_event: asyncio.Event | None = None


def _validate_env_vars() -> list[str]:
    """Validate production env vars and return warnings."""
    warnings = []

    # Numeric env vars that must parse correctly
    numeric_vars = {
        "YUNSHU_MAX_MEMORY_GB": (float, False),
        "YUNSHU_DRAIN_TIMEOUT": (float, False),
        "YUNSHU_MEMORY_POLL_INTERVAL": (float, False),
        "YUNSHU_MODEL_TTL_SECONDS": (float, False),
        "YUNSHU_MAX_CONCURRENT": (int, False),
        "YUNSHU_RATE_LIMIT_RPM": (float, False),
    }
    for var, (type_fn, required) in numeric_vars.items():
        val = os.environ.get(var)
        if val is not None:
            try:
                type_fn(val)
            except (ValueError, TypeError):
                warnings.append(f"Invalid value for {var}: '{val}' (expected {type_fn.__name__})")
        elif required:
            warnings.append(f"Required env var {var} is not set")

    # Boolean-ish env vars
    bool_vars = [
        "YUNSHU_MULTI_MODEL", "YUNSHU_DATA_PARALLEL", "YUNSHU_DISTRIBUTED",
        "YUNSHU_AUTH_DISABLED", "YUNSHU_RESPONSE_CACHE", "YUNSHU_PROCESS_ISOLATION",
    ]
    for var in bool_vars:
        val = os.environ.get(var)
        if val is not None and val.lower() not in ("0", "1", "true", "false", "yes", "no"):
            warnings.append(f"Invalid boolean value for {var}: '{val}'")

    # Conflicting config: both YUNSHU_MODEL and YUNSHU_MULTI_MODEL set
    if DEFAULT_MODEL and os.environ.get("YUNSHU_MULTI_MODEL"):
        warnings.append(
            "Both YUNSHU_MODEL and YUNSHU_MULTI_MODEL are set — "
            "YUNSHU_MODEL takes precedence (single-model mode)"
        )

    return warnings


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
        logger.debug("failed to read sysctl hw.memsize", exc_info=True)
        return 0  # unlimited


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown lifecycle with production hardening.

    Production features:
    - Startup validation of env vars with warning log
    - Startup timeout for model loading (configurable via YUNSHU_STARTUP_TIMEOUT)
    - Graceful shutdown: reject new -> drain -> cleanup -> model manager shutdown
    - Background task tracking for clean cancellation
    """
    global _memory_enforcer, _drain_event, _server_state, _startup_time
    global _background_tasks

    # Always reset state on lifespan entry — handles test isolation
    # where module-level globals persist between TestClient instances
    _drain_event = asyncio.Event()
    _server_state = ServerState.RUNNING
    _background_tasks = []

    # Store state on app for middleware to read (avoids stale module-level
    # state when TestClient doesn't trigger lifespan in Starlette 1.0+)
    app.state.server_state = ServerState.RUNNING

    # ── Startup validation ──
    env_warnings = _validate_env_vars()
    for w in env_warnings:
        logger.warning("CONFIG: %s", w)

    # ── Startup timeout ──
    startup_timeout = float(os.environ.get("YUNSHU_STARTUP_TIMEOUT", "300"))

    _startup_time = time.time()

    if DEFAULT_MODEL:
        # Single-model mode: use BatchedEngine directly
        from yunshu_engine.batched_engine import BatchedEngine
        from .engine import set_engine

        engine = BatchedEngine(model_name=DEFAULT_MODEL)
        set_engine(engine)
        try:
            await asyncio.wait_for(engine.start(), timeout=startup_timeout)
            logger.info(
                "Startup complete: model '%s' loaded (%.1fs)",
                DEFAULT_MODEL, time.time() - _startup_time,
            )
        except asyncio.TimeoutError:
            logger.error(
                "FATAL: model '%s' load timed out after %.0fs — server not ready",
                DEFAULT_MODEL, startup_timeout,
            )
            await engine.stop()
            # Server starts but /health/ready will report not-ready
        except Exception as e:
            logger.error(
                "FATAL: model '%s' load failed: %s — server not ready",
                DEFAULT_MODEL, e,
            )
            await engine.stop()

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
                        logger.debug(f"failed to register model {mid}", exc_info=True)
                if discovered:
                    logger.info(
                        "Auto-discovered %d models from %s",
                        len(discovered), models_path,
                    )
            except Exception as e:
                logger.warning("Model discovery failed: %s", e, exc_info=True)

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
            logger.info(
                "ProcessMemoryEnforcer started (limit=%.1fGB, poll=%.1fs)",
                max_bytes / 1024**3,
                float(os.environ.get("YUNSHU_MEMORY_POLL_INTERVAL", "2.0")),
            )

        logger.info(
            "Startup complete: multi-model mode, %d models registered (%.1fs)",
            len(manager.list_entries()), time.time() - _startup_time,
        )

    # Initialize MCP client manager (LLM → external MCP tool servers)
    mcp_config_path = os.environ.get("YUNSHU_MCP_CONFIG")
    mcp_servers_env = os.environ.get("YUNSHU_MCP_SERVERS", "")
    if mcp_config_path or mcp_servers_env:
        try:
            from yunshu_engine.mcp_client import init_mcp_client
            mcp_mgr = await init_mcp_client(config_path=mcp_config_path)
            if mcp_mgr and hasattr(app, 'state'):
                app.state.mcp_client = mcp_mgr
            stats = mcp_mgr.get_stats()
            logger.info(
                "MCP client manager initialized: %d servers, %d tools",
                stats["connected_servers"], stats["total_tools"],
            )
        except Exception:
            logger.warning("MCP client initialization failed", exc_info=True)

    yield

    # ═══ Graceful shutdown ═══
    logger.info(
        "Shutdown initiated: %d active requests, draining...",
        _active_requests,
    )
    _server_state = ServerState.REQUESTED
    app.state.server_state = ServerState.REQUESTED

    # Wait for active requests to drain
    drain_timeout = float(os.environ.get("YUNSHU_DRAIN_TIMEOUT", "30"))
    if _active_requests > 0 and _drain_event is not None:
        try:
            await asyncio.wait_for(_drain_event.wait(), timeout=drain_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Shutdown drain timed out after %.0fs: %d requests still active",
                drain_timeout, _active_requests,
            )

    # Disconnect MCP client manager
    try:
        from yunshu_engine.mcp_client import get_mcp_client_manager
        mcp = get_mcp_client_manager()
        if mcp is not None:
            await mcp.disconnect_all()
    except Exception:
        logger.debug("MCP client shutdown failed", exc_info=True)

    # Stop ProcessMemoryEnforcer
    if _memory_enforcer is not None:
        try:
            await _memory_enforcer.stop()
        except Exception:
            logger.debug("Memory enforcer stop failed", exc_info=True)
        _memory_enforcer = None

    # Cancel any remaining background tasks
    for task in _background_tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    _background_tasks.clear()

    _server_state = ServerState.SHUTTING_DOWN
    app.state.server_state = ServerState.SHUTTING_DOWN

    # Stop single-engine mode
    engine = get_engine()
    if engine and getattr(engine, 'is_loaded', False):
        try:
            await engine.stop()
        except Exception:
            logger.warning("Engine stop failed", exc_info=True)

    # Stop model manager — use proper shutdown() for ordered unload
    manager = get_model_manager()
    if manager:
        try:
            await manager.shutdown()
        except Exception:
            logger.warning("Model manager shutdown failed", exc_info=True)

    logger.info(
        "Shutdown complete (%.1fs uptime)",
        time.time() - _startup_time,
    )
    _startup_time = 0.0


def create_app() -> FastAPI:
    app = FastAPI(
        title="Yunshu",
        version="0.1.0-dev",
        description="Production-grade MLX inference platform for Apple Silicon",
        lifespan=lifespan,
    )

    # ── Custom exception handlers for consistent error formats ──

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """Return OpenAI-format errors for validation failures.

        Anthropic endpoints (/v1/messages, /messages) get Anthropic format instead.
        """
        errors = exc.errors()
        messages = []
        for err in errors:
            loc = ".".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "Invalid request")
            messages.append(f"{loc}: {msg}" if loc else msg)
        detail = "; ".join(messages)

        # Anthropic endpoints: return Anthropic error format
        path = request.url.path
        if path.endswith("/messages") or path.endswith("/messages/count_tokens"):
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": detail,
                    },
                },
            )

        # Default: OpenAI error format
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": detail,
                    "type": "invalid_request_error",
                    "code": "validation_error",
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        """Ensure all HTTP errors follow the correct format for the endpoint."""
        path = request.url.path

        # Anthropic endpoints: return Anthropic error format
        if path.endswith("/messages") or path.endswith("/messages/count_tokens"):
            error_type = "invalid_request_error"
            if exc.status_code == 404:
                error_type = "not_found_error"
            elif exc.status_code == 503:
                error_type = "overloaded_error"
            elif exc.status_code == 500:
                error_type = "api_error"
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "type": "error",
                    "error": {
                        "type": error_type,
                        "message": str(exc.detail),
                    },
                },
            )

        # Default: OpenAI error format
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": str(exc.detail), "type": "server_error"}},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Catch-all for unhandled exceptions — prevents stack trace leakage.

        Logs full traceback server-side, returns generic 500 to client.
        Handles MemoryError, RuntimeError, and any other unexpected exception.
        """
        import traceback
        logger.error(
            "Unhandled exception on %s %s: %s\n%s",
            request.method, request.url.path, exc,
            traceback.format_exc(),
        )
        path = request.url.path
        # Anthropic endpoints: return Anthropic error format
        if path.endswith("/messages") or path.endswith("/messages/count_tokens"):
            return JSONResponse(
                status_code=500,
                content={
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "Internal server error",
                    },
                },
            )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "type": "server_error",
                    "code": "internal_error",
                }
            },
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
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "Accept", "X-Request-ID"],
    )

    # Gateway middleware (order: outermost first)
    from .middleware.metrics import MetricsMiddleware
    from .middleware.rate_limit import RateLimitMiddleware
    from .middleware.request_logging import RequestLoggingMiddleware
    from .middleware.tenant_auth import TenantAuthMiddleware

    app.add_middleware(MetricsMiddleware)

    # Response cache middleware: caches non-streaming responses when enabled
    if os.environ.get("YUNSHU_RESPONSE_CACHE", "").lower() in ("1", "true", "yes"):
        from .middleware.gateway_optimizer import ResponseCacheMiddleware
        app.add_middleware(ResponseCacheMiddleware)
        logger.info("ResponseCache middleware registered (YUNSHU_RESPONSE_CACHE=1)")

    # Data-parallel middleware: initialize when YUNSHU_DATA_PARALLEL=1
    if os.environ.get("YUNSHU_DATA_PARALLEL", "").lower() in ("1", "true", "yes"):
        from .dp_middleware import setup_data_parallel, DPRouterMiddleware

        dp_lb = setup_data_parallel()
        app.add_middleware(DPRouterMiddleware)
        logger.info("DataParallel middleware registered (YUNSHU_DATA_PARALLEL=1)")

    # Shutdown rejection middleware: reject new inference requests during shutdown.
    # Reads state from app.state (set by lifespan) instead of module globals,
    # because Starlette 1.0+ TestClient may not trigger lifespan for non-context usage.
    @app.middleware("http")
    async def shutdown_guard(request: Request, call_next):
        state = getattr(request.app.state, 'server_state', ServerState.RUNNING)
        if state != ServerState.RUNNING:
            path = request.url.path
            # Allow health/monitoring/admin even during shutdown
            if not path.startswith(("/health", "/metrics", "/api/v1/admin", "/api/v1/monitoring")):
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "message": "Server is shutting down",
                            "type": "server_error",
                            "code": "shutdown_in_progress",
                        }
                    },
                    headers={"Retry-After": "5"},
                )
        return await call_next(request)

    # Sleep middleware: reject inference requests while sleeping
    @app.middleware("http")
    async def sleep_guard(request: Request, call_next):
        from .routers.sleep import is_sleeping
        path = request.url.path
        if is_sleeping() and not path.startswith(("/sleep", "/wake-up", "/health", "/api/v1/admin", "/api/v1/monitoring")):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=503, content={"detail": "Server is sleeping. POST /wake-up to resume."})
        return await call_next(request)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(TenantAuthMiddleware)

    # ── Wave 43: Gateway optimizer wiring ──
    from yunshu_engine.gateway_optimizer import (
        get_request_coalescer, get_streaming_buffer,
        get_response_cache, get_connection_pool,
    )
    app.state.request_coalescer = get_request_coalescer()
    app.state.streaming_buffer = get_streaming_buffer()
    # Response cache: opt-in via YUNSHU_RESPONSE_CACHE=1
    if os.environ.get("YUNSHU_RESPONSE_CACHE", "").lower() in ("1", "true", "yes"):
        app.state.response_cache = get_response_cache()
        logger.info("ResponseCache enabled (YUNSHU_RESPONSE_CACHE=1)")
    # Connection pool for distributed mode
    if os.environ.get("YUNSHU_DISTRIBUTED", "").lower() in ("1", "true", "yes"):
        app.state.connection_pool = get_connection_pool()
        logger.info("GatewayConnectionPool enabled (YUNSHU_DISTRIBUTED=1)")

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
    from .routers import anthropic, audio, batch_inference, bench, chat, completions, embeddings, images, mcp, models, monitoring as gw_monitoring, profiling, realtime, scoring, tokenize

    # Wave 43: MCP client manager (LLM → external MCP tool servers)
    # Actual initialization happens in lifespan() since init_mcp_client is async.
    # Here we just check if the singleton was already initialized (e.g. in tests).
    mcp_servers_env = os.environ.get("YUNSHU_MCP_SERVERS", "")
    if mcp_servers_env:
        try:
            from yunshu_engine.mcp_client import get_mcp_client_manager
            mcp_mgr = get_mcp_client_manager()
            if mcp_mgr:
                app.state.mcp_client = mcp_mgr
                logger.info(f"MCP client manager already initialized")
        except Exception:
            logger.debug("MCP client setup deferred to lifespan", exc_info=True)

    # Routes — L1 Gateway
    from .routers import sleep as sleep_mod
    from .routers import responses as responses_mod
    from .routers import cancel as cancel_mod
    from .routers import ocr as ocr_mod
    from .routers import disaggregate
    app.include_router(chat.router, prefix="/v1")
    app.include_router(completions.router, prefix="/v1")
    app.include_router(responses_mod.router, prefix="/v1")
    app.include_router(embeddings.router, prefix="/v1")
    app.include_router(models.router, prefix="/v1")
    app.include_router(anthropic.router, prefix="/v1")
    app.include_router(audio.router, prefix="/v1")
    app.include_router(images.router, prefix="/v1")
    app.include_router(tokenize.router, prefix="/v1")
    app.include_router(batch_inference.router, prefix="/v1")
    app.include_router(mcp.router, prefix="/v1")
    app.include_router(scoring.router, prefix="/v1")
    app.include_router(cancel_mod.router, prefix="/v1")
    app.include_router(ocr_mod.router)
    app.include_router(profiling.router, prefix="/v1")
    app.include_router(realtime.router)
    app.include_router(bench.router)
    app.include_router(disaggregate.router)

    from .routers import video as video_mod
    app.include_router(video_mod.router, prefix="/v1")
    app.include_router(sleep_mod.router)

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
        engine_info = {"loaded": False}
        if engine:
            try:
                engine_info = engine.get_stats()
            except Exception:
                engine_info = {"loaded": getattr(engine, 'is_loaded', False)}
        result = {
            "status": "ok",
            "engine": engine_info,
            "model_manager": manager.memory_usage if manager else None,
            "server_state": _server_state,
            "active_requests": _active_requests,
            "uptime_seconds": round(time.time() - _startup_time, 1) if _startup_time > 0 else 0,
        }
        if _memory_enforcer is not None:
            result["memory_enforcer"] = _memory_enforcer.get_status()

        # Add server metrics summary
        try:
            from yunshu_engine.server_metrics import get_server_metrics
            result["metrics"] = get_server_metrics().get_snapshot()
        except Exception:
            logger.debug("server metrics unavailable", exc_info=True)

        # Add MCP client status
        try:
            from yunshu_engine.mcp_client import get_mcp_client_manager
            mcp = get_mcp_client_manager()
            if mcp is not None:
                result["mcp_client"] = mcp.get_stats()
        except Exception:
            logger.debug("mcp client stats unavailable", exc_info=True)

        # Add model registry status
        try:
            from yunshu_engine.model_registry import get_registry
            result["model_registry"] = get_registry().get_stats()
        except Exception:
            logger.debug("model registry stats unavailable", exc_info=True)

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
        elif engine and getattr(engine, 'is_loaded', False):
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
                import asyncio as _asyncio
                import subprocess
                r = await _asyncio.to_thread(
                    subprocess.run,
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True,
                )
                total_uma = int(r.stdout.strip())
            except Exception:
                logger.debug("failed", exc_info=True)
            if total_uma > 0:
                mem_pct = active / total_uma
                checks["gpu_memory_ok"] = mem_pct < 0.95
                if mem_pct >= 0.95:
                    ready = False
            else:
                checks["gpu_memory_ok"] = True
        except Exception:
            logger.debug("gpu memory check failed in readiness probe", exc_info=True)
            checks["gpu_memory_ok"] = True

        # Check if shutting down
        checks["not_shutting_down"] = _server_state == ServerState.RUNNING
        if _server_state != ServerState.RUNNING:
            ready = False

        return {
            "ready": ready,
            "checks": checks,
        }

    @app.get("/health/live")
    async def liveness() -> dict:
        """Liveness probe — is the server alive?"""
        return {"alive": True, "state": _server_state}

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
