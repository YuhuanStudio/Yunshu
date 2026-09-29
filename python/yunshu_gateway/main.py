"""Yunshu L1 API Gateway — FastAPI app factory."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from starlette.requests import Request

logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from yunshu_engine import settings
from yunshu_engine.model_discovery import hf_repo_id_for, resolve_model_ref
from yunshu_engine.paths import models_dir
from yunshu_engine.version import yunshu_version

from .engine import get_engine, get_model_manager, init_model_manager

# Default model (YUNSHU_MODEL) or None (requires explicit load via API). A name
# under the models directory or a repo id already in the Hugging Face cache
# resolves to that local folder, so nothing is downloaded again.
DEFAULT_MODEL = resolve_model_ref(settings.get("YUNSHU_MODEL"))
MODELS_DIR = str(models_dir())

# ProcessMemoryEnforcer instance (multi-model mode only)
_memory_enforcer = None

# Background tasks tracked for clean shutdown
_background_tasks: list[asyncio.Task] = []

# Cache UMA size once at module load to avoid sysctl per-request
try:
    import subprocess as _sp

    _total_uma_bytes: int = int(
        _sp.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    )
except Exception:
    _total_uma_bytes: int = 0

# Startup timestamp for uptime tracking
_startup_time: float = 0.0


# Shutdown state machine (RUNNING → REQUESTED → SHUTTING_DOWN)
class ServerState:
    RUNNING = "running"
    REQUESTED = "shutdown_requested"
    SHUTTING_DOWN = "shutting_down"


_server_state = ServerState.RUNNING
_active_requests = 0
_drain_event: asyncio.Event | None = None


def _is_multi_model_enabled() -> bool:
    """True when YUNSHU_MULTI_MODEL is on."""
    return settings.get_bool("YUNSHU_MULTI_MODEL")


def _validate_settings() -> list[str]:
    """Parse every YUNSHU_* setting (raises SettingError on a bad value) and
    return warnings: unknown YUNSHU_* names and conflicting modes."""
    warnings = settings.validate(warn=False)
    if DEFAULT_MODEL and _is_multi_model_enabled():
        warnings.append(
            "Both YUNSHU_MODEL and YUNSHU_MULTI_MODEL are set — "
            "YUNSHU_MODEL takes precedence (single-model mode)"
        )
    return warnings


def _get_memory_limit_bytes() -> int:
    """Compute memory limit from YUNSHU_MAX_MEMORY_GB, or 80% of UMA."""
    limit_gb = settings.get("YUNSHU_MAX_MEMORY_GB")
    if limit_gb is not None:
        return int(limit_gb * 1024**3)  # 0 = 'disabled' (unlimited)

    # Default: 80% of UMA (reserve for system + KV cache)
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        uma = int(result.stdout.strip())
        return int(uma * 0.8)
    except Exception:
        logger.debug("failed to read sysctl hw.memsize", exc_info=True)
        return 0  # unlimited


def describe_load_error(model: str, exc: BaseException) -> str:
    """One readable line saying why ``model`` did not load."""
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    name = type(exc).__name__
    lowered = f"{name} {text}".lower()
    if (
        isinstance(exc, FileNotFoundError)
        or "no such file" in lowered
        or (model.startswith(("/", ".", "~")) and not Path(model).expanduser().exists())
    ):
        return (
            f"{model}: model path not found; check the path, list local models "
            "with `yunshu model list`, or download it with `yunshu pull`"
        )
    elif (
        "out of memory" in lowered
        or "insufficient memory" in lowered
        or (isinstance(exc, MemoryError))
    ):
        hint = "not enough memory for this model; use a smaller or more quantized one"
    elif "safetensors" in lowered or "header" in lowered or "truncated" in lowered:
        hint = "weights are damaged or incomplete; run `yunshu pull` again"
    else:
        hint = "see the server log for the traceback"
    return f"{model}: {name}: {text[:300]} ({hint})"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown lifecycle with production hardening.

    Production features:
    - Startup validation of settings (bad values abort startup)
    - Startup timeout for model loading (configurable via YUNSHU_STARTUP_TIMEOUT)
    - Graceful shutdown: reject new -> drain -> cleanup -> model manager shutdown
    - Background task tracking for clean cancellation
    """
    global _memory_enforcer, _drain_event, _server_state, _startup_time
    global _background_tasks, _active_requests

    # Always reset state on lifespan entry — handles test isolation
    # where module-level globals persist between TestClient instances
    _drain_event = asyncio.Event()
    _server_state = ServerState.RUNNING
    _background_tasks = []
    _active_requests = 0

    # Store state on app for middleware to read (avoids stale module-level
    # state when TestClient doesn't trigger lifespan in Starlette 1.0+)
    app.state.server_state = ServerState.RUNNING
    app.state.load_error = None

    # ── Startup validation ──
    env_warnings = _validate_settings()
    for w in env_warnings:
        logger.warning("CONFIG: %s", w)

    # Validate model path exists in single-model mode
    if DEFAULT_MODEL:
        from pathlib import Path as _P

        model_path = _P(DEFAULT_MODEL)
        if not model_path.exists() and not DEFAULT_MODEL.startswith(
            ("hf://", "mlx-community/", "Qwen")
        ):
            logger.warning(
                "CONFIG: Model path '%s' does not exist locally and doesn't look like a HuggingFace ID — "
                "model loading may fail",
                DEFAULT_MODEL,
            )

    # ── Startup timeout ──
    startup_timeout = settings.get("YUNSHU_STARTUP_TIMEOUT")

    _startup_time = time.monotonic()

    if DEFAULT_MODEL:
        # Single-model mode. Pick the engine by model type: BatchedEngine can only
        # load text models via mlx_lm, so a VLM/omni checkpoint (e.g. gemma-4-e4b
        # vision, NVIDIA Nemotron-Omni) hard-fails "model type not supported" —
        # those need VLMEngine (mlx_vlm loader). The chat/completions routers
        # already handle a non-BatchedEngine global (is_batched=False → the
        # VLM-compatible path), so this just removes the "VLM needs multi-model
        # mode" caveat. Native-speech realtime keeps its own OmniEngine (omni.py).
        from .engine import set_engine

        _engine_mt = None
        try:
            from yunshu_engine.model_manager import _detect_model_type

            _engine_mt = _detect_model_type(DEFAULT_MODEL)
        except Exception:
            logger.debug(
                "model-type detection failed; defaulting to LLM", exc_info=True
            )

        _engine_name = _engine_mt.name if _engine_mt is not None else "LLM"
        # Build the right engine for ANY modality via the shared map (the same path
        # multi-model loading uses), so `serve -m <model>` works for text, VLM, image,
        # video, TTS, ASR, OCR, and embedding/reranker models alike — not just the few
        # types previously special-cased (everything else fell to BatchedEngine and
        # hard-failed "model type … not supported"). instantiate_engine also starts it.
        from yunshu_engine.model_manager import instantiate_engine

        logger.info("Single-model mode: %s detected as %s", DEFAULT_MODEL, _engine_name)
        try:
            engine = await asyncio.wait_for(
                instantiate_engine(_engine_mt, DEFAULT_MODEL),
                timeout=startup_timeout,
            )
            # A Hugging Face cache snapshot is listed under its repo id.
            set_engine(engine, display_id=hf_repo_id_for(DEFAULT_MODEL))
            logger.info(
                "Startup complete: model '%s' loaded (%.1fs)",
                DEFAULT_MODEL,
                time.monotonic() - _startup_time,
            )
        except TimeoutError:
            logger.error(
                "FATAL: model '%s' load timed out after %.0fs — server not ready",
                DEFAULT_MODEL,
                startup_timeout,
            )
            # No engine was bound (wait_for cancelled instantiate_engine), so
            # there is nothing to stop. Server starts; /health/ready reports
            # not-ready.
            app.state.load_error = (
                f"model load timed out after {startup_timeout:.0f}s "
                "(raise YUNSHU_STARTUP_TIMEOUT)"
            )
        except BaseException as e:
            if isinstance(e, asyncio.CancelledError | KeyboardInterrupt):
                raise
            app.state.load_error = describe_load_error(DEFAULT_MODEL, e)
            logger.error(
                "FATAL: model '%s' load failed: %s — server not ready",
                DEFAULT_MODEL,
                e,
            )

    elif _is_multi_model_enabled():
        # Multi-model mode: auto-discover and register models from models_dir.
        # init_model_manager(models_dir=...) already calls _discover_models()
        # internally — no second discovery pass needed here.
        max_bytes = _get_memory_limit_bytes()
        manager = init_model_manager(
            max_memory_bytes=max_bytes if max_bytes > 0 else None,
            models_dir=MODELS_DIR,
        )

        # Start ProcessMemoryEnforcer ()
        if max_bytes > 0:
            from yunshu_engine.process_memory_enforcer import ProcessMemoryEnforcer

            _memory_enforcer = ProcessMemoryEnforcer(
                model_manager=manager,
                max_bytes=max_bytes,
                poll_interval=2.0,
                ttl_seconds=settings.get("YUNSHU_MODEL_TTL_SECONDS"),
            )
            _memory_enforcer.start()
            logger.info(
                "ProcessMemoryEnforcer started (limit=%.1fGB)", max_bytes / 1024**3
            )

        logger.info(
            "Startup complete: multi-model mode, %d models registered (%.1fs)",
            len(manager.list_entries()),
            time.monotonic() - _startup_time,
        )

    # Initialize MCP client manager (LLM → external MCP tool servers)
    mcp_config_path = settings.get("YUNSHU_MCP_CONFIG")
    if mcp_config_path or settings.is_set("YUNSHU_MCP_SERVERS"):
        try:
            from yunshu_engine.mcp_client import init_mcp_client

            mcp_mgr = await init_mcp_client(config_path=mcp_config_path)
            if mcp_mgr and hasattr(app, "state"):
                app.state.mcp_client = mcp_mgr
            stats = mcp_mgr.get_stats()
            logger.info(
                "MCP client manager initialized: %d servers, %d tools",
                stats["connected_servers"],
                stats["total_tools"],
            )
        except Exception:
            logger.warning("MCP client initialization failed", exc_info=True)

    # Preload + warm the native-omni model so the first voice request is warm
    # (~4s) instead of cold (~30s). Opt out with YUNSHU_OMNI_PRELOAD=0.
    try:
        from .routers.omni import preload_and_warmup

        await preload_and_warmup()
    except Exception:
        logger.warning("Omni preload hook failed", exc_info=True)

    yield

    # ═══ Graceful shutdown ═══
    logger.info(
        "Shutdown initiated: %d active requests, draining...",
        _active_requests,
    )
    _server_state = ServerState.REQUESTED
    app.state.server_state = ServerState.REQUESTED

    # Wait for active requests to drain. poll the LIVE _active_requests counter
    # instead of awaiting the one-shot _drain_event. asyncio.Event, once .set() (the count
    # touched 0 at ANY point in the server's life), stays set forever — so the old
    # _drain_event.wait() returned IMMEDIATELY even when a request was in-flight at shutdown,
    # letting engine.stop() tear the model out from under a running generation. Loop until
    # the count is genuinely 0 or the timeout elapses.
    drain_timeout = settings.get("YUNSHU_DRAIN_TIMEOUT")
    _drain_deadline = time.monotonic() + drain_timeout
    while _active_requests > 0 and time.monotonic() < _drain_deadline:
        await asyncio.sleep(0.1)
    if _active_requests > 0:
        logger.warning(
            "Shutdown drain timed out after %.0fs: %d requests still active",
            drain_timeout,
            _active_requests,
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
            with suppress(asyncio.CancelledError):
                await task
    _background_tasks.clear()

    _server_state = ServerState.SHUTTING_DOWN
    app.state.server_state = ServerState.SHUTTING_DOWN

    # Reset active request counter (drain is complete or timed out)
    _active_requests = 0

    # Stop single-engine mode — call stop() even if not fully loaded
    # to clean up partial state from a failed start()
    engine = get_engine()
    if engine:
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
        time.monotonic() - _startup_time,
    )
    _startup_time = 0.0


_YUNSHU_LOGGERS = (
    "yunshu",
    "yunshu_gateway",
    "yunshu_engine",
    "yunshu_kv",
    "yunshu_control",
    "yunshu_cli",
)


def _configure_logging() -> None:
    """Give Yunshu's own loggers an output.

    uvicorn only configures its own loggers, so without a root handler every
    yunshu_* INFO/WARNING line (engine choice, runner/kernel state, fallbacks)
    was silently dropped. An existing root configuration (an embedding app,
    pytest) is left untouched; third-party libraries stay at WARNING.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    level = settings.get("YUNSHU_LOG_LEVEL")

    for name in _YUNSHU_LOGGERS:
        logging.getLogger(name).setLevel(level)


def create_app() -> FastAPI:
    _configure_logging()
    # Anthropic API paths — exact matching to avoid overmatching routes that
    # merely end in '/messages' (e.g. /api/v1/admin/messages).
    _ANTHROPIC_PATHS = frozenset(
        {
            "/v1/messages",
            "/messages",
            "/v1/messages/count_tokens",
            "/messages/count_tokens",
        }
    )

    app = FastAPI(
        title="Yunshu",
        version=yunshu_version(),
        description="Local LLM / VLM inference engine for Apple Silicon",
        lifespan=lifespan,
    )

    # ── Custom exception handlers for consistent error formats ──

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """Return OpenAI-format errors for validation failures.

        Anthropic endpoints (/v1/messages, /messages) get Anthropic format instead.
        """
        errors = exc.errors()
        first_loc = [str(x) for x in (errors[0].get("loc", []) if errors else [])]
        if first_loc and first_loc[0] == "body":
            first_loc = first_loc[1:]
        param = ".".join(first_loc) or None
        messages = []
        for err in errors:
            loc = ".".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "Invalid request")
            messages.append(f"{loc}: {msg}" if loc else msg)
        detail = "; ".join(messages)

        # Anthropic endpoints: return Anthropic error format
        path = request.url.path
        if path in _ANTHROPIC_PATHS:
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
                    "param": param,
                    "code": "validation_error",
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        """Ensure all HTTP errors follow the correct format for the endpoint."""
        path = request.url.path

        # Anthropic endpoints: return Anthropic error format
        if path in _ANTHROPIC_PATHS:
            error_type = {
                401: "authentication_error",
                403: "permission_error",
                404: "not_found_error",
                413: "request_too_large",
                429: "rate_limit_error",
                500: "api_error",
                503: "overloaded_error",
                529: "overloaded_error",
            }.get(exc.status_code, "invalid_request_error")
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

        # Default: OpenAI error format with code field
        error_type = (
            "invalid_request_error" if exc.status_code < 500 else "server_error"
        )
        _code_map = {
            400: "bad_request",
            401: "authentication_required",
            403: "permission_denied",
            404: "model_not_found",
            409: "conflict",
            413: "request_too_large",
            429: "rate_limit_exceeded",
            500: "internal_server_error",
            503: "service_unavailable",
        }
        error_code = _code_map.get(exc.status_code)
        if exc.status_code == 404 and str(exc.detail) == "Not Found":
            error_code = "not_found"  # unknown route, not an unknown model
        if exc.status_code == 400 and "exceeds max context window" in str(exc.detail):
            error_code = "context_length_exceeded"
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": str(exc.detail),
                    "type": error_type,
                    "param": None,
                    "code": error_code,
                }
            },
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
            request.method,
            request.url.path,
            exc,
            traceback.format_exc(),
        )
        path = request.url.path
        # Anthropic endpoints: return Anthropic error format
        if path in _ANTHROPIC_PATHS:
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
                    "param": None,
                    "code": "internal_error",
                }
            },
        )

    # CORS: configurable via YUNSHU_CORS_ORIGINS (comma-separated).
    # Defaults to ["*"] in dev, should be restricted in production.
    # Note: allow_credentials=True is invalid with allow_origins=["*"] per CORS spec;
    # browsers will reject the response. Use specific origins in production.
    cors_origins_str = settings.get("YUNSHU_CORS_ORIGINS")
    cors_origins = cors_origins_str.split(",") if cors_origins_str != "*" else ["*"]
    allow_credentials = cors_origins != ["*"]
    if cors_origins == ["*"]:
        logger.warning(
            "CORS: allow_origins=['*'] — set YUNSHU_CORS_ORIGINS for production"
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Accept",
            "X-Request-ID",
            # Anthropic SDK headers
            "anthropic-version",
            "anthropic-beta",
            "x-api-key",
            # OpenAI SDK headers
            "OpenAI-Organization",
            "OpenAI-Beta",
        ],
        max_age=3600,  # Cache preflight for 1 hour to reduce OPTIONS overhead
    )

    # Gateway middleware (order: outermost first)
    from .middleware.auth import AuthMiddleware
    from .middleware.metrics import MetricsMiddleware
    from .middleware.rate_limit import RateLimitMiddleware
    from .middleware.request_logging import RequestLoggingMiddleware

    # MetricsMiddleware is registered LAST (below, after AuthMiddleware) so it is the
    # OUTERMOST middleware. add_middleware prepends, so the last-added wraps everything —
    # registering it here (first) made it the INNERMOST, so it only saw requests that passed
    # auth/rate-limit/body-size and EVERY gateway-level rejection (401/429/413/503) was
    # invisible to yunshu_request_count / error_count / the duration histogram / the
    # aggregator's error-rate. Its own comments + the cardinality cap already assume it
    # runs before auth on attacker-controlled paths; this makes the registration match. Its
    # recording reads only path/method/status/latency (no pre-call auth state), so it is safe
    # outermost, and /metrics serving has its own _check_metrics_auth.

    # Response cache middleware: caches non-streaming responses when enabled
    if settings.get_bool("YUNSHU_RESPONSE_CACHE"):
        from .middleware.gateway_optimizer import ResponseCacheMiddleware

        app.add_middleware(ResponseCacheMiddleware)
        logger.info("ResponseCache middleware registered (YUNSHU_RESPONSE_CACHE=1)")

    # Shutdown rejection middleware: reject new inference requests during shutdown.
    # Reads state from app.state (set by lifespan) instead of module globals,
    # because Starlette 1.0+ TestClient may not trigger lifespan for non-context usage.
    @app.middleware("http")
    async def shutdown_guard(request: Request, call_next):
        state = getattr(request.app.state, "server_state", ServerState.RUNNING)
        if state != ServerState.RUNNING:
            path = request.url.path
            # Allow health/metrics even during shutdown
            if not path.startswith(("/health", "/metrics")):
                # Anthropic endpoints: return Anthropic error format
                if path in _ANTHROPIC_PATHS:
                    return JSONResponse(
                        status_code=503,
                        content={
                            "type": "error",
                            "error": {
                                "type": "overloaded_error",
                                "message": "Server is shutting down",
                            },
                        },
                        headers={"Retry-After": "5"},
                    )
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

    # Active request tracking for graceful shutdown drain
    _INFERENCE_PATHS = (
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/messages",
        "/v1/responses",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1/images/variations",
        # these ALSO hit an engine but were missing from the active-request
        # counter the sleep guard relies on (sleep could tear the model out from under
        # an in-flight one). The sleep guard now ALSO consults engine.has_active_requests
        # as the source of truth, but keep this list complete so the counter is accurate.
        "/v1/score",
        "/v1/rerank",
        "/v1/pooling",
        "/v1/classify",
        "/v1/ocr",
        "/v1/audio/translations",
        "/v1/audio/speech/stream",
    )

    @app.middleware("http")
    async def profile_capture_guard(request: Request, call_next):
        path = request.url.path
        if path in _INFERENCE_PATHS:
            try:
                from .routers.profiling import is_profile_capture_active

                if is_profile_capture_active():
                    return JSONResponse(
                        status_code=503,
                        content={
                            "error": {
                                "message": "Metal capture active — inference is serialized during profiling.",
                                "type": "server_error",
                                "code": "profile_capture_active",
                            }
                        },
                        headers={"Retry-After": "5"},
                    )
            except ImportError:
                pass
        return await call_next(request)

    @app.middleware("http")
    async def track_active_requests(request: Request, call_next):
        global _active_requests
        if request.url.path in _INFERENCE_PATHS:
            _active_requests += 1
            # Update Prometheus gauge for active requests.
            try:
                from .middleware.prometheus_exporter import get_prometheus_metrics

                get_prometheus_metrics().set_gauge(
                    "gateway_active_requests", float(_active_requests)
                )
            except Exception:
                pass
            try:
                return await call_next(request)
            finally:
                _active_requests -= 1
                try:
                    from .middleware.prometheus_exporter import get_prometheus_metrics

                    get_prometheus_metrics().set_gauge(
                        "gateway_active_requests", float(_active_requests)
                    )
                except Exception:
                    pass
                if _active_requests == 0 and _drain_event is not None:
                    _drain_event.set()
        return await call_next(request)

    # Request body size limit middleware (reject oversized payloads early)
    max_request_size = settings.get("YUNSHU_MAX_REQUEST_SIZE")

    @app.middleware("http")
    async def request_size_limit(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > max_request_size:
                    path = request.url.path
                    if path in _ANTHROPIC_PATHS:
                        return JSONResponse(
                            status_code=413,
                            content={
                                "type": "error",
                                "error": {
                                    "type": "invalid_request_error",
                                    "message": f"Request body too large: {content_length} bytes (max {max_request_size})",
                                },
                            },
                        )
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "message": f"Request body too large: {content_length} bytes (max {max_request_size})",
                                "type": "invalid_request_error",
                                "code": "request_too_large",
                            }
                        },
                    )
            except (ValueError, TypeError):
                pass  # Malformed content-length — let downstream handle it

        # Any request WITHOUT a Content-Length header (chunked transfer-encoding, HTTP/2
        # DATA frames, or a body with no declared length) must be byte-counted from the
        # stream — the header check above can't bound it. the old condition only
        # covered `transfer-encoding: chunked`, so a no-Content-Length HTTP/2 body (or any
        # framing that isn't literally "chunked") bypassed the limit entirely and was read
        # whole into memory by request.json(). Enforce on every no-Content-Length request.
        # (A request WITH Content-Length is bounded by the server to the declared length, and
        # the header check above rejects a declared length over the limit.)
        if content_length is None:
            try:
                chunks: list[bytes] = []
                total_size = 0
                async for chunk in request.stream():
                    total_size += len(chunk)
                    if total_size > max_request_size:
                        path = request.url.path
                        if path in _ANTHROPIC_PATHS:
                            return JSONResponse(
                                status_code=413,
                                content={
                                    "type": "error",
                                    "error": {
                                        "type": "invalid_request_error",
                                        "message": f"Request body too large: {total_size} bytes (max {max_request_size})",
                                    },
                                },
                            )
                        return JSONResponse(
                            status_code=413,
                            content={
                                "error": {
                                    "message": f"Request body too large: {total_size} bytes (max {max_request_size})",
                                    "type": "invalid_request_error",
                                    "code": "request_too_large",
                                }
                            },
                        )
                    chunks.append(chunk)
                body_bytes = b"".join(chunks)

                # Re-inject the body so downstream handlers (Pydantic validators)
                # can access it via request.body() or request.json().
                async def _receive_with_body():
                    return {
                        "type": "http.request",
                        "body": body_bytes,
                        "more_body": False,
                    }

                request._receive = _receive_with_body
            except Exception:
                pass  # Body read failed — let downstream handle it
        return await call_next(request)

    app.add_middleware(RequestLoggingMiddleware)
    if settings.get("YUNSHU_RATE_LIMIT_RPM") > 0:
        app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthMiddleware)
    # OUTERMOST — added last so it wraps auth/rate-limit and records their
    # rejections (401/429/413/503) into the metrics, which it could not see when innermost.
    app.add_middleware(MetricsMiddleware)
    # Outermost: sees the client's disconnect (BaseHTTPMiddleware hides it).
    from .middleware.disconnect import DisconnectWatchMiddleware

    app.add_middleware(DisconnectWatchMiddleware)

    # ── Gateway optimizer wiring ──
    from yunshu_engine.gateway_optimizer import (
        get_request_coalescer,
        get_response_cache,
        get_streaming_buffer,
    )

    app.state.request_coalescer = get_request_coalescer()
    app.state.streaming_buffer = get_streaming_buffer()

    # Auth: single optional bearer-token gate (YUNSHU_AUTH_TOKEN). The
    # multi-tenant RBACManager / TenantManager initialization has been removed
    # — this engine serves a single consumer. AuthMiddleware now only
    # validates the static token; the per-tenant/RBAC branches are gone.

    # Response cache: opt-in via YUNSHU_RESPONSE_CACHE=1
    if settings.get_bool("YUNSHU_RESPONSE_CACHE"):
        app.state.response_cache = get_response_cache()
        logger.info("ResponseCache enabled (YUNSHU_RESPONSE_CACHE=1)")

    # Startup warnings
    if settings.get_bool("YUNSHU_AUTH_DISABLED"):
        logger.warning(
            "SECURITY: AUTH IS DISABLED — all endpoints are publicly accessible "
            "without any authentication. "
            "This is INSECURE and should ONLY be used in development. "
            "Set YUNSHU_AUTH_TOKEN=<secret> or remove YUNSHU_AUTH_DISABLED for production."
        )
    elif not settings.get("YUNSHU_AUTH_TOKEN"):
        logger.warning(
            "SECURITY: No YUNSHU_AUTH_TOKEN set — admin endpoints ("
            "model load/unload, benchmarks, dashboard) are DENIED by default. "
            "Inference endpoints (chat/completions) remain accessible without auth. "
            "Set YUNSHU_AUTH_TOKEN=<secret> to enable full Bearer token auth."
        )

    # Import routers lazily to reduce startup memory
    from .routers import (
        anthropic,
        audio,
        chat,
        completions,
        embeddings,
        images,
        mcp,
        models,
        realtime,
        scoring,
        stream_ws,
        tokenize,
    )
    from .routers import monitoring as gw_monitoring

    # MCP client manager (LLM → external MCP tool servers)
    # Actual initialization happens in lifespan() since init_mcp_client is async.
    # Here we just check if the singleton was already initialized (e.g. in tests).
    if settings.is_set("YUNSHU_MCP_SERVERS"):
        try:
            from yunshu_engine.mcp_client import get_mcp_client_manager

            mcp_mgr = get_mcp_client_manager()
            if mcp_mgr:
                app.state.mcp_client = mcp_mgr
                logger.info("MCP client manager already initialized")
        except Exception:
            logger.debug("MCP client setup deferred to lifespan", exc_info=True)

    # Routes — L1 Gateway
    from .routers import cancel as cancel_mod
    from .routers import ocr as ocr_mod
    from .routers import responses as responses_mod

    app.include_router(chat.router, prefix="/v1")
    from .routers import omni as omni_mod

    app.include_router(completions.router, prefix="/v1")
    app.include_router(responses_mod.router, prefix="/v1")
    app.include_router(embeddings.router, prefix="/v1")
    app.include_router(models.router, prefix="/v1")
    app.include_router(anthropic.router, prefix="/v1")
    app.include_router(audio.router, prefix="/v1")
    app.include_router(images.router, prefix="/v1")
    app.include_router(tokenize.router, prefix="/v1")
    app.include_router(tokenize.router)  # vLLM-native /tokenize, /detokenize
    app.include_router(mcp.router, prefix="/v1")
    app.include_router(scoring.router, prefix="/v1")
    app.include_router(cancel_mod.router, prefix="/v1")
    app.include_router(ocr_mod.router)
    app.include_router(realtime.router)
    app.include_router(stream_ws.router)
    from .routers import ollama as ollama_mod

    app.include_router(ollama_mod.router)  # Ollama-compatible /api/*
    app.include_router(omni_mod.router)

    # Routes — L1 Gateway Monitoring (system, models, requests, prometheus)
    app.include_router(gw_monitoring.metrics_router)
    if settings.get_bool("YUNSHU_DEBUG_ROUTES"):
        app.include_router(gw_monitoring.router)

    def _safe_memory_usage(manager):
        if manager is None:
            return None
        mu = getattr(manager, "memory_usage", None)
        if callable(mu):
            return mu()
        return mu

    @app.get("/health")
    async def health() -> dict:
        # SECURITY: /health is in PUBLIC_PATHS (auth never runs), so it must NOT
        # leak internal state to an unauthenticated caller. It previously returned full
        # engine.get_stats() (loaded model IDs, queue depth), model_manager + memory_
        # enforcer memory figures, server-metrics snapshot, and MCP/registry internals —
        # reconnaissance for any anonymous probe. Reduce to a minimal liveness payload
        # (status + engine-loaded bool + server_state + uptime). The rich detail is
        # available, properly gated, via the authenticated /api/v1/monitoring/* router.
        from .engine import get_engine, get_model_manager

        engine = get_engine()
        _loaded = bool(getattr(engine, "is_loaded", False)) if engine else False
        if not _loaded:
            # Multi-model mode has no global engine — report loaded when the model
            # manager holds any loaded model (mirrors /health/ready), else /health
            # reported loaded=false despite serving models.
            try:
                _mgr = get_model_manager()
                if _mgr is not None and any(
                    getattr(e, "is_loaded", False) for e in _mgr.list_entries()
                ):
                    _loaded = True
            except Exception:
                pass
        result = {
            "status": "ok",
            "engine": {"loaded": _loaded},
            "server_state": _server_state,
            "uptime_seconds": round(time.monotonic() - _startup_time, 1)
            if _startup_time > 0
            else 0,
        }
        return result

    @app.get("/health/ready")
    async def readiness(request: Request) -> dict:
        """Readiness probe — is the server ready to accept traffic?"""
        engine = get_engine()
        manager = get_model_manager()

        # Use app.state for server state (reliable across TestClient
        # instances that may not trigger lifespan in Starlette 1.0+)
        current_state = getattr(request.app.state, "server_state", _server_state)

        checks = {}
        ready = True

        # Check if shutting down (check BEFORE other checks so we
        # report not-ready immediately during shutdown drain)
        checks["not_shutting_down"] = current_state == ServerState.RUNNING
        if current_state != ServerState.RUNNING:
            ready = False

        # Check if at least one model is loaded
        has_loaded_model = False
        try:
            if manager is not None:
                for entry in manager.list_entries():
                    if getattr(entry, "is_loaded", False):
                        has_loaded_model = True
                        break
            elif engine and getattr(engine, "is_loaded", False):
                has_loaded_model = True
        except Exception:
            logger.debug("model_loaded check failed", exc_info=True)
            has_loaded_model = False

        checks["model_loaded"] = has_loaded_model
        if not has_loaded_model:
            ready = False

        # Check GPU memory available (uses cached UMA size)
        try:
            import mlx.core as mx

            active = mx.get_active_memory()
            total_uma = _total_uma_bytes
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

        # K8s / load-balancer readiness probes key off the HTTP STATUS CODE,
        # not the JSON body. Returning a plain dict made FastAPI emit 200 even when
        # ready=False (no model loaded / GPU ≥95% / draining), so the LB kept routing
        # traffic to a dead/OOM/shutting-down node. Return 503 when not ready.
        from fastapi.responses import JSONResponse as _JSONResponse

        content: dict = {"ready": ready, "checks": checks}
        load_error = getattr(request.app.state, "load_error", None)
        if not ready and not has_loaded_model and load_error:
            content["reason"] = load_error
        return _JSONResponse(status_code=200 if ready else 503, content=content)

    @app.get("/health/live")
    async def liveness(request: Request) -> dict:
        """Liveness probe — is the server alive?"""
        current_state = getattr(request.app.state, "server_state", _server_state)
        return {"alive": True, "state": current_state}

    @app.get("/version")
    async def version() -> dict:
        """Server version info."""
        return {
            "version": yunshu_version(),
            "service": "yunshu",
            "description": "Local LLM / VLM inference engine for Apple Silicon",
        }

    # Anthropic SDK sends requests to /v1/messages without /v1 prefix
    app.include_router(anthropic.router)

    return app


app = create_app()
