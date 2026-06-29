from __future__ import annotations

"""Sleep/wake endpoints for power saving and resource management.

3-level sleep:
- L0 (pause): Stop accepting requests, keep model + KV loaded
- L1 (unload): Unload model weights, keep KV cache
- L2 (deep): Unload everything — minimum memory footprint
"""
import contextlib
import logging
import os
import threading

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..engine import get_engine, get_model_manager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["sleep"])

from yunshu_control.audit_log import log_operation, resolve_actor


def _check_permission(request: Request, permission: str) -> None:
    """Check RBAC permission on sleep/wake endpoints.

    Security (deny-by-default):
    1. If YUNSHU_AUTH_DISABLED=true, allow (dev opt-in, logged at startup)
    2. If rbac_key is set (from TenantAuthMiddleware), check has_permission()
    3. If tenant is set (from TenantAuthMiddleware), allow (legacy tenant auth)
    4. If YUNSHU_AUTH_TOKEN is set, verify request actually presents it
    5. If no auth configured and not disabled — DENY access (secure default)
    """
    if os.environ.get("YUNSHU_AUTH_DISABLED", "").lower() in ("true", "1", "yes"):
        return
    rbac_key = getattr(request.state, "rbac_key", None)
    if rbac_key is not None:
        if not rbac_key.has_permission(permission):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return
    # Tenant set by TenantAuthMiddleware (legacy tenant auth)
    tenant = getattr(request.state, "tenant", None)
    if tenant is not None:
        # Don't let a legacy tenant inherit admin-class permissions via the
        # blanket tenant-allow. Sleep/wake guard can_unload_models, which is
        # privileged → require a real admin role.
        from .models import _TENANT_DENIED_PERMISSIONS

        if permission not in _TENANT_DENIED_PERMISSIONS:
            return
        _role = str(getattr(request.state, "role", "") or "")
        if _role.lower() in ("admin", "system", "owner") or _role.upper().endswith(
            "ADMIN"
        ):
            return
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    # Static token auth — must verify the request actually provides it
    auth_token = os.environ.get("YUNSHU_AUTH_TOKEN")
    if auth_token is not None and auth_token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            import hmac

            if hmac.compare_digest(auth[7:], auth_token):
                return
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # No auth configured and not explicitly disabled — deny by default
    logger.warning(
        "Sleep endpoint access denied: no auth configured. "
        "Set YUNSHU_AUTH_TOKEN or YUNSHU_AUTH_DISABLED=true to control access."
    )
    raise HTTPException(
        status_code=401,
        detail="No authentication configured. Set YUNSHU_AUTH_TOKEN or YUNSHU_AUTH_DISABLED=true.",
    )


class SleepRequest(BaseModel):
    # 0=pause (in-process), 1=unload weights, 2=deep sleep (full teardown).
    # Validate at schema boundary so level=3+ returns 422 instead of being
    # silently clamped to 2 (which produced confusing post-call state).
    level: int = Field(default=0, ge=0, le=2)


# ── Sleep state ──
_sleeping = False
_sleep_level = -1  # -1 = awake
_sleep_transitioning = False  # True during sleep/wake transition
_sleep_lock = threading.Lock()
_saved_model_name: str | None = None  # Model name saved before L1/L2 unload


@router.post("/sleep")
async def sleep_server(req: SleepRequest, request: Request):
    """Put server to sleep at the specified level."""
    actor = resolve_actor(request)
    _check_permission(request, "can_unload_models")
    global _sleeping, _sleep_level, _sleep_transitioning, _saved_model_name

    level = max(0, min(2, req.level))

    with _sleep_lock:
        if _sleeping:
            raise HTTPException(
                status_code=409, detail=f"Already sleeping at level {_sleep_level}"
            )
        if _sleep_transitioning:
            raise HTTPException(status_code=409, detail="Sleep transition in progress")

        # Check for active requests inside the lock to prevent TOCTOU race:
        # without this, a request could arrive between the check and setting
        # _sleeping=True, causing the engine to be torn down mid-generation.
        if level >= 1:
            import yunshu_gateway.main as _main

            # The HTTP _active_requests counter only covers _INFERENCE_PATHS,
            # which MISSES many engine-hitting endpoints (/v1/score, /rerank, /pooling,
            # /classify, /ocr, /audio/translations, /audio/speech/stream, /audio/speech-
            # to-speech/*, /audio/voice-pipeline). Those resolve to the SAME default
            # engine L1/L2 sleep tears down, so a sleep during one of them pulls the
            # model out mid-generation → crash / teardown-vs-GPU-work race. ALSO consult
            # the engine's authoritative has_active_requests() (the unload source of
            # truth), so a request the counter missed still blocks the sleep.
            _eng_busy = False
            try:
                from yunshu_gateway.engine import get_engine as _ge

                _eng = _ge()
                _hac = (
                    getattr(_eng, "has_active_requests", None)
                    if _eng is not None
                    else None
                )
                if callable(_hac):
                    _eng_busy = bool(_hac())
            except Exception:
                _eng_busy = True  # can't prove idle → fail safe (refuse the sleep)
            if _main._active_requests > 0 or _eng_busy:
                log_operation(
                    "server_sleep",
                    "server",
                    "failure",
                    actor=actor,
                    level=level,
                    detail="active_requests",
                )
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Cannot sleep: active requests in progress "
                        f"(http={_main._active_requests}, engine_busy={_eng_busy}). "
                        f"Wait for them to complete or use force shutdown."
                    ),
                )

        # Set sleep flag inside the lock so the active-request middleware
        # sees it atomically — no request can slip in between the check
        # and the state change.
        _sleep_level = level
        _sleeping = True
        _sleep_transitioning = True
        os.environ["YUNSHU_SLEEPING"] = "1"

    try:
        engine = get_engine()
        manager = get_model_manager()

        if level >= 1 and engine:
            # Save model name before unloading so wake-up can restore it
            _saved_model_name = (
                getattr(engine, "_model_name", None)
                or getattr(engine, "model_name", None)
                or os.environ.get("YUNSHU_MODEL")
            )

        logger.info("Server entering L%d sleep", level)

        if level >= 1 and engine and engine.is_loaded:
            # stop() the engine to tear down ALL subsystems (engine_core loop +
            # thread, KV/prefix caches, KV-transfer sockets, lora manager, spec decoders)
            # before nulling. Nulling only _model/_loaded/_running and leaving
            # everything else alive means wake builds a BRAND-NEW engine, abandoning the old
            # subsystems (thread/socket/KV-buffer leak across every sleep/wake cycle), and a
            # "KV cache retained" claim would be false (the fresh engine starts empty). L1 and
            # L2 both do a clean teardown; wake rebuilds.
            with contextlib.suppress(Exception):
                await engine.stop()
            if hasattr(engine, "_model"):
                engine._model = None
            if hasattr(engine, "_loaded"):
                engine._loaded = False
            if hasattr(engine, "_running"):
                engine._running = False
            logger.info("L1 sleep: engine stopped, weights and caches released")

        # Free manager-tracked engines for L1 too (not level>=2 only). In
        # multi-model mode get_engine() is None, so the singleton block above does nothing —
        # otherwise L1 is a pure no-op that flips _sleeping=True while every model stays resident.
        if level >= 1:
            # Unload everything tracked by the manager.
            if manager:
                # Route manager-tracked engines through unload_model(force=True)
                # so is_loaded AND _current_memory_bytes stay consistent. Calling
                # entry.engine.stop() directly leaves every entry is_loaded=True
                # pointing at a stopped engine and the manager's memory accounting never
                # decremented — accounting drift that later mis-triggers/blocks eviction,
                # and in multi-model mode get_engine() would hand a request the stale
                # (gutted) entry. force=True bypasses the active-request guard (this is a
                # deliberate full teardown).
                for entry in list(manager.list_entries()):
                    if entry.is_loaded and entry.engine:
                        try:
                            await manager.unload_model(entry.model_id, force=True)
                        except Exception:
                            logger.debug("manager unload failed", exc_info=True)
            logger.info("L%d sleep: all models and caches released", level)

        log_operation("server_sleep", "server", "success", actor=actor, level=level)
        return {"status": "sleeping", "level": level}
    except Exception:
        # Roll back sleep state on error so the server isn't stuck
        with _sleep_lock:
            _sleeping = False
            _sleep_level = -1
            os.environ.pop("YUNSHU_SLEEPING", None)
        log_operation("server_sleep", "server", "failure", actor=actor, level=req.level)
        raise
    finally:
        _sleep_transitioning = False


@router.post("/wake-up")
async def wake_up_server(request: Request):
    """Wake up server from sleep, reload model if needed."""
    actor = resolve_actor(request)
    _check_permission(request, "can_unload_models")
    global _sleeping, _sleep_level, _sleep_transitioning, _saved_model_name

    with _sleep_lock:
        if not _sleeping:
            return {"status": "awake", "message": "Server is already awake"}
        if _sleep_transitioning:
            raise HTTPException(status_code=409, detail="Sleep transition in progress")
        _sleep_transitioning = True

    try:
        level = _sleep_level
        model_name = _saved_model_name or os.environ.get("YUNSHU_MODEL")

        if level >= 1 and model_name:
            # Reload the model
            from yunshu_engine.batched_engine import BatchedEngine

            from ..engine import set_engine

            engine = BatchedEngine(model_name=model_name)
            set_engine(engine)
            await engine.start()
            logger.info("Woke up: model reloaded from %s", model_name)

        # Clear sleep state only after all operations succeed
        _sleeping = False
        _sleep_level = -1
        _saved_model_name = None
        os.environ.pop("YUNSHU_SLEEPING", None)

        log_operation(
            "server_wake", "server", "success", actor=actor, previous_level=level
        )
        return {"status": "awake", "previous_level": level}
    except Exception:
        # If wake-up fails, clear transitioning flag but leave sleeping=True
        # so the server is still marked as sleeping (it wasn't fully woken up).
        _sleep_transitioning = False
        log_operation(
            "server_wake", "server", "failure", actor=actor, detail="reload_failed"
        )
        raise
    finally:
        _sleep_transitioning = False


@router.get("/sleep/status")
async def sleep_status(request: Request):
    """Check current sleep state.

    SECURITY: gated by `can_view_system` so unauthenticated callers cannot
    fingerprint operational state.
    """
    import os

    if os.environ.get("YUNSHU_AUTH_DISABLED", "").lower() not in ("true", "1", "yes"):
        rbac_key = getattr(request.state, "rbac_key", None)
        if rbac_key is not None and not rbac_key.has_permission("can_view_system"):
            from fastapi import HTTPException

            raise HTTPException(status_code=403, detail="Requires can_view_system")
    return {
        "sleeping": _sleeping,
        "level": _sleep_level if _sleeping else -1,
        "transitioning": _sleep_transitioning,
    }


def is_sleeping() -> bool:
    """Check if server is in sleep mode (used by request middleware)."""
    return _sleeping


def get_sleep_state() -> dict:
    """Return a snapshot of the current sleep state for /health.

    Same shape as GET /sleep/status — exported so the main app can
    surface sleep state without an extra HTTP round-trip.
    """
    return {
        "sleeping": _sleeping,
        "level": _sleep_level if _sleeping else -1,
        "transitioning": _sleep_transitioning,
    }
