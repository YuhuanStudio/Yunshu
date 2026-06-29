from __future__ import annotations

"""OpenAI Models API compatible router."""

import asyncio
import logging
import os
import time

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)
from pydantic import BaseModel, model_validator

from yunshu_control.audit_log import log_operation, resolve_actor

from ..engine import get_engine, get_model_manager

router = APIRouter(tags=["models"])


def _check_permission(request: Request, permission: str) -> None:
    """Check auth on gateway endpoints (deny-by-default).

    Single-consumer model: the multi-tenant RBAC machinery is gone, so this
    enforces the static-token contract.

    1. If YUNSHU_AUTH_DISABLED=true, allow (dev opt-in, logged at startup)
    2. If YUNSHU_AUTH_TOKEN is set, verify request actually presents it
    3. If no auth configured and not disabled — DENY access (secure default)
    """
    if os.environ.get("YUNSHU_AUTH_DISABLED", "").lower() in ("true", "1", "yes"):
        return
    # Static token auth — must verify the request actually provides it
    auth_token = os.environ.get("YUNSHU_AUTH_TOKEN")
    if auth_token is not None and auth_token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            import hmac

            if hmac.compare_digest(auth[7:], auth_token):
                return  # Valid static token
        # Token is configured but request doesn't provide a valid one
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # No auth configured and not explicitly disabled — deny by default
    logger.warning(
        "Admin endpoint access denied: no auth configured. "
        "Set YUNSHU_AUTH_TOKEN or YUNSHU_AUTH_DISABLED=true to control access."
    )
    raise HTTPException(
        status_code=401,
        detail="No authentication configured. Set YUNSHU_AUTH_TOKEN or YUNSHU_AUTH_DISABLED=true.",
    )


def _check_model_access(request: Request, model: str | None) -> None:
    """No-op model-access gate (retained as a stable call seam).

    The per-key model-isolation it once enforced was part of the multi-tenant
    RBAC machinery, which has been removed (single-consumer model). Kept so the
    many model-serving routes that call it stay unchanged.
    """
    return None


# Guard against concurrent load/unload of the same model
_model_ops_lock = asyncio.Lock()
_model_ops_inflight: set[str] = set()


class LoadModelRequest(BaseModel):
    model: str
    pin: bool = False

    @model_validator(mode="before")
    @classmethod
    def _accept_model_id_alias(cls, data):
        """Accept 'model_id' as an alias for 'model' (doc parity).

        If both are provided, 'model' wins (explicit canonical name).
        """
        if isinstance(data, dict):
            if not data.get("model") and data.get("model_id"):
                data = dict(data)
                data["model"] = data["model_id"]
        return data


@router.get("/models")
async def list_models(request: Request) -> dict:
    """List available models (OpenAI-compatible).

    Public endpoint per OpenAI spec.  Detailed info (stats, sizes, loaded
    status) is only included for authenticated requests.
    """
    models = []
    # A static-token holder authenticates with role="admin"; an authenticated
    # request gets the detailed listing (loaded/type/size_gb/stats), the public
    # OpenAI-compat list stays minimal.
    _authenticated = getattr(request.state, "role", None) == "admin"

    # Multi-model mode
    manager = get_model_manager()
    if manager is not None:
        for entry in manager.list_entries():
            model_info = {
                "id": entry.model_id,
                "object": "model",
                "created": int(entry.load_time)
                if entry.load_time > 0
                else int(time.time()),
                "owned_by": "yunshu",
            }
            if _authenticated:
                model_info["loaded"] = entry.is_loaded
                model_info["type"] = entry.model_type.name
                model_info["size_gb"] = round(entry.estimated_bytes / 1e9, 1)
            if _authenticated and entry.is_loaded and entry.engine is not None:
                try:
                    stats = (
                        entry.engine.get_stats()
                        if hasattr(entry.engine, "get_stats")
                        else {}
                    )
                    model_info["stats"] = stats
                except Exception:
                    logger.debug(
                        f"failed to get stats for {entry.model_id}", exc_info=True
                    )
            models.append(model_info)
        result = {"object": "list", "data": models}
        # Include model registry stats for debugging/monitoring — but NOT for an
        # unauthenticated caller (the public OpenAI-compat list must not leak
        # global total_entries/active_owners).
        if _authenticated:
            try:
                from yunshu_engine.model_registry import get_registry

                result["registry"] = get_registry().get_stats()
            except Exception:
                logger.debug("model_registry stats unavailable", exc_info=True)
        return result

    # Single-engine mode
    engine = get_engine()
    if engine and engine.is_loaded:
        _load_time = getattr(engine, "_load_time", None) or getattr(
            engine, "load_time", None
        )
        models.append(
            {
                "id": engine.model_name,
                "object": "model",
                "created": int(_load_time) if _load_time else int(time.time()),
                "owned_by": "yunshu",
            }
        )
    return {"object": "list", "data": models}


@router.get("/models/{model_id:path}")
async def get_model(model_id: str, request: Request) -> dict:
    """Get details for a specific model."""
    _check_permission(request, "can_infer")
    manager = get_model_manager()
    if manager is not None:
        # Resolve aliases / case / org-prefix the same way inference does,
        # so an id that successfully runs chat/completions doesn't 404 here.
        _resolved = manager.resolve_model_id(model_id) or model_id
        entry = manager.get_entry(_resolved)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        return {
            "id": entry.model_id,
            "object": "model",
            "owned_by": "yunshu",
            "loaded": entry.is_loaded,
            "type": entry.model_type.name,
            "size_gb": round(entry.estimated_bytes / 1e9, 1),
            # OpenAI spec: `created` is required int — fall back to wall-clock if model
            # hasn't been loaded yet so we never emit null for a required field.
            "created": int(entry.load_time)
            if entry.load_time > 0
            else int(time.time()),
        }

    engine = get_engine()
    if not engine or not engine.resolve_model_id(model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return {
        "id": model_id,
        "object": "model",
        "owned_by": "yunshu",
        # OpenAI spec: `created` is a required int.
        "created": int(time.time()),
    }


@router.post("/models/load")
async def load_model(req: LoadModelRequest, request: Request) -> dict:
    """Load a model (supports both single-engine and multi-model modes)."""
    actor = resolve_actor(request)
    _check_permission(request, "can_load_models")
    if not req.model or not req.model.strip():
        log_operation(
            "model_load", req.model or "", "failure", actor=actor, detail="empty_model"
        )
        raise HTTPException(status_code=400, detail="model field cannot be empty")

    # Guard against concurrent load/unload of the same model
    async with _model_ops_lock:
        if req.model in _model_ops_inflight:
            log_operation(
                "model_load",
                req.model,
                "failure",
                actor=actor,
                detail="already_inflight",
            )
            raise HTTPException(
                status_code=409,
                detail=f"Model '{req.model}' is already being loaded or unloaded",
            )
        _model_ops_inflight.add(req.model)

    try:
        manager = get_model_manager()

        if manager is not None:
            try:
                engine = await manager.get_engine(req.model)
                if hasattr(engine, "is_running") and not engine.is_running:
                    await engine.start()
                log_operation("model_load", req.model, "success", actor=actor)
                return {"status": "loaded", "model": req.model}
            except KeyError:
                log_operation(
                    "model_load",
                    req.model,
                    "failure",
                    actor=actor,
                    detail="not_registered",
                )
                raise HTTPException(
                    status_code=404, detail=f"Model '{req.model}' not registered"
                ) from None
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Model load error: {e}", exc_info=True)
                log_operation(
                    "model_load", req.model, "failure", actor=actor, detail=str(e)[:120]
                )
                # Surface load-time RuntimeErrors verbatim: these are
                # deliberate "model unsupported in this build" errors raised by
                # engines (e.g. quantize shape mismatches for 4-bit Omni)
                # and the user needs the precise reason, not "Model loading
                # failed".
                if isinstance(e, RuntimeError):
                    raise HTTPException(
                        status_code=400, detail=f"Model loading failed: {e}"
                    ) from None
                raise HTTPException(
                    status_code=500, detail="Model loading failed"
                ) from None

        # Single-engine mode
        engine = get_engine()
        if engine is None:
            log_operation(
                "model_load",
                req.model,
                "failure",
                actor=actor,
                detail="engine_not_initialized",
            )
            raise HTTPException(status_code=503, detail="Engine not initialized")

        from yunshu_engine.batched_engine import BatchedEngine

        if isinstance(engine, BatchedEngine):
            log_operation(
                "model_load",
                req.model,
                "failure",
                actor=actor,
                detail="single_engine_not_supported_in_batched",
            )
            raise HTTPException(
                status_code=400,
                detail="Single-engine load not supported in batched mode — use model manager",
            )
        from yunshu_engine.mlx_executor import get_mlx_executor

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(get_mlx_executor(), engine.load, req.model)
        await engine.start()

        log_operation("model_load", req.model, "success", actor=actor)
        return {"status": "loaded", "model": req.model}
    finally:
        async with _model_ops_lock:
            _model_ops_inflight.discard(req.model)


@router.post("/models/unload/{model_id:path}")
async def unload_model(model_id: str, request: Request) -> dict:
    """Unload a model and release memory."""
    actor = resolve_actor(request)
    _check_permission(request, "can_unload_models")
    # Guard against concurrent load/unload of the same model
    async with _model_ops_lock:
        if model_id in _model_ops_inflight:
            log_operation(
                "model_unload",
                model_id,
                "failure",
                actor=actor,
                detail="already_inflight",
            )
            raise HTTPException(
                status_code=409,
                detail=f"Model '{model_id}' is already being loaded or unloaded",
            )
        _model_ops_inflight.add(model_id)

    try:
        manager = get_model_manager()
        if manager is None:
            log_operation(
                "model_unload",
                model_id,
                "failure",
                actor=actor,
                detail="multi_model_not_active",
            )
            raise HTTPException(status_code=400, detail="Multi-model mode not active")

        # Resolve aliases / case / org-prefix the way retrieve and
        # inference do, so unloading by an id that successfully ran chat doesn't 404.
        # Keep `model_id` as the raw key the _model_ops_inflight set was keyed on
        # (the finally block discards it) — only the entry/unload use the resolved id.
        _resolved_id = manager.resolve_model_id(model_id) or model_id
        entry = manager.get_entry(_resolved_id)
        if entry is None:
            log_operation(
                "model_unload", model_id, "failure", actor=actor, detail="not_found"
            )
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

        # Safety: refuse unload if the engine has active requests that would
        # crash when the model is torn down mid-generation.
        if entry.is_loaded and entry.engine is not None:
            # Engines expose has_active_requests() — a
            # getattr(engine,'_active_requests',0) would read a NON-EXISTENT attribute
            # (always 0), leaving this safety net dead so an unload could tear the
            # model down mid-generation (use-after-unload crash).
            try:
                active = entry.engine.has_active_requests()
            except Exception:
                active = False
            if active:
                # NOTE: do NOT modify _model_ops_inflight here — the finally
                # block handles cleanup under the lock.  Modifying the set
                # without the lock is a race with concurrent load/unload.
                log_operation(
                    "model_unload",
                    model_id,
                    "failure",
                    actor=actor,
                    detail="active_requests",
                )
                raise HTTPException(
                    status_code=409,
                    detail=f"Cannot unload '{model_id}': active requests in progress. Wait for them to complete or cancel them first.",
                )

        # Honor the manager's return value. The router's has_active_requests
        # pre-check (above) and the manager's own locked re-check sit either side
        # of a TOCTOU window: a request can start in between, so the manager may REFUSE
        # the unload (returns False) even though the pre-check passed. The result was
        # discarded and we always reported "unloaded" — telling the client a still-loaded,
        # actively-serving model was gone. A False also covers already-unloaded/not-found.
        _unloaded = await manager.unload_model(_resolved_id)
        if not _unloaded:
            log_operation(
                "model_unload",
                model_id,
                "skipped",
                actor=actor,
                detail="in_use_or_absent",
            )
            raise HTTPException(
                status_code=409,
                detail=f"Model '{model_id}' was not unloaded — it became active or was already unloaded. Retry after in-flight requests complete.",
            )
        log_operation("model_unload", model_id, "success", actor=actor)
        return {"status": "unloaded", "model": model_id}
    except HTTPException:
        raise
    except Exception as e:
        log_operation(
            "model_unload", model_id, "failure", actor=actor, detail=str(e)[:120]
        )
        raise
    finally:
        async with _model_ops_lock:
            _model_ops_inflight.discard(model_id)
