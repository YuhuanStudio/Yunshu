from __future__ import annotations

"""OpenAI Models API compatible router."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

from pydantic import BaseModel, model_validator

from yunshu_control.audit_log import log_operation, resolve_actor
from yunshu_engine import settings

from ..engine import get_engine, get_model_manager

router = APIRouter(tags=["models"])


def _check_permission(request: Request, permission: str) -> None:
    """Check auth on gateway endpoints (deny-by-default).

    Single-consumer model: the multi-tenant RBAC machinery is gone, so this
    enforces the static-token contract.

    1. If YUNSHU_AUTH_DISABLED=true, allow (dev opt-in, logged at startup)
    2. If YUNSHU_AUTH_TOKEN is set, verify request actually presents it
       (applies to EVERY permission — setting a token locks the whole server)
    3. If no auth configured and not disabled:
       - ``can_infer`` (inference endpoints) → ALLOW. A local drop-in OpenAI
         server must serve inference out of the box, like Ollama/LM Studio;
         the startup banner promises exactly this. Bind to localhost (default)
         or set a token to lock it down.
       - anything else (model load/unload + admin ops) → DENY (secure default).
    """
    if settings.get_bool("YUNSHU_AUTH_DISABLED"):
        return
    # Static token auth — must verify the request actually provides it.
    # When a token IS configured it gates everything, inference included.
    auth_token = settings.get("YUNSHU_AUTH_TOKEN")
    if auth_token is not None and auth_token:
        import hmac

        auth = request.headers.get("Authorization", "")
        # Anthropic SDKs send the key as x-api-key instead of a bearer token.
        presented = (
            auth[7:] if auth.startswith("Bearer ") else request.headers.get("x-api-key")
        )
        if presented and hmac.compare_digest(presented, auth_token):
            return  # Valid static token
        # Token is configured but request doesn't provide a valid one
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # No auth configured and not explicitly disabled. Inference stays open
    # (matches the startup banner + the drop-in-local-server contract);
    # privileged ops fall through to deny-by-default.
    if permission == "can_infer":
        return
    logger.warning(
        "Privileged endpoint (%s) access denied: no auth configured. "
        "Set YUNSHU_AUTH_TOKEN or YUNSHU_AUTH_DISABLED=true to control access.",
        permission,
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


def _model_payload(card, entry, authenticated: bool) -> dict:
    """One ``/v1/models`` item: the card in every wire format, plus admin-only detail."""
    from ..model_card_formats import openai_model

    item = openai_model(card, detailed=authenticated)
    if authenticated and entry is not None:
        item["loaded"] = entry.is_loaded
        item["size_gb"] = round(entry.estimated_bytes / 1e9, 1)
        if entry.is_loaded and entry.engine is not None:
            try:
                stats = (
                    entry.engine.get_stats()
                    if hasattr(entry.engine, "get_stats")
                    else {}
                )
                item["stats"] = stats
            except Exception:
                logger.debug(f"failed to get stats for {entry.model_id}", exc_info=True)
    return item


@router.get("/models")
async def list_models(request: Request) -> dict:
    """List available models (OpenAI + Anthropic compatible, with the full model card).

    Every item carries the OpenAI fields, the Anthropic ``ModelInfo`` fields, the OpenRouter /
    vLLM / LM Studio metadata fields and the complete card under ``yunshu`` (see
    ``yunshu_engine/model_card.py``). Loaded state, size and engine stats are only included for
    authenticated requests.
    """
    from ..model_cards import all_cards, entry_card

    # A static-token holder authenticates with role="admin".
    _authenticated = getattr(request.state, "role", None) == "admin"
    manager = get_model_manager()
    models = []
    if manager is not None:
        for entry in manager.list_entries():
            models.append(_model_payload(entry_card(entry), entry, _authenticated))
    else:
        models = [_model_payload(c, None, _authenticated) for c in all_cards()]
    result = {
        "object": "list",
        "data": models,
        "has_more": False,
        "first_id": models[0]["id"] if models else None,
        "last_id": models[-1]["id"] if models else None,
    }
    # Registry stats are not part of the public list (they would leak global counters).
    if manager is not None and _authenticated:
        try:
            from yunshu_engine.model_registry import get_registry

            result["registry"] = get_registry().get_stats()
        except Exception:
            logger.debug("model_registry stats unavailable", exc_info=True)
    return result


@router.get("/models/{model_id:path}")
async def get_model(model_id: str, request: Request) -> dict:
    """Get one model with its full card (404 when unknown)."""
    from ..model_cards import find_card

    _check_permission(request, "can_infer")
    card = find_card(model_id)
    if card is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    manager = get_model_manager()
    entry = (
        manager.get_entry(manager.resolve_model_id(model_id) or model_id)
        if manager is not None
        else None
    )
    item = _model_payload(card, entry, getattr(request.state, "role", None) == "admin")
    if entry is not None:
        item.pop("stats", None)  # detail view: the card is enough
    return item


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
