"""Model management for the console: downloads, local inventory, fit-before-load.

- ``POST   /v1/yunshu/downloads``        start (or resume) a Hugging Face download -> job
- ``GET    /v1/yunshu/downloads``        all jobs: state, bytes, rate, ETA (small, poll-safe)
- ``GET    /v1/yunshu/downloads/{id}``   one job with every file's progress
- ``DELETE /v1/yunshu/downloads/{id}``   cancel (partial files stay; POST again resumes)
- ``GET    /v1/yunshu/models/local``     models on disk (models dir + HF cache), registered or not
- ``GET    /v1/yunshu/models/{id}/fit``  dry run of the load-time memory check
- ``POST   /v1/yunshu/models/{id}/reload`` unload + load the same model (applies ``reload`` settings)

Reads follow the inference endpoints' access; starting or cancelling a download needs the
``admin`` permission (a token when one is configured, otherwise denied).
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from yunshu_control.audit_log import log_operation, resolve_actor
from yunshu_engine import paths

from .. import downloads as dl
from ..engine import get_engine, get_model_manager
from . import models as models_router
from .models import _check_permission

logger = logging.getLogger(__name__)

router = APIRouter(tags=["yunshu"])

_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
_REVISION = re.compile(r"^[\w][\w./-]{0,127}$")


class DownloadRequest(BaseModel):
    repo: str
    revision: str | None = None
    allow_patterns: list[str] | None = Field(default=None, max_length=32)


def local_target(repo: str) -> Path:
    """``<models dir>/<org>/<name>``: the layout ``yunshu pull`` and discovery use."""
    org, name = repo.split("/")
    return paths.models_dir() / org / name


def _on_complete(job: dl.Job) -> None:
    """After the bytes land: prove the folder is a finished model, then register it."""
    from yunshu_cli.model import weights_complete

    if (
        job.patterns
    ):  # a partial fetch (a file or two) is not a model: nothing to verify
        return
    target = Path(job.path or "")
    done, reason = weights_complete(target)
    if not done:
        raise RuntimeError(
            f"download finished but the model looks incomplete: {reason}"
        )
    manager = get_model_manager()
    if manager is None:
        return
    if any(e.model_id == job.repo for e in manager.list_entries()):
        return
    try:
        from yunshu_engine.model_discovery import estimate_model_size

        manager.register_model(
            job.repo, str(target), estimated_bytes=estimate_model_size(target)
        )
        job.registered = True
    except Exception:  # noqa: BLE001 - the files are fine; a restart discovers them
        logger.warning("registering %s failed", job.repo, exc_info=True)


def _models_free_bytes() -> int | None:
    p = paths.models_dir()
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return None


def _job_view(job: dl.Job, detail: bool = False) -> dict[str, Any]:
    return job.to_dict(detail)


@router.post("/yunshu/downloads", status_code=202)
async def start_download(body: DownloadRequest, request: Request) -> dict[str, Any]:
    _check_permission(request, "admin")
    if not _REPO.match(body.repo):
        raise HTTPException(400, "repo must be a Hugging Face id like org/name")
    if body.revision is not None and not _REVISION.match(body.revision):
        raise HTTPException(400, "invalid revision")
    reg = dl.get_registry()
    target = local_target(body.repo)

    if body.revision is None and not body.allow_patterns:
        from yunshu_cli.model import weights_complete

        if weights_complete(target)[0]:
            return _job_view(reg.record_present(body.repo, str(target)))

    try:
        await asyncio.to_thread(
            reg.preflight, body.repo, body.revision, body.allow_patterns, target
        )
    except dl.InsufficientDiskError as exc:
        raise HTTPException(
            507,
            {
                "message": str(exc),
                "needed_bytes": exc.needed,
                "free_bytes": exc.free,
                "path": exc.path,
            },
        ) from exc
    except Exception as exc:  # noqa: BLE001 - an unknown repo, no network, a gated repo
        raise HTTPException(
            502, f"cannot read {body.repo} from the hub: {type(exc).__name__}: {exc}"
        ) from exc
    job = reg.submit(
        body.repo,
        revision=body.revision,
        allow_patterns=body.allow_patterns,
        local_dir=target,
        on_complete=_on_complete,
    )
    return _job_view(job)


@router.get("/yunshu/downloads")
async def list_downloads(request: Request) -> dict[str, Any]:
    _check_permission(request, "can_infer")
    jobs = dl.get_registry().jobs()
    return {
        "downloads": [_job_view(j) for j in jobs],
        "active": sum(1 for j in jobs if j.state in dl.ACTIVE),
        "free_bytes": _models_free_bytes(),
        "models_dir": str(paths.models_dir()),
    }


@router.get("/yunshu/downloads/{job_id}")
async def get_download(job_id: str, request: Request) -> dict[str, Any]:
    _check_permission(request, "can_infer")
    job = dl.get_registry().get(job_id)
    if job is None:
        raise HTTPException(404, f"download '{job_id}' not found")
    return _job_view(job, detail=True)


@router.delete("/yunshu/downloads/{job_id}")
async def cancel_download(job_id: str, request: Request) -> dict[str, Any]:
    _check_permission(request, "admin")
    job = dl.get_registry().cancel(job_id)
    if job is None:
        raise HTTPException(404, f"download '{job_id}' not found")
    return _job_view(job)


# ── local inventory ────────────────────────────────────────────────────

_INV_TTL_S = 15.0
_inv_lock = threading.Lock()
_inv_cache: tuple[float, str, list[dict]] | None = None


def _scan_disk(refresh: bool) -> list[dict]:
    """Directory sizes walk every file, so the scan is cached for a few seconds."""
    global _inv_cache
    base = str(paths.models_dir())
    now = time.monotonic()
    with _inv_lock:
        if (
            not refresh
            and _inv_cache is not None
            and _inv_cache[1] == base
            and now - _inv_cache[0] < _INV_TTL_S
        ):
            return _inv_cache[2]
    from yunshu_cli.model import scan_hf_cache, scan_models_dir

    rows = scan_models_dir(paths.models_dir()) + scan_hf_cache()
    with _inv_lock:
        _inv_cache = (time.monotonic(), base, rows)
    return rows


def _loaded_paths() -> dict[str, tuple[str, bool]]:
    """resolved model path -> (registered id, loaded)."""
    out: dict[str, tuple[str, bool]] = {}
    manager = get_model_manager()
    try:
        if manager is not None:
            for e in manager.list_entries():
                out[str(Path(e.model_path).resolve())] = (e.model_id, bool(e.is_loaded))
        else:
            eng = get_engine()
            p = getattr(eng, "_model_path", None) or getattr(eng, "model_path", None)
            if eng is not None and p:
                out[str(Path(str(p)).resolve())] = (
                    str(getattr(eng, "model_name", p)),
                    bool(getattr(eng, "is_loaded", False)),
                )
    except Exception:  # noqa: BLE001
        logger.debug("loaded-model lookup failed", exc_info=True)
    return out


def _describe(row: dict, loaded: dict[str, tuple[str, bool]]) -> dict[str, Any]:
    from yunshu_cli.model import weights_complete
    from yunshu_gateway.model_cards import _safe_card

    path = Path(row["path"])
    complete, reason = weights_complete(path)
    card = _safe_card(str(path), model_id=row["name"])
    reg_id, is_loaded = loaded.get(str(path.resolve()), (None, False))
    q = card.quantization or {}
    return {
        "id": row["name"],
        "path": row["path"],
        "source": row["source"],
        "size_bytes": row["size"],
        "model_type": row["type"],
        "kind": card.kind,
        "architecture": card.architecture,
        "family": card.family,
        "parameters": card.parameters,
        "quantization": {"bits": q.get("bits"), "group_size": q.get("group_size")}
        if q
        else None,
        "context_length": (card.context or {}).get("length"),
        "capabilities": card.capabilities(),
        "complete": complete,
        "complete_reason": reason,
        "registered_as": reg_id,
        "loaded": is_loaded,
    }


@router.get("/yunshu/models/local")
async def local_models(
    request: Request, refresh: bool = Query(False)
) -> dict[str, Any]:
    _check_permission(request, "can_infer")

    def build() -> dict[str, Any]:
        from yunshu_cli.model import _detect_model_type, _dir_size

        rows = list(_scan_disk(refresh))
        loaded = _loaded_paths()
        known = {str(Path(r["path"]).resolve()) for r in rows}
        # A model started from an explicit path lives outside the models dir and the
        # Hugging Face cache; it is still the model the console must show as loaded.
        for path, (reg_id, _is_loaded) in loaded.items():
            if path not in known and Path(path).is_dir():
                rows.append(
                    {
                        "name": reg_id,
                        "path": path,
                        "type": _detect_model_type(Path(path)),
                        "size": _dir_size(Path(path)),
                        "source": "path",
                    }
                )
        models = [_describe(r, loaded) for r in rows]
        return {
            "models": models,
            "total_bytes": sum(m["size_bytes"] for m in models),
            "models_dir": str(paths.models_dir()),
            "free_bytes": _models_free_bytes(),
        }

    return await asyncio.to_thread(build)


@router.get("/yunshu/models/{model_id:path}/fit")
async def model_fit(model_id: str, request: Request) -> dict[str, Any]:
    _check_permission(request, "can_infer")
    manager = get_model_manager()
    if manager is None:
        raise HTTPException(
            400, "fit-before-load needs multi-model mode (--models-dir)"
        )
    resolved = manager.resolve_model_id(model_id) or model_id
    try:
        return manager.fit_check(resolved)
    except KeyError:
        raise HTTPException(404, f"model '{model_id}' not registered") from None


class ReloadRequest(BaseModel):
    force: bool = Field(
        default=False,
        description="reload even while requests are running (they are torn down)",
    )


@router.post("/yunshu/models/{model_id:path}/reload")
async def reload_model(
    model_id: str, request: Request, body: ReloadRequest | None = None
) -> dict[str, Any]:
    """Unload and load the same model so ``reload`` settings (``needs_reload`` from
    ``PATCH /v1/yunshu/config``) take effect. Refuses while the model has running
    requests unless ``force``. Multi-model mode only; a single-engine server
    restarts instead."""
    _check_permission(request, "admin")
    actor = resolve_actor(request)
    force = bool(body and body.force)
    manager = get_model_manager()
    if manager is None:
        raise HTTPException(
            400,
            "reload needs multi-model mode (--models-dir); restart the server to "
            "apply reload settings (POST /v1/yunshu/service/restart)",
        )
    resolved = manager.resolve_model_id(model_id) or model_id
    entry = manager.get_entry(resolved)
    if entry is None:
        raise HTTPException(404, f"model '{model_id}' not registered")
    async with models_router._model_ops_lock:
        if resolved in models_router._model_ops_inflight:
            raise HTTPException(
                409, f"Model '{model_id}' is already being loaded or unloaded"
            )
        models_router._model_ops_inflight.add(resolved)
    t0 = time.monotonic()
    try:
        was_loaded = bool(entry.is_loaded)
        if was_loaded:
            if not force:
                active = False
                if entry.engine is not None:
                    try:
                        active = bool(entry.engine.has_active_requests())
                    except Exception:
                        active = True  # cannot prove it is idle
                if active or entry.leases > 0:
                    log_operation(
                        "model_reload",
                        model_id,
                        "failure",
                        actor=actor,
                        detail="active_requests",
                    )
                    raise HTTPException(
                        409,
                        f"Cannot reload '{model_id}': requests are running. Wait for "
                        "them or send force=true to tear them down.",
                    )
            if not await manager.unload_model(resolved, force=force):
                raise HTTPException(
                    409,
                    f"Model '{model_id}' was not unloaded (it became active). Retry.",
                )
        try:
            engine = await manager.get_engine(resolved)
            if hasattr(engine, "is_running") and not engine.is_running:
                await engine.start()
        except Exception as e:
            logger.error("Model reload failed: %s", e, exc_info=True)
            log_operation(
                "model_reload", model_id, "failure", actor=actor, detail=str(e)[:120]
            )
            raise HTTPException(
                500, f"Model '{model_id}' was unloaded but loading it again failed: {e}"
            ) from None
        log_operation("model_reload", model_id, "success", actor=actor)
        return {
            "status": "reloaded",
            "model": resolved,
            "was_loaded": was_loaded,
            "forced": force,
            "elapsed_s": round(time.monotonic() - t0, 2),
        }
    finally:
        async with models_router._model_ops_lock:
            models_router._model_ops_inflight.discard(resolved)
