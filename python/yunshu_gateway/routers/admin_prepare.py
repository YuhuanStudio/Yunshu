"""Prepare (faster cold start) for the console: the one-time PLE repack of a Flash-Next model.

- ``GET    /v1/yunshu/models/{id}/prepare``  applicability, prepared or not, active storage, disk, job
- ``POST   /v1/yunshu/models/{id}/prepare``  start the repack in the background (202; 507 when the disk is short)
- ``DELETE /v1/yunshu/models/{id}/prepare``  cancel a running repack, else remove the prepared copy

The same job class backs ``yunshu prepare``. Reads follow the inference endpoints' access; the
rest needs ``admin``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from yunshu_control.audit_log import log_operation, resolve_actor
from yunshu_engine import ple_prepare

from ..engine import get_model_manager
from .admin_models import _loaded_paths, _scan_disk
from .models import _check_permission

router = APIRouter(tags=["yunshu"])


def _resolve(model_id: str) -> Path:
    manager = get_model_manager()
    if manager is not None:
        entry = manager.get_entry(manager.resolve_model_id(model_id) or model_id)
        if entry is not None:
            return Path(entry.model_path)
    for row in _scan_disk(False):
        if row["name"] == model_id:
            return Path(row["path"])
    for path, (registered, _loaded) in _loaded_paths().items():
        if registered == model_id:
            return Path(path)
    raise HTTPException(404, f"model '{model_id}' not found")


def _view(path: Path) -> dict[str, Any]:
    info = ple_prepare.status(path)
    job = ple_prepare.job_for(path)
    info["job"] = job.poll() if job is not None else None
    return info


@router.get("/yunshu/models/{model_id:path}/prepare")
async def prepare_status(model_id: str, request: Request) -> dict[str, Any]:
    _check_permission(request, "can_infer")
    path = _resolve(model_id)
    return await asyncio.to_thread(_view, path)


@router.post("/yunshu/models/{model_id:path}/prepare", status_code=202)
async def prepare_start(model_id: str, request: Request) -> dict[str, Any]:
    _check_permission(request, "admin")
    path = _resolve(model_id)
    info = await asyncio.to_thread(ple_prepare.status, path)
    if not info["applicable"]:
        raise HTTPException(400, "this model has no PLE table to prepare")
    if info["prepared"]:
        return await asyncio.to_thread(_view, path)
    try:
        await asyncio.to_thread(ple_prepare.start_job, path)
    except ple_prepare.InsufficientDiskError as exc:
        raise HTTPException(
            507,
            {
                "message": str(exc),
                "needed_bytes": exc.needed,
                "free_bytes": exc.free,
                "path": exc.path,
            },
        ) from exc
    log_operation("model_prepare", model_id, "success", actor=resolve_actor(request))
    return await asyncio.to_thread(_view, path)


@router.delete("/yunshu/models/{model_id:path}/prepare")
async def prepare_cancel_or_remove(model_id: str, request: Request) -> dict[str, Any]:
    _check_permission(request, "admin")
    path = _resolve(model_id)
    job = ple_prepare.job_for(path)
    if job is not None and job.state() == "running":
        await asyncio.to_thread(job.cancel)
        return await asyncio.to_thread(_view, path)
    if _loaded_paths().get(str(path.resolve()), (None, False))[1]:
        raise HTTPException(
            409, "unload the model first: it may be reading the prepared copy"
        )
    await asyncio.to_thread(ple_prepare.remove, path)
    log_operation(
        "model_prepare_remove", model_id, "success", actor=resolve_actor(request)
    )
    return await asyncio.to_thread(_view, path)
