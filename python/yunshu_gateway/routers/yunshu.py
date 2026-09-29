"""Yunshu-native endpoints (beyond the OpenAI / Anthropic / Ollama specs).

- ``GET /v1/yunshu/status``      one-call server state: models, memory, requests, throughput
- ``GET /v1/requests``           in-flight requests with phase and prefill progress
- ``GET /v1/requests/{id}``      one request (poll this for a non-streaming long prefill)
- ``DELETE /v1/requests/{id}``   cancel by the ``X-Request-Id`` the client sent (or the completion id)
- ``POST /v1/yunshu/warmup``     load a model, compile its kernels and (optionally) prefill a
                                 prompt so the first real request is warm; sets ``keep_alive``

Access follows the inference endpoints: open on a default local server, gated by
``YUNSHU_AUTH_TOKEN`` when one is set.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from yunshu_engine.request_tracker import get_request_tracker
from yunshu_engine.version import yunshu_version

from ..engine import get_display_model_id, get_engine, get_model_manager
from ..x_yunshu import (
    parse_keep_alive,
    progress_payload,
    recent_rates,
    registry,
)
from .models import _check_permission

logger = logging.getLogger(__name__)

router = APIRouter(tags=["yunshu"])

_STARTED = time.monotonic()


def _memory() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import mlx.core as mx

        out["active_gb"] = round(mx.get_active_memory() / 1e9, 2)
        out["cache_gb"] = round(mx.get_cache_memory() / 1e9, 2)
        out["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    except Exception:
        logger.debug("mlx memory unavailable", exc_info=True)
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        out["total_gb"] = round(total / 1e9, 1)
        if "active_gb" in out and total:
            out["pressure"] = round(out["active_gb"] * 1e9 / total, 3)
    except (ValueError, OSError, AttributeError):
        pass
    return out


def _models() -> list[dict[str, Any]]:
    manager = get_model_manager()
    rows: list[dict[str, Any]] = []
    if manager is not None:
        now = time.monotonic()
        for e in manager.list_entries():
            ka = manager.effective_keep_alive(e)
            rows.append(
                {
                    "id": e.model_id,
                    "type": e.model_type.name,
                    "loaded": e.is_loaded,
                    "loading": e.is_loading,
                    "pinned": e.is_pinned,
                    "size_gb": round(e.estimated_bytes / 1e9, 1),
                    "idle_s": round(now - e.last_access, 1)
                    if e.is_loaded and e.last_access
                    else None,
                    "keep_alive_s": ka,
                    "expires_in_s": (
                        round(x, 1)
                        if (x := manager.expires_in(e)) is not None
                        else None
                    ),
                    "error": e.load_error,
                }
            )
        return rows
    engine = get_engine()
    if engine is not None:
        rows.append(
            {
                "id": get_display_model_id() or getattr(engine, "model_name", None),
                "type": type(engine).__name__,
                "loaded": bool(getattr(engine, "is_loaded", False)),
                "loading": False,
                "pinned": True,  # single-model mode never frees its model
                "keep_alive_s": None,
                "expires_in_s": None,
            }
        )
    return rows


def _requests() -> list[dict[str, Any]]:
    tracker = get_request_tracker()
    out = []
    for info in registry.active():
        row = progress_payload(info)
        row["path"] = info.path
        row["stream"] = info.stream
        gen = info.gen
        if gen is not None:
            row["model"] = gen.model
            row["engine_request_id"] = gen.request_id
            row["cancelled"] = gen.cancel_event.is_set()
        out.append(row)
    known = {r.get("engine_request_id") for r in out}
    # Generations that did not come through a tracked HTTP path (batch jobs, Ollama proxy).
    for gen in tracker.all_active():
        if gen.request_id not in known and gen.client_request_id not in {
            r["request_id"] for r in out
        }:
            out.append(
                {
                    "request_id": gen.client_request_id or gen.request_id,
                    "engine_request_id": gen.request_id,
                    "phase": "running",
                    "model": gen.model,
                    "elapsed_s": round(gen.elapsed_s, 2),
                }
            )
    return out


@router.get("/yunshu/status")
async def status(request: Request) -> dict:
    """Everything an operator or a client wants to know before sending a big request."""
    _check_permission(request, "can_infer")
    reqs = _requests()
    phases: dict[str, int] = {}
    for r in reqs:
        phases[r["phase"]] = phases.get(r["phase"], 0) + 1
    window = registry.recent(60.0)
    live = 0.0
    for info in registry.active():
        st = info.stats
        if st is not None and st.phase == "decode" and st.t_last > st.t_first:
            live += (st.generated - 1) / (st.t_last - st.t_first)
    rates = recent_rates()
    state = getattr(request.app.state, "server_state", "running")
    return {
        "object": "yunshu.status",
        "version": yunshu_version(),
        "state": str(getattr(state, "value", state)),
        "uptime_s": round(time.monotonic() - _STARTED, 1),
        "load_error": getattr(request.app.state, "load_error", None),
        "models": _models(),
        "memory": _memory(),
        "requests": {
            "active": len(reqs),
            "queued": phases.get("queued", 0),
            "prefill": phases.get("prefill", 0),
            "decode": phases.get("decode", 0),
            "items": reqs,
        },
        "throughput": {
            "window_s": 60,
            "requests": len(window),
            "prompt_tokens": sum(r["prompt_tokens"] for r in window),
            "completion_tokens": sum(r["completion_tokens"] for r in window),
            "live_decode_tps": round(live, 1) if live else None,
            "mean_prefill_tps": rates["prefill_tps"],
            "mean_decode_tps": rates["decode_tps"],
        },
    }


@router.get("/requests")
async def list_requests(request: Request) -> dict:
    _check_permission(request, "can_infer")
    data = _requests()
    return {"object": "list", "data": data, "count": len(data)}


@router.get("/requests/{request_id}")
async def get_request(request_id: str, request: Request) -> dict:
    """State of one request by its ``X-Request-Id``: queued, prefill (tokens, %, ETA) or decode."""
    _check_permission(request, "can_infer")
    info = registry.get(request_id)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Request '{request_id}' not found or already completed",
        )
    row = progress_payload(info)
    row["path"] = info.path
    row["stream"] = info.stream
    if info.gen is not None:
        row["model"] = info.gen.model
    return {"object": "yunshu.request", **row}


@router.delete("/requests/{request_id}")
async def cancel_request(request_id: str, request: Request) -> dict:
    """Cancel a generation by the ``X-Request-Id`` the client sent (or the completion id)."""
    _check_permission(request, "can_infer")
    tracker = get_request_tracker()
    if tracker.cancel(request_id):
        return {"object": "yunshu.request", "id": request_id, "status": "cancelled"}
    info = registry.get(request_id)
    if info is not None:
        # Admitted but not yet handed to the engine (tokenizing, loading media).
        info.cancel_requested = True  # type: ignore[attr-defined]
        return {"object": "yunshu.request", "id": request_id, "status": "cancelling"}
    raise HTTPException(
        status_code=404,
        detail=f"Request '{request_id}' not found or already completed",
    )


class WarmupRequest(BaseModel):
    model: str | None = None
    prompt: str | None = None
    messages: list[dict] | None = None
    keep_alive: str | int | float | None = None
    max_tokens: int = 1


@router.post("/yunshu/warmup")
async def warmup(req: WarmupRequest, request: Request) -> dict:
    """Load the model, run one tiny generation (kernel compile, allocator warm-up) and, when
    ``prompt`` / ``messages`` is given, leave that prefix in the prompt cache."""
    _check_permission(request, "can_infer")
    try:
        keep_alive = parse_keep_alive(req.keep_alive)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    manager = get_model_manager()
    model = req.model or get_display_model_id()
    engine = get_engine()
    if not model and engine is not None:
        model = getattr(engine, "model_name", None)
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    t0 = time.perf_counter()
    loaded_now = False
    load_ms = 0.0
    generate = True
    if manager is not None:
        resolved = manager.resolve_model_id(model)
        entry = manager.get_entry(resolved) if resolved else None
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Model '{model}' not found")
        loaded_now = not entry.is_loaded
        try:
            eng = await manager.get_engine(resolved)
            if hasattr(eng, "is_running") and not eng.is_running:
                await eng.start()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        load_ms = round((time.perf_counter() - t0) * 1000, 1)
        model = resolved
        generate = entry.model_type.name in ("LLM", "VLM")
        if keep_alive is not None:
            manager.set_keep_alive(resolved, keep_alive)

    body: dict[str, Any] = {
        "model": model,
        "messages": req.messages
        or [{"role": "user", "content": req.prompt or "Hello"}],
        "max_tokens": max(int(req.max_tokens), 1),
        "temperature": 0,
    }
    result: dict[str, Any] = {
        "object": "yunshu.warmup",
        "model": model,
        "loaded_now": loaded_now,
        "load_ms": load_ms,
        "keep_alive_s": keep_alive,
        "generated": False,
    }
    if generate:
        headers = {}
        for h in ("authorization", "x-api-key"):
            if request.headers.get(h):
                headers[h] = request.headers[h]
        base = f"{request.url.scheme}://{request.url.netloc}"
        t1 = time.perf_counter()
        try:
            async with httpx.AsyncClient(
                base_url=base, headers=headers, timeout=None
            ) as client:
                resp = await client.post("/v1/chat/completions", json=body)
            if resp.status_code == 200:
                data = resp.json()
                result["generated"] = True
                result["x_yunshu"] = data.get("x_yunshu")
            else:
                result["warning"] = (
                    f"warm-up generation failed: HTTP {resp.status_code}"
                )
        except Exception as exc:
            result["warning"] = f"warm-up generation failed: {exc}"
        result["warmup_ms"] = round((time.perf_counter() - t1) * 1000, 1)
    return result
