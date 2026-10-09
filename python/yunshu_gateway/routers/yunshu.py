"""Yunshu-native endpoints (beyond the OpenAI / Anthropic / Ollama specs).

- ``GET /v1/yunshu/status``      one-call server state: models, memory, requests, throughput
- ``GET /v1/requests``           in-flight requests with phase and prefill progress
- ``GET /v1/requests/{id}``      one request (poll this for a non-streaming long prefill)
- ``DELETE /v1/requests/{id}``   cancel by the ``X-Request-Id`` the client sent (or the completion id)
- ``GET /v1/yunshu/requests/recent``  finished requests with phase timestamps (``offsets_ms``)
- ``GET /v1/yunshu/history``     server-side ring of throughput / counts / memory (charts survive reload)
- ``GET /v1/yunshu/memory``      the unified-memory ledger by owner, with OS pressure and swap
- ``GET /v1/yunshu/config``      effective settings and where each value came from (secrets masked)
- ``POST /v1/yunshu/warmup``     load a model, compile its kernels and (optionally) prefill a
                                 prompt so the first real request is warm; sets ``keep_alive``

Access follows the inference endpoints: open on a default local server, gated by
``YUNSHU_AUTH_TOKEN`` when one is set.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from yunshu_engine.request_tracker import get_request_tracker
from yunshu_engine.units import put_gb
from yunshu_engine.version import yunshu_version

from ..console_contracts import (
    BundleManifest,
    CacheClear,
    CacheView,
    DiagnosticsBundle,
    HistoryPage,
    ModelImpact,
    SpecAggregate,
)
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

        put_gb(out, "active", mx.get_active_memory())
        put_gb(out, "cache", mx.get_cache_memory())
        put_gb(out, "peak", mx.get_peak_memory())
    except Exception:
        logger.debug("mlx memory unavailable", exc_info=True)
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        put_gb(out, "total", total, 1)
        if "active_bytes" in out and total:
            out["pressure"] = round(out["active_bytes"] / total, 3)
    except (ValueError, OSError, AttributeError):
        pass
    return out


_WEIGHT_GB: dict[str, float | None] = {}


def _weights_gb(model_path: str | None) -> float | None:
    """On-disk safetensors size of a checkpoint in GB, stat'ed once per path."""
    if not model_path:
        return None
    if model_path not in _WEIGHT_GB:
        total = 0
        try:
            total = sum(
                f.stat().st_size for f in Path(model_path).rglob("*.safetensors")
            )
        except OSError:
            total = 0
        _WEIGHT_GB[model_path] = round(total / 1e9, 1) if total else None
    return _WEIGHT_GB[model_path]


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
                    "model": e.model_id,
                    "unload_in_flight_policy": "reject",
                    "type": e.model_type.name,
                    "loaded": e.is_loaded,
                    "loading": e.is_loading,
                    "pinned": e.is_pinned,
                    "size_gb": round(e.estimated_bytes / (1 << 30), 1),
                    "size_bytes": int(e.estimated_bytes),
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
                "model": get_display_model_id() or getattr(engine, "model_name", None),
                "unload_in_flight_policy": "unsupported",
                "type": type(engine).__name__,
                "loaded": bool(getattr(engine, "is_loaded", False)),
                "loading": False,
                "pinned": True,  # single-model mode never frees its model
                "size_gb": _weights_gb(getattr(engine, "_model_path", None)),
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
        row["model"] = getattr(info.gen, "model", None) or info.model
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


def _gpu_busy() -> dict[str, Any]:
    """Cumulative GPU-busy seconds and idle fraction per loaded engine (the idle fraction is
    since the runner was built; take two ``busy_seconds`` samples for a window). An engine that
    does not account busy time is absent, not zero."""
    from .monitoring import _collect_engines

    out: dict[str, Any] = {}
    try:
        for model_id, engine in _collect_engines(get_engine(), get_model_manager()):
            fn = getattr(engine, "busy_snapshot", None)
            snap = fn() if callable(fn) else None
            if snap:
                out[model_id] = snap
    except Exception:
        logger.debug("busy snapshot failed", exc_info=True)
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
        "last": registry.last(),
        "gpu": _gpu_busy(),
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


@router.get("/yunshu/requests/recent")
async def recent_requests(
    request: Request,
    limit: int = 100,
    model: str | None = None,
    since: float | None = None,
) -> dict:
    """Finished requests, newest first (the 512-entry ring), each with ``offsets_ms``: the
    phase timestamps (arrive, admit, first token, last token, done) in ms after arrival."""
    _check_permission(request, "can_infer")
    if not 1 <= limit <= 512:
        raise HTTPException(400, "limit must be between 1 and 512")
    rows = registry.recent_entries(limit=limit, model=model, since=since)
    return {"object": "list", "data": rows, "count": len(rows), "capacity": 512}


@router.get("/yunshu/history")
async def history(
    request: Request,
    since: float | None = None,
    step: float | None = Query(None, ge=0),
) -> dict:
    """Columnar samples (``t`` epoch seconds plus one array per field), oldest first. ``since``
    keeps rows newer than that epoch; ``step`` averages them into buckets of that many seconds."""
    _check_permission(request, "can_infer")
    from .. import history as _history

    sampler = _history.get()
    if sampler is None:
        return {
            "object": "yunshu.history",
            "enabled": False,
            "interval_s": None,
            "ring": {"capacity": 0, "rows": 0, "bytes": 0},
            "fields": list(_history.FIELDS),
            "series": {"t": [], **{f: [] for f in _history.FIELDS}},
        }
    return {"enabled": True, **sampler.payload(since, step)}


@router.get("/yunshu/memory")
async def memory_ledger(request: Request) -> dict:
    """Unified memory by owner (weights, prefix cache, MLX cache, residual), peak, limits and
    OS pressure / swap. Every figure comes from a counter; unknown is null."""
    _check_permission(request, "can_infer")
    from .. import memory_ledger as _ledger

    return _ledger.collect(get_model_manager(), get_engine(), get_display_model_id())


@router.get("/yunshu/config")
async def effective_config(
    request: Request, include: Literal["stable", "all"] = "stable"
) -> dict:
    """Every setting with its effective value, default and source (cli / env / file /
    default), as ``yunshu config`` shows them. Secrets are masked."""
    _check_permission(request, "can_infer")
    from yunshu_engine import settings as _settings

    levels = (
        ("stable",) if include == "stable" else ("stable", "experimental", "internal")
    )
    rows = _settings.effective(levels)
    for r in rows:
        r["description"] = (r["description"] or "")[:240]
        if _settings.REGISTRY[r["name"]].secret:
            r["default"] = None if r["default"] is None else "***"
    try:
        warnings = _settings.validate(warn=False)
    except _settings.SettingError as exc:
        warnings = [str(exc)]
    experimental = [
        s for s in _settings.REGISTRY.values() if s.stability == "experimental"
    ]
    return {
        "object": "yunshu.config",
        "include": include,
        "settings": rows,
        "warnings": warnings,
        "experimental_count": len(experimental),
        "experimental_max": _settings.MAX_EXPERIMENTAL,
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
    row["model"] = getattr(info.gen, "model", None) or info.model
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


@router.get("/yunshu/host")
async def host(request: Request) -> dict:
    """CPU-only thermal, power, OS memory pressure and swap; cached for 15 seconds.

    Unsupported probes return state=unknown with a reason. No root or MLX required.
    """
    import asyncio

    from ..host_state import snapshot

    _check_permission(request, "can_manage_models")
    return await asyncio.to_thread(snapshot)


class RegisterLocalRequest(BaseModel):
    model: str
    path: str


@router.post("/yunshu/models/register")
async def register_local(req: RegisterLocalRequest, request: Request) -> dict:
    """Register a local checkpoint or HF snapshot directory without loading or copying weights."""
    import asyncio
    import json
    from pathlib import Path

    from yunshu_cli.model import weights_complete

    from ..ollama_models import manager_for, model_link, registered_entry

    manager = manager_for(request, "can_load_models")
    model_link(req.model)  # validate the identifier; no filesystem mutation
    if registered_entry(manager, req.model):
        raise HTTPException(409, "Model already registered")

    def validate():
        path = Path(req.path).expanduser().resolve(strict=True)
        config = json.loads((path / "config.json").read_text())
        if (
            not isinstance(config, dict)
            or not isinstance(config.get("model_type"), str)
            or not config["model_type"].strip()
        ):
            raise ValueError("config.json requires a model_type")
        weight_files = list(path.glob("*.safetensors"))
        index = path / "model.safetensors.index.json"
        if index.exists():
            index_config = json.loads(index.read_text())
            weight_map = (
                index_config.get("weight_map")
                if isinstance(index_config, dict)
                else None
            )
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("weights index requires a nonempty weight_map")
            for shard in weight_map.values():
                if (
                    not isinstance(shard, str)
                    or not shard.endswith(".safetensors")
                    or Path(shard).is_absolute()
                    or ".." in Path(shard).parts
                    or not (path / shard).is_file()
                ):
                    raise ValueError("invalid or missing weights index shard")
            # HF snapshot shards legitimately point outside snapshots/ into blobs/.
            weight_files = [path / name for name in set(weight_map.values())]
        import importlib.metadata
        import re

        from yunshu_engine.model_manager import _MODEL_TYPE_REMAP

        kind = config["model_type"].lower().replace("-", "_")
        kind = _MODEL_TYPE_REMAP.get(kind, kind)
        if not re.fullmatch(r"[a-z0-9_]+", kind):
            raise ValueError("invalid model_type")
        supported = False
        for package, prefix in (
            ("mlx-lm", "mlx_lm/models"),
            ("mlx-vlm", "mlx_vlm/models"),
            ("mlx-audio", "mlx_audio"),
        ):
            try:
                files = importlib.metadata.distribution(package).files or []
            except importlib.metadata.PackageNotFoundError:
                continue
            if any(
                str(f) == f"{prefix}/{kind}.py"
                or str(f).startswith(f"{prefix}/{kind}/")
                or (package == "mlx-audio" and f"/models/{kind}/" in str(f))
                for f in files
            ):
                supported = True
                break
        if not supported:
            raise ValueError("model_type has no installed MLX implementation")
        complete, reason = weights_complete(path)
        if not complete:
            raise ValueError(reason)
        size = sum(f.stat().st_size for f in weight_files)
        return path, size

    try:
        path, size = await asyncio.to_thread(validate)
        if registered_entry(manager, req.model):
            raise HTTPException(409, "Model already registered")
        manager.register_model(req.model, str(path), estimated_bytes=int(size * 1.8))
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from None
    entry = manager.get_entry(req.model)
    return {
        "object": "yunshu.model",
        "id": req.model,
        "path": str(path),
        "type": entry.model_type.name,
        "loaded": False,
    }


@router.delete("/yunshu/models/register/{model_id:path}")
async def unregister_local(model_id: str, request: Request) -> dict:
    """Remove an unloaded registration only; checkpoint files and HF cache are preserved."""
    from ..ollama_models import manager_for

    manager = manager_for(request, "can_unload_models")
    try:
        removed = manager.unregister_model(model_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    if not removed:
        raise HTTPException(404, "Model not registered")
    return {"object": "yunshu.model", "id": model_id, "status": "unregistered"}


class CancelModelRequest(BaseModel):
    model: str


@router.post("/yunshu/models/cancel")
async def cancel_model_operation(req: CancelModelRequest, request: Request) -> dict:
    """Cancel a pending load/download cooperatively; executing work is safely drained before cleanup."""
    from ..ollama_models import cancel_download, manager_for

    manager = manager_for(request, "can_load_models")
    loading = manager.cancel_load(req.model)
    downloading = cancel_download(req.model)
    if not loading and not downloading:
        raise HTTPException(404, "No in-progress load or download")
    return {
        "object": "yunshu.model",
        "id": req.model,
        "status": "cancelling",
        "load": loading,
        "download": downloading,
    }


@router.get("/yunshu/requests/history", response_model=HistoryPage)
async def request_history(
    request: Request, limit: int = 50, before: str | None = None
) -> dict:
    """Metadata-only cursor pages over the opt-in, rotated serve log."""
    import asyncio

    from yunshu_engine import settings

    from ..serve_log import get_log

    _check_permission(request, "can_manage_models")
    if not 1 <= limit <= 512:
        raise HTTPException(400, "limit must be between 1 and 512")
    log = get_log()
    if log is None:
        return {
            "object": "list",
            "enabled": False,
            "data": [],
            "count": 0,
            "next_cursor": None,
        }
    try:
        return await asyncio.to_thread(
            log.history,
            limit=limit,
            before=before,
            retention_days=settings.get_int("YUNSHU_SERVE_LOG_RETENTION_DAYS") or 0,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None


@router.get("/yunshu/spec-decode", response_model=SpecAggregate)
async def speculative_metrics(request: Request) -> dict:
    """VLM MTP/DFlash lifetime counters, including accepted/drafted per depth."""
    _check_permission(request, "can_manage_models")
    from yunshu_engine.spec_metrics import snapshot

    return snapshot()


def bundle_manifest() -> dict:
    return {
        "object": "yunshu.bundle.manifest",
        "format": "application/json",
        "included": [
            "created",
            "version",
            "platform",
            "packages",
            "settings_changed",
            "paths",
            "doctor",
            "caches",
            "errors",
            "excluded",
        ],
        "redacted": [
            "credentials",
            "opaque_mcp_configuration",
            "log_payload_fragments",
        ],
        "excluded": ["prompts", "completions", "request_bodies", "model_weights"],
        "error_line_limit": 100,
        "error_line_max_chars": 300,
        "generator": "yunshu_cli.bundle.build",
    }


@router.get("/yunshu/bundle/manifest", response_model=BundleManifest)
async def support_bundle_manifest(request: Request) -> dict:
    """Describe the same local diagnostics JSON produced by the CLI."""
    _check_permission(request, "can_manage_models")
    return bundle_manifest()


async def _cache_data() -> dict:
    import asyncio

    from yunshu_engine.mlx_executor import get_mlx_executor

    from .monitoring import _collect_engines

    def collect() -> dict:
        entries: list[dict] = []
        events: list[dict] = []
        models: list[str] = []
        for model, engine in _collect_engines(get_engine(), get_model_manager()):
            apc = getattr(engine, "_apc_backend", None)
            fn = getattr(apc, "console_snapshot", None)
            if callable(fn):
                snap = fn()
                models.append(model)
                entries.extend({"model": model, **row} for row in snap["entries"])
                events.extend({"model": model, **row} for row in snap["events"])
        return {
            "object": "yunshu.cache",
            "data": entries,
            "count": len(entries),
            "events": sorted(events, key=lambda row: row["t"], reverse=True)[:512],
            "event_capacity": 512,
            "models": models,
            "scope": "apc",
        }

    return await asyncio.get_running_loop().run_in_executor(get_mlx_executor(), collect)


@router.get("/yunshu/models/impact", response_model=ModelImpact)
async def model_impact(request: Request, model: str) -> dict:
    """Advisory load eviction preview and actual non-forced unload policy."""
    _check_permission(request, "can_manage_models")
    manager = get_model_manager()
    if manager is None:
        raise HTTPException(409, "Model management requires multi-model mode")
    try:
        return manager.console_impact(model)
    except KeyError:
        raise HTTPException(404, "Model not found") from None


@router.get("/yunshu/bundle", response_model=DiagnosticsBundle)
async def support_bundle(request: Request):
    """Download the CLI diagnostics bundle; nothing is uploaded."""
    import asyncio

    from fastapi.responses import JSONResponse

    from yunshu_cli.bundle import build

    _check_permission(request, "can_manage_models")
    host, port = request.scope.get("server") or ("127.0.0.1", 8000)
    data = await asyncio.to_thread(build, host=host, port=port)
    return JSONResponse(
        data,
        headers={
            "Content-Disposition": 'attachment; filename="yunshu-diagnostics.json"',
            "Cache-Control": "no-store",
        },
    )


@router.get("/yunshu/cache", response_model=CacheView)
async def cache_entries(request: Request) -> dict:
    """APC entries and bounded lifecycle events; no prompt tokens are exposed."""
    _check_permission(request, "can_manage_models")
    return await _cache_data()


@router.post("/yunshu/cache/clear", response_model=CacheClear)
async def clear_cache(request: Request) -> dict:
    """Clear idle loaded models' resident APC; persistent SSD files remain usable."""
    import asyncio

    from yunshu_engine.mlx_executor import get_mlx_executor

    from .monitoring import _collect_engines

    _check_permission(request, "can_manage_models")
    engines = list(_collect_engines(get_engine(), get_model_manager()))
    from yunshu_engine.request_tracker import current_request_id

    request_id = current_request_id.get()

    def clear():
        # Recheck on the same executor as generation before touching the pool.
        for model, engine in engines:
            active = getattr(engine, "has_active_requests", None)
            if not callable(active) or active():
                raise HTTPException(409, f"Model {model} cannot be proven idle")
        cleared = []
        for model, engine in engines:
            apc = getattr(engine, "_apc_backend", None)
            if apc is not None:
                # Warm inflight work must settle before removal, otherwise it can
                # repopulate the resident cache after a successful clear.
                warm = getattr(apc, "warm", None)
                if warm is not None:
                    warm.clear()
                observation = getattr(apc, "observation", None)
                if observation is not None:
                    observation.request_id = request_id
                apc.clear()
                if observation is not None:
                    observation.request_id = None
                cleared.append(model)
        return {
            "object": "yunshu.cache.clear",
            "models": cleared,
            "scope": "resident_apc",
            "persistent_cleared": False,
        }

    return await asyncio.get_running_loop().run_in_executor(get_mlx_executor(), clear)
