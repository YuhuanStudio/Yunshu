from __future__ import annotations

"""Profiling endpoints for Metal performance analysis.

Provides /v1/start_profile and /v1/stop_profile endpoints for
MLX Metal GPU command buffer tracing.

Security: All profiling endpoints require authentication (deny-by-default).
Set YUNSHU_AUTH_TOKEN or YUNSHU_AUTH_DISABLED=true for access.
"""
import hmac
import logging
import os
import threading
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["profiling"])

from yunshu_control.audit_log import log_operation, resolve_actor

_profiling_active = False
_profile_start_time = 0.0
_profiling_lock = threading.Lock()


def _check_permission(request: Request) -> None:
    """Check auth on profiling endpoints (deny-by-default).

    Profiling controls are admin-sensitive: they can cause performance
    degradation and the output_path write is filesystem-sensitive.
    """
    if os.environ.get("YUNSHU_AUTH_DISABLED", "").lower() in ("true", "1", "yes"):
        return
    auth_token = os.environ.get("YUNSHU_AUTH_TOKEN")
    if auth_token:
        # Verify the request actually provides a valid Bearer token
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and hmac.compare_digest(auth[7:], auth_token):
            return
    raise HTTPException(
        status_code=401,
        detail="Profiling requires authentication. Set YUNSHU_AUTH_TOKEN or YUNSHU_AUTH_DISABLED=true.",
    )


class ProfileRequest(BaseModel):
    duration_seconds: float | None = None
    output_path: str | None = None


@router.post("/start_profile", response_model=None)
async def start_profile(req: ProfileRequest, request: Request):
    """Start Metal performance profiling capture."""
    actor = resolve_actor(request)
    _check_permission(request)
    global _profiling_active, _profile_start_time

    with _profiling_lock:
        if _profiling_active:
            log_operation(
                "profiling_start",
                "metal_capture",
                "failure",
                actor=actor,
                detail="already_active",
            )
            raise HTTPException(status_code=409, detail="Profiling already active")

        try:
            import mlx.core as mx

            if hasattr(mx.metal, "start_capture"):
                from pathlib import Path

                _profile_dir = Path("/tmp/yunshu_profiles")
                _profile_dir.mkdir(parents=True, exist_ok=True)
                # Apple's mx.metal.start_capture refuses file names that
                # aren't .gputrace (the Instruments-bundle extension). Default
                # to a timestamped .gputrace under the profile dir.
                output = req.output_path or str(
                    _profile_dir / f"yunshu_{int(time.time())}.gputrace"
                )
                # auto-append `.gputrace` if missing — Apple's
                # start_capture refuses any other extension with an opaque
                # error. Make the API forgiving instead of returning 500.
                if not output.endswith(".gputrace"):
                    output = output + ".gputrace"
                # Validate output_path is under the dedicated profile directory
                resolved = Path(output).resolve()
                if not str(resolved).startswith(str(_profile_dir.resolve()) + "/"):
                    log_operation(
                        "profiling_start",
                        "metal_capture",
                        "failure",
                        actor=actor,
                        detail="invalid_output_path",
                    )
                    raise HTTPException(
                        status_code=400,
                        detail=f"output_path must be under {_profile_dir}, got: {output}",
                    )
                # surface MTL_CAPTURE_ENABLED requirement as a
                # clear actionable 400 instead of letting the raw MLX error
                # bubble as 500.
                import os as _os

                if not _os.environ.get("MTL_CAPTURE_ENABLED"):
                    log_operation(
                        "profiling_start",
                        "metal_capture",
                        "failure",
                        actor=actor,
                        detail="MTL_CAPTURE_ENABLED unset",
                    )
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Metal capture requires environment variable "
                            "MTL_CAPTURE_ENABLED=1 to be set before launching the gateway. "
                            "Restart with: MTL_CAPTURE_ENABLED=1 uv run uvicorn yunshu_gateway.main:app ..."
                        ),
                    )
                mx.metal.start_capture(output)
                _profiling_active = True
                _profile_start_time = time.perf_counter()
                log_operation(
                    "profiling_start",
                    "metal_capture",
                    "success",
                    actor=actor,
                    output=output,
                )
                return JSONResponse(
                    {
                        "status": "started",
                        "output_path": output,
                        "duration_seconds": req.duration_seconds,
                    }
                )
            else:
                log_operation(
                    "profiling_start",
                    "metal_capture",
                    "failure",
                    actor=actor,
                    detail="not_available",
                )
                raise HTTPException(
                    status_code=501, detail="Metal profiling not available"
                )
        except HTTPException:
            raise
        except Exception as e:
            log_operation(
                "profiling_start",
                "metal_capture",
                "failure",
                actor=actor,
                detail=str(e)[:120],
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to start profiling: {e}"
            ) from None


@router.post("/stop_profile", response_model=None)
async def stop_profile(request: Request):
    """Stop Metal performance profiling capture."""
    actor = resolve_actor(request)
    _check_permission(request)
    global _profiling_active

    with _profiling_lock:
        if not _profiling_active:
            log_operation(
                "profiling_stop",
                "metal_capture",
                "failure",
                actor=actor,
                detail="not_active",
            )
            raise HTTPException(status_code=409, detail="No active profiling session")

        try:
            import mlx.core as mx

            mx.metal.stop_capture()
            elapsed = time.perf_counter() - _profile_start_time
            _profiling_active = False
            log_operation(
                "profiling_stop",
                "metal_capture",
                "success",
                actor=actor,
                elapsed=f"{elapsed:.3f}",
            )
            return JSONResponse(
                {
                    "status": "stopped",
                    "elapsed_seconds": round(elapsed, 3),
                }
            )
        except Exception as e:
            _profiling_active = False
            log_operation(
                "profiling_stop",
                "metal_capture",
                "failure",
                actor=actor,
                detail=str(e)[:120],
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to stop profiling: {e}"
            ) from None


@router.get("/profile/status", response_model=None)
async def profile_status(request: Request):
    """Get profiling status."""
    _check_permission(request)
    with _profiling_lock:
        active = _profiling_active
        start = _profile_start_time
        elapsed = time.perf_counter() - start if active else 0
    return JSONResponse(
        {
            "active": active,
            "elapsed_seconds": round(elapsed, 3) if active else None,
        }
    )


@router.get("/profile/engine", response_model=None)
async def engine_profiling_stats(request: Request):
    """Get engine-level profiling stats from PerformanceProfiler + ProfilingMixin."""
    _check_permission(request)
    from yunshu_engine.batched_engine import BatchedEngine

    from ..engine import get_engine, get_model_manager

    results = []
    manager = get_model_manager()
    engines = []

    if manager is not None:
        for entry in manager.list_entries():
            if entry.is_loaded and isinstance(
                getattr(entry, "engine", None), BatchedEngine
            ):
                engines.append((entry.model_id, entry.engine))
    else:
        engine = get_engine()
        if engine and isinstance(engine, BatchedEngine):
            engines.append((engine.model_name, engine))

    for model_id, engine in engines:
        info = {"model_id": model_id}
        stats: dict = getattr(engine, "get_stats", lambda: {})()
        if "profiler" in stats:
            info["profiler"] = stats["profiler"]
        if "auto_tuner" in stats:
            info["auto_tuner"] = stats["auto_tuner"]
        if "slo" in stats:
            info["slo"] = stats["slo"]

        # Check for CompositionScheduler profiling mixin
        core = getattr(engine, "_engine_core", None)
        if core is not None:
            cs = getattr(core, "_composition_scheduler", None)
            if cs is not None:
                mixin_stats = cs.get_stats()
                if "ProfilingMixin" in mixin_stats:
                    info["scheduler_profiling"] = mixin_stats["ProfilingMixin"]
        results.append(info)

    return JSONResponse({"engines": results})


# ── profile-state introspection for other modules ──


def is_profile_capture_active() -> bool:
    """Return True if Metal capture is currently active.

    Inference requests during active GPU
    capture can stall (HTTP 000 — capture-driven I/O serializes Metal
    work and blocks the event loop). Other modules can check this and
    return a 503 or queue the request explicitly instead of hanging.
    """
    return _profiling_active
