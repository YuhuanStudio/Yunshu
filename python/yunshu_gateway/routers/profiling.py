"""Profiling endpoints for Metal performance analysis.

Provides /v1/start_profile and /v1/stop_profile endpoints for
MLX Metal GPU command buffer tracing (vLLM pattern).
"""
import logging
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(tags=["profiling"])

_profiling_active = False
_profile_start_time = 0.0


class ProfileRequest(BaseModel):
    duration_seconds: Optional[float] = None
    output_path: Optional[str] = None


@router.post("/start_profile", response_model=None)
async def start_profile(req: ProfileRequest):
    """Start Metal performance profiling capture."""
    global _profiling_active, _profile_start_time

    if _profiling_active:
        raise HTTPException(status_code=409, detail="Profiling already active")

    try:
        import mlx.core as mx
        if hasattr(mx.metal, 'start_capture'):
            output = req.output_path or "/tmp/yunshu_profile.bin"
            mx.metal.start_capture(output)
            _profiling_active = True
            _profile_start_time = time.perf_counter()
            return JSONResponse({
                "status": "started",
                "output_path": output,
                "duration_seconds": req.duration_seconds,
            })
        else:
            raise HTTPException(status_code=501, detail="Metal profiling not available")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start profiling: {e}")


@router.post("/stop_profile", response_model=None)
async def stop_profile():
    """Stop Metal performance profiling capture."""
    global _profiling_active

    if not _profiling_active:
        raise HTTPException(status_code=409, detail="No active profiling session")

    try:
        import mlx.core as mx
        mx.metal.stop_capture()
        elapsed = time.perf_counter() - _profile_start_time
        _profiling_active = False
        return JSONResponse({
            "status": "stopped",
            "elapsed_seconds": round(elapsed, 3),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop profiling: {e}")


@router.get("/profile/status", response_model=None)
async def profile_status():
    """Get profiling status."""
    elapsed = time.perf_counter() - _profile_start_time if _profiling_active else 0
    return JSONResponse({
        "active": _profiling_active,
        "elapsed_seconds": round(elapsed, 3) if _profiling_active else None,
    })
