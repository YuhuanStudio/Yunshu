from __future__ import annotations
"""Sleep/wake endpoints for power saving and resource management.

3-level sleep (vLLM pattern):
- L0 (pause): Stop accepting requests, keep model + KV loaded
- L1 (unload): Unload model weights, keep KV cache
- L2 (deep): Unload everything — minimum memory footprint
"""
import asyncio
import logging
import os
import threading

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..engine import get_engine, get_model_manager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["sleep"])


class SleepRequest(BaseModel):
    level: int = 0  # 0=pause, 1=unload weights, 2=deep sleep


# ── Sleep state ──
_sleeping = False
_sleep_level = -1  # -1 = awake
_sleep_transitioning = False  # True during sleep/wake transition
_sleep_lock = threading.Lock()
_saved_model_name: str | None = None  # Model name saved before L1/L2 unload


@router.post("/sleep")
async def sleep_server(req: SleepRequest, request: Request):
    """Put server to sleep at the specified level."""
    global _sleeping, _sleep_level, _sleep_transitioning, _saved_model_name

    with _sleep_lock:
        if _sleeping:
            raise HTTPException(status_code=409, detail=f"Already sleeping at level {_sleep_level}")
        if _sleep_transitioning:
            raise HTTPException(status_code=409, detail="Sleep transition in progress")
        _sleep_transitioning = True

    try:
        level = max(0, min(2, req.level))
        engine = get_engine()
        manager = get_model_manager()

        if level >= 1 and engine:
            # Save model name before unloading so wake-up can restore it
            _saved_model_name = (
                getattr(engine, '_model_name', None)
                or getattr(engine, 'model_name', None)
                or os.environ.get("YUNSHU_MODEL")
            )

        # Set sleep level BEFORE any fallible operations so the state
        # is always consistent even if engine.stop() raises.
        _sleep_level = level
        _sleeping = True
        os.environ["YUNSHU_SLEEPING"] = "1"
        logger.info("Server entering L%d sleep", level)

        if level >= 1 and engine:
            # L1: unload model weights but keep KV cache
            # Safety: refuse L1 sleep if there are active requests that
            # would crash when engine._model is set to None mid-generation.
            # The sleep_guard middleware will reject new requests, but we
            # must also check for in-flight requests here.
            import yunshu_gateway.main as _main
            if _main._active_requests > 0:
                # Roll back sleep state
                _sleeping = False
                _sleep_level = -1
                os.environ.pop("YUNSHU_SLEEPING", None)
                raise HTTPException(
                    status_code=409,
                    detail=f"Cannot sleep: {_main._active_requests} active requests in progress. Wait for them to complete or use force shutdown.",
                )
            if engine.is_loaded:
                # Release model weights from GPU
                if hasattr(engine, '_model'):
                    engine._model = None
                if hasattr(engine, '_loaded'):
                    engine._loaded = False
                if hasattr(engine, '_running'):
                    engine._running = False
                logger.info("L1 sleep: model weights unloaded, KV cache retained")

        if level >= 2:
            # L2: unload everything
            if engine:
                await engine.stop()
            if manager:
                for entry in manager.list_entries():
                    if entry.is_loaded and entry.engine:
                        try:
                            await entry.engine.stop()
                        except Exception:
                            logger.debug("failed", exc_info=True)
            logger.info("L2 sleep: all models and caches released")

        return {"status": "sleeping", "level": level}
    except Exception:
        # Roll back sleep state on error so the server isn't stuck
        _sleeping = False
        _sleep_level = -1
        os.environ.pop("YUNSHU_SLEEPING", None)
        raise
    finally:
        _sleep_transitioning = False


@router.post("/wake-up")
async def wake_up_server(request: Request):
    """Wake up server from sleep, reload model if needed."""
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

        return {"status": "awake", "previous_level": level}
    except Exception:
        # If wake-up fails, clear transitioning flag but leave sleeping=True
        # so the server is still marked as sleeping (it wasn't fully woken up).
        _sleep_transitioning = False
        raise
    finally:
        _sleep_transitioning = False


@router.get("/sleep/status")
async def sleep_status(request: Request):
    """Check current sleep state."""
    return {
        "sleeping": _sleeping,
        "level": _sleep_level if _sleeping else -1,
        "transitioning": _sleep_transitioning,
    }


def is_sleeping() -> bool:
    """Check if server is in sleep mode (used by request middleware)."""
    return _sleeping
