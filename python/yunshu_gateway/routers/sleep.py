"""Sleep/wake endpoints for power saving and resource management.

3-level sleep (vLLM pattern):
- L0 (pause): Stop accepting requests, keep model + KV loaded
- L1 (unload): Unload model weights, keep KV cache
- L2 (deep): Unload everything — minimum memory footprint
"""
import logging
import os

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


@router.post("/sleep")
async def sleep_server(req: SleepRequest, request: Request):
    """Put server to sleep at the specified level."""
    global _sleeping, _sleep_level

    if _sleeping:
        raise HTTPException(status_code=409, detail=f"Already sleeping at level {_sleep_level}")

    level = max(0, min(2, req.level))
    engine = get_engine()
    manager = get_model_manager()

    if level >= 0:
        # L0: pause — stop accepting new requests
        _sleeping = True
        os.environ["YUNSHU_SLEEPING"] = "1"
        logger.info(f"Server entering L0 sleep (pause)")

    if level >= 1 and engine:
        # L1: unload model weights but keep KV cache
        if engine.is_loaded:
            # Release model weights from GPU
            if hasattr(engine, '_model'):
                engine._model = None
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

    _sleep_level = level
    return {"status": "sleeping", "level": level}


@router.post("/wake-up")
async def wake_up_server(request: Request):
    """Wake up server from sleep, reload model if needed."""
    global _sleeping, _sleep_level

    if not _sleeping:
        return {"status": "awake", "message": "Server is already awake"}

    level = _sleep_level
    default_model = os.environ.get("YUNSHU_MODEL")

    if level >= 1 and default_model:
        # Reload the model
        from yunshu_engine.batched_engine import BatchedEngine
        from ..engine import set_engine

        engine = BatchedEngine(model_name=default_model)
        set_engine(engine)
        await engine.start()
        logger.info("Woke up: model reloaded")

    _sleeping = False
    _sleep_level = -1
    os.environ.pop("YUNSHU_SLEEPING", None)

    return {"status": "awake", "previous_level": level}


@router.get("/sleep/status")
async def sleep_status(request: Request):
    """Check current sleep state."""
    return {
        "sleeping": _sleeping,
        "level": _sleep_level if _sleeping else -1,
    }


def is_sleeping() -> bool:
    """Check if server is in sleep mode (used by request middleware)."""
    return _sleeping
