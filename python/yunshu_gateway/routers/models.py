"""OpenAI Models API compatible router."""

import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
from pydantic import BaseModel

from ..engine import get_engine, get_model_manager

router = APIRouter(tags=["models"])


class LoadModelRequest(BaseModel):
    model: str
    pin: bool = False


@router.get("/models")
async def list_models() -> dict:
    """List available models (OpenAI-compatible)."""
    models = []

    # Multi-model mode
    manager = get_model_manager()
    if manager is not None:
        for entry in manager.list_entries():
            model_info = {
                "id": entry.model_id,
                "object": "model",
                "created": 0,
                "owned_by": "yunshu",
                "loaded": entry.is_loaded,
                "type": entry.model_type.name,
                "size_gb": round(entry.estimated_bytes / 1e9, 1),
            }
            if entry.is_loaded and entry.engine is not None:
                try:
                    stats = entry.engine.get_stats() if hasattr(entry.engine, 'get_stats') else {}
                    model_info["stats"] = stats
                except Exception:
                    logger.debug(f"failed to get stats for {entry.model_id}", exc_info=True)
            models.append(model_info)
        return {"object": "list", "data": models}

    # Single-engine mode
    engine = get_engine()
    if engine and engine.is_loaded:
        models.append({
            "id": engine.model_name,
            "object": "model",
            "created": 0,
            "owned_by": "yunshu",
        })
    return {"object": "list", "data": models}


@router.get("/models/{model_id}")
async def get_model(model_id: str) -> dict:
    """Get details for a specific model."""
    manager = get_model_manager()
    if manager is not None:
        entry = manager.get_entry(model_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        return {
            "id": entry.model_id,
            "object": "model",
            "owned_by": "yunshu",
            "loaded": entry.is_loaded,
        }

    engine = get_engine()
    if not engine or not engine.resolve_model_id(model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return {
        "id": engine.model_name,
        "object": "model",
        "owned_by": "yunshu",
    }


@router.post("/models/load")
async def load_model(req: LoadModelRequest) -> dict:
    """Load a model (supports both single-engine and multi-model modes)."""
    manager = get_model_manager()

    if manager is not None:
        try:
            engine = await manager.get_engine(req.model)
            if hasattr(engine, 'is_running') and not engine.is_running:
                await engine.start()
            return {"status": "loaded", "model": req.model}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Model '{req.model}' not registered")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Single-engine mode
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    import asyncio
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, engine.load, req.model)
    await engine.start()

    return {"status": "loaded", "model": req.model}


@router.post("/models/unload/{model_id}")
async def unload_model(model_id: str) -> dict:
    """Unload a model and release memory."""
    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=400, detail="Multi-model mode not active")

    entry = manager.get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    await manager.unload_model(model_id)
    return {"status": "unloaded", "model": model_id}
