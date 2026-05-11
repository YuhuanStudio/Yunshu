"""OpenAI Images API compatible router — image generation endpoint."""

import base64
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from ..engine import get_model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["images"])


class ImageGenerateRequest(BaseModel):
    prompt: str
    model: str = "Z-Image-Turbo-MLX-4bit"
    n: int = 1
    size: str = "1024x1024"  # WxH format
    response_format: str = "b64_json"  # b64_json or url
    negative_prompt: str = ""
    num_inference_steps: int = 4
    guidance_scale: float = 3.5
    seed: Optional[int] = None


@router.post("/images/generations")
async def create_image(req: ImageGenerateRequest) -> JSONResponse:
    """Generate images from text prompt (OpenAI /v1/images/generations compatible)."""
    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    from yunshu_engine.image_engine import ImageGenEngine

    # Find image gen engine
    img_engine = None
    for entry in manager.list_entries():
        if entry.is_loaded and isinstance(getattr(entry, 'engine', None), ImageGenEngine):
            if req.model in {entry.model_id, entry.model_id.lower()}:
                img_engine = entry.engine
                break

    if img_engine is None:
        # Try loading by model name
        try:
            engine = await manager.get_engine(req.model)
            if isinstance(engine, ImageGenEngine):
                img_engine = engine
        except (KeyError, Exception):
            pass

    if img_engine is None:
        # Find any loaded ImageGenEngine
        for entry in manager.list_entries():
            if entry.is_loaded and isinstance(getattr(entry, 'engine', None), ImageGenEngine):
                img_engine = entry.engine
                break

    if img_engine is None:
        raise HTTPException(
            status_code=404,
            detail=f"Image generation model '{req.model}' not found",
        )

    # Parse size
    try:
        width, height = map(int, req.size.split("x"))
    except (ValueError, AttributeError):
        width, height = 1024, 1024

    images_data = []
    for i in range(req.n):
        try:
            png_bytes = await img_engine.generate_image(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=width,
                height=height,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                seed=(req.seed + i) if req.seed is not None else None,
            )
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        if req.response_format == "b64_json":
            b64 = base64.b64encode(png_bytes).decode("ascii")
            images_data.append({
                "b64_json": b64,
            })
        else:
            images_data.append({
                "url": f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}",
            })

    return JSONResponse({
        "created": int(time.time()),
        "data": images_data,
    })


@router.post("/images/generations/stream")
async def stream_image_generation(req: ImageGenerateRequest):
    """Stream image generation progress as SSE events."""
    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    from yunshu_engine.image_engine import ImageGenEngine

    img_engine = None
    for entry in manager.list_entries():
        if entry.is_loaded and isinstance(getattr(entry, 'engine', None), ImageGenEngine):
            img_engine = entry.engine
            break

    if img_engine is None:
        raise HTTPException(status_code=404, detail="No image generation engine available")

    async def _progress_stream():
        async for chunk in img_engine.generate_image_stream(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=int(req.size.split("x")[0]) if "x" in req.size else 1024,
            height=int(req.size.split("x")[1]) if "x" in req.size else 1024,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
        ):
            if chunk.get("is_final") and chunk.get("image"):
                b64 = base64.b64encode(chunk["image"]).decode("ascii")
                yield f"data: {json.dumps({'step': chunk['step'], 'progress': 1.0, 'image': b64, 'is_final': True})}\n\n"
            elif not chunk.get("is_final"):
                yield f"data: {json.dumps({'step': chunk['step'], 'total_steps': chunk['total_steps'], 'progress': chunk['progress'], 'is_final': False})}\n\n"

    return StreamingResponse(
        _progress_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
