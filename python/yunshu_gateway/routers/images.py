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
    size: str = "1024x1024"
    response_format: str = "b64_json"
    num_inference_steps: int = 4
    seed: Optional[int] = None
    preview_interval: int = 0  # Decode & emit intermediate preview every N steps (0=off)
    # Note: guidance_scale and negative_prompt removed — Turbo models
    # don't support classifier-free guidance. These params were accepted
    # but silently ignored by the engine.


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
            logger.debug(f"failed to load image engine for {req.model}", exc_info=True)

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

    # Validate dimensions: multiples of 64, within 64–2048
    if width < 64 or width > 2048 or height < 64 or height > 2048:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions must be between 64 and 2048, got {width}x{height}",
        )
    if width % 64 != 0 or height % 64 != 0:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions must be multiples of 64, got {width}x{height}",
        )

    images_data = []
    for i in range(req.n):
        try:
            png_bytes = await img_engine.generate_image(
                prompt=req.prompt,
                width=width,
                height=height,
                num_inference_steps=req.num_inference_steps,
                seed=(req.seed + i) if req.seed is not None else None,
            )
        except Exception as e:
            logger.error(f"Image generation error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Image generation failed")

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
            width=int(req.size.split("x")[0]) if "x" in req.size else 1024,
            height=int(req.size.split("x")[1]) if "x" in req.size else 1024,
            num_inference_steps=req.num_inference_steps,
            seed=req.seed,
            preview_interval=req.preview_interval,
        ):
            if chunk.get("is_final") and chunk.get("image"):
                b64 = base64.b64encode(chunk["image"]).decode("ascii")
                yield f"data: {json.dumps({'step': chunk['step'], 'progress': 1.0, 'image': b64, 'is_final': True})}\n\n"
            elif not chunk.get("is_final") and chunk.get("image"):
                b64 = base64.b64encode(chunk["image"]).decode("ascii")
                yield f"data: {json.dumps({'step': chunk['step'], 'total_steps': chunk['total_steps'], 'progress': chunk['progress'], 'image': b64, 'is_final': False, 'is_preview': True})}\n\n"
            elif not chunk.get("is_final"):
                yield f"data: {json.dumps({'step': chunk['step'], 'total_steps': chunk['total_steps'], 'progress': chunk['progress'], 'is_final': False})}\n\n"

    return StreamingResponse(
        _progress_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


class ImageVariationsRequest(BaseModel):
    image: str  # base64 encoded image
    model: str = "Z-Image-Turbo-MLX-4bit"
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    num_inference_steps: int = 4
    seed: Optional[int] = None


@router.post("/images/variations")
async def create_image_variation(req: ImageVariationsRequest) -> JSONResponse:
    """Generate variations of an input image (OpenAI /v1/images/variations compatible).

    Uses the input image as a conditioning signal for the diffusion model.
    The image is decoded and used as a starting point for the generation.
    """
    import base64

    try:
        image_bytes = base64.b64decode(req.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

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
        raise HTTPException(status_code=404, detail="No image generation model available")

    try:
        # Generate variation using the input image as conditioning
        images = []
        for i in range(req.n):
            seed = (req.seed + i) if req.seed is not None else None
            result = await img_engine.generate(
                prompt="",  # Variation: no text prompt, image-conditioned
                num_inference_steps=req.num_inference_steps,
                seed=seed,
                image=image_bytes,
            )
            if isinstance(result, list):
                images.extend(result)
            else:
                images.append(result)

        data = []
        for img in images:
            b64 = base64.b64encode(img).decode("ascii")
            data.append({"b64_json": b64})

        return JSONResponse({
            "created": int(time.time()),
            "data": data,
        })
    except Exception as e:
        logger.error(f"Image variation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Image variation failed")


class ImageEditsRequest(BaseModel):
    image: str  # base64 encoded source image
    prompt: str  # edit instruction
    model: str = "Z-Image-Turbo-MLX-4bit"
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    num_inference_steps: int = 4
    seed: Optional[int] = None


@router.post("/images/edits")
async def create_image_edit(req: ImageEditsRequest) -> JSONResponse:
    """Edit an image based on a text prompt (OpenAI /v1/images/edits compatible).

    Combines the input image with a text prompt to generate an edited version.
    """
    import base64

    try:
        image_bytes = base64.b64decode(req.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

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
        raise HTTPException(status_code=404, detail="No image generation model available")

    try:
        images = []
        for i in range(req.n):
            seed = (req.seed + i) if req.seed is not None else None
            result = await img_engine.generate(
                prompt=req.prompt,
                num_inference_steps=req.num_inference_steps,
                seed=seed,
                image=image_bytes,
            )
            if isinstance(result, list):
                images.extend(result)
            else:
                images.append(result)

        data = []
        for img in images:
            b64 = base64.b64encode(img).decode("ascii")
            data.append({"b64_json": b64})

        return JSONResponse({
            "created": int(time.time()),
            "data": data,
        })
    except Exception as e:
        logger.error(f"Image edit error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Image edit failed")

