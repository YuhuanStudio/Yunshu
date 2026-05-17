from __future__ import annotations
"""OpenAI Images API compatible router — image generation endpoint."""

import base64
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from ..engine import get_model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["images"])


class ImageGenerateRequest(BaseModel):
    prompt: str
    model: str = "Z-Image-Turbo-MLX-4bit"
    n: int = Field(default=1, ge=1, le=10)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    num_inference_steps: int = Field(default=4, ge=1, le=100)
    seed: Optional[int] = None
    preview_interval: int = 0  # Decode & emit intermediate preview every N steps (0=off)
    # Note: guidance_scale and negative_prompt removed — Turbo models
    # don't support classifier-free guidance. These params were accepted
    # but silently ignored by the engine.

    @model_validator(mode="after")
    def validate_request(self):
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt: field is required and cannot be empty")
        if self.response_format not in ("b64_json", "url"):
            raise ValueError(f"response_format: must be 'b64_json' or 'url', got '{self.response_format}'")
        return self


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
    try:
        for i in range(req.n):
            png_bytes = await img_engine.generate_image(
                prompt=req.prompt,
                width=width,
                height=height,
                num_inference_steps=req.num_inference_steps,
                seed=(req.seed + i) if req.seed is not None else None,
            )

            if req.response_format == "b64_json":
                b64 = base64.b64encode(png_bytes).decode("ascii")
                images_data.append({
                    "b64_json": b64,
                })
            else:
                images_data.append({
                    "url": f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}",
                })
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"Image generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Image generation failed")

    return JSONResponse({
        "created": int(time.time()),
        "data": images_data,
    })


@router.post("/images/generations/stream")
async def stream_image_generation(req: ImageGenerateRequest, request: Request):
    """Stream image generation progress as SSE events."""
    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    from yunshu_engine.image_engine import ImageGenEngine

    img_engine = None
    # Match by model name first (consistent with non-streaming path)
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
        # Fallback: first loaded ImageGenEngine
        for entry in manager.list_entries():
            if entry.is_loaded and isinstance(getattr(entry, 'engine', None), ImageGenEngine):
                img_engine = entry.engine
                break

    if img_engine is None:
        raise HTTPException(status_code=404, detail="No image generation engine available")

    # Parse and validate size
    try:
        width, height = map(int, req.size.split("x"))
    except (ValueError, AttributeError):
        width, height = 1024, 1024
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

    # Register with request tracker for cancellation support
    import uuid as _uuid
    _img_id = f"img-{_uuid.uuid4().hex[:24]}"
    from yunshu_engine.request_tracker import get_request_tracker
    _img_tracker = get_request_tracker()
    _img_gen = _img_tracker.register(_img_id, req.model)

    async def _progress_stream():
        async for chunk in img_engine.generate_image_stream(
            prompt=req.prompt,
            width=width,
            height=height,
            num_inference_steps=req.num_inference_steps,
            seed=req.seed,
            preview_interval=req.preview_interval,
        ):
            if _img_gen.cancel_event.is_set():
                yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                return
            if chunk.get("is_final") and chunk.get("image"):
                b64 = base64.b64encode(chunk["image"]).decode("ascii")
                yield f"data: {json.dumps({'step': chunk['step'], 'progress': 1.0, 'image': b64, 'is_final': True})}\n\n"
            elif not chunk.get("is_final") and chunk.get("image"):
                b64 = base64.b64encode(chunk["image"]).decode("ascii")
                yield f"data: {json.dumps({'step': chunk['step'], 'total_steps': chunk['total_steps'], 'progress': chunk['progress'], 'image': b64, 'is_final': False, 'is_preview': True})}\n\n"
            elif not chunk.get("is_final"):
                yield f"data: {json.dumps({'step': chunk['step'], 'total_steps': chunk['total_steps'], 'progress': chunk['progress'], 'is_final': False})}\n\n"

    from ..streaming import with_sse_keepalive

    async def _wrapped_stream():
        try:
            async for event in with_sse_keepalive(
                _progress_stream(),
                http_request=request,
                cancel_event=_img_gen.cancel_event,
            ):
                yield event.encode("utf-8") if isinstance(event, str) else event
        except MemoryError:
            yield f"data: {json.dumps({'error': {'message': 'Out of GPU memory', 'type': 'memory_error'}})}\n\n".encode("utf-8")
        except Exception as e:
            logger.error(f"Image streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': {'message': 'Image generation failed', 'type': 'server_error'}})}\n\n".encode("utf-8")
        finally:
            _img_tracker.unregister(_img_id)

    return StreamingResponse(
        _wrapped_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


class ImageVariationsRequest(BaseModel):
    image: str  # base64 encoded image
    model: str = "Z-Image-Turbo-MLX-4bit"
    n: int = Field(default=1, ge=1, le=10)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    num_inference_steps: int = Field(default=4, ge=1, le=100)
    seed: Optional[int] = None

    @model_validator(mode="after")
    def validate_request(self):
        if not self.image or not self.image.strip():
            raise ValueError("image: field is required and cannot be empty")
        return self


@router.post("/images/variations")
async def create_image_variation(req: ImageVariationsRequest) -> JSONResponse:
    """Generate variations of an input image (OpenAI /v1/images/variations compatible).

    Uses the input image as a conditioning signal for the diffusion model.
    The image is decoded and used as a starting point for the generation.
    """
    import base64

    try:
        image_bytes = base64.b64decode(req.image, validate=True)
    except Exception:
        logger.debug("invalid base64 image data in variations request", exc_info=True)
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
        width, height = map(int, req.size.split("x"))
    except (ValueError, AttributeError):
        width, height = 1024, 1024

    try:
        # Generate variation using the input image as conditioning
        images = []
        for i in range(req.n):
            seed = (req.seed + i) if req.seed is not None else None
            result = await img_engine.generate(
                prompt="",
                num_inference_steps=req.num_inference_steps,
                seed=seed,
                image=image_bytes,
                width=width,
                height=height,
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
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"Image variation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Image variation failed")


class ImageEditsRequest(BaseModel):
    image: str  # base64 encoded source image
    prompt: str  # edit instruction
    model: str = "Z-Image-Turbo-MLX-4bit"
    n: int = Field(default=1, ge=1, le=10)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    num_inference_steps: int = Field(default=4, ge=1, le=100)
    seed: Optional[int] = None

    @model_validator(mode="after")
    def validate_request(self):
        if not self.image or not self.image.strip():
            raise ValueError("image: field is required and cannot be empty")
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt: field is required and cannot be empty")
        return self


@router.post("/images/edits")
async def create_image_edit(req: ImageEditsRequest) -> JSONResponse:
    """Edit an image based on a text prompt (OpenAI /v1/images/edits compatible).

    Combines the input image with a text prompt to generate an edited version.
    """
    import base64

    try:
        image_bytes = base64.b64decode(req.image, validate=True)
    except Exception:
        logger.debug("invalid base64 image data in edits request", exc_info=True)
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
        width, height = map(int, req.size.split("x"))
    except (ValueError, AttributeError):
        width, height = 1024, 1024

    try:
        images = []
        for i in range(req.n):
            seed = (req.seed + i) if req.seed is not None else None
            result = await img_engine.generate(
                prompt=req.prompt,
                num_inference_steps=req.num_inference_steps,
                seed=seed,
                image=image_bytes,
                width=width,
                height=height,
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
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"Image edit error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Image edit failed")


class ImageInpaintRequest(BaseModel):
    image: str = Field(description="Base64-encoded source image (PNG/JPEG)")
    prompt: str = Field(description="Text description of what to fill in the masked region")
    mask: Optional[str] = Field(default=None, description="Base64-encoded mask image (white=fill, black=keep)")
    model: str = "Z-Image-Turbo-MLX-4bit"
    n: int = Field(default=1, ge=1, le=10)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    num_inference_steps: int = Field(default=4, ge=1, le=100)
    seed: Optional[int] = None
    denoise_strength: float = Field(default=1.0, ge=0.0, le=1.0, description="How much to re-denoise (1.0=full)")

    @model_validator(mode="after")
    def validate_request(self):
        if not self.image or not self.image.strip():
            raise ValueError("image: field is required and cannot be empty")
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt: field is required and cannot be empty")
        return self


@router.post("/images/inpaint")
async def create_image_inpaint(req: ImageInpaintRequest) -> JSONResponse:
    """Inpaint masked regions of an image using a text prompt.

    Accepts a source image and a mask (white=fill, black=preserve).
    The masked region is re-generated guided by the text prompt while
    the unmasked region is preserved from the original image.
    """
    try:
        image_bytes = base64.b64decode(req.image, validate=True)
    except Exception:
        logger.debug("invalid base64 image data in inpaint request", exc_info=True)
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
        width, height = map(int, req.size.split("x"))
    except (ValueError, AttributeError):
        width, height = 1024, 1024

    try:
        data = []
        for i in range(req.n):
            seed = (req.seed + i) if req.seed is not None else None
            png = await img_engine.inpaint(
                prompt=req.prompt,
                image=image_bytes,
                mask_base64=req.mask,
                width=width,
                height=height,
                num_inference_steps=req.num_inference_steps,
                seed=seed,
                denoise_strength=req.denoise_strength,
            )
            b64 = base64.b64encode(png).decode("ascii")
            if req.response_format == "b64_json":
                data.append({"b64_json": b64})
            else:
                data.append({"url": f"data:image/png;base64,{b64}"})

        return JSONResponse({
            "created": int(time.time()),
            "data": data,
        })
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"Image inpaint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Image inpainting failed")


class ImageControlNetRequest(BaseModel):
    prompt: str = Field(description="Text prompt for generation")
    image: str = Field(description="Base64-encoded conditioning image (edges, depth map, etc.)")
    condition_type: str = Field(default="canny", description="Conditioning type: canny, depth, raw")
    model: str = "Z-Image-Turbo-MLX-4bit"
    n: int = Field(default=1, ge=1, le=10)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    num_inference_steps: int = Field(default=4, ge=1, le=100)
    seed: Optional[int] = None
    controlnet_strength: float = Field(default=1.0, ge=0.0, le=2.0, description="Conditioning strength")
    canny_low: int = Field(default=100, ge=0, le=255, description="Canny lower threshold")
    canny_high: int = Field(default=200, ge=0, le=255, description="Canny upper threshold")

    @model_validator(mode="after")
    def validate_request(self):
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt: field is required and cannot be empty")
        if not self.image or not self.image.strip():
            raise ValueError("image: field is required and cannot be empty")
        if self.condition_type not in ("canny", "depth", "raw"):
            raise ValueError(f"condition_type: must be 'canny', 'depth', or 'raw', got '{self.condition_type}'")
        return self


@router.post("/images/controlnet")
async def create_image_controlnet(req: ImageControlNetRequest) -> JSONResponse:
    """Generate an image with ControlNet spatial conditioning.

    Accepts a conditioning image (edge map, depth map, etc.) and a text prompt.
    The conditioning image guides the spatial structure of the generated output.
    """
    try:
        image_bytes = base64.b64decode(req.image, validate=True)
    except Exception:
        logger.debug("invalid base64 image data in controlnet request", exc_info=True)
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
        width, height = map(int, req.size.split("x"))
    except (ValueError, AttributeError):
        width, height = 1024, 1024

    try:
        data = []
        for i in range(req.n):
            seed = (req.seed + i) if req.seed is not None else None
            png = await img_engine.generate_controlled(
                prompt=req.prompt,
                condition_image=image_bytes,
                condition_type=req.condition_type,
                width=width,
                height=height,
                num_inference_steps=req.num_inference_steps,
                seed=seed,
                controlnet_strength=req.controlnet_strength,
                canny_low=req.canny_low,
                canny_high=req.canny_high,
            )
            b64 = base64.b64encode(png).decode("ascii")
            if req.response_format == "b64_json":
                data.append({"b64_json": b64})
            else:
                data.append({"url": f"data:image/png;base64,{b64}"})

        return JSONResponse({
            "created": int(time.time()),
            "data": data,
        })
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"ControlNet gen error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="ControlNet generation failed")


class ImageDepthGuidedRequest(BaseModel):
    prompt: str = Field(description="Text prompt for generation")
    depth_image: str = Field(description="Base64-encoded depth visualization image")
    model: str = "Z-Image-Turbo-MLX-4bit"
    n: int = Field(default=1, ge=1, le=10)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    num_inference_steps: int = Field(default=4, ge=1, le=100)
    seed: Optional[int] = None
    depth_strength: float = Field(default=1.0, ge=0.0, le=2.0, description="Depth conditioning strength")

    @model_validator(mode="after")
    def validate_request(self):
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt: field is required and cannot be empty")
        if not self.depth_image or not self.depth_image.strip():
            raise ValueError("depth_image: field is required and cannot be empty")
        return self


@router.post("/images/depth-guided")
async def create_image_depth_guided(req: ImageDepthGuidedRequest) -> JSONResponse:
    """Generate a depth-guided image using a depth map for spatial control.

    The depth map provides structural guidance — areas with similar depth values
    will maintain spatial coherence in the generated image.
    """
    try:
        depth_bytes = base64.b64decode(req.depth_image, validate=True)
    except Exception:
        logger.debug("invalid base64 depth image data", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid base64 depth image data")

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
        width, height = map(int, req.size.split("x"))
    except (ValueError, AttributeError):
        width, height = 1024, 1024

    try:
        data = []
        for i in range(req.n):
            seed = (req.seed + i) if req.seed is not None else None
            png = await img_engine.generate_depth_guided(
                prompt=req.prompt,
                depth_image=depth_bytes,
                width=width,
                height=height,
                num_inference_steps=req.num_inference_steps,
                seed=seed,
                depth_strength=req.depth_strength,
            )
            b64 = base64.b64encode(png).decode("ascii")
            if req.response_format == "b64_json":
                data.append({"b64_json": b64})
            else:
                data.append({"url": f"data:image/png;base64,{b64}"})

        return JSONResponse({
            "created": int(time.time()),
            "data": data,
        })
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"Depth-guided gen error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Depth-guided generation failed")

