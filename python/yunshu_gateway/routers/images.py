from __future__ import annotations

"""OpenAI Images API compatible router — image generation endpoint."""

import base64
import contextlib
import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from ..engine import get_model_manager
from ..streaming import run_with_disconnect_guard  # non-streaming cancel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["images"])

MAX_IMAGE_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


def _rand_seed() -> int:
    """A distinct random seed for an image when the request supplied no seed, so n>1
    yields DIFFERENT images. : the old `else None` made every image fall to
    the engine's hardcoded default seed (42) → N byte-identical images."""
    import secrets

    return secrets.randbits(48)


def _decode_image_b64(data: str) -> bytes:
    """Decode a base64 image, stripping any data URL prefix (data:image/png;base64,...)."""
    import base64

    if isinstance(data, str) and data.startswith("data:"):
        comma = data.find(",")
        if comma >= 0:
            data = data[comma + 1 :]
    return base64.b64decode(data, validate=True)


def _select_image_engine(manager, model):
    """Select the loaded ImageGenEngine whose model_id matches `model`.

     the img2img-family routes (variation/edit/inpaint/
    controlnet/depth) and the t2i fallbacks grabbed the FIRST loaded image engine, ignoring
    req.model. With ≥2 image models loaded that served the WRONG model.
    Match by model_id. Fall back to the lone loaded engine when (a) `model` is empty, or
    (b) `model` was given but unmatched AND exactly one image engine is loaded.

    case (b) restores single-model deployments. req.model defaults to a hardcoded
    "Z-Image-Turbo-MLX-4bit" on every image route, so an operator who loaded their image
    model under any other id (a custom alias, or a different model) and a client that omits
    `model` would 404 under the strict match even though exactly one image model is
    loaded and unambiguous. The wrong-model protection only matters when ≥2 are loaded
    (where a sole-fallback can't apply).
    """
    from yunshu_engine.image_engine import ImageGenEngine

    first = None
    n_loaded = 0
    ml = model.lower() if model else ""
    for entry in manager.list_entries():
        if not (
            entry.is_loaded
            and isinstance(getattr(entry, "engine", None), ImageGenEngine)
        ):
            continue
        n_loaded += 1
        if first is None:
            first = entry.engine
        if ml and (entry.model_id == model or entry.model_id.lower() == ml):
            return entry.engine
    if not ml:
        return first  # no model requested → first-of-type (legacy default)
    # model requested but unmatched: serve the lone engine only when unambiguous.
    return first if n_loaded == 1 else None


def _register_image_cancel(model):
    """Register a non-streaming image request with the request tracker and
    return (cancel_event, tracker, id). The cancel keystone — the
    non-streaming edit/variation/inpaint routes never had it; only the streaming generate
    path did. Best-effort: returns (None, None, id) if the tracker is unavailable."""
    import uuid as _uuid

    _id = f"img-{_uuid.uuid4().hex[:24]}"
    try:
        from yunshu_engine.request_tracker import get_request_tracker

        _tracker = get_request_tracker()
        return _tracker.register(_id, model or "image").cancel_event, _tracker, _id
    except Exception:
        return None, None, _id


def _unregister_image_cancel(tracker, _id) -> None:
    if tracker is not None:
        with contextlib.suppress(Exception):
            tracker.unregister(_id)


class ImageGenerateRequest(BaseModel):
    prompt: str
    model: str = "Z-Image-Turbo-MLX-4bit"
    n: int = Field(default=1, ge=1, le=10)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    num_inference_steps: int = Field(default=4, ge=1, le=100)
    seed: int | None = None
    preview_interval: int = (
        0  # Decode & emit intermediate preview every N steps (0=off)
    )
    # ControlNet (Z-Image-Fun-Controlnet-Union): a control map (canny/depth/pose/edge)
    # as base64 PNG/JPEG. When set, structure follows the control map. LoRA and
    # (word:weight) emphasis need no field — they're written inline in `prompt`
    # (<lora:name:0.7>, (word:1.3)) and applied transparently.
    control_image: str | None = None
    control_scale: float = Field(default=0.8, ge=0.0, le=2.0)
    # Note: guidance_scale and negative_prompt removed — Turbo models
    # don't support classifier-free guidance. These params were accepted
    # but silently ignored by the engine.

    @model_validator(mode="after")
    def validate_request(self):
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt: field is required and cannot be empty")
        if self.response_format not in ("b64_json", "url"):
            raise ValueError(
                f"response_format: must be 'b64_json' or 'url', got '{self.response_format}'"
            )
        return self


@router.post("/images/generations")
async def create_image(req: ImageGenerateRequest, request: Request) -> JSONResponse:
    """Generate images from text prompt (OpenAI /v1/images/generations compatible)."""
    from .models import _check_permission

    _check_permission(request, "can_infer")
    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    from yunshu_engine.image_engine import ImageGenEngine

    # Find image gen engine — match by model_id. Was: primary model_id match
    # then a first-of-type fallback that served the WRONG model. Dropped that fallback.
    img_engine = _select_image_engine(manager, req.model)

    if img_engine is None:
        # Try loading by model name
        try:
            engine = await manager.get_engine(req.model)
            if isinstance(engine, ImageGenEngine):
                img_engine = engine
        except (KeyError, Exception):
            logger.debug(f"failed to load image engine for {req.model}", exc_info=True)

    if img_engine is None:
        raise HTTPException(
            status_code=404,
            detail=f"Image generation model '{req.model}' not found",
        )

    # Parse size — reject malformed values rather than silently fall back so
    # callers don't get a 1024x1024 image when they asked for "512x512" with a
    # typo or wrong type.
    try:
        width, height = map(int, req.size.split("x"))
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid size '{req.size}': expected '<width>x<height>' (e.g. '1024x1024')",
        ) from None

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

    # Decode the optional control image INSIDE a guard: _decode_image_b64 raises
    # binascii.Error (a ValueError) on malformed base64. Left bare, that propagated
    # to the global handler as a 500, while every sibling image route correctly
    # returns 400 for bad base64. Match them.
    try:
        control_bytes = (
            _decode_image_b64(req.control_image) if req.control_image else None
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400, detail="Invalid base64 control_image data"
        ) from None
    images_data = []
    # client-disconnect cancellation for the control-image branch (the plain t2i
    # branch is fast 4-step and left as-is). Register once; unregister in finally.
    _img_cancel, _img_tracker, _img_id = _register_image_cancel(req.model)
    try:
        for i in range(req.n):
            _seed = (
                ((req.seed + i) & 0x7FFFFFFFFFFFFFFF)
                if req.seed is not None
                else _rand_seed()
            )
            if control_bytes is not None:
                png_bytes = await run_with_disconnect_guard(
                    request,
                    img_engine.generate_controlled_image(
                        req.prompt,
                        control_bytes,
                        control_scale=req.control_scale,
                        width=width,
                        height=height,
                        num_inference_steps=req.num_inference_steps,
                        seed=_seed,
                        cancel_event=_img_cancel,
                    ),
                    cancel_event=_img_cancel,
                )
                if png_bytes is None:  # client disconnected → stop the n-loop
                    break
            else:
                png_bytes = await img_engine.generate_image(
                    prompt=req.prompt,
                    width=width,
                    height=height,
                    num_inference_steps=req.num_inference_steps,
                    seed=_seed,
                )

            b64 = base64.b64encode(png_bytes).decode("ascii")
            if req.response_format == "b64_json":
                # OpenAI spec: only b64_json field populated
                images_data.append(
                    {
                        "b64_json": b64,
                    }
                )
            else:
                # response_format == "url": OpenAI spec wants HTTP(S) URL in url field.
                # Yunshu does not host generated images, so we surface the same data via
                # b64_json (spec-correct field for inline payloads) and keep the data URI
                # in url for backwards-compat with existing clients.
                images_data.append(
                    {
                        "url": f"data:image/png;base64,{b64}",
                        "b64_json": b64,
                    }
                )
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory") from None
    except NotImplementedError as e:
        logger.warning("Image generation not implemented for this model: %s", e)
        raise HTTPException(status_code=501, detail=str(e)) from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Image generation error: {e}", exc_info=True)
        # SECURITY: don't echo the raw exception (leaks filesystem paths /
        # model internals / library error text) to the client — log it server-side,
        # return a generic message like the sibling routes (scoring/audio).
        raise HTTPException(status_code=500, detail="Image generation failed") from None
    finally:
        _unregister_image_cancel(_img_tracker, _img_id)

    return JSONResponse(
        {
            "created": int(time.time()),
            "data": images_data,
        }
    )


@router.post("/images/generations/stream")
async def stream_image_generation(req: ImageGenerateRequest, request: Request):
    """Stream image generation progress as SSE events."""
    from .models import _check_permission

    _check_permission(request, "can_infer")
    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    from yunshu_engine.image_engine import ImageGenEngine

    # Match by model_id; dropped the first-of-type fallback that served the wrong
    # (unauthorized) model. Consistent with the non-streaming path.
    img_engine = _select_image_engine(manager, req.model)

    if img_engine is None:
        # Try loading by model name
        try:
            engine = await manager.get_engine(req.model)
            if isinstance(engine, ImageGenEngine):
                img_engine = engine
        except (KeyError, Exception):
            logger.debug(f"failed to load image engine for {req.model}", exc_info=True)

    if img_engine is None:
        raise HTTPException(
            status_code=404, detail="No image generation engine available"
        )

    # Parse and validate size — reject malformed values rather than silently
    # fall back so callers see an explicit 400 (consistent with /generations).
    try:
        width, height = map(int, req.size.split("x"))
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid size '{req.size}': expected '<width>x<height>' (e.g. '1024x1024')",
        ) from None
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
    _img_tracker = None
    _img_gen = None
    _cancel_event = None
    try:
        from yunshu_engine.request_tracker import get_request_tracker

        _img_tracker = get_request_tracker()
        _img_gen = _img_tracker.register(_img_id, req.model)
        _cancel_event = _img_gen.cancel_event
    except Exception:
        _img_tracker = None

    # an unseeded request must get a FRESH random seed (matching every
    # other image endpoint). Passing req.seed=None straight through made
    # _stream_sync fall back to a hardcoded seed 42 → identical image on every
    # call, contradicting the _rand_seed contract (the n-identical fix).
    _stream_seed = req.seed if req.seed is not None else _rand_seed()

    async def _progress_stream():
        async for chunk in img_engine.generate_image_stream(
            prompt=req.prompt,
            width=width,
            height=height,
            num_inference_steps=req.num_inference_steps,
            seed=_stream_seed,
            preview_interval=req.preview_interval,
        ):
            if _cancel_event is not None and _cancel_event.is_set():
                yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                # emit the SSE terminator on cancel too. Every other terminal
                # path here (final-image, error) ends with [DONE]; the cancel path
                # returned without it, so a strict SSE client waiting for [DONE] could
                # hang until socket close.
                yield "data: [DONE]\n\n"
                return
            if chunk.get("is_final") and chunk.get("image"):
                b64 = base64.b64encode(chunk["image"]).decode("ascii")
                yield f"data: {json.dumps({'step': chunk['step'], 'progress': 1.0, 'image': b64, 'is_final': True})}\n\n"
                yield "data: [DONE]\n\n"
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
                cancel_event=_cancel_event,
            ):
                yield event.encode("utf-8") if isinstance(event, str) else event
        except MemoryError:
            yield f"data: {json.dumps({'error': {'message': 'Out of GPU memory', 'type': 'memory_error'}})}\n\n".encode()
            yield b"data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Image streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': {'message': 'Image generation failed', 'type': 'server_error'}})}\n\n".encode()
            yield b"data: [DONE]\n\n"
        finally:
            if _img_tracker is not None:
                with contextlib.suppress(Exception):
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
    seed: int | None = None
    # Variations carry NO prompt, so the source image is the only signal. A high
    # denoise_strength + empty prompt destroys the subject and the unguided
    # denoise produces an unrelated image; keep it low so the subject is varied,
    # not replaced. (Edits use 0.8 because their prompt re-guides the denoise.)
    denoise_strength: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="How much to re-denoise (0=keep source, 1=full)",
    )

    @model_validator(mode="after")
    def validate_request(self):
        if not self.image or not self.image.strip():
            raise ValueError("image: field is required and cannot be empty")
        if self.response_format not in ("b64_json", "url"):
            raise ValueError(
                f"response_format: must be 'b64_json' or 'url', got '{self.response_format}'"
            )
        return self


@router.post("/images/variations")
async def create_image_variation(
    req: ImageVariationsRequest, request: Request
) -> JSONResponse:
    """Generate variations of an input image (OpenAI /v1/images/variations compatible).

    Uses the input image as a conditioning signal for the diffusion model.
    The image is decoded and used as a starting point for the generation.
    """
    from .models import _check_permission

    _check_permission(request, "can_infer")
    import base64

    try:
        image_bytes = _decode_image_b64(req.image)
        if len(image_bytes) > MAX_IMAGE_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413, detail=f"Image too large ({len(image_bytes)} bytes)"
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400, detail="Invalid base64 image data"
        ) from None

    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    # match by model_id (was first-of-type, ignoring req.model — wrong-model serving).
    img_engine = _select_image_engine(manager, req.model)

    if img_engine is None:
        raise HTTPException(
            status_code=404, detail="No image generation model available"
        )

    try:
        width, height = map(int, req.size.split("x"))
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid size '{req.size}': expected '<width>x<height>' (e.g. '1024x1024')",
        ) from None

    # Validate dimensions (same rules as /images/generations)
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

    # client-disconnect cancellation (the cancel keystone — the
    # NON-streaming image routes never got it; only the streaming generate path did,
    # despite commit note). An abandoned n>1 request otherwise runs all n diffusions
    # to completion on the serial GPU executor, head-of-line-blocking other work. The
    # engine's generate() honors cancel_event mid-diffusion, so wrapping each call in the
    # disconnect guard aborts the in-flight image and (via the persisted event) skips the
    # rest.
    _img_cancel, _img_tracker, _img_id = _register_image_cancel(req.model)
    try:
        # Generate variation using the input image as conditioning
        images = []
        for i in range(req.n):
            seed = (
                ((req.seed + i) & 0x7FFFFFFFFFFFFFFF)
                if req.seed is not None
                else _rand_seed()
            )
            result = await run_with_disconnect_guard(
                request,
                img_engine.generate(
                    prompt="",
                    num_inference_steps=req.num_inference_steps,
                    seed=seed,
                    image=image_bytes,
                    width=width,
                    height=height,
                    denoise_strength=req.denoise_strength,
                    cancel_event=_img_cancel,
                ),
                cancel_event=_img_cancel,
            )
            if result is None:  # client disconnected → stop the n-loop
                break
            if isinstance(result, list):
                images.extend(result)
            else:
                images.append(result)

        data = []
        for img in images:
            b64 = base64.b64encode(img).decode("ascii")
            # honor response_format="url" (variations + edits validated it as a
            # legal value but always returned b64_json only, unlike /generations//inpaint/
            # /controlnet//depth-guided — a client keying on `url` got a 200 missing it).
            if req.response_format == "b64_json":
                data.append({"b64_json": b64})
            else:
                data.append({"url": f"data:image/png;base64,{b64}"})

        return JSONResponse(
            {
                "created": int(time.time()),
                "data": data,
            }
        )
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory") from None
    except Exception as e:
        logger.error(f"Image variation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Image variation failed") from None
    finally:
        _unregister_image_cancel(_img_tracker, _img_id)


class ImageEditsRequest(BaseModel):
    image: str  # base64 encoded source image
    prompt: str  # edit instruction
    model: str = "Z-Image-Turbo-MLX-4bit"
    n: int = Field(default=1, ge=1, le=10)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    num_inference_steps: int = Field(default=4, ge=1, le=100)
    seed: int | None = None
    denoise_strength: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="How much to re-denoise (1.0=full, 0.0=keep source)",
    )

    @model_validator(mode="after")
    def validate_request(self):
        if not self.image or not self.image.strip():
            raise ValueError("image: field is required and cannot be empty")
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt: field is required and cannot be empty")
        if self.response_format not in ("b64_json", "url"):
            raise ValueError(
                f"response_format: must be 'b64_json' or 'url', got '{self.response_format}'"
            )
        return self


@router.post("/images/edits")
async def create_image_edit(req: ImageEditsRequest, request: Request) -> JSONResponse:
    """Edit an image based on a text prompt (OpenAI /v1/images/edits compatible).

    Combines the input image with a text prompt to generate an edited version.
    """
    from .models import _check_permission

    _check_permission(request, "can_infer")
    import base64

    try:
        image_bytes = _decode_image_b64(req.image)
        if len(image_bytes) > MAX_IMAGE_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413, detail=f"Image too large ({len(image_bytes)} bytes)"
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400, detail="Invalid base64 image data"
        ) from None

    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    # match by model_id (was first-of-type, ignoring req.model — wrong-model serving).
    img_engine = _select_image_engine(manager, req.model)

    if img_engine is None:
        raise HTTPException(
            status_code=404, detail="No image generation model available"
        )

    try:
        width, height = map(int, req.size.split("x"))
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid size '{req.size}': expected '<width>x<height>' (e.g. '1024x1024')",
        ) from None

    # Validate dimensions
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

    # client-disconnect cancellation (see _register_image_cancel / the variation
    # route) — the engine's generate() honors cancel_event mid-diffusion.
    _img_cancel, _img_tracker, _img_id = _register_image_cancel(req.model)
    try:
        images = []
        for i in range(req.n):
            seed = (
                ((req.seed + i) & 0x7FFFFFFFFFFFFFFF)
                if req.seed is not None
                else _rand_seed()
            )
            result = await run_with_disconnect_guard(
                request,
                img_engine.generate(
                    prompt=req.prompt,
                    num_inference_steps=req.num_inference_steps,
                    seed=seed,
                    image=image_bytes,
                    width=width,
                    height=height,
                    denoise_strength=req.denoise_strength,
                    cancel_event=_img_cancel,
                ),
                cancel_event=_img_cancel,
            )
            if result is None:  # client disconnected → stop the n-loop
                break
            if isinstance(result, list):
                images.extend(result)
            else:
                images.append(result)

        data = []
        for img in images:
            b64 = base64.b64encode(img).decode("ascii")
            # honor response_format="url" (variations + edits validated it as a
            # legal value but always returned b64_json only, unlike /generations//inpaint/
            # /controlnet//depth-guided — a client keying on `url` got a 200 missing it).
            if req.response_format == "b64_json":
                data.append({"b64_json": b64})
            else:
                data.append({"url": f"data:image/png;base64,{b64}"})

        return JSONResponse(
            {
                "created": int(time.time()),
                "data": data,
            }
        )
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory") from None
    except Exception as e:
        logger.error(f"Image edit error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Image edit failed") from None
    finally:
        _unregister_image_cancel(_img_tracker, _img_id)


class ImageInpaintRequest(BaseModel):
    image: str = Field(description="Base64-encoded source image (PNG/JPEG)")
    prompt: str = Field(
        description="Text description of what to fill in the masked region"
    )
    mask: str | None = Field(
        default=None, description="Base64-encoded mask image (white=fill, black=keep)"
    )
    model: str = "Z-Image-Turbo-MLX-4bit"
    n: int = Field(default=1, ge=1, le=10)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    # Inpaint fills the masked region starting from noise; at strength 1.0 it
    # starts from PURE noise, which the few default turbo steps can't converge
    # (produced a blank white blob). 0.75 starts from mostly-noised KNOWN latents
    # so the fast steps reach coherent content. Raise toward 1.0 + more steps for
    # a fuller replacement.
    num_inference_steps: int = Field(default=8, ge=1, le=100)
    seed: int | None = None
    denoise_strength: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="How much to re-denoise (1.0=full replace; lower keeps more of the source)",
    )

    @model_validator(mode="after")
    def validate_request(self):
        if not self.image or not self.image.strip():
            raise ValueError("image: field is required and cannot be empty")
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt: field is required and cannot be empty")
        if self.response_format not in ("b64_json", "url"):
            raise ValueError(
                f"response_format: must be 'b64_json' or 'url', got '{self.response_format}'"
            )
        if self.mask is not None and not self.mask.strip():
            raise ValueError("mask: if provided, cannot be empty or whitespace-only")
        return self


@router.post("/images/inpaint")
async def create_image_inpaint(
    req: ImageInpaintRequest, request: Request
) -> JSONResponse:
    """Inpaint masked regions of an image using a text prompt.

    Accepts a source image and a mask (white=fill, black=preserve).
    The masked region is re-generated guided by the text prompt while
    the unmasked region is preserved from the original image.
    """
    from .models import _check_permission

    _check_permission(request, "can_infer")
    try:
        image_bytes = _decode_image_b64(req.image)
        if len(image_bytes) > MAX_IMAGE_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413, detail=f"Image too large ({len(image_bytes)} bytes)"
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400, detail="Invalid base64 image data"
        ) from None

    # Sanitize mask: strip data URL prefix so engine receives clean base64
    mask_clean: str | None = None
    if req.mask is not None:
        try:
            mask_bytes = _decode_image_b64(req.mask)
            if len(mask_bytes) > MAX_IMAGE_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413, detail=f"Mask too large ({len(mask_bytes)} bytes)"
                )
            mask_clean = base64.b64encode(mask_bytes).decode("ascii")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=400, detail="Invalid base64 mask data"
            ) from None

    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    # match by model_id (was first-of-type, ignoring req.model — wrong-model serving).
    img_engine = _select_image_engine(manager, req.model)

    if img_engine is None:
        raise HTTPException(
            status_code=404, detail="No image generation model available"
        )

    try:
        width, height = map(int, req.size.split("x"))
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid size '{req.size}': expected '<width>x<height>' (e.g. '1024x1024')",
        ) from None

    # Validate dimensions
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

    # client-disconnect cancellation (pattern, swept to inpaint).
    _img_cancel, _img_tracker, _img_id = _register_image_cancel(req.model)
    try:
        data = []
        for i in range(req.n):
            seed = (
                ((req.seed + i) & 0x7FFFFFFFFFFFFFFF)
                if req.seed is not None
                else _rand_seed()
            )
            png = await run_with_disconnect_guard(
                request,
                img_engine.inpaint(
                    prompt=req.prompt,
                    image=image_bytes,
                    mask_base64=mask_clean,
                    width=width,
                    height=height,
                    num_inference_steps=req.num_inference_steps,
                    seed=seed,
                    denoise_strength=req.denoise_strength,
                    cancel_event=_img_cancel,
                ),
                cancel_event=_img_cancel,
            )
            if png is None:  # client disconnected → stop the n-loop
                break
            b64 = base64.b64encode(png).decode("ascii")
            if req.response_format == "b64_json":
                data.append({"b64_json": b64})
            else:
                data.append({"url": f"data:image/png;base64,{b64}"})

        return JSONResponse(
            {
                "created": int(time.time()),
                "data": data,
            }
        )
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory") from None
    except Exception as e:
        logger.error(f"Image inpaint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Image inpainting failed") from None
    finally:
        _unregister_image_cancel(_img_tracker, _img_id)


class ImageControlNetRequest(BaseModel):
    prompt: str = Field(description="Text prompt for generation")
    image: str = Field(
        description="Base64-encoded conditioning image (edges, depth map, etc.)"
    )
    condition_type: str = Field(
        default="canny", description="Conditioning type: canny, depth, raw"
    )
    model: str = "Z-Image-Turbo-MLX-4bit"
    n: int = Field(default=1, ge=1, le=10)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    num_inference_steps: int = Field(default=4, ge=1, le=100)
    seed: int | None = None
    controlnet_strength: float = Field(
        default=1.0, ge=0.0, le=2.0, description="Conditioning strength"
    )
    canny_low: int = Field(
        default=100, ge=0, le=255, description="Canny lower threshold"
    )
    canny_high: int = Field(
        default=200, ge=0, le=255, description="Canny upper threshold"
    )

    @model_validator(mode="after")
    def validate_request(self):
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt: field is required and cannot be empty")
        if not self.image or not self.image.strip():
            raise ValueError("image: field is required and cannot be empty")
        if self.condition_type not in ("canny", "depth", "raw"):
            raise ValueError(
                f"condition_type: must be 'canny', 'depth', or 'raw', got '{self.condition_type}'"
            )
        if self.response_format not in ("b64_json", "url"):
            raise ValueError(
                f"response_format: must be 'b64_json' or 'url', got '{self.response_format}'"
            )
        return self


@router.post("/images/controlnet")
async def create_image_controlnet(
    req: ImageControlNetRequest, request: Request
) -> JSONResponse:
    """Generate an image with ControlNet spatial conditioning.

    Accepts a conditioning image (edge map, depth map, etc.) and a text prompt.
    The conditioning image guides the spatial structure of the generated output.
    """
    from .models import _check_permission

    _check_permission(request, "can_infer")
    try:
        image_bytes = _decode_image_b64(req.image)
        if len(image_bytes) > MAX_IMAGE_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413, detail=f"Image too large ({len(image_bytes)} bytes)"
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400, detail="Invalid base64 image data"
        ) from None

    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    # match by model_id (was first-of-type, ignoring req.model — wrong-model serving).
    img_engine = _select_image_engine(manager, req.model)

    if img_engine is None:
        raise HTTPException(
            status_code=404, detail="No image generation model available"
        )

    try:
        width, height = map(int, req.size.split("x"))
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid size '{req.size}': expected '<width>x<height>' (e.g. '1024x1024')",
        ) from None

    # Validate dimensions
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

    # client-disconnect cancellation (pattern, swept to controlnet).
    _img_cancel, _img_tracker, _img_id = _register_image_cancel(req.model)
    try:
        data = []
        for i in range(req.n):
            seed = (
                ((req.seed + i) & 0x7FFFFFFFFFFFFFFF)
                if req.seed is not None
                else _rand_seed()
            )
            png = await run_with_disconnect_guard(
                request,
                img_engine.generate_controlled(
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
                    cancel_event=_img_cancel,
                ),
                cancel_event=_img_cancel,
            )
            if png is None:  # client disconnected → stop the n-loop
                break
            b64 = base64.b64encode(png).decode("ascii")
            if req.response_format == "b64_json":
                data.append({"b64_json": b64})
            else:
                data.append({"url": f"data:image/png;base64,{b64}"})

        return JSONResponse(
            {
                "created": int(time.time()),
                "data": data,
            }
        )
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory") from None
    except Exception as e:
        logger.error(f"ControlNet gen error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="ControlNet generation failed"
        ) from None
    finally:
        _unregister_image_cancel(_img_tracker, _img_id)


class ImageDepthGuidedRequest(BaseModel):
    prompt: str = Field(description="Text prompt for generation")
    depth_image: str = Field(description="Base64-encoded depth visualization image")
    model: str = "Z-Image-Turbo-MLX-4bit"
    n: int = Field(default=1, ge=1, le=10)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    num_inference_steps: int = Field(default=4, ge=1, le=100)
    seed: int | None = None
    depth_strength: float = Field(
        default=1.0, ge=0.0, le=2.0, description="Depth conditioning strength"
    )

    @model_validator(mode="after")
    def validate_request(self):
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt: field is required and cannot be empty")
        if not self.depth_image or not self.depth_image.strip():
            raise ValueError("depth_image: field is required and cannot be empty")
        if self.response_format not in ("b64_json", "url"):
            raise ValueError(
                f"response_format: must be 'b64_json' or 'url', got '{self.response_format}'"
            )
        return self


@router.post("/images/depth-guided")
async def create_image_depth_guided(
    req: ImageDepthGuidedRequest, request: Request
) -> JSONResponse:
    """Generate a depth-guided image using a depth map for spatial control.

    The depth map provides structural guidance — areas with similar depth values
    will maintain spatial coherence in the generated image.
    """
    from .models import _check_permission

    _check_permission(request, "can_infer")
    try:
        depth_bytes = _decode_image_b64(req.depth_image)
        if len(depth_bytes) > MAX_IMAGE_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Depth image too large ({len(depth_bytes)} bytes)",
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400, detail="Invalid base64 depth image data"
        ) from None

    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    # match by model_id (was first-of-type, ignoring req.model — wrong-model serving).
    img_engine = _select_image_engine(manager, req.model)

    if img_engine is None:
        raise HTTPException(
            status_code=404, detail="No image generation model available"
        )

    try:
        width, height = map(int, req.size.split("x"))
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid size '{req.size}': expected '<width>x<height>' (e.g. '1024x1024')",
        ) from None

    # Validate dimensions
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

    # client-disconnect cancellation (pattern, swept to depth-guided).
    _img_cancel, _img_tracker, _img_id = _register_image_cancel(req.model)
    try:
        data = []
        for i in range(req.n):
            seed = (
                ((req.seed + i) & 0x7FFFFFFFFFFFFFFF)
                if req.seed is not None
                else _rand_seed()
            )
            png = await run_with_disconnect_guard(
                request,
                img_engine.generate_depth_guided(
                    prompt=req.prompt,
                    depth_image=depth_bytes,
                    width=width,
                    height=height,
                    num_inference_steps=req.num_inference_steps,
                    seed=seed,
                    depth_strength=req.depth_strength,
                    cancel_event=_img_cancel,
                ),
                cancel_event=_img_cancel,
            )
            if png is None:  # client disconnected → stop the n-loop
                break
            b64 = base64.b64encode(png).decode("ascii")
            if req.response_format == "b64_json":
                data.append({"b64_json": b64})
            else:
                data.append({"url": f"data:image/png;base64,{b64}"})

        return JSONResponse(
            {
                "created": int(time.time()),
                "data": data,
            }
        )
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory") from None
    except Exception as e:
        logger.error(f"Depth-guided gen error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Depth-guided generation failed"
        ) from None
    finally:
        _unregister_image_cancel(_img_tracker, _img_id)
