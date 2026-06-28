from __future__ import annotations

"""OpenAI Video API compatible router — video generation endpoint.

Supports:
- /video/generations — text-to-video and image-to-video generation
- Streaming frame delivery via SSE when stream=true
"""
import base64
import contextlib
import json
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..engine import get_model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["video"])


class VideoGenerateRequest(BaseModel):
    model: str = "wan-2.2-t2v"
    prompt: str = Field(description="Text description of the video to generate")
    negative_prompt: str = ""
    image: str | None = Field(default=None, description="Base64-encoded source image for I2V mode")
    width: int = Field(default=1280, ge=64, le=2048)
    height: int = Field(default=704, ge=64, le=2048)
    num_frames: int = Field(default=81, ge=1, le=257)
    num_inference_steps: int = Field(default=20, ge=1, le=100)
    guide_scale: float = 5.0
    fps: int = Field(default=16, ge=1, le=60)
    seed: int | None = None
    scheduler: str = "unipc"
    response_format: str = "mp4"  # mp4 or frames
    stream: bool = False  # SSE streaming — delivers frames as they're generated


@router.post("/video/generations")
async def create_video(req: VideoGenerateRequest, request: Request):
    """Generate video from text prompt (and optionally an image for I2V).

    OpenAI-compatible video generation endpoint.
    When stream=true, delivers frames via SSE as they're generated.
    """
    from .models import _check_model_access, _check_permission
    _check_permission(request, "can_infer")
    _check_model_access(request, req.model)
    from yunshu_engine.video_engine import VideoEngine

    # Validate base64 image early
    image_bytes = None
    if req.image:
        try:
            # reuse the images-router helper so a standard data-URI image
            # (data:image/png;base64,...) — accepted by every /v1/images endpoint —
            # also works for video I2V. The old inline b64decode rejected the prefix.
            from .images import _decode_image_b64
            image_bytes = _decode_image_b64(req.image)
        except Exception:
            logger.debug("invalid base64 image data in video request", exc_info=True)
            raise HTTPException(status_code=400, detail="Invalid base64 image data") from None

    # Try to find a registered video engine
    video_engine = None
    manager = get_model_manager()
    if manager is not None:
        for entry in manager.list_entries():
            if entry.is_loaded and isinstance(getattr(entry, 'engine', None), VideoEngine):
                if req.model in {entry.model_id, entry.model_id.lower()}:
                    video_engine = entry.engine
                    break

        if video_engine is None:
            try:
                engine = await manager.get_engine(req.model)
                if isinstance(engine, VideoEngine):
                    video_engine = engine
            except (KeyError, Exception):
                logger.debug(f"Failed to load video engine for {req.model}", exc_info=True)

        if video_engine is None:
            # the W818/W823/W824 wrong-model keystone, unswept for video. The
            # old fallback grabbed the FIRST loaded VideoEngine, ignoring req.model — and
            # since _check_model_access(req.model) was verified above, that served a model
            # the key may NOT be authorized for. With ≥2 video models loaded and none
            # matching, the requested model is unavailable → 404; do NOT silently serve a
            # different one.
            _loaded_video = [
                e.engine for e in manager.list_entries()
                if e.is_loaded and isinstance(getattr(e, 'engine', None), VideoEngine)
            ]
            if len(_loaded_video) == 1:
                # single-model deployments are unambiguous — req.model defaults to
                # the hardcoded "wan-2.2-t2v", so an operator who loaded a video model under
                # another id (with a client that omits model) would 404 even though exactly
                # one video model is loaded. Serve it (mirrors _select_image_engine W835).
                video_engine = _loaded_video[0]
            elif len(_loaded_video) >= 2:
                raise HTTPException(
                    status_code=404,
                    detail=f"Video model '{req.model}' not found",
                )

    # Fallback: create a standalone engine (only when no video model is registered)
    if video_engine is None:
        video_engine = VideoEngine()

    # client-disconnect cancellation (the W758/W845 cancel keystone,
    # un-propagated to video — images/audio already had it; OCR closed in W859, video was
    # the last single-shot-media sibling). Register with the request tracker for accounting.
    _vid_id = f"vid-{uuid.uuid4().hex[:24]}"
    _vid_tracker = None
    _vid_cancel = None
    try:
        from yunshu_engine.request_tracker import get_request_tracker
        _vid_tracker = get_request_tracker()
        _vid_cancel = _vid_tracker.register(_vid_id, req.model or "video").cancel_event
    except Exception:
        _vid_tracker = None

    # Streaming path — deliver frames via SSE as generated.
    if req.stream:
        # with_sse_keepalive polls is_disconnected and, on disconnect, sets cancel_event
        # AND aclose()s the source generator → that fires generate_stream's finally → its
        # internal _cancel.set() → the executor's diffusion/frame loop stops promptly
        # instead of running the whole (expensive) video to completion.
        from ..streaming import with_sse_keepalive

        async def _guarded_video_stream():
            try:
                async for _ev in with_sse_keepalive(
                    _stream_video_frames(video_engine, req, image_bytes),
                    http_request=request,
                    cancel_event=_vid_cancel,
                ):
                    yield _ev.encode("utf-8") if isinstance(_ev, str) else _ev
            finally:
                if _vid_tracker is not None:
                    with contextlib.suppress(Exception):
                        _vid_tracker.unregister(_vid_id)

        return StreamingResponse(
            _guarded_video_stream(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    try:
        from ..streaming import run_with_disconnect_guard
        # Non-streaming generate() has no mid-gen cancel hook, so the bounded diffusion
        # runs to completion on the executor; the guard still frees the handler promptly on
        # disconnect (no event-loop head-of-line block) and None → client gone, discarded.
        result = await run_with_disconnect_guard(
            request,
            video_engine.generate(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                image=image_bytes,
                width=req.width,
                height=req.height,
                num_frames=req.num_frames,
                num_steps=req.num_inference_steps,
                guide_scale=req.guide_scale,
                fps=req.fps,
                seed=req.seed,
                scheduler=req.scheduler,
                output_format=req.response_format,
            ),
            cancel_event=_vid_cancel,
        )
        if result is None:
            # Client disconnected mid-generation; the response is discarded by uvicorn.
            return JSONResponse({"created": int(time.time()), "data": []})
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory") from None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Video generation failed") from None
    finally:
        if _vid_tracker is not None:
            with contextlib.suppress(Exception):
                _vid_tracker.unregister(_vid_id)

    # When the engine degraded to the fallback path it returns synthetic
    # PLACEHOLDER frames ("Frame i/N") — not real generation — and the mp4 encode
    # yields empty bytes. Do NOT hand back a misleading HTTP 200; signal clearly
    # via the method marker the engine sets.
    if "fallback" in (result.method or ""):
        raise HTTPException(
            status_code=503,
            detail=(
                "Video backend unavailable: only placeholder frames were produced "
                f"(method={result.method!r}). The mlx_video dependency is missing, "
                "or this model architecture is not supported for video generation."
            ),
        )

    if req.response_format == "frames":
        # Guard non-bytes frames (matches the streaming path) — a stray mx.array/None
        # would otherwise crash b64encode into a 500 instead of returning what we have.
        frames_b64 = [
            base64.b64encode(f).decode("ascii")
            for f in (result.frames or []) if isinstance(f, (bytes, bytearray))
        ]
        return JSONResponse({
            "created": int(time.time()),
            "data": [{
                "frames": frames_b64,
                "num_frames": result.num_frames,
                "fps": result.fps,
                "width": result.width,
                "height": result.height,
                "method": result.method,
            }],
        })
    else:
        video_b64 = base64.b64encode(result.video_data).decode("ascii")
        return JSONResponse({
            "created": int(time.time()),
            "data": [{
                "video": video_b64,
                "num_frames": result.num_frames,
                "fps": result.fps,
                "width": result.width,
                "height": result.height,
                "method": result.method,
            }],
        })


async def _stream_video_frames(video_engine, req: VideoGenerateRequest, image_bytes: bytes | None):
    """SSE generator that streams frames as they are generated and decoded.

    Uses generate_stream() which yields frames progressively instead of
    waiting for the entire video to be generated first. For the native MLX
    pipeline, frames are decoded one-by-one by the VAE and delivered
    immediately. For mlx-video, the MP4 is generated first but then decoded
    frame-by-frame via ffmpeg pipe (better than buffering all frames).
    """
    try:
        frame_count = 0
        async for frame_data in video_engine.generate_stream(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image=image_bytes,
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            num_steps=req.num_inference_steps,
            guide_scale=req.guide_scale,
            fps=req.fps,
            seed=req.seed,
            scheduler=req.scheduler,
        ):
            if not isinstance(frame_data, dict):
                continue
            # : do NOT stream synthetic placeholder frames as if real (the
            # non-stream path returns 503 for "fallback"; the stream path must
            # signal failure too, not hand back fake frames with HTTP 200).
            if "fallback" in (frame_data.get("method") or ""):
                yield f"data: {json.dumps({'error': {'message': 'Video backend unavailable: only placeholder frames were produced (mlx_video missing or model unloadable). No real video generated.', 'type': 'service_unavailable'}})}\n\n"
                yield "data: [DONE]\n\n"
                return

            frame_bytes = frame_data.get("frame")
            if not frame_bytes or not isinstance(frame_bytes, bytes):
                continue

            frame_b64 = base64.b64encode(frame_bytes).decode("ascii")
            is_final = frame_data.get("is_final", False)
            chunk = {
                "created": int(time.time()),
                "data": [{
                    "frame": frame_b64,
                    "index": frame_data.get("index", frame_count),
                    "width": frame_data.get("width", 0),
                    "height": frame_data.get("height", 0),
                    "method": frame_data.get("method", ""),
                    "type": "frame",
                }],
            }
            frame_count += 1
            yield f"data: {json.dumps(chunk)}\n\n"

            if is_final:
                break

        # Send done event with frame count
        done_chunk = {
            "created": int(time.time()),
            "data": [{
                "type": "done",
                "frames_delivered": frame_count,
            }],
        }
        yield f"data: {json.dumps(done_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except MemoryError:
        yield f"data: {json.dumps({'error': {'message': 'Out of GPU memory', 'type': 'memory_error'}})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Video streaming error: {e}", exc_info=True)
        yield f"data: {json.dumps({'error': {'message': 'Video generation failed', 'type': 'server_error'}})}\n\n"
        yield "data: [DONE]\n\n"
