"""OpenAI Video API compatible router — video generation endpoint.

Supports:
- /video/generations — text-to-video and image-to-video generation
- Streaming frame delivery via SSE when stream=true
"""
import base64
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..engine import get_model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["video"])


class VideoGenerateRequest(BaseModel):
    model: str = "wan-2.2-t2v"
    prompt: str = Field(description="Text description of the video to generate")
    negative_prompt: str = ""
    image: Optional[str] = Field(default=None, description="Base64-encoded source image for I2V mode")
    width: int = 1280
    height: int = 704
    num_frames: int = 81  # Must be 4n+1
    num_inference_steps: int = 20
    guide_scale: float = 5.0
    fps: int = 16
    seed: Optional[int] = None
    scheduler: str = "unipc"
    response_format: str = "mp4"  # mp4 or frames
    stream: bool = False  # SSE streaming — delivers frames as they're generated


@router.post("/video/generations")
async def create_video(req: VideoGenerateRequest):
    """Generate video from text prompt (and optionally an image for I2V).

    OpenAI-compatible video generation endpoint.
    When stream=true, delivers frames via SSE as they're generated.
    """
    from yunshu_engine.video_engine import VideoEngine

    # Validate base64 image early
    image_bytes = None
    if req.image:
        try:
            image_bytes = base64.b64decode(req.image, validate=True)
        except Exception:
            logger.debug("invalid base64 image data in video request", exc_info=True)
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

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
            for entry in manager.list_entries():
                if entry.is_loaded and isinstance(getattr(entry, 'engine', None), VideoEngine):
                    video_engine = entry.engine
                    break

    # Fallback: create a standalone engine
    if video_engine is None:
        video_engine = VideoEngine()

    # Streaming path — deliver frames via SSE as generated
    if req.stream:
        return StreamingResponse(
            _stream_video_frames(video_engine, req, image_bytes),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    try:
        result = await video_engine.generate(
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
        )
    except Exception as e:
        logger.error(f"Video generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Video generation failed")

    if req.response_format == "frames":
        frames_b64 = [base64.b64encode(f).decode("ascii") for f in result.frames]
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
    """SSE generator that streams video frames as they're generated."""
    try:
        async for frame_data in video_engine.stream_frames(
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
            if isinstance(frame_data, bytes):
                frame_b64 = base64.b64encode(frame_data).decode("ascii")
            elif isinstance(frame_data, dict):
                frame_b64 = frame_data
            else:
                continue
            chunk = {
                "created": int(time.time()),
                "data": [{"frame": frame_b64 if isinstance(frame_b64, str) else None, "type": "frame"}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Send done event
        yield f"data: {json.dumps({'created': int(time.time()), 'data': [{'type': 'done'}]})}\n\n"
    except Exception as e:
        logger.error(f"Video streaming error: {e}", exc_info=True)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
