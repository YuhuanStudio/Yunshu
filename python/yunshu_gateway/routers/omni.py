"""Omni voice router — exposes OmniEngine (native Qwen3-Omni Thinker+Talker)
as a streaming SSE endpoint.

POST /v1/omni/speech/stream  →  text (+ optional image/audio) in,
SSE stream of {text deltas, audio chunks @24kHz, done-stats} out.

This is the forward differentiation: native speech-in/speech-out from ONE
unified omni model, not an ASR→LLM→TTS cascade. No other MLX server exposes
Qwen3-Omni's Talker audio-out.

Config: set YUNSHU_OMNI_MODEL to a local Thinker+Talker model path
(e.g. mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit). Without it the endpoint
returns 503 (no honest placeholder audio).
"""

from __future__ import annotations

import base64
import json
import logging
import os

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["omni"])

AUDIO_SAMPLE_RATE = 24000  # Qwen3-Omni Talker output

# Module-level singleton (single consumer; one resident omni model).
_omni_engine = None


def _get_omni_engine():
    global _omni_engine
    if _omni_engine is None:
        path = os.environ.get("YUNSHU_OMNI_MODEL")
        if not path:
            raise HTTPException(
                status_code=503,
                detail="No omni model configured. Set YUNSHU_OMNI_MODEL to a "
                "Thinker+Talker model (e.g. mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit).",
            )
        from yunshu_engine.omni_engine import OmniEngine

        _omni_engine = OmniEngine(path)
    return _omni_engine


async def preload_and_warmup() -> None:
    """Boot-time hook: if YUNSHU_OMNI_MODEL is set (and YUNSHU_OMNI_PRELOAD != "0"),
    load the omni model and compile its kernels now so the first request is warm
    (~4s) instead of cold (~30s). Best-effort — failures are logged, not fatal."""
    if not os.environ.get("YUNSHU_OMNI_MODEL"):
        return
    if os.environ.get("YUNSHU_OMNI_PRELOAD", "1") == "0":
        return
    try:
        eng = _get_omni_engine()
        logger.info("Preloading omni model (warmup at boot)…")
        await eng.warmup()
    except Exception:  # noqa: BLE001
        logger.warning(
            "Omni preload/warmup failed (first request will be cold)", exc_info=True
        )


class OmniSpeechRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    speaker: str | None = None  # Ethan | Chelsie | Aiden | ...
    thinker_max_new_tokens: int | None = Field(default=None, ge=0, le=512)
    # Local file paths for multimodal INPUT (base64/URL decoding is a TODO).
    image_path: str | None = None
    audio_path: str | None = None


def _pcm16_b64(wav_f32: np.ndarray) -> str:
    """float32 [-1,1] mono → int16 little-endian PCM → base64 (for SSE)."""
    clipped = np.clip(wav_f32, -1.0, 1.0)
    return base64.b64encode((clipped * 32767.0).astype("<i2").tobytes()).decode("ascii")


@router.post("/v1/omni/speech/stream")
async def omni_speech_stream(req: OmniSpeechRequest) -> StreamingResponse:
    """Stream native Thinker text + Talker audio as SSE.

    Events: ``{"type":"text","delta":"..."}``, ``{"type":"audio","delta":"<b64 pcm16>","sr":24000}``,
    ``{"type":"done",...}``, then ``data: [DONE]``.
    """
    eng = _get_omni_engine()

    async def sse():
        try:
            async for ch in eng.stream(
                req.text,
                image_path=req.image_path,
                audio_path=req.audio_path,
                speaker=req.speaker,
                thinker_max_new_tokens=req.thinker_max_new_tokens,
            ):
                if ch.kind == "text":
                    yield _sse({"type": "text", "delta": ch.data})
                elif ch.kind == "audio":
                    yield _sse(
                        {
                            "type": "audio",
                            "delta": _pcm16_b64(ch.data),
                            "sr": eng.sample_rate,
                        }
                    )
                elif ch.kind == "done":
                    yield _sse({"type": "done", **ch.data})
            yield b"data: [DONE]\n\n"
        except Exception as e:  # noqa: BLE001
            logger.exception("omni speech stream failed")
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(sse(), media_type="text/event-stream")


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()
