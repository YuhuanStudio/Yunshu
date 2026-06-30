"""Omni voice router — exposes OmniEngine (native Qwen3-Omni Thinker+Talker)
as a streaming SSE endpoint.

POST /v1/omni/speech/stream  →  text (+ optional image/audio) in,
SSE stream of {text deltas, audio chunks @24kHz, done-stats} out.

This is the forward differentiation: native speech-in/speech-out from ONE
unified omni model, not an ASR→LLM→TTS cascade. No other MLX server exposes
Qwen3-Omni's Talker audio-out.

Config: when the served model is itself an omni model (has a Talker), the voice
path reuses it — no extra config, no second copy in memory. To point the voice
path at a *different* model than the one served for text, set YUNSHU_OMNI_MODEL.
With no speakable model available the endpoint returns 503 (no placeholder audio).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import tempfile
import urllib.request
from contextlib import suppress
from urllib.parse import urlparse

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["omni"])

# Multimodal-input media (image/audio) for the omni endpoint may arrive as a local
# path, a data: URI, or an http(s) URL. URL/base64 media is decoded to a temp file
# (cleaned up after the stream). Cap the size to bound memory / abuse.
_MEDIA_MAX_BYTES = 64 * 1024 * 1024  # 64 MiB
_MEDIA_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/webm": ".webm",
}


def _media_ext(mime: str, kind: str) -> str:
    return _MEDIA_EXT.get((mime or "").lower(), ".png" if kind == "image" else ".wav")


def _write_temp_media(raw: bytes, ext: str, tmp: list[str]) -> str:
    if len(raw) > _MEDIA_MAX_BYTES:
        raise HTTPException(status_code=413, detail="media exceeds 64 MiB limit")
    fd, path = tempfile.mkstemp(suffix=ext, prefix="yunshu_omni_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
    except Exception:
        with suppress(OSError):
            os.unlink(path)
        raise
    tmp.append(path)
    return path


def _download_media(url: str) -> tuple[bytes, str]:
    """Fetch http(s) media (size-capped). Returns (bytes, content-type)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Yunshu/omni"})
    with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310 — scheme checked
        mime = (resp.headers.get_content_type() or "").lower()
        raw = resp.read(_MEDIA_MAX_BYTES + 1)
    if len(raw) > _MEDIA_MAX_BYTES:
        raise HTTPException(status_code=413, detail="remote media exceeds 64 MiB limit")
    return raw, mime


async def _resolve_media(value: str | None, kind: str, tmp: list[str]) -> str | None:
    """Resolve an image/audio reference to a local file path.

    Accepts a local path (unchanged), a ``data:`` URI, or an ``http(s)`` URL.
    Decoded/downloaded bytes are written to a temp file (recorded in ``tmp`` for
    cleanup after the stream). Bad input raises a clean HTTP 4xx before streaming.
    """
    if not value:
        return value
    v = value.strip()
    if v.startswith("data:"):
        # data:[<mime>][;base64],<payload>
        header, _, payload = v[5:].partition(",")
        if not payload or "base64" not in header:
            raise HTTPException(
                status_code=400, detail=f"{kind} data URI must be base64-encoded"
            )
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as e:
            raise HTTPException(
                status_code=400, detail=f"invalid base64 in {kind} data URI"
            ) from e
        return _write_temp_media(raw, _media_ext(header.split(";")[0], kind), tmp)
    if urlparse(v).scheme in ("http", "https"):
        raw, mime = await asyncio.to_thread(_download_media, v)
        return _write_temp_media(raw, _media_ext(mime, kind), tmp)
    # Plain base64 (no data: prefix) is ambiguous vs a path — require the data: URI
    # form for inline media. Otherwise treat as a local filesystem path (unchanged).
    return v


def _cleanup_media(tmp: list[str]) -> None:
    for p in tmp:
        with suppress(OSError):
            os.unlink(p)


AUDIO_SAMPLE_RATE = 24000  # Qwen3-Omni Talker output

# Module-level singleton (single consumer; one resident omni model).
_omni_engine = None


def _shared_speakable_model():
    """If the gateway's served model is loaded AND has a Talker, return its
    ``(model, processor)`` so the voice path can reuse it — one omni model, two
    endpoints, no second copy in memory. Returns None otherwise.

    For an omni model the served engine is a VLMEngine that loaded the full model
    (Talker included) via the same ``mlx_vlm.load`` OmniEngine would use, so the
    object is directly reusable."""
    try:
        from ..engine import get_engine

        eng = get_engine()
    except Exception:  # noqa: BLE001 - no/By-name engine → no reuse, fall through
        return None
    if eng is None:
        return None
    model = getattr(eng, "_model", None)
    processor = getattr(eng, "_processor", None)
    if (
        model is not None
        and processor is not None
        and getattr(model, "has_talker", False)
    ):
        return model, processor
    return None


def _get_omni_engine():
    global _omni_engine
    if _omni_engine is None:
        from yunshu_engine.omni_engine import OmniEngine

        # Prefer reusing the already-served model (no second copy). Fall back to a
        # separately-configured YUNSHU_OMNI_MODEL only when the served model can't
        # speak (e.g. a text-only main model + a dedicated omni model).
        shared = _shared_speakable_model()
        path = os.environ.get("YUNSHU_OMNI_MODEL")
        if shared is not None:
            model, processor = shared
            _omni_engine = OmniEngine(model=model, processor=processor)
        elif path:
            _omni_engine = OmniEngine(path)
        else:
            raise HTTPException(
                status_code=503,
                detail="No omni model available. Serve a Thinker+Talker model "
                "(e.g. mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit), or set "
                "YUNSHU_OMNI_MODEL to a separate one.",
            )
    return _omni_engine


async def preload_and_warmup() -> None:
    """Boot-time hook: compile the Talker kernels now so the first voice request is
    warm (~4s) instead of cold (~30s). Runs whenever a speakable model is available
    — the served model itself (reuse) or a separate YUNSHU_OMNI_MODEL. Opt out with
    YUNSHU_OMNI_PRELOAD=0. Best-effort — failures are logged, not fatal."""
    if os.environ.get("YUNSHU_OMNI_PRELOAD", "1") == "0":
        return
    if not (os.environ.get("YUNSHU_OMNI_MODEL") or _shared_speakable_model()):
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
    # Multimodal INPUT: a local file path, a data: URI (base64), or an http(s) URL.
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

    # Validate the requested voice up front (before the stream opens) so an
    # unknown speaker returns a clean 400 with the valid set — not a silent
    # wrong-voice fallback, and not a 500 buried in the SSE error event.
    if req.speaker is not None and eng.resolve_speaker(req.speaker) is None:
        from yunshu_engine.omni_engine import _VOICE_ALIASES

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown speaker '{req.speaker}'. Valid speakers: "
                f"{sorted(eng.valid_speakers)}; aliases: {sorted(_VOICE_ALIASES)}"
            ),
        )

    # Resolve multimodal inputs (data:/URL → temp file) BEFORE the stream opens so
    # bad media returns a clean HTTP 4xx, not a buried SSE error. Temp files are
    # cleaned after the stream completes.
    _tmp: list[str] = []
    try:
        image_path = await _resolve_media(req.image_path, "image", _tmp)
        audio_path = await _resolve_media(req.audio_path, "audio", _tmp)
    except HTTPException:
        _cleanup_media(_tmp)
        raise
    except Exception as e:  # noqa: BLE001 — network/decoding failure → clean 400
        _cleanup_media(_tmp)
        raise HTTPException(
            status_code=400, detail=f"could not load input media: {e}"
        ) from e

    async def sse():
        try:
            async for ch in eng.stream(
                req.text,
                image_path=image_path,
                audio_path=audio_path,
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
        finally:
            _cleanup_media(_tmp)

    return StreamingResponse(sse(), media_type="text/event-stream")


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()
