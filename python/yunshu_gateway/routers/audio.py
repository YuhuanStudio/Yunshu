"""OpenAI Audio API compatible router — TTS and ASR endpoints.

Supports:
- /audio/speech — full WAV synthesis (OpenAI-compatible)
- /audio/speech/stream — chunked PCM streaming via SSE
- /audio/transcriptions — ASR (OpenAI-compatible)
- /audio/voices — list available TTS voices
"""

import base64
import json
import os
import tempfile
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from ..engine import get_model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["audio"])

MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

AVAILABLE_VOICES = ["alloy", "chelsie", "ethan", "aiden"]


# ── TTS (Text-to-Speech) ──


class TTSRequest(BaseModel):
    model: str
    input: str
    voice: str = "alloy"
    speed: float = 1.0
    response_format: str = "wav"  # Only "wav" currently supported
    temperature: Optional[float] = None
    instruct: Optional[str] = None  # Voice description for VoiceDesign models


@router.post("/audio/speech", response_class=Response)
async def create_speech(req: TTSRequest) -> Response:
    """Generate speech from text (OpenAI /v1/audio/speech compatible)."""
    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    # Find the TTS engine
    tts_engine = None
    for entry in manager.list_entries():
        if entry.is_loaded and hasattr(entry, '_engine') and entry.engine:
            # Check if it's a TTS engine
            engine_type = type(entry.engine).__name__
            if engine_type == "TTSEngine":
                if req.model in {
                    entry.model_id, entry.model_id.lower(),
                } or entry.model_id.lower() == req.model.lower():
                    tts_engine = entry.engine
                    break

    if tts_engine is None:
        # Try loading by model name
        try:
            tts_engine = await manager.get_engine(req.model)
        except (KeyError, Exception) as e:
            raise HTTPException(
                status_code=404,
                detail=f"TTS model '{req.model}' not found. Error: {e}",
            )

    from yunshu_engine.audio_engine import TTSEngine
    if not isinstance(tts_engine, TTSEngine):
        # Try to find any loaded TTSEngine
        for entry in manager.list_entries():
            if entry.is_loaded and isinstance(getattr(entry, 'engine', None), TTSEngine):
                tts_engine = entry.engine
                break
        if not isinstance(tts_engine, TTSEngine):
            raise HTTPException(
                status_code=404,
                detail=f"No TTS engine available for '{req.model}'",
            )

    try:
        # VoiceDesign models require 'instruct' for voice description
        instruct = req.instruct
        if instruct is None:
            # Provide sensible defaults based on voice name
            voice_defaults = {
                "chelsie": "A cheerful young female voice with clear pronunciation",
                "ethan": "A calm young male voice with warm tone",
                "aiden": "A neutral young voice with moderate pace",
            }
            instruct = voice_defaults.get(req.voice.lower(),
                f"A clear {req.voice} voice with natural intonation")

        wav_bytes = await tts_engine.synthesize(
            text=req.input,
            voice=req.voice,
            speed=req.speed,
            temperature=req.temperature,
            instruct=instruct,
        )
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if req.response_format not in ("wav",):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{req.response_format}'. Only 'wav' is supported.",
        )

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=speech.wav",
        },
    )


@router.post("/audio/speech/stream")
async def stream_speech(req: TTSRequest, request: Request):
    """Stream TTS synthesis as SSE events with audio chunks."""
    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    from yunshu_engine.audio_engine import TTSEngine, DEFAULT_SAMPLE_RATE, make_wav_header

    tts_engine = None
    for entry in manager.list_entries():
        if entry.is_loaded and isinstance(getattr(entry, 'engine', None), TTSEngine):
            tts_engine = entry.engine
            break

    if tts_engine is None:
        raise HTTPException(status_code=404, detail="No TTS engine available")

    async def _audio_stream():
        # Emit a WAV header in the first event so the client can construct
        # a playable stream.  data_size=0 signals "unknown length" which most
        # WAV players handle gracefully.
        wav_hdr_b64 = base64.b64encode(
            make_wav_header(0, DEFAULT_SAMPLE_RATE)
        ).decode("ascii")
        yield f"data: {json.dumps({'type': 'header', 'wav_header': wav_hdr_b64, 'sample_rate': DEFAULT_SAMPLE_RATE})}\n\n"

        # Mirror non-streaming instruct logic
        stream_instruct = req.instruct
        if stream_instruct is None:
            voice_defaults = {
                "chelsie": "A cheerful young female voice with clear pronunciation",
                "ethan": "A calm young male voice with warm tone",
                "aiden": "A neutral young voice with moderate pace",
            }
            stream_instruct = voice_defaults.get(req.voice.lower(),
                f"A clear {req.voice} voice with natural intonation")

        async for chunk in tts_engine.synthesize_stream(
            text=req.input,
            voice=req.voice,
            speed=req.speed,
            temperature=req.temperature,
            instruct=stream_instruct,
        ):
            if chunk.get("is_final"):
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                break
            pcm_b64 = base64.b64encode(chunk["audio"]).decode("ascii")
            yield f"data: {json.dumps({'type': 'audio', 'audio': pcm_b64, 'text': chunk.get('text', '')})}\n\n"

    return StreamingResponse(
        _audio_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Audio-Format": "pcm-s16le",
            "X-Sample-Rate": str(DEFAULT_SAMPLE_RATE),
        },
    )


# ── ASR (Speech-to-Text / Transcriptions) ──


class TranscriptionResponse(BaseModel):
    text: str


@router.post("/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
) -> dict:
    """Transcribe audio file (OpenAI /v1/audio/transcriptions compatible)."""
    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    # Find ASR engine
    from yunshu_engine.audio_engine import ASREngine
    asr_engine = None
    for entry in manager.list_entries():
        if entry.is_loaded and isinstance(getattr(entry, 'engine', None), ASREngine):
            asr_engine = entry.engine
            break

    if asr_engine is None:
        try:
            asr_engine = await manager.get_engine(model)
        except (KeyError, Exception) as e:
            raise HTTPException(
                status_code=404,
                detail=f"ASR model '{model}' not found. Error: {e}",
            )

    if not isinstance(asr_engine, ASREngine):
        raise HTTPException(status_code=404, detail=f"No ASR engine for '{model}'")

    # Save uploaded file to temp location
    content = await file.read()
    if len(content) > MAX_AUDIO_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large: {len(content)} bytes (max {MAX_AUDIO_UPLOAD_BYTES})",
        )

    # Whitelist safe audio extensions
    _SAFE_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm", ".aac"}
    raw_suffix = os.path.splitext(file.filename or "audio.wav")[1].lower()
    suffix = raw_suffix if raw_suffix in _SAFE_EXTENSIONS else ".wav"

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)

        result = await asr_engine.transcribe(
            audio_path=tmp_path,
            language=language,
        )
    except Exception as e:
        logger.error(f"ASR transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return {
        "text": result.get("text", ""),
        "language": result.get("language"),
    }


@router.get("/audio/voices")
async def list_voices() -> dict:
    """List available TTS voices."""
    return {
        "object": "list",
        "data": [{"id": v, "object": "voice"} for v in AVAILABLE_VOICES],
    }
