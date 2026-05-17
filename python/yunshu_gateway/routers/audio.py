from __future__ import annotations
"""OpenAI Audio API compatible router — TTS and ASR endpoints.

Supports:
- /audio/speech — full WAV synthesis (OpenAI-compatible)
- /audio/speech/stream — chunked PCM streaming via SSE
- /audio/transcriptions — ASR (OpenAI-compatible)
- /audio/voices — list available TTS voices
"""

import asyncio
import base64
import json
import os
import tempfile
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from ..engine import get_model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["audio"])

MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

AVAILABLE_VOICES = ["alloy", "chelsie", "ethan", "aiden"]


async def _extract_audio_from_video(video_path: str) -> str:
    """Extract audio track from a video file using ffmpeg.

    Returns path to a temporary WAV file. Caller is responsible for cleanup.
    """
    fd, audio_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", "-y", audio_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg extraction failed with code {proc.returncode}")
    except FileNotFoundError:
        raise HTTPException(status_code=501, detail="ffmpeg not installed — cannot extract audio from video")

    return audio_path


def _split_text_segments(text: str, max_chars: int = 300) -> list[str]:
    """Split text into segments at sentence/phrase boundaries.

    Prefers splitting at sentence-ending punctuation (.!?) or commas.
    Falls back to word boundaries, then to hard split.
    """
    if len(text) <= max_chars:
        return [text]

    segments = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            segments.append(remaining)
            break

        # Look for sentence boundary within max_chars
        split_pos = -1
        for i in range(min(len(remaining), max_chars), max_chars // 2, -1):
            if i < len(remaining) and remaining[i - 1] in '.!?。！？':
                split_pos = i
                break

        # Fall back to comma or semicolon
        if split_pos == -1:
            for i in range(min(len(remaining), max_chars), max_chars // 2, -1):
                if i < len(remaining) and remaining[i - 1] in ',;，、':
                    split_pos = i
                    break

        # Fall back to word/space boundary
        if split_pos == -1:
            for i in range(min(len(remaining), max_chars), max_chars // 2, -1):
                if i < len(remaining) and remaining[i - 1] in ' \t\n':
                    split_pos = i
                    break

        # Hard split as last resort
        if split_pos == -1:
            split_pos = max_chars

        segments.append(remaining[:split_pos])
        remaining = remaining[split_pos:].lstrip()

    return segments


# ── TTS (Text-to-Speech) ──


class TTSRequest(BaseModel):
    model: str
    input: str
    voice: str = "alloy"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    response_format: str = "wav"  # Only "wav" currently supported
    temperature: Optional[float] = None
    instruct: Optional[str] = None  # Voice description for VoiceDesign models
    # Extended parameters (oMLX pattern)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=32768)
    # Voice cloning parameters (mlx-audio pattern)
    ref_audio: Optional[str] = None  # Reference audio path for voice cloning
    ref_text: Optional[str] = None  # Reference text for voice cloning
    # Additional mlx-audio parameters
    language: Optional[str] = None  # Language code for multilingual TTS
    seed: Optional[int] = None  # Random seed for reproducibility
    # Segmented streaming (oMLX pattern: 300-char chunks)
    segment_size: int = Field(default=300, ge=50, le=2000)

    @model_validator(mode="after")
    def validate_request(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        if not self.input or not self.input.strip():
            raise ValueError("input: field is required and cannot be empty")
        if self.response_format not in ("wav", "mp3", "opus", "aac", "flac"):
            raise ValueError(f"response_format: unsupported format '{self.response_format}'")
        return self


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

    # Validate format before burning GPU time
    if req.response_format not in ("wav",):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{req.response_format}'. Only 'wav' is supported.",
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
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            max_tokens=req.max_tokens,
            ref_audio=req.ref_audio,
            ref_text=req.ref_text,
            language=req.language,
            seed=req.seed,
        )
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Speech synthesis failed")

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

    # Register with request tracker for cancellation support
    import uuid as _uuid
    _tts_id = f"tts-{_uuid.uuid4().hex[:24]}"
    from yunshu_engine.request_tracker import get_request_tracker
    _tts_tracker = get_request_tracker()
    _tts_gen = _tts_tracker.register(_tts_id, req.model)

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

        # Split long text into segments for progressive synthesis (oMLX pattern)
        text = req.input
        if len(text) > req.segment_size:
            segments = _split_text_segments(text, req.segment_size)
        else:
            segments = [text]

        for seg_idx, segment in enumerate(segments):
            if _tts_gen.cancel_event.is_set():
                yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                return
            async for chunk in tts_engine.synthesize_stream(
                text=segment,
                voice=req.voice,
                speed=req.speed,
                temperature=req.temperature,
                instruct=stream_instruct,
                top_k=req.top_k,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                max_tokens=req.max_tokens,
                ref_audio=req.ref_audio,
                ref_text=req.ref_text,
                language=req.language,
                seed=req.seed,
            ):
                if _tts_gen.cancel_event.is_set():
                    yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                    return
                if chunk.get("is_final"):
                    if seg_idx == len(segments) - 1:
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    continue
                pcm_b64 = base64.b64encode(chunk["audio"]).decode("ascii")
                yield f"data: {json.dumps({'type': 'audio', 'audio': pcm_b64, 'text': chunk.get('text', ''), 'segment': seg_idx})}\n\n"

    from ..streaming import with_sse_keepalive

    async def _wrapped_stream():
        try:
            async for event in with_sse_keepalive(
                _audio_stream(),
                http_request=request,
                cancel_event=_tts_gen.cancel_event,
            ):
                yield event.encode("utf-8") if isinstance(event, str) else event
        except MemoryError:
            yield f"data: {json.dumps({'error': {'message': 'Out of GPU memory', 'type': 'memory_error'}})}\n\n".encode("utf-8")
        except Exception as e:
            logger.error(f"TTS streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': {'message': 'TTS synthesis failed', 'type': 'server_error'}})}\n\n".encode("utf-8")
        finally:
            _tts_tracker.unregister(_tts_id)

    return StreamingResponse(
        _wrapped_stream(),
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

    # Whitelist safe audio + video extensions
    _SAFE_AUDIO = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm", ".aac"}
    _VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".ts", ".mts"}
    raw_suffix = os.path.splitext(file.filename or "audio.wav")[1].lower()

    fd, tmp_path = tempfile.mkstemp(suffix=raw_suffix if raw_suffix in _SAFE_AUDIO | _VIDEO_EXTENSIONS else ".wav")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)

        # If video file, extract audio track via ffmpeg
        asr_path = tmp_path
        if raw_suffix in _VIDEO_EXTENSIONS:
            asr_path = await _extract_audio_from_video(tmp_path)

        result = await asr_engine.transcribe(
            audio_path=asr_path,
            language=language,
        )
    except HTTPException:
        raise
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"ASR transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Audio transcription failed")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        # Clean up extracted audio if it was a video
        if asr_path != tmp_path:
            try:
                os.unlink(asr_path)
            except OSError:
                pass

    resp = {
        "text": result.get("text", ""),
        "language": result.get("language"),
    }
    if result.get("segments"):
        resp["segments"] = result["segments"]
    if result.get("duration"):
        resp["duration"] = result["duration"]
    return resp


@router.get("/audio/voices")
async def list_voices() -> dict:
    """List available TTS voices."""
    return {
        "object": "list",
        "data": [{"id": v, "object": "voice"} for v in AVAILABLE_VOICES],
    }


class VoicePipelineRequest(BaseModel):
    """Request for the STT → LLM → TTS pipeline."""
    file: UploadFile = File(...)
    llm_model: str = ""
    voice: str | None = None
    speed: float = 1.0
    llm_temperature: float = 0.7
    llm_max_tokens: int = 256
    system_prompt: str = "You are a helpful voice assistant. Keep responses concise."
    stream: bool = False


@router.post("/audio/voice-pipeline")
async def voice_pipeline(
    request: Request,
    file: UploadFile = File(...),
    llm_model: str = Form(""),
    voice: Optional[str] = Form(None),
    speed: float = Form(1.0),
    llm_temperature: float = Form(0.7),
    llm_max_tokens: int = Form(256),
    system_prompt: str = Form("You are a helpful voice assistant. Keep responses concise."),
    stream: bool = Form(False),
):
    """STT → LLM → TTS end-to-end voice pipeline.

    Accepts audio input, transcribes it, generates an LLM response,
    and synthesizes the response as audio.
    """
    from yunshu_engine.voice_pipeline import VoicePipeline, VoicePipelineConfig

    audio_data = await file.read()
    if len(audio_data) > MAX_AUDIO_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large (max 25MB)")

    # Write to temp file for ASR
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    config = VoicePipelineConfig(
        llm_model=llm_model,
        tts_voice=voice,
        tts_speed=speed,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        system_prompt=system_prompt,
    )
    pipeline = VoicePipeline(config)

    try:
        if stream:
            # Register with request tracker for cancellation support
            import uuid as _uuid
            _vp_id = f"vp-{_uuid.uuid4().hex[:24]}"
            from yunshu_engine.request_tracker import get_request_tracker
            _vp_tracker = get_request_tracker()
            _vp_gen = _vp_tracker.register(_vp_id, llm_model)
            from ..streaming import with_sse_keepalive

            async def _event_stream():
                try:
                    async for event in with_sse_keepalive(
                        (
                            f"data: {json.dumps({'stage': e.stage, 'data': e.data if isinstance(e.data, str) else ''})}\n\n"
                            async for e in pipeline.process_stream(tmp_path)
                        ),
                        http_request=request,
                        cancel_event=_vp_gen.cancel_event,
                    ):
                        yield event.encode("utf-8") if isinstance(event, str) else event
                finally:
                    _vp_tracker.unregister(_vp_id)

            return StreamingResponse(
                _event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        result = await pipeline.process(tmp_path)
        return {
            "text": result.get("text", ""),
            "audio": base64.b64encode(result.get("audio", b"")).decode("ascii") if result.get("audio") else None,
            "transcription": result.get("transcription", {}),
        }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"Voice pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Voice pipeline failed")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── STS (Speech-to-Speech) Endpoints ──


class STSEnhanceRequest(BaseModel):
    audio: str = Field(description="Base64-encoded audio data (WAV format)")
    method: Optional[str] = None  # spectral_gating, deep_filter, minimal
    noise_floor_db: Optional[float] = None

    @model_validator(mode="after")
    def validate_request(self):
        if not self.audio or not self.audio.strip():
            raise ValueError("audio: field is required and cannot be empty")
        return self


class STSSeparateRequest(BaseModel):
    audio: str = Field(description="Base64-encoded audio data (WAV format)")
    source_text: Optional[str] = None  # Text description of source to isolate
    method: Optional[str] = None

    @model_validator(mode="after")
    def validate_request(self):
        if not self.audio or not self.audio.strip():
            raise ValueError("audio: field is required and cannot be empty")
        return self


class STSTransformRequest(BaseModel):
    audio: str = Field(description="Base64-encoded audio data (WAV format)")
    pitch_shift: Optional[float] = None  # Semitones
    formant_ratio: Optional[float] = None  # Formant frequency ratio

    @model_validator(mode="after")
    def validate_request(self):
        if not self.audio or not self.audio.strip():
            raise ValueError("audio: field is required and cannot be empty")
        return self


@router.post("/audio/speech-to-speech/enhance")
async def sts_enhance(req: STSEnhanceRequest, request: Request):
    """Enhance audio quality — noise reduction and dereverberation."""
    try:
        from yunshu_engine.sts_engine import STSEngine
        engine = _get_sts_engine(request)
        audio_bytes = base64.b64decode(req.audio)
        result = await engine.enhance(
            audio_bytes, method=req.method, noise_floor_db=req.noise_floor_db,
        )
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"STS enhance error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Audio enhancement failed")
    return {
        "audio": base64.b64encode(result.audio_data).decode("ascii"),
        "sample_rate": result.sample_rate,
        "method": result.method,
        "metadata": result.metadata,
    }


@router.post("/audio/speech-to-speech/separate")
async def sts_separate(req: STSSeparateRequest, request: Request):
    """Separate audio sources — isolate specific sounds."""
    try:
        from yunshu_engine.sts_engine import STSEngine
        engine = _get_sts_engine(request)
        audio_bytes = base64.b64decode(req.audio)
        result = await engine.separate(
            audio_bytes, source_text=req.source_text, method=req.method,
        )
        return {
            "audio": base64.b64encode(result.audio_data).decode("ascii"),
            "sample_rate": result.sample_rate,
            "method": result.method,
            "metadata": result.metadata,
        }
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"STS separate error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Audio separation failed")


@router.post("/audio/speech-to-speech/transform")
async def sts_transform(req: STSTransformRequest, request: Request):
    """Transform voice characteristics — pitch shifting, formant modification."""
    try:
        from yunshu_engine.sts_engine import STSEngine
        engine = _get_sts_engine(request)
        audio_bytes = base64.b64decode(req.audio)
        result = await engine.transform(
            audio_bytes, pitch_shift=req.pitch_shift, formant_ratio=req.formant_ratio,
        )
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"STS transform error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Audio transform failed")
    return {
        "audio": base64.b64encode(result.audio_data).decode("ascii"),
        "sample_rate": result.sample_rate,
        "method": result.method,
        "metadata": result.metadata,
    }


def _get_sts_engine(request: Request):
    """Get or create the STS engine."""
    from yunshu_engine.sts_engine import STSEngine
    sts = getattr(request.app.state, "sts_engine", None)
    if sts is None:
        sts = STSEngine()
        sts.start()
        request.app.state.sts_engine = sts
    return sts
