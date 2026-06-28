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
import contextlib
import json
import logging
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from yunshu_engine.audio_engine import list_voices as _list_tts_voices

from ..engine import get_model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["audio"])

MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

AVAILABLE_VOICES = ["alloy", "chelsie", "ethan", "aiden"]

# Content-Type per OpenAI /audio/speech response_format.
_AUDIO_MIME = {
    "mp3": "audio/mpeg",
    # the opus path transcodes with `-f ogg -c:a libopus` → an Ogg-ENCAPSULATED
    # Opus stream (magic "OggS"), so the MIME is audio/ogg (matches OpenAI /audio/speech).
    # "audio/opus" denotes RAW/CMAF Opus packets — strict clients keying on Content-Type
    # mis-handled the Ogg bytes.
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/L16",  # raw 16-bit little-endian samples (OpenAI semantics)
}

# ffmpeg codec/format args per target. None → handled specially (wav/pcm).
_FFMPEG_FMT = {
    "mp3": ["-f", "mp3"],
    "opus": ["-f", "ogg", "-c:a", "libopus"],
    "aac": ["-f", "adts", "-c:a", "aac"],
    "flac": ["-f", "flac"],
}


def _strip_leading_wav_header(data: bytes) -> bytes:
    """Return raw PCM, stripping a leading RIFF/WAVE header if present.

    synthesize_stream prepends a WAV header to its FIRST chunk,
    but the streaming SSE route sends a separate 'header' event, so the audio payloads
    must be pure PCM — otherwise a client concatenating them gets a 44-byte RIFF blob
    mid-stream. Walks chunks to find 'data' (tolerates non-standard header sizes).
    """
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        off = 12
        while off + 8 <= len(data):
            cid = data[off:off + 4]
            sz = int.from_bytes(data[off + 4:off + 8], "little")
            if cid == b"data":
                return data[off + 8:]
            off += 8 + sz + (sz & 1)
        return data[44:]  # fallback: standard 44-byte PCM header
    return data


def _transcode_wav(wav_bytes: bytes, fmt: str) -> tuple[bytes, str, bool]:
    """Convert engine WAV output to the client-requested ``fmt``.

    Returns ``(payload, media_type, transcoded)``. Previously every
    response_format returned raw WAV bytes labelled ``audio/wav`` regardless of
    what the client asked for — a silent format mismatch (an ``mp3`` request got
    WAV bytes). We now actually transcode via ffmpeg, fall back to ``pcm`` by
    stripping the WAV header with the stdlib ``wave`` module, and — only when
    ffmpeg is genuinely unavailable for a compressed format — return honest WAV
    with ``transcoded=False`` so the caller can flag the degradation in a header
    rather than lying about the bytes.
    """
    if fmt == "wav":
        return wav_bytes, _AUDIO_MIME["wav"], True

    if fmt == "pcm":
        # OpenAI 'pcm' = raw 16-bit signed little-endian, header-stripped. Emit the
        # ACTUAL sample rate + channels in the L16 mime (RFC 2586) so the client can
        # play it back at the right speed — non-24k TTS models (e.g. dia=44100) were
        # otherwise unplayable since the bare "audio/L16" carried no rate.
        import io
        import wave
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                _sr = wf.getframerate()
                _ch = wf.getnchannels()
                _mime = f"audio/L16; rate={_sr}; channels={_ch}"
                return wf.readframes(wf.getnframes()), _mime, True
        except Exception:
            logger.warning("pcm extraction failed; returning WAV", exc_info=True)
            return wav_bytes, _AUDIO_MIME["wav"], False

    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg unavailable; cannot transcode WAV→%s, returning WAV", fmt)
        return wav_bytes, _AUDIO_MIME["wav"], False
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             *_FFMPEG_FMT[fmt], "pipe:1"],
            input=wav_bytes, capture_output=True, timeout=30, check=True,
        )
        return proc.stdout, _AUDIO_MIME[fmt], True
    except Exception:
        logger.warning("ffmpeg WAV→%s transcode failed; returning WAV", fmt, exc_info=True)
        return wav_bytes, _AUDIO_MIME["wav"], False


def _resolve_loaded_model_id(manager, requested: str) -> str | None:
    """Return the canonical model_id of a loaded entry matching `requested`.

    Performs case-insensitive matching against registered model_ids and
    returns the canonical id only if the entry is currently loaded.
    Returns None if no loaded entry matches.
    """
    if not requested:
        return None
    requested_lower = requested.lower()
    for entry in manager.list_entries():
        if not entry.is_loaded:
            continue
        if entry.model_id == requested or entry.model_id.lower() == requested_lower:
            return entry.model_id
    return None


def _select_audio_engine(manager, model: str, engine_cls):
    """Pick the loaded engine of type `engine_cls` whose model_id matches `model`.

    The ASR and streaming-TTS paths grabbed the FIRST loaded engine of the
    right type and ignored `model` entirely — so with two ASR (or two TTS) models
    loaded, a request for `whisper` could be served by `qwen3-asr` (wrong model, wrong
    sample-rate). _enforce_no_auto_load only proves the model is loaded; it does NOT
    bind engine selection to it. Match by model_id like the non-streaming create_speech
    already does. When `model` is empty (legacy "any loaded engine" default), fall back
    to the first of type to preserve that behavior.
    """
    first_of_type = None
    model_lower = model.lower() if model else ""
    for entry in manager.list_entries():
        if not (entry.is_loaded and isinstance(getattr(entry, "engine", None), engine_cls)):
            continue
        if first_of_type is None:
            first_of_type = entry.engine
        if model_lower and (entry.model_id == model or entry.model_id.lower() == model_lower):
            return entry.engine
    return first_of_type if not model_lower else None


def _enforce_no_auto_load(manager, model: str) -> None:
    """Reject the request unless `model` is already loaded.

    Bypassed when YUNSHU_ALLOW_AUTO_LOAD env var is truthy (1/true/yes).
    """
    allow = os.environ.get("YUNSHU_ALLOW_AUTO_LOAD", "").strip().lower()
    if allow in ("1", "true", "yes", "on"):
        return
    if not model or not model.strip():
        # Empty model — leave to downstream validators
        return
    if _resolve_loaded_model_id(manager, model) is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"model '{model}' not loaded — POST /v1/models/load first "
                "(or set YUNSHU_ALLOW_AUTO_LOAD=1 to permit on-demand loading)"
            ),
        )


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
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_bytes = await proc.communicate()
        if proc.returncode != 0:
            # Clean up the empty/invalid temp file before raising
            with contextlib.suppress(OSError):
                os.unlink(audio_path)
            stderr_text = (stderr_bytes or b"").decode("utf-8", errors="replace")
            # Detect "no audio stream" — video-only file, not a real error
            lowered = stderr_text.lower()
            if (
                "does not contain any stream" in lowered
                or "output file does not contain any stream" in lowered
                or "stream map" in lowered and "matches no streams" in lowered
            ):
                raise HTTPException(
                    status_code=415,
                    detail=(
                        "Video file contains no audio stream — nothing to transcribe. "
                        "Upload a video with an audio track OR extract audio to .wav first."
                    ),
                )
            # Other ffmpeg failure — surface a short tail of stderr for debugging
            tail = stderr_text.strip().splitlines()[-3:] if stderr_text else []
            detail = "ffmpeg failed to extract audio from video"
            if tail:
                detail = f"{detail}: {' | '.join(tail)[:300]}"
            raise HTTPException(status_code=415, detail=detail)
    except FileNotFoundError:
        # Clean up temp file before raising HTTPException
        with contextlib.suppress(OSError):
            os.unlink(audio_path)
        raise HTTPException(
            status_code=415,
            detail=(
                "Video transcription requires ffmpeg — install via `brew install ffmpeg` "
                "OR upload extracted .wav file directly."
            ),
        ) from None

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
    # temperature was unbounded → NaN/Inf accepted, downstream
    # sampler produced garbage tokens / crashed. Match chat.py bounds.
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    instruct: str | None = None  # Voice description for VoiceDesign models
    # Extended parameters
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=32768)
    # Voice cloning parameters (mlx-audio pattern)
    ref_audio: str | None = None  # Reference audio path for voice cloning
    ref_text: str | None = None  # Reference text for voice cloning
    # Additional mlx-audio parameters
    language: str | None = None  # Language code for multilingual TTS
    seed: int | None = None  # Random seed for reproducibility
    # Segmented streaming (300-char chunks)
    segment_size: int = Field(default=300, ge=50, le=2000)

    @model_validator(mode="after")
    def validate_request(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        if not self.input or not self.input.strip():
            raise ValueError("input: field is required and cannot be empty")
        # OpenAI TTS API limits input to 4096 chars; allow some slack for legitimate use
        if len(self.input) > 32768:
            raise ValueError(
                f"input: maximum 32768 characters, got {len(self.input)}"
            )
        if not self.voice or not self.voice.strip():
            raise ValueError("voice: field is required and cannot be empty")
        # was hard-rejecting anything but "wav".
        # Broke OpenAI SDK which defaults to mp3. Fixed only the
        # OUTER validator; this INNER Pydantic validator was missed.
        _OPENAI_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")
        if self.response_format not in _OPENAI_FORMATS:
            raise ValueError(
                f"response_format: unsupported format '{self.response_format}'. "
                f"Allowed: {', '.join(_OPENAI_FORMATS)}."
            )
        # Reject obviously suspicious ref_audio paths (path traversal / non-local schemes).
        # ref_audio is meant for trusted local voice-clone reference files; treat URI
        # schemes (http, file, ftp, etc.) and traversal sequences as invalid input.
        if self.ref_audio is not None:
            ra = self.ref_audio.strip()
            if not ra:
                raise ValueError("ref_audio: must not be blank")
            if "://" in ra or ra.startswith("file:"):
                raise ValueError("ref_audio: URI schemes are not allowed; provide a local filesystem path")
            if ".." in ra.replace("\\", "/").split("/"):
                raise ValueError("ref_audio: path traversal segments ('..') are not allowed")
            # SECURITY: the '..' check alone let an ABSOLUTE host path
            # (e.g. "/etc/passwd") through to load_audio → arbitrary file read.
            # Reject absolute paths unless the operator opts into local files.
            import os as _os
            if _os.path.isabs(ra) and _os.environ.get(
                "YUNSHU_ALLOW_LOCAL_FILES", "").lower() not in ("1", "true", "yes"):
                raise ValueError(
                    "ref_audio: absolute paths are not allowed "
                    "(set YUNSHU_ALLOW_LOCAL_FILES=1 to permit local files)")
        return self


async def _parse_tts_request(request: Request) -> TTSRequest:
    """Parse a TTS request from either JSON or multipart form body.

    OpenAI's /v1/audio/speech accepts JSON; some clients (and our own WebUI
    wave) post multipart to share the same form-encoded contract used by
    /audio/transcriptions. Accept both, mapping common form aliases (e.g.
    "format" → "response_format") for OpenAI parity.
    """
    ctype = (request.headers.get("content-type") or "").lower()
    if ctype.startswith("multipart/form-data") or ctype.startswith(
        "application/x-www-form-urlencoded"
    ):
        form = await request.form()

        def _f(*names, default=None):
            for n in names:
                if n in form:
                    return form[n]
            return default

        def _to_float(v, default):
            if v is None or v == "":
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"invalid numeric value: {v!r}") from None

        def _to_int(v, default):
            if v is None or v == "":
                return default
            try:
                return int(v)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"invalid integer value: {v!r}") from None

        payload = {
            "model": _f("model", default=""),
            "input": _f("input", "text", default=""),
            "voice": _f("voice", default="alloy"),
            "speed": _to_float(_f("speed"), 1.0),
            # OpenAI uses "response_format"; accept "format" as a permissive alias.
            "response_format": _f("response_format", "format", default="wav"),
            "instruct": _f("instruct"),
            "ref_audio": _f("ref_audio"),
            "ref_text": _f("ref_text"),
            "language": _f("language"),
            "top_k": _to_int(_f("top_k"), 50),
            "top_p": _to_float(_f("top_p"), 0.95),
            "repetition_penalty": _to_float(_f("repetition_penalty"), 1.0),
            "max_tokens": _to_int(_f("max_tokens"), 4096),
            "segment_size": _to_int(_f("segment_size"), 300),
        }
        # Optional numerics — only set when provided so Optional[None] survives.
        if (_t := _f("temperature")) is not None and _t != "":
            payload["temperature"] = _to_float(_t, None)
        if (_s := _f("seed")) is not None and _s != "":
            payload["seed"] = _to_int(_s, None)
        try:
            return TTSRequest(**payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid form body: {exc}") from None

    # Default: JSON
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    try:
        return TTSRequest(**body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid TTS request: {exc}") from None


@router.post("/audio/speech", response_class=Response)
async def create_speech(request: Request) -> Response:
    """Generate speech from text (OpenAI /v1/audio/speech compatible).

    Accepts JSON body (OpenAI default) or multipart/form-data (OpenAI parity
    with /audio/transcriptions). Form fields mirror the JSON schema; "format"
    is accepted as an alias for "response_format".
    """
    req = await _parse_tts_request(request)
    from .models import _check_model_access, _check_permission
    _check_permission(request, "can_infer")
    _check_model_access(request, req.model)
    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    # Reject if requested model is not already loaded (unless auto-load enabled)
    _enforce_no_auto_load(manager, req.model)

    # Find the TTS engine
    tts_engine = None
    for entry in manager.list_entries():
        if entry.is_loaded and hasattr(entry, '_engine') and entry.engine:
            # Check if it's a TTS engine
            engine_type = type(entry.engine).__name__
            if engine_type == "TTSEngine" and (req.model in {
                entry.model_id, entry.model_id.lower(),
            } or entry.model_id.lower() == req.model.lower()):
                tts_engine = entry.engine
                break

    if tts_engine is None:
        # Try loading by model name
        try:
            tts_engine = await manager.get_engine(req.model)
        except (KeyError, Exception) as e:
            logger.error(f"TTS model '{req.model}' load failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=404,
                detail=f"TTS model '{req.model}' not found.",
            ) from None

    from yunshu_engine.audio_engine import TTSEngine
    if not isinstance(tts_engine, TTSEngine):
        # re-select by model_id (was: first loaded TTSEngine, wrong-model
        # sibling of the streaming/ASR keystone) rather than grabbing any of the type.
        _matched = _select_audio_engine(manager, req.model, TTSEngine)
        if _matched is not None:
            tts_engine = _matched
        if not isinstance(tts_engine, TTSEngine):
            raise HTTPException(
                status_code=404,
                detail=f"No TTS engine available for '{req.model}'",
            )

    # OpenAI default response_format is `mp3`. Prior code
    # rejected anything other than `wav` → vanilla `openai` SDK calls of
    # `client.audio.speech.create(...)` returned HTTP 400. Now accept the
    # full OpenAI spec set (mp3/opus/aac/flac/wav/pcm).
    _OPENAI_SUPPORTED = ("mp3", "opus", "aac", "flac", "wav", "pcm")
    if req.response_format not in _OPENAI_SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{req.response_format}'. "
                   f"Allowed: {', '.join(_OPENAI_SUPPORTED)}.",
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
        raise HTTPException(status_code=507, detail="Out of GPU memory") from None
    except ValueError as e:
        # bad-input ValueError → 400 not 500
        logger.warning(f"TTS validation: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Speech synthesis failed") from None

    # Transcode to the requested format (honest WAV fallback if ffmpeg absent).
    payload, media_type, transcoded = _transcode_wav(wav_bytes, req.response_format)
    _ext = "wav" if not transcoded else req.response_format
    _headers = {"Content-Disposition": f"attachment; filename=speech.{_ext}"}
    if req.response_format not in ("wav", "pcm") and not transcoded:
        # Be explicit that we degraded rather than silently mislabel the bytes.
        _headers["X-Yunshu-Audio-Format-Fallback"] = "wav"
    return Response(content=payload, media_type=media_type, headers=_headers)


@router.post("/audio/speech/stream")
async def stream_speech(req: TTSRequest, request: Request):
    """Stream TTS synthesis as SSE events with audio chunks."""
    from .models import _check_model_access, _check_permission
    _check_permission(request, "can_infer")
    _check_model_access(request, req.model)
    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    _enforce_no_auto_load(manager, req.model)

    from yunshu_engine.audio_engine import (
        DEFAULT_SAMPLE_RATE,
        TTSEngine,
        make_wav_header,
    )

    # match by model_id (was: first loaded TTSEngine, ignoring req.model —
    # a `dia` stream could be served by `kokoro`, defeating the per-model rate).
    tts_engine = _select_audio_engine(manager, req.model, TTSEngine)

    if tts_engine is None:
        raise HTTPException(
            status_code=404,
            detail=f"TTS model '{req.model}' not found." if req.model else "No TTS engine available",
        )

    # advertise the model's REAL output rate (was hardcoded 24000
    # in both the SSE header event and X-Sample-Rate — non-24k models like dia=44100
    # played at the wrong speed/pitch).
    _sr = int(getattr(tts_engine, "sample_rate", DEFAULT_SAMPLE_RATE))

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
            make_wav_header(0, _sr)
        ).decode("ascii")
        yield f"data: {json.dumps({'type': 'header', 'wav_header': wav_hdr_b64, 'sample_rate': _sr})}\n\n"
        # The engine's synthesize_stream prepends its OWN WAV header to the first audio
        # chunk OF EACH segment (each segment is a separate synthesize_stream call); we
        # already sent one standalone 'header' event, so strip that embedded header from
        # the first audio chunk of EVERY segment, else clients concatenating the audio
        # payloads get a spurious 44-byte RIFF blob at each segment boundary → audible
        # click/garbage. (Bug: this flag used to be set ONCE before the loop, so only
        # segment 0's header was stripped — every later segment leaked its header.)

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

        # Split long text into segments for progressive synthesis
        text = req.input
        if len(text) > req.segment_size:
            segments = _split_text_segments(text, req.segment_size)
        else:
            segments = [text]

        for seg_idx, segment in enumerate(segments):
            if _tts_gen.cancel_event.is_set():
                yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                return
            _first_audio = True  # strip the embedded WAV header on THIS segment's first chunk
            async for chunk in tts_engine.synthesize_stream(
                text=segment,
                voice=req.voice,
                speed=req.speed,
                temperature=req.temperature,
                instruct=stream_instruct,
                cancel_event=_tts_gen.cancel_event,
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
                if chunk.get("error"):
                    # the engine signalled a mid-synthesis failure (previously a bare
                    # None that silently truncated the stream with NO terminal event). Surface a
                    # proper error event + [DONE] and stop — don't fall through to more segments.
                    # (Checked BEFORE is_final because the error chunk carries both.)
                    yield f"data: {json.dumps({'type': 'error', 'error': {'message': 'TTS synthesis failed', 'type': 'server_error'}})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                if chunk.get("is_final"):
                    if seg_idx == len(segments) - 1:
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                        yield "data: [DONE]\n\n"
                    continue
                _audio_bytes = chunk["audio"]
                if _first_audio:
                    _audio_bytes = _strip_leading_wav_header(_audio_bytes)
                    _first_audio = False
                pcm_b64 = base64.b64encode(_audio_bytes).decode("ascii")
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
            yield f"data: {json.dumps({'error': {'message': 'Out of GPU memory', 'type': 'memory_error'}})}\n\n".encode()
            yield b"data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"TTS streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': {'message': 'TTS synthesis failed', 'type': 'server_error'}})}\n\n".encode()
            yield b"data: [DONE]\n\n"
        finally:
            _tts_tracker.unregister(_tts_id)

    return StreamingResponse(
        _wrapped_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Audio-Format": "pcm-s16le",
            "X-Sample-Rate": str(_sr),
        },
    )


# ── ASR (Speech-to-Text / Transcriptions) ──


class TranscriptionResponse(BaseModel):
    text: str


@router.post("/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(None),
    response_format: str = Form("json"),
    prompt: str | None = Form(None),
    # default None (not 0.0). Forcing a scalar 0.0 on the ASR path disabled
    # Whisper's temperature-FALLBACK decoding (its default is a tuple 0.0,0.2,…,1.0 that
    # retries on compression/logprob failure) — hurting robustness on hard audio. None →
    # the engine omits the param so the model uses its own (fallback) default; an explicit
    # value is still honored.
    temperature: float | None = Form(None, ge=0.0, le=1.0),
    timestamp_granularities: list[str] | None = Form(None),
    # the official OpenAI SDK sends the multipart key literally as
    # `timestamp_granularities[]` (PHP-style array). FastAPI binds by exact key, so
    # without this alias the SDK's request never reached the param → word_timestamps
    # was never enabled → verbose_json.words came back empty. Accept both spellings.
    timestamp_granularities_bracket: list[str] | None = Form(None, alias="timestamp_granularities[]"),
) -> dict:
    """Transcribe audio file (OpenAI /v1/audio/transcriptions compatible)."""
    from .models import _check_model_access, _check_permission
    timestamp_granularities = timestamp_granularities or timestamp_granularities_bracket
    _check_permission(request, "can_infer")
    if not model or not model.strip():
        raise HTTPException(status_code=400, detail="model: field is required and cannot be empty")
    # validate response_format (mirrors TTS/STS). The handler is an
    # if/elif chain with no final else, so an unknown value (a typo, or a format we
    # don't support) silently fell through to the default json shape with a 200 — the
    # client never learned its request was wrong.
    _VALID_ASR_FORMATS = {"json", "text", "srt", "vtt", "verbose_json"}
    if response_format not in _VALID_ASR_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"response_format must be one of {sorted(_VALID_ASR_FORMATS)}, got '{response_format}'",
        )
    _check_model_access(request, model)

    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    _enforce_no_auto_load(manager, model)

    # Find ASR engine — match by model_id (was: first loaded ASREngine,
    # ignoring `model` → a request for whisper served by qwen3-asr when both loaded).
    from yunshu_engine.audio_engine import ASREngine
    asr_engine = _select_audio_engine(manager, model, ASREngine)

    if asr_engine is None:
        try:
            asr_engine = await manager.get_engine(model)
        except (KeyError, Exception) as e:
            logger.error(f"ASR model '{model}' load failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=404,
                detail=f"ASR model '{model}' not found.",
            ) from None

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
    asr_path = tmp_path  # Initialize before try so finally can always access it
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)

        # If video file, extract audio track via ffmpeg
        if raw_suffix in _VIDEO_EXTENSIONS:
            asr_path = await _extract_audio_from_video(tmp_path)

        result = await asr_engine.transcribe(
            audio_path=asr_path,
            language=language,
            prompt=prompt,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
        )
    except HTTPException:
        raise
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory") from None
    except ValueError as e:
        # Caller-supplied validation failures (bad language code, prompt, etc.)
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"ASR transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Audio transcription failed") from None
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        # Clean up extracted audio if it was a video
        if asr_path != tmp_path:
            with contextlib.suppress(OSError):
                os.unlink(asr_path)

    resp = {
        "text": result.get("text", ""),
        "language": result.get("language"),
    }
    if result.get("segments"):
        resp["segments"] = result["segments"]
    if result.get("duration"):
        resp["duration"] = result["duration"]

    # Handle response_format variants per OpenAI API
    if response_format == "verbose_json":
        # verbose_json always includes these fields (even if empty)
        resp.setdefault("task", "transcribe")  # OpenAI verbose_json carries task
        resp.setdefault("segments", [])
        resp.setdefault("duration", 0.0)
        # pad segments to the full OpenAI verbose_json segment schema so strict
        # SDK consumers that read avg_logprob/no_speech_prob/tokens don't get a partial obj.
        resp["segments"] = _normalize_verbose_segments(resp.get("segments"))
        if "words" not in resp and result.get("words"):
            resp["words"] = result["words"]
    elif response_format == "text":
        return Response(content=result.get("text", ""), media_type="text/plain")
    elif response_format == "srt":
        return Response(
            content=_format_srt(result.get("segments", [])),
            media_type="text/plain",
        )
    elif response_format == "vtt":
        return Response(
            content=_format_vtt(result.get("segments", [])),
            media_type="text/vtt",
        )
    return resp


@router.post("/audio/translations")
async def create_translation(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    response_format: str = Form("json"),
    prompt: str | None = Form(None),
    temperature: float = Form(0.0, ge=0.0, le=1.0),
):
    """Translate audio file to English (OpenAI /v1/audio/translations compatible).

    OpenAI semantics: transcribe non-English speech and emit the text in English.
    This requires a translate-capable ASR model (Whisper-style). Yunshu's current
    ASR backends (e.g. Qwen3-ASR) do **not** support cross-lingual translation —
    setting ``language="en"`` only changes the output hint, not the actual decode.

    Rather than silently mis-translating, we return HTTP 501. To use this endpoint,
    load a translate-capable ASR model (Whisper variants) once supported.
    """
    from .models import _check_permission
    _check_permission(request, "can_infer")

    # Discover the currently loaded ASR engine (if any) to give a precise error.
    manager = get_model_manager()
    asr_loaded_id: str | None = None
    if manager is not None:
        try:
            from yunshu_engine.audio_engine import (
                ASREngine,  # local import to avoid cycles
            )
            for entry in manager.list_entries():
                if entry.is_loaded and isinstance(getattr(entry, "engine", None), ASREngine):
                    asr_loaded_id = entry.model_id
                    break
        except Exception:
            asr_loaded_id = None

    raise HTTPException(
        status_code=501,
        detail=(
            "Translation not supported by loaded ASR model"
            + (f" ('{asr_loaded_id}')" if asr_loaded_id else "")
            + ". Yunshu ASR engines (Qwen3-ASR) transcribe in the source language only. "
            "Load a translate-capable ASR model (Whisper) to enable /v1/audio/translations, "
            "or use /v1/audio/transcriptions followed by an LLM translation step."
        ),
    )


@router.get("/audio/voices")
async def list_voices(request: Request) -> dict:
    """List available TTS voices."""
    from .models import _check_permission
    _check_permission(request, "can_infer")
    return {
        "object": "list",
        "data": [{"id": v, "object": "voice"} for v in _list_tts_voices()],
    }


@router.post("/audio/voice-pipeline")
async def voice_pipeline(
    request: Request,
    file: UploadFile = File(...),
    llm_model: str = Form(""),
    voice: str | None = Form(None),
    speed: float = Form(1.0, ge=0.25, le=4.0),
    llm_temperature: float = Form(0.7, ge=0.0, le=2.0),
    llm_max_tokens: int = Form(256, ge=1, le=131072),
    system_prompt: str = Form("You are a helpful voice assistant. Keep responses concise."),
    stream: bool = Form(False),
):
    """STT → LLM → TTS end-to-end voice pipeline.

    Accepts audio input, transcribes it, generates an LLM response,
    and synthesizes the response as audio.
    """
    from .models import _check_model_access, _check_permission
    _check_permission(request, "can_infer")
    if not llm_model or not llm_model.strip():
        raise HTTPException(status_code=400, detail="llm_model: field is required and cannot be empty")
    _check_model_access(request, llm_model)

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
    _streaming_returned = False

    try:
        if stream:
            # Register with request tracker for cancellation support
            import uuid as _uuid
            _vp_id = f"vp-{_uuid.uuid4().hex[:24]}"
            from yunshu_engine.request_tracker import get_request_tracker
            _vp_tracker = get_request_tracker()
            _vp_gen = _vp_tracker.register(_vp_id, llm_model)
            import base64 as _b64

            from ..streaming import with_sse_keepalive

            def _vp_sse(e) -> str:
                # the TTS stage yields audio as BYTES — the old code
                # coerced non-str data to '' and silently dropped all audio (the
                # whole point of the endpoint). Base64-encode bytes instead.
                if isinstance(e.data, (bytes, bytearray)):
                    payload = {"stage": e.stage, "audio_b64": _b64.b64encode(bytes(e.data)).decode("ascii")}
                else:
                    payload = {"stage": e.stage, "data": e.data if isinstance(e.data, str) else ""}
                return f"data: {json.dumps(payload)}\n\n"

            async def _event_stream():
                try:
                    async for event in with_sse_keepalive(
                        (
                            _vp_sse(e)
                            async for e in pipeline.process_stream(tmp_path)
                        ),
                        http_request=request,
                        cancel_event=_vp_gen.cancel_event,
                    ):
                        yield event.encode("utf-8") if isinstance(event, str) else event
                    yield b"data: [DONE]\n\n"
                finally:
                    _vp_tracker.unregister(_vp_id)
                    # Clean up temp file after streaming completes — cannot
                    # use the outer finally because the generator hasn't started
                    # executing when StreamingResponse is returned.
                    with contextlib.suppress(OSError):
                        os.unlink(tmp_path)

            _streaming_returned = True
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
        raise HTTPException(status_code=503, detail=str(e)) from None
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory") from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Voice pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Voice pipeline failed") from None
    finally:
        # Clean up temp file unless the StreamingResponse took ownership.
        # If streaming was requested but setup failed before the return,
        # _streaming_returned is still False and we clean up here.
        if not _streaming_returned:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


def _normalize_verbose_segments(segments: list) -> list[dict]:
    """Pad each transcription segment to the OpenAI verbose_json segment schema.

    OpenAI verbose_json segments carry id/seek/start/end/text/tokens/temperature/
    avg_logprob/compression_ratio/no_speech_prob. The engine emits only whatever the
    underlying mlx-audio model produced (typically text/start/end[/words]), so strict
    OpenAI-SDK consumers that type-validate segments or read avg_logprob/no_speech_prob got
    a partial object. Pad the missing fields with neutral defaults while preserving every
    real value the model provided.
    """
    out = []
    for i, seg in enumerate(segments or []):
        s = dict(seg) if isinstance(seg, dict) else {"text": str(seg)}
        s.setdefault("id", i)
        s.setdefault("seek", 0)
        s.setdefault("start", 0.0)
        s.setdefault("end", 0.0)
        s.setdefault("text", "")
        s.setdefault("tokens", [])
        s.setdefault("temperature", 0.0)
        s.setdefault("avg_logprob", 0.0)
        s.setdefault("compression_ratio", 0.0)
        s.setdefault("no_speech_prob", 0.0)
        out.append(s)
    return out


# ── Subtitle formatters ──


def _format_srt(segments: list[dict]) -> str:
    """Format transcription segments as SRT subtitle format."""
    lines = []
    for i, seg in enumerate(segments, 1):
        # Parakeet/NeMo ASR models emit start_time/end_time, Whisper
        # emits start/end. The engine normalizes now, but fall back here too so a
        # raw non-Whisper segment never collapses the whole cue to 00:00:00.
        start = seg.get("start", seg.get("start_time", 0.0))
        end = seg.get("end", seg.get("end_time", 0.0))
        text = seg.get("text", "")
        start_ts = _seconds_to_srt_timestamp(start)
        end_ts = _seconds_to_srt_timestamp(end)
        lines.append(f"{i}")
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _format_vtt(segments: list[dict]) -> str:
    """Format transcription segments as WebVTT subtitle format."""
    lines = ["WEBVTT", ""]
    for seg in segments:
        # see _format_srt — tolerate Parakeet/NeMo start_time/end_time.
        start = seg.get("start", seg.get("start_time", 0.0))
        end = seg.get("end", seg.get("end_time", 0.0))
        text = seg.get("text", "")
        start_ts = _seconds_to_vtt_timestamp(start)
        end_ts = _seconds_to_vtt_timestamp(end)
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _format_timestamp(seconds: float, sep: str) -> str:
    """Format seconds as HH:MM:SS<sep>mmm for SRT (sep=',') / VTT (sep='.').

    Round ONCE to total milliseconds and derive every field from that one
    integer (mirrors the whisper-standard writer). The old code computed each field
    independently with truncation (int((seconds-int(seconds))*1000)), which drifted by
    up to ~1ms and rendered e.g. 5.999999 as "05,999" instead of "06,000" and 12.555 as
    ",554" instead of ",555". Tolerate a None/str segment start (dict.get
    returns None for a null value) instead of 500-ing the whole srt/vtt transcription.
    """
    try:
        seconds = float(seconds or 0.0)
    except (TypeError, ValueError):
        seconds = 0.0
    ms_total = max(0, round(seconds * 1000))
    h, ms_total = divmod(ms_total, 3_600_000)
    m, ms_total = divmod(ms_total, 60_000)
    s, ms = divmod(ms_total, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _seconds_to_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    return _format_timestamp(seconds, ",")


def _seconds_to_vtt_timestamp(seconds: float) -> str:
    """Convert seconds to WebVTT timestamp format HH:MM:SS.mmm."""
    return _format_timestamp(seconds, ".")


# ── STS (Speech-to-Speech) Endpoints ──


# Supported STS response formats. "json" (default) returns base64 inside JSON
# envelope; "wav"/"mp3"/"pcm16"/"base64" honour OpenAI-style raw responses.
_STS_RESPONSE_FORMATS = {"json", "wav", "mp3", "pcm16", "base64"}


def _validate_sts_response_format(fmt: str) -> str:
    if fmt is None:
        return "json"
    if fmt not in _STS_RESPONSE_FORMATS:
        raise ValueError(
            f"response_format: unsupported format '{fmt}'. "
            f"Must be one of {sorted(_STS_RESPONSE_FORMATS)}."
        )
    return fmt


def _sts_response(result, response_format: str):
    """Render an STS engine result according to the requested response_format.

    - "json"  → JSON envelope with base64 audio (legacy default)
    - "wav"   → raw audio/wav bytes
    - "pcm16" → raw PCM16 bytes (audio/L16 with sample_rate parameter)
    - "mp3"   → audio bytes re-wrapped (engine emits WAV; we transcode via ffmpeg if available)
    - "base64"→ text/plain body containing only the base64 audio string
    """
    audio_bytes = result.audio_data
    sample_rate = result.sample_rate

    if response_format == "json":
        return {
            "audio": base64.b64encode(audio_bytes).decode("ascii"),
            "sample_rate": sample_rate,
            "method": result.method,
            "metadata": result.metadata,
        }
    if response_format == "wav":
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=sts.wav"},
        )
    if response_format == "base64":
        return Response(
            content=base64.b64encode(audio_bytes).decode("ascii"),
            media_type="text/plain",
        )
    if response_format == "pcm16":
        # STSEngine emits a WAV container — strip the 44-byte RIFF header to get
        # raw little-endian PCM16 samples. Fall back to full bytes if header is
        # missing/malformed.
        body = audio_bytes
        if len(body) > 44 and body[:4] == b"RIFF" and body[8:12] == b"WAVE":
            body = body[44:]
        return Response(
            content=body,
            media_type=f"audio/L16; rate={sample_rate}; channels=1",
        )
    if response_format == "mp3":
        # Best-effort: invoke ffmpeg if available; otherwise fall back to WAV.
        import shutil as _shutil
        import subprocess as _subprocess
        if _shutil.which("ffmpeg"):
            try:
                proc = _subprocess.run(
                    ["ffmpeg", "-loglevel", "error", "-f", "wav", "-i", "pipe:0",
                     "-f", "mp3", "-codec:a", "libmp3lame", "-q:a", "4", "pipe:1"],
                    input=audio_bytes, capture_output=True, timeout=30,
                )
                if proc.returncode == 0 and proc.stdout:
                    return Response(
                        content=proc.stdout,
                        media_type="audio/mpeg",
                        headers={"Content-Disposition": "attachment; filename=sts.mp3"},
                    )
            except Exception as exc:
                logger.warning(f"ffmpeg WAV→MP3 transcode failed: {exc}")
        # Fallback — emit WAV with a header advertising the actual content type.
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=sts.wav",
                "X-Requested-Format": "mp3",
                "X-Format-Fallback": "wav",
            },
        )
    # Should never reach here — validator gates the set.
    raise HTTPException(status_code=400, detail=f"Unsupported response_format '{response_format}'")


class STSEnhanceRequest(BaseModel):
    audio: str = Field(description="Base64-encoded audio data (WAV format)")
    method: str | None = None  # spectral_gating, deep_filter, minimal
    noise_floor_db: float | None = None
    response_format: str = "json"

    @model_validator(mode="after")
    def validate_request(self):
        if not self.audio or not self.audio.strip():
            raise ValueError("audio: field is required and cannot be empty")
        # Validate base64 decode won't exceed size limit
        try:
            import base64
            _decoded_len = len(base64.b64decode(self.audio, validate=True))
            if _decoded_len > MAX_AUDIO_UPLOAD_BYTES:
                raise ValueError(f"audio: decoded size ({_decoded_len}) exceeds limit ({MAX_AUDIO_UPLOAD_BYTES})")
        except Exception as e:
            if "exceeds limit" in str(e):
                raise
            raise ValueError("audio: invalid base64 data") from e
        _validate_sts_response_format(self.response_format)
        return self


class STSSeparateRequest(BaseModel):
    audio: str = Field(description="Base64-encoded audio data (WAV format)")
    source_text: str | None = None  # Text description of source to isolate
    method: str | None = None
    response_format: str = "json"

    @model_validator(mode="after")
    def validate_request(self):
        if not self.audio or not self.audio.strip():
            raise ValueError("audio: field is required and cannot be empty")
        try:
            import base64
            _decoded_len = len(base64.b64decode(self.audio, validate=True))
            if _decoded_len > MAX_AUDIO_UPLOAD_BYTES:
                raise ValueError(f"audio: decoded size ({_decoded_len}) exceeds limit ({MAX_AUDIO_UPLOAD_BYTES})")
        except Exception as e:
            if "exceeds limit" in str(e):
                raise
            raise ValueError("audio: invalid base64 data") from e
        _validate_sts_response_format(self.response_format)
        return self


class STSTransformRequest(BaseModel):
    audio: str = Field(description="Base64-encoded audio data (WAV format)")
    pitch_shift: float | None = None  # Semitones
    formant_ratio: float | None = None  # Formant frequency ratio
    response_format: str = "json"

    @model_validator(mode="after")
    def validate_request(self):
        if not self.audio or not self.audio.strip():
            raise ValueError("audio: field is required and cannot be empty")
        try:
            import base64
            _decoded_len = len(base64.b64decode(self.audio, validate=True))
            if _decoded_len > MAX_AUDIO_UPLOAD_BYTES:
                raise ValueError(f"audio: decoded size ({_decoded_len}) exceeds limit ({MAX_AUDIO_UPLOAD_BYTES})")
        except Exception as e:
            if "exceeds limit" in str(e):
                raise
            raise ValueError("audio: invalid base64 data") from e
        _validate_sts_response_format(self.response_format)
        return self


@router.post("/audio/speech-to-speech/enhance")
async def sts_enhance(req: STSEnhanceRequest, request: Request):
    """Enhance audio quality — noise reduction and dereverberation."""
    from .models import _check_permission
    _check_permission(request, "can_infer")
    try:
        engine = _get_sts_engine(request)
        audio_bytes = base64.b64decode(req.audio)
        if len(audio_bytes) > MAX_AUDIO_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Audio too large ({len(audio_bytes)} bytes)")
        result = await engine.enhance(
            audio_bytes, method=req.method, noise_floor_db=req.noise_floor_db,
        )
    except HTTPException:
        raise
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory") from None
    except ValueError as e:
        # Caller-supplied validation failures (bad method, etc.) → 400 not 500
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"STS enhance error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Audio enhancement failed") from None
    return _sts_response(result, req.response_format)


@router.post("/audio/speech-to-speech/separate")
async def sts_separate(req: STSSeparateRequest, request: Request):
    """Separate audio sources — isolate specific sounds."""
    from .models import _check_permission
    _check_permission(request, "can_infer")
    try:
        engine = _get_sts_engine(request)
        audio_bytes = base64.b64decode(req.audio)
        if len(audio_bytes) > MAX_AUDIO_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Audio too large ({len(audio_bytes)} bytes)")
        result = await engine.separate(
            audio_bytes, source_text=req.source_text, method=req.method,
        )
        return _sts_response(result, req.response_format)
    except HTTPException:
        raise
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory") from None
    except ValueError as e:
        # Caller-supplied validation failures (bad source_text, method, etc.) → 400
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"STS separate error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Audio separation failed") from None


@router.post("/audio/speech-to-speech/transform")
async def sts_transform(req: STSTransformRequest, request: Request):
    """Transform voice characteristics — pitch shifting, formant modification."""
    from .models import _check_permission
    _check_permission(request, "can_infer")
    try:
        engine = _get_sts_engine(request)
        audio_bytes = base64.b64decode(req.audio)
        if len(audio_bytes) > MAX_AUDIO_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Audio too large ({len(audio_bytes)} bytes)")
        result = await engine.transform(
            audio_bytes, pitch_shift=req.pitch_shift, formant_ratio=req.formant_ratio,
        )
    except HTTPException:
        raise
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory") from None
    except ValueError as e:
        # Caller-supplied validation failures (bad pitch/formant range) → 400
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"STS transform error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Audio transform failed") from None
    return _sts_response(result, req.response_format)


def _get_sts_engine(request: Request):
    """Get or create the STS engine.

    Registers a shutdown callback on app.state so the engine is stopped
    when the FastAPI application shuts down (via lifespan or atexit).
    """
    from yunshu_engine.sts_engine import STSEngine
    sts = getattr(request.app.state, "sts_engine", None)
    if sts is None:
        sts = STSEngine()
        sts.start()
        request.app.state.sts_engine = sts
        # Register shutdown hook so the engine is cleaned up on app teardown
        _prev_shutdown = getattr(request.app.state, "_sts_shutdown_hook", None)
        if _prev_shutdown is None:
            async def _shutdown_sts():
                engine = getattr(request.app.state, "sts_engine", None)
                if engine is not None:
                    try:
                        engine.stop()
                    except Exception:
                        logger.debug("STS engine shutdown error", exc_info=True)
                    request.app.state.sts_engine = None
            request.app.state._sts_shutdown_hook = _shutdown_sts
    return sts
