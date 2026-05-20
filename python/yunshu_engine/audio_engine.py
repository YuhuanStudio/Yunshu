from __future__ import annotations
"""Yunshu Audio Engine — MLX-native TTS and ASR.

Deeply adapted from oMLX's TTSEngine, using mlx-audio for inference:
- Model loading via mlx_audio.tts.utils.load_model / mlx_audio.stt.utils.load_model
- Audio synthesis via model.generate()
- PCM output -> WAV encoding
- All GPU work serialized on the shared MLX executor

Supports Qwen3-TTS, Whisper, and other mlx-audio compatible models.

Public audio helpers:
- transcribe(audio_path) -> dict
- synthesize(text, voice, speed) -> bytes
- list_voices() -> list[str]
- make_wav_header(data_size, sample_rate, num_channels) -> bytes
"""


import asyncio
import gc
import io
import logging
import struct
import threading
import time
from typing import Any

import numpy as np

from .types import EngineConfig

logger = logging.getLogger(__name__)

# Standard WAV header for 16-bit mono PCM
DEFAULT_SAMPLE_RATE = 24000


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    """Convert float audio array to WAV bytes (16-bit mono PCM)."""
    audio = np.array(audio).flatten()
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    return _pcm_to_wav(pcm, sample_rate)


def _wav_chunk_size(data_size: int) -> int:
    """Calculate RIFF chunk size with overflow protection.

    Standard WAV uses 32-bit unsigned ints, so max RIFF chunk size is
    0xFFFFFFFF (4,294,967,295 bytes).  The RIFF size field equals
    ``36 + data_size``.  If the audio exceeds ~4 GB this overflows and
    produces a corrupt header.

    For data that would overflow, raise ``ValueError`` — callers should
    split into multiple files or switch to a streaming format.
    """
    riff_size = 36 + data_size
    if riff_size > 0xFFFFFFFF:
        raise ValueError(
            f"WAV data_size={data_size} exceeds 4 GB limit "
            f"(RIFF chunk would overflow 32-bit field).  "
            f"Split the audio or use a streaming-capable container format."
        )
    return riff_size


def _pcm_to_wav(pcm: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE, num_channels: int = 1) -> bytes:
    """Encode 16-bit PCM samples into a WAV byte string."""
    buf = io.BytesIO()
    sample_width = 2
    num_frames = len(pcm)
    data_size = num_frames * num_channels * sample_width

    # RIFF header
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', _wav_chunk_size(data_size)))
    buf.write(b'WAVE')
    # fmt chunk
    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))  # chunk size
    buf.write(struct.pack('<H', 1))   # PCM format
    buf.write(struct.pack('<H', num_channels))
    buf.write(struct.pack('<I', sample_rate))
    buf.write(struct.pack('<I', sample_rate * num_channels * sample_width))
    buf.write(struct.pack('<H', num_channels * sample_width))
    buf.write(struct.pack('<H', sample_width * 8))
    # data chunk
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(pcm.tobytes())

    return buf.getvalue()


def make_wav_header(
    data_size: int,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    num_channels: int = 1,
    streaming: bool = False,
) -> bytes:
    """Generate a WAV header for raw PCM data.

    Useful for streaming: prepend this header to raw PCM bytes so the client
    can play the audio as a standard WAV file.

    Args:
        data_size: Number of bytes of raw PCM data that will follow.
            Ignored when *streaming* is True.
        sample_rate: Sample rate in Hz.
        num_channels: Number of audio channels.
        streaming: If True, write ``data_size=0`` to signal an unknown-length
            stream.  Some WAV players (and the Yunshu gateway) interpret a
            zero-size data chunk as "read until EOF".  This avoids lying to
            the client about the total length.

    Returns:
        44-byte WAV header.
    """
    sample_width = 2  # 16-bit
    if streaming:
        effective_data_size = 0  # signals unknown length
    else:
        effective_data_size = data_size

    buf = io.BytesIO()
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', _wav_chunk_size(effective_data_size)))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))
    buf.write(struct.pack('<H', 1))  # PCM
    buf.write(struct.pack('<H', num_channels))
    buf.write(struct.pack('<I', sample_rate))
    buf.write(struct.pack('<I', sample_rate * num_channels * sample_width))
    buf.write(struct.pack('<H', num_channels * sample_width))
    buf.write(struct.pack('<H', sample_width * 8))
    buf.write(b'data')
    buf.write(struct.pack('<I', effective_data_size))
    return buf.getvalue()


# ── Voice catalogue ───────────────────────────────────────────────────────────

# Built-in voice names for VoiceDesign-style models.  When mlx-audio or a
# loaded TTS model exposes its own voice list we return that instead.
DEFAULT_VOICES: list[str] = ["alloy", "chelsie", "ethan", "aiden"]


def list_voices() -> list[str]:
    """Return available TTS voice names.

    Resolution order:
    1. If an mlx-audio TTS model is loaded, query its ``voices`` attribute.
    2. Fall back to the built-in DEFAULT_VOICES list.

    Returns:
        List of voice name strings.
    """
    # Try to discover voices from any loaded TTS models via the model manager
    try:
        from ..yunshu_gateway.engine import get_model_manager  # type: ignore[import-not-found]
        manager = get_model_manager()
        if manager is not None:
            for entry in getattr(manager, "_entries", {}).values():
                engine = getattr(entry, "engine", None)
                if engine is not None and hasattr(engine, "list_voices"):
                    return engine.list_voices()
    except Exception:
        logger.debug("voice discovery from model manager failed", exc_info=True)

    return list(DEFAULT_VOICES)


class TTSEngine:
    """MLX-native Text-to-Speech engine using mlx-audio.

    Architecture follows oMLX's TTSEngine:
    - Lazy import of mlx_audio (avoids import errors if not installed)
    - All GPU work on shared MLX executor (prevents Metal stream conflicts)
    - model.generate() returns iterable of audio results
    """

    def __init__(self, model_path: str, config: EngineConfig | None = None) -> None:
        self._model_path = model_path
        self._model = None
        self._running = False
        from .mlx_executor import get_mlx_executor
        self._executor = get_mlx_executor()
        # Metrics (protected by _stats_lock for thread safety)
        self._stats_lock = threading.Lock()
        self._synth_count = 0
        self._stream_count = 0
        self._total_synth_ms = 0.0
        self._total_stream_ms = 0.0

    @property
    def model_name(self) -> str:
        return self._model_path.rsplit("/", 1)[-1] if "/" in self._model_path else self._model_path

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the TTS model (sync, called from executor)."""
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as e:
            raise ImportError(
                "mlx-audio is required for TTS. Install with: pip install mlx-audio"
            ) from e

        try:
            self._model = load_model(self._model_path, strict=True)
        except ValueError:
            logger.warning(f"Strict loading failed for {self._model_path}, retrying with strict=False")
            self._model = load_model(self._model_path, strict=False)

        logger.info(f"TTS engine loaded: {self._model_path}")

    async def start(self) -> None:
        """Start the engine (load model on MLX executor)."""
        if self._model is not None:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self.load)
        self._running = True

    async def stop(self) -> None:
        """Stop and cleanup (oMLX EngineCore.close pattern).

        Idempotent: safe to call multiple times.
        """
        if not self._running and self._model is None:
            return
        self._model = None
        self._running = False
        gc.collect()
        loop = asyncio.get_running_loop()
        from .mlx_executor import sync_and_clear_cache
        await loop.run_in_executor(self._executor, sync_and_clear_cache)

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        temperature: float | None = None,
        instruct: str | None = None,
        **kwargs,
    ) -> bytes:
        """Synthesize speech from text, returns WAV bytes.

        Args:
            text: Text to synthesize
            voice: Voice name or speaker identifier
            speed: Speech speed multiplier
            temperature: Sampling temperature
            instruct: Voice description for VoiceDesign models
                (e.g., "A cheerful young female voice with high pitch")
        """
        if self._model is None:
            raise RuntimeError("Engine not started")

        import inspect

        model = self._model
        gen_kwargs: dict[str, Any] = {"text": text, "verbose": False}

        # Route parameters based on model's generate() signature (oMLX pattern)
        gen_params = inspect.signature(model.generate).parameters
        has_instruct_param = "instruct" in gen_params

        if voice is not None:
            if "voice" in gen_params:
                gen_kwargs["voice"] = voice
            elif has_instruct_param:
                gen_kwargs["instruct"] = voice
        if instruct is not None and has_instruct_param:
            gen_kwargs["instruct"] = instruct
        elif has_instruct_param and "instruct" not in gen_kwargs:
            # VoiceDesign models require instruct — provide sensible default
            gen_kwargs["instruct"] = "A clear, natural-sounding voice"
        if speed != 1.0 and "speed" in gen_params:
            gen_kwargs["speed"] = speed
        if temperature is not None and "temperature" in gen_params:
            gen_kwargs["temperature"] = temperature
        gen_kwargs.update(kwargs)

        def _synthesize_sync() -> bytes:
            results = model.generate(**gen_kwargs)
            sample_rate = getattr(model, "sample_rate", DEFAULT_SAMPLE_RATE)
            audio_chunks = []
            for result in results:
                audio_chunks.append(np.array(result.audio))
            if not audio_chunks:
                raise RuntimeError("TTS model produced no audio output")
            audio = np.concatenate(audio_chunks, axis=0)
            return _audio_to_wav_bytes(audio, int(sample_rate))

        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self._executor, _synthesize_sync)
        elapsed = time.monotonic() - t0
        with self._stats_lock:
            self._synth_count += 1
            self._total_synth_ms += elapsed * 1000.0
        return result

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        temperature: float | None = None,
        instruct: str | None = None,
        cancel_event: asyncio.Event | None = None,
        **kwargs,
    ):
        """Streaming TTS synthesis — yields audio chunks as they're produced.

        Each yielded chunk is a dict with:
        - "audio": WAV bytes for this chunk
        - "text": Text segment that was synthesized
        - "is_final": True for the last chunk

        Args:
            cancel_event: Optional asyncio.Event — when set, aborts the TTS
                generation loop mid-stream.
        """
        if self._model is None:
            raise RuntimeError("Engine not started")

        import inspect

        model = self._model
        gen_kwargs: dict[str, Any] = {"text": text, "verbose": False}

        gen_params = inspect.signature(model.generate).parameters
        has_instruct_param = "instruct" in gen_params

        if voice is not None:
            if "voice" in gen_params:
                gen_kwargs["voice"] = voice
            elif has_instruct_param:
                gen_kwargs["instruct"] = voice
        if instruct is not None and has_instruct_param:
            gen_kwargs["instruct"] = instruct
        elif has_instruct_param and "instruct" not in gen_kwargs:
            gen_kwargs["instruct"] = "A clear, natural-sounding voice"
        if speed != 1.0 and "speed" in gen_params:
            gen_kwargs["speed"] = speed
        if temperature is not None and "temperature" in gen_params:
            gen_kwargs["temperature"] = temperature
        gen_kwargs.update(kwargs)

        import queue as _queue_mod
        _thread_queue: _queue_mod.Queue[dict | None] = _queue_mod.Queue(maxsize=64)
        sample_rate = getattr(model, "sample_rate", DEFAULT_SAMPLE_RATE)
        # Thread-safe cancel flag — asyncio.Event.is_set() reads a bool but
        # calling it from the executor thread is technically unsafe in older
        # Python.  Mirror the state into a threading.Event for safe cross-thread
        # access.
        _cancel = threading.Event()
        if cancel_event is not None and cancel_event.is_set():
            _cancel.set()

        def _stream_sync():
            try:
                gen_fn = (
                    model.stream_generate
                    if hasattr(model, 'stream_generate') and callable(model.stream_generate)
                    else model.generate
                )
                _first_chunk = True
                for result in gen_fn(**gen_kwargs):
                    # Check cancel flag between chunks (thread-safe)
                    if _cancel.is_set():
                        logger.info("TTS stream cancelled mid-generation")
                        break
                    audio = np.array(result.audio).flatten()
                    audio = np.clip(audio, -1.0, 1.0)
                    pcm = (audio * 32767).astype(np.int16)
                    raw_bytes = pcm.tobytes()
                    if _first_chunk:
                        # Send a WAV header with data_size=0 to signal
                        # unknown length.  Clients read until the stream
                        # ends.  This avoids a fabricated size that could
                        # be wrong or overflow the 32-bit RIFF field.
                        wav_header = make_wav_header(
                            data_size=0,
                            sample_rate=int(sample_rate),
                            num_channels=1,
                            streaming=True,
                        )
                        _thread_queue.put_nowait({
                            "audio": wav_header + raw_bytes,
                            "text": getattr(result, "text", ""),
                            "is_final": False,
                        })
                        _first_chunk = False
                    else:
                        try:
                            _thread_queue.put_nowait({
                                "audio": raw_bytes,
                                "text": getattr(result, "text", ""),
                                "is_final": False,
                            })
                        except _queue_mod.Full:
                            logger.warning("TTS stream queue full, dropping chunk")
                # Send is_final sentinel — drain one item if full so the client
                # always receives the completion marker and doesn't hang.
                try:
                    _thread_queue.put_nowait({"audio": b"", "text": "", "is_final": True})
                except _queue_mod.Full:
                    try:
                        _thread_queue.get_nowait()
                    except _queue_mod.Empty:
                        pass
                    try:
                        _thread_queue.put_nowait({"audio": b"", "text": "", "is_final": True})
                    except _queue_mod.Full:
                        pass
            except Exception as e:
                logger.error(f"TTS stream error: {e}", exc_info=True)
                try:
                    _thread_queue.put_nowait(None)
                except _queue_mod.Full:
                    # Queue is full and we can't signal error — drain one
                    # item and retry so the client sees the error sentinel.
                    try:
                        _thread_queue.get_nowait()
                    except _queue_mod.Empty:
                        pass
                    try:
                        _thread_queue.put_nowait(None)
                    except _queue_mod.Full:
                        pass

        loop = asyncio.get_running_loop()
        stream_task = loop.run_in_executor(self._executor, _stream_sync)

        stream_t0 = time.monotonic()
        try:
            while True:
                # Propagate cancel_event to the thread-safe flag so the
                # executor thread can pick it up without touching asyncio.
                if cancel_event is not None and cancel_event.is_set():
                    _cancel.set()
                try:
                    chunk = _thread_queue.get_nowait()
                except _queue_mod.Empty:
                    await asyncio.sleep(0.01)  # Brief yield to event loop
                    continue
                if chunk is None:
                    break
                yield chunk
                if chunk.get("is_final"):
                    break
        finally:
            # Signal the executor thread to stop (in case it hasn't yet)
            _cancel.set()
            stream_elapsed = time.monotonic() - stream_t0
            with self._stats_lock:
                self._stream_count += 1
                self._total_stream_ms += stream_elapsed * 1000.0
            if not stream_task.done():
                stream_task.cancel()
                try:
                    await stream_task
                except (asyncio.CancelledError, Exception):
                    pass
            # Drain remaining queue items to unblock the executor thread
            while True:
                try:
                    _thread_queue.get_nowait()
                except _queue_mod.Empty:
                    break

    def list_voices(self) -> list[str]:
        """Return voices available on this TTS model.

        Checks the loaded model for a ``voices`` attribute.  If the model
        does not expose one, falls back to the module-level DEFAULT_VOICES.
        """
        if self._model is not None:
            model_voices = getattr(self._model, "voices", None)
            if isinstance(model_voices, (list, tuple)) and model_voices:
                return list(model_voices)
        return list(DEFAULT_VOICES)

    def get_stats(self) -> dict:
        with self._stats_lock:
            synth_count = self._synth_count
            stream_count = self._stream_count
            total_synth_ms = self._total_synth_ms
            total_stream_ms = self._total_stream_ms
        return {
            "model": self._model_path,
            "loaded": self.is_loaded,
            "running": self._running,
            "synth_count": synth_count,
            "stream_count": stream_count,
            "total_synth_ms": round(total_synth_ms, 1),
            "total_stream_ms": round(total_stream_ms, 1),
            "avg_synth_ms": round(total_synth_ms / synth_count, 1) if synth_count > 0 else 0.0,
            "avg_stream_ms": round(total_stream_ms / stream_count, 1) if stream_count > 0 else 0.0,
        }


class ASREngine:
    """MLX-native Speech-to-Text engine using mlx-audio.

    Architecture follows oMLX's STTEngine:
    - model.generate() for transcription
    - Lazy import of mlx_audio
    - GPU work on shared executor
    """

    def __init__(self, model_path: str, config: EngineConfig | None = None) -> None:
        self._model_path = model_path
        self._model = None
        self._running = False
        from .mlx_executor import get_mlx_executor
        self._executor = get_mlx_executor()
        # Wave 43: VAD for voice activity detection
        from .vad import create_vad
        self._vad = create_vad()
        # Metrics (protected by _stats_lock for thread safety)
        self._stats_lock = threading.Lock()
        self._transcribe_count = 0
        self._total_transcribe_ms = 0.0

    @property
    def model_name(self) -> str:
        return self._model_path.rsplit("/", 1)[-1] if "/" in self._model_path else self._model_path

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the ASR model (sync)."""
        try:
            from mlx_audio.stt.utils import load_model
        except ImportError as e:
            raise ImportError(
                "mlx-audio is required for ASR. Install with: pip install mlx-audio"
            ) from e

        self._model = load_model(self._model_path)
        logger.info(f"ASR engine loaded: {self._model_path}")

    async def start(self) -> None:
        if self._model is not None:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self.load)
        self._running = True

    async def stop(self) -> None:
        """Stop and cleanup.

        Idempotent: safe to call multiple times.
        """
        if not self._running and self._model is None:
            return
        self._model = None
        self._running = False
        gc.collect()
        loop = asyncio.get_running_loop()
        from .mlx_executor import sync_and_clear_cache
        await loop.run_in_executor(self._executor, sync_and_clear_cache)

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Transcribe audio file. Returns {text, language, segments, duration}."""
        if self._model is None:
            raise RuntimeError("Engine not started")

        # Read audio file once — reuse the bytes for both VAD and LID to avoid
        # double I/O and double memory allocation for large audio files.
        _audio_raw: bytes | None = None

        # VAD pre-check: skip transcription if no speech detected.
        # Only run VAD on WAV files — non-WAV formats (MP3, FLAC, etc.)
        # are compressed and cannot be decoded by np.frombuffer, which
        # would produce garbage data and false VAD results.
        if self._vad is not None:
            try:
                import numpy as np
                with open(audio_path, "rb") as f:
                    _audio_raw = f.read()
                # Try to detect WAV header and extract raw PCM + sample rate
                file_sr: int | None = None
                is_wav = _audio_raw[:4] == b"RIFF"
                if is_wav:
                    # Extract sample rate from the fmt chunk (bytes 24-27)
                    if len(_audio_raw) >= 28 and _audio_raw[12:16] == b"fmt ":
                        file_sr = struct.unpack_from('<I', _audio_raw, 24)[0]
                    # Find the 'data' chunk — skip any extra chunks
                    data_offset = _audio_raw.find(b"data")
                    if data_offset != -1 and len(_audio_raw) > data_offset + 8:
                        data_size = int.from_bytes(_audio_raw[data_offset+4:data_offset+8], "little")
                        pcm = _audio_raw[data_offset+8:data_offset+8+data_size]
                    else:
                        pcm = _audio_raw
                else:
                    # Non-WAV: skip VAD — compressed audio cannot be decoded
                    # as raw PCM.  Let the ASR model handle it directly.
                    pcm = None
                samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0 if pcm is not None else np.array([], dtype=np.float32)
                # Resample to VAD's expected sample rate if different
                # (VAD models typically expect 16kHz)
                vad_sr = self._vad.sample_rate
                effective_file_sr = file_sr if file_sr is not None else 16000
                if effective_file_sr != vad_sr and effective_file_sr > 0 and len(samples) > 0:
                    try:
                        import scipy.signal
                        num_samples = int(len(samples) * vad_sr / effective_file_sr)
                        samples = scipy.signal.resample(samples, num_samples)
                    except ImportError:
                        pass  # No scipy — proceed with native rate, VAD may be less accurate
                if len(samples) > 0:
                    frame_samples = int(self._vad.sample_rate * self._vad.frame_duration_ms / 1000)
                    speech_detected = False
                    offset = 0
                    while offset + frame_samples <= len(samples):
                        frame_data = samples[offset:offset + frame_samples]
                        # EnergyVAD.process_frame expects raw int16 bytes
                        frame_bytes = (frame_data * 32768.0).astype(np.int16).tobytes()
                        vad_result = self._vad.process_frame(frame_bytes, sample_rate=self._vad.sample_rate)
                        if vad_result.is_speech:
                            speech_detected = True
                            break
                        offset += frame_samples
                    if not speech_detected:
                        logger.debug("VAD: no speech detected, skipping transcription")
                        return {"text": "", "language": language or "und", "segments": [], "duration": 0.0}
            except Exception:
                logger.debug("VAD pre-check failed, continuing with transcription", exc_info=True)

        # LID: auto-detect language if not specified
        detected_lang = language
        if language is None:
            try:
                from .lid import detect_language_from_audio
                if _audio_raw is None:
                    with open(audio_path, "rb") as f:
                        _audio_raw = f.read()
                audio_bytes = _audio_raw
                lid_result = detect_language_from_audio(audio_bytes)
                if lid_result.language != "und" and lid_result.confidence > 0.3:
                    detected_lang = lid_result.language
                    logger.debug(f"LID detected language: {detected_lang} (confidence={lid_result.confidence})")
            except Exception:
                logger.debug("LID failed, using default language", exc_info=True)

        model = self._model

        def _transcribe_sync() -> dict:
            gen_kwargs = dict(kwargs)
            if detected_lang:
                gen_kwargs.setdefault("language", detected_lang)
            result = model.generate(audio_path, **gen_kwargs)

            if hasattr(result, "text"):
                raw_lang = getattr(result, "language", None)
                if isinstance(raw_lang, list):
                    raw_lang = raw_lang[0] if raw_lang else None
                if isinstance(raw_lang, str) and raw_lang.lower() == "none":
                    raw_lang = language

                raw_segs = getattr(result, "segments", None)
                segments = []
                if raw_segs:
                    for s in raw_segs:
                        if isinstance(s, dict):
                            segments.append(s)
                        elif hasattr(s, "__dict__"):
                            segments.append(vars(s))
                        else:
                            segments.append({"text": str(s)})

                # Resolve language — must always be a string for API contracts.
                # Priority: model result > caller-provided hint > "und" (unknown).
                resolved_lang = raw_lang or language or "und"

                return {
                    "text": result.text or "",
                    "language": resolved_lang,
                    "segments": segments,
                    "duration": getattr(result, "total_time", 0.0),
                }

            return {"text": str(result), "language": language or "und", "segments": [], "duration": 0.0}

        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self._executor, _transcribe_sync)
        elapsed = time.monotonic() - t0
        with self._stats_lock:
            self._transcribe_count += 1
            self._total_transcribe_ms += elapsed * 1000.0
        return result

    def get_stats(self) -> dict:
        with self._stats_lock:
            transcribe_count = self._transcribe_count
            total_transcribe_ms = self._total_transcribe_ms
        return {
            "model": self._model_path,
            "loaded": self.is_loaded,
            "running": self._running,
            "transcribe_count": transcribe_count,
            "total_transcribe_ms": round(total_transcribe_ms, 1),
            "avg_transcribe_ms": round(total_transcribe_ms / transcribe_count, 1) if transcribe_count > 0 else 0.0,
        }


# ── Module-level convenience functions ─────────────────────────────────────────


def _find_engine(engine_type: str):
    """Locate a loaded engine of *engine_type* via the model manager.

    Returns the engine instance or ``None``.
    """
    try:
        from yunshu_gateway.engine import get_model_manager  # type: ignore[import-not-found]
        manager = get_model_manager()
        if manager is None:
            return None
        for entry in getattr(manager, "_entries", {}).values():
            engine = getattr(entry, "engine", None)
            if engine is None:
                continue
            if engine_type == "asr" and isinstance(engine, ASREngine):
                return engine
            if engine_type == "tts" and isinstance(engine, TTSEngine):
                return engine
    except Exception:
        logger.debug("engine lookup from model manager failed", exc_info=True)
    return None


def _find_asr_engine() -> "ASREngine | None":
    return _find_engine("asr")


def _find_tts_engine() -> "TTSEngine | None":
    return _find_engine("tts")


# ── Whisper model cache for transcription fallback ─────────────────────────────
_whisper_model_cache = None
_whisper_model_lock = threading.Lock()


async def transcribe(audio_path: str, language: str | None = None) -> dict[str, Any]:
    """Transcribe an audio file using the best available ASR engine.

    Resolution order:
    1. A loaded ``ASREngine`` from the model manager.
    2. If *mlx-audio* is installed, create a temporary ASREngine around
       mlx-audio's default whisper model.
    3. As a last resort, attempt a simple wav-header inspection fallback
       that returns empty transcription with metadata.

    Args:
        audio_path: Path to the audio file (wav, mp3, flac, etc.).
        language: Optional ISO-639-1 language hint.

    Returns:
        Dict with keys ``text``, ``language``, ``segments``, ``duration``.
    """
    import os

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1. Loaded ASREngine
    asr = _find_asr_engine()
    if asr is not None and asr.is_loaded:
        return await asr.transcribe(audio_path, language=language)

    # 2. mlx-audio fallback — try to load & use inline (cached model)
    try:
        from mlx_audio.stt.utils import load_model as _load_stt  # type: ignore[import-untyped]

        def _get_whisper_model():
            """Load and cache the whisper model to avoid reloading on every call."""
            global _whisper_model_cache
            if _whisper_model_cache is not None:
                return _whisper_model_cache
            with _whisper_model_lock:
                if _whisper_model_cache is None:
                    _whisper_model_cache = _load_stt("mlx-community/whisper-small")
                return _whisper_model_cache

        def _sync_transcribe() -> dict:
            model = _get_whisper_model()
            result = model.generate(audio_path)
            text = getattr(result, "text", str(result))
            lang = getattr(result, "language", language)
            if isinstance(lang, list):
                lang = lang[0] if lang else language
            return {
                "text": text or "",
                "language": lang or language,
                "segments": [],
                "duration": getattr(result, "total_time", 0.0),
            }

        loop = asyncio.get_running_loop()
        from .mlx_executor import get_mlx_executor
        return await loop.run_in_executor(get_mlx_executor(), _sync_transcribe)
    except ImportError:
        logger.debug("mlx-audio not available for transcription fallback")
    except Exception as exc:
        logger.warning("mlx-audio transcription fallback failed: %s", exc)

    # 3. Minimal fallback — just report that no ASR engine is loaded
    logger.warning(
        "No ASR engine available for transcription of %s. "
        "Install mlx-audio or load an ASR model.",
        audio_path,
    )
    return {
        "text": "",
        "language": language,
        "segments": [],
        "duration": 0.0,
    }


async def synthesize(
    text: str,
    voice: str | None = None,
    speed: float = 1.0,
    temperature: float | None = None,
    instruct: str | None = None,
) -> bytes:
    """Synthesize speech from text using the best available TTS engine.

    Resolution order:
    1. A loaded ``TTSEngine`` from the model manager.
    2. If *mlx-audio* is installed, create a temporary TTSEngine.
    3. Return silence as a last resort.

    Args:
        text: Text to speak.
        voice: Voice name (model-dependent).
        speed: Speed multiplier (default 1.0).
        temperature: Sampling temperature.
        instruct: Voice description for VoiceDesign models.

    Returns:
        WAV bytes (16-bit mono PCM).
    """
    if not text:
        # Return a short silence
        silence = np.zeros(int(DEFAULT_SAMPLE_RATE * 0.1), dtype=np.int16)
        return _pcm_to_wav(silence, DEFAULT_SAMPLE_RATE)

    # 1. Loaded TTSEngine
    tts = _find_tts_engine()
    if tts is not None and tts.is_loaded:
        return await tts.synthesize(
            text=text,
            voice=voice,
            speed=speed,
            temperature=temperature,
            instruct=instruct,
        )

    # 2. mlx-audio fallback
    try:
        from mlx_audio.tts.utils import load_model as _load_tts  # type: ignore[import-untyped]

        def _sync_synth() -> bytes:
            model = _load_tts("mlx-community/kokoro-82m", strict=False)
            gen_kwargs: dict[str, Any] = {"text": text, "verbose": False}
            sr = getattr(model, "sample_rate", DEFAULT_SAMPLE_RATE)
            results = model.generate(**gen_kwargs)
            chunks = [np.array(r.audio) for r in results]
            if not chunks:
                raise RuntimeError("TTS produced no audio")
            audio = np.concatenate(chunks, axis=0)
            return _audio_to_wav_bytes(audio, int(sr))

        loop = asyncio.get_running_loop()
        from .mlx_executor import get_mlx_executor
        return await loop.run_in_executor(get_mlx_executor(), _sync_synth)
    except ImportError:
        logger.debug("mlx-audio not available for synthesis fallback")
    except Exception as exc:
        logger.warning("mlx-audio synthesis fallback failed: %s", exc)

    # 3. Silence fallback
    logger.warning(
        "No TTS engine available for synthesis. "
        "Install mlx-audio or load a TTS model."
    )
    duration_sec = max(0.5, min(len(text) * 0.06, 30.0))  # rough heuristic
    silence = np.zeros(int(DEFAULT_SAMPLE_RATE * duration_sec), dtype=np.int16)
    return _pcm_to_wav(silence, DEFAULT_SAMPLE_RATE)
