from __future__ import annotations

"""Yunshu Audio Engine — MLX-native TTS and ASR.

Uses mlx-audio for inference:
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
import base64 as _base64
import contextlib
import gc
import io
import logging
import os as _os
import struct
import tempfile as _tempfile
import threading
import time
from typing import Any

import numpy as np

from .types import EngineConfig

logger = logging.getLogger(__name__)

# Standard WAV header for 16-bit mono PCM
DEFAULT_SAMPLE_RATE = 24000


def _materialize_ref_audio(ref_audio: str | None) -> tuple[str | None, str | None]:
    """Resolve a ref_audio argument to a local filesystem path.

    ``ref_audio`` may be either a path to an existing audio file or a raw
    base64-encoded audio blob (with or without ``data:audio/...;base64,``
    prefix). For base64 input we decode and write to a temporary WAV file so
    that ``mlx_audio.utils.load_audio`` (which only accepts paths) can read it.

    Returns:
        (path, tmpfile_to_delete) — caller is responsible for unlinking
        ``tmpfile_to_delete`` when it is not None.
    """
    if not ref_audio:
        return None, None
    # Treat as path if it already names an existing file.
    if _os.path.isfile(ref_audio):
        return ref_audio, None
    # Strip data URL prefix if present.
    payload = ref_audio
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    # Heuristic: extremely long strings without filesystem separators are
    # treated as base64 candidates. Otherwise the caller almost certainly
    # supplied a (broken) path and we surface that error unchanged.
    if "/" in payload or "\\" in payload:
        return ref_audio, None
    try:
        audio_bytes = _base64.b64decode(payload, validate=True)
    except Exception:
        # Not valid base64 — let the downstream loader raise the original
        # FileNotFoundError so the user sees a familiar message.
        return ref_audio, None
    if not audio_bytes:
        return ref_audio, None
    fd, tmp_path = _tempfile.mkstemp(prefix="yunshu_refaudio_", suffix=".wav")
    try:
        with _os.fdopen(fd, "wb") as fp:
            fp.write(audio_bytes)
    except Exception:
        with contextlib.suppress(OSError):
            _os.unlink(tmp_path)
        raise
    return tmp_path, tmp_path


def _audio_to_wav_bytes(
    audio: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> bytes:
    """Convert float audio array to WAV bytes (16-bit mono PCM)."""
    audio = np.array(audio).flatten()
    # Guard NaN/Inf before clip — np.clip passes NaN through and NaN.astype(
    # int16) becomes 0 with a RuntimeWarning (silent click + log noise). Inf → ±1.0.
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
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


def _pcm_to_wav(
    pcm: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE, num_channels: int = 1
) -> bytes:
    """Encode 16-bit PCM samples into a WAV byte string."""
    buf = io.BytesIO()
    sample_width = 2
    num_frames = len(pcm)
    data_size = num_frames * num_channels * sample_width

    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", _wav_chunk_size(data_size)))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))  # PCM format
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * num_channels * sample_width))
    buf.write(struct.pack("<H", num_channels * sample_width))
    buf.write(struct.pack("<H", sample_width * 8))
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
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
    # 0 signals unknown length for streaming
    effective_data_size = 0 if streaming else data_size

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", _wav_chunk_size(effective_data_size)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))  # PCM
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * num_channels * sample_width))
    buf.write(struct.pack("<H", num_channels * sample_width))
    buf.write(struct.pack("<H", sample_width * 8))
    buf.write(b"data")
    buf.write(struct.pack("<I", effective_data_size))
    return buf.getvalue()


# ── WAV inspection ────────────────────────────────────────────────────────────


def _wav_audio_duration_s(audio_path: str) -> float | None:
    """Return the audio duration (seconds) by walking WAV chunks.

    Tolerates JUNK/LIST/other auxiliary chunks before fmt/data.  Returns
    None for non-WAV files or malformed headers (caller should fall back).
    """
    try:
        with open(audio_path, "rb") as f:
            head = f.read(4096)
    except Exception:
        return None
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return None
    sr = None
    channels = 1
    bits_per_sample = 16
    off = 12
    # We may need to read more than the first 4 KB if the data chunk is
    # earlier than that; but to get sr we usually only need the fmt chunk
    # which is small (16 bytes typical).
    while off + 8 <= len(head):
        chunk_id = head[off : off + 4]
        chunk_sz = int.from_bytes(head[off + 4 : off + 8], "little")
        if chunk_id == b"fmt " and chunk_sz >= 16 and off + 8 + 16 <= len(head):
            channels = int.from_bytes(head[off + 10 : off + 12], "little") or 1
            sr = int.from_bytes(head[off + 12 : off + 16], "little")
            bits_per_sample = int.from_bytes(head[off + 22 : off + 24], "little") or 16
        elif chunk_id == b"data":
            data_size = chunk_sz
            if sr and sr > 0:
                bytes_per_sample = max(1, bits_per_sample // 8)
                return data_size / (bytes_per_sample * channels) / sr
            return None
        off += 8 + chunk_sz + (chunk_sz & 1)
    # Data chunk past the prefix we read — peek further
    if sr is None:
        return None
    try:
        import os

        total = os.path.getsize(audio_path)
        # Approximate: subtract typical header overhead (~46-100 bytes); for
        # int16 mono this gives a close-enough estimate.
        approx_data = max(0, total - 100)
        return approx_data / (max(1, bits_per_sample // 8) * channels) / sr
    except Exception:
        return None


def _decoded_audio_duration_s(audio_path: str) -> float | None:
    """Best-effort ACTUAL audio length (seconds) for compressed formats.

    The transcription `duration` field is defined (OpenAI) as the
    audio's length, but for non-WAV uploads (mp3/flac/ogg/m4a) the code fell back to the
    model's `total_time` — which is inference WALL-CLOCK, not audio duration — so a 3 s
    MP3 that took 1.2 s to decode reported duration 1.2. soundfile (libsndfile) covers
    flac/ogg/wav and mp3 where the lib supports it; m4a/aac may still fail → caller falls
    through to the segment-timestamp / last-resort path. Returns None on any failure.
    """
    try:
        import soundfile as sf

        info = sf.info(audio_path)
        if info.samplerate > 0:
            return info.frames / float(info.samplerate)
    except Exception:
        return None
    return None


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
        manager = _resolve_model_manager()
        if manager is not None:
            for entry in getattr(manager, "_entries", {}).values():
                engine = getattr(entry, "engine", None)
                if engine is not None and hasattr(engine, "list_voices"):
                    return engine.list_voices()
    except Exception:
        logger.debug("voice discovery from model manager failed", exc_info=True)

    return list(DEFAULT_VOICES)


from .active_tracking import ActiveRequestMixin, tracks_active, tracks_active_gen


class TTSEngine(ActiveRequestMixin):
    """MLX-native Text-to-Speech engine using mlx-audio.

    Architecture:
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
        return (
            self._model_path.rsplit("/", 1)[-1]
            if "/" in self._model_path
            else self._model_path
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def sample_rate(self) -> int:
        """The loaded model's TRUE output sample rate (Hz).

        The streaming TTS route hardcoded 24000 in its SSE header
        event + X-Sample-Rate, but models vary (dia=44100, voxcpm/indextts are
        config-driven). Exposing the real rate lets the route advertise it correctly so
        non-24k output isn't played at the wrong speed/pitch.
        """
        return int(getattr(self._model, "sample_rate", DEFAULT_SAMPLE_RATE))

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
            logger.warning(
                f"Strict loading failed for {self._model_path}, retrying with strict=False"
            )
            self._model = load_model(self._model_path, strict=False)

        logger.info(f"TTS engine loaded: {self._model_path}")

    async def start(self) -> None:
        """Start the engine (load model on MLX executor)."""
        if self._model is not None:
            self._running = True
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
        cleanup_model_caches()
        gc.collect()
        loop = asyncio.get_running_loop()
        from .mlx_executor import sync_and_clear_cache

        await loop.run_in_executor(self._executor, sync_and_clear_cache)

    @tracks_active
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

        # Route parameters based on model's generate() signature
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

        # Seed: the API advertises `seed` for reproducibility, but most TTS models'
        # generate() has no seed param and silently swallow it via **kwargs → output was
        # NOT reproducible. Apply it as the global mx seed before generation (and pass it
        # through only if the model genuinely accepts it).
        _seed = gen_kwargs.pop("seed", None)
        if _seed is not None and "seed" in gen_params:
            gen_kwargs["seed"] = _seed

        # Materialize base64 ref_audio (voice cloning) into a temp WAV path
        # since mlx_audio.utils.load_audio only accepts filesystem paths.
        _ref_tmp: str | None = None
        if "ref_audio" in gen_kwargs and gen_kwargs["ref_audio"]:
            resolved, _ref_tmp = _materialize_ref_audio(gen_kwargs["ref_audio"])
            gen_kwargs["ref_audio"] = resolved

        def _synthesize_sync() -> bytes:
            if _seed is not None:
                import mlx.core as _mx

                _mx.random.seed(int(_seed))
            results = model.generate(**gen_kwargs)
            sample_rate = getattr(model, "sample_rate", DEFAULT_SAMPLE_RATE)
            audio_chunks = []
            for result in results:
                audio_chunks.append(np.array(result.audio))
            if not audio_chunks:
                raise RuntimeError("TTS model produced no audio output")
            audio = np.concatenate(audio_chunks, axis=0)
            # OpenAI guarantees `speed`. When the model has no native speed control
            # (so it wasn't forwarded to generate), resample the output to change the
            # playback rate (speed>1 → fewer samples = faster) instead of ignoring it.
            if speed != 1.0 and "speed" not in gen_kwargs:
                import scipy.signal

                audio = scipy.signal.resample(audio, max(1, round(len(audio) / speed)))
            return _audio_to_wav_bytes(audio, int(sample_rate))

        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(self._executor, _synthesize_sync)
        finally:
            if _ref_tmp is not None:
                with contextlib.suppress(OSError):
                    _os.unlink(_ref_tmp)
        elapsed = time.monotonic() - t0
        with self._stats_lock:
            self._synth_count += 1
            self._total_synth_ms += elapsed * 1000.0
        return result

    @tracks_active_gen
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

        # Seed: the API advertises `seed` for reproducibility, but most TTS models'
        # generate() has no seed param and silently swallow it via **kwargs → output was
        # NOT reproducible. Apply it as the global mx seed before generation (and pass it
        # through only if the model genuinely accepts it).
        _seed = gen_kwargs.pop("seed", None)
        if _seed is not None and "seed" in gen_params:
            gen_kwargs["seed"] = _seed

        # Materialize base64 ref_audio (voice cloning) into a temp WAV path.
        _ref_tmp: str | None = None
        if "ref_audio" in gen_kwargs and gen_kwargs["ref_audio"]:
            resolved, _ref_tmp = _materialize_ref_audio(gen_kwargs["ref_audio"])
            gen_kwargs["ref_audio"] = resolved

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
                if _seed is not None:
                    import mlx.core as _mx

                    _mx.random.seed(int(_seed))
                gen_fn = (
                    model.stream_generate
                    if hasattr(model, "stream_generate")
                    and callable(model.stream_generate)
                    else model.generate
                )
                _first_chunk = True
                for result in gen_fn(**gen_kwargs):
                    # Check cancel flag between chunks (thread-safe)
                    if _cancel.is_set():
                        logger.info("TTS stream cancelled mid-generation")
                        break
                    audio = np.array(result.audio).flatten()
                    # Propagate the NaN/Inf guard to the streaming
                    # encoder — this third encode site was missed (np.clip passes NaN
                    # through and (NaN*32767).astype(int16) is a garbage sample + a
                    # RuntimeWarning mid-stream). Mirrors _audio_to_wav_bytes.
                    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
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
                        _thread_queue.put_nowait(
                            {
                                "audio": wav_header + raw_bytes,
                                "text": getattr(result, "text", ""),
                                "is_final": False,
                            }
                        )
                        _first_chunk = False
                    else:
                        try:
                            _thread_queue.put(
                                {
                                    "audio": raw_bytes,
                                    "text": getattr(result, "text", ""),
                                    "is_final": False,
                                },
                                timeout=5.0,
                            )
                        except _queue_mod.Full:
                            logger.warning(
                                "TTS stream queue full after timeout -- consumer likely gone"
                            )
                            break
                # Send is_final sentinel — drain one item if full so the client
                # always receives the completion marker and doesn't hang.
                try:
                    _thread_queue.put_nowait(
                        {"audio": b"", "text": "", "is_final": True}
                    )
                except _queue_mod.Full:
                    with contextlib.suppress(_queue_mod.Empty):
                        _thread_queue.get_nowait()
                    with contextlib.suppress(_queue_mod.Full):
                        _thread_queue.put_nowait(
                            {"audio": b"", "text": "", "is_final": True}
                        )
            except Exception as e:
                logger.error(f"TTS stream error: {e}", exc_info=True)
                # Enqueue an ERROR-tagged terminal chunk, NOT a bare None. A bare
                # None tripped synthesize_stream's `if chunk is None: break` drain WITHOUT
                # yielding an is_final marker, so the /v1/audio/speech route never reached its
                # terminal error/done block → a mid-synthesis failure produced a TRUNCATED SSE
                # stream indistinguishable from a successful short clip. The error chunk
                # (is_final=True) is yielded to the consumer, which surfaces a proper error
                # event + [DONE]. The realtime consumer treats empty-audio+is_final as a clean
                # turn end (its finally still emits the terminal audio.done).
                _err = {"audio": b"", "text": "", "error": str(e), "is_final": True}
                try:
                    _thread_queue.put_nowait(_err)
                except _queue_mod.Full:
                    # Queue is full and we can't signal error — drain one
                    # item and retry so the client sees the error sentinel.
                    with contextlib.suppress(_queue_mod.Empty):
                        _thread_queue.get_nowait()
                    with contextlib.suppress(_queue_mod.Full):
                        _thread_queue.put_nowait(_err)

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
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await stream_task
            # Drain remaining queue items to unblock the executor thread
            while True:
                try:
                    _thread_queue.get_nowait()
                except _queue_mod.Empty:
                    break
            # Clean up temp ref_audio file (if any)
            if _ref_tmp is not None:
                with contextlib.suppress(OSError):
                    _os.unlink(_ref_tmp)

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
            "avg_synth_ms": round(total_synth_ms / synth_count, 1)
            if synth_count > 0
            else 0.0,
            "avg_stream_ms": round(total_stream_ms / stream_count, 1)
            if stream_count > 0
            else 0.0,
        }


class ASREngine(ActiveRequestMixin):
    """MLX-native Speech-to-Text engine using mlx-audio.

    Architecture:
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
        # VAD for voice activity detection
        from .vad import create_vad

        self._vad = create_vad()
        # Metrics (protected by _stats_lock for thread safety)
        self._stats_lock = threading.Lock()
        self._transcribe_count = 0
        self._total_transcribe_ms = 0.0

    @property
    def model_name(self) -> str:
        return (
            self._model_path.rsplit("/", 1)[-1]
            if "/" in self._model_path
            else self._model_path
        )

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
            self._running = True
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
        cleanup_model_caches()
        gc.collect()
        loop = asyncio.get_running_loop()
        from .mlx_executor import sync_and_clear_cache

        await loop.run_in_executor(self._executor, sync_and_clear_cache)

    @tracks_active
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
        # Actual audio duration in seconds from WAV header (not inference time).
        # Computed while reading the WAV for VAD; used in the returned dict.
        _audio_duration_s: float | None = None

        # VAD pre-check: skip transcription if no speech detected.
        # Only run VAD on WAV files — non-WAV formats (MP3, FLAC, etc.)
        # are compressed and cannot be decoded by np.frombuffer, which
        # would produce garbage data and false VAD results.
        if self._vad is not None:
            try:
                import numpy as np

                with open(audio_path, "rb") as f:
                    _audio_raw = f.read()
                # Try to detect WAV header and extract raw PCM + sample rate.
                # Real-world WAV files may have JUNK / LIST / etc. chunks before
                # the fmt and data chunks — walk chunks instead of assuming
                # fixed offsets.
                file_sr: int | None = None
                num_channels = 1  # parse from fmt, don't assume mono/16-bit
                bits_per_sample = 16
                is_wav = _audio_raw[:4] == b"RIFF" and _audio_raw[8:12] == b"WAVE"
                pcm = None
                if is_wav:
                    _off = 12  # skip RIFF<size>WAVE
                    data_size = 0
                    while _off + 8 <= len(_audio_raw):
                        chunk_id = _audio_raw[_off : _off + 4]
                        chunk_sz = int.from_bytes(
                            _audio_raw[_off + 4 : _off + 8], "little"
                        )
                        if chunk_id == b"fmt " and chunk_sz >= 16:
                            # PCM fmt: AudioFormat(2) NumChannels(2) SampleRate(4) ByteRate(4) BlockAlign(2) BitsPerSample(2)
                            num_channels = (
                                struct.unpack_from("<H", _audio_raw, _off + 8 + 2)[0]
                                or 1
                            )
                            file_sr = struct.unpack_from(
                                "<I", _audio_raw, _off + 8 + 4
                            )[0]
                            bits_per_sample = (
                                struct.unpack_from("<H", _audio_raw, _off + 8 + 14)[0]
                                or 16
                            )
                        elif chunk_id == b"data":
                            data_size = chunk_sz
                            pcm = _audio_raw[_off + 8 : _off + 8 + chunk_sz]
                            if file_sr and file_sr > 0:
                                # bytes-per-frame = bytes/sample × channels (was
                                # hardcoded /2 mono → stereo/24-bit were off by an
                                # integer factor).
                                _bpf = max(
                                    1, (bits_per_sample // 8) * max(1, num_channels)
                                )
                                _audio_duration_s = data_size / _bpf / file_sr
                            break
                        _off += 8 + chunk_sz + (chunk_sz & 1)  # word-align
                    if pcm is None:
                        pcm = _audio_raw  # last-resort: feed raw bytes to VAD
                else:
                    # Non-WAV: skip VAD — compressed audio cannot be decoded
                    # as raw PCM.  Let the ASR model handle it directly.
                    pass
                # Decode PCM by the ACTUAL bit depth + downmix channels (was
                # hardcoded int16 mono → stereo/24-/32-bit decoded to garbage).
                if pcm is not None and len(pcm) > 0:
                    if bits_per_sample == 32:
                        samples = (
                            np.frombuffer(pcm, dtype=np.int32).astype(np.float32)
                            / 2147483648.0
                        )
                    elif bits_per_sample == 24:
                        # 24-bit was decoded as misaligned int16 → garbage waveform
                        # fed to the VAD gate, which could false-negative and SILENTLY return
                        # an empty transcript on valid speech. Decode 3-byte LE frames →
                        # sign-extended int32, scaled by 2^23.
                        _n24 = len(pcm) // 3
                        if _n24 > 0:
                            _b = (
                                np.frombuffer(pcm[: _n24 * 3], dtype=np.uint8)
                                .reshape(_n24, 3)
                                .astype(np.int32)
                            )
                            _i24 = _b[:, 0] | (_b[:, 1] << 8) | (_b[:, 2] << 16)
                            _i24 = np.where(_i24 >= (1 << 23), _i24 - (1 << 24), _i24)
                            samples = _i24.astype(np.float32) / 8388608.0
                        else:
                            samples = np.array([], dtype=np.float32)
                    elif bits_per_sample == 8:
                        samples = (
                            np.frombuffer(pcm, dtype=np.uint8).astype(np.float32)
                            - 128.0
                        ) / 128.0
                    else:  # 16-bit
                        samples = (
                            np.frombuffer(
                                pcm[: len(pcm) // 2 * 2], dtype=np.int16
                            ).astype(np.float32)
                            / 32768.0
                        )
                    if num_channels > 1 and len(samples) >= num_channels:
                        _n = len(samples) // num_channels * num_channels
                        samples = samples[:_n].reshape(-1, num_channels).mean(axis=1)
                else:
                    samples = np.array([], dtype=np.float32)
                # Resample to VAD's expected sample rate if different
                # (VAD models typically expect 16kHz)
                vad_sr = self._vad.sample_rate
                effective_file_sr = file_sr if file_sr is not None else 16000
                if (
                    effective_file_sr != vad_sr
                    and effective_file_sr > 0
                    and len(samples) > 0
                ):
                    try:
                        import scipy.signal

                        num_samples = int(len(samples) * vad_sr / effective_file_sr)
                        samples = scipy.signal.resample(samples, num_samples)
                    except ImportError:
                        pass  # No scipy — proceed with native rate, VAD may be less accurate
                if len(samples) > 0:
                    # Reset the VAD's adaptive state per file. The
                    # single shared EnergyVAD accumulates a noise floor
                    # (_noise_level = 0.95*_noise_level + 0.05*energy) and latches
                    # _is_speaking; reset() was never called, so a loud/long request
                    # polluted the threshold for the NEXT request — a quieter (but valid)
                    # speech file could then fail the gate and return empty text, or a
                    # left-over speaking latch could flip a silent file to "speech". Each
                    # upload is an independent stream, so state must not carry over.
                    self._vad.reset()
                    frame_samples = int(
                        self._vad.sample_rate * self._vad.frame_duration_ms / 1000
                    )
                    speech_detected = False
                    offset = 0
                    while offset + frame_samples <= len(samples):
                        frame_data = samples[offset : offset + frame_samples]
                        # EnergyVAD.process_frame expects raw int16 bytes.
                        # *32767 not *32768 — a full-scale +1.0 sample * 32768 = 32768
                        # overflows int16 and wraps to -32768 (every other conversion in
                        # this file uses 32767). Benign for the energy VAD but a latent
                        # footgun; match the rest.
                        frame_bytes = (
                            (np.clip(frame_data, -1.0, 1.0) * 32767.0)
                            .astype(np.int16)
                            .tobytes()
                        )
                        vad_result = self._vad.process_frame(
                            frame_bytes, sample_rate=self._vad.sample_rate
                        )
                        if vad_result.is_speech:
                            speech_detected = True
                            break
                        # is_speech only latches after ~3 CONSECUTIVE loud frames
                        # (the streaming turn-detection design), so a short utterance (<~90ms,
                        # e.g. "yes"/"了") never latches → the file returns an empty transcript
                        # on real speech. For a ONE-SHOT file pre-gate, any single frame clearly
                        # above the base energy threshold is enough to proceed to ASR.
                        if vad_result.energy > self._vad.threshold:
                            speech_detected = True
                            break
                        offset += frame_samples
                    if not speech_detected:
                        logger.debug("VAD: no speech detected, skipping transcription")
                        return {
                            "text": "",
                            "language": language or "und",
                            "segments": [],
                            "duration": 0.0,
                        }
            except Exception:
                logger.debug(
                    "VAD pre-check failed, continuing with transcription", exc_info=True
                )

        # LID: the spectral-heuristic LID in lid.py is unreliable and was
        # feeding the raw WAV file (including header bytes) to the FFT,
        # producing garbage results.  Trust the ASR model's own language
        # detection instead — pass `language=None` and use `result.language`.
        # If the caller did provide a language, honor it as a hint.
        detected_lang = language

        model = self._model

        def _transcribe_sync() -> dict:
            gen_kwargs = dict(kwargs)
            # A None temperature means "use the model's own default decoding".
            # For Whisper that's its temperature-FALLBACK schedule (a tuple 0.0,0.2,…,1.0
            # that retries on compression/logprob failure); forcing a scalar (the old router
            # default 0.0) disabled that fallback. Drop the key so the model keeps its
            # default; an explicitly-provided value still flows through.
            if gen_kwargs.get("temperature") is None:
                gen_kwargs.pop("temperature", None)
            if detected_lang:
                gen_kwargs.setdefault("language", detected_lang)
            # Map OpenAI-style transcription params to the model's actual arg names.
            # Whisper's biasing hint is `initial_prompt` (not `prompt`), and word-level
            # timestamps require `word_timestamps=True`. Without this mapping a caller's
            # `prompt` / `timestamp_granularities=["word"]` were silently swallowed by the
            # model's **kwargs and had no effect. Non-whisper ASR models ignore the extras.
            _prompt = gen_kwargs.pop("prompt", None)
            if _prompt and "initial_prompt" not in gen_kwargs:
                gen_kwargs["initial_prompt"] = _prompt
            _granularities = gen_kwargs.pop("timestamp_granularities", None)
            if _granularities and "word" in _granularities:
                gen_kwargs.setdefault("word_timestamps", True)
            result = model.generate(audio_path, **gen_kwargs)

            if hasattr(result, "text"):
                raw_lang = getattr(result, "language", None)
                if isinstance(raw_lang, list):
                    raw_lang = raw_lang[0] if raw_lang else None
                if isinstance(raw_lang, str) and raw_lang.lower() == "none":
                    raw_lang = language

                raw_segs = getattr(result, "segments", None)
                # Parakeet/NeMo return an AlignedResult whose container is
                # `.sentences` (with .start/.end and per-word data on .tokens), NOT
                # `.segments` — so raw_segs was always None and SRT/VTT/verbose_json
                # timestamps came out empty for that whole model family. Synthesize
                # segments from sentences.
                if raw_segs is None and getattr(result, "sentences", None):
                    raw_segs = []
                    for _s in result.sentences:
                        _toks = getattr(_s, "tokens", None) or []
                        raw_segs.append(
                            {
                                "text": getattr(_s, "text", ""),
                                "start": getattr(_s, "start", None),
                                "end": getattr(_s, "end", None),
                                "words": [
                                    {
                                        "word": getattr(_t, "text", ""),
                                        "start": getattr(_t, "start", None),
                                        "end": getattr(_t, "end", None),
                                    }
                                    for _t in _toks
                                ],
                            }
                        )
                segments = []
                if raw_segs:
                    for s in raw_segs:
                        if isinstance(s, dict):
                            seg = dict(s)
                        elif hasattr(s, "__dict__"):
                            seg = vars(s)
                        else:
                            seg = {"text": str(s)}
                        # Normalize timestamp keys to the OpenAI contract
                        # (start/end). mlx-audio ASR models disagree: Whisper emits
                        # start/end, Parakeet/NeMo emit start_time/end_time. Without
                        # this, SRT/VTT subtitle formatters (which read start/end)
                        # collapsed every cue to 00:00:00 for non-Whisper models, and
                        # verbose_json segments carried non-OpenAI keys.
                        if "start" not in seg and "start_time" in seg:
                            seg["start"] = seg["start_time"]
                        if "end" not in seg and "end_time" in seg:
                            seg["end"] = seg["end_time"]
                        segments.append(seg)

                # Resolve language — must always be a string for API contracts.
                # Priority: model result > caller-provided hint > "und" (unknown).
                resolved_lang = raw_lang or language or "und"

                # Duration: prefer ACTUAL audio length over inference wall-clock.
                # Order: WAV-header (from VAD pre-check) → WAV-header here → decode the
                # file (soundfile, covers mp3/flac/ogg) → last segment end timestamp →
                # model.total_time (inference wall-clock — wrong, true last resort only).
                duration = _audio_duration_s
                if duration is None:
                    duration = _wav_audio_duration_s(audio_path)
                if duration is None:
                    duration = _decoded_audio_duration_s(audio_path)
                if duration is None and segments:
                    # Derive from the last segment's end timestamp if the model emitted one.
                    try:
                        _ends = [
                            float(s.get("end", s.get("end_time", 0.0)))
                            for s in segments
                            if isinstance(s, dict)
                        ]
                        _max_end = max(_ends) if _ends else 0.0
                        if _max_end > 0:
                            duration = _max_end
                    except Exception:
                        pass
                if duration is None:
                    duration = getattr(result, "total_time", 0.0)

                # Surface word-level timestamps. word_timestamps=True is
                # passed to the model when timestamp_granularities=["word"], but the
                # result was never plumbed back, so the verbose_json `words` array
                # (audio.py:773) was always empty — the feature looked wired but was
                # a no-op end to end. Collect from a top-level result.words first,
                # else flatten per-segment word lists. Normalize to {word,start,end}.
                def _norm_word(w):
                    if isinstance(w, dict):
                        d = dict(w)
                    elif hasattr(w, "__dict__"):
                        d = vars(w)
                    else:
                        return None
                    text = d.get("word", d.get("text"))
                    if text is None:
                        return None
                    start = d.get("start", d.get("start_time"))
                    end = d.get("end", d.get("end_time"))
                    out = {"word": text}
                    if start is not None:
                        out["start"] = start
                    if end is not None:
                        out["end"] = end
                    return out

                words = []
                _raw_words = getattr(result, "words", None)
                if _raw_words:
                    words = [w for w in (_norm_word(x) for x in _raw_words) if w]
                else:
                    for seg in segments:
                        for x in seg.get("words") or seg.get("word") or []:
                            nw = _norm_word(x)
                            if nw:
                                words.append(nw)

                out = {
                    "text": result.text or "",
                    "language": resolved_lang,
                    "segments": segments,
                    "duration": duration,
                }
                if words:
                    out["words"] = words
                return out

            return {
                "text": str(result),
                "language": language or "und",
                "segments": [],
                "duration": 0.0,
            }

        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self._executor, _transcribe_sync)
        elapsed = time.monotonic() - t0
        with self._stats_lock:
            self._transcribe_count += 1
            self._total_transcribe_ms += elapsed * 1000.0

        # CJK post-processing override: ASR models occasionally misclassify long
        # Chinese audio as English (e.g. Qwen3-ASR on >30s clips). If the output
        # text is dominantly CJK (>30%), override language to "zh". Skip if the
        # caller passed an explicit non-Chinese language hint.
        try:
            text = result.get("text", "") if isinstance(result, dict) else ""
            current_lang = (
                (result.get("language") or "") if isinstance(result, dict) else ""
            )
            caller_hint_is_zh = (language or "").lower().startswith(
                "zh"
            ) or not language
            if text and caller_hint_is_zh and not current_lang.lower().startswith("zh"):
                # Count CJK Unified Ideographs + extensions (excl. punctuation/digits)
                cjk_chars = sum(
                    1
                    for c in text
                    if "一" <= c <= "鿿"  # CJK Unified Ideographs
                    or "㐀" <= c <= "䶿"  # Extension A
                    or "豈" <= c <= "﫿"  # Compatibility Ideographs
                )
                # Denominator: non-whitespace chars (avoids penalising punctuated text)
                non_ws = sum(1 for c in text if not c.isspace())
                if non_ws > 0 and (cjk_chars / non_ws) > 0.30:
                    result["language"] = "zh"
                    segs = result.get("segments")
                    if isinstance(segs, list):
                        for seg in segs:
                            if (
                                isinstance(seg, dict)
                                and seg.get("language")
                                and not str(seg["language"]).lower().startswith("zh")
                            ):
                                seg["language"] = "zh"
        except Exception:
            # Never let language post-processing break a successful transcription
            pass

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
            "avg_transcribe_ms": round(total_transcribe_ms / transcribe_count, 1)
            if transcribe_count > 0
            else 0.0,
        }


# ── Module-level convenience functions ─────────────────────────────────────────


def _resolve_gateway_engine_module():
    """Return the LIVE ``yunshu_gateway.engine`` module, robust to dual imports.

    The app may be launched as either ``yunshu_gateway.main`` or
    ``python.yunshu_gateway.main``. With both the repo root and ``python/`` on
    sys.path, ``yunshu_gateway.engine`` and ``python.yunshu_gateway.engine``
    resolve to DIFFERENT module objects, each with its own ``_model_manager`` /
    ``_engine`` singletons. A plain ``from yunshu_gateway.engine import
    get_model_manager`` inside a yunshu_engine module can therefore hit a
    duplicate, empty manager (or auto-create one) so loaded engines look absent
    — this is why voice-pipeline reported "No ASR engine available" while
    /audio/transcriptions (which uses the gateway's own relative import) found
    the engine fine.

    Probe both already-imported variants via sys.modules (without importing /
    creating anything) and return whichever holds an initialised manager,
    preferring one that actually has registered entries.
    """
    import sys

    fallback = None
    for name in ("python.yunshu_gateway.engine", "yunshu_gateway.engine"):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        mgr = getattr(mod, "_model_manager", None)
        if mgr is not None and getattr(mgr, "_entries", None):
            return mod
        if fallback is None and (
            mgr is not None or getattr(mod, "_engine", None) is not None
        ):
            fallback = mod
    if fallback is not None:
        return fallback
    try:
        import importlib

        return importlib.import_module("yunshu_gateway.engine")
    except Exception:
        return None


def _resolve_model_manager():
    """Return the live model manager (see _resolve_gateway_engine_module)."""
    mod = _resolve_gateway_engine_module()
    if mod is None:
        return None
    try:
        return mod.get_model_manager()
    except Exception:
        return getattr(mod, "_model_manager", None)


def _find_engine(engine_type: str):
    """Locate a loaded engine of *engine_type* via the model manager.

    Returns the engine instance or ``None``.
    """
    try:
        manager = _resolve_model_manager()
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


def _find_asr_engine() -> ASREngine | None:
    return _find_engine("asr")


def _find_tts_engine() -> TTSEngine | None:
    return _find_engine("tts")


# ── Whisper model cache for transcription fallback ─────────────────────────────
_whisper_model_cache = None
_whisper_model_lock = threading.Lock()

# ── Kokoro model cache for synthesis fallback ────────────────────────────────
_kokoro_model_cache = None
_kokoro_model_lock = threading.Lock()


def cleanup_model_caches() -> None:
    """Release cached fallback models (Whisper + Kokoro).

    Called from TTSEngine.stop() and ASREngine.stop() to free GPU memory
    when the audio subsystem shuts down.  The caches are lazily reloaded
    on next use.
    """
    global _whisper_model_cache, _kokoro_model_cache
    with _whisper_model_lock:
        _whisper_model_cache = None
    with _kokoro_model_lock:
        _kokoro_model_cache = None


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
        from mlx_audio.stt.utils import (
            load_model as _load_stt,  # type: ignore[import-untyped]
        )

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
            # Actual audio duration: WAV header → decode (mp3/flac/ogg) → wall-clock
            # last resort (total_time is inference latency, not audio length, so it
            # was wrong for every non-WAV upload here too).
            duration: float = (
                _wav_audio_duration_s(audio_path)
                or _decoded_audio_duration_s(audio_path)
                or 0.0
            )
            if duration == 0.0:
                duration = getattr(result, "total_time", 0.0)
            return {
                "text": text or "",
                "language": lang or language,
                "segments": [],
                "duration": duration,
            }

        loop = asyncio.get_running_loop()
        from .mlx_executor import get_mlx_executor

        return await loop.run_in_executor(get_mlx_executor(), _sync_transcribe)
    except ImportError:
        logger.debug("mlx-audio not available for transcription fallback")
    except Exception as exc:
        logger.warning("mlx-audio transcription fallback failed: %s", exc)

    # 3. No backend → ERROR: previously this returned an EMPTY
    # transcription as a successful result, indistinguishable from "the audio was
    # silent". A missing ASR backend is a failure, not an empty transcript — fail
    # loudly, like ASREngine.transcribe.
    raise RuntimeError(
        "No ASR backend available for transcription — load an ASR model or "
        "install mlx-audio. (Refusing to return an empty transcript as a fake success.)"
    )


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
        from mlx_audio.tts.utils import (
            load_model as _load_tts,  # type: ignore[import-untyped]
        )

        def _get_kokoro_model():
            global _kokoro_model_cache
            if _kokoro_model_cache is not None:
                return _kokoro_model_cache
            with _kokoro_model_lock:
                if _kokoro_model_cache is None:
                    _kokoro_model_cache = _load_tts(
                        "mlx-community/kokoro-82m", strict=False
                    )
                return _kokoro_model_cache

        def _sync_synth() -> bytes:
            model = _get_kokoro_model()
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

    # 3. No backend → ERROR: previously this returned a slab
    # of SILENCE as a successful WAV, masquerading a total backend failure as a
    # 200. A caller (and the user) had no way to tell "the model said nothing"
    # from "there is no TTS model". Fail loudly instead, like TTSEngine.synthesize.
    raise RuntimeError(
        "No TTS backend available for synthesis — load a TTS model or install "
        "mlx-audio. (Refusing to return silence as a fake success.)"
    )
