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

from __future__ import annotations

import asyncio
import gc
import io
import logging
import struct
import time
from typing import Any, Optional

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


def _pcm_to_wav(pcm: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE, num_channels: int = 1) -> bytes:
    """Encode 16-bit PCM samples into a WAV byte string."""
    buf = io.BytesIO()
    sample_width = 2
    num_frames = len(pcm)
    data_size = num_frames * num_channels * sample_width

    # RIFF header
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
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
) -> bytes:
    """Generate a WAV header for raw PCM data.

    Useful for streaming: prepend this header to raw PCM bytes so the client
    can play the audio as a standard WAV file.

    Args:
        data_size: Number of bytes of raw PCM data that will follow.
        sample_rate: Sample rate in Hz.
        num_channels: Number of audio channels.

    Returns:
        44-byte WAV header.
    """
    sample_width = 2  # 16-bit
    buf = io.BytesIO()
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
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
    buf.write(struct.pack('<I', data_size))
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
        """Stop and cleanup (oMLX EngineCore.close pattern)."""
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

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _synthesize_sync)

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        temperature: float | None = None,
        instruct: str | None = None,
        **kwargs,
    ):
        """Streaming TTS synthesis — yields audio chunks as they're produced.

        Each yielded chunk is a dict with:
        - "audio": WAV bytes for this chunk
        - "text": Text segment that was synthesized
        - "is_final": True for the last chunk
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

        queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=64)
        sample_rate = getattr(model, "sample_rate", DEFAULT_SAMPLE_RATE)

        def _stream_sync():
            try:
                # Use native stream_generate when available (chatterbox_turbo, pocket_tts)
                if hasattr(model, 'stream_generate') and callable(model.stream_generate):
                    for result in model.stream_generate(**gen_kwargs):
                        audio = np.array(result.audio)
                        wav = _audio_to_wav_bytes(audio, int(sample_rate))
                        segment_text = getattr(result, "text", "")
                        queue.put_nowait({
                            "audio": wav,
                            "text": segment_text,
                            "is_final": False,
                        })
                else:
                    results = model.generate(**gen_kwargs)
                    for result in results:
                        audio = np.array(result.audio)
                        wav = _audio_to_wav_bytes(audio, int(sample_rate))
                        segment_text = getattr(result, "text", "")
                        queue.put_nowait({
                            "audio": wav,
                            "text": segment_text,
                            "is_final": False,
                        })
                queue.put_nowait({"audio": b"", "text": "", "is_final": True})
            except Exception as e:
                logger.error(f"TTS stream error: {e}")
                queue.put_nowait(None)

        loop = asyncio.get_running_loop()
        loop.run_in_executor(self._executor, _stream_sync)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
            if chunk.get("is_final"):
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
        return {
            "model": self._model_path,
            "loaded": self.is_loaded,
            "running": self._running,
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

        # VAD pre-check: skip transcription if no speech detected
        if self._vad is not None:
            try:
                import numpy as np
                with open(audio_path, "rb") as f:
                    raw = f.read()
                # Try to detect WAV header and extract raw PCM
                if raw[:4] == b"RIFF" and len(raw) > 44:
                    pcm = raw[44:]
                else:
                    pcm = raw
                samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                if len(samples) > 0 and not self._vad.is_speech(samples):
                    logger.debug("VAD: no speech detected, skipping transcription")
                    return {"text": "", "language": language or "und", "segments": [], "duration": 0.0}
            except Exception:
                logger.debug("VAD pre-check failed, continuing with transcription", exc_info=True)

        # LID: auto-detect language if not specified
        detected_lang = language
        if language is None:
            try:
                from .lid import detect_language_from_audio
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
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

                return {
                    "text": result.text or "",
                    "language": raw_lang or language,
                    "segments": segments,
                    "duration": getattr(result, "total_time", 0.0),
                }

            return {"text": str(result), "language": language, "segments": [], "duration": 0.0}

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _transcribe_sync)

    def get_stats(self) -> dict:
        return {
            "model": self._model_path,
            "loaded": self.is_loaded,
            "running": self._running,
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

    # 2. mlx-audio fallback — try to load & use inline
    try:
        from mlx_audio.stt.utils import load_model as _load_stt  # type: ignore[import-untyped]

        def _sync_transcribe() -> dict:
            model = _load_stt("mlx-community/whisper-small")  # small, fast
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
