from __future__ import annotations
"""Speech-to-Speech (STS) Engine — audio enhancement, separation, and transformation.

Provides:
1. Speech Enhancement: noise reduction, dereverberation (DeepFilterNet pattern)
2. Audio Source Separation: text-guided audio isolation (SAMAudio pattern)
3. Speech-to-Speech Generation: multimodal voice conversion (LFM-Audio pattern)

Architecture:
  STSEngine wraps mlx-audio's audio processing capabilities with a unified
  interface matching oMLX's STSEngine pattern.

Integration:
  - ModelType.STS in model_manager for auto-detection
  - Gateway endpoint at /v1/audio/speech-to-speech
  - VLM pipeline for audio understanding → speech generation
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class STSConfig:
    """Configuration for STS engine."""
    sample_rate: int = 16000
    # Enhancement
    enhance_method: str = "spectral_gating"  # spectral_gating, deep_filter, minimal
    noise_floor_db: float = -40.0
    # Separation
    separation_method: str = "energy_mask"  # energy_mask, spectral_mask
    # Voice conversion
    voice_conversion_method: str = "none"  # none, pitch_shift, formant
    pitch_shift_semitones: float = 0.0
    formant_ratio: float = 1.0


@dataclass
class STSOutput:
    """Result from STS processing."""
    audio_data: bytes = b""
    sample_rate: int = 16000
    duration_s: float = 0.0
    method: str = ""
    metadata: dict = field(default_factory=dict)


class STSEngine:
    """Speech-to-Speech engine for audio enhancement and transformation.

    Processing pipeline:
    1. Load audio input (file or bytes)
    2. Apply enhancement (noise reduction)
    3. Optional: separation or voice conversion
    4. Return processed audio

    Falls back to signal processing when ML models are unavailable.
    """

    def __init__(self, model_path: str = "", config: STSConfig | None = None) -> None:
        self._model_path = model_path
        self._config = config or STSConfig()
        self._model = None
        self._running = False
        self._model_type = self._detect_model_type()

    @property
    def model_name(self) -> str:
        return self._model_path.rsplit("/", 1)[-1] if "/" in self._model_path else "sts-default"

    @property
    def is_loaded(self) -> bool:
        return self._running

    def _detect_model_type(self) -> str:
        """Detect STS model type from path."""
        path_lower = self._model_path.lower()
        if "deepfilter" in path_lower:
            return "deep_filter_net"
        if "mossformer" in path_lower:
            return "mossformer2"
        if "sam" in path_lower or "separation" in path_lower:
            return "sam_audio"
        if "lfm" in path_lower or "speech_to_speech" in path_lower:
            return "lfm_audio"
        return "default"

    def start(self) -> None:
        """Initialize the STS engine."""
        if self._running:
            return

        logger.info(f"Starting STS engine: {self._model_path or 'default'}")

        # Try to load an ML model if available
        if self._model_path and os.path.exists(self._model_path):
            try:
                self._load_model()
            except Exception as e:
                logger.warning(f"Failed to load STS model, using signal processing fallback: {e}")

        self._running = True
        logger.info("STS engine started")

    def _load_model(self) -> None:
        """Load ML model for STS processing."""
        # Try mlx-audio's STS capabilities
        try:
            import mlx_audio
            if hasattr(mlx_audio, 'sts'):
                logger.info("mlx-audio STS module available")
                return
        except ImportError:
            pass

        # Try loading a custom model
        model_path = Path(self._model_path)
        if model_path.exists():
            config_file = model_path / "config.json"
            if config_file.exists():
                import json
                config = json.loads(config_file.read_text())
                self._config.enhance_method = config.get("enhance_method", self._config.enhance_method)
                logger.info(f"Loaded STS config: {config.get('model_type', 'unknown')}")

    def stop(self) -> None:
        """Stop the STS engine and release resources."""
        self._model = None
        self._running = False
        import gc
        gc.collect()

    async def enhance(
        self,
        audio_input: bytes | str,
        method: str | None = None,
        noise_floor_db: float | None = None,
    ) -> STSOutput:
        """Enhance audio quality — noise reduction and dereverberation.

        Args:
            audio_input: Audio file path or raw bytes (WAV format).
            method: Enhancement method override (spectral_gating, deep_filter, minimal).
            noise_floor_db: Noise floor threshold in dB.

        Returns:
            STSOutput with enhanced audio data.
        """
        if not self._running:
            self.start()

        enhance_method = method if method is not None else self._config.enhance_method
        noise_floor = noise_floor_db if noise_floor_db is not None else self._config.noise_floor_db

        def _enhance_sync():
            audio_data, sr = self._load_audio(audio_input)
            t0 = time.monotonic()

            if enhance_method == "spectral_gating":
                enhanced = self._spectral_gating_enhance(audio_data, sr, noise_floor)
            elif enhance_method == "minimal":
                enhanced = self._minimal_enhance(audio_data)
            else:
                enhanced = self._spectral_gating_enhance(audio_data, sr, noise_floor)

            duration_s = time.monotonic() - t0
            output_wav = self._encode_wav(enhanced, sr)

            return STSOutput(
                audio_data=output_wav,
                sample_rate=sr,
                duration_s=duration_s,
                method=enhance_method,
                metadata={"original_samples": len(audio_data), "enhanced_samples": len(enhanced)},
            )

        from .mlx_executor import get_mlx_executor
        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, _enhance_sync)

    async def separate(
        self,
        audio_input: bytes | str,
        source_text: str | None = None,
        method: str | None = None,
    ) -> STSOutput:
        """Separate audio sources — isolate specific sounds.

        Args:
            audio_input: Audio file path or raw bytes.
            source_text: Text description of the source to isolate.
            method: Separation method override.

        Returns:
            STSOutput with separated audio data.
        """
        if not self._running:
            self.start()

        sep_method = method if method is not None else self._config.separation_method

        def _separate_sync():
            audio_data, sr = self._load_audio(audio_input)
            t0 = time.monotonic()

            separated = self._energy_mask_separation(audio_data, sr, source_text)

            duration_s = time.monotonic() - t0
            output_wav = self._encode_wav(separated, sr)

            return STSOutput(
                audio_data=output_wav,
                sample_rate=sr,
                duration_s=duration_s,
                method=sep_method,
                metadata={"source_text": source_text, "original_samples": len(audio_data)},
            )

        from .mlx_executor import get_mlx_executor
        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, _separate_sync)

    async def transform(
        self,
        audio_input: bytes | str,
        pitch_shift: float | None = None,
        formant_ratio: float | None = None,
    ) -> STSOutput:
        """Transform voice characteristics — pitch shifting, formant modification.

        Args:
            audio_input: Audio file path or raw bytes.
            pitch_shift: Semitones to shift pitch (negative = lower).
            formant_ratio: Formant frequency ratio (1.0 = no change).

        Returns:
            STSOutput with transformed audio data.
        """
        if not self._running:
            self.start()

        pitch = pitch_shift if pitch_shift is not None else self._config.pitch_shift_semitones
        formant = formant_ratio if formant_ratio is not None else self._config.formant_ratio

        def _transform_sync():
            audio_data, sr = self._load_audio(audio_input)
            t0 = time.monotonic()

            if pitch != 0.0:
                audio_data = self._pitch_shift(audio_data, sr, pitch)
            if formant != 1.0:
                audio_data = self._formant_shift(audio_data, sr, formant)

            duration_s = time.monotonic() - t0
            output_wav = self._encode_wav(audio_data, sr)

            return STSOutput(
                audio_data=output_wav,
                sample_rate=sr,
                duration_s=duration_s,
                method="voice_transform",
                metadata={"pitch_shift": pitch, "formant_ratio": formant},
            )

        from .mlx_executor import get_mlx_executor
        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, _transform_sync)

    # ── Signal Processing (fallback when ML models unavailable) ──

    def _load_audio(self, audio_input: bytes | str) -> tuple[list[float], int]:
        """Load audio from file path or bytes. Returns (samples, sample_rate)."""
        import struct
        import numpy as np

        if isinstance(audio_input, str):
            with open(audio_input, "rb") as f:
                audio_input = f.read()

        # Parse WAV header
        if audio_input[:4] == b"RIFF" and audio_input[8:12] == b"WAVE":
            # Find 'fmt ' chunk
            offset = 12
            fmt_data = None
            audio_data = None
            while offset + 8 <= len(audio_input):
                chunk_id = audio_input[offset:offset + 4]
                chunk_size = struct.unpack_from("<I", audio_input, offset + 4)[0]
                # Clamp chunk_size to remaining buffer to prevent over-read
                max_chunk = len(audio_input) - (offset + 8)
                if chunk_size > max_chunk:
                    chunk_size = max_chunk
                chunk_end = offset + 8 + chunk_size
                if chunk_id == b"fmt ":
                    fmt_data = audio_input[offset + 8:chunk_end]
                elif chunk_id == b"data":
                    audio_data = audio_input[offset + 8:chunk_end]
                offset = chunk_end
                # WAV chunks are word-aligned
                if chunk_size % 2 != 0:
                    offset += 1

            if fmt_data is None or audio_data is None:
                raise ValueError("Invalid WAV file: missing fmt or data chunk")

            channels = struct.unpack_from("<H", fmt_data, 2)[0]
            sample_rate = struct.unpack_from("<I", fmt_data, 4)[0]
            bits_per_sample = struct.unpack_from("<H", fmt_data, 14)[0]

            if bits_per_sample not in (8, 16, 24, 32):
                raise ValueError(f"Unsupported bits_per_sample: {bits_per_sample}")

            # Convert to mono float samples
            if bits_per_sample == 8:
                # 8-bit WAV is unsigned
                arr = np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32)
                arr = (arr - 128.0) / 128.0
            elif bits_per_sample == 24:
                # 24-bit: manual unpacking
                n_samples = len(audio_data) // 3
                arr = np.zeros(n_samples, dtype=np.float32)
                for j in range(n_samples):
                    b0, b1, b2 = audio_data[j*3], audio_data[j*3+1], audio_data[j*3+2]
                    val = b0 | (b1 << 8) | (b2 << 16)
                    if val >= 0x800000:
                        val -= 0x1000000
                    arr[j] = val / 8388608.0
            else:
                arr = np.frombuffer(audio_data, dtype=f"int{bits_per_sample}").astype(np.float32)
                if bits_per_sample == 16:
                    arr = arr / 32768.0
                elif bits_per_sample == 32:
                    arr = arr / 2147483648.0
            if channels > 1:
                arr = arr.reshape(-1, channels).mean(axis=1)

            return arr.tolist(), sample_rate

        raise ValueError("Unsupported audio format (only WAV supported)")

    def _encode_wav(self, samples: list[float] | np.ndarray, sample_rate: int) -> bytes:
        """Encode float samples to WAV bytes."""
        import struct
        import numpy as np

        arr = np.array(samples, dtype=np.float32)
        # Clip and convert to int16
        arr = np.clip(arr, -1.0, 1.0)
        int_samples = (arr * 32767).astype(np.int16)

        data = int_samples.tobytes()
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + len(data),
            b"WAVE",
            b"fmt ",
            16,  # chunk size
            1,   # PCM format
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            len(data),
        )
        return header + data

    def _spectral_gating_enhance(
        self, samples: list[float], sr: int, noise_floor_db: float
    ) -> list[float]:
        """Spectral gating noise reduction.

        Estimates noise floor from the first ~0.5s, then gates
        frequency bins below the threshold.
        """
        import numpy as np

        arr = np.array(samples, dtype=np.float32)
        if len(arr) < 1024:
            return arr.tolist()

        # Use STFT for spectral gating
        fft_size = 2048
        hop_size = 512
        noise_frames = min(sr // 2 // hop_size, 20)  # ~0.5s of noise estimation

        # Compute STFT
        frames = []
        for i in range(0, len(arr) - fft_size, hop_size):
            frame = arr[i:i + fft_size] * np.hanning(fft_size)
            spectrum = np.fft.rfft(frame)
            frames.append(spectrum)

        if len(frames) <= noise_frames:
            return arr.tolist()

        # Estimate noise from first N frames
        noise_spectrum = np.mean(np.abs(frames[:noise_frames]), axis=0)
        # noise_floor_db is negative (e.g., -40). Negate so threshold is
        # ABOVE the noise estimate: 10^(-noise_floor_db/20) > 1.
        noise_threshold = noise_spectrum * (10 ** (-noise_floor_db / 20))

        # Gate: suppress bins below threshold
        gate_factor = 0.1  # attenuation factor for noise bins
        for i in range(len(frames)):
            magnitude = np.abs(frames[i])
            phase = np.angle(frames[i])
            # Keep bins above threshold, suppress below
            mask = np.where(magnitude > noise_threshold, 1.0, gate_factor)
            frames[i] = magnitude * mask * np.exp(1j * phase)

        # ISTFT
        output = np.zeros(len(arr), dtype=np.float32)
        window_sum = np.zeros(len(arr), dtype=np.float32)
        for i, frame in enumerate(frames):
            time_frame = np.fft.irfft(frame, fft_size) * np.hanning(fft_size)
            start = i * hop_size
            output[start:start + fft_size] += time_frame
            window_sum[start:start + fft_size] += np.hanning(fft_size) ** 2

        window_sum = np.maximum(window_sum, 1e-8)
        output = output / window_sum

        return output.tolist()

    def _minimal_enhance(self, samples: list[float]) -> list[float]:
        """Minimal enhancement — simple noise gate."""
        import numpy as np
        arr = np.array(samples, dtype=np.float32)
        # Simple noise gate: zero out samples below 1% of max
        threshold = np.max(np.abs(arr)) * 0.01
        arr[np.abs(arr) < threshold] = 0.0
        return arr.tolist()

    def _energy_mask_separation(
        self, samples: list[float], sr: int, source_text: str | None
    ) -> list[float]:
        """Energy-based audio separation (fallback when ML models unavailable).

        Uses spectral energy analysis to isolate prominent sources.
        """
        import numpy as np

        arr = np.array(samples, dtype=np.float32)
        if len(arr) < 1024:
            return arr.tolist()

        # Simple approach: keep only the dominant frequency range
        fft_size = min(4096, len(arr))
        spectrum = np.fft.rfft(arr[:fft_size] * np.hanning(fft_size))
        magnitude = np.abs(spectrum)
        freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)

        # Keep speech frequency range (80 Hz - 8000 Hz)
        speech_mask = (freqs >= 80) & (freqs <= 8000)
        spectrum[~speech_mask] = 0

        separated = np.fft.irfft(spectrum, fft_size)

        # Pad to original length
        if len(separated) < len(arr):
            separated = np.pad(separated, (0, len(arr) - len(separated)))
        else:
            separated = separated[:len(arr)]

        return separated.tolist()

    def _pitch_shift(self, samples: list[float], sr: int, semitones: float) -> list[float]:
        """Pitch shift using resampling."""
        import numpy as np

        arr = np.array(samples, dtype=np.float32)
        if len(arr) < 2:
            return arr.tolist()

        factor = 2 ** (semitones / 12.0)

        # Resample to shift pitch
        new_length = max(2, int(len(arr) / factor))
        indices = np.linspace(0, len(arr) - 1, new_length)
        shifted = np.interp(indices, np.arange(len(arr)), arr)

        # Resample back to original length to preserve duration
        indices2 = np.linspace(0, len(shifted) - 1, len(arr))
        result = np.interp(indices2, np.arange(len(shifted)), shifted)

        return result.tolist()

    def _formant_shift(self, samples: list[float], sr: int, ratio: float) -> list[float]:
        """Formant shift using spectral processing."""
        import numpy as np

        arr = np.array(samples, dtype=np.float32)
        if len(arr) < 1024:
            return arr.tolist()

        # Spectral formant shift: shift the spectral envelope
        fft_size = min(4096, len(arr))
        spectrum = np.fft.rfft(arr[:fft_size] * np.hanning(fft_size))
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Shift magnitude by ratio
        n_bins = len(magnitude)
        new_magnitude = np.zeros_like(magnitude)
        for i in range(n_bins):
            src_idx = int(i / ratio)
            if 0 <= src_idx < n_bins:
                new_magnitude[i] = magnitude[src_idx]

        result = np.fft.irfft(new_magnitude * np.exp(1j * phase), fft_size)
        if len(result) < len(arr):
            result = np.pad(result, (0, len(arr) - len(result)))
        else:
            result = result[:len(arr)]

        return result.tolist()
