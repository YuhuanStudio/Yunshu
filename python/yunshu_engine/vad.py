from __future__ import annotations
"""Voice Activity Detection (VAD) module.

Provides multiple VAD implementations for speech detection in audio streams:
- EnergyVAD: Simple RMS energy threshold (fast, no ML required)
- WebRTCVD: WebRTC-based VAD (when webrtcvad is available)

Used by:
- Realtime API for automatic speech endpoint detection
- ASR pipeline for pre-filtering silence
- STS pipeline for turn-taking
"""

import logging
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VADResult:
    is_speech: bool
    energy: float
    confidence: float = 1.0


class VADBase(ABC):
    @abstractmethod
    def process_frame(self, audio_bytes: bytes, sample_rate: int = 16000) -> VADResult:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


class EnergyVAD(VADBase):
    """RMS energy-based VAD. Simple, fast, no dependencies.

    Detects speech by comparing audio frame energy against an adaptive
    threshold. Tracks background noise level for automatic threshold
    adjustment.
    """

    def __init__(
        self,
        threshold: float = 0.01,
        frame_duration_ms: int = 30,
        sample_rate: int = 16000,
        silence_duration_ms: int = 600,
        speech_duration_ms: int = 100,
    ) -> None:
        self.threshold = threshold
        self.frame_duration_ms = frame_duration_ms
        self.sample_rate = sample_rate
        self.silence_frames_needed = int(silence_duration_ms / frame_duration_ms)
        self.speech_frames_needed = int(speech_duration_ms / frame_duration_ms)

        self._noise_level = 0.0
        self._silence_count = 0
        self._speech_count = 0
        self._is_speaking = False

    def process_frame(self, audio_bytes: bytes, sample_rate: int = 16000) -> VADResult:
        energy = self._compute_energy(audio_bytes)

        # Adaptive threshold: track noise floor
        if not self._is_speaking:
            self._noise_level = 0.95 * self._noise_level + 0.05 * energy

        adaptive_threshold = max(self.threshold, self._noise_level * 3)

        if energy > adaptive_threshold:
            self._speech_count += 1
            self._silence_count = 0
            if self._speech_count >= self.speech_frames_needed:
                self._is_speaking = True
        else:
            self._silence_count += 1
            self._speech_count = 0
            if self._silence_count >= self.silence_frames_needed:
                self._is_speaking = False

        confidence = min(1.0, energy / (adaptive_threshold * 2)) if adaptive_threshold > 0 else 0.0
        return VADResult(
            is_speech=self._is_speaking,
            energy=energy,
            confidence=confidence,
        )

    def reset(self) -> None:
        self._noise_level = 0.0
        self._silence_count = 0
        self._speech_count = 0
        self._is_speaking = False

    @staticmethod
    def _compute_energy(audio_bytes: bytes) -> float:
        if len(audio_bytes) < 2:
            return 0.0
        n_samples = len(audio_bytes) // 2
        samples = struct.unpack(f"<{n_samples}h", audio_bytes[:n_samples * 2])
        if not samples:
            return 0.0
        rms = sum(s * s for s in samples) / n_samples
        return (rms ** 0.5) / 32768.0


class WebRTCVAD(VADBase):
    """WebRTC-based VAD wrapper (requires webrtcvad package).

    More accurate than EnergyVAD but requires external dependency.
    """

    def __init__(self, aggressiveness: int = 3) -> None:
        self._aggressiveness = aggressiveness
        self._vad = None
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(aggressiveness)
        except ImportError:
            logger.warning("webrtcvad not installed, falling back to EnergyVAD")

    def process_frame(self, audio_bytes: bytes, sample_rate: int = 16000) -> VADResult:
        if self._vad is None:
            # Fallback to energy-based
            energy = EnergyVAD._compute_energy(audio_bytes)
            return VADResult(is_speech=energy > 0.01, energy=energy)

        # WebRTC VAD only accepts frames of 10, 20, or 30 ms duration.
        # Validate frame size to avoid silent fallback on every call.
        n_samples = len(audio_bytes) // 2  # 16-bit audio
        frame_duration_ms = (n_samples / sample_rate) * 1000 if sample_rate > 0 else 0
        valid_durations = {10, 20, 30}
        if frame_duration_ms not in valid_durations:
            logger.warning(
                "WebRTC VAD requires frame duration of 10/20/30 ms, got %.1f ms "
                "(%d samples at %d Hz). Falling back to energy-based.",
                frame_duration_ms, n_samples, sample_rate,
            )
            energy = EnergyVAD._compute_energy(audio_bytes)
            return VADResult(is_speech=energy > 0.01, energy=energy)

        try:
            is_speech = self._vad.is_speech(audio_bytes, sample_rate)
            energy = EnergyVAD._compute_energy(audio_bytes)
            return VADResult(is_speech=is_speech, energy=energy)
        except Exception:
            logger.debug("WebRTC VAD failed, falling back to energy-based", exc_info=True)
            energy = EnergyVAD._compute_energy(audio_bytes)
            return VADResult(is_speech=energy > 0.01, energy=energy)

    def reset(self) -> None:
        pass


def create_vad(engine: str = "energy", **kwargs) -> VADBase:
    """Factory function to create a VAD instance.

    Args:
        engine: "energy" or "webrtc"
        **kwargs: VAD-specific parameters
    """
    if engine == "webrtc":
        return WebRTCVAD(aggressiveness=kwargs.get("aggressiveness", 3))
    return EnergyVAD(
        threshold=kwargs.get("threshold", 0.01),
        frame_duration_ms=kwargs.get("frame_duration_ms", 30),
        sample_rate=kwargs.get("sample_rate", 16000),
        silence_duration_ms=kwargs.get("silence_duration_ms", 600),
        speech_duration_ms=kwargs.get("speech_duration_ms", 100),
    )
