"""Tests for VAD (Voice Activity Detection)."""
import struct
import pytest


def _make_pcm_samples(samples, amplitude=0.5):
    """Generate PCM16 bytes from a list of amplitudes (-1.0 to 1.0)."""
    pcm = b""
    for s in samples:
        val = int(s * amplitude * 32767)
        val = max(-32768, min(32767, val))
        pcm += struct.pack("<h", val)
    return pcm


def _silence_pcm(duration_ms=30, sample_rate=16000):
    """Generate silence PCM bytes."""
    n = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * n


def _speech_pcm(duration_ms=30, sample_rate=16000, freq=440):
    """Generate speech-like PCM bytes (sine wave)."""
    import math
    n = int(sample_rate * duration_ms / 1000)
    samples = [math.sin(2 * math.pi * freq * i / sample_rate) * 0.5 for i in range(n)]
    return _make_pcm_samples(samples, amplitude=0.8)


class TestEnergyVAD:
    def test_silence_not_speech(self):
        from yunshu_engine.vad import EnergyVAD
        vad = EnergyVAD(threshold=0.01)
        frame = _silence_pcm()
        result = vad.process_frame(frame)
        assert result.is_speech is False

    def test_speech_detected(self):
        from yunshu_engine.vad import EnergyVAD
        vad = EnergyVAD(threshold=0.01, speech_duration_ms=30, silence_duration_ms=600)
        # Feed enough speech frames to trigger
        speech = _speech_pcm()
        for _ in range(5):
            result = vad.process_frame(speech)
        assert result.is_speech is True

    def test_energy_computation(self):
        from yunshu_engine.vad import EnergyVAD
        silence = _silence_pcm()
        speech = _speech_pcm()
        silence_energy = EnergyVAD._compute_energy(silence)
        speech_energy = EnergyVAD._compute_energy(speech)
        assert speech_energy > silence_energy

    def test_reset(self):
        from yunshu_engine.vad import EnergyVAD
        vad = EnergyVAD()
        vad._is_speaking = True
        vad.reset()
        assert vad._is_speaking is False

    def test_empty_audio(self):
        from yunshu_engine.vad import EnergyVAD
        vad = EnergyVAD()
        result = vad.process_frame(b"")
        assert result.energy == 0.0

    def test_result_fields(self):
        from yunshu_engine.vad import VADResult
        r = VADResult(is_speech=True, energy=0.5, confidence=0.9)
        assert r.is_speech is True
        assert r.energy == 0.5
        assert r.confidence == 0.9


class TestCreateVAD:
    def test_create_energy(self):
        from yunshu_engine.vad import create_vad, EnergyVAD
        vad = create_vad("energy")
        assert isinstance(vad, EnergyVAD)

    def test_create_webrtc_fallback(self):
        from yunshu_engine.vad import create_vad, WebRTCVAD
        vad = create_vad("webrtc")
        assert isinstance(vad, WebRTCVAD)

    def test_create_with_params(self):
        from yunshu_engine.vad import create_vad
        vad = create_vad("energy", threshold=0.05, sample_rate=8000)
        assert vad.threshold == 0.05
        assert vad.sample_rate == 8000
