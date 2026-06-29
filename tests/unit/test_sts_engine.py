"""Tests for STSEngine — speech-to-speech infrastructure."""

import struct

import numpy as np
import pytest
from python.yunshu_engine.sts_engine import STSConfig, STSEngine, STSOutput


def _make_wav(samples, sample_rate=16000):
    """Create a WAV bytes buffer from float samples."""
    arr = np.array(samples, dtype=np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    int_samples = (arr * 32767).astype(np.int16)
    data = int_samples.tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        len(data),
    )
    return header + data


def _make_sine_wav(freq=440, duration=0.1, sr=16000):
    """Create a sine wave WAV buffer."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    samples = 0.5 * np.sin(2 * np.pi * freq * t)
    return _make_wav(samples, sr), samples, sr


class TestSTSConfig:
    def test_defaults(self):
        c = STSConfig()
        assert c.sample_rate == 16000
        assert c.enhance_method == "spectral_gating"
        assert c.pitch_shift_semitones == 0.0

    def test_custom(self):
        c = STSConfig(enhance_method="deep_filter", noise_floor_db=-30.0)
        assert c.enhance_method == "deep_filter"
        assert c.noise_floor_db == -30.0


class TestSTSOutput:
    def test_defaults(self):
        out = STSOutput()
        assert out.audio_data == b""
        assert out.sample_rate == 16000
        assert out.method == ""

    def test_with_data(self):
        out = STSOutput(audio_data=b"fake", sample_rate=22050, method="spectral_gating")
        assert out.audio_data == b"fake"
        assert out.sample_rate == 22050


class TestSTSEngineInit:
    def test_default_init(self):
        engine = STSEngine()
        assert engine.model_name == "sts-default"
        assert not engine.is_loaded

    def test_with_path(self):
        engine = STSEngine(model_path="/models/deepfilter-v2")
        assert "deepfilter" in engine.model_name

    def test_detect_model_type_deepfilter(self):
        engine = STSEngine(model_path="/models/deepfilter-v2")
        assert engine._model_type == "deep_filter_net"

    def test_detect_model_type_mossformer(self):
        engine = STSEngine(model_path="/models/mossformer2")
        assert engine._model_type == "mossformer2"

    def test_detect_model_type_sam(self):
        engine = STSEngine(model_path="/models/sam-audio")
        assert engine._model_type == "sam_audio"

    def test_detect_model_type_lfm(self):
        engine = STSEngine(model_path="/models/lfm-audio")
        assert engine._model_type == "lfm_audio"

    def test_detect_model_type_default(self):
        engine = STSEngine(model_path="/models/unknown")
        assert engine._model_type == "default"


class TestSTSEngineLifecycle:
    def test_start_stop(self):
        engine = STSEngine()
        engine.start()
        assert engine.is_loaded
        engine.stop()
        assert not engine.is_loaded

    def test_double_start(self):
        engine = STSEngine()
        engine.start()
        engine.start()  # Should not raise
        assert engine.is_loaded
        engine.stop()


class TestSTSEngineEnhance:
    @pytest.mark.asyncio
    async def test_enhance_spectral_gating(self):
        wav_data, samples, sr = _make_sine_wav(freq=440, duration=0.1)
        engine = STSEngine()
        engine.start()
        result = await engine.enhance(wav_data, method="spectral_gating")
        assert isinstance(result, STSOutput)
        assert result.method == "spectral_gating"
        assert len(result.audio_data) > 0
        assert result.sample_rate == sr
        engine.stop()

    @pytest.mark.asyncio
    async def test_enhance_minimal(self):
        wav_data, samples, sr = _make_sine_wav(freq=440, duration=0.1)
        engine = STSEngine()
        engine.start()
        result = await engine.enhance(wav_data, method="minimal")
        assert result.method == "minimal"
        assert len(result.audio_data) > 0
        engine.stop()

    @pytest.mark.asyncio
    async def test_enhance_auto_starts(self):
        wav_data, _, _ = _make_sine_wav()
        engine = STSEngine()
        assert not engine.is_loaded
        result = await engine.enhance(wav_data)
        assert engine.is_loaded
        assert len(result.audio_data) > 0
        engine.stop()


class TestSTSEngineSeparate:
    @pytest.mark.asyncio
    async def test_separate_basic(self):
        wav_data, samples, sr = _make_sine_wav(freq=440, duration=0.2)
        engine = STSEngine()
        engine.start()
        result = await engine.separate(wav_data, source_text="voice")
        assert isinstance(result, STSOutput)
        assert result.method == "energy_mask"
        assert len(result.audio_data) > 0
        engine.stop()


class TestSTSEngineTransform:
    @pytest.mark.asyncio
    async def test_transform_pitch_shift(self):
        wav_data, samples, sr = _make_sine_wav(freq=440, duration=0.1)
        engine = STSEngine()
        engine.start()
        result = await engine.transform(wav_data, pitch_shift=2.0)
        assert isinstance(result, STSOutput)
        assert result.method == "voice_transform"
        assert result.metadata["pitch_shift"] == 2.0
        engine.stop()

    @pytest.mark.asyncio
    async def test_transform_no_change(self):
        wav_data, _, _ = _make_sine_wav()
        engine = STSEngine()
        engine.start()
        result = await engine.transform(wav_data)
        assert len(result.audio_data) > 0
        engine.stop()


class TestSTSSignalProcessing:
    def test_spectral_gating(self):
        sr = 16000
        t = np.linspace(0, 0.5, sr // 2, dtype=np.float32)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        noise = 0.05 * np.random.randn(len(signal)).astype(np.float32)
        samples = signal + noise

        engine = STSEngine()
        engine.start()
        enhanced = engine._spectral_gating_enhance(samples.tolist(), sr, -40.0)
        assert len(enhanced) > 0
        assert isinstance(enhanced, list)

    def test_minimal_enhance(self):
        samples = [0.5, 0.0, 0.001, -0.5, 0.0001, 0.3]
        engine = STSEngine()
        engine.start()
        enhanced = engine._minimal_enhance(samples)
        assert len(enhanced) == len(samples)

    def test_pitch_shift(self):
        sr = 16000
        samples = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 1600)).tolist()
        engine = STSEngine()
        engine.start()
        shifted = engine._pitch_shift(samples, sr, 2.0)
        assert len(shifted) == len(samples)

    def test_formant_shift(self):
        sr = 16000
        samples = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 1600)).tolist()
        engine = STSEngine()
        engine.start()
        shifted = engine._formant_shift(samples, sr, 1.2)
        assert len(shifted) == len(samples)


class TestWavEncodeDecode:
    def test_roundtrip(self):
        sr = 16000
        samples = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 1600))).tolist()
        engine = STSEngine()
        engine.start()
        wav = engine._encode_wav(samples, sr)
        decoded, decoded_sr = engine._load_audio(wav)
        assert decoded_sr == sr
        assert len(decoded) == len(samples)
        # Check signal is approximately preserved
        np.testing.assert_allclose(decoded, samples, atol=0.01)

    def test_load_invalid_format(self):
        engine = STSEngine()
        engine.start()
        with pytest.raises(ValueError, match="Unsupported"):
            engine._load_audio(b"not a wav file")
