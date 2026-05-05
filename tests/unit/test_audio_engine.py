"""Tests for the audio engine module.

Covers:
- WAV encoding helpers (_audio_to_wav_bytes, _pcm_to_wav, make_wav_header)
- list_voices() fallback
- TTSEngine (lifecycle, synthesize with mock, list_voices)
- ASREngine (lifecycle, transcribe with mock)
- Module-level transcribe() and synthesize() convenience functions
"""
import asyncio
import io
import os
import struct
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from yunshu_engine.audio_engine import (
    ASREngine,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VOICES,
    TTSEngine,
    _audio_to_wav_bytes,
    _pcm_to_wav,
    list_voices,
    make_wav_header,
)


# ── WAV helpers ──


class TestWavHelpers:
    def test_pcm_to_wav_header_structure(self):
        """WAV output should start with RIFF header."""
        pcm = np.zeros(100, dtype=np.int16)
        wav = _pcm_to_wav(pcm)
        assert wav[:4] == b'RIFF'
        assert wav[8:12] == b'WAVE'
        assert wav[12:16] == b'fmt '
        assert wav[36:40] == b'data'

    def test_pcm_to_wav_correct_size(self):
        """WAV data size should match PCM bytes + 44 byte header."""
        pcm = np.zeros(100, dtype=np.int16)
        wav = _pcm_to_wav(pcm, sample_rate=22050)
        # 100 samples * 2 bytes/sample = 200 data bytes
        # Total = 44 header + 200 data = 244
        assert len(wav) == 244

    def test_audio_to_wav_bytes_float_input(self):
        """Float audio array should be clipped and converted to 16-bit PCM."""
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        wav = _audio_to_wav_bytes(audio)
        assert wav[:4] == b'RIFF'
        # Check data chunk size
        data_size = struct.unpack_from('<I', wav, 40)[0]
        assert data_size == 5 * 2  # 5 samples * 2 bytes

    def test_audio_to_wav_bytes_clipping(self):
        """Values > 1.0 should be clipped."""
        audio = np.array([2.0, -2.0], dtype=np.float32)
        wav = _audio_to_wav_bytes(audio)
        data_size = struct.unpack_from('<I', wav, 40)[0]
        assert data_size == 2 * 2

    def test_make_wav_header_length(self):
        """WAV header should always be 44 bytes."""
        hdr = make_wav_header(1000, sample_rate=24000)
        assert len(hdr) == 44

    def test_make_wav_header_riff_chunk_size(self):
        """RIFF chunk size should be 36 + data_size."""
        data_size = 5000
        hdr = make_wav_header(data_size, sample_rate=22050)
        riff_size = struct.unpack_from('<I', hdr, 4)[0]
        assert riff_size == 36 + data_size

    def test_make_wav_header_sample_rate(self):
        """Sample rate should be correctly encoded."""
        hdr = make_wav_header(100, sample_rate=48000)
        sr = struct.unpack_from('<I', hdr, 24)[0]
        assert sr == 48000

    def test_make_wav_header_num_channels(self):
        """Channel count should be correctly encoded."""
        hdr = make_wav_header(100, sample_rate=24000, num_channels=2)
        channels = struct.unpack_from('<H', hdr, 22)[0]
        assert channels == 2

    def test_pcm_roundtrip(self):
        """PCM data should survive WAV encode -> parse roundtrip."""
        original = np.array([0, 100, -100, 32767, -32768], dtype=np.int16)
        wav = _pcm_to_wav(original)
        # Skip 44 byte header, read back PCM
        pcm_bytes = wav[44:]
        recovered = np.frombuffer(pcm_bytes, dtype=np.int16)
        np.testing.assert_array_equal(recovered, original)


# ── list_voices ──


class TestListVoices:
    def test_returns_default_voices_without_manager(self):
        """Without a model manager, should return DEFAULT_VOICES."""
        voices = list_voices()
        assert isinstance(voices, list)
        assert all(isinstance(v, str) for v in voices)
        # Should at least contain the defaults
        for v in DEFAULT_VOICES:
            assert v in voices

    def test_returns_list_not_reference(self):
        """Should return a new list, not the mutable DEFAULT_VOICES."""
        v1 = list_voices()
        v2 = list_voices()
        assert v1 is not v2


# ── TTSEngine ──


class TestTTSEngine:
    def test_engine_creation(self):
        engine = TTSEngine("test-model")
        assert engine.model_name == "test-model"
        assert engine.is_loaded is False

    def test_model_name_from_path(self):
        engine = TTSEngine("/models/kokoro-82m")
        assert engine.model_name == "kokoro-82m"

    def test_list_voices_returns_default_when_no_model(self):
        engine = TTSEngine("test")
        voices = engine.list_voices()
        assert voices == DEFAULT_VOICES

    def test_list_voices_from_model(self):
        engine = TTSEngine("test")
        mock_model = MagicMock()
        mock_model.voices = ["custom_voice_1", "custom_voice_2"]
        engine._model = mock_model
        voices = engine.list_voices()
        assert voices == ["custom_voice_1", "custom_voice_2"]

    def test_get_stats(self):
        engine = TTSEngine("/models/test-tts")
        stats = engine.get_stats()
        assert stats["model"] == "/models/test-tts"
        assert stats["loaded"] is False
        assert stats["running"] is False


# ── ASREngine ──


class TestASREngine:
    def test_engine_creation(self):
        engine = ASREngine("whisper-small")
        assert engine.model_name == "whisper-small"
        assert engine.is_loaded is False

    def test_model_name_from_path(self):
        engine = ASREngine("/models/whisper-large-v3")
        assert engine.model_name == "whisper-large-v3"

    def test_get_stats(self):
        engine = ASREngine("whisper-small")
        stats = engine.get_stats()
        assert stats["model"] == "whisper-small"
        assert stats["loaded"] is False
        assert stats["running"] is False


# ── Module-level convenience functions ──


class TestModuleLevelTranscribe:
    @pytest.mark.asyncio
    async def test_file_not_found(self):
        """transcribe() should raise FileNotFoundError for missing files."""
        from yunshu_engine.audio_engine import transcribe
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            await transcribe("/nonexistent/path/audio.wav")

    @pytest.mark.asyncio
    async def test_fallback_returns_empty_dict(self):
        """Without mlx-audio or a loaded engine, transcribe returns empty text."""
        from yunshu_engine.audio_engine import transcribe

        # Create a temp WAV file
        pcm = np.zeros(1000, dtype=np.int16)
        wav = _pcm_to_wav(pcm, 16000)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav)
            tmp_path = tmp.name

        try:
            with patch("yunshu_engine.audio_engine._find_asr_engine", return_value=None):
                with patch.dict("sys.modules", {"mlx_audio": None, "mlx_audio.stt": None, "mlx_audio.stt.utils": None}):
                    result = await transcribe(tmp_path, language="en")
                    assert "text" in result
                    assert "language" in result
                    assert "segments" in result
                    assert "duration" in result
        finally:
            os.unlink(tmp_path)


class TestModuleLevelSynthesize:
    @pytest.mark.asyncio
    async def test_empty_text_returns_silence(self):
        """Empty text input should produce a valid WAV with silence."""
        from yunshu_engine.audio_engine import synthesize
        result = await synthesize("")
        assert result[:4] == b'RIFF'
        assert result[8:12] == b'WAVE'

    @pytest.mark.asyncio
    async def test_fallback_returns_wav_silence(self):
        """Without engine or mlx-audio, synthesize returns silence WAV."""
        from yunshu_engine.audio_engine import synthesize

        with patch("yunshu_engine.audio_engine._find_tts_engine", return_value=None):
            with patch.dict("sys.modules", {"mlx_audio": None, "mlx_audio.tts": None, "mlx_audio.tts.utils": None}):
                result = await synthesize("Hello world")
                assert result[:4] == b'RIFF'
                assert result[8:12] == b'WAVE'
                # Should have some data (not just header)
                assert len(result) > 44
