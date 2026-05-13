"""Tests for extended TTS parameters."""
import pytest


class TestTTSRequestExtended:
    def test_default_params(self):
        from yunshu_gateway.routers.audio import TTSRequest
        req = TTSRequest(model="test", input="hello")
        assert req.top_k == 50
        assert req.top_p == 0.95
        assert req.repetition_penalty == 1.0
        assert req.max_tokens == 4096
        assert req.ref_audio is None
        assert req.ref_text is None
        assert req.segment_size == 300

    def test_voice_cloning_params(self):
        from yunshu_gateway.routers.audio import TTSRequest
        req = TTSRequest(
            model="test",
            input="hello",
            ref_audio="/path/to/ref.wav",
            ref_text="reference text",
        )
        assert req.ref_audio == "/path/to/ref.wav"
        assert req.ref_text == "reference text"

    def test_custom_sampling_params(self):
        from yunshu_gateway.routers.audio import TTSRequest
        req = TTSRequest(
            model="test",
            input="hello",
            top_k=100,
            top_p=0.9,
            repetition_penalty=1.2,
            temperature=0.8,
        )
        assert req.top_k == 100
        assert req.top_p == 0.9
        assert req.repetition_penalty == 1.2
