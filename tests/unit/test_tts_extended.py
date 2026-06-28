"""Tests for extended TTS parameters."""


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
        # SECURITY: ref_audio is a relative path under the media dir;
        # absolute host paths are rejected (arbitrary-file-read defense).
        req = TTSRequest(
            model="test",
            input="hello",
            ref_audio="voices/ref.wav",
            ref_text="reference text",
        )
        assert req.ref_audio == "voices/ref.wav"
        assert req.ref_text == "reference text"

    def test_voice_cloning_absolute_path_rejected(self):
        """an absolute ref_audio path must be rejected (would let a
        client read any host file as a 'voice reference')."""
        import pydantic
        import pytest

        from yunshu_gateway.routers.audio import TTSRequest
        with pytest.raises((pydantic.ValidationError, ValueError)):
            TTSRequest(model="test", input="hello", ref_audio="/etc/passwd")

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
