"""Tests for video understanding via VLM frame extraction."""

import os
import tempfile

import pytest
from python.yunshu_engine.vlm_engine import VLMEngine

# ── Helpers ──


class FakeTokenizer:
    """Minimal tokenizer for VLMEngine init."""

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, ids, **kw):
        return "decoded"

    @property
    def eos_token_id(self):
        return 2


class FakeModel:
    """Minimal model object."""

    pass


def _make_engine():
    """Create a VLMEngine with minimal fakes."""
    engine = VLMEngine.__new__(VLMEngine)
    engine._model = FakeModel()
    engine._tokenizer = FakeTokenizer()
    engine._model_name = "test-vlm"
    engine._config = {"model_type": "qwen2_vl"}
    engine._temp_files = []
    import threading

    engine._temp_files_lock = threading.Lock()
    engine._is_vlm = True
    engine._has_vision = True
    engine._active_count = 0
    engine._enable_thinking = None
    engine._executor = None
    return engine


# ── Video Detection Tests ──


class TestVideoDetection:
    def test_has_video_with_video_url(self):
        from python.yunshu_gateway.routers.chat import _has_video

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video"},
                    {"type": "video_url", "video_url": {"url": "file:///tmp/test.mp4"}},
                ],
            }
        ]
        assert _has_video(messages) is True

    def test_has_video_with_video_file(self):
        from python.yunshu_gateway.routers.chat import _has_video

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze"},
                    {"type": "video_file", "video_file": {"file_id": "/tmp/vid.mp4"}},
                ],
            }
        ]
        assert _has_video(messages) is True

    def test_no_video_text_only(self):
        from python.yunshu_gateway.routers.chat import _has_video

        messages = [{"role": "user", "content": "Hello"}]
        assert _has_video(messages) is False

    def test_no_video_image_only(self):
        from python.yunshu_gateway.routers.chat import _has_video

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://example.com/img.png"},
                    },
                ],
            }
        ]
        assert _has_video(messages) is False

    def test_has_video_mixed_content(self):
        from python.yunshu_gateway.routers.chat import _has_video

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://example.com/img.png"},
                    },
                    {"type": "video_url", "video_url": {"url": "file:///tmp/test.mp4"}},
                ],
            }
        ]
        assert _has_video(messages) is True


# ── Frame Extraction Tests ──


class TestVideoFrameExtraction:
    @pytest.mark.asyncio
    async def test_extract_no_video(self):
        engine = _make_engine()
        messages = [{"role": "user", "content": "Hello"}]
        frames = await engine._extract_video_frames(messages)
        assert frames == []

    @pytest.mark.asyncio
    async def test_extract_base64_video(self):
        """Base64 video content should be saved and attempted for frame extraction."""
        import base64

        engine = _make_engine()
        # Minimal valid-ish base64 video data (won't produce frames without real video)
        fake_data = base64.b64encode(b"fake video data").decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{fake_data}"},
                    },
                ],
            }
        ]
        # This will fail to extract frames (not a real video) but should not crash
        frames = await engine._extract_video_frames(messages)
        assert isinstance(frames, list)

    @pytest.mark.asyncio
    async def test_extract_nonexistent_file(self):
        # a REFERENCED-but-unloadable video must FAIL LOUD (matching image/audio
        # + anti-hallucination), not silently return [] — the old behavior let the model
        # answer about a video it never saw.
        engine = _make_engine()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": "file:///nonexistent/video.mp4"},
                    },
                ],
            }
        ]
        with pytest.raises(ValueError):
            await engine._extract_video_frames(messages)

    @pytest.mark.asyncio
    async def test_extract_video_file_type(self):
        # nonexistent video_file file_id → fail loud (was silent [])
        engine = _make_engine()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_file",
                        "video_file": {"file_id": "/nonexistent/video.mp4"},
                    },
                ],
            }
        ]
        with pytest.raises(ValueError):
            await engine._extract_video_frames(messages)

    @pytest.mark.asyncio
    async def test_max_frames_limit(self):
        """max_frames parameter should limit total frames."""
        engine = _make_engine()
        messages = [{"role": "user", "content": "Hello"}]
        frames = await engine._extract_video_frames(messages, max_frames=4)
        assert len(frames) <= 4

    @pytest.mark.asyncio
    async def test_fps_parameter(self):
        """fps parameter should be accepted without error."""
        engine = _make_engine()
        messages = [{"role": "user", "content": "Hello"}]
        frames = await engine._extract_video_frames(messages, fps=2.0)
        assert isinstance(frames, list)


# ── Save Base64 File Tests ──


class TestSaveBase64File:
    @pytest.mark.asyncio
    async def test_save_mp4(self):
        import base64

        engine = _make_engine()
        data = base64.b64encode(b"fake video content").decode()
        path = await engine._save_base64_file(data, "mp4")
        assert os.path.exists(path)
        assert path.endswith(".mp4")
        with open(path, "rb") as f:
            assert f.read() == b"fake video content"
        engine._cleanup_temp_files()
        assert not os.path.exists(path)

    @pytest.mark.asyncio
    async def test_save_webm(self):
        import base64

        engine = _make_engine()
        data = base64.b64encode(b"webm data").decode()
        path = await engine._save_base64_file(data, "webm")
        assert path.endswith(".webm")
        engine._cleanup_temp_files()


# ── ffmpeg Extraction Tests ──


class TestFFmpegExtraction:
    @pytest.mark.asyncio
    async def test_extract_from_nonexistent_video(self):
        engine = _make_engine()
        frames = await engine._extract_frames_from_file("/nonexistent/file.mp4")
        assert frames == []

    @pytest.mark.asyncio
    async def test_extract_from_empty_file(self):
        engine = _make_engine()
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)  # noqa: SIM115
        tmp.write(b"not a real video")
        tmp.close()
        try:
            frames = await engine._extract_frames_from_file(tmp.name)
            # ffmpeg will fail on invalid data, should return empty
            assert isinstance(frames, list)
        finally:
            os.unlink(tmp.name)


# ── Integration: Messages with Video Content ──


class TestVideoMessageFormats:
    @pytest.mark.asyncio
    async def test_multiple_video_urls(self):
        engine = _make_engine()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare videos"},
                    {
                        "type": "video_url",
                        "video_url": {"url": "file:///nonexistent/a.mp4"},
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": "file:///nonexistent/b.mp4"},
                    },
                ],
            }
        ]
        # an unloadable referenced video fails loud (was silent [])
        with pytest.raises(ValueError):
            await engine._extract_video_frames(messages)

    @pytest.mark.asyncio
    async def test_video_and_image_mixed(self):
        engine = _make_engine()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Mixed"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://example.com/img.png"},
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": "file:///nonexistent/v.mp4"},
                    },
                ],
            }
        ]
        # the referenced (nonexistent) video fails loud — it must not be
        # silently dropped just because an image is also present.
        with pytest.raises(ValueError):
            await engine._extract_video_frames(messages)

    @pytest.mark.asyncio
    async def test_empty_content_parts(self):
        engine = _make_engine()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Just text"},
                ],
            }
        ]
        frames = await engine._extract_video_frames(messages)
        assert frames == []

    @pytest.mark.asyncio
    async def test_string_content_no_video(self):
        engine = _make_engine()
        messages = [{"role": "user", "content": "Plain string message"}]
        frames = await engine._extract_video_frames(messages)
        assert frames == []
