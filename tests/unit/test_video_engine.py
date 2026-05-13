"""Tests for video engine and gateway — Wan2.2/LTX2 video generation."""

import base64
import io
import pytest
import numpy as np

from PIL import Image as PILImage


def _make_png(width=64, height=64, color=(128, 64, 32)) -> bytes:
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    pil = PILImage.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


# ── VideoGenConfig tests ──


class TestVideoGenConfig:
    def test_defaults(self):
        from yunshu_engine.video_engine import VideoGenConfig

        cfg = VideoGenConfig()
        assert cfg.width == 1280
        assert cfg.height == 704
        assert cfg.num_frames == 81
        assert cfg.fps == 16

    def test_custom(self):
        from yunshu_engine.video_engine import VideoGenConfig

        cfg = VideoGenConfig(width=640, height=480, num_frames=41, num_steps=10)
        assert cfg.width == 640
        assert cfg.num_frames == 41


class TestVideoGenOutput:
    def test_defaults(self):
        from yunshu_engine.video_engine import VideoGenOutput

        out = VideoGenOutput()
        assert out.video_data == b""
        assert out.frames == []
        assert out.method == ""

    def test_with_data(self):
        from yunshu_engine.video_engine import VideoGenOutput

        out = VideoGenOutput(
            video_data=b"fake_mp4",
            width=1280,
            height=704,
            num_frames=81,
            method="wan_2_2",
        )
        assert out.video_data == b"fake_mp4"
        assert out.num_frames == 81


# ── VideoEngine tests ──


class TestVideoEngineInit:
    def test_default_init(self):
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine()
        assert engine.model_name == "video-default"
        assert not engine.is_loaded

    def test_with_path(self):
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine(model_path="/models/Wan2.2-T2V-1.3B")
        assert "wan" in engine.model_name.lower() or "Wan" in engine.model_name

    def test_detect_wan(self):
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine(model_path="/models/wan-2.2-t2v")
        assert engine._model_type == "wan_2_2"

    def test_detect_ltx(self):
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine(model_path="/models/ltx-video-2.0")
        assert engine._model_type == "ltx_2"

    def test_detect_default(self):
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine(model_path="/models/unknown")
        assert engine._model_type == "wan_2_2"  # Defaults to wan

    def test_lifecycle(self):
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine()
        engine.start()
        assert engine.is_loaded
        engine.stop()
        assert not engine.is_loaded

    def test_double_start(self):
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine()
        engine.start()
        engine.start()
        assert engine.is_loaded
        engine.stop()

    def test_stats(self):
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine(model_path="/models/test-video")
        stats = engine.get_stats()
        assert "model_type" in stats
        assert stats["model_type"] == "wan_2_2"


class TestVideoEngineFallback:
    @pytest.mark.asyncio
    async def test_fallback_generation(self):
        """Fallback should produce placeholder frames without a model."""
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine()
        engine.start()
        result = await engine.generate(
            prompt="a cat running",
            width=64,
            height=64,
            num_frames=9,
            output_format="frames",
        )
        assert len(result.frames) > 0
        assert result.method == "fallback"
        assert result.width == 64
        assert result.height == 64
        engine.stop()

    @pytest.mark.asyncio
    async def test_fallback_with_image(self):
        """Fallback should work even with an image input (I2V)."""
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine()
        engine.start()
        image = _make_png(64, 64)
        result = await engine.generate(
            prompt="animate this",
            image=image,
            width=64,
            height=64,
            output_format="frames",
        )
        assert len(result.frames) > 0
        engine.stop()

    @pytest.mark.asyncio
    async def test_auto_start(self):
        """Engine should auto-start when generate is called."""
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine()
        assert not engine.is_loaded
        result = await engine.generate(
            prompt="test",
            width=64,
            height=64,
            output_format="frames",
        )
        assert engine.is_loaded
        engine.stop()


# ── Gateway endpoint tests ──


class TestVideoEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from yunshu_gateway.main import create_app
        app = create_app()
        return TestClient(app)

    def test_video_no_model(self, client):
        """Video endpoint should work with fallback engine."""
        resp = client.post("/v1/video/generations", json={
            "prompt": "a beautiful sunset",
            "width": 64,
            "height": 64,
            "num_frames": 9,
            "response_format": "frames",
        })
        # Should succeed with fallback frames
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) > 0
        assert "frames" in data["data"][0]

    def test_video_invalid_image(self, client):
        resp = client.post("/v1/video/generations", json={
            "prompt": "test",
            "image": "not-valid-base64!!!",
        })
        assert resp.status_code == 400

    def test_video_request_model(self):
        from yunshu_gateway.routers.video import VideoGenerateRequest

        req = VideoGenerateRequest(
            prompt="a mountain scene",
            width=640,
            height=480,
            num_frames=41,
            num_inference_steps=10,
            guide_scale=7.0,
            fps=24,
            seed=42,
            scheduler="euler",
        )
        assert req.width == 640
        assert req.num_frames == 41
        assert req.scheduler == "euler"


# ── Model manager integration ──


class TestVideoModelManager:
    def test_video_model_type(self):
        from yunshu_engine.model_manager import ModelType

        assert hasattr(ModelType, "VIDEO")

    def test_detect_wan_model(self):
        from yunshu_engine.model_manager import _detect_model_type, ModelType

        assert _detect_model_type("/models/Wan2.2-T2V-1.3B") == ModelType.VIDEO

    def test_detect_ltx_model(self):
        from yunshu_engine.model_manager import _detect_model_type, ModelType

        assert _detect_model_type("/models/ltx-video-2.0") == ModelType.VIDEO

    def test_detect_video_keyword(self):
        from yunshu_engine.model_manager import _detect_model_type, ModelType

        assert _detect_model_type("/models/text-to-video") == ModelType.VIDEO
