"""Tests for video engine and gateway — Wan2.2/LTX2 video generation."""

import base64
import io
import os
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


# ── VideoStats tests ──


class TestVideoStats:
    def test_stats_dataclass(self):
        from yunshu_engine.video_engine import VideoStats

        stats = VideoStats()
        assert stats.frames_processed == 0
        assert stats.frames_streamed == 0
        assert stats.batches_processed == 0
        assert stats.total_generate_calls == 0
        assert stats.total_stream_calls == 0
        assert stats.lora_loaded is False
        assert stats.lora_merged is False
        assert stats.avg_fps == 0.0
        assert stats.avg_decode_fps == 0.0

    def test_avg_fps_calculation(self):
        from yunshu_engine.video_engine import VideoStats

        stats = VideoStats(frames_processed=100, total_generate_ms=5000.0)
        assert stats.avg_fps == 20.0

    def test_avg_decode_fps_calculation(self):
        from yunshu_engine.video_engine import VideoStats

        stats = VideoStats(frames_streamed=50, total_frame_decode_ms=2500.0)
        assert stats.avg_decode_fps == 20.0


# ── FrameBatch tests ──


class TestFrameBatch:
    def test_frame_batch_defaults(self):
        from yunshu_engine.video_engine import FrameBatch

        batch = FrameBatch()
        assert batch.frames == []
        assert batch.frame_indices == []
        assert batch.batch_size == 0

    def test_frame_batch_with_data(self):
        from yunshu_engine.video_engine import FrameBatch

        batch = FrameBatch(
            frames=[b"frame1", b"frame2", b"frame3"],
            frame_indices=[0, 2, 4],
            width=128,
            height=128,
            batch_size=3,
        )
        assert len(batch.frames) == 3
        assert batch.frame_indices == [0, 2, 4]
        assert batch.batch_size == 3


# ── Enhanced stats tests ──


class TestVideoEngineStats:
    def test_stats_include_new_fields(self):
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine(model_path="/models/test-video")
        stats = engine.get_stats()
        assert "frames_processed" in stats
        assert "frames_streamed" in stats
        assert "batches_processed" in stats
        assert "avg_fps" in stats
        assert "avg_decode_fps" in stats
        assert "lora_loaded" in stats
        assert "lora_merged" in stats
        assert "lora_adapter_id" in stats
        assert "total_generate_calls" in stats
        assert "total_stream_calls" in stats

    def test_stats_update_after_generate(self):
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine()
        engine.start()
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            engine.generate(
                prompt="test stats",
                width=64,
                height=64,
                output_format="frames",
            )
        )
        stats = engine.get_stats()
        assert stats["total_generate_calls"] == 1
        assert stats["frames_processed"] > 0
        assert stats["total_generate_ms"] > 0
        assert stats["avg_fps"] > 0
        engine.stop()


# ── LoRA adapter tests ──


class TestVideoLoRA:
    def test_load_lora_no_config(self):
        """Loading LoRA with no adapter_config.json should fail gracefully."""
        import tempfile
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine()
        engine.start()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = engine.load_lora_adapter(tmpdir)
            assert result is False

        engine.stop()

    def test_load_lora_not_loaded_twice(self):
        """Loading LoRA twice without unloading should warn and return False."""
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine()
        engine.start()
        engine._lora_loaded = True  # Simulate already loaded

        result = engine.load_lora_adapter("/fake/path")
        assert result is False

        engine._lora_loaded = False
        engine.stop()

    def test_unload_lora_when_not_loaded(self):
        """Unloading when nothing is loaded should return False."""
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine()
        assert engine.unload_lora_adapter() is False

    def test_merge_lora_when_not_loaded(self):
        """Merging when nothing is loaded should return False."""
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine()
        assert engine.merge_lora_adapter() is False

    def test_unload_merged_lora_fails(self):
        """Unloading a merged LoRA should fail."""
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine()
        engine._lora_loaded = True
        engine._lora_merged = True

        result = engine.unload_lora_adapter()
        assert result is False

        engine._lora_loaded = False
        engine._lora_merged = False

    def test_list_lora_status_empty(self):
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine()
        status = engine.list_lora_status()
        assert status["loaded"] is False
        assert status["merged"] is False
        assert status["adapter_path"] == ""
        assert status["rank"] == 8
        assert status["scale"] == 20.0

    def test_load_lora_lazy_queuing(self):
        """When model is None, LoRA should be queued for lazy loading."""
        import tempfile
        import json
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "adapter_config.json")
            with open(config_path, "w") as f:
                json.dump({
                    "lora_parameters": {"rank": 4, "scale": 10.0},
                    "num_layers": 8,
                }, f)

            result = engine.load_lora_adapter(tmpdir)
            assert result is True
            assert engine._lora_adapter_path == tmpdir
            assert engine._lora_rank == 4
            assert engine._lora_scale == 10.0

    def test_lora_lifecycle(self):
        """Full LoRA lifecycle: load, status, unload."""
        import tempfile
        import json
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "adapter_config.json")
            with open(config_path, "w") as f:
                json.dump({
                    "lora_parameters": {"rank": 16, "scale": 30.0},
                    "num_layers": 4,
                }, f)

            # Load (will queue since no model)
            assert engine.load_lora_adapter(tmpdir) is True
            status = engine.list_lora_status()
            assert status["adapter_path"] == tmpdir

    def test_stop_clears_lora(self):
        """Stopping engine should clear LoRA state."""
        from yunshu_engine.video_engine import VideoEngine

        engine = VideoEngine()
        engine.start()
        engine._lora_loaded = True
        engine._lora_adapter_path = "/fake/lora"
        engine.stop()

        assert engine._lora_loaded is False
        assert engine._lora_adapter_path == ""
        assert engine._base_model_weights is None


# ── Frame batching tests ──


class TestFrameBatching:
    def test_process_frame_batch_sync(self):
        from yunshu_engine.video_engine import VideoEngine, FrameBatch

        engine = VideoEngine()
        batch = FrameBatch(
            frames=[b"frame1", b"frame2"],
            frame_indices=[0, 1],
            width=64,
            height=64,
            batch_size=2,
        )
        results = engine.process_frame_batch_sync(batch)
        assert len(results) == 2
        assert results[0]["frame_index"] == 0
        assert results[1]["frame_index"] == 1
        assert results[0]["width"] == 64
        assert results[0]["height"] == 64

    def test_process_batch_frame_size(self):
        from yunshu_engine.video_engine import VideoEngine, FrameBatch

        engine = VideoEngine()
        batch = FrameBatch(
            frames=[b"x" * 100, b"y" * 200],
            frame_indices=[0, 5],
            width=128,
            height=128,
            batch_size=2,
        )
        results = engine.process_frame_batch_sync(batch)
        assert results[0]["frame_size_bytes"] == 100
        assert results[1]["frame_size_bytes"] == 200


# ── Env var tests ──


class TestVideoEnvVars:
    def test_lora_env_var_set(self):
        """YUNSHU_VIDEO_LORA should set adapter path."""
        import tempfile
        from yunshu_engine.video_engine import VideoEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["YUNSHU_VIDEO_LORA"] = tmpdir
            try:
                engine = VideoEngine()
                assert engine._lora_adapter_path == tmpdir
            finally:
                del os.environ["YUNSHU_VIDEO_LORA"]

    def test_lora_env_var_empty(self):
        """Empty YUNSHU_VIDEO_LORA should not set path."""
        from yunshu_engine.video_engine import VideoEngine

        old_val = os.environ.pop("YUNSHU_VIDEO_LORA", None)
        try:
            engine = VideoEngine()
            assert engine._lora_adapter_path == ""
        finally:
            if old_val is not None:
                os.environ["YUNSHU_VIDEO_LORA"] = old_val

    def test_lora_env_var_nonexistent_path(self):
        """Non-existent path in YUNSHU_VIDEO_LORA should not set path."""
        os.environ["YUNSHU_VIDEO_LORA"] = "/nonexistent/path/lora"
        try:
            from yunshu_engine.video_engine import VideoEngine
            engine = VideoEngine()
            assert engine._lora_adapter_path == ""
        finally:
            del os.environ["YUNSHU_VIDEO_LORA"]
