"""Tests for ControlNet + Depth-guided generation infrastructure."""

import base64
import io

import numpy as np
import pytest
from PIL import Image as PILImage


def _make_png(width=64, height=64, color=(128, 64, 32)) -> bytes:
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    pil = PILImage.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


# ── ConditioningPreprocessor tests ──


class TestCannyEdges:
    def test_canny_output_shape(self):
        from yunshu_engine.controlnet_engine import ConditioningPreprocessor

        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        edges = ConditioningPreprocessor.canny_edges(img, 100, 200)
        assert edges.shape == (64, 64, 3)

    def test_canny_output_dtype(self):
        from yunshu_engine.controlnet_engine import ConditioningPreprocessor

        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        edges = ConditioningPreprocessor.canny_edges(img)
        assert edges.dtype == np.uint8

    def test_canny_solid_image_minimal_edges(self):
        """Solid color image should produce few/no edges."""
        from yunshu_engine.controlnet_engine import ConditioningPreprocessor

        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        edges = ConditioningPreprocessor.canny_edges(img)
        # Solid image should have mostly black (no edges)
        assert np.sum(edges > 0) < 64 * 64 * 3 * 0.1  # Less than 10% edge pixels

    def test_canny_high_contrast(self):
        """High contrast image should produce more edges."""
        from yunshu_engine.controlnet_engine import ConditioningPreprocessor

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:32, :, :] = 255  # Top half white, bottom half black
        edges = ConditioningPreprocessor.canny_edges(img)
        # Should detect the horizontal boundary
        assert np.sum(edges > 0) > 0


class TestNormalizeDepth:
    def test_normalize_shape(self):
        from yunshu_engine.controlnet_engine import ConditioningPreprocessor

        depth = np.random.rand(64, 64).astype(np.float32)
        result = ConditioningPreprocessor.normalize_depth(depth)
        assert result.shape == (64, 64, 3)

    def test_normalize_range(self):
        from yunshu_engine.controlnet_engine import ConditioningPreprocessor

        depth = np.random.rand(32, 48).astype(np.float32) * 100
        result = ConditioningPreprocessor.normalize_depth(depth)
        assert result.min() >= 0.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_normalize_flat_depth(self):
        """Constant depth map should normalize to all zeros."""
        from yunshu_engine.controlnet_engine import ConditioningPreprocessor

        depth = np.full((32, 32), 5.0, dtype=np.float32)
        result = ConditioningPreprocessor.normalize_depth(depth)
        assert np.allclose(result, 0.0)

    def test_normalize_3channel_input(self):
        from yunshu_engine.controlnet_engine import ConditioningPreprocessor

        depth = np.random.rand(32, 32, 3).astype(np.float32)
        result = ConditioningPreprocessor.normalize_depth(depth)
        assert result.shape == (32, 32, 3)


# ── ControlNetBlock tests ──


class TestControlNetBlock:
    def test_init(self):
        from yunshu_engine.controlnet_engine import ControlNetBlock, ControlNetConfig

        config = ControlNetConfig(condition_type="canny", controlnet_strength=0.8)
        block = ControlNetBlock(config)
        assert block._config.controlnet_strength == 0.8

    def test_inject_condition(self):
        """Test condition injection modifies latents.

        The no-weights path standardizes the conditioning to zero-mean/unit-std
        before adding it (so off-distribution VAE latents can't collapse the
        output to black). A UNIFORM condition therefore correctly contributes
        nothing — real conditioning maps (canny/depth) have spatial variance, so
        use a structured condition here. step=0 keeps the early-step fade weight
        at maximum so the bias is applied.
        """
        import mlx.core as mx

        from yunshu_engine.controlnet_engine import ControlNetBlock

        block = ControlNetBlock()
        latents = mx.zeros((16, 1, 8, 8), dtype=mx.float16)
        condition = mx.random.normal((16, 1, 8, 8)).astype(mx.float16)
        result = block.inject_condition(latents, condition, step=0, total_steps=4)
        # Structured condition + max fade weight → result must differ from base.
        assert not np.allclose(np.array(result), np.array(latents))
        mx.eval(result)

    def test_inject_outside_range(self):
        """Conditioning should be identity when step is outside range."""
        import mlx.core as mx

        from yunshu_engine.controlnet_engine import ControlNetBlock, ControlNetConfig

        config = ControlNetConfig(start_step=0.5, end_step=1.0)
        block = ControlNetBlock(config)
        latents = mx.ones((16, 1, 8, 8), dtype=mx.float16)
        condition = mx.ones((16, 1, 8, 8), dtype=mx.float16) * 5.0
        result = block.inject_condition(latents, condition, step=0, total_steps=4)
        # Step 0/4 = 0.0 < start_step 0.5, should be identity
        np.testing.assert_allclose(np.array(result), np.array(latents), rtol=1e-5)

    def test_inject_zero_strength(self):
        """Zero strength should produce minimal modification."""
        import mlx.core as mx

        from yunshu_engine.controlnet_engine import ControlNetBlock, ControlNetConfig

        config = ControlNetConfig(controlnet_strength=0.0)
        block = ControlNetBlock(config)
        latents = mx.ones((16, 1, 8, 8), dtype=mx.float16)
        condition = mx.ones((16, 1, 8, 8), dtype=mx.float16) * 10.0
        result = block.inject_condition(latents, condition, step=1, total_steps=4)
        np.testing.assert_allclose(np.array(result), np.array(latents), atol=1e-5)

    def test_inject_3d_condition(self):
        """Test with 3D condition latents (no frame dim)."""
        import mlx.core as mx

        from yunshu_engine.controlnet_engine import ControlNetBlock

        block = ControlNetBlock()
        latents = mx.zeros((16, 1, 8, 8), dtype=mx.float16)
        condition = mx.ones((16, 8, 8), dtype=mx.float16)
        result = block.inject_condition(latents, condition, step=1, total_steps=4)
        assert result.shape == latents.shape
        mx.eval(result)


# ── ControlNetConfig tests ──


class TestControlNetConfig:
    def test_defaults(self):
        from yunshu_engine.controlnet_engine import ControlNetConfig

        config = ControlNetConfig()
        assert config.condition_type == "canny"
        assert config.controlnet_strength == 1.0
        assert config.condition_channels == 16

    def test_custom(self):
        from yunshu_engine.controlnet_engine import ControlNetConfig

        config = ControlNetConfig(
            condition_type="depth",
            controlnet_strength=0.5,
            start_step=0.2,
            end_step=0.8,
        )
        assert config.condition_type == "depth"
        assert config.controlnet_strength == 0.5
        assert config.start_step == 0.2


# ── ConditioningResult tests ──


class TestConditioningResult:
    def test_creation(self):
        import mlx.core as mx

        from yunshu_engine.controlnet_engine import ConditioningResult

        result = ConditioningResult(
            condition_latents=mx.zeros((16, 8, 8)),
            condition_type="canny",
            strength=0.8,
            metadata={"width": 1024},
        )
        assert result.condition_type == "canny"
        assert result.strength == 0.8
        assert result.metadata["width"] == 1024


# ── DepthGuider tests ──


class TestDepthGuider:
    def test_prepare_depth_latents_no_encoder(self):
        """Should produce zeros without VAE encoder."""
        import mlx.core as mx

        from yunshu_engine.controlnet_engine import DepthGuider

        depth_png = _make_png(64, 64, color=(128, 128, 128))
        result = DepthGuider.prepare_depth_latents(
            depth_image=depth_png,
            vae=None,
            width=64,
            height=64,
        )
        assert result.shape == (16, 8, 8)
        mx.eval(result)

    def test_concatenate_depth(self):
        """Test depth concatenation."""
        import mlx.core as mx

        from yunshu_engine.controlnet_engine import DepthGuider

        noise = mx.zeros((16, 1, 8, 8))
        depth = mx.ones((16, 1, 8, 8))
        result = DepthGuider.concatenate_depth(noise, depth)
        assert result.shape == (32, 1, 8, 8), f"Shape: {result.shape}"
        mx.eval(result)

    def test_concatenate_depth_3d(self):
        """Test concatenation with 3D depth latents."""
        import mlx.core as mx

        from yunshu_engine.controlnet_engine import DepthGuider

        noise = mx.zeros((16, 1, 8, 8))
        depth = mx.ones((16, 8, 8))
        result = DepthGuider.concatenate_depth(noise, depth)
        assert result.shape == (32, 1, 8, 8)
        mx.eval(result)


# ── Gateway endpoint tests ──


class TestControlNetEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from yunshu_gateway.main import create_app

        app = create_app()
        return TestClient(app)

    def test_controlnet_no_model(self, client):
        """ControlNet endpoint should 404 when no image engine loaded."""
        img_b64 = base64.b64encode(_make_png(64, 64)).decode()
        resp = client.post(
            "/v1/images/controlnet",
            json={
                "prompt": "a cat sitting on a table",
                "image": img_b64,
                "condition_type": "canny",
            },
        )
        assert resp.status_code in (404, 503)

    def test_controlnet_invalid_base64(self, client):
        resp = client.post(
            "/v1/images/controlnet",
            json={
                "prompt": "test",
                "image": "not-valid!!!",
            },
        )
        assert resp.status_code == 400

    def test_controlnet_request_model(self):
        from yunshu_gateway.routers.images import ImageControlNetRequest

        req = ImageControlNetRequest(
            prompt="a landscape",
            image="abc",
            condition_type="depth",
            controlnet_strength=0.7,
            canny_low=50,
            canny_high=150,
        )
        assert req.condition_type == "depth"
        assert req.controlnet_strength == 0.7


class TestDepthGuidedEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from yunshu_gateway.main import create_app

        app = create_app()
        return TestClient(app)

    def test_depth_guided_no_model(self, client):
        img_b64 = base64.b64encode(_make_png(64, 64)).decode()
        resp = client.post(
            "/v1/images/depth-guided",
            json={
                "prompt": "a mountain scene",
                "depth_image": img_b64,
            },
        )
        assert resp.status_code in (404, 503)

    def test_depth_guided_invalid_base64(self, client):
        resp = client.post(
            "/v1/images/depth-guided",
            json={
                "prompt": "test",
                "depth_image": "not-valid!!!",
            },
        )
        assert resp.status_code == 400

    def test_depth_guided_request_model(self):
        from yunshu_gateway.routers.images import ImageDepthGuidedRequest

        req = ImageDepthGuidedRequest(
            prompt="test",
            depth_image="abc",
            depth_strength=0.5,
        )
        assert req.depth_strength == 0.5
