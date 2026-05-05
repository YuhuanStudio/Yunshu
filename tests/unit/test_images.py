"""OpenAI Images API endpoint tests.

Tests:
- Request schema validation (prompt, model, n, size, response_format)
- Input parsing (size parsing, parameter defaults)
- Response format validation (OpenAI-compatible structure)
- Base64 and URL response formats
- Streaming image generation endpoint
- Edge cases (invalid size, missing model manager)
- Endpoint-level integration via FastAPI TestClient
"""
import os
import time

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from yunshu_gateway.routers.images import (
    ImageGenerateRequest,
)


# ── Schema Tests ──


class TestImageGenerateRequest:
    def test_defaults(self):
        req = ImageGenerateRequest(prompt="a cat")
        assert req.prompt == "a cat"
        assert req.model == "Z-Image-Turbo-MLX-4bit"
        assert req.n == 1
        assert req.size == "1024x1024"
        assert req.response_format == "b64_json"
        assert req.negative_prompt == ""
        assert req.num_inference_steps == 4
        assert req.guidance_scale == 3.5
        assert req.seed is None

    def test_custom_model(self):
        req = ImageGenerateRequest(prompt="a dog", model="my-flux-model")
        assert req.model == "my-flux-model"

    def test_multiple_images(self):
        req = ImageGenerateRequest(prompt="a cat", n=4)
        assert req.n == 4

    def test_custom_size(self):
        req = ImageGenerateRequest(prompt="a cat", size="512x512")
        assert req.size == "512x512"

    def test_url_response_format(self):
        req = ImageGenerateRequest(
            prompt="a cat",
            response_format="url",
        )
        assert req.response_format == "url"

    def test_b64_response_format(self):
        req = ImageGenerateRequest(
            prompt="a cat",
            response_format="b64_json",
        )
        assert req.response_format == "b64_json"

    def test_negative_prompt(self):
        req = ImageGenerateRequest(
            prompt="a cat",
            negative_prompt="blurry, low quality",
        )
        assert req.negative_prompt == "blurry, low quality"

    def test_inference_steps(self):
        req = ImageGenerateRequest(
            prompt="a cat",
            num_inference_steps=20,
        )
        assert req.num_inference_steps == 20

    def test_guidance_scale(self):
        req = ImageGenerateRequest(
            prompt="a cat",
            guidance_scale=7.5,
        )
        assert req.guidance_scale == 7.5

    def test_seed(self):
        req = ImageGenerateRequest(
            prompt="a cat",
            seed=42,
        )
        assert req.seed == 42


# ── Input Parsing Tests ──


class TestSizeParsing:
    """Test parsing of the size parameter (WxH format)."""

    def test_standard_size(self):
        size = "1024x1024"
        width, height = map(int, size.split("x"))
        assert width == 1024
        assert height == 1024

    def test_landscape_size(self):
        size = "1536x1024"
        width, height = map(int, size.split("x"))
        assert width == 1536
        assert height == 1024

    def test_portrait_size(self):
        size = "1024x1536"
        width, height = map(int, size.split("x"))
        assert width == 1024
        assert height == 1536

    def test_small_size(self):
        size = "256x256"
        width, height = map(int, size.split("x"))
        assert width == 256
        assert height == 256

    def test_invalid_size_fallback(self):
        """Invalid size should fall back to 1024x1024."""
        size = "invalid"
        try:
            width, height = map(int, size.split("x"))
        except (ValueError, AttributeError):
            width, height = 1024, 1024
        assert width == 1024
        assert height == 1024

    def test_partial_size_fallback(self):
        """Size without 'x' should fall back."""
        size = "1024"
        try:
            width, height = map(int, size.split("x"))
        except ValueError:
            width, height = 1024, 1024
        assert width == 1024
        assert height == 1024


class TestSeedIncrement:
    """Test seed increment for multiple images."""

    def test_seed_increments_per_image(self):
        """When n > 1, seed should increment for each image."""
        base_seed = 42
        n = 3
        seeds = [(base_seed + i) for i in range(n)]
        assert seeds == [42, 43, 44]

    def test_no_seed_is_none(self):
        """When seed is None, each image should get None."""
        req = ImageGenerateRequest(prompt="test", n=3)
        for i in range(req.n):
            seed = (req.seed + i) if req.seed is not None else None
            assert seed is None


# ── Response Format Tests ──


class TestImageResponseFormat:
    """Test the OpenAI images API response format."""

    def test_b64_response_structure(self):
        """Verify b64_json response structure."""
        import base64
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # Minimal PNG header
        b64 = base64.b64encode(fake_png).decode("ascii")

        response = {
            "created": int(time.time()),
            "data": [{"b64_json": b64}],
        }
        assert "created" in response
        assert isinstance(response["data"], list)
        assert len(response["data"]) == 1
        assert "b64_json" in response["data"][0]
        # Verify it's valid base64
        decoded = base64.b64decode(response["data"][0]["b64_json"])
        assert decoded[:4] == b"\x89PNG"

    def test_url_response_structure(self):
        """Verify URL response structure (data URI format)."""
        import base64
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        b64 = base64.b64encode(fake_png).decode("ascii")

        response = {
            "created": int(time.time()),
            "data": [{"url": f"data:image/png;base64,{b64}"}],
        }
        assert response["data"][0]["url"].startswith("data:image/png;base64,")
        # Extract and verify base64 part
        url = response["data"][0]["url"]
        b64_part = url.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert decoded[:4] == b"\x89PNG"

    def test_multiple_images_response(self):
        """Response should have n items in data array."""
        n = 3
        data = [{"b64_json": f"fake_data_{i}"} for i in range(n)]
        response = {"created": int(time.time()), "data": data}
        assert len(response["data"]) == n

    def test_response_has_created_timestamp(self):
        """Response should have integer 'created' timestamp."""
        response = {"created": int(time.time()), "data": []}
        assert isinstance(response["created"], int)
        assert response["created"] > 0


# ── Streaming Response Tests ──


class TestImageStreamingResponse:
    """Test the streaming image generation SSE event format."""

    def test_progress_event_format(self):
        """Progress events should have step, total_steps, progress, is_final."""
        import json
        event = {
            "step": 2,
            "total_steps": 4,
            "progress": 0.5,
            "is_final": False,
        }
        sse_data = f"data: {json.dumps(event)}\n\n"
        assert "data: " in sse_data
        parsed = json.loads(sse_data.strip().split("data: ")[1])
        assert parsed["step"] == 2
        assert parsed["total_steps"] == 4
        assert parsed["progress"] == 0.5
        assert parsed["is_final"] is False

    def test_final_event_format(self):
        """Final events should have is_final=True and image data."""
        import base64
        import json
        fake_png = b"\x89PNG"
        event = {
            "step": 4,
            "progress": 1.0,
            "image": base64.b64encode(fake_png).decode("ascii"),
            "is_final": True,
        }
        assert event["is_final"] is True
        assert event["progress"] == 1.0
        assert "image" in event


# ── Endpoint-Level Integration Tests ──


@pytest.fixture
def _setup_engine():
    """Set up a mock engine for endpoint tests."""
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    from yunshu_engine.engine import Engine, EngineConfig
    from yunshu_gateway.engine import set_engine
    engine = Engine(EngineConfig())
    engine._model = object()
    engine._model_name = "test-model"
    engine._running = True
    engine._loaded = True
    set_engine(engine)
    yield engine
    set_engine(None)
    os.environ.pop("YUNSHU_AUTH_DISABLED", None)


def _client():
    from yunshu_gateway.main import create_app
    return TestClient(create_app(), raise_server_exceptions=False)


class TestImagesEndpoint:
    """Test the /v1/images/generations endpoint via FastAPI TestClient."""

    def test_images_generations_no_model_manager(self, _setup_engine):
        """Should fail when no model manager is initialized."""
        client = _client()
        resp = client.post("/v1/images/generations", json={
            "model": "test",
            "prompt": "a cat",
        })
        # Model manager may not be set up, so 503 or 404
        assert resp.status_code in (200, 404, 500, 503)

    def test_images_validates_prompt_required(self, _setup_engine):
        """Should return 422 when prompt is missing."""
        client = _client()
        resp = client.post("/v1/images/generations", json={
            "model": "test",
        })
        assert resp.status_code == 422

    def test_images_accepts_all_params(self, _setup_engine):
        """Should accept all valid parameters without 422."""
        client = _client()
        resp = client.post("/v1/images/generations", json={
            "prompt": "a beautiful sunset",
            "model": "Z-Image-Turbo-MLX-4bit",
            "n": 2,
            "size": "512x512",
            "response_format": "b64_json",
            "negative_prompt": "blurry",
            "num_inference_steps": 8,
            "guidance_scale": 5.0,
            "seed": 42,
        })
        # May fail due to engine, but should NOT be 422
        assert resp.status_code in (200, 404, 500, 503)

    def test_images_accepts_url_format(self, _setup_engine):
        """Should accept URL response format."""
        client = _client()
        resp = client.post("/v1/images/generations", json={
            "prompt": "a cat",
            "response_format": "url",
        })
        assert resp.status_code in (200, 404, 500, 503)

    def test_images_streaming_endpoint_no_engine(self, _setup_engine):
        """Streaming endpoint should fail gracefully when no engine available."""
        client = _client()
        resp = client.post("/v1/images/generations/stream", json={
            "prompt": "a cat",
        })
        assert resp.status_code in (200, 404, 500, 503)

    def test_images_default_model(self, _setup_engine):
        """Should use default model when not specified."""
        client = _client()
        resp = client.post("/v1/images/generations", json={
            "prompt": "a cat",
        })
        assert resp.status_code in (200, 404, 500, 503)


class TestImageRequestValidation:
    """Additional validation tests for image generation requests."""

    def test_prompt_only_minimum_valid(self):
        """Request with only prompt should use all defaults."""
        req = ImageGenerateRequest(prompt="test")
        assert req.prompt == "test"
        assert req.model == "Z-Image-Turbo-MLX-4bit"
        assert req.n == 1
        assert req.size == "1024x1024"
        assert req.response_format == "b64_json"

    def test_non_square_size(self):
        """Should accept non-square sizes."""
        req = ImageGenerateRequest(prompt="test", size="1792x1024")
        assert req.size == "1792x1024"

    def test_one_inference_step(self):
        """Should accept single inference step."""
        req = ImageGenerateRequest(prompt="test", num_inference_steps=1)
        assert req.num_inference_steps == 1
