"""Tests for image variation and edit endpoints."""
import pytest


class TestImageVariationsRequest:
    def test_defaults(self):
        from yunshu_gateway.routers.images import ImageVariationsRequest
        req = ImageVariationsRequest(image="dGVzdA==")
        assert req.model == "Z-Image-Turbo-MLX-4bit"
        assert req.n == 1
        assert req.num_inference_steps == 4

    def test_with_params(self):
        from yunshu_gateway.routers.images import ImageVariationsRequest
        req = ImageVariationsRequest(
            image="dGVzdA==",
            n=3,
            size="512x512",
            seed=42,
        )
        assert req.n == 3
        assert req.seed == 42


class TestImageEditsRequest:
    def test_requires_prompt(self):
        from yunshu_gateway.routers.images import ImageEditsRequest
        req = ImageEditsRequest(image="dGVzdA==", prompt="make it blue")
        assert req.prompt == "make it blue"

    def test_defaults(self):
        from yunshu_gateway.routers.images import ImageEditsRequest
        req = ImageEditsRequest(image="dGVzdA==", prompt="edit")
        assert req.model == "Z-Image-Turbo-MLX-4bit"
        assert req.n == 1
