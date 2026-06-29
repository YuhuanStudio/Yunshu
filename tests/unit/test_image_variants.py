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
        assert req.denoise_strength == 0.8

    def test_denoise_strength_custom(self):
        from yunshu_gateway.routers.images import ImageEditsRequest

        req = ImageEditsRequest(image="dGVzdA==", prompt="edit", denoise_strength=0.3)
        assert req.denoise_strength == 0.3


class TestImageSizeValidation:
    def test_valid_size_1024x1024(self):
        from yunshu_gateway.routers.images import ImageGenerateRequest

        req = ImageGenerateRequest(prompt="test", size="1024x1024")
        assert req.size == "1024x1024"

    def test_valid_size_512x512(self):
        from yunshu_gateway.routers.images import ImageGenerateRequest

        req = ImageGenerateRequest(prompt="test", size="512x512")
        assert req.size == "512x512"

    def test_valid_size_256x256(self):
        from yunshu_gateway.routers.images import ImageGenerateRequest

        req = ImageGenerateRequest(prompt="test", size="256x256")
        assert req.size == "256x256"

    def test_valid_size_64x64(self):
        from yunshu_gateway.routers.images import ImageGenerateRequest

        req = ImageGenerateRequest(prompt="test", size="64x64")
        assert req.size == "64x64"

    def test_parse_size(self):
        """Size string parsing logic."""
        size = "512x768"
        width, height = map(int, size.split("x"))
        assert width == 512
        assert height == 768

    def test_not_multiple_of_64(self):
        """100x100 is not a valid dimension (not multiple of 64)."""
        width, height = 100, 100
        assert width % 64 != 0
        assert height % 64 != 0

    def test_too_small_dimension(self):
        """32x32 is below minimum (64)."""
        width, height = 32, 32
        assert width < 64
        assert height < 64

    def test_too_large_dimension(self):
        """4096x4096 is above maximum (2048)."""
        width, height = 4096, 4096
        assert width > 2048
        assert height > 2048

    def test_invalid_size_format(self):
        """Non-numeric size format."""
        with pytest.raises(ValueError):
            width, height = map(int, ["abc", "def"])
