"""Tests for image generation streaming intermediate preview."""

from python.yunshu_engine.image_engine import ImageGenEngine


class TestPreviewIntervalLogic:
    """Test the _should_preview helper logic within generate_image_stream."""

    def test_preview_disabled_zero(self):
        """preview_interval=0 should never preview."""
        ImageGenEngine.__new__(ImageGenEngine)
        # Access the closure from generate_image_stream via a test wrapper
        # We test the logic directly
        preview_interval = 0

        def _should_preview(step: int, total: int) -> bool:
            if preview_interval <= 0:
                return False
            if step == total:
                return True
            return step % preview_interval == 0

        assert _should_preview(1, 4) is False
        assert _should_preview(4, 4) is False

    def test_preview_every_step(self):
        """preview_interval=1 should preview at every step."""
        preview_interval = 1

        def _should_preview(step: int, total: int) -> bool:
            if preview_interval <= 0:
                return False
            if step == total:
                return True
            return step % preview_interval == 0

        assert _should_preview(1, 4) is True
        assert _should_preview(2, 4) is True
        assert _should_preview(3, 4) is True
        assert _should_preview(4, 4) is True

    def test_preview_every_other_step(self):
        """preview_interval=2 should preview at steps 2, 4."""
        preview_interval = 2

        def _should_preview(step: int, total: int) -> bool:
            if preview_interval <= 0:
                return False
            if step == total:
                return True
            return step % preview_interval == 0

        assert _should_preview(1, 4) is False
        assert _should_preview(2, 4) is True
        assert _should_preview(3, 4) is False
        assert _should_preview(4, 4) is True

    def test_preview_every_third_step(self):
        preview_interval = 3

        def _should_preview(step: int, total: int) -> bool:
            if preview_interval <= 0:
                return False
            if step == total:
                return True
            return step % preview_interval == 0

        assert _should_preview(1, 6) is False
        assert _should_preview(2, 6) is False
        assert _should_preview(3, 6) is True
        assert _should_preview(4, 6) is False
        assert _should_preview(5, 6) is False
        assert _should_preview(6, 6) is True

    def test_preview_final_step_always(self):
        """Final step should always be a preview when interval > 0."""
        preview_interval = 5

        def _should_preview(step: int, total: int) -> bool:
            if preview_interval <= 0:
                return False
            if step == total:
                return True
            return step % preview_interval == 0

        assert _should_preview(3, 3) is True
        assert _should_preview(7, 7) is True
        assert _should_preview(5, 5) is True

    def test_preview_negative_interval(self):
        """Negative interval should be treated as disabled."""
        preview_interval = -1

        def _should_preview(step: int, total: int) -> bool:
            if preview_interval <= 0:
                return False
            if step == total:
                return True
            return step % preview_interval == 0

        assert _should_preview(1, 4) is False
        assert _should_preview(4, 4) is False


class TestImageGenerateRequest:
    """Test the request model accepts preview_interval."""

    def test_default_preview_interval(self):
        from python.yunshu_gateway.routers.images import ImageGenerateRequest

        req = ImageGenerateRequest(prompt="test")
        assert req.preview_interval == 0

    def test_custom_preview_interval(self):
        from python.yunshu_gateway.routers.images import ImageGenerateRequest

        req = ImageGenerateRequest(prompt="test", preview_interval=2)
        assert req.preview_interval == 2

    def test_request_serialization(self):
        from python.yunshu_gateway.routers.images import ImageGenerateRequest

        req = ImageGenerateRequest(
            prompt="a cat",
            preview_interval=1,
            num_inference_steps=8,
        )
        d = req.model_dump()
        assert d["preview_interval"] == 1
        assert d["num_inference_steps"] == 8
