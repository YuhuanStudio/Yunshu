"""Upstream mlx-vlm Qwen3-Omni audio formatter behaviour Yunshu relies on. No model load."""

import pytest

pytest.importorskip("mlx_vlm")


def test_upstream_formatter_inserts_audio_token():
    """LIST_WITH_IMAGE_FIRST formatter must insert an audio placeholder."""
    from mlx_vlm.prompt_utils import MessageFormatter

    mf = MessageFormatter("qwen3_omni_moe")
    msg = mf._format_list_with_image(
        "hello", "user", False, False, num_images=0, num_audios=1, image_first=True
    )
    types = [c.get("type") for c in msg["content"]]
    assert "audio" in types


def test_upstream_formatter_no_op_without_audio():
    """No audio token when num_audios == 0 (text/image-only unaffected)."""
    from mlx_vlm.prompt_utils import MessageFormatter

    mf = MessageFormatter("qwen3_omni_moe")
    msg = mf._format_list_with_image(
        "hi", "user", False, False, num_images=0, num_audios=0, image_first=True
    )
    types = [c.get("type") for c in msg["content"]]
    assert "audio" not in types


def test_upstream_formatter_no_op_when_skip_audio_token():
    from mlx_vlm.prompt_utils import MessageFormatter

    mf = MessageFormatter("qwen3_omni_moe")
    msg = mf._format_list_with_image(
        "hi", "user", False, True, num_images=0, num_audios=2, image_first=True
    )
    types = [c.get("type") for c in msg["content"]]
    assert "audio" not in types
