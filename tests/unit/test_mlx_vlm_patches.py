"""Tests for the pinned-mlx-vlm Omni audio runtime patches.

These verify the two monkeypatches that make the Qwen3-Omni audio path work
against mlx-vlm 0.5.0 (upstream fix submitted, not yet released). No model load
required — pure patch logic.
"""

import pytest

pytest.importorskip("mlx_vlm")

from python.yunshu_engine.mlx_vlm_patches import apply_mlx_vlm_patches


def test_patches_apply_and_are_idempotent():
    # First application reports the patches it installed (or [] if already done
    # by an earlier test in the session — both are valid).
    apply_mlx_vlm_patches()
    # A subsequent call must be a no-op.
    assert apply_mlx_vlm_patches() == []


def test_fix1_audio_mask_key_accepted():
    """get_input_embeddings must accept the mask under feature_attention_mask."""
    apply_mlx_vlm_patches()
    from mlx_vlm.models.qwen3_omni_moe import qwen3_omni_moe as omni

    assert getattr(omni.Model, "_yunshu_audio_mask_patched", False) is True


def test_fix2_formatter_inserts_audio_token():
    """LIST_WITH_IMAGE_FIRST formatter must insert an audio placeholder."""
    apply_mlx_vlm_patches()
    from mlx_vlm.prompt_utils import MessageFormatter

    mf = MessageFormatter("qwen3_omni_moe")
    msg = mf._format_list_with_image(
        "hello", "user", False, False, num_images=0, num_audios=1, image_first=True
    )
    types = [c.get("type") for c in msg["content"]]
    assert "audio" in types


def test_fix2_no_op_without_audio():
    """No audio token when num_audios == 0 (text/image-only unaffected)."""
    apply_mlx_vlm_patches()
    from mlx_vlm.prompt_utils import MessageFormatter

    mf = MessageFormatter("qwen3_omni_moe")
    msg = mf._format_list_with_image(
        "hi", "user", False, False, num_images=0, num_audios=0, image_first=True
    )
    types = [c.get("type") for c in msg["content"]]
    assert "audio" not in types


def test_fix2_no_op_when_skip_audio_token():
    apply_mlx_vlm_patches()
    from mlx_vlm.prompt_utils import MessageFormatter

    mf = MessageFormatter("qwen3_omni_moe")
    msg = mf._format_list_with_image(
        "hi", "user", False, True, num_images=0, num_audios=2, image_first=True
    )
    types = [c.get("type") for c in msg["content"]]
    assert "audio" not in types
