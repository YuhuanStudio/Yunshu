"""Tests for OCREngine image-token accounting (Bug 3)."""

from __future__ import annotations

from types import SimpleNamespace


def _make_engine_with_vision_config(patch_size: int = 14, merge_size: int = 2):
    """Build a partially-mocked OCREngine whose vision_config behaves like
    GLM-OCR's (patch=14, merge=2). No actual model weights are loaded.
    """
    from yunshu_engine.ocr_engine import OCREngine

    engine = OCREngine("/fake/glm-ocr")
    fake_model = SimpleNamespace(
        config=SimpleNamespace(
            vision_config={"patch_size": patch_size, "spatial_merge_size": merge_size}
        )
    )
    engine._model = fake_model
    return engine


def _save_image(tmp_path, width: int, height: int):
    from PIL import Image, ImageDraw

    path = tmp_path / f"img_{width}x{height}.png"
    img = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(img).text((10, 10), "TEST", fill="black")
    img.save(str(path))
    return str(path)


def test_estimate_image_tokens_glm_ocr_500x200(tmp_path):
    engine = _make_engine_with_vision_config(patch_size=14, merge_size=2)
    img_path = _save_image(tmp_path, 500, 200)
    tokens = engine._estimate_image_tokens(img_path)
    # (500 // 14) * (200 // 14) // 4 == 35 * 14 // 4 == 122
    assert tokens == 122


def test_estimate_image_tokens_scales_with_image(tmp_path):
    engine = _make_engine_with_vision_config(patch_size=14, merge_size=2)
    small = engine._estimate_image_tokens(_save_image(tmp_path, 224, 224))
    large = engine._estimate_image_tokens(_save_image(tmp_path, 1024, 1024))
    assert large > small > 0
    # Roughly quadratic: 4x linear → ~16x tokens. Allow generous slack.
    assert large >= small * 8


def test_estimate_image_tokens_minimum_one(tmp_path):
    """Tiny images still report at least one image token."""
    engine = _make_engine_with_vision_config(patch_size=14, merge_size=2)
    tokens = engine._estimate_image_tokens(_save_image(tmp_path, 8, 8))
    assert tokens >= 1


def test_estimate_image_tokens_handles_missing_config(tmp_path):
    """No vision_config → fall back to defaults (patch=14, merge=1)."""
    from yunshu_engine.ocr_engine import OCREngine

    engine = OCREngine("/fake/no-config")
    engine._model = SimpleNamespace(config=None)
    tokens = engine._estimate_image_tokens(_save_image(tmp_path, 224, 224))
    # 224//14 == 16, 16*16 == 256 (no merge division when merge_size==1)
    assert tokens == 256


def test_estimate_image_tokens_unreadable_returns_zero(tmp_path):
    engine = _make_engine_with_vision_config()
    # Non-existent path → PIL raises, we return 0.
    assert engine._estimate_image_tokens(str(tmp_path / "does_not_exist.png")) == 0


def test_get_vision_config_handles_dict(tmp_path):
    from yunshu_engine.ocr_engine import OCREngine

    engine = OCREngine("/fake")
    engine._model = SimpleNamespace(
        config={"vision_config": {"patch_size": 16, "spatial_merge_size": 1}}
    )
    cfg = engine._get_vision_config()
    assert cfg["patch_size"] == 16
    assert cfg["spatial_merge_size"] == 1


def test_estimate_image_tokens_qwen_style_patch16(tmp_path):
    """A Qwen-style VLM with patch=16, merge=2 on a 768x768 image."""
    engine = _make_engine_with_vision_config(patch_size=16, merge_size=2)
    img_path = _save_image(tmp_path, 768, 768)
    # (768//16)^2 / 4 == 48^2 / 4 == 576
    assert engine._estimate_image_tokens(img_path) == 576
