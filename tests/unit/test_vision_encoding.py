"""Tests for Vision Encoding Strategies — pluggable vision encoders.

Covers:
  - VisionEncodingStrategy enum values
  - VisionEncoder ABC contract
  - MLXVLMEncoder: supports_model, encode_image fallbacks
  - QwenVLEncoder: supports_model detection, Qwen-VL specific path
  - LLaVAEncoder: supports_model detection, pooling strategies, anyres
  - CustomEncoder: user-provided encoder function
  - VisionEncoderFactory: create_encoder, register_encoder, auto-detection
  - Environment variable override (YUNSHU_VISION_ENCODER)
  - Integration helpers: create_vision_encoder_for_model, get_vision_encoding_strategy
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import mlx.core as mx

from yunshu_engine.vision_encoding import (
    CustomEncoder,
    LLaVAEncoder,
    MLXVLMEncoder,
    QwenVLEncoder,
    VisionEncoder,
    VisionEncoderFactory,
    VisionEncodingStrategy,
    create_vision_encoder_for_model,
    get_vision_encoding_strategy,
)

# ── Helpers ──


class FakeConfig:
    """Fake model config with model_type."""

    def __init__(self, model_type="qwen2_vl", hidden_size=4096):
        self.model_type = model_type
        self.hidden_size = hidden_size


class FakeModel:
    """Fake model for testing supports_model and encode."""

    def __init__(
        self, model_type="qwen2_vl", module="mlx_vlm.models.qwen2_vl", extra_attrs=None
    ):
        self.config = FakeConfig(model_type)
        self._module = module
        if extra_attrs:
            for k, v in extra_attrs.items():
                setattr(self, k, v)

    def __class__(self):
        # Override module for type(model).__module__
        pass


def _make_model_with_module(module_name, model_type="qwen2_vl", extra=None):
    """Create a mock model with a specific __module__.

    Uses spec=[] on MagicMock to prevent attribute auto-creation, then
    explicitly sets only the needed attributes.
    """
    model = MagicMock(spec=[])
    model.config = FakeConfig(model_type)
    type(model).__module__ = module_name
    if extra:
        for k, v in extra.items():
            setattr(model, k, v)
    return model


# ── VisionEncodingStrategy ──


class TestVisionEncodingStrategy:
    def test_enum_values(self):
        assert VisionEncodingStrategy.MLX_VLM == "mlx_vlm"
        assert VisionEncodingStrategy.QWEN_VL == "qwen_vl"
        assert VisionEncodingStrategy.LLAVA == "llava"
        assert VisionEncodingStrategy.CUSTOM == "custom"

    def test_from_string(self):
        assert VisionEncodingStrategy("mlx_vlm") == VisionEncodingStrategy.MLX_VLM
        assert VisionEncodingStrategy("qwen_vl") == VisionEncodingStrategy.QWEN_VL

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            VisionEncodingStrategy("invalid")


# ── VisionEncoder ABC ──


class TestVisionEncoderABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            VisionEncoder()

    def test_subclass_must_implement_encode_image(self):
        class Incomplete(VisionEncoder):
            def supports_model(self, model):
                return True

        with pytest.raises(TypeError):
            Incomplete()

    def test_complete_subclass(self):
        class Complete(VisionEncoder):
            def encode_image(self, image, model, processor=None):
                return mx.zeros((1, 1, 10))

            def supports_model(self, model):
                return True

        enc = Complete()
        assert enc.supports_model(None)
        result = enc.encode_image(None, None)
        assert result.shape == (1, 1, 10)

    def test_default_encode_images(self):
        """encode_images should default to sequential encode_image calls."""

        class Single(VisionEncoder):
            def encode_image(self, image, model, processor=None):
                return mx.ones((1, 1, 4))

            def supports_model(self, model):
                return True

        enc = Single()
        results = enc.encode_images([1, 2, 3], None)
        assert len(results) == 3
        for r in results:
            assert r.shape == (1, 1, 4)


# ── MLXVLMEncoder ──


class TestMLXVLMEncoder:
    def test_supports_mlx_vlm_model(self):
        model = _make_model_with_module("mlx_vlm.models.qwen2_vl")
        enc = MLXVLMEncoder()
        assert enc.supports_model(model)

    def test_supports_model_with_vision_tower(self):
        model = MagicMock()
        model.vision_tower = MagicMock()
        enc = MLXVLMEncoder()
        assert enc.supports_model(model)

    def test_does_not_support_plain_model(self):
        model = MagicMock(spec=[])
        enc = MLXVLMEncoder()
        assert not enc.supports_model(model)

    def test_encode_image_without_processor_raises(self):
        enc = MLXVLMEncoder()
        with pytest.raises(ValueError, match="processor"):
            enc.encode_image("test.jpg", MagicMock())

    def test_encode_image_returns_array_on_failure(self):
        """encode_image should return an mx.array even on failure."""
        model = MagicMock(spec=[])
        model.config = FakeConfig(hidden_size=256)
        enc = MLXVLMEncoder()
        # Pass a processor but force failure via nonexistent image processing
        processor = MagicMock()
        processor._prepare_image_inputs.side_effect = Exception("test failure")
        result = enc.encode_image("test.jpg", model, processor=processor)
        # Should return zeros array from fallback
        assert isinstance(result, mx.array)
        assert result.shape[-1] == 256

    def test_encode_image_fallback_on_import_error(self):
        model = MagicMock()
        model.config = FakeConfig(hidden_size=128)
        enc = MLXVLMEncoder()
        with patch.dict("sys.modules", {"mlx_vlm.utils": None}):
            # Force import failure path
            result = enc.encode_image("test.jpg", model, processor=MagicMock())
            # Should return zeros array due to import error
            assert isinstance(result, mx.array)


# ── QwenVLEncoder ──


class TestQwenVLEncoder:
    def test_supports_qwen2_vl(self):
        model = _make_model_with_module("mlx_vlm.models.qwen2_vl", "qwen2_vl")
        enc = QwenVLEncoder()
        assert enc.supports_model(model)

    def test_supports_qwen3_vl(self):
        model = _make_model_with_module("mlx_vlm.models.qwen3_vl", "qwen3_vl")
        enc = QwenVLEncoder()
        assert enc.supports_model(model)

    def test_supports_qwen3_omni_moe(self):
        model = _make_model_with_module(
            "mlx_vlm.models.qwen3_omni_moe", "qwen3_omni_moe"
        )
        enc = QwenVLEncoder()
        assert enc.supports_model(model)

    def test_does_not_support_llava(self):
        model = _make_model_with_module("mlx_vlm.models.llava", "llava")
        enc = QwenVLEncoder()
        assert not enc.supports_model(model)

    def test_does_not_support_unknown(self):
        model = _make_model_with_module("mlx_vlm.models.unknown", "unknown")
        enc = QwenVLEncoder()
        assert not enc.supports_model(model)

    def test_supports_via_vision_tower_and_rope(self):
        model = MagicMock()
        model.config = FakeConfig("unknown")
        model.vision_tower = MagicMock()
        model.rope = MagicMock()
        enc = QwenVLEncoder()
        assert enc.supports_model(model)

    def test_encode_image_fallback(self):
        """When Qwen-VL specific path fails, should fall back to MLXVLMEncoder."""
        model = MagicMock(spec=[])
        model.config = FakeConfig(hidden_size=64)
        enc = QwenVLEncoder()
        # No vision_tower -> QwenVL path returns None -> falls to MLXVLMEncoder
        # MLXVLMEncoder requires processor; without it returns zeros via exception handling
        processor = MagicMock()
        processor._prepare_image_inputs.side_effect = Exception("no image")
        result = enc.encode_image("test.jpg", model, processor=processor)
        assert isinstance(result, mx.array)

    def test_encode_images_batch(self):
        model = MagicMock(spec=[])
        model.config = FakeConfig(hidden_size=64)
        enc = QwenVLEncoder()
        processor = MagicMock()
        processor._prepare_image_inputs.side_effect = Exception("no image")
        results = enc.encode_images(["img1.jpg", "img2.jpg"], model, processor)
        assert len(results) == 2

    def test_encode_images_empty(self):
        enc = QwenVLEncoder()
        assert enc.encode_images([], None) == []

    def test_encode_count_increments(self):
        enc = QwenVLEncoder()
        model = MagicMock(spec=[])
        model.config = FakeConfig(hidden_size=64)
        processor = MagicMock()
        processor._prepare_image_inputs.side_effect = Exception("no image")
        assert enc._encode_count == 0
        enc.encode_image("test.jpg", model, processor=processor)
        assert enc._encode_count == 1
        enc.encode_image("test2.jpg", model, processor=processor)
        assert enc._encode_count == 2


# ── LLaVAEncoder ──


class TestLLaVAEncoder:
    def test_supports_llava(self):
        model = _make_model_with_module("mlx_vlm.models.llava", "llava")
        enc = LLaVAEncoder()
        assert enc.supports_model(model)

    def test_supports_llava_next(self):
        model = _make_model_with_module("mlx_vlm.models.llava_next", "llava_next")
        enc = LLaVAEncoder()
        assert enc.supports_model(model)

    def test_supports_phi3_v(self):
        model = _make_model_with_module("mlx_vlm.models.phi3_v", "phi3_v")
        enc = LLaVAEncoder()
        assert enc.supports_model(model)

    def test_does_not_support_qwen(self):
        model = _make_model_with_module("mlx_vlm.models.qwen2_vl", "qwen2_vl")
        enc = LLaVAEncoder()
        assert not enc.supports_model(model)

    def test_does_not_support_unknown(self):
        model = _make_model_with_module("mlx_vlm.models.unknown", "unknown")
        enc = LLaVAEncoder()
        assert not enc.supports_model(model)

    def test_supports_via_vision_tower_and_projector(self):
        model = MagicMock(spec=[])
        model.config = FakeConfig("unknown")
        model.vision_tower = MagicMock()
        model.multi_modal_projector = MagicMock()
        enc = LLaVAEncoder()
        assert enc.supports_model(model)

    def test_default_pooling(self):
        enc = LLaVAEncoder(pooling_strategy="default")
        features = mx.ones((1, 5, 64))
        result = enc._apply_pooling(features)
        assert mx.array_equal(result, features)

    def test_spatial_avg_pooling(self):
        enc = LLaVAEncoder(pooling_strategy="spatial_avg")
        features = mx.ones((1, 5, 64))
        result = enc._apply_pooling(features)
        assert result.shape[1] == 1  # Averaged over spatial dim
        assert result.shape[0] == 1
        assert result.shape[2] == 64

    def test_attention_pooling(self):
        enc = LLaVAEncoder(pooling_strategy="attention_pool")
        features = mx.ones((1, 5, 64))
        result = enc._apply_pooling(features)
        assert result.shape[1] == 1  # Pooled to single token
        assert result.shape[2] == 64

    def test_pooling_2d_passthrough(self):
        enc = LLaVAEncoder(pooling_strategy="spatial_avg")
        features = mx.ones((5, 64))  # 2D input
        result = enc._apply_pooling(features)
        assert result.shape == (5, 64)  # Unchanged

    def test_encode_image_fallback(self):
        model = MagicMock(spec=[])
        model.config = FakeConfig(hidden_size=64)
        enc = LLaVAEncoder()
        processor = MagicMock()
        processor._prepare_image_inputs.side_effect = Exception("no image")
        result = enc.encode_image("test.jpg", model, processor=processor)
        assert isinstance(result, mx.array)

    def test_encode_images_batch(self):
        model = MagicMock(spec=[])
        model.config = FakeConfig(hidden_size=64)
        enc = LLaVAEncoder()
        processor = MagicMock()
        processor._prepare_image_inputs.side_effect = Exception("no image")
        results = enc.encode_images(
            ["img1.jpg", "img2.jpg", "img3.jpg"], model, processor
        )
        assert len(results) == 3

    def test_encode_images_empty(self):
        enc = LLaVAEncoder()
        assert enc.encode_images([], None) == []

    def test_anyres_config(self):
        enc = LLaVAEncoder(anyres=True, max_crops=4)
        assert enc._anyres is True
        assert enc._max_crops == 4

    def test_anyres_disabled(self):
        enc = LLaVAEncoder(anyres=False)
        assert enc._anyres is False

    def test_encode_count_increments(self):
        enc = LLaVAEncoder()
        model = MagicMock(spec=[])
        model.config = FakeConfig(hidden_size=64)
        processor = MagicMock()
        processor._prepare_image_inputs.side_effect = Exception("no image")
        assert enc._encode_count == 0
        enc.encode_image("test.jpg", model, processor=processor)
        assert enc._encode_count == 1


# ── CustomEncoder ──


class TestCustomEncoder:
    def test_basic_custom_encoder(self):
        def my_encoder(image, model, processor):
            return mx.zeros((1, 1, 32))

        enc = CustomEncoder(my_encoder)
        result = enc.encode_image("test.jpg", None)
        assert result.shape == (1, 1, 32)

    def test_custom_encoder_with_model(self):
        def my_encoder(image, model, processor):
            return mx.ones((1, 1, model.config.hidden_size))

        model = MagicMock()
        model.config = FakeConfig(hidden_size=128)
        enc = CustomEncoder(my_encoder)
        result = enc.encode_image("test.jpg", model)
        assert result.shape == (1, 1, 128)

    def test_custom_supports_model_always_true(self):
        enc = CustomEncoder(lambda i, m, p: mx.zeros((1,)))
        assert enc.supports_model(None)  # Always True by default

    def test_custom_supports_model_with_checker(self):
        def checker(model):
            return getattr(model, "my_attr", False)

        enc = CustomEncoder(
            encoder_fn=lambda i, m, p: mx.zeros((1,)),
            model_checker=checker,
        )
        model1 = MagicMock()
        model1.my_attr = True
        model2 = MagicMock(spec=[])
        assert enc.supports_model(model1)
        assert not enc.supports_model(model2)


# ── VisionEncoderFactory ──


class TestVisionEncoderFactory:
    def setup_method(self):
        """Clear registry before each test."""
        VisionEncoderFactory._registry.clear()
        VisionEncoderFactory._custom_encoders.clear()

    def test_default_is_mlx_vlm(self):
        model = MagicMock(spec=[])
        enc = VisionEncoderFactory.create_encoder(model)
        assert isinstance(enc, MLXVLMEncoder)

    def test_explicit_mlx_vlm_strategy(self):
        model = MagicMock(spec=[])
        enc = VisionEncoderFactory.create_encoder(
            model, strategy=VisionEncodingStrategy.MLX_VLM
        )
        assert isinstance(enc, MLXVLMEncoder)

    def test_explicit_qwen_vl_strategy(self):
        model = MagicMock(spec=[])
        enc = VisionEncoderFactory.create_encoder(
            model, strategy=VisionEncodingStrategy.QWEN_VL
        )
        assert isinstance(enc, QwenVLEncoder)

    def test_explicit_llava_strategy(self):
        model = MagicMock(spec=[])
        enc = VisionEncoderFactory.create_encoder(
            model, strategy=VisionEncodingStrategy.LLAVA
        )
        assert isinstance(enc, LLaVAEncoder)

    def test_explicit_llava_with_pooling(self):
        model = MagicMock(spec=[])
        enc = VisionEncoderFactory.create_encoder(
            model,
            strategy=VisionEncodingStrategy.LLAVA,
            pooling_strategy="spatial_avg",
        )
        assert isinstance(enc, LLaVAEncoder)
        assert enc._pooling_strategy == "spatial_avg"

    def test_custom_strategy_without_registration(self):
        model = MagicMock(spec=[])
        enc = VisionEncoderFactory.create_encoder(
            model, strategy=VisionEncodingStrategy.CUSTOM
        )
        # Falls back to MLXVLMEncoder when no custom registered
        assert isinstance(enc, MLXVLMEncoder)

    def test_custom_strategy_with_registration(self):
        def my_enc(img, model, proc):
            return mx.zeros((1, 1, 16))

        VisionEncoderFactory.register_custom_encoder("test", my_enc)
        model = MagicMock(spec=[])
        enc = VisionEncoderFactory.create_encoder(
            model,
            strategy=VisionEncodingStrategy.CUSTOM,
            custom_name="test",
        )
        assert isinstance(enc, CustomEncoder)

    def test_register_encoder(self):
        VisionEncoderFactory.register_encoder("my_model", QwenVLEncoder)
        model = _make_model_with_module("test", "my_model")
        enc = VisionEncoderFactory.create_encoder(model)
        assert isinstance(enc, QwenVLEncoder)

    def test_registry_overrides_auto_detect(self):
        """Registry match takes priority over auto-detection."""
        VisionEncoderFactory.register_encoder("qwen2_vl", LLaVAEncoder)
        model = _make_model_with_module("mlx_vlm.models.qwen2_vl", "qwen2_vl")
        enc = VisionEncoderFactory.create_encoder(model)
        assert isinstance(enc, LLaVAEncoder)

    def test_auto_detect_qwen(self):
        model = _make_model_with_module("mlx_vlm.models.qwen2_vl", "qwen2_vl")
        enc = VisionEncoderFactory.create_encoder(model)
        assert isinstance(enc, QwenVLEncoder)

    def test_auto_detect_llava(self):
        model = _make_model_with_module("mlx_vlm.models.llava", "llava")
        enc = VisionEncoderFactory.create_encoder(model)
        assert isinstance(enc, LLaVAEncoder)

    def test_auto_detect_unknown_falls_back(self):
        model = _make_model_with_module("some.random.module", "unknown_model")
        enc = VisionEncoderFactory.create_encoder(model)
        assert isinstance(enc, MLXVLMEncoder)

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_VISION_ENCODER", "llava")
        model = MagicMock(spec=[])
        enc = VisionEncoderFactory.create_encoder(model)
        assert isinstance(enc, LLaVAEncoder)

    def test_env_var_qwen(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_VISION_ENCODER", "qwen_vl")
        model = MagicMock(spec=[])
        enc = VisionEncoderFactory.create_encoder(model)
        assert isinstance(enc, QwenVLEncoder)

    def test_env_var_unknown_falls_back(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_VISION_ENCODER", "nonexistent")
        model = MagicMock(spec=[])
        # Should warn and fall back to auto-detect -> MLXVLMEncoder
        enc = VisionEncoderFactory.create_encoder(model)
        assert isinstance(enc, MLXVLMEncoder)

    def test_explicit_strategy_overrides_env_var(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_VISION_ENCODER", "llava")
        model = MagicMock(spec=[])
        enc = VisionEncoderFactory.create_encoder(
            model, strategy=VisionEncodingStrategy.QWEN_VL
        )
        # Explicit strategy should override env var
        assert isinstance(enc, QwenVLEncoder)


# ── Integration helpers ──


class TestIntegrationHelpers:
    def setup_method(self):
        VisionEncoderFactory._registry.clear()
        VisionEncoderFactory._custom_encoders.clear()

    def test_create_vision_encoder_for_model(self):
        model = _make_model_with_module("mlx_vlm.models.qwen2_vl", "qwen2_vl")
        enc = create_vision_encoder_for_model(model)
        assert isinstance(enc, QwenVLEncoder)

    def test_create_vision_encoder_default(self):
        model = MagicMock(spec=[])
        enc = create_vision_encoder_for_model(model)
        assert isinstance(enc, MLXVLMEncoder)

    def test_get_vision_encoding_strategy_auto_qwen(self):
        model = _make_model_with_module("mlx_vlm.models.qwen2_vl", "qwen2_vl")
        strategy = get_vision_encoding_strategy(model)
        assert strategy == VisionEncodingStrategy.QWEN_VL

    def test_get_vision_encoding_strategy_auto_llava(self):
        model = _make_model_with_module("mlx_vlm.models.llava", "llava")
        strategy = get_vision_encoding_strategy(model)
        assert strategy == VisionEncodingStrategy.LLAVA

    def test_get_vision_encoding_strategy_default(self):
        model = MagicMock(spec=[])
        strategy = get_vision_encoding_strategy(model)
        assert strategy == VisionEncodingStrategy.MLX_VLM

    def test_get_vision_encoding_strategy_env_override(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_VISION_ENCODER", "llava")
        model = MagicMock(spec=[])
        strategy = get_vision_encoding_strategy(model)
        assert strategy == VisionEncodingStrategy.LLAVA


# ── Edge cases ──


class TestEdgeCases:
    def setup_method(self):
        VisionEncoderFactory._registry.clear()
        VisionEncoderFactory._custom_encoders.clear()

    def test_model_with_no_config(self):
        model = MagicMock(spec=[])
        enc = QwenVLEncoder()
        # No config attribute -> should not support
        assert not enc.supports_model(model)

    def test_model_with_config_no_model_type(self):
        model = MagicMock(spec=[])
        config = MagicMock(spec=[])
        model.config = config
        enc = QwenVLEncoder()
        # Config exists but has no real model_type (MagicMock spec=[] has nothing)
        assert not enc.supports_model(model)

    def test_factory_with_none_model(self):
        """Factory should handle None model gracefully."""
        enc = VisionEncoderFactory.create_encoder(None)
        assert isinstance(enc, MLXVLMEncoder)

    def test_register_multiple_custom_encoders(self):
        VisionEncoderFactory.register_custom_encoder(
            "enc_a", lambda i, m, p: mx.zeros((1, 1, 10))
        )
        VisionEncoderFactory.register_custom_encoder(
            "enc_b", lambda i, m, p: mx.zeros((1, 1, 20))
        )
        assert "enc_a" in VisionEncoderFactory._custom_encoders
        assert "enc_b" in VisionEncoderFactory._custom_encoders

    def test_overwrite_registry(self):
        VisionEncoderFactory.register_encoder("test_type", QwenVLEncoder)
        VisionEncoderFactory.register_encoder("test_type", LLaVAEncoder)
        assert VisionEncoderFactory._registry["test_type"] == LLaVAEncoder
