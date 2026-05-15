"""Tests for model_preprocessor.py — model-specific input preprocessors."""

import pytest

from yunshu_engine.model_preprocessor import (
    CosyVoicePhonemePreprocessor,
    DeepSeekOCRPreprocessor,
    GLMOCRPreprocessor,
    LLaVAImagePreprocessor,
    ModelPreprocessor,
    PreprocessedInput,
    PreprocessorRegistry,
    PreprocessorType,
    QwenOmniAudioPreprocessor,
    QwenVLImagePreprocessor,
    WanVideoPreprocessor,
    WhisperSpeechPreprocessor,
)


class TestQwenOmniAudioPreprocessor:
    def test_raw_bytes_input(self):
        p = QwenOmniAudioPreprocessor()
        result = p.preprocess(b"\x00" * 3200)
        assert result.input_type == PreprocessorType.AUDIO
        assert result.model_family == "qwen3_omni"
        # Placeholder returns empty token_ids (real codec not integrated)
        assert result.token_ids == []
        assert result.original_tokens == 10  # 3200 // 320
        assert len(result.warnings) > 0

    def test_token_ids_input(self):
        p = QwenOmniAudioPreprocessor()
        result = p.preprocess([1, 2, 3, 4, 5])
        assert result.token_ids == [1, 2, 3, 4, 5]
        assert result.processed_tokens == 5

    def test_dict_input(self):
        p = QwenOmniAudioPreprocessor()
        result = p.preprocess({"token_ids": [10, 20, 30]})
        assert result.token_ids == [10, 20, 30]

    def test_dict_audio_bytes(self):
        p = QwenOmniAudioPreprocessor()
        result = p.preprocess({"audio": b"\x00" * 640})
        # dict without token_ids key — returns empty
        assert result.token_ids == []

    def test_unsupported_input(self):
        p = QwenOmniAudioPreprocessor()
        result = p.preprocess(12345)
        assert len(result.warnings) == 1
        assert result.token_ids == []

    def test_detect_model(self):
        p = QwenOmniAudioPreprocessor()
        assert p.detect_model({"model_type": "qwen3_omni"})
        assert p.detect_model({"model_type": "qwen2_5_omni"})
        assert not p.detect_model({"model_type": "llama"})


class TestCosyVoicePhonemePreprocessor:
    def test_text_input(self):
        p = CosyVoicePhonemePreprocessor()
        result = p.preprocess("你好世界")
        assert result.input_type == PreprocessorType.SPEECH
        assert result.token_ids == []  # Placeholder: no real phoneme encoder
        assert result.features["phoneme_count"] > 0
        assert len(result.warnings) > 0

    def test_dict_input(self):
        p = CosyVoicePhonemePreprocessor()
        result = p.preprocess({"text": "hello"})
        assert result.model_family == "cosyvoice"

    def test_empty_text(self):
        p = CosyVoicePhonemePreprocessor()
        result = p.preprocess("")
        assert result.token_ids == []

    def test_detect_model(self):
        p = CosyVoicePhonemePreprocessor()
        assert p.detect_model({"model_type": "cosyvoice"})
        assert not p.detect_model({"model_type": "whisper"})


class TestLLaVAImagePreprocessor:
    def test_default_preprocessing(self):
        p = LLaVAImagePreprocessor()
        result = p.preprocess(None)
        assert result.input_type == PreprocessorType.IMAGE
        assert result.model_family == "llava"
        expected_patches = (336 // 14) ** 2
        assert result.token_ids == []  # Placeholder: no real vision encoder
        assert result.features["num_patches"] == expected_patches

    def test_custom_image_size(self):
        p = LLaVAImagePreprocessor(image_size=384, patch_size=16)
        result = p.preprocess(None)
        expected_patches = (384 // 16) ** 2
        assert result.token_ids == []  # Placeholder
        assert result.features["num_patches"] == expected_patches

    def test_features(self):
        p = LLaVAImagePreprocessor()
        result = p.preprocess(None)
        assert result.features["image_size"] == 336
        assert result.features["patch_size"] == 14
        assert result.features["num_patches"] > 0

    def test_detect_model(self):
        p = LLaVAImagePreprocessor()
        assert p.detect_model({"model_type": "llava"})
        assert p.detect_model({"model_type": "llava_next"})
        assert p.detect_model({"model_type": "llava_onevision"})
        assert not p.detect_model({"model_type": "qwen2_vl"})


class TestQwenVLImagePreprocessor:
    def test_default_resolution(self):
        p = QwenVLImagePreprocessor()
        result = p.preprocess(None)
        assert result.input_type == PreprocessorType.IMAGE
        assert result.model_family == "qwen_vl"

    def test_custom_resolution(self):
        p = QwenVLImagePreprocessor()
        result = p.preprocess(None, resolution=672)
        assert result.features["resolution"] == 672
        assert result.token_ids == []  # Placeholder: no real ViT encoder

    def test_special_vision_tokens(self):
        p = QwenVLImagePreprocessor()
        result = p.preprocess(None)
        assert result.token_ids == []  # Placeholder: no real ViT encoder
        assert result.features["num_patches"] > 0

    def test_detect_model(self):
        p = QwenVLImagePreprocessor()
        assert p.detect_model({"model_type": "qwen2_vl"})
        assert p.detect_model({"model_type": "qwen2_5_vl"})
        assert p.detect_model({"model_type": "qwen3_vl"})
        assert not p.detect_model({"model_type": "llava"})


class TestWanVideoPreprocessor:
    def test_default_params(self):
        p = WanVideoPreprocessor()
        result = p.preprocess(None)
        assert result.input_type == PreprocessorType.VIDEO
        assert result.model_family == "wan_video"
        assert result.features["num_frames"] == 16

    def test_custom_params(self):
        p = WanVideoPreprocessor(num_frames=32, frame_size=768)
        result = p.preprocess(None, num_frames=32, frame_size=768)
        assert result.features["num_frames"] == 32
        assert result.features["frame_size"] == 768

    def test_tokens_per_frame(self):
        p = WanVideoPreprocessor(num_frames=8, frame_size=256)
        result = p.preprocess(None, num_frames=8, frame_size=256)
        expected_per_frame = (256 // 16) ** 2
        assert result.features["tokens_per_frame"] == expected_per_frame

    def test_detect_model(self):
        p = WanVideoPreprocessor()
        assert p.detect_model({"model_type": "wan"})
        assert p.detect_model({"model_type": "wan_video"})
        assert not p.detect_model({"model_type": "llama"})


class TestGLMOCRPreprocessor:
    def test_basic_preprocessing(self):
        p = GLMOCRPreprocessor()
        result = p.preprocess(None)
        assert result.input_type == PreprocessorType.OCR
        assert result.model_family == "glm_ocr"
        assert result.features["layout_aware"] is True

    def test_max_patches(self):
        p = GLMOCRPreprocessor()
        result = p.preprocess(None, max_patches=512)
        assert result.token_ids == []  # Placeholder: no real OCR encoder
        assert result.features["max_patches"] == 512

    def test_detect_model(self):
        p = GLMOCRPreprocessor()
        assert p.detect_model({"model_type": "glm_ocr"})
        assert not p.detect_model({"model_type": "glm"})


class TestDeepSeekOCRPreprocessor:
    def test_basic_preprocessing(self):
        p = DeepSeekOCRPreprocessor()
        result = p.preprocess(None)
        assert result.input_type == PreprocessorType.OCR
        assert result.model_family == "deepseek_ocr"

    def test_detect_model(self):
        p = DeepSeekOCRPreprocessor()
        assert p.detect_model({"model_type": "deepseek_ocr"})
        assert not p.detect_model({"model_type": "deepseek"})


class TestWhisperSpeechPreprocessor:
    def test_raw_audio_bytes(self):
        p = WhisperSpeechPreprocessor()
        result = p.preprocess(b"\x00" * 32000)  # 16000 samples * 2 bytes
        assert result.input_type == PreprocessorType.SPEECH
        assert result.model_family == "whisper"
        assert result.features["mel_frames"] > 0

    def test_dict_input(self):
        p = WhisperSpeechPreprocessor()
        result = p.preprocess({"audio_length": 16000})
        assert result.features["mel_frames"] == 100  # 16000 / 160

    def test_max_duration_cap(self):
        p = WhisperSpeechPreprocessor(max_audio_seconds=30.0)
        result = p.preprocess({"audio_length": 1000000})
        max_frames = int(30.0 * 16000 / 160)
        assert result.features["mel_frames"] == max_frames

    def test_features(self):
        p = WhisperSpeechPreprocessor()
        result = p.preprocess(b"\x00" * 3200)
        assert result.features["sample_rate"] == 16000
        assert result.features["n_mels"] == 128

    def test_detect_model(self):
        p = WhisperSpeechPreprocessor()
        assert p.detect_model({"model_type": "whisper"})
        assert not p.detect_model({"model_type": "cosyvoice"})


class TestPreprocessorRegistry:
    def test_default_registration(self):
        reg = PreprocessorRegistry()
        stats = reg.get_stats()
        assert stats["registered"] >= 8

    def test_get_by_family(self):
        reg = PreprocessorRegistry()
        p = reg.get("qwen3_omni")
        assert p is not None
        assert isinstance(p, QwenOmniAudioPreprocessor)

    def test_get_nonexistent(self):
        reg = PreprocessorRegistry()
        assert reg.get("nonexistent_model") is None

    def test_detect_from_config(self):
        reg = PreprocessorRegistry()
        p = reg.detect({"model_type": "llava"})
        assert p is not None
        assert isinstance(p, LLaVAImagePreprocessor)

    def test_detect_no_match(self):
        reg = PreprocessorRegistry()
        assert reg.detect({"model_type": "unknown_model_xxx"}) is None

    def test_preprocess_by_family(self):
        reg = PreprocessorRegistry()
        result = reg.preprocess(b"\x00" * 320, model_family="qwen3_omni")
        assert result.input_type == PreprocessorType.AUDIO
        assert result.token_ids == []  # Placeholder: no real codec

    def test_preprocess_by_config(self):
        reg = PreprocessorRegistry()
        result = reg.preprocess(
            None,
            model_config={"model_type": "llava"},
        )
        assert result.input_type == PreprocessorType.IMAGE
        assert result.token_ids == []  # Placeholder: no real vision encoder

    def test_preprocess_unknown_fallback(self):
        reg = PreprocessorRegistry()
        result = reg.preprocess(None, model_family="unknown_xxx")
        assert result.model_family == "unknown"
        assert len(result.warnings) == 1

    def test_list_preprocessors(self):
        reg = PreprocessorRegistry()
        entries = reg.list_preprocessors()
        assert len(entries) >= 8
        families = [e["family"] for e in entries]
        assert "qwen3_omni" in families
        assert "llava" in families
        assert "whisper" in families

    def test_custom_preprocessor(self):
        class CustomPreprocessor(ModelPreprocessor):
            model_family = "custom_test"
            input_type = PreprocessorType.MULTIMODAL
            def preprocess(self, raw_input, **kwargs):
                return PreprocessedInput(
                    input_type=self.input_type,
                    model_family=self.model_family,
                    token_ids=[1, 2, 3],
                )

        reg = PreprocessorRegistry()
        reg.register(CustomPreprocessor())
        p = reg.get("custom_test")
        assert p is not None
        result = p.preprocess(None)
        assert result.token_ids == [1, 2, 3]

    def test_type_index(self):
        reg = PreprocessorRegistry()
        stats = reg.get_stats()
        assert "IMAGE" in stats["by_type"]
        assert "AUDIO" in stats["by_type"]
        assert stats["by_type"]["IMAGE"] >= 2  # LLaVA + QwenVL

    def test_stats(self):
        reg = PreprocessorRegistry()
        stats = reg.get_stats()
        assert stats["registered"] >= 8
        assert "by_type" in stats
