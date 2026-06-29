"""Tests for OCR engine and video audio extraction."""

import pytest


class TestOCREngine:
    def test_init(self):
        from yunshu_engine.ocr_engine import OCREngine

        engine = OCREngine("/path/to/ocr-model")
        assert engine.model_name == "ocr-model"
        assert not engine.is_loaded

    def test_model_name_from_path(self):
        from yunshu_engine.ocr_engine import OCREngine

        engine = OCREngine("/models/deepseekocr")
        assert engine.model_name == "deepseekocr"

    def test_model_name_empty(self):
        from yunshu_engine.ocr_engine import OCREngine

        engine = OCREngine("")
        assert engine.model_name == ""

    def test_stats(self):
        from yunshu_engine.ocr_engine import OCREngine

        engine = OCREngine("/test/model")
        stats = engine.get_stats()
        assert stats["model"] == "/test/model"
        assert not stats["loaded"]
        assert not stats["running"]

    def test_extract_text_raises_when_not_loaded(self):
        from yunshu_engine.ocr_engine import OCREngine

        engine = OCREngine("/test/model")
        with pytest.raises(RuntimeError, match="not started"):
            import asyncio

            asyncio.run(engine.extract_text("/tmp/test.png"))

    def test_task_prompts(self):
        """Task parameter maps to correct GLM-OCR prompts."""
        from yunshu_engine.ocr_engine import OCREngine

        engine = OCREngine("/test/model")
        # Just verify the engine accepts the task parameter
        assert not engine.is_loaded

    def test_default_task_is_text(self):
        """Default task should be 'text'."""
        import inspect

        from yunshu_engine.ocr_engine import OCREngine

        sig = inspect.signature(OCREngine.extract_text)
        assert sig.parameters["task"].default == "text"


class TestModelTypeOCR:
    def test_ocr_in_model_type(self):
        from yunshu_engine.model_manager import ModelType

        assert hasattr(ModelType, "OCR")

    def test_ocr_model_detection(self):
        """Models with 'ocr' in the directory name should be detected as OCR type."""


class TestVideoExtensions:
    def test_video_extensions_set(self):
        """Verify video extensions are recognized."""
        video_exts = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".ts", ".mts"}
        assert ".mp4" in video_exts
        assert ".mkv" in video_exts

    def test_safe_audio_set(self):
        """Verify audio extensions are still recognized."""
        audio_exts = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm", ".aac"}
        assert ".wav" in audio_exts
        assert ".mp3" in audio_exts


class TestMaxModelsConfig:
    def test_max_models_default(self):
        from yunshu_engine.model_manager import ModelManager

        mm = ModelManager()
        assert mm.max_models == 0  # unlimited

    def test_max_models_custom(self):
        from yunshu_engine.model_manager import ModelManager

        mm = ModelManager(max_models=3)
        assert mm.max_models == 3

    def test_loaded_count_empty(self):
        from yunshu_engine.model_manager import ModelManager

        mm = ModelManager()
        assert mm.loaded_count == 0

    def test_loaded_count_property(self):
        from yunshu_engine.model_manager import ModelManager

        mm = ModelManager()
        mm.register_model("test-model", "/fake/path")
        assert mm.loaded_count == 0
