"""Tests for yunshu_engine.vlm_engine — VLM Engine."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yunshu_engine.vlm_engine import VLMEngine, _is_mlx_vlm_model


class TestVLMEngineInit:
    def test_default_state(self):
        engine = VLMEngine("/models/test-model")
        assert engine.model_name == "test-model"
        assert engine.is_loaded is False
        assert engine.is_running is False
        assert engine.has_active_requests() is False
        assert engine.has_vision is False

    def test_model_name_nested_path(self):
        engine = VLMEngine("/home/user/models/qwen3-vl")
        assert engine.model_name == "qwen3-vl"

    def test_model_name_no_slash(self):
        engine = VLMEngine("local-model")
        assert engine.model_name == "local-model"


class TestExtractText:
    def test_string_content(self):
        assert VLMEngine._extract_text("hello") == "hello"

    def test_list_with_text_dicts(self):
        content = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        assert VLMEngine._extract_text(content) == "hello world"

    def test_list_with_strings(self):
        assert VLMEngine._extract_text(["a", "b"]) == "a b"

    def test_list_with_image_url_ignored(self):
        content = [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "http://img"}},
        ]
        assert VLMEngine._extract_text(content) == "describe"

    def test_empty_list(self):
        assert VLMEngine._extract_text([]) == ""

    def test_non_string_non_list(self):
        assert VLMEngine._extract_text(42) == "42"

    def test_none(self):
        assert VLMEngine._extract_text(None) == "None"


class TestFormatPrompt:
    def test_fallback_without_tokenizer(self):
        engine = VLMEngine("/models/test")
        engine._tokenizer = None
        result = engine._format_prompt([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": [{"type": "text", "text": "How are you?"}]},
        ])
        assert "User: Hello" in result
        assert "Assistant: Hi" in result
        assert "User: How are you?" in result
        assert result.endswith("Assistant:")

    def test_with_chat_template(self):
        engine = VLMEngine("/models/test")
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>user\nHi<|im_end|>"
        engine._tokenizer = mock_tokenizer
        result = engine._format_prompt([{"role": "user", "content": "Hi"}])
        assert "<|im_start|>" in result
        mock_tokenizer.apply_chat_template.assert_called_once()

    def test_chat_template_exception_falls_back(self):
        engine = VLMEngine("/models/test")
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = RuntimeError("no template")
        engine._tokenizer = mock_tokenizer
        result = engine._format_prompt([{"role": "user", "content": "Hi"}])
        assert "User: Hi" in result


class TestExtractImages:
    @pytest.mark.asyncio
    async def test_no_images(self):
        engine = VLMEngine("/models/test")
        messages = [{"role": "user", "content": "Hello"}]
        result = await engine._extract_images(messages)
        assert result == []

    @pytest.mark.asyncio
    async def test_text_only_content_list(self):
        engine = VLMEngine("/models/test")
        messages = [{"role": "user", "content": [{"type": "text", "text": "describe"}]}]
        result = await engine._extract_images(messages)
        assert result == []

    @pytest.mark.asyncio
    async def test_base64_image(self):
        engine = VLMEngine("/models/test")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
            ],
        }]
        result = await engine._extract_images(messages)
        assert len(result) == 1
        assert os.path.exists(result[0])
        engine._cleanup_temp_files()
        assert not os.path.exists(result[0])

    @pytest.mark.asyncio
    async def test_nonexistent_url_skipped(self):
        engine = VLMEngine("/models/test")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "/nonexistent/path.png"}},
            ],
        }]
        result = await engine._extract_images(messages)
        assert result == []

    @pytest.mark.asyncio
    async def test_file_path_image(self, tmp_path):
        engine = VLMEngine("/models/test")
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": str(img)}},
            ],
        }]
        result = await engine._extract_images(messages)
        assert len(result) == 1
        assert result[0] == str(img)


class TestCleanupTempFiles:
    def test_cleanup(self):
        engine = VLMEngine("/models/test")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"test")
            tmp_path = f.name
        engine._temp_files = [tmp_path]
        engine._cleanup_temp_files()
        assert not os.path.exists(tmp_path)
        assert engine._temp_files == []

    def test_cleanup_no_temp_files(self):
        engine = VLMEngine("/models/test")
        engine._cleanup_temp_files()  # should not raise

    def test_cleanup_none_temp_files(self):
        engine = VLMEngine("/models/test")
        engine._temp_files = None
        engine._cleanup_temp_files()  # should not raise


class TestGetStats:
    def test_initial_stats(self):
        engine = VLMEngine("/models/test-model")
        stats = engine.get_stats()
        assert stats["model"] == "/models/test-model"
        assert stats["loaded"] is False
        assert stats["running"] is False
        assert stats["has_vision"] is False
        assert stats["is_vlm"] is False
        assert stats["num_requests_processed"] == 0
        assert "uptime_seconds" in stats

    def test_stats_after_requests(self):
        engine = VLMEngine("/models/test")
        engine._num_requests_processed = 5
        engine._start_time = 100.0
        stats = engine.get_stats()
        assert stats["num_requests_processed"] == 5


class TestIsMlxVlmModel:
    def test_mlx_vlm_model(self):
        model = MagicMock()
        type(model).__module__ = "mlx_vlm.models.qwen3_vl"
        assert _is_mlx_vlm_model(model) is True

    def test_mlx_lm_model(self):
        model = MagicMock()
        type(model).__module__ = "mlx_lm.models.qwen3"
        assert _is_mlx_vlm_model(model) is False

    def test_custom_model(self):
        model = MagicMock()
        type(model).__module__ = "custom.models.my_model"
        assert _is_mlx_vlm_model(model) is False
