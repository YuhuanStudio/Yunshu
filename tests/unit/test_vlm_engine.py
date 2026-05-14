"""Tests for yunshu_engine.vlm_engine — VLM Engine."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yunshu_engine.vlm_engine import VLMEngine, _is_mlx_vlm_model, _MlxVlmVisionCacheAdapter


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


class TestMlxVlmVisionCacheAdapter:
    """Tests for the adapter that bridges VisionFeatureCache to mlx_vlm's interface."""

    def test_get_returns_cached_features(self, tmp_path):
        """Adapter.get() should return features from the underlying cache."""
        from yunshu_engine.vision_feature_cache import VisionFeatureCache

        cache = VisionFeatureCache(cache_dir=None, max_memory_entries=5)
        adapter = _MlxVlmVisionCacheAdapter(cache, "test-model")

        # Create a test image file
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n fake image data")

        # Pre-populate the cache
        from yunshu_engine.vision_feature_cache import compute_image_hash
        with open(str(img), "rb") as f:
            img_hash = compute_image_hash(f.read())
        features = MagicMock()
        cache.put(img_hash, "test-model", features)

        # Adapter should find it
        result = adapter.get(str(img))
        assert result is features

    def test_get_returns_none_on_miss(self, tmp_path):
        """Adapter.get() should return None for unknown images."""
        from yunshu_engine.vision_feature_cache import VisionFeatureCache

        cache = VisionFeatureCache(cache_dir=None, max_memory_entries=5)
        adapter = _MlxVlmVisionCacheAdapter(cache, "test-model")

        img = tmp_path / "unknown.png"
        img.write_bytes(b"unknown image")
        assert adapter.get(str(img)) is None

    def test_put_stores_features(self, tmp_path):
        """Adapter.put() should store features in the underlying cache."""
        from yunshu_engine.vision_feature_cache import VisionFeatureCache

        cache = VisionFeatureCache(cache_dir=None, max_memory_entries=5)
        adapter = _MlxVlmVisionCacheAdapter(cache, "test-model")

        img = tmp_path / "store.png"
        img.write_bytes(b"image to store")

        features = MagicMock()
        adapter.put(str(img), features)

        # Verify it's retrievable via the adapter
        result = adapter.get(str(img))
        assert result is features

    def test_get_with_nonexistent_path_hashes_path_string(self):
        """Adapter.get() should handle non-file paths gracefully."""
        from yunshu_engine.vision_feature_cache import VisionFeatureCache

        cache = VisionFeatureCache(cache_dir=None, max_memory_entries=5)
        adapter = _MlxVlmVisionCacheAdapter(cache, "test-model")

        # Non-existent path should not crash
        result = adapter.get("/nonexistent/path.png")
        assert result is None

    def test_get_with_list_of_images(self, tmp_path):
        """Adapter.get() should handle multi-image (list) input."""
        from yunshu_engine.vision_feature_cache import VisionFeatureCache

        cache = VisionFeatureCache(cache_dir=None, max_memory_entries=5)
        adapter = _MlxVlmVisionCacheAdapter(cache, "test-model")

        img1 = tmp_path / "a.png"
        img2 = tmp_path / "b.png"
        img1.write_bytes(b"image a")
        img2.write_bytes(b"image b")

        # Store via list
        features = MagicMock()
        adapter.put([str(img1), str(img2)], features)

        # Retrieve via same list
        result = adapter.get([str(img1), str(img2)])
        assert result is features

    def test_exception_in_get_returns_none(self, tmp_path):
        """Adapter.get() should return None on unexpected errors."""
        from yunshu_engine.vision_feature_cache import VisionFeatureCache

        cache = VisionFeatureCache(cache_dir=None, max_memory_entries=5)
        adapter = _MlxVlmVisionCacheAdapter(cache, "test-model")

        # Even if the cache is broken, get should not raise
        result = adapter.get(None)
        assert result is None

    def test_different_models_isolated(self, tmp_path):
        """Adapters for different models should be isolated."""
        from yunshu_engine.vision_feature_cache import VisionFeatureCache

        cache = VisionFeatureCache(cache_dir=None, max_memory_entries=5)
        adapter_a = _MlxVlmVisionCacheAdapter(cache, "model-a")
        adapter_b = _MlxVlmVisionCacheAdapter(cache, "model-b")

        img = tmp_path / "shared.png"
        img.write_bytes(b"shared image")

        features_a = "features-a"
        adapter_a.put(str(img), features_a)

        # model-b should not see model-a's features
        assert adapter_b.get(str(img)) is None
        assert adapter_a.get(str(img)) == features_a


class TestVLMKVPrefixCache:
    """Tests for per-image KV prefix cache state management."""

    def test_get_kv_prefix_state_returns_none_for_new_image(self):
        engine = VLMEngine("/models/test")
        assert engine._get_kv_prefix_state("new_hash") is None
        assert engine._vlm_kv_prefix_misses == 1

    def test_get_kv_prefix_state_returns_existing(self):
        engine = VLMEngine("/models/test")
        # Create a state entry
        state = engine._ensure_kv_prefix_state("img_hash")
        # Now get should find it
        result = engine._get_kv_prefix_state("img_hash")
        assert result is state
        assert engine._vlm_kv_prefix_hits == 1

    def test_get_kv_prefix_state_none_hash_returns_none(self):
        engine = VLMEngine("/models/test")
        assert engine._get_kv_prefix_state(None) is None

    def test_ensure_kv_prefix_state_creates_entry(self):
        engine = VLMEngine("/models/test")
        state = engine._ensure_kv_prefix_state("hash1")
        assert "hash1" in engine._kv_prefix_states
        assert engine._kv_prefix_states["hash1"] is state

    def test_ensure_kv_prefix_state_evicts_when_full(self):
        engine = VLMEngine("/models/test")
        engine._kv_prefix_max_entries = 4

        # Fill up to max
        for i in range(4):
            engine._ensure_kv_prefix_state(f"hash_{i}")

        assert len(engine._kv_prefix_states) == 4

        # Add one more — should evict old entries
        engine._ensure_kv_prefix_state("hash_4")
        assert len(engine._kv_prefix_states) <= 4
        assert "hash_4" in engine._kv_prefix_states


class TestVLMCacheStats:
    """Tests for VLM cache statistics tracking."""

    def test_initial_stats_have_new_fields(self):
        engine = VLMEngine("/models/test")
        stats = engine.get_stats()
        assert "vlm_vision_feature_hits" in stats
        assert "vlm_vision_feature_misses" in stats
        assert "vlm_vision_feature_hit_rate" in stats
        assert "vlm_kv_prefix_entries" in stats
        assert "vlm_kv_prefix_hits" in stats
        assert "vlm_kv_prefix_misses" in stats
        assert "vlm_kv_prefix_hit_rate" in stats

    def test_stats_reflect_kv_prefix_activity(self):
        engine = VLMEngine("/models/test")

        # Miss
        engine._get_kv_prefix_state("hash_a")
        assert engine.get_stats()["vlm_kv_prefix_misses"] == 1
        assert engine.get_stats()["vlm_kv_prefix_hit_rate"] == 0.0

        # Create and hit
        engine._ensure_kv_prefix_state("hash_a")
        engine._get_kv_prefix_state("hash_a")
        assert engine.get_stats()["vlm_kv_prefix_hits"] == 1
        assert engine.get_stats()["vlm_kv_prefix_hit_rate"] == pytest.approx(0.5)

    def test_stats_reflect_vision_feature_activity(self):
        engine = VLMEngine("/models/test")
        engine._vlm_vision_hits = 3
        engine._vlm_vision_misses = 1
        stats = engine.get_stats()
        assert stats["vlm_vision_feature_hits"] == 3
        assert stats["vlm_vision_feature_misses"] == 1
        assert stats["vlm_vision_feature_hit_rate"] == pytest.approx(0.75)

    def test_vision_cache_enabled_in_stats(self):
        engine = VLMEngine("/models/test")
        # Vision cache is now enabled by default (unless explicitly disabled)
        stats = engine.get_stats()
        assert isinstance(stats["vision_cache_enabled"], bool)
