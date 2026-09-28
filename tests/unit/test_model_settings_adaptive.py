"""Tests for ModelSettings adaptive defaults integration."""

from yunshu_engine.model_settings import load_model_settings


class TestLoadModelSettingsAdaptive:
    def test_adaptive_applied_by_default(self, tmp_path):
        settings = load_model_settings(str(tmp_path), "test-model", use_adaptive=True)
        # Should have adaptive overrides applied (batch_size, max_kv_cache_memory, etc.)
        assert settings.max_kv_cache_memory > 0

    def test_json_overrides_adaptive(self, tmp_path):
        import json

        (tmp_path / "model_settings.json").write_text(json.dumps({"batch_size": 99}))
        settings = load_model_settings(str(tmp_path), "test-model", use_adaptive=True)
        assert settings.batch_size == 99

    def test_use_adaptive_false(self, tmp_path):
        settings = load_model_settings(str(tmp_path), "test-model", use_adaptive=False)
        assert settings.max_kv_cache_memory == 0
