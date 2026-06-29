"""Tests for ModelSettings adaptive defaults integration."""

from yunshu_engine.model_settings import load_model_settings


class TestLoadModelSettingsAdaptive:
    def test_adaptive_applied_by_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("YUNSHU_ADAPTIVE_DEFAULTS", raising=False)
        settings = load_model_settings(str(tmp_path), "test-model", use_adaptive=True)
        # Should have adaptive overrides applied (batch_size, max_kv_cache_memory, etc.)
        assert settings.max_kv_cache_memory > 0

    def test_adaptive_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("YUNSHU_ADAPTIVE_DEFAULTS", "0")
        settings = load_model_settings(str(tmp_path), "test-model", use_adaptive=True)
        # Should use raw defaults (max_kv_cache_memory=0)
        assert settings.max_kv_cache_memory == 0

    def test_adaptive_disabled_by_false(self, tmp_path, monkeypatch):
        monkeypatch.setenv("YUNSHU_ADAPTIVE_DEFAULTS", "false")
        settings = load_model_settings(str(tmp_path), "test-model", use_adaptive=True)
        assert settings.max_kv_cache_memory == 0

    def test_env_vars_override_adaptive(self, tmp_path, monkeypatch):
        monkeypatch.delenv("YUNSHU_ADAPTIVE_DEFAULTS", raising=False)
        monkeypatch.setenv("YUNSHU_MODEL_TEST_MODEL_BATCH_SIZE", "42")
        settings = load_model_settings(str(tmp_path), "test-model", use_adaptive=True)
        assert settings.batch_size == 42

    def test_json_overrides_adaptive(self, tmp_path, monkeypatch):
        import json

        monkeypatch.delenv("YUNSHU_ADAPTIVE_DEFAULTS", raising=False)
        (tmp_path / "model_settings.json").write_text(json.dumps({"batch_size": 99}))
        settings = load_model_settings(str(tmp_path), "test-model", use_adaptive=True)
        assert settings.batch_size == 99

    def test_use_adaptive_false(self, tmp_path):
        settings = load_model_settings(str(tmp_path), "test-model", use_adaptive=False)
        assert settings.max_kv_cache_memory == 0
