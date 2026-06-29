"""Tests for per-model settings integration with BatchedEngine."""

import os


class TestModelSettingsLoadApply:
    def test_load_model_settings_defaults(self, tmp_path):
        from yunshu_engine.model_settings import load_model_settings

        settings = load_model_settings(str(tmp_path), "test-model", use_adaptive=False)
        assert settings.max_tokens == 4096
        assert settings.temperature == 0.7
        assert settings.kv_cache_quant_bits is None

    def test_load_from_json(self, tmp_path):
        import json

        from yunshu_engine.model_settings import load_model_settings

        settings_file = tmp_path / "model_settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "max_tokens": 8192,
                    "temperature": 0.5,
                    "kv_cache_quant_bits": 4,
                }
            )
        )
        settings = load_model_settings(str(tmp_path), "test-model")
        assert settings.max_tokens == 8192
        assert settings.temperature == 0.5
        assert settings.kv_cache_quant_bits == 4

    def test_env_overrides_json(self, tmp_path):
        import json

        from yunshu_engine.model_settings import load_model_settings

        settings_file = tmp_path / "model_settings.json"
        settings_file.write_text(json.dumps({"max_tokens": 8192}))
        os.environ["YUNSHU_MODEL_TEST_MODEL_MAX_TOKENS"] = "16384"
        try:
            settings = load_model_settings(str(tmp_path), "test-model")
            assert settings.max_tokens == 16384
        finally:
            del os.environ["YUNSHU_MODEL_TEST_MODEL_MAX_TOKENS"]

    def test_apply_overrides(self):
        from yunshu_engine.model_settings import ModelSettings

        settings = ModelSettings()
        changed = settings.apply_overrides(
            {
                "max_tokens": 2048,
                "temperature": 0.1,
            }
        )
        assert "max_tokens" in changed
        assert "temperature" in changed
        assert settings.max_tokens == 2048
        assert settings.temperature == 0.1

    def test_apply_overrides_ignores_unknown(self):
        from yunshu_engine.model_settings import ModelSettings

        settings = ModelSettings()
        changed = settings.apply_overrides({"nonexistent_field": 42})
        assert len(changed) == 0

    def test_to_dict(self):
        from yunshu_engine.model_settings import ModelSettings

        settings = ModelSettings(max_tokens=2048, temperature=0.5)
        d = settings.to_dict()
        assert d["max_tokens"] == 2048
        assert d["temperature"] == 0.5


class TestBatchedEngineSettingsAccess:
    def test_settings_none_before_load(self):
        from yunshu_engine.batched_engine import BatchedEngine

        engine = BatchedEngine(model_name="test")
        assert engine.get_settings() is None
