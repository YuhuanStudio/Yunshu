"""Tests for per-model settings (ModelSettings)."""
import json
import os
import tempfile

from yunshu_engine.model_settings import ModelSettings, load_model_settings


class TestModelSettings:
    def test_defaults(self):
        s = ModelSettings()
        assert s.max_tokens == 4096
        assert s.temperature == 0.7
        assert s.repetition_penalty == 1.0
        assert s.spec_decode_enabled is False
        assert s.prefix_cache_enabled is True
        assert s.kv_cache_quant_bits is None

    def test_to_dict(self):
        s = ModelSettings(max_tokens=2048, temperature=0.5)
        d = s.to_dict()
        assert d["max_tokens"] == 2048
        assert d["temperature"] == 0.5
        # Empty lists should not appear
        assert "stop" not in d

    def test_apply_overrides(self):
        s = ModelSettings()
        changed = s.apply_overrides({
            "max_tokens": 8192,
            "temperature": 0.3,
            "prefix_cache_enabled": False,
        })
        assert "max_tokens" in changed
        assert "temperature" in changed
        assert "prefix_cache_enabled" in changed
        assert s.max_tokens == 8192
        assert s.temperature == 0.3
        assert s.prefix_cache_enabled is False

    def test_apply_overrides_type_coercion(self):
        s = ModelSettings()
        s.apply_overrides({"max_tokens": 4096.0})  # float → int
        assert s.max_tokens == 4096
        assert isinstance(s.max_tokens, int)

    def test_apply_overrides_ignores_unknown(self):
        s = ModelSettings()
        changed = s.apply_overrides({"nonexistent_field": 42})
        assert changed == []
        assert not hasattr(s, "nonexistent_field")

    def test_apply_overrides_ignores_none(self):
        s = ModelSettings()
        changed = s.apply_overrides({"max_tokens": None})
        assert changed == []

    def test_load_from_model_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = os.path.join(tmpdir, "model_settings.json")
            with open(settings_path, "w") as f:
                json.dump({"max_tokens": 8192, "temperature": 0.1}, f)

            s = load_model_settings(tmpdir, "test-model")
            assert s.max_tokens == 8192
            assert s.temperature == 0.1

    def test_load_defaults_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = load_model_settings(tmpdir, "test-model")
            assert s.max_tokens == 4096  # default

    def test_load_invalid_json_uses_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = os.path.join(tmpdir, "model_settings.json")
            with open(settings_path, "w") as f:
                f.write("invalid json{")

            s = load_model_settings(tmpdir, "test-model")
            assert s.max_tokens == 4096  # still default

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_MODEL_MY_MODEL_MAX_TOKENS", "16384")
        monkeypatch.setenv("YUNSHU_MODEL_MY_MODEL_TEMPERATURE", "0.9")
        monkeypatch.setenv("YUNSHU_MODEL_MY_MODEL_SPEC_DECODE_ENABLED", "true")

        s = load_model_settings("/tmp/nonexistent", "my-model")
        assert s.max_tokens == 16384
        assert s.temperature == 0.9
        assert s.spec_decode_enabled is True

    def test_env_var_bool_false(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_MODEL_TEST_PREFIX_CACHE_ENABLED", "false")
        s = load_model_settings("/tmp/nonexistent", "test")
        assert s.prefix_cache_enabled is False

    def test_all_fields_have_defaults(self):
        """Ensure all fields have usable defaults."""
        s = ModelSettings()
        for name in s.__dataclass_fields__:
            val = getattr(s, name)
            assert val is not None or name in (
                "kv_cache_quant_bits", "thinking_budget", "enable_thinking", "seed"
            ), f"{name} should have a non-None default"
