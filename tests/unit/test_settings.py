"""Unit tests for settings module."""
import json
import os
import tempfile
from pathlib import Path

import pytest

from yunshu_engine.settings import (
    YunshuSettings,
    ServerSettings,
    ModelSettings,
    CacheSettings,
    EngineSettings,
    init_settings,
    get_settings,
    get_system_memory,
    get_ssd_capacity,
)


@pytest.fixture(autouse=True)
def _reset_settings():
    import yunshu_engine.settings as mod
    old = mod._settings
    mod._settings = None
    yield
    mod._settings = old


class TestServerSettings:
    def test_defaults(self):
        s = ServerSettings()
        assert s.host == "127.0.0.1"
        assert s.port == 8000

    def test_from_dict(self):
        s = ServerSettings.from_dict({"host": "0.0.0.0", "port": 9000})
        assert s.host == "0.0.0.0"
        assert s.port == 9000

    def test_from_dict_ignores_unknown(self):
        s = ServerSettings.from_dict({"host": "0.0.0.0", "unknown": 123})
        assert s.host == "0.0.0.0"

    def test_to_dict_roundtrip(self):
        s = ServerSettings(host="0.0.0.0", port=9999)
        d = s.to_dict()
        s2 = ServerSettings.from_dict(d)
        assert s2.host == "0.0.0.0"
        assert s2.port == 9999


class TestYunshuSettings:
    def test_defaults(self):
        s = YunshuSettings()
        assert s.version == "1.0"
        assert s.server.port == 8000
        assert s.engine.max_num_seqs == 256

    def test_from_dict(self):
        s = YunshuSettings.from_dict({
            "server": {"port": 9000},
            "engine": {"max_num_seqs": 128},
        })
        assert s.server.port == 9000
        assert s.engine.max_num_seqs == 128

    def test_to_dict(self):
        s = YunshuSettings()
        d = s.to_dict()
        assert "server" in d
        assert "models" in d
        assert "cache" in d
        assert "engine" in d

    def test_roundtrip(self):
        s = YunshuSettings()
        s.server.port = 7777
        s.cache.paged_cache_block_size = 128
        d = s.to_dict()
        s2 = YunshuSettings.from_dict(d)
        assert s2.server.port == 7777
        assert s2.cache.paged_cache_block_size == 128


class TestInitSettings:
    def test_init_creates_settings(self, tmp_path):
        s = init_settings(base_path=str(tmp_path))
        assert s is not None
        assert s.base_path == str(tmp_path)

    def test_settings_json_persisted(self, tmp_path):
        init_settings(base_path=str(tmp_path))
        p = tmp_path / "settings.json"
        assert p.exists()
        with open(p) as f:
            d = json.load(f)
        assert "server" in d

    def test_settings_json_loaded(self, tmp_path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"server": {"port": 9999}}))
        s = init_settings(base_path=str(tmp_path))
        assert s.server.port == 9999

    def test_env_override(self, tmp_path):
        os.environ["YUNSHU_PORT"] = "5555"
        try:
            s = init_settings(base_path=str(tmp_path))
            assert s.server.port == 5555
        finally:
            del os.environ["YUNSHU_PORT"]

    def test_cli_overrides(self, tmp_path):
        s = init_settings(
            base_path=str(tmp_path),
            cli_overrides={"server.port": 3333},
        )
        assert s.server.port == 3333

    def test_get_settings_returns_singleton(self, tmp_path):
        s1 = init_settings(base_path=str(tmp_path))
        s2 = get_settings()
        assert s1 is s2


class TestSystemUtils:
    def test_get_system_memory_positive(self):
        mem = get_system_memory()
        assert mem > 0

    def test_get_ssd_capacity(self):
        cap = get_ssd_capacity("/tmp")
        assert cap > 0


class TestModelSettings:
    def test_resolved_model_dirs_default(self, tmp_path):
        ms = ModelSettings()
        dirs = ms.resolved_model_dirs(tmp_path)
        assert dirs == [tmp_path / "models"]

    def test_resolved_model_dirs_custom(self):
        ms = ModelSettings(model_dirs=["/data/models", "/opt/models"])
        dirs = ms.resolved_model_dirs(Path("/base"))
        assert dirs == [Path("/data/models"), Path("/opt/models")]
