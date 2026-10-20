"""get_ane_processor() must follow the live YUNSHU_ANE_EMBEDDING_MODEL setting."""

from __future__ import annotations

from yunshu_engine import ane_embedding, settings


class _FakeProc:
    def __init__(self, config):
        self._config = config

    def is_compiled(self):
        return True


def test_setting_change_rebuilds_the_processor(monkeypatch):
    monkeypatch.setattr(ane_embedding, "is_ane_available", lambda: True)
    monkeypatch.setattr(ane_embedding, "ANEEmbeddingProcessor", _FakeProc)
    monkeypatch.setattr(ane_embedding, "_ane_processor", None)
    try:
        settings.set_override("YUNSHU_ANE_EMBEDDING_MODEL", "model-a")
        a = ane_embedding.get_ane_processor()
        assert a._config.model_name == "model-a"
        assert ane_embedding.get_ane_processor() is a  # unchanged: reused
        settings.set_override("YUNSHU_ANE_EMBEDDING_MODEL", "model-b")
        b = ane_embedding.get_ane_processor()
        assert b is not a and b._config.model_name == "model-b"
    finally:
        settings.clear_overrides()
