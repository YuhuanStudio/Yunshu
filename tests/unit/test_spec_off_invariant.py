"""Spec off on a spec family must run the spec lane's plain-decode arithmetic."""

from types import SimpleNamespace

import pytest

from yunshu_engine import spec_select
from yunshu_engine.vlm_engine import VLMEngine


class _StopError(Exception):
    pass


def _build(monkeypatch, model_type):
    calls = []
    monkeypatch.setattr(
        "yunshu_engine.utils.hardware.is_paravirtual_metal", lambda: False
    )
    monkeypatch.setattr(VLMEngine, "_round_driver_wanted", lambda *_: False)
    monkeypatch.setattr(
        spec_select,
        "choose",
        lambda *a, **k: spec_select.SpecChoice("none", None, "off"),
    )
    import yunshu_engine.kernels.batch_invariant as bi
    import yunshu_engine.kernels.omlx as omlx
    import yunshu_engine.kernels.tensorfold.lane_qmm as qmm
    from yunshu_engine.kernels import buffer_cache

    monkeypatch.setattr(
        omlx, "apply", lambda row_exact=False: calls.append("omlx") or {}
    )
    monkeypatch.setattr(bi, "install", lambda *a, **k: calls.append("install") or True)
    monkeypatch.setattr(bi, "set_active", lambda v: calls.append(("active", v)))
    monkeypatch.setattr(qmm, "ready", lambda: False)

    def stop(*_a, **_k):
        raise _StopError

    monkeypatch.setattr(buffer_cache, "install", stop)
    monkeypatch.setattr(buffer_cache, "auto_limit_gib", lambda *_: 0)
    eng = object.__new__(VLMEngine)
    eng._config = {"model_type": model_type}
    eng._model = SimpleNamespace(language_model=SimpleNamespace())
    eng._apc_backend = None
    with pytest.raises(_StopError):
        eng._build_batch_runner("/nonexistent")
    return eng, calls


def test_spec_off_spec_family_installs_invariant(monkeypatch):
    eng, calls = _build(monkeypatch, "qwen3_5")
    assert "install" in calls and "omlx" in calls
    assert ("active", False) in calls
    assert eng._prefix_invariant_dispatch is True
    assert eng._apc_prefill_stride > 0


def test_non_spec_family_untouched(monkeypatch):
    eng, calls = _build(monkeypatch, "gemma4")
    assert calls == []
    assert eng._prefix_invariant_dispatch is False
    assert eng._apc_prefill_stride == 0
