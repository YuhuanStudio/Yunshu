"""Native voice reuses the already-served model — one omni model, two endpoints,
no second copy in memory.

These cover the wiring (engine injection + reuse selection + realtime auto-enable),
not the GPU path; the end-to-end speech run needs a real Qwen3-Omni model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


# ── OmniEngine: adopt an already-loaded model instead of loading a 2nd copy ──
def test_omni_engine_adopts_injected_model_without_loading(monkeypatch):
    from yunshu_engine import omni_engine
    from yunshu_engine.omni_engine import OmniEngine

    # vlmload must NOT be called when a model is injected.
    def _boom(*a, **k):  # pragma: no cover - asserts it isn't reached
        raise AssertionError("vlmload should not run when a model is injected")

    monkeypatch.setattr("mlx_vlm.load", _boom, raising=False)

    fake_model = SimpleNamespace(has_talker=True)  # no real thinker → prime no-ops
    eng = OmniEngine(model=fake_model, processor=object())
    assert eng._shared is True
    assert eng.is_loaded() is True

    eng.load()  # runs setup on the injected model; _compile_kernels is best-effort
    assert eng._setup_done is True
    assert eng.model is fake_model
    assert isinstance(omni_engine, type(omni_engine))  # module import sanity


def test_omni_engine_requires_a_model_or_path():
    from yunshu_engine.omni_engine import OmniEngine

    with pytest.raises(ValueError, match="model_path or an injected model"):
        OmniEngine().load()


def test_omni_engine_rejects_a_model_without_a_talker():
    from yunshu_engine.omni_engine import OmniEngine

    eng = OmniEngine(model=SimpleNamespace(has_talker=False), processor=object())
    with pytest.raises(ValueError, match="no Talker"):
        eng.load()


# ── omni router: detect + reuse the served model ────────────────────────────
def test_shared_speakable_model_returns_served_talker_model(monkeypatch):
    from yunshu_gateway.routers import omni

    model = SimpleNamespace(has_talker=True)
    processor = object()
    monkeypatch.setattr(
        omni, "_shared_speakable_model", omni._shared_speakable_model
    )  # ensure real fn
    monkeypatch.setattr(
        "yunshu_gateway.engine.get_engine",
        lambda: SimpleNamespace(_model=model, _processor=processor),
        raising=False,
    )
    assert omni._shared_speakable_model() == (model, processor)


def test_shared_speakable_model_none_without_talker(monkeypatch):
    from yunshu_gateway.routers import omni

    monkeypatch.setattr(
        "yunshu_gateway.engine.get_engine",
        lambda: SimpleNamespace(_model=SimpleNamespace(has_talker=False), _processor=1),
        raising=False,
    )
    assert omni._shared_speakable_model() is None


def test_get_omni_engine_reuses_shared_model(monkeypatch):
    from yunshu_gateway.routers import omni

    omni._omni_engine = None  # reset singleton
    model = SimpleNamespace(has_talker=True)
    processor = object()
    monkeypatch.setattr(omni, "_shared_speakable_model", lambda: (model, processor))

    eng = omni._get_omni_engine()
    assert eng.model is model  # adopted, not loaded from a path
    assert eng.model_path is None
    omni._omni_engine = None


def test_get_omni_engine_503_when_no_model(monkeypatch):
    from fastapi import HTTPException

    from yunshu_gateway.routers import omni

    omni._omni_engine = None
    monkeypatch.setattr(omni, "_shared_speakable_model", lambda: None)
    monkeypatch.delenv("YUNSHU_OMNI_MODEL", raising=False)
    with pytest.raises(HTTPException) as ei:
        omni._get_omni_engine()
    assert ei.value.status_code == 503
    omni._omni_engine = None


# ── realtime: native voice is on by default when a model can speak ──────────
def test_realtime_enabled_auto_on_when_served_model_speaks(monkeypatch):
    from yunshu_gateway.routers import omni, realtime

    monkeypatch.delenv("YUNSHU_REALTIME_OMNI", raising=False)
    monkeypatch.delenv("YUNSHU_OMNI_MODEL", raising=False)
    monkeypatch.setattr(omni, "_shared_speakable_model", lambda: (object(), object()))
    assert realtime._omni_realtime_enabled() is True


def test_realtime_disabled_for_nonspeaking_model(monkeypatch):
    from yunshu_gateway.routers import omni, realtime

    monkeypatch.delenv("YUNSHU_REALTIME_OMNI", raising=False)
    monkeypatch.delenv("YUNSHU_OMNI_MODEL", raising=False)
    monkeypatch.setattr(omni, "_shared_speakable_model", lambda: None)
    assert realtime._omni_realtime_enabled() is False


def test_realtime_env_zero_forces_cascade(monkeypatch):
    from yunshu_gateway.routers import omni, realtime

    monkeypatch.setenv("YUNSHU_REALTIME_OMNI", "0")
    monkeypatch.setattr(omni, "_shared_speakable_model", lambda: (object(), object()))
    assert realtime._omni_realtime_enabled() is False
