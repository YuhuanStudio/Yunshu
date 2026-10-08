"""Every engine *_gb field is binary GB (1024**3) with an exact *_bytes sibling."""

from __future__ import annotations

from types import SimpleNamespace

from yunshu_engine.units import GIB, gb, put_gb

TOTAL = 128 * GIB


def test_helpers():
    assert gb(TOTAL, 1) == 128.0
    assert gb(None) is None
    out: dict = {}
    put_gb(out, "total", TOTAL, 1)
    assert out == {"total_gb": 128.0, "total_bytes": TOTAL}


def test_model_manager_fields():
    from yunshu_engine.model_manager import ModelManager, ModelType

    m = ModelManager.__new__(ModelManager)
    e = SimpleNamespace(
        model_id="a",
        model_type=ModelType.LLM,
        is_loaded=True,
        is_pinned=False,
        is_loading=False,
        estimated_bytes=TOTAL,
        last_access=0.0,
        load_error=None,
        engine=None,
    )
    m._entries = {"a": e}
    m._current_memory_bytes = TOTAL
    m.max_memory_bytes = TOTAL
    m._eviction_stats = {}
    m.memory_pressure_threshold = 0.9
    row = m.list_models()[0]
    assert row["size_gb"] == 128.0 and row["size_bytes"] == TOTAL
    mu = m.memory_usage
    assert mu["current_gb"] == mu["max_gb"] == 128.0
    assert mu["current_bytes"] == mu["max_bytes"] == TOTAL
    st = m.get_status()
    assert st["max_memory_gb"] == st["current_memory_gb"] == 128.0
    assert st["max_memory_bytes"] == st["current_memory_bytes"] == TOTAL
    assert st["models"][0]["size_gb"] == 128.0
    assert st["models"][0]["size_bytes"] == TOTAL


def test_status_memory(monkeypatch):
    import sys
    import types

    from yunshu_gateway.routers import yunshu

    mx = types.ModuleType("mlx.core")
    mx.get_active_memory = lambda: 64 * GIB
    mx.get_cache_memory = lambda: GIB
    mx.get_peak_memory = lambda: 65 * GIB
    pkg = types.ModuleType("mlx")
    pkg.core = mx
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setattr(
        yunshu.os, "sysconf", lambda n: 16384 if n == "SC_PAGE_SIZE" else TOTAL // 16384
    )
    out = yunshu._memory()
    assert out["total_gb"] == 128.0 and out["total_bytes"] == TOTAL
    assert out["active_gb"] == 64.0 and out["active_bytes"] == 64 * GIB
    assert out["cache_gb"] == 1.0 and out["peak_gb"] == 65.0
    assert out["pressure"] == 0.5


def test_models_route_size(monkeypatch):
    from yunshu_gateway.routers import models

    class Card:
        pass

    monkeypatch.setattr(
        "yunshu_gateway.model_card_formats.openai_model",
        lambda card, detailed=False: {},
    )
    entry = SimpleNamespace(estimated_bytes=TOTAL, is_loaded=False, engine=None)
    item = models._model_payload(Card(), entry, True)
    assert item["size_gb"] == 128.0 and item["size_bytes"] == TOTAL
