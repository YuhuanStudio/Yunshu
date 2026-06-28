"""unregister_model must refuse while a model is LOADING, symmetric
with register_model's existing is_loading guard.

Deleting an entry mid-load orphans the in-flight loader's local `entry` reference:
the load completes outside `_entries`, so the memory counter is credited for a model
that no longer exists (permanent drift → eventually blocks all loads), the loaded
engine is never reachable by shutdown()/eviction (engine.stop() never runs, the W957
leak class), and the finally's `_loading_events.pop` can pop a re-registered model's
NEW event so its fresh waiters hang. The is_loaded guard alone left this window open.
"""
from __future__ import annotations

import pytest

from yunshu_engine.model_manager import ModelEntry, ModelManager, ModelType


def _mgr_with_state(*, is_loaded=False, is_loading=False):
    mgr = ModelManager()
    mgr._entries["m"] = ModelEntry(
        model_id="m", model_path="/x", model_type=ModelType.LLM,
        is_loaded=is_loaded, estimated_bytes=0,
    )
    mgr._entries["m"].is_loading = is_loading
    return mgr


def test_unregister_refuses_while_loading():
    mgr = _mgr_with_state(is_loading=True)
    with pytest.raises(ValueError, match="loading"):
        mgr.unregister_model("m")
    assert "m" in mgr._entries  # NOT deleted — loader's entry ref stays valid


def test_unregister_still_refuses_while_loaded():
    mgr = _mgr_with_state(is_loaded=True)
    with pytest.raises(ValueError, match="loaded"):
        mgr.unregister_model("m")
    assert "m" in mgr._entries


def test_unregister_succeeds_when_idle_registered():
    mgr = _mgr_with_state()  # registered, neither loaded nor loading
    assert mgr.unregister_model("m") is True
    assert "m" not in mgr._entries


def test_unregister_missing_returns_false():
    mgr = ModelManager()
    assert mgr.unregister_model("nope") is False
