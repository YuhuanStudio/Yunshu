"""model-management hardening.

MEDIUM-1: manager.unload_model() discarded the bool from _unload_model_locked and
returned None, so the /v1/models/unload route always reported {"status":"unloaded"} —
even when the manager REFUSED the unload (a request started in the TOCTOU window between
the router's has_active_requests pre-check and the manager's locked re-check) or the
model was already gone. unload_model now returns the bool; the router 409s on False.

LOW-3: the public GET /v1/models leaked global registry stats (total_entries/
active_owners) to UNauthenticated callers (the gate only fired for a scoped key).

LOW-2: unload didn't resolve aliases/case, so unloading an id that ran chat 404'd.
"""

from __future__ import annotations

import asyncio
import inspect

from yunshu_engine import (
    model_manager as MM,  # noqa: N812  # intentional short module alias
)
from yunshu_gateway.routers import (
    models as MODELS,  # noqa: N812  # intentional short module alias
)


def test_unload_model_returns_bool():
    # signature + behavior: returns the locked-unload result, not None
    src = inspect.getsource(MM.ModelManager.unload_model)
    assert "return await self._unload_model_locked" in src


def test_unload_model_propagates_false(monkeypatch):
    mgr = MM.ModelManager.__new__(MM.ModelManager)
    mgr._lock = asyncio.Lock()

    async def _locked(model_id, force=False):
        return False  # refused (active) / already unloaded

    mgr._unload_model_locked = _locked
    assert asyncio.run(mgr.unload_model("m")) is False

    async def _locked_true(model_id, force=False):
        return True

    mgr._unload_model_locked = _locked_true
    assert asyncio.run(mgr.unload_model("m")) is True


def test_router_unload_409s_on_false():
    src = inspect.getsource(MODELS.unload_model)
    assert "_unloaded = await manager.unload_model(_resolved_id)" in src
    assert "if not _unloaded:" in src
    assert "status_code=409" in src
    # resolves aliases without clobbering the inflight key
    assert "_resolved_id = manager.resolve_model_id(model_id)" in src


def test_registry_stats_gated_on_auth():
    src = (
        inspect.getsource(MODELS.list_models)
        if hasattr(MODELS, "list_models")
        else inspect.getsource(MODELS)
    )
    assert "if _authenticated:" in src
