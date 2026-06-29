"""(HIGH): the MCP resources surface (resources/list + resources/read) leaked
per-model metadata to a model-scoped RBAC key — the per-key isolation that /v1/models
applies was never propagated to MCP. resources/list enumerated every model's id/type/size;
resources/read returned full status/size/load-error for an arbitrary yunshu://models/<id>.
Both reached their handler behind only can_infer (the _check_model_access gate covered
only tools/call). Now resources/list filters by the key and resources/read is gated."""
from __future__ import annotations

import asyncio
import inspect
import types

from yunshu_gateway import engine as eng_mod
from yunshu_gateway.routers import (
    mcp as MCP,  # noqa: N812  # intentional short module alias
)


class _Key:
    def __init__(self, allowed):
        self._allowed = allowed

    def can_access_model(self, model_id):
        return model_id in self._allowed


def _patch_manager(monkeypatch):
    models = [
        {"id": "model-A", "type": "llm", "size_gb": 1.0},
        {"id": "model-B", "type": "llm", "size_gb": 2.0},
    ]
    mgr = types.SimpleNamespace(list_models=lambda: models)
    monkeypatch.setattr(eng_mod, "get_model_manager", lambda: mgr)


def test_resources_list_filters_by_key(monkeypatch):
    _patch_manager(monkeypatch)
    key = _Key({"model-A"})  # scoped away from model-B
    resp = asyncio.run(MCP._handle_resources_list(None, 1, key))
    names = {r["name"] for r in resp["result"]["resources"]}
    assert names == {"model-A"}, f"scoped key must not see model-B, got {names}"


def test_resources_list_no_rbac_sees_all(monkeypatch):
    _patch_manager(monkeypatch)
    resp = asyncio.run(MCP._handle_resources_list(None, 1, None))
    names = {r["name"] for r in resp["result"]["resources"]}
    assert names == {"model-A", "model-B"}


def test_resources_read_gated_in_endpoint_source():
    # the endpoint access-gate must cover resources/read (resolve the model id from the
    # uri and _check_model_access) — mirrors the tools/call gate.
    src = inspect.getsource(MCP)
    assert 'req.method == "resources/read"' in src
    assert 'yunshu://models/' in src
    # resources/list is dispatched with the caller's rbac_key
    assert "_handle_resources_list(\n                req.params, req.id, getattr(request.state, \"rbac_key\", None))" in src \
        or "_handle_resources_list(req.params, req.id, getattr(request.state" in src
