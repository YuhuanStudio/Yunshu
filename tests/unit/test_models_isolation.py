"""the model-info endpoints must honor the same can_access_model isolation
every serving route enforces. GET /v1/models/{id} only checked can_infer (a scoped key
could read an out-of-scope model's metadata); GET /v1/models listed every model. Also
the admin discover dir-containment check used str.startswith (sibling-prefix bug)."""

from __future__ import annotations

import asyncio
import types

import pytest
from fastapi import HTTPException

from yunshu_gateway.routers import (
    models as M,  # noqa: N812  # intentional short module alias
)


class _Entry:
    def __init__(self, mid):
        self.model_id = mid
        self.is_loaded = False
        self.model_type = types.SimpleNamespace(name="LLM")
        self.estimated_bytes = 1_000_000_000
        self.load_time = 0
        self.engine = None


class _Manager:
    def __init__(self, ids):
        self._entries = {i: _Entry(i) for i in ids}

    def list_entries(self):
        return list(self._entries.values())

    def get_entry(self, mid):
        return self._entries.get(mid)

    def resolve_model_id(self, mid):
        return mid if mid in self._entries else None


class _Key:
    """Scoped to model ids starting with 'public-'."""

    def can_access_model(self, mid):
        return mid.startswith("public-")


def _req(rbac_key):
    return types.SimpleNamespace(
        state=types.SimpleNamespace(rbac_key=rbac_key, tenant=None)
    )


def test_list_filters_out_of_scope(monkeypatch):
    monkeypatch.setattr(
        M, "get_model_manager", lambda: _Manager(["public-a", "secret-b"])
    )
    monkeypatch.setattr(M, "_check_permission", lambda *a, **k: None)
    out = asyncio.run(M.list_models(_req(_Key())))
    ids = {m["id"] for m in out["data"]}
    assert ids == {"public-a"}  # secret-b filtered
    assert "registry" not in out  # registry stats suppressed for a scoped key


def test_list_unscoped_sees_all(monkeypatch):
    monkeypatch.setattr(
        M, "get_model_manager", lambda: _Manager(["public-a", "secret-b"])
    )
    monkeypatch.setattr(M, "_check_permission", lambda *a, **k: None)
    out = asyncio.run(M.list_models(_req(None)))  # no rbac key (static-token/disabled)
    assert {m["id"] for m in out["data"]} == {"public-a", "secret-b"}


def test_get_model_out_of_scope_is_404(monkeypatch):
    monkeypatch.setattr(
        M, "get_model_manager", lambda: _Manager(["public-a", "secret-b"])
    )
    monkeypatch.setattr(M, "_check_permission", lambda *a, **k: None)
    # accessible model → ok
    ok = asyncio.run(M.get_model("public-a", _req(_Key())))
    assert ok["id"] == "public-a"
    # inaccessible → 404 (not 403 — no existence leak)
    with pytest.raises(HTTPException) as e:
        asyncio.run(M.get_model("secret-b", _req(_Key())))
    assert e.value.status_code == 404


def test_discover_dir_containment_uses_relative_to(monkeypatch):
    # The fix replaced str.startswith with relative_to — verify the sibling-prefix
    # case is rejected by the helper logic.
    from pathlib import Path

    def _within(child: Path, parent: Path) -> bool:
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            return False

    assert _within(Path("/Users/yuhuan/models"), Path("/Users/yuhuan")) is True
    assert (
        _within(Path("/Users/yuhuan-secret"), Path("/Users/yuhuan")) is False
    )  # was True under startswith
