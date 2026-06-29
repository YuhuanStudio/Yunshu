"""the Responses-API hunt found DELETE /v1/responses/{id} was missing entirely —
a stored response/conversation could never be explicitly deleted (only LRU-evicted), so
clients got 405 and a stored conversation lingered until eviction (a privacy gap). Added an
OpenAI-compatible delete handler, ownership-gated exactly like GET (cross-tenant delete
denied as not-found, the IDOR class).
"""
from __future__ import annotations

import asyncio
import json
import types

import pytest

from yunshu_gateway.routers import (
    responses as R,  # noqa: N812  # intentional short module alias
)


def _req(actor="", role=""):
    r = types.SimpleNamespace()
    r.state = types.SimpleNamespace(role=role)
    r.headers = {}
    r._actor = actor
    return r


def _body(resp):
    return json.loads(bytes(resp.body).decode())


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    import yunshu_control.audit_log as al
    monkeypatch.setattr(al, "resolve_actor", lambda request: getattr(request, "_actor", ""))
    monkeypatch.setattr(R, "_check_permission", lambda request, perm: None)
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    R._response_store.clear()
    yield
    R._response_store.clear()


def test_delete_removes_stored_response():
    R._store_response("resp-1", {"id": "resp-1", "object": "response", "_owner": "alice"})
    resp = asyncio.run(R.delete_response("resp-1", _req(actor="alice")))
    assert resp.status_code == 200
    body = _body(resp)
    assert body == {"id": "resp-1", "object": "response.deleted", "deleted": True}
    # gone: a follow-up GET 404s
    g = asyncio.run(R.get_response("resp-1", _req(actor="alice")))
    assert g.status_code == 404


def test_delete_unknown_id_404():
    resp = asyncio.run(R.delete_response("resp-nope", _req(actor="alice")))
    assert resp.status_code == 404


def test_delete_cross_tenant_denied_and_preserves_entry():
    R._store_response("resp-2", {"id": "resp-2", "object": "response", "_owner": "alice"})
    # bob may not delete alice's response → 404, and it must remain for alice
    resp = asyncio.run(R.delete_response("resp-2", _req(actor="bob")))
    assert resp.status_code == 404
    assert R._get_stored_response("resp-2") is not None
    # alice can still delete it
    ok = asyncio.run(R.delete_response("resp-2", _req(actor="alice")))
    assert ok.status_code == 200


def test_delete_admin_bypass():
    R._store_response("resp-3", {"id": "resp-3", "object": "response", "_owner": "alice"})
    resp = asyncio.run(R.delete_response("resp-3", _req(actor="bob", role="admin")))
    assert resp.status_code == 200
    assert R._get_stored_response("resp-3") is None
