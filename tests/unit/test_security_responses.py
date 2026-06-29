"""lock the Responses-API store/cancel cross-tenant IDOR fixes
and the multi-hop chaining duplication fix.

The OpenAI Responses store (`store: true`) keeps responses for retrieval via
GET /v1/responses/{id}, cancellation, and previous_response_id chaining. Before
the store recorded NO owner, so any authenticated tenant could read,
cancel, or chain off another tenant's stored response by id (the SSE stream
leaks the resp- id). Every other per-handle router enforces ownership;
Responses was missed.
"""

from __future__ import annotations

import asyncio
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


@pytest.fixture(autouse=True)
def _stub_actor(monkeypatch):
    """resolve_actor(request) -> request._actor (mirrors the real per-request actor)."""
    import yunshu_control.audit_log as al

    monkeypatch.setattr(
        al, "resolve_actor", lambda request: getattr(request, "_actor", "")
    )
    # Permission check is orthogonal to ownership — stub it permissive.
    monkeypatch.setattr(R, "_check_permission", lambda request, perm: None)
    # Ensure auth-disabled env doesn't auto-grant admin.
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    R._response_store.clear()
    yield
    R._response_store.clear()


# ── _owns_stored / _public_stored helpers ──


def test_owns_stored_owner_match():
    assert R._owns_stored(_req(actor="alice"), {"_owner": "alice"}) is True


def test_owns_stored_cross_tenant_denied():
    assert R._owns_stored(_req(actor="bob"), {"_owner": "alice"}) is False


def test_owns_stored_admin_bypass():
    assert R._owns_stored(_req(actor="bob", role="admin"), {"_owner": "alice"}) is True


def test_owns_stored_unowned_is_permissive():
    # Legacy entries with no _owner stay readable (don't break pre-stores).
    assert R._owns_stored(_req(actor="bob"), {"id": "x"}) is True


def test_public_stored_strips_internal_keys():
    pub = R._public_stored(
        {"id": "r1", "_owner": "alice", "_input_messages": [1], "status": "completed"}
    )
    assert pub == {"id": "r1", "status": "completed"}
    assert "_owner" not in pub and "_input_messages" not in pub


# ── GET /v1/responses/{id} ownership gate ──


def _run(coro):
    return asyncio.run(coro)


def test_get_response_cross_tenant_is_404():
    R._store_response(
        "resp-1", {"id": "resp-1", "status": "completed", "_owner": "alice"}
    )
    resp = _run(R.get_response("resp-1", _req(actor="bob")))
    assert resp.status_code == 404


def test_get_response_owner_ok_and_stripped():
    R._store_response(
        "resp-2",
        {
            "id": "resp-2",
            "status": "completed",
            "_owner": "alice",
            "_input_messages": [{"role": "user", "content": "SECRET"}],
        },
    )
    resp = _run(R.get_response("resp-2", _req(actor="alice")))
    assert resp.status_code == 200
    import json

    body = json.loads(bytes(resp.body))
    assert body["id"] == "resp-2"
    # Internal plumbing must not leak even to the owner.
    assert "_owner" not in body and "_input_messages" not in body


def test_get_response_admin_can_read_any():
    R._store_response(
        "resp-3", {"id": "resp-3", "status": "completed", "_owner": "alice"}
    )
    resp = _run(R.get_response("resp-3", _req(actor="bob", role="admin")))
    assert resp.status_code == 200


# ── cancel ownership gate ──


def test_cancel_cross_tenant_in_flight_is_404(monkeypatch):
    """Tenant B cannot cancel tenant A's in-flight (tracker-owned) response."""

    class _Tracker:
        def get_owner(self, rid):
            return "alice"

        def cancel(self, rid):
            raise AssertionError(
                "cancel() must not be reached for a cross-tenant request"
            )

    import yunshu_engine.request_tracker as rt

    monkeypatch.setattr(rt, "get_request_tracker", lambda: _Tracker())
    resp = _run(R.cancel_response("resp-x", _req(actor="bob")))
    assert resp.status_code == 404


def test_cancel_owner_in_flight_ok(monkeypatch):
    class _Tracker:
        def get_owner(self, rid):
            return "alice"

        def cancel(self, rid):
            return True

    import yunshu_engine.request_tracker as rt

    monkeypatch.setattr(rt, "get_request_tracker", lambda: _Tracker())
    resp = _run(R.cancel_response("resp-y", _req(actor="alice")))
    # In-flight cancelled → synthetic cancelled envelope (200).
    assert resp.status_code == 200
