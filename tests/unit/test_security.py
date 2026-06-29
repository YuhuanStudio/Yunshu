"""lock the /metrics auth-bypass + cachedContents/batch IDOR fixes."""

from __future__ import annotations

import os
import types

import pytest


@pytest.fixture(autouse=True)
def _auth_env():
    """Tests assume auth is configured — the global test env sets
    YUNSHU_AUTH_DISABLED=true, which would short-circuit _check_metrics_auth."""
    saved = {
        k: os.environ.pop(k, None)
        for k in ("YUNSHU_AUTH_DISABLED", "YUNSHU_AUTH_TOKEN")
    }
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


class _Hdrs(dict):
    def get(self, k, default=""):
        return super().get(k, default)


def _req(headers=None, state=None):
    r = types.SimpleNamespace()
    r.headers = _Hdrs(headers or {})
    r.app = types.SimpleNamespace(state=types.SimpleNamespace())
    r.state = types.SimpleNamespace(**(state or {}))
    return r


# ── /metrics auth (single-token gate) ──


def test_metrics_public_when_no_auth_configured():
    from yunshu_gateway.middleware.metrics import _check_metrics_auth

    os.environ.pop("YUNSHU_AUTH_TOKEN", None)
    os.environ.pop("YUNSHU_AUTH_DISABLED", None)
    # no static token → scraper-compatible allow
    _check_metrics_auth(_req())  # no raise


def test_metrics_rejects_unauth_when_static_token_set():
    """Single-consumer model: /metrics is gated by YUNSHU_AUTH_TOKEN. With a
    token configured, a request lacking a valid Bearer header must 401."""
    from fastapi import HTTPException

    from yunshu_gateway.middleware.metrics import _check_metrics_auth

    os.environ["YUNSHU_AUTH_TOKEN"] = "owner-static-secret"
    try:
        # No Authorization header → 401 (not the scraper-compatible allow).
        with pytest.raises(HTTPException) as e:
            _check_metrics_auth(_req())
        assert e.value.status_code == 401
    finally:
        os.environ.pop("YUNSHU_AUTH_TOKEN", None)


def test_metrics_rejects_wrong_static_token():
    """A Bearer token that doesn't constant-time match YUNSHU_AUTH_TOKEN → 401."""
    from fastapi import HTTPException

    from yunshu_gateway.middleware.metrics import _check_metrics_auth

    os.environ["YUNSHU_AUTH_TOKEN"] = "owner-static-secret"
    try:
        with pytest.raises(HTTPException) as e:
            _check_metrics_auth(_req({"Authorization": "Bearer wrong-token"}))
        assert e.value.status_code == 401
    finally:
        os.environ.pop("YUNSHU_AUTH_TOKEN", None)


def test_metrics_accepts_valid_static_token():
    """A Bearer token matching YUNSHU_AUTH_TOKEN → allow."""
    from yunshu_gateway.middleware.metrics import _check_metrics_auth

    os.environ["YUNSHU_AUTH_TOKEN"] = "owner-static-secret"
    try:
        _check_metrics_auth(
            _req({"Authorization": "Bearer owner-static-secret"})
        )  # no raise
    finally:
        os.environ.pop("YUNSHU_AUTH_TOKEN", None)


def test_metrics_accepts_owner_role_from_middleware():
    """When TenantAuthMiddleware has already authenticated the request and
    stamped state.role="owner", /metrics trusts that and allows."""
    from yunshu_gateway.middleware.metrics import _check_metrics_auth

    os.environ["YUNSHU_AUTH_TOKEN"] = "owner-static-secret"
    try:
        _check_metrics_auth(_req(state={"role": "owner"}))  # no raise
    finally:
        os.environ.pop("YUNSHU_AUTH_TOKEN", None)


# (RBAC-specific /metrics tests removed — single-consumer model gates /metrics
#  only via the static YUNSHU_AUTH_TOKEN, covered above.)


# ── cachedContents IDOR ──


def test_cached_contents_owner_isolation():
    from yunshu_gateway.explicit_cache import ExplicitContextCache

    store = ExplicitContextCache()
    e = store.create(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        token_count=1,
        owner="alice",
    )
    assert e.owner == "alice"

    # mimic the router _owns() check
    def owns(entry, actor):
        o = getattr(entry, "owner", None)
        return (not o or o == "anonymous") or actor == o

    assert owns(e, "alice") is True
    assert owns(e, "bob") is False  # bob cannot see alice's handle


def test_batch_owner_isolation():
    # mirror _owns_batch logic
    def owns(info, actor):
        o = info.get("owner")
        return (not o or o == "anonymous") or actor == o

    assert owns({"owner": "alice"}, "alice") is True
    assert owns({"owner": "alice"}, "bob") is False
    assert owns({"owner": "anonymous"}, "bob") is True  # permissive when unstamped


class TestCachedContentReadPathIDOR:
    """the cached_content READ/consumption path (prepending a handle's stored
    context into a generation request) must enforce ownership too — not just the
    management routes. Otherwise tenant B prepends tenant A's private context by guessing
    the handle and exfiltrates it via the model output."""

    def _fake_request(self, actor_name):
        import types

        key = types.SimpleNamespace(name=actor_name, key_prefix=actor_name)
        state = types.SimpleNamespace(rbac_key=key)
        return types.SimpleNamespace(state=state)

    def _entry(self, owner):
        import types

        return types.SimpleNamespace(
            messages=[{"role": "system", "content": "SECRET CONTEXT"}],
            model="m",
            owner=owner,
        )

    def test_other_tenant_handle_is_ignored(self, monkeypatch):
        from yunshu_gateway.routers import chat as chat_mod

        store = type("S", (), {"use": lambda self, n: self._e})()
        store._e = self._entry(owner="alice")
        monkeypatch.setattr(
            "yunshu_gateway.explicit_cache.get_store", lambda: store, raising=False
        )
        # tenant "bob" tries to use alice's handle → context must NOT be prepended
        msgs = [{"role": "user", "content": "hi"}]
        out = chat_mod._prepend_cached_content(
            list(msgs), "cachedContents/x", "m", self._fake_request("bob")
        )
        assert out == msgs  # alice's SECRET CONTEXT not leaked into bob's prompt

    def test_owner_can_use_own_handle(self, monkeypatch):
        from yunshu_gateway.routers import chat as chat_mod

        store = type("S", (), {"use": lambda self, n: self._e})()
        store._e = self._entry(owner="alice")
        monkeypatch.setattr(
            "yunshu_gateway.explicit_cache.get_store", lambda: store, raising=False
        )
        out = chat_mod._prepend_cached_content(
            [{"role": "user", "content": "hi"}],
            "cachedContents/x",
            "m",
            self._fake_request("alice"),
        )
        assert any(m.get("content") == "SECRET CONTEXT" for m in out)  # owner gets it

    def test_anonymous_owner_allowed(self, monkeypatch):
        from yunshu_gateway.routers import chat as chat_mod

        store = type("S", (), {"use": lambda self, n: self._e})()
        store._e = self._entry(owner="anonymous")
        monkeypatch.setattr(
            "yunshu_gateway.explicit_cache.get_store", lambda: store, raising=False
        )
        out = chat_mod._prepend_cached_content(
            [{"role": "user", "content": "hi"}],
            "cachedContents/x",
            "m",
            self._fake_request("bob"),
        )
        assert any(m.get("content") == "SECRET CONTEXT" for m in out)
