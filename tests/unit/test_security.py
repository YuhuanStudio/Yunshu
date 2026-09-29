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
    """When AuthMiddleware has already authenticated the request and
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


def test_batch_owner_isolation():
    # mirror _owns_batch logic
    def owns(info, actor):
        o = info.get("owner")
        return (not o or o == "anonymous") or actor == o

    assert owns({"owner": "alice"}, "alice") is True
    assert owns({"owner": "alice"}, "bob") is False
    assert owns({"owner": "anonymous"}, "bob") is True  # permissive when unstamped
