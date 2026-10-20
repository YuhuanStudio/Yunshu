"""CORS must wrap auth / rate limit / metrics so their error responses carry CORS headers."""

from __future__ import annotations

from yunshu_gateway.main import create_app


def test_live_cors_is_outside_auth_and_rate_limit(monkeypatch):
    monkeypatch.setenv("YUNSHU_CORS_ORIGINS", "*")
    names = [m.cls.__name__ for m in create_app().user_middleware]  # outermost first
    cors = names.index("LiveCORSMiddleware")
    for inner in ("AuthMiddleware", "YunshuExtensionsMiddleware", "MetricsMiddleware"):
        assert cors < names.index(inner), (inner, names)
