"""Tests for Yunshu Gateway middleware."""

import os
from unittest.mock import patch

from fastapi.testclient import TestClient

from yunshu_gateway.main import create_app


class TestMetricsMiddleware:
    """Test Prometheus metrics endpoint."""

    def test_metrics_endpoint(self):
        with patch.dict(os.environ, {"YUNSHU_AUTH_DISABLED": "true"}):
            app = create_app()
            client = TestClient(app)
            resp = client.get("/metrics")
            assert resp.status_code == 200
            text = resp.text
            assert "yunshu_uptime_seconds" in text
            assert "yunshu_request_count" in text
            assert "yunshu_tokens_total" in text

    def test_metrics_record_requests(self):
        with patch.dict(os.environ, {"YUNSHU_AUTH_DISABLED": "true"}):
            app = create_app()
            client = TestClient(app)
            client.get("/health")
            client.get("/health")
            client.get("/v1/models")
            resp = client.get("/metrics")
            text = resp.text
            assert "yunshu_request_count" in text


class TestRateLimitMiddleware:
    """Test rate limiting."""

    def test_health_not_limited(self):
        app = create_app()
        client = TestClient(app)
        # Health should never be rate limited
        for _ in range(50):
            resp = client.get("/health")
            assert resp.status_code == 200


class TestAuthMiddleware:
    """Test tenant authentication."""

    def test_no_auth_without_token(self):
        """Auth is off when YUNSHU_AUTH_TOKEN is not set — requests pass through."""
        import os

        os.environ.pop("YUNSHU_AUTH_TOKEN", None)
        app = create_app()
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 200

    def test_auth_disabled_via_env(self):
        """YUNSHU_AUTH_DISABLED=true allows all requests."""
        import os

        os.environ["YUNSHU_AUTH_DISABLED"] = "true"
        os.environ.pop("YUNSHU_AUTH_TOKEN", None)
        try:
            app = create_app()
            client = TestClient(app)
            resp = client.get("/v1/models")
            assert resp.status_code == 200
        finally:
            os.environ.pop("YUNSHU_AUTH_DISABLED", None)
