"""Tests for new admin monitoring endpoints."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from yunshu_gateway.main import create_app
    app = create_app()
    return TestClient(app)


class TestHealthWithMetrics:
    def test_health_does_not_leak_metrics(self, client):
        # SECURITY: public /health must NOT leak the server-metrics snapshot
        # (or model ids / memory / registry internals) to an unauthenticated caller.
        # The rich detail lives behind the authenticated /api/v1/monitoring/* router.
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "metrics" not in data
        assert "model_registry" not in data and "mcp_client" not in data
        assert data["status"] in ("ok", "sleeping")  # minimal liveness payload still present
