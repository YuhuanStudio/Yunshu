"""Tests for new admin monitoring endpoints."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from yunshu_gateway.main import create_app
    app = create_app()
    return TestClient(app)


class TestAdminHardware:
    def test_hardware_endpoint(self, client):
        resp = client.get("/api/v1/admin/hardware")
        assert resp.status_code == 200
        data = resp.json()
        assert "hardware" in data
        assert "chip" in data["hardware"]
        assert "total_memory_gb" in data["hardware"]

    def test_memory_endpoint(self, client):
        resp = client.get("/api/v1/admin/memory")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_bytes" in data
        assert "active_bytes" in data


class TestAdminMetrics:
    def test_metrics_endpoint(self, client):
        resp = client.get("/api/v1/admin/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data
        assert "uptime_seconds" in data

    def test_metrics_clear(self, client):
        resp = client.post("/api/v1/admin/metrics/clear")
        assert resp.status_code == 200

    def test_prefill_endpoint(self, client):
        resp = client.get("/api/v1/admin/metrics/prefill")
        assert resp.status_code == 200

    def test_registry_endpoint(self, client):
        resp = client.get("/api/v1/admin/registry")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_entries" in data

    def test_discover_endpoint(self, client):
        resp = client.get("/api/v1/admin/models/discover")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data


class TestHealthWithMetrics:
    def test_health_includes_metrics(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "metrics" in data
        assert data["metrics"]["total_requests"] == 0
