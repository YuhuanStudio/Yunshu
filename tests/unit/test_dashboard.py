"""Tests for Dashboard API endpoints."""

import pytest
from fastapi.testclient import TestClient

from yunshu_engine.engine import Engine, EngineConfig


class TestDashboard:
    """Test dashboard configuration and summary endpoints."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from yunshu_gateway.engine import set_engine
        self._engine = Engine(EngineConfig())
        self._engine._model = object()
        self._engine._model_name = "test-model"
        self._engine._running = True
        set_engine(self._engine)

    def test_get_config(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/v1/dashboard/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "theme" in data
        assert data["theme"] == "dark"

    def test_update_config(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.put("/api/v1/dashboard/config", json={
            "theme": "light",
            "refresh_interval_ms": 10000,
            "show_gpu_stats": True,
        })
        assert resp.status_code == 200
        assert resp.json()["theme"] == "light"

        # Verify it persists
        resp = client.get("/api/v1/dashboard/config")
        assert resp.json()["theme"] == "light"

    def test_summary(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/v1/dashboard/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "engine" in data
        assert "models" in data
        assert "gpu" in data
        assert "auth" in data

    def test_usage(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/v1/dashboard/usage")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data
        assert "prompt_tokens" in data
        assert "completion_tokens" in data
