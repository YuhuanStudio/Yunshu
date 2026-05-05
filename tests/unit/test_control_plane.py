"""Tests for Yunshu Control Plane API."""

import pytest
from fastapi.testclient import TestClient

from yunshu_engine.engine import Engine, EngineConfig


class TestControlPlane:
    """Test the control plane admin/monitoring endpoints."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from yunshu_gateway.engine import set_engine
        self._engine = Engine(EngineConfig())
        self._engine._model = object()
        self._engine._model_name = "test-model"
        self._engine._running = True
        set_engine(self._engine)

    def test_health_endpoint(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_admin_models_list(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/v1/admin/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_monitoring_system(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/v1/monitoring/system")
        assert resp.status_code == 200
        data = resp.json()
        assert "gpu" in data
        assert "python_version" in data
        assert "mlx_version" in data

    def test_monitoring_engine(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/v1/monitoring/engine")
        assert resp.status_code == 200
        data = resp.json()
        assert "loaded" in data
        assert "running" in data
        assert "step_counter" in data

    def test_monitoring_requests(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/v1/monitoring/requests")
        assert resp.status_code == 200
        data = resp.json()
        assert "tokens_per_second" in data
        assert "requests_per_second" in data

    def test_engine_config_get(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/v1/admin/config/engine")
        assert resp.status_code == 200
        data = resp.json()
        assert "completion_batch_size" in data
        assert "prefill_batch_size" in data

    def test_engine_config_update(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.patch(
            "/api/v1/admin/config/engine",
            json={"completion_batch_size": 64},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "updated"
        assert "completion_batch_size" in data["fields"]

    def test_token_crud(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        # Create token
        resp = client.post(
            "/api/v1/admin/tokens",
            json={"name": "test-token", "expires_days": 30},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["token"].startswith("ys_")
        assert data["name"] == "test-token"
        token = data["token"]

        # List tokens
        resp = client.get("/api/v1/admin/tokens")
        assert resp.status_code == 200
        tokens = resp.json()
        assert len(tokens) >= 1

        # Revoke token
        resp = client.delete(f"/api/v1/admin/tokens/{token[:8]}")
        assert resp.status_code == 200
