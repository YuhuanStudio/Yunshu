"""Tests for production health checks and graceful shutdown."""

import pytest
from fastapi.testclient import TestClient

from yunshu_engine.engine import Engine, EngineConfig


class TestHealthChecks:
    """Test health, readiness, and liveness probes."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from yunshu_gateway.engine import set_engine
        import yunshu_gateway.main as _main

        _main._shutting_down = False
        _main._active_requests = 0

        self._engine = Engine(EngineConfig())
        self._engine._model = object()
        self._engine._model_name = "test-model"
        self._engine._running = True
        set_engine(self._engine)
        self._main = _main

    def test_health_endpoint(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "engine" in data

    def test_liveness_probe(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.get("/health/live")
        assert resp.status_code == 200
        data = resp.json()
        assert data["alive"] is True
        assert "shutting_down" in data

    def test_readiness_probe_with_model(self):
        from yunshu_gateway.main import create_app
        from yunshu_gateway.engine import set_engine

        # Fresh engine to avoid shared state from other tests
        engine = Engine(EngineConfig())
        engine._model = object()
        engine._model_name = "test-model"
        engine._running = True
        set_engine(engine)

        app = create_app()
        client = TestClient(app)

        resp = client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["checks"]["model_loaded"] is True
        assert data["checks"]["gpu_memory_ok"] is True
        assert data["checks"]["not_shutting_down"] is True
        assert data["ready"] is True

    def test_readiness_probe_no_model(self):
        from yunshu_gateway.main import create_app
        from yunshu_gateway.engine import set_engine
        set_engine(None)

        app = create_app()
        client = TestClient(app)

        resp = client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["checks"]["model_loaded"] is False
        assert data["ready"] is False


class TestRequestTracking:
    """Test request ID tracking and active request counting."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from yunshu_gateway.engine import set_engine
        self._engine = Engine(EngineConfig())
        self._engine._model = object()
        self._engine._model_name = "test-model"
        self._engine._running = True
        set_engine(self._engine)

    def test_request_id_header(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.get("/health", headers={"X-Request-ID": "my-custom-id"})
        assert resp.headers.get("X-Request-ID") == "my-custom-id"

    def test_request_id_auto_generated(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/v1/admin/models")
        request_id = resp.headers.get("X-Request-ID")
        assert request_id is not None
        assert request_id.startswith("req_")

    def test_active_requests_tracking(self):
        import yunshu_gateway.main as _main
        _main._active_requests = 0

        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        client.get("/health")
        assert _main._active_requests == 0
