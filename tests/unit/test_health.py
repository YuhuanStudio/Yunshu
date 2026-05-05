"""Tests for health check and version info router.

Tests the Phase 5 health router at python/yunshu_gateway/routers/health.py:
- GET /health — aggregate status
- GET /health/ready — readiness probe
- GET /health/live — liveness probe
- GET /version — version and runtime info
"""

import sys

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from yunshu_gateway.routers.health import (
    HealthResponse,
    LiveResponse,
    ReadyCheckDetail,
    ReadyResponse,
    VersionResponse,
    _check_gpu_memory,
    _get_version,
    _is_mlx_available,
    _is_model_loaded,
    router,
)


# ── Helper: create a FastAPI app with the health router ──

def _make_app():
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client():
    app = _make_app()
    yield TestClient(app)


# ── /health ──


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_health_timestamp_is_iso_format(self, client):
        resp = client.get("/health")
        timestamp = resp.json()["timestamp"]
        # Should contain 'T' for ISO 8601
        assert "T" in timestamp

    def test_health_response_model(self, client):
        resp = client.get("/health")
        # Validate it matches the schema
        HealthResponse(**resp.json())


# ── /health/ready ──


class TestReadinessEndpoint:
    def test_readiness_without_engine(self, client):
        """No engine loaded -> not ready."""
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        # model_loaded should be False since no engine is set
        assert "ready" in data
        assert "checks" in data

    def test_readiness_with_engine(self):
        """With a loaded engine -> ready."""
        import yunshu_gateway.main as _main
        from yunshu_engine.engine import Engine, EngineConfig
        from yunshu_gateway.engine import set_engine, get_engine

        # Reset global shutdown state that may be polluted by other tests
        _main._shutting_down = False

        engine = Engine(EngineConfig())
        engine._model = object()
        engine._model_name = "test-model"
        engine._running = True
        set_engine(engine)

        try:
            # Verify the engine is actually set before testing
            assert get_engine() is not None
            assert get_engine().is_loaded

            app = _make_app()
            client = TestClient(app)
            resp = client.get("/health/ready")
            data = resp.json()
            assert data["checks"]["model_loaded"] is True
            assert data["ready"] is True
        finally:
            set_engine(None)
            _main._shutting_down = False

    def test_readiness_response_model(self, client):
        resp = client.get("/health/ready")
        ReadyResponse(**resp.json())

    def test_readiness_checks_fields(self, client):
        resp = client.get("/health/ready")
        checks = resp.json()["checks"]
        assert "model_loaded" in checks
        assert "gpu_memory_ok" in checks
        assert "not_shutting_down" in checks

    def test_readiness_gpu_memory_check(self, client):
        resp = client.get("/health/ready")
        checks = resp.json()["checks"]
        # gpu_memory_ok should be True (MLX not in CI or well below 95%)
        assert checks["gpu_memory_ok"] is True


# ── /health/live ──


class TestLivenessEndpoint:
    def test_liveness_returns_alive(self, client):
        resp = client.get("/health/live")
        assert resp.status_code == 200
        data = resp.json()
        assert data["alive"] is True
        assert "shutting_down" in data

    def test_liveness_response_model(self, client):
        resp = client.get("/health/live")
        LiveResponse(**resp.json())

    def test_liveness_always_ok(self, client):
        """Liveness should always return 200 even when not ready."""
        resp = client.get("/health/live")
        assert resp.status_code == 200


# ── /version ──


class TestVersionEndpoint:
    def test_version_returns_info(self, client):
        resp = client.get("/version")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert "python" in data
        assert "mlx_available" in data
        assert "platform" in data

    def test_version_python_format(self, client):
        resp = client.get("/version")
        python_ver = resp.json()["python"]
        # Should be major.minor.micro
        parts = python_ver.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_version_python_matches_sys(self, client):
        resp = client.get("/version")
        expected = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        assert resp.json()["python"] == expected

    def test_version_response_model(self, client):
        resp = client.get("/version")
        VersionResponse(**resp.json())

    def test_mlx_available_is_boolean(self, client):
        resp = client.get("/version")
        assert isinstance(resp.json()["mlx_available"], bool)


# ── Helper functions ──


class TestHelperFunctions:
    def test_is_model_loaded_no_engine(self):
        assert _is_model_loaded(None, None) is False

    def test_is_model_loaded_with_engine(self):
        from yunshu_engine.engine import Engine, EngineConfig
        engine = Engine(EngineConfig())
        engine._model = object()
        engine._model_name = "test"
        engine._running = True
        assert _is_model_loaded(engine, None) is True

    def test_is_model_loaded_unloaded_engine(self):
        from yunshu_engine.engine import Engine, EngineConfig
        engine = Engine(EngineConfig())
        assert _is_model_loaded(engine, None) is False

    def test_check_gpu_memory_returns_bool(self):
        result = _check_gpu_memory()
        assert isinstance(result, bool)

    def test_is_mlx_available_returns_bool(self):
        result = _is_mlx_available()
        assert isinstance(result, bool)

    def test_get_version_returns_string(self):
        version = _get_version()
        assert isinstance(version, str)
        assert len(version) > 0
