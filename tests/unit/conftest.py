"""Shared fixtures for unit tests."""

import os
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _disable_auth(request):
    """Disable auth for all unit tests by default.

    Auth-related tests (auth, rbac, middleware) manage their own auth env
    and are exempted from the global disable.
    """
    modname = request.module.__name__ if request.module else ""

    is_auth_test = any(kw in modname.lower() for kw in ("auth", "rbac", "middleware"))
    if is_auth_test:
        yield
        return

    old = os.environ.get("YUNSHU_AUTH_DISABLED")
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    yield
    if old is None:
        os.environ.pop("YUNSHU_AUTH_DISABLED", None)
    else:
        os.environ["YUNSHU_AUTH_DISABLED"] = old


@pytest.fixture(autouse=True)
def _no_user_config(tmp_path, monkeypatch):
    """Tests never read the developer's ~/.yunshu/config.toml."""
    from yunshu_engine import settings

    monkeypatch.setattr(
        settings, "user_config_path", lambda: tmp_path / "user-config.toml"
    )


@pytest.fixture(autouse=True)
def _no_m3_idle_probe(monkeypatch):
    """gpuq daemon-loop tests never ssh to the laptop for its idle time."""
    monkeypatch.setenv("GPUQ_M3_IDLE_S", "0")


@pytest.fixture(autouse=True, scope="session")
def _fast_drain():
    """Zero drain timeout for all tests — avoids 30s wait on app teardown."""
    os.environ["YUNSHU_DRAIN_TIMEOUT"] = "0"
    yield
    os.environ.pop("YUNSHU_DRAIN_TIMEOUT", None)


@pytest.fixture
def mock_engine():
    """A minimal mock Engine that simulates a loaded+running state."""
    from yunshu_engine.types import EngineConfig

    engine = MagicMock()
    engine._model_name = "test-model"
    engine._model = MagicMock()
    engine._running = True
    engine._loaded = True
    engine._model_path = "/models/test-model"
    engine.config = EngineConfig()
    yield engine


@pytest.fixture
def mock_batched_engine():
    """A minimal mock BatchedEngine that simulates a loaded+running state."""
    engine = MagicMock()
    engine._model_name = "test-model"
    engine._running = True
    engine._loaded = True
    yield engine


@pytest.fixture
def set_engine(mock_engine):
    """Set a mock engine as the global engine, yield it, then clear.

    Uses the engine module's set_engine/get_engine pattern.
    """
    from yunshu_gateway.engine import set_engine as _set_engine

    _set_engine(mock_engine)
    yield mock_engine
    _set_engine(None)


@pytest.fixture
def client(set_engine):
    """FastAPI TestClient with mock engine and auth disabled."""
    from fastapi.testclient import TestClient

    from yunshu_gateway.main import create_app

    with TestClient(create_app(), raise_server_exceptions=False) as c:
        yield c
