"""Shared fixtures for unit tests."""

import os as _os
import sys as _sys
import tempfile as _tempfile

# The dev tools (yv, agentbench, the agentic runner) default to this machine's build volume and main
# checkout venv. Unit tests must never write there, and CI has neither: point them at a throwaway root
# and the running interpreter before any test module imports them.
_DEV_ROOT = _tempfile.mkdtemp(prefix="yunshu-unit-devroot-")
_os.environ["YV_PY"] = _sys.executable
_os.environ["YV_ROOT"] = _os.path.join(_DEV_ROOT, "verify")
_os.environ["AGENTIC_BUILD"] = _os.path.join(_DEV_ROOT, "agentic")
_os.environ["AGENTBENCH_ROOT"] = _os.path.join(_DEV_ROOT, "agentbench")
# Stores, keys, caches and config default to ~/.yunshu (Path.home()). A unit test must never write
# into the developer's real home (scripted chat completions piled up in ~/.yunshu/chat_completions),
# so every test runs with HOME pointed at the throwaway root, removed when the session ends.
_os.makedirs(_os.path.join(_DEV_ROOT, "home"), exist_ok=True)
_os.environ["HOME"] = _os.path.join(_DEV_ROOT, "home")
import atexit as _atexit
import shutil as _shutil

_atexit.register(_shutil.rmtree, _DEV_ROOT, True)

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


@pytest.fixture(autouse=True)
def _mcp_connector_loopback(monkeypatch):
    """The fake MCP servers in the tests listen on 127.0.0.1; the product default blocks that."""
    monkeypatch.setenv("YUNSHU_MCP_CONNECTOR_ALLOW_PRIVATE", "1")


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


@pytest.fixture(autouse=True)
def _isolate_native_tool_context():
    """Direct router helpers must not leave request context for the next unit test."""
    from yunshu_engine.batched_engine import _REQUEST_TOOL_USE, _REQUEST_TOOLS

    tools = _REQUEST_TOOLS.set(None)
    choice = _REQUEST_TOOL_USE.set(None)
    yield
    _REQUEST_TOOLS.reset(tools)
    _REQUEST_TOOL_USE.reset(choice)


@pytest.fixture(autouse=True)
def _restore_mlx_default_device():
    """A test that moves MLX to the CPU must not leave it there: a later test's Metal kernels would then run
    on the CPU (test_lane_linear failed only in suite order this way)."""
    mx = _sys.modules.get("mlx.core")
    previous = mx.default_device() if mx is not None else None
    yield
    mx = _sys.modules.get("mlx.core")
    if mx is None:
        return
    if previous is None:
        # mlx was first imported during this test; the process default before it was the GPU when there is one
        previous = mx.gpu if mx.metal.is_available() else mx.cpu
    if mx.default_device() != previous:
        mx.set_default_device(previous)
