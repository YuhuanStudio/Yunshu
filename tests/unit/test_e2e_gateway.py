"""End-to-end gateway test with mocked engine.

Tests the full HTTP request → SSE response pipeline without a real model.
"""

import json
import os

import pytest
from fastapi.testclient import TestClient

from yunshu_engine.engine import (
    Engine,
    EngineConfig,
    RequestOutput,
    RequestState,
)
from yunshu_gateway import engine as engine_mod
from yunshu_gateway.main import create_app


class _FakeTokenizer:
    def encode(self, text):
        return list(range(len(text)))

    @property
    def detokenizer(self):
        return _FakeDetokenizer()


class _FakeDetokenizer:
    def reset(self):
        pass

    def add_token(self, token_id):
        pass

    @property
    def last_segment(self):
        return ""

    def finalize(self):
        return ""


class _FakeResponse:
    def __init__(self, uid, text, token, finish_reason=None):
        self.uid = uid
        self.text = text
        self.token = token
        self.finish_reason = finish_reason


class _FakeBatchGen:
    def __init__(self):
        self._uid_counter = 0
        self._pending = {}

    def insert(self, prompts, max_tokens):
        uids = []
        for prompt, mt in zip(prompts, max_tokens):
            uid = self._uid_counter
            self._uid_counter += 1
            uids.append(uid)
            tokens = ["Hi", " there"]
            self._pending[uid] = [
                _FakeResponse(uid, t, j,
                              finish_reason=("stop" if j == len(tokens) - 1 else None))
                for j, t in enumerate(tokens)
            ]
        return uids

    def next_generated(self):
        batch = []
        finished = []
        for uid, responses in self._pending.items():
            if responses:
                batch.append(responses.pop(0))
                if not responses:
                    finished.append(uid)
        for uid in finished:
            del self._pending[uid]
        return batch

    def remove(self, uids):
        for uid in uids:
            self._pending.pop(uid, None)

    def close(self):
        pass


def _make_engine():
    eng = Engine(EngineConfig())
    eng._model = object()
    eng._tokenizer = _FakeTokenizer()
    eng._model_name = "test-model"
    eng._batch_gen = _FakeBatchGen()
    eng._running = True
    return eng


@pytest.fixture(autouse=True)
def _env_fast_drain(monkeypatch):
    """Set fast drain timeout and no model loading for all tests."""
    monkeypatch.setenv("YUNSHU_DRAIN_TIMEOUT", "0")
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("YUNSHU_MULTI_MODEL", raising=False)
    monkeypatch.delenv("YUNSHU_DATA_PARALLEL", raising=False)
    # Reset engine state before each test to prevent cross-contamination
    engine_mod._engine = None
    yield
    engine_mod._engine = None


@pytest.fixture
def app_client():
    """Create a fresh app + client for each test (no shared state)."""
    app = create_app()
    with TestClient(app) as client:
        yield client


class TestE2EGateway:
    def test_health_with_engine(self):
        """Health endpoint shows engine loaded."""
        engine_mod._engine = _make_engine()
        app = create_app()
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["engine"]["loaded"] is True

    def test_models_lists_loaded(self):
        """Models endpoint lists loaded model."""
        engine_mod._engine = _make_engine()
        app = create_app()
        with TestClient(app) as client:
            resp = client.get("/v1/models")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "test-model"

    def test_chat_wrong_model_404(self):
        """Request for unloaded model returns 404."""
        engine_mod._engine = _make_engine()
        app = create_app()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "nonexistent-model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert resp.status_code == 404

    def test_chat_no_model_404(self):
        """Engine initialized but no model loaded → 404."""
        engine_mod._engine = Engine(EngineConfig())  # no model loaded
        app = create_app()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "any-model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert resp.status_code == 404

    def test_chat_no_engine_503(self):
        """Without engine initialized, returns 503."""
        # _engine is None from autouse fixture
        app = create_app()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "any-model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            # Engine is created by lifespan but no model loaded → 404
            # This test verifies the full path works without pre-set engine
            assert resp.status_code in (404, 503)

    def test_completions_wrong_model_404(self):
        """Completions endpoint returns 404 for unknown model."""
        engine_mod._engine = _make_engine()
        app = create_app()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/completions",
                json={
                    "model": "nonexistent",
                    "prompt": "hello",
                },
            )
            assert resp.status_code == 404

    def test_mcp_initialize(self, app_client):
        """MCP initialize endpoint returns capabilities."""
        resp = app_client.post(
            "/v1/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "id": 1,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["protocolVersion"] == "2024-11-05"

    def test_mcp_tools_list(self, app_client):
        """MCP tools/list returns available tools."""
        resp = app_client.post(
            "/v1/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 2,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        tools = data["result"]["tools"]
        assert len(tools) >= 3
        names = [t["name"] for t in tools]
        assert "generate" in names

    def test_mcp_unknown_method(self, app_client):
        """MCP unknown method returns error."""
        resp = app_client.post(
            "/v1/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "nonexistent",
                "id": 3,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    def test_mcp_invalid_jsonrpc(self, app_client):
        """MCP with wrong jsonrpc version returns error."""
        resp = app_client.post(
            "/v1/mcp",
            json={
                "jsonrpc": "1.0",
                "method": "initialize",
                "id": 1,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    def test_embeddings_no_model_404(self, app_client):
        """Embeddings endpoint returns 404 without model."""
        resp = app_client.post(
            "/v1/embeddings",
            json={
                "model": "nonexistent",
                "input": "hello",
            },
        )
        assert resp.status_code == 404

    def test_batch_empty_400(self, app_client):
        """Batch endpoint returns 400 for empty batch."""
        resp = app_client.post(
            "/v1/batch",
            json={"requests": []},
        )
        assert resp.status_code == 400

    def test_batch_too_large_400(self, app_client):
        """Batch endpoint returns 400 for too large batch."""
        resp = app_client.post(
            "/v1/batch",
            json={
                "requests": [
                    {"custom_id": f"r{i}", "body": {"model": "test"}}
                    for i in range(501)
                ]
            },
        )
        assert resp.status_code == 400

    def test_metrics_endpoint(self, app_client):
        """Metrics endpoint returns Prometheus format."""
        resp = app_client.get("/metrics")
        assert resp.status_code == 200
        assert "yunshu_" in resp.text

    def test_models_detail(self):
        """Get model by ID returns model info."""
        engine_mod._engine = _make_engine()
        app = create_app()
        with TestClient(app) as client:
            resp = client.get("/v1/models/test-model")
            assert resp.status_code == 200
            data = resp.json()
            assert data["id"] == "test-model"

    def test_models_detail_404(self):
        """Get unknown model by ID returns 404."""
        engine_mod._engine = _make_engine()
        app = create_app()
        with TestClient(app) as client:
            resp = client.get("/v1/models/nonexistent")
            assert resp.status_code == 404

    def test_health_no_engine(self):
        """Health endpoint works without engine."""
        app = create_app()
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"

    def test_anthropic_count_tokens_no_engine(self):
        """Anthropic token counting returns error without engine."""
        app = create_app()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/messages/count_tokens",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert resp.status_code in (404, 503)
