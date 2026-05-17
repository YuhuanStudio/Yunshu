"""Comprehensive end-to-end HTTP tests for Yunshu gateway.

Tests the full HTTP request/response pipeline without real models,
using mocked engine components. Covers:

  - Chat completion error cases (wrong model, no model, bad request)
  - Text completions error cases
  - Model listing and detail
  - Embeddings endpoint
  - Health/version/metrics endpoints
  - Authentication behavior
  - Rate limiting behavior
  - MCP protocol (initialize, tools/list, errors)
  - Batch API (validation)
  - Anthropic compatibility
  - Response format validation

Note: Full generation flow tests (streaming + non-streaming with actual
token output) require a running server with real models. See
scripts/test_e2e_http.py for those tests. Generation-triggering tests
with a valid model name are excluded because the mocked engine's async
step loop does not produce tokens.
"""

import json

import pytest
from fastapi.testclient import TestClient

from yunshu_engine.engine import (
    Engine,
    EngineConfig,
)
from yunshu_gateway import engine as engine_mod
from yunshu_gateway.main import create_app


# ─── Fake Components ───────────────────────────────────────────

class _FakeDetokenizer:
    """Fake streaming detokenizer for per-request use."""

    def __init__(self):
        self._segments = []
        self.last_segment = ""

    def reset(self):
        self._segments = []
        self.last_segment = ""

    def add_token(self, token_id):
        text = f"[tok{token_id}]"
        self._segments.append(text)
        self.last_segment = text

    def finalize(self):
        self.last_segment = "".join(self._segments) if self._segments else ""


class _FakeTokenizer:
    """Minimal tokenizer that encodes text as token IDs."""

    def encode(self, text):
        return list(range(len(text)))

    @property
    def eos_token_id(self):
        return 2

    @property
    def eos_token_ids(self):
        return [2]

    @property
    def detokenizer(self):
        return _FakeDetokenizer()


def _make_engine(model_name="test-model"):
    """Create a fake engine with mocked components."""
    eng = Engine(EngineConfig())
    eng._model = object()
    eng._tokenizer = _FakeTokenizer()
    eng._model_name = model_name
    eng._running = True
    eng._batch_gen = None
    return eng


@pytest.fixture
def _reset_engine():
    """Reset module-level engine state between tests."""
    old = engine_mod._engine
    engine_mod._engine = None
    yield
    engine_mod._engine = old


def _client_with_engine(model_name="test-model"):
    """Create a TestClient with a pre-loaded fake engine."""
    engine_mod._engine = _make_engine(model_name)
    app = create_app()
    return TestClient(app)


def _client_without_engine():
    """Create a TestClient without engine loaded."""
    engine_mod._engine = Engine(EngineConfig())
    app = create_app()
    return TestClient(app)


# ═══════════════════════════════════════════════════════════════
# 1. Chat Completion Error Tests
# ═══════════════════════════════════════════════════════════════

class TestChatCompletion:
    """Test /v1/chat/completions endpoint error cases.

    Note: Successful generation tests require real models.
    See scripts/test_e2e_http.py.
    """

    @pytest.fixture(autouse=True)
    def _reset(self, _reset_engine):
        pass

    def test_chat_wrong_model_404(self):
        """Request for unloaded model returns 404."""
        client = _client_with_engine()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 404

    def test_chat_no_model_404(self):
        """Engine initialized but no model loaded returns 404."""
        client = _client_without_engine()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "any-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code in (404, 503)

    def test_chat_no_engine_503(self):
        """Without engine initialized, returns 404 or 503."""
        app = create_app()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "any-model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert resp.status_code in (404, 503)

    def test_chat_missing_messages_422(self):
        """Request without messages returns validation error (400 or 422)."""
        client = _client_with_engine()
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test-model"},
        )
        assert resp.status_code in (400, 422)

    def test_invalid_json_body_422(self):
        """Invalid JSON body returns 400 or 422."""
        client = _client_with_engine()
        resp = client.post(
            "/v1/chat/completions",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code in (400, 422)


# ═══════════════════════════════════════════════════════════════
# 2. Text Completions Tests
# ═══════════════════════════════════════════════════════════════

class TestCompletions:
    """Test /v1/completions endpoint."""

    @pytest.fixture(autouse=True)
    def _reset(self, _reset_engine):
        pass

    def test_completions_wrong_model(self):
        """Completions with unknown model returns 404."""
        client = _client_with_engine()
        resp = client.post(
            "/v1/completions",
            json={
                "model": "nonexistent",
                "prompt": "Hello",
            },
        )
        assert resp.status_code == 404

    def test_completions_no_model(self):
        """Completions without loaded model returns error."""
        client = _client_without_engine()
        resp = client.post(
            "/v1/completions",
            json={
                "model": "test",
                "prompt": "Hello",
            },
        )
        assert resp.status_code in (404, 503)


# ═══════════════════════════════════════════════════════════════
# 3. Model Management Tests
# ═══════════════════════════════════════════════════════════════

class TestModelManagement:
    """Test model listing and detail endpoints."""

    @pytest.fixture(autouse=True)
    def _reset(self, _reset_engine):
        pass

    def test_models_list(self):
        """Models endpoint lists loaded model."""
        client = _client_with_engine()
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        assert data["data"][0]["id"] == "test-model"

    def test_models_list_empty(self):
        """Models endpoint returns empty list when no engine."""
        app = create_app()
        with TestClient(app) as client:
            resp = client.get("/v1/models")
            assert resp.status_code == 200
            data = resp.json()
            assert data["data"] == []

    def test_model_detail(self):
        """Get model by ID returns model info."""
        client = _client_with_engine()
        resp = client.get("/v1/models/test-model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "test-model"

    def test_model_detail_404(self):
        """Get unknown model by ID returns 404."""
        client = _client_with_engine()
        resp = client.get("/v1/models/nonexistent")
        assert resp.status_code == 404

    def test_models_list_format(self):
        """Models list follows OpenAI format."""
        client = _client_with_engine()
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        model = data["data"][0]
        assert "id" in model
        assert "object" in model
        assert model["object"] == "model"
        assert "created" in model


# ═══════════════════════════════════════════════════════════════
# 4. Embeddings Tests
# ═══════════════════════════════════════════════════════════════

class TestEmbeddings:
    """Test /v1/embeddings endpoint."""

    @pytest.fixture(autouse=True)
    def _reset(self, _reset_engine):
        pass

    def test_embeddings_no_model(self):
        """Embeddings returns 404 without model."""
        app = create_app()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/embeddings",
                json={
                    "model": "nonexistent",
                    "input": "hello",
                },
            )
            assert resp.status_code == 404

    def test_embeddings_missing_input(self):
        """Embeddings returns error without input."""
        client = _client_with_engine()
        resp = client.post(
            "/v1/embeddings",
            json={"model": "test-model"},
        )
        assert resp.status_code in (400, 422)


# ═══════════════════════════════════════════════════════════════
# 5. Health / Version / Metrics Tests
# ═══════════════════════════════════════════════════════════════

class TestHealthVersion:
    """Test health check, version, and metrics endpoints."""

    @pytest.fixture(autouse=True)
    def _reset(self, _reset_engine):
        pass

    def test_health_no_engine(self):
        """Health endpoint works without engine."""
        app = create_app()
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"

    def test_health_with_engine(self):
        """Health endpoint shows engine loaded."""
        client = _client_with_engine()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["engine"]["loaded"] is True

    def test_health_format(self):
        """Health response has expected structure."""
        client = _client_with_engine()
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "engine" in data
        # GPU memory is nested under engine.gpu_memory
        assert "gpu_memory" in data["engine"]

    def test_metrics_endpoint(self):
        """Metrics endpoint returns Prometheus format."""
        app = create_app()
        with TestClient(app) as client:
            resp = client.get("/metrics")
            assert resp.status_code == 200
            assert "yunshu_" in resp.text

    def test_version_endpoint(self):
        """Version endpoint returns version info."""
        app = create_app()
        with TestClient(app) as client:
            resp = client.get("/version")
            assert resp.status_code == 200
            data = resp.json()
            assert "version" in data


# ═══════════════════════════════════════════════════════════════
# 6. Authentication Tests
# ═══════════════════════════════════════════════════════════════

class TestAuthentication:
    """Test API authentication behavior."""

    @pytest.fixture(autouse=True)
    def _reset(self, _reset_engine):
        pass

    def test_no_auth_by_default(self):
        """With auth disabled, health requests work without auth header."""
        client = _client_with_engine()
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_no_auth_required(self):
        """Health endpoint does not require authentication."""
        app = create_app()
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_chat_not_401_without_auth(self):
        """Chat endpoint doesn't return 401 when auth is disabled."""
        client = _client_with_engine()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code != 401


# ═══════════════════════════════════════════════════════════════
# 7. MCP Protocol Tests
# ═══════════════════════════════════════════════════════════════

class TestMCP:
    """Test Model Context Protocol endpoints."""

    @pytest.fixture(autouse=True)
    def _reset(self, _reset_engine):
        pass

    def test_mcp_initialize(self):
        """MCP initialize endpoint returns capabilities."""
        app = create_app()
        with TestClient(app) as client:
            resp = client.post(
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

    def test_mcp_tools_list(self):
        """MCP tools/list returns available tools."""
        app = create_app()
        with TestClient(app) as client:
            resp = client.post(
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

    def test_mcp_unknown_method(self):
        """MCP unknown method returns error."""
        app = create_app()
        with TestClient(app) as client:
            resp = client.post(
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

    def test_mcp_invalid_jsonrpc(self):
        """MCP with wrong jsonrpc version returns error."""
        app = create_app()
        with TestClient(app) as client:
            resp = client.post(
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


# ═══════════════════════════════════════════════════════════════
# 8. Batch API Tests
# ═══════════════════════════════════════════════════════════════

class TestBatchAPI:
    """Test /v1/batch endpoint validation."""

    @pytest.fixture(autouse=True)
    def _reset(self, _reset_engine):
        pass

    def test_batch_empty_400(self):
        """Batch endpoint returns 400 for empty batch."""
        app = create_app()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/batch",
                json={"requests": []},
            )
            assert resp.status_code == 400

    def test_batch_too_large_400(self):
        """Batch endpoint returns 400 for too large batch."""
        app = create_app()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/batch",
                json={
                    "requests": [
                        {"custom_id": f"r{i}", "body": {"model": "test"}}
                        for i in range(501)
                    ]
                },
            )
            assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════════
# 9. Anthropic Compatibility Tests
# ═══════════════════════════════════════════════════════════════

class TestAnthropic:
    """Test Anthropic Messages API compatibility."""

    @pytest.fixture(autouse=True)
    def _reset(self, _reset_engine):
        pass

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

    def test_anthropic_messages_no_model(self):
        """Anthropic messages returns error without loaded model."""
        client = _client_without_engine()
        resp = client.post(
            "/v1/messages",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
            },
        )
        assert resp.status_code in (404, 503)


# ═══════════════════════════════════════════════════════════════
# 10. Rate Limiting Tests
# ═══════════════════════════════════════════════════════════════

class TestRateLimiting:
    """Test rate limiting behavior."""

    @pytest.fixture(autouse=True)
    def _reset(self, _reset_engine):
        pass

    def test_no_rate_limit_by_default(self):
        """Without rate limit configured, many requests succeed."""
        client = _client_with_engine()
        for _ in range(20):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_burst_requests_ok(self):
        """Burst of requests to lightweight endpoints succeeds."""
        app = create_app()
        with TestClient(app) as client:
            for _ in range(50):
                resp = client.get("/health")
                assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════
# 11. Response Format Validation Tests
# ═══════════════════════════════════════════════════════════════

class TestResponseFormat:
    """Validate response format compliance with OpenAI spec."""

    @pytest.fixture(autouse=True)
    def _reset(self, _reset_engine):
        pass

    def test_models_response_format(self):
        """Models list response follows OpenAI format."""
        client = _client_with_engine()
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()

        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        model = data["data"][0]
        assert "id" in model
        assert "object" in model
        assert model["object"] == "model"
        assert "created" in model

    def test_model_detail_response_format(self):
        """Model detail response has required fields."""
        client = _client_with_engine()
        resp = client.get("/v1/models/test-model")
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert "object" in data
        assert data["id"] == "test-model"

    def test_health_response_format(self):
        """Health response has expected structure."""
        client = _client_with_engine()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "engine" in data
        assert "gpu_memory" in data["engine"]

    def test_error_response_format(self):
        """Error responses contain error information."""
        client = _client_with_engine()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data or "detail" in data

    def test_version_response_format(self):
        """Version response has expected fields."""
        app = create_app()
        with TestClient(app) as client:
            resp = client.get("/version")
            assert resp.status_code == 200
            data = resp.json()
            assert "version" in data
