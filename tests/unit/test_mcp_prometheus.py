"""Tests for MCP protocol and Prometheus metrics."""

import pytest
from fastapi.testclient import TestClient

from yunshu_engine.engine import Engine, EngineConfig


class TestMCPProtocol:
    """Test MCP JSON-RPC protocol endpoints."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from yunshu_gateway.engine import set_engine

        self._engine = Engine(EngineConfig())
        self._engine._model = object()
        self._engine._model_name = "test-model"
        self._engine._running = True
        set_engine(self._engine)

    def test_initialize(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

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
        assert "tools" in data["result"]["capabilities"]

    def test_tools_list(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.post(
            "/v1/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 2,
            },
        )
        assert resp.status_code == 200
        tools = resp.json()["result"]["tools"]
        names = {t["name"] for t in tools}
        assert "generate" in names
        assert "synthesize_speech" in names
        assert "generate_image" in names

    def test_tools_discovery_rest(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.get("/v1/mcp/tools")
        assert resp.status_code == 200
        tools = resp.json()["tools"]
        assert len(tools) >= 3

    def test_ping(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.post(
            "/v1/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "ping",
                "id": 3,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["result"] == {}

    def test_method_not_found(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.post(
            "/v1/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "nonexistent",
                "id": 4,
            },
        )
        assert resp.status_code == 200
        assert "error" in resp.json()
        assert resp.json()["error"]["code"] == -32601

    def test_invalid_jsonrpc_version(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.post(
            "/v1/mcp",
            json={
                "jsonrpc": "1.0",
                "method": "ping",
                "id": 5,
            },
        )
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_prompts_list(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.post(
            "/v1/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "prompts/list",
                "id": 6,
            },
        )
        assert resp.status_code == 200
        prompts = resp.json()["result"]["prompts"]
        assert len(prompts) >= 2
        names = {p["name"] for p in prompts}
        assert "summarize" in names
        assert "translate" in names

    def test_prompts_get(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.post(
            "/v1/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "prompts/get",
                "params": {
                    "name": "summarize",
                    "arguments": {"text": "Hello world"},
                },
                "id": 7,
            },
        )
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert "messages" in result
        assert "Hello world" in result["messages"][0]["content"]["text"]

    def test_prompts_get_unknown(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.post(
            "/v1/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "prompts/get",
                "params": {"name": "nonexistent"},
                "id": 8,
            },
        )
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_resources_list(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.post(
            "/v1/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "resources/list",
                "id": 9,
            },
        )
        assert resp.status_code == 200

    def test_tools_call_unknown_tool(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.post(
            "/v1/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "unknown_tool", "arguments": {}},
                "id": 10,
            },
        )
        assert resp.status_code == 200
        assert "error" in resp.json()


class TestPrometheusMetrics:
    """Test Prometheus /metrics endpoint."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from yunshu_gateway.engine import set_engine

        self._engine = Engine(EngineConfig())
        self._engine._model = object()
        self._engine._model_name = "test-model"
        self._engine._running = True
        set_engine(self._engine)

    def test_metrics_endpoint(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "yunshu_uptime_seconds" in text
        assert "yunshu_request_count" in text
        assert "yunshu_tokens_total" in text

    def test_metrics_after_requests(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        # Make some requests to generate metrics
        client.get("/health")
        client.get("/health")

        resp = client.get("/metrics")
        text = resp.text
        assert 'method="GET"' in text
        assert 'endpoint="/health"' in text
        assert 'status="200"' in text
        assert "yunshu_request_latency_seconds" in text

    def test_metrics_record_tokens(self):
        # Use a fresh _Metrics() (like the sibling tests) instead of the shared
        # get_metrics() singleton, which accumulates token counts across the suite
        # and made this assertion order-dependent.
        from yunshu_gateway.middleware.metrics import _Metrics

        metrics = _Metrics()
        metrics.record_tokens(prompt=100, completion=50)

        text = metrics.to_prometheus()
        assert 'yunshu_tokens_total{type="prompt"} 100' in text
        assert 'yunshu_tokens_total{type="completion"} 50' in text

    def test_metrics_record_inference(self):
        from yunshu_gateway.middleware.metrics import _Metrics

        metrics = _Metrics()
        metrics.record_inference()
        metrics.record_inference()

        text = metrics.to_prometheus()
        assert "yunshu_inference_count 2" in text

    def test_metrics_gpu_memory(self):
        from yunshu_gateway.middleware.metrics import get_metrics

        metrics = get_metrics()
        text = metrics.to_prometheus()
        assert "yunshu_gpu_memory_bytes" in text

    def test_metrics_p99_latency(self):
        from yunshu_gateway.middleware.metrics import get_metrics

        metrics = get_metrics()
        metrics.record_request("/v1/chat/completions", "POST", 200, 0.1)
        metrics.record_request("/v1/chat/completions", "POST", 200, 0.5)

        text = metrics.to_prometheus()
        assert 'quantile="0.5"' in text
        assert 'quantile="0.99"' in text

    def test_metrics_content_type(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.get("/metrics")
        assert "text/plain" in resp.headers.get("content-type", "")
