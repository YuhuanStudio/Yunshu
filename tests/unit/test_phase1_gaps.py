"""Tests for Phase 1 gap fills: stream_options, auth middleware, /version."""

import json

from fastapi.testclient import TestClient

# ── stream_options / include_usage ──


class TestStreamOptionsFormat:
    """Tests for format_openai_usage_chunk."""

    def test_usage_chunk_format(self):
        from yunshu_gateway.streaming import format_openai_usage_chunk

        chunk = format_openai_usage_chunk(
            completion_id="chatcmpl-123",
            model="test-model",
            prompt_tokens=10,
            completion_tokens=5,
        )
        assert chunk.startswith("data: ")
        assert chunk.endswith("\n\n")
        data = json.loads(chunk[len("data: "):])
        assert data["id"] == "chatcmpl-123"
        assert data["object"] == "chat.completion.chunk"
        assert data["choices"] == []
        assert data["usage"]["prompt_tokens"] == 10
        assert data["usage"]["completion_tokens"] == 5
        assert data["usage"]["total_tokens"] == 15

    def test_usage_chunk_zero_tokens(self):
        from yunshu_gateway.streaming import format_openai_usage_chunk

        chunk = format_openai_usage_chunk(
            completion_id="chatcmpl-abc",
            model="model-x",
            prompt_tokens=0,
            completion_tokens=0,
        )
        data = json.loads(chunk[len("data: "):])
        assert data["usage"]["total_tokens"] == 0

    def test_usage_chunk_large_tokens(self):
        from yunshu_gateway.streaming import format_openai_usage_chunk

        chunk = format_openai_usage_chunk(
            completion_id="chatcmpl-456",
            model="big-model",
            prompt_tokens=50000,
            completion_tokens=10000,
        )
        data = json.loads(chunk[len("data: "):])
        assert data["usage"]["total_tokens"] == 60000

    def test_usage_chunk_reasoning_not_double_counted(self):
        # completion_tokens already includes reasoning (engine n_tok
        # counts every generated token). reasoning_tokens is the detail SUBSET,
        # not an addend — previously it was added, inflating completion/total
        # for thinking models with include_usage=true.
        from yunshu_gateway.streaming import format_openai_usage_chunk

        chunk = format_openai_usage_chunk(
            completion_id="chatcmpl-r",
            model="thinking-model",
            prompt_tokens=10,
            completion_tokens=100,  # already includes the 30 reasoning tokens
            reasoning_tokens=30,
        )
        data = json.loads(chunk[len("data: "):])
        assert data["usage"]["completion_tokens"] == 100  # NOT 130
        assert data["usage"]["total_tokens"] == 110  # NOT 140
        assert data["usage"]["completion_tokens_details"]["reasoning_tokens"] == 30

    def test_completion_usage_chunk_reasoning_not_double_counted(self):
        from yunshu_gateway.streaming import format_openai_completion_usage_chunk

        chunk = format_openai_completion_usage_chunk(
            completion_id="cmpl-r",
            model="thinking-model",
            prompt_tokens=10,
            completion_tokens=100,
            reasoning_tokens=30,
        )
        data = json.loads(chunk[len("data: "):])
        assert data["usage"]["completion_tokens"] == 100
        assert data["usage"]["total_tokens"] == 110


class TestStreamOptionsChatRequest:
    """Tests for stream_options in ChatCompletionRequest."""

    def test_stream_options_default_none(self):
        from yunshu_gateway.routers.chat import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert req.stream_options is None

    def test_stream_options_include_usage(self):
        from yunshu_gateway.routers.chat import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            stream_options={"include_usage": True},
        )
        assert req.stream_options is not None
        assert req.stream_options.include_usage is True

    def test_stream_options_exclude_usage(self):
        from yunshu_gateway.routers.chat import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            stream_options={"include_usage": False},
        )
        assert req.stream_options is not None
        assert req.stream_options.include_usage is False


class TestStreamOptionsCompletionRequest:
    """Tests for stream_options in CompletionRequest."""

    def test_stream_options_default_none(self):
        from yunshu_gateway.routers.completions import CompletionRequest

        req = CompletionRequest(model="test", prompt="hello")
        assert req.stream_options is None

    def test_stream_options_include_usage(self):
        from yunshu_gateway.routers.completions import CompletionRequest

        req = CompletionRequest(
            model="test",
            prompt="hello",
            stream_options={"include_usage": True},
        )
        assert req.stream_options is not None
        assert req.stream_options.include_usage is True


# ── Auth middleware public paths ──


class TestGatewayTenantAuthMiddleware:
    """Tests for gateway TenantAuthMiddleware public paths."""

    def test_all_health_paths_public(self):
        from yunshu_gateway.middleware.tenant_auth import TenantAuthMiddleware

        assert "/health" in TenantAuthMiddleware.PUBLIC_PATHS
        assert "/health/live" in TenantAuthMiddleware.PUBLIC_PATHS
        assert "/health/ready" in TenantAuthMiddleware.PUBLIC_PATHS
        assert "/version" in TenantAuthMiddleware.PUBLIC_PATHS


# ── /version endpoint ──


class TestVersionEndpoint:
    """Tests for the /version endpoint."""

    def test_version_endpoint(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.get("/version")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert data["service"] == "yunshu"
        assert isinstance(data["version"], str)


# ── Verify existing tests still pass with stream_options addition ──


class TestStreamFormattersStillWork:
    """Regression tests to verify existing streaming formatters still work."""

    def test_openai_chunk_unchanged(self):
        from yunshu_gateway.streaming import format_openai_chunk

        chunk = format_openai_chunk(
            completion_id="chatcmpl-1",
            model="test",
            delta_content="Hello",
        )
        data = json.loads(chunk[len("data: "):])
        assert data["choices"][0]["delta"]["content"] == "Hello"

    def test_openai_done_unchanged(self):
        from yunshu_gateway.streaming import format_openai_done

        assert format_openai_done() == "data: [DONE]\n\n"

    def test_non_stream_unchanged(self):
        from yunshu_gateway.streaming import format_openai_non_stream

        resp = format_openai_non_stream(
            completion_id="chatcmpl-1",
            model="test",
            content="Hello",
            prompt_tokens=5,
            completion_tokens=1,
        )
        assert resp["choices"][0]["message"]["content"] == "Hello"
        assert resp["usage"]["total_tokens"] == 6
