"""Tests for Phase 1 gap fills: stream_options, auth middleware, request queue, /version."""

import json
import os
import time

import pytest
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


class TestControlPlaneAuthMiddleware:
    """Tests for yunshu_api AuthMiddleware."""

    def test_health_no_auth_needed(self):
        from yunshu_api.middleware.auth import AuthMiddleware
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        async def health(request):
            return JSONResponse({"status": "ok"})

        async def secure(request):
            return JSONResponse({"secret": "data"})

        app = Starlette(
            routes=[
                Route("/health", health),
                Route("/health/live", health),
                Route("/health/ready", health),
                Route("/version", health),
                Route("/secure", secure),
            ],
        )
        app.add_middleware(AuthMiddleware, token="test-secret")
        client = TestClient(app)

        # Public paths should not require auth
        assert client.get("/health").status_code == 200
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        assert client.get("/version").status_code == 200

        # Secure paths should require auth
        assert client.get("/secure").status_code == 401

        # With valid auth
        assert client.get(
            "/secure", headers={"Authorization": "Bearer test-secret"}
        ).status_code == 200

        # With wrong auth
        assert client.get(
            "/secure", headers={"Authorization": "Bearer wrong"}
        ).status_code == 401

    def test_docs_and_openapi_no_auth(self):
        from yunshu_api.middleware.auth import AuthMiddleware, PUBLIC_PATHS

        assert "/docs" in PUBLIC_PATHS
        assert "/openapi.json" in PUBLIC_PATHS
        assert "/redoc" in PUBLIC_PATHS


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


# ── Request Queue Management ──


class TestRequestQueueManager:
    """Tests for yunshu_control.request_queue."""

    def test_on_request_arrival(self):
        from yunshu_control.request_queue import RequestQueueManager, QueueAction

        mgr = RequestQueueManager(max_queue_size=10)
        action = mgr.on_request_arrival(
            request_id="req-1",
            tenant_id="tenant-a",
            model_id="model-x",
        )
        assert action == QueueAction.ACCEPT

    def test_queue_full_rejects(self):
        from yunshu_control.request_queue import RequestQueueManager, QueueAction

        mgr = RequestQueueManager(max_queue_size=2)
        mgr.on_request_arrival(request_id="req-1")
        mgr.on_request_arrival(request_id="req-2")
        action = mgr.on_request_arrival(request_id="req-3")
        assert action == QueueAction.REJECT_QUEUE_FULL

    def test_on_request_scheduled(self):
        from yunshu_control.request_queue import RequestQueueManager

        mgr = RequestQueueManager()
        mgr.on_request_arrival(request_id="req-1")
        mgr.on_request_scheduled(request_id="req-1")

        entry = mgr._entries["req-1"]
        assert entry.scheduled_time is not None

    def test_on_request_completed(self):
        from yunshu_control.request_queue import RequestQueueManager

        mgr = RequestQueueManager()
        mgr.on_request_arrival(request_id="req-1")
        mgr.on_request_scheduled(request_id="req-1")
        mgr.on_request_completed(
            request_id="req-1",
            prompt_tokens=10,
            completion_tokens=5,
        )

        entry = mgr._entries["req-1"]
        assert entry.completion_time is not None
        assert entry.prompt_tokens == 10
        assert entry.completion_tokens == 5

    def test_on_request_cancelled(self):
        from yunshu_control.request_queue import RequestQueueManager

        mgr = RequestQueueManager()
        mgr.on_request_arrival(request_id="req-1")
        mgr.on_request_cancelled(request_id="req-1")

        assert "req-1" not in mgr._entries

    def test_get_queue_stats(self):
        from yunshu_control.request_queue import RequestQueueManager

        mgr = RequestQueueManager()
        mgr.on_request_arrival(request_id="req-1")
        mgr.on_request_arrival(request_id="req-2")
        mgr.on_request_arrival(request_id="req-cancel")
        mgr.on_request_scheduled(request_id="req-1")
        mgr.on_request_completed(request_id="req-1", prompt_tokens=10, completion_tokens=5)
        mgr.on_request_cancelled(request_id="req-cancel")
        mgr._rejected_count = 3

        stats = mgr.get_queue_stats()
        assert stats.waiting == 1  # req-2 is still waiting (not scheduled)
        assert stats.running == 0  # req-1 completed, req-2 not scheduled
        assert stats.completed == 1
        # cancelled entry is removed from _entries but tracked via _cancelled_count
        assert stats.cancelled == 1
        assert stats.total_prompt_tokens == 10
        assert stats.total_completion_tokens == 5

    def test_get_pending_requests(self):
        from yunshu_control.request_queue import RequestQueueManager

        mgr = RequestQueueManager()
        mgr.on_request_arrival(request_id="req-1", priority=5)
        mgr.on_request_arrival(request_id="req-2", priority=10)
        mgr.on_request_arrival(request_id="req-3", priority=0)

        pending = mgr.get_pending_requests()
        assert len(pending) == 3
        # Should be sorted by priority descending
        assert pending[0]["priority"] == 10
        assert pending[1]["priority"] == 5
        assert pending[2]["priority"] == 0

    def test_get_active_requests(self):
        from yunshu_control.request_queue import RequestQueueManager

        mgr = RequestQueueManager()
        mgr.on_request_arrival(request_id="req-1")
        mgr.on_request_scheduled(request_id="req-1")
        mgr.on_request_arrival(request_id="req-2")

        active = mgr.get_active_requests()
        assert len(active) == 1
        assert active[0]["request_id"] == "req-1"

    def test_priority_override(self):
        from yunshu_control.request_queue import RequestQueueManager, QueueAction, QueuePriority

        mgr = RequestQueueManager()
        mgr.set_priority_override("req-1", QueuePriority.CRITICAL.value)
        action = mgr.on_request_arrival(request_id="req-1")
        assert action == QueueAction.ACCEPT
        entry = mgr._entries["req-1"]
        assert entry.priority == QueuePriority.CRITICAL.value

    def test_get_scheduling_priority(self):
        from yunshu_control.request_queue import RequestQueueManager

        mgr = RequestQueueManager()
        # No override set
        assert mgr.get_scheduling_priority("req-x", default=5) == 5
        # Set override
        mgr.set_priority_override("req-y", 10)
        assert mgr.get_scheduling_priority("req-y", default=5) == 10
        # Override is consumed
        assert mgr.get_scheduling_priority("req-y", default=5) == 5

    def test_is_queue_full(self):
        from yunshu_control.request_queue import RequestQueueManager

        mgr = RequestQueueManager(max_queue_size=2)
        assert not mgr.is_queue_full()
        mgr.on_request_arrival(request_id="req-1")
        assert not mgr.is_queue_full()
        mgr.on_request_arrival(request_id="req-2")
        assert mgr.is_queue_full()

    def test_clear_completed(self):
        from yunshu_control.request_queue import RequestQueueManager

        mgr = RequestQueueManager()
        mgr.on_request_arrival(request_id="req-1")
        mgr.on_request_scheduled(request_id="req-1")
        mgr.on_request_completed(request_id="req-1")
        mgr.on_request_arrival(request_id="req-2")

        removed = mgr.clear_completed()
        assert removed == 1
        assert "req-1" not in mgr._entries
        assert "req-2" in mgr._entries

    def test_queue_entry_status(self):
        from yunshu_control.request_queue import QueueEntry

        entry = QueueEntry(request_id="test")
        assert entry.status == "waiting"

        entry.scheduled_time = time.monotonic()
        assert entry.status == "running"

        entry.completion_time = time.monotonic()
        assert entry.status == "completed"

    def test_queue_entry_wait_time(self):
        from yunshu_control.request_queue import QueueEntry

        entry = QueueEntry(
            request_id="test",
            arrival_time=time.monotonic() - 1.0,  # arrived 1s ago
        )
        assert entry.wait_time_ms >= 900  # approximately 1000ms

    def test_queue_entry_total_time(self):
        from yunshu_control.request_queue import QueueEntry

        now = time.monotonic()
        entry = QueueEntry(
            request_id="test",
            arrival_time=now - 2.0,
            completion_time=now,
        )
        assert entry.total_time_ms >= 1900

    def test_get_request_queue_manager_singleton(self):
        from yunshu_control.request_queue import (
            get_request_queue_manager,
            set_request_queue_manager,
            RequestQueueManager,
        )

        mgr = RequestQueueManager()
        set_request_queue_manager(mgr)
        assert get_request_queue_manager() is mgr

        # Reset
        set_request_queue_manager(None)
        new_mgr = get_request_queue_manager()
        assert isinstance(new_mgr, RequestQueueManager)

    def test_priority_constants(self):
        from yunshu_control.request_queue import QueuePriority

        assert QueuePriority.LOW.value == 0
        assert QueuePriority.NORMAL.value == 5
        assert QueuePriority.HIGH.value == 10
        assert QueuePriority.CRITICAL.value == 15


class TestRequestQueueIntegration:
    """Integration test: request lifecycle through queue."""

    def test_full_lifecycle(self):
        from yunshu_control.request_queue import RequestQueueManager, QueueAction

        mgr = RequestQueueManager(max_queue_size=100)

        # Arrival
        action = mgr.on_request_arrival(
            request_id="req-lifecycle",
            tenant_id="tenant-1",
            model_id="llama-3b",
            priority=10,
            slo_class="premium",
        )
        assert action == QueueAction.ACCEPT

        # Check pending
        pending = mgr.get_pending_requests()
        assert len(pending) == 1
        assert pending[0]["request_id"] == "req-lifecycle"
        assert pending[0]["priority"] == 10

        # Scheduled
        mgr.on_request_scheduled("req-lifecycle")
        active = mgr.get_active_requests()
        assert len(active) == 1

        # Completed
        mgr.on_request_completed(
            "req-lifecycle",
            prompt_tokens=100,
            completion_tokens=50,
        )

        stats = mgr.get_queue_stats()
        assert stats.completed == 1
        assert stats.total_prompt_tokens == 100
        assert stats.total_completion_tokens == 50

        # Cleanup
        removed = mgr.clear_completed()
        assert removed == 1


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
