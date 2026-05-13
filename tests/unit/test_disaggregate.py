"""Tests for disaggregated serving endpoints."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from python.yunshu_gateway.routers.disaggregate import (
    PrefillRequest,
    PrefillResponse,
    DecodeRequest,
    DecodeResponse,
    _cache_handles,
    router,
)


class TestPrefillRequest:
    def test_defaults(self):
        req = PrefillRequest()
        assert req.model == ""
        assert req.prompt == ""
        assert req.max_prefill_tokens is None
        assert req.temperature == 0.0

    def test_with_params(self):
        req = PrefillRequest(
            model="test-model",
            prompt="Hello world",
            max_prefill_tokens=1024,
            temperature=0.5,
        )
        assert req.model == "test-model"
        assert req.max_prefill_tokens == 1024


class TestDecodeRequest:
    def test_defaults(self):
        req = DecodeRequest()
        assert req.cache_handle == ""
        assert req.max_tokens == 256
        assert req.stream is False

    def test_with_handle(self):
        req = DecodeRequest(cache_handle="pf-abc123", max_tokens=512)
        assert req.cache_handle == "pf-abc123"
        assert req.max_tokens == 512


class TestCacheHandles:
    def setup_method(self):
        _cache_handles.clear()

    def test_store_and_retrieve(self):
        _cache_handles["test-1"] = {
            "cache": None,
            "token_ids": [1, 2, 3],
            "prompt_tokens": 3,
            "cached_tokens": 0,
            "created_at": 0.0,
        }
        assert "test-1" in _cache_handles
        assert _cache_handles["test-1"]["prompt_tokens"] == 3

    def test_delete(self):
        _cache_handles["test-2"] = {"cache": None, "token_ids": [], "prompt_tokens": 0, "cached_tokens": 0, "created_at": 0.0}
        del _cache_handles["test-2"]
        assert "test-2" not in _cache_handles

    def test_cleanup_after_use(self):
        _cache_handles["test-3"] = {"cache": None, "token_ids": [], "prompt_tokens": 0, "cached_tokens": 0, "created_at": 0.0}
        _cache_handles.pop("test-3", None)
        assert "test-3" not in _cache_handles


class TestPrefillResponse:
    def test_response_model(self):
        resp = PrefillResponse(
            id="pf-test",
            cache_handle="pf-test",
            prompt_tokens=100,
            cached_tokens=50,
            duration_s=0.5,
        )
        assert resp.cache_handle == "pf-test"
        assert resp.prompt_tokens == 100
        assert resp.cached_tokens == 50


class TestDecodeResponse:
    def test_response_model(self):
        resp = DecodeResponse(
            id="dec-test",
            text="Hello world",
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
        )
        assert resp.text == "Hello world"
        assert resp.finish_reason == "stop"


class TestRouterDefinition:
    def test_router_prefix(self):
        assert router.prefix == "/v1"

    def test_router_tags(self):
        assert "disaggregate" in router.tags

    def test_has_routes(self):
        route_paths = [r.path for r in router.routes]
        assert "/v1/prefill" in route_paths
        assert "/v1/decode" in route_paths
        assert "/v1/cache-handles" in route_paths

    def test_prefill_is_post(self):
        for r in router.routes:
            if r.path == "/prefill":
                assert "POST" in r.methods
                break

    def test_decode_is_post(self):
        for r in router.routes:
            if r.path == "/decode":
                assert "POST" in r.methods
                break

    def test_cache_handles_list_is_get(self):
        for r in router.routes:
            if r.path == "/cache-handles":
                assert "GET" in r.methods
                break

    def test_cache_handles_delete(self):
        for r in router.routes:
            if r.path == "/cache-handles/{handle_id}":
                assert "DELETE" in r.methods
                break
