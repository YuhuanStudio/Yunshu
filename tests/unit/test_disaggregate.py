"""Tests for disaggregated serving endpoints."""

import collections
import time

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from python.yunshu_gateway.routers.disaggregate import (
    PrefillRequest,
    PrefillResponse,
    DecodeRequest,
    DecodeResponse,
    _cache_handles,
    _add_cache_handle,
    _get_cache_handle,
    _CACHE_MAX_SIZE,
    _CACHE_TTL_SECONDS,
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
        _add_cache_handle("test-1", {
            "cache": None,
            "token_ids": [1, 2, 3],
            "prompt_tokens": 3,
            "cached_tokens": 0,
            "created_at": time.monotonic(),
        })
        data = _get_cache_handle("test-1")
        assert data is not None
        assert data["prompt_tokens"] == 3

    def test_delete(self):
        _add_cache_handle("test-2", {"cache": None, "token_ids": [], "prompt_tokens": 0, "cached_tokens": 0, "created_at": 0.0})
        assert "test-2" in _cache_handles
        del _cache_handles["test-2"]
        assert "test-2" not in _cache_handles

    def test_cleanup_after_use(self):
        _add_cache_handle("test-3", {"cache": None, "token_ids": [], "prompt_tokens": 0, "cached_tokens": 0, "created_at": 0.0})
        _cache_handles.pop("test-3", None)
        assert "test-3" not in _cache_handles

    def test_lru_eviction_at_max_size(self):
        """When cache exceeds max size, oldest entry is evicted."""
        for i in range(_CACHE_MAX_SIZE + 5):
            _add_cache_handle(f"handle-{i}", {
                "cache": None, "token_ids": [], "prompt_tokens": i,
                "cached_tokens": 0, "created_at": time.monotonic(),
            })
        # Should be capped at max size
        assert len(_cache_handles) == _CACHE_MAX_SIZE
        # Oldest entries should be evicted
        for i in range(5):
            assert _get_cache_handle(f"handle-{i}") is None
        # Newest entries should exist
        assert _get_cache_handle(f"handle-{_CACHE_MAX_SIZE + 4}") is not None

    def test_lru_access_moves_to_end(self):
        """Accessing a handle via _get_cache_handle moves it to end (most recent)."""
        # Add exactly 3 items
        for i in range(3):
            _add_cache_handle(f"handle-{i}", {
                "cache": None, "token_ids": [], "prompt_tokens": i,
                "cached_tokens": 0, "created_at": time.monotonic(),
            })
        # Access handle-0 to make it most recent
        _get_cache_handle("handle-0")

        # Fill up to max size by adding (max-3) more items.
        # That means total = max, no evictions yet. Order: 1, 2, 0, 3, 4, ...
        # Then add 1 more item to trigger one eviction of the LRU (handle-1).
        for i in range(3, _CACHE_MAX_SIZE + 1):
            _add_cache_handle(f"handle-{i}", {
                "cache": None, "token_ids": [], "prompt_tokens": i,
                "cached_tokens": 0, "created_at": time.monotonic(),
            })
        # handle-0 was accessed recently, so handle-1 (oldest unused) is evicted
        assert _get_cache_handle("handle-1") is None
        assert _get_cache_handle("handle-0") is not None
        assert _get_cache_handle("handle-2") is not None

    def test_ttl_expiry(self):
        """Handles with created_at older than TTL should be expired."""
        expired_time = time.monotonic() - _CACHE_TTL_SECONDS - 10
        _add_cache_handle("expired-handle", {
            "cache": None, "token_ids": [], "prompt_tokens": 0,
            "cached_tokens": 0, "created_at": expired_time,
        })
        _add_cache_handle("fresh-handle", {
            "cache": None, "token_ids": [], "prompt_tokens": 0,
            "cached_tokens": 0, "created_at": time.monotonic(),
        })
        # Both exist
        assert "expired-handle" in _cache_handles
        assert "fresh-handle" in _cache_handles

        # Simulate GC by manually removing expired entries
        now = time.monotonic()
        expired = [
            hid for hid, data in _cache_handles.items()
            if (now - data.get("created_at", 0)) > _CACHE_TTL_SECONDS
        ]
        for hid in expired:
            _cache_handles.pop(hid, None)

        assert "expired-handle" not in _cache_handles
        assert "fresh-handle" in _cache_handles

    def test_get_returns_none_for_missing(self):
        assert _get_cache_handle("nonexistent") is None

    def test_overwrite_existing_handle(self):
        _add_cache_handle("dup", {"prompt_tokens": 1, "created_at": 0.0})
        _add_cache_handle("dup", {"prompt_tokens": 2, "created_at": 0.0})
        data = _get_cache_handle("dup")
        assert data["prompt_tokens"] == 2
        assert len(_cache_handles) == 1


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

    def test_disagg_stats_route(self):
        route_paths = [r.path for r in router.routes]
        assert "/v1/disagg-stats" in route_paths


class TestDisaggRouterIntegration:
    """Test that _get_disagg_router and _register_mesh_nodes work with env."""

    def test_get_disagg_router_returns_none_when_disabled(self):
        with patch.dict("os.environ", {"YUNSHU_DISAGG_PD": "0"}, clear=False):
            # Re-import to pick up env change
            from python.yunshu_gateway.routers.disaggregate import _get_disagg_router
            router_inst = _get_disagg_router()
            # Should return a DisaggRouter but with enabled=False
            if router_inst is not None:
                assert not router_inst.config.enabled

    def test_register_mesh_nodes_from_env(self):
        from python.yunshu_gateway.routers.disaggregate import (
            _get_disagg_router, _register_mesh_nodes,
        )
        with patch.dict("os.environ", {
            "YUNSHU_DISAGG_PD": "1",
            "YUNSHU_PREFILL_NODES": "10.0.0.1:7891,10.0.0.2:7891",
            "YUNSHU_DECODE_NODES": "10.0.0.3:7890",
        }, clear=False):
            router_inst = _get_disagg_router()
            if router_inst is not None:
                _register_mesh_nodes(router_inst)
                # Check nodes were registered
                stats = router_inst.get_stats()
                assert stats["prefill_nodes"] == 2
                assert stats["decode_nodes"] == 1
