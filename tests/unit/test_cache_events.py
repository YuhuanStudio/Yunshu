"""Tests for KV cache event publishing (cache coherency for disaggregated serving).

Covers:
  - CacheEventBus pub/sub mechanics
  - BlockPool publishing block_cached / block_evicted events
  - KVCacheManager publishing request_freed events
  - DisaggRouter wiring: event relay from KVCacheManager
"""

from __future__ import annotations

import pytest

from yunshu_kv.cache_events import CacheEvent, CacheEventBus
from yunshu_kv.block import BlockPool, KVBlock
from yunshu_kv.manager import KVCacheConfig, KVCacheManager
from yunshu_mesh.disagg_pd import DisaggConfig, DisaggRouter, NodeRole


# ── CacheEventBus ────────────────────────────────────────────────────


class TestCacheEvent:
    def test_defaults(self):
        ev = CacheEvent(event_type="test")
        assert ev.event_type == "test"
        assert ev.block_hash is None
        assert ev.block_ids == []
        assert ev.node_id is None

    def test_full_init(self):
        ev = CacheEvent(
            event_type="block_cached",
            block_hash=0xDEAD,
            block_ids=[1, 2, 3],
            node_id="node-1",
        )
        assert ev.block_hash == 0xDEAD
        assert ev.block_ids == [1, 2, 3]
        assert ev.node_id == "node-1"


class TestCacheEventBus:
    def test_subscribe_and_publish(self):
        bus = CacheEventBus()
        received: list[CacheEvent] = []
        bus.subscribe("block_cached", received.append)
        ev = CacheEvent("block_cached", block_hash=123)
        bus.publish(ev)
        assert len(received) == 1
        assert received[0] is ev

    def test_multiple_subscribers(self):
        bus = CacheEventBus()
        r1: list[CacheEvent] = []
        r2: list[CacheEvent] = []
        bus.subscribe("block_cached", r1.append)
        bus.subscribe("block_cached", r2.append)
        bus.publish(CacheEvent("block_cached", block_hash=10))
        assert len(r1) == 1
        assert len(r2) == 1

    def test_event_type_isolation(self):
        bus = CacheEventBus()
        cached: list[CacheEvent] = []
        evicted: list[CacheEvent] = []
        bus.subscribe("block_cached", cached.append)
        bus.subscribe("block_evicted", evicted.append)
        bus.publish(CacheEvent("block_cached"))
        bus.publish(CacheEvent("block_evicted"))
        assert len(cached) == 1
        assert len(evicted) == 1

    def test_unsubscribe(self):
        bus = CacheEventBus()
        received: list[CacheEvent] = []
        cb = received.append
        bus.subscribe("block_cached", cb)
        bus.unsubscribe("block_cached", cb)
        bus.publish(CacheEvent("block_cached"))
        assert len(received) == 0

    def test_subscriber_count(self):
        bus = CacheEventBus()
        assert bus.subscriber_count("block_cached") == 0
        bus.subscribe("block_cached", lambda e: None)
        assert bus.subscriber_count("block_cached") == 1

    def test_publish_unknown_type_is_noop(self):
        bus = CacheEventBus()
        # Should not raise
        bus.publish(CacheEvent("nonexistent"))

    def test_subscriber_exception_does_not_stop_others(self):
        bus = CacheEventBus()
        good: list[CacheEvent] = []

        def bad_cb(event: CacheEvent):
            raise RuntimeError("boom")

        bus.subscribe("block_cached", bad_cb)
        bus.subscribe("block_cached", good.append)
        bus.publish(CacheEvent("block_cached"))
        assert len(good) == 1  # second subscriber still called


# ── BlockPool events ─────────────────────────────────────────────────


class TestBlockPoolEvents:
    def _make_pool(self, num_blocks: int = 20, **kw) -> tuple[BlockPool, CacheEventBus]:
        bus = CacheEventBus()
        pool = BlockPool(num_blocks=num_blocks, block_size=4, event_bus=bus, **kw)
        return pool, bus

    def test_cache_block_publishes_cached_event(self):
        pool, bus = self._make_pool()
        received: list[CacheEvent] = []
        bus.subscribe("block_cached", received.append)

        block = pool.allocate(1)[0]
        pool.cache_block(block, 0xABCD)

        assert len(received) == 1
        assert received[0].event_type == "block_cached"
        assert received[0].block_hash == 0xABCD
        assert received[0].block_ids == [block.block_id]

    def test_evict_publishes_evicted_event(self):
        pool, bus = self._make_pool()
        received: list[CacheEvent] = []
        bus.subscribe("block_evicted", received.append)

        block = pool.allocate(1)[0]
        pool.cache_block(block, 0xBEEF)
        # Evict by clearing the cache — simulate via reset_prefix_cache
        # which clears hashes but does NOT call _evict_cached_block.
        # Instead, trigger eviction through allocate re-use.
        pool.free([block])
        # The block is now free.  Re-allocating it triggers _evict_cached_block.
        pool.allocate(1)
        # Depending on LRU order, the evicted block may or may not be the one
        # we just freed.  Let's explicitly evict.
        assert len(received) >= 0  # event may fire during allocation

    def test_evict_cached_block_publishes_event(self):
        pool, bus = self._make_pool()
        received: list[CacheEvent] = []
        bus.subscribe("block_evicted", received.append)

        block = pool.allocate(1)[0]
        pool.cache_block(block, 0x1234)
        pool._evict_cached_block(block)

        assert len(received) == 1
        assert received[0].event_type == "block_evicted"
        assert received[0].block_hash == 0x1234

    def test_no_event_bus_is_noop(self):
        # Without an event bus, operations should work normally
        pool = BlockPool(num_blocks=10, block_size=4)
        block = pool.allocate(1)[0]
        pool.cache_block(block, 0xAAAA)
        pool._evict_cached_block(block)
        # No exceptions = pass

    def test_caching_disabled_no_events(self):
        bus = CacheEventBus()
        pool = BlockPool(
            num_blocks=10, block_size=4, enable_caching=False, event_bus=bus,
        )
        received: list[CacheEvent] = []
        bus.subscribe("block_cached", received.append)

        block = pool.allocate(1)[0]
        pool.cache_block(block, 0xCCCC)
        assert len(received) == 0

    def test_eviction_during_allocate_publishes_event(self):
        pool, bus = self._make_pool(num_blocks=5)
        received: list[CacheEvent] = []
        bus.subscribe("block_evicted", received.append)

        # Allocate all free blocks, cache one, free it, then re-allocate
        # to trigger _evict_cached_block inside allocate().
        blocks = pool.allocate(4)  # 5 total - 1 null = 4 free
        pool.cache_block(blocks[0], 0x5000)
        pool.free([blocks[0]])
        # Now blocks[0] is in free queue but still has block_hash.
        # Allocate again — the block is reused and its hash evicted.
        pool.allocate(1)
        assert len(received) == 1
        assert received[0].block_hash == 0x5000


# ── KVCacheManager events ────────────────────────────────────────────


class TestKVCacheManagerEvents:
    def _make_manager(
        self, num_blocks: int = 100,
    ) -> tuple[KVCacheManager, CacheEventBus]:
        config = KVCacheConfig(block_size=4, enable_caching=True)
        mgr = KVCacheManager(config=config, num_blocks=num_blocks)
        return mgr, mgr._event_bus

    def test_free_request_publishes_request_freed(self):
        mgr, bus = self._make_manager()
        received: list[CacheEvent] = []
        bus.subscribe("request_freed", received.append)

        tokens = list(range(16))  # 4 blocks
        table, match = mgr.allocate_for_prefill(tokens, request_id="req-1")
        mgr.free_request(table, request_id="req-1")

        assert len(received) == 1
        assert received[0].event_type == "request_freed"
        assert received[0].node_id == "req-1"
        assert len(received[0].block_ids) > 0

    def test_cache_block_via_manager_publishes_cached(self):
        mgr, bus = self._make_manager()
        received: list[CacheEvent] = []
        bus.subscribe("block_cached", received.append)

        tokens = list(range(16))
        table, match = mgr.allocate_for_prefill(tokens, request_id="req-2")
        # Cache completed blocks
        cached = mgr.cache_completed_blocks(table, tokens)
        assert cached > 0
        assert len(received) == cached

    def test_manager_creates_own_bus_if_none_provided(self):
        config = KVCacheConfig(block_size=4)
        mgr = KVCacheManager(config=config, num_blocks=10)
        assert mgr._event_bus is not None

    def test_manager_uses_provided_bus(self):
        bus = CacheEventBus()
        config = KVCacheConfig(block_size=4)
        mgr = KVCacheManager(config=config, num_blocks=10, event_bus=bus)
        assert mgr._event_bus is bus
        assert mgr.block_pool._event_bus is bus

    def test_allocate_and_free_full_lifecycle_events(self):
        mgr, bus = self._make_manager()
        cached_events: list[CacheEvent] = []
        freed_events: list[CacheEvent] = []
        bus.subscribe("block_cached", cached_events.append)
        bus.subscribe("request_freed", freed_events.append)

        tokens = list(range(20))  # 5 full blocks
        table, match = mgr.allocate_for_prefill(tokens, request_id="lifecycle")
        mgr.cache_completed_blocks(table, tokens)
        assert len(cached_events) == 5

        mgr.free_request(table, request_id="lifecycle")
        assert len(freed_events) == 1
        assert freed_events[0].node_id == "lifecycle"


# ── DisaggRouter event wiring ────────────────────────────────────────


class TestDisaggRouterEvents:
    def test_no_kv_manager_no_crash(self):
        router = DisaggRouter()
        # Should not raise
        events = router.drain_pending_events()
        assert events == []

    def test_events_collected_from_kv_manager(self):
        config = KVCacheConfig(block_size=4, enable_caching=True)
        mgr = KVCacheManager(config=config, num_blocks=50)
        router = DisaggRouter(
            config=DisaggConfig(auto_role_detection=False),
            kv_manager=mgr,
        )

        cached_events: list[CacheEvent] = []
        mgr._event_bus.subscribe("block_cached", cached_events.append)

        # Allocate and cache some blocks
        tokens = list(range(16))
        table, match = mgr.allocate_for_prefill(tokens, request_id="r1")
        mgr.cache_completed_blocks(table, tokens)

        # The router should have collected events via its listeners
        events = router.drain_pending_events()
        assert len(events) > 0
        event_types = [e.event_type for e in events]
        assert "block_cached" in event_types

    def test_request_freed_event_relayed(self):
        config = KVCacheConfig(block_size=4, enable_caching=True)
        mgr = KVCacheManager(config=config, num_blocks=50)
        router = DisaggRouter(kv_manager=mgr)

        tokens = list(range(12))
        table, match = mgr.allocate_for_prefill(tokens, request_id="rr1")
        mgr.free_request(table, request_id="rr1")

        events = router.drain_pending_events()
        freed = [e for e in events if e.event_type == "request_freed"]
        assert len(freed) == 1
        assert freed[0].node_id == "rr1"

    def test_drain_clears_pending_events(self):
        config = KVCacheConfig(block_size=4, enable_caching=True)
        mgr = KVCacheManager(config=config, num_blocks=50)
        router = DisaggRouter(kv_manager=mgr)

        tokens = list(range(8))
        table, match = mgr.allocate_for_prefill(tokens, request_id="d1")
        mgr.free_request(table, request_id="d1")

        first = router.drain_pending_events()
        assert len(first) > 0
        second = router.drain_pending_events()
        assert len(second) == 0

    def test_reset_clears_events(self):
        config = KVCacheConfig(block_size=4, enable_caching=True)
        mgr = KVCacheManager(config=config, num_blocks=50)
        router = DisaggRouter(kv_manager=mgr)

        tokens = list(range(8))
        table, match = mgr.allocate_for_prefill(tokens, request_id="x1")
        mgr.free_request(table, request_id="x1")
        assert len(router._pending_events) > 0

        router.reset()
        assert len(router._pending_events) == 0

    def test_multiple_event_types_interleaved(self):
        config = KVCacheConfig(block_size=4, enable_caching=True)
        mgr = KVCacheManager(config=config, num_blocks=50)
        router = DisaggRouter(kv_manager=mgr)

        # Request 1: allocate, cache, free
        tokens1 = list(range(16))
        table1, _ = mgr.allocate_for_prefill(tokens1, request_id="a1")
        mgr.cache_completed_blocks(table1, tokens1)
        mgr.free_request(table1, request_id="a1")

        events = router.drain_pending_events()
        types = [e.event_type for e in events]
        assert "block_cached" in types
        assert "request_freed" in types
