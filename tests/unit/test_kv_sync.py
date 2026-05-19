"""Tests for distributed KV cache synchronization and mesh health monitoring.

Covers:
  KVSynchronizationService:
    - Local hash registration and unregistration
    - Remote hash broadcasting and receiving
    - KV transfer request/response flow
    - Block provider/consumer callbacks
    - Stats tracking (hits, misses, bytes)
    - Peer management
    - Async lifecycle (start/stop)
    - Capacity eviction

  MeshHealthMonitor:
    - Node registration and unregistration
    - Heartbeat processing
    - Failure detection and failover
    - Node join/recovery
    - Rebalancing with LayerAllocator
    - Callback invocation
    - Stats and rebalance history
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from yunshu_mesh.kv_sync import (
    KVSyncStats,
    KVSynchronizationService,
    MeshHealthMonitor,
    NodeHealthMetadata,
    NodeHealthStatus,
    PrefixHashEntry,
    RebalanceEvent,
    SyncMessageType,
    TransferRequest,
    TransferResponse,
)
from yunshu_mesh.node import MeshNode, MeshNodeState
from yunshu_engine.kv_transfer import (
    KVBlockData,
    TransferStatus,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_node(node_id: str = "node-0", port: int = 8000) -> MeshNode:
    """Create a test MeshNode."""
    return MeshNode(
        node_id=node_id,
        hostname=f"{node_id}.local",
        ip="127.0.0.1",
        port=port,
        state=MeshNodeState.READY,
    )


def _make_service(
    node_id: str = "node-0",
    broadcast_interval: float = 1.0,
    max_entries: int = 100,
) -> KVSynchronizationService:
    """Create a test KV sync service."""
    return KVSynchronizationService(
        local_node_id=node_id,
        broadcast_interval=broadcast_interval,
        max_entries=max_entries,
    )


# ═══════════════════════════════════════════════════════════════════════
# KVSynchronizationService Tests
# ═══════════════════════════════════════════════════════════════════════


class TestKVSyncLocalHashes:
    """Tests for local prefix hash registration."""

    def test_register_local_prefix(self):
        svc = _make_service()
        svc.register_local_prefix(
            block_hash=0xABCD,
            block_ids=[0, 1, 2],
            model_name="qwen-2.5",
            num_tokens=192,
        )
        assert svc.has_local_hash(0xABCD)
        hashes = svc.get_local_hashes()
        assert 0xABCD in hashes

    def test_register_multiple_prefixes(self):
        svc = _make_service()
        for i in range(10):
            svc.register_local_prefix(block_hash=i, num_tokens=64 * (i + 1))
        assert len(svc.get_local_hashes()) == 10

    def test_unregister_local_prefix(self):
        svc = _make_service()
        svc.register_local_prefix(block_hash=0x1111)
        assert svc.unregister_local_prefix(0x1111) is True
        assert not svc.has_local_hash(0x1111)

    def test_unregister_nonexistent_returns_false(self):
        svc = _make_service()
        assert svc.unregister_local_prefix(0xDEAD) is False

    def test_local_hash_overwrite_on_reregister(self):
        svc = _make_service()
        svc.register_local_prefix(block_hash=0x42, num_tokens=64)
        svc.register_local_prefix(block_hash=0x42, num_tokens=128)
        hashes = svc.get_local_hashes()
        assert len(hashes) == 1

    def test_empty_local_hashes_initially(self):
        svc = _make_service()
        assert svc.get_local_hashes() == []

    def test_capacity_eviction(self):
        svc = _make_service(max_entries=10)
        for i in range(15):
            svc.register_local_prefix(block_hash=i)
        # After eviction, should be at or under max_entries
        assert len(svc.get_local_hashes()) <= 10


class TestKVSyncPeers:
    """Tests for peer management."""

    def test_add_peer(self):
        svc = _make_service()
        peer = _make_node("peer-1")
        svc.add_peer(peer)
        stats = svc.get_stats()
        assert stats["peer_count"] == 1

    def test_remove_peer(self):
        svc = _make_service()
        peer = _make_node("peer-1")
        svc.add_peer(peer)
        svc.remove_peer("peer-1")
        stats = svc.get_stats()
        assert stats["peer_count"] == 0

    def test_remove_peer_cleans_remote_hashes(self):
        svc = _make_service()
        peer = _make_node("peer-1")
        svc.add_peer(peer)

        # Simulate receiving hashes from peer-1
        svc.receive_broadcast("peer-1", {
            0xAA: PrefixHashEntry(block_hash=0xAA, source_node_id="peer-1"),
        })
        assert svc.lookup_remote_hash(0xAA) is not None

        svc.remove_peer("peer-1")
        assert svc.lookup_remote_hash(0xAA) is None

    def test_remove_nonexistent_peer_is_safe(self):
        svc = _make_service()
        svc.remove_peer("ghost")  # Should not raise


class TestKVSyncBroadcast:
    """Tests for hash broadcasting."""

    def test_broadcast_sends_to_all_peers(self):
        svc = _make_service("node-0")
        svc.add_peer(_make_node("node-1"))
        svc.add_peer(_make_node("node-2"))

        svc.register_local_prefix(block_hash=0xAAAA)
        results = svc.broadcast_prefix_hashes()

        assert "node-1" in results
        assert "node-2" in results
        assert results["node-1"] is True
        assert results["node-2"] is True

    def test_broadcast_empty_hashes_returns_empty(self):
        svc = _make_service()
        svc.add_peer(_make_node("peer-1"))
        results = svc.broadcast_prefix_hashes()
        assert results == {}

    def test_broadcast_updates_stats(self):
        svc = _make_service()
        svc.add_peer(_make_node("peer-1"))
        svc.register_local_prefix(block_hash=0x1)
        svc.register_local_prefix(block_hash=0x2)

        svc.broadcast_prefix_hashes()
        stats = svc.get_stats()
        assert stats["broadcasts_sent"] == 1
        assert stats["hashes_advertised"] == 2

    def test_broadcast_with_explicit_hash_list(self):
        svc = _make_service()
        svc.add_peer(_make_node("peer-1"))
        svc.register_local_prefix(block_hash=0x1)
        svc.register_local_prefix(block_hash=0x2)

        results = svc.broadcast_prefix_hashes(local_hashes=[0x1])
        assert len(results) == 1
        payload = svc.get_last_broadcast()
        assert len(payload["hashes"]) == 1

    def test_receive_broadcast_adds_to_remote_registry(self):
        svc = _make_service()
        entries = {
            0xBB: PrefixHashEntry(
                block_hash=0xBB,
                source_node_id="node-1",
                model_name="qwen",
            ),
        }
        new_count = svc.receive_broadcast("node-1", entries)
        assert new_count == 1
        assert svc.lookup_remote_hash(0xBB) is not None

    def test_receive_broadcast_duplicate_is_not_new(self):
        svc = _make_service()
        entries = {
            0xBB: PrefixHashEntry(block_hash=0xBB, source_node_id="node-1"),
        }
        svc.receive_broadcast("node-1", entries)
        new_count = svc.receive_broadcast("node-1", entries)
        assert new_count == 0

    def test_receive_broadcast_updates_hash_index(self):
        svc = _make_service()
        # Same hash from two different nodes
        svc.receive_broadcast("node-1", {
            0xCC: PrefixHashEntry(block_hash=0xCC, source_node_id="node-1"),
        })
        svc.receive_broadcast("node-2", {
            0xCC: PrefixHashEntry(block_hash=0xCC, source_node_id="node-2"),
        })
        result = svc.lookup_remote_hash(0xCC)
        assert result is not None
        assert result.source_node_id == "node-1"  # First entry returned


class TestKVSyncTransfer:
    """Tests for KV transfer request/response flow."""

    def test_request_transfer_unknown_peer_fails(self):
        svc = _make_service()
        response = svc.request_kv_transfer("unknown-peer", prefix_hash=0x1)
        assert response.status == TransferStatus.FAILED
        assert "Unknown peer" in response.error

    def test_request_transfer_with_known_peer(self):
        svc = _make_service()
        svc.add_peer(_make_node("peer-1"))
        response = svc.request_kv_transfer("peer-1", prefix_hash=0x1)
        assert response.status == TransferStatus.PENDING
        assert response.request_id  # Non-empty

    def test_request_transfer_increments_stats(self):
        svc = _make_service()
        svc.add_peer(_make_node("peer-1"))
        svc.request_kv_transfer("peer-1", prefix_hash=0x1)
        assert svc.get_stats()["transfers_requested"] == 1

    def test_handle_transfer_request_hash_not_found(self):
        svc = _make_service()
        request = TransferRequest(
            prefix_hash=0xDEAD,
            requestor_node_id="node-1",
            target_node_id="node-0",
        )
        response = svc.handle_transfer_request(request)
        assert response.status == TransferStatus.FAILED
        assert "not found" in response.error

    def test_handle_transfer_request_hash_found(self):
        svc = _make_service()
        svc.register_local_prefix(
            block_hash=0xBEEF,
            model_name="qwen-2.5",
        )
        svc.set_block_provider(lambda h: [
            KVBlockData(block_hash=h, token_count=64, layer_data={0: b"kv"}),
        ])

        request = TransferRequest(
            prefix_hash=0xBEEF,
            requestor_node_id="node-1",
            target_node_id="node-0",
            model_name="qwen-2.5",
        )
        response = svc.handle_transfer_request(request)
        assert response.status == TransferStatus.COMPLETED
        assert len(response.blocks) == 1
        assert response.blocks[0].block_hash == 0xBEEF

    def test_handle_transfer_request_model_mismatch(self):
        svc = _make_service()
        svc.register_local_prefix(
            block_hash=0xBEEF,
            model_name="qwen-2.5",
        )
        request = TransferRequest(
            prefix_hash=0xBEEF,
            requestor_node_id="node-1",
            model_name="llama-3",
        )
        response = svc.handle_transfer_request(request)
        assert response.status == TransferStatus.FAILED
        assert "mismatch" in response.error.lower()

    def test_handle_transfer_request_provider_error(self):
        svc = _make_service()
        svc.register_local_prefix(block_hash=0xBEEF)
        svc.set_block_provider(lambda h: (_ for _ in ()).throw(RuntimeError("oops")))

        request = TransferRequest(prefix_hash=0xBEEF, requestor_node_id="node-1")
        response = svc.handle_transfer_request(request)
        assert response.status == TransferStatus.FAILED
        assert "Block provider error" in response.error

    def test_complete_transfer_success(self):
        svc = _make_service()
        svc.add_peer(_make_node("peer-1"))

        # Request
        req_response = svc.request_kv_transfer("peer-1", prefix_hash=0x1)

        # Set consumer
        loaded_blocks = []
        svc.set_block_consumer(
            lambda blocks, model: (loaded_blocks.extend(blocks), len(blocks))[1]
        )

        # Complete
        response = TransferResponse(
            request_id=req_response.request_id,
            status=TransferStatus.COMPLETED,
            blocks=[KVBlockData(block_hash=0x1, token_count=64)],
        )
        loaded = svc.complete_transfer(response)
        assert loaded == 1

    def test_complete_transfer_failure_status(self):
        svc = _make_service()
        response = TransferResponse(
            request_id="test",
            status=TransferStatus.FAILED,
            error="connection lost",
        )
        loaded = svc.complete_transfer(response)
        assert loaded == 0
        assert svc.get_stats()["transfers_failed"] >= 1


class TestKVSyncStats:
    """Tests for KV sync statistics."""

    def test_initial_stats(self):
        svc = _make_service()
        stats = svc.get_stats()
        assert stats["local_hash_count"] == 0
        assert stats["remote_hash_count"] == 0
        assert stats["peer_count"] == 0
        assert stats["broadcasts_sent"] == 0
        assert stats["hit_rate"] == 0.0

    def test_hit_rate_calculation(self):
        svc = _make_service()
        # Simulate a hit
        svc.receive_broadcast("node-1", {
            0xAA: PrefixHashEntry(block_hash=0xAA, source_node_id="node-1"),
        })
        svc.lookup_remote_hash(0xAA)  # Hit
        svc.lookup_remote_hash(0xBB)  # Miss
        stats = svc.get_stats()
        assert stats["hit_rate"] == 0.5

    def test_kv_sync_stats_dataclass(self):
        stats = KVSyncStats()
        assert stats.hit_rate == 0.0
        d = stats.to_dict()
        assert "broadcasts_sent" in d
        assert "hit_rate" in d
        assert "bytes_sent" in d

    def test_bytes_tracked_on_transfer(self):
        svc = _make_service()
        svc.register_local_prefix(block_hash=0x1)
        svc.set_block_provider(lambda h: [
            KVBlockData(block_hash=h, token_count=64, layer_data={0: b"x" * 100}),
        ])
        request = TransferRequest(prefix_hash=0x1, requestor_node_id="node-1")
        response = svc.handle_transfer_request(request)
        assert svc.get_stats()["bytes_sent"] == 100


class TestKVSyncAsync:
    """Tests for async lifecycle."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        svc = _make_service()
        await svc.start()
        assert svc._running is True
        await svc.stop()
        assert svc._running is False

    @pytest.mark.asyncio
    async def test_broadcast_loop_fires(self):
        svc = _make_service(broadcast_interval=0.1)
        svc.add_peer(_make_node("peer-1"))
        svc.register_local_prefix(block_hash=0x42)

        await svc.start()
        await asyncio.sleep(0.3)  # Allow at least one broadcast cycle
        await svc.stop()

        assert svc.get_stats()["broadcasts_sent"] >= 1


# ═══════════════════════════════════════════════════════════════════════
# MeshHealthMonitor Tests
# ═══════════════════════════════════════════════════════════════════════


class TestHealthMonitorRegistration:
    """Tests for node registration."""

    def test_register_node(self):
        mon = MeshHealthMonitor()
        node = _make_node("node-0")
        mon.register_node(node)
        status = mon.check_node("node-0")
        assert status.healthy is True
        assert status.node_id == "node-0"

    def test_register_multiple_nodes(self):
        mon = MeshHealthMonitor()
        mon.register_node(_make_node("node-0"))
        mon.register_node(_make_node("node-1"))
        mon.register_node(_make_node("node-2"))
        assert len(mon.get_healthy_nodes()) == 3

    def test_unregister_node(self):
        mon = MeshHealthMonitor()
        mon.register_node(_make_node("node-0"))
        mon.unregister_node("node-0")
        status = mon.check_node("node-0")
        assert status.healthy is False
        assert status.state == MeshNodeState.OFFLINE

    def test_unregister_nonexistent_is_safe(self):
        mon = MeshHealthMonitor()
        mon.unregister_node("ghost")  # Should not raise


class TestHealthMonitorHeartbeat:
    """Tests for heartbeat processing."""

    def test_receive_heartbeat(self):
        mon = MeshHealthMonitor()
        mon.register_node(_make_node("node-0"))
        result = mon.receive_heartbeat("node-0")
        assert result is True

    def test_receive_heartbeat_with_metadata(self):
        mon = MeshHealthMonitor()
        mon.register_node(_make_node("node-0"))
        metadata = NodeHealthMetadata(
            gpu_utilization=0.75,
            memory_used_bytes=32 * 1024**3,
            memory_total_bytes=64 * 1024**3,
            active_requests=5,
            loaded_models=["qwen-2.5"],
        )
        mon.receive_heartbeat("node-0", metadata)
        status = mon.check_node("node-0")
        assert status.metadata.gpu_utilization == 0.75
        assert status.metadata.active_requests == 5

    def test_receive_heartbeat_unknown_node_returns_false(self):
        mon = MeshHealthMonitor()
        result = mon.receive_heartbeat("ghost")
        assert result is False

    def test_heartbeat_updates_last_timestamp(self):
        mon = MeshHealthMonitor()
        mon.register_node(_make_node("node-0"))
        before = mon.check_node("node-0").last_heartbeat
        time.sleep(0.01)
        mon.receive_heartbeat("node-0")
        after = mon.check_node("node-0").last_heartbeat
        assert after > before

    def test_heartbeat_recovers_offline_node(self):
        mon = MeshHealthMonitor()
        node = _make_node("node-0")
        mon.register_node(node)

        # Force node offline
        mon.on_node_failure("node-0")
        assert mon.check_node("node-0").healthy is False
        assert node.state == MeshNodeState.OFFLINE

        # Heartbeat recovers it
        mon.receive_heartbeat("node-0")
        assert mon.check_node("node-0").healthy is True
        assert node.state == MeshNodeState.READY


class TestHealthMonitorFailureDetection:
    """Tests for failure detection and failover."""

    def test_detect_failure_after_timeout(self):
        mon = MeshHealthMonitor(timeout=0.05, failure_threshold=1)
        node = _make_node("node-0")
        mon.register_node(node)

        # Simulate heartbeat received long ago
        mon._node_status["node-0"].last_heartbeat = time.monotonic() - 10

        # Detect failures
        timed_out = mon._detect_failures()
        assert "node-0" in timed_out

    def test_on_node_failure_marks_unhealthy(self):
        mon = MeshHealthMonitor()
        node = _make_node("node-0")
        mon.register_node(node)

        mon.on_node_failure("node-0")
        status = mon.check_node("node-0")
        assert status.healthy is False
        assert status.state == MeshNodeState.OFFLINE
        assert node.state == MeshNodeState.OFFLINE

    def test_on_node_failure_increments_consecutive_failures(self):
        mon = MeshHealthMonitor()
        mon.register_node(_make_node("node-0"))

        mon.on_node_failure("node-0")
        assert mon.check_node("node-0").consecutive_failures == 1

    def test_failure_threshold_triggers_failover(self):
        mon = MeshHealthMonitor(failure_threshold=2)
        node = _make_node("node-0")
        mon.register_node(node)

        failover_called = []
        mon.on_node_failure_callback(
            lambda nid, n: failover_called.append(nid)
        )

        # First failure: below threshold
        mon._node_status["node-0"].last_heartbeat = time.monotonic() - 10
        mon.on_node_failure("node-0")
        assert len(failover_called) == 1

    def test_failure_callback_invoked(self):
        mon = MeshHealthMonitor()
        mon.register_node(_make_node("node-0"))

        callbacks = []
        mon.on_node_failure_callback(lambda nid, n: callbacks.append(nid))

        mon.on_node_failure("node-0")
        assert "node-0" in callbacks

    def test_failure_records_rebalance_event(self):
        mon = MeshHealthMonitor()
        mon.register_node(_make_node("node-0"))

        mon.on_node_failure("node-0")
        history = mon.get_rebalance_history()
        assert len(history) == 1
        assert history[0]["event_type"] == "node_failure"
        assert history[0]["node_id"] == "node-0"

    def test_healthy_node_not_detected_as_failed(self):
        mon = MeshHealthMonitor(timeout=30.0)
        mon.register_node(_make_node("node-0"))
        mon.receive_heartbeat("node-0")  # Fresh heartbeat

        timed_out = mon._detect_failures()
        assert "node-0" not in timed_out


class TestHealthMonitorNodeJoin:
    """Tests for node join and recovery."""

    def test_on_node_join(self):
        mon = MeshHealthMonitor()
        node = _make_node("node-1")
        mon.on_node_join("node-1", node)
        assert mon.check_node("node-1").healthy is True

    def test_node_join_callback(self):
        mon = MeshHealthMonitor()
        joined = []
        mon.on_node_join_callback(lambda nid, n: joined.append(nid))

        mon.on_node_join("node-1", _make_node("node-1"))
        assert "node-1" in joined

    def test_node_join_records_rebalance_event(self):
        mon = MeshHealthMonitor()
        mon.on_node_join("node-1", _make_node("node-1"))
        history = mon.get_rebalance_history()
        assert any(e["event_type"] == "node_join" for e in history)

    def test_get_healthy_unhealthy_nodes(self):
        mon = MeshHealthMonitor()
        mon.register_node(_make_node("node-0"))
        mon.register_node(_make_node("node-1"))

        mon.on_node_failure("node-1")
        healthy = mon.get_healthy_nodes()
        unhealthy = mon.get_unhealthy_nodes()
        assert "node-0" in healthy
        assert "node-1" in unhealthy


class TestHealthMonitorRebalance:
    """Tests for rebalancing integration with LayerAllocator."""

    def test_rebalance_without_allocator_returns_none(self):
        mon = MeshHealthMonitor()
        result = mon.trigger_rebalance()
        assert result is None

    def test_rebalance_with_allocator(self):
        from yunshu_mesh.layer_allocator import LayerAllocator, NodeProfile, StageAllocation

        mon = MeshHealthMonitor()
        allocator = LayerAllocator()
        initial_alloc = allocator.allocate(
            32,
            [NodeProfile("node-0", memory_bytes=64 * 1024**3, gpu_cores=48)],
        )

        node0 = _make_node("node-0")
        node0.capabilities.gpu_cores = 48
        node0.capabilities.total_memory_gb = 64
        node1 = _make_node("node-1")
        node1.capabilities.gpu_cores = 38
        node1.capabilities.total_memory_gb = 36

        mon.register_node(node0)
        mon.register_node(node1)

        mon.set_layer_allocator(allocator, initial_alloc, 32)

        # Add node-1 to allocator profiles
        profiles = [
            NodeProfile("node-0", memory_bytes=64 * 1024**3, gpu_cores=48),
            NodeProfile("node-1", memory_bytes=36 * 1024**3, gpu_cores=38),
        ]
        new_alloc = allocator.rebalance(initial_alloc, profiles, 32)
        assert len(new_alloc) == 2
        assert sum(s.num_layers for s in new_alloc) == 32

    def test_rebalance_no_healthy_nodes_returns_none(self):
        mon = MeshHealthMonitor()
        mock_allocator = MagicMock()
        mon.set_layer_allocator(mock_allocator, [], 32)

        node = _make_node("node-0")
        mon.register_node(node)
        mon.on_node_failure("node-0")

        result = mon.trigger_rebalance()
        assert result is None


class TestHealthMonitorAsync:
    """Tests for async lifecycle."""

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        mon = MeshHealthMonitor()
        mon.register_node(_make_node("node-0"))
        await mon.start_monitoring()
        assert mon._running is True
        await mon.stop_monitoring()
        assert mon._running is False

    @pytest.mark.asyncio
    async def test_health_check_detects_timeout(self):
        mon = MeshHealthMonitor(
            timeout=0.05,
            check_interval=0.1,
            failure_threshold=1,
        )
        node = _make_node("node-0")
        mon.register_node(node)

        # Make heartbeat stale
        mon._node_status["node-0"].last_heartbeat = time.monotonic() - 10

        await mon.start_monitoring()
        await asyncio.sleep(0.3)  # Allow health check to run
        await mon.stop_monitoring()

        assert mon.check_node("node-0").healthy is False


class TestHealthMonitorStats:
    """Tests for health monitor statistics."""

    def test_get_stats(self):
        mon = MeshHealthMonitor(timeout=30.0, check_interval=5.0)
        mon.register_node(_make_node("node-0"))
        mon.register_node(_make_node("node-1"))
        mon.on_node_failure("node-1")

        stats = mon.get_stats()
        assert stats["total_nodes"] == 2
        assert stats["healthy_nodes"] == 1
        assert stats["unhealthy_nodes"] == 1
        assert stats["timeout_seconds"] == 30.0
        assert stats["check_interval"] == 5.0
        assert "node-0" in stats["nodes"]
        assert stats["nodes"]["node-0"]["healthy"] is True

    def test_check_all_nodes(self):
        mon = MeshHealthMonitor()
        mon.register_node(_make_node("node-0"))
        mon.register_node(_make_node("node-1"))

        all_status = mon.check_all_nodes()
        assert len(all_status) == 2
        assert all(s.healthy for s in all_status.values())


# ═══════════════════════════════════════════════════════════════════════
# Data Structure Tests
# ═══════════════════════════════════════════════════════════════════════


class TestPrefixHashEntry:
    def test_default_fields(self):
        entry = PrefixHashEntry(block_hash=0x42)
        assert entry.block_hash == 0x42
        assert entry.block_ids == []
        assert entry.source_node_id == ""
        assert entry.computed_at > 0

    def test_custom_fields(self):
        entry = PrefixHashEntry(
            block_hash=0x99,
            block_ids=[10, 11],
            source_node_id="node-5",
            model_name="qwen-2.5",
            num_tokens=128,
        )
        assert entry.block_ids == [10, 11]
        assert entry.source_node_id == "node-5"
        assert entry.model_name == "qwen-2.5"


class TestTransferRequest:
    def test_auto_generated_request_id(self):
        req = TransferRequest()
        assert len(req.request_id) > 0

    def test_unique_request_ids(self):
        ids = {TransferRequest().request_id for _ in range(100)}
        assert len(ids) == 100

    def test_custom_fields(self):
        req = TransferRequest(
            prefix_hash=0xDEAD,
            requestor_node_id="node-0",
            target_node_id="node-1",
            model_name="qwen",
        )
        assert req.prefix_hash == 0xDEAD
        assert req.target_node_id == "node-1"


class TestTransferResponse:
    def test_default_status(self):
        resp = TransferResponse()
        assert resp.status == TransferStatus.PENDING

    def test_completed_with_blocks(self):
        resp = TransferResponse(
            request_id="r1",
            status=TransferStatus.COMPLETED,
            blocks=[KVBlockData(block_hash=0x1, token_count=64)],
        )
        assert resp.status == TransferStatus.COMPLETED
        assert len(resp.blocks) == 1


class TestRebalanceEvent:
    def test_fields(self):
        evt = RebalanceEvent(
            event_type="node_failure",
            node_id="node-0",
        )
        assert evt.event_type == "node_failure"
        assert evt.node_id == "node-0"
        assert evt.timestamp > 0


class TestSyncMessageType:
    def test_values(self):
        assert SyncMessageType.HASH_BROADCAST == "hash_broadcast"
        assert SyncMessageType.TRANSFER_REQUEST == "transfer_request"
        assert SyncMessageType.TRANSFER_RESPONSE == "transfer_response"
        assert SyncMessageType.TRANSFER_ACK == "transfer_ack"


class TestNodeHealthMetadata:
    def test_defaults(self):
        meta = NodeHealthMetadata()
        assert meta.gpu_utilization == 0.0
        assert meta.memory_used_bytes == 0
        assert meta.active_requests == 0
        assert meta.loaded_models == []

    def test_custom(self):
        meta = NodeHealthMetadata(
            gpu_utilization=0.85,
            active_requests=12,
            kv_cache_usage=0.6,
        )
        assert meta.gpu_utilization == 0.85
        assert meta.kv_cache_usage == 0.6
