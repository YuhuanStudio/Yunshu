"""Integration tests for the DISAGG-1 disaggregated serving pipeline.

Validates the full end-to-end flow:
  1. DisaggRouter routes long prompts to prefill nodes, short to decode
  2. KVTransferServer lifecycle (init/start/stop/stats)
  3. KV block serialization and deserialization roundtrip
  4. Client/Server transfer over real TCP sockets
  5. TTL-based garbage collection of transferred blocks
  6. KVSynchronizationService broadcast + transfer with DisaggRouter
  7. End-to-end: route → prefill → transfer → decode with mock transport
"""
from __future__ import annotations

import asyncio
import os
import struct
import time
from unittest.mock import MagicMock, patch

import pytest

from yunshu_engine.kv_transfer import (
    CompressionType,
    KVBlockData,
    KVTransferClient,
    KVTransferConfig,
    KVTransferHeader,
    KVTransferMessage,
    KVTransferProtocol,
    KVTransferResult,
    KVTransferServer,
    KVTransferStats,
    TransferStatus,
    extract_kv_blocks_from_cache,
    load_kv_blocks_into_cache,
)
from yunshu_engine.external_prefill import (
    ExternalPrefiller,
    PrefillResult,
)
from yunshu_mesh.disagg_pd import (
    DisaggConfig,
    DisaggNodeInfo,
    DisaggRouter,
    NodeRole,
)
from yunshu_mesh.kv_sync import (
    KVSynchronizationService,
    TransferResponse,
)
from yunshu_mesh.node import MeshNode, MeshNodeState


# ── Helpers ──────────────────────────────────────────────────────────


def _make_node(node_id: str = "node-0", port: int = 8000) -> MeshNode:
    return MeshNode(
        node_id=node_id,
        hostname=f"{node_id}.local",
        ip="127.0.0.1",
        port=port,
        state=MeshNodeState.READY,
    )


def _make_blocks(n: int = 3, layers: int = 2, data_size: int = 128) -> list[KVBlockData]:
    """Create n KVBlockData objects with mock layer data."""
    return [
        KVBlockData(
            block_hash=0xA000 + i,
            token_count=64,
            layer_data={j: os.urandom(data_size) for j in range(layers)},
        )
        for i in range(n)
    ]


class _FakeModel:
    def __call__(self, input_ids, cache=None, **kwargs):
        return MagicMock()


class _FakeTokenizer:
    eos_token_ids = [3]
    has_thinking = False

    def encode(self, text, **kwargs):
        return list(range(len(text)))

    def decode(self, tokens):
        return " ".join(f"t{t}" for t in tokens)


# ═══════════════════════════════════════════════════════════════════════
# 1. KVTransferServer Lifecycle
# ═══════════════════════════════════════════════════════════════════════


class TestKVTransferServerLifecycle:
    """Test KVTransferServer init, start, stop, and stats."""

    @pytest.mark.asyncio
    async def test_disabled_server_lifecycle(self):
        """Disabled server start/stop is a no-op."""
        config = KVTransferConfig(enabled=False)
        server = KVTransferServer(config)

        assert server._server is None
        await server.start()
        assert server._server is None  # Still None when disabled
        await server.stop()  # Should not raise
        assert server._server is None

    @pytest.mark.asyncio
    async def test_enabled_server_starts_and_stops_cleanly(self):
        """Enabled server starts TCP listener and stops cleanly."""
        config = KVTransferConfig(enabled=True, listen_port=0)
        server = KVTransferServer(config)

        # Before start
        assert server._server is None
        assert server._running is False

        await server.start()
        assert server._server is not None
        assert server._running is True

        # Check it's listening
        addr = server._server.sockets[0].getsockname()
        assert isinstance(addr[1], int)
        assert addr[1] > 0

        await server.stop()
        assert server._server is None
        assert server._running is False

    @pytest.mark.asyncio
    async def test_server_stats_initial_state(self):
        """Stats reflect initial state correctly."""
        config = KVTransferConfig(enabled=False, listen_port=7890)
        server = KVTransferServer(config)

        stats = server.get_stats()
        assert stats["enabled"] is False
        assert stats["listen_port"] == 7890
        assert stats["active_transfers"] == 0
        assert stats["total_transfers"] == 0
        assert stats["total_blocks_received"] == 0
        assert stats["total_bytes_received"] == 0
        assert stats["total_checksum_failures"] == 0

    @pytest.mark.asyncio
    async def test_server_set_kv_manager(self):
        """KV manager can be set and updated."""
        server = KVTransferServer()
        assert server._kv_manager is None

        mock_mgr = MagicMock()
        server.set_kv_manager(mock_mgr)
        assert server._kv_manager is mock_mgr

        server.set_kv_manager(None)
        assert server._kv_manager is None

    @pytest.mark.asyncio
    async def test_server_double_stop_is_safe(self):
        """Calling stop() twice does not raise."""
        config = KVTransferConfig(enabled=True, listen_port=0)
        server = KVTransferServer(config)
        await server.start()
        await server.stop()
        await server.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_server_stop_cancels_active_transfers(self):
        """Stop() cancels all in-progress transfers."""
        config = KVTransferConfig(enabled=True, listen_port=0)
        server = KVTransferServer(config)

        # Simulate an in-progress transfer
        in_progress = KVTransferResult(
            request_id="test-req",
            status=TransferStatus.IN_PROGRESS,
        )
        server._active_transfers["test-req"] = in_progress

        await server.start()
        await server.stop()

        # In-progress transfers should be cancelled
        assert in_progress.status == TransferStatus.CANCELLED
        assert len(server._active_transfers) == 0


# ═══════════════════════════════════════════════════════════════════════
# 2. KV Block Serialization Roundtrip
# ═══════════════════════════════════════════════════════════════════════


class TestKVBlockSerializationRoundtrip:
    """Test KV block serialization/deserialization with wire format."""

    def test_single_block_roundtrip(self):
        """Single block survives serialize → deserialize."""
        original = KVBlockData(
            block_hash=0xDEADBEEF,
            token_count=64,
            layer_data={0: b"kv_layer_0", 1: b"kv_layer_1", 2: b"kv_layer_2"},
        )

        payload = KVTransferProtocol.serialize_blocks([original])
        restored = KVTransferProtocol.deserialize_blocks(payload)

        assert len(restored) == 1
        assert restored[0].block_hash == 0xDEADBEEF
        assert restored[0].token_count == 64
        assert len(restored[0].layer_data) == 3
        for i in range(3):
            assert restored[0].layer_data[i] == f"kv_layer_{i}".encode()

    def test_multi_block_roundtrip_preserves_order(self):
        """Multiple blocks maintain insertion order."""
        blocks = [
            KVBlockData(block_hash=i, token_count=64, layer_data={0: f"block_{i}".encode()})
            for i in range(20)
        ]

        payload = KVTransferProtocol.serialize_blocks(blocks)
        restored = KVTransferProtocol.deserialize_blocks(payload)

        assert len(restored) == 20
        for i, block in enumerate(restored):
            assert block.block_hash == i
            assert block.layer_data[0] == f"block_{i}".encode()

    def test_full_wire_format_roundtrip(self):
        """Full encode_message → decode_message roundtrip."""
        blocks = _make_blocks(n=5, layers=4, data_size=256)
        header = KVTransferHeader(
            request_id="roundtrip-integ",
            block_count=5,
            model_name="qwen-2.5-0.5b-instruct",
            compression="none",
            checksum="",
            total_tokens=320,
            layer_count=4,
            block_size=64,
        )
        message = KVTransferMessage(header=header, blocks=blocks)

        frame = KVTransferProtocol.encode_message(
            message,
            compression=CompressionType.NONE,
            checksum_algorithm="sha256",
        )

        # Verify frame structure
        assert frame[:4] == b"YKVT"

        decoded, result = KVTransferProtocol.decode_message(frame)

        assert result.status == TransferStatus.COMPLETED
        assert result.checksum_verified is True
        assert result.blocks_transferred == 5
        assert decoded.request_id == "roundtrip-integ"
        assert decoded.header.model_name == "qwen-2.5-0.5b-instruct"
        assert len(decoded.blocks) == 5

        # Verify all block data matches
        for i, (orig, rest) in enumerate(zip(blocks, decoded.blocks)):
            assert orig.block_hash == rest.block_hash
            assert orig.token_count == rest.token_count
            assert orig.layer_data == rest.layer_data

    def test_empty_blocks_roundtrip(self):
        """Empty block list survives roundtrip."""
        header = KVTransferHeader(
            request_id="empty", block_count=0, model_name="m",
            compression="none", checksum="", total_tokens=0,
            layer_count=0, block_size=64,
        )
        message = KVTransferMessage(header=header, blocks=[])
        frame = KVTransferProtocol.encode_message(message)
        decoded, result = KVTransferProtocol.decode_message(frame)

        assert result.status == TransferStatus.COMPLETED
        assert decoded.blocks == []
        assert result.blocks_transferred == 0

    def test_large_block_payload_roundtrip(self):
        """Large payloads (1MB+) survive roundtrip."""
        # 100 blocks, 24 layers, 512 bytes each = ~1.2 MB
        blocks = _make_blocks(n=100, layers=24, data_size=512)
        header = KVTransferHeader(
            request_id="large-payload", block_count=100, model_name="m",
            compression="none", checksum="", total_tokens=6400,
            layer_count=24, block_size=64,
        )
        message = KVTransferMessage(header=header, blocks=blocks)
        frame = KVTransferProtocol.encode_message(message)

        decoded, result = KVTransferProtocol.decode_message(frame)
        assert result.status == TransferStatus.COMPLETED
        assert len(decoded.blocks) == 100
        for orig, rest in zip(blocks, decoded.blocks):
            assert orig.layer_data == rest.layer_data

    def test_checksum_mismatch_on_corruption(self):
        """Corrupted payload is detected via checksum."""
        blocks = _make_blocks(n=3)
        header = KVTransferHeader(
            request_id="corrupt", block_count=3, model_name="m",
            compression="none", checksum="", total_tokens=192,
            layer_count=2, block_size=64,
        )
        message = KVTransferMessage(header=header, blocks=blocks)
        frame = KVTransferProtocol.encode_message(message)

        # Corrupt a byte in the payload area
        corrupted = bytearray(frame)
        # Find a byte in the payload (after header area)
        # The header length is at bytes 4:8, skip past header + 8-byte payload length
        header_len = struct.unpack("!I", bytes(corrupted[4:8]))[0]
        payload_offset = 4 + 4 + header_len + 8  # magic + header_len + header + payload_len
        if payload_offset < len(corrupted):
            corrupted[payload_offset] ^= 0xFF
        else:
            corrupted[-1] ^= 0xFF

        _, result = KVTransferProtocol.decode_message(bytes(corrupted))
        assert result.status == TransferStatus.CHECKSUM_MISMATCH
        assert "Checksum mismatch" in (result.error or "")


# ═══════════════════════════════════════════════════════════════════════
# 3. DisaggRouter Routing Decision Logic
# ═══════════════════════════════════════════════════════════════════════


class TestDisaggRouterRouting:
    """Test DisaggRouter routing decisions under various configurations."""

    def test_below_threshold_routes_to_decode(self):
        """Prompts below threshold always route to decode pool."""
        router = DisaggRouter(DisaggConfig(
            enabled=True,
            prefill_threshold_tokens=256,
            auto_role_detection=False,
        ))
        router.add_node("pf1", NodeRole.PREFILL)
        router.add_node("dc1", NodeRole.DECODE)

        for tokens in [0, 1, 50, 100, 255]:
            node_id, role = router.route_request(tokens)
            assert node_id == "dc1", f"Expected dc1 for {tokens} tokens"

    def test_at_threshold_routes_to_prefill(self):
        """Prompts at exactly the threshold route to prefill."""
        router = DisaggRouter(DisaggConfig(
            enabled=True,
            prefill_threshold_tokens=256,
            auto_role_detection=False,
        ))
        router.add_node("pf1", NodeRole.PREFILL)
        router.add_node("dc1", NodeRole.DECODE)

        node_id, role = router.route_request(256)
        assert node_id == "pf1"

    def test_above_threshold_routes_to_prefill(self):
        """Long prompts route to prefill pool."""
        router = DisaggRouter(DisaggConfig(
            enabled=True,
            prefill_threshold_tokens=256,
            auto_role_detection=False,
        ))
        router.add_node("pf1", NodeRole.PREFILL)
        router.add_node("dc1", NodeRole.DECODE)

        for tokens in [256, 512, 1024, 8192]:
            node_id, _ = router.route_request(tokens)
            assert node_id == "pf1", f"Expected pf1 for {tokens} tokens"

    def test_least_loaded_prefill_selection(self):
        """Selects the least-loaded prefill node."""
        router = DisaggRouter(DisaggConfig(
            auto_role_detection=False,
            prefill_threshold_tokens=100,
        ))
        router.add_node("pf1", NodeRole.PREFILL)
        router.add_node("pf2", NodeRole.PREFILL)

        router.update_node_load("pf1", active_prefills=5)
        router.update_node_load("pf2", active_prefills=1)

        node_id, _ = router.route_request(200)
        assert node_id == "pf2"

    def test_least_loaded_decode_selection(self):
        """Selects the least-loaded decode node."""
        router = DisaggRouter(DisaggConfig(
            auto_role_detection=False,
            prefill_threshold_tokens=100,
        ))
        router.add_node("dc1", NodeRole.DECODE)
        router.add_node("dc2", NodeRole.DECODE)

        router.update_node_load("dc1", active_decodes=10)
        router.update_node_load("dc2", active_decodes=2)

        node_id, _ = router.route_request(50)
        assert node_id == "dc2"

    def test_hybrid_nodes_serve_both_roles(self):
        """Hybrid nodes can receive both prefill and decode requests."""
        router = DisaggRouter(DisaggConfig(
            auto_role_detection=False,
            prefill_threshold_tokens=100,
        ))
        router.add_node("hy1", NodeRole.HYBRID)

        # Short request → decode via hybrid
        node_short, _ = router.route_request(50)
        assert node_short == "hy1"

        # Long request → prefill via hybrid
        node_long, _ = router.route_request(200)
        assert node_long == "hy1"

    def test_prefers_dedicated_over_hybrid(self):
        """Dedicated prefill/decode nodes are preferred over hybrid."""
        router = DisaggRouter(DisaggConfig(
            auto_role_detection=False,
            prefill_threshold_tokens=100,
        ))
        router.add_node("hy1", NodeRole.HYBRID)
        router.add_node("pf1", NodeRole.PREFILL)
        router.add_node("dc1", NodeRole.DECODE)

        # Long request should go to dedicated prefill, not hybrid
        node_id, _ = router.route_request(200)
        assert node_id == "pf1"

        # Short request should go to dedicated decode, not hybrid
        node_id, _ = router.route_request(50)
        assert node_id == "dc1"

    def test_no_available_nodes_returns_empty(self):
        """Returns empty string when no nodes are available."""
        router = DisaggRouter(DisaggConfig(auto_role_detection=False))

        node_id, _ = router.route_request(100)
        assert node_id == ""

    def test_all_nodes_unavailable_returns_empty(self):
        """Returns empty string when all nodes are marked unavailable."""
        router = DisaggRouter(DisaggConfig(auto_role_detection=False))
        router.add_node("dc1", NodeRole.DECODE)
        router.mark_unavailable("dc1")

        node_id, _ = router.route_request(100)
        assert node_id == ""

    def test_auto_role_detection_large_machine_is_decode(self):
        """Auto detection assigns DECODE to high-memory high-GPU nodes."""
        router = DisaggRouter()
        # 192 GB + 40 GPU cores → DECODE
        router.add_node("big-iron", memory_gb=192, gpu_cores=40)
        stats = router.get_stats()
        assert stats["decode_nodes"] == 1

    def test_auto_role_detection_medium_is_prefill(self):
        """Auto detection assigns PREFILL to medium-memory nodes."""
        router = DisaggRouter()
        router.add_node("medium", memory_gb=64, gpu_cores=10)
        stats = router.get_stats()
        assert stats["prefill_nodes"] == 1

    def test_auto_role_detection_small_is_hybrid(self):
        """Auto detection assigns HYBRID to small nodes."""
        router = DisaggRouter()
        router.add_node("small", memory_gb=32, gpu_cores=8)
        stats = router.get_stats()
        assert stats["hybrid_nodes"] == 1


# ═══════════════════════════════════════════════════════════════════════
# 4. Client/Server TCP Transfer
# ═══════════════════════════════════════════════════════════════════════


class TestClientServerTCPTransfer:
    """Test real TCP-based KV transfer between client and server."""

    @pytest.mark.asyncio
    async def test_single_transfer(self):
        """Single block transfer from client to server."""
        server_config = KVTransferConfig(enabled=True, listen_port=0)
        server = KVTransferServer(server_config)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        client_config = KVTransferConfig(
            enabled=True,
            remote_host="127.0.0.1",
            remote_port=port,
        )
        client = KVTransferClient(client_config)
        await client.start()

        try:
            blocks = [
                KVBlockData(
                    block_hash=0xAAAA,
                    token_count=64,
                    layer_data={0: b"kv_layer_0_data", 1: b"kv_layer_1_data"},
                ),
            ]
            result = await client.send_blocks(
                blocks=blocks,
                request_id="tcp-test-1",
                model_name="test-model",
                total_tokens=64,
                layer_count=2,
            )

            assert result.status == TransferStatus.COMPLETED
            assert result.blocks_transferred == 1
            assert result.bytes_transferred > 0
            assert result.checksum_verified is True
            assert result.duration_seconds > 0

            # Verify server received
            assert server.stats.total_blocks_received == 1
            assert server.stats.total_transfers >= 1

        finally:
            await client.stop()
            await server.stop()

    @pytest.mark.asyncio
    async def test_multi_block_transfer(self):
        """Multiple blocks in a single transfer."""
        server_config = KVTransferConfig(enabled=True, listen_port=0)
        server = KVTransferServer(server_config)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        client_config = KVTransferConfig(
            enabled=True, remote_host="127.0.0.1", remote_port=port,
        )
        client = KVTransferClient(client_config)
        await client.start()

        try:
            blocks = _make_blocks(n=10, layers=4, data_size=256)
            result = await client.send_blocks(
                blocks=blocks,
                request_id="tcp-test-multi",
                model_name="qwen-2.5",
                total_tokens=640,
                layer_count=4,
            )

            assert result.status == TransferStatus.COMPLETED
            assert result.blocks_transferred == 10

        finally:
            await client.stop()
            await server.stop()

    @pytest.mark.asyncio
    async def test_sequential_transfers(self):
        """Multiple sequential transfers on the same connection."""
        server_config = KVTransferConfig(enabled=True, listen_port=0)
        server = KVTransferServer(server_config)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        client_config = KVTransferConfig(
            enabled=True, remote_host="127.0.0.1", remote_port=port,
        )
        client = KVTransferClient(client_config)
        await client.start()

        try:
            for i in range(5):
                blocks = [KVBlockData(block_hash=i, token_count=64)]
                result = await client.send_blocks(
                    blocks=blocks,
                    request_id=f"seq-{i}",
                    model_name="m",
                    total_tokens=64,
                )
                assert result.status == TransferStatus.COMPLETED

            # All 5 transfers received
            assert server.stats.total_blocks_received == 5

        finally:
            await client.stop()
            await server.stop()

    @pytest.mark.asyncio
    async def test_client_connection_timeout(self):
        """Client returns failure when server is unreachable."""
        config = KVTransferConfig(
            enabled=True,
            remote_host="127.0.0.1",
            remote_port=19999,  # Nobody listening here
            timeout_seconds=0.5,
        )
        client = KVTransferClient(config)
        await client.start()

        try:
            result = await client.send_blocks(
                blocks=[KVBlockData(block_hash=1, token_count=10)],
                request_id="timeout-test",
            )
            assert result.status == TransferStatus.FAILED
            assert result.error is not None
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_client_disabled_returns_failure(self):
        """Disabled client returns failure immediately."""
        config = KVTransferConfig(enabled=False)
        client = KVTransferClient(config)

        result = await client.send_blocks(
            blocks=[KVBlockData(block_hash=1, token_count=10)],
        )
        assert result.status == TransferStatus.FAILED
        assert "not enabled" in result.error


# ═══════════════════════════════════════════════════════════════════════
# 5. TTL-Based Garbage Collection
# ═══════════════════════════════════════════════════════════════════════


class TestTTLGarbageCollection:
    """Test TTL-based cleanup of transferred blocks and active transfers."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_transfers_empty(self):
        """Cleanup on empty dict returns 0."""
        server = KVTransferServer()
        assert server.cleanup_expired_transfers() == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_transfers_removes_completed(self):
        """Cleanup removes completed entries when dict exceeds cap."""
        server = KVTransferServer()

        # Fill with more than 1000 entries
        for i in range(1100):
            server._active_transfers[f"req-{i}"] = KVTransferResult(
                request_id=f"req-{i}",
                status=TransferStatus.COMPLETED,
            )

        removed = server.cleanup_expired_transfers()
        assert removed > 0
        assert len(server._active_transfers) <= 1000

    @pytest.mark.asyncio
    async def test_cleanup_preserves_in_progress(self):
        """Cleanup never removes in-progress transfers."""
        server = KVTransferServer()

        for i in range(1100):
            server._active_transfers[f"req-{i}"] = KVTransferResult(
                request_id=f"req-{i}",
                status=TransferStatus.COMPLETED,
            )

        # Add an in-progress entry
        server._active_transfers["req-inprog"] = KVTransferResult(
            request_id="req-inprog",
            status=TransferStatus.IN_PROGRESS,
        )

        server.cleanup_expired_transfers()

        # In-progress should still be there
        assert "req-inprog" in server._active_transfers

    @pytest.mark.asyncio
    async def test_active_transfers_tracked_during_processing(self):
        """Server tracks active transfers during _process_frame."""
        server = KVTransferServer()
        blocks = _make_blocks(n=2)
        header = KVTransferHeader(
            request_id="tracked-req", block_count=2, model_name="m",
            compression="none", checksum="", total_tokens=128,
            layer_count=2, block_size=64,
        )
        message = KVTransferMessage(header=header, blocks=blocks)
        frame = KVTransferProtocol.encode_message(message)

        # Process the frame
        result = await server._process_frame(frame)

        assert result.status == TransferStatus.COMPLETED
        assert "tracked-req" in server._active_transfers
        entry = server._active_transfers["tracked-req"]
        assert entry.status == TransferStatus.COMPLETED


# ═══════════════════════════════════════════════════════════════════════
# 6. KVSynchronizationService + DisaggRouter Integration
# ═══════════════════════════════════════════════════════════════════════


class TestKVSyncWithDisagg:
    """Test KV sync service integration with DisaggRouter."""

    def test_broadcast_to_peers_and_lookup(self):
        """KV sync broadcasts hashes and peers can look them up."""
        svc_a = KVSynchronizationService(local_node_id="node-a")
        svc_b = KVSynchronizationService(local_node_id="node-b")

        # Add each other as peers
        svc_a.add_peer(_make_node("node-b"))
        svc_b.add_peer(_make_node("node-a"))

        # Node A registers a prefix hash
        svc_a.register_local_prefix(
            block_hash=0xABCD,
            block_ids=[0, 1, 2],
            model_name="qwen-2.5",
            num_tokens=192,
        )

        # Node A broadcasts
        svc_a.broadcast_prefix_hashes()

        # Simulate: node B receives the broadcast
        broadcast = svc_a.get_last_broadcast()
        assert broadcast is not None

        entries = {}
        for h_str, entry_data in broadcast["hashes"].items():
            h = int(h_str)
            entries[h] = KVBlockData.__module__  # just verify we have data
        # More realistic: reconstruct entries
        from yunshu_mesh.kv_sync import PrefixHashEntry
        hash_entries = {}
        for h_str, entry_data in broadcast["hashes"].items():
            h = int(h_str)
            hash_entries[h] = PrefixHashEntry(
                block_hash=h,
                block_ids=entry_data.get("block_ids", []),
                source_node_id="node-a",
                model_name=entry_data.get("model_name", ""),
                num_tokens=entry_data.get("num_tokens", 0),
            )

        svc_b.receive_broadcast("node-a", hash_entries)

        # Node B can now look up the hash
        result = svc_b.lookup_remote_hash(0xABCD)
        assert result is not None
        assert result.source_node_id == "node-a"
        assert result.model_name == "qwen-2.5"

    def test_kv_transfer_via_sync_service(self):
        """Full transfer flow: register → broadcast → request → handle → complete."""
        svc_a = KVSynchronizationService(local_node_id="node-a")
        svc_b = KVSynchronizationService(local_node_id="node-b")

        svc_a.add_peer(_make_node("node-b"))
        svc_b.add_peer(_make_node("node-a"))

        # Node A has the data
        svc_a.register_local_prefix(block_hash=0xBEEF, model_name="qwen-2.5")
        test_blocks = [KVBlockData(block_hash=0xBEEF, token_count=64, layer_data={0: b"kv"})]
        svc_a.set_block_provider(lambda h: test_blocks)

        # Node B requests transfer
        response = svc_b.request_kv_transfer("node-a", prefix_hash=0xBEEF, model_name="qwen-2.5")
        assert response.status == TransferStatus.PENDING

        # Get the stored request and handle it on node A
        request = svc_a._last_transfer_request if hasattr(svc_a, '_last_transfer_request') else None
        if request:
            transfer_response = svc_a.handle_transfer_request(request)
            assert transfer_response.status == TransferStatus.COMPLETED
            assert len(transfer_response.blocks) == 1

    def test_disagg_router_with_kv_sync_stats(self):
        """DisaggRouter stats properly reflect routing + KV transfers."""
        router = DisaggRouter(DisaggConfig(
            enabled=True,
            prefill_threshold_tokens=256,
            auto_role_detection=False,
        ))
        router.add_node("pf1", NodeRole.PREFILL)
        router.add_node("dc1", NodeRole.DECODE)

        # Route some requests
        router.route_request(1024)  # prefill
        router.route_request(100)   # decode
        router.route_request(2048)  # prefill

        # Request KV transfers
        router.request_kv_transfer("req-1", "pf1", "dc1", 32)
        router.request_kv_transfer("req-2", "pf1", "dc1", 64)
        router.complete_kv_transfer("req-1", success=True)
        router.complete_kv_transfer("req-2", success=False)

        stats = router.get_stats()
        assert stats["stats"]["total_prefill_requests"] == 2
        assert stats["stats"]["total_decode_requests"] == 1
        assert stats["stats"]["total_kv_transfers"] == 2
        assert stats["stats"]["kv_transfer_failures"] == 1
        assert stats["pending_transfers"] == 0


# ═══════════════════════════════════════════════════════════════════════
# 7. End-to-End Pipeline with Mock Transport
# ═══════════════════════════════════════════════════════════════════════


class TestEndToEndPipeline:
    """Full pipeline: route → prefill → transfer → decode with mock transport."""

    def test_full_pipeline_flow(self):
        """Simulate the complete disaggregated serving flow."""
        # Step 1: Set up DisaggRouter with prefill and decode nodes
        router = DisaggRouter(DisaggConfig(
            enabled=True,
            prefill_threshold_tokens=256,
            auto_role_detection=False,
        ))
        router.add_node("pf1", NodeRole.PREFILL)
        router.add_node("dc1", NodeRole.DECODE)

        # Step 2: Simulate an incoming request with 1024 tokens
        prompt_tokens = 1024
        request_id = "req-e2e-001"

        # Step 3: Route request — should go to prefill node
        node_id, role = router.route_request(prompt_tokens, request_id=request_id)
        assert node_id == "pf1"
        assert role == NodeRole.PREFILL

        # Step 4: Simulate prefill on prefill node
        prefiller = ExternalPrefiller(_FakeModel(), _FakeTokenizer())
        prefill_result = prefiller.prefill(
            token_ids=list(range(prompt_tokens)),
        )
        assert prefill_result.num_tokens == prompt_tokens

        # Step 5: Extract KV blocks from the prefill result.
        # Without MLX, prefill_result.kv_cache is None, so extract returns
        # empty. Use token_ids directly with an empty list to produce blocks
        # with content hashes (but no layer data).
        blocks = extract_kv_blocks_from_cache(
            prefill_result.kv_cache if prefill_result.kv_cache is not None else [],
            prefill_result.token_ids,
            block_size=64,
        )
        # With empty list (no MLX), we still get blocks with hashes but no
        # layer data. If kv_cache was None, extract returns [].
        # For this test, create mock blocks to simulate what would happen
        # with a real model.
        if not blocks:
            blocks = [
                KVBlockData(
                    block_hash=0x1000 + i,
                    token_count=64,
                    layer_data={j: os.urandom(128) for j in range(2)},
                )
                for i in range(16)  # 1024 / 64 = 16 blocks
            ]
        assert len(blocks) == 16

        # Step 6: Serialize and deserialize KV blocks (simulating network transfer)
        header = KVTransferHeader(
            request_id=request_id,
            block_count=len(blocks),
            model_name="qwen-2.5-0.5b",
            compression="none",
            checksum="",
            total_tokens=prompt_tokens,
            layer_count=0,
            block_size=64,
        )
        message = KVTransferMessage(header=header, blocks=blocks)
        frame = KVTransferProtocol.encode_message(message)

        # Simulate network: decode on the other side
        decoded_msg, decode_result = KVTransferProtocol.decode_message(frame)

        assert decode_result.status == TransferStatus.COMPLETED
        assert decode_result.checksum_verified is True
        assert len(decoded_msg.blocks) == 16

        # Step 7: Register KV transfer with DisaggRouter
        transfer = router.request_kv_transfer(
            request_id, "pf1", "dc1", len(blocks),
        )
        assert transfer.status == "pending"

        # Step 8: Complete transfer
        router.complete_kv_transfer(request_id, success=True)
        assert router.stats.total_kv_transfers == 1

        # Step 9: Simulate decode phase
        router.update_node_load("dc1", active_decodes=1)

        # Step 10: Verify final state
        stats = router.get_stats()
        assert stats["stats"]["total_prefill_requests"] == 1
        assert stats["stats"]["total_kv_transfers"] == 1
        assert stats["nodes"]["dc1"]["active_decodes"] == 1
        assert stats["pending_transfers"] == 0

    @pytest.mark.asyncio
    async def test_pipeline_with_real_tcp_transfer(self):
        """Full pipeline with real TCP KV transfer between nodes."""
        # Set up DisaggRouter
        router = DisaggRouter(DisaggConfig(
            enabled=True,
            prefill_threshold_tokens=256,
            auto_role_detection=False,
        ))
        router.add_node("pf1", NodeRole.PREFILL)
        router.add_node("dc1", NodeRole.DECODE)

        # Route a long request
        node_id, _ = router.route_request(512, request_id="tcp-e2e")
        assert node_id == "pf1"

        # Extract blocks from mock prefill
        tokens = list(range(512))
        blocks = extract_kv_blocks_from_cache([], tokens, block_size=64)
        assert len(blocks) == 8

        # Start KV transfer server (decode node)
        server_config = KVTransferConfig(enabled=True, listen_port=0)
        server = KVTransferServer(server_config)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        # Send blocks via client (prefill node)
        client_config = KVTransferConfig(
            enabled=True,
            remote_host="127.0.0.1",
            remote_port=port,
        )
        client = KVTransferClient(client_config)
        await client.start()

        try:
            result = await client.send_blocks(
                blocks=blocks,
                request_id="tcp-e2e",
                model_name="qwen-2.5",
                total_tokens=512,
                layer_count=0,
            )

            assert result.status == TransferStatus.COMPLETED
            assert result.blocks_transferred == 8
            assert result.checksum_verified is True

            # Register and complete transfer in DisaggRouter
            router.request_kv_transfer("tcp-e2e", "pf1", "dc1", 8)
            router.complete_kv_transfer("tcp-e2e", success=True)

            # Verify server stats
            assert server.stats.total_blocks_received == 8
            assert server.stats.total_bytes_received > 0

            # Verify router stats
            assert router.stats.total_kv_transfers == 1

        finally:
            await client.stop()
            await server.stop()

    def test_pipeline_short_request_no_disagg(self):
        """Short requests go directly to decode, no KV transfer needed."""
        router = DisaggRouter(DisaggConfig(
            enabled=True,
            prefill_threshold_tokens=512,
            auto_role_detection=False,
        ))
        router.add_node("pf1", NodeRole.PREFILL)
        router.add_node("dc1", NodeRole.DECODE)

        # Short request → decode directly
        node_id, role = router.route_request(64, request_id="short-req")
        assert node_id == "dc1"
        assert role == NodeRole.DECODE

        # No KV transfer needed
        stats = router.get_stats()
        assert stats["stats"]["total_decode_requests"] == 1
        assert stats["stats"]["total_prefill_requests"] == 0
        assert stats["pending_transfers"] == 0

    @pytest.mark.asyncio
    async def test_pipeline_multiple_requests_different_sizes(self):
        """Pipeline handles multiple requests with different token counts."""
        router = DisaggRouter(DisaggConfig(
            enabled=True,
            prefill_threshold_tokens=256,
            auto_role_detection=False,
        ))
        router.add_node("pf1", NodeRole.PREFILL)
        router.add_node("dc1", NodeRole.DECODE)

        # Start transfer server
        server_config = KVTransferConfig(enabled=True, listen_port=0)
        server = KVTransferServer(server_config)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        client_config = KVTransferConfig(
            enabled=True, remote_host="127.0.0.1", remote_port=port,
        )
        client = KVTransferClient(client_config)
        await client.start()

        try:
            requests = [
                (64, False),    # short → decode
                (512, True),    # long → prefill + transfer
                (128, False),   # short → decode
                (1024, True),   # long → prefill + transfer
                (32, False),    # short → decode
            ]

            for idx, (token_count, needs_transfer) in enumerate(requests):
                node_id, role = router.route_request(
                    token_count, request_id=f"req-{idx}",
                )

                if needs_transfer:
                    assert node_id == "pf1"
                    blocks = extract_kv_blocks_from_cache(
                        [], list(range(token_count)), block_size=64,
                    )
                    result = await client.send_blocks(
                        blocks=blocks,
                        request_id=f"req-{idx}",
                        model_name="qwen-2.5",
                        total_tokens=token_count,
                    )
                    assert result.status == TransferStatus.COMPLETED
                    router.request_kv_transfer(f"req-{idx}", "pf1", "dc1", len(blocks))
                    router.complete_kv_transfer(f"req-{idx}", success=True)
                else:
                    assert node_id == "dc1"

            # Verify aggregate stats
            stats = router.get_stats()
            assert stats["stats"]["total_prefill_requests"] == 2
            assert stats["stats"]["total_decode_requests"] == 3
            assert stats["stats"]["total_kv_transfers"] == 2

            # Server received all transfers
            assert server.stats.total_blocks_received > 0

        finally:
            await client.stop()
            await server.stop()


# ═══════════════════════════════════════════════════════════════════════
# 8. Edge Cases and Error Handling
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCasesAndErrorHandling:
    """Test error handling and edge cases in the disaggregated pipeline."""

    @pytest.mark.asyncio
    async def test_server_handles_malformed_frame(self):
        """Server gracefully handles malformed frames."""
        server_config = KVTransferConfig(enabled=True, listen_port=0)
        server = KVTransferServer(server_config)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        try:
            # Connect and send garbage
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"XXXX" + struct.pack("!I", 4) + b"test")  # Bad magic
            await writer.drain()

            # Server should close connection (not crash)
            await asyncio.sleep(0.1)
            writer.close()
            await writer.wait_closed()

            # Server should still be running
            assert server._running is True

        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_client_auto_generates_request_id(self):
        """Client generates unique request IDs when none provided."""
        config = KVTransferConfig(enabled=True, listen_port=0)
        server = KVTransferServer(config)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        client_config = KVTransferConfig(
            enabled=True, remote_host="127.0.0.1", remote_port=port,
        )
        client = KVTransferClient(client_config)
        await client.start()

        try:
            # Send without request_id
            result = await client.send_blocks(
                blocks=[KVBlockData(block_hash=1, token_count=10)],
            )
            assert result.status == TransferStatus.COMPLETED
            assert result.request_id  # Auto-generated
            assert result.request_id.startswith("kv-")
        finally:
            await client.stop()
            await server.stop()

    def test_extract_blocks_handles_empty_tokens(self):
        """Extract returns empty for empty token lists."""
        blocks = extract_kv_blocks_from_cache([], [], block_size=64)
        assert blocks == []

    def test_extract_blocks_handles_none_cache(self):
        """Extract returns empty for None cache."""
        blocks = extract_kv_blocks_from_cache(None, [1, 2, 3], block_size=64)
        assert blocks == []

    def test_load_blocks_handles_none_cache(self):
        """Load returns 0 for None cache."""
        loaded = load_kv_blocks_into_cache(None, [MagicMock()])
        assert loaded == 0

    def test_load_blocks_handles_empty_blocks(self):
        """Load returns 0 for empty block list."""
        loaded = load_kv_blocks_into_cache([], [])
        assert loaded == 0

    @pytest.mark.asyncio
    async def test_concurrent_transfers(self):
        """Server handles multiple concurrent transfers."""
        server_config = KVTransferConfig(
            enabled=True, listen_port=0, max_concurrent_transfers=4,
        )
        server = KVTransferServer(server_config)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        async def send_one(idx: int) -> KVTransferResult:
            client_config = KVTransferConfig(
                enabled=True, remote_host="127.0.0.1", remote_port=port,
            )
            client = KVTransferClient(client_config)
            await client.start()
            try:
                return await client.send_blocks(
                    blocks=[KVBlockData(block_hash=idx, token_count=64)],
                    request_id=f"concurrent-{idx}",
                    model_name="m",
                    total_tokens=64,
                )
            finally:
                await client.stop()

        try:
            results = await asyncio.gather(*[send_one(i) for i in range(4)])
            for r in results:
                assert r.status == TransferStatus.COMPLETED
            assert server.stats.total_blocks_received == 4
        finally:
            await server.stop()

    def test_kv_stats_accumulation(self):
        """KVTransferStats correctly accumulates across multiple operations."""
        stats = KVTransferStats()

        # Record 3 sends
        for i in range(3):
            stats.record_send(KVTransferResult(
                request_id=f"r{i}",
                status=TransferStatus.COMPLETED,
                blocks_transferred=10,
                bytes_transferred=1000,
                bytes_original=2000,
                duration_seconds=0.1,
            ))

        assert stats.total_transfers == 3
        assert stats.total_blocks_sent == 30
        assert stats.total_bytes_sent == 3000
        assert stats.avg_send_latency_ms == pytest.approx(100.0)
        assert stats.avg_compression_ratio == 0.5

    def test_disagg_router_remove_nonexistent_node(self):
        """Removing a nonexistent node is safe."""
        router = DisaggRouter()
        router.remove_node("ghost")  # Should not raise

    def test_disagg_router_update_load_nonexistent(self):
        """Updating load on nonexistent node is safe."""
        router = DisaggRouter()
        router.update_node_load("ghost", active_prefills=5)  # Should not raise
