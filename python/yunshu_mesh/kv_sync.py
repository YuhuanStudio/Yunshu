from __future__ import annotations
"""Yunshu Mesh — Distributed KV cache synchronization and health monitoring.

Architecture:
  KVSynchronizationService — cross-node KV prefix hash broadcast and transfer
  MeshHealthMonitor — continuous health monitoring with failover and rebalancing

KV Sync Flow:
  1. Each node periodically broadcasts KV prefix hashes it has computed
  2. When a node receives a request for a prefix another node already has,
     it requests a KV transfer from that peer
  3. The peer serves the KV blocks from its local cache using the
     kv_transfer.py wire protocol
  4. Stats (hits/misses/bytes) are tracked for monitoring

Health Monitor Flow:
  1. Periodic heartbeat with metadata (GPU util, memory, active requests)
  2. Failure detection within configurable timeout
  3. Failover triggers: mark node offline, update topology, rebalance
  4. Node join triggers: rebalance via LayerAllocator
"""

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .node import MeshNode, MeshNodeState

logger = logging.getLogger(__name__)

# Re-use the KV transfer wire format from the engine layer.
from yunshu_engine.kv_transfer import (
    KVBlockData,
    KVTransferConfig,
    KVTransferHeader,
    KVTransferMessage,
    KVTransferProtocol,
    KVTransferResult,
    KVTransferStats,
    TransferStatus,
)


# ── Data Structures ──────────────────────────────────────────────────


class SyncMessageType(str, Enum):
    """Types of messages exchanged in KV sync protocol."""
    HASH_BROADCAST = "hash_broadcast"
    TRANSFER_REQUEST = "transfer_request"
    TRANSFER_RESPONSE = "transfer_response"
    TRANSFER_ACK = "transfer_ack"


@dataclass
class PrefixHashEntry:
    """A single prefix hash known to this node.

    Attributes:
        block_hash: Content hash of the KV block.
        block_ids: Physical block IDs on the source node.
        source_node_id: Node that computed this prefix.
        model_name: Model used to compute the prefix.
        num_tokens: Total tokens in the prefix.
        computed_at: Timestamp when this prefix was computed.
        last_verified: Last time we confirmed the prefix is still in cache.
    """
    block_hash: int
    block_ids: list[int] = field(default_factory=list)
    source_node_id: str = ""
    model_name: str = ""
    num_tokens: int = 0
    computed_at: float = field(default_factory=time.time)
    last_verified: float = field(default_factory=time.time)


@dataclass
class TransferRequest:
    """A request to transfer KV blocks from a peer.

    Attributes:
        request_id: Unique request identifier.
        prefix_hash: Hash of the prefix to transfer.
        requestor_node_id: Node making the request.
        target_node_id: Node that has the prefix.
        model_name: Model name for compatibility check.
        timestamp: When the request was created.
    """
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    prefix_hash: int = 0
    requestor_node_id: str = ""
    target_node_id: str = ""
    model_name: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class TransferResponse:
    """Response to a KV transfer request.

    Attributes:
        request_id: Matches the TransferRequest.
        status: Whether the transfer succeeded.
        blocks: KV block data (empty if declined or error).
        error: Error message if status is not COMPLETED.
    """
    request_id: str = ""
    status: TransferStatus = TransferStatus.PENDING
    blocks: list[KVBlockData] = field(default_factory=list)
    error: str | None = None


@dataclass
class KVSyncStats:
    """Aggregate statistics for KV synchronization."""
    broadcasts_sent: int = 0
    broadcasts_received: int = 0
    hashes_advertised: int = 0  # Total unique hashes advertised to peers
    transfers_requested: int = 0
    transfers_completed: int = 0
    transfers_failed: int = 0
    transfers_declined: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    cache_hits: int = 0  # Requests that found a prefix on another node
    cache_misses: int = 0  # Requests that didn't find a prefix anywhere
    hash_lookups: int = 0  # Total lookup attempts

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that found a cached prefix."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def to_dict(self) -> dict:
        return {
            "broadcasts_sent": self.broadcasts_sent,
            "broadcasts_received": self.broadcasts_received,
            "hashes_advertised": self.hashes_advertised,
            "transfers_requested": self.transfers_requested,
            "transfers_completed": self.transfers_completed,
            "transfers_failed": self.transfers_failed,
            "transfers_declined": self.transfers_declined,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hash_lookups": self.hash_lookups,
            "hit_rate": round(self.hit_rate, 4),
        }


# ── Health Monitor Data Structures ───────────────────────────────────


@dataclass
class NodeHealthMetadata:
    """Metadata included in heartbeat messages.

    Attributes:
        gpu_utilization: GPU utilization 0.0-1.0.
        memory_used_bytes: Current memory usage in bytes.
        memory_total_bytes: Total memory in bytes.
        active_requests: Number of active inference requests.
        loaded_models: List of loaded model names.
        kv_cache_usage: Fraction of KV cache blocks in use (0.0-1.0).
    """
    gpu_utilization: float = 0.0
    memory_used_bytes: int = 0
    memory_total_bytes: int = 0
    active_requests: int = 0
    loaded_models: list[str] = field(default_factory=list)
    kv_cache_usage: float = 0.0


@dataclass
class NodeHealthStatus:
    """Health status of a mesh node.

    Attributes:
        node_id: Identifier of the node.
        healthy: Whether the node is considered healthy.
        last_heartbeat: Timestamp of the last received heartbeat.
        metadata: Latest metadata from heartbeat.
        consecutive_failures: Number of consecutive health check failures.
        state: Current lifecycle state of the node.
    """
    node_id: str = ""
    healthy: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    metadata: NodeHealthMetadata = field(default_factory=NodeHealthMetadata)
    consecutive_failures: int = 0
    state: MeshNodeState = MeshNodeState.READY


@dataclass
class RebalanceEvent:
    """Event triggered when nodes join/leave requiring rebalancing.

    Attributes:
        event_type: "node_join" or "node_failure".
        node_id: The node that triggered the event.
        timestamp: When the event occurred.
    """
    event_type: str = ""  # "node_join" or "node_failure"
    node_id: str = ""
    timestamp: float = field(default_factory=time.time)


# ── KVSynchronizationService ─────────────────────────────────────────


class KVSynchronizationService:
    """Cross-node KV cache prefix synchronization.

    Runs on each mesh node. Periodically broadcasts KV cache prefix hashes
    to other nodes and handles transfer requests/responses.

    Architecture:
    - Local hash registry: maps block_hash -> PrefixHashEntry for local prefixes
    - Remote hash registry: maps (node_id, block_hash) -> PrefixHashEntry for remote
    - On broadcast: send local hashes to all peers
    - On transfer request: look up remote registry, request from peer
    - On transfer response: decode and load into local cache

    Thread safety:
    - All mutable state is protected by _lock
    - Transfer operations are async-safe via asyncio

    Usage:
        service = KVSynchronizationService(local_node_id="node-0")
        service.add_peer(node_id="node-1")
        service.register_local_prefix(block_hash=0xABCD, block_ids=[0,1,2])
        service.broadcast_prefix_hashes()
        response = service.request_kv_transfer(peer_node_id="node-1", prefix_hash=0xABCD)
    """

    def __init__(
        self,
        local_node_id: str = "",
        broadcast_interval: float = 10.0,
        max_entries: int = 100_000,
    ) -> None:
        self._local_node_id = local_node_id
        self._broadcast_interval = broadcast_interval
        self._max_entries = max_entries

        # Local prefix hashes: block_hash -> PrefixHashEntry
        self._local_hashes: dict[int, PrefixHashEntry] = {}
        # Remote prefix hashes: (node_id, block_hash) -> PrefixHashEntry
        self._remote_hashes: dict[tuple[str, int], PrefixHashEntry] = {}
        # All remote hashes indexed by block_hash for fast lookup
        self._remote_hash_index: dict[int, list[PrefixHashEntry]] = {}

        # Peers: node_id -> MeshNode
        self._peers: dict[str, MeshNode] = {}

        # Pending transfers: request_id -> TransferRequest
        self._pending_transfers: dict[str, TransferRequest] = {}
        # Transfer history: request_id -> TransferResponse (recent)
        self._transfer_history: dict[str, TransferResponse] = {}

        # Stats
        self._stats = KVSyncStats()

        # Callback for serving KV blocks from local cache
        self._block_provider: Optional[Callable[[int], list[KVBlockData]]] = None
        # Callback for loading received KV blocks into local cache
        self._block_consumer: Optional[Callable[[list[KVBlockData], str], int]] = None

        # Broadcast state
        self._running = False
        self._broadcast_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

    # ── Configuration ────────────────────────────────────────────────

    @property
    def local_node_id(self) -> str:
        return self._local_node_id

    @property
    def stats(self) -> KVSyncStats:
        return self._stats

    def set_block_provider(
        self, provider: Callable[[int], list[KVBlockData]]
    ) -> None:
        """Set the callback that serves KV blocks from local cache.

        Args:
            provider: Callable that takes a block_hash and returns KVBlockData list.
        """
        self._block_provider = provider

    def set_block_consumer(
        self, consumer: Callable[[list[KVBlockData], str], int]
    ) -> None:
        """Set the callback that loads received KV blocks into local cache.

        Args:
            consumer: Callable that takes (blocks, model_name) and returns count loaded.
        """
        self._block_consumer = consumer

    def add_peer(self, node: MeshNode) -> None:
        """Register a peer node for hash broadcasts and transfers."""
        with self._lock:
            self._peers[node.node_id] = node

    def remove_peer(self, node_id: str) -> None:
        """Remove a peer node."""
        with self._lock:
            self._peers.pop(node_id, None)
            # Clean up remote hashes from this node
            keys_to_remove = [
                k for k in self._remote_hashes if k[0] == node_id
            ]
            for k in keys_to_remove:
                entry = self._remote_hashes.pop(k, None)
                if entry:
                    hash_list = self._remote_hash_index.get(entry.block_hash, [])
                    hash_list = [e for e in hash_list if e.source_node_id != node_id]
                    if hash_list:
                        self._remote_hash_index[entry.block_hash] = hash_list
                    else:
                        self._remote_hash_index.pop(entry.block_hash, None)

    # ── Local Hash Management ────────────────────────────────────────

    def register_local_prefix(
        self,
        block_hash: int,
        block_ids: list[int] | None = None,
        model_name: str = "",
        num_tokens: int = 0,
    ) -> None:
        """Register a locally-computed KV prefix hash.

        Args:
            block_hash: Content hash of the prefix.
            block_ids: Physical block IDs for this prefix.
            model_name: Model used to compute the prefix.
            num_tokens: Total tokens in the prefix.
        """
        with self._lock:
            self._local_hashes[block_hash] = PrefixHashEntry(
                block_hash=block_hash,
                block_ids=block_ids or [],
                source_node_id=self._local_node_id,
                model_name=model_name,
                num_tokens=num_tokens,
            )
            # Evict oldest entries if at capacity
            if len(self._local_hashes) > self._max_entries:
                self._evict_oldest(self._local_hashes)

    def unregister_local_prefix(self, block_hash: int) -> bool:
        """Remove a locally registered prefix hash.

        Returns True if the hash was found and removed.
        """
        with self._lock:
            return self._local_hashes.pop(block_hash, None) is not None

    def get_local_hashes(self) -> list[int]:
        """Return all locally registered prefix hashes."""
        with self._lock:
            return list(self._local_hashes.keys())

    # ── Remote Hash Lookup ───────────────────────────────────────────

    def lookup_remote_hash(self, block_hash: int) -> Optional[PrefixHashEntry]:
        """Look up a prefix hash in the remote registry.

        Returns the first matching entry, or None if not found.
        """
        with self._lock:
            self._stats.hash_lookups += 1
            entries = self._remote_hash_index.get(block_hash, [])
            if entries:
                self._stats.cache_hits += 1
                return entries[0]
            self._stats.cache_misses += 1
            return None

    def has_local_hash(self, block_hash: int) -> bool:
        """Check if a prefix hash exists locally."""
        with self._lock:
            return block_hash in self._local_hashes

    # ── Broadcast ────────────────────────────────────────────────────

    def broadcast_prefix_hashes(
        self, local_hashes: list[int] | None = None
    ) -> dict[str, bool]:
        """Broadcast local KV prefix hashes to all peers.

        In production, this would serialize hashes and send via the mesh
        transport. For this implementation, we simulate by updating the
        remote registries of peer services directly.

        Args:
            local_hashes: Optional explicit list. If None, uses all registered.

        Returns:
            Dict of peer_node_id -> success status.
        """
        with self._lock:
            hashes = local_hashes if local_hashes is not None else list(
                self._local_hashes.keys()
            )
            entries = {
                h: self._local_hashes[h]
                for h in hashes
                if h in self._local_hashes
            }

        if not entries:
            return {}

        self._stats.broadcasts_sent += 1
        self._stats.hashes_advertised += len(entries)

        results: dict[str, bool] = {}
        # In production: serialize and send over network
        # For testing: we return the broadcast payload so tests can
        # call receive_broadcast on peer services
        self._last_broadcast = {
            "source_node_id": self._local_node_id,
            "hashes": {str(h): {
                "block_hash": e.block_hash,
                "block_ids": e.block_ids,
                "model_name": e.model_name,
                "num_tokens": e.num_tokens,
            } for h, e in entries.items()},
            "timestamp": time.time(),
        }

        for peer_id in self._peers:
            results[peer_id] = True  # Simulated success

        return results

    def receive_broadcast(
        self,
        source_node_id: str,
        hash_entries: dict[int, PrefixHashEntry],
    ) -> int:
        """Process a received hash broadcast from a peer.

        Args:
            source_node_id: Node that sent the broadcast.
            hash_entries: Dict of block_hash -> PrefixHashEntry.

        Returns:
            Number of new hashes received.
        """
        with self._lock:
            self._stats.broadcasts_received += 1
            new_count = 0
            for block_hash, entry in hash_entries.items():
                key = (source_node_id, block_hash)
                if key not in self._remote_hashes:
                    new_count += 1
                self._remote_hashes[key] = entry
                # Update the hash index
                if block_hash not in self._remote_hash_index:
                    self._remote_hash_index[block_hash] = []
                entries = self._remote_hash_index[block_hash]
                # Replace existing entry from same source if any
                entries = [e for e in entries if e.source_node_id != source_node_id]
                entries.append(entry)
                self._remote_hash_index[block_hash] = entries
            return new_count

    def get_last_broadcast(self) -> dict | None:
        """Return the last broadcast payload (for testing/diagnostics)."""
        return getattr(self, '_last_broadcast', None)

    # ── Transfer ─────────────────────────────────────────────────────

    def request_kv_transfer(
        self,
        peer_node_id: str,
        prefix_hash: int,
        model_name: str = "",
    ) -> TransferResponse:
        """Request KV blocks for a prefix from a peer node.

        Args:
            peer_node_id: Node that has the prefix.
            prefix_hash: Content hash of the prefix to transfer.
            model_name: Model name for compatibility check.

        Returns:
            TransferResponse with blocks or error.
        """
        with self._lock:
            if peer_node_id not in self._peers:
                return TransferResponse(
                    request_id="",
                    status=TransferStatus.FAILED,
                    error=f"Unknown peer: {peer_node_id}",
                )

            request = TransferRequest(
                prefix_hash=prefix_hash,
                requestor_node_id=self._local_node_id,
                target_node_id=peer_node_id,
                model_name=model_name,
            )
            self._pending_transfers[request.request_id] = request
            self._stats.transfers_requested += 1

        # In production: serialize request and send over network
        # For this implementation: store request for simulated handling
        self._last_transfer_request = request

        # Check if we have a block provider that can serve this locally
        # (for unit testing simulated transfers)
        return TransferResponse(
            request_id=request.request_id,
            status=TransferStatus.PENDING,
        )

    def handle_transfer_request(
        self, request: TransferRequest
    ) -> TransferResponse:
        """Serve a KV transfer request from the local cache.

        Args:
            request: The incoming transfer request.

        Returns:
            TransferResponse with blocks if found, or declined.
        """
        with self._lock:
            entry = self._local_hashes.get(request.prefix_hash)

        if entry is None:
            self._stats.transfers_declined += 1
            return TransferResponse(
                request_id=request.request_id,
                status=TransferStatus.FAILED,
                error=f"Prefix hash 0x{request.prefix_hash:x} not found locally",
            )

        # Check model compatibility
        if (
            request.model_name
            and entry.model_name
            and request.model_name != entry.model_name
        ):
            self._stats.transfers_declined += 1
            return TransferResponse(
                request_id=request.request_id,
                status=TransferStatus.FAILED,
                error=(
                    f"Model mismatch: request={request.model_name}, "
                    f"cached={entry.model_name}"
                ),
            )

        # Get blocks via provider callback
        blocks: list[KVBlockData] = []
        if self._block_provider is not None:
            try:
                blocks = self._block_provider(request.prefix_hash)
            except Exception as e:
                logger.debug("Block provider failed: %s", e, exc_info=True)
                self._stats.transfers_failed += 1
                return TransferResponse(
                    request_id=request.request_id,
                    status=TransferStatus.FAILED,
                    error=f"Block provider error: {e}",
                )

        self._stats.transfers_completed += 1
        total_bytes = sum(b.data_size for b in blocks)
        self._stats.bytes_sent += total_bytes

        return TransferResponse(
            request_id=request.request_id,
            status=TransferStatus.COMPLETED,
            blocks=blocks,
        )

    def complete_transfer(self, response: TransferResponse) -> int:
        """Process a completed transfer response.

        Loads received blocks into the local cache.

        Args:
            response: The transfer response from the peer.

        Returns:
            Number of blocks loaded into local cache.
        """
        with self._lock:
            if response.status != TransferStatus.COMPLETED:
                self._stats.transfers_failed += 1
                self._transfer_history[response.request_id] = response
                return 0

            loaded = 0
            if self._block_consumer is not None and response.blocks:
                try:
                    model_name = ""
                    # Try to get model from the original request
                    req = self._pending_transfers.get(response.request_id)
                    if req:
                        model_name = req.model_name
                    loaded = self._block_consumer(response.blocks, model_name)
                except Exception as e:
                    logger.debug("Block consumer failed: %s", e, exc_info=True)

            total_bytes = sum(b.data_size for b in response.blocks)
            self._stats.bytes_received += total_bytes

            self._transfer_history[response.request_id] = response
            self._pending_transfers.pop(response.request_id, None)

        return loaded

    # ── Async Lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        """Start the periodic broadcast loop."""
        self._running = True
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        logger.info(
            "KV sync service started (node=%s, interval=%.1fs, peers=%d)",
            self._local_node_id,
            self._broadcast_interval,
            len(self._peers),
        )

    async def stop(self) -> None:
        """Stop the service."""
        self._running = False
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        logger.info("KV sync service stopped")

    async def _broadcast_loop(self) -> None:
        """Periodically broadcast local hashes to peers."""
        while self._running:
            try:
                self.broadcast_prefix_hashes()
            except Exception:
                logger.debug("Broadcast failed", exc_info=True)
            await asyncio.sleep(self._broadcast_interval)

    # ── Stats ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return sync statistics."""
        stats = self._stats.to_dict()
        with self._lock:
            stats["local_hash_count"] = len(self._local_hashes)
            stats["remote_hash_count"] = len(self._remote_hashes)
            stats["peer_count"] = len(self._peers)
            stats["pending_transfers"] = len(self._pending_transfers)
        return stats

    # ── Internal ─────────────────────────────────────────────────────

    def _evict_oldest(self, registry: dict) -> None:
        """Evict the least-recently-verified entries from a hash registry."""
        # Sort by last_verified (LRU) and remove the 10% least recently used
        sorted_entries = sorted(
            registry.items(), key=lambda x: x[1].last_verified
        )
        to_remove = max(1, len(sorted_entries) // 10)
        for key, _ in sorted_entries[:to_remove]:
            registry.pop(key, None)


# ── MeshHealthMonitor ────────────────────────────────────────────────


class MeshHealthMonitor:
    """Continuous health monitoring of all mesh nodes.

    Features:
    - Periodic heartbeat collection with metadata
    - Node failure detection within configurable timeout
    - Automatic failover when nodes go offline
    - Rebalancing trigger when nodes join/leave
    - Integration with LayerAllocator for pipeline rebalancing

    Thread safety:
    - Health state protected by _lock
    - Callbacks invoked without lock held

    Usage:
        monitor = MeshHealthMonitor(timeout=30.0, check_interval=5.0)
        monitor.register_node(node)
        monitor.start_monitoring()
        status = monitor.check_node(node.node_id)
        monitor.stop_monitoring()
    """

    def __init__(
        self,
        timeout: float = 30.0,
        check_interval: float = 5.0,
        failure_threshold: int = 3,
    ) -> None:
        """Initialize the health monitor.

        Args:
            timeout: Seconds without heartbeat before marking unhealthy.
            check_interval: Seconds between health checks.
            failure_threshold: Consecutive failures before triggering failover.
        """
        self._timeout = timeout
        self._check_interval = check_interval
        self._failure_threshold = failure_threshold

        # Node health status: node_id -> NodeHealthStatus
        self._node_status: dict[str, NodeHealthStatus] = {}
        # Registered nodes: node_id -> MeshNode
        self._nodes: dict[str, MeshNode] = {}

        # Callbacks
        self._on_node_failure_callbacks: list[Callable] = []
        self._on_node_join_callbacks: list[Callable] = []

        # Rebalance events log
        self._rebalance_events: list[RebalanceEvent] = []

        # Layer allocator integration
        self._layer_allocator: Any = None
        self._current_alloc: list[Any] = []
        self._total_layers: int = 0

        # State
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def check_interval(self) -> float:
        return self._check_interval

    # ── Node Registration ────────────────────────────────────────────

    def register_node(self, node: MeshNode) -> None:
        """Register a node for health monitoring.

        Args:
            node: The mesh node to monitor.
        """
        with self._lock:
            self._nodes[node.node_id] = node
            if node.node_id not in self._node_status:
                self._node_status[node.node_id] = NodeHealthStatus(
                    node_id=node.node_id,
                    healthy=True,
                    last_heartbeat=time.time(),
                    state=node.state,
                )

    def unregister_node(self, node_id: str) -> None:
        """Stop monitoring a node."""
        with self._lock:
            self._nodes.pop(node_id, None)
            self._node_status.pop(node_id, None)

    # ── Heartbeat ────────────────────────────────────────────────────

    def receive_heartbeat(
        self,
        node_id: str,
        metadata: NodeHealthMetadata | None = None,
    ) -> bool:
        """Process a heartbeat from a node.

        Args:
            node_id: The node that sent the heartbeat.
            metadata: Optional metadata (GPU util, memory, etc.).

        Returns:
            True if heartbeat was processed, False if node unknown.
        """
        with self._lock:
            if node_id not in self._node_status:
                return False

            status = self._node_status[node_id]
            status.last_heartbeat = time.time()
            status.healthy = True
            status.consecutive_failures = 0
            if metadata:
                status.metadata = metadata

            # Update node state if it was offline (recovery)
            node = self._nodes.get(node_id)
            if node and node.state == MeshNodeState.OFFLINE:
                node.state = MeshNodeState.READY
                status.state = MeshNodeState.READY

            return True

    def get_local_heartbeat_metadata(self) -> NodeHealthMetadata:
        """Collect heartbeat metadata for the local node.

        Returns a NodeHealthMetadata with best-effort system stats.
        """
        metadata = NodeHealthMetadata()
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                metadata.memory_total_bytes = int(result.stdout.strip())
        except Exception:
            logger.debug("operation failed", exc_info=True)
            pass
        return metadata

    # ── Health Checking ──────────────────────────────────────────────

    def check_node(self, node_id: str) -> NodeHealthStatus:
        """Check the current health status of a node.

        Args:
            node_id: The node to check.

        Returns:
            Current NodeHealthStatus. Returns an offline status for unknown nodes.
        """
        with self._lock:
            if node_id not in self._node_status:
                return NodeHealthStatus(
                    node_id=node_id,
                    healthy=False,
                    state=MeshNodeState.OFFLINE,
                )
            return self._node_status[node_id]

    def check_all_nodes(self) -> dict[str, NodeHealthStatus]:
        """Check health status of all monitored nodes."""
        with self._lock:
            return dict(self._node_status)

    def get_healthy_nodes(self) -> list[str]:
        """Return node IDs of all healthy nodes."""
        with self._lock:
            return [
                nid for nid, status in self._node_status.items()
                if status.healthy
            ]

    def get_unhealthy_nodes(self) -> list[str]:
        """Return node IDs of all unhealthy nodes."""
        with self._lock:
            return [
                nid for nid, status in self._node_status.items()
                if not status.healthy
            ]

    # ── Failure Detection & Failover ─────────────────────────────────

    def _detect_failures(self) -> list[str]:
        """Detect nodes that have exceeded the heartbeat timeout.

        Returns:
            List of node IDs that are currently healthy but whose last
            heartbeat exceeds the timeout.
        """
        now = time.time()
        timed_out: list[str] = []

        with self._lock:
            for node_id, status in self._node_status.items():
                if not status.healthy:
                    continue
                elapsed = now - status.last_heartbeat
                if elapsed > self._timeout:
                    timed_out.append(node_id)

        return timed_out

    def on_node_failure(self, node_id: str) -> None:
        """Handle a node failure — mark offline, trigger failover.

        Args:
            node_id: The node that has failed.
        """
        with self._lock:
            status = self._node_status.get(node_id)
            if status is None:
                return

            # If the node was healthy (direct call from external code or
            # tests), count this as a failure.  If already unhealthy
            # (called from _run_health_check after threshold counting),
            # consecutive_failures was already set.
            if status.healthy:
                status.consecutive_failures += 1

            status.healthy = False
            status.state = MeshNodeState.OFFLINE

            node = self._nodes.get(node_id)
            if node:
                node.state = MeshNodeState.OFFLINE

        # Record event
        event = RebalanceEvent(
            event_type="node_failure",
            node_id=node_id,
        )
        self._rebalance_events.append(event)

        # Fire callbacks (outside lock)
        for cb in self._on_node_failure_callbacks:
            try:
                cb(node_id, self._nodes.get(node_id))
            except Exception:
                logger.debug("on_node_failure callback error", exc_info=True)

        logger.warning(
            "Node failure detected: %s (failures=%d)",
            node_id,
            status.consecutive_failures if status else 0,
        )

    def on_node_join(self, node_id: str, node: MeshNode | None = None) -> None:
        """Handle a node joining — register and trigger rebalancing.

        Args:
            node_id: The joining node.
            node: Optional MeshNode with capabilities.
        """
        if node is not None:
            self.register_node(node)

        with self._lock:
            status = self._node_status.get(node_id)
            if status:
                status.healthy = True
                status.state = MeshNodeState.READY
                status.last_heartbeat = time.time()
                status.consecutive_failures = 0

        # Record event
        event = RebalanceEvent(
            event_type="node_join",
            node_id=node_id,
        )
        self._rebalance_events.append(event)

        # Fire callbacks (outside lock)
        for cb in self._on_node_join_callbacks:
            try:
                cb(node_id, node)
            except Exception:
                logger.debug("on_node_join callback error", exc_info=True)

        logger.info("Node joined: %s", node_id)

    # ── Callbacks ────────────────────────────────────────────────────

    def on_node_failure_callback(self, callback: Callable) -> None:
        """Register a callback for node failure events.

        Callback signature: callback(node_id: str, node: MeshNode | None)
        """
        self._on_node_failure_callbacks.append(callback)

    def on_node_join_callback(self, callback: Callable) -> None:
        """Register a callback for node join events.

        Callback signature: callback(node_id: str, node: MeshNode | None)
        """
        self._on_node_join_callbacks.append(callback)

    # ── Rebalancing Integration ──────────────────────────────────────

    def set_layer_allocator(
        self,
        allocator: Any,
        current_alloc: list[Any],
        total_layers: int,
    ) -> None:
        """Configure integration with LayerAllocator for rebalancing.

        Args:
            allocator: LayerAllocator instance.
            current_alloc: Current StageAllocation list.
            total_layers: Total transformer layers.
        """
        self._layer_allocator = allocator
        self._current_alloc = current_alloc
        self._total_layers = total_layers

    def trigger_rebalance(self) -> list[Any] | None:
        """Trigger a rebalance using the configured LayerAllocator.

        Returns:
            New StageAllocation list, or None if allocator not configured.
        """
        if self._layer_allocator is None:
            logger.warning("Cannot rebalance: no LayerAllocator configured")
            return None

        # Collect healthy node profiles
        from .layer_allocator import NodeProfile

        healthy_nodes = self.get_healthy_nodes()
        profiles = []
        for nid in healthy_nodes:
            node = self._nodes.get(nid)
            if node:
                profiles.append(NodeProfile(
                    node_id=nid,
                    memory_bytes=int(
                        node.capabilities.total_memory_gb * (1024 ** 3)
                    ),
                    gpu_cores=node.capabilities.gpu_cores,
                ))

        if not profiles:
            logger.warning("Cannot rebalance: no healthy nodes")
            return None

        new_alloc = self._layer_allocator.rebalance(
            self._current_alloc,
            profiles,
            self._total_layers,
        )
        self._current_alloc = new_alloc

        logger.info(
            "Rebalanced %d layers across %d nodes",
            self._total_layers,
            len(profiles),
        )
        return new_alloc

    # ── Async Lifecycle ──────────────────────────────────────────────

    async def start_monitoring(self) -> None:
        """Start the periodic health monitoring loop."""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(
            "Health monitor started (timeout=%.1fs, interval=%.1fs, nodes=%d)",
            self._timeout,
            self._check_interval,
            len(self._node_status),
        )

    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitor stopped")

    async def _monitor_loop(self) -> None:
        """Periodically check node health and detect failures."""
        while self._running:
            try:
                self._run_health_check()
            except Exception:
                logger.debug("Health check failed", exc_info=True)
            await asyncio.sleep(self._check_interval)

    def _run_health_check(self) -> None:
        """Execute a single health check cycle.

        For each timed-out healthy node, increments its consecutive_failures
        counter.  When the counter reaches the failure threshold, triggers
        failover via on_node_failure.
        """
        timed_out = self._detect_failures()
        for node_id in timed_out:
            with self._lock:
                status = self._node_status.get(node_id)
                if status is None:
                    continue
                status.consecutive_failures += 1
                if status.consecutive_failures < self._failure_threshold:
                    continue
            # Trigger failover outside the lock to avoid deadlock.
            self.on_node_failure(node_id)

    # ── Stats ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return health monitoring statistics."""
        with self._lock:
            healthy = sum(
                1 for s in self._node_status.values() if s.healthy
            )
            unhealthy = sum(
                1 for s in self._node_status.values() if not s.healthy
            )
            return {
                "total_nodes": len(self._node_status),
                "healthy_nodes": healthy,
                "unhealthy_nodes": unhealthy,
                "timeout_seconds": self._timeout,
                "check_interval": self._check_interval,
                "failure_threshold": self._failure_threshold,
                "rebalance_events": len(self._rebalance_events),
                "nodes": {
                    nid: {
                        "healthy": s.healthy,
                        "state": s.state.name,
                        "last_heartbeat_age": round(
                            time.time() - s.last_heartbeat, 2
                        ),
                        "consecutive_failures": s.consecutive_failures,
                        "gpu_utilization": s.metadata.gpu_utilization,
                        "memory_used_bytes": s.metadata.memory_used_bytes,
                        "active_requests": s.metadata.active_requests,
                    }
                    for nid, s in self._node_status.items()
                },
            }

    def get_rebalance_history(self) -> list[dict]:
        """Return history of rebalance events."""
        return [
            {
                "event_type": e.event_type,
                "node_id": e.node_id,
                "timestamp": e.timestamp,
            }
            for e in self._rebalance_events
        ]
