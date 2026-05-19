from __future__ import annotations
"""Yunshu Disaggregated Prefill/Decode — C20: separate prefill/decode nodes.

Studied from exo and vLLM's disaggregated serving pattern:
- Prefill nodes: handle long prompt tokenization + KV cache filling
- Decode nodes: handle autoregressive token generation
- KV transfer: prefill nodes send filled KV blocks to decode nodes

Benefits:
  - Prefill-heavy workloads don't block decode latency
  - Decode nodes maintain stable TTFT/ITL (no prefill interruptions)
  - Prefill nodes can use cheaper hardware (lower memory bandwidth OK)
  - Throughput improvement: 1.5-2x in mixed workloads (vLLM benchmark)

Architecture:
  DisaggRouter → routes requests to prefill or decode pool
  PrefillPool → set of nodes optimized for prefill throughput
  DecodePool → set of nodes optimized for decode latency
  KVTransfer → moves KV cache blocks from prefill to decode node

Limitations on Apple Silicon:
  - Thunderbolt networking limits KV transfer bandwidth (~20 Gbps)
  - Unified memory means no real CPU/GPU memory split
  - Best suited for: M4 Ultra (decode) + M2/M3 (prefill) clusters
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yunshu_kv.manager import KVCacheManager

logger = logging.getLogger(__name__)


class NodeRole(Enum):
    """Role of a node in disaggregated serving."""
    PREFILL = auto()   # Optimized for prefill throughput
    DECODE = auto()    # Optimized for decode latency
    HYBRID = auto()    # Can handle both (default)


@dataclass
class DisaggNodeInfo:
    """Node info for disaggregated routing."""

    node_id: str
    role: NodeRole
    memory_gb: float = 0.0
    gpu_cores: int = 0
    active_prefills: int = 0
    active_decodes: int = 0
    last_prefill_time: float = 0.0
    last_decode_time: float = 0.0
    kv_transfer_queue: int = 0  # Pending KV transfers
    available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "role": self.role.name,
            "memory_gb": self.memory_gb,
            "gpu_cores": self.gpu_cores,
            "active_prefills": self.active_prefills,
            "active_decodes": self.active_decodes,
            "kv_transfer_queue": self.kv_transfer_queue,
            "available": self.available,
        }


@dataclass
class DisaggConfig:
    """Configuration for disaggregated prefill/decode serving.

    Attributes:
        enabled: Whether to enable disaggregated mode.
        prefill_threshold_tokens: Prompts longer than this go to prefill pool.
        kv_transfer_batch_size: Number of KV blocks per transfer batch.
        kv_transfer_timeout_ms: Maximum wait for KV transfer.
        auto_role_detection: Auto-assign roles based on node capabilities.
    """

    enabled: bool = False
    prefill_threshold_tokens: int = 512
    kv_transfer_batch_size: int = 16
    kv_transfer_timeout_ms: float = 5000.0
    auto_role_detection: bool = True

    @classmethod
    def from_env(cls) -> DisaggConfig:
        return cls(
            enabled=os.environ.get("YUNSHU_DISAGG_PD", "0") == "1",
            prefill_threshold_tokens=int(
                os.environ.get("YUNSHU_PREFILL_THRESHOLD", "512")
            ),
            kv_transfer_batch_size=int(
                os.environ.get("YUNSHU_KV_TRANSFER_BATCH", "16")
            ),
            kv_transfer_timeout_ms=float(
                os.environ.get("YUNSHU_KV_TRANSFER_TIMEOUT_MS", "5000")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "prefill_threshold_tokens": self.prefill_threshold_tokens,
            "kv_transfer_batch_size": self.kv_transfer_batch_size,
            "kv_transfer_timeout_ms": self.kv_transfer_timeout_ms,
            "auto_role_detection": self.auto_role_detection,
        }


@dataclass
class KVTransferRequest:
    """A pending KV cache transfer from prefill to decode node."""

    request_id: str
    source_node: str
    target_node: str
    num_blocks: int
    created_at: float = 0.0
    status: str = "pending"  # pending, transferring, completed, failed

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.monotonic()


@dataclass
class DisaggStats:
    """Statistics for disaggregated serving."""

    total_prefill_requests: int = 0
    total_decode_requests: int = 0
    total_kv_transfers: int = 0
    kv_transfer_bytes: int = 0
    kv_transfer_failures: int = 0
    avg_kv_transfer_time_ms: float = 0.0
    prefill_node_utilization: float = 0.0
    decode_node_utilization: float = 0.0

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_prefill_requests": self.total_prefill_requests,
            "total_decode_requests": self.total_decode_requests,
            "total_kv_transfers": self.total_kv_transfers,
            "kv_transfer_failures": self.kv_transfer_failures,
            "avg_kv_transfer_time_ms": round(self.avg_kv_transfer_time_ms, 2),
            "prefill_utilization": round(self.prefill_node_utilization, 3),
            "decode_utilization": round(self.decode_node_utilization, 3),
        }

    def reset(self) -> None:
        self.total_prefill_requests = 0
        self.total_decode_requests = 0
        self.total_kv_transfers = 0
        self.kv_transfer_bytes = 0
        self.kv_transfer_failures = 0
        self.avg_kv_transfer_time_ms = 0.0
        self.prefill_node_utilization = 0.0
        self.decode_node_utilization = 0.0


class DisaggRouter:
    """Routes requests to prefill or decode pools in disaggregated mode.

    Request flow:
    1. Incoming request → classify (prefill or decode)
    2. Short requests (< threshold): route directly to decode pool
    3. Long requests (>= threshold): route to prefill pool first
    4. After prefill completes → KV transfer to decode node
    5. Decode node continues autoregressive generation

    When no dedicated prefill nodes exist, all nodes act as hybrid.
    """

    def __init__(
        self,
        config: DisaggConfig | None = None,
        kv_manager: KVCacheManager | None = None,
    ) -> None:
        self._config = config or DisaggConfig.from_env()
        self._nodes: dict[str, DisaggNodeInfo] = {}
        self._pending_transfers: list[KVTransferRequest] = []
        self._stats = DisaggStats()
        self._kv_manager = kv_manager
        self._lock = threading.Lock()
        # Event relay buffers: cache coherency events collected from the
        # local KV manager and awaiting relay to peer nodes.
        self._pending_events: list[Any] = []
        self._setup_cache_listeners()

    @property
    def config(self) -> DisaggConfig:
        return self._config

    @property
    def stats(self) -> DisaggStats:
        return self._stats

    # -- Cache coherency event handling ------------------------------------

    def _setup_cache_listeners(self) -> None:
        """Subscribe to KV cache events from the local manager.

        Collected events are buffered in ``_pending_events`` for relay
        to peer nodes via the mesh layer.
        """
        if self._kv_manager is None:
            return
        bus = self._kv_manager._event_bus
        bus.subscribe("block_cached", self._on_block_cached)
        bus.subscribe("block_evicted", self._on_block_evicted)
        bus.subscribe("request_freed", self._on_request_freed)

    def _on_block_cached(self, event: Any) -> None:
        """Handle a block-cached event from the local KV manager."""
        with self._lock:
            self._pending_events.append(event)
        logger.debug(
            "Cache event: block_cached hash=0x%x block_ids=%s",
            event.block_hash or 0,
            event.block_ids,
        )

    def _on_block_evicted(self, event: Any) -> None:
        """Handle a block-evicted event from the local KV manager."""
        with self._lock:
            self._pending_events.append(event)
        logger.debug(
            "Cache event: block_evicted hash=0x%x block_ids=%s",
            event.block_hash or 0,
            event.block_ids,
        )

    def _on_request_freed(self, event: Any) -> None:
        """Handle a request-freed event from the local KV manager."""
        with self._lock:
            self._pending_events.append(event)
        logger.debug(
            "Cache event: request_freed node_id=%s block_ids=%s",
            event.node_id,
            event.block_ids,
        )

    def drain_pending_events(self) -> list[Any]:
        """Return and clear buffered cache coherency events.

        The caller (mesh relay loop) is expected to batch-send these
        to peer nodes over the mesh connection.
        """
        with self._lock:
            events = list(self._pending_events)
            self._pending_events.clear()
        return events

    def add_node(
        self,
        node_id: str,
        role: NodeRole = NodeRole.HYBRID,
        memory_gb: float = 0.0,
        gpu_cores: int = 0,
    ) -> None:
        """Register a node in the disaggregated pool."""
        with self._lock:
            if self._config.auto_role_detection and role == NodeRole.HYBRID:
                role = self._auto_detect_role(memory_gb, gpu_cores)

            self._nodes[node_id] = DisaggNodeInfo(
                node_id=node_id,
                role=role,
                memory_gb=memory_gb,
                gpu_cores=gpu_cores,
            )
        logger.info(f"Added node {node_id} as {role.name} ({memory_gb}GB, {gpu_cores} GPU cores)")

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the pool."""
        with self._lock:
            self._nodes.pop(node_id, None)

    def mark_unavailable(self, node_id: str) -> None:
        """Mark a node as unavailable."""
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].available = False

    def mark_available(self, node_id: str) -> None:
        """Mark a node as available."""
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].available = True

    def route_request(
        self,
        prompt_tokens: int,
        request_id: str = "",
    ) -> tuple[str, NodeRole]:
        """Route a request to the appropriate pool.

        Increments active load counters so subsequent routing decisions
        account for in-flight requests.
        """
        with self._lock:
            if prompt_tokens >= self._config.prefill_threshold_tokens:
                node_id = self._select_prefill_node()
                role = NodeRole.PREFILL
                self._stats.total_prefill_requests += 1
            else:
                node_id = self._select_decode_node()
                role = NodeRole.DECODE
                self._stats.total_decode_requests += 1

            # Fallback to any available node
            if node_id is None:
                node_id = self._select_any_node()
                role = NodeRole.HYBRID

            # Increment load counter on the selected node
            if node_id and node_id in self._nodes:
                node = self._nodes[node_id]
                if role == NodeRole.PREFILL or role == NodeRole.HYBRID:
                    node.active_prefills += 1
                if role == NodeRole.DECODE or role == NodeRole.HYBRID:
                    node.active_decodes += 1

        return node_id or None, role

    def request_completed(self, node_id: str, role: NodeRole) -> None:
        """Decrement load counters after a request finishes."""
        with self._lock:
            if node_id not in self._nodes:
                return
            node = self._nodes[node_id]
            if role == NodeRole.PREFILL or role == NodeRole.HYBRID:
                node.active_prefills = max(0, node.active_prefills - 1)
            if role == NodeRole.DECODE or role == NodeRole.HYBRID:
                node.active_decodes = max(0, node.active_decodes - 1)

    def request_kv_transfer(
        self,
        request_id: str,
        source_node: str,
        target_node: str,
        num_blocks: int,
    ) -> KVTransferRequest:
        """Request a KV cache transfer from prefill to decode node.

        In production, this would use Thunderbolt RDMA or TCP.
        Here we track the transfer lifecycle.
        """
        with self._lock:
            transfer = KVTransferRequest(
                request_id=request_id,
                source_node=source_node,
                target_node=target_node,
                num_blocks=num_blocks,
            )
            self._pending_transfers.append(transfer)

            if source_node in self._nodes:
                self._nodes[source_node].kv_transfer_queue += 1

        logger.debug(
            f"KV transfer requested: {source_node} → {target_node}, "
            f"{num_blocks} blocks for {request_id}"
        )
        return transfer

    def complete_kv_transfer(self, request_id: str, success: bool = True) -> None:
        """Mark a KV transfer as completed or failed."""
        with self._lock:
            for t in self._pending_transfers:
                if t.request_id == request_id and t.status == "pending":
                    t.status = "completed" if success else "failed"
                    if t.source_node in self._nodes:
                        self._nodes[t.source_node].kv_transfer_queue = max(
                            0, self._nodes[t.source_node].kv_transfer_queue - 1
                        )
                    self._stats.total_kv_transfers += 1
                    if not success:
                        self._stats.kv_transfer_failures += 1
                    break

            # Prune completed/failed transfers to prevent unbounded growth.
            # Keep only pending transfers (capped at 500) and recent completions.
            if len(self._pending_transfers) > 200:
                active = [t for t in self._pending_transfers if t.status in ("pending", "transferring")]
                terminal = [t for t in self._pending_transfers if t.status not in ("pending", "transferring")]
                self._pending_transfers = active + terminal[-100:]

    def get_pending_transfers(self) -> list[KVTransferRequest]:
        """Get all pending KV transfers."""
        with self._lock:
            return [t for t in self._pending_transfers if t.status == "pending"]

    def _select_prefill_node(self) -> str | None:
        """Select the best prefill node (least loaded)."""
        prefill_nodes = [
            n for n in self._nodes.values()
            if n.available and n.role in (NodeRole.PREFILL, NodeRole.HYBRID)
        ]
        if not prefill_nodes:
            return None
        # Prefer dedicated prefill nodes, then hybrid
        dedicated = [n for n in prefill_nodes if n.role == NodeRole.PREFILL]
        pool = dedicated if dedicated else prefill_nodes
        return min(pool, key=lambda n: n.active_prefills).node_id

    def _select_decode_node(self) -> str | None:
        """Select the best decode node (least loaded)."""
        decode_nodes = [
            n for n in self._nodes.values()
            if n.available and n.role in (NodeRole.DECODE, NodeRole.HYBRID)
        ]
        if not decode_nodes:
            return None
        dedicated = [n for n in decode_nodes if n.role == NodeRole.DECODE]
        pool = dedicated if dedicated else decode_nodes
        return min(pool, key=lambda n: n.active_decodes).node_id

    def _select_any_node(self) -> str | None:
        """Select any available node (fallback)."""
        available = [n for n in self._nodes.values() if n.available]
        if not available:
            return None
        total = lambda n: n.active_prefills + n.active_decodes
        return min(available, key=total).node_id

    @staticmethod
    def _auto_detect_role(memory_gb: float, gpu_cores: int) -> NodeRole:
        """Auto-detect node role based on hardware capabilities.

        High memory bandwidth (more GPU cores) → Decode
        Large memory → Prefill
        """
        if gpu_cores >= 24 and memory_gb >= 128:
            return NodeRole.DECODE
        elif memory_gb >= 64:
            return NodeRole.PREFILL
        else:
            return NodeRole.HYBRID

    def update_node_load(
        self,
        node_id: str,
        active_prefills: int | None = None,
        active_decodes: int | None = None,
    ) -> None:
        """Update a node's current load for routing decisions."""
        with self._lock:
            if node_id not in self._nodes:
                return
            node = self._nodes[node_id]
            if active_prefills is not None:
                node.active_prefills = active_prefills
            if active_decodes is not None:
                node.active_decodes = active_decodes

    def compute_utilization(self) -> None:
        """Compute pool utilization for stats."""
        prefill_nodes = [n for n in self._nodes.values() if n.role == NodeRole.PREFILL]
        decode_nodes = [n for n in self._nodes.values() if n.role == NodeRole.DECODE]

        if prefill_nodes:
            self._stats.prefill_node_utilization = sum(
                n.active_prefills for n in prefill_nodes
            ) / len(prefill_nodes)
        if decode_nodes:
            self._stats.decode_node_utilization = sum(
                n.active_decodes for n in decode_nodes
            ) / len(decode_nodes)

    def get_stats(self) -> dict[str, Any]:
        """Return disaggregated serving statistics."""
        with self._lock:
            # Inline compute_utilization to avoid nested lock acquire
            prefill_nodes = [n for n in self._nodes.values() if n.role == NodeRole.PREFILL]
            decode_nodes = [n for n in self._nodes.values() if n.role == NodeRole.DECODE]
            if prefill_nodes:
                self._stats.prefill_node_utilization = sum(
                    n.active_prefills for n in prefill_nodes
                ) / len(prefill_nodes)
            if decode_nodes:
                self._stats.decode_node_utilization = sum(
                    n.active_decodes for n in decode_nodes
                ) / len(decode_nodes)
            pending = [t for t in self._pending_transfers if t.status == "pending"]
            return {
                "config": self._config.to_dict(),
                "nodes": {
                    nid: node.to_dict() for nid, node in self._nodes.items()
                },
                "pending_transfers": len(pending),
                "prefill_nodes": sum(
                    1 for n in self._nodes.values()
                    if n.role == NodeRole.PREFILL
                ),
                "decode_nodes": sum(
                    1 for n in self._nodes.values()
                    if n.role == NodeRole.DECODE
                ),
                "hybrid_nodes": sum(
                    1 for n in self._nodes.values()
                    if n.role == NodeRole.HYBRID
                ),
                "stats": self._stats.get_stats(),
            }

    def reset(self) -> None:
        """Reset all stats."""
        with self._lock:
            self._stats.reset()
            self._pending_transfers.clear()
            self._pending_events.clear()
