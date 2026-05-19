from __future__ import annotations
"""Yunshu Mesh — Data parallelism for distributed inference.

Each node loads a full copy of the model. Incoming requests are
distributed across nodes using round-robin or load-aware routing.
No weight sharding needed — scales throughput linearly with nodes.

API:
  DataParallelRouter — distributes requests across engine replicas
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class NodeLoad:
    """Track load on a single node for routing decisions."""
    node_id: str
    rank: int
    active_requests: int = 0
    total_requests: int = 0
    avg_latency_ms: float = 0.0
    last_request_time: float = 0.0
    available: bool = True
    # C17: Memory-proportional capacity weight (1.0 = default)
    # Set >1.0 for nodes with more memory, <1.0 for constrained nodes
    capacity_weight: float = 1.0
    # Node memory info (bytes)
    memory_bytes: int = 0
    gpu_cores: int = 0

    def record_request_start(self) -> None:
        self.active_requests += 1
        self.total_requests += 1
        self.last_request_time = time.monotonic()

    def record_request_end(self, latency_ms: float) -> None:
        self.active_requests = max(0, self.active_requests - 1)
        # Exponential moving average
        alpha = 0.3
        self.avg_latency_ms = (
            alpha * latency_ms + (1 - alpha) * self.avg_latency_ms
            if self.avg_latency_ms > 0 else latency_ms
        )


class DataParallelRouter:
    """Routes requests across data-parallel engine replicas.

    Strategies:
    - round_robin: simple cycling through nodes
    - least_loaded: pick node with fewest active requests
    - latency_aware: weight by average latency (lower is better)
    """

    def __init__(self, strategy: str = "least_loaded"):
        self._strategy = strategy
        self._nodes: dict[str, NodeLoad] = {}
        self._rr_index = 0
        self._lock = threading.Lock()

    def add_node(self, node_id: str, rank: int) -> None:
        """Register a data-parallel node. Idempotent — preserves existing load stats."""
        with self._lock:
            if node_id not in self._nodes:
                self._nodes[node_id] = NodeLoad(node_id=node_id, rank=rank)

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the routing pool."""
        with self._lock:
            self._nodes.pop(node_id, None)

    def mark_unavailable(self, node_id: str) -> None:
        """Temporarily mark a node as unavailable."""
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].available = False

    def mark_available(self, node_id: str) -> None:
        """Mark a node as available again."""
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].available = True

    def select_node(self) -> Optional[str]:
        """Select the best node for the next request.

        Atomically selects a node and records the request start to prevent
        TOCTOU races where two concurrent callers both select the same
        least-loaded node before either increments active_requests.

        Returns:
            node_id of the selected node, or None if no nodes available.
        """
        with self._lock:
            available = [n for n in self._nodes.values() if n.available]
            if not available:
                return None

            if self._strategy == "round_robin":
                node_id = self._select_round_robin(available)
            elif self._strategy == "least_loaded":
                node_id = self._select_least_loaded(available)
            elif self._strategy == "latency_aware":
                node_id = self._select_latency_aware(available)
            elif self._strategy == "capacity_aware":
                node_id = self._select_capacity_aware(available)
            else:
                node_id = self._select_least_loaded(available)

            # Atomically record the request so the load counter is
            # accurate for the next select_node() call.
            if node_id and node_id in self._nodes:
                self._nodes[node_id].record_request_start()
            return node_id

    def set_node_capacity(self, node_id: str, memory_bytes: int, gpu_cores: int = 0) -> None:
        """C17: Set node capacity for memory-proportional routing.

        Automatically computes capacity_weight based on memory relative
        to the smallest node in the cluster.
        """
        with self._lock:
            if node_id not in self._nodes:
                return
            node = self._nodes[node_id]
            node.memory_bytes = memory_bytes
            node.gpu_cores = gpu_cores
            self._recompute_weights()

    def _recompute_weights(self) -> None:
        """Recompute capacity weights based on node memory."""
        nodes_with_mem = [n for n in self._nodes.values() if n.memory_bytes > 0]
        if not nodes_with_mem:
            return
        min_mem = min(n.memory_bytes for n in nodes_with_mem)
        for n in nodes_with_mem:
            n.capacity_weight = max(n.memory_bytes / min_mem, 0.1)
        # Set default weight for nodes without memory info
        for n in self._nodes.values():
            if n.memory_bytes == 0:
                n.capacity_weight = 1.0

    def _select_round_robin(self, available: list[NodeLoad]) -> str:
        # Sort by node_id for stable ordering regardless of dict iteration
        sorted_avail = sorted(available, key=lambda n: n.node_id)
        idx = self._rr_index % len(sorted_avail)
        self._rr_index = (self._rr_index + 1) % (1 << 30)
        return sorted_avail[idx].node_id

    def _select_least_loaded(self, available: list[NodeLoad]) -> str:
        return min(available, key=lambda n: n.active_requests).node_id

    def _select_latency_aware(self, available: list[NodeLoad]) -> str:
        """Weight by latency, prefer lower latency nodes when equally loaded."""
        def score(n: NodeLoad) -> float:
            lat = max(n.avg_latency_ms, 1.0)
            # Active requests dominate, latency breaks ties
            return n.active_requests * 1000 + lat
        return min(available, key=score).node_id

    def _select_capacity_aware(self, available: list[NodeLoad]) -> str:
        """C17: Memory-proportional + bandwidth-aware selection.

        Nodes with more memory get proportionally more requests.
        Bandwidth-aware: prefer nodes with lower current utilization
        relative to their capacity.
        """
        if not any(n.capacity_weight > 0 for n in available):
            return self._select_least_loaded(available)

        def score(n: NodeLoad) -> float:
            weight = max(n.capacity_weight, 0.1)
            utilization = n.active_requests / weight
            return utilization

        return min(available, key=score).node_id

    def record_request_start(self, node_id: str) -> None:
        """Manually record request start for externally-routed requests.

        Note: select_node() already increments active_requests atomically.
        Only call this if the node was chosen outside of select_node().
        """
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].record_request_start()

    def record_request_end(self, node_id: str, latency_ms: float) -> None:
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].record_request_end(latency_ms)

    def get_stats(self) -> dict:
        """Return routing statistics."""
        with self._lock:
            return {
                "strategy": self._strategy,
                "total_nodes": len(self._nodes),
                "available_nodes": sum(
                    1 for n in self._nodes.values() if n.available
                ),
                "nodes": {
                    nid: {
                        "rank": n.rank,
                        "active_requests": n.active_requests,
                        "total_requests": n.total_requests,
                        "avg_latency_ms": round(n.avg_latency_ms, 2),
                        "available": n.available,
                        "capacity_weight": round(n.capacity_weight, 2),
                        "memory_gb": round(n.memory_bytes / (1024**3), 1) if n.memory_bytes else 0,
                        "gpu_cores": n.gpu_cores,
                    }
                    for nid, n in self._nodes.items()
                },
            }

    @property
    def num_nodes(self) -> int:
        with self._lock:
            return len(self._nodes)
