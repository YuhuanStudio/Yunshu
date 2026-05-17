from __future__ import annotations
"""Yunshu Mesh — Topology management.

Manages the physical and logical arrangement of nodes in the mesh.
Supports Ring, Fully-Connected Mesh, and Pipeline Parallel topologies.

Topologies are selected based on:
  - Ring: for low-node-count (< 4) TB5 clusters, minimal bandwidth
  - Fully-Connected: for 2-4 node TB5 clusters (JACCL), max bandwidth
  - Pipeline: for long-sequence models split across nodes

Following SGLang's parallel_state pattern but adapted for mx.distributed.
"""


import logging
from enum import Enum, auto
from typing import Optional

from .node import MeshNode

logger = logging.getLogger(__name__)


class TopologyType(Enum):
    RING = "ring"
    FULLY_CONNECTED = "fully_connected"
    PIPELINE = "pipeline"
    SINGLE = "single"  # Single node, no distributed


class MeshTopology:
    """Manages node arrangement in the compute mesh.

    The topology determines how data flows between nodes during
    collective operations (all_reduce, all_gather, etc.).
    """

    def __init__(self, topo_type: TopologyType = TopologyType.RING):
        self.topo_type = topo_type
        self._nodes: list[MeshNode] = []
        self._rank_map: dict[int, MeshNode] = {}  # rank -> node

    @property
    def size(self) -> int:
        return len(self._nodes)

    @property
    def nodes(self) -> list[MeshNode]:
        return list(self._nodes)

    def add_node(self, node: MeshNode) -> int:
        """Add a node and assign it a rank. Returns the assigned rank.

        If a node with the same node_id already exists, returns its
        existing rank without adding a duplicate.
        """
        for existing in self._nodes:
            if existing.node_id == node.node_id:
                return existing.rank
        if node.rank >= 0 and node.rank not in self._rank_map:
            rank = node.rank
        else:
            # Assign the first unused rank (handles sparse rank maps).
            rank = len(self._nodes)
            while rank in self._rank_map:
                rank += 1
        node.rank = rank
        self._nodes.append(node)
        self._rank_map[rank] = node
        return rank

    def remove_node(self, node_id: str) -> bool:
        """Remove a node by node_id. Re-ranks remaining nodes."""
        for i, n in enumerate(self._nodes):
            if n.node_id == node_id:
                self._nodes.pop(i)
                break
        else:
            return False
        # Re-rank
        self._rank_map.clear()
        for i, n in enumerate(self._nodes):
            n.rank = i
            self._rank_map[i] = n
        return True

    def get_node(self, rank: int) -> Optional[MeshNode]:
        return self._rank_map.get(rank)

    def get_rank(self, node_id: str) -> int:
        for n in self._nodes:
            if n.node_id == node_id:
                return n.rank
        return -1

    def get_neighbors(self, rank: int) -> list[int]:
        """Get neighbor ranks based on topology type."""
        if self.topo_type == TopologyType.RING:
            return [
                (rank - 1) % self.size,
                (rank + 1) % self.size,
            ]
        elif self.topo_type == TopologyType.FULLY_CONNECTED:
            return [i for i in range(self.size) if i != rank]
        elif self.topo_type == TopologyType.PIPELINE:
            neighbors = []
            if rank > 0:
                neighbors.append(rank - 1)
            if rank < self.size - 1:
                neighbors.append(rank + 1)
            return neighbors
        return []

    def auto_select(self) -> TopologyType:
        """Auto-select topology based on node count and capabilities.

        Heuristics from whitepaper §4.3:
        - 1 node: SINGLE
        - 2-4 nodes with JACCL: FULLY_CONNECTED
        - 2-4 nodes without JACCL: RING
        - >4 nodes: RING (scalability)
        """
        if self.size <= 1:
            return TopologyType.SINGLE

        all_jaccl = all(
            n.capabilities.supports_jaccl for n in self._nodes
        )

        if self.size <= 4 and all_jaccl:
            return TopologyType.FULLY_CONNECTED
        return TopologyType.RING

    def get_send_recv_pairs(self, step: int) -> list[tuple[int, int]]:
        """Get (src, dst) pairs for a given communication step.

        For Ring all_reduce, this follows the ring pattern:
        step i: rank sends to (rank + 1) % size, receives from (rank - 1) % size
        """
        if self.topo_type == TopologyType.RING:
            # Ring reduce-scatter phase
            send_rank = (step + 1) % self.size
            recv_rank = (step - 1) % self.size
            return [(recv_rank, send_rank)]

        elif self.topo_type == TopologyType.FULLY_CONNECTED:
            # All-to-all: everyone sends to everyone
            pairs = []
            for src in range(self.size):
                for dst in range(self.size):
                    if src != dst:
                        pairs.append((src, dst))
            return pairs

        elif self.topo_type == TopologyType.PIPELINE:
            # Pipeline: each stage sends to the next
            pairs = []
            for i in range(self.size - 1):
                pairs.append((i, i + 1))
            return pairs

        return []

    def to_dict(self) -> dict:
        return {
            "topology_type": self.topo_type.value,
            "size": self.size,
            "nodes": [n.to_dict() for n in self._nodes],
        }
