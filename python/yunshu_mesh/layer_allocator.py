"""Yunshu Mesh — Intelligent layer allocation for pipeline parallelism.

Upgrades from equal-split to memory-proportional, bandwidth-aware, and
latency-optimal layer allocation strategies.  Inspired by:
  - exo's allocate_layers_proportionally() (memory-based split)
  - Parallax's DP layer allocation (latency-minimizing split)
  - PipeDream's balanced pipeline partitioning

Part of §16.2/§16.3 gap closure from the exo/Parallax comparison.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class LayerAllocationStrategy(Enum):
    """Strategy for distributing transformer layers across pipeline nodes."""
    EQUAL = "equal"
    MEMORY_PROPORTIONAL = "memory_proportional"
    BANDWIDTH_AWARE = "bandwidth_aware"
    LATENCY_OPTIMAL = "latency_optimal"


@dataclass
class NodeProfile:
    """Hardware profile of a node relevant to layer allocation.

    All fields use physical units so the allocator can reason about
    real capacity constraints.
    """
    node_id: str
    memory_bytes: int = 0
    bandwidth_mbps: float = 0.0
    latency_ms: float = 0.0
    gpu_cores: int = 0

    # Weights for capacity_score.  Defaults follow whitepaper §16.3:
    # memory is dominant, bandwidth matters for inter-node traffic,
    # GPU cores provide compute headroom.
    _weight_memory: float = field(default=0.6, repr=False)
    _weight_bandwidth: float = field(default=0.25, repr=False)
    _weight_gpu: float = field(default=0.15, repr=False)

    def capacity_score(self) -> float:
        """Weighted capacity metric in normalized [0, 1+] range.

        Uses log-scale normalization so a node with 2x memory doesn't
        completely dominate one with 1x — the relationship is sub-linear.
        """
        mem_part = math.log1p(self.memory_bytes) if self.memory_bytes > 0 else 0.0
        bw_part = math.log1p(self.bandwidth_mbps) if self.bandwidth_mbps > 0 else 0.0
        gpu_part = math.log1p(self.gpu_cores) if self.gpu_cores > 0 else 0.0

        return (
            self._weight_memory * mem_part
            + self._weight_bandwidth * bw_part
            + self._weight_gpu * gpu_part
        )


@dataclass
class StageAllocation:
    """Allocation result for a single pipeline stage."""
    node_id: str
    start_layer: int
    end_layer: int
    num_layers: int


@dataclass
class AllocationStats:
    """Quality metrics for a layer allocation."""
    total_layers: int
    num_stages: int
    balance_ratio: float  # min_layers / max_layers (1.0 = perfect)
    max_stage_layers: int
    min_stage_layers: int
    strategy: str
    stages: list[StageAllocation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_layers": self.total_layers,
            "num_stages": self.num_stages,
            "balance_ratio": round(self.balance_ratio, 4),
            "max_stage_layers": self.max_stage_layers,
            "min_stage_layers": self.min_stage_layers,
            "strategy": self.strategy,
        }


class WaterFillingRebalancer:
    """Incrementally adjusts layer allocations when topology changes.

    Moves layers from over-provisioned to under-provisioned nodes using
    a water-filling approach.  Only transfers layers if the improvement
    exceeds a configurable threshold.

    Algorithm:
    1. Compute each node's ideal share = total_layers * (capacity_i / sum_capacities)
    2. Identify donors (over-provisioned) and acceptors (under-provisioned)
    3. Transfer layers one-at-a-time (granularity=1) or in chunks
    4. Stop when improvement < threshold or no more transfers help
    """

    def __init__(
        self,
        transfer_granularity: int = 1,
        improvement_threshold: float = 0.01,
    ):
        self._granularity = max(1, transfer_granularity)
        self._threshold = improvement_threshold

    def rebalance(
        self,
        current_alloc: list[StageAllocation],
        nodes: list[NodeProfile],
        total_layers: int,
    ) -> list[StageAllocation]:
        """Rebalance current allocation to reflect changed node topology.

        Args:
            current_alloc: Current stage allocations (may be stale).
            nodes: Updated node profiles (may have new/removed nodes).
            total_layers: Total number of layers to distribute.

        Returns:
            New list of StageAllocation reflecting the updated topology.
        """
        if not nodes or total_layers <= 0:
            return []

        num_nodes = len(nodes)
        total_capacity = sum(n.capacity_score() for n in nodes)

        if total_capacity <= 0:
            # All nodes have zero capacity — fall back to equal split
            return self._equal_fallback(nodes, total_layers)

        # Compute ideal layer counts
        ideal = []
        for node in nodes:
            share = total_layers * (node.capacity_score() / total_capacity)
            ideal.append(share)

        # Start from current allocation mapped to surviving nodes
        alloc_map: dict[str, int] = {}
        for sa in current_alloc:
            alloc_map[sa.node_id] = sa.num_layers

        # Initialize allocation: keep current if node still exists, else 0
        current = []
        for node in nodes:
            current.append(alloc_map.get(node.node_id, 0))

        # Ensure total is correct — scale proportionally if nodes changed
        current_total = sum(current)
        if current_total != total_layers:
            # Redistribute excess/deficit proportionally
            diff = total_layers - current_total
            self._adjust_total(current, ideal, diff)

        # Water-filling: move layers from donors to acceptors
        improved = True
        max_iterations = total_layers * 2  # safety bound
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Find most under-provisioned and most over-provisioned
            donors = []  # (index, excess)
            acceptors = []  # (index, deficit)

            for i, node in enumerate(nodes):
                diff = ideal[i] - current[i]
                if diff >= self._threshold:
                    acceptors.append((i, diff))
                elif diff <= -self._threshold:
                    donors.append((i, -diff))

            if not donors or not acceptors:
                break

            # Transfer from largest donor to largest acceptor
            donors.sort(key=lambda x: x[1], reverse=True)
            acceptors.sort(key=lambda x: x[1], reverse=True)

            donor_idx, donor_excess = donors[0]
            acceptor_idx, acceptor_deficit = acceptors[0]

            transfer = min(
                self._granularity,
                int(donor_excess),
                int(acceptor_deficit),
            )

            if transfer <= 0:
                break

            # Verify donor has layers to give
            if current[donor_idx] - transfer < 1:
                transfer = current[donor_idx] - 1
                if transfer <= 0:
                    break

            current[donor_idx] -= transfer
            current[acceptor_idx] += transfer
            improved = True

        # Build new StageAllocation list
        return self._build_stages(nodes, current, total_layers)

    def _adjust_total(
        self,
        current: list[int],
        ideal: list[float],
        diff: int,
    ) -> None:
        """Adjust current allocations to match total_layers."""
        if diff == 0:
            return

        if diff > 0:
            # Add layers to most under-provisioned
            for _ in range(diff):
                worst_idx = -1
                worst_gap = float("-inf")
                for i in range(len(current)):
                    gap = ideal[i] - current[i]
                    if gap > worst_gap:
                        worst_gap = gap
                        worst_idx = i
                if worst_idx >= 0:
                    current[worst_idx] += 1
        else:
            # Remove layers from most over-provisioned
            for _ in range(-diff):
                worst_idx = -1
                worst_gap = float("-inf")
                for i in range(len(current)):
                    gap = current[i] - ideal[i]
                    if gap > worst_gap:
                        worst_gap = gap
                        worst_idx = i
                if worst_idx >= 0:
                    current[worst_idx] = max(0, current[worst_idx] - 1)

    def _equal_fallback(
        self,
        nodes: list[NodeProfile],
        total_layers: int,
    ) -> list[StageAllocation]:
        """Fall back to equal split when no capacity info available."""
        n = len(nodes)
        base = total_layers // n
        remainder = total_layers % n
        current = []
        for i in range(n):
            extra = 1 if i < remainder else 0
            current.append(base + extra)
        return self._build_stages(nodes, current, total_layers)

    def _build_stages(
        self,
        nodes: list[NodeProfile],
        layer_counts: list[int],
        total_layers: int,
    ) -> list[StageAllocation]:
        """Convert a list of layer counts into StageAllocation objects."""
        stages = []
        offset = 0
        for i, node in enumerate(nodes):
            count = layer_counts[i]
            stages.append(StageAllocation(
                node_id=node.node_id,
                start_layer=offset,
                end_layer=offset + count,
                num_layers=count,
            ))
            offset += count
        return stages


class LayerAllocator:
    """Allocates transformer layers across pipeline-parallel nodes.

    Supports four allocation strategies, from simple to sophisticated:
    - EQUAL: Baseline equal split (original auto_partition_model behavior)
    - MEMORY_PROPORTIONAL: Layers proportional to available memory (exo pattern)
    - BANDWIDTH_AWARE: Accounts for inter-node bandwidth on pipeline links
    - LATENCY_OPTIMAL: DP that minimizes max stage latency (Parallax pattern)

    Usage:
        allocator = LayerAllocator()
        nodes = [NodeProfile("n0", memory_bytes=64*GB, ...), ...]
        stages = allocator.allocate(32, nodes, LayerAllocationStrategy.MEMORY_PROPORTIONAL)
        stats = allocator.get_stats()
    """

    def __init__(self) -> None:
        self._last_alloc: Optional[list[StageAllocation]] = None
        self._last_stats: Optional[AllocationStats] = None
        self._last_strategy: Optional[LayerAllocationStrategy] = None
        self._rebalancer = WaterFillingRebalancer()

    def allocate(
        self,
        num_layers: int,
        nodes: list[NodeProfile],
        strategy: LayerAllocationStrategy = LayerAllocationStrategy.MEMORY_PROPORTIONAL,
    ) -> list[StageAllocation]:
        """Main allocation entry point.

        Args:
            num_layers: Total transformer layers to distribute.
            nodes: Hardware profiles for each pipeline node.
            strategy: Which allocation strategy to use.

        Returns:
            List of StageAllocation, one per node, in pipeline order.

        Raises:
            ValueError: If num_layers <= 0 or nodes is empty.
        """
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if not nodes:
            raise ValueError("nodes must be non-empty")
        if len(nodes) > num_layers:
            logger.warning(
                f"More nodes ({len(nodes)}) than layers ({num_layers}); "
                "some nodes will get 0 layers"
            )

        # Check if any node has meaningful profile data
        has_profiles = any(n.memory_bytes > 0 or n.gpu_cores > 0 for n in nodes)

        # Backward compatibility: if no profiles, fall back to EQUAL
        if not has_profiles and strategy != LayerAllocationStrategy.EQUAL:
            logger.info(
                f"No node profiles available; falling back to EQUAL from {strategy.value}"
            )
            effective_strategy = LayerAllocationStrategy.EQUAL
        else:
            effective_strategy = strategy

        dispatch = {
            LayerAllocationStrategy.EQUAL: self._allocate_equal,
            LayerAllocationStrategy.MEMORY_PROPORTIONAL: self._allocate_memory_proportional,
            LayerAllocationStrategy.BANDWIDTH_AWARE: self._allocate_bandwidth_aware,
            LayerAllocationStrategy.LATENCY_OPTIMAL: self._allocate_latency_optimal,
        }
        allocator_fn = dispatch[effective_strategy]
        result = allocator_fn(num_layers, nodes)

        self._last_alloc = result
        self._last_strategy = effective_strategy
        self._last_stats = self._compute_stats(num_layers, result, effective_strategy)

        return result

    def rebalance(
        self,
        current_alloc: list[StageAllocation],
        changed_nodes: list[NodeProfile],
        total_layers: int,
    ) -> list[StageAllocation]:
        """Incrementally rebalance when nodes join or leave.

        Delegates to WaterFillingRebalancer for the actual adjustment.
        """
        result = self._rebalancer.rebalance(current_alloc, changed_nodes, total_layers)
        self._last_alloc = result
        strategy = self._last_strategy or LayerAllocationStrategy.EQUAL
        self._last_stats = self._compute_stats(total_layers, result, strategy)
        return result

    def get_stats(self) -> Optional[AllocationStats]:
        """Return quality metrics from the most recent allocation."""
        return self._last_stats

    # ------------------------------------------------------------------
    # Allocation strategies
    # ------------------------------------------------------------------

    def _allocate_equal(
        self,
        num_layers: int,
        nodes: list[NodeProfile],
    ) -> list[StageAllocation]:
        """Equal split — each node gets (roughly) the same number of layers.

        Remainder layers are distributed one-per-node to the first nodes.
        This is the original auto_partition_model behavior.
        """
        n = len(nodes)
        base = num_layers // n
        remainder = num_layers % n

        stages = []
        offset = 0
        for i, node in enumerate(nodes):
            extra = 1 if i < remainder else 0
            count = base + extra
            stages.append(StageAllocation(
                node_id=node.node_id,
                start_layer=offset,
                end_layer=offset + count,
                num_layers=count,
            ))
            offset += count

        return stages

    def _allocate_memory_proportional(
        self,
        num_layers: int,
        nodes: list[NodeProfile],
    ) -> list[StageAllocation]:
        """Distribute layers proportionally to each node's available memory.

        Follows exo's allocate_layers_proportionally pattern:
        layers_i = floor(num_layers * mem_i / total_mem)
        Remaining layers go to nodes with largest rounding deficit.
        """
        total_mem = sum(n.memory_bytes for n in nodes)

        if total_mem <= 0:
            return self._allocate_equal(num_layers, nodes)

        # Compute raw proportional shares
        raw_shares = [num_layers * n.memory_bytes / total_mem for n in nodes]
        floored = [int(s) for s in raw_shares]
        remainders = [raw_shares[i] - floored[i] for i in range(len(nodes))]

        # Ensure every node gets at least 1 layer (if possible)
        for i in range(len(nodes)):
            if floored[i] < 1 and num_layers >= len(nodes):
                floored[i] = 1

        assigned = sum(floored)
        leftover = num_layers - assigned

        # Distribute leftover to nodes with largest fractional remainder
        if leftover > 0:
            indices_by_remainder = sorted(
                range(len(nodes)),
                key=lambda i: remainders[i],
                reverse=True,
            )
            for idx in indices_by_remainder:
                if leftover <= 0:
                    break
                floored[idx] += 1
                leftover -= 1

        # Clamp: ensure total doesn't exceed num_layers
        while sum(floored) > num_layers:
            # Remove from node with most layers
            max_idx = max(range(len(nodes)), key=lambda i: floored[i])
            if floored[max_idx] > 1:
                floored[max_idx] -= 1

        return self._build_stages(nodes, floored)

    def _allocate_bandwidth_aware(
        self,
        num_layers: int,
        nodes: list[NodeProfile],
    ) -> list[StageAllocation]:
        """Bandwidth-aware allocation that considers inter-node link speed.

        Pipeline stages that are neighbors should be sized to account for
        the activation transfer cost between them.  Stages connected by
        low-bandwidth links should be larger (fewer transfers), while
        stages connected by high-bandwidth links can be smaller.

        Algorithm:
        1. Start with memory-proportional allocation
        2. For each pipeline boundary, compute transfer penalty
        3. Shift layers toward nodes with lower downstream bandwidth
        """
        # Start with memory-proportional as base
        base_alloc = self._allocate_memory_proportional(num_layers, nodes)
        counts = [s.num_layers for s in base_alloc]

        if len(nodes) <= 1:
            return base_alloc

        # Compute bandwidth-weighted adjustments
        # Higher bandwidth -> fewer layers needed (more transfers are cheap)
        # Lower bandwidth -> more layers (avoid transfers)
        total_bw = sum(n.bandwidth_mbps for n in nodes)
        if total_bw <= 0:
            # No bandwidth info — fall back to memory-proportional
            return base_alloc

        # Normalize bandwidth: nodes with low BW get more layers
        # Use inverse-bw weighting
        inv_weights = []
        for node in nodes:
            bw = max(node.bandwidth_mbps, 1.0)
            inv_weights.append(1.0 / bw)

        total_inv = sum(inv_weights)
        bw_alloc = [int(num_layers * w / total_inv) for w in inv_weights]

        # Ensure at least 1 per node
        for i in range(len(bw_alloc)):
            if bw_alloc[i] < 1 and num_layers >= len(nodes):
                bw_alloc[i] = 1

        # Blend: 70% memory-proportional + 30% bandwidth-weighted
        blended = []
        for i in range(len(nodes)):
            blended_count = int(0.7 * counts[i] + 0.3 * bw_alloc[i])
            blended.append(max(1, blended_count) if num_layers >= len(nodes) else blended_count)

        # Adjust total to match num_layers
        self._fix_total(blended, num_layers, len(nodes))

        return self._build_stages(nodes, blended)

    def _allocate_latency_optimal(
        self,
        num_layers: int,
        nodes: list[NodeProfile],
    ) -> list[StageAllocation]:
        """Parallax-style DP allocation minimizing max stage latency.

        The latency of a pipeline stage depends on:
        - Computation: proportional to num_layers / gpu_cores
        - Communication: proportional to latency_ms (inter-node hop cost)

        We use DP to find the partition that minimizes the maximum
        stage latency across all stages.

        DP formulation:
          dp[k][j] = min max stage latency using k stages for first j layers
          transition: dp[k][j] = min over p of max(dp[k-1][p], stage_latency(p..j, node_k))
        """
        n = len(nodes)
        if n == 1:
            return [StageAllocation(
                node_id=nodes[0].node_id,
                start_layer=0,
                end_layer=num_layers,
                num_layers=num_layers,
            )]

        # Compute per-layer, per-node compute time (ms)
        # Approximate: compute_time = num_layers * (base_compute / gpu_cores)
        # Communication cost per stage = latency_ms (fixed hop cost)
        INF = float("inf")

        def stage_cost(layer_start: int, layer_end: int, node: NodeProfile) -> float:
            """Estimated latency for a pipeline stage."""
            n_layers = layer_end - layer_start
            if n_layers <= 0:
                return 0.0
            # Compute cost: inversely proportional to GPU cores
            compute = n_layers * (1000.0 / max(node.gpu_cores, 1))
            # Communication cost: inter-node latency
            comm = node.latency_ms
            return compute + comm

        # dp[k][j] = min over all partitions of first j layers into k stages
        # of the maximum stage latency
        # k ranges from 0..n-1 (stage index), j ranges from 0..num_layers
        dp = [[INF] * (num_layers + 1) for _ in range(n + 1)]
        split = [[0] * (num_layers + 1) for _ in range(n + 1)]

        # Base case: 0 stages, 0 layers = 0 latency
        dp[0][0] = 0.0

        for k in range(1, n + 1):
            node = nodes[k - 1]
            for j in range(1, num_layers + 1):
                # Try all possible split points for stage k
                # Stage k covers layers p..j
                for p in range(0, j):
                    cost = stage_cost(p, j, node)
                    candidate = max(dp[k - 1][p], cost)
                    if candidate < dp[k][j]:
                        dp[k][j] = candidate
                        split[k][j] = p

        # Backtrack to find the partition
        # Walk backwards from dp[n][num_layers]
        boundaries = []
        j = num_layers
        for k in range(n, 0, -1):
            p = split[k][j]
            boundaries.append((p, j))
            j = p

        boundaries.reverse()

        # If we didn't use all stages (e.g. not enough layers), redistribute
        while len(boundaries) < n:
            # Split the largest stage
            max_idx = max(range(len(boundaries)), key=lambda i: boundaries[i][1] - boundaries[i][0])
            s, e = boundaries[max_idx]
            mid = (s + e) // 2
            if mid == s:
                break
            boundaries[max_idx] = (s, mid)
            boundaries.insert(max_idx + 1, (mid, e))

        # Build allocations
        stages = []
        for i, (start, end) in enumerate(boundaries):
            if i < n:
                stages.append(StageAllocation(
                    node_id=nodes[i].node_id,
                    start_layer=start,
                    end_layer=end,
                    num_layers=end - start,
                ))

        # If we have more stages than nodes (shouldn't happen), truncate
        stages = stages[:n]

        return stages

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_stages(
        self,
        nodes: list[NodeProfile],
        layer_counts: list[int],
    ) -> list[StageAllocation]:
        """Convert a list of layer counts into StageAllocation objects."""
        stages = []
        offset = 0
        for i, node in enumerate(nodes):
            count = layer_counts[i] if i < len(layer_counts) else 0
            stages.append(StageAllocation(
                node_id=node.node_id,
                start_layer=offset,
                end_layer=offset + count,
                num_layers=count,
            ))
            offset += count
        return stages

    def _fix_total(
        self,
        counts: list[int],
        total: int,
        min_per_node: int = 1,
    ) -> None:
        """Adjust layer counts in-place so they sum to total."""
        current = sum(counts)
        if current == total:
            return

        diff = total - current
        if diff > 0:
            # Add to nodes with fewest layers
            for _ in range(diff):
                min_idx = min(range(len(counts)), key=lambda i: counts[i])
                counts[min_idx] += 1
        else:
            # Remove from nodes with most layers (but keep >= min_per_node)
            for _ in range(-diff):
                max_idx = max(
                    range(len(counts)),
                    key=lambda i: counts[i],
                )
                if counts[max_idx] > min_per_node:
                    counts[max_idx] -= 1

    def _compute_stats(
        self,
        num_layers: int,
        stages: list[StageAllocation],
        strategy: Optional[LayerAllocationStrategy],
    ) -> AllocationStats:
        """Compute quality metrics for the allocation."""
        strategy_str = strategy.value if strategy else "unknown"

        if not stages:
            return AllocationStats(
                total_layers=num_layers,
                num_stages=0,
                balance_ratio=0.0,
                max_stage_layers=0,
                min_stage_layers=0,
                strategy=strategy_str,
            )

        layer_counts = [s.num_layers for s in stages if s.num_layers > 0]
        if not layer_counts:
            layer_counts = [0]

        return AllocationStats(
            total_layers=num_layers,
            num_stages=len(stages),
            balance_ratio=min(layer_counts) / max(layer_counts) if max(layer_counts) > 0 else 0.0,
            max_stage_layers=max(layer_counts),
            min_stage_layers=min(layer_counts),
            strategy=strategy_str,
            stages=stages,
        )
