from __future__ import annotations
"""Yunshu Mesh — Pipeline parallelism for multi-node inference.

Splits a transformer model across N nodes where each node handles
a contiguous set of layers. Activations flow through the pipeline
via mx.distributed send/recv.

Follows vLLM's pipeline parallel + SGLang's model parallel patterns
but using mx.distributed point-to-point communication.

Pipeline parallelism is ideal for:
- Models too large for a single node (e.g., 70B across 4×M3 Ultra)
- Long sequence generation where KV cache exceeds single-node memory
- Latency-tolerant batch workloads
"""


import logging
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx

from .layer_allocator import (
    LayerAllocationStrategy,
    LayerAllocator,
    NodeProfile,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """One stage of a pipeline-parallel model.

    Each stage is responsible for a contiguous range of transformer layers.
    """
    stage_id: int
    start_layer: int
    end_layer: int
    rank: int  # Which node this stage runs on
    _is_last: bool = field(init=False, default=False)

    @property
    def num_layers(self) -> int:
        return self.end_layer - self.start_layer

    def is_first(self) -> bool:
        return self.start_layer == 0

    @property
    def is_last(self) -> bool:
        return self._is_last

    @is_last.setter
    def is_last(self, value: bool) -> None:
        self._is_last = value


class PipelineParallel:
    """Manages pipeline-parallel execution across mesh nodes.

    Splits a model's layers into stages and coordinates the flow of
    activations between stages using mx.distributed send/recv.

    Supports:
    - GPipe-style synchronous pipeline (simple, higher latency)
    - 1F1B (one forward one backward) for training (future)
    """

    def __init__(
        self,
        num_layers: int,
        num_stages: int,
        collective: Optional[object] = None,
    ):
        """
        Args:
            num_layers: Total number of transformer layers in the model.
            num_stages: Number of pipeline stages (= number of nodes).
            collective: CollectiveOps instance for send/recv.
        """
        self._num_layers = num_layers
        self._num_stages = num_stages
        self._collective = collective
        self._stages: list[PipelineStage] = []
        self._build_stages()

    def _build_stages(self) -> None:
        """Split layers into roughly equal stages."""
        layers_per_stage = self._num_layers // self._num_stages
        remainder = self._num_layers % self._num_stages

        current_layer = 0
        for i in range(self._num_stages):
            extra = 1 if i < remainder else 0
            stage_layers = layers_per_stage + extra
            stage = PipelineStage(
                stage_id=i,
                start_layer=current_layer,
                end_layer=current_layer + stage_layers,
                rank=i,
            )
            stage.is_last = (stage.end_layer == self._num_layers)
            self._stages.append(stage)
            current_layer += stage_layers

    @property
    def stages(self) -> list[PipelineStage]:
        return list(self._stages)

    def get_stage(self, rank: int) -> Optional[PipelineStage]:
        for stage in self._stages:
            if stage.rank == rank:
                return stage
        return None

    @property
    def num_stages(self) -> int:
        return self._num_stages

    def send_activations(
        self,
        hidden_states: mx.array,
        dst_rank: int,
    ) -> None:
        """Send hidden states to the next pipeline stage."""
        if self._collective is None:
            return
        self._collective.send(hidden_states, dst_rank)

    def recv_activations(
        self,
        src_rank: int,
        shape: tuple[int, ...],
        dtype: mx.Dtype = mx.float16,
    ) -> mx.array:
        """Receive hidden states from the previous pipeline stage."""
        if self._collective is None:
            return mx.zeros(shape, dtype=dtype)
        return self._collective.recv(shape, dtype, src_rank)

    def pipeline_forward_step(
        self,
        stage: PipelineStage,
        hidden_states: mx.array,
        model_layers: list,
        cache: Optional[list] = None,
    ) -> mx.array:
        """Run forward through one pipeline stage's layers.

        Args:
            stage: The pipeline stage to execute.
            hidden_states: Input activations (from previous stage or embedding).
            model_layers: All model layers (we slice by stage range).
            cache: Optional KV cache for each layer.

        Returns:
            Output activations after this stage's layers.
        """
        for i in range(stage.start_layer, stage.end_layer):
            layer = model_layers[i]
            if cache is not None and i < len(cache):
                hidden_states = layer(hidden_states, cache[i])
            else:
                hidden_states = layer(hidden_states)
        return hidden_states

    def to_dict(self) -> dict:
        return {
            "num_layers": self._num_layers,
            "num_stages": self._num_stages,
            "stages": [
                {
                    "stage_id": s.stage_id,
                    "start_layer": s.start_layer,
                    "end_layer": s.end_layer,
                    "rank": s.rank,
                    "num_layers": s.num_layers,
                }
                for s in self._stages
            ],
        }


def _build_node_profiles(
    node_memory_gb: list[float],
) -> list[NodeProfile]:
    """Convert legacy node_memory_gb list to NodeProfile objects.

    Preserves backward compatibility for callers that only provide memory info.
    """
    profiles = []
    for i, mem_gb in enumerate(node_memory_gb):
        profiles.append(NodeProfile(
            node_id=f"node_{i}",
            memory_bytes=int(mem_gb * (1024 ** 3)),
            bandwidth_mbps=0.0,
            latency_ms=0.0,
            gpu_cores=0,
        ))
    return profiles


def auto_partition_model(
    num_layers: int,
    num_nodes: int,
    node_memory_gb: list[float],
    model_memory_per_layer_gb: float,
    strategy: LayerAllocationStrategy = LayerAllocationStrategy.MEMORY_PROPORTIONAL,
    node_bandwidths_mbps: Optional[list[float]] = None,
    node_latencies_ms: Optional[list[float]] = None,
    node_gpu_cores: Optional[list[int]] = None,
) -> PipelineParallel:
    """Auto-partition a model across nodes using intelligent layer allocation.

    Follows Megatron-LM's pipeline partitioning but accounts for
    Apple Silicon's UMA (memory is shared, not device-specific).

    Now uses LayerAllocator with configurable strategies:
    - EQUAL: simple equal split (legacy behavior)
    - MEMORY_PROPORTIONAL: layers proportional to node memory (exo pattern)
    - BANDWIDTH_AWARE: considers inter-node bandwidth (default)
    - LATENCY_OPTIMAL: DP minimizing max stage latency (Parallax pattern)

    Backward compatible: if only node_memory_gb is provided and no profiles
    can be built, falls back to EQUAL.

    Args:
        num_layers: Total transformer layers.
        num_nodes: Number of available nodes.
        node_memory_gb: Memory available on each node (GB).
        model_memory_per_layer_gb: Memory per layer (GB).
        strategy: Allocation strategy (default: MEMORY_PROPORTIONAL).
        node_bandwidths_mbps: Optional inter-node bandwidth for each node.
        node_latencies_ms: Optional inter-node latency for each node.
        node_gpu_cores: Optional GPU core count for each node.

    Returns:
        PipelineParallel with optimal stage assignments.
    """
    total_model_gb = num_layers * model_memory_per_layer_gb
    total_available_gb = sum(node_memory_gb)

    if total_model_gb > total_available_gb:
        logger.warning(
            f"Model requires {total_model_gb:.1f} GB but only "
            f"{total_available_gb:.1f} GB available across {num_nodes} nodes"
        )

    # Build NodeProfile objects from available information
    profiles = []
    for i, mem_gb in enumerate(node_memory_gb):
        bw = node_bandwidths_mbps[i] if node_bandwidths_mbps and i < len(node_bandwidths_mbps) else 0.0
        lat = node_latencies_ms[i] if node_latencies_ms and i < len(node_latencies_ms) else 0.0
        cores = node_gpu_cores[i] if node_gpu_cores and i < len(node_gpu_cores) else 0
        profiles.append(NodeProfile(
            node_id=f"node_{i}",
            memory_bytes=int(mem_gb * (1024 ** 3)),
            bandwidth_mbps=bw,
            latency_ms=lat,
            gpu_cores=cores,
        ))

    # Use LayerAllocator
    allocator = LayerAllocator()
    stage_allocs = allocator.allocate(num_layers, profiles, strategy)

    # Build PipelineParallel directly from allocation results (skip
    # PipelineParallel.__init__'s _build_stages to avoid wasted work).
    pp = object.__new__(PipelineParallel)
    pp._num_layers = num_layers
    pp._num_stages = num_nodes
    pp._collective = None
    pp._stages = []
    for i, sa in enumerate(stage_allocs):
        if i >= num_nodes:
            break
        stage = PipelineStage(
            stage_id=i,
            start_layer=sa.start_layer,
            end_layer=sa.end_layer,
            rank=i,
        )
        stage.is_last = (sa.end_layer == num_layers)
        pp._stages.append(stage)

    return pp
