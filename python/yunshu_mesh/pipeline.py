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

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx

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


def auto_partition_model(
    num_layers: int,
    num_nodes: int,
    node_memory_gb: list[float],
    model_memory_per_layer_gb: float,
) -> PipelineParallel:
    """Auto-partition a model across nodes based on available memory.

    Follows Megatron-LM's pipeline partitioning but accounts for
    Apple Silicon's UMA (memory is shared, not device-specific).

    Args:
        num_layers: Total transformer layers.
        num_nodes: Number of available nodes.
        node_memory_gb: Memory available on each node (GB).
        model_memory_per_layer_gb: Memory per layer (GB).

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

    # Distribute layers proportionally to memory
    layer_allocations = []
    remaining_layers = num_layers
    for i, mem_gb in enumerate(node_memory_gb):
        if i == num_nodes - 1:
            # Last node gets all remaining layers
            layer_allocations.append(remaining_layers)
        else:
            proportion = mem_gb / total_available_gb
            alloc = max(1, int(num_layers * proportion))
            alloc = min(alloc, remaining_layers - (num_nodes - i - 1))
            layer_allocations.append(alloc)
            remaining_layers -= alloc

    # Build stages from allocations
    pp = PipelineParallel(num_layers, num_nodes)
    current = 0
    for i, n_layers in enumerate(layer_allocations):
        pp._stages[i] = PipelineStage(
            stage_id=i,
            start_layer=current,
            end_layer=current + n_layers,
            rank=i,
        )
        current += n_layers

    return pp
