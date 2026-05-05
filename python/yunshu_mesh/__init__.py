"""Yunshu L3 Compute Mesh — mx.distributed integration for Apple Silicon clusters.

Architecture (whitepaper §4.3):
  - Node discovery via mDNS/DNS-SD on Thunderbolt 5 / Ethernet
  - Transport: JACCL (TB5 direct), Ring (TCP fallback), MPI (HPC)
  - Topologies: Ring, Fully-Connected Mesh, Pipeline Parallel
  - Collective ops: all_reduce, all_gather, send/recv (via mx.distributed)

Design follows SGLang's parallel_state + vLLM's distributed architecture
but adapted for mx.distributed (not torch.distributed).

mx.distributed API (verified on MLX 0.31+):
  - init(backend='any'|'mpi'|'nccl'|'jaccl'|'ring') -> Group
  - all_sum(x, group=None) -> array
  - all_gather(x, group=None) -> array
  - all_max(x, group=None) -> array
  - all_min(x, group=None) -> array
  - sum_scatter(x, group=None) -> array
  - send(x, dst, group=None) -> array
  - recv(shape, dtype, src, group=None) -> array
  - Group.rank() -> int
  - Group.size() -> int
  - Group.split(key) -> Group
"""

from .node import MeshNode, MeshNodeState
from .topology import MeshTopology, TopologyType
from .collective import CollectiveOps
from .pipeline import PipelineStage, PipelineParallel
from .manager import MeshManager

__all__ = [
    "CollectiveOps",
    "MeshManager",
    "MeshNode",
    "MeshNodeState",
    "MeshTopology",
    "PipelineParallel",
    "PipelineStage",
    "TopologyType",
]
