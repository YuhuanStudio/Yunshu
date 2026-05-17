"""Yunshu Mesh — Collective operations via mx.distributed.

Wraps mx.distributed collective ops with mesh-aware group management.
Supports Ring AllReduce, AllGather, Send/Recv point-to-point.

Following vLLM's custom_all_reduce + SGLang's communication ops patterns
but using mlx.core.distributed (not torch.distributed).

mx.distributed API:
  init(backend='any'|'jaccl'|'ring'|'mpi') -> Group
  all_sum(x, group=None) -> array
  all_gather(x, group=None) -> array
  all_max/all_min(x, group=None) -> array
  sum_scatter(x, group=None) -> array
  send(x, dst, group=None) -> array
  recv(shape, dtype, src, group=None) -> array
"""


import logging
import time
from typing import Optional, Sequence

import mlx.core as mx

from .topology import MeshTopology

logger = logging.getLogger(__name__)


class CollectiveOps:
    """High-level collective operations for the compute mesh.

    Manages mx.distributed backend initialization and provides
    mesh-aware collective operations.
    """

    def __init__(self, backend: str = "any"):
        self._backend = backend
        self._group: Optional[mx.distributed.Group] = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def rank(self) -> int:
        if self._group is None:
            return 0
        return self._group.rank()

    @property
    def size(self) -> int:
        if self._group is None:
            return 1
        return self._group.size()

    @property
    def group(self) -> Optional[mx.distributed.Group]:
        return self._group

    def initialize(self, backend: Optional[str] = None) -> bool:
        """Initialize mx.distributed backend.

        Args:
            backend: 'any', 'jaccl', 'ring', 'mpi'. Default: auto-detect.
        """
        if self._initialized:
            return True

        backend = backend or self._backend

        try:
            if not mx.distributed.is_available():
                logger.warning("mx.distributed not available — running in single-node mode")
                return False

            self._group = mx.distributed.init(backend=backend)
            self._initialized = True
            logger.info(
                f"Initialized mx.distributed: backend={backend}, "
                f"rank={self.rank}, size={self.size}"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize mx.distributed: {e}")
            return False

    def shutdown(self) -> None:
        """Clean up distributed resources."""
        self._group = None
        self._initialized = False

    # ── Collective Operations ──

    def all_reduce_sum(
        self,
        x: mx.array,
        stream: Optional[mx.Stream] = None,
    ) -> mx.array:
        """All-reduce sum across all nodes.

        Uses mx.distributed.all_sum which automatically handles
        Ring AllReduce or direct exchange depending on backend.
        """
        if not self._initialized:
            return x
        kwargs = {"group": self._group}
        if stream is not None:
            kwargs["stream"] = stream
        return mx.distributed.all_sum(x, **kwargs)

    def all_reduce_mean(
        self,
        x: mx.array,
        stream: Optional[mx.Stream] = None,
    ) -> mx.array:
        """All-reduce mean across all nodes."""
        result = self.all_reduce_sum(x, stream)
        return result / self.size

    def all_gather(
        self,
        x: mx.array,
        stream: Optional[mx.Stream] = None,
    ) -> mx.array:
        """Gather arrays from all nodes.

        Returns array with first dim multiplied by world size.
        """
        if not self._initialized:
            return x
        kwargs = {"group": self._group}
        if stream is not None:
            kwargs["stream"] = stream
        return mx.distributed.all_gather(x, **kwargs)

    def all_max(
        self,
        x: mx.array,
        stream: Optional[mx.Stream] = None,
    ) -> mx.array:
        """All-reduce max across all nodes."""
        if not self._initialized:
            return x
        kwargs = {"group": self._group}
        if stream is not None:
            kwargs["stream"] = stream
        return mx.distributed.all_max(x, **kwargs)

    def send(
        self,
        x: mx.array,
        dst: int,
        stream: Optional[mx.Stream] = None,
    ) -> None:
        """Send array to a specific rank."""
        if not self._initialized:
            return
        kwargs = {"dst": dst, "group": self._group}
        if stream is not None:
            kwargs["stream"] = stream
        mx.distributed.send(x, **kwargs)

    def recv(
        self,
        shape: Sequence[int],
        dtype: mx.Dtype,
        src: int,
        stream: Optional[mx.Stream] = None,
    ) -> mx.array:
        """Receive array from a specific rank."""
        if not self._initialized:
            return mx.zeros(shape, dtype=dtype)
        kwargs = {"src": src, "group": self._group}
        if stream is not None:
            kwargs["stream"] = stream
        return mx.distributed.recv(shape, dtype, **kwargs)

    def all_min(
        self,
        x: mx.array,
        stream: Optional[mx.Stream] = None,
    ) -> mx.array:
        """All-reduce min across all nodes."""
        if not self._initialized:
            return x
        kwargs = {"group": self._group}
        if stream is not None:
            kwargs["stream"] = stream
        return mx.distributed.all_min(x, **kwargs)

    def sum_scatter(
        self,
        x: mx.array,
        stream: Optional[mx.Stream] = None,
    ) -> mx.array:
        """Sum-scatter: sum across all nodes, then scatter the result."""
        if not self._initialized:
            return x
        kwargs = {"group": self._group}
        if stream is not None:
            kwargs["stream"] = stream
        return mx.distributed.sum_scatter(x, **kwargs)

    # ── Ring AllReduce (explicit step-by-step) ──

    def ring_all_reduce(
        self,
        x: mx.array,
        topology: MeshTopology,
    ) -> mx.array:
        """Explicit Ring AllReduce for monitoring/debugging.

        Two phases:
        1. Reduce-scatter: N-1 steps, each step sends a chunk to next rank
        2. All-gather: N-1 steps, each step sends a chunk to next rank

        For production, use all_reduce_sum() which uses the optimized backend.
        This explicit version is for diagnostics and benchmarks.
        """
        if topology.size <= 1 or not self._initialized:
            return x

        rank = self.rank
        world_size = self.size
        n = world_size

        # Split tensor into chunks
        chunks = list(mx.split(x, n, axis=0))

        # Phase 1: Reduce-scatter
        # At step s, rank r sends chunk (r-s)%n to (r+1)%n, receives chunk
        # (r-s-1)%n from (r-1)%n, and accumulates into the received chunk.
        for step in range(n - 1):
            send_rank = (rank + 1) % n
            recv_rank = (rank - 1 + n) % n
            send_chunk_idx = (rank - step + n) % n
            recv_chunk_idx = (rank - step - 1 + n) % n

            # Avoid deadlock: even ranks send first, odd ranks recv first.
            if rank % 2 == 0:
                self.send(chunks[send_chunk_idx].astype(mx.float32), send_rank)
                received = self.recv(
                    chunks[recv_chunk_idx].shape,
                    mx.float32,
                    recv_rank,
                )
            else:
                received = self.recv(
                    chunks[recv_chunk_idx].shape,
                    mx.float32,
                    recv_rank,
                )
                self.send(chunks[send_chunk_idx].astype(mx.float32), send_rank)

            # Accumulate received partial sum into the recv chunk index
            chunks[recv_chunk_idx] = chunks[recv_chunk_idx].astype(mx.float32) + received

        # Phase 2: All-gather
        # At step s, rank r sends chunk (r-s+1)%n to (r+1)%n, receives chunk
        # (r-s)%n from (r-1)%n, and overwrites with the received (fully reduced) value.
        for step in range(n - 1):
            send_rank = (rank + 1) % n
            recv_rank = (rank - 1 + n) % n
            send_chunk_idx = (rank - step + 1 + n) % n
            recv_chunk_idx = (rank - step + n) % n

            if rank % 2 == 0:
                self.send(chunks[send_chunk_idx].astype(mx.float32), send_rank)
                received = self.recv(
                    chunks[recv_chunk_idx].shape,
                    mx.float32,
                    recv_rank,
                )
            else:
                received = self.recv(
                    chunks[recv_chunk_idx].shape,
                    mx.float32,
                    recv_rank,
                )
                self.send(chunks[send_chunk_idx].astype(mx.float32), send_rank)

            chunks[recv_chunk_idx] = received

        return mx.concatenate(chunks, axis=0).astype(x.dtype)

    # ── Benchmark ──

    def bench_collective(
        self,
        op_name: str,
        tensor_size: int = 1024 * 1024,
        num_iters: int = 100,
        group: Optional[mx.distributed.Group] = None,
        dtype: mx.Dtype = mx.float32,
    ) -> dict:
        """Benchmark a single collective operation over *num_iters* iterations.

        Measures per-iteration latency and computes avg, p50, p99, and
        effective bandwidth. Works in single-process mode (returns local
        operation latency when mx.distributed is not initialised).

        Args:
            op_name: One of "all_reduce", "all_gather", "send_recv".
            tensor_size: Number of elements in the test tensor.
            num_iters: Number of timed iterations.
            group: mx.distributed Group (uses default if None).
            dtype: Tensor dtype for the benchmark.

        Returns:
            Dict with keys: op, tensor_size, avg_latency_ms, p50_ms,
            p99_ms, bandwidth_gbs, mode.
        """
        group = group or self._group
        x = mx.ones((tensor_size,), dtype=dtype)
        mx.eval(x)

        lats: list[float] = []
        # Compute actual byte size from dtype, not hardcoded float32
        dtype_sizes = {
            mx.float16: 2, mx.float32: 4, mx.float64: 8,
            mx.int8: 1, mx.int32: 4, mx.int64: 8,
            mx.bfloat16: 2,
        }
        elem_size = dtype_sizes.get(dtype, 4)
        nbytes = tensor_size * elem_size

        for _ in range(num_iters):
            t0 = time.monotonic()
            result = self._exec_bench_op(op_name, x, group)
            mx.eval(result)
            lats.append((time.monotonic() - t0) * 1000.0)

        sorted_lats = sorted(lats)
        avg_ms = sum(lats) / len(lats) if lats else 0.0
        p50_ms = sorted_lats[len(sorted_lats) // 2] if sorted_lats else 0.0
        p99_idx = min(int(len(sorted_lats) * 0.99), len(sorted_lats) - 1) if sorted_lats else 0
        p99_ms = sorted_lats[p99_idx] if sorted_lats else 0.0

        # Bandwidth: data volume transferred / time
        data_volume = nbytes * 2 if op_name == "all_reduce" else nbytes
        bandwidth_gbs = (data_volume / (avg_ms / 1000.0)) / (1024**3) if avg_ms > 0 else 0.0

        return {
            "op": op_name,
            "tensor_size": tensor_size,
            "avg_latency_ms": round(avg_ms, 4),
            "p50_ms": round(p50_ms, 4),
            "p99_ms": round(p99_ms, 4),
            "bandwidth_gbs": round(bandwidth_gbs, 3),
            "mode": "distributed" if self._initialized else "single_node",
        }

    def _exec_bench_op(
        self,
        op_name: str,
        x: mx.array,
        group: Optional[mx.distributed.Group],
    ) -> mx.array:
        """Execute a single op for benchmarking (distributed or local)."""
        if not self._initialized:
            # Single-process fallback: local operation
            if op_name == "all_reduce":
                return x + mx.zeros_like(x)
            elif op_name == "all_gather":
                return mx.concatenate([x], axis=0)
            elif op_name == "send_recv":
                return x + mx.zeros_like(x)
            else:
                raise ValueError(f"Unknown collective op: {op_name}")

        kwargs = {"group": group}
        if op_name == "all_reduce":
            return mx.distributed.all_sum(x, **kwargs)
        elif op_name == "all_gather":
            return mx.distributed.all_gather(x, **kwargs)
        elif op_name == "send_recv":
            rank = self.rank
            world = self.size
            dst = (rank + 1) % world
            src = (rank - 1) % world
            mx.distributed.send(x, dst=dst, **kwargs)
            return mx.distributed.recv(x.shape, x.dtype, src=src, **kwargs)
        else:
            raise ValueError(f"Unknown collective op: {op_name}")

    def run_benchmark(self, size: int = 1024 * 1024, dtype: mx.Dtype = mx.float32) -> dict:
        """Run a quick collective ops benchmark.

        Returns latencies for all_reduce, all_gather, and sum_scatter.
        """
        if not self._initialized:
            return {"error": "not initialized"}

        x = mx.ones((size,), dtype=dtype)
        mx.eval(x)

        results = {}

        # all_reduce_sum
        t0 = time.monotonic()
        for _ in range(10):
            r = self.all_reduce_sum(x)
            mx.eval(r)
        t_all_reduce = (time.monotonic() - t0) / 10
        results["all_reduce_sum_ms"] = t_all_reduce * 1000
        results["all_reduce_sum_size"] = size

        # all_gather
        t0 = time.monotonic()
        for _ in range(10):
            r = self.all_gather(x)
            mx.eval(r)
        t_all_gather = (time.monotonic() - t0) / 10
        results["all_gather_ms"] = t_all_gather * 1000

        # sum_scatter
        t0 = time.monotonic()
        for _ in range(10):
            r = self.sum_scatter(x)
            mx.eval(r)
        t_sum_scatter = (time.monotonic() - t0) / 10
        results["sum_scatter_ms"] = t_sum_scatter * 1000

        results["world_size"] = self.size
        results["rank"] = self.rank
        return results
