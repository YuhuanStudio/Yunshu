#!/usr/bin/env python3
"""ANE vs GPU Micro-Benchmark for Phase 0 Platform Validation.

Comprehensive benchmark comparing Apple Neural Engine (ANE) vs GPU performance
for embedding and small-model workloads using MLX and CoreML.

Usage:
    PYTHONPATH=. uv run python scripts/bench_ane.py
    PYTHONPATH=. uv run python scripts/bench_ane.py --output bench_ane_report.json
    PYTHONPATH=. uv run python scripts/bench_ane.py --warmup 5 --iters 100

This script is part of Phase 0 (W0-W2) platform validation. It measures:
  1. Embedding inference: small model forward pass on GPU vs estimated ANE
  2. Linear layer: matmul performance at various dimensions
  3. Transformer layer: attention + FFN block performance

For GPU workloads, it uses mx.array operations directly. For ANE estimates,
it uses CoreML compilation where possible and analytical estimation otherwise.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure project root on sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "python"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("bench_ane")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ANEBenchConfig:
    """Configuration for the ANE vs GPU benchmark.

    Attributes:
        model_sizes: Model sizes in millions of parameters to benchmark.
        seq_lengths: Sequence lengths in tokens to benchmark.
        num_warmup: Number of warmup iterations before timing.
        num_iters: Number of timed iterations for each benchmark.
    """

    model_sizes: list[int] = field(
        default_factory=lambda: [10, 50, 100, 300],
    )
    seq_lengths: list[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512],
    )
    num_warmup: int = 3
    num_iters: int = 50
    quick: bool = False  # skip Linear and Transformer inner sweeps


# ---------------------------------------------------------------------------
# ANE Benchmark
# ---------------------------------------------------------------------------


class ANEBenchmark:
    """Comprehensive ANE vs GPU benchmark for embedding and small-model workloads.

    Benchmarks three categories of operations:
      1. Embedding model inference (embedding lookup + projection)
      2. Linear layer matmul (core compute building block)
      3. Transformer layer (attention + feed-forward network)

    For GPU: uses mx.array operations directly with MLX.
    For ANE: attempts CoreML compilation, falls back to analytical estimation
    based on known ANE characteristics.
    """

    def __init__(self, config: ANEBenchConfig | None = None) -> None:
        self.config = config or ANEBenchConfig()
        self._mlx_available = False
        self._coreml_available = False

        # Try to import MLX
        try:
            import mlx.core as mx

            self._mx = mx
            self._mlx_available = True
        except ImportError:
            self._mx = None

        # Try to import coremltools
        try:
            import coremltools as ct  # type: ignore[import-untyped]

            self._ct = ct
            self._coreml_available = True
        except ImportError:
            self._ct = None

    # ── Single-benchmark methods ────────────────────────────────────────────────

    def bench_embedding_inference(
        self,
        model_size_m: int,
        seq_length: int,
        device: str = "gpu",
    ) -> dict[str, Any]:
        """Benchmark embedding model inference on GPU or ANE.

        Simulates an embedding model forward pass: token embedding lookup
        followed by a few transformer encoder layers and mean pooling.

        Args:
            model_size_m: Model size in millions of parameters.
            seq_length: Input sequence length in tokens.
            device: "gpu" for MLX GPU, "ane" for CoreML ANE.

        Returns:
            dict with keys: model_size_m, seq_length, device, latency_ms,
                            throughput_seq_per_s, status
        """
        result: dict[str, Any] = {
            "model_size_m": model_size_m,
            "seq_length": seq_length,
            "device": device,
            "latency_ms": None,
            "throughput_seq_per_s": None,
            "status": "unknown",
        }

        if device == "gpu":
            if not self._mlx_available:
                result["status"] = "mlx_unavailable"
                return result
            return self._bench_embedding_gpu(model_size_m, seq_length, result)

        if device == "ane":
            if self._coreml_available:
                return self._bench_embedding_ane_coreml(
                    model_size_m, seq_length, result,
                )
            return self._bench_embedding_ane_estimated(
                model_size_m, seq_length, result,
            )

        result["status"] = f"unknown_device_{device}"
        return result

    def _bench_embedding_gpu(
        self,
        model_size_m: int,
        seq_length: int,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Run embedding inference benchmark on GPU via MLX."""
        mx = self._mx

        # Derive dimensions from model size.
        # Typical embedding model: vocab=30000, hidden ~= sqrt(4 * params / num_layers)
        vocab_size = 30000
        hidden_dim = max(64, int((model_size_m * 1e6 * 4 / 6) ** 0.5))
        num_layers = max(1, min(12, model_size_m // 10))

        # Build simple embedding model weights (random, for timing only)
        embedding_table = mx.random.normal(shape=(vocab_size, hidden_dim)) * 0.01
        # Simulate encoder layers: QKV projection + FFN
        q_weight = mx.random.normal(shape=(hidden_dim, hidden_dim)) * 0.01
        k_weight = mx.random.normal(shape=(hidden_dim, hidden_dim)) * 0.01
        v_weight = mx.random.normal(shape=(hidden_dim, hidden_dim)) * 0.01
        ffn_up = mx.random.normal(shape=(hidden_dim, hidden_dim * 4)) * 0.01
        ffn_down = mx.random.normal(shape=(hidden_dim * 4, hidden_dim)) * 0.01

        # Generate random token IDs
        input_ids = mx.random.randint(0, vocab_size, shape=(1, seq_length))

        def _forward() -> Any:
            """Single forward pass of a simplified embedding model."""
            h = embedding_table[input_ids]  # (1, seq_len, hidden)
            for _ in range(num_layers):
                q = h @ q_weight
                k = h @ k_weight
                v = h @ v_weight
                # Simplified attention: element-wise (not full softmax attention)
                attn = (q * k).sum(axis=-1, keepdims=True) / (hidden_dim**0.5)
                h = h + attn * v
                # FFN
                f = mx.maximum(h @ ffn_up, 0)
                h = h + f @ ffn_down
            # Mean pooling
            return h.mean(axis=1)

        # Warmup
        for _ in range(self.config.num_warmup):
            _ = _forward()
            mx.eval(_)

        # Timed iterations
        latencies = []
        for _ in range(self.config.num_iters):
            t0 = time.perf_counter()
            output = _forward()
            mx.eval(output)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        result["latency_ms"] = round(avg_latency, 3)
        result["throughput_seq_per_s"] = round(1000.0 / avg_latency, 2)
        result["status"] = "ok"
        return result

    def _bench_embedding_ane_coreml(
        self,
        model_size_m: int,
        seq_length: int,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Try CoreML ANE compilation for embedding inference."""
        try:
            import numpy as np

            ct = self._ct
            hidden_dim = max(64, int((model_size_m * 1e6 * 4 / 6) ** 0.5))

            # Build a simple MLP model for CoreML tracing
            import torch  # type: ignore[import-untyped]

            class SimpleEmbedding(torch.nn.Module):
                def __init__(self, hidden: int):
                    super().__init__()
                    self.linear1 = torch.nn.Linear(hidden, hidden)
                    self.linear2 = torch.nn.Linear(hidden, hidden)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x = torch.relu(self.linear1(x))
                    return self.linear2(x)

            model = SimpleEmbedding(hidden_dim)
            model.eval()

            sample_input = torch.randn(1, seq_length, hidden_dim)

            traced = torch.jit.trace(model, sample_input)

            coreml_model = ct.convert(
                traced,
                inputs=[
                    ct.TensorType(
                        name="input",
                        shape=(1, seq_length, hidden_dim),
                        dtype=np.float32,
                    )
                ],
                convert_to="mlprogram",
                compute_units=ct.ComputeUnit.ALL,
            )

            # Warmup
            for _ in range(self.config.num_warmup):
                coreml_model.predict({"input": np.random.randn(1, seq_length, hidden_dim).astype(np.float32)})

            # Timed iterations
            latencies = []
            for _ in range(self.config.num_iters):
                inp = np.random.randn(1, seq_length, hidden_dim).astype(np.float32)
                t0 = time.perf_counter()
                coreml_model.predict({"input": inp})
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

            avg_latency = sum(latencies) / len(latencies)
            result["latency_ms"] = round(avg_latency, 3)
            result["throughput_seq_per_s"] = round(1000.0 / avg_latency, 2)
            result["status"] = "ok"
            return result

        except Exception as exc:
            logger.warning(
                "CoreML ANE benchmark failed, falling back to estimation: %s", exc,
            )
            return self._bench_embedding_ane_estimated(
                model_size_m, seq_length, result,
            )

    def _bench_embedding_ane_estimated(
        self,
        model_size_m: int,
        seq_length: int,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Estimate ANE performance based on known characteristics.

        ANE characteristics used for estimation:
        - Peak throughput: ~11 TOPS (M1) to ~38 TOPS (M4 Max) for dense matmul
        - Memory bandwidth: ~68 GB/s (M1) to ~546 GB/s (M4 Max)
        - Optimal for: batch=1, seq_len <= 256, model < 100M params
        - Overhead: ~0.05ms fixed dispatch cost per operation
        """
        from yunshu_engine.ane_embedding import estimate_ane_speedup

        # Estimate based on GPU result if available, or analytical model
        # Analytical model: ANE latency ~= GPU_latency / speedup
        # Use a baseline: small embedding model GPU latency estimate
        # Based on empirical MLX data: ~0.5ms for 10M model at seq_len=32

        base_latency_ms = 0.5  # ms for 10M model at seq_len=32 on GPU
        # Scale by model size and sequence length
        gpu_estimated_ms = base_latency_ms * (model_size_m / 10.0) * (seq_length / 32.0) ** 0.7

        speedup = estimate_ane_speedup(float(model_size_m), seq_length)
        ane_latency_ms = gpu_estimated_ms / speedup

        result["latency_ms"] = round(ane_latency_ms, 3)
        result["throughput_seq_per_s"] = round(1000.0 / ane_latency_ms, 2) if ane_latency_ms > 0 else 0
        result["status"] = "estimated"
        return result

    def bench_linear_layer(
        self,
        in_dim: int,
        out_dim: int,
        seq_length: int,
        device: str = "gpu",
    ) -> dict[str, Any]:
        """Benchmark a linear layer (matmul + bias) on GPU or ANE.

        Args:
            in_dim: Input dimension.
            out_dim: Output dimension.
            seq_length: Sequence length (batch dimension).
            device: "gpu" for MLX GPU, "ane" for CoreML ANE.

        Returns:
            dict with keys: in_dim, out_dim, seq_length, device,
                            latency_ms, gflops, status
        """
        result: dict[str, Any] = {
            "in_dim": in_dim,
            "out_dim": out_dim,
            "seq_length": seq_length,
            "device": device,
            "latency_ms": None,
            "gflops": None,
            "status": "unknown",
        }

        if device == "gpu":
            if not self._mlx_available:
                result["status"] = "mlx_unavailable"
                return result

            mx = self._mx
            weight = mx.random.normal(shape=(in_dim, out_dim)) * 0.01
            bias = mx.zeros(shape=(out_dim,))
            x = mx.random.normal(shape=(1, seq_length, in_dim))

            def _forward() -> Any:
                return x @ weight + bias

            # Warmup
            for _ in range(self.config.num_warmup):
                _ = _forward()
                mx.eval(_)

            # Timed
            latencies = []
            for _ in range(self.config.num_iters):
                t0 = time.perf_counter()
                out = _forward()
                mx.eval(out)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

            avg_latency = sum(latencies) / len(latencies)
            flops = 2.0 * seq_length * in_dim * out_dim
            gflops = flops / (avg_latency * 1e-3) / 1e9

            result["latency_ms"] = round(avg_latency, 3)
            result["gflops"] = round(gflops, 2)
            result["status"] = "ok"
            return result

        if device == "ane":
            if self._coreml_available:
                return self._bench_linear_ane_coreml(
                    in_dim, out_dim, seq_length, result,
                )
            return self._bench_linear_ane_estimated(
                in_dim, out_dim, seq_length, result,
            )

        result["status"] = f"unknown_device_{device}"
        return result

    def _bench_linear_ane_coreml(
        self,
        in_dim: int,
        out_dim: int,
        seq_length: int,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Try CoreML ANE for linear layer."""
        try:
            import numpy as np

            ct = self._ct

            class LinearModule:
                """Minimal linear model for CoreML tracing."""
                pass

            import torch  # type: ignore[import-untyped]

            model = torch.nn.Linear(in_dim, out_dim)
            model.eval()

            sample_input = torch.randn(1, seq_length, in_dim)
            traced = torch.jit.trace(model, sample_input)

            coreml_model = ct.convert(
                traced,
                inputs=[
                    ct.TensorType(
                        name="input",
                        shape=(1, seq_length, in_dim),
                        dtype=np.float32,
                    )
                ],
                convert_to="mlprogram",
                compute_units=ct.ComputeUnit.ALL,
            )

            for _ in range(self.config.num_warmup):
                coreml_model.predict(
                    {"input": np.random.randn(1, seq_length, in_dim).astype(np.float32)},
                )

            latencies = []
            for _ in range(self.config.num_iters):
                inp = np.random.randn(1, seq_length, in_dim).astype(np.float32)
                t0 = time.perf_counter()
                coreml_model.predict({"input": inp})
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

            avg_latency = sum(latencies) / len(latencies)
            flops = 2.0 * seq_length * in_dim * out_dim
            gflops = flops / (avg_latency * 1e-3) / 1e9

            result["latency_ms"] = round(avg_latency, 3)
            result["gflops"] = round(gflops, 2)
            result["status"] = "ok"
            return result

        except Exception as exc:
            logger.warning(
                "CoreML ANE linear bench failed, falling back: %s", exc,
            )
            return self._bench_linear_ane_estimated(
                in_dim, out_dim, seq_length, result,
            )

    def _bench_linear_ane_estimated(
        self,
        in_dim: int,
        out_dim: int,
        seq_length: int,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Estimate ANE linear layer performance."""
        # ANE matmul estimate: ~11-38 TOPS depending on chip
        # Assume mid-range: 15 TOPS for matmul operations
        ane_tops = 15.0
        flops = 2.0 * seq_length * in_dim * out_dim
        # Add 0.05ms dispatch overhead per op
        estimated_ms = flops / (ane_tops * 1e12 / 1e3) + 0.05
        gflops = flops / (estimated_ms * 1e-3) / 1e9

        result["latency_ms"] = round(estimated_ms, 3)
        result["gflops"] = round(gflops, 2)
        result["status"] = "estimated"
        return result

    def bench_transformer_layer(
        self,
        hidden_dim: int,
        seq_length: int,
        device: str = "gpu",
    ) -> dict[str, Any]:
        """Benchmark a transformer encoder layer on GPU or ANE.

        Simulates a single transformer layer: multi-head attention + FFN.
        Uses simplified attention (no causal mask, single head for speed).

        Args:
            hidden_dim: Hidden dimension of the transformer layer.
            seq_length: Input sequence length.
            device: "gpu" for MLX GPU, "ane" for CoreML ANE.

        Returns:
            dict with keys: hidden_dim, seq_length, device, latency_ms,
                            throughput_seq_per_s, status
        """
        result: dict[str, Any] = {
            "hidden_dim": hidden_dim,
            "seq_length": seq_length,
            "device": device,
            "latency_ms": None,
            "throughput_seq_per_s": None,
            "status": "unknown",
        }

        if device == "gpu":
            if not self._mlx_available:
                result["status"] = "mlx_unavailable"
                return result

            mx = self._mx

            # Transformer layer weights
            qkv_weight = mx.random.normal(shape=(hidden_dim, hidden_dim * 3)) * 0.01
            out_weight = mx.random.normal(shape=(hidden_dim, hidden_dim)) * 0.01
            ffn_up_weight = mx.random.normal(shape=(hidden_dim, hidden_dim * 4)) * 0.01
            ffn_down_weight = mx.random.normal(shape=(hidden_dim * 4, hidden_dim)) * 0.01

            input_x = mx.random.normal(shape=(1, seq_length, hidden_dim))

            def _forward() -> Any:
                """Single transformer layer forward pass."""
                h = input_x
                # Attention
                qkv = h @ qkv_weight  # (1, seq, 3*hidden)
                q = qkv[..., :hidden_dim]
                k = qkv[..., hidden_dim : 2 * hidden_dim]
                v = qkv[..., 2 * hidden_dim :]

                # Simplified attention scores (no reshape for multi-head)
                scores = (q @ k.transpose(0, 2, 1)) / (hidden_dim**0.5)
                attn_weights = mx.softmax(scores, axis=-1)
                attn_out = attn_weights @ v

                # Output projection + residual
                h = h + attn_out @ out_weight

                # FFN + residual
                f = mx.maximum(h @ ffn_up_weight, 0)
                h = h + f @ ffn_down_weight

                return h

            # Warmup
            for _ in range(self.config.num_warmup):
                _ = _forward()
                mx.eval(_)

            # Timed
            latencies = []
            for _ in range(self.config.num_iters):
                t0 = time.perf_counter()
                out = _forward()
                mx.eval(out)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

            avg_latency = sum(latencies) / len(latencies)
            result["latency_ms"] = round(avg_latency, 3)
            result["throughput_seq_per_s"] = round(1000.0 / avg_latency, 2)
            result["status"] = "ok"
            return result

        if device == "ane":
            if self._coreml_available:
                return self._bench_transformer_ane_coreml(
                    hidden_dim, seq_length, result,
                )
            return self._bench_transformer_ane_estimated(
                hidden_dim, seq_length, result,
            )

        result["status"] = f"unknown_device_{device}"
        return result

    def _bench_transformer_ane_coreml(
        self,
        hidden_dim: int,
        seq_length: int,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Try CoreML ANE for transformer layer."""
        try:
            import numpy as np

            ct = self._ct

            import torch  # type: ignore[import-untyped]

            class TransformerLayer(torch.nn.Module):
                def __init__(self, hidden: int):
                    super().__init__()
                    self.qkv = torch.nn.Linear(hidden, hidden * 3)
                    self.out_proj = torch.nn.Linear(hidden, hidden)
                    self.ffn_up = torch.nn.Linear(hidden, hidden * 4)
                    self.ffn_down = torch.nn.Linear(hidden * 4, hidden)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    qkv = self.qkv(x)
                    q = qkv[..., : x.shape[-1]]
                    k = qkv[..., x.shape[-1] : 2 * x.shape[-1]]
                    v = qkv[..., 2 * x.shape[-1] :]
                    scores = (q @ k.transpose(-2, -1)) / (x.shape[-1] ** 0.5)
                    attn = torch.softmax(scores, dim=-1) @ v
                    x = x + self.out_proj(attn)
                    x = x + self.ffn_down(torch.relu(self.ffn_up(x)))
                    return x

            model = TransformerLayer(hidden_dim)
            model.eval()

            sample_input = torch.randn(1, seq_length, hidden_dim)
            traced = torch.jit.trace(model, sample_input)

            coreml_model = ct.convert(
                traced,
                inputs=[
                    ct.TensorType(
                        name="input",
                        shape=(1, seq_length, hidden_dim),
                        dtype=np.float32,
                    )
                ],
                convert_to="mlprogram",
                compute_units=ct.ComputeUnit.ALL,
            )

            for _ in range(self.config.num_warmup):
                coreml_model.predict(
                    {"input": np.random.randn(1, seq_length, hidden_dim).astype(np.float32)},
                )

            latencies = []
            for _ in range(self.config.num_iters):
                inp = np.random.randn(1, seq_length, hidden_dim).astype(np.float32)
                t0 = time.perf_counter()
                coreml_model.predict({"input": inp})
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

            avg_latency = sum(latencies) / len(latencies)
            result["latency_ms"] = round(avg_latency, 3)
            result["throughput_seq_per_s"] = round(1000.0 / avg_latency, 2)
            result["status"] = "ok"
            return result

        except Exception as exc:
            logger.warning(
                "CoreML ANE transformer bench failed, falling back: %s", exc,
            )
            return self._bench_transformer_ane_estimated(
                hidden_dim, seq_length, result,
            )

    def _bench_transformer_ane_estimated(
        self,
        hidden_dim: int,
        seq_length: int,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Estimate ANE transformer layer performance."""
        # Transformer layer FLOPs: ~4 * seq * hidden^2 (attention) + 8 * seq * hidden^2 (FFN)
        flops = 12.0 * seq_length * hidden_dim * hidden_dim
        # ANE tops estimate
        ane_tops = 15.0
        # Multiple dispatch overhead for sub-operations
        num_ops = 8  # QKV, attention score, softmax, attn_out, out_proj, ffn_up, relu, ffn_down
        estimated_ms = flops / (ane_tops * 1e12 / 1e3) + num_ops * 0.05

        result["latency_ms"] = round(estimated_ms, 3)
        result["throughput_seq_per_s"] = round(1000.0 / estimated_ms, 2) if estimated_ms > 0 else 0
        result["status"] = "estimated"
        return result

    # ── Full benchmark run ─────────────────────────────────────────────────────

    def run(self) -> list[dict[str, Any]]:
        """Run all configured benchmarks.

        Returns:
            List of result dicts, one per benchmark configuration.
        """
        results: list[dict[str, Any]] = []

        logger.info(
            "Starting ANE vs GPU benchmark: %d model sizes x %d seq lengths x 2 devices",
            len(self.config.model_sizes),
            len(self.config.seq_lengths),
        )

        # 1. Embedding inference benchmarks
        for model_size in self.config.model_sizes:
            for seq_len in self.config.seq_lengths:
                logger.info("  Embedding: %dM params, seq=%d, GPU...", model_size, seq_len)
                gpu_result = self.bench_embedding_inference(model_size, seq_len, device="gpu")
                results.append(gpu_result)

                logger.info("  Embedding: %dM params, seq=%d, ANE...", model_size, seq_len)
                ane_result = self.bench_embedding_inference(model_size, seq_len, device="ane")
                results.append(ane_result)

        if self.config.quick:
            logger.info("Benchmark complete (quick mode, embedding-only): %d results", len(results))
            return results

        # 2. Linear layer benchmarks (select subset of sizes)
        linear_configs = [
            (128, 128, 64),
            (256, 256, 128),
            (512, 512, 128),
            (768, 768, 256),
        ]
        for in_dim, out_dim, seq_len in linear_configs:
            logger.info("  Linear: (%d->%d), seq=%d, GPU...", in_dim, out_dim, seq_len)
            gpu_result = self.bench_linear_layer(in_dim, out_dim, seq_len, device="gpu")
            results.append(gpu_result)

            logger.info("  Linear: (%d->%d), seq=%d, ANE...", in_dim, out_dim, seq_len)
            ane_result = self.bench_linear_layer(in_dim, out_dim, seq_len, device="ane")
            results.append(ane_result)

        # 3. Transformer layer benchmarks (select hidden dims)
        for hidden_dim in [128, 256, 512]:
            for seq_len in [32, 128, 512]:
                logger.info("  Transformer: hidden=%d, seq=%d, GPU...", hidden_dim, seq_len)
                gpu_result = self.bench_transformer_layer(hidden_dim, seq_len, device="gpu")
                results.append(gpu_result)

                logger.info("  Transformer: hidden=%d, seq=%d, ANE...", hidden_dim, seq_len)
                ane_result = self.bench_transformer_layer(hidden_dim, seq_len, device="ane")
                results.append(ane_result)

        logger.info("Benchmark complete: %d results", len(results))
        return results

    # ── Formatting ─────────────────────────────────────────────────────────────

    @staticmethod
    def format_results(results: list[dict[str, Any]]) -> str:
        """Format benchmark results as a readable comparison table.

        Args:
            results: List of result dicts from run() or individual bench methods.

        Returns:
            Formatted string with comparison tables.
        """
        if not results:
            return "(no results)"

        lines: list[str] = []
        lines.append("")
        lines.append("=" * 90)
        lines.append("  ANE vs GPU Micro-Benchmark Results")
        lines.append("=" * 90)

        # ── Embedding inference table ──
        emb_results = [r for r in results if "model_size_m" in r]
        if emb_results:
            lines.append("")
            lines.append("  [Embedding Inference]")
            lines.append(
                f"  {'Model (M)':>10s} {'Seq Len':>8s} "
                f"{'GPU (ms)':>10s} {'ANE (ms)':>10s} {'Speedup':>8s} {'Status':>12s}"
            )
            lines.append(f"  {'-' * 10} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 12}")

            # Group by model_size + seq_length
            grouped: dict[tuple, dict[str, dict]] = {}
            for r in emb_results:
                key = (r["model_size_m"], r["seq_length"])
                if key not in grouped:
                    grouped[key] = {}
                grouped[key][r["device"]] = r

            for (model_size, seq_len), devices in sorted(grouped.items()):
                gpu_ms = devices.get("gpu", {}).get("latency_ms", None)
                ane_ms = devices.get("ane", {}).get("latency_ms", None)
                ane_status = devices.get("ane", {}).get("status", "N/A")

                gpu_str = f"{gpu_ms:.3f}" if gpu_ms is not None else "N/A"
                ane_str = f"{ane_ms:.3f}" if ane_ms is not None else "N/A"

                speedup_str = "N/A"
                if gpu_ms and ane_ms and ane_ms > 0:
                    speedup = gpu_ms / ane_ms
                    speedup_str = f"{speedup:.2f}x"

                lines.append(
                    f"  {model_size:>10d} {seq_len:>8d} "
                    f"{gpu_str:>10s} {ane_str:>10s} {speedup_str:>8s} {ane_status:>12s}"
                )

        # ── Linear layer table ──
        lin_results = [r for r in results if "in_dim" in r]
        if lin_results:
            lines.append("")
            lines.append("  [Linear Layer]")
            lines.append(
                f"  {'In Dim':>8s} {'Out Dim':>8s} {'Seq Len':>8s} "
                f"{'GPU (ms)':>10s} {'ANE (ms)':>10s} {'Speedup':>8s} {'Status':>12s}"
            )
            lines.append(
                f"  {'-' * 8} {'-' * 8} {'-' * 8} "
                f"{'-' * 10} {'-' * 10} {'-' * 8} {'-' * 12}"
            )

            grouped_lin: dict[tuple, dict[str, dict]] = {}
            for r in lin_results:
                key = (r["in_dim"], r["out_dim"], r["seq_length"])
                if key not in grouped_lin:
                    grouped_lin[key] = {}
                grouped_lin[key][r["device"]] = r

            for (in_d, out_d, seq_l), devices in sorted(grouped_lin.items()):
                gpu_ms = devices.get("gpu", {}).get("latency_ms", None)
                ane_ms = devices.get("ane", {}).get("latency_ms", None)
                ane_status = devices.get("ane", {}).get("status", "N/A")

                gpu_str = f"{gpu_ms:.3f}" if gpu_ms is not None else "N/A"
                ane_str = f"{ane_ms:.3f}" if ane_ms is not None else "N/A"

                speedup_str = "N/A"
                if gpu_ms and ane_ms and ane_ms > 0:
                    speedup = gpu_ms / ane_ms
                    speedup_str = f"{speedup:.2f}x"

                lines.append(
                    f"  {in_d:>8d} {out_d:>8d} {seq_l:>8d} "
                    f"{gpu_str:>10s} {ane_str:>10s} {speedup_str:>8s} {ane_status:>12s}"
                )

        # ── Transformer layer table ──
        tf_results = [r for r in results if "hidden_dim" in r]
        if tf_results:
            lines.append("")
            lines.append("  [Transformer Layer]")
            lines.append(
                f"  {'Hidden':>8s} {'Seq Len':>8s} "
                f"{'GPU (ms)':>10s} {'ANE (ms)':>10s} {'Speedup':>8s} {'Status':>12s}"
            )
            lines.append(f"  {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 12}")

            grouped_tf: dict[tuple, dict[str, dict]] = {}
            for r in tf_results:
                key = (r["hidden_dim"], r["seq_length"])
                if key not in grouped_tf:
                    grouped_tf[key] = {}
                grouped_tf[key][r["device"]] = r

            for (hidden, seq_l), devices in sorted(grouped_tf.items()):
                gpu_ms = devices.get("gpu", {}).get("latency_ms", None)
                ane_ms = devices.get("ane", {}).get("latency_ms", None)
                ane_status = devices.get("ane", {}).get("status", "N/A")

                gpu_str = f"{gpu_ms:.3f}" if gpu_ms is not None else "N/A"
                ane_str = f"{ane_ms:.3f}" if ane_ms is not None else "N/A"

                speedup_str = "N/A"
                if gpu_ms and ane_ms and ane_ms > 0:
                    speedup = gpu_ms / ane_ms
                    speedup_str = f"{speedup:.2f}x"

                lines.append(
                    f"  {hidden:>8d} {seq_l:>8d} "
                    f"{gpu_str:>10s} {ane_str:>10s} {speedup_str:>8s} {ane_status:>12s}"
                )

        lines.append("")
        lines.append("=" * 90)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run ANE micro-benchmark, print results, and optionally save JSON report."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ANE vs GPU Micro-Benchmark for Phase 0 Platform Validation",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Save results as JSON to this file",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Number of timed iterations (default: 50)",
    )
    parser.add_argument(
        "--model-sizes",
        type=str,
        default=None,
        help="Comma-separated model sizes in M params (default: 10,50,100,300)",
    )
    parser.add_argument(
        "--seq-lengths",
        type=str,
        default=None,
        help="Comma-separated sequence lengths (default: 32,64,128,256,512)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke run: 10M params, single seq=64, 2 warmup + 5 iters (under 60s)",
    )
    args = parser.parse_args()

    # Build config
    if args.quick:
        config = ANEBenchConfig(num_warmup=2, num_iters=5, quick=True)
        config.model_sizes = [10]
        config.seq_lengths = [64]
    else:
        config = ANEBenchConfig(
            num_warmup=args.warmup,
            num_iters=args.iters,
        )
        if args.model_sizes:
            config.model_sizes = [int(x.strip()) for x in args.model_sizes.split(",")]
        if args.seq_lengths:
            config.seq_lengths = [int(x.strip()) for x in args.seq_lengths.split(",")]

    # Run benchmarks
    bench = ANEBenchmark(config)
    results = bench.run()

    # Print formatted table
    print(ANEBenchmark.format_results(results))

    # Save JSON report
    if args.output:
        report = {
            "benchmark": "ane_vs_gpu",
            "config": {
                "model_sizes": config.model_sizes,
                "seq_lengths": config.seq_lengths,
                "num_warmup": config.num_warmup,
                "num_iters": config.num_iters,
            },
            "results": results,
        }
        args.output.write_text(json.dumps(report, indent=2))
        print(f"JSON report saved to: {args.output}")


if __name__ == "__main__":
    main()
