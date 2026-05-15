from __future__ import annotations
"""Apple Silicon Roofline Model — predict LLM inference throughput bounds.

.. deprecated:: This module is not used in the production pipeline. Kept for reference only.


Analyzes compute vs memory-boundedness of transformer operations
(GEMM, attention, FFN) against Apple GPU bandwidth and FLOP ceilings.

Usage::

    from yunshu_engine.roofline import RooflineModel

    rm = RooflineModel("M4_Max")
    report = rm.compute_gemm_roofline(M=1, N=4096, K=4096)
    rm.plot_roofline("roofline.png")
"""

import logging
import re
from dataclasses import dataclass, field

from yunshu_engine.utils.hardware import get_chip_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chip database
# ---------------------------------------------------------------------------

CHIP_PARAMS: dict[str, dict[str, float]] = {
    "M1":      {"bandwidth_gbps": 68,  "compute_tflops_fp16": 2.6},
    "M1_Pro":  {"bandwidth_gbps": 200, "compute_tflops_fp16": 5.2},
    "M1_Max":  {"bandwidth_gbps": 400, "compute_tflops_fp16": 10.4},
    "M1_Ultra":{"bandwidth_gbps": 800, "compute_tflops_fp16": 20.8},
    "M2":      {"bandwidth_gbps": 100, "compute_tflops_fp16": 3.6},
    "M2_Pro":  {"bandwidth_gbps": 200, "compute_tflops_fp16": 7.8},
    "M2_Max":  {"bandwidth_gbps": 400, "compute_tflops_fp16": 13.6},
    "M2_Ultra":{"bandwidth_gbps": 800, "compute_tflops_fp16": 27.2},
    "M3":      {"bandwidth_gbps": 150, "compute_tflops_fp16": 3.6},
    "M3_Pro":  {"bandwidth_gbps": 300, "compute_tflops_fp16": 7.8},
    "M3_Max":  {"bandwidth_gbps": 400, "compute_tflops_fp16": 14.0},
    "M3_Ultra":{"bandwidth_gbps": 800, "compute_tflops_fp16": 27.0},
    "M4":      {"bandwidth_gbps": 120, "compute_tflops_fp16": 5.0},
    "M4_Pro":  {"bandwidth_gbps": 273, "compute_tflops_fp16": 10.0},
    "M4_Max":  {"bandwidth_gbps": 546, "compute_tflops_fp16": 18.0},
}

# dtype size in bytes
DTYPE_BYTES: dict[str, int] = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "int8": 1,
    "int4": 0,
}

# Well-known model architectures for estimate_max_throughput
MODEL_CONFIGS: dict[str, dict] = {
    "Qwen2.5-0.5B": {
        "hidden_size": 896, "num_layers": 24, "num_heads": 14,
        "head_dim": 64, "ffn_dim": 4864, "vocab_size": 151936,
    },
    "Qwen2.5-1.5B": {
        "hidden_size": 1536, "num_layers": 28, "num_heads": 12,
        "head_dim": 128, "ffn_dim": 8960, "vocab_size": 151936,
    },
    "Qwen2.5-3B": {
        "hidden_size": 2048, "num_layers": 36, "num_heads": 16,
        "head_dim": 128, "ffn_dim": 11008, "vocab_size": 151936,
    },
    "Qwen2.5-7B": {
        "hidden_size": 3584, "num_layers": 28, "num_heads": 28,
        "head_dim": 128, "ffn_dim": 18944, "vocab_size": 152064,
    },
    "Qwen2.5-9B": {
        "hidden_size": 3584, "num_layers": 36, "num_heads": 28,
        "head_dim": 128, "ffn_dim": 18944, "vocab_size": 152064,
    },
    "Qwen2.5-14B": {
        "hidden_size": 5120, "num_layers": 40, "num_heads": 40,
        "head_dim": 128, "ffn_dim": 13824, "vocab_size": 152064,
    },
    "Qwen2.5-32B": {
        "hidden_size": 5120, "num_layers": 64, "num_heads": 40,
        "head_dim": 128, "ffn_dim": 27648, "vocab_size": 152064,
    },
    "Qwen2.5-72B": {
        "hidden_size": 8192, "num_layers": 80, "num_heads": 64,
        "head_dim": 128, "ffn_dim": 29568, "vocab_size": 152064,
    },
    "Llama-3.1-8B": {
        "hidden_size": 4096, "num_layers": 32, "num_heads": 32,
        "head_dim": 128, "ffn_dim": 14336, "vocab_size": 128256,
    },
    "Llama-3.1-70B": {
        "hidden_size": 8192, "num_layers": 80, "num_heads": 64,
        "head_dim": 128, "ffn_dim": 28672, "vocab_size": 128256,
    },
    "Mistral-7B": {
        "hidden_size": 4096, "num_layers": 32, "num_heads": 32,
        "head_dim": 128, "ffn_dim": 14336, "vocab_size": 32000,
    },
    "Phi-3.5-mini": {
        "hidden_size": 3072, "num_layers": 32, "num_heads": 32,
        "head_dim": 96, "ffn_dim": 8192, "vocab_size": 32064,
    },
    "Gemma-2-9B": {
        "hidden_size": 3584, "num_layers": 42, "num_heads": 16,
        "head_dim": 256, "ffn_dim": 14336, "vocab_size": 256000,
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RooflineResult:
    """Result of a single roofline computation."""
    flops: int
    bytes_accessed: int
    operational_intensity: float        # FLOP / byte
    peak_gflops: float                  # compute ceiling
    bandwidth_gbs: float                # memory ceiling
    predicted_gflops: float             # min(bw * OI, peak)
    bound: str                          # "memory" or "compute"
    label: str = ""


@dataclass
class DecodeRooflineResult:
    """Aggregate roofline result for a full decode step."""
    total_flops: int
    total_bytes: int
    operational_intensity: float
    predicted_gflops: float
    bound: str
    ops: list[RooflineResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Normalise chip name
# ---------------------------------------------------------------------------

def _normalise_chip(raw: str) -> str:
    """Turn 'Apple M3 Max' or 'm3_max' into 'M3_Max'."""
    s = raw.strip()
    # Strip leading "Apple "
    s = re.sub(r"^Apple\s+", "", s, flags=re.IGNORECASE)
    # Lowercase and replace spaces / dashes with underscore
    s = s.lower().replace(" ", "_").replace("-", "_")
    # Map common variants
    s = re.sub(r"^(m\d+)_(pro|max|ultra)$", lambda m: f"{m.group(1).upper()}_{m.group(2).capitalize()}", s)
    # Simple chip without tier: just uppercase
    s = re.sub(r"^(m\d+)$", lambda m: m.group(1).upper(), s)
    # Try exact
    if s in CHIP_PARAMS:
        return s
    # Try fuzzy: "M3MAX" -> "M3_Max"
    m = re.search(r"(M\d+)\s*(Pro|Max|Ultra)?", s, re.IGNORECASE)
    if m:
        tier = m.group(2)
        if tier:
            return f"M{m.group(1)[1]}_{tier.capitalize()}"
        return f"M{m.group(1)[1]}"
    return s


# ---------------------------------------------------------------------------
# RooflineModel
# ---------------------------------------------------------------------------

class RooflineModel:
    """Apple Silicon roofline model for LLM inference throughput prediction."""

    def __init__(self, chip_name: str | None = None) -> None:
        if chip_name is None:
            chip_name = get_chip_name()
        self._raw_chip = chip_name
        self.chip_key = _normalise_chip(chip_name)

        if self.chip_key not in CHIP_PARAMS:
            # Fall back to closest match
            logger.warning(
                "Unknown chip %r (normalised: %r). Falling back to M3_Max.",
                chip_name, self.chip_key,
            )
            self.chip_key = "M3_Max"

        params = CHIP_PARAMS[self.chip_key]
        self.bandwidth_gbs: float = params["bandwidth_gbps"]
        self.compute_tflops: float = params["compute_tflops_fp16"]

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _dtype_bytes(self, dtype: str) -> int:
        b = DTYPE_BYTES.get(dtype)
        if b is None:
            raise ValueError(f"Unknown dtype {dtype!r}. Supported: {list(DTYPE_BYTES)}")
        if b == 0:
            # int4 sub-byte: treat as 0.5 bytes
            return 0  # caller must handle
        return b

    def _roofline(self, flops: int, bytes_accessed: int, label: str = "") -> RooflineResult:
        """Compute roofline from raw FLOPs and bytes."""
        oi = flops / bytes_accessed if bytes_accessed > 0 else float("inf")
        peak_gflops = self.compute_tflops * 1000.0
        bandwidth_bound = self.bandwidth_gbs * oi   # GB/s * FLOP/byte = GFLOP/s
        predicted_gflops = min(bandwidth_bound, peak_gflops)
        bound = "memory" if bandwidth_bound <= peak_gflops else "compute"
        return RooflineResult(
            flops=flops,
            bytes_accessed=bytes_accessed,
            operational_intensity=oi,
            peak_gflops=peak_gflops,
            bandwidth_gbs=self.bandwidth_gbs,
            predicted_gflops=predicted_gflops,
            bound=bound,
            label=label,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_attention_roofline(
        self,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        batch_size: int = 1,
        dtype: str = "fp16",
    ) -> RooflineResult:
        """Roofline for attention (Q@K^T + softmax@V).

        Compute: 2 * seq_len^2 * num_heads * head_dim  (matmul FLOPs)
        Memory:  load Q, K, V + write output
                  = seq_len * num_heads * head_dim * 4 * dtype_bytes
                  (the *4 accounts for Q, K, V, and output matrices)
        """
        db = self._dtype_bytes(dtype)
        # Handle sub-byte types
        if db == 0:
            db = 1  # rough approx for int4 — treat as 1 byte for roofline

        B = batch_size
        S = seq_len
        H = num_heads
        D = head_dim

        # FLOPs: Q @ K^T  = 2*B*S*S*H*D   and  softmax @ V = 2*B*S*S*H*D
        flops = 2 * 2 * B * S * S * H * D  # total = 4 * B * S^2 * H * D

        # Memory traffic
        # Q: B*S*H*D, K: B*S*H*D, V: B*S*H*D, Output: B*S*H*D
        bytes_accessed = int(B * S * H * D * 4 * db)

        return self._roofline(
            flops=flops,
            bytes_accessed=bytes_accessed,
            label=f"attention B={B} S={S} H={H} D={D}",
        )

    def compute_gemm_roofline(
        self,
        M: int,
        N: int,
        K: int,
        dtype: str = "fp16",
    ) -> RooflineResult:
        """Roofline for a GEMM operation (M x K) @ (K x N) -> (M x N).

        Compute:  2 * M * N * K  FLOPs
        Memory:   (M*K + K*N + M*N) * dtype_bytes
        """
        db = self._dtype_bytes(dtype)
        if db == 0:
            db = 1

        flops = 2 * M * N * K
        bytes_accessed = int((M * K + K * N + M * N) * db)

        return self._roofline(
            flops=flops,
            bytes_accessed=bytes_accessed,
            label=f"GEMM ({M}x{K})@({K}x{N})",
        )

    def compute_decode_roofline(
        self,
        model_config: dict,
        batch_size: int = 1,
        context_len: int = 0,
        dtype: str = "fp16",
    ) -> DecodeRooflineResult:
        """Roofline for one full transformer decode step (batch=1 token generation).

        model_config keys: hidden_size, num_layers, num_heads, head_dim,
                           ffn_dim, vocab_size (optional).
        """
        h = model_config["hidden_size"]
        L = model_config["num_layers"]
        nh = model_config["num_heads"]
        hd = model_config["head_dim"]
        ffn = model_config["ffn_dim"]
        V = model_config.get("vocab_size", h * 4)
        B = batch_size

        ops: list[RooflineResult] = []

        total_flops = 0
        total_bytes = 0

        def _add(r: RooflineResult) -> None:
            nonlocal total_flops, total_bytes
            ops.append(r)
            total_flops += r.flops
            total_bytes += r.bytes_accessed

        for layer_i in range(L):
            # QKV projection: 3 GEMMs (1xK) @ (Kxh) each -> effectively one big GEMM
            # Computation: input (B, h) -> QKV (B, 3*h)
            # This is: 3 * (2 * B * h * h) = 6 * B * h^2 ... but we decompose as
            # three projections: (1, h)@(h, h) each
            qkv = self.compute_gemm_roofline(M=B, N=h, K=h, dtype=dtype)
            qkv.flops *= 3
            qkv.bytes_accessed = int(
                (B * h + h * h) * self._dtype_bytes(dtype) * 3  # 3 projections
                + B * h * 3 * self._dtype_bytes(dtype)           # output
            )
            qkv.label = f"L{layer_i}/QKV_proj"
            _add(qkv)

            # Attention: for decode (seq_len=1), memory-bound
            # Effective seq_len includes context for K/V reads
            attn_seq = max(context_len, 1) + 1  # context + new token
            # Attention compute: 2 * B * attn_seq * nh * hd (Q@K) + 2 * B * attn_seq * nh * hd (att@V)
            attn_flops = 2 * 2 * B * attn_seq * nh * hd
            # Attention memory: read Q(B*nh*hd), K(attn_seq*nh*hd), V(attn_seq*nh*hd), write O(B*nh*hd)
            db = self._dtype_bytes(dtype)
            attn_bytes = int((B * nh * hd + attn_seq * nh * hd * 2 + B * nh * hd) * db)
            attn_r = self._roofline(attn_flops, attn_bytes, label=f"L{layer_i}/attention")
            _add(attn_r)

            # Output projection: (1, h) @ (h, h) -> (1, h)
            o_proj = self.compute_gemm_roofline(M=B, N=h, K=h, dtype=dtype)
            o_proj.label = f"L{layer_i}/O_proj"
            _add(o_proj)

            # FFN: gate+up projection (2 GEMMs) + down projection (1 GEMM)
            # For SwiGLU: input(h) -> gate(ffn), up(ffn), then gate*up -> down -> h
            # 3 GEMMs: (1,h)@(h,ffn), (1,h)@(h,ffn), (1,ffn)@(ffn,h)
            gate = self.compute_gemm_roofline(M=B, N=ffn, K=h, dtype=dtype)
            gate.label = f"L{layer_i}/FFN_gate"
            _add(gate)

            up = self.compute_gemm_roofline(M=B, N=ffn, K=h, dtype=dtype)
            up.label = f"L{layer_i}/FFN_up"
            _add(up)

            down = self.compute_gemm_roofline(M=B, N=h, K=ffn, dtype=dtype)
            down.label = f"L{layer_i}/FFN_down"
            _add(down)

        # LM head: (1, h) @ (h, V) -> (1, V)
        lm_head = self.compute_gemm_roofline(M=B, N=V, K=h, dtype=dtype)
        lm_head.label = "LM_head"
        _add(lm_head)

        oi = total_flops / total_bytes if total_bytes > 0 else float("inf")
        peak_gflops = self.compute_tflops * 1000.0
        bw_bound = self.bandwidth_gbs * oi
        predicted_gflops = min(bw_bound, peak_gflops)
        bound = "memory" if bw_bound <= peak_gflops else "compute"

        return DecodeRooflineResult(
            total_flops=total_flops,
            total_bytes=total_bytes,
            operational_intensity=oi,
            predicted_gflops=predicted_gflops,
            bound=bound,
            ops=ops,
        )

    def estimate_max_throughput(
        self,
        model_name: str,
        context_len: int = 0,
        dtype: str = "fp16",
    ) -> dict:
        """Estimate maximum decode tokens/sec for a model.

        Returns dict with keys:
            model, chip, tokens_per_sec, gflops_per_token, bound,
            total_flops, total_bytes, operational_intensity
        """
        config = self._resolve_model_config(model_name)
        result = self.compute_decode_roofline(
            config, batch_size=1, context_len=context_len, dtype=dtype,
        )
        gflops_per_token = result.total_flops / 1e9
        tokens_per_sec = result.predicted_gflops / gflops_per_token if gflops_per_token > 0 else 0.0

        return {
            "model": model_name,
            "chip": self.chip_key,
            "tokens_per_sec": round(tokens_per_sec, 2),
            "gflops_per_token": round(gflops_per_token, 2),
            "bound": result.bound,
            "total_flops": result.total_flops,
            "total_bytes": result.total_bytes,
            "operational_intensity": round(result.operational_intensity, 4),
            "bandwidth_gbs": self.bandwidth_gbs,
            "compute_tflops": self.compute_tflops,
        }

    def _resolve_model_config(self, model_name: str) -> dict:
        """Look up or fuzzy-match model config."""
        # Exact match
        for key, cfg in MODEL_CONFIGS.items():
            if key.lower() == model_name.lower():
                return cfg

        # Fuzzy: strip common suffixes and prefixes
        cleaned = re.sub(
            r"[-_]?((Instruct|Chat|Base|it|4bit|8bit|GPTQ|AWQ)[-_]?)",
            "", model_name, flags=re.IGNORECASE,
        ).strip("-_")

        for key, cfg in MODEL_CONFIGS.items():
            if cleaned.lower() in key.lower() or key.lower() in cleaned.lower():
                return cfg

        raise ValueError(
            f"Unknown model {model_name!r}. "
            f"Known models: {sorted(MODEL_CONFIGS.keys())}"
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_roofline(self, output_path: str = "roofline.png") -> str:
        """Generate a roofline plot and save to *output_path*.

        Requires matplotlib (``pip install matplotlib`` or ``pip install yunshu[bench]``).
        Returns the output path.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_roofline(). "
                "Install it with: pip install matplotlib  or  pip install yunshu[bench]"
            ) from exc

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        peak_gflops = self.compute_tflops * 1000.0
        bw = self.bandwidth_gbs  # GB/s

        # X range: operational intensity (FLOP/byte)
        x_min, x_max = 0.5, 500.0
        xs = [x_min]
        # Find ridge point: bw * OI = peak  =>  OI = peak / bw
        ridge_oi = peak_gflops / bw if bw > 0 else 100.0
        xs.append(ridge_oi)
        xs.append(x_max)

        # Memory-bound slope: y = bw * x
        y_bw = [bw * x for x in xs]
        # Compute ceiling: y = peak
        y_peak = [peak_gflops] * len(xs)

        # Fill regions
        ax.fill_between(
            [x_min, ridge_oi],
            [0, 0],
            [bw * x_min, bw * ridge_oi],
            alpha=0.10, color="tab:blue", label="Memory-bound region",
        )
        ax.fill_between(
            [ridge_oi, x_max],
            [0, 0],
            [peak_gflops, peak_gflops],
            alpha=0.10, color="tab:red", label="Compute-bound region",
        )

        # Roofline lines
        ax.plot(
            [x_min, ridge_oi],
            [bw * x_min, bw * ridge_oi],
            "b-", linewidth=2, label=f"Bandwidth: {bw:.0f} GB/s",
        )
        ax.plot(
            [ridge_oi, x_max],
            [peak_gflops, peak_gflops],
            "r-", linewidth=2, label=f"Peak FP16: {self.compute_tflops:.1f} TFLOP/s",
        )

        # Mark ridge point
        ax.plot(ridge_oi, peak_gflops, "ko", markersize=8, zorder=5)
        ax.annotate(
            f"Ridge point\nOI={ridge_oi:.1f}",
            xy=(ridge_oi, peak_gflops),
            xytext=(ridge_oi * 1.5, peak_gflops * 0.85),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9,
        )

        # Mark representative operations
        markers = self._compute_marker_points()
        colors = ["tab:green", "tab:orange", "tab:purple", "tab:brown", "tab:cyan", "tab:pink"]
        for i, m in enumerate(markers):
            oi = m["oi"]
            perf = m["perf_gflops"]
            c = colors[i % len(colors)]
            ax.plot(oi, perf, "D", color=c, markersize=8, zorder=6)
            ax.annotate(
                m["label"],
                xy=(oi, perf),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=8,
                color=c,
                fontweight="bold",
            )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.set_xlabel("Operational Intensity (FLOP / byte)", fontsize=12)
        ax.set_ylabel("Performance (GFLOP/s)", fontsize=12)
        ax.set_title(
            f"Roofline Model — {self.chip_key} "
            f"(BW={bw:.0f} GB/s, Peak={self.compute_tflops:.1f} TFLOP/s FP16)",
            fontsize=13,
        )
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    def _compute_marker_points(self) -> list[dict]:
        """Compute representative operations to mark on the roofline."""
        points: list[dict] = []
        peak_gflops = self.compute_tflops * 1000.0
        bw = self.bandwidth_gbs

        # 1. Attention prefill (seq=1024)
        r = self.compute_attention_roofline(seq_len=1024, num_heads=32, head_dim=128, batch_size=1)
        perf = min(bw * r.operational_intensity, peak_gflops)
        points.append({
            "label": "Attn prefill\nS=1024",
            "oi": r.operational_intensity,
            "perf_gflops": perf,
        })

        # 2. Attention decode (seq=1)
        r = self.compute_attention_roofline(seq_len=1, num_heads=32, head_dim=128, batch_size=1)
        perf = min(bw * r.operational_intensity, peak_gflops)
        points.append({
            "label": "Attn decode\nS=1",
            "oi": r.operational_intensity,
            "perf_gflops": perf,
        })

        # 3. Small GEMM (decode linear, 1x4096 @ 4096x4096)
        r = self.compute_gemm_roofline(M=1, N=4096, K=4096)
        perf = min(bw * r.operational_intensity, peak_gflops)
        points.append({
            "label": "GEMM decode\n1x4K@4Kx4K",
            "oi": r.operational_intensity,
            "perf_gflops": perf,
        })

        # 4. Large GEMM (prefill, 1024x4096 @ 4096x4096)
        r = self.compute_gemm_roofline(M=1024, N=4096, K=4096)
        perf = min(bw * r.operational_intensity, peak_gflops)
        points.append({
            "label": "GEMM prefill\n1024x4K",
            "oi": r.operational_intensity,
            "perf_gflops": perf,
        })

        # 5. FFN decode (1x4096 @ 4096x14336)
        r = self.compute_gemm_roofline(M=1, N=14336, K=4096)
        perf = min(bw * r.operational_intensity, peak_gflops)
        points.append({
            "label": "FFN decode\n1x4K->14K",
            "oi": r.operational_intensity,
            "perf_gflops": perf,
        })

        return points


# ---------------------------------------------------------------------------
# MLX-based measurement utilities
# ---------------------------------------------------------------------------

def measure_roofline(
    chip_name: str | None = None,
    bandwidth_bytes: int = 256 * 1024 * 1024,
    gemm_sizes: list[int] | None = None,
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> dict:
    """Run empirical MLX roofline benchmarks and return measurement results.

    This performs actual GPU work to measure:
    - **Memory bandwidth**: large array copy (GB/s)
    - **Compute throughput**: GEMM at various sizes (TFLOPS)
    - **Roofline analysis**: compute operational intensity and bound type

    Returns a dict with keys:
        bandwidth_gbs, bandwidth_details,
        tflops_peak, tflops_details,
        peak_theoretical_gbs, peak_theoretical_tflops,
        bound_type, chip

    Requires MLX to be available. Raises ``ImportError`` if MLX is missing.

    Usage::

        from yunshu_engine.roofline import measure_roofline

        results = measure_roofline()
        print(f"Bandwidth: {results['bandwidth_gbs']:.1f} GB/s")
        print(f"Peak compute: {results['tflops_peak']:.2f} TFLOPS")
        print(f"Bound: {results['bound_type']}")
    """
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise ImportError(
            "MLX is required for measure_roofline(). "
            "Install with: pip install mlx"
        ) from exc

    import time

    if chip_name is None:
        chip_name = get_chip_name()
    rm = RooflineModel(chip_name)

    if gemm_sizes is None:
        gemm_sizes = [256, 512, 1024, 2048, 4096]

    # -- Memory bandwidth benchmark --
    # Copy a large array to measure sustained memory throughput
    src = mx.random.normal((bandwidth_bytes // 4,))  # float32 = 4 bytes
    mx.synchronize()

    # Warmup
    for _ in range(warmup_iters):
        dst = mx.array(src)
    mx.synchronize()

    t0 = time.perf_counter()
    for _ in range(bench_iters):
        dst = mx.array(src)
    mx.synchronize()
    elapsed_bw = time.perf_counter() - t0

    total_bytes = bandwidth_bytes * bench_iters * 2  # read + write
    bandwidth_gbs = total_bytes / elapsed_bw / 1e9

    bandwidth_details = {
        "method": "array_copy",
        "size_bytes": bandwidth_bytes,
        "iters": bench_iters,
        "elapsed_s": round(elapsed_bw, 4),
        "bandwidth_gbs": round(bandwidth_gbs, 2),
    }

    # -- Compute throughput benchmark --
    # GEMM at various sizes to find peak TFLOPS
    tflops_details = []
    peak_tflops = 0.0

    for size in gemm_sizes:
        a = mx.random.normal((size, size), dtype=mx.float16)
        b = mx.random.normal((size, size), dtype=mx.float16)
        # Warmup
        _ = a @ b
        mx.synchronize()

        iters = max(1, min(bench_iters, 2**20 // (size * size)))
        t0 = time.perf_counter()
        for _ in range(iters):
            c = a @ b
        mx.synchronize()
        elapsed = time.perf_counter() - t0

        flops = 2.0 * size ** 3 * iters
        tflops = flops / elapsed / 1e12
        tflops_details.append({
            "size": size,
            "iters": iters,
            "elapsed_s": round(elapsed, 4),
            "tflops": round(tflops, 2),
        })
        if tflops > peak_tflops:
            peak_tflops = tflops

    # -- Roofline analysis --
    # Use the peak GEMM to compute operational intensity and bound type
    # For a square GEMM at peak size: OI = FLOPs / bytes
    # bytes ≈ (M*K + K*N + M*N) * 2 (fp16) ≈ 3 * size^2 * 2
    # FLOPs = 2 * size^3
    # OI = 2*size^3 / (6*size^2) = size/3
    # We use the analytical model's ridge point for bound classification
    peak_gflops_analytical = rm.compute_tflops * 1000.0
    ridge_oi = peak_gflops_analytical / rm.bandwidth_gbs if rm.bandwidth_gbs > 0 else 100.0

    # Determine overall bound type based on measured vs theoretical ratios
    measured_tflops_gflops = peak_tflops * 1000.0
    if measured_tflops_gflops > 0 and rm.bandwidth_gbs > 0:
        # If peak TFLOPS is close to theoretical peak, we're compute-bound at large sizes
        compute_ratio = measured_tflops_gflops / peak_gflops_analytical if peak_gflops_analytical > 0 else 0
        # If bandwidth achieved is close to theoretical, we're bandwidth-efficient
        bw_ratio = bandwidth_gbs / rm.bandwidth_gbs if rm.bandwidth_gbs > 0 else 0
        bound_type = "compute" if compute_ratio > bw_ratio else "memory"
    else:
        bound_type = "unknown"

    return {
        "bandwidth_gbs": round(bandwidth_gbs, 2),
        "bandwidth_details": bandwidth_details,
        "tflops_peak": round(peak_tflops, 2),
        "tflops_details": tflops_details,
        "peak_theoretical_gbs": rm.bandwidth_gbs,
        "peak_theoretical_tflops": rm.compute_tflops,
        "bound_type": bound_type,
        "chip": rm.chip_key,
    }
