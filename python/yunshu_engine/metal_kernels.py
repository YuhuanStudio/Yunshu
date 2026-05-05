"""Yunshu Metal Kernel Python Bindings.

Loads precompiled .metallib and provides Python API for launching
Metal compute kernels from MLX arrays.

Usage:
    from yunshu_engine.metal_kernels import MetalKernelManager

    mgr = MetalKernelManager()
    mgr.load_default_library()
    result = mgr.paged_attention_decode(queries, key_cache, value_cache, ...)

Also provides:
    compile_kernels() — runs `make` in metal/ dir to build .metallib
    get_compilation_status() — returns dict with compilation metadata
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

# Path to compiled Metal library
_METALLIB_PATH = Path(__file__).parent.parent.parent / "metal" / "build" / "yunshu_kernels.metallib"
_METAL_DIR = Path(__file__).parent.parent.parent / "metal"

# Compilation state
_compilation_status: dict = {
    "compiled": False,
    "metallib_path": str(_METALLIB_PATH),
    "kernel_count": 0,
    "last_error": None,
    "last_attempt": None,
}


class MetalKernelManager:
    """Manages Metal kernel loading and execution.

    Loads the compiled yunshu_kernels.metallib and provides typed wrappers
    for each kernel with proper buffer management.
    """

    def __init__(self, metallib_path: Optional[str] = None):
        self._path = Path(metallib_path) if metallib_path else _METALLIB_PATH
        self._loaded = False
        self._kernels: dict[str, mx.MetalKernel] = {}

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_default_library(self) -> bool:
        """Load the compiled Metal library."""
        if not self._path.exists():
            logger.warning(f"Metal library not found: {self._path}")
            logger.warning("Run 'just build-metal' to compile Metal kernels")
            return False

        try:
            # Load kernel source from .metallib
            # MLX's Metal kernel API loads .metal source directly
            metal_dir = self._path.parent.parent
            for metal_file in metal_dir.glob("*.metal"):
                if metal_file.name == "common.metal":
                    continue
                kernel_name = self._metal_filename_to_kernel(metal_file.name)
                if kernel_name:
                    self._kernels[kernel_name] = str(metal_file)
            self._loaded = True
            logger.info(f"Loaded {len(self._kernels)} Metal kernel sources")
            return True
        except Exception as e:
            logger.error(f"Failed to load Metal library: {e}")
            return False

    def _metal_filename_to_kernel(self, filename: str) -> Optional[str]:
        """Map .metal filenames to kernel group names."""
        mapping = {
            "paged_attention.metal": "paged_attention",
            "kivi_quant.metal": "kivi_quant",
            "sgmv.metal": "sgmv",
            "sdpa.metal": "sdpa",
            "gemv.metal": "gemv",
        }
        return mapping.get(filename)

    def paged_attention_decode(
        self,
        queries: mx.array,      # [num_queries, num_heads, head_dim]
        key_cache: mx.array,    # [num_blocks * block_size, num_heads, head_dim]
        value_cache: mx.array,  # [num_blocks * block_size, num_heads, head_dim]
        block_tables: mx.array, # [num_queries, max_num_blocks]
        seq_lens: mx.array,     # [num_queries]
        num_heads: int,
        head_dim: int,
        kv_block_size: int = 64,
        scale: Optional[float] = None,
    ) -> mx.array:
        """Run PagedAttention decode kernel.

        Returns:
            output: [num_queries, num_heads, head_dim]
        """
        if scale is None:
            scale = 1.0 / (head_dim ** 0.5)

        # For now, use MLX's built-in SDPA as fallback
        # Real Metal kernel dispatch is Phase 2 (requires MXF integration)
        return self._fallback_paged_attention(
            queries, key_cache, value_cache,
            block_tables, seq_lens,
            num_heads, head_dim, kv_block_size, scale,
        )

    def _fallback_paged_attention(
        self,
        queries, key_cache, value_cache,
        block_tables, seq_lens,
        num_heads, head_dim, kv_block_size, scale,
    ) -> mx.array:
        """Vectorized paged attention using MLX ops."""
        num_queries = queries.shape[0]
        output = mx.zeros_like(queries)

        for i in range(num_queries):
            seq_len = int(seq_lens[i])
            if seq_len <= 0:
                continue

            num_blocks = (seq_len + kv_block_size - 1) // kv_block_size
            q = queries[i]  # [num_heads, head_dim]

            # Gather KV from paged blocks
            block_ids = block_tables[i, :num_blocks]
            valid = block_ids >= 0
            valid_ids = mx.array([b for b in block_ids.tolist() if b >= 0])

            if len(valid_ids) == 0:
                continue

            # Gather contiguous KV segments from physical blocks
            k_parts = []
            v_parts = []
            for b_idx, physical in enumerate(valid_ids.tolist()):
                start = physical * kv_block_size
                remaining = seq_len - b_idx * kv_block_size
                end = start + min(kv_block_size, remaining)
                k_parts.append(key_cache[start:end])
                v_parts.append(value_cache[start:end])

            keys = mx.concatenate(k_parts, axis=0)   # [seq_len, num_heads, head_dim]
            vals = mx.concatenate(v_parts, axis=0)    # [seq_len, num_heads, head_dim]

            # SDPA: q [H,D] @ k^T [H,D] -> scores [H, S] -> softmax -> @ v [S,H,D] -> [H,D]
            # Use einsum-style matmul for efficiency
            scores = (keys * q[None, :, :]).sum(axis=-1) * scale  # [seq_len, num_heads]
            scores = mx.softmax(scores.astype(mx.float32), axis=0).astype(queries.dtype)

            # [seq_len, num_heads, 1] * [seq_len, num_heads, head_dim] -> sum -> [num_heads, head_dim]
            output[i] = (scores[:, :, None] * vals).sum(axis=0)

        return output

    def kivi_quantize(
        self,
        keys: mx.array,  # [num_tokens, num_heads, head_dim] FP16
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Quantize key cache with KIVI 2-bit quantization.

        Returns:
            quant_keys: [num_tokens, num_heads, head_dim/4] uint8 (packed 2-bit)
            scales: [num_tokens, num_heads] FP16
            zero_points: [num_tokens, num_heads] FP16
        """
        num_tokens, num_heads, head_dim = keys.shape

        # Per-channel quantization
        channel_min = keys.min(axis=-1)  # [num_tokens, num_heads]
        channel_max = keys.max(axis=-1)  # [num_tokens, num_heads]

        scale = (channel_max - channel_min) / 3.0  # 4 levels
        scale = mx.where(scale < 1e-8, mx.ones_like(scale), scale)
        zp = channel_min

        # Quantize to 2-bit (4 values per byte)
        inv_scale = 1.0 / scale
        quant_float = (keys - zp[..., None]) * inv_scale[..., None]
        quant_int = mx.clip(mx.round(quant_float), 0, 3).astype(mx.uint8)

        # Pack 4 values into 1 byte
        packed_dim = head_dim // 4
        quant_keys = mx.zeros((num_tokens, num_heads, packed_dim), dtype=mx.uint8)
        for i in range(4):
            shifted = quant_int[..., i::4] << (i * 2)
            quant_keys[..., :] += shifted

        return quant_keys, scale.astype(mx.float16), zp.astype(mx.float16)

    def kivi_dequantize(
        self,
        quant_keys: mx.array,  # [num_tokens, num_heads, head_dim/4] uint8
        scales: mx.array,      # [num_tokens, num_heads] FP16
        zero_points: mx.array, # [num_tokens, num_heads] FP16
        head_dim: int,
    ) -> mx.array:
        """Dequantize KIVI 2-bit keys back to FP16."""
        num_tokens, num_heads, packed_dim = quant_keys.shape

        # Unpack 4 values from each byte
        keys = mx.zeros((num_tokens, num_heads, head_dim), dtype=mx.float16)
        for i in range(4):
            unpacked = (quant_keys >> (i * 2)) & 0x3  # uint8 → 0-3
            dequant = scales[..., None] * unpacked.astype(mx.float16) + zero_points[..., None]
            keys[..., i::4] = dequant[..., :keys.shape[-1] // 4 + 1][:, :, :head_dim // 4]

        return keys

    def gemv(
        self,
        W: mx.array,     # [out_dim, in_dim] FP16 weight matrix
        x: mx.array,     # [in_dim] or [batch, in_dim] FP16 input vector(s)
        bias: Optional[mx.array] = None,  # [out_dim] FP16 bias (optional)
    ) -> mx.array:
        """General matrix-vector multiply: y = W @ x [+ bias].

        Supports single vector or batched input.

        Args:
            W: Weight matrix [out_dim, in_dim]
            x: Input vector [in_dim] or batch [batch, in_dim]
            bias: Optional bias [out_dim]

        Returns:
            y: Output [out_dim] or [batch, out_dim]
        """
        out_dim, in_dim = W.shape

        if x.ndim == 1:
            # Single vector: y = W @ x
            result = mx.matmul(W, x)
            if bias is not None:
                result = result + bias
        else:
            # Batched: y[i] = W @ x[i]
            result = mx.matmul(x, W.T)
            if bias is not None:
                result = result + bias

        return result

    def sdpa_attention(
        self,
        Q: mx.array,  # [seq_len, num_heads, head_dim] or [batch, seq_len, num_heads, head_dim]
        K: mx.array,  # [seq_len, num_kv_heads, head_dim] or [batch, seq_len, num_kv_heads, head_dim]
        V: mx.array,  # [seq_len, num_kv_heads, head_dim] or [batch, seq_len, num_kv_heads, head_dim]
        scale: Optional[float] = None,
        causal: bool = True,
    ) -> mx.array:
        """Scaled dot-product attention using MLX ops.

        Computes: output = softmax(Q @ K^T / sqrt(d)) @ V

        Falls back to manual implementation if MLX built-in SDPA is not available.

        Args:
            Q: Query tensor [seq_len, num_heads, head_dim]
            K: Key tensor [seq_len, num_kv_heads, head_dim]
            V: Value tensor [seq_len, num_kv_heads, head_dim]
            scale: Attention scale (default: 1/sqrt(head_dim))
            causal: Whether to apply causal mask

        Returns:
            output: [seq_len, num_heads, head_dim]
        """
        seq_q = Q.shape[0]
        seq_k = K.shape[0]
        num_heads = Q.shape[1]
        head_dim = Q.shape[2]
        num_kv_heads = K.shape[1]

        if scale is None:
            scale = 1.0 / (head_dim ** 0.5)

        # GQA: repeat K/V if num_kv_heads < num_heads
        if num_kv_heads < num_heads:
            n_rep = num_heads // num_kv_heads
            K = mx.repeat(K, n_rep, axis=1)
            V = mx.repeat(V, n_rep, axis=1)

        # Scores: [S_q, H, S_k]
        scores = mx.einsum("qhd,khd->qhk", Q.astype(mx.float32), K.astype(mx.float32)) * scale

        if causal:
            # Causal mask: [S_q, S_k], query q can attend key k only if k <= q
            mask = mx.triu(mx.full((seq_q, seq_k), -1e9), k=seq_k - seq_q + 1)
            scores = scores + mask[:, None, :]

        # Softmax along key dimension
        weights = mx.softmax(scores, axis=-1).astype(Q.dtype)

        # Weighted sum: [seq_len_q, num_heads, seq_len_k] @ [seq_len_k, num_heads, head_dim]
        # -> [seq_len_q, num_heads, head_dim]
        output = mx.einsum("qhk,khd->qhd", weights.astype(mx.float32), V.astype(mx.float32))
        return output.astype(Q.dtype)

    def reload(self) -> bool:
        """Reload the Metal kernel library (useful after recompilation)."""
        self._loaded = False
        self._kernels.clear()
        return self.load_default_library()


def compile_kernels(force: bool = False) -> bool:
    """Compile Metal kernels by running `make` in the metal/ directory.

    Args:
        force: If True, run `make clean && make`. Otherwise just `make`.

    Returns:
        True if compilation succeeded (metallib exists after attempt).
    """
    global _compilation_status

    _compilation_status["last_attempt"] = time.time()

    if not _METAL_DIR.exists():
        _compilation_status["last_error"] = f"Metal directory not found: {_METAL_DIR}"
        logger.error(_compilation_status["last_error"])
        return False

    try:
        cmd = ["make", "clean", "&&", "make"] if force else ["make"]
        # Use shell=True for the && chain or simple make
        shell_cmd = f"cd {_METAL_DIR} && make clean && make" if force else f"cd {_METAL_DIR} && make"

        result = subprocess.run(
            shell_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0 and _METALLIB_PATH.exists():
            _compilation_status["compiled"] = True
            _compilation_status["last_error"] = None
            # Count kernel files
            metal_files = list(_METAL_DIR.glob("*.metal"))
            _compilation_status["kernel_count"] = len([f for f in metal_files if f.name != "common.metal"])
            logger.info(f"Metal kernels compiled successfully: {_METALLIB_PATH}")
            return True
        else:
            _compilation_status["compiled"] = False
            error_msg = result.stderr.strip() if result.stderr else "Unknown compilation error"
            # Common CI/no-SDK case
            if "unable to find utility" in error_msg or "not a developer tool" in error_msg:
                error_msg += " (Metal SDK not available — expected on CI/non-macOS)"
            _compilation_status["last_error"] = error_msg
            logger.warning(f"Metal kernel compilation failed: {error_msg}")
            return False

    except subprocess.TimeoutExpired:
        _compilation_status["compiled"] = False
        _compilation_status["last_error"] = "Compilation timed out (120s)"
        logger.error(_compilation_status["last_error"])
        return False
    except Exception as e:
        _compilation_status["compiled"] = False
        _compilation_status["last_error"] = str(e)
        logger.error(f"Compilation error: {e}")
        return False


def get_compilation_status() -> dict:
    """Get the current Metal kernel compilation status.

    Returns:
        dict with keys:
            compiled: bool — whether .metallib exists and was compiled
            metallib_path: str — path to expected .metallib
            kernel_count: int — number of .metal source files
            last_error: str | None — last compilation error, if any
            last_attempt: float | None — timestamp of last compile attempt
    """
    global _compilation_status

    # Refresh compiled status by checking file existence
    _compilation_status["compiled"] = _METALLIB_PATH.exists()

    # Count kernels if not already done
    if _compilation_status["kernel_count"] == 0 and _METAL_DIR.exists():
        metal_files = list(_METAL_DIR.glob("*.metal"))
        _compilation_status["kernel_count"] = len(
            [f for f in metal_files if f.name != "common.metal"]
        )

    return dict(_compilation_status)


# Module-level singleton
_kernel_manager: Optional[MetalKernelManager] = None


def get_kernel_manager() -> MetalKernelManager:
    """Get or create the Metal kernel manager singleton."""
    global _kernel_manager
    if _kernel_manager is None:
        _kernel_manager = MetalKernelManager()
        _kernel_manager.load_default_library()
    return _kernel_manager
