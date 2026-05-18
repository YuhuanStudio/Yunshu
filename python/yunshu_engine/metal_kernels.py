"""Yunshu Metal Kernel Python Bindings.

Uses MLX's mx.fast.metal_kernel API to JIT-compile custom Metal kernels
that integrate directly with MLX's computation graph and memory management.

This is the best approach because:
- Zero-copy integration with mx.arrays (no buffer copying)
- Automatic memory management via MLX's graph
- JIT specialization per dtype/shape combination
- Template metaprogramming for type-generic kernels

Kernels:
- paged_attention_decode: Single-query decode against paged KV cache
- gemv_fp16: FP16 matrix-vector multiply
- gemv_q4: 4-bit quantized GEMV with on-the-fly dequant
- kivi_quantize/dequantize: 2-bit KV cache compression
"""


import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_METAL_DIR = Path(__file__).parent.parent.parent / "metal"
_METALLIB_PATH = _METAL_DIR / "build" / "yunshu_kernels.metallib"

# ── Common Metal source headers ──

_SIMD_REDUCE_HEADER = """
// Warp-level sum reduction (Apple GPU 32-wide SIMD)
inline float simd_reduce_sum(float val) {
    for (uint16_t offset = 16; offset > 0; offset >>= 1) {
        val += simd_shuffle_down(val, offset);
    }
    return val;
}

inline float simd_reduce_max(float val) {
    for (uint16_t offset = 16; offset > 0; offset >>= 1) {
        val = metal::max(val, simd_shuffle_down(val, offset));
    }
    return val;
}
"""


# ── PagedAttention Decode ──

def _paged_attention_decode_kernel(head_dim: int):
    """PagedAttention decode: single query token per sequence against paged KV cache.

    Grid: (num_heads, num_queries, 1), threadgroup: (32, 1, 1).
    Each threadgroup handles one (query, head) pair.
    Online softmax with FP32 accumulation over KV blocks.
    """
    source = r"""
        uint query_idx = threadgroup_position_in_grid.y;
        uint head_idx = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;

        int seq_len = seq_lens_ptr[query_idx];
        if (seq_len <= 0) return;

        int num_kv_blocks = (seq_len + KV_BLOCK_SZ - 1) / KV_BLOCK_SZ;
        device const int* bt = block_tables_ptr + query_idx * max_blocks;

        // Load query into registers
        float q[HEAD_DIM];
        for (uint d = lane; d < HEAD_DIM; d += 32) {
            q[d] = (float)queries_ptr[(query_idx * NUM_HEADS + head_idx) * HEAD_DIM + d];
        }

        // Online softmax accumulator
        float acc[HEAD_DIM];
        for (uint d = 0; d < HEAD_DIM; d++) acc[d] = 0.0f;
        float running_max = -1e9f;
        float running_sum = 0.0f;

        for (int block = 0; block < num_kv_blocks; block++) {
            int phys = bt[block];
            if (phys < 0) continue;

            int block_start = block * KV_BLOCK_SZ;
            int block_end = metal::min((int)KV_BLOCK_SZ, seq_len - block_start);

            // Each lane processes one KV row, stride by SIMD width
            for (int kv_idx = lane; kv_idx < block_end; kv_idx += 32) {
                uint kv_base = (phys * KV_BLOCK_SZ + kv_idx) * NUM_HEADS * HEAD_DIM
                               + head_idx * HEAD_DIM;

                // Q·K score
                float score = 0.0f;
                for (uint d = 0; d < HEAD_DIM; d++) {
                    score += q[d] * (float)key_cache_ptr[kv_base + d];
                }
                score *= attn_scale;

                // Online softmax update
                float old_max = running_max;
                running_max = metal::max(running_max, score);
                float correction = metal::exp(old_max - running_max);

                for (uint d = 0; d < HEAD_DIM; d++) acc[d] *= correction;
                running_sum *= correction;

                float weight = metal::exp(score - running_max);
                running_sum += weight;

                // Accumulate weighted value
                for (uint d = 0; d < HEAD_DIM; d++) {
                    acc[d] += weight * (float)value_cache_ptr[kv_base + d];
                }
            }
        }

        // Finalize
        float inv_sum = 1.0f / metal::max(running_sum, 1e-8f);
        uint out_base = (query_idx * NUM_HEADS + head_idx) * HEAD_DIM;
        for (uint d = lane; d < HEAD_DIM; d += 32) {
            output_ptr[out_base + d] = (half)(acc[d] * inv_sum);
        }
    """

    return mx.fast.metal_kernel(
        name="yunshu_paged_attn_decode",
        input_names=["queries_ptr", "key_cache_ptr", "value_cache_ptr",
                      "block_tables_ptr", "seq_lens_ptr"],
        output_names=["output_ptr"],
        source=source,
    )


# ── GEMV Kernels via mx.fast.metal_kernel ──

def _gemv_fp16_kernel():
    """FP16 GEMV: y = W @ x + bias. Each threadgroup (32 threads) handles one output row."""
    source = r"""
        uint row = threadgroup_position_in_grid.x;
        if (row >= OUT_DIM) return;

        float sum = 0.0f;
        for (uint d = thread_index_in_simdgroup; d < IN_DIM; d += 32) {
            sum += (float)W_ptr[row * IN_DIM + d] * (float)x_ptr[d];
        }
        for (uint16_t offset = 16; offset > 0; offset >>= 1) {
            sum += simd_shuffle_down(sum, offset);
        }
        if (thread_index_in_simdgroup == 0) {
            y_ptr[row] = (half)sum;
        }
    """

    return mx.fast.metal_kernel(
        name="yunshu_gemv_fp16",
        input_names=["W_ptr", "x_ptr"],
        output_names=["y_ptr"],
        source=source,
    )


def _gemv_q4_kernel():
    """4-bit quantized GEMV: dequantize + dot product fused."""
    source = r"""
        uint row = threadgroup_position_in_grid.x;
        if (row >= OUT_DIM) return;

        float scale = (float)scales_ptr[row];
        float zp = (float)zp_ptr[row];
        uint packed_dim = IN_DIM / 2;
        uint base = row * packed_dim;

        float sum = 0.0f;
        for (uint d = thread_index_in_simdgroup; d < packed_dim; d += 32) {
            uchar packed = W_q_ptr[base + d];
            float val_lo = (float)(packed & 0xF) - zp;
            float val_hi = (float)((packed >> 4) & 0xF) - zp;
            sum += scale * val_lo * (float)x_ptr[d * 2];
            sum += scale * val_hi * (float)x_ptr[d * 2 + 1];
        }
        for (uint16_t offset = 16; offset > 0; offset >>= 1) {
            sum += simd_shuffle_down(sum, offset);
        }
        if (thread_index_in_simdgroup == 0) {
            y_ptr[row] = (half)sum;
        }
    """

    return mx.fast.metal_kernel(
        name="yunshu_gemv_q4",
        input_names=["W_q_ptr", "scales_ptr", "zp_ptr", "x_ptr"],
        output_names=["y_ptr"],
        source=source,
    )


def _kivi_quantize_kernel():
    """KIVI 2-bit quantize: FP16 → packed 2-bit with per-channel scale/zp."""
    source = r"""
        uint token_idx = thread_position_in_grid.x;
        uint head_idx = thread_position_in_grid.y;
        if (token_idx >= num_tokens_val || head_idx >= num_heads_val) return;

        uint base = token_idx * num_heads_val * head_dim_val + head_idx * head_dim_val;
        uint packed_dim = head_dim_val / 4;
        uint channel_idx = token_idx * num_heads_val + head_idx;

        // Find min/max
        float ch_min = (float)keys_ptr[base];
        float ch_max = (float)keys_ptr[base];
        for (uint d = 1; d < head_dim_val; d++) {
            float v = (float)keys_ptr[base + d];
            ch_min = metal::min(ch_min, v);
            ch_max = metal::max(ch_max, v);
        }

        float scale = (ch_max - ch_min) / 3.0f;
        float inv_scale = (scale > 1e-8f) ? (1.0f / scale) : 0.0f;

        scales_out[channel_idx] = (half)scale;
        zp_out[channel_idx] = (half)ch_min;

        // Pack 4 values per byte
        for (uint d = 0; d < packed_dim; d++) {
            uchar packed = 0;
            for (uint i = 0; i < 4; i++) {
                float val = (float)keys_ptr[base + d * 4 + i];
                float q = (val - ch_min) * inv_scale;
                int qi = (int)metal::clamp(q + 0.5f, 0.0f, 3.0f);
                packed |= (uchar)(qi << (i * 2));
            }
            quant_out[token_idx * num_heads_val * packed_dim + head_idx * packed_dim + d] = packed;
        }
    """

    return mx.fast.metal_kernel(
        name="yunshu_kivi_quantize",
        input_names=["keys_ptr"],
        output_names=["quant_out", "scales_out", "zp_out"],
        source=source,
    )


def _kivi_dequantize_kernel():
    """KIVI 2-bit dequantize: packed 2-bit → FP16."""
    source = r"""
        uint token_idx = thread_position_in_grid.x;
        uint head_idx = thread_position_in_grid.y;
        if (token_idx >= num_tokens_val || head_idx >= num_heads_val) return;

        uint channel_idx = token_idx * num_heads_val + head_idx;
        float scale = (float)scales_ptr[channel_idx];
        float zp = (float)zp_ptr[channel_idx];

        uint packed_dim = head_dim_val / 4;
        uint in_base = token_idx * num_heads_val * packed_dim + head_idx * packed_dim;
        uint out_base = token_idx * num_heads_val * head_dim_val + head_idx * head_dim_val;

        for (uint d = 0; d < packed_dim; d++) {
            uchar packed = quant_ptr[in_base + d];
            for (uint i = 0; i < 4; i++) {
                int q = (packed >> (i * 2)) & 0x3;
                keys_out[out_base + d * 4 + i] = (half)(scale * (float)q + zp);
            }
        }
    """

    return mx.fast.metal_kernel(
        name="yunshu_kivi_dequantize",
        input_names=["quant_ptr", "scales_ptr", "zp_ptr"],
        output_names=["keys_out"],
        source=source,
    )


# ── Kernel Cache (lazy JIT compilation) ──

_kernels: dict[str, object] = {}


def _get_kernel(name: str):
    """Lazily compile and cache a Metal kernel."""
    if name not in _kernels:
        try:
            if name == "gemv_fp16":
                _kernels[name] = _gemv_fp16_kernel()
            elif name == "gemv_q4":
                _kernels[name] = _gemv_q4_kernel()
            elif name == "kivi_quantize":
                _kernels[name] = _kivi_quantize_kernel()
            elif name == "kivi_dequantize":
                _kernels[name] = _kivi_dequantize_kernel()
            elif name.startswith("paged_attn_decode_h"):
                import re
                m = re.match(r"paged_attn_decode_h(\d+)_k(\d+)", name)
                head_dim = int(m.group(1))
                _kernels[name] = _paged_attention_decode_kernel(head_dim)
            else:
                raise ValueError(f"Unknown kernel: {name}")
            logger.info(f"JIT-compiled Metal kernel: {name}")
        except Exception as e:
            logger.warning(f"Failed to compile Metal kernel {name}: {e}")
            return None
    return _kernels.get(name)


# ── High-level API ──

class MetalKernelManager:
    """Manages Metal kernel loading and execution.

    Uses mx.fast.metal_kernel for JIT-compiled kernels that integrate
    directly with MLX's computation graph. Falls back to pure MLX ops
    when Metal kernel compilation is unavailable.
    """

    def __init__(self, metallib_path: Optional[str] = None):
        self._loaded = False
        self._kernels: dict[str, object] = {}

    @property
    def is_loaded(self) -> bool:
        return True  # Always available via JIT

    def load_default_library(self) -> bool:
        """Pre-compile all kernels. Optional — kernels are lazily compiled."""
        for name in ["gemv_fp16", "gemv_q4", "kivi_quantize", "kivi_dequantize"]:
            _get_kernel(name)
        self._loaded = True
        return True

    def gemv(
        self,
        W: mx.array,
        x: mx.array,
        bias: Optional[mx.array] = None,
    ) -> mx.array:
        """FP16 GEMV: y = W @ x [+ bias].

        Uses Metal kernel for single-vector GEMV, MLX matmul for batched.
        """
        out_dim, in_dim = W.shape

        if x.ndim == 1:
            kernel = _get_kernel("gemv_fp16")
            if kernel is not None:
                try:
                    outputs = kernel(
                        inputs=[W, x],
                        template=[("IN_DIM", in_dim), ("OUT_DIM", out_dim)],
                        grid=(out_dim * 32, 1, 1),  # total threads = rows * SIMD width
                        threadgroup=(32, 1, 1),       # 1 SIMD group per row
                        output_shapes=[(out_dim,)],
                        output_dtypes=[mx.float16],
                    )
                    result = outputs[0]
                    if bias is not None:
                        result = result + bias
                    return result
                except Exception as e:
                    logger.debug(f"Metal GEMV fallback: {e}")

            # MLX fallback
            result = mx.matmul(W, x)
            if bias is not None:
                result = result + bias
            return result
        else:
            result = mx.matmul(x, W.T)
            if bias is not None:
                result = result + bias
            return result

    def gemv_q4(
        self,
        W_q: mx.array,
        scales: mx.array,
        zero_points: mx.array,
        x: mx.array,
        in_dim: int,
    ) -> mx.array:
        """4-bit quantized GEMV: dequantize + dot product fused."""
        out_dim = W_q.shape[0]

        kernel = _get_kernel("gemv_q4")
        if kernel is not None:
            try:
                outputs = kernel(
                    inputs=[W_q, scales, zero_points, x],
                    template=[("IN_DIM", in_dim), ("OUT_DIM", out_dim)],
                    grid=(out_dim * 32, 1, 1),
                    threadgroup=(32, 1, 1),
                    output_shapes=[(out_dim,)],
                    output_dtypes=[mx.float16],
                )
                return outputs[0]
            except Exception as e:
                logger.debug(f"Metal GEMV-Q4 fallback: {e}")

        # MLX fallback: dequantize then matmul
        packed_dim = in_dim // 2
        lo = (W_q & 0xF).astype(mx.float16)
        hi = ((W_q >> 4) & 0xF).astype(mx.float16)
        W_dequant = mx.zeros((out_dim, in_dim), dtype=mx.float16)
        for d in range(packed_dim):
            W_dequant[:, d * 2] = scales * lo[:, d] + zero_points
            W_dequant[:, d * 2 + 1] = scales * hi[:, d] + zero_points
        return mx.matmul(W_dequant, x)

    def paged_attention_decode(
        self,
        queries: mx.array,
        key_cache: mx.array,
        value_cache: mx.array,
        block_tables: mx.array,
        seq_lens: mx.array,
        num_heads: int,
        head_dim: int,
        kv_block_size: int = 16,
        scale: Optional[float] = None,
    ) -> mx.array:
        """PagedAttention decode: single-query against paged KV cache.

        Tries Metal kernel first, falls back to MLX ops.
        """
        if scale is None:
            scale = 1.0 / (head_dim ** 0.5)

        num_queries = queries.shape[0]
        if num_queries == 0:
            return mx.zeros_like(queries)

        kernel = _get_kernel(f"paged_attn_decode_h{head_dim}_k{kv_block_size}")
        if kernel is not None:
            try:
                max_blocks = block_tables.shape[1]
                output = kernel(
                    inputs=[queries, key_cache, value_cache, block_tables, seq_lens],
                    template=[
                        ("HEAD_DIM", head_dim),
                        ("NUM_HEADS", num_heads),
                        ("KV_BLOCK_SZ", kv_block_size),
                        ("max_blocks", max_blocks),
                        ("attn_scale", scale),
                    ],
                    grid=(num_heads, num_queries, 1),
                    threadgroup=(32, 1, 1),
                    output_shapes=[queries.shape],
                    output_dtypes=[queries.dtype],
                )
                return output[0]
            except Exception as e:
                logger.debug(f"Metal paged_attn_decode fallback: {e}")

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
            q = queries[i]

            block_ids = block_tables[i, :num_blocks]
            valid_ids = [b for b in block_ids.tolist() if b >= 0]

            if not valid_ids:
                continue

            k_parts = []
            v_parts = []
            for b_idx, physical in enumerate(valid_ids):
                start = physical * kv_block_size
                remaining = seq_len - b_idx * kv_block_size
                end = start + min(kv_block_size, remaining)
                k_parts.append(key_cache[start:end])
                v_parts.append(value_cache[start:end])

            keys = mx.concatenate(k_parts, axis=0)
            vals = mx.concatenate(v_parts, axis=0)

            scores = (keys * q[None, :, :]).sum(axis=-1) * scale
            scores = mx.softmax(scores.astype(mx.float32), axis=0).astype(queries.dtype)
            output[i] = (scores[:, :, None] * vals).sum(axis=0)

        return output

    def kivi_quantize(
        self,
        keys: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """KIVI 2-bit quantize: FP16 keys → packed 2-bit + scale + zp."""
        num_tokens, num_heads, head_dim = keys.shape

        if num_tokens == 0 or num_heads == 0:
            packed_dim = head_dim // 4
            return (
                mx.zeros((num_tokens, num_heads, packed_dim), dtype=mx.uint8),
                mx.zeros((num_tokens, num_heads), dtype=mx.float16),
                mx.zeros((num_tokens, num_heads), dtype=mx.float16),
            )

        kernel = _get_kernel("kivi_quantize")
        if kernel is not None:
            try:
                packed_dim = head_dim // 4
                outputs = kernel(
                    inputs=[keys],
                    template=[
                        ("T", mx.float16),
                        ("num_tokens_val", num_tokens),
                        ("num_heads_val", num_heads),
                        ("head_dim_val", head_dim),
                    ],
                    grid=(num_tokens, num_heads, 1),
                    threadgroup=(1, 1, 1),
                    output_shapes=[
                        (num_tokens, num_heads, packed_dim),
                        (num_tokens, num_heads),
                        (num_tokens, num_heads),
                    ],
                    output_dtypes=[mx.uint8, mx.float16, mx.float16],
                )
                return outputs[0], outputs[1], outputs[2]
            except Exception as e:
                logger.debug(f"Metal KIVI quantize fallback: {e}")

        # MLX fallback
        channel_min = keys.min(axis=-1)
        channel_max = keys.max(axis=-1)
        scale = (channel_max - channel_min) / 3.0
        scale = mx.where(scale < 1e-8, mx.ones_like(scale), scale)
        zp = channel_min

        inv_scale = 1.0 / scale
        quant_float = (keys - zp[..., None]) * inv_scale[..., None]
        quant_int = mx.clip(mx.round(quant_float), 0, 3).astype(mx.uint8)

        packed_dim = head_dim // 4
        quant_keys = mx.zeros((num_tokens, num_heads, packed_dim), dtype=mx.uint8)
        for i in range(4):
            shifted = quant_int[..., i::4] << (i * 2)
            quant_keys = quant_keys + shifted

        return quant_keys, scale.astype(mx.float16), zp.astype(mx.float16)

    def kivi_dequantize(
        self,
        quant_keys: mx.array,
        scales: mx.array,
        zero_points: mx.array,
        head_dim: int,
    ) -> mx.array:
        """KIVI 2-bit dequantize: packed 2-bit → FP16 keys."""
        num_tokens, num_heads, packed_dim = quant_keys.shape

        if num_tokens == 0 or num_heads == 0:
            return mx.zeros((num_tokens, num_heads, head_dim), dtype=mx.float16)

        kernel = _get_kernel("kivi_dequantize")
        if kernel is not None:
            try:
                outputs = kernel(
                    inputs=[quant_keys, scales, zero_points],
                    template=[
                        ("T", mx.float16),
                        ("num_tokens_val", num_tokens),
                        ("num_heads_val", num_heads),
                        ("head_dim_val", head_dim),
                    ],
                    grid=(num_tokens, num_heads, 1),
                    threadgroup=(1, 1, 1),
                    output_shapes=[(num_tokens, num_heads, head_dim)],
                    output_dtypes=[mx.float16],
                )
                return outputs[0]
            except Exception as e:
                logger.debug(f"Metal KIVI dequantize fallback: {e}")

        # MLX fallback
        keys = mx.zeros((num_tokens, num_heads, head_dim), dtype=mx.float16)
        for i in range(4):
            unpacked = (quant_keys >> (i * 2)) & 0x3
            dequant = scales[..., None] * unpacked.astype(mx.float16) + zero_points[..., None]
            n = head_dim // 4
            keys[..., i::4] = dequant[..., :n]

        return keys

    def sdpa_attention(
        self,
        Q: mx.array,
        K: mx.array,
        V: mx.array,
        scale: Optional[float] = None,
        causal: bool = True,
    ) -> mx.array:
        """Scaled dot-product attention using MLX einsum."""
        head_dim = Q.shape[-1]
        if scale is None:
            scale = 1.0 / (head_dim ** 0.5)

        seq_q = Q.shape[0]
        seq_k = K.shape[0]
        num_heads = Q.shape[1]
        num_kv_heads = K.shape[1]

        if num_kv_heads < num_heads:
            n_rep = num_heads // num_kv_heads
            K = mx.repeat(K, n_rep, axis=1)
            V = mx.repeat(V, n_rep, axis=1)

        scores = mx.einsum("qhd,khd->qhk", Q.astype(mx.float32), K.astype(mx.float32)) * scale

        if causal:
            mask = mx.triu(mx.full((seq_q, seq_k), -1e9), k=seq_k - seq_q + 1)
            scores = scores + mask[:, None, :]

        weights = mx.softmax(scores, axis=-1).astype(Q.dtype)
        output = mx.einsum("qhk,khd->qhd", weights.astype(mx.float32), V.astype(mx.float32))
        return output.astype(Q.dtype)

    def reload(self) -> bool:
        """Clear kernel cache and re-JIT on next use."""
        _kernels.clear()
        return True


# ── Compilation ──

_compilation_status: dict = {
    "compiled": False,
    "metallib_path": str(_METALLIB_PATH),
    "kernel_count": 0,
    "last_error": None,
    "last_attempt": None,
}


def compile_kernels(force: bool = False) -> bool:
    """Compile Metal kernels via `make` in metal/ directory."""
    global _compilation_status

    _compilation_status["last_attempt"] = time.time()

    if not _METAL_DIR.exists():
        _compilation_status["last_error"] = f"Metal directory not found: {_METAL_DIR}"
        return False

    try:
        if force:
            subprocess.run(["make", "clean"], cwd=str(_METAL_DIR), capture_output=True, text=True, timeout=60)
        result = subprocess.run(["make"], cwd=str(_METAL_DIR), capture_output=True, text=True, timeout=120)

        if result.returncode == 0 and _METALLIB_PATH.exists():
            _compilation_status.update(compiled=True, last_error=None)
            metal_files = [f for f in _METAL_DIR.glob("*.metal") if f.name != "common.metal"]
            _compilation_status["kernel_count"] = len(metal_files)
            return True
        else:
            _compilation_status["compiled"] = False
            _compilation_status["last_error"] = result.stderr.strip() or "Unknown error"
            return False
    except Exception as e:
        _compilation_status["compiled"] = False
        _compilation_status["last_error"] = str(e)
        return False


def get_compilation_status() -> dict:
    """Get Metal kernel compilation status."""
    _compilation_status["compiled"] = _METALLIB_PATH.exists()
    if _compilation_status["kernel_count"] == 0 and _METAL_DIR.exists():
        _compilation_status["kernel_count"] = len(
            [f for f in _METAL_DIR.glob("*.metal") if f.name != "common.metal"]
        )
    return dict(_compilation_status)


# ── Singleton ──

_kernel_manager: Optional[MetalKernelManager] = None


def get_kernel_manager() -> MetalKernelManager:
    """Get or create the Metal kernel manager singleton."""
    global _kernel_manager
    if _kernel_manager is None:
        _kernel_manager = MetalKernelManager()
    return _kernel_manager
