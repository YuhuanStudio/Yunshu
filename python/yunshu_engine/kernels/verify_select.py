"""Faster exact kernel choice for 5-bit layers in mlx-vlm's multi-row verify.

mlx-vlm's exact verify matmul picks its kernel by row count: rows <= 5 use the
base kernel (per-row register accumulators, collapses from ~5 rows), and only
4-bit layers get the ``streamed`` kernel for >= 6 rows. Its 5-bit ``streamed``
path is broken (the thread-array ``qdot_exact`` overload only implements 4-bit,
so it returns the bias term alone), so mixed 4/5-bit checkpoints (Qwen3.8-27B
oQ4e: 166 5-bit projections incl. every GDN ``out_proj`` and 27 ``down_proj``)
run every 5-bit projection on the collapsing base kernel.

This module carries a fixed streamed kernel (5-bit weights read through the
device-pointer ``qdot_exact`` the base kernel uses, so each row's accumulation
order equals single-row decode) and routes 5-bit layers with >= 5 verify rows to
it. M5 Max, 17408->5120 5-bit projection
(docs/research/runs/2026-09-28-matrix/verify-streamed-fixed.jsonl):

    rows              5        6        8
    base           503 us   873 us  2434 us
    streamed-fixed 261 us   323 us   406 us   (bit-identical to decode)

5-bit grouped projections at those widths skip the fused base kernel and use
the per-projection path. Only the kernel choice changes, never the output.
"""

from __future__ import annotations

import mlx.core as mx

MIN_STREAMED_ROWS_5BIT = 5

# mlx-vlm's streamed verify kernel with the 5-bit weight read fixed: upstream
# preloads weights into a ``thread uint16_t`` array and calls the qdot_exact
# overload that only implements 4-bit, so for 5-bit it returned just the bias
# term (max rel. error ~1.8). Here every width reads the packed bytes through
# the device-pointer overload the base kernel uses, so each row's accumulation
# order is exactly the single-row decode order.
_STREAMED_FIXED_SOURCE = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint b_idx = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K_SIZE / GS;

    const device uint8_t* ws =
        (const device uint8_t*)w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* sc =
        scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bs =
        biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* xk =
        x + int(b_idx) * VERIFY_T * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

    float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        result[t][row] = 0.0f;
      }
    }

    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        float scale = float(sc[row * in_vec_size_g]);
        float bias = float(bs[row * in_vec_size_g]);
        for (int t = 0; t < VERIFY_T; ++t) {
          float x_thread[VALUES_PER_THREAD];
          float sum = load_vector_exact<T>(xk + t * K_SIZE, x_thread);
          result[t][row] += qdot_exact(wl, x_thread, scale, bias, sum);
        }
      }
      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xk += BLOCK_SIZE;
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      for (int t = 0; t < VERIFY_T; ++t) {
        float value = simd_sum(result[t][row]);
        if (simd_lid == 0) {
          y[(int(b_idx) * VERIFY_T + t) * N_SIZE + n] = T(value);
        }
      }
    }
"""

_KERNELS: dict = {}


def streamed_fixed_kernel(qv, bits, group_size, dtype, verify_t, k_size, n_size):
    key = (bits, group_size, dtype, verify_t, k_size, n_size)
    if key not in _KERNELS:
        dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
        _KERNELS[key] = mx.fast.metal_kernel(
            name=(
                "yunshu_verify_qmv_streamed_fixed_"
                f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{n_size}_{dtype_name}"
            ),
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            header=qv._target_verify_qlinear_header(bits, group_size, 1),
            source=_STREAMED_FIXED_SOURCE,
        )
    return _KERNELS[key]


def streamed_fixed(qv, linear, x: mx.array) -> mx.array:
    B, T, K = x.shape
    N = linear.weight.shape[0]
    x = mx.contiguous(x)
    kernel = streamed_fixed_kernel(qv, linear.bits, linear.group_size, x.dtype, T, K, N)
    out = kernel(
        inputs=[x, linear.weight, linear.scales, linear.biases],
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(T)),
            ("K_SIZE", int(K)),
            ("N_SIZE", int(N)),
        ],
        grid=(32, 2 * (N // 2), B),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, T, N)],
        output_dtypes=[x.dtype],
    )[0]
    if "bias" in linear:
        out = out + linear["bias"]
    return out


def install() -> bool:
    from mlx_vlm.models import quantized_verifier as qv
    from mlx_vlm.speculative.ops import linear as ops

    single = qv.optimized_affine_linear
    grouped = qv.optimized_affine_linears
    if getattr(single, "_yunshu_streamed5", False):
        return True

    def optimized_affine_linear(linear, x):
        if (
            getattr(linear, "bits", None) == 5
            and x.ndim == 3
            and x.shape[1] >= MIN_STREAMED_ROWS_5BIT
            and qv._can_optimized_affine_linear(linear, x)
        ):
            return streamed_fixed(qv, linear, x)
        return single(linear, x)

    def optimized_affine_linears(linears, x):
        if (
            linears
            and getattr(linears[0], "bits", None) == 5
            and x.ndim == 3
            and x.shape[1] >= MIN_STREAMED_ROWS_5BIT
        ):
            return None  # per-projection streamed path beats the fused base kernel
        return grouped(linears, x)

    optimized_affine_linear._yunshu_streamed5 = True
    qv.optimized_affine_linear = optimized_affine_linear
    qv.optimized_affine_linears = optimized_affine_linears
    ops._target_verify_optimized_affine_linear = optimized_affine_linear
    ops._target_verify_quantized_linears = optimized_affine_linears
    return True


# Experimental: unpack each weight block once per output row and reuse it for
# every verify row. Keeps qdot_exact's product and summation order term for
# term so each row still matches single-row decode bit for bit (checked by
# scripts/research/bench_verify_qmv_variants.py style parity tests).
_UNPACKED_SOURCE = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint b_idx = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K_SIZE / GS;

    const device uint8_t* ws =
        (const device uint8_t*)w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* sc =
        scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bs =
        biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* xk =
        x + int(b_idx) * VERIFY_T * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

    constexpr int TERMS = (BITS == 5) ? (VALUES_PER_THREAD / 8) * 12 : VALUES_PER_THREAD;

    float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        result[t][row] = 0.0f;
      }
    }

    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        float scale = float(sc[row * in_vec_size_g]);
        float bias = float(bs[row * in_vec_size_g]);
        float wf[TERMS];
        if (BITS == 4) {
          const device uint16_t* w16 = (const device uint16_t*)wl;
          for (int i = 0; i < VALUES_PER_THREAD / 4; ++i) {
            uint p = w16[i];
            wf[4 * i] = float(p & 0x000f);
            wf[4 * i + 1] = float((p >> 4) & 0x000f);
            wf[4 * i + 2] = float((p >> 8) & 0x000f);
            wf[4 * i + 3] = float((p >> 12) & 0x000f);
          }
        } else {
          for (int i = 0; i < VALUES_PER_THREAD / 8; ++i) {
            const device uint8_t* wb = wl + 5 * i;
            thread float* f = wf + 12 * i;
            f[0] = float(wb[0] & 0x1f);
            f[1] = float(wb[0] & 0xe0);
            f[2] = float(wb[1] & 0x3);
            f[3] = float(wb[1] & 0x7c);
            f[4] = float(wb[1] & 0x80);
            f[5] = float(wb[2] & 0xf);
            f[6] = float(wb[2] & 0xf0);
            f[7] = float(wb[3] & 0x1);
            f[8] = float(wb[3] & 0x3e);
            f[9] = float(wb[3] & 0xc0);
            f[10] = float(wb[4] & 0x7);
            f[11] = float(wb[4] & 0xf8);
          }
        }
        for (int t = 0; t < VERIFY_T; ++t) {
          float xt[VALUES_PER_THREAD];
          float sum = load_vector_exact<T>(xk + t * K_SIZE, xt);
          float accum = 0.0f;
          if (BITS == 4) {
            for (int i = 0; i < VALUES_PER_THREAD / 4; ++i) {
              accum +=
                  (xt[4 * i] * wf[4 * i] + xt[4 * i + 1] * wf[4 * i + 1] +
                   xt[4 * i + 2] * wf[4 * i + 2] + xt[4 * i + 3] * wf[4 * i + 3]);
            }
          } else {
            for (int i = 0; i < VALUES_PER_THREAD / 8; ++i) {
              const thread float* q = xt + 8 * i;
              const thread float* f = wf + 12 * i;
              accum += f[0] * q[0];
              accum += f[1] * q[1];
              accum += f[2] * (q[1] * 256.0f);
              accum += f[3] * q[2];
              accum += f[4] * q[3];
              accum += f[5] * (q[3] * 256.0f);
              accum += f[6] * q[4];
              accum += f[7] * (q[4] * 256.0f);
              accum += f[8] * q[5];
              accum += f[9] * q[6];
              accum += f[10] * (q[6] * 256.0f);
              accum += f[11] * q[7];
            }
          }
          result[t][row] += scale * accum + sum * bias;
        }
      }
      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xk += BLOCK_SIZE;
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      for (int t = 0; t < VERIFY_T; ++t) {
        float value = simd_sum(result[t][row]);
        if (simd_lid == 0) {
          y[(int(b_idx) * VERIFY_T + t) * N_SIZE + n] = T(value);
        }
      }
    }
"""


def unpacked_kernel(qv, bits, group_size, dtype, verify_t, k_size, n_size, rps=4):
    key = ("unpacked", bits, group_size, dtype, verify_t, k_size, n_size, rps)
    if key not in _KERNELS:
        dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
        _KERNELS[key] = mx.fast.metal_kernel(
            name=(
                "yunshu_verify_qmv_unpacked_"
                f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{n_size}_r{rps}_{dtype_name}"
            ),
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            header=qv._target_verify_qlinear_header(bits, group_size, rps),
            source=_UNPACKED_SOURCE,
        )
    return _KERNELS[key]


def unpacked(qv, linear, x: mx.array, rps: int = 4) -> mx.array:
    B, T, K = x.shape
    N = linear.weight.shape[0]
    x = mx.contiguous(x)
    kernel = unpacked_kernel(qv, linear.bits, linear.group_size, x.dtype, T, K, N, rps)
    out = kernel(
        inputs=[x, linear.weight, linear.scales, linear.biases],
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(T)),
            ("K_SIZE", int(K)),
            ("N_SIZE", int(N)),
        ],
        grid=(32, 2 * (N // (2 * rps)), B),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, T, N)],
        output_dtypes=[x.dtype],
    )[0]
    if "bias" in linear:
        out = out + linear["bias"]
    return out
