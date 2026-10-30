"""int8 KV for the tree-attention tile kernel: the same partials, bit for bit, as the bf16 kernel on the
dequantized cache.

The bf16 token-tile kernel (``ragged_attention._TILE``) feeds K and V straight from device memory into
MetalPerformancePrimitives ``matmul2d`` in 64-key steps. A dequantized 64 x 256 bf16 tile does not fit in
threadgroup memory next to the kernel's score and probability tiles, so each threadgroup dequantizes its
step's K and V rows into its own device scratch (``bf16(float(code) * float(scale))``, the definition of
``int8_kv.dequantize_kv``) and runs the unchanged ``matmul2d`` calls on that scratch: same shapes, same
layouts, same arithmetic, so the partials equal the bf16 kernel's on the dequantized buffer. The scratch
is re-read by the threadgroup that wrote it right away, so it stays in cache; only int8 codes and scales
come from DRAM.

``install()`` assigns ``tree_verify.TILE_PARTIALS_Q8``.
"""

from __future__ import annotations

import mlx.core as mx

N_STEP = 64  # keys per step of the tile kernel
GS = 32  # dims per scale group

# Replace the two device-memory views of the step's K / V rows with views of the scratch the
# threadgroup just filled from the int8 codes.
_LOAD_OLD = """    auto ks = tensor(kh + (int64_t)ts * D, dextents<int, 2>{D, N}, array<int, 2>{1, D})
        .template slice<D, N>(0, 0);
    auto vs = tensor(vh + (int64_t)ts * D, dextents<int, 2>{D, N}, array<int, 2>{1, D})
        .template slice<D, N>(0, 0);
"""
_LOAD_NEW = """    {
      // dequantize the step's N rows of K and V into this threadgroup's scratch
      for (int e = int(tid); e < N * D; e += 256) {
        const int r = e / D, d = e % D;
        const int64_t src = (row0 + ts + r) * D + d;
        const int64_t sidx = (row0 + ts + r) * (D / GS) + d / GS;
        kq[e] = bfloat16_t(float(keys[src]) * float(k_scales[sidx]));
        vq[e] = bfloat16_t(float(values[src]) * float(v_scales[sidx]));
      }
      threadgroup_barrier(mem_flags::mem_device);
    }
    auto ks = tensor(kq, dextents<int, 2>{D, N}, array<int, 2>{1, D})
        .template slice<D, N>(0, 0);
    auto vs = tensor(vq, dextents<int, 2>{D, N}, array<int, 2>{1, D})
        .template slice<D, N>(0, 0);
"""
# bf16 pointers into the cache become pointers into the scratch; the first V view (only used for the
# accumulator's type) points at the scratch too
_PTR_OLD = """  device bfloat16_t* kh = const_cast<device bfloat16_t*>(keys) + row0 * D;
  device bfloat16_t* vh = const_cast<device bfloat16_t*>(values) + row0 * D;
"""
_PTR_NEW = """  const int64_t tg = int64_t(threadgroup_position_in_grid.y);
  device bfloat16_t* kq = scratch + tg * (2 * N * D);
  device bfloat16_t* vq = kq + N * D;
  device bfloat16_t* vh = vq;
"""
_VFIRST_OLD = "auto v_first = tensor(vh, dextents"
_VFIRST_NEW = "auto v_first = tensor(vq, dextents"
# the scratch rewrite after the step's matmuls must wait until both matmuls have read it
_STEP_END_OLD = """    pv.run(p0, vs, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""
_STEP_END_NEW = """    pv.run(p0, vs, running);
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
"""

_KERNEL: list = []


def _source() -> str:
    from . import ragged_attention as ra

    src = ra._TILE
    for old, new in (
        (_PTR_OLD, _PTR_NEW),
        (_LOAD_OLD, _LOAD_NEW),
        (_VFIRST_OLD, _VFIRST_NEW),
        (_STEP_END_OLD, _STEP_END_NEW),
    ):
        if src.count(old) != 1:
            raise RuntimeError(
                "ragged_attention._TILE changed; int8_tile rewrite no longer applies"
            )
        src = src.replace(old, new)
    return src.replace("(D / GS)", f"(D / {GS})").replace("d / GS", f"d / {GS}")


def _kernel():
    if not _KERNEL:
        from . import ragged_attention as ra

        _KERNEL.append(
            mx.fast.metal_kernel(
                name="yunshu_ragged_attn_tile_q8",
                input_names=[
                    "queries",
                    "keys",
                    "k_scales",
                    "values",
                    "v_scales",
                    "lengths",
                    "slots",
                    "scale",
                    "work",
                    "wstart",
                ],
                output_names=["PO", "PM", "PL", "scratch"],
                source=_source(),
                header=ra._TILE_HEADER,
                ensure_row_contiguous=False,
            )
        )
    return _KERNEL[0]


def tile_partials_q8(
    q_fused, keys, k_scales, values, v_scales, lengths, slots, scale, plan, t_tokens, nc
):
    """``tree_verify._tile_partials`` on an int8 cache (codes [1, HKV, CAP, D], fp16 scales
    [1, HKV, CAP, D / 32]); returns PO, PM, PL and the per-(row, token) first slot."""
    from .. import tree_verify as tv

    hkv, cap, d = int(keys.shape[1]), int(keys.shape[2]), int(keys.shape[3])
    h = int(q_fused.shape[2]) // 8 * hkv
    g = h // hkv
    work, wstart, nslot = plan
    groups = hkv * int(work.size)
    po, pm, pl, _scratch = _kernel()(
        inputs=[
            q_fused,
            keys,
            k_scales,
            values,
            v_scales,
            lengths,
            slots,
            scale,
            work,
            wstart,
        ],
        template=[
            ("D", d),
            ("G", g),
            ("T", t_tokens),
            ("H", h),
            ("HKV", hkv),
            ("CAP", cap),
            ("CK", tv.CK),
            ("NC", nc),
            ("WL", True),
        ],
        grid=(256, groups, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (nslot * h * d,),
            (nslot * h,),
            (nslot * h,),
            (groups * 2 * N_STEP * d,),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.bfloat16],
    )
    return po, pm, pl, wstart


def tree_attention_fused_q8(
    queries, keys, k_scales, values, v_scales, shape, rc, scale_arr, dims
):
    """``tree_verify._tree_attention_fused`` on an int8 cache: the shared prefix through
    ``tile_partials_q8``, and the tail copy from a dequantized slice of the rows it reads (the last
    partial chunk ``[tail_start, n0)`` and the window rows ``n0 + win_idx``, at most CK + width rows)."""
    from .. import tree_verify as tv
    from .int8_kv import dequantize_kv

    start, end = rc.tail_start, rc.n0 + shape.width
    tail_keys = dequantize_kv(keys[:, :, start:end], k_scales[:, :, start:end])
    tail_values = dequantize_kv(values[:, :, start:end], v_scales[:, :, start:end])
    tail_meta = mx.array([rc.n0 - start, 0, rc.m], dtype=mx.int32)

    def shared(q, lengths, slots, scale, plan, t_tokens, nc):
        return tile_partials_q8(
            q,
            keys,
            k_scales,
            values,
            v_scales,
            lengths,
            slots,
            scale,
            plan,
            t_tokens,
            nc,
        )

    return tv._tree_attention_fused(
        queries,
        keys,
        values,
        shape,
        rc,
        scale_arr,
        dims,
        shared_partials=shared,
        tail_kv=(tail_keys, tail_values, tail_meta),
    )


def install() -> bool:
    """Install the int8 tree-attention kernels; False (nothing installed) where the bf16 tile kernel
    itself cannot run (no M5-class tensor ops), so ``require_int8_kernels`` refuses int8 there."""
    from .. import tree_verify as tv
    from . import ragged_attention as ra

    if not ra.tile_ready():
        return False
    tv.TILE_PARTIALS_Q8 = tile_partials_q8
    tv.TREE_ATTENTION_FUSED_Q8 = tree_attention_fused_q8
    return True
