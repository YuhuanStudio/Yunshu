# Upstream (derived body): TensorFold MIT lane_qmm, as vendored in Yunshu.
"""Own address-only grouping: two original buffers, one unchanged lane op.

Select the projection pointer and local column at each column tile. No packed
weight copy, module mutation, or model-owned cache is needed. The original
per-projection split-K, row block, fma and slice-add order remain fixed.
"""

import mlx.core as mx

from yunshu_engine.kernels.lane_linear import LaneLinear
from yunshu_engine.kernels.tensorfold import lane_qmm as q

_KERNEL = None


def grouped_linears(members, x):
    global _KERNEL
    members = tuple(members)
    if len(members) != 2 or not all(isinstance(m, LaneLinear) for m in members):
        return None
    a, b = members
    rows = x.size // a.input_dims
    if not 0 < rows <= 128 or not all(
        m.bits == 4
        and m.tiled
        and m.output_dims % q.NT == 0
        and "bias" not in m
        and m.input_dims == a.input_dims
        and m.group_size == a.group_size
        for m in members
    ):
        return None
    split = q.split_k(a.output_dims, a.input_dims)
    if split != q.split_k(b.output_dims, b.input_dims):
        return None
    blocks = {
        16 if 16 < rows <= 48 and m.output_dims < 100_000 else None for m in members
    }
    if len(blocks) != 1:
        return None
    if _KERNEL is None:
        q._resolve_variant()
        body = q._MAIN_TILED
        needle = "  const int n0 = threadgroup_position_in_grid.x * NT;"
        assert body.count(needle) == 1
        body = body.replace(
            needle,
            """
        const int offset = threadgroup_position_in_grid.x * NT < N0 ? 0 : N0;
        const int VN = offset == 0 ? N0 : N1;
        const int n0 = threadgroup_position_in_grid.x * NT - offset;
        auto Wq = offset == 0 ? W0 : W1;
        auto SBt = offset == 0 ? S0 : S1;
        auto Y = offset == 0 ? Y0 : Y1;
        """,
        )
        assert "threadgroup_position_in_grid.x * KG" in body
        body = body.replace("threadgroup_position_in_grid.x * KG", "(n0 / NT) * KG")
        body = body.replace("g * N +", "g * VN +")
        body = body.replace("dextents<int32_t, 2>(K, N)", "dextents<int32_t, 2>(K, VN)")
        assert "Y[m * N + n" in body
        body = body.replace("Y[m * N + n", "Y[m * VN + n")
        _KERNEL = q._Baked(
            "wide6_virtual_columns",
            body,
            ["X", "XS", "W0", "S0", "W1", "S1", "mdims"],
            ["Y0", "Y1"],
        )
    dtype = x.dtype
    x2 = x.reshape(rows, a.input_dims).astype(mx.bfloat16)
    mp = 16 * ((rows + 15) // 16)
    mdims = q._mdims(rows, mp)
    xs = q._kernel("xsum")(
        inputs=[x2, mdims],
        template=[("K", a.input_dims), ("GS", a.group_size)],
        grid=(a.input_dims // a.group_size, mp, 1),
        threadgroup=(min(a.input_dims // a.group_size, 256), 1, 1),
        output_shapes=[(a.input_dims // a.group_size, mp)],
        output_dtypes=[mx.float32],
    )[0]
    block = blocks.pop() or min(mp, q.ROW_BLOCK)
    n = a.output_dims + b.output_dims
    outputs = _KERNEL(
        inputs=[x2, xs, a.weight, a.sbt, b.weight, b.sbt, mdims],
        template=[
            ("TMR", block // 16),
            ("N", n),
            ("N0", a.output_dims),
            ("N1", b.output_dims),
            ("K", a.input_dims),
            ("NT", q.NT),
            ("SK", split),
            ("GS", a.group_size),
            ("EDGE", int(mp % block != 0)),
        ],
        grid=(n // q.NT * 32 * split, -(-mp // block), 1),
        threadgroup=(32 * split, 1, 1),
        output_shapes=[(rows, a.output_dims), (rows, b.output_dims)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )
    return tuple(
        out.reshape(*x.shape[:-1], member.output_dims).astype(dtype)
        for out, member in zip(outputs, members, strict=True)
    )
