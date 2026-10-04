"""Data-movement kernels for the fast tree's attention layers.

``tree_attention`` fed the tile kernel through about twenty small MLX ops per
layer (gathers, transposes, pads, concatenates). These kernels build the same
bytes in one launch each: the per-row tail copy of the keys and values, and the
token-fused query layouts. Values are copied or zero-filled, never computed, so
the attention arithmetic downstream is untouched.
"""

import mlx.core as mx

_KERNELS: dict = {}

_TAIL = """
    uint c = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    uint x = row % CAP2;
    uint rh = row / CAP2;
    uint h = rh % HKV;
    uint r = rh / HKV;
    int n0 = meta[0], start = meta[1], m = meta[2];
    int pos = -1;
    if (int(x) < m)
        pos = start + int(x);
    else if (int(x) < m + J)
        pos = n0 + win_idx[r * J + (x - uint(m))];
    long out = long(row) * D + c * 4;
    long src = (long(h) * CAP + pos) * D + c * 4;
    for (int i = 0; i < 4; ++i) {
        tk[out + i] = pos >= 0 ? keys[src + i] : T(0);
        tv[out + i] = pos >= 0 ? values[src + i] : T(0);
    }
"""

_QUERIES = """
    uint id = thread_position_in_grid.x;
    constexpr uint F = 8 * G;
    constexpr uint NA = HKV * F * D;
    constexpr uint NB = W * NA;
    if (id < NB) {
        uint dd = id % D;
        uint rest = id / D;
        uint f = rest % F;
        rest /= F;
        uint h = rest % HKV;
        uint r = rest / HKV;
        uint t = f / G, gh = f % G;
        qb[id] = t == 0 ? queries[((h * G + gh) * W + r) * D + dd] : T(0);
    } else {
        uint idx = id - NB;
        uint gi = idx / NA;
        idx %= NA;
        uint dd = idx % D;
        uint rest = idx / D;
        uint f = rest % F;
        uint h = rest / F;
        uint t = f / G, gh = f % G;
        uint token = gi * 8 + t;
        T v = (gi < NG && token < W) ? queries[((h * G + gh) * W + token) * D + dd] : T(0);
        if (gi == 0)
            qa0[idx] = v;
        else
            qa1[idx] = v;
    }
"""


def tail_copy(keys, values, win_idx, meta, *, width, depth_rows, cap2):
    """[W, HKV, cap2, D] key and value tails: ``meta[2]`` shared prefix rows, then
    each row's ancestor rows (``win_idx``), then zeros."""
    if "tail" not in _KERNELS:
        _KERNELS["tail"] = mx.fast.metal_kernel(
            name="yunshu_tree_tail_copy",
            input_names=["keys", "values", "win_idx", "meta"],
            output_names=["tk", "tv"],
            source=_TAIL,
        )
    hkv, cap, d = (int(s) for s in (keys.shape[1], keys.shape[2], keys.shape[3]))
    return _KERNELS["tail"](
        inputs=[keys, values, win_idx, meta],
        template=[
            ("T", keys.dtype),
            ("HKV", hkv),
            ("CAP", cap),
            ("D", d),
            ("J", depth_rows),
            ("CAP2", cap2),
        ],
        grid=(d // 4, width * hkv * cap2, 1),
        threadgroup=(min(64, d // 4), 1, 1),
        output_shapes=[(width, hkv, cap2, d)] * 2,
        output_dtypes=[keys.dtype] * 2,
    )


def fuse_queries(queries, hkv, groups):
    """The tile kernel's fused-token query layouts of [1, H, W, D] queries:
    two 8-token groups [1, HKV, 8G, D] (row = token * G + head) and the per-row
    single-token layout [W, HKV, 8G, D]; padding rows are zero."""
    if "queries" not in _KERNELS:
        _KERNELS["queries"] = mx.fast.metal_kernel(
            name="yunshu_tree_fuse_queries",
            input_names=["queries"],
            output_names=["qa0", "qa1", "qb"],
            source=_QUERIES,
        )
    _, h, w, d = (int(s) for s in queries.shape)
    g = h // hkv
    total = w * hkv * 8 * g * d + 2 * hkv * 8 * g * d
    return _KERNELS["queries"](
        inputs=[queries],
        template=[
            ("T", queries.dtype),
            ("H", h),
            ("HKV", hkv),
            ("G", g),
            ("W", w),
            ("D", d),
            ("NG", groups),
        ],
        grid=(total, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, hkv, 8 * g, d)] * 2 + [(w, hkv, 8 * g, d)],
        output_dtypes=[queries.dtype] * 3,
    )


_ADD_RMS_OLD_TAIL = """        // Sixteen lanes cover one 64-wide group of the normed row.
        for (int off = 1; off < 16; off <<= 1)
            part += simd_shuffle_xor(part, ushort(off));
        if ((lid & 15) == 0 && base < D)
            xs_out[long(row) * (D / 64) + base / 64] = part;
"""
_ADD_RMS_NEW_TAIL = ""

_ADD_RMS_SUMS = """
    // The lane matmul's group sums: one thread a 64-wide group, its normed
    // bf16 values added in order from 0.0f in a single float, exactly as the
    // lane xsum kernel does.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < D / 64) {
        float seq = 0.0f;
        for (int i = 0; i < 64; ++i)
            seq += float(tn[lid * 64 + i]);
        xs_out[lid * MP + row] = seq;
    }
"""


def add_rms_lane_source():
    """omlx's residual-add + RMSNorm kernel with the lane matmul's group sums.

    The original writes per-64 sums in an xor-tree order (rows, K/64); the lane
    projections take their own sequential sums as (K/64, MP). The wide fast tree
    therefore recomputed them in a separate launch after every norm. Rows past
    the window (padding to MP) write zeros like the xsum kernel does."""
    from .kernels.omlx import qwen35_verify_qmm as vq

    source = vq._ADD_RMS_SOURCE
    needle = "    uint row = threadgroup_position_in_grid.x;\n"
    loop = "            if (idx < D) {\n                T nv = w[idx] * static_cast<T>(vals[st * 4 + i] * inv);\n"
    keep = "                part += float(nv);\n"
    if (
        source.count(_ADD_RMS_OLD_TAIL) != 1
        or source.count(needle) != 1
        or source.count(loop) != 1
        or source.count(keep) != 1
    ):
        raise RuntimeError("add_rms source changed")
    source = source.replace(
        needle,
        needle
        + "    if (row >= M) {\n"
        + "        if (lid < D / 64) xs_out[lid * MP + row] = 0.0f;\n"
        + "        return;\n"
        + "    }\n",
    )
    source = source.replace(keep, keep + "                tn[idx] = nv;\n")
    source = source.replace(
        "    threadgroup float local_sums[32];\n",
        "    threadgroup float local_sums[32];\n    threadgroup T tn[D];\n",
    )
    return source.replace(_ADD_RMS_OLD_TAIL, "").rstrip("\n") + "\n" + _ADD_RMS_SUMS


def add_rms_lane(a, b, norm):
    """``(a + b, rms_norm(a + b), lane group sums)``; the first two are omlx's
    fused kernel's bytes, the sums the lane xsum kernel's bits. The sums are
    registered for the lane projections that read ``normed``."""
    from .kernels.tensorfold import lane_qmm as q

    if "add_rms" not in _KERNELS:
        _KERNELS["add_rms"] = mx.fast.metal_kernel(
            name="yunshu_tree_add_rms_sums",
            input_names=["a", "b", "w", "eps"],
            output_names=["s_out", "n_out", "xs_out"],
            source=add_rms_lane_source(),
        )
    d = int(a.shape[-1])
    rows = a.size // d
    mp = 16 * ((rows + 15) // 16)
    s_out, n_out, xs = _KERNELS["add_rms"](
        inputs=[a, b, norm.weight, mx.array([norm.eps], dtype=mx.float32)],
        template=[("T", a.dtype), ("D", d), ("M", rows), ("MP", mp)],
        grid=(1024 * mp, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[a.shape, a.shape, (d // 64, mp)],
        output_dtypes=[a.dtype, a.dtype, mx.float32],
    )
    q._xs_cache[id(n_out)] = (n_out, xs)
    while len(q._xs_cache) > 4:
        q._xs_cache.pop(next(iter(q._xs_cache)))
    return s_out, n_out, xs


_NG_OLD_TAIL = """    // Per-64 sums of the output feed the out projection.
    for (int off = 1; off < 16; off <<= 1)
        part += simd_shuffle_xor(part, ushort(off));
    if ((lane & 15) == 0)
        xs[row * 2 + lane / 16] = part;
"""
_NG_NEW_TAIL = """    // The lane out projection's group sums: the 128 outputs of this row are two
    // 64-wide groups, each added in order from 0.0f in one float exactly as the
    // lane xsum kernel does. XS is (K / 64, MP) with K = Hv * 128.
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 2) {
        float seq = 0.0f;
        for (int i = 0; i < 64; ++i)
            seq += float(tile[slot * 128 + lane * 64 + i]);
        xs[((row % Hv) * 2 + lane) * MP + row / Hv] = seq;
    }
"""


def norm_gate_lane_source(eps):
    """omlx's GDN norm-gate kernel writing the lane out projection's group sums."""
    from .kernels.omlx import qwen35_gdn_verify_fused as gv

    source = gv._NORM_GATE
    head = "    uint row = thread_position_in_grid.y;\n"
    if source.count(_NG_OLD_TAIL) != 1 or source.count(head) != 1:
        raise RuntimeError("norm-gate source changed")
    source = source.replace(
        head,
        head
        + "    if (row / Hv >= M) {\n"
        + "        if (lane < 2) xs[((row % Hv) * 2 + lane) * MP + row / Hv] = 0.0f;\n"
        + "        return;\n"
        + "    }\n"
        + "    threadgroup InT tile[8 * 128];\n"
        + "    const uint slot = thread_position_in_threadgroup.y;\n",
    )
    keep = "        part += float(o);\n"
    if source.count(keep) != 1:
        raise RuntimeError("norm-gate source changed")
    source = source.replace(
        keep, keep + "        tile[slot * 128 + lane * 4 + i] = o;\n"
    )
    source = source.replace(_NG_OLD_TAIL, _NG_NEW_TAIL)
    return source.replace("EPS", f"{float(eps)!r}f").replace(
        "SIGMOID_EXP", gv._sigmoid_exp()
    )


def norm_gate_lane(y, z, norm, hv):
    """The GDN output norm and gate; returns the out-projection input, whose lane
    group sums are registered for the projection that reads it."""
    from .kernels.tensorfold import lane_qmm as q

    eps = float(norm.eps)
    key = ("norm_gate", eps)
    if key not in _KERNELS:
        _KERNELS[key] = mx.fast.metal_kernel(
            name="yunshu_tree_norm_gate_sums_" + f"{eps:.0e}".replace("-", "m"),
            input_names=["y", "z", "norm_w"],
            output_names=["out", "xs"],
            source=norm_gate_lane_source(eps),
        )
    w = int(y.shape[1])
    mp = 16 * ((w + 15) // 16)
    width = hv * 128
    out, xs = _KERNELS[key](
        inputs=[y, z, norm.weight],
        template=[("InT", y.dtype), ("Hv", hv), ("M", w), ("MP", mp)],
        grid=(32, mp * hv, 1),
        threadgroup=(32, 8, 1),
        output_shapes=[(1, w, width), (width // 64, mp)],
        output_dtypes=[y.dtype, mx.float32],
    )
    q._xs_cache[id(out)] = (out, xs)
    while len(q._xs_cache) > 4:
        q._xs_cache.pop(next(iter(q._xs_cache)))
    return out
