"""Research-only: feed packed 8-bit groups directly to the uint8 tensor op.

Eight-bit codes already are bytes; avoid the shared-memory copy and its two
SIMD barriers per K group. Keep descriptor, fmas, split-K and reductions.
"""


def transform(source):
    begin = source.index("  threadgroup uint stage_all[")
    end = source.index("  const device uint* Wv", begin)
    source = source[:begin] + source[end:]
    begin = source.index("    uint w[WPG + 1];")
    end = source.index("    float s[NF][4]", begin)
    source = source[:begin] + source[end:]
    source = source.replace(
        "    auto a = tA.slice(g * 64, 0);",
        """    auto a = tA.slice(g * 64, 0);
    tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> wb(
        (device uint8_t*)Wq, dextents<int32_t, 2>(K, N));
    auto b = wb.slice(g * 64, n0);
    tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> bt(
        (device uint8_t*)Wq + (int64_t)(threadgroup_position_in_grid.x * KG + g) * NT * 64,
        dextents<int32_t, 2>(64, NT));""",
    )
    source = source.replace(
        "    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();\n    op.run(a, b, P);",
        "    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();\n    if (TILED) op.run(a, bt, P); else op.run(a, b, P);",
    )
    source = source.replace(
        "    simdgroup_barrier(mem_flags::mem_threadgroup);   // the op has read the stage before the next group's widening\n",
        "",
    )
    if "stage[" in source or "simdgroup_barrier" in source:
        raise RuntimeError("unexpected staged source")
    return source


def direct_for(n, k, rows):
    """Screened schedule: narrow rows, plus FC where all tested widths improved."""
    return rows <= 16 or ((n, k) == (5120, 25600) and rows <= 32)


def install(*, selective=False):
    from yunshu_engine.kernels.tensorfold import lane_qmm as q
    from yunshu_engine.kernels.tensorfold import lane_widen

    if q._resolve_variant() != "m5":
        raise RuntimeError("M5 experiment only")
    base = q._kernel("bytes")
    fast = q._Baked(
        "lane_direct8",
        transform(lane_widen.BYTES),
        ["X", "XS", "Wq", "SBt", "mdims"],
        ["Y"],
    )

    def dispatch(*, template=(), **kwargs):
        config = dict(template)
        rows = int(kwargs["inputs"][0].shape[0])
        use = config["BITS"] == 8 and (
            not selective or direct_for(config["N"], config["K"], rows)
        )
        return (fast if use else base)(template=template, **kwargs)

    q._kernels["bytes"] = dispatch

    def undo():
        q._kernels["bytes"] = base

    return undo
