"""Research-only 16-output tiles over the existing 32-output tiled buffer.

Keep the original split-K and K-group order. Halve each SIMD group's output
registers and shared reduction footprint, doubling the number of threadgroups.
The 16-row tensor op retains K=64; parity must certify N=16 vs N=32.
"""


def transform(source):
    old = "(int64_t)(threadgroup_position_in_grid.x * KG + g) * (NT * GS / 2)"
    new = (
        "(int64_t)((threadgroup_position_in_grid.x / 2) * KG + g) * (32 * GS / 2)"
        " + (threadgroup_position_in_grid.x % 2) * (16 * GS / 2)"
    )
    if source.count(old) != 1:
        raise RuntimeError("unexpected tiled lane addressing")
    return source.replace(old, new)


def install(*, row_block=None):
    from yunshu_engine.kernels.tensorfold import lane_qmm as q

    if q._resolve_variant() != "m5":
        raise RuntimeError("M5 experiment only")
    base = q._kernel("main_tiled")
    half = q._Baked(
        "lane_half_tile",
        transform(q._MAIN_TILED),
        ["X", "XS", "Wq", "SBt", "mdims"],
        ["Y"],
    )
    original = q.lane_matmul

    def dispatch(*, template=(), **kwargs):
        return (half if dict(template)["NT"] == 16 else base)(
            template=template, **kwargs
        )

    def matmul(x, weight, sbt, **kwargs):
        if (
            kwargs.get("tiled")
            and q.weight_bits(weight, int(x.shape[-1])) == 4
            and kwargs.get("nt", 32) == 32
        ):
            kwargs["nt"] = 16
            rows = int(x.size) // int(x.shape[-1])
            if row_block is not None and 16 < rows <= 32:
                kwargs["row_block"] = row_block
        return original(x, weight, sbt, **kwargs)

    q._kernels["main_tiled"] = dispatch
    q.lane_matmul = matmul

    def undo():
        q._kernels["main_tiled"] = base
        q.lane_matmul = original

    return undo
