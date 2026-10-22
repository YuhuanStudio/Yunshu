"""Research: two original M16 ops per K group, sharing scales and staging.

A M32 row block retains C[2] but only one M16 destination P at a time, reducing
live temporary registers relative to a M32 tensor op. K accumulation/reduction
order and each M16 op stay unchanged. Only target 4/5-bit rows17..32 participate.
"""

import hashlib
from pathlib import Path


def transform(source):
    descriptor = "matmul2d_descriptor(16 * TMR, NT,"
    if source.count(descriptor) != 1:
        raise RuntimeError("unexpected tensor descriptor")
    source = source.replace(descriptor, "matmul2d_descriptor(16, NT,")
    starts = ["    auto a = tA.slice(g * GS, 0);", "    auto a = tA.slice(g * 64, 0);"]
    needle = next((s for s in starts if source.count(s) == 1), None)
    if needle is None:
        raise RuntimeError("unexpected input slice")
    begin = source.index(needle)
    end = source.index("\n  }\n", begin)
    original = source[begin:end]
    body = original.replace(
        needle,
        "    for (int tb = 0; tb < TMR; tb++) {\n"
        + needle.replace(", 0);", ", tb * 16);"),
    )
    body = body.replace("for (int t = 0; t < TMR; t++)", "for (int t = 0; t < 1; t++)")
    body = body.replace("rb + t * 16", "rb + (tb + t) * 16").replace(
        "C[t][i]", "C[tb + t][i]"
    )
    barrier = "    simdgroup_barrier(mem_flags::mem_threadgroup);   // the op has read the stage before the next group's widening\n"
    staged = barrier in body
    body = body.replace(barrier, "") + "\n    }"
    if staged:
        body += "\n" + barrier.rstrip("\n")
    return source[:begin] + body + source[end:]


def preflight(root):
    """Execute only source-string generators, without importing MLX."""
    folder = Path(root) / "python/yunshu_engine/kernels/tensorfold"
    main, widen = {}, {}
    exec(compile((folder / "lane_m5.py").read_text(), "lane_m5_source", "exec"), main)
    exec(
        compile((folder / "lane_widen_m5.py").read_text(), "lane_widen_source", "exec"),
        widen,
    )
    hashes = {}
    for name, source in [("main", main["_MAIN"]), ("bytes", widen["BYTES"])]:
        new = transform(source)
        suffix = "  threadgroup float part["
        assert source[source.index(suffix) :] == new[new.index(suffix) :]
        assert source.count("simdgroup_barrier") == new.count("simdgroup_barrier")
        assert "matmul2d_descriptor(16, NT," in new
        assert (
            "C[tb + t][i] = fma(s[f][j], P[t * NF * 8 + i], fma(bb[f][j], r ? xs1 : xs0, C[tb + t][i]));"
            in new
        )
        hashes[name] = hashlib.sha256(new.encode()).hexdigest()
    return hashes


def install():
    from yunshu_engine.kernels.tensorfold import lane_qmm as q
    from yunshu_engine.kernels.tensorfold import lane_widen

    if q._resolve_variant() != "m5":
        raise RuntimeError("M5 experiment only")
    originals = {}
    for name, source in [
        ("main", q._MAIN),
        ("main_tiled", q._MAIN_TILED),
        ("bytes", lane_widen.BYTES),
    ]:
        base = q._kernel(name)
        originals[name] = base
        fast = q._Baked(
            "lane_pair16_" + name,
            transform(source),
            ["X", "XS", "Wq", "SBt", "mdims"],
            ["Y"],
        )

        def dispatch(*, template=(), _base=base, _fast=fast, **kwargs):
            cfg = dict(template)
            rows = int(kwargs["inputs"][0].shape[0])
            use = (
                16 < rows <= 32
                and cfg["N"] < 100000
                and cfg.get("BITS", 4) in (4, 5)
                and cfg.get("GS", 64) == 64
            )
            return (_fast if use else _base)(template=template, **kwargs)

        q._kernels[name] = dispatch
    original = q.lane_matmul

    def matmul(x, weight, sbt, **kwargs):
        rows = int(x.size) // int(x.shape[-1])
        bits = q.weight_bits(weight, int(x.shape[-1]))
        if (
            16 < rows <= 32
            and int(weight.shape[0]) < 100000
            and bits in (4, 5)
            and kwargs.get("group", 64) == 64
            and kwargs.get("nt", 32) == 32
        ):
            kwargs["row_block"] = 32
        return original(x, weight, sbt, **kwargs)

    q.lane_matmul = matmul

    def undo():
        q._kernels.update(originals)
        q.lane_matmul = original

    return undo
