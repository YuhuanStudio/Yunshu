"""Research-only two fixed 32-row ops per group, sharing a weight group.

Unlike a 64-row matmul descriptor, each op retains the shipped 32-row descriptor
and per-group fma/reduction order. Only the row blocks' scheduling changes.
"""


def transform(source):
    source = source.replace(
        "matmul2d_descriptor(16 * TMR, NT, GS",
        "matmul2d_descriptor(16 * ((TMR < 2) ? TMR : 2), NT, GS",
    )
    begin = source.index("    auto a = tA.slice(g * GS, 0);")
    end = source.index("\n  }\n  // K slices", begin)
    original = source[begin:end]
    replacement = original.replace(
        "    auto a = tA.slice(g * GS, 0);",
        "    for (int tb = 0; tb < TMR; tb += 2) {\n    auto a = tA.slice(g * GS, tb * 16);",
    )
    replacement = replacement.replace(
        "for (int t = 0; t < TMR; t++)", "for (int t = 0; t < min(TMR, 2); t++)"
    )
    replacement = replacement.replace("rb + t * 16", "rb + (tb + t) * 16")
    replacement = replacement.replace("C[t][i]", "C[tb + t][i]")
    replacement += "\n    }"
    if replacement == original:
        raise RuntimeError("paired32 transform did not engage")
    return source[:begin] + replacement + source[end:]


def install():
    from yunshu_engine.kernels.tensorfold import lane_qmm

    original_main = lane_qmm._MAIN
    original_tiled = lane_qmm._MAIN_TILED
    original_matmul = lane_qmm.lane_matmul
    lane_qmm._MAIN = transform(original_main)
    lane_qmm._MAIN_TILED = transform(original_tiled)
    lane_qmm._kernels.clear()

    def matmul(x, weight, sbt, **kwargs):
        if (
            int(x.size) // int(x.shape[-1]) > 32
            and lane_qmm.weight_bits(weight, int(x.shape[-1])) == 4
            and kwargs.get("nt", 32) == 32
        ):
            kwargs["row_block"] = 64
        return original_matmul(x, weight, sbt, **kwargs)

    lane_qmm.lane_matmul = matmul

    def uninstall():
        lane_qmm._MAIN = original_main
        lane_qmm._MAIN_TILED = original_tiled
        lane_qmm.lane_matmul = original_matmul
        lane_qmm._kernels.clear()

    return uninstall
