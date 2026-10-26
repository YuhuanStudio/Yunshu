"""Pointer-induction experiment for the existing NAX lane weight loader.

Changes address generation only: same byte addresses, dequantization operations,
threadgroup layout and NAX accumulation order. CPU import never initializes MLX.
"""

from __future__ import annotations


def induction_header(source):
    replacements = (
        (
            "      bi(READS*(sg*32+lane)/CP), bj(READS*(sg*32+lane)%CP), g(0) {}",
            "      bi(READS*(sg*32+lane)/CP), bj(READS*(sg*32+lane)%CP), g(0) {\n"
            "    W += (((n/32)*KG)*32+n%32)*(64*bits/8) + bj*BP;\n"
            "    SB += 2*n;\n  }",
        ),
        (
            "    const device uint8_t* src = W + (((n/32)*KG+g)*32+n%32)*(64*bits/8) + bj*BP;",
            "    const device uint8_t* src = W;",
        ),
        (
            "    const T scale = SB[2*(g*N+n)], bias = SB[2*(g*N+n)+1];",
            "    const T scale = SB[0], bias = SB[1];",
        ),
        (
            "  void next() thread { g++; }",
            "  void next() thread { W += 32*(64*bits/8); SB += 2*N; }",
        ),
    )
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError("NAX loader source changed: " + before)
        source = source.replace(before, after)
    return source


def install():
    from functools import cache

    from yunshu_engine.kernels import nax_prefill

    original = nax_prefill.lane_header
    nax_prefill.lane_header = cache(lambda: induction_header(original()))
    nax_prefill.compiled.cache_clear()
