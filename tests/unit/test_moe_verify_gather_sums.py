"""Yunshu's promoted bias sums in the gathered qmv header equal omlx's MLX 0.32.4 transcription
(omlx 94363f89: every operand of a multi-term sum is cast to float before the additions), and the
gathered kernel keeps requiring N % 8 == 0 whatever the ordinary qmv rule says."""

import re

from yunshu_engine.kernels.omlx import moe_verify_gather as gather
from yunshu_engine.kernels.qmv_compat import stock_load_vector

FOUR = "sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);"
EIGHT = (
    "sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) + "
    "float(x[i + 4]) + float(x[i + 5]) + float(x[i + 6]) + float(x[i + 7]);"
)


def _sums(header):
    return [" ".join(s.split()) for s in re.findall(r"sum \+= .*?;", header, re.DOTALL)]


def test_promoted_sums_match_the_upstream_transcription():
    sums = _sums(stock_load_vector(gather._HEADER, True))
    # load_vector and load_vector_safe: 4-bit, 5-bit, 6-bit multi-term sums, 8-bit single term
    assert sums.count(FOUR) == 4  # 4-bit and 6-bit, in both functions
    assert sums.count(EIGHT) == 2  # 5-bit, in both functions
    assert [s for s in sums if s not in (FOUR, EIGHT)] == ["sum += float(x[i]);"] * 2


def test_unpromoted_header_is_the_pre_0_32_4_arithmetic():
    sums = _sums(stock_load_vector(gather._HEADER, False))
    assert all("float(" not in s for s in sums)


def test_gathered_kernel_still_requires_n_multiple_of_8():
    # the ordinary-qmv rule drops the N % 8 condition on 0.32.4 (f8aaf49d); the gathered one keeps it,
    # and supported() checks it on its own, so an N = 12 weight is never routed here
    import mlx.core as mx

    def linear(n):
        class Weight:
            shape = (4, n, 64)
            ndim = 3

        class Scales:
            shape = (4, n, 1)
            dtype = mx.bfloat16

        parts = {"weight": Weight(), "scales": Scales(), "biases": Scales()}
        return type("L", (dict,), {"bits": 4, "group_size": 64, "mode": "affine"})(
            parts
        )

    assert gather.supported(linear(16), mx.bfloat16) is True
    assert gather.supported(linear(12), mx.bfloat16) is False
