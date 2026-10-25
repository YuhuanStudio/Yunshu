"""The grouped lane kernel's source is built for any member count and weight format (CPU)."""

import pytest

pytest.importorskip("mlx.core")

from yunshu_engine.kernels import lane_virtual
from yunshu_engine.kernels.tensorfold import lane_qmm as q


@pytest.fixture(autouse=True)
def no_device(monkeypatch):
    monkeypatch.setattr(q, "_resolve_variant", lambda: "portable")


@pytest.mark.parametrize("count", [2, 3, 4])
@pytest.mark.parametrize("bits", [4, 5])
def test_source_selects_member_and_keeps_arithmetic(count, bits):
    src = lane_virtual._source(count, bits)
    assert "auto Y = " in src and f"Y{count - 1}" in src
    assert "(n0 / NT) * KG" in src and "threadgroup_position_in_grid.x * KG" not in src
    assert f"gcol >= O{count - 1}" in src and f"O{count}" not in src
    assert "const int N = " in src
    base = (
        q._MAIN_TILED
        if bits == 4
        else __import__(
            "yunshu_engine.kernels.tensorfold.lane_widen", fromlist=["BYTES"]
        ).BYTES
    )
    # the matmul op and the fma epilogue are the original's, untouched
    for token in ("matmul2d", "fma(", "part["):
        assert src.count(token) == base.count(token)


def test_group_partition_ignores_unmatched_members():
    class Fake:
        def __init__(self, n, bits):
            self.output_dims, self.bits, self.group_size = n, bits, 64
            self.input_dims = 5120

        def __contains__(self, key):
            return False

    assert lane_virtual._eligible(Fake(48, 5))
    assert not lane_virtual._eligible(Fake(48, 3))
    assert q.split_k(6144, 5120) == q.split_k(48, 5120) == 8  # z, b, a share a split
    assert q.split_k(1024, 5120) == 8  # k and v
    assert q.split_k(10240, 5120) != q.split_k(6144, 5120)  # qkv keeps its own launch


@pytest.mark.parametrize("count", [2, 3])
def test_pipelined_body_groups_and_keeps_arithmetic(count):
    from yunshu_engine.kernels.tensorfold import lane_m5, lane_pipe

    src = lane_virtual._source(count, 4, pipe=True)
    assert (
        "(n0 / NT) * KG * TILE_V" in src
        and "threadgroup_position_in_grid.x * KG" not in src
    )
    assert f"gcol >= O{count - 1}" in src
    # same op descriptor, epilogue fma and slice-add order as the unpipelined M5 body
    for token in (
        "matmul2d_descriptor(16 * TMR, NT, GS",
        "fma(s[f][j], P[t * NF * 8 + i]",
        "part[((s2 - 1)",
    ):
        assert token in lane_pipe._PIPE and token in lane_m5._MAIN


def test_pipelined_body_is_off_by_default():
    assert q.PIPE == [False]
