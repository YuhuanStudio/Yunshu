"""Fast-tree glue kernels: the sources differ from their omlx originals only where intended."""

import pytest

pytest.importorskip("mlx.core")
from yunshu_engine import tree_glue  # noqa: E402
from yunshu_engine.kernels.omlx import qwen35_gdn_verify_fused as gv  # noqa: E402
from yunshu_engine.kernels.omlx import qwen35_verify_qmm as vq  # noqa: E402


def test_add_rms_keeps_the_original_arithmetic_and_adds_sequential_sums():
    source = tree_glue.add_rms_lane_source()
    original = vq._ADD_RMS_SOURCE
    # every arithmetic line of the original kernel survives
    for line in (
        "v = ap[idx] + bp[idx];",
        "acc += xi * xi;",
        "T nv = w[idx] * static_cast<T>(vals[st * 4 + i] * inv);",
        "n_out[long(row) * D + idx] = nv;",
        "local_inv[0] = metal::precise::rsqrt(acc / D + eps[0]);",
    ):
        assert line in original and line in source
    # the xor-tree sums are gone; one thread adds a group's 64 values in order
    assert "simd_shuffle_xor(part" not in source
    assert "seq += float(tn[lid * 64 + i]);" in source
    assert "float seq = 0.0f;" in source
    # rows past the window write zero sums, like the xsum kernel
    assert "if (row >= M)" in source and "xs_out[lid * MP + row] = 0.0f;" in source


def test_norm_gate_keeps_the_original_arithmetic_and_adds_sequential_sums(monkeypatch):
    monkeypatch.setattr(gv, "_sigmoid_exp", lambda: "metal::exp")
    source = tree_glue.norm_gate_lane_source(1e-6)
    for line in (
        "InT normed = norm_w[lane * 4 + i] * static_cast<InT>(x[i] * inv);",
        "InT o = static_cast<InT>((g * sig) * static_cast<float>(normed));",
        "float sy = 1 / (1 + metal::exp(metal::abs(g)));",
    ):
        assert line in source
    assert "simd_shuffle_xor" not in source.replace("simd_shuffle_xor(ss", "")
    assert "seq += float(tile[slot * 128 + lane * 64 + i]);" in source
    assert "EPS" not in source and "SIGMOID_EXP" not in source


def test_query_and_tail_kernels_only_move_or_zero_values():
    # no arithmetic operator on the values: copies, selects and zeros
    for source in (tree_glue._TAIL, tree_glue._QUERIES):
        assert "exp(" not in source and "sqrt" not in source and " * 0" not in source
