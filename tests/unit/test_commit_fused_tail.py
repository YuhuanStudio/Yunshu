"""The fused replay + conv-tail kernel equals the separate replay kernel and conv-tail gather."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from yunshu_engine import tree_verify as tv

HK, HV, DK, DV, W = 2, 4, 128, 64, 16


def _layer():
    mx.random.seed(1)
    return SimpleNamespace(
        num_k_heads=HK,
        num_v_heads=HV,
        head_k_dim=DK,
        head_v_dim=DV,
        A_log=mx.random.normal((HV,)).astype(mx.bfloat16),
        dt_bias=mx.random.normal((HV,)).astype(mx.bfloat16),
    )


def _rows():
    mx.random.seed(2)
    pk = mx.random.normal((1, W, HK, DK)).astype(mx.bfloat16)
    pv = mx.random.normal((1, W, HV, DV)).astype(mx.bfloat16)
    pa = mx.random.normal((1, W, HV)).astype(mx.bfloat16)
    pb = mx.random.normal((1, W, HV)).astype(mx.bfloat16)
    return pk, pv, pa, pb


@pytest.mark.parametrize(
    "path", [[0], [0, 1], [0, 3, 4], [0, 1, 2, 3, 4], [0, 5, 9, 14], list(range(16))]
)
def test_fused_commit_is_bit_identical_to_the_separate_ops(path):
    layer, rows = _layer(), _rows()
    channels = 2 * HK * DK + HV * DV
    assert tv.tail_fits(
        layer,
        mx.zeros((1, 3, channels), dtype=mx.bfloat16),
        mx.zeros((1, W, channels), dtype=mx.bfloat16),
    )
    state = mx.random.normal((1, HV, DV, DK)).astype(mx.float32)
    conv_prev = mx.random.normal((1, 3, channels)).astype(mx.bfloat16)
    mixed = mx.random.normal((1, W, channels)).astype(mx.bfloat16)
    m = len(path)
    path_arr = mx.array(path + [0] * (W - m), dtype=mx.int32)
    count_arr = mx.array([m], dtype=mx.int32)
    conv_idx = mx.array(([0, 1, 2] + [3 + r for r in path])[-3:], dtype=mx.int32)
    tail_idx = mx.array(path[-3:], dtype=mx.int32) if m >= 3 else None

    want_state = tv.replay_path(layer, state, rows, path_arr, count_arr)
    want_tail = tv.conv_tail(conv_prev, mixed, path, conv_idx, tail_idx)
    got_state, got_tail = tv.replay_path_tail(
        layer, state, rows, path_arr, count_arr, conv_prev, mixed, conv_idx
    )
    assert mx.array_equal(got_state, want_state).item()
    assert (
        got_tail.shape == want_tail.shape and mx.array_equal(got_tail, want_tail).item()
    )


def test_a_conv_state_wider_than_the_grid_falls_back():
    layer = _layer()
    wide = mx.zeros(
        (1, 3, 32 * DV * HV), dtype=mx.bfloat16
    )  # 3 * C exceeds the grid's threads
    assert not tv.tail_fits(layer, wide, wide)
    assert not tv.tail_fits(
        layer,
        mx.zeros((1, 3, 64), dtype=mx.bfloat16),
        mx.zeros((1, 4, 64), dtype=mx.float32),
    )


@pytest.mark.parametrize("path", [[0, 2], [0, 1, 2, 3], [0, 4, 7, 11, 15]])
def test_gate_rows_fused_commit_is_bit_identical(path):
    from yunshu_engine import dflash_fast

    layer = _layer()
    pk, pv, pa, pb = _rows()
    g, beta = dflash_fast._gate_values(pa, pb, layer.A_log, layer.dt_bias)
    rows = dflash_fast.GateRows(pk, pv, pa, pb, g, beta)
    channels = 2 * HK * DK + HV * DV
    state = mx.random.normal((1, HV, DV, DK)).astype(mx.float32)
    conv_prev = mx.random.normal((1, 3, channels)).astype(mx.bfloat16)
    mixed = mx.random.normal((1, W, channels)).astype(mx.bfloat16)
    m = len(path)
    path_arr = mx.array(path + [0] * (W - m), dtype=mx.int32)
    count_arr = mx.array([m], dtype=mx.int32)
    conv_idx = mx.array(([0, 1, 2] + [3 + r for r in path])[-3:], dtype=mx.int32)
    tail_idx = mx.array(path[-3:], dtype=mx.int32) if m >= 3 else None
    want_state = rows.replay(layer, state, path_arr, count_arr)
    want_tail = tv.conv_tail(conv_prev, mixed, path, conv_idx, tail_idx)
    got_state, got_tail = rows.replay_tail(
        layer, state, path_arr, count_arr, conv_prev, mixed, conv_idx
    )
    assert mx.array_equal(got_state, want_state).item()
    assert mx.array_equal(got_tail, want_tail).item()
