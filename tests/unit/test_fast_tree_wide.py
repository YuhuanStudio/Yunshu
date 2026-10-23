"""Fast-tree GDN kernels past one simdgroup's 32 lanes: a 48-row chain equals 32 + 16 rows
(the kernels alone; the verify window itself stays at 32 rows).

Rows 32.. read their gates directly instead of through the lane shuffle; any slip there
changes their outputs or the replayed state."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from yunshu_engine import dflash_fast as fast
from yunshu_engine import tree_verify as tv
from yunshu_engine.dflash_plan import FastShape

HK, HV, DK, DV = 2, 4, 128, 64


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


def _inputs(w):
    mx.random.seed(2)
    q = mx.random.normal((1, w, HK, DK)).astype(mx.bfloat16)
    k = (0.1 * mx.random.normal((1, w, HK, DK))).astype(mx.bfloat16)
    v = mx.random.normal((1, w, HV, DV)).astype(mx.bfloat16)
    a = mx.random.normal((1, w, HV)).astype(mx.bfloat16)
    b = mx.random.normal((1, w, HV)).astype(mx.bfloat16)
    return q, k, v, a, b


def _chain(w):
    return FastShape(mx.arange(-1, w - 1, dtype=mx.int32), w - 1, is_chain=True)


def _forward(layer, state, rows):
    w = int(rows[0].shape[1])
    return fast.gdn_forward(_chain(w), layer, state, *rows)


def _replay(gate_rows, layer, state, keep):
    path = mx.arange(keep, dtype=mx.int32)
    return gate_rows.replay(layer, state, path, mx.array([keep], dtype=mx.int32))


@pytest.mark.parametrize("wide", [40, 48])
def test_wide_chain_equals_32_then_the_rest(wide):
    layer = _layer()
    state = (0.1 * mx.random.normal((1, HV, DV, DK))).astype(mx.float32)
    rows = _inputs(wide)
    y_wide, g_wide = _forward(layer, state, rows)
    head = tuple(x[:, :32] for x in rows)
    tail = tuple(x[:, 32:] for x in rows)
    y_head, g_head = _forward(layer, state, head)
    state32 = _replay(g_head, layer, state, 32)
    y_tail, g_tail = _forward(layer, state32, tail)
    assert mx.array_equal(y_wide[:, :32], y_head).item()
    assert mx.array_equal(y_wide[:, 32:], y_tail).item()
    # the commit's replay over the whole wide path equals 32 rows then the rest
    want = _replay(g_tail, layer, state32, wide - 32)
    assert mx.array_equal(_replay(g_wide, layer, state, wide), want).item()


def test_verify_windows_stay_at_32_rows():
    # wider windows are not bit-equal to decode yet (see tree_verify.FAST_MAX_ROWS)
    assert tv.MAX_ROWS == tv.FAST_MAX_ROWS == 32
