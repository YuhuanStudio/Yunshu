"""Batched gated-delta kernel with per-row lengths equals the upstream
kernel run one row at a time (bit for bit), including rollback states."""

import pytest

mx = pytest.importorskip("mlx.core")


def test_rows_match_upstream_per_row():
    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    from mlx_vlm.models.qwen3_5.gated_delta import (
        _compute_g_beta,
        gated_delta_update_with_states,
    )

    from yunshu_engine.kernels.gdn_rows import gated_delta_rows

    B, T, Hk, Hv, Dk, Dv = 4, 5, 2, 4, 128, 128
    lens = [1, 5, 3, 2]
    mx.random.seed(1)
    q = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    k = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    v = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
    a = mx.random.normal((B, T, Hv)).astype(mx.bfloat16)
    b = mx.random.normal((B, T, Hv)).astype(mx.bfloat16)
    A_log = mx.random.normal((Hv,))
    dt_bias = mx.random.normal((Hv,))
    state = mx.random.normal((B, Hv, Dv, Dk))
    g, beta = _compute_g_beta(A_log, a, b, dt_bias)
    y, st, hist = gated_delta_rows(q, k, v, g, beta, state, mx.array(lens))
    mx.eval(y, st, hist)
    for r, w in enumerate(lens):
        sl = slice(r, r + 1)
        ry, rs, rh = gated_delta_update_with_states(
            q[sl, :w],
            k[sl, :w],
            v[sl, :w],
            a[sl, :w],
            b[sl, :w],
            A_log,
            dt_bias,
            state[sl],
            state_steps=w - 1,
        )
        assert mx.array_equal(y[r, :w], ry[0])
        assert mx.array_equal(st[r], rs[0])
        assert not mx.any(y[r, w:]).item()
        for j in range(w - 1):
            assert mx.array_equal(hist[r, j], rh[0, j])
