"""Span invariance of the prefill kernels the round driver may use: a token's
bits must not depend on whether its atom (512 tokens on the absolute grid) ran
alone or inside a longer run of full atoms."""

import pytest

mx = pytest.importorskip("mlx.core")


def _inputs(T, seed=0):
    mx.random.seed(seed)
    B, Hk, Hv, Dk, Dv = 1, 4, 8, 128, 128
    q = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16) * 0.1
    k = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16) * 0.1
    v = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
    g = -mx.abs(mx.random.normal((B, T, Hv))).astype(mx.bfloat16) * 0.1
    beta = mx.sigmoid(mx.random.normal((B, T, Hv))).astype(mx.bfloat16)
    state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.bfloat16)
    return q, k, v, g, beta, state


@pytest.mark.skipif(
    not hasattr(mx.fast, "gated_delta_update"), reason="no fused GDN update"
)
def test_chunked_gdn_atoms_equal_one_run_of_atoms():
    q, k, v, g, beta, state = _inputs(1536)
    whole, s_whole = mx.fast.gated_delta_update(q, k, v, g, beta, state, None)
    outs, s = [], state
    for i in range(0, 1536, 512):
        sl = slice(i, i + 512)
        o, s = mx.fast.gated_delta_update(
            q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], s, None
        )
        outs.append(o)
    mx.eval(whole, s_whole, outs, s)
    assert mx.array_equal(mx.concatenate(outs, axis=1), whole).item()
    assert mx.array_equal(s, s_whole).item()


def test_nax_prefill_rows_do_not_depend_on_the_run_length():
    import mlx.nn as nn

    from yunshu_engine.kernels import lane_linear, nax_prefill

    if not nax_prefill.enable():
        pytest.skip("NAX prefill needs the measured M5 / MLX pair")
    try:
        mx.random.seed(3)
        lin = nn.Linear(1024, 4096, bias=False)
        lin.set_dtype(mx.bfloat16)
        q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=4)
        lane = lane_linear.LaneLinear.from_quantized(q)
        x = mx.random.normal((2048, 1024)).astype(mx.bfloat16)
        whole = lane.prefill([x])[0]
        parts = lane.prefill([x[i : i + 512] for i in range(0, 2048, 512)])
        mx.eval(whole, parts)
        assert mx.array_equal(mx.concatenate(parts, axis=0), whole).item()
        assert nax_prefill._dispatches >= 2  # whole run and atoms ran NAX
    finally:
        nax_prefill.disable()
