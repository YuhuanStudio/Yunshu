"""int8 key-parallel decode attention at the 27B's shape (24 query heads, 4 KV heads, head_dim 256):
a row's output bits depend on that row alone, not on the other rows, their lengths, the batch's
``max_length`` or the window width (solo == concurrent, plain decode == verify window)."""

import pytest

mx = pytest.importorskip("mlx.core")

H, HKV, D, CAP = 24, 4, 256, 3072
LENS = [700, 1300, 2100, 129]


@pytest.fixture(scope="module")
def buffers():
    from yunshu_engine.kernels import ragged_attention as ra

    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    mx.random.seed(3)
    k = mx.random.normal((len(LENS), HKV, CAP, D)).astype(mx.bfloat16)
    v = mx.random.normal((len(LENS), HKV, CAP, D)).astype(mx.bfloat16)
    kq, ks, vq, vs = ra.quantize_kv_pair(k, v)
    q = mx.random.normal((len(LENS), H, 8, D)).astype(mx.bfloat16)
    mx.eval(kq, ks, vq, vs, q)
    return ra, kq, ks, vq, vs, q


def _attend(b, rows, t, impl="auto", max_length=None, qfrom=0):
    ra, kq, ks, vq, vs, q = b
    lengths = [LENS[r] for r in rows]
    slots = mx.array(rows, dtype=mx.int32)
    out = ra.ragged_decode_attention(
        q[mx.array(rows)][:, :, qfrom : qfrom + t],
        kq,
        vq,
        mx.array(lengths, dtype=mx.int32),
        D**-0.5,
        max_length=max_length or max(lengths),
        k_scales=ks,
        v_scales=vs,
        row_lengths=lengths,
        slots=slots,
        impl=impl,
    )
    mx.eval(out)
    return out


@pytest.mark.parametrize("t", [1, 3, 8])
@pytest.mark.parametrize("row", [0, 1, 2, 3])
def test_row_bits_do_not_depend_on_the_other_rows(buffers, row, t):
    solo = _attend(buffers, [row], t)
    for rows in ([0, 1, 2, 3], [3, 2, 1, 0], [row, 2], [2, row]):
        out = _attend(buffers, rows, t)
        assert mx.array_equal(out[rows.index(row)], solo[0]), (rows, row, t)
    wide = _attend(buffers, [row], t, max_length=CAP)
    assert mx.array_equal(wide[0], solo[0])


@pytest.mark.parametrize("row", [0, 2])
def test_a_token_attends_the_same_whatever_the_window_width(buffers, row):
    # the last token of an 8-token window has the same causal limit (the row's length) and the
    # same query as a one-token decode of query 7
    wide = _attend(buffers, [row], 8)
    one = _attend(buffers, [row], 1, qfrom=7)
    assert mx.array_equal(wide[0, :, 7], one[0, :, 0])
