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


@pytest.mark.parametrize("n", [700, 32700])
@pytest.mark.parametrize("t", [1, 8])
def test_decode_attention_bits_do_not_depend_on_the_buffer_capacity(n, t):
    """A hit restores a row into a buffer whose capacity differs from a cold row's: same codes,
    capacity 64-padded vs several thousand rows more, must give the same attention bits."""
    from yunshu_engine.kernels import ragged_attention as ra

    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    mx.random.seed(11)
    k = mx.random.normal((1, HKV, n, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, HKV, n, D)).astype(mx.bfloat16)
    q = mx.random.normal((1, H, t, D)).astype(mx.bfloat16)
    kq, ks, vq, vs = ra.quantize_kv_pair(k, v)

    def padded(x, cap):
        z = mx.zeros((1, HKV, cap - n, x.shape[-1]), dtype=x.dtype)
        return mx.concatenate([x, z], axis=2)

    outs = []
    for cap in (-(-n // 64) * 64, n + 3000, n + 4096 + 17):
        out = ra.ragged_decode_attention(
            q,
            padded(kq, cap),
            padded(vq, cap),
            mx.array([n], dtype=mx.int32),
            D**-0.5,
            max_length=n,
            k_scales=padded(ks, cap),
            v_scales=padded(vs, cap),
            row_lengths=[n],
        )
        mx.eval(out)
        outs.append(out)
    assert all(mx.array_equal(outs[0], o) for o in outs[1:])


@pytest.mark.parametrize("width", [1, 4, 8])
@pytest.mark.parametrize("n0", [300, 32700])
def test_tree_attention_int8_bits_do_not_depend_on_the_cache_capacity(
    monkeypatch, n0, width
):
    from mlx_vlm.models.cache import KVCache

    from yunshu_engine import tree_verify as tv
    from yunshu_engine.kernels.int8_kv import Int8KVCache
    from yunshu_engine.kernels.ragged_attention import tile_ready

    if not mx.metal.is_available() or not tile_ready():
        pytest.skip("needs the int8 tile kernel")
    from yunshu_engine.kernels import int8_tile

    monkeypatch.setattr(tv, "TILE_PARTIALS_Q8", int8_tile.tile_partials_q8)
    monkeypatch.setattr(
        tv, "TREE_ATTENTION_FUSED_Q8", int8_tile.tree_attention_fused_q8
    )
    mx.random.seed(2)
    n = n0 + width
    c = KVCache()
    c.update_and_fetch(
        mx.random.normal((1, HKV, n, D)).astype(mx.bfloat16),
        mx.random.normal((1, HKV, n, D)).astype(mx.bfloat16),
    )
    q = mx.random.normal((1, H, width, D)).astype(mx.bfloat16)
    outs = []
    for extra in (0, 3000, 4096 + 17):
        row = Int8KVCache.from_kv(c)
        if extra:
            row._reserve(n + extra, row.keys[..., :1, :])
        out = tv.tree_attention(
            q, row, D**-0.5, tv.TreeShape(list(range(-1, width - 1))), n0
        )
        mx.eval(out)
        outs.append(out)
    assert all(mx.array_equal(outs[0], o) for o in outs[1:])


@pytest.mark.parametrize("t", [1, 4, 8])
def test_int8_tile_decode_equals_the_bf16_tile_kernel_on_the_dequantized_cache(
    buffers, t
):
    """int8 plain decode / verify windows use the token-tile family: bit-identical to the bf16 tile
    kernel on the dequantized buffers (so int8 decode == int8 tree verify, like bf16)."""
    from yunshu_engine.kernels.int8_kv import dequantize_kv

    ra, kq, ks, vq, vs, q = buffers
    if not ra.tile_ready():
        pytest.skip("needs the tile kernel")
    rows = [0, 1, 2, 3]
    lengths = [LENS[r] for r in rows]
    kw = dict(
        max_length=max(lengths),
        row_lengths=lengths,
        slots=mx.array(rows, dtype=mx.int32),
        impl="tile",
    )
    got = ra.ragged_decode_attention(
        q[:, :, :t],
        kq,
        vq,
        mx.array(lengths, dtype=mx.int32),
        D**-0.5,
        k_scales=ks,
        v_scales=vs,
        **kw,
    )
    ref = ra.ragged_decode_attention(
        q[:, :, :t],
        dequantize_kv(kq, ks),
        dequantize_kv(vq, vs),
        mx.array(lengths, dtype=mx.int32),
        D**-0.5,
        **kw,
    )
    mx.eval(got, ref)
    assert mx.array_equal(got, ref)
