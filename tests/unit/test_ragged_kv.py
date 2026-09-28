"""Per-row-length KV cache and ragged decode attention (YUNSHU_RAGGED_KV)."""

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CI without Apple GPU
    pytest.skip("needs an Apple GPU", allow_module_level=True)

from mlx_lm.models.cache import BatchKVCache  # noqa: E402

from yunshu_engine.kernels import ragged_kv  # noqa: E402
from yunshu_engine.kernels.ragged_attention import (  # noqa: E402
    ragged_decode_attention,
    tile_ready,
)

H, HKV, D = 24, 4, 256


def _ref(q, k, v, lengths, T):
    """Per-row stock SDPA over each row's own keys (causal inside the window)."""
    rows = []
    for b, n in enumerate(lengths):
        toks = []
        for t in range(T):
            m = n - (T - 1 - t)
            toks.append(
                mx.fast.scaled_dot_product_attention(
                    q[b : b + 1, :, t : t + 1],
                    k[b : b + 1, :, :m],
                    v[b : b + 1, :, :m],
                    scale=D**-0.5,
                )
            )
        rows.append(mx.concatenate(toks, axis=2))
    return mx.concatenate(rows, axis=0)


@pytest.mark.parametrize("T", [1, 4])
@pytest.mark.parametrize(
    "lengths", [[37], [1030, 37], [5000, 1030, 37, 600], [17000, 37]]
)
def test_kernel_matches_sdpa_per_row(lengths, T):
    mx.random.seed(len(lengths) * 10 + T)
    cap = -(-max(lengths) // 256) * 256 + 256
    B = len(lengths)
    k = mx.random.normal((B, HKV, cap, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, HKV, cap, D)).astype(mx.bfloat16)
    q = mx.random.normal((B, H, T, D)).astype(mx.bfloat16)
    out = ragged_decode_attention(q, k, v, mx.array(lengths), D**-0.5, max(lengths))
    ref = _ref(q, k, v, lengths, T)
    err = mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item()
    assert err < 4e-3


def test_row_result_independent_of_batch():
    mx.random.seed(3)
    lengths = [900, 4100, 37]
    cap = 4352
    k = mx.random.normal((3, HKV, cap, D)).astype(mx.bfloat16)
    v = mx.random.normal((3, HKV, cap, D)).astype(mx.bfloat16)
    q = mx.random.normal((3, H, 1, D)).astype(mx.bfloat16)
    full = ragged_decode_attention(q, k, v, mx.array(lengths), D**-0.5, 4100)
    alone = ragged_decode_attention(
        q[:1], k[:1], v[:1], mx.array(lengths[:1]), D**-0.5, 900
    )
    assert mx.array_equal(full[:1], alone)


def _stock_cache(lengths, T=1):
    """A left-padded mlx-lm BatchKVCache holding random keys for rows of ``lengths``."""
    L = max(lengths)
    pads = [L - n for n in lengths]
    c = BatchKVCache(pads)
    k = mx.random.normal((len(lengths), HKV, L, 64)).astype(mx.bfloat16)
    v = mx.random.normal((len(lengths), HKV, L, 64)).astype(mx.bfloat16)
    c.update_and_fetch(k, v)
    return c, k, v, pads


def test_from_cache_right_aligns_rows():
    mx.random.seed(4)
    lengths = [10, 3, 7]
    stock, k, v, pads = _stock_cache(lengths)
    rc = ragged_kv.RaggedKVCache.from_cache(stock)
    assert rc.lengths == lengths and rc.offset.tolist() == lengths and rc._idx == 10
    assert rc.rows == 4  # headroom: next power of two
    for b in range(len(lengths)):
        rk, rv = rc.row(b)
        assert mx.array_equal(rk, k[b, :, pads[b] :])
        assert mx.array_equal(rv, v[b, :, pads[b] :])


def test_update_filter_extend_trim():
    mx.random.seed(5)
    stock, *_ = _stock_cache([10, 3])
    rc = ragged_kv.RaggedKVCache.from_cache(stock)
    new_k = mx.random.normal((2, HKV, 1, 64)).astype(mx.bfloat16)
    keys, _ = rc.update_and_fetch(new_k, new_k)
    assert rc.lengths == [11, 4]
    assert mx.array_equal(rc.row(0)[0][:, 10], new_k[0, :, 0]) and mx.array_equal(
        rc.row(1)[0][:, 3], new_k[1, :, 0]
    )
    assert keys is rc.keys  # the whole buffer, never a slice
    other, ok, _, opads = _stock_cache([5])
    before = rc.row(0)[0]
    rc.extend(other)
    assert rc.lengths == [11, 4, 5] and mx.array_equal(
        rc.row(2)[0], ok[0, :, opads[0] :]
    )
    assert mx.array_equal(rc.row(0)[0], before)
    rc.filter(mx.array([1, 2]))
    assert rc.lengths == [4, 5] and rc.offset.tolist() == [4, 5]
    assert mx.array_equal(rc.row(1)[0], ok[0, :, opads[0] :])
    # a joining row takes a free slot; live rows stay where they are
    slots = list(rc.slots)
    rc.extend(_stock_cache([2])[0])
    assert rc.slots[:2] == slots and rc.slots[2] not in slots
    assert rc.trim(2) == 2 and rc.lengths == [2, 3, 0]


def test_update_writes_every_row_at_its_own_length():
    """One scatter writes T tokens per row at each row's own position, for
    rows in arbitrary slots (after filter/extend), and leaves other keys."""
    mx.random.seed(21)
    stock, *_ = _stock_cache([9, 2, 5, 7])
    rc = ragged_kv.RaggedKVCache.from_cache(stock)
    rc.filter(mx.array([3, 1]))
    rc.extend(_stock_cache([4])[0])
    before = [rc.row(b)[0] for b in range(3)]
    T = 3
    x = mx.random.normal((3, HKV, T, 64)).astype(mx.bfloat16)
    rc.update_and_fetch(x, -x)
    for b, n in enumerate([7, 2, 4]):
        k, v = rc.row(b)
        assert rc.lengths[b] == n + T
        assert mx.array_equal(k[:, :n], before[b])
        assert mx.array_equal(k[:, n:], x[b]) and mx.array_equal(v[:, n:], -x[b])


def test_join_copies_only_the_new_row_when_room():
    """With a free slot and key room, extend writes into the existing buffer
    (same shape, no reallocation of the live rows)."""
    mx.random.seed(22)
    rc = ragged_kv.RaggedKVCache.from_cache(_stock_cache([30, 3, 7])[0])
    shape = rc.keys.shape
    rc.extend(_stock_cache([12])[0])
    assert rc.keys.shape == shape and rc.lengths == [30, 3, 7, 12]
    rc.extend(_stock_cache([5])[0])  # a fifth row: rows grow to 8
    assert rc.rows == 8 and rc.lengths[-1] == 5


def test_filter_releases_capacity_when_longest_row_leaves():
    mx.random.seed(6)
    stock, *_ = _stock_cache([3000, 10])
    rc = ragged_kv.RaggedKVCache.from_cache(stock)
    assert rc.capacity >= 3000
    rc.filter(mx.array([1]))
    assert rc.capacity <= 512 and rc.lengths == [10] and rc.slots == [0]


def test_qwen3_5_attention_matches_stock_left_padded_path():
    """The patched attention with a ragged cache equals the stock module over a
    left-padded BatchKVCache, decode step by step, rows of different lengths."""
    from mlx_vlm.models.qwen3_5.config import TextConfig
    from mlx_vlm.models.qwen3_5.language import Qwen3_5Attention

    cfg = TextConfig(
        model_type="qwen3_5_text",
        hidden_size=256,
        intermediate_size=512,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=6,
        rms_norm_eps=1e-6,
        vocab_size=100,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        head_dim=64,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
        },
    )
    mx.random.seed(7)
    attn = Qwen3_5Attention(cfg)
    attn.set_dtype(mx.bfloat16)
    lengths = [9, 3]
    L = max(lengths)
    pads = [L - n for n in lengths]
    prompt = mx.random.normal((2, L, 256)).astype(mx.bfloat16)
    stock = BatchKVCache(pads)
    ragged_kv.install()
    attn(
        prompt, cache=stock, mask=None
    )  # prefill (stock path); padded rows' garbage is masked later
    rc = ragged_kv.RaggedKVCache.from_cache(stock)
    for step in range(4):
        x = mx.random.normal((2, 1, 256)).astype(mx.bfloat16)
        ref_rows = []
        for b in range(2):
            # stock reference: row alone over its own keys (no padding)
            row = BatchKVCache([0])
            row.update_and_fetch(
                stock.keys[b : b + 1, :, pads[b] : stock._idx],
                stock.values[b : b + 1, :, pads[b] : stock._idx],
            )
            ref_rows.append(attn(x[b : b + 1], cache=row, mask=None))
        stock.update_and_fetch(
            *(mx.zeros((2, 2, 1, 64), mx.bfloat16),) * 2
        )  # keep indices aligned only
        out = attn(x, cache=rc)
        ref = mx.concatenate(ref_rows, axis=0)
        err = mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item()
        assert err < 2e-2, (step, err)
        # feed the ragged cache's new keys to the stock copy for the next step
        for b in range(2):
            n = rc.lengths[b]
            rk, rv = rc.row(b)
            stock.keys[b, :, stock._idx - 1] = rk[:, n - 1]
            stock.values[b, :, stock._idx - 1] = rv[:, n - 1]


def test_enable_makes_generation_batch_merge_ragged():
    """mlx-vlm's BatchGenerator merges a joining row's caches with upstream
    ``_extend_cache``; with ragged KV enabled that builds a RaggedKVCache from
    the decoding row's KVCache (the first join) or extends the ragged one, and
    leaves other caches (GatedDeltaNet ArraysCache) on the upstream path."""
    from mlx_vlm.generate import ar
    from mlx_vlm.models.cache import ArraysCache, KVCache
    from mlx_vlm.models.cache import BatchKVCache as VlmBatchKVCache

    from yunshu_engine.kernels.ragged_kv import RaggedKVCache, enable

    def lone(n, seed):
        mx.random.seed(seed)
        c = KVCache()
        k = mx.random.normal((1, HKV, n, 64)).astype(mx.bfloat16)
        c.update_and_fetch(k, -k)
        return c, k

    def joining(n, seed):
        mx.random.seed(seed)
        c = VlmBatchKVCache([0])
        k = mx.random.normal((1, HKV, n, 64)).astype(mx.bfloat16)
        c.update_and_fetch(k, -k)
        return c, k

    def gdn():
        a = ArraysCache(size=2)
        a.cache = [mx.zeros((1, 3, 8)), mx.zeros((1, 2, 4, 4))]
        return a

    try:
        enable("bf16")
        (ca, ka), (cb, kb) = lone(9, 1), joining(4, 2)
        out = ar._extend_cache([ca, gdn()], [cb, gdn()])
        rc = out[0]
        assert isinstance(rc, RaggedKVCache) and rc.lengths == [9, 4]
        assert mx.array_equal(rc.row(0)[0], ka[0]) and mx.array_equal(
            rc.row(1)[0], kb[0]
        )
        assert mx.array_equal(rc.row(1)[1], -kb[0])
        assert isinstance(out[1], ArraysCache) and out[1].cache[0].shape[0] == 2
        cc, kc = joining(6, 3)
        out2 = ar._extend_cache([rc], [cc])
        assert out2[0] is rc and rc.lengths == [9, 4, 6]
        assert mx.array_equal(rc.row(2)[0], kc[0])
        enable(None)
        out3 = ar._extend_cache([lone(5, 4)[0]], [joining(3, 5)[0]])
        assert type(out3[0]) is VlmBatchKVCache
    finally:
        enable(None)


# --- int8 KV (YUNSHU_KV_PRECISION=int8) -------------------------------------


def _q8(x):
    from yunshu_engine.kernels.ragged_attention import quantize_kv_reference

    return quantize_kv_reference(x)


def _dequant(codes, scales):
    *lead, d = codes.shape
    g = codes.astype(mx.float32).reshape(*lead, scales.shape[-1], -1)
    return (g * scales.astype(mx.float32)[..., None]).reshape(*lead, d)


def _cos(a, b):
    a = a.astype(mx.float32).reshape(-1)
    b = b.astype(mx.float32).reshape(-1)
    return (mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b))).item()


@pytest.mark.parametrize("outliers", [False, True])
@pytest.mark.parametrize("T", [1, 4])
@pytest.mark.parametrize(
    "lengths", [[37], [1030, 37], [5000, 1030, 37, 600], [17000, 37]]
)
def test_int8_kernel_close_to_bf16_reference(lengths, T, outliers):
    """int8 codes + group scales vs stock SDPA on the original bf16 K/V, per
    row: attention output (softmax-weighted V) and the pre-softmax logits."""
    mx.random.seed(len(lengths) * 10 + T + 100 * outliers)
    cap = -(-max(lengths) // 256) * 256 + 256
    B = len(lengths)
    k = mx.random.normal((B, HKV, cap, D))
    v = mx.random.normal((B, HKV, cap, D))
    if outliers:  # a few 20x channels, as real K caches have
        boost = mx.where(mx.arange(D) % 37 == 0, 20.0, 1.0)
        k, v = k * boost, v * boost
    k, v = k.astype(mx.bfloat16), v.astype(mx.bfloat16)
    q = mx.random.normal((B, H, T, D)).astype(mx.bfloat16)
    kq, ks = _q8(k)
    vq, vs = _q8(v)
    out = ragged_decode_attention(
        q, kq, vq, mx.array(lengths), D**-0.5, max(lengths), ks, vs
    )
    ref = _ref(q, k, v, lengths, T)
    err = mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item()
    for b in range(B):
        assert _cos(out[b], ref[b]) > (0.997 if outliers else 0.9999)
    if not outliers:
        assert err < 0.03
    # logits: q . k over the dequantized keys vs the bf16 keys
    kd = _dequant(kq, ks)[:, :, : max(lengths)]
    qg = q.astype(mx.float32).reshape(B, HKV, H // HKV * T, D)
    lg = qg @ k[:, :, : max(lengths)].astype(mx.float32).swapaxes(-1, -2) * D**-0.5
    lq = qg @ kd.swapaxes(-1, -2) * D**-0.5
    rel = (mx.max(mx.abs(lg - lq)) / mx.max(mx.abs(lg))).item()
    assert rel < (0.03 if outliers else 0.01), rel


def test_int8_row_result_independent_of_batch():
    mx.random.seed(13)
    lengths = [900, 4100, 37]
    cap = 4352
    kq, ks = _q8(mx.random.normal((3, HKV, cap, D)).astype(mx.bfloat16))
    vq, vs = _q8(mx.random.normal((3, HKV, cap, D)).astype(mx.bfloat16))
    q = mx.random.normal((3, H, 1, D)).astype(mx.bfloat16)
    full = ragged_decode_attention(q, kq, vq, mx.array(lengths), D**-0.5, 4100, ks, vs)
    for b, n in enumerate(lengths):
        alone = ragged_decode_attention(
            q[b : b + 1],
            kq[b : b + 1],
            vq[b : b + 1],
            mx.array([n]),
            D**-0.5,
            n,
            ks[b : b + 1],
            vs[b : b + 1],
        )
        assert mx.array_equal(full[b : b + 1], alone)


@pytest.mark.parametrize("shape", [(3, HKV, 5, 64), (2, HKV, 1, D), (1, 2, 300, D)])
def test_quantize_kernel_matches_reference(shape):
    """The fused K/V quantize launch writes the reference's codes and scales,
    including all-zero groups (scale 0) and non-contiguous inputs."""
    from yunshu_engine.kernels.ragged_attention import quantize_kv_pair

    mx.random.seed(sum(shape))
    k = (mx.random.normal(shape) * 3).astype(mx.bfloat16)
    k = mx.where(mx.arange(shape[-1]) < 32, 0.0, k).astype(mx.bfloat16)
    v = mx.random.normal((shape[0], shape[2], shape[1], shape[3])).astype(mx.bfloat16)
    v = v.transpose(0, 2, 1, 3)
    kq, ks, vq, vs = quantize_kv_pair(k, v)
    for got, ref in zip((kq, ks, vq, vs), (*_q8(k), *_q8(v)), strict=True):
        assert got.dtype == ref.dtype and mx.array_equal(got, ref)


def test_int8_needs_scales():
    kq = mx.zeros((1, HKV, 64, D), mx.int8)
    q = mx.zeros((1, H, 1, D), mx.bfloat16)
    with pytest.raises(ValueError):
        ragged_decode_attention(q, kq, kq, mx.array([8]), D**-0.5)


def test_int8_cache_quantizes_on_write():
    mx.random.seed(14)
    lengths = [10, 3, 7]
    stock, k, v, pads = _stock_cache(lengths)
    rc = ragged_kv.RaggedKVCache.from_cache(stock, "int8")
    assert rc.quantized and rc.keys.dtype == mx.int8
    assert rc.k_scales.shape[:3] == rc.keys.shape[:3] and rc.k_scales.shape[3] == 2
    for b, n in enumerate(lengths):
        s = rc.slots[b]
        kd = _dequant(rc.keys[s, :, :n], rc.k_scales[s, :, :n])
        ref = k[b, :, pads[b] :].astype(mx.float32)
        assert mx.max(mx.abs(kd - ref)).item() <= mx.max(mx.abs(ref)).item() / 127
    new_k = mx.random.normal((3, HKV, 1, 64)).astype(mx.bfloat16)
    rc.update_and_fetch(new_k, new_k)
    assert rc.lengths == [11, 4, 8]
    wq, ws = _q8(new_k)
    for b, n in enumerate(rc.lengths):
        s = rc.slots[b]
        assert mx.array_equal(rc.keys[s, :, n - 1], wq[b, :, 0])
        assert mx.array_equal(rc.v_scales[s, :, n - 1], ws[b, :, 0])
    other, *_ = _stock_cache([5])
    rc.extend(other)
    assert rc.lengths == [11, 4, 8, 5] and rc.k_scales.shape[0] == 4
    with pytest.raises(ValueError):
        rc.extend(ragged_kv.RaggedKVCache.from_cache(_stock_cache([2])[0]))
    rc.filter(mx.array([1, 3]))
    assert rc.lengths == [4, 5] and rc.slots == [1, 3]
    assert rc.trim(2) == 2 and rc.lengths == [2, 3]


def test_int8_cache_grows_past_capacity():
    mx.random.seed(15)
    stock, *_ = _stock_cache([5, 2])
    rc = ragged_kv.RaggedKVCache.from_cache(stock, "int8")
    cap = rc.capacity
    for _ in range(cap):
        x = mx.random.normal((2, HKV, 1, 64)).astype(mx.bfloat16)
        rc.update_and_fetch(x, x)
    assert rc.capacity > cap and rc.k_scales.shape[2] == rc.capacity
    assert rc.lengths == [5 + cap, 2 + cap]


def test_int8_from_cache_quantizes_rows():
    from mlx_vlm.models.cache import BatchKVCache as VlmBatchKVCache

    c = VlmBatchKVCache([2, 0])
    k = mx.random.normal((2, 4, 5, 64)).astype(mx.bfloat16)
    c.update_and_fetch(k, k)
    rc = ragged_kv.RaggedKVCache.from_cache(c, "int8")
    assert rc.quantized and rc.keys.dtype == mx.int8 and rc.lengths == [3, 5]


# --- kernel variants: key-parallel (default, D=256) vs per-head ---------------


def _rand_kv(lengths, T, seed):
    mx.random.seed(seed)
    cap = -(-max(lengths) // 256) * 256 + 256
    B = len(lengths)
    k = mx.random.normal((B, HKV, cap, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, HKV, cap, D)).astype(mx.bfloat16)
    q = mx.random.normal((B, H, T, D)).astype(mx.bfloat16)
    return q, k, v


def _attn(q, k, v, lengths, q8, **kw):
    if q8:
        kq, ks = _q8(k)
        vq, vs = _q8(v)
        kw.setdefault("max_length", None)
        return ragged_decode_attention(
            q, kq, vq, mx.array(lengths), D**-0.5, k_scales=ks, v_scales=vs, **kw
        )
    return ragged_decode_attention(q, k, v, mx.array(lengths), D**-0.5, **kw)


@pytest.mark.parametrize("q8", [False, True])
@pytest.mark.parametrize("T", [1, 4, 8])
def test_variants_match_reference_and_each_other(T, q8):
    """Chunk-boundary lengths (512, 513, 1024) and a short row; the key-parallel
    kernel (dense grid and work-list launch) and the per-head kernel agree
    with stock SDPA and with each other."""
    lengths = [1030, 37, 512, 513, 1024]
    q, k, v = _rand_kv(lengths, T, 40 + T)
    ref = _ref(q, k, v, lengths, T).astype(mx.float32)
    outs = {
        "per_head": _attn(q, k, v, lengths, q8, impl="per_head"),
        "key_parallel": _attn(q, k, v, lengths, q8, impl="key_parallel"),
        "key_parallel_wl": _attn(
            q, k, v, lengths, q8, impl="key_parallel", row_lengths=lengths
        ),
    }
    if not q8 and tile_ready():  # tensor-op tile kernel: bf16 only
        outs["tile"] = _attn(q, k, v, lengths, q8, impl="tile")
        outs["tile_wl"] = _attn(q, k, v, lengths, q8, impl="tile", row_lengths=lengths)
    for name, out in outs.items():
        err = mx.max(mx.abs(out.astype(mx.float32) - ref)).item()
        assert err < (0.04 if q8 else 1e-2), (name, err)
    # the work-list launches compute the same chunks: same bits
    assert mx.array_equal(outs["key_parallel"], outs["key_parallel_wl"])
    if "tile" in outs:
        assert mx.array_equal(outs["tile"], outs["tile_wl"])


@pytest.mark.parametrize("impl", ["key_parallel", "tile", "key_parallel_q8"])
@pytest.mark.parametrize("T", [2, 6, 8])
def test_verify_tokens_equal_one_token_decode(T, impl):
    """Verify token t's bits (T tokens in one call) equal a one-token call at
    that token's position — the property that makes speculative verify equal
    plain decode — for the key-parallel (bf16 / int8) and tile kernels, in a
    batch with slots. Lengths straddle chunk (512) and step (32 / 64)
    boundaries."""
    q8 = impl.endswith("_q8")
    kind = impl.removesuffix("_q8")
    if kind == "tile" and not tile_ready():
        pytest.skip("tensor-op tile kernel unavailable on this GPU")
    lengths = [1030, 37, 513, 520]
    q, k, v = _rand_kv(lengths, T, 70 + T)
    full = _attn(q, k, v, lengths, q8, impl=kind, row_lengths=lengths)
    for b, n in enumerate(lengths):
        for t in range(T):
            m = n - (T - 1 - t)
            one = _attn(
                q[b : b + 1, :, t : t + 1],
                k[b : b + 1],
                v[b : b + 1],
                [m],
                q8,
                impl=kind,
                row_lengths=[m],
            )
            assert mx.array_equal(full[b : b + 1, :, t : t + 1], one), (b, t)


@pytest.mark.parametrize("q8", [False, True])
@pytest.mark.parametrize("T", [1, 3, 8])
def test_key_parallel_row_bits_independent_of_batch(T, q8):
    """A row's output bits do not depend on its batch neighbours, the grid
    size (``max_length``) or the launch mode — the property speculative verify
    (T <= 8) vs plain decode relies on."""
    lengths = [2049, 700, 37, 512]
    q, k, v = _rand_kv(lengths, T, 60 + T)
    full = _attn(q, k, v, lengths, q8, row_lengths=lengths)
    dense = _attn(q, k, v, lengths, q8, max_length=8192)
    assert mx.array_equal(full, dense)
    for b, n in enumerate(lengths):
        sl = slice(b, b + 1)
        alone = _attn(q[sl], k[sl], v[sl], [n], q8, row_lengths=[n])
        assert mx.array_equal(full[sl], alone), b


def test_other_head_dim_falls_back_to_per_head():
    mx.random.seed(7)
    d = 128
    k = mx.random.normal((2, HKV, 512, d)).astype(mx.bfloat16)
    v = mx.random.normal((2, HKV, 512, d)).astype(mx.bfloat16)
    q = mx.random.normal((2, H, 1, d)).astype(mx.bfloat16)
    lengths = [300, 40]
    out = ragged_decode_attention(
        q, k, v, mx.array(lengths), d**-0.5, row_lengths=lengths
    )
    for b, n in enumerate(lengths):
        ref = mx.fast.scaled_dot_product_attention(
            q[b : b + 1], k[b : b + 1, :, :n], v[b : b + 1, :, :n], scale=d**-0.5
        )
        err = mx.max(mx.abs(out[b : b + 1].astype(mx.float32) - ref.astype(mx.float32)))
        assert err.item() < 4e-3
    with pytest.raises(ValueError):
        ragged_decode_attention(
            q, k, v, mx.array(lengths), d**-0.5, impl="key_parallel"
        )


@pytest.mark.parametrize("q8", [False, True])
def test_slots_address_buffer_rows(q8):
    """Query row b reads buffer row ``slots[b]``: a permuted, over-allocated
    buffer gives the same bits as the dense one."""
    lengths = [700, 37, 1030]
    q, k, v = _rand_kv(lengths, 2, 80)
    perm = [2, 0, 3]  # buffer row of each query row; row 1 is a free slot
    z = mx.zeros_like(k[:1])
    kb = mx.concatenate([k[1:2], z, k[0:1], k[2:3]])
    vb = mx.concatenate([v[1:2], z, v[0:1], v[2:3]])
    slots = mx.array(perm, dtype=mx.int32)
    for kw in ({"row_lengths": lengths}, {"max_length": 2048}, {"impl": "per_head"}):
        out = _attn(q, kb, vb, lengths, q8, slots=slots, **kw)
        assert mx.array_equal(out, _attn(q, k, v, lengths, q8, **kw)), kw


@pytest.mark.parametrize("batch_cache", [False, True])
def test_dense_lane_matches_ragged_cache_bits(batch_cache):
    """The speculative lane's one-row cache (KVCache, or the verify path's
    one-row BatchKVCache) through the lane kernel gives the same bits as the
    same row in a batch call of that kernel (decode T=1 and verify T=4), and
    nothing changes while the lane is off."""
    from mlx_vlm.models.cache import BatchKVCache as VlmBatchKVCache
    from mlx_vlm.models.cache import KVCache

    from yunshu_engine.kernels.ragged_kv import dense_lane_attention, set_dense_lane

    mx.random.seed(90)
    n = 1300
    k = mx.random.normal((1, HKV, n, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, HKV, n, D)).astype(mx.bfloat16)
    for T in (1, 4):
        c = VlmBatchKVCache([0]) if batch_cache else KVCache()
        c.update_and_fetch(k, v)
        q = mx.random.normal((1, H, T, D)).astype(mx.bfloat16)
        nk = mx.random.normal((1, HKV, T, D)).astype(mx.bfloat16)
        c.update_and_fetch(nk, nk)
        assert dense_lane_attention(q, c, D**-0.5) is None  # lane off
        set_dense_lane(True)
        try:
            out = dense_lane_attention(q, c, D**-0.5)
        finally:
            set_dense_lane(False)
        kk = mx.concatenate([k, nk], axis=2)
        vv = mx.concatenate([v, nk], axis=2)
        other = mx.random.normal((1, HKV, n + T, D)).astype(mx.bfloat16)
        batch = ragged_decode_attention(
            mx.concatenate([q, q]),
            mx.concatenate([other, kk]),
            mx.concatenate([other, vv]),
            mx.array([40, n + T]),
            D**-0.5,
            row_lengths=[40, n + T],
            impl="tile" if tile_ready() else "auto",
        )
        assert mx.array_equal(out, batch[1:])


@pytest.mark.parametrize("prev", [1000, 1017, 1020, 1023])
@pytest.mark.parametrize("T", [1, 3, 7])
def test_dense_lane_verify_equals_decode_after_rollback_growth(prev, T):
    """After a rollback, a stock cache grows to ``offset + 256`` (a capacity
    that is not a multiple of 64). Each verify token through the lane kernel
    must still equal a one-token decode at its position over an aligned
    buffer — the tile kernel's key windows may not depend on the capacity."""
    from mlx_vlm.models.cache import KVCache

    from yunshu_engine.kernels.ragged_kv import dense_lane_attention, set_dense_lane

    if not tile_ready():
        pytest.skip("tensor-op tile kernel unavailable on this GPU")
    mx.random.seed(prev + T)
    total = prev + T
    k = mx.random.normal((1, HKV, 1280, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, HKV, 1280, D)).astype(mx.bfloat16)
    q = mx.random.normal((1, H, T, D)).astype(mx.bfloat16)
    c = KVCache()
    c.update_and_fetch(k[:, :, :1024], v[:, :, :1024])  # capacity 1024
    c.trim(1024 - prev)  # rollback to ``prev``
    c.update_and_fetch(k[:, :, prev:total], v[:, :, prev:total])
    assert c.offset == total
    set_dense_lane(True)
    try:
        out = dense_lane_attention(q, c, D**-0.5)
    finally:
        set_dense_lane(False)
    assert c.keys.shape[2] % 64 == 0
    for t in range(T):
        m = total - (T - 1 - t)
        one = ragged_decode_attention(
            q[:, :, t : t + 1],
            k,
            v,
            mx.array([m]),
            D**-0.5,
            row_lengths=[m],
            impl="tile",
        )
        assert mx.array_equal(out[:, :, t : t + 1], one), t


def test_tile_bits_independent_of_capacity():
    """Same keys in buffers of different capacities (multiples of 64): same
    bits (the lane pads other capacities to one)."""
    if not tile_ready():
        pytest.skip("tensor-op tile kernel unavailable on this GPU")
    mx.random.seed(5)
    k = mx.random.normal((1, HKV, 1600, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, HKV, 1600, D)).astype(mx.bfloat16)
    for n in (1000, 1023, 1025, 1087):
        q = mx.random.normal((1, H, 1, D)).astype(mx.bfloat16)
        outs = [
            ragged_decode_attention(
                q,
                mx.contiguous(k[:, :, :cap]),
                mx.contiguous(v[:, :, :cap]),
                mx.array([n]),
                D**-0.5,
                row_lengths=[n],
                impl="tile",
            )
            for cap in (-(-n // 64) * 64, 1536, 1600)
        ]
        assert all(mx.array_equal(outs[0], o) for o in outs[1:]), n
