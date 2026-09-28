"""Per-row-length KV cache and ragged decode attention (YUNSHU_RAGGED_KV)."""

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CI without Apple GPU
    pytest.skip("needs an Apple GPU", allow_module_level=True)

from mlx_lm.models.cache import BatchKVCache  # noqa: E402

from yunshu_engine.kernels import ragged_kv  # noqa: E402
from yunshu_engine.kernels.ragged_attention import ragged_decode_attention  # noqa: E402

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


def test_from_batch_kv_right_aligns_rows():
    mx.random.seed(4)
    lengths = [10, 3, 7]
    stock, k, v, pads = _stock_cache(lengths)
    rc = ragged_kv.RaggedKVCache.from_batch_kv(stock)
    assert rc.lengths == lengths and rc.offset.tolist() == lengths and rc._idx == 10
    for b, n in enumerate(lengths):
        assert mx.array_equal(rc.keys[b, :, :n], k[b, :, pads[b] :])
        assert mx.array_equal(rc.values[b, :, :n], v[b, :, pads[b] :])


def test_update_filter_extend_trim():
    mx.random.seed(5)
    stock, *_ = _stock_cache([10, 3])
    rc = ragged_kv.RaggedKVCache.from_batch_kv(stock)
    new_k = mx.random.normal((2, HKV, 1, 64)).astype(mx.bfloat16)
    keys, _ = rc.update_and_fetch(new_k, new_k)
    assert rc.lengths == [11, 4]
    assert mx.array_equal(keys[0, :, 10], new_k[0, :, 0]) and mx.array_equal(
        keys[1, :, 3], new_k[1, :, 0]
    )
    other, ok, _, opads = _stock_cache([5])
    rc.extend(other)
    assert rc.lengths == [11, 4, 5] and mx.array_equal(
        rc.keys[2, :, :5], ok[0, :, opads[0] :]
    )
    rc.filter(mx.array([1, 2]))
    assert rc.lengths == [4, 5] and rc.offset.tolist() == [4, 5]
    assert rc.trim(2) == 2 and rc.lengths == [2, 3]


def test_filter_releases_capacity_when_longest_row_leaves():
    mx.random.seed(6)
    stock, *_ = _stock_cache([3000, 10])
    rc = ragged_kv.RaggedKVCache.from_batch_kv(stock)
    assert rc.capacity >= 3000
    rc.filter(mx.array([1]))
    assert rc.capacity <= 512


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
    rc = ragged_kv.RaggedKVCache.from_batch_kv(stock)
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
            stock.keys[b, :, stock._idx - 1] = rc.keys[b, :, n - 1]
            stock.values[b, :, stock._idx - 1] = rc.values[b, :, n - 1]


def test_convert_batch_accepts_mlx_vlm_batch_cache():
    """mlx-vlm's BatchGenerator uses its own BatchKVCache class; the runner
    path must convert it (the mlx-lm class alone never matched in serving)."""
    from mlx_vlm.models.cache import BatchKVCache as VlmBatchKVCache

    from yunshu_engine.kernels.ragged_kv import RaggedKVCache, convert_batch

    c = VlmBatchKVCache([2, 0])
    k = mx.random.normal((2, 4, 5, 8)).astype(mx.bfloat16)
    c.update_and_fetch(k, k)
    caches = [c]
    assert convert_batch(caches) == 1
    assert isinstance(caches[0], RaggedKVCache)
    assert caches[0].lengths == [3, 5]


# --- int8 KV (YUNSHU_RAGGED_KV=int8) ----------------------------------------


def _q8(x):
    from yunshu_engine.kernels.ragged_attention import quantize_kv

    return quantize_kv(x)


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


def test_int8_needs_scales():
    kq = mx.zeros((1, HKV, 64, D), mx.int8)
    q = mx.zeros((1, H, 1, D), mx.bfloat16)
    with pytest.raises(ValueError):
        ragged_decode_attention(q, kq, kq, mx.array([8]), D**-0.5)


def test_int8_cache_quantizes_on_write():
    mx.random.seed(14)
    lengths = [10, 3, 7]
    stock, k, v, pads = _stock_cache(lengths)
    rc = ragged_kv.RaggedKVCache.from_batch_kv(stock, "int8")
    assert rc.quantized and rc.keys.dtype == mx.int8
    assert rc.k_scales.shape[:3] == rc.keys.shape[:3] and rc.k_scales.shape[3] == 2
    for b, n in enumerate(lengths):
        kd = _dequant(rc.keys[b, :, :n], rc.k_scales[b, :, :n])
        ref = k[b, :, pads[b] :].astype(mx.float32)
        assert mx.max(mx.abs(kd - ref)).item() <= mx.max(mx.abs(ref)).item() / 127
    new_k = mx.random.normal((3, HKV, 1, 64)).astype(mx.bfloat16)
    rc.update_and_fetch(new_k, new_k)
    assert rc.lengths == [11, 4, 8]
    wq, ws = _q8(new_k)
    for b, n in enumerate(rc.lengths):
        assert mx.array_equal(rc.keys[b, :, n - 1], wq[b, :, 0])
        assert mx.array_equal(rc.v_scales[b, :, n - 1], ws[b, :, 0])
    other, *_ = _stock_cache([5])
    rc.extend(other)
    assert rc.lengths == [11, 4, 8, 5] and rc.k_scales.shape[0] == 4
    with pytest.raises(ValueError):
        rc.extend(ragged_kv.RaggedKVCache.from_batch_kv(_stock_cache([2])[0]))
    rc.filter(mx.array([1, 3]))
    assert rc.lengths == [4, 5] and rc.v_scales.shape[0] == 2
    assert rc.trim(2) == 2 and rc.lengths == [2, 3]


def test_int8_cache_grows_past_capacity():
    mx.random.seed(15)
    stock, *_ = _stock_cache([5, 2])
    rc = ragged_kv.RaggedKVCache.from_batch_kv(stock, "int8")
    cap = rc.capacity
    for _ in range(cap):
        x = mx.random.normal((2, HKV, 1, 64)).astype(mx.bfloat16)
        rc.update_and_fetch(x, x)
    assert rc.capacity > cap and rc.k_scales.shape[2] == rc.capacity
    assert rc.lengths == [5 + cap, 2 + cap]


def test_convert_batch_int8():
    from mlx_vlm.models.cache import BatchKVCache as VlmBatchKVCache

    from yunshu_engine.kernels.ragged_kv import RaggedKVCache, convert_batch

    c = VlmBatchKVCache([2, 0])
    k = mx.random.normal((2, 4, 5, 64)).astype(mx.bfloat16)
    c.update_and_fetch(k, k)
    caches = [c]
    assert convert_batch(caches, "int8") == 1
    assert isinstance(caches[0], RaggedKVCache) and caches[0].quantized
    assert caches[0].lengths == [3, 5]


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
        "work_list": _attn(q, k, v, lengths, q8, row_lengths=lengths),
    }
    for name, out in outs.items():
        err = mx.max(mx.abs(out.astype(mx.float32) - ref)).item()
        assert err < (0.04 if q8 else 4e-3), (name, err)
    # the work-list launch computes the same chunks: same bits
    assert mx.array_equal(outs["key_parallel"], outs["work_list"])


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
