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
