"""/747: KV-quant auto-gate. corrected the gate from token-count
to estimated KV BYTES after a real-model benchmark showed token-count was
net-negative for small models (small KV doesn't dominate bandwidth)."""

from __future__ import annotations

import types

from yunshu_engine.batched_engine import BatchedEngine


def _eng(explicit_bits=None, model=None, auto=True):
    e = BatchedEngine.__new__(BatchedEngine)
    e._kv_quant_bits = explicit_bits
    e._kv_quant_auto = auto
    e._kv_quant_start = 0
    e._model = model
    return e


def _fake_model(n_layers, n_kv, head_dim, n_heads=None):
    cfg = types.SimpleNamespace(
        num_hidden_layers=n_layers,
        num_key_value_heads=n_kv,
        num_attention_heads=n_heads or n_kv,
        hidden_size=(n_heads or n_kv) * head_dim,
        head_dim=head_dim,
    )
    return types.SimpleNamespace(config=cfg)


def test_explicit_bits_always_win():
    e = _eng(explicit_bits=4, model=_fake_model(64, 8, 128))
    assert e._effective_kv_quant_bits(100000) == 4


def test_opt_out():
    """YUNSHU_KV_QUANT_BITS=off disables the auto gate."""
    e = _eng(model=_fake_model(64, 8, 128), auto=False)
    assert e._effective_kv_quant_bits(10**6) is None


def test_small_model_long_ctx_NOT_quantized():
    """a 0.5B-like model (24L, 2 kv-heads, 64 head_dim) at 9k tokens has
    ~108MB KV << 2GB → must NOT quantize (benchmark showed net-negative)."""
    e = _eng(model=_fake_model(24, 2, 64))
    assert e._effective_kv_quant_bits(9000) is None


def test_large_model_long_ctx_quantized():
    """A 32B-like model (64L, 8 kv-heads, 128 head_dim) = 256KB/token; at 16k
    tokens that's ~4GB KV >= 2GB → quantize."""
    e = _eng(model=_fake_model(64, 8, 128))
    assert e._effective_kv_quant_bits(16384) == 8
    assert e._effective_kv_quant_bits(100) is None


def test_fallback_token_threshold_when_dims_unreadable():
    """No model → fall back to the conservative token threshold (default 16384)."""
    e = _eng(model=None)
    assert e._effective_kv_quant_bits(16384) == 8
    assert e._effective_kv_quant_bits(8192) is None


def test_auto_quantized_kv_is_only_lossy_when_the_cache_really_is_quantized():
    """The RAM prefix/prompt caches have no precision in their key: KV that really got
    quantized must not be stored (a hit would differ from a miss). The max_tokens-inflated
    estimate alone, a sliding-window cache (never quantizes) or a fixed precision are storable."""
    from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache

    auto = _eng(model=_fake_model(64, 8, 128))
    bits = auto._effective_kv_quant_bits(16384)
    assert bits == 8 and auto._auto_quantized_kv(
        bits
    )  # decided, nothing known about the cache
    assert not auto._auto_quantized_kv(auto._effective_kv_quant_bits(100))
    assert not auto._auto_quantized_kv(bits, [KVCache(), KVCache()])
    assert not auto._auto_quantized_kv(bits, [RotatingKVCache(max_size=8)])
    assert auto._auto_quantized_kv(bits, [KVCache(), QuantizedKVCache()])
    fixed = _eng(explicit_bits=4, model=_fake_model(64, 8, 128))
    assert not fixed._auto_quantized_kv(fixed._effective_kv_quant_bits(16384))
    assert not fixed._auto_quantized_kv(4, [QuantizedKVCache()])


def test_auto_quantization_starts_where_the_kv_estimate_reaches_the_gate():
    auto = _eng(model=_fake_model(64, 8, 128))  # 256 KiB/token -> 8192 tokens = 2 GiB
    assert auto._kv_quant_start_for(8) == 8192
    assert auto._kv_quant_start_for(None) == 0
    auto._kv_quant_start = 100
    assert auto._kv_quant_start_for(8) == 8192
    fixed = _eng(explicit_bits=4, model=_fake_model(64, 8, 128))
    assert fixed._kv_quant_start_for(4) == 0
    assert _eng(model=None)._kv_quant_start_for(8) == 16384


def _gated_engine():
    """Tiny real engine whose KV estimate (50 MiB/token -> gate at 41 tokens) crosses 2 GiB for
    prompt + max_tokens, like a 70B model asked for a long max_tokens."""
    from unittest.mock import MagicMock

    import mlx.core as mx
    from mlx_lm.models import llama

    from .tiny_text_engine import make_engine

    eng = make_engine()
    mx.random.seed(0)
    # head_dim 32: the smallest KV group size mlx can quantize
    eng._model = llama.Model(
        llama.ModelArgs(
            model_type="llama",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=2,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=64,
            max_position_embeddings=512,
        )
    )
    mx.eval(eng._model.parameters())
    eng._kv_quant_bits = None
    eng._kv_quant_auto = True
    eng._kv_bytes_per_token = lambda: 50 * 1024**2
    eng._kv_quant_group_size = 32
    eng._kv_prefix_cache = MagicMock()
    eng._kv_prefix_cache.get.return_value = (None, None, 0)
    eng._kv_prefix_cache._snapshot_cache.side_effect = lambda cache, trim=0: cache
    return eng


_KW = dict(prompt="the quick brown fox", temperature=0.0)


async def _short_stop(eng):
    """A stop string the greedy continuation hits within its first 3 tokens."""
    probe = await _gated_engine()._generate_fast(max_tokens=6, **_KW)
    return [probe.text[2]]


async def test_short_reply_under_a_long_max_tokens_is_still_stored_non_stream():
    eng = _gated_engine()
    assert eng._effective_kv_quant_bits(19 + 40) == 8  # the cap alone trips the gate
    out = await eng._generate_fast(max_tokens=40, stop=await _short_stop(eng), **_KW)
    assert out.error is None and out.completion_tokens < 10
    assert (
        eng._kv_prefix_cache.add.call_count >= 1
    )  # was 0: the estimate skipped the store


async def test_really_quantized_cache_is_not_stored_non_stream():
    eng = _gated_engine()
    out = await eng._generate_fast(max_tokens=50, **_KW)  # 19 + 50 > the 41-token gate
    assert out.error is None and out.completion_tokens >= 40
    assert eng._kv_prefix_cache.add.call_count == 0


async def test_short_reply_under_a_long_max_tokens_is_still_stored_stream():
    from .tiny_text_engine import collect_stream

    eng = _gated_engine()
    await collect_stream(eng, max_tokens=40, stop=await _short_stop(eng), **_KW)
    assert eng._kv_prefix_cache.add.call_count >= 1


async def test_really_quantized_cache_is_not_stored_stream():
    from .tiny_text_engine import collect_stream

    eng = _gated_engine()
    await collect_stream(eng, max_tokens=50, **_KW)
    assert eng._kv_prefix_cache.add.call_count == 0


def test_ngram_path_only_stores_fixed_precision_kv():
    """The n-gram path quantizes with the engine-wide ``_kv_quant_bits`` only (never
    the per-request length gate), so what it stores is the same for every request."""
    import inspect

    from yunshu_engine import engine_ngram

    src = inspect.getsource(engine_ngram)
    assert "_effective_kv_quant_bits" not in src
    assert "if self._kv_quant_bits is not None:" in src
