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


def test_auto_quantized_kv_is_never_stored_in_token_keyed_caches():
    """The RAM prefix/prompt caches have no precision in their key: KV that the length gate
    quantized must not be stored (a hit would differ from a miss); a fixed precision may."""
    import inspect

    from yunshu_engine import engine_fast

    auto = _eng(model=_fake_model(64, 8, 128))
    assert auto._auto_quantized_kv(auto._effective_kv_quant_bits(16384))
    assert not auto._auto_quantized_kv(auto._effective_kv_quant_bits(100))
    fixed = _eng(explicit_bits=4, model=_fake_model(64, 8, 128))
    assert not fixed._auto_quantized_kv(fixed._effective_kv_quant_bits(16384))
    src = inspect.getsource(engine_fast)
    assert "_lossy_kv = self._auto_quantized_kv(_req_kv_bits)" in src
    assert src.count("and not _lossy_kv") >= 3


def test_streaming_path_never_stores_auto_quantized_kv():
    """Every store on the streaming fast path (post-generation add and the
    breakpoint prefixes) goes through ``store_cache``, which the length-gated 8-bit
    KV disables, like ``_lossy_kv`` on the non-stream path."""
    import inspect
    import re

    from yunshu_engine import engine_stream

    src = inspect.getsource(engine_stream)
    assert re.search(
        r"store_cache = \(\s*None\s*if _stream_hybrid or "
        r"self\._auto_quantized_kv\(_req_kv_bits\)\s*else prefix_cache\s*\)",
        src,
    )
    # no other KV prefix-cache store may bypass store_cache
    assert len(re.findall(r"\bprefix_cache\.add\(", src)) == 0


def test_ngram_path_only_stores_fixed_precision_kv():
    """The n-gram path quantizes with the engine-wide ``_kv_quant_bits`` only (never
    the per-request length gate), so what it stores is the same for every request."""
    import inspect

    from yunshu_engine import engine_ngram

    src = inspect.getsource(engine_ngram)
    assert "_effective_kv_quant_bits" not in src
    assert "if self._kv_quant_bits is not None:" in src
