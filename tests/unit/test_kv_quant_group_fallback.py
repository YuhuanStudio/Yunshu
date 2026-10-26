"""KV quantisation on a model whose head_dim does not divide the group must fall back to the
unquantised cache (one warning), never fail the request into an empty 200."""

from __future__ import annotations

import logging

import mlx.core as mx
import pytest

from yunshu_engine.batched_engine import BatchedEngine


def _tiny(head_dim=32):
    from mlx_lm.models import llama

    mx.random.seed(0)
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=head_dim * 2,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=64,
        head_dim=head_dim,
    )
    m = llama.Model(args)
    mx.eval(m.parameters())
    return m


def _engine(model, bits=4, group=64):
    e = BatchedEngine.__new__(BatchedEngine)
    e._kv_quant_bits = bits
    e._kv_quant_auto = False
    e._kv_quant_start = 0
    e._kv_quant_group_size = group
    e._model = model
    return e


def _tokens(model, kv_bits, group):
    from mlx_lm.generate import generate_step

    out = []
    for tok, _ in generate_step(
        mx.array([1, 2, 3, 4, 5]),
        model,
        max_tokens=8,
        kv_bits=kv_bits,
        kv_group_size=group,
        quantized_kv_start=0,
    ):
        out.append(int(tok))
    return out


def test_misaligned_head_dim_falls_back_to_unquantised_with_same_tokens(caplog):
    mx.set_default_device(mx.cpu)
    model = _tiny(32)
    eng = _engine(model)
    with caplog.at_level(logging.WARNING):
        bits = eng._effective_kv_quant_bits(100)
        assert eng._effective_kv_quant_bits(100) is None
    assert bits is None
    assert sum("head_dim" in r.getMessage() for r in caplog.records) == 1
    # the raw mlx-lm call with the configured bits really fails ...
    with pytest.raises(ValueError):
        _tokens(model, 4, 64)
    # ... and the fallback bits give exactly the quant-off tokens
    assert _tokens(model, bits, 64) == _tokens(model, None, 64)


def test_aligned_head_dim_keeps_quantisation():
    mx.set_default_device(mx.cpu)
    eng = _engine(_tiny(64))
    assert eng._effective_kv_quant_bits(100) == 4


def test_maybe_quantize_skips_misaligned_layers():
    from mlx_lm.models.cache import KVCache

    mx.set_default_device(mx.cpu)
    from yunshu_engine.engine_cache import _maybe_quantize_kv_cache

    c = KVCache()
    c.update_and_fetch(mx.zeros((1, 2, 4, 32)), mx.zeros((1, 2, 4, 32)))
    cache = [c]
    _maybe_quantize_kv_cache(cache, 0, 64, 4)
    assert cache[0] is c
