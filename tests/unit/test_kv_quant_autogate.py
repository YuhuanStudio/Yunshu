"""/747: KV-quant auto-gate. W747 corrected the gate from token-count
to estimated KV BYTES after a real-model benchmark showed token-count was
net-negative for small models (small KV doesn't dominate bandwidth)."""
from __future__ import annotations

import types

from yunshu_engine.batched_engine import BatchedEngine


def _eng(explicit_bits=None, model=None):
    e = BatchedEngine.__new__(BatchedEngine)
    e._kv_quant_bits = explicit_bits
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


def test_explicit_bits_always_win(monkeypatch):
    monkeypatch.setenv("YUNSHU_KV_QUANT_AUTO", "1")
    e = _eng(explicit_bits=4, model=_fake_model(64, 8, 128))
    assert e._effective_kv_quant_bits(100000) == 4


def test_opt_out(monkeypatch):
    monkeypatch.setenv("YUNSHU_KV_QUANT_AUTO", "0")
    e = _eng(model=_fake_model(64, 8, 128))
    assert e._effective_kv_quant_bits(10**6) is None


def test_small_model_long_ctx_NOT_quantized(monkeypatch):
    """W747: a 0.5B-like model (24L, 2 kv-heads, 64 head_dim) at 9k tokens has
    ~108MB KV << 2GB → must NOT quantize (benchmark showed net-negative)."""
    monkeypatch.delenv("YUNSHU_KV_QUANT_AUTO", raising=False)
    monkeypatch.delenv("YUNSHU_KV_QUANT_AUTO_MIN_BYTES", raising=False)
    e = _eng(model=_fake_model(24, 2, 64))
    assert e._effective_kv_quant_bits(9000) is None


def test_large_model_long_ctx_quantized(monkeypatch):
    """A 32B-like model (64L, 8 kv-heads, 128 head_dim) = 256KB/token; at 16k
    tokens that's ~4GB KV >= 2GB → quantize."""
    monkeypatch.delenv("YUNSHU_KV_QUANT_AUTO", raising=False)
    monkeypatch.delenv("YUNSHU_KV_QUANT_AUTO_MIN_BYTES", raising=False)
    e = _eng(model=_fake_model(64, 8, 128))
    assert e._effective_kv_quant_bits(16384) == 8
    assert e._effective_kv_quant_bits(100) is None


def test_custom_min_bytes(monkeypatch):
    monkeypatch.delenv("YUNSHU_KV_QUANT_AUTO", raising=False)
    monkeypatch.setenv("YUNSHU_KV_QUANT_AUTO_MIN_BYTES", "100000000")  # 100MB
    e = _eng(model=_fake_model(24, 2, 64))  # ~12KB/token → 100MB ~8138 tokens
    assert e._effective_kv_quant_bits(9000) == 8
    assert e._effective_kv_quant_bits(1000) is None


def test_fallback_token_threshold_when_dims_unreadable(monkeypatch):
    """No model → fall back to the conservative token threshold (default 16384)."""
    monkeypatch.delenv("YUNSHU_KV_QUANT_AUTO", raising=False)
    monkeypatch.delenv("YUNSHU_KV_QUANT_AUTO_THRESHOLD", raising=False)
    e = _eng(model=None)
    assert e._effective_kv_quant_bits(16384) == 8
    assert e._effective_kv_quant_bits(8192) is None
