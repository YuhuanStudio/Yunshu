"""Spilling a quantized entry to SSD dequantizes each layer once, not once per block."""

import mlx.core as mx
import pytest

from yunshu_engine import kv_prefix_cache as kpc


class _SSD:
    def __init__(self):
        self.saved = []

    def has_block(self, key):
        return False

    def save_block(self, block_hash, cache_data, token_count, model_name):
        self.saved.append((token_count, cache_data[0].keys.shape[-2]))


def _quantized_entry(n_tokens, layers=2):
    from mlx_lm.models.cache import KVCache

    out = []
    for _ in range(layers):
        c = KVCache()
        c.update_and_fetch(
            mx.random.normal((1, 2, n_tokens, 64)),
            mx.random.normal((1, 2, n_tokens, 64)),
        )
        out.append(c.to_quantized(group_size=64, bits=4))
    return out


def _prefix_cache(monkeypatch, n_tokens):
    calls = []
    real = mx.dequantize
    monkeypatch.setattr(
        kpc.mx, "dequantize", lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    )
    c = kpc.KVPrefixCache(min_prefix_length=4)
    c._ssd_cache = _SSD()
    c._ssd_model_name = "m"
    prompt = mx.arange(n_tokens, dtype=mx.int32)
    c._prompts.append(prompt)
    c._caches.append(_quantized_entry(n_tokens))
    c._block_hashes.append(list(range(1, n_tokens // kpc._BLOCK_SIZE + 1)))
    return c, calls


@pytest.mark.parametrize("path", ["evict", "flush"])
def test_spill_dequantizes_each_layer_once(monkeypatch, path):
    n = 8 * kpc._BLOCK_SIZE
    c, calls = _prefix_cache(monkeypatch, n)
    if path == "evict":
        for name in ("_last_used", "_warm_flags", "_priorities", "_access_counts"):
            getattr(c, name).append(0)
        c._remove_entry(0)
    else:
        monkeypatch.setattr(c, "_snapshot_cache", lambda cache, trim=0: cache)
        c.flush_to_ssd()
    assert c._ssd_cache.saved == [(kpc._BLOCK_SIZE, kpc._BLOCK_SIZE)] * 8
    assert len(calls) == 2 * 2  # keys + values, per layer
