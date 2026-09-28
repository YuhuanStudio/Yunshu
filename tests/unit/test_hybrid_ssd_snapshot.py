"""HybridSnapshotStore save/load round-trip (whole-snapshot SSD tier for hybrid /
recurrent models, e.g. Qwen3.5).

Regression guard for the →688 bug: the atomic write used a temp name that
did NOT end in `.safetensors`; mx.save_safetensors APPENDS that extension, so the
file landed at `{path}.tmp.{pid}.safetensors` and the os.replace of `{path}.tmp.{pid}`
raised (swallowed) → EVERY save silently failed → has()→False → every restore fell
back to a full prefill (the hybrid SSD tier was dead for days; measured Qwen3.5
SSD restore 400ms → 2880ms ≈ cold). These tests assert the file lands at the FINAL
path, has() is True, load round-trips, and no temp file leaks.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from yunshu_engine.hybrid_ssd_snapshot import HybridSnapshotStore


def _kv(n=8, layers=2):
    from mlx_lm.models.cache import KVCache

    out = []
    for _ in range(layers):
        c = KVCache()
        c.keys = mx.ones((1, 2, n, 4))
        c.values = mx.ones((1, 2, n, 4)) * 2
        c.offset = n
        out.append(c)
    return out


def test_save_lands_at_final_path_and_has_is_true(tmp_path):
    store = HybridSnapshotStore(str(tmp_path))
    key = b"abcd1234efgh5678"
    store.save(key, _kv(), token_count=8)
    assert store.has(key), "save must register the key (it silently failed pre-fix)"
    p = store._path(key.hex())
    assert p.exists(), "snapshot must land at the FINAL .safetensors path"
    # no temp file left behind
    leftovers = [f for f in p.parent.iterdir() if ".tmp." in f.name]
    assert not leftovers, f"atomic temp file leaked: {leftovers}"


def test_load_roundtrips(tmp_path):
    store = HybridSnapshotStore(str(tmp_path))
    key = b"deadbeefcafef00d"
    store.save(key, _kv(n=8), token_count=8)
    cache_list, tok = store.load(key)
    assert cache_list is not None and tok == 8
    assert len(cache_list) == 2
    # values were 2.0 (bit-exact by default)
    v = cache_list[0].values
    assert v is not None and abs(float(mx.max(v)) - 2.0) < 0.05


def test_persists_across_reopen(tmp_path):
    key = b"0123456789abcdef"
    HybridSnapshotStore(str(tmp_path)).save(key, _kv(), token_count=8)
    # a fresh store scans the dir on init — the entry must be discoverable
    assert HybridSnapshotStore(str(tmp_path)).has(key)


def test_scan_skips_and_cleans_temp_files(tmp_path):
    store = HybridSnapshotStore(str(tmp_path))
    key = b"feedface00001111"
    store.save(key, _kv(), token_count=8)
    # drop a stray temp leftover next to the real file
    sub = store._path(key.hex()).parent
    stray = sub / "deadc0de.99.tmp.safetensors"
    stray.write_bytes(b"junk")
    fresh = HybridSnapshotStore(str(tmp_path))
    assert fresh.has(key)
    assert not stray.exists(), "stray .tmp. file must be cleaned, not indexed"
    assert "deadc0de.99.tmp" not in fresh._index


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def _hybrid(n=8):
    from mlx_lm.models.cache import ArraysCache, KVCache

    kv = KVCache()
    kv.keys = (mx.random.normal((1, 2, n, 8)) * 50).astype(mx.bfloat16)
    kv.values = (mx.random.normal((1, 2, n, 8)) * 0.01).astype(mx.bfloat16)
    kv.offset = n
    ac = ArraysCache(2)
    ac.cache = [mx.random.normal((1, 3, 16)), mx.random.normal((1, 4, 8, 8))]
    return [kv, ac]


def test_default_roundtrip_is_bit_exact(tmp_path):
    """Lossy storage is opt-in: the default store keeps KV and recurrent state exact."""
    store = HybridSnapshotStore(str(tmp_path))
    key = b"exactexactexact0"
    src = _hybrid()
    store.save(key, src, token_count=8)
    out, _ = store.load(key)
    assert out[0].keys.dtype == mx.bfloat16
    assert mx.array_equal(out[0].keys, src[0].keys).item()
    assert mx.array_equal(out[0].values, src[0].values).item()
    for a, b in zip(out[1].cache, src[1].cache, strict=True):
        assert mx.array_equal(a, b).item()


def test_int8_option_is_lossy_but_close(tmp_path):
    store = HybridSnapshotStore(str(tmp_path), precision="int8")
    key = b"int8int8int8int8"
    src = _hybrid()
    store.save(key, src, token_count=8)
    out, _ = store.load(key)
    a, b = out[1].cache[0], src[1].cache[0]
    err = mx.max(mx.abs(a - b)).item()
    assert 0 < err < 0.1


def test_unknown_precision_rejected(tmp_path):
    with pytest.raises(ValueError):
        HybridSnapshotStore(str(tmp_path), precision="fp4")
