"""APC SSD tier (SpillDiskStore): corrupt, torn or foreign checkpoints invalidate and
fall back cold, never crash a request or serve wrong state (R07); a state that does not
fit the live model is rejected (R06)."""

from __future__ import annotations

import os

import mlx.core as mx
import pytest

from yunshu_engine.apc_manager import SpillDiskStore, check_loaded_cache

TOKENS = tuple(range(1, 33))
BLOCK = 16


def _cache(n=32, heads=2, dim=4):
    from mlx_vlm.models.cache import ArraysCache, KVCache

    kv = KVCache()
    kv.keys = mx.ones((1, heads, n, dim), dtype=mx.bfloat16)
    kv.values = mx.ones((1, heads, n, dim), dtype=mx.bfloat16) * 2
    kv.offset = n
    ac = ArraysCache(2)
    ac.cache = [mx.ones((1, 3, 4), dtype=mx.bfloat16), mx.ones((1, 2, 2, 2))]
    return [ac, kv]


def _store(tmp_path):
    return SpillDiskStore(tmp_path, namespace="ns", num_workers=1)


def _put(store, tokens=TOKENS):
    from mlx_vlm.apc import _sequence_hash

    key = _sequence_hash(tuple(tokens), 0, BLOCK)
    assert store.write_now(key, tokens, 0, _cache(len(tokens)), True)
    store.flush()
    return key


def _load(store, key, tokens=TOKENS):
    return store.load_exact_cache(key, prefix_len=len(tokens))


def test_roundtrip_baseline(tmp_path):
    store = _store(tmp_path)
    key = _put(store)
    tokens, extra, cache = _load(store, key)
    assert tokens == TOKENS and extra == 0 and len(cache) == 2
    assert check_loaded_cache(tokens, cache, None, kv_heads=2, head_dim=4)


def _path(store, key):
    return store._exact_index[key]


def _gone(store, key, path):
    assert _load(store, key) is None
    assert not path.exists()
    assert key not in store._exact_index
    assert _load(store, key) is None  # never retried


def test_truncated_file_invalidates(tmp_path):
    store = _store(tmp_path)
    key = _put(store)
    path = _path(store, key)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 40])
    store._header_cache.clear()
    _gone(store, key, path)
    assert store.invalidated == 1


def test_garbage_header_invalidates(tmp_path):
    store = _store(tmp_path)
    key = _put(store)
    path = _path(store, key)
    path.write_bytes(os.urandom(512))
    store._header_cache.clear()
    assert _load(store, key) is None  # header unreadable: cold, no exception
    assert key not in store._exact_index


def test_load_exceptions_are_contained(tmp_path, monkeypatch):
    store = _store(tmp_path)
    key = _put(store)
    path = _path(store, key)

    def boom(self, *a, **k):
        raise RuntimeError("corrupt dtype")

    monkeypatch.setattr(SpillDiskStore.__mro__[1], "load_exact_cache", boom)
    assert _load(store, key) is None
    assert not path.exists()


def test_index_failures_go_cold(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _put(store)

    def boom(self, *a, **k):
        raise OSError("disk went away")

    monkeypatch.setattr(SpillDiskStore.__mro__[1], "find_exact_prefix", boom)
    assert store.find_exact_prefix(TOKENS, extra_hash=0) is None
    monkeypatch.setattr(SpillDiskStore.__mro__[1], "load_layer_major_prefix", boom)
    assert store.load_layer_major_prefix([1, 2]) is None


@pytest.mark.parametrize(
    "template_kwargs",
    [
        {"kv_heads": 4},  # another model's attention geometry
        {"head_dim": 8},
    ],
)
def test_validator_rejects_foreign_geometry(tmp_path, template_kwargs):
    store = _store(tmp_path)
    store.validator = lambda tokens, cache: check_loaded_cache(
        tokens, cache, None, **template_kwargs
    )
    key = _put(store)
    path = _path(store, key)
    _gone(store, key, path)


def test_validator_rejects_wrong_layer_structure(tmp_path):
    from mlx_vlm.models.cache import KVCache

    store = _store(tmp_path)
    store.validator = lambda tokens, cache: check_loaded_cache(
        tokens,
        cache,
        [KVCache(), KVCache()],  # model expects two attention layers
    )
    key = _put(store)
    _gone(store, key, _path(store, key))


def test_validator_exception_counts_as_invalid(tmp_path):
    store = _store(tmp_path)

    def raising(tokens, cache):
        raise ValueError("bad")

    store.validator = raising
    key = _put(store)
    _gone(store, key, _path(store, key))


def test_check_loaded_cache_cases():
    toks = tuple(range(32))
    good = _cache()
    assert check_loaded_cache(toks, good)
    assert not check_loaded_cache(toks, [])
    assert not check_loaded_cache((), good)
    short = _cache()
    short[1].offset = 40  # more than tokens / than the keys hold
    assert not check_loaded_cache(toks, short)
    zero = _cache()
    zero[1].offset = 0
    assert not check_loaded_cache(toks, zero)
    batched = _cache()
    batched[1].keys = mx.ones((2, 2, 32, 4), dtype=mx.bfloat16)
    batched[1].values = batched[1].keys
    assert not check_loaded_cache(toks, batched)
    mixed = _cache()
    mixed[1].values = mixed[1].values.astype(mx.float16)
    assert not check_loaded_cache(toks, mixed)
    flat = _cache()
    flat[0].cache[0] = mx.ones((4,))
    assert not check_loaded_cache(toks, flat)


def test_namespace_follows_checkpoint_identity(tmp_path):
    from yunshu_engine.vlm_engine import VLMEngine

    ckpt = tmp_path / "m"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    (ckpt / "tokenizer.json").write_text("{}")
    (ckpt / "model.safetensors").write_bytes(b"a" * 10)
    eng = VLMEngine.__new__(VLMEngine)
    eng._model_path = str(ckpt)
    first = eng._apc_disk_namespace()
    (ckpt / "tokenizer.json").write_text('{"changed": true}')
    second = eng._apc_disk_namespace()
    assert second != first  # tokenizer revision
    (ckpt / "model.safetensors").write_bytes(b"b" * 11)
    assert eng._apc_disk_namespace() != second  # weights replaced in place


def test_apc_keys_depend_on_the_checkpoint(tmp_path):
    """Same prompt, same shapes, different checkpoint identity: the APC extra hash (what
    every RAM / SSD entry is keyed with) differs, via mlx-vlm's apc_key_dependencies."""
    from mlx_vlm.apc import semantic_extra_hash

    from yunshu_engine.vlm_engine import VLMEngine

    class LM:
        pass

    def salt(model_dir):
        eng = VLMEngine.__new__(VLMEngine)
        eng._model_path = str(model_dir)
        eng._apc_backend = None
        lm = LM()
        eng._install_apc_identity(lm)
        return semantic_extra_hash(image_hash=0, media={"audio": None}, model=lm)

    ckpt = tmp_path / "m"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    (ckpt / "model.safetensors").write_bytes(b"a" * 10)
    before = salt(ckpt)
    assert salt(ckpt) == before
    (ckpt / "model.safetensors").write_bytes(b"b" * 10)
    os.utime(ckpt / "model.safetensors", ns=(1, 2))
    assert salt(ckpt) != before
