"""the tiered/paged SSDCacheStore (engine_core) must namespace its disk
dir per model. fixed cross-model disk-KV pollution for the default SSDKVCache
path but never propagated to this store — it keyed blocks by content hash with
model_hash always 0, so two models sharing one YUNSHU_SSD_CACHE_DIR + a common token
prefix would serve each other's KV. This locks the per-model isolation contract the
fix relies on (engine_core derives a model key and scopes via _scoped_ssd_dir)."""

from __future__ import annotations

from yunshu_engine.kv_prefix_cache import KVPrefixCache


def _model_key(cfg: dict) -> str:
    """Mirror engine_core's derivation: exact path if present, else an
    architecture signature."""
    key = str(cfg.get("_name_or_path", "") or cfg.get("name_or_path", ""))
    if not key:
        key = "-".join(
            str(cfg.get(a, ""))
            for a in ("model_type", "hidden_size", "num_hidden_layers", "vocab_size")
        )
    return key


def test_distinct_models_get_distinct_dirs():
    base = "/tmp/ssd-test"
    a = KVPrefixCache._scoped_ssd_dir(
        base, _model_key({"_name_or_path": "Qwen2.5-0.5B"})
    )
    b = KVPrefixCache._scoped_ssd_dir(base, _model_key({"_name_or_path": "Llama-3-8B"}))
    assert a != b
    assert a.startswith(base) and b.startswith(base)


def test_same_arch_different_path_isolated():
    base = "/tmp/ssd-test"
    a = KVPrefixCache._scoped_ssd_dir(base, _model_key({"_name_or_path": "ckpt-A"}))
    b = KVPrefixCache._scoped_ssd_dir(base, _model_key({"_name_or_path": "ckpt-B"}))
    assert a != b


def test_arch_signature_fallback_distinguishes():
    base = "/tmp/ssd-test"
    # No _name_or_path → architecture signature; different arch → different dir.
    qwen = _model_key(
        {
            "model_type": "qwen2",
            "hidden_size": 896,
            "num_hidden_layers": 24,
            "vocab_size": 151936,
        }
    )
    llama = _model_key(
        {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "vocab_size": 128256,
        }
    )
    assert qwen != llama
    assert KVPrefixCache._scoped_ssd_dir(base, qwen) != KVPrefixCache._scoped_ssd_dir(
        base, llama
    )


def test_empty_key_keeps_base_dir():
    # Back-compat: no derivable model identity → base dir unchanged.
    assert KVPrefixCache._scoped_ssd_dir("/tmp/ssd-test", "") == "/tmp/ssd-test"
