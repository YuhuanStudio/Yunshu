"""Persisted KV states are only valid for the checkpoint that wrote them: same name /
path with different weights, config, tokenizer, adapter or cache layout must select a
different namespace and never read the old states back (R04 / R06)."""

from __future__ import annotations

import os

from yunshu_engine.kv_prefix_cache import KVPrefixCache
from yunshu_kv.fingerprint import checkpoint_fingerprint


def _ckpt(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text('{"hidden_size": 8}')
    (d / "tokenizer.json").write_text("{}")
    (d / "tokenizer_config.json").write_text('{"chat_template": "a"}')
    (d / "model.safetensors").write_bytes(b"w" * 100)
    return d


def test_same_files_same_fingerprint(tmp_path):
    d = _ckpt(tmp_path)
    assert checkpoint_fingerprint(d) == checkpoint_fingerprint(str(d))


def test_each_identity_part_changes_the_fingerprint(tmp_path):
    d = _ckpt(tmp_path)
    seen = {checkpoint_fingerprint(d)}

    def check():
        fp = checkpoint_fingerprint(d)
        assert fp not in seen
        seen.add(fp)

    (d / "model.safetensors").write_bytes(b"x" * 101)  # weights replaced in place
    check()
    st = (d / "model.safetensors").stat()
    os.utime(d / "model.safetensors", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    check()  # same size, newer mtime
    (d / "config.json").write_text('{"hidden_size": 16}')
    check()
    (d / "tokenizer_config.json").write_text('{"chat_template": "b"}')  # template
    check()
    (d / "tokenizer.json").write_text('{"v": 1}')
    check()
    (d / "chat_template.jinja").write_text("{{ x }}")
    check()


def test_adapter_extra_and_format_version(tmp_path, monkeypatch):
    d = _ckpt(tmp_path)
    adapter = tmp_path / "lora"
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"a")
    base = checkpoint_fingerprint(d)
    with_adapter = checkpoint_fingerprint(d, adapter_paths=(adapter,))
    assert with_adapter != base
    (adapter / "adapters.safetensors").write_bytes(b"ab")
    assert checkpoint_fingerprint(d, adapter_paths=(adapter,)) != with_adapter
    assert checkpoint_fingerprint(d, extra={"kv_precision": "int8"}) != base
    assert checkpoint_fingerprint(d, extra={"a": 1, "b": 2}) == checkpoint_fingerprint(
        d, extra={"b": 2, "a": 1}
    )
    import yunshu_kv.fingerprint as fp

    monkeypatch.setattr(fp, "FORMAT_VERSION", fp.FORMAT_VERSION + 1)
    assert checkpoint_fingerprint(d) != base  # a layout change invalidates everything


def test_non_local_path_is_identified_by_string_and_extra():
    a = checkpoint_fingerprint("org/model-a")
    assert a == checkpoint_fingerprint("org/model-a")
    assert a != checkpoint_fingerprint("org/model-b")


def test_lm_ssd_dir_follows_the_fingerprint(tmp_path):
    d = _ckpt(tmp_path)
    base = str(tmp_path / "kv")
    a = KVPrefixCache._scoped_ssd_dir(base, str(d), checkpoint_fingerprint(d))
    (d / "model.safetensors").write_bytes(b"new weights")
    b = KVPrefixCache._scoped_ssd_dir(base, str(d), checkpoint_fingerprint(d))
    assert a != b  # same name, different revision: a different (empty) directory


def test_lm_block_store_drops_blocks_of_another_fingerprint(tmp_path):
    """Even in a shared directory, a block written under another fingerprint is never
    served (header check on recovery and on load)."""
    import mlx.core as mx

    from yunshu_engine.ssd_kv_cache import SSDKVCache

    block = [(mx.ones((1, 2, 4, 4)), mx.ones((1, 2, 4, 4)))]
    key = b"k" * 16
    old = SSDKVCache(str(tmp_path), fingerprint="rev-A", hot_cache_size=0)
    old.save_block(key, block, token_count=4)
    old.close()
    same = SSDKVCache(str(tmp_path), fingerprint="rev-A", hot_cache_size=0)
    assert same.load_block(key) is not None
    same.close()
    other = SSDKVCache(str(tmp_path), fingerprint="rev-B", hot_cache_size=0)
    assert other.load_block(key) is None
    assert not other.has_block(key)
    other.close()
