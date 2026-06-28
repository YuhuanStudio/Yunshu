"""two bugs in the opt-in SSD/cold KV tier (YUNSHU_SSD_CACHE=1).

(1) The disk budget (max_size_bytes / YUNSHU_SSD_CACHE_MAX_GB) was accepted, stored, and
reported in get_stats but NEVER enforced — save_block wrote unconditionally and the only
"eviction" dropped RAM-hot entries to disk (never deleting files), so the SSD KV grew
without bound until the disk filled. Now _enforce_disk_budget_locked evicts LRU on-disk
blocks after each write.

(2) Per-block int8 dequant forced np.float16, so an int8*scale product exceeding fp16's
65504 max (large-model KV outliers / attention sinks; bf16 is the default KV dtype)
silently overflowed to inf → NaN attention on restore. Now dequantizes in float32.
"""
from __future__ import annotations

import inspect
import os
import tempfile
import time

from yunshu_engine import (
    ssd_kv_cache as M,  # noqa: N812  # intentional short module alias
)
from yunshu_engine.ssd_kv_cache import SSDKVCache, _BlockMeta


def _cache(max_bytes):
    d = tempfile.mkdtemp()
    return SSDKVCache(cache_dir=d, max_size_bytes=max_bytes, backend="json")


def _add_block(cache, name, size, age_s):
    """Create a real on-disk file + index entry of `size` bytes, written `age_s` ago."""
    fp = os.path.join(str(cache._cache_dir), name[0], name + ".bin")
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "wb") as f:
        f.write(b"\0" * size)
    cache._index[name] = _BlockMeta(
        block_hash=name.encode(),
        file_path=fp, token_count=10, model_name="m",
        created_at=time.time() - age_s, file_size=size, last_accessed=time.time() - age_s,
    )
    return fp


def test_disk_budget_evicts_lru_until_under():
    cache = _cache(max_bytes=250)
    # three 100-byte blocks (total 300 > 250); a=oldest, c=newest
    fa = _add_block(cache, "aaaaaaaaaaaaaaaa", 100, age_s=300)
    fb = _add_block(cache, "bbbbbbbbbbbbbbbb", 100, age_s=200)
    fc = _add_block(cache, "cccccccccccccccc", 100, age_s=10)
    cache._enforce_disk_budget_locked()
    # the oldest (a) is evicted → 200 <= 250; b and c kept
    assert "aaaaaaaaaaaaaaaa" not in cache._index
    assert not os.path.exists(fa)
    assert "bbbbbbbbbbbbbbbb" in cache._index and os.path.exists(fb)
    assert "cccccccccccccccc" in cache._index and os.path.exists(fc)
    assert cache._evictions == 1


def test_disk_budget_skips_just_written_block():
    cache = _cache(max_bytes=150)
    _add_block(cache, "oldoldoldoldold0", 100, age_s=500)
    fnew = _add_block(cache, "newnewnewnewnew0", 100, age_s=0)
    # even though both are over budget, the just-written block must survive
    cache._enforce_disk_budget_locked(skip_hash="newnewnewnewnew0")
    assert "newnewnewnewnew0" in cache._index and os.path.exists(fnew)
    assert "oldoldoldoldold0" not in cache._index


def test_disk_budget_disabled_when_zero():
    cache = _cache(max_bytes=0)  # 0 = unlimited
    _add_block(cache, "xxxxxxxxxxxxxxxx", 1000, age_s=100)
    cache._enforce_disk_budget_locked()
    assert "xxxxxxxxxxxxxxxx" in cache._index  # nothing evicted


def test_dequant_uses_float32_not_float16():
    src = inspect.getsource(M)
    # the fp16 dequant (overflow→inf) is gone; dequant is float32
    assert "dtype=np.float16)" not in src
    assert "np.float16(k_scale)" not in src and "np.float16(s_scale)" not in src
    assert "np.float32(k_scale)" in src
