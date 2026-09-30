"""(HIGH): the KVPrefixCache SSD tier was dead-coded on a cold/empty RAM cache.

Both _get_unlocked (per-block restore) and _get_no_trim_unlocked (hybrid whole-snapshot
probe) started with `if not self._prompts: return None` BEFORE the SSD restore logic. The
SSD tier's whole purpose is to serve prefix hits after RAM eviction or a process restart —
i.e. exactly when self._prompts is empty — so every SSD restore on a cold cache was a
silent miss (proven: blocks written, 0 read back). Fix: skip the early-exit when an SSD
tier exists so the empty-cache fall-through reaches the SSD restore.
"""

from __future__ import annotations

import mlx.core as mx

from yunshu_engine.kv_prefix_cache import KVPrefixCache


def _cache():
    c = KVPrefixCache.__new__(KVPrefixCache)
    c._prompts = []
    c._caches = []
    c._hash_index = {}
    c._min_prefix = 4
    c._ssd_cache = None
    c._hybrid_ssd = None
    c._ssd_restore_min_tokens = 0
    c._prefill_tps = None
    c._ssd_prefill_tps_ceil = 4000.0
    return c


def test_get_unlocked_consults_ssd_on_empty_ram_cache():
    c = _cache()
    c._ssd_cache = object()  # an SSD tier is configured
    c._find_prefix_via_hash_chain = lambda qb: (-1, 0)
    c.restore_prefix_from_ssd = lambda qb: (["RESTORED"], 64)
    prompt = mx.arange(128, dtype=mx.int32)
    result, remaining, matched = c._get_unlocked(prompt)
    # the SSD restore ran despite the empty RAM cache (was an instant None before)
    assert result == ["RESTORED"]
    assert matched == 64
    assert remaining == 128 - 64


def test_get_unlocked_still_fast_exits_with_no_ssd():
    c = _cache()  # no SSD tier
    c._find_prefix_via_hash_chain = lambda qb: (-1, 0)
    prompt = mx.arange(128, dtype=mx.int32)
    result, remaining, matched = c._get_unlocked(prompt)
    assert result is None and matched == 0 and remaining == 128


def test_get_no_trim_consults_hybrid_ssd_on_empty_ram_cache():
    c = _cache()

    class _HybridSSD:
        def candidate_token_counts(self):  # probe iterates exact stored counts
            return [128]

        def has(self, key):
            return True

        def load(self, key):
            return (["HYBRID_RESTORED"], 128)

    c._hybrid_ssd = _HybridSSD()
    prompt = mx.arange(128, dtype=mx.int32)
    result, remaining, matched = c._get_no_trim_unlocked(prompt)
    assert result == ["HYBRID_RESTORED"]
    assert matched == 128
    assert remaining == 0


def test_get_no_trim_still_fast_exits_with_no_hybrid_ssd():
    c = _cache()
    prompt = mx.arange(128, dtype=mx.int32)
    result, remaining, matched = c._get_no_trim_unlocked(prompt)
    assert result is None and matched == 0 and remaining == 128


def test_ssd_longer_than_ram_hit_wins_and_short_ram_hit_does_not_hide_it():
    class _SSD:
        def __init__(self, blocks):
            self.blocks = blocks

        def has_block(self, key):
            return key in self.blocks

    c = _cache()
    keys = [i.to_bytes(8, "little") for i in range(40)]
    c._ssd_cache = _SSD(set(keys[:30]))  # 30 blocks = 1920 tokens on disk
    seen = {}

    def restore(qb):
        seen["n"] = len(qb)
        return ["RESTORED"], len(qb) * 64

    c.restore_prefix_from_ssd = restore
    query = list(range(40))
    # RAM holds only a 256-token prefix: the disk run is much longer -> SSD wins
    assert c._ssd_longer_than_ram(query, 256) == (["RESTORED"], 30 * 64)
    assert seen["n"] == 30
    # RAM hit within the minimum gain of the disk run: keep RAM
    assert c._ssd_longer_than_ram(query, 30 * 64 - 100) is None
    # prefill-speed gate: a fast model re-prefills instead
    c._prefill_tps = 9000.0
    assert c._ssd_longer_than_ram(query, 256) is None
    # no SSD tier: nothing to do
    c._ssd_cache = None
    assert c._ssd_longer_than_ram(query, 256) is None
