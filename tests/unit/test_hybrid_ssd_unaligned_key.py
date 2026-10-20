"""(MED): the hybrid SSD whole-snapshot store saved snapshots keyed by the hash
of the FULL (unaligned) prompt, but _get_no_trim_unlocked probed only 64-aligned prefixes
(_token_hash(q[:b]) for b in {64,128,...}). So a snapshot for, say, a 100-token prompt was
written to disk but NEVER restorable (the keys agree only when len % block == 0) — the
class "wrote N MB / restored 0". Fix: probe the EXACT stored token-counts via a
new candidate_token_counts(), which resolves disk-discovered (post-restart) counts from a
cheap safetensors-header read.
"""

from __future__ import annotations

import mlx.core as mx

from yunshu_engine.hybrid_ssd_snapshot import HybridSnapshotStore
from yunshu_engine.kv_prefix_cache import KVPrefixCache, _token_hash


class _FakeKV:
    def __init__(self):
        self.keys = mx.zeros((1, 2, 100, 8))
        self.values = mx.zeros((1, 2, 100, 8))
        self.offset = 100


def test_store_roundtrip_resolves_count_across_restart(tmp_path):
    prompt = mx.arange(100, dtype=mx.int32)  # UNALIGNED length (100 % 64 != 0)
    key = _token_hash(prompt).encode("utf-8")[:32]

    store = HybridSnapshotStore(str(tmp_path))
    store.save(key, [_FakeKV()], token_count=100)
    assert store.has(key)
    assert store.candidate_token_counts() == [100]

    # Simulate a process restart: a fresh store re-scans the dir → counts lazily 0.
    store2 = HybridSnapshotStore(str(tmp_path))
    # the header-read resolves the real count even though nothing has been load()ed
    assert store2.candidate_token_counts() == [100]
    cache_list, tok = store2.load(key)
    assert cache_list is not None and tok == 100


def test_get_no_trim_restores_unaligned_snapshot(tmp_path):
    # The PROBE must hit a 100-token snapshot for a 150-token query whose first 100
    # tokens match — the old 64-aligned probe would only try b=128,64 and miss.
    prompt150 = mx.arange(150, dtype=mx.int32)
    expected_key = _token_hash(prompt150[:100]).encode("utf-8")[:32]

    class _Hybrid:
        def candidate_token_counts(self):
            return [100]

        def has(self, key):
            return key == expected_key

        def load(self, key):
            return (["SNAPSHOT"], 100) if key == expected_key else (None, 0)

    c = KVPrefixCache.__new__(KVPrefixCache)
    c._prompts = []
    c._min_prefix = 4
    c._hybrid_ssd = _Hybrid()

    result, remaining, matched = c._get_no_trim_unlocked(prompt150)
    assert result == ["SNAPSHOT"]
    assert matched == 100
    assert remaining == 50


def test_get_no_trim_no_false_hit_when_prefix_differs(tmp_path):
    # A stored 100-token count exists, but the query's first 100 tokens DON'T match it
    # (different hash) → has() is False → no restore (no corruption from a count-only match).
    query = mx.arange(150, dtype=mx.int32)
    other_key = _token_hash(mx.arange(100, 200, dtype=mx.int32)).encode("utf-8")[:32]

    class _Hybrid:
        def candidate_token_counts(self):
            return [100]

        def has(self, key):
            return key == other_key  # the stored snapshot is for a DIFFERENT prompt

        def load(self, key):
            return (["WRONG"], 100)

    c = KVPrefixCache.__new__(KVPrefixCache)
    c._prompts = []
    c._min_prefix = 4
    c._hybrid_ssd = _Hybrid()

    result, remaining, matched = c._get_no_trim_unlocked(query)
    assert result is None and matched == 0 and remaining == 150


def test_snapshot_shorter_than_its_token_count_is_rejected(tmp_path):
    # The attention layer holds 50 positions but the record claims 100 tokens: a hit
    # would skip 50 prompt tokens that were never prefilled.
    key = _token_hash(mx.arange(100, dtype=mx.int32)).encode("utf-8")[:32]
    short = _FakeKV()
    short.offset = 50
    store = HybridSnapshotStore(str(tmp_path))
    store.save(key, [short], token_count=100)
    cache_list, tok = HybridSnapshotStore(str(tmp_path)).load(key)
    assert cache_list is None and tok == 0


def _ram_and_ssd(ram_len, ssd_len, query):
    ssd_key = _token_hash(query[:ssd_len]).encode("utf-8")[:32]

    class _Hybrid:
        def candidate_token_counts(self):
            return [ssd_len]

        def has(self, key):
            return key == ssd_key

        def load(self, key):
            return (["SSD"], ssd_len)

    c = KVPrefixCache.__new__(KVPrefixCache)
    c._prompts = [query[:ram_len]]
    c._caches = ["RAM"]
    c._min_prefix = 4
    c._hybrid_ssd = _Hybrid()
    c._snapshot_cache = lambda cached, trim: [cached]
    c._touch = lambda i: None
    return c


def test_longer_ssd_snapshot_beats_a_short_ram_hit(monkeypatch):
    # RAM keeps a 128-token system prefix; SSD holds the 4K conversation checkpoint.
    import yunshu_engine.kv_prefix_cache as kpc

    monkeypatch.setattr(kpc, "cache_length", lambda cached: 128)
    query = mx.arange(5000, dtype=mx.int32)
    result, remaining, matched = _ram_and_ssd(128, 4096, query)._get_no_trim_unlocked(
        query
    )
    assert result == ["SSD"] and matched == 4096 and remaining == 904


def test_ssd_snapshot_barely_longer_than_ram_keeps_the_ram_hit(monkeypatch):
    import yunshu_engine.kv_prefix_cache as kpc

    monkeypatch.setattr(kpc, "cache_length", lambda cached: 4000)
    query = mx.arange(5000, dtype=mx.int32)
    result, remaining, matched = _ram_and_ssd(4000, 4096, query)._get_no_trim_unlocked(
        query
    )
    assert result == ["RAM"] and matched == 4000
