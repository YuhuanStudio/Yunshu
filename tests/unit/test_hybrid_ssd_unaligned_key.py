"""(MED): the hybrid SSD whole-snapshot store saved snapshots keyed by the hash
of the FULL (unaligned) prompt, but _get_no_trim_unlocked probed only 64-aligned prefixes
(_token_hash(q[:b]) for b in {64,128,...}). So a snapshot for, say, a 100-token prompt was
written to disk but NEVER restorable (the keys agree only when len % block == 0) — the
W843/W851-class "wrote N MB / restored 0". Fix: probe the EXACT stored token-counts via a
new candidate_token_counts(), which resolves disk-discovered (post-restart) counts from a
cheap safetensors-header read.
"""
from __future__ import annotations

import mlx.core as mx

from yunshu_engine.hybrid_ssd_snapshot import HybridSnapshotStore
from yunshu_engine.kv_prefix_cache import KVPrefixCache, _token_hash


class _FakeKV:
    def __init__(self):
        self.keys = mx.zeros((1, 2, 4, 8))
        self.values = mx.zeros((1, 2, 4, 8))
        self.offset = 4


def test_store_roundtrip_resolves_count_across_restart(tmp_path):
    prompt = mx.arange(100, dtype=mx.int32)        # UNALIGNED length (100 % 64 != 0)
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
            return key == other_key   # the stored snapshot is for a DIFFERENT prompt

        def load(self, key):
            return (["WRONG"], 100)

    c = KVPrefixCache.__new__(KVPrefixCache)
    c._prompts = []
    c._min_prefix = 4
    c._hybrid_ssd = _Hybrid()

    result, remaining, matched = c._get_no_trim_unlocked(query)
    assert result is None and matched == 0 and remaining == 150
