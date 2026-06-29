"""PromptCacheManager TTL was a FIXED-LIFETIME timer (created_at), so a hot
prompt — e.g. a long shared system prefix reused on every request — expired 1h after its
FIRST store no matter how often it was hit, forcing a needless full re-prefill. The TTL is
meant to be an IDLE timeout: now _is_expired measures time since last_accessed (already
refreshed on every lookup), so a frequently-reused entry survives indefinitely while a
genuinely idle one still expires. LRU + max_entries still bound total retention.

(Following a CLEAN keying audit — compute_messages_hash includes all prompt-affecting
fields; this is the one efficiency fix the hunt surfaced.)

Tests drive the real clock and adjust each entry's created_at / last_accessed directly
(the dataclass default_factory captures the real time.monotonic at class definition, so
patching the module clock would not affect store-time timestamps).
"""

from __future__ import annotations

import time

from yunshu_engine.prompt_cache import PromptCacheManager, compute_messages_hash


def _entry(cache, h):
    return cache._cache[h]


def test_hot_entry_survives_despite_ancient_created_at():
    """created_at far in the past must NOT expire an entry that was just accessed."""
    cache = PromptCacheManager(ttl_seconds=10.0)
    h = compute_messages_hash([{"role": "system", "content": "shared prefix"}])
    cache.store(h, kv_state=[b"x" * 16], token_count=4)

    now = time.monotonic()
    e = _entry(cache, h)
    e.created_at = now - 100_000.0  # stored a day ago by the OLD (broken) semantics
    e.last_accessed = now - 1.0  # but reused 1s ago → still hot

    assert cache.lookup(h) is not None, "hot entry wrongly expired on created_at"


def test_idle_entry_still_expires():
    """A genuinely idle entry must expire after ttl_seconds since its last access."""
    cache = PromptCacheManager(ttl_seconds=10.0)
    h = compute_messages_hash([{"role": "user", "content": "one-shot"}])
    cache.store(h, kv_state=[b"y" * 16])

    e = _entry(cache, h)
    e.last_accessed = time.monotonic() - 11.0  # idle longer than ttl

    assert cache.lookup(h) is None, "idle entry should have expired"
    stats = cache.get_stats()
    assert stats["misses"] == 1
    assert stats["entries"] == 0


def test_prune_expired_uses_idle_window():
    """prune_expired honours the same idle semantics: hot stays, cold goes."""
    cache = PromptCacheManager(ttl_seconds=10.0)
    hot = compute_messages_hash([{"role": "system", "content": "hot"}])
    cold = compute_messages_hash([{"role": "system", "content": "cold"}])
    cache.store(hot, kv_state=[b"h" * 16])
    cache.store(cold, kv_state=[b"c" * 16])

    now = time.monotonic()
    _entry(cache, hot).last_accessed = now - 3.0  # idle 3s < ttl
    _entry(cache, cold).last_accessed = now - 12.0  # idle 12s > ttl

    pruned = cache.prune_expired()
    assert pruned == 1
    assert cache.has(hot)
    assert not cache.has(cold)
