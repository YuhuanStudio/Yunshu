"""KV-prefix-cache eviction strategies must honor an `exclude` set so
_evict_if_full can skip entries the block-evict checker pinned and still evict the
NEXT-best victim. Previously a deterministic strategy kept returning the same pinned
oldest entry, so eviction gave up after one entry and the cache stayed over capacity."""
from __future__ import annotations

from yunshu_engine.kv_prefix_cache import (
    EvictionStrategy,
    FILOStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
)

# last_used: entry 0 oldest ... entry 4 newest
_LAST_USED = [10, 20, 30, 40, 50]
_PRIORITIES = [0, 0, 0, 0, 0]
_ENTRIES = [None] * 5


def test_base_lru_excludes_pinned_oldest():
    s = EvictionStrategy()
    assert s.select_victim(_ENTRIES, _LAST_USED, 0, _PRIORITIES) == 0  # oldest
    # pin entry 0 → must pick the next-oldest (1), not give up on 0
    assert s.select_victim(_ENTRIES, _LAST_USED, 0, _PRIORITIES, exclude={0}) == 1
    assert s.select_victim(_ENTRIES, _LAST_USED, 0, _PRIORITIES, exclude={0, 1}) == 2


def test_lru_strategy_excludes():
    s = LRUStrategy()
    assert s.select_victim(_ENTRIES, _LAST_USED, 0, _PRIORITIES, exclude={0}) == 1


def test_mru_strategy_excludes_newest():
    s = MRUStrategy()
    assert s.select_victim(_ENTRIES, _LAST_USED, 0, _PRIORITIES) == 4  # newest
    assert s.select_victim(_ENTRIES, _LAST_USED, 0, _PRIORITIES, exclude={4}) == 3


def test_filo_strategy_excludes_newest():
    s = FILOStrategy()
    assert s.select_victim(_ENTRIES, _LAST_USED, 0, _PRIORITIES, exclude={4}) == 3


def test_priority_strategy_excludes():
    s = PriorityStrategy()
    # all same priority → LRU tiebreak; pin the lowest-priority-oldest, get next
    assert s.select_victim(_ENTRIES, _LAST_USED, 0, _PRIORITIES, exclude={0}) == 1


def test_priority_all_min_excluded_falls_back():
    s = PriorityStrategy()
    prios = [0, 0, 5, 5, 5]  # entries 0,1 are min-priority
    # exclude both min-priority entries → must still pick a victim (oldest of the rest)
    v = s.select_victim(_ENTRIES, _LAST_USED, 0, prios, exclude={0, 1})
    assert v == 2  # oldest among the remaining


def test_slru_strategy_excludes():
    s = SLRUStrategy()
    s.update_access_counts([0, 0, 0, 0, 0])  # all probationary
    assert s.select_victim(_ENTRIES, _LAST_USED, 0, _PRIORITIES, exclude={0}) == 1
