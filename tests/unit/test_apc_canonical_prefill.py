"""A restore uses a prefix computed at the same forward boundary as a miss."""

import pytest

mx = pytest.importorskip("mlx.core")

from tests.unit.test_apc_manager import _cache, _coordinator, _mgr  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_arrays():
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(device)


def lookup(coordinator, ids):
    return coordinator.lookup(
        ids,
        extra_hash=7,
        safe_lookup_min=0,
        suffix_is_text_only=lambda _: True,
        prefix_has_media=lambda _: False,
    )


def test_split_grid_does_not_store_every_split(monkeypatch):
    manager = _mgr(prefill_stride=64, prefill_boundary_tokens=(121,))
    coordinator = _coordinator(manager)
    ids = list(range(200))
    assert coordinator.checkpoint_lengths(ids, set()) == [64, 122, 128, 192, 199]
    calls = []
    monkeypatch.setattr(manager, "store_exact_cache", lambda *a, **kw: calls.append(a))
    cache = _cache(64)
    assert not coordinator.store_checkpoint(ids[:64], cache)
    assert calls == []
    assert coordinator.store_checkpoint(ids[:199], cache) is False  # mock returns None
    assert len(calls) == 1


def test_changed_prompt_skips_a_noncanonical_guard():
    manager = _mgr(prefill_stride=64)
    ids = list(range(512))
    manager.store_exact_cache(ids[:320], _cache(320), extra_hash=7)
    manager.store_exact_cache(ids[:350], _cache(350), extra_hash=7)
    hit = lookup(_coordinator(manager), ids + [999])
    assert hit["prefix_len"] == 320


def test_thinking_tag_frontier_stays_reusable_across_turns():
    manager = _mgr(prefill_stride=64, prefill_boundary_tokens=(349,))
    ids = list(range(512))
    manager.store_exact_cache(ids[:350], _cache(350), extra_hash=7)
    hit = lookup(_coordinator(manager), ids + [999])
    assert hit["prefix_len"] == 350


def test_exact_revisit_accepts_its_original_guard():
    manager = _mgr(prefill_stride=64)
    ids = list(range(351))
    manager.store_exact_cache(ids[:350], _cache(350), extra_hash=7)
    assert lookup(_coordinator(manager), ids)["prefix_len"] == 350


def test_no_eligible_checkpoint_is_a_miss():
    manager = _mgr(prefill_stride=64)
    ids = list(range(512))
    manager.store_exact_cache(ids[:350], _cache(350), extra_hash=7)
    assert lookup(_coordinator(manager), ids + [999]) is None


def test_eviction_cannot_fall_back_to_an_arbitrary_guard(monkeypatch):
    manager = _mgr(prefill_stride=64)
    ids = list(range(512))
    manager.store_exact_cache(ids[:320], _cache(320), extra_hash=7)
    manager.store_exact_cache(ids[:300], _cache(300), extra_hash=7)
    choose = manager._canonical_bounds

    def evict(*args):
        n = choose(*args)
        for key, entry in list(manager._exact_cache.items()):
            if len(entry.token_ids) == n:
                del manager._exact_cache[key]
        return n

    monkeypatch.setattr(manager, "_canonical_bounds", evict)
    assert lookup(_coordinator(manager), ids + [999]) is None


def test_disk_metadata_skips_noncanonical_frontiers():
    from types import SimpleNamespace

    manager = _mgr(prefill_stride=64)
    bounds = []

    def find(tokens, *, max_prefix_tokens, min_prefix_tokens, **kw):
        bounds.append((max_prefix_tokens, min_prefix_tokens))
        n = next(
            (n for n in (350, 320) if min_prefix_tokens < n <= max_prefix_tokens), 0
        )
        return (n, n) if n else None

    manager.disk = SimpleNamespace(find_exact_prefix=find)
    policy = (id(manager), 64, 512, 0, frozenset())
    assert manager._canonical_bounds(tuple(range(513)), 7, 512, 0, policy) == 320
    assert bounds == [(512, 0), (320, 0)]


def test_media_keeps_its_existing_boundary_policy():
    manager = _mgr(prefill_stride=64)
    coordinator = _coordinator(manager)
    ids = list(range(200))
    assert coordinator.checkpoint_lengths(ids, {121}) == [199]


def test_user_quoted_thinking_tags_do_not_split_prefill():
    manager = _mgr(
        prefill_stride=64,
        prefill_boundary_tokens=(777, 778),
        prefill_assistant_header=(900, 74455),
        prefill_message_end=901,
    )
    ids = [900, 846, 777, 778, 901, 900, 74455, 777, 10, 778, 901]
    assert manager.semantic_boundaries(ids) == {8, 10}
