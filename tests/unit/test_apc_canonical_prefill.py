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


def test_turn_boundary_marks_the_end_of_prior_assistant_turns_only():
    # user turn, assistant turn (ended), user turn, assistant generation prompt (open)
    ids = [900, 846, 5, 901, 900, 74455, 6, 901, 900, 846, 7, 901, 900, 74455, 8]
    on = _mgr(
        prefill_stride=64,
        prefill_assistant_header=(900, 74455),
        prefill_message_end=901,
        prefill_turn_boundary=True,
    )
    off = _mgr(
        prefill_stride=64,
        prefill_assistant_header=(900, 74455),
        prefill_message_end=901,
    )
    assert on.semantic_boundaries(ids) == {
        8
    }  # the user header after the assistant turn
    assert off.semantic_boundaries(ids) == set()
    # a single-turn prompt (no prior assistant turn) gets no extra boundary
    assert on.semantic_boundaries([900, 846, 5, 901, 900, 74455, 8]) == set()
    # a user turn ending is not an assistant turn ending
    assert on.semantic_boundaries([900, 846, 5, 901, 900, 846, 7, 901]) == set()


def _two_turn_ids():
    history = (
        [900, 846]
        + list(range(10, 70))
        + [901]
        + [900, 74455]
        + list(range(100, 150))
        + [901]
    )
    boundary = len(history)  # the next turn's header index
    follow_up = history + [900, 846] + list(range(200, 230)) + [901, 900, 74455]
    return history, boundary, follow_up


def _turn_mgr(on):
    return _mgr(
        prefill_stride=64,
        prefill_assistant_header=(900, 74455),
        prefill_message_end=901,
        prefill_turn_boundary=on,
    )


def test_turn_boundary_is_a_cold_forward_split_and_the_rehearsal_final():
    history, boundary, follow_up = _two_turn_ids()
    on, off = _turn_mgr(True), _turn_mgr(False)
    # the follow-up's cold prefill splits at the end of the earlier assistant turn
    assert boundary in _coordinator(on).checkpoint_lengths(follow_up, set())
    assert boundary not in _coordinator(off).checkpoint_lengths(follow_up, set())
    # the rehearsal request ends at the header: its stored checkpoint (len - guard) is that boundary
    rehearsal = history + [900]
    assert _coordinator(on).checkpoint_lengths(rehearsal, set())[-1] == boundary
    # a single-turn prompt is split exactly as without the option
    single = history[: boundary - 51] + [900, 74455] + list(range(300, 340))
    assert _coordinator(on).checkpoint_lengths(single, set()) == _coordinator(
        off
    ).checkpoint_lengths(single, set())


def test_restore_at_the_turn_boundary_is_an_allowed_lookup(monkeypatch):
    history, boundary, follow_up = _two_turn_ids()
    on, off = _turn_mgr(True), _turn_mgr(False)
    for m, want in ((on, True), (off, False)):
        policy = (
            id(m),
            64,
            len(follow_up) - 1,
            0,
            frozenset(m.semantic_boundaries(follow_up)),
        )
        assert (boundary in policy[4]) is want


def test_split_invariant_checkpoints_are_always_absolute_64_aligned():
    manager = _mgr(prefill_stride=2048, prefill_alignment=64)
    manager.head_marker = (90, 91)
    ids = list(range(100)) + [90, 91] + list(range(100, 300))
    points = _coordinator(manager).checkpoint_lengths(ids, set())
    assert points and all(n % 64 == 0 for n in points)
    assert manager.head_boundary(ids) == 64
    assert max(points) == (len(ids) - 1) // 64 * 64


def test_split_invariant_lookup_accepts_rehearsed_subatom_but_not_unaligned():
    manager = _mgr(prefill_stride=2048, prefill_alignment=64)
    ids = list(range(1024))
    manager.store_exact_cache(ids[:640], _cache(640), extra_hash=7)
    manager.store_exact_cache(ids[:700], _cache(700), extra_hash=7)
    assert lookup(_coordinator(manager), ids)["prefix_len"] == 640


@pytest.mark.parametrize(
    "total,prefix",
    [(1, 0), (63, 0), (64, 0), (65, 0), (1001, 0), (1361, 960), (32769, 0)],
)
def test_prompt_dispatch_never_splits_inside_a_recurrence_block(
    monkeypatch, total, prefix
):
    from types import SimpleNamespace

    from yunshu_engine import cache_prefill
    from yunshu_engine.kernels import gdn_prefill

    class Base:
        def needs_processing(self):
            return self._inputs_embeds.shape[1] > 1

        def prompt_step(self):
            n = min(self.prefill_step_size, self._inputs_embeds.shape[1] - 1)
            self._processed_prompt_columns += n
            self._inputs_embeds.shape = (1, self._inputs_embeds.shape[1] - n, 1)
            return n

    ar = SimpleNamespace(PromptProcessingBatch=Base)
    root = SimpleNamespace()
    monkeypatch.setattr(
        cache_prefill.importlib,
        "import_module",
        lambda name: ar if name.endswith(".ar") else root,
    )
    monkeypatch.setitem(gdn_prefill._STATE, "enabled", True)
    monkeypatch.setitem(gdn_prefill._STATE, "split", True)
    cache_prefill.install()
    batch = ar.PromptProcessingBatch()
    batch.prefill_step_size = 2048
    batch._processed_prompt_columns = 0
    batch._cached_tokens_per_row = [prefix]
    batch._inputs_embeds = SimpleNamespace(shape=(1, total - prefix, 1))
    while batch.needs_processing():
        assert batch.prompt_step() > 0
        assert (prefix + batch._processed_prompt_columns) % 64 == 0
    assert 1 <= batch._inputs_embeds.shape[1] <= 64
    assert batch.prefill_step_size == 2048


def test_split_media_checkpoint_also_ends_at_a_complete_block():
    manager = _mgr(prefill_stride=2048, prefill_alignment=64)
    assert _coordinator(manager).checkpoint_lengths(list(range(200)), {121}) == [192]


def test_split_manager_rejects_incomplete_recurrence_states():
    manager = _mgr(prefill_stride=2048, prefill_alignment=64)
    assert not manager.store_exact_cache(list(range(700)), _cache(700), extra_hash=7)
    assert not manager._exact_cache
