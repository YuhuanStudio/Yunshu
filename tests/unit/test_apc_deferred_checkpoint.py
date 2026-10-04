"""Checkpoint snapshots precede mutation; their admission follows first-token delivery."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_vlm.models.cache import ArraysCache, KVCache  # noqa: E402

from tests.unit.test_apc_manager import _coordinator, _mgr  # noqa: E402
from yunshu_engine import vlm_batch_runner as vbr  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_arrays():
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(device)


def test_checkpoint_is_deferred_and_its_lazy_arrays_survive_live_mutation(monkeypatch):
    manager = _mgr()
    coordinator = _coordinator(manager)
    coordinator.defer_checkpoint_stores = True
    rec = ArraysCache(2)
    rec.cache = [mx.ones((1, 2, 3)), mx.ones((1, 2, 3)) * 7]
    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 1, 32, 4)), mx.ones((1, 1, 32, 4)) * 3)
    calls = []

    def store(ids, cache, **kwargs):
        calls.append((ids, cache, kwargs))
        return True

    monkeypatch.setattr(manager, "store_exact_cache", store)
    assert coordinator.store_checkpoint(list(range(32)), [rec, kv], extra_hash=91)
    assert calls == []
    # Mutate the same mx.array objects, replace state, and append KV before evaluating
    # the snapshot. A borrowed cache (or a shallow Python copy) fails this assertion.
    rec.cache[0][:] = 9
    rec.cache[1] = mx.zeros((1, 2, 3))
    kv.keys[:, :, :32, :] = 8
    kv.update_and_fetch(mx.zeros((1, 1, 1, 4)), mx.zeros((1, 1, 1, 4)))
    coordinator.flush_deferred_checkpoints()
    assert len(calls) == 1
    ids, frozen, kwargs = calls[0]
    assert ids == tuple(range(32))
    assert kwargs == {"extra_hash": 91, "_generation": 0, "_owned": True}
    assert frozen[1].offset == 32
    assert frozen[0].cache[0].tolist() == [[[1.0] * 3] * 2]
    assert frozen[0].cache[1].tolist() == [[[7.0] * 3] * 2]
    assert frozen[1].keys[:, :, :32, :].tolist() == [[[[1.0] * 4] * 32]]
    coordinator.flush_deferred_checkpoints()
    assert len(calls) == 1


def test_cancel_before_first_token_drops_pending_checkpoint(monkeypatch):
    manager = _mgr()
    coordinator = _coordinator(manager)
    coordinator.defer_checkpoint_stores = True
    rec = ArraysCache(1)
    rec.cache = [mx.ones((1, 2, 3))]
    calls = []
    monkeypatch.setattr(manager, "store_exact_cache", lambda *a, **kw: calls.append(a))
    coordinator.store_checkpoint(list(range(32)), [rec])
    coordinator.discard_deferred_checkpoints()
    coordinator.flush_deferred_checkpoints()
    assert calls == []


def test_zero_budget_uses_upstream_store_instead_of_retaining_snapshot(monkeypatch):
    manager = _mgr()
    manager.memory_max_bytes = 0
    coordinator = _coordinator(manager)
    coordinator.defer_checkpoint_stores = True
    rec = ArraysCache(1)
    rec.cache = [mx.ones((1, 2, 3))]
    calls = []
    monkeypatch.setattr(manager, "store_exact_cache", lambda *a, **kw: calls.append(a))
    coordinator.store_checkpoint(list(range(32)), [rec])
    assert len(calls) == 1
    assert getattr(coordinator, "_deferred_bytes", 0) == 0
    coordinator.flush_deferred_checkpoints()
    assert len(calls) == 1


def test_batch_rows_use_upstream_extraction(monkeypatch):
    from mlx_vlm.models.cache import BatchKVCache

    manager = _mgr()
    coordinator = _coordinator(manager)
    coordinator.defer_checkpoint_stores = True
    kv = BatchKVCache(mx.array([0, 0]))
    kv.update_and_fetch(mx.ones((2, 1, 32, 4)), mx.ones((2, 1, 32, 4)))
    calls = []
    monkeypatch.setattr(manager, "store_exact_cache", lambda *a, **kw: calls.append(a))
    coordinator.store_checkpoint(list(range(32)), [kv], batch_idx=1)
    assert len(calls) == 1
    assert getattr(coordinator, "_deferred_bytes", 0) == 0


def test_single_speculative_batch_captures_a_normalized_lazy_kv_row(monkeypatch):
    from mlx_vlm.models.cache import BatchKVCache

    manager = _mgr()
    coordinator = _coordinator(manager)
    coordinator.defer_checkpoint_stores = True
    kv = BatchKVCache([0])
    kv.update_and_fetch(mx.ones((1, 1, 32, 4)), mx.ones((1, 1, 32, 4)) * 2)
    calls = []
    monkeypatch.setattr(manager, "store_exact_cache", lambda *a, **kw: calls.append(a))
    assert coordinator.store_checkpoint(list(range(32)), [kv], batch_idx=0)
    assert calls == []
    kv.keys[:, :, :32, :] = 9
    coordinator.flush_deferred_checkpoints()
    snapshot = calls[0][1][0]
    assert type(snapshot) is KVCache
    assert snapshot.offset == 32
    assert snapshot.keys.tolist() == [[[[1.0] * 4] * 32]]
    assert snapshot.values.tolist() == [[[[2.0] * 4] * 32]]


def test_pending_captures_cannot_exceed_apc_budget(monkeypatch):
    from mlx_vlm.apc import _cache_nbytes

    manager = _mgr()
    coordinator = _coordinator(manager)
    coordinator.defer_checkpoint_stores = True
    rec = ArraysCache(1)
    rec.cache = [mx.ones((1, 2, 3))]
    manager.memory_max_bytes = _cache_nbytes([rec])
    calls = []
    monkeypatch.setattr(manager, "store_exact_cache", lambda *a, **kw: calls.append(a))
    coordinator.store_checkpoint(list(range(32)), [rec])
    coordinator.store_checkpoint(list(range(48)), [rec])
    assert len(calls) == 1  # second capture had to store synchronously
    assert coordinator._deferred_bytes <= manager.memory_max_bytes
    coordinator.flush_deferred_checkpoints()
    assert len(calls) == 2


def test_resident_entries_and_captures_share_one_budget(monkeypatch):
    from mlx_vlm.apc import _cache_nbytes

    manager = _mgr()
    coordinator = _coordinator(manager)
    coordinator.defer_checkpoint_stores = True
    rec = ArraysCache(1)
    rec.cache = [mx.ones((1, 2, 3))]
    size = _cache_nbytes([rec])
    manager.memory_max_bytes = size
    monkeypatch.setattr(manager, "resident_bytes", lambda: size)
    calls = []
    monkeypatch.setattr(manager, "store_exact_cache", lambda *a, **kw: calls.append(a))
    coordinator.store_checkpoint(list(range(32)), [rec])
    assert len(calls) == 1
    assert getattr(coordinator, "_deferred_bytes", 0) == 0


def test_publication_preserves_the_capturing_request_generation():
    manager = _mgr()
    coordinator = _coordinator(manager)
    coordinator.defer_checkpoint_stores = True
    rec = ArraysCache(1)
    rec.cache = [mx.ones((1, 2, 3))]
    manager.begin_request()
    coordinator.store_checkpoint(list(range(128)), [rec], extra_hash=91)
    manager.begin_request()  # another group starts while the snapshot is pending
    coordinator.flush_deferred_checkpoints()
    assert list(manager._born.values()) == [1]
    assert manager._generation == 2
    manager.store_exact_cache(list(range(160)), [rec], extra_hash=91)
    assert [len(e.token_ids) for e in manager._exact_cache.values()] == [160]
    assert manager._born[next(iter(manager._exact_cache))] == 2


def test_late_capture_keeps_its_groups_generation():
    manager = _mgr()
    first = _coordinator(manager)
    first.checkpoint_lengths(list(range(128)), set())
    other = _coordinator(manager)
    other.checkpoint_lengths(list(range(160)), set())
    first.defer_checkpoint_stores = True
    rec = ArraysCache(1)
    rec.cache = [mx.ones((1, 2, 3))]
    first.store_checkpoint(list(range(127)), [rec])
    first.flush_deferred_checkpoints()
    assert list(manager._born.values()) == [1]


def test_runner_emits_first_token_before_checkpoint_flush():
    events = []
    coordinator = SimpleNamespace(
        defer_checkpoint_stores=False,
        flush_deferred_checkpoints=lambda: events.append("flush"),
    )

    def step():
        assert coordinator.defer_checkpoint_stores
        events.append("step")
        return [], [SimpleNamespace(uid=1, token=42, finish_reason=None)]

    runner = vbr.VLMBatchRunner(SimpleNamespace(language_model=object()), None)
    runner._emit = lambda *args: events.append("emit")
    runner._note_cache = lambda *args: None
    runner._note_prefill = lambda *args: None
    job = SimpleNamespace(stats=vbr.RunStats(), start=0.0, logprobs=False)
    group = vbr._Group(SimpleNamespace(next=step, apc=coordinator), spec=False)
    group.jobs[1] = job
    runner._step_generator(group)
    assert events == ["step", "emit", "flush"]
    assert not coordinator.defer_checkpoint_stores


def test_lone_ar_uses_speculative_target_arithmetic(monkeypatch):
    from yunshu_engine.kernels import batch_invariant, ragged_kv

    runner = vbr.VLMBatchRunner(SimpleNamespace(language_model=object()), None)
    runner.ragged_kv = "bf16"
    monkeypatch.setitem(batch_invariant._STATE, "installed", True)
    monkeypatch.setitem(batch_invariant._STATE, "active", False)
    seen = []
    runner._step_generator = lambda group, **kwargs: seen.append(
        (batch_invariant._STATE["active"], ragged_kv._STATE["dense_lane"])
    )
    group = vbr._Group(SimpleNamespace(), spec=False)
    group.jobs[1] = SimpleNamespace(cancel_event=None, abandoned=False)
    runner._step_group(group)
    assert seen == [(True, True)]
    assert not batch_invariant._STATE["active"]
    assert not ragged_kv._STATE["dense_lane"]


def test_failed_prefill_discards_pending_captures():
    events = []
    coordinator = SimpleNamespace(
        defer_checkpoint_stores=False,
        flush_deferred_checkpoints=lambda: events.append("flush"),
        discard_deferred_checkpoints=lambda: events.append("discard"),
    )

    def step():
        assert coordinator.defer_checkpoint_stores
        raise RuntimeError("prefill failed")

    runner = vbr.VLMBatchRunner(SimpleNamespace(language_model=object()), None)
    group = vbr._Group(SimpleNamespace(next=step, apc=coordinator), spec=False)
    group.jobs[1] = SimpleNamespace()
    with pytest.raises(RuntimeError, match="prefill failed"):
        runner._step_generator(group)
    assert events == ["discard"]
    assert not coordinator.defer_checkpoint_stores


def test_lone_row_arithmetic_revision_is_in_apc_identity():
    from yunshu_engine.kernels import lane_linear
    from yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine.__new__(VLMEngine)
    engine._model = SimpleNamespace(
        language_model=SimpleNamespace(
            named_modules=lambda: [
                ("proj", lane_linear.LaneLinear.__new__(lane_linear.LaneLinear))
            ]
        )
    )
    assert engine._prefill_kernel_id().endswith("+single-row-invariant-v1")


def test_sg8_target_also_versions_its_lone_row_arithmetic():
    from yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine.__new__(VLMEngine)
    engine._model = SimpleNamespace(
        language_model=SimpleNamespace(
            named_modules=lambda: [("proj", SimpleNamespace(_yunshu_invariant=True))]
        )
    )
    assert engine._prefill_kernel_id().endswith("+single-row-invariant-v1")


def test_cancel_discards_capture_even_after_another_row_joins(monkeypatch):
    events = []
    runner = vbr.VLMBatchRunner(SimpleNamespace(language_model=object()), None)
    runner._finish = lambda group, uid, reason: group.jobs.pop(uid)
    runner._step_generator = lambda group, **kwargs: events.append("step")
    group = vbr._Group(
        SimpleNamespace(
            remove=lambda uid: None,
            apc=SimpleNamespace(
                discard_deferred_checkpoints=lambda: events.append("discard")
            ),
        ),
        spec=False,
    )
    group.jobs = {
        1: SimpleNamespace(cancel_event=None, abandoned=True),
        2: SimpleNamespace(cancel_event=None, abandoned=False),
    }
    runner._step_group(group)
    assert events == ["discard", "step"]
    assert list(group.jobs) == [2]


def test_single_restore_preserves_capacity_without_aliasing_array_handles():
    manager = _mgr()
    coordinator = _coordinator(manager)
    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 1, 32, 4)), mx.ones((1, 1, 32, 4)) * 2)
    rec = ArraysCache(1)
    rec.cache = [mx.ones((1, 2, 3))]
    mx.eval(kv.keys, kv.values, rec.state)
    caches, prefix = coordinator.merge_rows([{"warm_cache": [rec, kv]}], [32])
    batch = caches[1]
    assert prefix == batch._idx == 32
    assert batch.keys.shape == kv.keys.shape  # 256-token reservation survives
    assert batch.keys is not kv.keys
    assert batch.values is not kv.values
    batch.keys[..., :32, :] = 9
    assert kv.keys[..., :32, :].tolist() == [[[[1.0] * 4] * 32]]
    source_capacity = batch.keys.shape[2]
    batch.update_and_fetch(mx.zeros((1, 1, 1, 4)), mx.zeros((1, 1, 1, 4)))
    assert batch.keys.shape[2] == source_capacity
    assert kv.offset == 32


def test_multirow_restore_keeps_upstream_merge():
    manager = _mgr()
    coordinator = _coordinator(manager)
    rows = []
    for n in (32, 48):
        kv = KVCache()
        kv.update_and_fetch(mx.ones((1, 1, n, 4)), mx.ones((1, 1, n, 4)))
        rows.append({"warm_cache": [kv]})
    merged, prefix = coordinator.merge_rows(rows, [32, 48])
    assert prefix == 48
    assert merged[0].keys.shape[:3] == (2, 1, 48)


def test_deferred_policy_survives_prefill_descriptor_release():
    from mlx_vlm.apc import _sequence_hash

    manager = _mgr()
    coordinator = _coordinator(manager)
    ids = list(range(300))
    policy = {"points": [(80, 300), (113, 3600)], "writes": [(113, 3600)], "written": 0}
    coordinator.set_request(ids, policy)
    coordinator.checkpoint_lengths(ids, set())
    coordinator.defer_checkpoint_stores = True
    rec = ArraysCache(1)
    rec.cache = [mx.ones((1, 2, 3))]
    assert not coordinator.store_checkpoint(ids[:80], [rec])
    assert coordinator.store_checkpoint(ids[:113], [rec])
    assert policy["written"] == 0
    assert manager._span_plans == {}
    coordinator.release_request(ids, policy)
    coordinator.flush_deferred_checkpoints()
    key = _sequence_hash(tuple(ids[:113]), 0, manager.block_size)
    assert policy["written"] == 113
    assert manager._span_plans[key] == (80,)
    assert manager._retention[key][1] == 3600


def test_rejected_deferred_admission_does_not_publish_policy(monkeypatch):
    manager = _mgr()
    coordinator = _coordinator(manager)
    ids = list(range(300))
    policy = {"points": [(113, 300)], "written": 0}
    coordinator.set_request(ids, policy)
    coordinator.checkpoint_lengths(ids, set())
    coordinator.defer_checkpoint_stores = True
    rec = ArraysCache(1)
    rec.cache = [mx.ones((1, 2, 3))]
    assert coordinator.store_checkpoint(ids[:113], [rec])
    monkeypatch.setattr(manager, "store_exact_cache", lambda *a, **kw: False)
    coordinator.flush_deferred_checkpoints()
    assert policy["written"] == 0
    assert manager._retention == {}
    assert manager._span_plans == {}


def test_one_token_completion_waits_for_deferred_publication():
    events = []
    policy = {"written": 0}

    def flush():
        policy["written"] = 113
        events.append("flush")

    coordinator = SimpleNamespace(
        defer_checkpoint_stores=False, flush_deferred_checkpoints=flush
    )
    response = SimpleNamespace(uid=1, token=42, finish_reason="length")
    runner = vbr.VLMBatchRunner(SimpleNamespace(language_model=object()), None)
    runner._emit = lambda *args: events.append("emit")
    runner._note_cache = lambda *args: None
    runner._note_prefill = lambda *args: None

    def finish(*args):
        assert policy["written"] == 113
        events.append("done")

    runner._finish = finish
    job = SimpleNamespace(stats=vbr.RunStats(), start=0.0, logprobs=False)
    group = vbr._Group(
        SimpleNamespace(next=lambda: ([], [response]), apc=coordinator), spec=False
    )
    group.jobs[1] = job
    runner._step_generator(group)
    assert events == ["emit", "flush", "done"]


@pytest.mark.parametrize(
    "slots", [(False, False), (False, True), (True, False), (True, True)]
)
def test_private_arrays_restore_view_matches_upstream_state_and_metadata(slots):
    from yunshu_engine.apc_manager import _single_native_arrays_row

    source = ArraysCache(2)
    source.cache = [
        mx.full((1, 2, 3), i + 2) if present else None
        for i, present in enumerate(slots)
    ]
    source.lengths = mx.array([12])
    source.left_padding = mx.array([3])
    expected = ArraysCache.merge([source])
    restored = _single_native_arrays_row(source)
    assert restored is not None
    mx.eval(expected.state, restored.state)
    for a, b in zip(restored.state, expected.state, strict=True):
        if b is None:
            assert a is None
        else:
            assert mx.array_equal(a, b).item()
    assert restored.lengths is expected.lengths is None
    assert (restored.left_padding is None) == (expected.left_padding is None)
    if expected.left_padding is not None:
        assert restored.left_padding.tolist() == expected.left_padding.tolist()
    assert source.lengths.tolist() == [12]
    assert source.left_padding.tolist() == [3]
    if not source.empty():
        index = next(i for i, a in enumerate(source.cache) if a is not None)
        assert restored[index] is not source[index]
        original = source[index].tolist()
        restored[index][:] = 99
        assert source[index].tolist() == original
        source[index][:] = 17
        assert restored[index].tolist() == [[[99.0] * 3] * 2]


def test_private_arrays_restore_view_rejects_multirow_and_custom_contracts():
    from yunshu_engine.apc_manager import _single_native_arrays_row

    source = ArraysCache(1)
    source.cache = [mx.ones((2, 2, 3))]
    assert _single_native_arrays_row(source) is None

    class CustomArraysCache(ArraysCache):
        pass

    assert _single_native_arrays_row(CustomArraysCache(1)) is None


@pytest.mark.parametrize("fresh", [1, 63, 64])
def test_arrays_restore_view_only_for_known_suffix_prefill(monkeypatch, fresh):
    from mlx_vlm import apc_adapters

    manager = _mgr()
    coordinator = _coordinator(manager)
    coordinator.prepare_prefill([32 + fresh], prefill_step_size=2048)
    source = ArraysCache(1)
    source.cache = [mx.ones((1, 2, 3))]
    calls = []
    merge = apc_adapters.merge_cache_entries

    def observed(entries, prefix_lens):
        calls.append(entries[0])
        return merge(entries, prefix_lens)

    monkeypatch.setattr(apc_adapters, "merge_cache_entries", observed)
    restored, prefix = coordinator.merge_rows([{"warm_cache": [source]}], [32])
    assert calls == ([] if fresh >= 64 else [source])
    assert prefix == 32
    assert restored[0][0].tolist() == source[0].tolist()
    assert restored[0][0] is not source[0]


def test_flush_frees_superseded_checkpoints_before_copying_the_new_ones(monkeypatch):
    from yunshu_engine import apc_manager

    manager = _mgr()
    coordinator = _coordinator(manager)
    coordinator.defer_checkpoint_stores = True
    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 1, 32, 4)), mx.ones((1, 1, 32, 4)))
    log = []
    real_release = manager.release_superseded
    monkeypatch.setattr(
        manager,
        "release_superseded",
        lambda *a, **kw: (log.append("release"), real_release(*a, **kw))[1],
    )
    monkeypatch.setattr(
        apc_manager, "materialize", lambda lists, *a, **kw: log.append("copy")
    )
    monkeypatch.setattr(
        manager, "store_exact_cache", lambda *a, **kw: log.append("store") or True
    )
    assert coordinator.store_checkpoint(list(range(32)), [kv])
    coordinator.flush_deferred_checkpoints()
    assert log == ["release", "copy", "store"]


def test_superseded_checkpoint_is_gone_before_the_copy_is_made():
    manager = _mgr()
    coordinator = _coordinator(manager)
    coordinator.defer_checkpoint_stores = True
    old = KVCache()
    old.update_and_fetch(mx.ones((1, 1, 32, 4)), mx.ones((1, 1, 32, 4)))
    manager.begin_request()
    assert manager.store_exact_cache(list(range(32)), [old])
    assert len(manager._exact_cache) == 1
    manager.begin_request()
    live = KVCache()
    live.update_and_fetch(mx.ones((1, 1, 48, 4)), mx.ones((1, 1, 48, 4)))
    seen = []
    from yunshu_engine import apc_manager

    real = apc_manager.materialize
    apc_manager.materialize = lambda lists, *a, **kw: (
        seen.append(len(manager._exact_cache)),
        real(lists, *a, **kw),
    )[1]
    try:
        assert coordinator.store_checkpoint(list(range(48)), [live])
        coordinator.flush_deferred_checkpoints()
    finally:
        apc_manager.materialize = real
    assert seen == [0]  # the 32-token checkpoint was released before the copy
    assert [len(e.token_ids) for e in manager._exact_cache.values()] == [48]


def test_materialize_copies_a_few_positions_at_a_time_with_identical_values(
    monkeypatch,
):
    from yunshu_engine.apc_manager import materialize

    lists = [[mx.ones((2, 2)) * (10 * j + i) + 0 for i in range(10)] for j in range(2)]
    calls = []
    real = mx.eval
    monkeypatch.setattr(mx, "eval", lambda xs: (calls.append(len(xs)), real(xs))[1])
    materialize(lists, group=4)
    assert calls == [8, 8, 4]  # positions 0-3, 4-7, 8-9 of both checkpoints
    assert [x.tolist()[0][0] for x in lists[1]] == [10.0 + i for i in range(10)]
    # unequal lengths still evaluate everything
    ragged = [[mx.ones(3) + 1 for _ in range(5)], [mx.ones(3) + 2 for _ in range(2)]]
    materialize(ragged, group=2)
    assert ragged[1][1].tolist() == [3.0, 3.0, 3.0]


def test_materialize_of_nothing_is_a_noop():
    from yunshu_engine.apc_manager import materialize

    materialize([])
    materialize([[]])
