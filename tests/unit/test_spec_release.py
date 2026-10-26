import gc
import weakref
from types import SimpleNamespace

from yunshu_engine.spec_release import release_rounds


class Payload:
    pass


def _batch_with_cycle():
    """Mimic GenerationBatch._start_rounds: closure over the batch kept by its generator."""
    batch = SimpleNamespace(_rounds_iter=None)
    payload = Payload()

    def stop_check():
        return batch

    def rounds(held, check):
        yield 1
        yield 2

    batch._rounds_iter = rounds(payload, stop_check)
    next(batch._rounds_iter)
    return batch, weakref.ref(payload)


def test_cycle_holds_payload_without_release():
    gc.disable()
    try:
        batch, ref = _batch_with_cycle()
        gen = SimpleNamespace(_generation_batch=batch)
        del batch, gen
        assert ref() is not None  # the cycle keeps it alive until a GC pass
        gc.collect()
    finally:
        gc.enable()


def test_release_frees_payload_immediately():
    gc.disable()
    try:
        batch, ref = _batch_with_cycle()
        gen = SimpleNamespace(_generation_batch=batch)
        assert release_rounds(gen) is True
        assert batch._rounds_iter is None
        del batch, gen
        assert ref() is None  # no gc.collect(): refcount alone frees it
    finally:
        gc.enable()


def test_release_without_rounds_is_a_noop():
    assert release_rounds(SimpleNamespace(_generation_batch=None)) is False
    assert release_rounds(SimpleNamespace()) is False


class _FakeSpecBatch:
    """mlx-vlm 259802e SpeculativeGenerationBatch: the last finished row calls _release()."""

    def __init__(self, uids, cache, release_on_empty):
        self.uids = list(uids)
        self.prompt_cache = cache
        self.hidden = object()
        self.shared_kv_states = object()
        self.prompt_tokens = object()
        self._rounds_iter = None
        self._release_on_empty = release_on_empty

    def finish(self, uid):
        self.uids.remove(uid)
        if not self.uids and self._release_on_empty:
            self.prompt_cache = []
            self.hidden = self.shared_kv_states = self.prompt_tokens = None


def test_snapshot_survives_the_release_of_a_finished_batch():
    from yunshu_engine.spec_release import finished_caches, snapshot_caches

    cache = [object(), object()]
    batch = _FakeSpecBatch([7], cache, release_on_empty=True)
    gen = SimpleNamespace(_generation_batch=batch)
    snap = snapshot_caches(gen, [7])
    assert snap == cache
    batch.finish(7)
    assert batch.prompt_cache == []  # what the runner used to read at _finish
    assert finished_caches(gen, snap) == cache


def test_pre_release_batches_keep_their_own_list():
    from yunshu_engine.spec_release import finished_caches, snapshot_caches

    cache = [object()]
    batch = _FakeSpecBatch([7], cache, release_on_empty=False)
    gen = SimpleNamespace(_generation_batch=batch)
    snap = snapshot_caches(gen, [7])
    batch.finish(7)
    assert finished_caches(gen, snap) == cache


def test_snapshot_ignores_other_or_missing_batches():
    from yunshu_engine.spec_release import finished_caches, snapshot_caches

    cache = [object()]
    gen = SimpleNamespace(_generation_batch=_FakeSpecBatch([3], cache, True))
    assert snapshot_caches(gen, [7]) is None  # an older/other batch is never captured
    assert snapshot_caches(gen, [3, 4]) is None
    assert snapshot_caches(SimpleNamespace(_generation_batch=None), [7]) is None
    assert snapshot_caches(SimpleNamespace(), [7]) is None
    assert finished_caches(SimpleNamespace(), None) == []


def test_offer_after_release_uses_the_snapshot():
    """The runner's _finish offers the finished KV set; after 259802e only the snapshot has it."""
    import mlx.core as mx
    from mlx_vlm.models.cache import KVCache

    from yunshu_engine.kernels import buffer_cache
    from yunshu_engine.spec_release import finished_caches, snapshot_caches

    kv = KVCache()
    kv.keys = mx.zeros((1, 1, 8, 4))
    kv.values = mx.zeros((1, 1, 8, 4))
    mx.eval(kv.keys, kv.values)
    batch = _FakeSpecBatch([1], [kv], release_on_empty=True)
    gen = SimpleNamespace(_generation_batch=batch)
    snap = snapshot_caches(gen, [1])
    batch.finish(1)
    caches = finished_caches(gen, snap)
    assert caches == [kv]
    seen = {}
    orig = buffer_cache.restore_buffer_bytes
    try:
        buffer_cache._STATE["limit"] = 10**9
        buffer_cache._RESTORE_BUFFERS[:] = []
        buffer_cache.offer_restore_buffers(caches)
        seen["pool"] = len(buffer_cache._RESTORE_BUFFERS)
    finally:
        buffer_cache._RESTORE_BUFFERS[:] = []
        buffer_cache.restore_buffer_bytes = orig
    assert seen["pool"] == 2
    assert buffer_cache.offer_restore_buffers([]) is None
