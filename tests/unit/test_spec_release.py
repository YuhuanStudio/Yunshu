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
