"""Waiters remain on CPU until a producer can publish a complete checkpoint."""

import threading
from types import SimpleNamespace

from yunshu_engine.vlm_batch_runner import RunStats, VLMBatchRunner, _Job


def job(ids, *, salt=7, cancel=None):
    return _Job(
        ids=ids,
        max_tokens=8,
        greedy=True,
        sampling=None,
        use_draft=False,
        logprobs=False,
        top_logprobs=0,
        processors=[],
        prompt_kwargs={},
        salt=salt,
        seed=None,
        cancel_event=cancel,
        stats=RunStats(),
    )


def runner():
    return VLMBatchRunner(
        object(), object(), apc_manager=SimpleNamespace(), prefix_invariant=True
    )


def test_shared_prefix_waits_but_unrelated_or_other_media_does_not():
    r = runner()
    a = job(list(range(9000)))
    b = job(list(range(8500)) + [99999])
    r._prefix_producers = [(a, [8192])]
    assert r._prefix_wait(b)
    assert not r._prefix_wait(job(b.ids, salt=8))
    assert not r._prefix_wait(job([99999] * 9000))


def test_cancel_error_and_completed_prefill_release_waiter():
    r = runner()
    ev = threading.Event()
    a = job(list(range(9000)), cancel=ev)
    b = job(a.ids)
    r._prefix_producers = [(a, [8192])]
    ev.set()
    assert not r._prefix_wait(b)
    ev.clear()
    a.terminal = True
    assert not r._prefix_wait(b)
    a.terminal = False
    a.stats.t_first = 1.0
    assert not r._prefix_wait(b)


def test_small_shared_prefix_has_no_wait():
    r = runner()
    a = job(list(range(9000)))
    r._prefix_producers = [(a, [8192])]
    assert not r._prefix_wait(job(list(range(100)) + [99999]))


def test_prefill_progress_is_not_a_publication_receipt():
    r = runner()
    ready = [False]
    r.apc_manager.checkpoint_ready = lambda *args: ready[0]
    a = job(list(range(9000)))
    b = job(a.ids)
    a.stats.prefill_done = 8192
    r._prefix_producers = [(a, [8192])]
    assert r._prefix_wait(b)
    ready[0] = True
    assert not r._prefix_wait(b)


def test_unqualified_numerical_configuration_keeps_original_admission():
    r = runner()
    r.prefix_invariant = False
    a = job(list(range(9000)))
    r._prefix_producers = [(a, [8192])]
    assert not r._prefix_wait(job(a.ids))
