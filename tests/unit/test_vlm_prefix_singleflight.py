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


def test_driver_job_registers_its_planned_checkpoints_as_a_producer():
    """The round driver owns its APC: a driver job is a single-flight producer of the
    checkpoints its row plans (published after its first token), so a same-prefix waiter waits."""
    r = runner()
    a = job(list(range(9000)))
    a.stats.used_apc = True
    a.cache_store = True
    drv = SimpleNamespace(
        planned_checkpoints=lambda h: (123, [8192]) if h is a else None
    )
    a.driver = drv
    r._driver_jobs[id(a)] = a
    r._track_prefix_producer(a)
    assert r._prefix_producers == [(a, [8192])] and a.apc_salt == 123
    b = job(list(range(8500)) + [99999])
    assert r._prefix_wait(b)
    # nothing planned (no APC / no checkpoints): no producer
    r2 = runner()
    c = job(list(range(9000)))
    c.stats.used_apc = c.cache_store = True
    c.driver = SimpleNamespace(planned_checkpoints=lambda h: None)
    r2._driver_jobs[id(c)] = c
    r2._track_prefix_producer(c)
    assert r2._prefix_producers == []
