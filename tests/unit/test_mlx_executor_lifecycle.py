"""R12 / R13: never two GPU workers; cache sync runs on the executor thread."""

import threading

import pytest

from yunshu_engine import mlx_executor as me


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(me, "_init_mlx_thread", lambda: None)
    monkeypatch.setattr(me, "_executor", None)
    yield
    ex = me._executor
    if ex is not None:
        ex.shutdown(wait=True)


def test_r12_reset_with_running_task_keeps_single_worker(fresh):
    ex = me.get_mlx_executor()
    started, release = threading.Event(), threading.Event()
    fut = ex.submit(lambda: (started.set(), release.wait(5)))
    assert started.wait(5)
    again = me.reset_mlx_executor()
    assert again is ex and me.get_mlx_executor() is ex
    release.set()
    fut.result(5)
    workers = [t for t in threading.enumerate() if t.name.startswith("mlx-global")]
    assert len(workers) == 1


def test_r12_reset_replaces_shut_down_executor(fresh):
    ex = me.get_mlx_executor()
    ex.submit(lambda: 1).result(5)
    ex.shutdown(wait=True)
    new = me.reset_mlx_executor()
    assert new is not ex and new.submit(lambda: 2).result(5) == 2


def test_r13_shutdown_syncs_on_executor_thread(fresh, monkeypatch):
    seen = []
    monkeypatch.setattr(
        me, "sync_and_clear_cache", lambda: seen.append(threading.current_thread().name)
    )
    me.get_mlx_executor().submit(lambda: 0).result(5)
    me.shutdown_mlx_executor(wait=True)
    assert seen and seen[0].startswith("mlx-global")
