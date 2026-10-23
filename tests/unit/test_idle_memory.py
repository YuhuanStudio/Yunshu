"""Idle schedule that returns cycle garbage and the allocator pool once the runner drains."""

import threading
import time

from yunshu_engine.idle_memory import IdleMemory


def _wait(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def _make(busy_flag, log, collect_s=0.02, trim_s=0.06):
    return IdleMemory(
        lambda fn: fn(),
        lambda: busy_flag["busy"],
        collect_after_s=collect_s,
        trim_after_s=trim_s,
        collect=lambda: log.append("collect"),
        trim=lambda: log.append("trim"),
    )


def test_drain_collects_then_trims_in_order():
    log, busy = [], {"busy": False}
    idle = _make(busy, log)
    idle.drained()
    assert _wait(lambda: log == ["collect", "trim"])
    assert (idle.collections, idle.trims) == (1, 1)


def test_new_request_cancels_the_schedule():
    log, busy = [], {"busy": False}
    idle = _make(busy, log, collect_s=0.1, trim_s=0.2)
    idle.drained()
    idle.activity()
    time.sleep(0.35)
    assert log == []


def test_stage_skipped_while_runner_is_busy():
    log, busy = [], {"busy": True}
    idle = _make(busy, log)
    idle.drained()
    time.sleep(0.2)
    assert log == []  # busy at the first stage: nothing runs, the chain stops
    busy["busy"] = False
    idle.drained()
    assert _wait(lambda: log == ["collect", "trim"])


def test_trim_only_when_still_idle_after_collect():
    log, busy = [], {"busy": False}
    idle = _make(busy, log)

    def collect():
        log.append("collect")
        busy["busy"] = True  # a request slips in right after the collection

    idle._stages[0] = (idle._stages[0][0], collect, "collections")
    idle.drained()
    time.sleep(0.25)
    assert log == ["collect"]


def test_stages_run_on_the_submitted_executor_not_the_timer_thread():
    ran_on = []
    done = threading.Event()
    box = {}

    def submit(fn):
        box["fn"] = fn

    idle = IdleMemory(
        submit,
        lambda: False,
        collect_after_s=0.01,
        trim_after_s=0.02,
        collect=lambda: ran_on.append(threading.current_thread().name),
        trim=lambda: done.set(),
    )
    idle.drained()
    assert _wait(lambda: "fn" in box)
    assert ran_on == []  # the timer only submitted
    box.pop("fn")()
    assert ran_on == [threading.current_thread().name]


def _runner(monkeypatch):
    import importlib
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from tests.unit.test_vlm_runner_batching import FakeGen
    from yunshu_engine import vlm_batch_runner as vbr

    ar = importlib.import_module("mlx_vlm.generate.ar")
    FakeGen.instances = []
    monkeypatch.setattr(ar, "BatchGenerator", FakeGen)
    rn = vbr.VLMBatchRunner(SimpleNamespace(language_model=object()), processor=None)
    rn._executor = ThreadPoolExecutor(max_workers=1)
    return rn


def test_runner_drain_runs_collect_then_trim(monkeypatch):
    rn = _runner(monkeypatch)
    seen = []
    rn._idle_memory = IdleMemory(
        rn._executor.submit,
        rn.busy,
        collect_after_s=0.02,
        trim_after_s=0.05,
        collect=lambda: seen.append("collect"),
        trim=lambda: seen.append("trim"),
    )
    assert list(rn.iter_tokens([1, 2, 3], max_tokens=2, prompt_kwargs={}))
    assert _wait(lambda: seen == ["collect", "trim"])
    rn._executor.shutdown(wait=True)


def test_runner_request_cancels_pending_idle_schedule(monkeypatch):
    rn = _runner(monkeypatch)
    seen = []
    idle = IdleMemory(
        rn._executor.submit,
        rn.busy,
        collect_after_s=0.4,
        trim_after_s=0.8,
        collect=lambda: seen.append("collect"),
        trim=lambda: seen.append("trim"),
    )
    rn._idle_memory = idle
    idle.drained()
    assert idle._timer is not None
    # a request submitted to the runner cancels the schedule armed by the earlier drain;
    # its own drain then re-arms it, so only one pass runs after the last request
    assert list(rn.iter_tokens([1, 2, 3], max_tokens=1, prompt_kwargs={}))
    assert _wait(lambda: seen == ["collect", "trim"], timeout=5)
    assert seen == ["collect", "trim"]
    rn._executor.shutdown(wait=True)


def test_closing_a_group_breaks_the_speculative_round_cycle_without_the_collector():
    """A finished speculative batch is freed by reference counting, not by gc."""
    import gc
    import weakref

    from yunshu_engine.vlm_batch_runner import close_generator

    class Cache:
        pass

    class Batch:  # stands in for SpeculativeGenerationBatch
        def __init__(self):
            self.prompt_cache = [Cache()]
            self._rounds_iter = None

        def start(self):
            def stop_check(i):  # closes over the batch, as upstream's does
                return self.prompt_cache

            def rounds():
                transaction = (stop_check, self.prompt_cache)
                while True:
                    yield transaction

            self._rounds_iter = rounds()
            next(self._rounds_iter)

    class Gen:
        closed = False

        def __init__(self):
            self._generation_batch = Batch()
            self._generation_batch.start()

        def close(self):
            self.closed = True

    gen = Gen()
    cache_ref = weakref.ref(gen._generation_batch.prompt_cache[0])
    gc.collect()
    gc.disable()
    try:
        close_generator(gen)
        assert gen.closed
        del gen
        assert cache_ref() is None  # freed at once: no cycle left for gc to find
    finally:
        gc.enable()


def test_close_generator_tolerates_generators_without_speculation():
    from yunshu_engine.vlm_batch_runner import close_generator

    class Plain:
        closed = False

        def close(self):
            self.closed = True

    g = Plain()
    close_generator(g)
    assert g.closed


def test_idle_trim_releases_host_allocator_after_mlx(monkeypatch):
    from yunshu_engine import idle_memory, mlx_executor
    from yunshu_engine.kernels import buffer_cache

    calls = []
    monkeypatch.setattr(
        buffer_cache, "clear_restore_buffers", lambda: calls.append("spares")
    )
    monkeypatch.setattr(idle_memory.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(idle_memory, "_gib", lambda: (20, 0))
    monkeypatch.setattr(
        mlx_executor, "sync_and_clear_cache", lambda: calls.append("mlx")
    )
    monkeypatch.setattr(
        idle_memory, "_trim_host_allocator", lambda: calls.append("host") or 123
    )
    idle_memory._trim()
    assert calls == ["gc", "spares", "mlx", "host"]


def test_host_allocator_uses_all_zones_and_correct_abi(monkeypatch):
    import ctypes
    import sys
    from types import SimpleNamespace

    from yunshu_engine.idle_memory import _trim_host_allocator

    calls = []

    class Relief:
        def __call__(self, zone, goal):
            calls.append((zone, goal))
            return 42

    relief = Relief()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        ctypes, "CDLL", lambda path: SimpleNamespace(malloc_zone_pressure_relief=relief)
    )
    assert _trim_host_allocator() == 42
    assert calls == [(None, 0)]
    assert relief.argtypes == [ctypes.c_void_p, ctypes.c_size_t]
    assert relief.restype is ctypes.c_size_t


def _beating(busy_flag, beats, heartbeat_s=0.02, trim_s=0.1):
    return IdleMemory(
        lambda fn: fn(),
        lambda: busy_flag["busy"],
        collect_after_s=0.01,
        trim_after_s=trim_s,
        collect=lambda: None,
        trim=lambda: None,
        heartbeat_s=heartbeat_s,
        beat=lambda: beats.append(time.monotonic()),
        release=lambda: beats.append("release"),
    )


def test_heartbeat_keeps_the_gpu_warm_until_the_trim():
    beats, busy = [], {"busy": False}
    idle = _beating(busy, beats)
    idle.drained()
    assert _wait(lambda: idle.trims == 1)
    time.sleep(0.15)
    assert beats[-1] == "release"  # the trim gives the wired limit back
    # only beats before the trim (timer drift under load may drop the last one), then it stops
    assert 1 <= len(beats) - 1 == idle.beats <= 4


def test_request_stops_the_heartbeat():
    beats, busy = [], {"busy": False}
    idle = _beating(busy, beats, heartbeat_s=0.03, trim_s=1.0)
    idle.drained()
    assert _wait(lambda: len(beats) >= 2)
    idle.activity()
    n = len(beats)
    time.sleep(0.15)
    assert len(beats) == n


def test_heartbeat_skips_while_busy_and_restarts_on_the_next_drain():
    beats, busy = [], {"busy": True}
    idle = _beating(busy, beats)
    idle.drained()
    time.sleep(0.1)
    assert beats == []
    busy["busy"] = False
    idle.drained()
    assert _wait(lambda: len(beats) >= 2)


def test_heartbeat_disabled_with_zero_period():
    beats, busy = [], {"busy": False}
    idle = _beating(busy, beats, heartbeat_s=0.0)
    idle.drained()
    assert _wait(lambda: idle.trims == 1)
    time.sleep(0.05)
    assert beats == []  # nothing held, so nothing released
