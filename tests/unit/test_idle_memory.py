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
