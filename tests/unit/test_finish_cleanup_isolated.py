"""A finish-path cleanup step that raises must not strand the client without a terminal event
(mlx2 3dc6f0f): the runner emits its _DONE sentinel regardless, and the gateway's LoRA release in
the response `finally` never turns a finished response into an error."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from yunshu_engine import vlm_batch_runner as vbr
from yunshu_gateway.routers.chat import _release_lora_adapter


def _runner(**boom):
    r = vbr.VLMBatchRunner.__new__(vbr.VLMBatchRunner)
    r.prefix_invariant = False
    r.finish_cleanup_failures = {}
    emitted = []
    r._emit = lambda job, item: emitted.append(item)

    def observe(job):
        if "observe" in boom:
            raise RuntimeError("observe")

    r._observe_prefill = observe
    return r, emitted


def _group(job, spec=False):
    return SimpleNamespace(
        jobs={1: job}, sampler=None, spec=spec, gen=SimpleNamespace(apc=None)
    )


def _job():
    return SimpleNamespace(
        stats=SimpleNamespace(finish_reason=None, request_id="r1"),
        priority=0,
        abandoned=False,
        ids=[1, 2],
        cache_plan=None,
    )


def test_terminal_is_emitted_when_observe_prefill_raises(caplog):
    r, emitted = _runner(observe=True)
    job = _job()
    with caplog.at_level(logging.ERROR):
        r._finish(_group(job), 1, "stop")
    assert emitted == [vbr._DONE]
    assert job.stats.finish_reason == "stop"
    assert r.finish_cleanup_failures == {"observe_prefill": 1}


def test_terminal_is_emitted_when_apc_release_raises(monkeypatch):
    r, emitted = _runner()

    def boom(group, job):
        raise RuntimeError("apc release")

    monkeypatch.setattr(vbr.VLMBatchRunner, "_release_apc_request", staticmethod(boom))
    r._finish(_group(_job()), 1, None)
    assert emitted == [vbr._DONE]
    assert r.finish_cleanup_failures == {"release_apc_request": 1}


def test_every_step_still_runs_after_an_earlier_one_raised(monkeypatch):
    r, emitted = _runner(observe=True)
    ran = []
    monkeypatch.setattr(
        vbr.VLMBatchRunner,
        "_release_apc_request",
        staticmethod(lambda group, job: ran.append("apc")),
    )
    r._finish(_group(_job()), 1, None)
    assert ran == ["apc"] and emitted == [vbr._DONE]


def test_lora_release_failure_never_reaches_the_response(caplog):
    class Mgr:
        def release_adapter(self, adapter):
            raise RuntimeError("not loaded")

    engine = SimpleNamespace(get_lora_manager=lambda: Mgr())
    with caplog.at_level(logging.WARNING):
        _release_lora_adapter(engine, "adapter-1")  # must not raise
    assert "release failed" in caplog.text


def test_engine_loop_finish_step_isolates_a_raising_cleanup(caplog):
    from yunshu_engine.scheduler import Scheduler

    s = Scheduler.__new__(Scheduler)
    s.finish_cleanup_failures = {}

    def boom(*a, **k):
        raise RuntimeError("tracker gone")

    with caplog.at_level(logging.ERROR):
        s._finish_step("r1", "server_metrics", boom, 1, x=2)  # must not raise
    s._finish_step("r1", "server_metrics", lambda: None)
    assert s.finish_cleanup_failures == {"server_metrics": 1}
