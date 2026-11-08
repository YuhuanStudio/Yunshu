"""Captured fingerprints and lossless priority scheduling regressions."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.unit.test_vlm_runner_batching import FakeDriver
from tests.unit.test_vlm_runner_batching import runner as _runner  # noqa: F401
from yunshu_engine import vlm_batch_runner as vbr
from yunshu_gateway import admission


@pytest.fixture
def title():
    return json.loads(
        (
            Path(__file__).parents[1] / "fixtures/auxiliary/opencode_title.json"
        ).read_text()
    )


def test_captured_title(title):
    assert admission.auxiliary_kind(title) == "opencode_title"


@pytest.mark.parametrize("change", ["tools", "system", "role", "model", "budget"])
def test_near_matches_stay_interactive(title, change):
    if change == "tools":
        title["tools"] = [{"type": "function"}]
    elif change == "system":
        title["messages"][0]["content"] += " Answer the question too."
    elif change == "role":
        title["messages"][-1]["role"] = "assistant"
    elif change == "model":
        title["model"] = "claude-haiku"
    else:
        title["max_tokens"] = 1
    assert admission.auxiliary_kind(title) is None


@pytest.fixture
def runner(_runner):  # noqa: F811
    """A runner whose primary and auxiliary drivers are fakes (auxiliary rows never join the primary)."""
    _runner._aux_driver = FakeDriver()
    return _runner


def job(priority):
    request = vbr._Job(
        ids=[1, 2],
        max_tokens=2,
        greedy=True,
        sampling=None,
        use_draft=False,
        logprobs=False,
        top_logprobs=0,
        processors=[],
        prompt_kwargs={},
        salt=None,
        seed=None,
        cancel_event=None,
        stats=vbr.RunStats(),
    )
    request.priority = priority
    return request


def test_pending_primary_precedes_auxiliary_without_dropping(runner):
    aux, main = job(-1), job(0)
    runner._submit(aux)
    runner._submit(main)
    runner._drive_slice(resubmit=False)
    assert main.stats.generated == 1
    assert aux.stats.generated == 0
    assert aux in runner._pending
    runner._drive_slice(resubmit=False)
    assert main.stats.generated == 2
    runner._drive_slice(resubmit=False)
    assert aux.stats.generated == 1
    runner._drive_slice(resubmit=False)
    assert aux.stats.generated == 2
    assert not runner.busy()


def test_active_auxiliary_pauses_and_resumes_exact_state(runner):
    aux = job(-1)
    runner._submit(aux)
    runner._drive_slice(resubmit=False)
    assert aux.stats.generated == 1
    main = job(0)
    runner._submit(main)
    runner._drive_slice(resubmit=False)
    assert main.stats.generated == 1
    assert aux.stats.generated == 1
    runner._drive_slice(resubmit=False)
    runner._drive_slice(resubmit=False)
    assert aux.stats.generated == 2
    assert [aux.out.get()[0], aux.out.get()[0]] == [1, 2]


def test_auxiliary_cannot_touch_apc(runner):
    runner.apc_manager = SimpleNamespace()
    aux = job(-1)
    runner._submit(aux)
    runner._drive_slice(resubmit=False)
    assert not aux.stats.used_apc
    assert runner._aux_driver.requests[0].use_apc is False


def test_paused_auxiliary_cancel_releases_row(runner):
    import threading

    aux = job(-1)
    aux.cancel_event = threading.Event()
    runner._submit(aux)
    runner._drive_slice(resubmit=False)
    main = job(0)
    runner._submit(main)
    aux.cancel_event.set()
    runner._drive_slice(resubmit=False)
    assert aux.stats.finish_reason == "cancel"
    assert aux.stats.generated == 1
    assert len(runner._aux_driver.removed) == 1
    assert main.stats.generated == 1


def test_round_driver_auxiliary_never_joins_primary_driver(runner, monkeypatch):

    class Driver:
        head = None
        can_draft = False
        chunk = 2048

        def __init__(self):
            self.rows = []
            self.steps = 0

        def add(self, req):
            self.rows.append(SimpleNamespace(req=req, done=0, hit=0))
            return 0

        def step(self):
            self.steps += bool(self.rows)
            return []

        def remove(self, handle):
            self.rows = [r for r in self.rows if r.req.handle is not handle]

    runner.driver = Driver()
    runner._aux_driver = Driver()
    aux = job(-1)
    aux.prompt_kwargs = None
    runner._submit(aux)
    runner._drive_slice(resubmit=False)
    assert runner._aux_driver.steps == 1
    main = job(0)
    main.prompt_kwargs = None
    runner._submit(main)
    runner._drive_slice(resubmit=False)
    assert runner._aux_driver.steps == 1
    assert runner.driver.steps == 1
    assert runner.driver.rows[0].req.handle is main
    assert not runner._aux_driver.rows[0].req.use_apc
    main.abandoned = True
    runner._drive_slice(resubmit=False)
    runner._drive_slice(resubmit=False)
    assert runner._aux_driver.steps == 2


def test_tracker_transfers_priority_to_worker_event():
    from yunshu_engine.request_tracker import RequestTracker, current_request_info
    from yunshu_gateway.x_yunshu import RequestInfo

    info = RequestInfo("i8", "POST", "/v1/chat/completions", scheduling_priority=-1)
    token = current_request_info.set(info)
    try:
        generation = RequestTracker().register("i8-generation", "m")
        assert generation.priority == -1
        assert generation.cancel_event.scheduling_priority == -1
    finally:
        current_request_info.reset(token)


def test_opt_out_and_unknown_requests_keep_normal_priority(title, monkeypatch):
    from yunshu_engine.request_tracker import current_request_info
    from yunshu_gateway.admission import classify_request
    from yunshu_gateway.x_yunshu import RequestInfo

    info = RequestInfo("i8", "POST", "/v1/chat/completions")
    token = current_request_info.set(info)
    try:
        monkeypatch.setenv("YUNSHU_AUXILIARY_SCHEDULING", "0")
        classify_request(title)
        assert info.scheduling_priority == 0
        monkeypatch.setenv("YUNSHU_AUXILIARY_SCHEDULING", "1")
        classify_request(title)
        assert info.scheduling_priority == -1
        assert info.auxiliary_kind == "opencode_title"
        classify_request(
            {"messages": [{"role": "user", "content": "Generate a title"}]}
        )
        assert info.scheduling_priority == 0
    finally:
        current_request_info.reset(token)


def test_pending_auxiliary_cancel_does_not_wait_for_primary(runner):
    import threading

    aux, main = job(-1), job(0)
    aux.cancel_event = threading.Event()
    runner._submit(aux)
    runner._submit(main)
    runner._drive_slice(resubmit=False)
    aux.cancel_event.set()
    runner._drive_slice(resubmit=False)
    assert aux.stats.finish_reason == "cancel"
    assert aux.stats.generated == 0
    assert aux not in runner._pending
    assert aux.out.get_nowait() is vbr._DONE


@pytest.mark.asyncio
async def test_gateway_grace_then_waits_for_interactive_idle(monkeypatch):
    from yunshu_engine.request_tracker import current_request_info
    from yunshu_gateway.x_yunshu import RequestInfo, registry

    auxiliary = RequestInfo(
        "aux", "POST", "/v1/chat/completions", scheduling_priority=-1
    )
    primary = RequestInfo("main", "POST", "/v1/chat/completions")
    sleeps = []
    active = [auxiliary, primary]

    async def sleep(duration):
        sleeps.append(duration)
        if duration == 0.01:
            active.remove(primary)

    monkeypatch.setattr("asyncio.sleep", sleep)
    monkeypatch.setattr(registry, "active", lambda: list(active))
    token = current_request_info.set(auxiliary)
    try:
        await admission.defer_auxiliary()
        assert sleeps == [0.5, 0.01]
    finally:
        current_request_info.reset(token)


def test_auxiliary_does_not_train_apc_cost_model(runner):
    observed = []
    runner.apc_manager = SimpleNamespace(
        disk=SimpleNamespace(observe_prefill=lambda *args: observed.append(args))
    )
    aux = job(-1)
    aux.stats.t_admit, aux.stats.t_first = 1, 2
    runner._observe_prefill(aux)
    assert observed == []
    main = job(0)
    main.stats.t_admit, main.stats.t_first = 1, 2
    runner._observe_prefill(main)
    assert observed == [(2, 1)]


def test_queue_position_does_not_put_agent_behind_title(monkeypatch):
    from yunshu_gateway.x_yunshu import RequestInfo, queue_snapshot, registry

    aux = RequestInfo(
        "aux", "POST", "/v1/chat/completions", arrived=1, scheduling_priority=-1
    )
    older = RequestInfo("older", "POST", "/v1/chat/completions", arrived=0)
    newer = RequestInfo("newer", "POST", "/v1/chat/completions", arrived=2)
    monkeypatch.setattr(registry, "active", lambda: [aux, older, newer])
    assert queue_snapshot(newer)[0] == 1
    assert queue_snapshot(older)[0] == 0
    assert queue_snapshot(aux)[0] == 2


def test_auxiliary_waits_for_request_still_preparing(runner):
    runner.inflight = lambda: (
        2
    )  # auxiliary plus a primary not submitted to the runner yet
    aux = job(-1)
    runner._submit(aux)
    runner._drive_slice(resubmit=False)
    assert aux.stats.t_admit == 0
    assert aux in runner._pending
    runner.inflight = lambda: 1  # the other request left
    runner._drive_slice(resubmit=False)
    assert aux.stats.generated == 1
    runner._drive_slice(resubmit=False)
    assert aux.stats.generated == 2


def test_fingerprint_survives_real_gateway_schema(title):
    from yunshu_gateway.routers.chat import ChatCompletionRequest

    validated = ChatCompletionRequest.model_validate(title)
    assert admission.auxiliary_kind(validated.model_dump()) == "opencode_title"


@pytest.mark.parametrize("qualified", [False, True])
def test_auxiliary_default_uses_loaded_backend(title, monkeypatch, qualified):
    from yunshu_engine.request_tracker import current_request_info
    from yunshu_gateway import engine
    from yunshu_gateway.x_yunshu import RequestInfo

    monkeypatch.delenv("YUNSHU_AUXILIARY_SCHEDULING", raising=False)
    monkeypatch.setattr("yunshu_engine.settings._file", lambda: {})
    monkeypatch.setattr(engine, "get_model_manager", lambda: None)
    monkeypatch.setattr(
        engine,
        "get_engine",
        lambda: SimpleNamespace(_prefix_invariant_dispatch=qualified),
    )
    info = RequestInfo("i8", "POST", "/v1/chat/completions")
    token = current_request_info.set(info)
    try:
        admission.classify_request(title)
        assert info.scheduling_priority == (-1 if qualified else 0)
    finally:
        current_request_info.reset(token)


@pytest.mark.parametrize("qualified", [False, True])
def test_auxiliary_default_resolves_loaded_model_alias(title, monkeypatch, qualified):
    from yunshu_engine.request_tracker import current_request_info
    from yunshu_gateway import engine
    from yunshu_gateway.x_yunshu import RequestInfo

    monkeypatch.delenv("YUNSHU_AUXILIARY_SCHEDULING", raising=False)
    monkeypatch.setattr("yunshu_engine.settings._file", lambda: {})
    canonical = "Jundot/" + title["model"]
    loaded = SimpleNamespace(
        engine=SimpleNamespace(_prefix_invariant_dispatch=qualified)
    )
    manager = SimpleNamespace(
        resolve_model_id=lambda name: canonical if name == title["model"] else None,
        get_entry=lambda name: loaded if name == canonical else None,
    )
    monkeypatch.setattr(engine, "get_engine", lambda: None)
    monkeypatch.setattr(engine, "get_model_manager", lambda: manager)
    info = RequestInfo("i8", "POST", "/v1/chat/completions")
    token = current_request_info.set(info)
    try:
        admission.classify_request(title)
        assert info.scheduling_priority == (-1 if qualified else 0)
    finally:
        current_request_info.reset(token)
