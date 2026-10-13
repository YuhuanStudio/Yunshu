import asyncio
import concurrent.futures
import threading

import pytest

from yunshu_engine import next_turn, settings, vlm_engine

IM = 9
VOCAB = {
    9: "<|im_start|>",
    1: "user",
    2: "assistant",
    3: "\n",
    4: "A",
    5: "B",
    6: next_turn.SENTINEL,
}


class _Tok:
    def decode(self, ids):
        return "".join(VOCAB[i] for i in ids)


class _Arr:
    def __init__(self, ids):
        self._ids = ids

    def tolist(self):
        return list(self._ids)


def _engine(monkeypatch, *, on=True):
    from yunshu_engine.kernels import preempt

    monkeypatch.setattr(preempt, "supported", lambda: True)
    pass
    eng = object.__new__(vlm_engine.VLMEngine)
    eng._rehearsal_cancel = None
    eng._rehearsal_task = None
    eng._active_count = 1
    eng._batch_runner = object()
    eng._apc_backend = None
    eng._apc_prefill_stride = 0
    eng._tokenizer = _Tok()
    eng._executor = concurrent.futures.ThreadPoolExecutor(1)
    pool = concurrent.futures.ThreadPoolExecutor(1)
    eng._runner_consumers = lambda: pool
    eng._chatml_head_marker = lambda: (IM, 1)
    eng._tokenize_with_cache = lambda messages, **kw: _Arr(
        [9, 1, 3, 4, 9, 2, 3, 5, 9, 1, 3, 6, 9, 2, 3]
    )
    eng._runner_kwargs = lambda **kw: {
        "cancel_event": kw["cancel_event"],
        "max_tokens": kw["max_tokens"],
    }
    eng._resolve_prompt_cache_plan = lambda *a, **k: None
    calls = []

    def runner(ids, **kw):
        calls.append((list(ids.tolist() if hasattr(ids, "tolist") else ids), kw))
        return ("", 0, 1, True, False, 0)

    eng._generate_vlm_runner_text = runner
    monkeypatch.setattr(
        settings,
        "get",
        lambda k, *a, **kw: (
            ("stride" if on is True else on or "off")
            if k == "YUNSHU_NEXT_TURN_PREFILL"
            else None
        ),
    )
    return eng, calls


def _go(eng, **over):
    args = dict(
        finish_reason="stop",
        completion_tokens=100,
        enable_thinking=None,
        tpl_extra=None,
        has_media=False,
        kwargs={},
    )
    args.update(over)
    eng._maybe_rehearse([{"role": "user", "content": "q"}], "reply", **args)


def test_rehearsal_prefills_prefix_before_the_next_user_turn(monkeypatch):
    eng, calls = _engine(monkeypatch)

    async def main():
        _go(eng)
        await eng._rehearsal_task

    asyncio.run(main())
    assert len(calls) == 1
    ids, kw = calls[0]
    assert ids == [9, 1, 3, 4, 9, 2, 3, 5]
    assert kw["max_tokens"] == 1


@pytest.mark.parametrize(
    "over",
    [
        {"finish_reason": "error"},
        {"has_media": True},
        {"kwargs": {"tools": [1]}},
        {"completion_tokens": 2},
    ],
)
def test_ineligible_requests_do_not_rehearse(monkeypatch, over):
    eng, calls = _engine(monkeypatch)

    async def main():
        _go(eng, **over)
        assert eng._rehearsal_task is None

    asyncio.run(main())
    assert calls == []


def test_flag_off_and_busy_engine_do_not_rehearse(monkeypatch):
    eng, calls = _engine(monkeypatch, on=False)

    async def main():
        _go(eng)
        assert eng._rehearsal_task is None
        eng2, _ = _engine(monkeypatch)
        eng2._active_count = 2
        _go(eng2)
        assert eng2._rehearsal_task is None

    asyncio.run(main())


def test_new_request_cancels_the_rehearsal(monkeypatch):
    eng, calls = _engine(monkeypatch)
    ev = threading.Event()
    eng._rehearsal_cancel = ev
    eng._cancel_rehearsal()
    assert ev.is_set()


def test_counters_record_started_and_completed_tokens(monkeypatch):
    eng, calls = _engine(monkeypatch)
    before = next_turn.snapshot()

    def runner(ids, extras=None, **kw):
        extras["prompt_tokens"] = 8
        return ("", 0, 1, True, False, 0)

    eng._generate_vlm_runner_text = runner

    async def main():
        _go(eng)
        await eng._rehearsal_task

    asyncio.run(main())
    after = next_turn.snapshot()
    assert after["started"] == before["started"] + 1
    assert after["completed"] == before["completed"] + 1
    assert after["tokens"] == before["tokens"] + 8


def test_cancelled_rehearsal_is_counted(monkeypatch):
    eng, calls = _engine(monkeypatch)
    before = next_turn.snapshot()

    def runner(ids, extras=None, cancel_event=None, **kw):
        cancel_event.set()
        raise asyncio.CancelledError()

    eng._generate_vlm_runner_text = runner

    async def main():
        _go(eng)
        await eng._rehearsal_task

    asyncio.run(main())
    assert next_turn.snapshot()["cancelled"] == before["cancelled"] + 1
    assert next_turn.snapshot()["completed"] == before["completed"]


def test_ineligible_is_counted_with_reason(monkeypatch, caplog):
    eng, _ = _engine(monkeypatch)
    before = next_turn.snapshot()
    import logging

    with caplog.at_level(logging.DEBUG):

        async def main():
            _go(eng, completion_tokens=2)

        asyncio.run(main())
    assert next_turn.snapshot()["ineligible"] == before["ineligible"] + 1
    assert "too short" in caplog.text


def test_metrics_endpoint_exposes_the_counters():
    from yunshu_gateway.middleware.prometheus_exporter import get_prometheus_metrics

    next_turn.record("completed", 5)
    text = get_prometheus_metrics().generate()
    assert "yunshu_next_turn_rehearsals_total" in text
    assert "yunshu_next_turn_rehearsed_tokens_total" in text


def test_rehearsal_stops_at_the_stride_grid(monkeypatch):
    eng, calls = _engine(monkeypatch)
    eng._apc_prefill_stride = 4
    mgr = type("M", (), {"exact_cache_guard_tokens": 1})()
    eng._apc_backend = mgr

    async def main():
        _go(eng)
        await eng._rehearsal_task

    asyncio.run(main())
    ids, _ = calls[0]
    # prefix is 8 tokens; stride 4 -> S = 8 would pass the prefix with the guard, so S = 4
    assert ids == [9, 1, 3, 4, 9]


def test_exact_mode_rehearses_through_the_next_turn_header(monkeypatch):
    eng, calls = _engine(monkeypatch, on="exact")
    eng._apc_prefill_stride = 2048
    mgr = type("M", (), {"exact_cache_guard_tokens": 1})()
    eng._apc_backend = mgr

    async def main():
        _go(eng)
        await eng._rehearsal_task

    asyncio.run(main())
    ids, _ = calls[0]
    # prefix is 8 tokens (ends before the sentinel user header); exact adds the header token
    assert ids == [9, 1, 3, 4, 9, 2, 3, 5, 9]


def test_boundary_mode_never_rehearses(monkeypatch):
    eng, calls = _engine(monkeypatch, on="boundary")

    async def main():
        _go(eng)
        assert eng._rehearsal_task is None

    asyncio.run(main())
    assert calls == []


def _delay(monkeypatch, seconds):
    real = settings.get
    monkeypatch.setattr(
        settings,
        "get",
        lambda k, *a, **kw: (
            seconds if k == "YUNSHU_NEXT_TURN_DELAY_S" else real(k, *a, **kw)
        ),
    )


def test_request_during_the_start_delay_prevents_the_rehearsal(monkeypatch):
    eng, calls = _engine(monkeypatch)
    _delay(monkeypatch, 0.05)
    before = next_turn.snapshot()

    async def main():
        _go(eng)
        await asyncio.sleep(0.01)  # the rehearsal ids are known, the delay is running
        # even the follow-up itself arrives before any work started: nothing to adopt yet
        eng._settle_rehearsal([9, 1, 3, 4, 9, 2, 3, 5, 9, 1])
        await eng._rehearsal_task

    asyncio.run(main())
    assert calls == []
    assert next_turn.snapshot()["started"] == before["started"]
    assert next_turn.snapshot()["cancelled"] == before["cancelled"] + 1


def test_follow_up_adopts_a_running_rehearsal_and_a_diverging_request_cancels(
    monkeypatch,
):
    eng, calls = _engine(monkeypatch)
    release = threading.Event()
    seen = {}

    def runner(ids, extras=None, cancel_event=None, **kw):
        seen["ev"] = cancel_event
        release.wait(2)
        return ("", 0, 1, True, False, 0)

    eng._generate_vlm_runner_text = runner

    async def main():
        _go(eng)
        for _ in range(200):
            if "ev" in seen:
                break
            await asyncio.sleep(0.005)
        prefix = [9, 1, 3, 4, 9, 2, 3, 5]
        eng._settle_rehearsal(
            prefix + [9, 1, 3, 6]
        )  # same conversation, new user message
        assert not seen["ev"].is_set()  # adopted: the rehearsal keeps running
        eng._settle_rehearsal([9, 1, 3, 4, 9, 7])  # another prompt
        assert seen["ev"].is_set()  # cancelled at the next atom
        release.set()
        await eng._rehearsal_task

    asyncio.run(main())


def test_extends_and_state_rules():
    assert next_turn.extends([1, 2, 3, 4], [1, 2, 3])
    assert not next_turn.extends([1, 2], [1, 2, 3])
    assert not next_turn.extends([1, 9, 3, 4], [1, 2, 3])
    assert not next_turn.extends([1, 2, 3], [])
    ev = threading.Event()
    st = next_turn.RehearsalState(ev)
    st.set_ids([1, 2, 3])
    assert st.arrive([1, 2, 3, 4]) == "cancel"  # not submitted yet: no work to adopt
    assert ev.is_set() and not st.submit()


def test_unsupported_family_never_rehearses(monkeypatch):
    from yunshu_engine.kernels import preempt

    eng, calls = _engine(monkeypatch)
    monkeypatch.setattr(preempt, "supported", lambda: False)

    async def main():
        _go(eng)
        assert eng._rehearsal_task is None

    asyncio.run(main())
    assert calls == []


def test_rehearsal_cancel_event_is_preemptible(monkeypatch):
    eng, calls = _engine(monkeypatch)

    async def main():
        _go(eng)
        assert eng._rehearsal_cancel.preemptible is True
        await eng._rehearsal_task

    asyncio.run(main())


def _early(eng, messages, ids=None, fail=False):
    def peek(m, t, e):
        if fail:
            raise RuntimeError("tokenizer")
        return ids

    eng._peek_prompt_ids = peek
    asyncio.run(eng._settle_rehearsal_early(messages, None, None))


def test_early_settle_cancels_a_diverging_request_before_the_executor(monkeypatch):
    eng, _ = _engine(monkeypatch)
    ev = threading.Event()
    st = next_turn.RehearsalState(ev)
    st.set_ids([9, 1, 3])
    st.submit()
    eng._rehearsal_state = st
    _early(eng, [{"role": "user", "content": "other"}], ids=[9, 2, 2])
    assert ev.is_set()


def test_early_settle_adopts_the_follow_up(monkeypatch):
    eng, _ = _engine(monkeypatch)
    ev = threading.Event()
    st = next_turn.RehearsalState(ev)
    st.set_ids([9, 1, 3])
    st.submit()
    eng._rehearsal_state = st
    _early(eng, [{"role": "user", "content": "q"}], ids=[9, 1, 3, 5, 6])
    assert not ev.is_set()


def test_early_settle_fails_closed_on_media_and_peek_errors(monkeypatch):
    for kw, msgs in (
        (
            {"ids": [9, 1, 3, 5]},
            [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
        ),
        ({"fail": True}, [{"role": "user", "content": "q"}]),
    ):
        eng, _ = _engine(monkeypatch)
        ev = threading.Event()
        st = next_turn.RehearsalState(ev)
        st.set_ids([9, 1, 3])
        st.submit()
        eng._rehearsal_state = st
        _early(eng, msgs, **kw)
        assert ev.is_set()


def test_early_settle_is_free_without_a_rehearsal(monkeypatch):
    eng, _ = _engine(monkeypatch)
    eng._rehearsal_state = None

    def boom(*a):
        raise AssertionError("must not tokenize when no rehearsal runs")

    eng._peek_prompt_ids = boom
    asyncio.run(
        eng._settle_rehearsal_early([{"role": "user", "content": "q"}], None, None)
    )


def test_early_settle_does_not_tokenize_before_the_rehearsal_started(monkeypatch):
    eng, _ = _engine(monkeypatch)
    ev = threading.Event()
    st = next_turn.RehearsalState(ev)
    st.set_ids([9, 1, 3])  # not submitted: still inside the start delay
    eng._rehearsal_state = st

    def boom(*a):
        raise AssertionError("no token peek while nothing runs (costs a warm hit 7 ms)")

    eng._peek_prompt_ids = boom
    asyncio.run(
        eng._settle_rehearsal_early([{"role": "user", "content": "q"}], None, None)
    )
    assert ev.is_set()


def test_nothing_is_tokenized_during_the_start_delay(monkeypatch):
    eng, calls = _engine(monkeypatch)
    _delay(monkeypatch, 0.05)
    seen = []
    real = eng._tokenize_with_cache
    eng._tokenize_with_cache = lambda *a, **k: (seen.append(1), real(*a, **k))[1]

    async def main():
        _go(eng)
        await asyncio.sleep(0.01)
        assert (
            seen == []
        )  # a back-to-back request must not queue behind our tokenization
        eng._cancel_rehearsal()
        await eng._rehearsal_task

    asyncio.run(main())
    assert seen == [] and calls == []


def test_rehearsal_registers_its_ids_with_the_real_manager_object(monkeypatch):
    """The guard never engaged in the 27B gate because the engine looked for backend.manager;
    the backend IS the manager."""
    eng, calls = _engine(monkeypatch)
    mgr = type("M", (), {"exact_cache_guard_tokens": 1, "keep_prior_for": None})()
    eng._apc_backend = mgr

    async def main():
        _go(eng)
        await eng._rehearsal_task

    asyncio.run(main())
    assert mgr.keep_prior_for == (9, 1, 3, 4, 9, 2, 3, 5)


def test_default_rehearsal_uses_canonical_checkpoint_not_assistant_end(monkeypatch):
    # Exercise the shipped default, rather than the helper's forced stride setting.
    default = settings.REGISTRY["YUNSHU_NEXT_TURN_PREFILL"].default
    eng, calls = _engine(monkeypatch, on=default)
    eng._apc_prefill_stride = 4
    eng._apc_backend = type("M", (), {"exact_cache_guard_tokens": 1})()

    async def main():
        _go(eng)
        await eng._rehearsal_task

    asyncio.run(main())
    # The 8-token assistant end is not a canonical reusable checkpoint: the
    # guard must fit inside the known prefix. Rehearse through 4 + one guard.
    assert calls[0][0] == [9, 1, 3, 4, 9]
