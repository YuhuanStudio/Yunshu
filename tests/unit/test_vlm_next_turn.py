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
    monkeypatch.setattr(next_turn, "START_DELAY_S", 0.0)
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
    eng._apc_backend = type("B", (), {"manager": mgr})()

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
    eng._apc_backend = type("B", (), {"manager": mgr})()

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


def test_request_during_the_start_delay_prevents_the_rehearsal(monkeypatch):
    eng, calls = _engine(monkeypatch)
    monkeypatch.setattr(next_turn, "START_DELAY_S", 0.05)
    before = next_turn.snapshot()

    async def main():
        _go(eng)
        eng._cancel_rehearsal()  # a request arrives right after the reply
        await eng._rehearsal_task

    asyncio.run(main())
    assert calls == []
    assert next_turn.snapshot()["started"] == before["started"]
    assert next_turn.snapshot()["cancelled"] == before["cancelled"] + 1
