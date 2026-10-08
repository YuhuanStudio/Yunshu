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
    eng = object.__new__(vlm_engine.VLMEngine)
    eng._rehearsal_cancel = None
    eng._rehearsal_task = None
    eng._active_count = 1
    eng._batch_runner = object()
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
        lambda k, *a, **kw: on if k == "YUNSHU_NEXT_TURN_PREFILL" else None,
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
        {"finish_reason": "length"},
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
