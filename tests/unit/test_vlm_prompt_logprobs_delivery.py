"""Driver prompt reports cross runner/engine delivery without another model forward."""

import queue
from types import SimpleNamespace

from yunshu_engine.vlm_engine import VLMEngine

SCORES = [None, {"token_id": 2, "logprob": -1.0}]


def engine():
    def events(ids, *, stats, **kwargs):
        assert kwargs["prompt_logprobs"] == 0
        stats.prompt_logprobs = SCORES
        stats.prompt_tokens = len(ids)
        stats.generated = 2
        yield "a", 3, "normal", None, 0, None
        yield "b", 4, "normal", "length", 0, None

    return SimpleNamespace(_runner_events=events)


def test_nonstream_prompt_reports_in_extras():
    extras = {}
    result = VLMEngine._generate_vlm_runner_text(
        engine(), [1, 2], extras, prompt_logprobs=0
    )
    assert result[0] == "ab"
    assert extras["prompt_logprobs"] == SCORES


def test_stream_prompt_reports_once():
    outputs = queue.Queue()
    VLMEngine._stream_vlm_runner_text(engine(), [1, 2], "r", outputs, prompt_logprobs=0)
    first, second = outputs.get_nowait(), outputs.get_nowait()
    assert first.prompt_logprobs == SCORES
    assert second.prompt_logprobs is None
    assert outputs.empty()
