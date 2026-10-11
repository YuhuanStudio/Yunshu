"""A user stop string on the VLM runner path sets stopped_by_stop_sequence (Anthropic stop_reason)."""

from types import SimpleNamespace

from yunshu_engine import vlm_engine as ve
from yunshu_engine.vlm_batch_runner import RunStats


class Q:
    def __init__(self):
        self.items = []

    def put_nowait(self, o):
        self.items.append(o)


def fake_self(hit):
    def events(input_ids, stats, **kw):
        stats.generated = 2
        if hit:
            stats.stop_string_hit = True
        yield "1 2 ", 5, "normal", "stop", 0, None

    return SimpleNamespace(_runner_events=events)


def run(hit):
    q = Q()
    ve.VLMEngine._stream_vlm_runner_text(fake_self(hit), [1, 2, 3], "r", q)
    return q.items[-1]


def test_stop_string_sets_flag():
    out = run(True)
    assert out.finish_reason == "stop" and out.stopped_by_stop_sequence is True


def test_eos_leaves_flag_false():
    out = run(False)
    assert out.finish_reason == "stop" and out.stopped_by_stop_sequence is False
    assert RunStats().stop_string_hit is False
