"""--log-usage-sample must capture the first real benchmark request, not a warm-up probe (CPU only)."""

import json
from contextlib import contextmanager

import pytest
from scripts.research import tfbench as tb


class _Stream:
    def __init__(self, chunks):
        self._lines = [("data: " + json.dumps(c)).encode() for c in chunks] + [
            b"data: [DONE]"
        ]

    def __iter__(self):
        return iter(self._lines)


def _chunks(prompt_tokens):
    text = {"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}
    final = {
        "choices": [{"delta": {}, "finish_reason": "length"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 1,
            "total_tokens": prompt_tokens + 1,
        },
    }
    return [text, final]


@pytest.fixture
def fake_urlopen(monkeypatch):
    @contextmanager
    def _open(_req, timeout=None):
        yield _Stream(_chunks(15))

    monkeypatch.setattr(tb.urllib.request, "urlopen", _open)
    tb._USAGE_SAMPLE[0] = None
    yield
    tb._USAGE_SAMPLE[0] = None


def test_warmup_request_does_not_fill_usage_sample(fake_urlopen):
    tb.send(
        "http://127.0.0.1:1/v1/chat/completions",
        {"messages": [{"role": "user", "content": "Say hi."}]},
    )
    assert tb._USAGE_SAMPLE[0] is None


def test_benchmark_request_fills_usage_sample_with_final_usage(fake_urlopen):
    tb.send(
        "http://127.0.0.1:1/v1/chat/completions",
        {"messages": [{"role": "user", "content": "bench"}]},
        sample=True,
    )
    sample = tb._USAGE_SAMPLE[0]
    assert sample is not None
    assert sample["usage"]["prompt_tokens"] == 15
