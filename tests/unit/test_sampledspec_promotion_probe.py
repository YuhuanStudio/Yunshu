"""CPU fixtures keep the promotion probe from accepting a serialized self-comparison."""

import copy
import json
from types import SimpleNamespace

import pytest
from scripts.research import sampledspec_promotion as probe


def _line(event):
    return b"data: " + json.dumps(event).encode() + b"\n"


def _stream():
    return [
        _line(
            {"choices": [{"logprobs": {"content": [{"token": "a", "logprob": -1.0}]}}]}
        ),
        _line(
            {
                "choices": [{"finish_reason": "length"}],
                "usage": {"completion_tokens": 1},
            }
        ),
        b"data: [DONE]\n",
    ]


def test_stream_requires_done_usage_and_per_token_logprobs():
    calls = []
    result = probe.read_stream(
        _stream(), clock=lambda: 3.0, first=SimpleNamespace(set=lambda: calls.append(1))
    )
    assert result["lps"] == [-1.0] and result["times"] == [3.0] and calls == [1]
    with pytest.raises(RuntimeError, match="incomplete"):
        probe.read_stream(_stream()[:-1])
    bad = _stream()
    bad[1] = _line(
        {"choices": [{"finish_reason": "length"}], "usage": {"completion_tokens": 2}}
    )
    with pytest.raises(RuntimeError, match="count"):
        probe.read_stream(bad)


def _results():
    alone = dict(
        ids=list(range(256)),
        lps=[-1.0] * 256,
        tops=[[]] * 256,
        times=[i / 100 for i in range(256)],
        finish="length",
        usage={"prompt_tokens": 1024, "prompt_tokens_details": {"cached_tokens": 0}},
    )
    overlap = copy.deepcopy(alone)
    overlap["times"] = [10 + i / 100 for i in range(256)]
    return alone, overlap, {"ended": 11.0}


def _compare(alone, overlap, blocker, log="Promoted request onto the speculative lane"):
    return probe.compare(alone, overlap, blocker, log, tokens=256, min_tail_ratio=0.9)


def test_identity_requires_overlap_actual_promotion_and_a_long_surviving_tail():
    alone, overlap, blocker = _results()
    result = _compare(alone, overlap, blocker)
    assert result["identity"] and result["tokens_after_blocker"] == 155
    assert result["tail_speed_ratio"] == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match="engagement"):
        _compare(alone, overlap, blocker, log="")
    with pytest.raises(RuntimeError, match="did not overlap"):
        _compare(alone, overlap, {"ended": 9.0})
    with pytest.raises(RuntimeError, match="fewer than 128"):
        _compare(alone, overlap, {"ended": 12.0})


@pytest.mark.parametrize("field", ["ids", "lps", "tops"])
def test_parity_failure_is_fatal(field):
    alone, overlap, blocker = _results()
    overlap[field][40] = "changed"
    with pytest.raises(RuntimeError, match=f"{field} differ"):
        _compare(alone, overlap, blocker)


def test_tail_speed_failure_is_not_hidden_by_whole_request_identity():
    alone, overlap, blocker = _results()
    overlap["times"] = [10 + i / 50 for i in range(256)]
    with pytest.raises(RuntimeError, match="speed ratio"):
        _compare(alone, overlap, blocker)


def test_fresh_capture_boundary_ignores_old_identical_requests(tmp_path):
    alone, _, _ = _results()
    alone["tokens"] = ["token"] * 256
    capture = tmp_path / "capture.jsonl"
    rows = [
        dict(prompt_tokens=1024, token_ids=[99] * 256),
        dict(prompt_tokens=1024, token_ids=list(range(256))),
    ]
    capture.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert probe.captured_ids(capture, 1, alone) == list(range(256))
    with pytest.raises(RuntimeError, match="ambiguous"):
        probe.captured_ids(capture, 0, alone)


def test_request_normalizes_string_prompt_and_retains_sampling(tmp_path, monkeypatch):
    bodies = []

    class Response:
        def __enter__(self):
            return iter(_stream())

        def __exit__(self, *args):
            pass

    def open_request(request, timeout):
        bodies.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(probe.urllib.request, "urlopen", open_request)
    probe.ask(
        SimpleNamespace(url="http://fixture", model="fixture"),
        "hello",
        1,
        probe.SAMPLING[1],
    )
    assert bodies[0]["messages"] == [{"role": "user", "content": "hello"}]
    assert bodies[0]["seed"] == 17 and bodies[0]["top_k"] == 20
