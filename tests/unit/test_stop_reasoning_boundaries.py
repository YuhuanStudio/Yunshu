"""R22: stop strings and the reasoning split across token boundaries.

Property tests against the plain ``str.find`` semantics, plus tool-call JSON that
mentions the reasoning tags.
"""

from __future__ import annotations

import random

import pytest

from yunshu_engine.reasoning_parser import get_reasoning_parser
from yunshu_engine.text_utils import StopHoldbackBuffer

ALPHABET = [
    "a",
    "b",
    "\n",
    " ",
    "é",
    "龍",
    "😀",
    "<",
    "/",
    "s",
    "t",
    "o",
    "p",
    "S",
    "T",
]


def _stream(stops, text, cuts):
    buf = StopHoldbackBuffer(stops)
    out = []
    pos = 0
    for cut in cuts + [len(text)]:
        piece = text[pos:cut]
        pos = cut
        out.append(buf.feed(piece))
        if buf.contains_stop():
            return "".join(out) + buf.take_stopped()
    return "".join(out) + buf.flush()


def test_stop_holdback_equals_find_for_random_chunkings():
    rng = random.Random(1234)
    stops_pool = ["stop", "\n\n", "龍", "<s>", "ab", "😀😀", "oS", "t\n"]
    for _ in range(4000):
        stops = rng.sample(stops_pool, rng.randint(1, 3))
        text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 24)))
        cuts = sorted(
            rng.sample(range(len(text) + 1), rng.randint(0, min(len(text), 6)))
        )
        first = min((i for i in (text.find(s) for s in stops) if i >= 0), default=-1)
        want = text if first < 0 else text[:first]
        assert _stream(stops, text, cuts) == want, (stops, text, cuts)


def test_stop_holdback_logprob_variant_matches_plain():
    rng = random.Random(99)
    for _ in range(1500):
        stops = rng.sample(["stop", "\n\n", "龍", "ab"], rng.randint(1, 2))
        text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 20)))
        cuts = sorted(
            rng.sample(range(len(text) + 1), rng.randint(0, min(len(text), 5)))
        )
        plain = _stream(stops, text, cuts)
        buf = StopHoldbackBuffer(stops)
        out, pos = [], 0
        for cut in cuts + [len(text)]:
            for chunk, _lp in buf.feed_lp(text[pos:cut], object()):
                out.append(chunk)
            pos = cut
            if buf.contains_stop():
                out.append(buf.take_stopped())
                break
        else:
            out.append(buf.flush())
        assert "".join(out) == plain, (stops, text, cuts)


QWEN = "Qwen3.5-27B"


def test_think_tag_inside_tool_json_is_not_a_reasoning_boundary():
    parser = get_reasoning_parser(QWEN)
    text = '<think>plan</think>{"name":"write","arguments":{"text":"use <think> and </think> tags"}}'
    out = parser.parse(text)
    assert out.reasoning == "plan"
    assert (
        out.content
        == '{"name":"write","arguments":{"text":"use <think> and </think> tags"}}'
    )


def test_close_tag_inside_json_without_thinking_is_content():
    parser = get_reasoning_parser(QWEN)
    text = '{"name":"write","arguments":{"text":"close with </think>"}}'
    out = parser.parse(text)
    assert out.reasoning is None
    assert out.content == text


@pytest.mark.parametrize(
    "text,reasoning,content",
    [
        ("r1</think>answer", "r1", "answer"),
        ("<think>r1</think>answer", "r1", "answer"),
        ("r1</think>mid</think>tail", "r1", "midtail"),
        ("<think>a</think>x<think>b</think>y", "a\nb", "xy"),
        ("<think>unclosed", "unclosed", ""),
    ],
)
def test_existing_reasoning_split_behaviour_is_preserved(text, reasoning, content):
    out = get_reasoning_parser(QWEN).parse(text)
    assert (out.reasoning, out.content) == (reasoning, content)
