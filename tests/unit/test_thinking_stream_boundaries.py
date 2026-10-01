"""R22: the streaming ThinkingParser must split identically however the text is chunked."""

from __future__ import annotations

import random

from yunshu_gateway.streaming import ThinkingParser

TAGS = [
    "<think>",
    "</think>",
    "<think/>",
    "</think/>",
    "<think >",
    "</think >",
    "<think  >",
    "</think\n>",
    "<think\n>",
    "<thinker>",
    "<thin",
    "</thi",
    "<",
    "</",
]
WORDS = ["a", " b", "龍", "\n", "é", "x<y", "```", "{", "}"]


def _run(pieces):
    p = ThinkingParser()
    visible, thinking = [], []
    for piece in pieces:
        out = p.process_chunk(piece)
        visible.append(out["visible"])
        thinking.append(out["thinking"])
    out = p.finalize()
    visible.append(out["visible"])
    thinking.append(out["thinking"])
    return "".join(visible), "".join(thinking)


def test_stream_split_is_independent_of_chunking():
    rng = random.Random(7)
    for _ in range(3000):
        n = rng.randint(1, 10)
        text = "".join(
            rng.choice(TAGS) if rng.random() < 0.4 else rng.choice(WORDS)
            for _ in range(n)
        )
        whole = _run([text])
        cuts = sorted(
            rng.sample(range(len(text) + 1), rng.randint(0, min(len(text), 6)))
        )
        pieces, pos = [], 0
        for cut in cuts + [len(text)]:
            pieces.append(text[pos:cut])
            pos = cut
        assert _run(pieces) == whole, (text, pieces)


def test_tag_split_character_by_character():
    text = "pre<think>reason</think>post"
    assert _run(list(text)) == ("prepost", "reason")
