"""_json_object_end_after must equal the verbatim old per-brace tail-copy scan."""

import json
import random

from yunshu_engine.prompt_caching import _json_object_end_after


def _old(marked_prompt, at):
    decoder = json.JSONDecoder()
    candidates = []
    for start in range(at):
        if marked_prompt[start] != "{":
            continue
        try:
            _, length = decoder.raw_decode(marked_prompt[start:])
        except ValueError:
            continue
        if start + length > at:
            candidates.append(start + length)
    return max(candidates) if candidates else None


def test_nested_object_with_marker_inside():
    text = 'sys {"a": {"b": [1, 2, {"c": "x"}]}, "d": "MARK tail"} after {"z": 1}'
    at = text.index("MARK")
    assert _json_object_end_after(text, at) == _old(text, at)
    assert _json_object_end_after(text, at) is not None


def test_no_candidate():
    text = 'plain {"a": 1} MARK text'
    at = text.index("MARK")
    assert _json_object_end_after(text, at) is None
    assert _old(text, at) is None


def test_fuzz_identical():
    rng = random.Random(9)
    pieces = [
        "{",
        "}",
        '"',
        ":",
        ",",
        "1",
        "a",
        " ",
        "[",
        "]",
        "\\",
        '{"k": ',
        '"v"}',
        "null",
    ]
    for _ in range(3000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randrange(1, 60)))
        at = rng.randrange(0, len(text) + 1)
        assert _json_object_end_after(text, at) == _old(text, at), (text, at)
