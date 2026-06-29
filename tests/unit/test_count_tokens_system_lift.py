"""(MED): /v1/messages/count_tokens double-counted lifted role="system" messages.

Generation (create_message) LIFTS any role="system" entries out of messages[] and MERGES
them with the top-level `system` into ONE canonical system block. count_tokens counted only
the top-level system and left the in-message system entries in the converted list, rendering
a SECOND <|system|> wrapper per lifted message → an overcount vs the real prompt (proven ~+5
tokens per extra system message on the real Qwen tokenizer). count_tokens now mirrors the
lift so its estimate matches what generation actually prompts.
"""
from __future__ import annotations

import inspect

from yunshu_gateway.routers import (
    anthropic as A,  # noqa: N812  # intentional short module alias
)


def _merge_system_like_w1033(top_system, msgs):
    """Faithful replica of the lift: collect top-level + in-message system parts into
    ONE block, and the surviving non-system messages separately (mirrors generation)."""
    parts = []
    if top_system:
        parts.append(top_system)
    parts.extend(c for r, c in msgs if r == "system")
    parts = [p for p in parts if p]
    survivors = [(r, c) for r, c in msgs if r != "system"]
    return parts, survivors


def test_lifted_system_messages_merge_into_one_block():
    parts, survivors = _merge_system_like_w1033(
        "S0", [("system", "S1"), ("user", "hi"), ("system", "S2")])
    # all three system texts in ONE block, top-level first then in-message order
    assert parts == ["S0", "S1", "S2"]
    # the system messages are removed from the conversational turns (no double-count)
    assert survivors == [("user", "hi")]


def test_no_system_messages_unchanged():
    parts, survivors = _merge_system_like_w1033(None, [("user", "a"), ("assistant", "b")])
    assert parts == []
    assert survivors == [("user", "a"), ("assistant", "b")]


def test_count_tokens_source_lifts_and_filters_system():
    src = inspect.getsource(A.count_tokens)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # collects in-message role="system" into the canonical system parts
    assert 'getattr(_m, "role", None) == "system"' in code
    assert '"\\n\\n".join(_sys_parts)' in code
    # and drops them from the converted messages so the wrapper isn't counted twice
    assert 'm for m in converted_msgs if m.get("role") != "system"' in code
