"""(HIGH): anthropic.py gated thinking handling on a falsy `enable_thinking` that
the engine no longer respects — native-thinking models (Qwen3/DeepSeek-R1/GLM-Z1) emit CoT
by DEFAULT (W832) regardless of the request's thinking config. Result: NON-stream leaked
the raw <think>…</think> + entire CoT into the visible text block; streaming dropped the
reasoning tokens entirely (hit neither branch). chat.py/responses.py route reasoning
unconditionally — Anthropic was the lone outlier. Now both Anthropic paths separate/route
reasoning unconditionally (extract_thinking is a no-op without think tags).
"""
from __future__ import annotations

import inspect

from yunshu_gateway.routers import (
    anthropic as A,  # noqa: N812  # intentional short module alias
)
from yunshu_gateway.streaming import extract_thinking


def test_extract_thinking_separates_cot_premise():
    # the helper the non-stream path relies on: <think> CoT is split out of visible text
    thinking, visible = extract_thinking("<think>reasoning</think>the answer", "qwen3")
    assert "reasoning" in thinking
    assert "<think>" not in visible and visible.strip() == "the answer"


def test_nonstream_extract_thinking_not_gated_on_enable_thinking():
    # neither non-stream extract_thinking call (one uses `text`, one `result.text`) may be
    # gated by an `if enable_thinking:` guard immediately before it.
    src = inspect.getsource(A)
    for call in ("extract_thinking(text, req.model)", "extract_thinking(result.text, req.model)"):
        i = src.index(call)
        window = src[max(0, i - 160):i]
        assert "if enable_thinking:" not in window, f"thinking extraction still gated before {call!r}"


def test_streaming_routes_on_is_reasoning_alone():
    src = inspect.getsource(A)
    # BOTH streaming reasoning routings (batched + non-batched) drop the enable_thinking gate
    assert "if enable_thinking and _is_reasoning and _token_text:" not in src
    assert src.count("if _is_reasoning and _token_text:") >= 2
