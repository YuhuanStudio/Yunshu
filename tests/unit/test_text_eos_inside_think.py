"""A reply that ends inside the <think> block the prompt opened keeps its text as reasoning, not
content (the VLM path's rule; the streaming split already did this, the non-streaming parsers did not)."""

from __future__ import annotations

from yunshu_engine.reasoning_parser import (
    close_unterminated_reasoning,
    get_reasoning_parser,
)

from .tiny_text_engine import FakeTokenizer, make_engine

OPEN, CLOSE = 60, 61  # ':' and ';' encode to these ids in FakeTokenizer
PROMPT = "hello:"  # ends with the think-start token


class ThinkTok(FakeTokenizer):
    think_start_id = OPEN

    def encode(self, text, add_special_tokens=True, **kw):
        if text == "<think>":
            return [OPEN]
        if text == "</think>":
            return [CLOSE]
        return super().encode(text, add_special_tokens=add_special_tokens)


def _engine():
    eng = make_engine()
    eng._tokenizer = ThinkTok()
    return eng


def test_helper_closes_only_an_unterminated_prompt_opened_block():
    assert close_unterminated_reasoning("abc", True) == "abc</think>"
    assert close_unterminated_reasoning("abc", False) == "abc"
    assert close_unterminated_reasoning("a</think>b", True) == "a</think>b"
    assert close_unterminated_reasoning("<think>a", True) == "<think>a"
    assert close_unterminated_reasoning('{"a": 1}', True) == '{"a": 1}'
    out = get_reasoning_parser("x").parse(close_unterminated_reasoning("abc", True))
    assert out.reasoning == "abc" and out.content == ""


async def test_non_stream_fast_path_tail_inside_open_think_is_not_content():
    out = await _engine()._generate_fast(prompt=PROMPT, max_tokens=5, temperature=0.0)
    assert out.error is None and out.completion_tokens == 5
    assert out.text == "" and out.reasoning_tokens > 0


async def test_non_stream_without_an_open_think_is_content():
    out = await _engine()._generate_fast(prompt="hello", max_tokens=5, temperature=0.0)
    assert out.text and out.reasoning_tokens == 0


async def test_stream_tail_inside_open_think_is_reasoning():
    states = []
    async for o in _engine()._stream_generate_fast(
        prompt=PROMPT, max_tokens=5, temperature=0.0
    ):
        if o.new_text:
            states.append(o.current_state)
    assert states and set(states) == {"reasoning"}
