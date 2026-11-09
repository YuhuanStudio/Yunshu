"""Text (mlx-lm) path: reasoning_effort reaches the chat template (2026-11-09).

Before: the text path never rendered reasoning_effort; it only turned it into an invented
thinking-token cap (low 2048 / medium 8192 / ...), so Qwen3.8's own low/medium/xhigh
levels were unreachable and long reasoning was cut off.
"""

import pytest

from yunshu_engine import batched_engine as be

QWEN38 = (
    "{%- set reasoning_effort = reasoning_effort|default('xhigh') %}"
    "{%- if reasoning_effort not in ('xhigh', 'medium', 'low') %}{{ raise_exception('bad') }}{%- endif %}"
    "{{ messages }}"
)


class Tok:
    def __init__(self, template):
        self.chat_template = template
        self.seen = None

    def apply_chat_template(self, messages, **kwargs):
        self.seen = kwargs
        return "x"


@pytest.fixture
def engine():
    eng = be.BatchedEngine.__new__(be.BatchedEngine)
    eng.enable_thinking = None
    eng.model_name = "Qwen3.8-27B"
    return eng


@pytest.mark.parametrize(
    "effort, rendered",
    [("low", "low"), ("medium", "medium"), ("xhigh", "xhigh"), ("high", "xhigh")],
)
def test_effort_is_rendered_onto_the_templates_own_levels(engine, effort, rendered):
    engine._tokenizer = Tok(QWEN38)
    be.set_request_effort(effort)
    engine._apply_chat_template([{"role": "user", "content": "hi"}])
    assert engine._tokenizer.seen["reasoning_effort"] == rendered


def test_no_effort_and_thinking_off_efforts_render_nothing(engine):
    engine._tokenizer = Tok(QWEN38)
    for effort in (None, "none", "minimal"):
        be.set_request_effort(effort)
        engine._apply_chat_template([{"role": "user", "content": "hi"}])
        assert "reasoning_effort" not in engine._tokenizer.seen


def test_a_template_without_the_variable_gets_nothing(engine):
    engine._tokenizer = Tok("{{ messages }}")
    be.set_request_effort("high")
    engine._apply_chat_template([{"role": "user", "content": "hi"}])
    assert "reasoning_effort" not in engine._tokenizer.seen
