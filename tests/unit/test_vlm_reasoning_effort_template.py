"""reasoning_effort goes into a chat template that supports it (Qwen3.8)."""

from types import SimpleNamespace

from yunshu_engine.vlm_engine import VLMEngine


def _engine(template):
    eng = VLMEngine.__new__(VLMEngine)
    eng._tokenizer = SimpleNamespace(chat_template=template)
    eng._processor = SimpleNamespace(chat_template=None)
    return eng


def test_effort_moves_into_supporting_template():
    eng = _engine("{% set reasoning_effort = reasoning_effort|default('xhigh') %}")
    kwargs = {"reasoning_effort": "medium"}
    assert eng._template_effort_extra(kwargs) == {"reasoning_effort": "medium"}
    assert "reasoning_effort" not in kwargs  # no thinking-budget mapping later


def test_effort_from_chat_template_kwargs():
    eng = _engine("reasoning_effort")
    kwargs = {"chat_template_kwargs": {"reasoning_effort": "low"}}
    assert eng._template_effort_extra(kwargs) == {"reasoning_effort": "low"}


def test_effort_kept_as_budget_when_template_lacks_it():
    eng = _engine("{{ messages }}")
    kwargs = {"reasoning_effort": "medium"}
    assert eng._template_effort_extra(kwargs) is None
    assert kwargs["reasoning_effort"] == "medium"
