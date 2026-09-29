"""A constrained (JSON schema / grammar) VLM request defaults to thinking off.

The constraint masks output from the first token; with a template that opens
`<think>` the reasoning was forced into the schema and the content came back empty
(Ollama `format` on Qwen3.5 answered 422).
"""

from __future__ import annotations

from yunshu_engine.vlm_engine import VLMEngine


def _engine(model_name="qwen3.5-0.8b"):
    e = object.__new__(VLMEngine)
    e._config = {"model_type": "qwen3_5"}
    e._model_path = model_name
    return e


def test_constrained_defaults_thinking_off():
    assert _engine()._default_enable_thinking(None, constrained=True) is False


def test_unconstrained_leaves_default():
    assert _engine()._default_enable_thinking(None) is None


def test_explicit_choice_wins():
    assert _engine()._default_enable_thinking(True, constrained=True) is True
    assert _engine()._default_enable_thinking(False) is False
