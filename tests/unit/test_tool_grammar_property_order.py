"""Tool grammars are keyed by the request's declared property order (mlx-swift-lm d02948b): llguidance
emits object properties in the schema's order, so two tool sets that differ only in that order must not
share one compiled grammar."""

import json

import pytest

pytest.importorskip("llguidance")

from llguidance import LLMatcher  # noqa: E402

from yunshu_engine import tool_call_grammar as tcg  # noqa: E402


def _tools(order):
    props = {name: {"type": "integer"} for name in order}
    return [
        {
            "type": "function",
            "function": {
                "name": "Move",
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": list(order),
                },
            },
        }
    ]


def test_llguidance_follows_the_declared_property_order():
    def emitted(order):
        schema = {
            "type": "object",
            "properties": {k: {"type": "integer"} for k in order},
            "required": list(order),
            "additionalProperties": False,
        }
        grammar = LLMatcher.grammar_from_json_schema(json.dumps(schema))
        return grammar.find("xxx") < grammar.find("yyy")

    assert emitted(["xxx", "yyy"]) and not emitted(["yyy", "xxx"])


def test_property_order_is_part_of_the_grammar_key():
    assert tcg.grammar_key(_tools(["xxx", "yyy"]), "required", True) != tcg.grammar_key(
        _tools(["yyy", "xxx"]), "required", True
    )


def test_the_same_request_keeps_the_same_key():
    assert tcg.grammar_key(_tools(["xxx", "yyy"]), "required", True) == tcg.grammar_key(
        _tools(["xxx", "yyy"]), "required", True
    )
