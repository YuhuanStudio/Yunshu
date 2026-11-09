"""Parallel tool results reach the template in tool_calls order (2026-11-09).

Before: Qwen templates pair <tool_response> blocks with calls by position and never read
tool_call_id, so results returned in completion order were matched to the wrong calls.
"""

from yunshu_engine.message_adapter import adapt_messages, order_tool_results


def _turn(results):
    return [
        {"role": "user", "content": "weather in A and B"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "w", "arguments": "{}"},
                },
                {
                    "id": "b",
                    "type": "function",
                    "function": {"name": "w", "arguments": "{}"},
                },
                {
                    "id": "c",
                    "type": "function",
                    "function": {"name": "w", "arguments": "{}"},
                },
            ],
        },
        *[{"role": "tool", "tool_call_id": i, "content": t} for i, t in results],
        {"role": "user", "content": "thanks"},
    ]


def _tools(msgs):
    return [(m.get("tool_call_id"), m["content"]) for m in msgs if m["role"] == "tool"]


def test_completion_order_is_put_back_into_call_order():
    got = order_tool_results(_turn([("c", "C"), ("a", "A"), ("b", "B")]))
    assert _tools(got) == [("a", "A"), ("b", "B"), ("c", "C")]


def test_unknown_ids_follow_in_their_own_order():
    got = order_tool_results(_turn([("x", "X"), ("b", "B"), (None, "N"), ("a", "A")]))
    assert _tools(got) == [("a", "A"), ("b", "B"), ("x", "X"), (None, "N")]


def test_runs_only_reorder_within_their_own_turn_and_nothing_else_moves():
    msgs = _turn([("b", "B"), ("a", "A")]) + _turn([("c", "C"), ("b", "B")])
    got = order_tool_results(msgs)
    assert [m["role"] for m in got] == [m["role"] for m in msgs]
    assert _tools(got) == [("a", "A"), ("b", "B"), ("b", "B"), ("c", "C")]


def test_the_template_input_of_every_route_is_ordered():
    got = adapt_messages(_turn([("b", "B"), ("a", "A")]), "Qwen3.8-27B")
    assert _tools(got) == [("a", "A"), ("b", "B")]
