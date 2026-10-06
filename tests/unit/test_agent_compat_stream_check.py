"""The agent-compat stream validators accept spec-shaped streams and reject broken ones."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research/agent_compat"))
import stream_check as sc  # noqa: E402


def _anth(tool=False):
    ev = [
        ("message_start", {"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant", "model": "x", "content": [], "usage": {"input_tokens": 3, "output_tokens": 0}}}),
        ("ping", {"type": "ping"}),
    ]
    if tool:
        ev += [
            ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t", "name": "f", "input": {}}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"a":'}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "1}"}}),
        ]
    else:
        ev += [
            ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
        ]
    ev += [
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use" if tool else "end_turn"}, "usage": {"output_tokens": 2}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return ev


def test_anthropic_valid():
    assert sc.check_anthropic_stream(_anth()) == []
    assert sc.check_anthropic_stream(_anth(tool=True)) == []


def test_anthropic_breaks():
    ev = _anth()
    assert sc.check_anthropic_stream(ev[:-1])  # no message_stop
    bad = copy.deepcopy(ev)
    bad[2][1]["index"] = 1
    assert any("index" in x for x in sc.check_anthropic_stream(bad))
    bad = copy.deepcopy(_anth(tool=True))
    bad[4][1]["delta"]["partial_json"] = "1"  # '{"a":1' never closed
    assert any("JSON" in x for x in sc.check_anthropic_stream(bad))
    bad = copy.deepcopy(ev)
    bad[-2][1]["delta"]["stop_reason"] = "tool_use"  # no tool_use block
    assert any("disagrees" in x for x in sc.check_anthropic_stream(bad))
    assert sc.check_anthropic_stream([("error", {"type": "error", "error": {"type": "api_error", "message": "x"}})])
    assert sc.check_anthropic_stream([])


def _chunk(delta, finish=None, usage=None, choices=True):
    c = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}] if choices else []}
    if usage:
        c["usage"] = usage
    return (None, c)


def test_chat_valid_and_broken():
    u = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    ev = [_chunk({"role": "assistant", "content": ""}), _chunk({"content": "a"}), _chunk({}, "stop"), _chunk({}, usage=u, choices=False), (None, "[DONE]")]
    assert sc.check_chat_stream(ev, expect_usage=True) == []
    assert sc.check_chat_stream(ev[:-1])  # no [DONE]
    assert sc.check_chat_stream(ev[:3] + ev[4:], expect_usage=True)  # usage missing
    tool = [
        _chunk({"role": "assistant", "tool_calls": [{"index": 0, "id": "i", "type": "function", "function": {"name": "f", "arguments": ""}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"a":1}'}}]}),
        _chunk({}, "tool_calls"),
        (None, "[DONE]"),
    ]
    assert sc.check_chat_stream(tool) == []
    tool[2] = _chunk({}, "stop")
    assert any("disagrees" in x for x in sc.check_chat_stream(tool))


def _r(t, n, **k):
    return (t, {"type": t, "sequence_number": n, **k})


def test_responses_valid_and_broken():
    resp = {"id": "r", "object": "response", "status": "completed", "model": "m", "output": [{"type": "message"}], "usage": {}}
    ev = [
        _r("response.created", 0, response=resp),
        _r("response.output_item.added", 1, output_index=0, item={"type": "message"}),
        _r("response.output_text.delta", 2, delta="a"),
        _r("response.output_item.done", 3, output_index=0, item={"type": "message"}),
        _r("response.completed", 4, response=resp),
    ]
    assert sc.check_responses_stream(ev) == []
    assert any("terminal" in x for x in sc.check_responses_stream(ev[:-1]))
    bad = copy.deepcopy(ev)
    bad[3][1]["sequence_number"] = 1
    assert any("increasing" in x for x in sc.check_responses_stream(bad))
    assert any("added" in x for x in sc.check_responses_stream(ev[:3] + ev[4:]))


def test_body_checks_use_sdk_models():
    assert sc.check_body("/v1/messages", {"type": "error", "error": {}})
    assert sc.check_body("/v1/chat/completions", {"id": "x"})
    ok = {"id": "c", "object": "chat.completion", "created": 1, "model": "m", "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "a"}}]}
    assert sc.check_body("/v1/chat/completions", ok) == []


def test_responses_trailing_done_is_tolerated():
    resp = {"id": "r", "object": "response", "status": "completed", "model": "m", "output": [], "usage": {}}
    ev = [_r("response.created", 0, response=resp), _r("response.completed", 1, response=resp), (None, "[DONE]")]
    assert sc.check_responses_stream(ev) == []
