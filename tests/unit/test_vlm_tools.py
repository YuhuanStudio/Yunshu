"""VLM chat prompt dropped tool_calls + didn't normalize string args.

VLMEngine._format_prompt rebuilt each message as {role, content} ONLY, dropping
tool_calls/tool_call_id/name. For a VLM + tools conversation (e.g. GLM-4V) prior assistant
tool-call turns vanished from the prompt; and string-form tool args left intact would trip
a GLM template's `is not mapping` raise → the whole prompt collapsing to the plaintext
fallback. Now mirrors the BatchedEngine path: preserve the fields + normalize string args.
"""

from __future__ import annotations

import yunshu_engine.vlm_engine as v


def test_normalize_vlm_tool_calls():
    f = v.VLMEngine._normalize_vlm_tool_calls
    # string JSON args → dict
    out = f([{"function": {"name": "g", "arguments": '{"a": 1}'}}])
    assert out[0]["function"]["arguments"] == {"a": 1}
    # non-JSON string → wrapped, not dropped
    out = f([{"function": {"name": "g", "arguments": "oops"}}])
    assert out[0]["function"]["arguments"] == {"value": "oops"}
    # already a dict → untouched
    out = f([{"function": {"name": "g", "arguments": {"a": 1}}}])
    assert out[0]["function"]["arguments"] == {"a": 1}


def test_format_prompt_preserves_tool_fields():
    eng = v.VLMEngine.__new__(v.VLMEngine)
    captured = {}

    class FakeTok:
        def apply_chat_template(self, msgs, **kw):
            captured["msgs"] = msgs
            return "PROMPT"

    eng._tokenizer = FakeTok()
    eng._extract_text = lambda c: c if isinstance(c, str) else ""

    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_w", "arguments": '{"loc": "SF"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "get_w", "content": "sunny"},
    ]
    out = eng._format_prompt(msgs)
    assert out == "PROMPT"
    cap = captured["msgs"]
    # assistant tool_calls preserved, args normalized to dict
    assert "tool_calls" in cap[1]
    assert cap[1]["tool_calls"][0]["function"]["arguments"] == {"loc": "SF"}
    # tool message keeps tool_call_id + name
    assert cap[2].get("tool_call_id") == "c1"
    assert cap[2].get("name") == "get_w"
