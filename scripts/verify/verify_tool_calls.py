"""Tool-call parsing gate (no model needed — pure parser).

Verifies the model-aware tool-call parser extracts name+arguments correctly,
including the string-aware-brace edge case (structural `}{`/quotes INSIDE an
argument string value must not break JSON brace tracking) and multiple tool calls
in one response. Guards the tool-calling protocol surface.

Run: PYTHONPATH=. uv run python scripts/verify_tool_calls.py
"""
from __future__ import annotations

import json
import sys

from yunshu_engine.tool_call_parser import parse_tool_calls


def _name_args(c):
    name = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else None)
    args = (getattr(c, "arguments", None) or getattr(c, "args", None)
            or (c.get("arguments") if isinstance(c, dict) else None))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            pass
    return name, args


CASES = [
    ("hermes single",
     '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
     "Qwen2.5-3B",
     lambda n, a: n == "get_weather" and a.get("city") == "Paris"),
    # structural }{ AND escaped quotes inside the arg string must survive
    ("brace-in-string",
     '<tool_call>{"name": "echo", "arguments": {"text": "use }{ and \\"quotes\\""}}</tool_call>',
     "Qwen2.5-3B",
     lambda n, a: n == "echo" and "}{" in a.get("text", "")),
    ("multiple calls",
     '<tool_call>{"name":"a","arguments":{}}</tool_call>'
     '<tool_call>{"name":"b","arguments":{"x":1}}</tool_call>',
     "Qwen2.5-3B",
     None),  # checked by count below
]


def main() -> int:
    fails = 0
    for label, text, model, check in CASES:
        calls = parse_tool_calls(text, model)
        if label == "multiple calls":
            ok = len(calls) == 2 and {_name_args(c)[0] for c in calls} == {"a", "b"}
        else:
            ok = bool(calls) and check(*_name_args(calls[0]))
        print(f"  {'OK ' if ok else 'BAD'} {label}: {len(calls)} call(s)"
              + (f" → {_name_args(calls[0])}" if calls else ""))
        if not ok:
            fails += 1
    print(f"RESULT: {len(CASES) - fails} passed, {fails} failed")
    print("PASS" if fails == 0 else "FAIL")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
