"""Recover trailing GLM calls only when generation ends on observation EOS.

Reasoning streams normally until a possible tool marker. Complete trailing calls
are held until the actual terminal token proves the model is awaiting a tool.
Length, cancel, user stops, malformed calls and quoted examples stay reasoning.
"""

from __future__ import annotations

import json
import re

from .tool_format import openai_tools, parse_tool_output


def _complete_calls(text: str) -> bool:
    bodies = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not bodies or len(bodies) != text.count("<tool_call>"):
        return False
    for body in bodies:
        for tag in ("arg_key", "arg_value"):
            if body.count(f"<{tag}>") != body.count(f"</{tag}>"):
                return False
        if body.lstrip().startswith(("{", "[")):
            try:
                json.loads(body)
            except ValueError:
                return False
    return True


def recover_tool_events(events, *, observation_id, formats, tools):
    marker = "<tool_call>"
    pending = ""
    candidate = False
    disabled = False
    names = {tool["function"]["name"] for tool in openai_tools(tools)}

    def text_event(text, state="reasoning"):
        return text, None, state, None, last_thinking, None

    last_thinking = 0
    for event in events:
        text, token, state, finish, last_thinking, lp = event
        if disabled:
            yield event
            continue
        if finish is not None:
            # The detokenizer's last partial bytes may arrive on the EOS event.
            if candidate:
                pending += text
                text = ""
            rescued = False
            if (
                candidate
                and finish == "stop"
                and token == observation_id
                and _complete_calls(pending)
            ):
                calls, remainder = parse_tool_output(pending, formats, tools)
                rescued = (
                    bool(calls)
                    and not remainder.strip()
                    and all(c["name"] in names for c in calls)
                )
            if pending:
                yield text_event(pending, "normal" if rescued else "reasoning")
            yield text, token, state, finish, last_thinking, lp
            return
        if state != "reasoning":
            if text and pending:
                yield text_event(pending)
                pending = ""
                candidate = False
            yield event
            continue
        pending += text
        if not candidate:
            at = pending.find(marker)
            if at >= 0:
                head, pending = pending[:at], pending[at:]
                candidate = True
            else:
                # Hold only the marker's possible suffix across tokenizer chunks.
                keep = next(
                    (
                        n
                        for n in range(min(len(pending), len(marker) - 1), 0, -1)
                        if pending.endswith(marker[:n])
                    ),
                    0,
                )
                head, pending = (
                    (pending[:-keep], pending[-keep:]) if keep else (pending, "")
                )
            yield head, token, state, finish, last_thinking, lp
        else:
            yield "", token, state, finish, last_thinking, lp
        if len(pending) > 1024 * 1024:
            yield text_event(pending)
            pending = ""
            disabled = True
    if pending:
        yield text_event(pending)
