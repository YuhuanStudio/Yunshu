"""(MED): realtime never enforced tool_choice — a response.create with
tool_choice="none" still parsed the model's text for tool calls and emitted function_call
items, so a client asking for a plain-text turn got a tool call anyway. This is the realtime
sibling of the anthropic / chat tool_choice enforcement, which realtime lacked.

Fix: SessionConfig gains a tool_choice field (so session.update can set a session default),
and _generate_response snapshots a per-response tool_choice (override → session) and skips
tool-call parsing/emission when it is "none" (still stripping any stray markup, class).
"auto"/"required"/named keep parsing — post-gen forcing of required/named isn't feasible here.
"""

from __future__ import annotations

import inspect

from yunshu_gateway.routers import realtime
from yunshu_gateway.routers.realtime import SessionConfig


def test_session_default_tool_choice_is_auto():
    assert SessionConfig().tool_choice == "auto"


def test_session_update_accepts_valid_tool_choice():
    s = SessionConfig()
    assert "tool_choice" in s.update({"tool_choice": "none"})
    assert s.tool_choice == "none"
    s.update({"tool_choice": "required"})
    assert s.tool_choice == "required"
    s.update({"tool_choice": {"type": "function", "function": {"name": "f"}}})
    assert s.tool_choice == {"type": "function", "function": {"name": "f"}}


def test_session_update_rejects_garbage_tool_choice():
    s = SessionConfig()
    s.update({"tool_choice": "none"})
    # a non-string / non-dict (e.g. an int or a bogus string) must be dropped, not stored
    s.update({"tool_choice": 123})
    assert s.tool_choice == "none"  # unchanged
    s.update({"tool_choice": "bogus"})
    assert s.tool_choice == "none"


def test_generate_response_snapshots_and_gates_on_tool_choice():
    src = inspect.getsource(realtime.RealtimeSession._generate_response)
    # the per-response snapshot exists and falls back to the session default
    assert '_snap_tool_choice = config.get("tool_choice"' in src
    # the tool-call parse is gated on tool_choice != "none"
    snap = src.index("_snap_tool_choice")
    gate = src.index('_snap_tool_choice != "none"', snap)
    parse = src.index("parse_tool_calls(full_text", gate)
    assert snap < gate < parse, "parse must be gated on _snap_tool_choice != 'none'"
    # and the none path still strips stray markup (class)
    assert '_snap_tool_choice == "none"' in src
