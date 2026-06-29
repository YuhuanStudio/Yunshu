"""the two early-error returns in RealtimeSession._generate_response
(empty conversation / no engine) emitted only an `error` event and returned. In
the OpenAI Realtime protocol an `error` does NOT terminate a response — it stays
in_progress until `response.done` — so the client's response future hung forever
(and the SDK refused the next turn). Both paths must emit a terminal response.done."""

from __future__ import annotations

import asyncio

from yunshu_gateway.routers.realtime import RealtimeSession


def _session_capturing(events):
    s = RealtimeSession.__new__(RealtimeSession)
    s._response_item_open = False  # _close_response_item is a no-op (item not open)
    s._active_response = None  # so the finally's `is current_task` skips the clear

    async def _cap(ev):
        events.append(ev)

    s.send_event = _cap
    return s


def _run_generate(session):
    asyncio.run(session._generate_response("resp-1", "item-1", ["text"], {}))


def _has_terminal_done(events):
    dones = [e for e in events if e.get("type") == "response.done"]
    return dones


def test_empty_conversation_emits_response_done():
    events = []
    s = _session_capturing(events)
    s._build_messages = lambda **k: []  # → empty-messages early return
    _run_generate(s)
    assert any(e.get("type") == "error" for e in events)
    dones = _has_terminal_done(events)
    assert len(dones) == 1, (
        f"expected exactly one response.done, got {[e.get('type') for e in events]}"
    )
    assert dones[0]["response"]["status"] == "failed"
    assert s._response_done_emitted is True


def test_no_engine_emits_response_done():
    events = []
    s = _session_capturing(events)
    s._build_messages = lambda **k: [{"role": "user", "content": "hi"}]
    s._resolve_engine = lambda: None  # → engine-None early return
    _run_generate(s)
    assert any(e.get("type") == "error" for e in events)
    dones = _has_terminal_done(events)
    assert len(dones) == 1
    assert dones[0]["response"]["status"] == "failed"
    assert s._response_done_emitted is True
