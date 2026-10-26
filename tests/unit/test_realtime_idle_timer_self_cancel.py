"""An idle_timeout_ms response must not be aborted by the idle timer cancelling itself.

_handle_response_create cancels the idle timer; when the timer itself triggered the response
that is the running task, so the cancel landed on its next real suspension and the
response.created / generation after it never happened.
"""

from __future__ import annotations

import asyncio

import pytest

from yunshu_gateway.routers.realtime import ConversationItem

from .test_realtime_audio import _make_session


@pytest.mark.asyncio
async def test_idle_timer_response_is_not_cancelled_by_itself():
    s = _make_session()
    s.session.turn_detection = {"type": "server_vad", "idle_timeout_ms": 50}
    s.conversation.add_item(
        ConversationItem("item_a", "message", role="user", content=[])
    )
    real_send = s.ws.send_json

    async def yielding_send(data):  # a real socket suspends on send
        await asyncio.sleep(0)
        await real_send(data)

    s.ws.send_json = yielding_send
    s._idle_task = asyncio.create_task(s._idle_timeout_run(0.01))
    await asyncio.wait({s._idle_task}, timeout=2)
    assert "response.created" in [e["type"] for e in s.ws.sent]
