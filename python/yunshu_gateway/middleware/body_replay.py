"""Replay an already-consumed request body to a downstream ASGI app."""

from __future__ import annotations


def replay_receive(body: bytes, original_receive):
    """A ``receive`` that yields ``body`` once, then defers to the real receive.

    Later calls therefore see the client's ``http.disconnect`` (and wait for it)
    instead of the body being repeated forever.
    """
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await original_receive()

    return receive
