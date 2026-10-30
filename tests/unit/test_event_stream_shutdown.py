"""Event streams end when the server starts shutting down, not only when the client leaves.

A remote console held /v1/yunshu/events open while the gate stopped a server: uvicorn waited the
whole graceful timeout, then cancelled the stream and logged three tracebacks (gate log_clean FAIL)."""

import asyncio
import signal

from yunshu_gateway import event_hub


def test_open_stream_ends_after_close_streams(monkeypatch):
    monkeypatch.setattr(event_hub, "_CLOSING", event_hub.threading.Event())
    hub = event_hub.Hub({})

    async def never_disconnected():
        return False

    async def run():
        stream = hub.subscribe(
            [], None, boot_id="b", is_disconnected=never_disconnected
        )
        hello = await stream.__anext__()
        event_hub.close_streams()
        rest = [frame async for frame in stream]
        return hello, rest

    hello, rest = asyncio.run(asyncio.wait_for(run(), timeout=5))
    assert hello.startswith("event: hello") and rest == []


def test_signal_chain_closes_streams_then_calls_the_server_handler(monkeypatch):
    monkeypatch.setattr(event_hub, "_CLOSING", event_hub.threading.Event())
    calls = []
    previous = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, lambda s, f: calls.append(s))
        event_hub.close_streams_on_signals((signal.SIGTERM,))
        event_hub.close_streams_on_signals((signal.SIGTERM,))  # one wrapper, not two
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert calls == [signal.SIGTERM]
        assert event_hub._CLOSING.is_set()
    finally:
        signal.signal(signal.SIGTERM, previous)
