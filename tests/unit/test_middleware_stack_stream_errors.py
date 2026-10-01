"""R25: stream-header / error-envelope / cancel combinations through the full stack."""

from __future__ import annotations

import asyncio
import contextlib

from starlette.responses import StreamingResponse

from yunshu_gateway.main import create_app


def _app_with_routes():
    app = create_app()

    async def boom_after_headers(request):
        async def gen():
            yield b"data: one\n\n"
            raise RuntimeError("mid-stream failure")

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def slow_stream(request):
        async def gen():
            for _ in range(1000):
                yield b": tick\n\n"
                await asyncio.sleep(0.01)

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def boom_before_headers(request):
        raise RuntimeError("early failure")

    app.add_route("/_t/boom_after", boom_after_headers)
    app.add_route("/_t/slow", slow_stream)
    app.add_route("/_t/boom_before", boom_before_headers)
    return app


def _scope(path, app):
    return {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"host", b"t")],
        "client": ("127.0.0.1", 1),
        "server": ("t", 80),
        "scheme": "http",
        "app": app,
    }


def test_error_after_stream_headers_does_not_start_second_response():
    app = _app_with_routes()
    sent: list = []

    async def run():
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": b"", "more_body": False}
            await asyncio.sleep(5)
            return {"type": "http.disconnect"}

        async def send(msg):
            sent.append(msg)

        with contextlib.suppress(Exception):
            await asyncio.wait_for(app(_scope("/_t/boom_after", app), receive, send), 3)

    asyncio.run(run())
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1
    assert starts[0]["status"] == 200


def test_error_before_headers_is_one_envelope():
    app = _app_with_routes()
    sent: list = []

    async def run():
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": b"", "more_body": False}
            await asyncio.sleep(5)
            return {"type": "http.disconnect"}

        async def send(msg):
            sent.append(msg)

        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                app(_scope("/_t/boom_before", app), receive, send), 3
            )

    asyncio.run(run())
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1
    assert starts[0]["status"] == 500


def test_client_disconnect_stops_stream_promptly():
    app = _app_with_routes()
    chunks = 0

    async def run():
        nonlocal chunks
        gone = asyncio.Event()

        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": b"", "more_body": False}
            await gone.wait()
            return {"type": "http.disconnect"}

        async def send(msg):
            nonlocal chunks
            if msg["type"] == "http.response.body" and msg.get("body"):
                chunks += 1
                if chunks == 3:
                    gone.set()

        await asyncio.wait_for(app(_scope("/_t/slow", app), receive, send), 3)

    asyncio.run(run())
    assert 3 <= chunks < 50  # stopped soon after the disconnect, not 1000 ticks
