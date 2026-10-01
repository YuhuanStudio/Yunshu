"""R16 / R25: the size-limit middleware forwards a counted body exactly once."""

from __future__ import annotations

import asyncio

from starlette.responses import JSONResponse

from yunshu_gateway.main import create_app


def _scope(path, app, method):
    return {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"host", b"t")],
        "client": ("127.0.0.1", 1),
        "server": ("t", 80),
        "scheme": "http",
        "app": app,
    }


def test_body_without_content_length_reaches_handler():
    app = create_app()

    async def echo(request):
        return JSONResponse({"len": len(await request.body())})

    app.add_route("/_t/echo", echo, methods=["POST"])
    sent: list = []

    async def run():
        queue = [
            {"type": "http.request", "body": b"abc", "more_body": True},
            {"type": "http.request", "body": b"def", "more_body": False},
        ]

        async def receive():
            if queue:
                return queue.pop(0)
            await asyncio.sleep(5)
            return {"type": "http.disconnect"}

        async def send(msg):
            sent.append(msg)

        await asyncio.wait_for(app(_scope("/_t/echo", app, "POST"), receive, send), 3)

    asyncio.run(run())
    body = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    assert body == b'{"len":6}'
