"""R14 / R15: response-cache key hygiene and body replay that keeps disconnect."""

from __future__ import annotations

import asyncio
import json

from yunshu_gateway.middleware.gateway_optimizer import ResponseCacheMiddleware


class FakeCache:
    enabled = True

    def __init__(self):
        self.store: dict = {}

    async def get(self, key):
        return self.store.get(key)

    async def put(self, key, value):
        self.store[key] = value


class _State:
    pass


class _App:
    def __init__(self, cache):
        self.state = _State()
        self.state.response_cache = cache


def _scope(app, body: bytes):
    return {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "app": app,
    }


async def _call(mw, app, body: bytes, receive=None):
    sent = []

    async def default_receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(msg):
        sent.append(msg)

    await mw(_scope(app, body), receive or default_receive, send)
    return sent


def _downstream(counter, reads=None):
    async def app(scope, receive, send):
        counter.append(1)
        msg = await receive()
        if reads is not None:
            reads.append(msg)
            reads.append(await receive())
        payload = json.dumps({"n": len(counter)}).encode()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": payload})

    return app


BODY = json.dumps(
    {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0}
).encode()


def test_hit_then_generation_change_misses(monkeypatch):
    from yunshu_gateway import engine as engine_mod

    gen = {"v": "g1"}
    monkeypatch.setattr(engine_mod, "engine_generation", lambda: gen["v"])
    calls: list = []
    cache = FakeCache()
    app = _App(cache)
    mw = ResponseCacheMiddleware(_downstream(calls))

    async def run():
        await _call(mw, app, BODY)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await _call(mw, app, BODY)  # hit
        assert len(calls) == 1
        gen["v"] = "g2"  # model unloaded / reloaded
        await _call(mw, app, BODY)
        assert len(calls) == 2

    asyncio.run(run())


def test_file_and_previous_response_references_bypass_cache():
    calls: list = []
    cache = FakeCache()
    app = _App(cache)
    mw = ResponseCacheMiddleware(_downstream(calls))
    for extra in ({"file_id": "file-abc"}, {"previous_response_id": "resp_1"}):
        body = json.dumps(
            {"model": "m", "messages": [], "temperature": 0, "x": extra}
        ).encode()

        async def run(body=body):
            await _call(mw, app, body)
            await asyncio.sleep(0)
            await _call(mw, app, body)

        asyncio.run(run())
    assert len(calls) == 4
    assert cache.store == {}


def test_replayed_body_not_repeated_and_disconnect_preserved():
    calls: list = []
    reads: list = []
    cache = FakeCache()
    app = _App(cache)
    mw = ResponseCacheMiddleware(_downstream(calls, reads))
    queue = [{"type": "http.request", "body": BODY, "more_body": False}]

    async def receive():
        if queue:
            return queue.pop(0)
        return {"type": "http.disconnect"}

    asyncio.run(_call(mw, app, BODY, receive))
    assert reads[0]["body"] == BODY
    assert reads[1]["type"] == "http.disconnect"
