"""A model unloaded or reloaded while a request runs on it: the unload is refused (409) until the
request is done, the request is not harmed, and afterwards the unload works and a new request gets
a clean error instead of a hang."""

from __future__ import annotations

import asyncio

import pytest

from yunshu_engine.model_manager import ModelEntry, ModelManager, ModelType
from yunshu_gateway import engine as engine_mod

from .harness import (
    FakeBatchedEngine,
    Script,
    asgi_call,
    assert_one_terminal,
    request_for,
    running_app,
)


@pytest.fixture
def managed(make_app):
    eng = FakeBatchedEngine(Script.hang_after(1), model="fault-model")
    eng.has_active_requests = lambda: eng.inflight > 0  # type: ignore[method-assign]
    mgr = ModelManager()
    mgr._entries["fault-model"] = ModelEntry(
        model_id="fault-model",
        model_path="/does/not/exist",
        model_type=ModelType.LLM,
        engine=eng,
        is_loaded=True,
        estimated_bytes=0,
    )
    app = make_app(eng)
    engine_mod._engine = None  # multi-model mode: the manager resolves the engine
    engine_mod._model_manager = mgr
    yield app, eng, mgr
    engine_mod._model_manager = None


@pytest.mark.parametrize("dialect", ["openai", "anthropic", "responses"])
async def test_unload_is_refused_while_a_request_runs_then_works(
    managed, audit, dialect
):
    app, eng, mgr = managed
    path, body = request_for(dialect, True, model="fault-model")
    async with running_app(app, warm=False):
        a = audit(eng)
        t = asyncio.create_task(asgi_call(app, "POST", path, body))
        for _ in range(300):
            await asyncio.sleep(0.01)
            if eng.inflight:
                break
        assert eng.inflight == 1
        r = await asgi_call(app, "POST", "/v1/models/unload/fault-model")
        assert r.status == 409
        err = r.json()["error"]
        assert "active requests" in err["message"]
        assert mgr.get_entry("fault-model").is_loaded  # still there
        assert eng.inflight == 1  # and the request is untouched
        await asgi_call(app, "POST", "/v1/cancel", {"cancel_all": True})
        rep = await t
        assert_one_terminal(dialect, rep.text)
        await a.settle()
        a.assert_clean()
        r = await asgi_call(app, "POST", "/v1/models/unload/fault-model")
        assert r.status == 200, r.text
        assert not mgr.get_entry("fault-model").is_loaded


async def test_request_after_unload_is_a_clean_error_not_a_hang(managed):
    app, eng, mgr = managed
    path, body = request_for("openai", False, model="fault-model")
    async with running_app(app, warm=False):
        eng.script = Script.ok("x")
        assert (
            await asgi_call(app, "POST", "/v1/models/unload/fault-model")
        ).status == 200
        r = await asgi_call(app, "POST", path, body, timeout=20)
        assert r.status in (404, 500, 503)
        assert r.headers["content-type"].startswith("application/json")
        assert "error" in r.json()
