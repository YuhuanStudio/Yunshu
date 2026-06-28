"""OpenAI Responses API `background: true` — POST returns a queued
response immediately and the full generation runs asynchronously under the same
id (pollable via GET, cancellable). Previously `background` was silently dropped
and the request ran synchronously (blocking)."""
from __future__ import annotations

import asyncio
import types

from yunshu_gateway.routers import (
    responses as R,  # noqa: N812  # intentional short module alias
)
from yunshu_gateway.routers.responses import ResponsesRequest


def _fake_request():
    """A minimal Request stand-in — only request.state is read by the handler
    helpers (_resolve_owner/_check_permission/rbac_key)."""
    return types.SimpleNamespace(state=types.SimpleNamespace(rbac_key=None, role="admin"))


def _payload_json(resp):
    import json
    return json.loads(bytes(resp.body).decode())


def test_background_field_accepted():
    req = ResponsesRequest(model="qwen", input="hi", background=True)
    assert req.background is True
    # default is off so existing callers are unaffected
    assert ResponsesRequest(model="qwen", input="hi").background is False


def test_background_returns_queued_then_completes(monkeypatch):
    R._response_store.clear()

    async def _fake_create_response(req, request):
        # Mirror the real handler's id-forcing + store-on-completion contract.
        rid = request.state._forced_response_id
        assert rid is not None  # id propagated to the generation
        assert req.background is False and req.stream is False and req.store is True
        R._store_response(rid, {
            "id": rid, "object": "response", "status": "completed",
            "model": req.model, "output": [{"type": "message", "role": "assistant",
                                            "content": [{"type": "output_text", "text": "done"}]}],
            "_owner": "",
        })
        return None

    monkeypatch.setattr(R, "create_response", _fake_create_response)

    async def _run():
        req = ResponsesRequest(model="qwen", input="hi", background=True)
        resp = await R._start_background_response(req, _fake_request())
        body = _payload_json(resp)
        assert body["status"] == "queued"
        rid = body["id"]
        assert rid.startswith("resp-")
        assert "_owner" not in body  # internal plumbing stripped
        # Let the background runner finish.
        for _ in range(50):
            await asyncio.sleep(0.01)
            cur = R._get_stored_response(rid)
            if cur and cur.get("status") == "completed":
                break
        cur = R._get_stored_response(rid)
        assert cur is not None and cur["status"] == "completed"
        assert cur["output"][0]["content"][0]["text"] == "done"
        return rid

    asyncio.run(_run())


def test_background_failure_flips_to_failed(monkeypatch):
    R._response_store.clear()

    async def _boom(req, request):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(R, "create_response", _boom)

    async def _run():
        req = ResponsesRequest(model="qwen", input="hi", background=True)
        resp = await R._start_background_response(req, _fake_request())
        rid = _payload_json(resp)["id"]
        for _ in range(50):
            await asyncio.sleep(0.01)
            cur = R._get_stored_response(rid)
            if cur and cur.get("status") == "failed":
                break
        cur = R._get_stored_response(rid)
        assert cur is not None and cur["status"] == "failed"
        assert cur["error"]["code"] == "internal_error"

    asyncio.run(_run())


def test_background_no_store_no_terminal_flips_failed(monkeypatch):
    """If create_response returns without storing a terminal payload (e.g. it
    emitted a 500 JSONResponse), the runner must not leave the entry stuck."""
    R._response_store.clear()

    async def _silent(req, request):
        return None  # never stores → entry stays in_progress

    monkeypatch.setattr(R, "create_response", _silent)

    async def _run():
        req = ResponsesRequest(model="qwen", input="hi", background=True)
        resp = await R._start_background_response(req, _fake_request())
        rid = _payload_json(resp)["id"]
        for _ in range(50):
            await asyncio.sleep(0.01)
            cur = R._get_stored_response(rid)
            if cur and cur.get("status") == "failed":
                break
        cur = R._get_stored_response(rid)
        assert cur is not None and cur["status"] == "failed"

    asyncio.run(_run())
