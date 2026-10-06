"""POST /v1/responses/{id}/cancel on a running background response must answer status "cancelled"
(the real-server route check got the in_progress snapshot back), and the response stays cancelled
when the generation finishes afterwards."""

from __future__ import annotations

import asyncio
import json
import types

import pytest

from yunshu_gateway.routers import responses as R  # noqa: N812


class Tracker:
    def __init__(self, hit):
        self.hit = hit

    def cancel(self, rid):
        return self.hit

    def get_owner(self, rid):
        return None


def _req():
    r = types.SimpleNamespace()
    r.state = types.SimpleNamespace(role="owner")
    r.headers = {}
    r._actor = ""
    return r


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    import yunshu_control.audit_log as al
    import yunshu_engine.request_tracker as rt

    monkeypatch.setattr(al, "resolve_actor", lambda request: "")
    monkeypatch.setattr(R, "_check_permission", lambda request, perm: None)
    monkeypatch.setattr(rt, "get_request_tracker", lambda: Tracker(True))
    R._response_store.clear()
    yield
    R._response_store.clear()


def _cancel(rid):
    resp = asyncio.run(R.cancel_response(rid, _req()))
    return resp.status_code, json.loads(bytes(resp.body).decode())


@pytest.mark.parametrize("status", ["queued", "in_progress"])
def test_cancel_of_a_running_background_response_says_cancelled(status):
    R._store_response(
        "resp-bg",
        {"id": "resp-bg", "object": "response", "status": status, "output": []},
    )
    code, body = _cancel("resp-bg")
    assert code == 200 and body["status"] == "cancelled"
    assert R._get_stored_response("resp-bg")["status"] == "cancelled"  # a poll agrees


def test_cancel_of_a_finished_response_is_idempotent():
    R._store_response(
        "resp-done",
        {"id": "resp-done", "object": "response", "status": "completed", "output": []},
    )
    code, body = _cancel("resp-done")
    assert code == 200 and body["status"] == "completed"


def test_cancel_unknown_is_404(monkeypatch):
    import yunshu_engine.request_tracker as rt

    monkeypatch.setattr(rt, "get_request_tracker", lambda: Tracker(False))
    code, body = _cancel("resp-nope")
    assert code == 404 and body["error"]["code"] == "response_not_found"


def test_background_failure_path_does_not_overwrite_a_cancel(monkeypatch):
    """The engine raising on cancel made the runner mark the response failed (seen on the real
    server: a poll after the cancel said failed)."""
    import yunshu_gateway.routers.responses as mod

    async def boom(req, request):
        rid = request.state._forced_response_id
        mod._store_response(rid, {"id": rid, "status": "cancelled", "output": []})
        raise RuntimeError("generation cancelled")

    monkeypatch.setattr(mod, "create_response", boom)
    req = mod.ResponsesRequest(model="m", input="hi", background=True)

    async def go():
        import asyncio

        r = await mod._start_background_response(req, _req())
        await asyncio.sleep(0.3)
        return json.loads(bytes(r.body).decode())["id"]

    rid = asyncio.run(go())
    assert mod._get_stored_response(rid)["status"] == "cancelled"
