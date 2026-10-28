"""Responses WebSocket ``store: false``: a failed turn keeps the last good response.

A turn that fails (or is cancelled) after ``response.created`` never reaches the store; it
must not push the last completed ``store: false`` response out of the socket's cache, or the
client's next turn (chained off that response) is a 404.
"""

from __future__ import annotations

import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient

from yunshu_gateway.routers import responses as rsp
from yunshu_gateway.routers import stream_ws
from yunshu_gateway.streaming import ClosingStreamingResponse as StreamingResponse


def test_responses_ws_store_false_failed_turn_keeps_last_good_response():
    app = FastAPI()
    app.include_router(stream_ws.router)
    seq = iter(range(1, 100))

    @app.post("/v1/responses")
    async def create(request: Request):
        body = await request.json()
        prev = body.get("previous_response_id")
        if prev and rsp._get_stored_response(prev) is None:
            return JSONResponse(
                {
                    "error": {
                        "message": "not found",
                        "code": "previous_response_not_found",
                    }
                },
                status_code=404,
            )
        rid = f"resp_wsf_{next(seq)}"
        failing = body.get("input") == "fail"
        if body.get("store") and not failing:
            rsp._store_response(rid, {"id": rid})
        created = {"type": "response.created", "response": {"id": rid}}
        end = {
            "type": "response.failed" if failing else "response.completed",
            "response": {"id": rid, "store": body.get("store")},
        }

        async def gen():
            yield "event: response.created\ndata: " + json.dumps(created) + "\n\n"
            yield f"event: {end['type']}\ndata: " + json.dumps(end) + "\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    def turn(ws, prev=None, text="x"):
        msg = {"type": "response.create", "model": "m", "input": text, "store": False}
        if prev:
            msg["previous_response_id"] = prev
        ws.send_text(json.dumps(msg))
        out = [json.loads(ws.receive_text())]
        while out[-1]["type"] == "response.created":
            out.append(json.loads(ws.receive_text()))
        return out[-1]

    with TestClient(app) as c, c.websocket_connect("/v1/responses") as ws:
        good = turn(ws)["response"]["id"]
        assert turn(ws, good, "fail")["type"] == "response.failed"
        time.sleep(0.3)  # the failed job's cleanup runs after its last event
        assert rsp._get_stored_response(good) is not None
        third = turn(ws, good)
        assert third["type"] == "response.completed", third
