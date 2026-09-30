"""OpenAI Batch API and Anthropic Message Batches with a fake loopback target."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway import files_store
from yunshu_gateway.routers import batches as batches_router
from yunshu_gateway.routers import files as files_router

ANTH = {"anthropic-version": "2023-06-01"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("YUNSHU_FILES_DIR", str(tmp_path / "files"))
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    files_store.reset_store()
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(
            {"path": request.url.path, "body": body, "headers": dict(request.headers)}
        )
        if body.get("model") == "boom":
            return httpx.Response(
                400,
                json={
                    "error": {"message": "bad model", "type": "invalid_request_error"}
                },
            )
        if request.url.path == "/v1/messages":
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                },
            )
        return httpx.Response(
            200,
            headers={"x-request-id": "rid-1"},
            json={"id": "chatcmpl-1", "choices": [{"message": {"content": "ok"}}]},
        )

    runner = batches_router.runner
    runner.autostart = False
    runner.yield_seconds = 0
    state = {"handler": handler}
    runner.client_factory = lambda base, headers: httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: state["handler"](r)),
        base_url=base,
        headers=headers,
    )
    app = FastAPI()
    app.include_router(files_router.router, prefix="/v1")
    app.include_router(batches_router.router, prefix="/v1")
    client = TestClient(app)
    yield client, calls, state
    runner.autostart = True
    runner.client_factory = batches_router._default_client
    runner.yield_seconds = 0.05
    files_store.reset_store()


def drain():
    asyncio.run(batches_router.runner.process_all())


def jsonl(*rows) -> bytes:
    return b"\n".join(json.dumps(r).encode() for r in rows) + b"\n"


def chat_line(cid, model="m", url="/v1/chat/completions"):
    return {
        "custom_id": cid,
        "method": "POST",
        "url": url,
        "body": {"model": model, "messages": [{"role": "user", "content": "hi"}]},
    }


def upload_batch_file(client, data: bytes, purpose="batch") -> str:
    r = client.post(
        "/v1/files",
        data={"purpose": purpose},
        files={"file": ("in.jsonl", data, "application/x-jsonl")},
    )
    return r.json()["id"]


def create(client, fid, endpoint="/v1/chat/completions", **extra):
    return client.post(
        "/v1/batches",
        json={
            "input_file_id": fid,
            "endpoint": endpoint,
            "completion_window": "24h",
            **extra,
        },
    )


def test_openai_sdk_batch_flow(env):
    client, calls, _ = env
    openai = pytest.importorskip("openai")
    sdk = openai.OpenAI(
        base_url="http://testserver/v1", api_key="x", http_client=client
    )
    f = sdk.files.create(
        file=(
            "in.jsonl",
            jsonl(chat_line("a"), chat_line("b", "boom"), chat_line("c")),
            "application/x-jsonl",
        ),
        purpose="batch",
    )
    b = sdk.batches.create(
        input_file_id=f.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"k": "v"},
    )
    assert (
        b.id.startswith("batch_")
        and b.status == "validating"
        and b.request_counts.total == 3
    )
    assert b.metadata == {"k": "v"} and b.expires_at == b.created_at + 86400
    drain()
    b = sdk.batches.retrieve(b.id)
    assert b.status == "completed" and b.completed_at and b.in_progress_at
    assert (b.request_counts.completed, b.request_counts.failed) == (2, 1)
    out = [json.loads(x) for x in sdk.files.content(b.output_file_id).text.splitlines()]
    assert [o["custom_id"] for o in out] == ["a", "c"]
    assert out[0]["id"].startswith("batch_req_") and out[0]["error"] is None
    assert (
        out[0]["response"]["status_code"] == 200
        and out[0]["response"]["request_id"] == "rid-1"
    )
    assert out[0]["response"]["body"]["id"] == "chatcmpl-1"
    err = [json.loads(x) for x in sdk.files.content(b.error_file_id).text.splitlines()]
    assert err[0]["custom_id"] == "b" and err[0]["response"]["status_code"] == 400
    assert [x.id for x in sdk.batches.list()] == [b.id]
    assert calls[0]["body"]["stream"] is False and len(calls) == 3


def test_openai_bad_requests(env):
    client, _, _ = env
    fid = upload_batch_file(client, jsonl(chat_line("a")))
    assert create(client, fid, "/v1/nope").status_code == 400
    assert (
        client.post(
            "/v1/batches",
            json={
                "input_file_id": fid,
                "endpoint": "/v1/chat/completions",
                "completion_window": "1h",
            },
        ).status_code
        == 400
    )
    assert create(client, "file_" + "0" * 24).status_code == 400
    other = upload_batch_file(client, jsonl(chat_line("a")), purpose="assistants")
    assert create(client, other).status_code == 400
    assert client.post("/v1/batches", content=b"nope").status_code == 400
    assert create(client, fid, metadata={"a": 1}).status_code == 400
    assert client.get("/v1/batches/batch_" + "0" * 24).status_code == 404
    assert client.get("/v1/batches/../x").status_code in (404, 405)


@pytest.mark.parametrize(
    "payload,code",
    [
        (
            b'{"custom_id": "a", "method": "POST", "url": "/v1/chat/completions", "body": {}}\nnot json\n',
            "invalid_json_line",
        ),
        (
            jsonl({"method": "POST", "url": "/v1/chat/completions", "body": {}}),
            "missing_custom_id",
        ),
        (jsonl(chat_line("a"), chat_line("a")), "duplicate_custom_id"),
        (jsonl(chat_line("a", url="/v1/embeddings")), "invalid_url"),
        (b"\n\n", "empty_file"),
    ],
)
def test_bad_jsonl_fails_batch(env, payload, code):
    client, _, _ = env
    b = create(client, upload_batch_file(client, payload)).json()
    assert b["status"] == "failed" and b["failed_at"]
    assert code in [e["code"] for e in b["errors"]["data"]]
    drain()  # nothing to run
    assert client.get(f"/v1/batches/{b['id']}").json()["status"] == "failed"


def test_line_limit(env, monkeypatch):
    client, _, _ = env
    monkeypatch.setattr(batches_router, "MAX_OPENAI_LINES", 3)
    data = jsonl(*[chat_line(f"c{i}") for i in range(5)])
    b = create(client, upload_batch_file(client, data)).json()
    assert b["status"] == "failed"
    assert b["errors"]["data"][0]["code"] == "batch_request_limit_exceeded"


def test_cancel_before_start_and_mid_batch(env):
    client, calls, state = env
    b = create(
        client, upload_batch_file(client, jsonl(chat_line("a"), chat_line("b")))
    ).json()
    c = client.post(f"/v1/batches/{b['id']}/cancel").json()
    assert c["status"] == "cancelling" and c["cancelling_at"]
    drain()
    got = client.get(f"/v1/batches/{b['id']}").json()
    assert got["status"] == "cancelled" and got["cancelled_at"] and not calls
    assert client.post(f"/v1/batches/{b['id']}/cancel").status_code == 409

    b2 = create(
        client,
        upload_batch_file(client, jsonl(*[chat_line(f"r{i}") for i in range(4)])),
    ).json()
    inner = state["handler"]

    def cancelling(request):
        resp = inner(request)
        if len([c for c in calls if c["body"]]) == 2:
            client.post(f"/v1/batches/{b2['id']}/cancel")
        return resp

    state["handler"] = cancelling
    drain()
    got = client.get(f"/v1/batches/{b2['id']}").json()
    assert got["status"] == "cancelled" and got["request_counts"] == {
        "total": 4,
        "completed": 2,
        "failed": 0,
    }
    out = client.get(f"/v1/files/{got['output_file_id']}/content").text.splitlines()
    assert len(out) == 2


def test_resume_after_restart(env):
    client, calls, state = env
    b = create(
        client,
        upload_batch_file(client, jsonl(*[chat_line(f"r{i}") for i in range(4)])),
    ).json()
    inner = state["handler"]

    class Crash(BaseException):
        pass

    def crashing(request):
        if len(calls) >= 2:
            raise Crash
        return inner(request)

    state["handler"] = crashing
    with pytest.raises(Crash):
        drain()
    rec = files_store.get_store().load_batch(b["id"])
    assert rec["status"] == "in_progress" and rec["counts"]["succeeded"] == 2
    files_store.reset_store()  # simulate a fresh process
    state["handler"] = inner
    calls.clear()
    drain()
    got = client.get(f"/v1/batches/{b['id']}").json()
    assert got["status"] == "completed" and got["request_counts"]["completed"] == 4
    assert [c["body"]["messages"][0]["content"] for c in calls] == [
        "hi",
        "hi",
    ]  # only 2 remaining ran
    out = [
        json.loads(x)["custom_id"]
        for x in client.get(
            f"/v1/files/{got['output_file_id']}/content"
        ).text.splitlines()
    ]
    assert out == ["r0", "r1", "r2", "r3"]


def test_openai_list_pagination(env):
    client, _, _ = env
    fid = upload_batch_file(client, jsonl(chat_line("a")))
    ids = [create(client, fid).json()["id"] for _ in range(3)]
    p1 = client.get("/v1/batches?limit=2").json()
    assert p1["has_more"] is True and len(p1["data"]) == 2
    p2 = client.get(f"/v1/batches?limit=2&after={p1['last_id']}").json()
    assert len(p2["data"]) == 1 and p2["has_more"] is False
    assert sorted([x["id"] for x in p1["data"] + p2["data"]]) == sorted(ids)
    assert client.get("/v1/batches?limit=101").status_code == 400


def test_expiry(env):
    client, calls, _ = env
    b = create(client, upload_batch_file(client, jsonl(chat_line("a")))).json()
    store = files_store.get_store()
    rec = store.load_batch(b["id"])
    rec["expires_at"] = int(time.time()) - 1
    store.save_batch(rec)
    drain()
    got = client.get(f"/v1/batches/{b['id']}").json()
    assert got["status"] == "expired" and got["expired_at"] and not calls


def test_end_to_end_with_worker_task(env):
    client, _, _ = env
    batches_router.runner.autostart = True
    with TestClient(client.app) as c:
        b = c.post(
            "/v1/messages/batches",
            headers=ANTH,
            json={
                "requests": [
                    {
                        "custom_id": "x1",
                        "params": {
                            "model": "m",
                            "max_tokens": 5,
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    }
                ]
            },
        ).json()
        for _ in range(100):
            got = c.get(f"/v1/messages/batches/{b['id']}", headers=ANTH).json()
            if got["processing_status"] == "ended":
                break
            time.sleep(0.05)
        assert (
            got["processing_status"] == "ended"
            and got["request_counts"]["succeeded"] == 1
        )
        asyncio.run(batches_router.stop_runner())


# -- Anthropic ------------------------------------------------------------


def areq(cid, model="m"):
    return {
        "custom_id": cid,
        "params": {
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    }


def test_anthropic_batch_flow(env):
    client, calls, _ = env
    r = client.post(
        "/v1/messages/batches",
        headers=ANTH,
        json={"requests": [areq("ok-1"), areq("bad_2", "boom")]},
    )
    assert r.status_code == 200
    b = r.json()
    assert b["id"].startswith("msgbatch_") and b["type"] == "message_batch"
    assert b["processing_status"] == "in_progress" and b["results_url"] is None
    assert b["request_counts"] == {
        "processing": 2,
        "succeeded": 0,
        "errored": 0,
        "canceled": 0,
        "expired": 0,
    }
    assert (
        client.get(f"/v1/messages/batches/{b['id']}/results", headers=ANTH).status_code
        == 409
    )
    drain()
    got = client.get(f"/v1/messages/batches/{b['id']}", headers=ANTH).json()
    assert got["processing_status"] == "ended" and got["ended_at"]
    assert got["request_counts"] == {
        "processing": 0,
        "succeeded": 1,
        "errored": 1,
        "canceled": 0,
        "expired": 0,
    }
    assert (
        got["results_url"] == f"http://testserver/v1/messages/batches/{b['id']}/results"
    )
    res = client.get(got["results_url"], headers=ANTH)
    assert res.headers["content-type"].startswith("application/x-jsonl")
    rows = {j["custom_id"]: j["result"] for j in map(json.loads, res.text.splitlines())}
    assert (
        rows["ok-1"]["type"] == "succeeded"
        and rows["ok-1"]["message"]["role"] == "assistant"
    )
    assert rows["bad_2"]["type"] == "errored"
    assert (
        rows["bad_2"]["error"]["type"] == "error"
        and rows["bad_2"]["error"]["error"]["message"] == "bad model"
    )
    assert (
        calls[0]["headers"]["anthropic-version"] == "2023-06-01"
        and calls[0]["path"] == "/v1/messages"
    )
    lst = client.get("/v1/messages/batches", headers=ANTH).json()
    assert lst["first_id"] == b["id"] and lst["has_more"] is False
    assert client.delete(f"/v1/messages/batches/{b['id']}", headers=ANTH).json() == {
        "id": b["id"],
        "type": "message_batch_deleted",
    }
    assert (
        client.get(f"/v1/messages/batches/{b['id']}", headers=ANTH).status_code == 404
    )


def test_anthropic_validation_and_envelope(env):
    client, _, _ = env
    for payload in (
        {},
        {"requests": []},
        {"requests": [{"custom_id": "bad id!", "params": {}}]},
        {"requests": [areq("a"), areq("a")]},
        {"requests": [{"custom_id": "a", "params": {"model": "m"}}]},
    ):
        r = client.post(
            "/v1/messages/batches", json=payload
        )  # no anthropic-version header
        assert r.status_code == 400
        assert (
            r.json()["type"] == "error"
            and r.json()["error"]["type"] == "invalid_request_error"
        )
    assert (
        client.get("/v1/messages/batches/msgbatch_" + "0" * 24).json()["error"]["type"]
        == "not_found_error"
    )
    assert client.get("/v1/messages/batches?limit=0").status_code == 400


def test_anthropic_cancel_and_delete_rules(env):
    client, calls, _ = env
    b = client.post(
        "/v1/messages/batches", json={"requests": [areq("a"), areq("b")]}
    ).json()
    assert client.delete(f"/v1/messages/batches/{b['id']}").status_code == 400
    c = client.post(f"/v1/messages/batches/{b['id']}/cancel").json()
    assert c["processing_status"] == "canceling" and c["cancel_initiated_at"]
    drain()
    got = client.get(f"/v1/messages/batches/{b['id']}").json()
    assert got["processing_status"] == "ended"
    assert (
        got["request_counts"]["canceled"] == 2
        and got["request_counts"]["processing"] == 0
    )
    rows = [
        json.loads(x)
        for x in client.get(f"/v1/messages/batches/{b['id']}/results").text.splitlines()
    ]
    assert {r["result"]["type"] for r in rows} == {"canceled"} and not calls
    assert client.post(f"/v1/messages/batches/{b['id']}/cancel").status_code == 400


def test_anthropic_list_pagination(env):
    client, _, _ = env
    ids = [
        client.post("/v1/messages/batches", json={"requests": [areq("a")]}).json()["id"]
        for _ in range(3)
    ]
    p1 = client.get("/v1/messages/batches?limit=2").json()
    assert len(p1["data"]) == 2 and p1["has_more"] is True
    p2 = client.get(f"/v1/messages/batches?limit=2&after_id={p1['last_id']}").json()
    assert len(p2["data"]) == 1 and p2["has_more"] is False
    back = client.get(f"/v1/messages/batches?before_id={p2['data'][0]['id']}").json()
    assert [x["id"] for x in back["data"]] == [x["id"] for x in p1["data"]]
    assert sorted(x["id"] for x in p1["data"] + p2["data"]) == sorted(ids)


def test_openai_and_anthropic_batches_are_separate(env):
    client, _, _ = env
    client.post("/v1/messages/batches", json={"requests": [areq("a")]})
    assert client.get("/v1/batches").json()["data"] == []
    assert len(client.get("/v1/messages/batches").json()["data"]) == 1


def test_anthropic_sdk_shapes(env):
    anthropic = pytest.importorskip("anthropic")
    client, _, _ = env
    sdk = anthropic.Anthropic(
        base_url="http://testserver", api_key="x", http_client=client
    )
    b = sdk.messages.batches.create(
        requests=[
            {
                "custom_id": "s1",
                "params": {
                    "model": "m",
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            }
        ]
    )
    drain()
    assert sdk.messages.batches.retrieve(b.id).processing_status == "ended"
    results = list(sdk.messages.batches.results(b.id))
    assert results[0].custom_id == "s1" and results[0].result.type == "succeeded"
