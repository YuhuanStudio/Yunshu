"""Log ring (redacted, bounded, incremental), its routes and the diagnostics bundle route."""

from __future__ import annotations

import asyncio
import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway import log_ring
from yunshu_gateway.routers import admin_logs


@pytest.fixture
def ring():
    log_ring.uninstall()
    h = log_ring.install(capacity=50)
    lg = logging.getLogger("yunshu_test.logs")
    old = lg.level
    lg.setLevel(logging.DEBUG)
    yield h, lg
    lg.setLevel(old)
    log_ring.uninstall()


@pytest.fixture
def client(ring, monkeypatch):
    monkeypatch.setattr(
        "yunshu_gateway.routers.models._check_permission", lambda *a: None
    )
    app = FastAPI()
    app.include_router(admin_logs.router, prefix="/v1")
    return TestClient(app), ring[1]


def test_records_carry_level_time_logger_message_and_ids(client):
    c, lg = client
    lg.info("loaded %s in %d ms", "model", 12)
    lg.warning("slow")
    j = c.get("/v1/yunshu/logs").json()
    msgs = [r["msg"] for r in j["records"]]
    assert msgs == ["loaded model in 12 ms", "slow"]
    r = j["records"][0]
    assert set(r) == {"id", "t", "level", "logger", "msg"}
    assert r["level"] == "INFO" and r["logger"] == "yunshu_test.logs" and r["t"] > 1e9
    assert j["next_id"] == j["records"][-1]["id"] and j["dropped"] == 0


def test_level_since_id_q_and_limit_filters(client):
    c, lg = client
    lg.debug("d-one")
    lg.info("i-two")
    lg.error("e-three needle")
    lg.warning("w-four")
    get = lambda **kw: c.get("/v1/yunshu/logs", params=kw).json()  # noqa: E731
    assert [r["level"] for r in get(level="warning")["records"]] == ["ERROR", "WARNING"]
    assert [r["msg"] for r in get(q="NEEDLE")["records"]] == ["e-three needle"]
    first = get()["records"][0]["id"]
    assert [r["msg"] for r in get(since_id=first + 1)["records"]] == [
        "e-three needle",
        "w-four",
    ]
    assert [r["msg"] for r in get(limit=2)["records"]] == ["e-three needle", "w-four"]
    future = get(since=9e12)
    assert future["records"] == [] and future["next_id"] >= 4
    assert c.get("/v1/yunshu/logs?level=loud").status_code == 400
    assert c.get("/v1/yunshu/logs?limit=0").status_code == 422


def test_secrets_and_payloads_are_redacted_when_emitted(client):
    c, lg = client
    lg.error("auth failed Authorization: Bearer sk-abcdef0123456789abcdef0123456789")
    lg.error("validation: input_value={'content': 'my secret prompt'}, input_type=dict")
    lg.error("body: {'prompt': 'tell me a secret'}")
    blob = json.dumps(c.get("/v1/yunshu/logs").json())
    assert "sk-abcdef" not in blob
    assert "my secret prompt" not in blob and "tell me a secret" not in blob


def test_ring_is_bounded_and_counts_drops(client, ring):
    c, lg = client
    for i in range(80):
        lg.info("m%d", i)
    j = c.get("/v1/yunshu/logs?limit=2000").json()
    assert len(j["records"]) == 50 and j["records"][0]["msg"] == "m30"
    assert j["dropped"] == 30 and j["capacity"] == 50


def test_long_messages_are_cut(client):
    c, lg = client
    lg.info("x" * 10_000)
    assert len(c.get("/v1/yunshu/logs").json()["records"][0]["msg"]) == 2048


def test_exception_text_is_not_leaked_only_its_type(client):
    c, lg = client
    try:
        raise ValueError("secret detail in the exception")
    except ValueError:
        lg.exception("boom")
    msg = c.get("/v1/yunshu/logs").json()["records"][0]["msg"]
    assert msg == "boom [ValueError]"


def test_install_is_idempotent_and_attaches_to_the_root_once():
    log_ring.uninstall()
    a, b = log_ring.install(), log_ring.install()
    assert a is b
    assert (
        sum(isinstance(h, log_ring.RingHandler) for h in logging.getLogger().handlers)
        == 1
    )
    log_ring.uninstall()
    assert log_ring.handler() is None


class _Req:
    def __init__(self):
        self.calls = 0

    async def is_disconnected(self):
        self.calls += 1
        return self.calls > 3


def test_sse_tail_emits_new_records_only(ring, monkeypatch):
    h, lg = ring
    lg.info("old line")
    monkeypatch.setattr(admin_logs, "_STREAM_POLL_S", 0.0)
    monkeypatch.setattr(
        "yunshu_gateway.routers.models._check_permission", lambda *a: None
    )

    async def run():
        resp = await admin_logs.logs_stream(_Req(), level="info", q=None, since_id=None)
        lg.info("new line")
        out = []
        async for chunk in resp.body_iterator:
            out.append(chunk)
        return out

    chunks = asyncio.run(run())
    assert resp_ok(chunks) == ["new line"]


def resp_ok(chunks):
    out = []
    for c in chunks:
        for line in c.splitlines():
            if line.startswith("data: "):
                out.append(json.loads(line[6:])["msg"])
    return out


def test_sse_route_headers_and_bad_level(client):
    c, _ = client
    assert c.get("/v1/yunshu/logs/stream?level=loud").status_code == 400


@pytest.mark.parametrize("path", ["/v1/yunshu/logs", "/v1/yunshu/logs/stream"])
def test_logs_need_admin(path, monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    app = FastAPI()
    app.include_router(admin_logs.router, prefix="/v1")
    assert TestClient(app).get(path).status_code == 401
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "tok")
    c = TestClient(app)
    assert c.get(path, headers={"Authorization": "Bearer bad"}).status_code == 401
