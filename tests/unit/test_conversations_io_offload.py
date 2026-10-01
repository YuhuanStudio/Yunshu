"""B25: conversation store I/O runs off the event loop; RMW stays atomic."""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway import conversations_store as cs
from yunshu_gateway.routers import conversations as conv_router


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("YUNSHU_CONVERSATIONS_DIR", str(tmp_path / "convs"))
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    cs.reset_store()
    a = FastAPI()
    a.include_router(conv_router.router, prefix="/v1")
    yield a
    cs.reset_store()


def test_slow_fsync_does_not_block_event_loop(app, monkeypatch):
    real_fsync = os.fsync

    def slow_fsync(fd):
        time.sleep(0.4)
        real_fsync(fd)

    monkeypatch.setattr(cs.os, "fsync", slow_fsync)

    async def main():
        ticks = 0
        stop = False

        async def heartbeat():
            nonlocal ticks
            while not stop:
                await asyncio.sleep(0.02)
                ticks += 1

        hb = asyncio.create_task(heartbeat())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as ac:
            r = await ac.post("/v1/conversations", json={})
        stop = True
        await hb
        return r.status_code, ticks

    status, ticks = asyncio.run(main())
    assert status == 200
    assert ticks >= 10  # a blocked loop gives ~0-1 ticks


def test_concurrent_add_items_atomic(app):
    client = TestClient(app)
    cid = client.post("/v1/conversations", json={}).json()["id"]

    def add(i):
        return client.post(
            f"/v1/conversations/{cid}/items",
            json={"items": [{"role": "user", "content": f"t{i}"}]},
        ).status_code

    with ThreadPoolExecutor(8) as ex:
        assert set(ex.map(add, range(16))) == {200}
    got = client.get(f"/v1/conversations/{cid}/items?limit=100").json()
    assert len(got["data"]) == 16
