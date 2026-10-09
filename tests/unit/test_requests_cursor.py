"""The engine's finished-request ring can be followed by cursor: the console process reads it every
second and must neither miss nor repeat a request, across polls and across an engine restart."""

from __future__ import annotations

import threading
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway import x_yunshu
from yunshu_gateway.routers.yunshu import router


def entry(i: int) -> dict:
    return {
        "t": time.time(),
        "request_id": f"r{i}",
        "prompt_tokens": 1,
        "completion_tokens": 1,
    }


def test_entries_get_rising_sequence_numbers_and_the_cursor_filters():
    reg = x_yunshu._Registry()
    for i in range(5):
        reg.record_done(entry(i))
    rows, latest = reg.recent_cursor(limit=100)
    assert [r["seq"] for r in rows] == [5, 4, 3, 2, 1] and latest == 5
    rows, latest = reg.recent_cursor(limit=100, after_seq=3)
    assert [r["request_id"] for r in rows] == ["r4", "r3"] and latest == 5
    assert reg.recent_cursor(after_seq=5)[0] == []


def test_a_reader_following_the_cursor_sees_every_request_exactly_once_under_load():
    reg = x_yunshu._Registry()
    total = 3000
    done = threading.Event()

    def writer() -> None:
        for i in range(total):
            reg.record_done(entry(i))
            if i % 50 == 0:
                time.sleep(0.0005)
        done.set()

    seen: list[str] = []
    cursor: int | None = None
    thread = threading.Thread(target=writer)
    thread.start()
    while True:
        finished = done.is_set()
        rows, latest = reg.recent_cursor(limit=512, after_seq=cursor)
        seen.extend(r["request_id"] for r in reversed(rows))
        cursor = latest
        if finished and not rows:
            break
    thread.join()
    assert seen == [f"r{i}" for i in range(total)], (
        "none skipped, none repeated, in order"
    )


def test_the_route_reports_the_cursor_and_the_boot_id():
    x_yunshu.registry.clear()
    x_yunshu.registry._seq = 0
    for i in range(3):
        x_yunshu.registry.record_done(entry(i))
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    import os

    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    try:
        c = TestClient(app)
        body = c.get("/v1/yunshu/requests/recent").json()
        assert body["boot_id"] == x_yunshu.BOOT_ID and body["latest_seq"] == 3
        after = c.get("/v1/yunshu/requests/recent", params={"after_seq": 2}).json()
        assert [r["request_id"] for r in after["data"]] == ["r2"] and after[
            "latest_seq"
        ] == 3
        assert (
            c.get("/v1/yunshu/requests/recent", params={"after_seq": -1}).status_code
            == 422
        )
    finally:
        os.environ.pop("YUNSHU_AUTH_DISABLED", None)
        x_yunshu.registry.clear()


def test_the_status_names_the_engine_process():
    import os

    app = FastAPI()
    app.include_router(router, prefix="/v1")
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    try:
        body = TestClient(app).get("/v1/yunshu/status").json()
        assert body["pid"] == os.getpid()
    finally:
        os.environ.pop("YUNSHU_AUTH_DISABLED", None)
