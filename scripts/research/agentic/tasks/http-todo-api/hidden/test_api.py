import http.client
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

TASK = Path(os.environ["AGENTIC_TASK_DIR"])


@pytest.fixture()
def api():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, "server.py", "--port", str(port)],
        cwd=TASK,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("server did not start")

    def call(method, path, body=None, raw=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        data = (
            raw if raw is not None else (json.dumps(body) if body is not None else None)
        )
        c.request(method, path, body=data, headers={"Content-Type": "application/json"})
        r = c.getresponse()
        text = r.read().decode()
        c.close()
        return (
            r.status,
            (json.loads(text) if text else None),
            r.getheader("Content-Type"),
        )

    yield call
    proc.kill()
    proc.wait()


def test_health(api):
    st, body, ct = api("GET", "/health")
    assert st == 200 and body == {"status": "ok"} and "json" in ct


def test_create_and_get(api):
    st, body, _ = api("POST", "/todos", {"title": "  buy milk "})
    assert (
        st == 201
        and body["id"] == 1
        and body["done"] is False
        and body["title"].strip() == "buy milk"
    )
    st, b2, _ = api("POST", "/todos", {"title": "b", "done": True})
    assert st == 201 and b2["id"] == 2 and b2["done"] is True
    assert api("GET", "/todos/1")[1]["id"] == 1
    st, lst, _ = api("GET", "/todos")
    assert st == 200 and [t["id"] for t in lst] == [1, 2]


def test_filter_done(api):
    api("POST", "/todos", {"title": "a"})
    api("POST", "/todos", {"title": "b", "done": True})
    assert [t["id"] for t in api("GET", "/todos?done=true")[1]] == [2]
    assert [t["id"] for t in api("GET", "/todos?done=false")[1]] == [1]
    assert api("GET", "/todos?done=maybe")[0] == 400


def test_patch(api):
    api("POST", "/todos", {"title": "a"})
    st, body, _ = api("PATCH", "/todos/1", {"done": True, "junk": 1})
    assert st == 200 and body == {"id": 1, "title": "a", "done": True}
    st, body, _ = api("PATCH", "/todos/1", {"title": "renamed"})
    assert body == {"id": 1, "title": "renamed", "done": True}
    assert api("PATCH", "/todos/1", {"done": "yes"})[0] == 400
    assert api("PATCH", "/todos/1", {"title": ""})[0] == 400
    assert api("PATCH", "/todos/9", {"done": True})[0] == 404


def test_delete_and_ids_not_reused(api):
    api("POST", "/todos", {"title": "a"})
    api("POST", "/todos", {"title": "b"})
    conn = http.client.HTTPConnection
    st, body, _ = api("DELETE", "/todos/2")
    assert st == 204 and body is None
    assert api("GET", "/todos/2")[0] == 404
    assert api("DELETE", "/todos/2")[0] == 404
    st, body, _ = api("POST", "/todos", {"title": "c"})
    assert body["id"] == 3
    assert conn is not None


def test_validation_errors(api):
    for bad in (
        {},
        {"title": ""},
        {"title": "   "},
        {"title": 5},
        {"title": "x", "done": "no"},
    ):
        st, body, ct = api("POST", "/todos", bad)
        assert st == 400 and "error" in body and "json" in ct, bad
    assert api("POST", "/todos", raw="{not json")[0] == 400
    assert api("POST", "/todos", raw="[1, 2]")[0] == 400
    assert api("PATCH", "/todos/1", raw="nope")[0] in (400, 404)


def test_not_found_and_method(api):
    assert api("GET", "/nope")[0] == 404
    assert api("GET", "/todos/abc")[0] == 404
    st, body, _ = api("GET", "/todos/7")
    assert st == 404 and "error" in body
    assert api("PUT", "/todos")[0] == 405
    assert api("POST", "/health", {})[0] == 405
