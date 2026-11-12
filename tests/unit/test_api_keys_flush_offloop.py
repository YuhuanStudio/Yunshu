"""Usage flush: a due write happens off the calling (event-loop) thread, once at a time."""

import json
import threading

import pytest

from yunshu_gateway import api_keys
from yunshu_gateway.api_keys import Principal


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    s = api_keys.configure(tmp_path / "keys.json")
    yield s
    api_keys._store = None


def test_due_flush_runs_off_the_calling_thread(store, monkeypatch):
    rec, _secret = store.create("ci")
    principal = Principal(
        kind="key", scopes=frozenset({"infer"}), key_id=rec.id, name=rec.name
    )
    store.admit(principal)
    writes = []
    real_write = store._write_private

    def recording_write(path, payload):
        writes.append(threading.current_thread())
        real_write(path, payload)

    monkeypatch.setattr(store, "_write_private", recording_write)
    store._last_flush = 0.0  # the flush interval has passed

    store.account(rec.id, prompt=7, completion=3)

    store._flush_thread.join(5)
    assert writes, "the due usage write did not happen"
    assert all(t is not threading.main_thread() for t in writes)
    usage = json.loads(store.usage_path.read_text())
    assert usage["keys"][rec.id]["days"]


def test_flush_on_the_loop_is_not_repeated_while_a_write_runs(store, monkeypatch):
    rec, _secret = store.create("ci")
    principal = Principal(
        kind="key", scopes=frozenset({"infer"}), key_id=rec.id, name=rec.name
    )
    store.admit(principal)
    gate = threading.Event()
    starts = []
    real_write = store._write_private

    def slow_write(path, payload):
        starts.append(1)
        gate.wait(5)
        real_write(path, payload)

    monkeypatch.setattr(store, "_write_private", slow_write)
    store._last_flush = 0.0
    store.account(rec.id, prompt=1, completion=1)
    store.account(rec.id, prompt=1, completion=1)  # returns at once, no second thread
    gate.set()
    store._flush_thread.join(5)
    assert len(starts) == 1
