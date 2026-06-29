"""/v1/ocr had NO client-disconnect cancellation — the cancel
keystone (task.cancel can't stop a running generate on the max_workers=1 executor; the
engine must see its cancel_event) was propagated to VLM chat and audio/images, but
OCR + video were the missed single-shot-media siblings (grep: ocr.py had 0 cancel refs vs
audio 9 / images 6). Now OCR registers with the request tracker, runs both engine calls
under run_with_disconnect_guard, passes cancel_event to the VLM fallback (which honors it
between decode steps → real mid-gen stop), and unregisters in finally. The native path uses
a blocking mlx_vlm.generate (bounded, not interruptible) but the guard still frees the
handler + tracker slot promptly on disconnect.
"""
from __future__ import annotations

import inspect
import os
import threading
import types

import pytest
from fastapi.testclient import TestClient


class _FakeOCR:
    async def extract_text(self, path, language=None, task="text"):
        return {"text": "hello", "confidence": None,
                "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


class _Gen:
    def __init__(self):
        self.cancel_event = threading.Event()


class _Tracker:
    def __init__(self):
        self.registered = []
        self.unregistered = []

    def register(self, rid, model):
        self.registered.append((rid, model))
        return _Gen()

    def unregister(self, rid):
        self.unregistered.append(rid)


@pytest.fixture
def _client(monkeypatch):
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    from yunshu_engine import request_tracker as rt_mod
    from yunshu_engine.ocr_engine import OCREngine
    from yunshu_gateway import engine as eng_mod

    a = OCREngine.__new__(OCREngine)
    a.extract_text = _FakeOCR().extract_text
    mgr = types.SimpleNamespace(
        list_entries=lambda: [types.SimpleNamespace(model_id="ocr-A", engine=a, is_loaded=True)],
    )
    monkeypatch.setattr(eng_mod, "get_model_manager", lambda: mgr)

    tracker = _Tracker()
    monkeypatch.setattr(rt_mod, "get_request_tracker", lambda: tracker)

    from yunshu_gateway.main import create_app
    client = TestClient(create_app(), raise_server_exceptions=False)
    client._yunshu_tracker = tracker
    yield client
    os.environ.pop("YUNSHU_AUTH_DISABLED", None)


def test_ocr_registers_and_unregisters_with_tracker(_client):
    resp = _client.post("/v1/ocr",
                        data={"model": "ocr-A", "task": "text"},
                        files={"file": ("x.png", b"\x89PNG\r\n\x1a\n" + b"0" * 32, "image/png")})
    assert resp.status_code == 200, resp.text
    assert resp.json()["text"] == "hello"
    tracker = _client._yunshu_tracker
    # registered exactly once for this request, and unregistered in finally (no leak)
    assert len(tracker.registered) == 1
    assert tracker.registered[0][1] == "ocr-A"
    assert tracker.unregistered == [tracker.registered[0][0]]


def test_ocr_source_wires_disconnect_guard_and_fallback_cancel():
    from yunshu_gateway.routers import ocr
    src = inspect.getsource(ocr)
    # both engine calls run under the disconnect guard
    assert src.count("run_with_disconnect_guard(") >= 2
    # the VLM fallback receives cancel_event (real mid-gen cancellation)
    assert "cancel_event=_ocr_cancel" in src
    # registration + unregistration present
    assert ".register(_ocr_id" in src
    assert "_ocr_tracker.unregister(_ocr_id)" in src
