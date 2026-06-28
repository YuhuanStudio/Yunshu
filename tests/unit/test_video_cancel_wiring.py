"""/v1/video/generations had NO client-disconnect cancellation — the W758/W845
cancel keystone (audio/images had it, OCR closed in W859), and video was the last
single-shot-media sibling (grep: video.py 0 cancel refs). Video generation is expensive
(frames × diffusion steps), so an abandoned request kept the GPU busy to completion.

Now both paths register with the request tracker. The STREAMING path runs through
with_sse_keepalive, which on disconnect sets cancel_event AND aclose()s the source
generator → fires generate_stream's finally → its internal _cancel.set() → the executor's
diffusion/frame loop stops promptly (REAL cancellation). The NON-streaming path runs under
run_with_disconnect_guard (frees the handler promptly; generate() has no mid-gen hook so
the bounded diffusion finishes on the executor — same honest limit as OCR native).
"""
from __future__ import annotations

import inspect
import os
import threading
import types

import pytest
from fastapi.testclient import TestClient


class _Result:
    method = "native"
    video_data = b"\x00\x00\x00\x18ftypmp42"
    frames = []
    num_frames = 1
    fps = 16
    width = 64
    height = 64


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
    from yunshu_engine.video_engine import VideoEngine
    from yunshu_gateway.main import create_app
    from yunshu_gateway.routers import video as video_mod

    eng = VideoEngine.__new__(VideoEngine)

    async def _generate(**kwargs):
        return _Result()

    eng.generate = _generate
    entries = [types.SimpleNamespace(model_id="vid-A", engine=eng, is_loaded=True)]

    async def _get_engine(m):
        raise KeyError(m)

    mgr = types.SimpleNamespace(list_entries=lambda: entries, get_engine=_get_engine)
    monkeypatch.setattr(video_mod, "get_model_manager", lambda: mgr)

    tracker = _Tracker()
    monkeypatch.setattr(rt_mod, "get_request_tracker", lambda: tracker)

    client = TestClient(create_app(), raise_server_exceptions=False)
    client._yunshu_tracker = tracker
    yield client
    os.environ.pop("YUNSHU_AUTH_DISABLED", None)


def test_nonstream_video_registers_and_unregisters(_client):
    resp = _client.post("/v1/video/generations", json={"model": "vid-A", "prompt": "a cat"})
    assert resp.status_code == 200, resp.text
    tracker = _client._yunshu_tracker
    assert len(tracker.registered) == 1
    assert tracker.registered[0][1] == "vid-A"
    # unregistered in finally — no tracker leak
    assert tracker.unregistered == [tracker.registered[0][0]]


def test_source_wires_both_paths():
    from yunshu_gateway.routers import video
    src = inspect.getsource(video.create_video)
    # streaming → with_sse_keepalive (disconnect → aclose → internal _cancel)
    assert "with_sse_keepalive(" in src
    # non-streaming → run_with_disconnect_guard
    assert "run_with_disconnect_guard(" in src
    # register + unregister present on both paths
    assert ".register(_vid_id" in src
    assert src.count("_vid_tracker.unregister(_vid_id)") >= 2
