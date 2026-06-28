"""/v1/video/generations had the W818/W823/W824 wrong-model keystone, unswept.
After matching req.model and a get_engine fallback, it grabbed the FIRST loaded
VideoEngine — ignoring req.model. Since _check_model_access(req.model) is verified above,
that served a model the key may NOT be authorized for (W801 isolation class). Now: when
video models ARE loaded but none matches, 404 (don't serve a different one); only fall
back to the standalone default engine when NO video model is registered."""
from __future__ import annotations

import inspect
import os
import types

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def _client(monkeypatch):
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    from yunshu_engine.video_engine import VideoEngine
    from yunshu_gateway.main import create_app
    from yunshu_gateway.routers import video as video_mod

    a = VideoEngine.__new__(VideoEngine)
    b = VideoEngine.__new__(VideoEngine)
    entries = [
        types.SimpleNamespace(model_id="vid-A", engine=a, is_loaded=True),
        types.SimpleNamespace(model_id="vid-B", engine=b, is_loaded=True),
    ]

    async def _get_engine(m):
        raise KeyError(m)

    mgr = types.SimpleNamespace(list_entries=lambda: entries, get_engine=_get_engine)
    # video.py does `from ..engine import get_model_manager` at module scope, so the
    # binding lives in the router module's namespace — patch it there.
    monkeypatch.setattr(video_mod, "get_model_manager", lambda: mgr)
    yield TestClient(create_app(), raise_server_exceptions=False)
    os.environ.pop("YUNSHU_AUTH_DISABLED", None)


def test_unmatched_model_404s_not_wrong_model(_client):
    # two video models loaded, request a THIRD → must 404, NOT serve vid-A
    resp = _client.post("/v1/video/generations", json={
        "model": "vid-C-not-loaded",
        "prompt": "a cat playing",
    })
    assert resp.status_code == 404, resp.text
    assert "vid-C-not-loaded" in resp.text


def test_single_model_mismatch_not_404(monkeypatch):
    # exactly one video model loaded under a DIFFERENT id than the default
    # req.model — must NOT 404 (selection serves the lone engine; generation may then fail
    # with a different status, but the model was found).
    import os
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    from yunshu_engine.video_engine import VideoEngine
    from yunshu_gateway.main import create_app
    from yunshu_gateway.routers import video as video_mod

    only = VideoEngine.__new__(VideoEngine)
    entries = [types.SimpleNamespace(model_id="my-custom-video", engine=only, is_loaded=True)]

    async def _get_engine(m):
        raise KeyError(m)

    mgr = types.SimpleNamespace(list_entries=lambda: entries, get_engine=_get_engine)
    monkeypatch.setattr(video_mod, "get_model_manager", lambda: mgr)
    client = TestClient(create_app(), raise_server_exceptions=False)
    resp = client.post("/v1/video/generations", json={"prompt": "a cat"})  # default model id
    os.environ.pop("YUNSHU_AUTH_DISABLED", None)
    assert resp.status_code != 404, f"single loaded video model must not 404: {resp.text}"


def test_source_has_keystone_guard():
    from yunshu_gateway.routers import video
    src = inspect.getsource(video.create_video)
    # ≥2 loaded + no match → 404; exactly 1 loaded → served (W835 single-model fallback)
    assert "_loaded_video" in src
    assert "len(_loaded_video) == 1" in src
    assert "len(_loaded_video) >= 2" in src
