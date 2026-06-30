"""Omni SSE endpoint multimodal-input resolution: local path / data: URI / URL.

The /v1/omni/speech/stream endpoint accepts image_path/audio_path as a local path
(unchanged), a base64 data: URI, or an http(s) URL. URL/base64 media is decoded to
a temp file (cleaned up after the stream); bad input raises a clean HTTP 4xx.
"""

import asyncio
import base64
import os

import pytest
from fastapi import HTTPException

import yunshu_gateway.routers.omni as omni
from yunshu_gateway.routers.omni import _cleanup_media, _resolve_media


def _run(coro):
    return asyncio.run(coro)


def test_local_path_passthrough():
    tmp: list[str] = []
    assert _run(_resolve_media("/data/scene.png", "image", tmp)) == "/data/scene.png"
    assert tmp == []  # no temp file created for a plain path


def test_none_passthrough():
    assert _run(_resolve_media(None, "image", [])) is None


def test_data_uri_image_decoded_to_tempfile():
    raw = b"\x89PNG\r\n fake image bytes"
    uri = "data:image/png;base64," + base64.b64encode(raw).decode()
    tmp: list[str] = []
    path = _run(_resolve_media(uri, "image", tmp))
    assert path.endswith(".png") and os.path.exists(path)
    with open(path, "rb") as f:
        assert f.read() == raw
    assert tmp == [path]
    _cleanup_media(tmp)
    assert not os.path.exists(path)  # cleaned up


def test_data_uri_audio_extension():
    uri = "data:audio/wav;base64," + base64.b64encode(b"RIFFfake").decode()
    tmp: list[str] = []
    path = _run(_resolve_media(uri, "audio", tmp))
    assert path.endswith(".wav")
    _cleanup_media(tmp)


def test_bad_base64_raises_400():
    tmp: list[str] = []
    with pytest.raises(HTTPException) as e:
        _run(_resolve_media("data:image/png;base64,@@not-base64@@", "image", tmp))
    assert e.value.status_code == 400
    assert tmp == []


def test_data_uri_without_base64_marker_raises_400():
    with pytest.raises(HTTPException) as e:
        _run(_resolve_media("data:image/png,rawnotbase64", "image", []))
    assert e.value.status_code == 400


def test_size_cap_raises_413(monkeypatch):
    monkeypatch.setattr(omni, "_MEDIA_MAX_BYTES", 4)
    uri = "data:image/png;base64," + base64.b64encode(b"too-long-payload").decode()
    tmp: list[str] = []
    with pytest.raises(HTTPException) as e:
        _run(_resolve_media(uri, "image", tmp))
    assert e.value.status_code == 413
