"""Decoded temp media is deleted when a streamed request fails before generation starts."""

from __future__ import annotations

import asyncio
import base64
import binascii
import os
import threading

import pytest

from yunshu_engine.vlm_engine import VLMEngine

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 64).decode()


def _engine():
    eng = VLMEngine.__new__(VLMEngine)
    eng._temp_files = []
    eng._temp_files_lock = threading.Lock()
    eng._model = object()
    eng._config = {}
    eng._model_path = "local-model"
    created: list[str] = []
    real = eng._register_temp_file

    def spy(path):
        created.append(path)
        real(path)

    eng._register_temp_file = spy
    return eng, created


def _drain(eng, messages):
    async def go():
        async for _ in eng._generate_stream_impl(messages=messages):
            pass

    asyncio.run(go())


def _image(url):
    return {"type": "image_url", "image_url": {"url": url}}


def test_stream_failure_after_first_image_leaves_no_files():
    eng, created = _engine()
    messages = [
        {
            "role": "user",
            "content": [
                _image(f"data:image/png;base64,{PNG}"),
                _image("data:image/png;base64,A"),
            ],
        }
    ]
    with pytest.raises(binascii.Error):
        _drain(eng, messages)
    assert created, "the first image was decoded to a temp file"
    assert not any(os.path.exists(p) for p in created)
    assert not eng._temp_files


def test_stream_rejected_media_leaves_no_files():
    eng, created = _engine()
    messages = [{"role": "user", "content": [_image(f"data:image/png;base64,{PNG}")]}]

    def reject(*a, **k):
        raise ValueError("this model does not accept images")

    eng._check_request_supported = reject
    with pytest.raises(ValueError, match="does not accept"):
        _drain(eng, messages)
    assert created
    assert not any(os.path.exists(p) for p in created)


def test_bad_base64_leaves_no_file_behind():
    eng, _ = _engine()
    before = set(os.listdir(os.path.dirname(_tmp_path())))
    with pytest.raises(binascii.Error):
        asyncio.run(eng._save_base64_image("data:image/png;base64,A"))
    after = set(os.listdir(os.path.dirname(_tmp_path())))
    assert not [f for f in after - before if f.endswith(".png")]


def _tmp_path():
    import tempfile

    return os.path.join(tempfile.gettempdir(), "x")
