"""video input was the un-propagated sibling of the W724/W792 fail-loud
invariant. An http(s) video_url had NO handling branch → silently dropped; a rejected /
nonexistent file://, bare-path, or video_file.file_id was log-and-continue. So the model
answered about a video it never saw (hallucination), while image+audio raise ValueError on
the same conditions. Now a REFERENCED-but-unloadable video fails loud.
"""
from __future__ import annotations

import asyncio

import pytest

from yunshu_engine.vlm_engine import VLMEngine


def _run(parts):
    eng = VLMEngine.__new__(VLMEngine)
    msgs = [{"role": "user", "content": parts}]
    return asyncio.run(eng._extract_video_frames(msgs))


def test_http_video_url_fails_loud():
    with pytest.raises(ValueError, match="http"):
        _run([{"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}}])


def test_empty_video_url_fails_loud():
    with pytest.raises(ValueError):
        _run([{"type": "video_url", "video_url": {"url": ""}}])


def test_empty_video_file_id_fails_loud():
    with pytest.raises(ValueError):
        _run([{"type": "video_file", "video_file": {"file_id": ""}}])


def test_no_video_parts_returns_empty_not_raise():
    # a message with only text must NOT raise (no video referenced)
    assert _run([{"type": "text", "text": "hello"}]) == []


def test_subtypeless_data_video_url_is_clean_valueerror_not_indexerror():
    """W938 video sibling: data:video;base64,… (subtype-less) had no '/' in the
    header → header.split('/')[1] raised IndexError, which is NOT a ValueError, so
    the gateway returned an opaque 500 instead of a clean 400. After the fix the
    header is split defensively (default ext mp4); the only way to fail here is the
    base64 decode, which raises ValueError (binascii.Error) — never IndexError."""
    # A subtype-less data:video URL must NOT raise IndexError during header parse.
    # (A bare __new__ instance lacks the temp-file registry, so the *decode/save*
    # step raises AttributeError — but the point is the header parse is reached at
    # all, i.e. no IndexError; pre-fix the IndexError fired BEFORE _save_base64_file.)
    import base64
    payload = base64.b64encode(b"\x00\x00\x00\x18ftypmp42").decode()
    with pytest.raises(Exception) as ei:
        _run([{"type": "video_url", "video_url": {"url": f"data:video;base64,{payload}"}}])
    assert not isinstance(ei.value, IndexError), "header parse must not raise IndexError"


def test_commaless_data_video_url_fails_loud_valueerror():
    with pytest.raises(ValueError, match="payload separator"):
        _run([{"type": "video_url", "video_url": {"url": "data:video/mp4;base64"}}])
