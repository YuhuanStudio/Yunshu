"""video input was the un-propagated sibling of the fail-loud
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


def test_http_video_url_fetch_failure_is_not_silently_dropped(monkeypatch):
    async def fail(self, url):
        raise ValueError("http video fetch failed")

    monkeypatch.setattr(VLMEngine, "_download_video", fail)
    with pytest.raises(ValueError, match="http"):
        _run(
            [
                {
                    "type": "video_url",
                    "video_url": {"url": "https://example.com/clip.mp4"},
                }
            ]
        )


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
    """video sibling: data:video;base64,… (subtype-less) had no '/' in the
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
        _run(
            [
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video;base64,{payload}"},
                }
            ]
        )
    assert not isinstance(ei.value, IndexError), (
        "header parse must not raise IndexError"
    )


def test_commaless_data_video_url_fails_loud_valueerror():
    with pytest.raises(ValueError, match="payload separator"):
        _run([{"type": "video_url", "video_url": {"url": "data:video/mp4;base64"}}])


def _engine():
    eng = VLMEngine.__new__(VLMEngine)
    eng._register_temp_file = lambda p: None
    return eng


def _write_mp4(path, n=8):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (64, 64))
    for _ in range(n):
        w.write(np.full((64, 64, 3), (0, 0, 200), dtype=np.uint8))  # BGR red
    w.release()


def test_video_frames_without_ffmpeg_come_from_opencv(tmp_path, monkeypatch):
    """Neither the M5 nor the M3 has ffmpeg: video_url used to log a warning and return no frames,
    so the model answered about a video it never saw. OpenCV (a locked dependency) decodes it."""
    import subprocess

    def no_ffmpeg(*a, **k):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(subprocess, "run", no_ffmpeg)
    mp4 = tmp_path / "v.mp4"
    _write_mp4(mp4)
    frames = asyncio.run(
        _engine()._extract_frames_from_file(str(mp4), fps=1.0, max_frames=4)
    )
    assert 1 <= len(frames) <= 4 and all(f.endswith(".jpg") for f in frames)


def test_undecodable_video_fails_loud_not_silent(tmp_path, monkeypatch):
    import subprocess

    def no_ffmpeg(*a, **k):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(subprocess, "run", no_ffmpeg)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="frames"):
        asyncio.run(
            _engine()._extract_frames_from_file(str(bad), fps=1.0, max_frames=4)
        )
