"""Two image/video generation API-contract consistency fixes.
(1) /v1/images/variations + /edits accepted response_format="url" but always returned
    b64_json only (the other 4 image endpoints honor url).
(2) /v1/video/generations rejected a data-URL-prefixed base64 image that every image
    endpoint accepts (it now reuses _decode_image_b64)."""

from __future__ import annotations

import pathlib

from yunshu_gateway.routers.images import _decode_image_b64


def test_decode_image_b64_strips_data_uri():
    assert _decode_image_b64("data:image/png;base64,QUJD") == b"ABC"
    assert _decode_image_b64("QUJD") == b"ABC"  # raw still works


def test_variations_edits_honor_response_format():
    """The unconditional b64_json-only block must be gone; the url branch present."""
    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "python/yunshu_gateway/routers/images.py").read_text()
    # The old unconditional block (no response_format branch) must no longer exist.
    bad = '        for img in images:\n            b64 = base64.b64encode(img).decode("ascii")\n            data.append({"b64_json": b64})'
    assert bad not in src
    # Both endpoints now branch on response_format.
    assert src.count("honor response_format") == 2  # variations + edits


def test_video_uses_shared_image_decoder():
    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "python/yunshu_gateway/routers/video.py").read_text()
    assert "_decode_image_b64(req.image)" in src
    # the old prefix-rejecting inline decode is gone
    assert "base64.b64decode(req.image, validate=True)" not in src
