"""Waves 999-1000: VLM media — subtype-less data:video → clean 400 + remote image size cap.

W999 (HIGH): a subtype-less `data:video;base64,…` URL has no '/' in the header, so
  header.split("/")[1] raised IndexError — and unlike the image/audio paths this is NOT a
  ValueError, so it surfaced as an opaque 500 instead of a clean 400. Guarded the comma +
  defensive subtype split, mirroring the W894/W938 image/audio fixes. (Functional regression
  in test_vlm_video_failloud_w874.py.)
W1000 (MEDIUM): _download_image had SSRF/redirect/timeout guards but NO size cap →
  shutil.copyfileobj streamed a multi-GB remote image_url to disk (then PIL into memory),
  bypassing the W953 body-size middleware (out-of-band bytes). Added a Content-Length check +
  byte-count cap (YUNSHU_VLM_MAX_IMAGE_BYTES, default 25MB), and a size violation skips the
  insecure-SSL retry.
"""
from __future__ import annotations

import inspect

from yunshu_engine import vlm_engine


def test_w999_video_subtype_split_is_defensive():
    src = inspect.getsource(vlm_engine.VLMEngine._extract_video_frames)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # the comma separator is guarded (clean ValueError, not an IndexError)
    assert 'if "," not in url:' in code
    assert 'raise ValueError("malformed data:video URL' in code
    # the subtype split is defensive (split("/", 1) + default), not header.split("/")[1]
    assert 'header.split("/", 1)' in code
    assert 'header.split("/")[1]' not in code


def test_w1000_download_has_size_cap_and_skips_insecure_retry_on_oversize():
    src = inspect.getsource(vlm_engine.VLMEngine._download_image)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # a configurable cap is read and enforced
    assert "YUNSHU_VLM_MAX_IMAGE_BYTES" in code
    # Content-Length pre-check AND streaming byte-count enforcement
    assert 'resp.headers.get("Content-Length")' in code
    assert "_written > _cap" in code
    # the unbounded copyfileobj is gone
    assert "shutil.copyfileobj(resp, fh)" not in code
    # a size violation bypasses the insecure-SSL retry (no re-download of the oversized body)
    assert 'if "size limit" in str(e):' in code
