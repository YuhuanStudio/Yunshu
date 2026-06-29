"""the fail-loud invariant (a media content part routed to the VLM
must resolve to a path or RAISE — never silently drop, which desyncs the unconditional
placeholder → masked_scatter crash, hallucination, or image-order shift) only covered
the file:// branches. Extend it to the sibling branches: audio with no data / http url,
and image/image_data parts with no usable url."""
from __future__ import annotations

import pytest

from yunshu_engine.vlm_engine import VLMEngine
from yunshu_gateway.routers.chat import _normalize_image_part

# ── _extract_audio fail-loud ──

@pytest.mark.asyncio
async def test_input_audio_without_data_raises():
    engine = VLMEngine("/models/test")
    messages = [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"format": "wav"}},  # no 'data'
    ]}]
    with pytest.raises(ValueError):
        await engine._extract_audio(messages)


@pytest.mark.asyncio
async def test_audio_url_http_raises():
    engine = VLMEngine("/models/test")
    messages = [{"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url": "https://example.com/a.wav"}},
    ]}]
    with pytest.raises(ValueError):
        await engine._extract_audio(messages)


@pytest.mark.asyncio
async def test_audio_text_only_no_raise():
    engine = VLMEngine("/models/test")
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert await engine._extract_audio(messages) == []


# ── _normalize_image_part fail-loud ──

def test_normalize_image_data_without_url_raises():
    with pytest.raises(ValueError):
        _normalize_image_part({"type": "image_data"})  # no url/image_url/data
    with pytest.raises(ValueError):
        _normalize_image_part({"type": "image", "url": ""})


def test_normalize_valid_image_data_passthrough():
    # A bare base64 blob is wrapped as a data: URL the loader can decode.
    out = _normalize_image_part({"type": "image_data", "data": "QUJD"})
    assert out["type"] == "image_url"
    assert out["image_url"]["url"].startswith("data:image/png;base64,")
    # A normal data: url is preserved.
    out2 = _normalize_image_part({"type": "image", "url": "data:image/png;base64,QUJD"})
    assert out2["image_url"]["url"] == "data:image/png;base64,QUJD"


def test_normalize_non_image_part_untouched():
    p = {"type": "text", "text": "hello"}
    assert _normalize_image_part(p) is p
