"""(MED): the OpenAI bare-STRING image_url variant 500'd on the inference path.

A client may send {"type":"image_url","image_url":"https://…"} (a bare string, not the
{"url":…} object). The engine's _extract_images does part["image_url"].get("url") →
AttributeError → opaque 500. coerced this only in _check_has_media (the SSRF pre-check),
NOT in _normalize_image_part on the path that actually feeds the engine. Now the gateway
normalizes the bare string to the canonical object, and the engine extraction guards against
a non-dict image_url defensively.
"""
from __future__ import annotations

import inspect

from yunshu_engine import vlm_engine
from yunshu_gateway.routers.chat import _normalize_image_part


def test_bare_string_image_url_coerced_to_object():
    part = {"type": "image_url", "image_url": "https://example.com/cat.png"}
    out = _normalize_image_part(part)
    assert isinstance(out["image_url"], dict)
    assert out["image_url"]["url"] == "https://example.com/cat.png"
    assert out["type"] == "image_url"


def test_object_form_image_url_unchanged():
    part = {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
    out = _normalize_image_part(part)
    assert out["image_url"] == {"url": "https://example.com/x.png"}


def test_image_and_image_data_types_still_normalized():
    # the pre-existing behavior must be preserved
    assert _normalize_image_part(
        {"type": "image", "url": "https://e/x.png"})["image_url"]["url"] == "https://e/x.png"
    assert "abc" in _normalize_image_part(
        {"type": "image_data", "data": "abc"})["image_url"]["url"]


def test_engine_extraction_guards_non_dict_image_url():
    # the engine extraction must not do .get("url") on a bare string (AttributeError → 500)
    src = inspect.getsource(vlm_engine.VLMEngine._extract_images)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert 'part.get("image_url", {}).get("url", "")' not in code  # the crashing form is gone
    assert "isinstance(_iu, dict)" in code and "isinstance(_iu, str)" in code
