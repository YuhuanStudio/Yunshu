"""+ 879 (parallel hunt round).

(HIGH): Anthropic /v1/messages/count_tokens dropped image tokens — a text tokenizer
renders an image_url block to ~0 tokens, but the real /messages path routes to the VLM
engine and bills _estimate_image_tokens() per image, so count_tokens grossly undercounted
(defeating client budgeting). Plus a MEDIUM: the chat-template fallback did
str.join([m["content"] ...]) which TypeError'd (→500) when a content was a LIST (image
message). Now: add 576 (== IMAGE_TOKEN_ESTIMATE == VLM default) per converted image, and
coerce non-str content to text in the fallback.
"""

from __future__ import annotations

import inspect

from yunshu_control.token_counter import IMAGE_TOKEN_ESTIMATE
from yunshu_gateway.routers import (
    anthropic as A,  # noqa: N812  # intentional short module alias
)


def test_count_tokens_adds_per_image_estimate():
    src = inspect.getsource(A.count_tokens)
    # per-image token estimate is added to the returned count
    assert "len(_ct_temp_files) * 576" in src
    assert "len(tokens) + _img_tokens" in src
    # 576 is the shared constant (VLM default + gateway budgeting), kept in sync
    assert IMAGE_TOKEN_ESTIMATE == 576


def test_count_tokens_fallback_coerces_list_content():
    src = inspect.getsource(A.count_tokens)
    # the fallback no longer does a raw str.join over possibly-list content
    assert 'text_parts = [m["content"] for m in messages]' not in src
    assert '_extract_text_from_content(m["content"])' in src
