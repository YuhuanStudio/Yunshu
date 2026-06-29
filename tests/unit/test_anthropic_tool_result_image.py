"""Anthropic allows image content inside a tool_result block (a tool returning
a screenshot — vision-tool / computer-use agents). Previously (1) _has_image_blocks didn't
recurse into tool_result so the VLM path was bypassed when the only image lived there, and
(2) the tool_result branch flattened images to the literal "[Image: …]" text → the model
was blind to the tool-returned image. Both must now route the image to the VLM."""

from __future__ import annotations

from yunshu_gateway.routers.anthropic import (
    AnthropicMessage,
    _convert_anthropic_messages,
    _has_image_blocks,
)

_IMG = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
}


def test_has_image_blocks_recurses_into_tool_result():
    # top-level image (existing) still True
    assert _has_image_blocks([_IMG]) is True
    # image nested in a tool_result → now True (was False)
    tr = {"type": "tool_result", "tool_use_id": "toolu_1", "content": [_IMG]}
    assert _has_image_blocks([tr]) is True
    # a text-only tool_result → False
    tr_text = {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": [{"type": "text", "text": "ok"}],
    }
    assert _has_image_blocks([tr_text]) is False


def test_tool_result_image_becomes_image_url_not_placeholder():
    temp_files: list[str] = []
    msgs = [
        AnthropicMessage(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "screenshot",
                    "input": {},
                },
            ],
        ),
        AnthropicMessage(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [{"type": "text", "text": "here is the screen"}, _IMG],
                },
            ],
        ),
    ]
    out, _ = _convert_anthropic_messages(msgs, has_images=True, temp_files=temp_files)
    # The tool message carries the TEXT and the right tool_call_id.
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "toolu_1"
    assert "here is the screen" in tool_msgs[0]["content"]
    # An image_url user message was emitted (the VLM will pick it up) — not "[Image: …]".
    img_parts = [
        p
        for m in out
        if isinstance(m.get("content"), list)
        for p in m["content"]
        if isinstance(p, dict) and p.get("type") == "image_url"
    ]
    assert len(img_parts) == 1
    assert img_parts[0]["image_url"]["url"].startswith("file://")
    # the placeholder literal must NOT appear anywhere in the converted text
    all_text = " ".join(m["content"] for m in out if isinstance(m.get("content"), str))
    assert "[Image:" not in all_text
    # a temp file was registered for cleanup
    assert len(temp_files) == 1
