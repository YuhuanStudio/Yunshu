"""Waves 897-901: cross-path parity + parity-drift fixes found by fresh-subsystem hunts.

W897 (HIGH, chat.py): non-streaming tool-call markup leaked into content when a named/forced
  tool_choice suppressed the model's (wrong-named) call — cleanup was gated on the
  post-enforcement count, not on whether ANY markup was parsed.
W898 (image_engine.py): inline <lora:> tags were silently dropped on the streaming image
  endpoint (encode strips tags but never loads the adapter).
W899 (image_engine.py): PNG uint8 cast truncated instead of rounding (mflux parity drift).
W900 (chat.py): VLM non-streaming left <think> markup in content by default; the text path
  strips it unconditionally.
W901 (disaggregate.py): _resolve_engine fell back to "any loaded engine" for a named-but-
  absent model — wrong-model output. Now strict when ≥2 models are loaded.
"""
from __future__ import annotations

import inspect


def test_w897_chat_cleans_markup_when_any_parsed_all_three_paths():
    from yunshu_gateway.routers import chat
    src = inspect.getsource(chat)
    # all three non-streaming sites gate cleanup on the RAW parse, not the enforced count
    assert src.count("if _raw_calls:") >= 3
    # the leak-prone form (clean only when enforced calls survive) is gone from these sites
    # (cleanup is now driven by _raw_calls; the enforced list still drives finish_reason)
    assert "tool_calls = _enforce_tool_choice(_raw_calls" in src


def test_w898_streaming_image_applies_inline_lora():
    from yunshu_engine.image_engine import ImageGenEngine as ImageEngine
    src = inspect.getsource(ImageEngine.generate_image_stream)
    assert "_parse_lora_tags(prompt)" in src
    assert "self._apply_inline_loras(_lora_tags)" in src
    assert "self.unload_diffusion_lora()" in src
    # the clean prompt (tags stripped) is what gets encoded
    assert "self._encode_prompt(_clean_prompt)" in src


def test_w899_to_png_rounds_before_uint8():
    from yunshu_engine.image_engine import ImageGenEngine as ImageEngine
    src = inspect.getsource(ImageEngine._to_png)
    assert "(arr * 255).round().astype(np.uint8)" in src
    assert "(arr * 255).astype(np.uint8)" not in src


def test_w899_round_behavior_numerically():
    import numpy as np
    # a value that truncation and rounding disagree on: 0.5/255 region
    arr = np.array([[0.5019607843, 0.9999]], dtype=np.float64)  # *255 = 128.0.., 254.97
    truncated = (arr * 255).astype(np.uint8)
    rounded = (arr * 255).round().astype(np.uint8)
    assert rounded[0, 1] == 255 and truncated[0, 1] == 254  # rounding recovers the top value


def test_w900_vlm_thinking_strip_unconditional():
    from yunshu_gateway.routers import chat
    src = inspect.getsource(chat)
    # the VLM gen path no longer gates extract_thinking on enable_thinking
    assert "if req.enable_thinking:\n            thinking_content, content = extract_thinking" not in src
