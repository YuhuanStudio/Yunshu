"""C03: media token cost from the processor's patch grid, not a flat 576."""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

from yunshu_control.token_counter import count_message_tokens
from yunshu_engine.context_window import ContextWindowManager
from yunshu_engine.media_tokens import (
    check_template_overhead,
    estimate_template_overhead,
    make_media_token_counter,
)


def _qwen(patch=14, merge=2, temporal=2, min_px=56 * 56, max_px=14 * 14 * 4 * 1280):
    ip = SimpleNamespace(
        patch_size=patch,
        merge_size=merge,
        temporal_patch_size=temporal,
        min_pixels=min_px,
        max_pixels=max_px,
    )
    return SimpleNamespace(image_processor=ip), {"vision_start_token_id": 1}


def _png_url(w, h):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_image_tokens_follow_patch_grid():
    proc, cfg = _qwen()
    c = make_media_token_counter(proc, cfg)
    # 448x448 -> 16x16 patches... (448/14)^2 / 4 = 256, + 2 markers
    assert c({"type": "image_url", "width": 448, "height": 448}) == 258
    assert c({"type": "image_url", "width": 224, "height": 224}) == 64 + 2
    assert c({"type": "image_url", "width": 896, "height": 448}) == 512 + 2


def test_max_pixels_caps_and_min_pixels_floors():
    proc, cfg = _qwen(max_px=28 * 28 * 100)
    c = make_media_token_counter(proc, cfg)
    big = c({"type": "image", "width": 4000, "height": 4000})
    assert big <= 100 + 2
    tiny = c({"type": "image", "width": 10, "height": 10})
    assert tiny >= 4 + 2


def test_dimensions_read_from_data_url_when_not_given():
    proc, cfg = _qwen()
    c = make_media_token_counter(proc, cfg)
    n = c({"type": "image_url", "image_url": {"url": _png_url(448, 224)}})
    assert n == 128 + 2


def test_video_scales_with_frames_and_temporal_patch():
    proc, cfg = _qwen(temporal=2)
    c = make_media_token_counter(proc, cfg)
    one = c({"type": "video", "width": 224, "height": 224, "num_frames": 2})
    four = c({"type": "video", "width": 224, "height": 224, "num_frames": 8})
    assert one == 66 and four == 4 * 66
    odd = c({"type": "video", "width": 224, "height": 224, "frames": [0] * 3})
    assert odd == 2 * 66  # ceil(3/2) temporal groups


def test_fixed_sequence_models_use_it():
    c = make_media_token_counter(SimpleNamespace(image_seq_length=256), {})
    assert c({"type": "image_url", "width": 4000, "height": 100}) == 256


def test_unknown_geometry_falls_back_to_legacy_estimate():
    c = make_media_token_counter(None, {})
    assert c({"type": "image_url", "image_url": {"url": "http://x/y.png"}}) == 576


def test_context_window_uses_processor_cost():
    proc, cfg = _qwen()
    msgs = [
        {
            "role": "user",
            "content": [{"type": "image_url", "width": 224, "height": 224}],
        }
    ]
    flat = ContextWindowManager(token_counter=lambda t: 0)._count_messages_tokens(msgs)
    real = ContextWindowManager.for_processor(
        proc, cfg, token_counter=lambda t: 0
    )._count_messages_tokens(msgs)
    assert flat == 576 + 4
    assert real == 66 + 4


def test_count_message_tokens_media_counter():
    proc, cfg = _qwen()
    msgs = [
        {"role": "user", "content": [{"type": "image", "width": 448, "height": 448}]}
    ]
    flat = count_message_tokens(msgs)
    real = count_message_tokens(msgs, media_counter=make_media_token_counter(proc, cfg))
    assert flat - real == 576 - 258


class _Tok:
    def encode(self, text):
        return text.split()

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=True):
        out = []
        for m in msgs:
            out += ["<s>", m["role"], *m["content"].split(), "</s>"]
        return out + ["<gen>"]


def test_template_overhead_measured_and_checked():
    msgs = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": [{"type": "text", "text": "hi there"}]},
    ]
    # 2 msgs * (<s>, role, </s>) + <gen> = 7 framing tokens
    assert estimate_template_overhead(_Tok(), msgs) == 7
    # text=4, overhead=7 -> need 11; an estimate of 8 undercounts by 3
    assert check_template_overhead(8, _Tok(), msgs) == {"overhead": 7, "undercount": 3}
    assert check_template_overhead(40, _Tok(), msgs)["undercount"] == 0
    assert estimate_template_overhead(object(), msgs) is None
