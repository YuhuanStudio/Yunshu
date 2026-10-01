"""Media token cost derived from the model's processor / config.

The context budget used to charge every image / video a flat 576 tokens. The real
cost is the patch grid the processor produces: for Qwen-VL style models
``smart_resize`` snaps the image to a multiple of ``patch_size * merge_size``
(bounded by ``min_pixels`` / ``max_pixels``), then
``tokens = (H / patch) * (W / patch) / merge**2``; video multiplies that by
``ceil(frames / temporal_patch_size)``. Models with a fixed per-image sequence
(``image_seq_length`` / ``mm_tokens_per_image``) charge exactly that.

``make_media_token_counter(processor, config)`` returns ``block -> tokens`` for
``ContextWindowManager(media_token_counter=...)`` and
``count_message_tokens(media_counter=...)``. Unknown dimensions fall back to the
legacy flat estimate. ``estimate_template_overhead`` measures the chat template's own
framing so the budget can be checked against it.
"""

from __future__ import annotations

import base64
import io
import math
from collections.abc import Callable
from typing import Any

DEFAULT_IMAGE_TOKENS = 576
DEFAULT_VIDEO_FRAMES = 16
VIDEO_BLOCK_TYPES = ("video", "video_url")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _first(sources: list[Any], *keys: str) -> Any:
    for src in sources:
        for k in keys:
            v = _get(src, k)
            if v is not None:
                return v
    return None


def _smart_resize(
    h: int, w: int, factor: int, min_pixels: int | None, max_pixels: int | None
) -> tuple[int, int]:
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if max_pixels and h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif min_pixels and h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


class MediaTokenModel:
    """Patch-grid arithmetic for one model, built from its processor / config."""

    def __init__(self, processor: Any = None, config: dict | None = None) -> None:
        cfg = config or {}
        ip = _get(processor, "image_processor", processor)
        vp = _get(processor, "video_processor")
        vcfg = (
            cfg.get("vision_config")
            or (cfg.get("thinker_config") or {}).get("vision_config")
            or {}
        )
        srcs = [ip, vcfg, cfg]
        self.fixed_tokens: int | None = _first(
            [processor, ip, cfg, vcfg], "image_seq_length", "mm_tokens_per_image"
        )
        self.patch = _first(srcs, "patch_size")
        self.merge = _first(srcs, "merge_size", "spatial_merge_size") or 1
        self.temporal = _first([vp, ip, vcfg], "temporal_patch_size") or 1
        size = _get(ip, "size")
        min_px = _first([ip], "min_pixels")
        max_px = _first([ip], "max_pixels")
        if isinstance(size, dict):
            min_px = min_px or size.get("shortest_edge")
            max_px = max_px or size.get("longest_edge")
        self.min_pixels = min_px if isinstance(min_px, int) else None
        self.max_pixels = max_px if isinstance(max_px, int) else None
        # vision start/end markers wrapped around the pad run
        has_marker = (
            _first([cfg, vcfg], "vision_start_token_id", "image_start_token_id")
            is not None
        )
        self.wrapper = 2 if has_marker else 0
        self.derivable = isinstance(self.patch, int) and self.patch > 0

    def image_tokens(self, width: int | None, height: int | None) -> int | None:
        if self.fixed_tokens:
            return int(self.fixed_tokens) + self.wrapper
        if not self.derivable or not width or not height:
            return None
        patch, merge = int(self.patch), int(self.merge)
        h, w = _smart_resize(
            int(height), int(width), patch * merge, self.min_pixels, self.max_pixels
        )
        return (h // patch) * (w // patch) // (merge * merge) + self.wrapper

    def video_tokens(
        self, width: int | None, height: int | None, frames: int | None
    ) -> int | None:
        per = self.image_tokens(width, height)
        if per is None:
            return None
        frames = max(1, int(frames or DEFAULT_VIDEO_FRAMES))
        if self.fixed_tokens:
            return per * frames
        return per * math.ceil(frames / max(1, int(self.temporal)))


def _decode_dims(ref: str) -> tuple[int, int] | None:
    try:
        from PIL import Image  # lazy: header-only read
    except Exception:
        return None
    try:
        if ref.startswith("data:"):
            raw = base64.b64decode(ref.split(",", 1)[1], validate=False)
            return Image.open(io.BytesIO(raw)).size
        path = ref[7:] if ref.startswith("file://") else ref
        if path.startswith(("http://", "https://")):
            return None
        return Image.open(path).size
    except Exception:
        return None


def _block_ref(block: dict) -> str | None:
    for key in ("image_url", "video_url", "video", "image", "image_data"):
        v = block.get(key)
        if isinstance(v, dict):
            v = v.get("url")
        if isinstance(v, str):
            return str(v)
    src = block.get("source")
    if isinstance(src, dict) and isinstance(src.get("data"), str):
        return "data:;base64," + str(src["data"])
    return None


def make_media_token_counter(
    processor: Any = None,
    config: dict | None = None,
    fallback: int = DEFAULT_IMAGE_TOKENS,
) -> Callable[[dict], int]:
    """Return ``block -> estimated tokens`` for image / video content blocks."""
    model = MediaTokenModel(processor, config)

    def counter(block: dict) -> int:
        btype = block.get("type", "")
        is_video = btype in VIDEO_BLOCK_TYPES
        w = block.get("width")
        h = block.get("height")
        if not (w and h) and not is_video:
            ref = _block_ref(block)
            dims = _decode_dims(ref) if ref else None
            if dims:
                w, h = dims
        if is_video:
            frames = block.get("num_frames") or block.get("frames")
            if isinstance(frames, list):
                frames = len(frames)
            n = model.video_tokens(w, h, frames)
            if n is None:
                groups = math.ceil(
                    max(1, int(frames or DEFAULT_VIDEO_FRAMES))
                    / max(1, int(model.temporal))
                )
                n = fallback * groups
            return n
        n = model.image_tokens(w, h)
        return fallback if n is None else n

    return counter


def _text_tokens(tokenizer: Any, messages: list[dict]) -> tuple[int, list[dict]]:
    total = 0
    text_only = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, list):
            c = "".join(
                b.get("text", "")
                for b in c
                if isinstance(b, dict) and b.get("type") == "text"
            )
        c = c or ""
        total += len(tokenizer.encode(c))
        text_only.append({"role": m.get("role", "user"), "content": c})
    return total, text_only


def estimate_template_overhead(tokenizer: Any, messages: list[dict]) -> int | None:
    """Tokens the chat template adds beyond the raw message text (framing, role
    markers, generation prompt); ``None`` when the template cannot be rendered.
    Media blocks are stripped first so this is pure template framing."""
    try:
        raw, text_only = _text_tokens(tokenizer, messages)
        rendered = tokenizer.apply_chat_template(
            text_only, tokenize=True, add_generation_prompt=True
        )
        if isinstance(rendered, dict):
            rendered = rendered.get("input_ids", [])
        return max(0, len(rendered) - raw)
    except Exception:
        return None


def check_template_overhead(
    estimated_total: int, tokenizer: Any, messages: list[dict], media_tokens: int = 0
) -> dict:
    """Compare a budget estimate with text + template framing + media.

    Returns ``{"overhead", "undercount"}``; ``undercount`` is how many tokens the
    estimate is short of the template-rendered need (0 when it covers it).
    """
    overhead = estimate_template_overhead(tokenizer, messages)
    if overhead is None:
        return {"overhead": None, "undercount": 0}
    text, _ = _text_tokens(tokenizer, messages)
    need = text + overhead + media_tokens
    return {"overhead": overhead, "undercount": max(0, need - estimated_total)}
