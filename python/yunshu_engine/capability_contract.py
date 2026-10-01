"""Per-model capability contract: what a served model accepts, and the 400 for the rest.

The contract is derived from the :class:`~yunshu_engine.model_card.ModelCard` (which is
derived from the checkpoint), so ``/v1/models`` states exactly what the request gate
enforces. A request that uses a feature the model does not have gets an explicit 400 naming
the field, never a silent ignore. A card with no checkpoint evidence (unreadable config)
carries no contract and gates nothing.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from . import settings

_TEXT_KINDS = ("chat", "vlm", "omni")
# Chat content part type -> input modality it needs.
_PART_MODALITY = {
    "image_url": "image",
    "image": "image",
    "image_data": "image",
    "input_image": "image",
    "input_audio": "audio",
    "audio_url": "audio",
    "video_url": "video",
    "video": "video",
}
_CONSTRAINT_FIELDS = (
    "tools",
    "tool_choice",
    "response_format",
    "guided_json",
    "guided_regex",
    "guided_choice",
    "guided_grammar",
    "grammar",
)


def _llguidance() -> bool:
    return importlib.util.find_spec("llguidance") is not None


def _cache_tiers(card: Any) -> list[dict]:
    pc = card.prefix_cache or {}
    if not pc.get("supported"):
        return []
    tiers: list[dict] = [{"tier": "ram", "kind": pc.get("kind")}]
    if card.kind in ("vlm", "omni"):
        if settings.get_bool("YUNSHU_VLM_APC_DISK"):
            tiers.append({"tier": "ssd", "kind": "apc"})
    elif settings.get_bool("YUNSHU_SSD_CACHE"):
        tiers.append({"tier": "ssd", "kind": "kv_prefix"})
    return tiers


def build_contract(card: Any) -> dict:
    """The contract object served under ``yunshu.contract`` for ``card``."""
    text = card.kind in _TEXT_KINDS
    spec = card.speculative or {}
    so = card.structured_output or {}
    llg = _llguidance()
    return {
        "tools": {
            "supported": text,
            "mode": ("template" if card.tools.get("supported") else "prompt")
            if text
            else None,
            "parallel": bool(card.tools.get("parallel")),
            "tool_choice": ["auto", "none", "required", "function"] if text else [],
        },
        "structured_output": {
            "json_object": bool(so.get("json_object")),
            # In-house subset first; schemas outside it are enforced by llguidance.
            "json_schema": {
                "supported": bool(so.get("json_schema")),
                "engines": ["in-house", "llguidance"] if llg else ["in-house"],
            },
            "regex": bool(so.get("regex")),
            "choice": bool(so.get("choice")),
            "grammar": {
                "supported": bool(so.get("grammar")) and llg,
                "syntax": "lark" if llg else None,
                "engine": "llguidance" if llg else None,
            },
        },
        "logprobs": dict(card.logprobs),
        "media": {
            "input": [m for m in card.input_modalities if m != "text"],
            "output": [m for m in card.output_modalities if m != "text"],
        },
        "reasoning": {
            "supported": bool(card.reasoning.get("supported")),
            "effort_levels": card.reasoning.get("effort_levels") or [],
        },
        "speculative": {
            "mode": spec.get("method") or "none",
            "lossless": spec.get("lossless"),
        },
        "cache_tiers": _cache_tiers(card),
        "context": {
            "length": (card.context or {}).get("length"),
            "max_output_tokens": card.max_output_tokens,
        },
        "unsupported_fields": "400",
    }


def has_evidence(card: Any) -> bool:
    return bool(card.family or card.architecture)


def unsupported(card: Any, body: dict) -> list[str]:
    """Reasons ``body`` (a chat request) uses what ``card``'s model lacks."""
    if card is None or not has_evidence(card):
        return []
    out: list[str] = []
    text = card.kind in _TEXT_KINDS
    if not text:
        for f in _CONSTRAINT_FIELDS + ("logprobs",):
            if body.get(f):
                out.append(f"{f}: a {card.kind} model does not generate text")
    # reasoning_effort / thinking_budget on a model without a reasoning mode are
    # accepted and have no effect (the card says so): coding agents send an effort on
    # every request whatever model they point at, and a 400 would break them.
    allowed = set(card.input_modalities)
    seen: set[str] = set()
    for msg in body.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            need = _PART_MODALITY.get(str(part.get("type")))
            if need and need not in allowed and need not in seen:
                seen.add(need)
                out.append(
                    f"messages.content[type={part['type']}]: this model accepts "
                    f"{'/'.join(sorted(allowed)) or 'no'} input, not {need}"
                )
    return out
