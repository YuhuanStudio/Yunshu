"""Anthropic `context_management` (beta context editing): clear_tool_uses_20250919 and
clear_thinking_20251015, applied to the request's messages before generation.

The edits follow the Anthropic spec: ``clear_thinking`` first, ``clear_tool_uses`` triggered by input
tokens (default 100000) or tool uses, the most recent ``keep`` tool uses kept (default 3), cleared
tool results replaced by a placeholder, ``clear_at_least`` honoured, ``exclude_tools`` and
``clear_tool_inputs`` supported. The response reports ``context_management.applied_edits``.
Token numbers come from the served tokenizer when the caller passes one (an estimate otherwise).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

CLEARED_PLACEHOLDER = "[cleared]"
DEFAULT_TRIGGER_TOKENS = 100000
DEFAULT_KEEP_TOOL_USES = 3
_TOOL_EDIT = "clear_tool_uses_20250919"
_THINK_EDIT = "clear_thinking_20251015"
# Server-side compaction: the served model summarizes the earlier turns (see routers/anthropic.py).
COMPACT_EDIT = "compact_20260112"
_COMPACT_EDIT = COMPACT_EDIT
COMPACT_DEFAULT_TRIGGER = 150000
COMPACT_MIN_TRIGGER = 50000


class ContextEditError(ValueError):
    """An invalid or unsupported `context_management` config (a 400)."""


def estimate_tokens(text: str) -> int:
    return (len(text.encode("utf-8")) + 3) // 4


def _blocks(msg: dict) -> list[dict]:
    c = msg.get("content")
    return [b for b in c if isinstance(b, dict)] if isinstance(c, list) else []


def tokens(messages: list[dict], count: Callable[[str], int]) -> int:
    return sum(
        count(json.dumps(m.get("content"), ensure_ascii=False, default=str))
        for m in messages
    )


def validate(cfg: Any) -> list[dict]:
    """The edits list of a config, or raise ContextEditError."""
    if cfg is None:
        return []
    if not isinstance(cfg, dict):
        raise ContextEditError("context_management must be an object")
    edits = cfg.get("edits", [])
    if not isinstance(edits, list):
        raise ContextEditError("context_management.edits must be an array")
    for i, e in enumerate(edits):
        t = e.get("type") if isinstance(e, dict) else None
        if t not in (_TOOL_EDIT, _THINK_EDIT, _COMPACT_EDIT):
            raise ContextEditError(
                f"context_management.edits.{i}.type: {t!r} is not supported; "
                f"use {_TOOL_EDIT} or {_THINK_EDIT}"
            )
        if t == _THINK_EDIT and i != 0:
            raise ContextEditError(
                f"context_management.edits.{i}: {_THINK_EDIT} must be the first edit"
            )
        if t == _COMPACT_EDIT:
            trig = e.get("trigger")
            if trig is not None:
                v = trig.get("value") if isinstance(trig, dict) else None
                if (
                    not isinstance(trig, dict)
                    or trig.get("type") != "input_tokens"
                    or not isinstance(v, int)
                    or v < COMPACT_MIN_TRIGGER
                ):
                    raise ContextEditError(
                        f"context_management.edits.{i}.trigger: must be "
                        f"{{type: input_tokens, value: >= {COMPACT_MIN_TRIGGER}}}"
                    )
    return edits


def _clear_thinking(
    messages: list[dict], edit: dict, count: Callable[[str], int]
) -> dict | None:
    keep = edit.get("keep", {"type": "thinking_turns", "value": 1})
    if keep == "all" or (isinstance(keep, dict) and keep.get("type") == "all"):
        return None
    n = keep.get("value", 1) if isinstance(keep, dict) else 1
    if not isinstance(n, int) or n < 0:
        raise ContextEditError(
            "context_management.edits.0.keep.value must be a non-negative integer"
        )
    turns = [
        i
        for i, m in enumerate(messages)
        if m.get("role") == "assistant"
        and any(b.get("type") in ("thinking", "redacted_thinking") for b in _blocks(m))
    ]
    old = turns[: len(turns) - n] if n else turns
    if not old:
        return None
    freed = 0
    for i in old:
        before = tokens([messages[i]], count)
        messages[i]["content"] = [
            b
            for b in messages[i]["content"]
            if not (
                isinstance(b, dict)
                and b.get("type") in ("thinking", "redacted_thinking")
            )
        ] or [{"type": "text", "text": ""}]
        freed += before - tokens([messages[i]], count)
    return {
        "type": _THINK_EDIT,
        "cleared_input_tokens": max(freed, 0),
        "cleared_thinking_turns": len(old),
    }


def _clear_tool_uses(
    messages: list[dict], edit: dict, count: Callable[[str], int]
) -> dict | None:
    trigger = edit.get("trigger") or {
        "type": "input_tokens",
        "value": DEFAULT_TRIGGER_TOKENS,
    }
    uses = [
        (i, b)
        for i, m in enumerate(messages)
        if m.get("role") == "assistant"
        for b in _blocks(m)
        if b.get("type") == "tool_use"
    ]
    value = trigger.get("value", DEFAULT_TRIGGER_TOKENS)
    if trigger.get("type") == "tool_uses":
        if len(uses) <= value:
            return None
    elif tokens(messages, count) <= value:
        return None
    keep = (edit.get("keep") or {}).get("value", DEFAULT_KEEP_TOOL_USES)
    candidates = uses[: len(uses) - keep] if keep else uses
    exclude = set(edit.get("exclude_tools") or [])
    ids = {b.get("id"): b for _, b in candidates if b.get("name") not in exclude}
    if not ids:
        return None
    ci = edit.get("clear_tool_inputs")
    names = set(ci) if isinstance(ci, list) else set()
    before = tokens(messages, count)
    cleared = 0
    for m in messages:
        for b in _blocks(m):
            if b.get("type") == "tool_use" and b.get("id") in ids:
                if ci is True or b.get("name") in names:
                    b["input"] = {}
                cleared += 1
            elif b.get("type") == "tool_result" and b.get("tool_use_id") in ids:
                b["content"] = CLEARED_PLACEHOLDER
    freed = before - tokens(messages, count)
    least = (edit.get("clear_at_least") or {}).get("value", 0)
    if freed < least or freed <= 0:
        return None  # signalled to the caller, which restores the snapshot
    return {
        "type": _TOOL_EDIT,
        "cleared_input_tokens": freed,
        "cleared_tool_uses": cleared,
    }


def apply_edits(
    messages: list[dict],
    cfg: Any,
    count: Callable[[str], int] = estimate_tokens,
) -> tuple[list[dict], list[dict]]:
    """(edited copy of messages, applied_edits). An edit that clears nothing is not reported."""
    edits = validate(cfg)
    out = copy.deepcopy(messages)
    applied: list[dict] = []
    for edit in edits:
        snapshot = copy.deepcopy(out)
        if edit["type"] == _COMPACT_EDIT:
            continue  # run by the router: it needs the model
        fn = _clear_thinking if edit["type"] == _THINK_EDIT else _clear_tool_uses
        result = fn(out, edit, count)
        if result is None:
            out = snapshot
        else:
            applied.append(result)
    return out, applied
