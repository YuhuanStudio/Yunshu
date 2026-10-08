"""Schemas for Anthropic's client-executed tools (``bash_*``, ``text_editor_*``, ``memory_*``).

These tools are declared by ``type`` and name only (``{"type": "bash_20250124", "name": "bash"}``): the
schema lives inside Claude, which was trained on it. A local model has no such training, so without a
schema it cannot produce a valid ``tool_use`` input. The gateway fills in the documented input schema and
a short description; the client still executes the tool and sends the ``tool_result`` back.
"""

from __future__ import annotations

import re
from typing import Any

_EDITOR_COMMANDS = ["view", "create", "str_replace", "insert", "undo_edit"]

BASH: dict[str, Any] = {
    "description": "Run commands in a bash shell. State (working directory, variables) persists between calls.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to run."},
            "restart": {
                "type": "boolean",
                "description": "Set true to restart the shell.",
            },
        },
    },
}

TEXT_EDITOR: dict[str, Any] = {
    "description": (
        "View, create and edit text files. `view` shows a file (optionally a line range) or lists a directory; "
        "`create` writes `file_text`; `str_replace` replaces the exact text `old_str` with `new_str`; "
        "`insert` adds `insert_text` after line `insert_line`."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "enum": _EDITOR_COMMANDS},
            "path": {
                "type": "string",
                "description": "Absolute path of the file or directory.",
            },
            "file_text": {"type": "string", "description": "Content for `create`."},
            "old_str": {
                "type": "string",
                "description": "Exact text to replace (`str_replace`).",
            },
            "new_str": {
                "type": "string",
                "description": "Replacement text (`str_replace`).",
            },
            "insert_line": {
                "type": "integer",
                "description": "Line after which to insert (`insert`).",
            },
            "insert_text": {
                "type": "string",
                "description": "Text to insert (`insert`).",
            },
            "view_range": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "[start, end] line numbers for `view`; end -1 means to the end.",
            },
        },
        "required": ["command", "path"],
    },
}

MEMORY: dict[str, Any] = {
    "description": (
        "Store and retrieve information across conversations in the /memories directory: "
        "`view`, `create`, `str_replace`, `insert`, `delete`, `rename`."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["view", "create", "str_replace", "insert", "delete", "rename"],
            },
            "path": {"type": "string"},
            "file_text": {"type": "string"},
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
            "insert_line": {"type": "integer"},
            "insert_text": {"type": "string"},
            "old_path": {"type": "string"},
            "new_path": {"type": "string"},
            "view_range": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["command"],
    },
}

COMPUTER: dict[str, Any] = {
    "description": "Control the client's computer display. Coordinates are pixels; execute actions on the client and return screenshots as tool results.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "key",
                    "type",
                    "mouse_move",
                    "left_click",
                    "left_click_drag",
                    "right_click",
                    "middle_click",
                    "double_click",
                    "screenshot",
                    "cursor_position",
                    "scroll",
                    "triple_click",
                    "left_mouse_down",
                    "left_mouse_up",
                    "hold_key",
                    "wait",
                ],
            },
            "text": {"type": "string"},
            "coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
            },
            "scroll_direction": {
                "type": "string",
                "enum": ["up", "down", "left", "right"],
            },
            "scroll_amount": {"type": "integer", "minimum": 0},
            "duration": {"type": "number", "minimum": 0},
        },
        "required": ["action"],
    },
}

_FAMILIES = (
    (re.compile(r"^computer_\d+$"), COMPUTER),
    (re.compile(r"^bash_\d+$"), BASH),
    (re.compile(r"^text_editor_\d+$"), TEXT_EDITOR),
    (re.compile(r"^memory_\d+$"), MEMORY),
)


def schema_for(tool_type: str | None) -> dict | None:
    """``{"description", "input_schema"}`` for a client-executed Anthropic tool type, else None."""
    for pat, spec in _FAMILIES:
        if tool_type and pat.match(tool_type):
            import copy

            result = copy.deepcopy(spec)
            if (
                tool_type.startswith("text_editor_")
                and tool_type >= "text_editor_20250429"
            ):
                result["input_schema"]["properties"]["command"]["enum"].remove(
                    "undo_edit"
                )
            if tool_type == "computer_20241022":
                result["input_schema"]["properties"]["action"]["enum"] = [
                    "key",
                    "type",
                    "mouse_move",
                    "left_click",
                    "left_click_drag",
                    "right_click",
                    "middle_click",
                    "double_click",
                    "screenshot",
                    "cursor_position",
                ]
            elif tool_type.startswith("computer_") and tool_type >= "computer_20251124":
                result["input_schema"]["properties"]["action"]["enum"].append("zoom")
                result["input_schema"]["properties"]["region"] = {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                }
            return result
    return None


def fill_client_tool_schemas(tools: list | None) -> bool:
    """Give schema-less ``bash_*`` / ``text_editor_*`` / ``memory_*`` tools their schema, in place.

    Returns True when something changed. Tools that already carry an ``input_schema`` are left alone.
    """
    changed = False
    for t in tools or []:
        spec = schema_for(getattr(t, "type", None))
        if spec is not None and not getattr(t, "input_schema", None):
            import copy

            t.input_schema = copy.deepcopy(spec["input_schema"])
            if str(getattr(t, "type", "")).startswith("computer_"):
                t.description = (
                    (t.description or spec["description"])
                    + f" Display: {getattr(t, 'display_width_px', '?')} x {getattr(t, 'display_height_px', '?')} pixels, display {getattr(t, 'display_number', 1)}."
                )
            if (
                str(getattr(t, "type", "")).startswith("computer_")
                and "zoom" in t.input_schema["properties"]["action"]["enum"]
                and not getattr(t, "enable_zoom", False)
            ):
                t.input_schema["properties"]["action"]["enum"].remove("zoom")
            if str(getattr(t, "type", "")).startswith("text_editor_") and getattr(
                t, "max_characters", None
            ):
                t.description = (
                    (t.description or spec["description"])
                    + f" View results are limited to {t.max_characters} characters by the client."
                )
            if not getattr(t, "description", None):
                t.description = spec["description"]
            changed = True
    return changed
