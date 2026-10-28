"""OpenAI ``strict`` structured outputs, one behaviour for Chat Completions and Responses.

``strict: true`` (chat ``response_format.json_schema.strict``, Responses ``text.format.strict``)
means the schema is enforced exactly: every object schema, at any depth, rejects keys it does not
declare (``additionalProperties: false``), as OpenAI requires of strict schemas. ``strict`` false or
absent (OpenAI's default) leaves the schema as the client wrote it. The decoder constrains to the
schema either way; strict only closes objects.
"""

from __future__ import annotations

import copy
from typing import Any

_SCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions")
_SCHEMA_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems")
_SCHEMA_ONE = ("items", "not", "if", "then", "else", "contains")


def _close(node: Any) -> None:
    if not isinstance(node, dict):
        return
    if node.get("type") == "object" or "properties" in node:
        # A schema-valued additionalProperties (a map type) is the client's explicit choice.
        if not isinstance(node.get("additionalProperties"), dict):
            node["additionalProperties"] = False
    for key in _SCHEMA_MAPS:
        children = node.get(key)
        if isinstance(children, dict):
            for child in children.values():
                _close(child)
    for key in _SCHEMA_LISTS:
        children = node.get(key)
        if isinstance(children, list):
            for child in children:
                _close(child)
    for key in _SCHEMA_ONE:
        _close(node.get(key))
    _close(node.get("additionalProperties"))


def strict_schema(schema: Any, strict: Any) -> Any:
    """``schema`` with every object closed when ``strict`` is true; otherwise unchanged."""
    if strict is not True or not isinstance(schema, dict):
        return schema
    out = copy.deepcopy(schema)
    _close(out)
    return out
