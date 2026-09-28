"""Type tool-call arguments with the request's JSON schema.

Formats such as Qwen's native ``<tool_call><function=f><parameter=days>3
</parameter>`` carry every parameter value as text, so ``days`` arrives as
``"3"``. Clients validate arguments against the schema they sent, so each
string value is converted to the type its parameter declares (integer, number,
boolean, array, object, null). Anything that does not convert cleanly, and
every value whose declared type is a string or unknown, is left unchanged.
"""

from __future__ import annotations

import json
from typing import Any


def tool_schemas(tools: Any) -> dict[str, dict]:
    """Map function name -> parameters schema from tool definitions: OpenAI
    chat (``{"function": {...}}``), Responses (flat) and Anthropic
    (``input_schema``); dicts or pydantic models with ``model_dump``."""
    out: dict[str, dict] = {}
    for tool in tools or []:
        if hasattr(tool, "model_dump"):
            tool = tool.model_dump()
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        params = fn.get("parameters") or fn.get("input_schema")
        if isinstance(name, str) and isinstance(params, dict):
            out[name] = params
    return out


def _types(schema: dict) -> list[str]:
    t = schema.get("type")
    if isinstance(t, str):
        return [t]
    if isinstance(t, list):
        return [x for x in t if isinstance(x, str)]
    return []


def _from_text(text: str, types: list[str]) -> tuple[bool, Any]:
    stripped = text.strip()
    for t in types:
        if t == "integer":
            try:
                value = json.loads(stripped)
            except ValueError:
                continue
            if isinstance(value, int) and not isinstance(value, bool):
                return True, value
            if isinstance(value, float) and value.is_integer():
                return True, int(value)
        elif t == "number":
            try:
                value = json.loads(stripped)
            except ValueError:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return True, value
        elif t == "boolean":
            if stripped.lower() in ("true", "false"):
                return True, stripped.lower() == "true"
        elif t in ("array", "object"):
            try:
                value = json.loads(stripped)
            except ValueError:
                continue
            if isinstance(value, list if t == "array" else dict):
                return True, value
        elif t == "null" and stripped.lower() in ("null", "none"):
            return True, None
    return False, text


def _coerce(value: Any, schema: Any) -> Any:
    if not isinstance(schema, dict):
        return value
    types = _types(schema)
    if isinstance(value, str) and types and "string" not in types:
        ok, converted = _from_text(value, types)
        if ok:
            value = converted
    if isinstance(value, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            return {k: _coerce(v, props.get(k)) for k, v in value.items()}
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        return [_coerce(v, schema["items"]) for v in value]
    return value


def coerce_tool_arguments(name: str, arguments: str, schemas: dict[str, dict]) -> str:
    """Return ``arguments`` (a JSON object string) with string values converted
    to the types declared in ``schemas[name]``. Never raises; returns the input
    unchanged when there is nothing to convert or it cannot be parsed."""
    schema = schemas.get(name) if schemas else None
    if not schema or not isinstance(arguments, str):
        return arguments
    try:
        args = json.loads(arguments)
    except (TypeError, ValueError):
        return arguments
    if not isinstance(args, dict):
        return arguments
    try:
        coerced = _coerce(args, schema)
    except Exception:
        return arguments
    out = json.dumps(coerced, ensure_ascii=False)
    if out == json.dumps(args, ensure_ascii=False):
        return arguments
    return out


def coerce_tool_calls(calls: list | None, tools: Any) -> list | None:
    """Return copies of ``{"name", "arguments"}`` dicts with typed arguments."""
    if not calls:
        return calls
    schemas = tool_schemas(tools)
    if not schemas:
        return calls
    out = []
    for call in calls:
        if isinstance(call, dict) and "arguments" in call:
            call = {
                **call,
                "arguments": coerce_tool_arguments(
                    call.get("name", ""), call["arguments"], schemas
                ),
            }
        out.append(call)
    return out


def arguments_json(arguments: Any) -> str:
    """Tool-call ``arguments`` as the wire string: strings pass through,
    ``None`` is ``"{}"``, anything else is JSON-encoded."""
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return "{}"
    return json.dumps(arguments, ensure_ascii=False)
