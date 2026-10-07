# Upstream (inspired): ml-explore/mlx-lm (MIT) mlx_lm/tool_parsers/qwen3_coder.py @ a537041
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
        return t if t and all(isinstance(x, str) for x in t) else []
    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list):
            kinds = [
                _types(branch) if isinstance(branch, dict) else []
                for branch in branches
            ]
            if not kinds or any(not types for types in kinds):
                return []
            return list(dict.fromkeys(t for types in kinds for t in types))
    values = schema.get("enum")
    if not isinstance(values, list) or not values:
        values = [schema["const"]] if "const" in schema else []
    kinds_by_type = {
        str: "string",
        bool: "boolean",
        int: "integer",
        float: "number",
        list: "array",
        dict: "object",
        type(None): "null",
    }
    if values:
        return list(dict.fromkeys(kinds_by_type.get(type(v), "string") for v in values))
    return []


def _from_text(text: str, types: list[str]) -> tuple[bool, Any]:
    stripped = text.strip()
    # XML parsers sometimes retain a JSON-quoted scalar. Unwrap only when the
    # schema excludes strings, so identifiers/leading zeroes remain exact.
    try:
        decoded = json.loads(stripped)
        if isinstance(decoded, str) and "string" not in types:
            stripped = decoded.strip()
    except ValueError:
        pass
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
    if isinstance(value, str) and types:
        if "string" not in types:
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


def coerce_tool_arguments(
    name: str,
    arguments: str,
    schemas: dict[str, dict],
    *,
    raw_text_values: bool = False,
) -> str:
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
    original = args
    try:
        if raw_text_values:
            # XML's unquoted null is ambiguous with a string; JSON's quoted
            # "null" is already a valid string and must never take this branch.
            props = schema.get("properties") or {}
            args = dict(args)
            for key, value in args.items():
                prop = props.get(key)
                types = _types(prop) if isinstance(prop, dict) else []
                if value == "null" and "string" in types and "null" in types:
                    args[key] = None
                elif (
                    isinstance(value, str)
                    and isinstance(prop, dict)
                    and "type" not in prop
                    and not types
                ):
                    # Untyped XML containers follow mlx-lm a537041. Keep
                    # scalars and explicit string unions as literal text.
                    try:
                        container = json.loads(value)
                    except ValueError:
                        continue
                    if isinstance(container, (dict, list)):
                        args[key] = container
        coerced = _coerce(args, schema)
    except Exception:
        return arguments
    out = json.dumps(coerced, ensure_ascii=False)
    if out == json.dumps(original, ensure_ascii=False):
        return arguments
    return out


def coerce_tool_calls(
    calls: list | None, tools: Any, *, raw_text_values: bool = False
) -> list | None:
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
                    call.get("name", ""),
                    call["arguments"],
                    schemas,
                    raw_text_values=raw_text_values,
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


def parser_tools(tools: Any) -> Any:
    """Teach upstream parsers about string unions before they deserialize text.

    Upstream GLM/Qwen readers only recognize type == 'string'. Once '123'
    became an integer, downstream schema coercion cannot recover the raw value.
    Copies keep the caller's schema and grammar cache untouched.
    """
    import copy

    if not tools:
        return tools
    result = copy.deepcopy(tools)
    for tool in result:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool)
        params = fn.get("parameters") or fn.get("input_schema") or {}
        for schema in (params.get("properties") or {}).values():
            if isinstance(schema, dict) and "string" in _types(schema):
                schema["type"] = "string"
    return result
