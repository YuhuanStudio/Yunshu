"""Explicit JSON ``null`` on an optional request parameter means "not set".

The OpenAI and Anthropic SDK types declare almost every parameter ``Optional[...]``, and LangChain, LiteLLM,
the Vercel AI SDK, Continue and Zed send ``"temperature": null`` / ``"max_tokens": null`` / ``"stream": null``
for parameters the caller left unset. The request models declare those fields with a concrete type and a
default, so pydantic answered 400 "Input should be a valid number". Dropping the key lets the default apply,
which is what the real APIs do.
"""

from __future__ import annotations

import types
import typing
from typing import Any


def _allows_none(annotation: Any) -> bool:
    if annotation is Any or annotation is None or annotation is type(None):
        return True
    args = typing.get_args(annotation)
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return any(a is type(None) or _allows_none(a) for a in args)
    return False


def clean_request(model_cls: type, data: Any) -> Any:
    """Return `data` without keys that are None where `model_cls` does not accept None; a negative
    ``top_k`` (the vLLM / llama.cpp spelling of "disabled", sent by LiteLLM and Cline) becomes 0."""
    if not isinstance(data, dict):
        return data
    if "structured_outputs" in data and "grammar" in model_cls.model_fields:
        from .structured_outputs import fold_structured_outputs

        data = fold_structured_outputs(data)
    if isinstance(data.get("top_k"), int) and data["top_k"] < 0:
        data = {**data, "top_k": 0}
    if None not in data.values():
        return data
    fields = model_cls.model_fields
    return {
        k: v
        for k, v in data.items()
        if v is not None or k not in fields or _allows_none(fields[k].annotation)
    }


def fold_allowed_tools(data: Any) -> Any:
    """Chat ``tool_choice={"type": "allowed_tools", "allowed_tools": {"mode", "tools"}}`` (the newer OpenAI spelling) ->
    ``tools`` restricted (Responses spells it flat, with ``name`` instead of ``function.name``) to the allowed names plus ``tool_choice`` = the mode (``auto`` / ``required``)."""
    if not isinstance(data, dict):
        return data
    tc = data.get("tool_choice")
    if not (isinstance(tc, dict) and tc.get("type") == "allowed_tools"):
        return data
    spec = (
        tc.get("allowed_tools") or tc
    )  # chat nests it; Responses puts mode / tools beside type
    names = {
        (t.get("function") or {}).get("name") or t.get("name")
        for t in spec.get("tools") or []
        if isinstance(t, dict)
    }
    out = {
        **data,
        "tool_choice": spec.get("mode")
        if spec.get("mode") in ("auto", "required")
        else "auto",
    }
    if names and isinstance(data.get("tools"), list):
        out["tools"] = [
            t
            for t in data["tools"]
            if isinstance(t, dict)
            and ((t.get("function") or {}).get("name") or t.get("name")) in names
        ]
    return out
