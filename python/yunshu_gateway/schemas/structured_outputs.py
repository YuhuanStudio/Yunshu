"""vLLM's ``structured_outputs`` request field, folded into Yunshu's native constraint fields.

Upstream: vllm-project/vllm ``StructuredOutputsParams`` (vllm/sampling_params.py) and the chat/completion
request field ``structured_outputs`` (vllm/entrypoints/openai/chat_completion/protocol.py); behaviour
cross-checked with tests/entrypoints/openai/chat_completion/test_chat_completion.py (invalid regex, empty
grammar and invalid schema are 400s). Independent code: kind "inspired" in vendor.json.

``{"structured_outputs": {"json": schema | "regex": str | "choice": [..] | "grammar": str |
"json_object": true}}`` becomes the same ``response_format`` / ``grammar`` the ``guided_*`` aliases use;
a native field the caller also set wins. Exactly one key may be given.
"""

from __future__ import annotations

from typing import Any

_KEYS = ("json", "regex", "choice", "grammar", "json_object", "structural_tag")


def fold_structured_outputs(data: Any) -> Any:
    if not isinstance(data, dict) or "structured_outputs" not in data:
        return data
    so = data["structured_outputs"]
    data = {k: v for k, v in data.items() if k != "structured_outputs"}
    if so is None:
        return data
    if not isinstance(so, dict):
        raise ValueError("structured_outputs: must be an object")
    given = [k for k in _KEYS if so.get(k) not in (None, False)]
    unknown = [k for k in so if k not in _KEYS and k != "whitespace_pattern"]
    if unknown:
        raise ValueError(f"structured_outputs: unknown field {unknown[0]!r}")
    if len(given) != 1:
        raise ValueError(
            "structured_outputs: exactly one of " + ", ".join(_KEYS) + " is required"
        )
    kind, val = given[0], so[given[0]]
    if kind == "json_object":
        data.setdefault("response_format", {"type": "json_object"})
    elif kind == "json":
        if isinstance(val, str):
            import json

            try:
                val = json.loads(val)
            except ValueError as exc:
                raise ValueError("structured_outputs.json: not valid JSON") from exc
        if not isinstance(val, dict):
            raise ValueError("structured_outputs.json: must be a JSON schema object")
        data.setdefault(
            "response_format",
            {"type": "json_schema", "json_schema": {"schema": val}},
        )
    elif kind == "regex":
        if not isinstance(val, str) or not val:
            raise ValueError("structured_outputs.regex: must be a non-empty string")
        data.setdefault("grammar", {"type": "regex", "pattern": val})
    elif kind == "choice":
        if (
            not isinstance(val, list)
            or not val
            or not all(isinstance(c, str) for c in val)
        ):
            raise ValueError(
                "structured_outputs.choice: must be a non-empty list of strings"
            )
        data.setdefault("grammar", {"type": "choice", "choices": val})
    elif kind == "grammar":
        if not isinstance(val, str) or not val.strip():
            raise ValueError("structured_outputs.grammar: must be a non-empty string")
        data.setdefault("grammar", {"type": "cfg", "grammar": val})
    else:
        raise ValueError("structured_outputs.structural_tag is not supported")
    return data
