# Upstream (inspired): vllm-project/vllm (Apache-2.0) vllm/sampling_params.py StructuredOutputsParams @ 68088ed
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
        import json

        if isinstance(val, str):
            try:
                val = json.loads(val)
            except ValueError as exc:
                raise ValueError("structural_tag: not valid JSON") from exc
        if not isinstance(val, dict):
            raise ValueError("structural_tag: must be an object")
        structural_tag_grammar(val)
        data.setdefault("response_format", {**val, "type": "structural_tag"})
    return data


def structural_tag_grammar(spec: Any) -> str:
    """Compile the OpenAI tags/triggers shape using llguidance's lazy lexemes.

    Ordinary text (including reasoning) remains free until a trigger appears.
    Treat markers as byte strings: they need not be single special tokens.
    """
    from llguidance import StructTag

    if not isinstance(spec, dict):
        raise ValueError("structural_tag: must be an object")
    tags, triggers = spec.get("structures"), spec.get("triggers")
    if not isinstance(tags, list) or not tags:
        raise ValueError("structural_tag.structures: must be a non-empty list")
    if (
        not isinstance(triggers, list)
        or not triggers
        or not all(isinstance(t, str) and t for t in triggers)
    ):
        raise ValueError("structural_tag.triggers: must be non-empty strings")
    compiled = []
    used = set()
    for tag in tags:
        if not isinstance(tag, dict):
            raise ValueError("structural_tag: each structure must be an object")
        begin, end, schema = tag.get("begin"), tag.get("end"), tag.get("schema")
        if not isinstance(begin, str) or not begin or not isinstance(end, str):
            raise ValueError(
                "structural_tag: begin/end must be strings, begin non-empty"
            )
        if not isinstance(schema, dict):
            raise ValueError("structural_tag: schema must be an object")
        matches = [t for t in triggers if begin.startswith(t)]
        if len(matches) != 1:
            raise ValueError("structural_tag: begin must match exactly one trigger")
        used.add(matches[0])
        compiled.append(
            StructTag(trigger=matches[0], begin=begin, end=end, grammar=schema)
        )
    if used != set(triggers):
        raise ValueError("structural_tag: every trigger must have a structure")
    lark = StructTag.to_grammar(compiled, assume_special=False)
    from yunshu_engine.grammar_constraint import validate_llg_cfg

    validate_llg_cfg(lark)
    return "// yunshu structural_tag\n" + lark
