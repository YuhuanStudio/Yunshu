"""Forced native tool envelopes compiled by llguidance.

The argument schema belongs to each named branch, including parallel calls.
Byte markers also work when a tokenizer splits them over several tokens.
"""

from __future__ import annotations

import json
import re
from typing import Any


def build_native_grammar(
    specs: Any, fmt: Any, tokenizer: Any, *, only: str | None, parallel: bool
) -> str:
    from .tool_call_grammar import _q, _schema_type, _token_id

    hf = getattr(tokenizer, "_tokenizer", tokenizer)
    specials = set(getattr(hf, "all_special_tokens", []))
    get_added = getattr(hf, "get_added_vocab", None)
    if get_added is not None:
        specials.update(get_added())

    def lit(text: str) -> str:
        tid = _token_id(tokenizer, text)
        if tid is not None:
            return f"<[{tid}]>"
        # Envelopes can combine special tokens and ordinary header text.
        known = sorted((t for t in specials if t in text), key=len, reverse=True)
        if not known:
            return _q(text)
        pieces = re.split("(" + "|".join(re.escape(t) for t in known) + ")", text)
        return " ".join(
            f"<[{_token_id(tokenizer, p)}]>" if p in specials else _q(p)
            for p in pieces
            if p
        )

    def js(schema: dict) -> str:
        return "%json " + json.dumps(schema, ensure_ascii=False)

    text_rules = []

    def value(schema: dict, close: str, *, python: bool = False) -> str:
        kind = _schema_type(schema)
        if python and kind == "boolean":
            return '("True" | "False")'
        if python and kind == "null":
            return '"None"'
        if kind == "enum" and not python:
            values = schema.get("enum", [schema.get("const")])
            return "(" + " | ".join(_q(v) for v in values) + ") " + lit(close)
        if kind == "string" and not python:
            # stop at the exact closing delimiter, retaining embedded newlines
            key = f"text_{len(text_rules)}"
            lo, hi = schema.get("minLength", 0), schema.get("maxLength", "")
            bounded = f"/(.|\\n){{{lo},{hi}}}/"
            if "pattern" in schema:
                pattern = schema["pattern"]
                begin, end = pattern.startswith("^"), pattern.endswith("$")
                pattern = pattern.removeprefix("^").removesuffix("$")
                expression = (
                    ("" if begin else "(.|\\n)*")
                    + "(?:"
                    + pattern
                    + ")"
                    + ("" if end else "(.|\\n)*")
                )
                bounded += " & /" + expression.replace("/", "\\/") + "/"
            text_rules.append(f"{key}[suffix={_q(close)}]: {bounded}")
            return key
        return js(schema) + ("" if python else " " + lit(close))

    rules = []
    branches = []
    for i, spec in enumerate(specs):
        if only is not None and spec.name != only:
            continue
        name, schema = spec.name, spec.parameters
        key = f"call_{i}"
        family = fmt.name
        if family in ("hermes", "llama3_json"):
            call_schema = {
                "type": "object",
                "properties": {
                    "name": {"const": name},
                    "arguments": schema,
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
            body = js(call_schema)
            envelope = (
                (lit(fmt.start) + " " if fmt.start else "")
                + body
                + (" " + lit(fmt.end) if fmt.end else "")
            )
        elif family == "harmony":
            envelope = f'{lit(fmt.start)} {_q(" to=" + name)} {lit("<|channel|>")} "commentary" {lit("<|message|>")} {js(schema)} {lit(fmt.end)}'
        elif family == "mistral":
            call_schema = {
                "type": "object",
                "properties": {"name": {"const": name}, "arguments": schema},
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
            envelope = js(call_schema)
        elif family == "kimi_k2":
            envelope = f"{lit('<|tool_call_begin|>')} {_q('functions.' + name + ':0')} {lit('<|tool_call_argument_begin|>')} {js(schema)} {lit('<|tool_call_end|>')}"
        elif family == "deepseek":
            envelope = f"{lit('<｜tool▁call▁begin｜>')} {_q(name)} {lit('<｜tool▁sep｜>')} {js(schema)} {lit('<｜tool▁call▁end｜>')}"
        elif family in ("glm47", "deepseek_v32", "deepseek_v4", "llama3_pythonic"):
            fields = []
            required = schema.get("required", [])
            props = schema.get("properties", {})
            for k, prop in props.items():
                if family == "glm47":
                    part = f"{lit('<arg_key>')} {_q(k)} {lit('</arg_key>')} {lit('<arg_value>')} {value(prop, '</arg_value>')}"
                elif family.startswith("deepseek"):
                    flag = (
                        "true" if _schema_type(prop) in ("string", "enum") else "false"
                    )
                    part = f"{lit(chr(60) + '｜DSML｜parameter name=' + json.dumps(k) + ' string=' + json.dumps(flag) + chr(62))} {value(prop, '</｜DSML｜parameter>')}"
                else:
                    part = f"{_q(k + '=')} {value(prop, '', python=True)}"
                if family == "llama3_pythonic":
                    # Fixed schema order, optional keys can be omitted without dangling commas.
                    fields.append(part)
                else:
                    fields.append(part if k in required else f"({part})?")
            if family == "glm47":
                envelope = (
                    f"{lit(fmt.start)} {_q(name)} "
                    + " ".join(fields)
                    + f" {lit(fmt.end)}"
                )
            elif family == "llama3_pythonic":
                # Required fields only; optional fields remain valid when omitted.
                fields = [
                    part for k, part in zip(props, fields, strict=True) if k in required
                ]
                envelope = _q(name + "(") + " " + ' "," '.join(fields) + ' ")"'
            else:
                envelope = (
                    lit('<｜DSML｜invoke name="' + name + '">')
                    + " "
                    + " ".join(fields)
                    + " "
                    + lit("</｜DSML｜invoke>")
                )
        else:
            raise ValueError(f"unsupported native grammar: {family}")
        rules.append(f"{key}: {envelope}")
        branches.append(key)
    if not branches:
        raise ValueError("no tool matches forced choice")
    union = " | ".join(branches)
    repeat = ' ("," call)*' if parallel else ""
    family = fmt.name
    if family in ("mistral", "llama3_pythonic"):
        start = (
            (lit(fmt.start) + " " if fmt.start else "") + '"[" call' + repeat + ' "]"'
        )
    elif family in ("kimi_k2", "deepseek", "deepseek_v32", "deepseek_v4"):
        start = (
            lit(fmt.start)
            + " call"
            + (" call*" if parallel else "")
            + " "
            + lit(fmt.end)
        )
    elif family == "llama3_json":
        start = "call"
        if _token_id(tokenizer, "<|python_tag|>") is not None:
            start = f"({lit('<|python_tag|>')})? call"
    else:
        start = "call" + (" call*" if parallel else "")
    return "\n".join(
        [
            "%ignore /[ \\t\\n\\r]+/",
            "start: " + start,
            "call: " + union,
            *rules,
            *text_rules,
        ]
    )
