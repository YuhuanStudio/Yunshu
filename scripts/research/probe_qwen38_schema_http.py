"""Real-model VLM JSON-schema probe against an already-running local gateway.

Each row includes syntax, schema and task correctness. It never loads/downloads
a model, and it distinguishes stream reconstruction from non-stream output.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from jsonschema import Draft202012Validator

CASES = [
    (
        "two_fields",
        "Return code COBALT and count 7.",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "enum": ["COBALT"]},
                "count": {"type": "integer", "enum": [7]},
            },
            "required": ["code", "count"],
            "additionalProperties": False,
        },
        {"code": "COBALT", "count": 7},
    ),
    (
        "nested",
        "Return city Taipei and district Xinyi as a nested location object.",
        {
            "type": "object",
            "properties": {
                "location": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "enum": ["Taipei"]},
                        "district": {"type": "string", "enum": ["Xinyi"]},
                    },
                    "required": ["city", "district"],
                    "additionalProperties": False,
                }
            },
            "required": ["location"],
            "additionalProperties": False,
        },
        {"location": {"city": "Taipei", "district": "Xinyi"}},
    ),
    (
        "array",
        "Return the ordered tags red then blue.",
        {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "prefixItems": [{"const": "red"}, {"const": "blue"}],
                    "minItems": 2,
                    "maxItems": 2,
                }
            },
            "required": ["tags"],
            "additionalProperties": False,
        },
        {"tags": ["red", "blue"]},
    ),
    (
        "boolean",
        "Return enabled true.",
        {
            "type": "object",
            "properties": {"enabled": {"type": "boolean", "const": True}},
            "required": ["enabled"],
            "additionalProperties": False,
        },
        {"enabled": True},
    ),
    (
        "nullable",
        "Return nickname null.",
        {
            "type": "object",
            "properties": {"nickname": {"type": ["string", "null"], "const": None}},
            "required": ["nickname"],
            "additionalProperties": False,
        },
        {"nickname": None},
    ),
    (
        "empty_object",
        "Return an empty object.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        {},
    ),
    (
        "integer_array",
        "Return the numbers 2 and 4 in that order.",
        {
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                }
            },
            "required": ["numbers"],
            "additionalProperties": False,
        },
        {"numbers": [2, 4]},
    ),
    (
        "unicode",
        "Return the Chinese label 藍色.",
        {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": ["藍色"]}},
            "required": ["label"],
            "additionalProperties": False,
        },
        {"label": "藍色"},
    ),
    (
        "three_fields",
        "Return name Yunshu, version 3 and stable false.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "const": "Yunshu"},
                "version": {"type": "integer", "const": 3},
                "stable": {"type": "boolean", "const": False},
            },
            "required": ["name", "version", "stable"],
            "additionalProperties": False,
        },
        {"name": "Yunshu", "version": 3, "stable": False},
    ),
    (
        "long_prefix",
        ("This archive sentence is irrelevant to the final JSON fields.\n" * 320)
        + "Return code COBALT and count 7.",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "enum": ["COBALT"]},
                "count": {"type": "integer", "enum": [7]},
            },
            "required": ["code", "count"],
            "additionalProperties": False,
        },
        {"code": "COBALT", "count": 7},
    ),
]


def call(url, body):
    started = time.perf_counter()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        if not body["stream"]:
            result = json.load(response)
            choice = result["choices"][0]
            return (
                choice["message"].get("content", ""),
                choice["finish_reason"],
                result.get("usage"),
                time.perf_counter() - started,
            )
        chunks = []
        finish = None
        usage = None
        for line in response:
            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                chunks.append(choice.get("delta", {}).get("content", ""))
                finish = choice.get("finish_reason") or finish
        return "".join(chunks), finish, usage, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, prompt, schema, expected in CASES:
        Draft202012Validator.check_schema(schema)
        for stream in (
            [False, True]
            if name in ("two_fields", "nested", "long_prefix")
            else [False]
        ):
            repeats = 2 if name == "long_prefix" else 1
            for repeat in range(repeats):
                body = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 160,
                    "stream": stream,
                    "stream_options": {"include_usage": True} if stream else None,
                    "reasoning_effort": "none",
                    "chat_template_kwargs": {"enable_thinking": False},
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": name, "strict": True, "schema": schema},
                    },
                }
                row = {
                    "case": name,
                    "stream": stream,
                    "repeat": repeat,
                    "expected": expected,
                    "schema": schema,
                }
                try:
                    content, finish, usage, wall = call(args.url, body)
                    row.update(
                        content=content, finish_reason=finish, usage=usage, wall_s=wall
                    )
                    value = json.loads(content)
                    row["syntax_ok"] = True
                    errors = list(Draft202012Validator(schema).iter_errors(value))
                    row["schema_ok"] = not errors
                    row["schema_errors"] = [error.message for error in errors]
                    row["task_ok"] = value == expected
                except Exception as exc:
                    row["error"] = repr(exc)
                    row.setdefault("syntax_ok", False)
                    row.setdefault("schema_ok", False)
                    row.setdefault("task_ok", False)
                with args.output.open("a") as file:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    {
                        key: row.get(key)
                        for key in (
                            "case",
                            "stream",
                            "repeat",
                            "syntax_ok",
                            "schema_ok",
                            "task_ok",
                            "finish_reason",
                            "wall_s",
                            "error",
                        )
                        if key in row
                    },
                    flush=True,
                )


if __name__ == "__main__":
    main()
