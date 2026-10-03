"""Offline-safe grading for real Claude Messages and Codex Responses tool shapes.

Runs only HTTP inference; emitted shell tools are never executed.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def cases(model):
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
        "required": ["command", "timeout"],
        "additionalProperties": False,
    }
    prompt = 'Call the provided shell tool once with command exactly "printf rapidmlx" and timeout exactly 1000. Do not run any other command.'
    for forced in (False, True):
        mode = "forced" if forced else "auto"
        yield (
            "openai-" + mode,
            "/v1/chat/completions",
            {
                "model": model,
                "temperature": 0,
                "max_tokens": 256,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [
                    {
                        "role": "developer",
                        "content": "Use the provided tool as requested.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "description": "Run a shell command.",
                            "parameters": parameters,
                        },
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "shell"}}
                if forced
                else "auto",
            },
        )
        yield (
            "claude-" + mode,
            "/v1/messages?beta=true",
            {
                "model": model,
                "temperature": 0,
                "max_tokens": 256,
                "system": [
                    {
                        "type": "text",
                        "text": "Use the provided tool as requested.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
                "thinking": {"type": "disabled"},
                "tools": [
                    {
                        "name": "shell",
                        "description": "Run a shell command.",
                        "input_schema": parameters,
                    }
                ],
                "tool_choice": {"type": "tool", "name": "shell"}
                if forced
                else {"type": "auto"},
            },
        )
        yield (
            "codex-" + mode,
            "/v1/responses",
            {
                "model": model,
                "temperature": 0,
                "max_output_tokens": 256,
                "input": [
                    {
                        "role": "developer",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Use the provided tool as requested.",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}],
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "shell",
                        "description": "Run a shell command.",
                        "parameters": parameters,
                    }
                ],
                "tool_choice": {"type": "function", "name": "shell"}
                if forced
                else "auto",
            },
        )


def grade(name, response):
    if name.startswith("claude-"):
        calls = [
            (block.get("name"), block.get("input"))
            for block in response.get("content", [])
            if block.get("type") == "tool_use"
        ]
    elif name.startswith("codex-"):
        calls = [
            (block.get("name"), block.get("arguments"))
            for block in response.get("output", [])
            if block.get("type") == "function_call"
        ]
    else:
        calls = [
            (
                call.get("function", {}).get("name"),
                call.get("function", {}).get("arguments"),
            )
            for choice in response.get("choices", [])
            for call in choice.get("message", {}).get("tool_calls", [])
        ]
    if len(calls) != 1:
        return False, f"expected exactly one call, got {len(calls)}"
    tool, arguments = calls[0]
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return False, "invalid arguments JSON"
    if tool != "shell" or not isinstance(arguments, dict):
        return False, "wrong tool or arguments shape"
    if set(arguments) != {"command", "timeout"}:
        return False, "missing or unexpected arguments"
    if (
        arguments["command"] != "printf rapidmlx"
        or type(arguments["timeout"]) is not int
        or arguments["timeout"] != 1000
    ):
        return False, "wrong command or integer timeout"
    return True, "exact tool and schema-correct arguments"


def run(url, model):
    rows = []
    for name, route, body in cases(model):
        row = {"case": name, "route": route, "request": body}
        try:
            req = urllib.request.Request(
                url + route,
                data=json.dumps(body).encode(),
                headers={
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "claude-code-20250219,prompt-caching-scope-2026-01-05",
                },
            )
            with urllib.request.urlopen(req, timeout=180) as response:
                row["response"] = json.load(response)
                row["http_status"] = response.status
            row["passed"], row["reason"] = grade(name, row["response"])
        except Exception as exc:
            row.update(passed=False, error=repr(exc))
            if hasattr(exc, "read"):
                row["error_body"] = exc.read().decode(errors="replace")
        rows.append(row)
    return {
        "complete": True,
        "passed": sum(r["passed"] for r in rows),
        "total": len(rows),
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.url.rstrip("/"), args.model)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}))


if __name__ == "__main__":
    main()
