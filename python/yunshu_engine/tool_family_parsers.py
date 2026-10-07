# Upstream (inspired): vllm-project/vllm (Apache-2.0) tests/tool_parsers
"""Independent readers for native tool wire formats; never execute model code."""

from __future__ import annotations

import ast
import json
import re
from typing import Any


def pythonic(body: str, tools: Any) -> list[dict[str, str]]:
    from .tool_format import _call

    body = body.strip().removeprefix("<|python_tag|>").strip()
    tree = ast.parse(body, mode="eval").body
    nodes = tree.elts if isinstance(tree, ast.List) else [tree]
    calls = []
    for node in nodes:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError("not a literal function call")
        if node.args or any(k.arg is None for k in node.keywords):
            raise ValueError("positional arguments and expansions are unsupported")
        args = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        if len(args) != len(node.keywords):
            raise ValueError("duplicate keyword")
        calls.append(_call(node.func.id, args))
    if not calls:
        raise ValueError("empty call list")
    return calls


def glm(body: str, tools: Any) -> list[dict[str, str]]:
    from .tool_arguments import coerce_tool_calls
    from .tool_format import _call

    match = re.match(r"\s*([\w.-]+)", body)
    if match is None:
        raise ValueError("missing GLM function name")
    name = match[1]
    rest = body[match.end() :].strip()
    if not rest:
        return [_call(name, {})]
    if rest.startswith("{"):
        return [_call(name, rest)]
    pairs = re.findall(
        r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", rest, re.S
    )
    if not pairs:
        raise ValueError("invalid GLM arguments")
    args = {key.strip(): value for key, value in pairs}
    return coerce_tool_calls([_call(name, args)], tools, raw_text_values=True) or []


def dsml(body: str, tools: Any) -> list[dict[str, str]]:
    from .tool_format import _call

    calls = []
    for invoke in re.finditer(
        r'<｜DSML｜invoke name="([^"]+)">(.*?)</｜DSML｜invoke>', body, re.S
    ):
        args = {}
        for param in re.finditer(
            r'<｜DSML｜parameter name="([^"]+)" string="(true|false)">(.*?)</｜DSML｜parameter>',
            invoke[2],
            re.S,
        ):
            if param[1] in args:
                raise ValueError("duplicate DSML parameter")
            args[param[1]] = param[3] if param[2] == "true" else json.loads(param[3])
        calls.append(_call(invoke[1], args))
    if not calls:
        raise ValueError("missing DSML invoke")
    return calls


def kimi(body: str, tools: Any) -> list[dict[str, str]]:
    from .tool_format import _call

    calls = []
    for match in re.finditer(
        r"<\|tool_call_begin\|>functions\.(.*?):\d+\s*<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>",
        body,
        re.S,
    ):
        calls.append(_call(match[1], match[2]))
    if not calls:
        raise ValueError("missing Kimi call")
    return calls


def harmony(body: str, tools: Any) -> list[dict[str, str]]:
    from .tool_format import _call

    header, sep, args = body.partition("<|message|>")
    match = re.search(r"(?:^|\s)to=([\w.-]+)", header)
    if not sep or match is None:
        raise ValueError("not a Harmony tool message")
    return [_call(match[1].removeprefix("functions."), args)]


def mistral(body: str, tools: Any) -> list[dict[str, str]]:
    from .tool_format import _calls_from

    if "[ARGS]" in body:
        from .tool_format import _call

        name, args = body.split("[ARGS]", 1)
        return [_call(name.strip(), args)]
    return _calls_from(json.loads(body.strip()))
