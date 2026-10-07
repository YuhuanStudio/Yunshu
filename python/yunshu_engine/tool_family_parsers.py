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

        class JsonLiterals(ast.NodeTransformer):
            def visit_Name(self, value):
                literals = {"true": True, "false": False, "null": None}
                if value.id not in literals:
                    raise ValueError("non-literal Python argument")
                return ast.copy_location(ast.Constant(literals[value.id]), value)

        args = {
            k.arg: ast.literal_eval(JsonLiterals().visit(k.value))
            for k in node.keywords
        }
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
    pair_pattern = re.compile(
        r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S
    )
    pairs = pair_pattern.findall(rest)
    if pair_pattern.sub("", rest).strip():
        raise ValueError("malformed GLM parameter")
    if not pairs:
        raise ValueError("invalid GLM arguments")
    args = {key.strip(): value for key, value in pairs}
    if len(args) != len(pairs):
        raise ValueError("duplicate GLM argument")
    return coerce_tool_calls([_call(name, args)], tools, raw_text_values=True) or []


def dsml(body: str, tools: Any) -> list[dict[str, str]]:
    from .tool_format import _call

    calls = []
    for invoke in re.finditer(
        r'<｜DSML｜invoke name="([^"]+)">(.*?)</｜DSML｜invoke>', body, re.S
    ):
        args = {}
        pattern = re.compile(
            r'<｜DSML｜parameter name="([^"]+)" string="(true|false)">(.*?)</｜DSML｜parameter>',
            re.S,
        )
        if pattern.sub("", invoke[2]).strip():
            raise ValueError("malformed DSML parameter")
        for param in pattern.finditer(invoke[2]):
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
    pos = 0
    decoder = json.JSONDecoder()
    while pos < len(body):
        while pos < len(body) and body[pos].isspace():
            pos += 1
        if pos == len(body):
            break
        header = re.compile(
            r"<\|tool_call_begin\|>functions\.(.*?):\d+\s*<\|tool_call_argument_begin\|>"
        ).match(body, pos)
        if header is None:
            raise ValueError("invalid Kimi call header")
        args_start = header.end()
        while args_start < len(body) and body[args_start].isspace():
            args_start += 1
        _, args_end = decoder.raw_decode(body, args_start)
        pos = args_end
        while pos < len(body) and body[pos].isspace():
            pos += 1
        marker = "<|tool_call_end|>"
        if not body.startswith(marker, pos):
            raise ValueError("missing Kimi call end")
        calls.append(_call(header[1], body[args_start:args_end]))
        pos += len(marker)
    if not calls:
        raise ValueError("missing Kimi call")
    return calls


def harmony(body: str, tools: Any) -> list[dict[str, str]]:
    from .tool_format import _call

    header, sep, args = body.partition("<|message|>")
    match = re.search(r"(?:^|\s)to=([\w.-]+)", header)
    if not sep or match is None:
        raise ValueError("not a Harmony tool message")
    args = re.split(r"<\|(?:ghissue|end|fim_suffix)\|>", args, maxsplit=1)[0]
    return [_call(match[1].removeprefix("functions."), args)]


def mistral(body: str, tools: Any) -> list[dict[str, str]]:
    from .tool_format import _calls_from

    if not body.lstrip().startswith(("[", "{")) and "[ARGS]" in body:
        from .tool_format import _call

        name, args = body.split("[ARGS]", 1)
        return [_call(name.strip(), args)]
    return _calls_from(json.loads(body.strip()))
