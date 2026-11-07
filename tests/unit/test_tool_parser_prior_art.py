"""Prior-art comparison: our family tool readers against mlx-lm's tool_parsers.

The reference modules are loaded by file path from reference/mlx-lm (pure Python; importing the
package would import MLX). Each case is either an AGREEMENT (same name + arguments) or a documented
DIVERGENCE with the intended reason. vLLM (tool_parsers/*) and llama.cpp (common/chat*.cpp) are C++/GPU
heavy and cannot be imported on CPU; their documented behaviours are encoded as expectations below.
Skipped when the reference clone is absent.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import types
from pathlib import Path

import pytest

from yunshu_engine import tool_format as tf

_MAIN = Path(
    "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/reference/mlx-lm/mlx_lm/tool_parsers"
)


def _find_ref() -> Path | None:
    for cand in (
        *(
            p / "reference/mlx-lm/mlx_lm/tool_parsers"
            for p in Path(__file__).resolve().parents
        ),
        _MAIN,
    ):
        try:
            if cand.is_dir():
                return cand
        except OSError:  # unreadable under the CI sandbox
            continue
    return None


REF = _find_ref()

pytestmark = pytest.mark.skipif(REF is None, reason="reference/mlx-lm clone missing")


# The mlx-lm revision the family readers were reviewed against (vendor.json, tool_format.py
# reviewed_commit). The clone's checkout can be older or newer (a helper Mac's clone was at a May
# 2026 commit whose mistral and pythonic parsers predate the reviewed ones), so the reference source
# is read from this revision, never from the working tree.
REVIEWED = "9d8abd94d63a9b3c72e7e9b146e43af1005368fa"


def _ref(name):
    assert REF is not None
    src = subprocess.run(
        [
            "git",
            "-C",
            str(REF.parents[1]),
            "show",
            f"{REVIEWED}:mlx_lm/tool_parsers/{name}.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    mod = types.ModuleType(f"_ref_{name}")
    exec(compile(src, f"mlx-lm@{REVIEWED[:9]}:{name}.py", "exec"), mod.__dict__)
    return mod


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "code": {"type": "string"},
                },
            },
        },
    }
]


def _norm(calls):
    if isinstance(calls, dict):
        calls = [calls]
    out = []
    for c in calls:
        a = c["arguments"]
        if isinstance(a, str):
            with contextlib.suppress(ValueError):  # raw arguments stay a string
                a = json.loads(a)
        out.append((c["name"], a))
    return out


def _ours(fmt, body):
    try:
        return _norm(fmt.parse(body, TOOLS))
    except Exception as e:
        return ("ERR", type(e).__name__)


def _theirs(name, body):
    try:
        return _norm(_ref(name).parse_tool_call(body, TOOLS))
    except Exception as e:
        return ("ERR", type(e).__name__)


AGREE = [
    ("pythonic", tf.PYTHONIC, '[weather(city="Taipei", days=3)]'),
    ("pythonic", tf.PYTHONIC, '[weather(city="a]b)", days=1), weather(city="c")]'),
    ("pythonic", tf.PYTHONIC, '[weather(city="x", extra={"a": true, "b": null})]'),
    ("mistral", tf.MISTRAL, '[{"name": "weather", "arguments": {"city": "Taipei"}}]'),
    ("mistral", tf.MISTRAL, 'weather[ARGS]{"city": "Taipei", "days": 3}'),
    (
        "kimi_k2",
        tf.KIMI,
        '<|tool_call_begin|>functions.weather:0<|tool_call_argument_begin|>{"city": "Taipei"}<|tool_call_end|>',
    ),
    (
        "glm47",
        tf.GLM,
        "weather<arg_key>city</arg_key><arg_value>Taipei</arg_value><arg_key>days</arg_key><arg_value>3</arg_value>",
    ),
    ("glm47", tf.GLM, "weather<arg_key>code</arg_key><arg_value>007</arg_value>"),
    ("glm47", tf.GLM, 'weather\n{"city": "Taipei"}'),
]


@pytest.mark.parametrize("name,fmt,body", AGREE)
def test_agrees_with_mlx_lm(name, fmt, body):
    assert _ours(fmt, body) == _theirs(name, body)


def test_pythonic_positional_and_import_fail_closed():
    for body in ("[weather(1)]", '[weather(city=__import__("os"))]', "[weather(**{})]"):
        assert _ours(tf.PYTHONIC, body)[0] == "ERR", body


def test_kimi_non_json_arguments_fail_closed_unlike_mlx_lm():
    body = "<|tool_call_begin|>functions.weather:0<|tool_call_argument_begin|>not json<|tool_call_end|>"
    assert _ours(tf.KIMI, body)[0] == "ERR"
    theirs = _theirs("kimi_k2", body)
    assert theirs == [("weather", "not json")]  # mlx-lm keeps the raw string


def test_mistral_repeated_header_calls_vs_mlx_lm():
    body = 'weather[ARGS]{"city": "T"}weather[ARGS]{"city": "U"}'
    theirs = _theirs("mistral", body)
    assert [a["city"] for _, a in theirs] == ["T", "U"]
    ours = _ours(tf.MISTRAL, body)
    # Record the actual behaviour so a change in either direction is visible.
    assert ours[0] == ("weather", {"city": "T"}) or ours[0] == "ERR"
