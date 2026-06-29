from __future__ import annotations

"""Tool Call Parser Factory.

Auto-detects model-specific tool call output formats and routes to
the correct parser. Replaces the static extract_tool_calls function
with a pluggable factory pattern.

Supported formats:
1. Hermes: <tool_call/>{"name": ..., "arguments": ...}</tool_call/>
2. Qwen/Llama XML: <function=name>{"param": value}</function>
3. Direct JSON: {"name": ..., "arguments": ...}
4. Code block: ```json\n{"name": ...}\n```
5. Mistral: {"function": {"name": ..., "arguments": ...}}
6. ChatML: [TOOL_CALLS] [{"name": ...}]
7. DeepSeek: ✿FUNCTION✿ {"name": ...} ✿
8. Anthropic: native tool_use blocks (pre-parsed)
9. Gemini: FunctionCall JSON in response parts
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    arguments: str  # JSON string


class ToolCallParser(ABC):
    @abstractmethod
    def parse(self, text: str) -> list[ToolCall]: ...

    @abstractmethod
    def format_name(self) -> str: ...


def _sanitize_arguments(args) -> str:
    if isinstance(args, str):
        return args
    return json.dumps(args, ensure_ascii=False)


def _extract_brace_block(text: str, start: int) -> str | None:
    """Extract a brace-balanced JSON object starting at position *start*.

    Handles braces inside JSON strings correctly.
    Returns the matched substring or None if unmatched.
    """
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# ── Format 1: Hermes <tool_call/> ──


class HermesToolCallParser(ToolCallParser):
    _RE = re.compile(r"<tool_call\s*/?\s*>(.*?)</tool_call\s*/?\s*>", re.DOTALL)

    def parse(self, text: str) -> list[ToolCall]:
        calls = []
        for m in self._RE.finditer(text):
            try:
                data = json.loads(m.group(1).strip())
                if "name" in data:
                    calls.append(
                        ToolCall(
                            name=data["name"],
                            arguments=_sanitize_arguments(
                                data.get("arguments") or data.get("parameters") or {}
                            ),
                        )
                    )
            except json.JSONDecodeError:
                continue
        return calls

    def format_name(self) -> str:
        return "hermes"


# ── Format 2: Qwen/Llama XML <function=name> ──


class QwenXMLToolCallParser(ToolCallParser):
    _FUNC_RE = re.compile(r"<function\s*=\s*([\w.\-]+)>(.*?)</function>", re.DOTALL)
    _PARAM_RE = re.compile(r"<parameter\s*=\s*([\w.\-]+)>(.*?)</parameter>", re.DOTALL)

    def parse(self, text: str) -> list[ToolCall]:
        calls = []
        for m in self._FUNC_RE.finditer(text):
            name = m.group(1).strip()
            body = m.group(2).strip()
            try:
                data = json.loads(body)
                calls.append(ToolCall(name=name, arguments=_sanitize_arguments(data)))
            except json.JSONDecodeError:
                params = {}
                for pm in self._PARAM_RE.finditer(body):
                    params[pm.group(1)] = pm.group(2).strip()
                if params:
                    calls.append(
                        ToolCall(
                            name=name, arguments=json.dumps(params, ensure_ascii=False)
                        )
                    )
        return calls

    def format_name(self) -> str:
        return "qwen_xml"


# ── Format 3: Direct JSON ──


class DirectJSONToolCallParser(ToolCallParser):
    """Parses bare JSON objects containing a 'name' key.

    Uses brace counting instead of a naive regex so that nested JSON
    objects in ``arguments`` are captured correctly.  The previous
    ``[^{}]*`` pattern could never match real tool calls whose
    ``arguments`` value is a dict (it contains ``{``/``}``).
    """

    # Quick pre-filter: must contain "name" key somewhere
    _HAS_NAME = re.compile(r'"name"\s*:', re.DOTALL)

    def parse(self, text: str) -> list[ToolCall]:
        calls: list[ToolCall] = []
        if not self._HAS_NAME.search(text):
            return calls

        depth = 0
        start = -1
        in_string = False
        escape_next = False
        for i, ch in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = text[start : i + 1]
                    try:
                        data = json.loads(candidate)
                        if isinstance(data, dict) and "name" in data:
                            calls.append(
                                ToolCall(
                                    name=data["name"],
                                    arguments=_sanitize_arguments(
                                        data.get("arguments")
                                        or data.get("parameters")
                                        or {}
                                    ),
                                )
                            )
                    except json.JSONDecodeError:
                        pass
                    start = -1
        return calls

    def format_name(self) -> str:
        return "direct_json"


# ── Format 5: Mistral ──


class MistralToolCallParser(ToolCallParser):
    def parse(self, text: str) -> list[ToolCall]:
        calls = []
        depth = 0
        start = -1
        in_string = False
        escape_next = False
        for i, ch in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        data = json.loads(text[start : i + 1])
                        func = data.get("function", {})
                        name = func.get("name", "") if isinstance(func, dict) else ""
                        if (
                            not name
                            and "name" in data
                            and isinstance(data.get("name"), str)
                        ):
                            name = data["name"]
                            func = data
                        if name:
                            args = func.get("arguments", {})
                            if isinstance(args, str):
                                args = json.loads(args)
                            calls.append(
                                ToolCall(name=name, arguments=_sanitize_arguments(args))
                            )
                    except (json.JSONDecodeError, KeyError):
                        pass
                    start = -1
        return calls

    def format_name(self) -> str:
        return "mistral"


# ── Format 6: ChatML [TOOL_CALLS] ──


class ChatMLToolCallParser(ToolCallParser):
    """Parses Mistral-style ``[TOOL_CALLS] [...]`` blocks.

    Uses bracket counting instead of ``\\[.*?\\]`` so that tool call
    arguments containing nested arrays (e.g. ``\"tags\": [\"a\"]``) are
    captured correctly.
    """

    _PREFIX_RE = re.compile(r"\[TOOL_CALLS\]\s*", re.DOTALL)

    def parse(self, text: str) -> list[ToolCall]:
        calls = []
        for m in self._PREFIX_RE.finditer(text):
            rest = text[m.end() :]
            # Extract the JSON array using bracket counting
            arr_text = self._extract_array(rest)
            if arr_text is None:
                continue
            try:
                arr = json.loads(arr_text)
                if not isinstance(arr, list):
                    continue
                for item in arr:
                    if isinstance(item, dict):
                        name = (
                            item.get("name")
                            or item.get("function", {}).get("name")
                            or ""
                        )
                        args = (
                            item.get("arguments")
                            or item.get("function", {}).get("arguments")
                            or {}
                        )
                        if name:
                            if isinstance(args, str):
                                args = json.loads(args)
                            calls.append(
                                ToolCall(name=name, arguments=_sanitize_arguments(args))
                            )
            except (json.JSONDecodeError, KeyError):
                continue
        return calls

    @staticmethod
    def _extract_array(text: str) -> str | None:
        """Extract a JSON array from the start of *text* using bracket counting.

        Handles brackets inside JSON strings correctly by tracking quote state.
        """
        if not text or text[0] != "[":
            return None
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[: i + 1]
        return None

    def format_name(self) -> str:
        return "chatml"


# ── Format 7: DeepSeek ✿FUNCTION✿ ──


class DeepSeekToolCallParser(ToolCallParser):
    """Parses DeepSeek-style ✿FUNCTION✿ blocks.

    Uses brace counting instead of ``\\{.*?\\}`` so that nested JSON
    objects in ``arguments`` are captured correctly.  The previous
    ``\\{.*?\\}`` pattern stopped at the first ``}`` and truncated
    any dict/array arguments.
    """

    _PREFIX = "✿FUNCTION✿"
    _SUFFIX = "✿"

    def parse(self, text: str) -> list[ToolCall]:
        calls = []
        idx = 0
        while True:
            start = text.find(self._PREFIX, idx)
            if start == -1:
                break
            brace_start = text.find("{", start + len(self._PREFIX))
            if brace_start == -1:
                break
            candidate = _extract_brace_block(text, brace_start)
            if candidate is None:
                idx = brace_start + 1
                continue
            try:
                data = json.loads(candidate)
                name = data.get("name", "")
                if name:
                    args = data.get("arguments") or data.get("parameters") or {}
                    if isinstance(args, str):
                        args = json.loads(args)
                    calls.append(
                        ToolCall(name=name, arguments=_sanitize_arguments(args))
                    )
            except (json.JSONDecodeError, KeyError):
                pass
            idx = brace_start + len(candidate)
        return calls

    def format_name(self) -> str:
        return "deepseek"


# ── Format 8: Anthropic native (already parsed) ──


class AnthropicToolCallParser(ToolCallParser):
    """For Anthropic models that output tool_use blocks (already parsed by API)."""

    _RE = re.compile(
        r"<tool_use\b[^>]*>(.*?)</tool_use>",
        re.DOTALL,
    )
    _NAME_RE = re.compile(r"<name>(.*?)</name>", re.DOTALL)
    _INPUT_RE = re.compile(r"<input>(.*?)</input>", re.DOTALL)

    def parse(self, text: str) -> list[ToolCall]:
        calls = []
        for m in self._RE.finditer(text):
            block = m.group(1)
            name_m = self._NAME_RE.search(block)
            input_m = self._INPUT_RE.search(block)
            if name_m:
                name = name_m.group(1).strip()
                input_str = input_m.group(1).strip() if input_m else "{}"
                try:
                    args = json.loads(input_str)
                except json.JSONDecodeError:
                    args = {}
                calls.append(ToolCall(name=name, arguments=_sanitize_arguments(args)))
        return calls

    def format_name(self) -> str:
        return "anthropic"


# ── Format 9: Gemini FunctionCall ──


class GeminiToolCallParser(ToolCallParser):
    """Parses Gemini-style functionCall JSON in response parts."""

    def parse(self, text: str) -> list[ToolCall]:
        calls = []
        idx = 0
        key = '"functionCall"'
        while True:
            pos = text.find(key, idx)
            if pos == -1:
                break
            brace_start = text.find("{", pos + len(key))
            if brace_start == -1:
                break
            # Use string-aware brace counting
            candidate = _extract_brace_block(text, brace_start)
            if candidate is None:
                idx = brace_start + 1
                continue
            try:
                data = json.loads(candidate)
                name = data.get("name", "")
                if name:
                    args = data.get("args") or data.get("arguments") or {}
                    calls.append(
                        ToolCall(name=name, arguments=_sanitize_arguments(args))
                    )
            except (json.JSONDecodeError, KeyError):
                pass
            idx = brace_start + len(candidate)
        return calls

    def format_name(self) -> str:
        return "gemini"


class GLMToolCallParser(ToolCallParser):
    """GLM-4.x tool calls — BOTH the older block form and the GLM-4.6/4.7 key/value form.

    This flush-parser registry (tool_call_parser.py) had drifted from the
    non-streaming registry in tool_call_parsers.py, which DOES have a GLM parser. So the
    streamer's BUFFER_ALL flush could not parse GLM output even after it was routed here —
    GLM streaming tool calls were silently dropped (or the raw markup leaked as text).
    Delegate to the authoritative parser in tool_call_parsers.parse_glm_tool_calls (handles
    `<|tool_call_block_begin|>name```json{...}``` ` and
    `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`).
    """

    def parse(self, text: str) -> list[ToolCall]:
        from .tool_call_parsers import parse_glm_tool_calls

        return [
            ToolCall(name=r.name, arguments=_sanitize_arguments(r.arguments))
            for r in parse_glm_tool_calls(text)
            if r.name
        ]

    def format_name(self) -> str:
        return "glm"


# ── Factory ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[ToolCallParser]] = {
    "hermes": HermesToolCallParser,
    "qwen_xml": QwenXMLToolCallParser,
    "direct_json": DirectJSONToolCallParser,
    "mistral": MistralToolCallParser,
    "chatml": ChatMLToolCallParser,
    "deepseek": DeepSeekToolCallParser,
    "anthropic": AnthropicToolCallParser,
    "gemini": GeminiToolCallParser,
    "glm": GLMToolCallParser,
}

_MODEL_HINTS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"qwen", re.IGNORECASE), ["qwen_xml", "hermes", "direct_json"]),
    (re.compile(r"deepseek", re.IGNORECASE), ["deepseek", "hermes", "direct_json"]),
    (re.compile(r"llama", re.IGNORECASE), ["hermes", "qwen_xml", "direct_json"]),
    (re.compile(r"mistral", re.IGNORECASE), ["mistral", "hermes", "direct_json"]),
    (re.compile(r"claude|anthropic", re.IGNORECASE), ["anthropic", "direct_json"]),
    (re.compile(r"gemma", re.IGNORECASE), ["hermes", "direct_json"]),
    (re.compile(r"gemin[iy]", re.IGNORECASE), ["gemini", "direct_json"]),
    # GLM-4.x has its own key/value tool-call form — route it to the GLM parser
    # first (then direct_json as a JSON-arg fallback).
    (re.compile(r"glm|chatglm", re.IGNORECASE), ["glm", "direct_json"]),
]


def parse_tool_calls(text: str, model_name: str | None = None) -> list[ToolCall]:
    """Parse tool calls from model output with auto-detection.

    Tries model-specific parsers first (based on model name), then
    falls back to trying all parsers. Returns the first non-empty result.
    """
    if model_name:
        for pattern, formats in _MODEL_HINTS:
            if pattern.search(model_name):
                for fmt in formats:
                    parser = _REGISTRY.get(fmt)
                    if parser:
                        calls = parser().parse(text)
                        if calls:
                            return calls
                break

    # Try all parsers in order
    for parser_cls in _REGISTRY.values():
        calls = parser_cls().parse(text)
        if calls:
            return calls

    return []


def _remove_chatml_blocks(text: str) -> str:
    """Remove [TOOL_CALLS] [...] blocks with proper bracket nesting."""
    result = []
    i = 0
    prefix = "[TOOL_CALLS]"
    while i < len(text):
        pos = text.find(prefix, i)
        if pos == -1:
            result.append(text[i:])
            break
        # Keep text before the block
        result.append(text[i:pos])
        # Find the JSON array after the prefix
        rest = text[pos + len(prefix) :]
        stripped = rest.lstrip()
        skip = len(rest) - len(stripped)
        if stripped and stripped[0] == "[":
            block = ChatMLToolCallParser._extract_array(stripped)
            if block is not None:
                i = pos + len(prefix) + skip + len(block)
                continue
        # If we can't extract, skip the prefix and move on
        i = pos + len(prefix)
    return "".join(result)


def clean_tool_markup(text: str) -> str:
    """Remove tool call markup from text, leaving clean content."""
    text = re.sub(
        r"<tool_call\s*/?\s*>.*?</tool_call\s*/?\s*>", "", text, flags=re.DOTALL
    )
    text = re.sub(r"<function\s*=\s*[\w.\-]+>.*?</function>", "", text, flags=re.DOTALL)
    text = re.sub(r"<tool_use\b[^>]*>.*?</tool_use>", "", text, flags=re.DOTALL)
    text = _remove_chatml_blocks(text)
    text = re.sub(r"✿FUNCTION✿.*?✿", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
