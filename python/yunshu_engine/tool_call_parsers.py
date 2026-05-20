from __future__ import annotations
"""Yunshu Tool Call Parsers — C15: multi-model tool call extraction.

Different LLM families emit tool calls in different formats. This module
provides a unified ToolCallParser that auto-detects the model's format
and extracts structured tool call results from generated text.

Supported formats:
  1. Qwen: <tool_call/>{"name": "...", "arguments": {...}}</tool_call/>
  2. DeepSeek: <｜tool▁callBegin｜>function<｜tool▁sep｜>name\n```json\n{...}\n```<｜tool▁callEnd｜>
  3. GLM: <|tool_call_block_begin|>name\n```json\n{...}\n```<|tool_call_block_end|>
  4. Llama: [TOOL_CALL] name arguments_json [/TOOL_CALL]
  5. Mistral: [TOOL_CALLS] [{"name": "...", "arguments": {...}}] [/TOOL_CALLS]
  6. Generic: fallback regex-based extraction

Each parser returns a list of ToolCallResult with (id, name, arguments).
The factory auto-selects based on model name or explicit config.
"""

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class ToolCallFormat(Enum):
    """Known tool call output formats."""
    QWEN = auto()
    DEEPSEEK = auto()
    GLM = auto()
    LLAMA = auto()
    MISTRAL = auto()
    GENERIC = auto()


@dataclass
class ToolCallResult:
    """A parsed tool call extracted from model output."""
    id: str = ""
    name: str = ""
    arguments: str = ""  # JSON string

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


# ── Format-specific parsers ──

def _fix_json_arguments(raw: str) -> str:
    """Try to fix common JSON issues in tool-call arguments.

    If the text can be repaired, returns the repaired JSON string.
    Otherwise returns ``"{}"`` (empty JSON object) so that ``arguments``
    is always valid JSON per the OpenAI spec.
    """
    # 1. Quick attempt — maybe it is already valid
    try:
        json.loads(raw)
        return raw
    except (json.JSONDecodeError, ValueError):
        pass

    text = raw.strip()

    # 2. Common fix: missing closing braces/brackets
    open_curly = text.count("{") - text.count("}")
    open_square = text.count("[") - text.count("]")
    if open_curly > 0:
        text += "}" * open_curly
    if open_square > 0:
        text += "]" * open_square

    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. Common fix: unescaped inner double quotes (heuristic — only
    #    attempt when the string is clearly a JSON object)
    if text.startswith("{") and text.endswith("}"):
        escaped = text.replace('\\"', '"')
        try:
            json.loads(escaped)
            return escaped
        except (json.JSONDecodeError, ValueError):
            pass

    # 4. Give up — return empty JSON object
    return "{}"


def _extract_brace_block(text: str, start: int) -> str | None:
    """Extract a brace-balanced block from *text* starting at *start*.

    Handles braces inside JSON strings correctly.
    """
    if start >= len(text) or text[start] != '{':
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_qwen_tool_calls(text: str) -> list[ToolCallResult]:
    """Parse Qwen-style tool calls: <tool_call/>{"name": "...", "arguments": {...}}</tool_call/>"""
    results = []
    tag_re = re.compile(r'<tool_call[^>]*>\s*', re.DOTALL)
    for i, m in enumerate(tag_re.finditer(text)):
        brace_start = text.find('{', m.end())
        if brace_start == -1:
            continue
        block = _extract_brace_block(text, brace_start)
        if block is None:
            continue
        try:
            data = json.loads(block)
            name = data.get("name", "")
            args = data.get("arguments", data.get("parameters", {}))
            args_str = json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args
            results.append(ToolCallResult(id=f"call_{i}", name=name, arguments=args_str))
        except json.JSONDecodeError:
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', block)
            if name_match:
                results.append(ToolCallResult(
                    id=f"call_{i}",
                    name=name_match.group(1),
                    arguments="{}",
                ))
    return results


def parse_deepseek_tool_calls(text: str) -> list[ToolCallResult]:
    """Parse DeepSeek-style tool calls with special token markers.

    DeepSeek uses fullwidth chars in markers:
    - ｜ (U+FF5C) for pipes
    - tool▁callBegin uses ▁ (U+2581 LOWER ONE EIGHTH BLOCK) for underscores
    """
    results = []
    # Primary pattern: fullwidth pipe markers with special token boundaries.
    # Note: the model uses ▁ (U+2581 LOWER ONE EIGHTH BLOCK) in place of _
    pattern = r'｜tool▁callBegin｜function｜tool▁sep｜([\w.\/\-:]+)\s*```json\s*'
    # Fallback: halfwidth variants
    if not re.search(pattern, text):
        pattern = r'[･|]tool_callBegin[･|]function[･|]tool_sep[･|]([\w.\/\-:]+)\s*```json\s*'
    # Last fallback: plain function form
    if not re.search(pattern, text):
        pattern = r'function\s*:\s*([\w.\/\-:]+)\s*```json\s*'
    for i, match in enumerate(re.finditer(pattern, text, re.DOTALL)):
        name = match.group(1)
        brace_start = text.find('{', match.end())
        if brace_start == -1:
            continue
        block = _extract_brace_block(text, brace_start)
        if block is None:
            continue
        try:
            args = json.loads(block)
            args_str = json.dumps(args, ensure_ascii=False)
        except json.JSONDecodeError:
            args_str = _fix_json_arguments(block)
        results.append(ToolCallResult(id=f"call_{i}", name=name, arguments=args_str))
    return results


def parse_glm_tool_calls(text: str) -> list[ToolCallResult]:
    """Parse GLM-style tool calls: <|tool_call_block_begin|>name\n```json\n{...}\n```"""
    results = []
    pattern = r'<\|tool_call_block_begin\|>\s*([\w.\/\-:]+)\s*```(?:json)?\s*'
    for i, match in enumerate(re.finditer(pattern, text, re.DOTALL)):
        name = match.group(1)
        brace_start = text.find('{', match.end())
        if brace_start == -1:
            continue
        block = _extract_brace_block(text, brace_start)
        if block is None:
            continue
        try:
            args = json.loads(block)
            args_str = json.dumps(args, ensure_ascii=False)
        except json.JSONDecodeError:
            args_str = _fix_json_arguments(block)
        results.append(ToolCallResult(id=f"call_{i}", name=name, arguments=args_str))
    return results


def parse_llama_tool_calls(text: str) -> list[ToolCallResult]:
    """Parse Llama-style: [TOOL_CALL] name arguments_json [/TOOL_CALL]"""
    results = []
    tag_re = re.compile(r'\[TOOL_CALL\]\s*([\w.\/\-:]+)\s*', re.DOTALL)
    for i, match in enumerate(tag_re.finditer(text)):
        name = match.group(1)
        brace_start = text.find('{', match.end())
        if brace_start == -1:
            continue
        block = _extract_brace_block(text, brace_start)
        if block is None:
            continue
        try:
            args = json.loads(block)
            args_str = json.dumps(args, ensure_ascii=False)
        except json.JSONDecodeError:
            args_str = _fix_json_arguments(block)
        results.append(ToolCallResult(id=f"call_{i}", name=name, arguments=args_str))
    return results


def _extract_bracket_block(text: str, start: int) -> str | None:
    """Extract a bracket-balanced block from *text* starting at *start*.

    Handles brackets inside JSON strings correctly.
    """
    if start >= len(text) or text[start] != '[':
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_mistral_tool_calls(text: str) -> list[ToolCallResult]:
    """Parse Mistral-style: [TOOL_CALLS] [{"name": "...", "arguments": {...}}] [/TOOL_CALLS]"""
    results = []
    tag_re = re.compile(r'\[TOOL_CALLS\]\s*', re.DOTALL)
    for m in tag_re.finditer(text):
        bracket_start = text.find('[', m.end())
        if bracket_start == -1:
            continue
        block = _extract_bracket_block(text, bracket_start)
        if block is None:
            continue
        try:
            calls = json.loads(block)
            if isinstance(calls, list):
                for i, call in enumerate(calls):
                    name = call.get("name", "")
                    args = call.get("arguments", call.get("parameters", {}))
                    args_str = json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args
                    results.append(ToolCallResult(id=f"call_{i}", name=name, arguments=args_str))
        except json.JSONDecodeError:
            pass
    return results


def parse_generic_tool_calls(text: str) -> list[ToolCallResult]:
    """Generic fallback: look for JSON objects with 'name' and 'arguments' keys."""
    results = []
    # Find "name": then extract the surrounding JSON object
    name_re = re.compile(r'"name"\s*:\s*"([^"]+)"')
    for i, match in enumerate(name_re.finditer(text)):
        name = match.group(1)
        # Walk backward with string-aware brace-depth tracking to find
        # the true enclosing brace.
        depth = 0
        brace_start = -1
        in_string = False
        escape_next = False
        for pos in range(match.start() - 1, -1, -1):
            ch = text[pos]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '}':
                depth += 1
            elif ch == '{':
                if depth == 0:
                    brace_start = pos
                    break
                depth -= 1
        if brace_start == -1:
            continue
        block = _extract_brace_block(text, brace_start)
        if block is None:
            continue
        try:
            data = json.loads(block)
            if not isinstance(data, dict) or "name" not in data:
                continue
            args = data.get("arguments", data.get("parameters", {}))
            args_str = json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args
            results.append(ToolCallResult(id=f"call_{i}", name=name, arguments=args_str))
        except json.JSONDecodeError:
            # JSON parse failed — try to extract arguments field with regex
            # as a last resort before giving up entirely.
            _args_match = re.search(r'"arguments"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', block)
            if _args_match:
                results.append(ToolCallResult(id=f"call_{i}", name=name, arguments=_args_match.group(1)))
            else:
                logger.debug("generic tool call JSON parse failed for name=%s", name)
    return results


# ── Format detection ──

_MODEL_FORMAT_MAP: dict[str, ToolCallFormat] = {
    "qwen": ToolCallFormat.QWEN,
    "qwq": ToolCallFormat.QWEN,
    "deepseek": ToolCallFormat.DEEPSEEK,
    "glm": ToolCallFormat.GLM,
    "chatglm": ToolCallFormat.GLM,
    "llama": ToolCallFormat.LLAMA,
    "mistral": ToolCallFormat.MISTRAL,
    "mixtral": ToolCallFormat.MISTRAL,
    "codestral": ToolCallFormat.MISTRAL,
}


def detect_tool_call_format(model_name: str) -> ToolCallFormat:
    """Auto-detect tool call format from model name."""
    name_lower = model_name.lower()
    for key, fmt in _MODEL_FORMAT_MAP.items():
        if key in name_lower:
            return fmt
    return ToolCallFormat.GENERIC


# ── Parser registry ──

_PARSERS: dict[ToolCallFormat, Any] = {
    ToolCallFormat.QWEN: parse_qwen_tool_calls,
    ToolCallFormat.DEEPSEEK: parse_deepseek_tool_calls,
    ToolCallFormat.GLM: parse_glm_tool_calls,
    ToolCallFormat.LLAMA: parse_llama_tool_calls,
    ToolCallFormat.MISTRAL: parse_mistral_tool_calls,
    ToolCallFormat.GENERIC: parse_generic_tool_calls,
}


class ToolCallParser:
    """Unified tool call parser with auto-detection.

    Usage:
        parser = ToolCallParser(model_name="Qwen2.5-7B-Instruct")
        calls = parser.parse(text)
        for call in calls:
            print(call.name, call.arguments)
    """

    def __init__(
        self,
        model_name: str = "",
        format: ToolCallFormat | None = None,
    ) -> None:
        self._format = format or detect_tool_call_format(model_name)
        self._parser = _PARSERS.get(self._format, parse_generic_tool_calls)
        self._parse_count: int = 0
        self._call_count: int = 0

    @property
    def format(self) -> ToolCallFormat:
        return self._format

    @property
    def format_name(self) -> str:
        return self._format.name

    def parse(self, text: str) -> list[ToolCallResult]:
        """Parse tool calls from generated text."""
        if not text:
            return []
        results = self._parser(text)
        self._parse_count += 1
        self._call_count += len(results)
        return results

    def get_stats(self) -> dict[str, Any]:
        return {
            "format": self.format_name,
            "parse_count": self._parse_count,
            "call_count": self._call_count,
            "avg_calls_per_parse": (
                round(self._call_count / max(self._parse_count, 1), 2)
            ),
        }

    @staticmethod
    def register_parser(format: ToolCallFormat, parser_fn: Any) -> None:
        """Register a custom parser for a format."""
        _PARSERS[format] = parser_fn
