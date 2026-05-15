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

def parse_qwen_tool_calls(text: str) -> list[ToolCallResult]:
    """Parse Qwen-style tool calls: <tool_call/>{"name": "...", "arguments": {...}}</tool_call/>"""
    results = []
    pattern = r'<tool_call[^>]*>\s*(\{.*?\})\s*</tool_call[^>]*>'
    for i, match in enumerate(re.finditer(pattern, text, re.DOTALL)):
        try:
            data = json.loads(match.group(1))
            name = data.get("name", "")
            args = data.get("arguments", data.get("parameters", {}))
            args_str = json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args
            results.append(ToolCallResult(id=f"call_{i}", name=name, arguments=args_str))
        except json.JSONDecodeError:
            # Fallback regex
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', match.group(1))
            if name_match:
                results.append(ToolCallResult(
                    id=f"call_{i}",
                    name=name_match.group(1),
                    arguments="{}",
                ))
    return results


def parse_deepseek_tool_calls(text: str) -> list[ToolCallResult]:
    """Parse DeepSeek-style tool calls with special token markers."""
    results = []
    pattern = r'･tool_callBegin･function･tool_sep･(\w+)\s*```json\s*(\{.*?\})\s*```'
    # Also try the visible form
    if not re.search(pattern, text):
        pattern = r'function\s*:\s*(\w+)\s*```json\s*(\{.*?\})\s*```'
    for i, match in enumerate(re.finditer(pattern, text, re.DOTALL)):
        name = match.group(1)
        try:
            args = json.loads(match.group(2))
            args_str = json.dumps(args, ensure_ascii=False)
        except json.JSONDecodeError:
            args_str = match.group(2)
        results.append(ToolCallResult(id=f"call_{i}", name=name, arguments=args_str))
    return results


def parse_glm_tool_calls(text: str) -> list[ToolCallResult]:
    """Parse GLM-style tool calls: <|tool_call_block_begin|>name\n```json\n{...}\n```"""
    results = []
    pattern = r'<\|tool_call_block_begin\|>\s*(\w+)\s*```(?:json)?\s*(\{.*?\})\s*```'
    for i, match in enumerate(re.finditer(pattern, text, re.DOTALL)):
        name = match.group(1)
        try:
            args = json.loads(match.group(2))
            args_str = json.dumps(args, ensure_ascii=False)
        except json.JSONDecodeError:
            args_str = match.group(2)
        results.append(ToolCallResult(id=f"call_{i}", name=name, arguments=args_str))
    return results


def parse_llama_tool_calls(text: str) -> list[ToolCallResult]:
    """Parse Llama-style: [TOOL_CALL] name arguments_json [/TOOL_CALL]"""
    results = []
    pattern = r'\[TOOL_CALL\]\s*(\w+)\s*(\{.*?\})\s*\[/TOOL_CALL\]'
    for i, match in enumerate(re.finditer(pattern, text, re.DOTALL)):
        name = match.group(1)
        try:
            args = json.loads(match.group(2))
            args_str = json.dumps(args, ensure_ascii=False)
        except json.JSONDecodeError:
            args_str = match.group(2)
        results.append(ToolCallResult(id=f"call_{i}", name=name, arguments=args_str))
    return results


def parse_mistral_tool_calls(text: str) -> list[ToolCallResult]:
    """Parse Mistral-style: [TOOL_CALLS] [{"name": "...", "arguments": {...}}] [/TOOL_CALLS]"""
    results = []
    pattern = r'\[TOOL_CALLS\]\s*(\[.*?\])\s*\[/TOOL_CALLS\]'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            calls = json.loads(match.group(1))
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
    # Look for function_name(args) pattern
    pattern = r'"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})'
    for i, match in enumerate(re.finditer(pattern, text, re.DOTALL)):
        name = match.group(1)
        try:
            args = json.loads(match.group(2))
            args_str = json.dumps(args, ensure_ascii=False)
        except json.JSONDecodeError:
            args_str = match.group(2)
        results.append(ToolCallResult(id=f"call_{i}", name=name, arguments=args_str))
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
