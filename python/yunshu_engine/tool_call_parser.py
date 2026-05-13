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
from __future__ import annotations

import json
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    arguments: str  # JSON string


class ToolCallParser(ABC):
    @abstractmethod
    def parse(self, text: str) -> list[ToolCall]:
        ...

    @abstractmethod
    def format_name(self) -> str:
        ...


def _sanitize_arguments(args) -> str:
    if isinstance(args, str):
        return args
    return json.dumps(args, ensure_ascii=False)


# ── Format 1: Hermes <tool_call/> ──

class HermesToolCallParser(ToolCallParser):
    _RE = re.compile(r"<tool_call\s*/?\s*>(.*?)</tool_call\s*/?\s*>", re.DOTALL)

    def parse(self, text: str) -> list[ToolCall]:
        calls = []
        for m in self._RE.finditer(text):
            try:
                data = json.loads(m.group(1).strip())
                if "name" in data:
                    calls.append(ToolCall(
                        name=data["name"],
                        arguments=_sanitize_arguments(data.get("arguments", data.get("parameters", {}))),
                    ))
            except json.JSONDecodeError:
                continue
        return calls

    def format_name(self) -> str:
        return "hermes"


# ── Format 2: Qwen/Llama XML <function=name> ──

class QwenXMLToolCallParser(ToolCallParser):
    _FUNC_RE = re.compile(r"<function\s*=\s*(\w+)>(.*?)</function>", re.DOTALL)
    _PARAM_RE = re.compile(r"<parameter\s*=\s*(\w+)>(.*?)</parameter>", re.DOTALL)

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
                    calls.append(ToolCall(name=name, arguments=json.dumps(params, ensure_ascii=False)))
        return calls

    def format_name(self) -> str:
        return "qwen_xml"


# ── Format 3: Direct JSON ──

class DirectJSONToolCallParser(ToolCallParser):
    _RE = re.compile(r'\{[^{}]*?"name"\s*:\s*"[^"]+?"[^{}]*?\}', re.DOTALL)

    def parse(self, text: str) -> list[ToolCall]:
        calls = []
        for m in self._RE.finditer(text):
            try:
                data = json.loads(m.group(0))
                if isinstance(data, dict) and "name" in data:
                    calls.append(ToolCall(
                        name=data["name"],
                        arguments=_sanitize_arguments(data.get("arguments", data.get("parameters", {}))),
                    ))
            except json.JSONDecodeError:
                continue
        return calls

    def format_name(self) -> str:
        return "direct_json"


# ── Format 5: Mistral ──

class MistralToolCallParser(ToolCallParser):
    def parse(self, text: str) -> list[ToolCall]:
        calls = []
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        data = json.loads(text[start:i + 1])
                        func = data.get("function", {})
                        name = func.get("name", "") if isinstance(func, dict) else ""
                        if not name and "name" in data and isinstance(data.get("name"), str):
                            name = data["name"]
                            func = data
                        if name:
                            args = func.get("arguments", {})
                            if isinstance(args, str):
                                args = json.loads(args)
                            calls.append(ToolCall(name=name, arguments=_sanitize_arguments(args)))
                    except (json.JSONDecodeError, KeyError):
                        pass
                    start = -1
        return calls

    def format_name(self) -> str:
        return "mistral"


# ── Format 6: ChatML [TOOL_CALLS] ──

class ChatMLToolCallParser(ToolCallParser):
    _RE = re.compile(r'\[TOOL_CALLS\]\s*(\[.*?\])', re.DOTALL)

    def parse(self, text: str) -> list[ToolCall]:
        calls = []
        for m in self._RE.finditer(text):
            try:
                arr = json.loads(m.group(1))
                for item in arr:
                    if isinstance(item, dict):
                        name = item.get("name", item.get("function", {}).get("name", ""))
                        args = item.get("arguments", item.get("function", {}).get("arguments", {}))
                        if name:
                            if isinstance(args, str):
                                args = json.loads(args)
                            calls.append(ToolCall(name=name, arguments=_sanitize_arguments(args)))
            except (json.JSONDecodeError, KeyError):
                continue
        return calls

    def format_name(self) -> str:
        return "chatml"


# ── Format 7: DeepSeek ✿FUNCTION✿ ──

class DeepSeekToolCallParser(ToolCallParser):
    _RE = re.compile(r'✿FUNCTION✿\s*(\{.*?\})\s*✿', re.DOTALL)

    def parse(self, text: str) -> list[ToolCall]:
        calls = []
        for m in self._RE.finditer(text):
            try:
                data = json.loads(m.group(1))
                name = data.get("name", "")
                if name:
                    args = data.get("arguments", data.get("parameters", {}))
                    if isinstance(args, str):
                        args = json.loads(args)
                    calls.append(ToolCall(name=name, arguments=_sanitize_arguments(args)))
            except (json.JSONDecodeError, KeyError):
                continue
        return calls

    def format_name(self) -> str:
        return "deepseek"


# ── Format 8: Anthropic native (already parsed) ──

class AnthropicToolCallParser(ToolCallParser):
    """For Anthropic models that output tool_use blocks (already parsed by API)."""

    _RE = re.compile(
        r'<tool_use\b[^>]*>(.*?)</tool_use>',
        re.DOTALL,
    )
    _NAME_RE = re.compile(r'<name>(.*?)</name>', re.DOTALL)
    _INPUT_RE = re.compile(r'<input>(.*?)</input>', re.DOTALL)

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
            brace_start = text.find('{', pos + len(key))
            if brace_start == -1:
                break
            depth = 0
            end = brace_start
            for end in range(brace_start, len(text)):
                if text[end] == '{':
                    depth += 1
                elif text[end] == '}':
                    depth -= 1
                    if depth == 0:
                        break
            candidate = text[brace_start:end + 1]
            try:
                data = json.loads(candidate)
                name = data.get("name", "")
                if name:
                    args = data.get("args", data.get("arguments", {}))
                    calls.append(ToolCall(name=name, arguments=_sanitize_arguments(args)))
            except (json.JSONDecodeError, KeyError):
                pass
            idx = end + 1
        return calls

    def format_name(self) -> str:
        return "gemini"


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
}

_MODEL_HINTS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"qwen", re.IGNORECASE), ["qwen_xml", "hermes", "direct_json"]),
    (re.compile(r"deepseek", re.IGNORECASE), ["deepseek", "hermes", "direct_json"]),
    (re.compile(r"llama", re.IGNORECASE), ["hermes", "qwen_xml", "direct_json"]),
    (re.compile(r"mistral", re.IGNORECASE), ["mistral", "hermes", "direct_json"]),
    (re.compile(r"claude|anthropic", re.IGNORECASE), ["anthropic", "direct_json"]),
    (re.compile(r"gemma", re.IGNORECASE), ["hermes", "direct_json"]),
    (re.compile(r"gemin[iy]", re.IGNORECASE), ["gemini", "direct_json"]),
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


def clean_tool_markup(text: str) -> str:
    """Remove tool call markup from text, leaving clean content."""
    text = re.sub(r"<tool_call\s*/?\s*>.*?</tool_call\s*/?\s*>", "", text, flags=re.DOTALL)
    text = re.sub(r"<function\s*=\s*\w+>.*?</function>", "", text, flags=re.DOTALL)
    text = re.sub(r"<tool_use\b[^>]*>.*?</tool_use>", "", text, flags=re.DOTALL)
    text = re.sub(r"\[TOOL_CALLS\]\s*\[.*?\]", "", text, flags=re.DOTALL)
    text = re.sub(r"✿FUNCTION✿.*?✿", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
