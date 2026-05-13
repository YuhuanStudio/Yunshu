"""Reasoning output parser factory.

Auto-detects and parses reasoning/thinking output from different model
families. Each parser handles the specific format used by a model family
to separate reasoning from the final answer.

Supported families:
- Qwen3/3.5: <think/>...</think/> tags
- DeepSeek-R1: <think/>...</think/> with \n patterns
- GLM-4/5: <think/>...</think/> with specific whitespace
- Harmony/gpt_oss: [REASONING]...[/REASONING] markers
- Gemma4: <start_think/>...</end_think/> tags
- Generic: fallback <think/>...</think/> parser
"""
from __future__ import annotations

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReasoningOutput:
    content: str
    reasoning: str | None = None
    reasoning_tokens: int = 0


class ReasoningParser(ABC):
    @abstractmethod
    def parse(self, text: str) -> ReasoningOutput:
        ...

    @abstractmethod
    def family_name(self) -> str:
        ...


class QwenReasoningParser(ReasoningParser):
    """Qwen3/3.5: <think/>...</think/> tags with optional \n."""

    _PATTERN = re.compile(
        r"<think\s*/?\s*>(.*?)</think\s*/?\s*>(.*)",
        re.DOTALL,
    )

    def parse(self, text: str) -> ReasoningOutput:
        m = self._PATTERN.match(text)
        if m:
            reasoning = m.group(1).strip()
            content = m.group(2).strip()
            return ReasoningOutput(
                content=content,
                reasoning=reasoning,
                reasoning_tokens=len(reasoning.split()),
            )
        return ReasoningOutput(content=text.strip())

    def family_name(self) -> str:
        return "qwen"


class DeepSeekReasoningParser(ReasoningParser):
    """DeepSeek-R1/V3: <think/>...</think/> with varied whitespace."""

    _PATTERN = re.compile(
        r"<think\s*/?\s*>\s*(.*?)\s*</think\s*/?\s*>(.*)",
        re.DOTALL,
    )

    def parse(self, text: str) -> ReasoningOutput:
        m = self._PATTERN.match(text)
        if m:
            reasoning = m.group(1).strip()
            content = m.group(2).strip()
            return ReasoningOutput(
                content=content,
                reasoning=reasoning,
                reasoning_tokens=len(reasoning.split()),
            )
        return ReasoningOutput(content=text.strip())

    def family_name(self) -> str:
        return "deepseek"


class GLMReasoningParser(ReasoningParser):
    """GLM-4/5: <think/>...</think/> with specific newline patterns."""

    _PATTERN = re.compile(
        r"<think\s*/?\s*>(.*?)</think\s*/?\s*>(.*)",
        re.DOTALL,
    )

    def parse(self, text: str) -> ReasoningOutput:
        m = self._PATTERN.match(text)
        if m:
            reasoning = m.group(1).strip()
            content = m.group(2).strip()
            return ReasoningOutput(
                content=content,
                reasoning=reasoning,
                reasoning_tokens=len(reasoning.split()),
            )
        return ReasoningOutput(content=text.strip())

    def family_name(self) -> str:
        return "glm"


class HarmonyReasoningParser(ReasoningParser):
    """Harmony/gpt_oss: [REASONING]...[/REASONING] markers."""

    _PATTERN = re.compile(
        r"\[REASONING\](.*?)\[/REASONING\](.*)",
        re.DOTALL | re.IGNORECASE,
    )

    def parse(self, text: str) -> ReasoningOutput:
        m = self._PATTERN.match(text)
        if m:
            reasoning = m.group(1).strip()
            content = m.group(2).strip()
            return ReasoningOutput(
                content=content,
                reasoning=reasoning,
                reasoning_tokens=len(reasoning.split()),
            )
        return ReasoningOutput(content=text.strip())

    def family_name(self) -> str:
        return "harmony"


class GemmaReasoningParser(ReasoningParser):
    """Gemma4: <start_think/>...</end_think/> tags."""

    _PATTERN = re.compile(
        r"<start_think\s*/?\s*>(.*?)</end_think\s*/?\s*>(.*)",
        re.DOTALL,
    )

    def parse(self, text: str) -> ReasoningOutput:
        m = self._PATTERN.match(text)
        if m:
            reasoning = m.group(1).strip()
            content = m.group(2).strip()
            return ReasoningOutput(
                content=content,
                reasoning=reasoning,
                reasoning_tokens=len(reasoning.split()),
            )
        return ReasoningOutput(content=text.strip())

    def family_name(self) -> str:
        return "gemma"


class GenericReasoningParser(ReasoningParser):
    """Generic fallback: tries <think/>...</think/> then returns raw text."""

    _PATTERN = re.compile(
        r"<think\s*/?\s*>(.*?)</think\s*/?\s*>(.*)",
        re.DOTALL,
    )

    def parse(self, text: str) -> ReasoningOutput:
        m = self._PATTERN.match(text)
        if m:
            reasoning = m.group(1).strip()
            content = m.group(2).strip()
            return ReasoningOutput(
                content=content,
                reasoning=reasoning,
                reasoning_tokens=len(reasoning.split()),
            )
        return ReasoningOutput(content=text.strip())

    def family_name(self) -> str:
        return "generic"


# ── Registry & Auto-Detection ────────────────────────────────────────────────

_REGISTRY: dict[str, type[ReasoningParser]] = {
    "qwen": QwenReasoningParser,
    "deepseek": DeepSeekReasoningParser,
    "glm": GLMReasoningParser,
    "harmony": HarmonyReasoningParser,
    "gemma": GemmaReasoningParser,
    "generic": GenericReasoningParser,
}

_MODEL_FAMILY_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"qwen", re.IGNORECASE), "qwen"),
    (re.compile(r"deepseek", re.IGNORECASE), "deepseek"),
    (re.compile(r"glm", re.IGNORECASE), "glm"),
    (re.compile(r"gemma", re.IGNORECASE), "gemma"),
    (re.compile(r"harmony|gpt.?oss", re.IGNORECASE), "harmony"),
]


def get_reasoning_parser(model_name: str | None = None) -> ReasoningParser:
    """Get the appropriate reasoning parser for a model.

    Auto-detects based on model name, falls back to generic.
    """
    family = _detect_family(model_name) if model_name else "generic"
    parser_cls = _REGISTRY.get(family, GenericReasoningParser)
    return parser_cls()


def _detect_family(model_name: str) -> str:
    for pattern, family in _MODEL_FAMILY_HINTS:
        if pattern.search(model_name):
            return family
    return "generic"


def register_reasoning_parser(family: str, parser_cls: type[ReasoningParser]) -> None:
    _REGISTRY[family] = parser_cls
