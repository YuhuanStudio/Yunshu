from __future__ import annotations
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

All parsers now:
- Use finditer/search instead of match() to handle leading whitespace
- Handle multiple think/unthink cycles (concatenating reasoning)
- Count reasoning_tokens by character length (not word split — wrong for CJK)
"""

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
    """Base class with shared extraction logic.

    Subclasses define _OPEN_RE and _CLOSE_RE for their tag format.
    The parse() method handles multiple think/unthink cycles and
    leading content before the first tag.
    """

    _OPEN_RE: re.Pattern  # opening tag pattern
    _CLOSE_RE: re.Pattern  # closing tag pattern

    @abstractmethod
    def family_name(self) -> str:
        ...

    def parse(self, text: str) -> ReasoningOutput:
        """Extract reasoning from text, handling multiple cycles.

        Handles:
        - Leading content before first think tag
        - Multiple think/unthink cycles (reasoning concatenated)
        - Unclosed think tags (rest treated as reasoning)
        """
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        pos = 0

        for m in self._OPEN_RE.finditer(text):
            # Everything before this open tag is content
            content_parts.append(text[pos:m.start()])
            think_start = m.end()
            # Find the matching close tag
            close_m = self._CLOSE_RE.search(text, think_start)
            if close_m:
                reasoning_parts.append(text[think_start:close_m.start()].strip())
                pos = close_m.end()
            else:
                # Unclosed tag — rest is reasoning
                reasoning_parts.append(text[think_start:].strip())
                pos = len(text)
                break

        # Any remaining text after the last close tag is content
        if pos < len(text):
            content_parts.append(text[pos:])

        if reasoning_parts:
            reasoning = "\n".join(r for r in reasoning_parts if r)
            content = "".join(content_parts).strip()
            return ReasoningOutput(
                content=content,
                reasoning=reasoning,
                reasoning_tokens=len(reasoning),
            )
        return ReasoningOutput(content=text.strip())


# ── Tag patterns ──────────────────────────────────────────────────────────────

_THINK_OPEN = re.compile(r"<think\s*/?\s*>", re.DOTALL)
_THINK_CLOSE = re.compile(r"</think\s*/?\s*>", re.DOTALL)
_REASONING_OPEN = re.compile(r"\[REASONING\]", re.DOTALL | re.IGNORECASE)
_REASONING_CLOSE = re.compile(r"\[/REASONING\]", re.DOTALL | re.IGNORECASE)
_START_THINK_OPEN = re.compile(r"<start_think\s*/?\s*>", re.DOTALL)
_END_THINK_CLOSE = re.compile(r"</end_think\s*/?\s*>", re.DOTALL)
_BRACKET_THINK_OPEN = re.compile(r"\[THINK\]", re.DOTALL)
_BRACKET_THINK_CLOSE = re.compile(r"\[/THINK\]", re.DOTALL)
_COHERE_OPEN = re.compile(r"<\|START_THINKING\|>", re.DOTALL)
_COHERE_CLOSE = re.compile(r"<\|END_THINKING\|>", re.DOTALL)


# ── Parser subclasses ────────────────────────────────────────────────────────

class QwenReasoningParser(ReasoningParser):
    """Qwen3/3.5: <think/>...</think/> tags."""
    _OPEN_RE = _THINK_OPEN
    _CLOSE_RE = _THINK_CLOSE

    def family_name(self) -> str:
        return "qwen"


class DeepSeekReasoningParser(ReasoningParser):
    """DeepSeek-R1/V3: <think/>...</think/> with varied whitespace."""
    _OPEN_RE = _THINK_OPEN
    _CLOSE_RE = _THINK_CLOSE

    def family_name(self) -> str:
        return "deepseek"


class GLMReasoningParser(ReasoningParser):
    """GLM-4/5: <think/>...</think/> with specific newline patterns."""
    _OPEN_RE = _THINK_OPEN
    _CLOSE_RE = _THINK_CLOSE

    def family_name(self) -> str:
        return "glm"


class HarmonyReasoningParser(ReasoningParser):
    """Harmony/gpt_oss: [REASONING]...[/REASONING] markers."""
    _OPEN_RE = _REASONING_OPEN
    _CLOSE_RE = _REASONING_CLOSE

    def family_name(self) -> str:
        return "harmony"


class GemmaReasoningParser(ReasoningParser):
    """Gemma4: <start_think/>...</end_think/> tags."""
    _OPEN_RE = _START_THINK_OPEN
    _CLOSE_RE = _END_THINK_CLOSE

    def family_name(self) -> str:
        return "gemma"


class MistralReasoningParser(ReasoningParser):
    """Mistral/Codestral: [THINK]...[/THINK] markers."""
    _OPEN_RE = _BRACKET_THINK_OPEN
    _CLOSE_RE = _BRACKET_THINK_CLOSE

    def family_name(self) -> str:
        return "mistral"


class PhiReasoningParser(ReasoningParser):
    """Phi-3/4: <think/>...</think/> tags."""
    _OPEN_RE = _THINK_OPEN
    _CLOSE_RE = _THINK_CLOSE

    def family_name(self) -> str:
        return "phi"


class CohereReasoningParser(ReasoningParser):
    """Cohere Command-R: <|START_THINKING|>...<|END_THINKING|> markers."""
    _OPEN_RE = _COHERE_OPEN
    _CLOSE_RE = _COHERE_CLOSE

    def family_name(self) -> str:
        return "cohere"


class LLamaReasoningParser(ReasoningParser):
    """LLaMA 3/4: <think/>...</think/> tags."""
    _OPEN_RE = _THINK_OPEN
    _CLOSE_RE = _THINK_CLOSE

    def family_name(self) -> str:
        return "llama"


class InternVLReasoningParser(ReasoningParser):
    """InternVL: <think/>...</think/> tags with image context markers."""
    _OPEN_RE = _THINK_OPEN
    _CLOSE_RE = _THINK_CLOSE

    def family_name(self) -> str:
        return "internvl"


class GenericReasoningParser(ReasoningParser):
    """Generic fallback: <think/>...</think/> tags."""
    _OPEN_RE = _THINK_OPEN
    _CLOSE_RE = _THINK_CLOSE

    def family_name(self) -> str:
        return "generic"


# ── Registry & Auto-Detection ────────────────────────────────────────────────

_REGISTRY: dict[str, type[ReasoningParser]] = {
    "qwen": QwenReasoningParser,
    "deepseek": DeepSeekReasoningParser,
    "glm": GLMReasoningParser,
    "harmony": HarmonyReasoningParser,
    "gemma": GemmaReasoningParser,
    "mistral": MistralReasoningParser,
    "phi": PhiReasoningParser,
    "cohere": CohereReasoningParser,
    "llama": LLamaReasoningParser,
    "internvl": InternVLReasoningParser,
    "generic": GenericReasoningParser,
}

_MODEL_FAMILY_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"qwen", re.IGNORECASE), "qwen"),
    (re.compile(r"deepseek", re.IGNORECASE), "deepseek"),
    (re.compile(r"glm", re.IGNORECASE), "glm"),
    (re.compile(r"gemma", re.IGNORECASE), "gemma"),
    (re.compile(r"harmony|gpt.?oss", re.IGNORECASE), "harmony"),
    (re.compile(r"mistral|codestral|mixtral|pixtral", re.IGNORECASE), "mistral"),
    (re.compile(r"phi[-_.]?[34]", re.IGNORECASE), "phi"),
    (re.compile(r"command[-_.]?r|cohere", re.IGNORECASE), "cohere"),
    (re.compile(r"llama", re.IGNORECASE), "llama"),
    (re.compile(r"intern[-_.]?vl", re.IGNORECASE), "internvl"),
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
