from __future__ import annotations
"""Output Parser Factory — model-specific output extraction.

oMLX §13.2 pattern: Each model family has its own output format for
reasoning, tool calls, and structured content. This factory auto-detects
the model family and applies the correct parser to extract:
- Reasoning/thinking content
- Tool calls (already handled by tool_call_parser.py)
- Clean content (stripping model-specific markup)

Complements reasoning_parser.py (which handles thinking tags) by also
stripping model-specific prefixes, suffixes, and formatting artifacts.
"""

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParsedOutput:
    content: str
    reasoning: str | None = None
    tool_call_text: str | None = None
    finish_reason: str | None = None


class OutputParser(ABC):
    @abstractmethod
    def parse(self, text: str) -> ParsedOutput:
        ...

    @abstractmethod
    def family_name(self) -> str:
        ...


class DeepSeekOutputParser(OutputParser):
    """DeepSeek V3/R1: Strip ✿FUNCTION✿ markers and reasoning tags."""

    _THINK_RE = re.compile(r"<think\s*/?\s*>(.*?)</think\s*/?\s*>", re.DOTALL)
    _FUNC_RE = re.compile(r"✿FUNCTION✿.*?✿", re.DOTALL)

    def parse(self, text: str) -> ParsedOutput:
        reasoning = None
        think_match = self._THINK_RE.search(text)
        if think_match:
            reasoning = think_match.group(1).strip()
            text = self._THINK_RE.sub("", text).strip()

        tool_text = None
        func_matches = self._FUNC_RE.findall(text)
        if func_matches:
            tool_text = "\n".join(func_matches)
            text = self._FUNC_RE.sub("", text).strip()

        return ParsedOutput(
            content=text,
            reasoning=reasoning,
            tool_call_text=tool_text,
        )

    def family_name(self) -> str:
        return "deepseek"


class QwenOutputParser(OutputParser):
    """Qwen 3/3.5: Extract <think/> blocks, strip tool XML."""

    _THINK_RE = re.compile(r"<think\s*/?\s*>(.*?)</think\s*/?\s*>", re.DOTALL)
    _TOOL_RE = re.compile(r"<tool_call\s*/?\s*>.*?</tool_call\s*/?\s*>", re.DOTALL)
    _FUNC_RE = re.compile(r"<function\s*=\s*\w+>.*?</function>", re.DOTALL)

    def parse(self, text: str) -> ParsedOutput:
        reasoning = None
        think_match = self._THINK_RE.search(text)
        if think_match:
            reasoning = think_match.group(1).strip()
            text = self._THINK_RE.sub("", text).strip()

        tool_text = None
        tool_matches = self._TOOL_RE.findall(text) + self._FUNC_RE.findall(text)
        if tool_matches:
            tool_text = "\n".join(tool_matches)
            text = self._TOOL_RE.sub("", text)
            text = self._FUNC_RE.sub("", text).strip()

        return ParsedOutput(content=text, reasoning=reasoning, tool_call_text=tool_text)

    def family_name(self) -> str:
        return "qwen"


class GemmaOutputParser(OutputParser):
    """Gemma 4: Extract <start_think/>...</end_think/> blocks."""

    _THINK_RE = re.compile(r"<start_think\s*/?\s*>(.*?)</end_think\s*/?\s*>", re.DOTALL)

    def parse(self, text: str) -> ParsedOutput:
        reasoning = None
        think_match = self._THINK_RE.search(text)
        if think_match:
            reasoning = think_match.group(1).strip()
            text = self._THINK_RE.sub("", text).strip()

        return ParsedOutput(content=text, reasoning=reasoning)

    def family_name(self) -> str:
        return "gemma"


class HarmonyOutputParser(OutputParser):
    """Harmony/gpt_oss: Extract [REASONING]...[/REASONING] markers."""

    _REASON_RE = re.compile(r"\[REASONING\](.*?)\[/REASONING\]", re.DOTALL | re.IGNORECASE)

    def parse(self, text: str) -> ParsedOutput:
        reasoning = None
        reason_match = self._REASON_RE.search(text)
        if reason_match:
            reasoning = reason_match.group(1).strip()
            text = self._REASON_RE.sub("", text).strip()

        return ParsedOutput(content=text, reasoning=reasoning)

    def family_name(self) -> str:
        return "harmony"


class GLMOutputParser(OutputParser):
    """GLM-4/5: Extract <think/> blocks, strip observation tags."""

    _THINK_RE = re.compile(r"<think\s*/?\s*>(.*?)</think\s*/?\s*>", re.DOTALL)
    _OBS_RE = re.compile(r"<observation\s*/?\s*>.*?</observation\s*/?\s*>", re.DOTALL)

    def parse(self, text: str) -> ParsedOutput:
        reasoning = None
        think_match = self._THINK_RE.search(text)
        if think_match:
            reasoning = think_match.group(1).strip()
            text = self._THINK_RE.sub("", text).strip()

        text = self._OBS_RE.sub("", text).strip()
        return ParsedOutput(content=text, reasoning=reasoning)

    def family_name(self) -> str:
        return "glm"


class MistralOutputParser(OutputParser):
    """Mistral/Codestral: Extract [THINK]...[/THINK] blocks."""

    _THINK_RE = re.compile(r"\[THINK\](.*?)\[/THINK\]", re.DOTALL)
    _TOOL_RE = re.compile(r"\[TOOL_CALLS\](.*?)$", re.DOTALL)

    def parse(self, text: str) -> ParsedOutput:
        reasoning = None
        think_match = self._THINK_RE.search(text)
        if think_match:
            reasoning = think_match.group(1).strip()
            text = self._THINK_RE.sub("", text).strip()

        tool_text = None
        tool_match = self._TOOL_RE.search(text)
        if tool_match:
            tool_text = tool_match.group(1).strip()
            text = self._TOOL_RE.sub("", text).strip()

        return ParsedOutput(content=text, reasoning=reasoning, tool_call_text=tool_text)

    def family_name(self) -> str:
        return "mistral"


class PhiOutputParser(OutputParser):
    """Phi-3/4: Extract <think/> blocks, strip tool markers."""

    _THINK_RE = re.compile(r"<think\s*/?\s*>(.*?)</think\s*/?\s*>", re.DOTALL)
    _TOOL_RE = re.compile(r"<\|tool_calls\|>(.*?)<\|/tool_calls\|>", re.DOTALL)

    def parse(self, text: str) -> ParsedOutput:
        reasoning = None
        think_match = self._THINK_RE.search(text)
        if think_match:
            reasoning = think_match.group(1).strip()
            text = self._THINK_RE.sub("", text).strip()

        tool_text = None
        tool_match = self._TOOL_RE.search(text)
        if tool_match:
            tool_text = tool_match.group(1).strip()
            text = self._TOOL_RE.sub("", text).strip()

        return ParsedOutput(content=text, reasoning=reasoning, tool_call_text=tool_text)

    def family_name(self) -> str:
        return "phi"


class CohereOutputParser(OutputParser):
    """Cohere Command-R: Extract <|START_THINKING|>...<|END_THINKING|> blocks."""

    _THINK_RE = re.compile(r"<\|START_THINKING\|>(.*?)<\|END_THINKING\|>", re.DOTALL)
    _ACTION_RE = re.compile(r"<\|START_ACTION\|>(.*?)<\|END_ACTION\|>", re.DOTALL)

    def parse(self, text: str) -> ParsedOutput:
        reasoning = None
        think_match = self._THINK_RE.search(text)
        if think_match:
            reasoning = think_match.group(1).strip()
            text = self._THINK_RE.sub("", text).strip()

        tool_text = None
        action_match = self._ACTION_RE.search(text)
        if action_match:
            tool_text = action_match.group(1).strip()
            text = self._ACTION_RE.sub("", text).strip()

        return ParsedOutput(content=text, reasoning=reasoning, tool_call_text=tool_text)

    def family_name(self) -> str:
        return "cohere"


class LLamaOutputParser(OutputParser):
    """LLaMA 3/4: Extract <think/> blocks, strip tool call markers."""

    _THINK_RE = re.compile(r"<think\s*/?\s*>(.*?)</think\s*/?\s*>", re.DOTALL)
    _TOOL_RE = re.compile(r"<\|python_tag\|>(.*?)$", re.DOTALL)

    def parse(self, text: str) -> ParsedOutput:
        reasoning = None
        think_match = self._THINK_RE.search(text)
        if think_match:
            reasoning = think_match.group(1).strip()
            text = self._THINK_RE.sub("", text).strip()

        tool_text = None
        tool_match = self._TOOL_RE.search(text)
        if tool_match:
            tool_text = tool_match.group(1).strip()
            text = self._TOOL_RE.sub("", text).strip()

        return ParsedOutput(content=text, reasoning=reasoning, tool_call_text=tool_text)

    def family_name(self) -> str:
        return "llama"


class InternVLOutputParser(OutputParser):
    """InternVL: Extract <think/> blocks, strip image artifacts."""

    _THINK_RE = re.compile(r"<think\s*/?\s*>(.*?)</think\s*/?\s*>", re.DOTALL)
    _IMG_RE = re.compile(r"<IMG_CONTEXT>.*?</IMG_CONTEXT>", re.DOTALL)

    def parse(self, text: str) -> ParsedOutput:
        reasoning = None
        think_match = self._THINK_RE.search(text)
        if think_match:
            reasoning = think_match.group(1).strip()
            text = self._THINK_RE.sub("", text).strip()

        text = self._IMG_RE.sub("", text).strip()

        return ParsedOutput(content=text, reasoning=reasoning)

    def family_name(self) -> str:
        return "internvl"


class GenericOutputParser(OutputParser):
    """Generic: try <think/> extraction, pass through otherwise."""

    _THINK_RE = re.compile(r"<think\s*/?\s*>(.*?)</think\s*/?\s*>", re.DOTALL)

    def parse(self, text: str) -> ParsedOutput:
        reasoning = None
        think_match = self._THINK_RE.search(text)
        if think_match:
            reasoning = think_match.group(1).strip()
            text = self._THINK_RE.sub("", text).strip()

        return ParsedOutput(content=text, reasoning=reasoning)

    def family_name(self) -> str:
        return "generic"


# ── Registry & Auto-Detection ────────────────────────────────────────────────

_REGISTRY: dict[str, type[OutputParser]] = {
    "deepseek": DeepSeekOutputParser,
    "qwen": QwenOutputParser,
    "gemma": GemmaOutputParser,
    "harmony": HarmonyOutputParser,
    "glm": GLMOutputParser,
    "mistral": MistralOutputParser,
    "phi": PhiOutputParser,
    "cohere": CohereOutputParser,
    "llama": LLamaOutputParser,
    "internvl": InternVLOutputParser,
    "generic": GenericOutputParser,
}

_MODEL_FAMILY_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"deepseek", re.IGNORECASE), "deepseek"),
    (re.compile(r"qwen", re.IGNORECASE), "qwen"),
    (re.compile(r"gemma", re.IGNORECASE), "gemma"),
    (re.compile(r"harmony|gpt.?oss", re.IGNORECASE), "harmony"),
    (re.compile(r"glm", re.IGNORECASE), "glm"),
    (re.compile(r"mistral|codestral|mixtral|pixtral", re.IGNORECASE), "mistral"),
    (re.compile(r"phi[-_.]?[34]", re.IGNORECASE), "phi"),
    (re.compile(r"command[-_.]?r|cohere", re.IGNORECASE), "cohere"),
    (re.compile(r"llama", re.IGNORECASE), "llama"),
    (re.compile(r"intern[-_.]?vl", re.IGNORECASE), "internvl"),
]


def get_output_parser(model_name: str | None = None) -> OutputParser:
    """Get the appropriate output parser for a model."""
    if model_name:
        for pattern, family in _MODEL_FAMILY_HINTS:
            if pattern.search(model_name):
                return _REGISTRY.get(family, GenericOutputParser)()
    return GenericOutputParser()


def parse_output(text: str, model_name: str | None = None) -> ParsedOutput:
    """Convenience: parse model output in one call."""
    parser = get_output_parser(model_name)
    return parser.parse(text)


def register_output_parser(family: str, parser_cls: type[OutputParser]) -> None:
    _REGISTRY[family] = parser_cls
