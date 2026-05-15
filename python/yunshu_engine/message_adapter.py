from __future__ import annotations
"""Message format adapters for model-specific chat template input.

oMLX §13.2 pattern: Some model families require special message formatting
before applying the chat template. These adapters transform the standard
OpenAI-style message list into the format expected by each model family.

Adapters are auto-detected from model name and applied in _apply_chat_template().
"""

import re
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MessageAdapter(ABC):
    @abstractmethod
    def adapt(self, messages: list[dict]) -> list[dict]:
        """Transform messages for this model family."""
        ...

    @abstractmethod
    def family_name(self) -> str:
        ...


class HarmonyMessageAdapter(MessageAdapter):
    """Harmony/gpt_oss: Restructure messages with system-level directives.

    Harmony models expect:
    - System messages as top-level 'developer' role directives
    - Tool results wrapped in 'tool' role messages
    - Specific whitespace handling in multi-turn
    """

    def adapt(self, messages: list[dict]) -> list[dict]:
        adapted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                adapted.append({
                    "role": "developer",
                    "content": content,
                })
            elif role == "tool":
                adapted.append({
                    "role": "tool",
                    "content": content,
                    "tool_call_id": msg.get("tool_call_id", ""),
                })
            elif role == "assistant":
                new_msg = {"role": "assistant", "content": content}
                if msg.get("tool_calls"):
                    new_msg["tool_calls"] = msg["tool_calls"]
                adapted.append(new_msg)
            else:
                adapted.append(dict(msg))
        return adapted

    def family_name(self) -> str:
        return "harmony"


class Gemma4MessageAdapter(MessageAdapter):
    """Gemma4: Special system message handling + turn structure.

    Gemma4 models expect:
    - System message merged into first user turn as context
    - Strict alternation (no consecutive same-role messages)
    - Tool calls as function_call blocks in assistant messages
    """

    def adapt(self, messages: list[dict]) -> list[dict]:
        adapted = []
        system_prefix = ""
        prev_role = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_prefix = content
                continue

            if role == "user" and system_prefix and not adapted:
                content = f"{system_prefix}\n\n{content}" if content else system_prefix
                system_prefix = ""

            # Merge consecutive same-role messages
            if role == prev_role and adapted and role in ("user", "assistant"):
                last = adapted[-1]
                last["content"] = last.get("content", "") + "\n" + content
                continue

            new_msg = {"role": role, "content": content}
            if role == "assistant" and msg.get("tool_calls"):
                new_msg["tool_calls"] = msg["tool_calls"]
            if role == "tool":
                new_msg["tool_call_id"] = msg.get("tool_call_id", "")

            adapted.append(new_msg)
            prev_role = role

        # Ensure starts with user
        if adapted and adapted[0]["role"] != "user":
            adapted.insert(0, {"role": "user", "content": ""})

        return adapted

    def family_name(self) -> str:
        return "gemma4"


class DeepSeekMessageAdapter(MessageAdapter):
    """DeepSeek V4: Chat template patches for proper formatting.

    DeepSeek models benefit from:
    - Preserving tool call structure exactly as-is
    - Ensuring system message is first if present
    - Trimming trailing whitespace from content
    """

    def adapt(self, messages: list[dict]) -> list[dict]:
        adapted = []
        has_system = False

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Trim trailing whitespace from string content
            if isinstance(content, str):
                content = content.rstrip()

            new_msg = {"role": role, "content": content}

            if role == "system":
                has_system = True
            if role == "assistant" and msg.get("tool_calls"):
                new_msg["tool_calls"] = msg["tool_calls"]
            if role == "tool":
                new_msg["tool_call_id"] = msg.get("tool_call_id", "")

            adapted.append(new_msg)

        # System must be first
        if has_system and adapted and adapted[0]["role"] != "system":
            sys_msgs = [m for m in adapted if m["role"] == "system"]
            other = [m for m in adapted if m["role"] != "system"]
            adapted = sys_msgs + other

        return adapted

    def family_name(self) -> str:
        return "deepseek"


class QwenMessageAdapter(MessageAdapter):
    """Qwen 3.5: Attention patch compatibility formatting.

    Qwen 3.5 models benefit from:
    - Preserving thinking/reasoning in assistant turns
    - Proper tool call formatting
    """

    def adapt(self, messages: list[dict]) -> list[dict]:
        adapted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            new_msg = {"role": role, "content": content}

            if role == "assistant":
                if msg.get("tool_calls"):
                    new_msg["tool_calls"] = msg["tool_calls"]
                if msg.get("reasoning_content"):
                    new_msg["reasoning_content"] = msg["reasoning_content"]
            if role == "tool":
                new_msg["tool_call_id"] = msg.get("tool_call_id", "")

            adapted.append(new_msg)
        return adapted

    def family_name(self) -> str:
        return "qwen"


class GenericMessageAdapter(MessageAdapter):
    """Generic: pass-through with minimal cleanup."""

    def adapt(self, messages: list[dict]) -> list[dict]:
        return [dict(m) for m in messages]

    def family_name(self) -> str:
        return "generic"


# ── Registry & Auto-Detection ────────────────────────────────────────────────

_REGISTRY: dict[str, type[MessageAdapter]] = {
    "harmony": HarmonyMessageAdapter,
    "gemma4": Gemma4MessageAdapter,
    "deepseek": DeepSeekMessageAdapter,
    "qwen": QwenMessageAdapter,
    "generic": GenericMessageAdapter,
}

_MODEL_FAMILY_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"harmony|gpt.?oss", re.IGNORECASE), "harmony"),
    (re.compile(r"gemma.?[4-9]", re.IGNORECASE), "gemma4"),
    (re.compile(r"deepseek", re.IGNORECASE), "deepseek"),
    (re.compile(r"qwen", re.IGNORECASE), "qwen"),
]


def get_message_adapter(model_name: str | None = None) -> MessageAdapter:
    """Get the appropriate message adapter for a model.

    Auto-detects based on model name, falls back to generic pass-through.
    """
    if model_name:
        for pattern, family in _MODEL_FAMILY_HINTS:
            if pattern.search(model_name):
                adapter_cls = _REGISTRY.get(family, GenericMessageAdapter)
                return adapter_cls()
    return GenericMessageAdapter()


def adapt_messages(messages: list[dict], model_name: str | None = None) -> list[dict]:
    """Convenience: adapt messages for a model in one call."""
    adapter = get_message_adapter(model_name)
    return adapter.adapt(messages)


def register_message_adapter(family: str, adapter_cls: type[MessageAdapter]) -> None:
    _REGISTRY[family] = adapter_cls
