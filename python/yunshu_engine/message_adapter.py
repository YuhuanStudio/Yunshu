from __future__ import annotations

"""Message format adapters for model-specific chat template input.

Some model families require special message formatting
before applying the chat template. These adapters transform the standard
OpenAI-style message list into the format expected by each model family.

Adapters are auto-detected from model name and applied in _apply_chat_template().
"""

import logging
import re
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
                tool_msg = {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": msg.get("tool_call_id", ""),
                }
                if msg.get("name"):
                    tool_msg["name"] = msg["name"]
                adapted.append(tool_msg)
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

    @staticmethod
    def _extract_text(content) -> str:
        """Extract plain text from content, handling list-format multi-part."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            return "\n".join(parts)
        return str(content)

    def adapt(self, messages: list[dict]) -> list[dict]:
        adapted: list[dict] = []
        system_prefix = ""
        prev_role = None

        for msg in messages:
            role = msg.get("role", "user")
            content = self._extract_text(msg.get("content", ""))

            if role == "system":
                # ACCUMULATE multiple system messages — the old
                # `system_prefix = content` overwrote, so with ≥2 system msgs (e.g. a
                # cached_content context doc prepended ahead of the request's own system
                # message) all but the LAST were silently dropped. Gemma-4 is the flagship
                # model + the cached_content→Gemma path is exactly explicit_cache's purpose.
                system_prefix = f"{system_prefix}\n\n{content}" if system_prefix else content
                # A system message is a TURN BOUNDARY — reset prev_role so a
                # following user/assistant isn't merged into the one BEFORE the system msg.
                # The old code left prev_role unchanged, so [user, system, user] collapsed
                # the two user turns into one merged turn (silent structure loss on the
                # flagship Gemma-4 path; mid-conversation system messages are common).
                prev_role = "system"
                continue

            if role == "user" and system_prefix and not adapted:
                content = f"{system_prefix}\n\n{content}" if content else system_prefix
                system_prefix = ""

            # Merge consecutive same-role messages (but only plain text;
            # if the current or previous message carries structured fields
            # like tool_calls or reasoning_content, keep them separate).
            if (
                role == prev_role
                and adapted
                and role in ("user", "assistant")
                and not msg.get("tool_calls")
                and not msg.get("reasoning_content")
                and not adapted[-1].get("tool_calls")
                and not adapted[-1].get("reasoning_content")
            ):
                last = adapted[-1]
                last["content"] = last.get("content", "") + "\n" + content
                continue

            new_msg = {"role": role, "content": content}
            if role == "assistant":
                if msg.get("tool_calls"):
                    new_msg["tool_calls"] = msg["tool_calls"]
                if msg.get("reasoning_content"):
                    new_msg["reasoning_content"] = msg["reasoning_content"]
            if role == "tool":
                new_msg["tool_call_id"] = msg.get("tool_call_id", "")
                if msg.get("name"):
                    new_msg["name"] = msg["name"]

            adapted.append(new_msg)
            prev_role = role

        # Ensure starts with user
        if adapted and adapted[0]["role"] != "user":
            adapted.insert(0, {"role": "user", "content": ""})

        # If system_prefix was never consumed (no user messages), surface it.
        # With turns present, prepend to the first; with NO turns at all (a
        # system-only "begin in this persona" request), INJECT a user turn carrying it —
        # the old `system_prefix and adapted` guard dropped the orphan system_prefix when
        # adapted was [], so adapt() returned [], the Gemma template raised on the empty
        # message list → plaintext fallback → the system prompt was 100% lost.
        if system_prefix:
            if adapted:
                adapted[0]["content"] = (
                    f"{system_prefix}\n\n{adapted[0]['content']}"
                    if adapted[0]["content"]
                    else system_prefix
                )
            else:
                adapted.append({"role": "user", "content": system_prefix})

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
                if msg.get("name"):
                    new_msg["name"] = msg["name"]

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
                if msg.get("name"):
                    new_msg["name"] = msg["name"]

            adapted.append(new_msg)

        # Qwen's chat template raises TemplateError "System message must be at
        # the beginning" when a system message appears mid-conversation. Hoist system
        # messages to the front (mirrors the Llama/GLM/DeepSeek/Phi adapters, which already
        # do this — a mid-system Qwen request otherwise raised → caught at
        # _apply_chat_template → collapsed to the plaintext fallback).
        if (adapted and adapted[0]["role"] != "system"
                and any(m["role"] == "system" for m in adapted)):
            sys_msgs = [m for m in adapted if m["role"] == "system"]
            other = [m for m in adapted if m["role"] != "system"]
            adapted = sys_msgs + other
        return adapted

    def family_name(self) -> str:
        return "qwen"


class MistralMessageAdapter(MessageAdapter):
    """Mistral/Codestral: Strict role alternation, no system role.

    Mistral models expect:
    - No system role (merge into first user turn)
    - Strict user/assistant alternation (insert empty turns if needed)
    - Tool calls as function_call blocks
    """

    def adapt(self, messages: list[dict]) -> list[dict]:
        adapted = []
        system_prefix = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_prefix = (system_prefix + "\n\n" + content).strip() if system_prefix else content
                continue

            new_msg = {"role": role, "content": content}

            if role == "assistant":
                if msg.get("tool_calls"):
                    new_msg["tool_calls"] = msg["tool_calls"]
                if msg.get("reasoning_content"):
                    new_msg["reasoning_content"] = msg["reasoning_content"]
            if role == "tool":
                new_msg["tool_call_id"] = msg.get("tool_call_id", "")
                if msg.get("name"):
                    new_msg["name"] = msg["name"]

            adapted.append(new_msg)

        # Merge system prefix into first user message. A system-only request (no
        # user/assistant turns) must still surface the system prompt — inject a user turn
        # carrying it, else adapted stays [] → Mistral template raise → plaintext fallback →
        # system prompt lost (same drop-on-empty as the Gemma4 sibling).
        if system_prefix:
            if adapted:
                first = adapted[0]
                if first["role"] == "user":
                    first["content"] = f"{system_prefix}\n\n{first['content']}" if first["content"] else system_prefix
                else:
                    adapted.insert(0, {"role": "user", "content": system_prefix})
            else:
                adapted.append({"role": "user", "content": system_prefix})

        # Ensure strict alternation starting with user
        if adapted and adapted[0]["role"] != "user":
            adapted.insert(0, {"role": "user", "content": ""})

        alternated = []
        for msg in adapted:
            role = msg["role"]
            # Tool messages are passthrough — they don't participate in alternation
            if role == "tool":
                alternated.append(msg)
                continue
            if alternated and alternated[-1]["role"] == role and role in ("user", "assistant"):
                # Insert empty opposite turn
                opposite = "assistant" if role == "user" else "user"
                alternated.append({"role": opposite, "content": ""})
            alternated.append(msg)

        return alternated

    def family_name(self) -> str:
        return "mistral"


class PhiMessageAdapter(MessageAdapter):
    """Phi-3/4: System-first with tool call and reasoning support.

    Phi models expect:
    - System message must come first if present
    - Tool results must include tool_call_id
    - Reasoning content preserved in assistant turns
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
                if msg.get("name"):
                    new_msg["name"] = msg["name"]

            adapted.append(new_msg)

        # System must be first if present
        sys_msgs = [m for m in adapted if m["role"] == "system"]
        other = [m for m in adapted if m["role"] != "system"]
        return sys_msgs + other

    def family_name(self) -> str:
        return "phi"


class CohereMessageAdapter(MessageAdapter):
    """Cohere Command-R: Tool call formatting with reasoning support.

    Command-R models expect:
    - Tool calls wrapped in specific format
    - System messages preserved as-is (Cohere supports system role)
    - Reasoning content in assistant turns
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
                if msg.get("name"):
                    new_msg["name"] = msg["name"]

            adapted.append(new_msg)

        return adapted

    def family_name(self) -> str:
        return "cohere"


class LLamaMessageAdapter(MessageAdapter):
    """LLaMA 3/4: System message handling with tool call support.

    LLaMA models expect:
    - System messages supported (native in LLaMA 3+)
    - Tool calls with function format
    - BOS/EOS handled by tokenizer
    """

    def adapt(self, messages: list[dict]) -> list[dict]:
        adapted = []
        has_system = False

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            new_msg = {"role": role, "content": content}

            if role == "system":
                has_system = True

            if role == "assistant":
                if msg.get("tool_calls"):
                    new_msg["tool_calls"] = msg["tool_calls"]
                if msg.get("reasoning_content"):
                    new_msg["reasoning_content"] = msg["reasoning_content"]
            if role == "tool":
                new_msg["tool_call_id"] = msg.get("tool_call_id", "")
                if msg.get("name"):
                    new_msg["name"] = msg["name"]

            adapted.append(new_msg)

        # System must be first
        if has_system and adapted and adapted[0]["role"] != "system":
            sys_msgs = [m for m in adapted if m["role"] == "system"]
            other = [m for m in adapted if m["role"] != "system"]
            adapted = sys_msgs + other

        return adapted

    def family_name(self) -> str:
        return "llama"


class InternVLMessageAdapter(MessageAdapter):
    """InternVL: Vision-language message formatting.

    InternVL models expect:
    - Image tokens as <img> placeholders in content
    - System messages supported
    - Multi-part content with image_url type
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
                if msg.get("name"):
                    new_msg["name"] = msg["name"]

            adapted.append(new_msg)

        return adapted

    def family_name(self) -> str:
        return "internvl"


class GLMMessageAdapter(MessageAdapter):
    """GLM-4/5: System message handling with tool call support.

    GLM models expect:
    - System messages supported
    - Tool calls with function format
    - Observation tags for tool results
    """

    def adapt(self, messages: list[dict]) -> list[dict]:
        adapted = []
        has_system = False

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            new_msg = {"role": role, "content": content}

            if role == "system":
                has_system = True

            if role == "assistant":
                if msg.get("tool_calls"):
                    new_msg["tool_calls"] = msg["tool_calls"]
                if msg.get("reasoning_content"):
                    new_msg["reasoning_content"] = msg["reasoning_content"]
            if role == "tool":
                new_msg["tool_call_id"] = msg.get("tool_call_id", "")
                if msg.get("name"):
                    new_msg["name"] = msg["name"]

            adapted.append(new_msg)

        # System must be first
        if has_system and adapted and adapted[0]["role"] != "system":
            sys_msgs = [m for m in adapted if m["role"] == "system"]
            other = [m for m in adapted if m["role"] != "system"]
            adapted = sys_msgs + other

        return adapted

    def family_name(self) -> str:
        return "glm"


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
    "mistral": MistralMessageAdapter,
    "phi": PhiMessageAdapter,
    "cohere": CohereMessageAdapter,
    "llama": LLamaMessageAdapter,
    "internvl": InternVLMessageAdapter,
    "glm": GLMMessageAdapter,
    "generic": GenericMessageAdapter,
}

_MODEL_FAMILY_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"harmony|gpt.?oss", re.IGNORECASE), "harmony"),
    (re.compile(r"gemma.?[4-9]", re.IGNORECASE), "gemma4"),
    (re.compile(r"deepseek", re.IGNORECASE), "deepseek"),
    (re.compile(r"qwen", re.IGNORECASE), "qwen"),
    (re.compile(r"mistral|codestral|mixtral|pixtral", re.IGNORECASE), "mistral"),
    (re.compile(r"phi[-_.]?[34]", re.IGNORECASE), "phi"),
    (re.compile(r"command[-_.]?r|cohere", re.IGNORECASE), "cohere"),
    (re.compile(r"llama", re.IGNORECASE), "llama"),
    (re.compile(r"intern[-_.]?vl", re.IGNORECASE), "internvl"),
    (re.compile(r"glm", re.IGNORECASE), "glm"),
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
