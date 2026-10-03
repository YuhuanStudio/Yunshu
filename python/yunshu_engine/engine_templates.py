from __future__ import annotations

"""Engine templates extracted from batched_engine.

Patchable helpers resolve through the compatibility facade. Concrete self types
retain the shared BatchedEngine state; misc ignores allow that mixin self type.
"""

from collections.abc import AsyncIterator


class EngineTemplatesMixin:
    async def chat(  # type: ignore[misc]
        self: _engine.BatchedEngine,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        stop: list[str] | None = None,
        enable_thinking: bool | None = None,
        **kwargs,
    ) -> _engine.GenerationOutput:
        """Non-streaming chat completion (messages → template → generate)."""
        prompt = self._apply_chat_template(messages, enable_thinking)
        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            stop=stop,
            **kwargs,
        )

    async def stream_chat(  # type: ignore[misc]
        self: _engine.BatchedEngine,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        stop: list[str] | None = None,
        enable_thinking: bool | None = None,
        **kwargs,
    ) -> AsyncIterator[_engine.GenerationOutput]:
        """Streaming chat completion (messages → template → stream_generate)."""
        prompt = self._apply_chat_template(messages, enable_thinking)
        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            stop=stop,
            **kwargs,
        ):
            yield output

    @staticmethod
    def _normalize_messages_for_chat_template(messages: list[dict]) -> list[dict]:
        """Normalize messages before chat-template rendering.

        - Closes dangling <think> spans before raw <tool_call> XML in assistant
          content (Qwen 3.6 produces history where <think> is left open when a
          tool call follows — conditioning the next turn as still-reasoning).
        - Converts tool-call argument JSON strings to dicts for templates that
          iterate argument keys.
        """
        import json as _json

        normalized = []
        for m in messages:
            if m.get("role") != "assistant":
                normalized.append(m)
                continue
            m = dict(m)
            content = m.get("content")
            if (
                isinstance(content, str)
                and "<tool_call>" in content
                and "<think>" in content
            ):
                last_think = content.rfind("<think>")
                last_close = content.rfind("</think>")
                tool_pos = content.find("<tool_call>")
                if (
                    not (last_close >= last_think and last_close != -1)
                    and tool_pos > last_think
                ):
                    m["content"] = content[:tool_pos] + "</think>" + content[tool_pos:]
            tool_calls = m.get("tool_calls")
            if isinstance(tool_calls, list):
                patched = []
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        patched.append(tc)
                        continue
                    func = tc.get("function")
                    if isinstance(func, dict):
                        args = func.get("arguments")
                        if args is None:
                            args = "{}"
                        if isinstance(args, str):
                            try:
                                parsed = _json.loads(args.strip() or "{}")
                            except Exception:
                                parsed = {"value": args}
                            tc = dict(tc)
                            tc["function"] = dict(func)
                            tc["function"]["arguments"] = (
                                parsed
                                if isinstance(parsed, dict)
                                else {}
                                if parsed is None
                                else {"value": parsed}
                            )
                    patched.append(tc)
                m["tool_calls"] = patched
            normalized.append(m)
        return normalized

    @staticmethod
    def _encode_prompt(tokenizer, text: str) -> list[int]:
        """Encode a prompt string, avoiding a DOUBLE-BOS.

        When ``text`` came from apply_chat_template(tokenize=False) for a BOS-prepending
        model (Gemma/Llama/Mistral), it already starts with the literal bos_token; a plain
        encode() defaults to add_special_tokens=True and prepends BOS *again* → [BOS, BOS,
        …], which corrupts the first-token distribution. Qwen has no BOS so it was immune
        (and hid this). Mirrors mlx-lm's own generate.py guard. A raw /v1/completions
        string (no template, no leading BOS) still correctly gets its single BOS.
        """
        bos = getattr(tokenizer, "bos_token", None)
        # add_special unless the text already opens with a real bos_token string.
        add_special = not (isinstance(bos, str) and bos and text.startswith(bos))
        try:
            return tokenizer.encode(text, add_special_tokens=add_special)
        except TypeError:
            # Tokenizer.encode doesn't accept the kwarg — fall back (no double-BOS guard).
            return tokenizer.encode(text)

    def _apply_chat_template(  # type: ignore[misc]
        self: _engine.BatchedEngine,
        messages: list[dict],
        enable_thinking: bool | None = None,
        tools: list | None = None,
    ) -> str:
        """Apply chat template to convert messages to text.

        ``tools`` (OpenAI function schemas), when the model's chat template natively
        supports a ``tools`` variable, are rendered in the model's OWN tool format +
        special tokens (better adherence/parse rates than a generic injected system
        prompt). Callers pass tools ONLY when they've skipped the prompt-injection
        fallback (see _engine_supports_native_tools); otherwise None keeps the old path.
        """
        thinking = (
            enable_thinking if enable_thinking is not None else self.enable_thinking
        )
        tokenizer = self._tokenizer

        # normalize OpenAI's `developer` role → `system` and the
        # legacy `function` role → `tool` BEFORE the family adapter + template run. Request
        # validation accepts both, but no chat template knows them → apply_chat_template
        # raises "Unknown role" → the WHOLE prompt collapses to the structureless plaintext
        # fallback (special tokens + chat structure lost → degraded generation). `developer`
        # is OpenAI's current recommended replacement for `system`, so clients send it
        # routinely. Doing this pre-adapter lets gemma4/mistral merge it as a system msg.
        if any(m.get("role") in ("developer", "function") for m in messages):
            _remapped = []
            for _m in messages:
                _r = _m.get("role")
                if _r in ("developer", "function"):
                    _m = dict(_m)
                    _m["role"] = "system" if _r == "developer" else "tool"
                _remapped.append(_m)
            messages = _remapped

        # Apply model-specific message adapter
        try:
            from yunshu_engine.message_adapter import adapt_messages

            messages = adapt_messages(messages, self.model_name)
        except Exception:
            _engine.logger.debug("message adapter failed", exc_info=True)

        # Safety normalization: close dangling <think> before <tool_call>,
        # convert tool-call argument strings to dicts (vllm-mlx pattern)
        messages = self._normalize_messages_for_chat_template(messages)

        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            try:
                clean = []
                for m in messages:
                    # coerce content=None → "" BEFORE the template. m.get("content",
                    # "") returns the default only when the key is MISSING; an explicit
                    # content=None (the canonical OpenAI agent-loop assistant turn
                    # {"role":"assistant","content":null,"tool_calls":[...]}) passes None through.
                    # A Jinja `{{ content }}` then renders Python None as the literal text "None"
                    # (verified) — corrupting every GLM/Llama/Qwen tool-calling turn with history;
                    # a `{{ "x" + content }}` template instead raises TypeError → plaintext
                    # fallback. The Gemma adapter and VLMEngine._format_prompt already coerce
                    # None→"" via _extract_text; this is the un-swept BatchedEngine text sibling.
                    _content = m.get("content")
                    msg = {
                        "role": m.get("role", "user"),
                        "content": "" if _content is None else _content,
                    }
                    # Preserve tool-related fields for correct template rendering
                    if m.get("tool_calls"):
                        msg["tool_calls"] = m["tool_calls"]
                    if m.get("tool_call_id"):
                        msg["tool_call_id"] = m["tool_call_id"]
                    if m.get("name"):
                        msg["name"] = m["name"]
                    # preserve reasoning_content — the family
                    # adapters thread it through, but this clean step dropped it, so
                    # DeepSeek-v3.2's thinking template (which asserts
                    # `reasoning_content or tool_calls` for an assistant turn after the
                    # last user msg) raised AssertionError → whole prompt collapsed to the
                    # plaintext fallback.
                    if m.get("reasoning_content"):
                        msg["reasoning_content"] = m["reasoning_content"]
                    clean.append(msg)
                # ASSISTANT PREFILL. A trailing assistant message means
                # "continue THIS turn" (Anthropic prefill, also OpenAI's) — the model
                # must continue from the prefilled text and NOT have a fresh assistant
                # turn opened after it. add_generation_prompt=True closes the prefill and
                # opens an empty turn (the prefix is ignored, output restarts). When the
                # last message is assistant, use continue_final_message=True instead so
                # the template keeps the turn open. Gated on trailing-assistant only, so
                # the normal case (last msg user/tool) is byte-identical to before.
                # prefill ONLY when the trailing assistant has non-empty STRING
                # content to continue. The gate (any trailing assistant) also matched
                # the canonical OpenAI agent-loop shape {"role":"assistant","content":null,
                # "tool_calls":[...]}, where continue_final_message makes the Jinja template
                # raise ValueError ("no content to continue") → not caught (only TypeError
                # was) → the WHOLE prompt collapsed to the plaintext fallback. A trailing
                # assistant with null/empty content (e.g. a tool_calls-only turn) is treated
                # as a completed turn → normal add_generation_prompt.
                _last = clean[-1] if clean else None
                _is_prefill = (
                    _last is not None
                    and _last.get("role") == "assistant"
                    and isinstance(_last.get("content"), str)
                    and _last["content"] != ""
                )
                kwargs = {"tokenize": False}
                if _is_prefill:
                    kwargs["continue_final_message"] = True
                else:
                    kwargs["add_generation_prompt"] = True
                if thinking is not None:
                    kwargs["enable_thinking"] = thinking
                # Native tool rendering: explicit param wins, else the per-request
                # contextvar the router set (task-side). Only pass when the template
                # actually references a `tools` variable.
                _tools = tools if tools is not None else _engine._REQUEST_TOOLS.get()
                if _tools and _engine._template_supports_tools(tokenizer):
                    kwargs["tools"] = _tools
                try:
                    text = tokenizer.apply_chat_template(clean, **kwargs)
                except (TypeError, ValueError) as e:
                    _es = str(e)
                    if "tools" in kwargs and (
                        "tool" in _es.lower() or isinstance(e, TypeError)
                    ):
                        # Some templates declare a `tools` var but choke on the schema
                        # shape / lack the kwarg. Drop tools and retry — the request still
                        # generates (tool-calling just isn't natively templated this turn).
                        _engine.logger.warning(
                            f"Model {self.model_name} rejected native tools ({_es[:80]}); "
                            "retrying without"
                        )
                        kwargs.pop("tools", None)
                        text = tokenizer.apply_chat_template(clean, **kwargs)
                    elif "continue_final_message" in _es:
                        # TypeError = tokenizer too old for the kwarg; ValueError
                        # = template rejects continue_final_message (e.g. "no content to
                        # continue"). Either way, retry without it — don't open a NEW turn
                        # after the prefix, and never collapse to the plaintext fallback.
                        _engine.logger.warning(
                            f"Model {self.model_name} rejected continue_final_message ({_es[:80]}); retrying without"
                        )
                        kwargs.pop("continue_final_message", None)
                        kwargs["add_generation_prompt"] = False
                        try:
                            text = tokenizer.apply_chat_template(clean, **kwargs)
                        except (TypeError, ValueError) as e2:
                            if "enable_thinking" in str(e2):
                                kwargs.pop("enable_thinking", None)
                                text = tokenizer.apply_chat_template(clean, **kwargs)
                            else:
                                raise
                    elif "enable_thinking" in _es:
                        _engine.logger.warning(
                            f"Model {self.model_name} doesn't support enable_thinking, retrying without"
                        )
                        kwargs.pop("enable_thinking", None)
                        text = tokenizer.apply_chat_template(clean, **kwargs)
                    else:
                        raise
                if text:
                    return text
            except Exception:
                _engine.logger.debug(
                    "chat template failed, using fallback", exc_info=True
                )

        # Generic fallback
        parts = []
        for m in messages:
            # same None→"" coercion as the template clean step, so the plaintext
            # fallback doesn't print a literal "None" for a content=null assistant turn.
            _fc = m.get("content")
            parts.append(
                f"{m.get('role', 'user').capitalize()}: {'' if _fc is None else _fc}"
            )
        parts.append("Assistant:")
        return "\n".join(parts)


# Resolve the facade after definitions to also support direct module imports.
from . import batched_engine as _engine
