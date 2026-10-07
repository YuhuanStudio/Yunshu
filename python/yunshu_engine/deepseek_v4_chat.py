"""Use the pinned official encoder when a V4 checkpoint omits its template."""

from __future__ import annotations

import copy
import json

from ._vendor.deepseek_v4 import encoding as ref

START = "<｜DSML｜tool_calls>"
END = "</｜DSML｜tool_calls>"
# Metadata for Yunshu's template/format detection; rendering uses the callable.
TEMPLATE_MARKERS = "{{ tools }} {{ reasoning_effort }} " + START + END


def render(
    messages,
    *,
    enable_thinking=True,
    add_generation_prompt=False,
    continue_final_message=False,
    tools=None,
    reasoning_effort=None,
    **kwargs,
):
    if add_generation_prompt and continue_final_message:
        raise ValueError("Conflicting generation/continuation controls")
    prepared = copy.deepcopy(messages)
    for message in prepared:
        for call in message.get("tool_calls") or []:
            args = call["function"].get("arguments", {})
            if isinstance(args, dict):
                call["function"]["arguments"] = json.dumps(args, ensure_ascii=False)
    if tools:
        if not prepared or prepared[0].get("role") != "system":
            prepared.insert(0, dict(role="system", content=""))
        prepared[0]["tools"] = copy.deepcopy(tools)
    # The official source only accepts low/high/max, whereas clients also use
    # the OpenAI vocabulary. Thinking itself is controlled separately.
    effort = {
        "none": "low",
        "off": "low",
        "minimal": "low",
        "medium": "high",
        "xhigh": "max",
    }.get(reasoning_effort, reasoning_effort)
    if effort not in (None, "low", "high", "max"):
        effort = "low"
    mode = "thinking" if enable_thinking else "chat"
    text = ref.encode_messages(prepared, thinking_mode=mode, reasoning_effort=effort)
    primer = ref.ASSISTANT_SP_TOKEN + (
        ref.thinking_start_token if enable_thinking else ref.thinking_end_token
    )
    if continue_final_message:
        return text.removesuffix(ref.eos_token)
    if add_generation_prompt:
        return text if text.endswith(primer) else text + primer
    return text.removesuffix(primer)


def install(tokenizer, processor=None):
    """Install only on a template-less V4 wrapper, keeping published templates."""
    if getattr(tokenizer, "chat_template", None) or getattr(
        tokenizer, "_chat_template", None
    ):
        return False
    tokenizer._chat_template = render
    tokenizer._thinking_kwarg = "enable_thinking"
    tokenizer.has_chat_template = True
    tokenizer.chat_template = TEMPLATE_MARKERS
    if processor is not None:
        # load_processor adds this VLM contract to its HF tokenizer. Keep it
        # when sharing our renderer wrapper with BatchGenerator.
        criteria = getattr(
            getattr(processor, "tokenizer", None), "stopping_criteria", None
        )
        if criteria is not None:
            tokenizer.stopping_criteria = criteria
        processor.tokenizer = tokenizer
    return True


def parse_calls(body: str) -> list[dict[str, str]]:
    # The reference parser starts after the outer marker's name, before its
    # closing bracket. ToolFormat supplies only the block's inner text.
    text = ">\n" + body.strip("\n") + "\n" + END
    _, _, calls = ref.parse_tool_calls(0, text)
    return calls
