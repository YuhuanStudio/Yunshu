"""(HIGH): content=None rendered the literal string "None" into the prompt.

_apply_chat_template's clean step did `m.get("content", "")`, which returns the default ONLY
when the key is missing — an explicit content=None (the canonical OpenAI agent-loop assistant
turn {"role":"assistant","content":null,"tool_calls":[...]}) passed None straight through. A
Jinja `{{ content }}` renders Python None as the literal text "None" (verified), corrupting
every GLM/Llama/Qwen tool-calling turn with prior history; a `{{ "x"+content }}` template
instead raises TypeError → plaintext fallback. The Gemma adapter and VLMEngine._format_prompt
already coerce None→"" (_extract_text); this was the un-swept BatchedEngine text sibling.
"""

from __future__ import annotations

from jinja2 import Environment

from yunshu_engine.batched_engine import BatchedEngine


class _JinjaTok:
    """Faithfully renders each message's content via `{{ content }}` — exactly the construct
    that turns Python None into the literal 'None' in real GLM/Llama/Qwen templates."""

    bos_token = None

    def apply_chat_template(self, messages, **kwargs):
        env = Environment()
        t = env.from_string(
            "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
        )
        return t.render(messages=messages)


class _RaiseTok:
    bos_token = None

    def apply_chat_template(self, messages, **kwargs):
        raise RuntimeError("template boom → plaintext fallback")


def _engine(tok):
    e = BatchedEngine.__new__(BatchedEngine)
    e._tokenizer = tok
    e.model_name = "test-model"
    e.enable_thinking = None
    return e


_AGENT_LOOP = [
    {"role": "user", "content": "weather?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
]


def test_null_content_not_rendered_as_None_literal_via_template():
    out = _engine(_JinjaTok())._apply_chat_template(_AGENT_LOOP)
    assert "None" not in out, (
        out
    )  # the assistant turn must render EMPTY content, not "None"
    assert "<|assistant|>\n" in out  # role marker present, content blank


def test_null_content_not_rendered_as_None_literal_via_fallback():
    out = _engine(_RaiseTok())._apply_chat_template(_AGENT_LOOP)
    assert "None" not in out, out
    assert (
        "Assistant: " in out
    )  # plaintext fallback, blank content (not "Assistant: None")


def test_string_content_still_rendered():
    # regression: a normal string content must pass through unchanged.
    out = _engine(_JinjaTok())._apply_chat_template(
        [{"role": "user", "content": "hello world"}]
    )
    assert "hello world" in out


def test_missing_content_key_still_empty():
    # a message with NO content key (not None) must also render blank, as before.
    out = _engine(_JinjaTok())._apply_chat_template([{"role": "user"}])
    assert "None" not in out and "<|user|>" in out
