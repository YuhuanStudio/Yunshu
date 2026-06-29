"""(HIGH): VLMEngine._format_prompt (the text-only chat path served by a VLM
model) was a stale clone missing three BatchedEngine chat-template fixes:

1. assistant-prefill: it hardcoded add_generation_prompt=True, so a trailing
   assistant message (Anthropic/OpenAI prefill) had its turn CLOSED and the model
   restarted the answer instead of continuing the prefilled text.
2. developer/function role normalization: a `developer` (OpenAI's current system
   alias) or `function` role reached the template as an unknown role → apply_chat_template
   raised → collapsed to the lossy plaintext fallback (chat structure + special tokens lost).
3. family adapter: no adapt_messages, so a mid-conversation system message raised
   ("System message must be at the beginning") → same lossy fallback.

Mirrors BatchedEngine._apply_chat_template.
"""

from __future__ import annotations

from yunshu_engine.vlm_engine import VLMEngine


class _FakeTok:
    def __init__(self):
        self.last_clean = None
        self.last_kwargs = None

    def apply_chat_template(self, clean, **kwargs):
        self.last_clean = clean
        self.last_kwargs = kwargs
        return "RENDERED"


def _engine(tok):
    eng = VLMEngine.__new__(VLMEngine)
    eng._tokenizer = tok
    eng._model_name = "test-model"
    return eng


def test_developer_role_remapped_to_system():
    tok = _FakeTok()
    out = _engine(tok)._format_prompt(
        [
            {"role": "developer", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert out == "RENDERED"
    roles = [m["role"] for m in tok.last_clean]
    assert "developer" not in roles
    assert "system" in roles  # developer → system


def test_assistant_prefill_uses_continue_final_message():
    tok = _FakeTok()
    _engine(tok)._format_prompt(
        [
            {"role": "user", "content": "Write a poem"},
            {"role": "assistant", "content": "Roses are red,"},
        ]
    )
    assert tok.last_kwargs.get("continue_final_message") is True
    assert "add_generation_prompt" not in tok.last_kwargs  # turn kept open


def test_normal_trailing_user_uses_add_generation_prompt():
    tok = _FakeTok()
    _engine(tok)._format_prompt([{"role": "user", "content": "hi"}])
    assert tok.last_kwargs.get("add_generation_prompt") is True
    assert "continue_final_message" not in tok.last_kwargs


def test_tool_calls_only_assistant_is_not_a_prefill():
    # trailing assistant with empty string content (tool_calls-only turn, the canonical
    # agent-loop shape) is a COMPLETED turn → normal add_generation_prompt.
    tok = _FakeTok()
    _engine(tok)._format_prompt(
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
        ]
    )
    assert tok.last_kwargs.get("add_generation_prompt") is True
    assert "continue_final_message" not in tok.last_kwargs


def test_continue_final_message_rejection_retries_without_it():
    # If the template raises ValueError on continue_final_message, retry without it
    # rather than collapsing to the plaintext fallback.
    class _RejectTok:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, clean, **kwargs):
            self.calls.append(dict(kwargs))
            if kwargs.get("continue_final_message"):
                raise ValueError("continue_final_message: no content to continue")
            return "RENDERED2"

    tok = _RejectTok()
    out = _engine(tok)._format_prompt(
        [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "partial"},
        ]
    )
    assert out == "RENDERED2"  # did NOT fall to plaintext fallback
    assert len(tok.calls) == 2  # tried prefill, then retried
    assert tok.calls[1].get("add_generation_prompt") is False
