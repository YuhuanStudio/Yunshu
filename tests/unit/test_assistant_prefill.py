"""assistant prefill was broken. A trailing assistant message means
"continue THIS turn" (Anthropic prefill, also OpenAI's) — the model must continue the
prefilled text. _apply_chat_template unconditionally passed add_generation_prompt=True,
which CLOSES the prefilled assistant turn and opens an empty new one → the prefix is
ignored and generation restarts from scratch. Now a trailing-assistant message routes to
continue_final_message=True; the normal case is byte-identical to before."""

from __future__ import annotations

from yunshu_engine.batched_engine import BatchedEngine


class _RecordingTokenizer:
    """Records the kwargs apply_chat_template was called with, and emulates a real
    template: continue_final_message keeps the assistant turn open (no end token)."""

    def __init__(self):
        self.last_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.last_kwargs = kwargs
        out = []
        for m in messages:
            out.append(f"<|{m['role']}|>{m.get('content', '')}")
        if kwargs.get("add_generation_prompt"):
            out.append("<|assistant|>")  # fresh empty turn opened
        elif kwargs.get("continue_final_message"):
            pass  # turn stays open — last assistant content is the live continuation point
        return "".join(out)


def _engine(tok):
    eng = BatchedEngine.__new__(BatchedEngine)
    eng._tokenizer = tok
    eng.model_name = "test-model"
    eng.enable_thinking = None
    return eng


def test_trailing_assistant_uses_continue_final_message():
    tok = _RecordingTokenizer()
    eng = _engine(tok)
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is"},
    ]
    out = eng._apply_chat_template(msgs)
    assert tok.last_kwargs.get("continue_final_message") is True
    assert "add_generation_prompt" not in tok.last_kwargs
    # prefilled text stays at the end (no fresh empty turn appended)
    assert out.endswith("The answer is")
    assert "<|assistant|><|assistant|>" not in out  # no doubled turn


def test_normal_request_unchanged():
    tok = _RecordingTokenizer()
    eng = _engine(tok)
    msgs = [{"role": "user", "content": "hi"}]
    eng._apply_chat_template(msgs)
    assert tok.last_kwargs.get("add_generation_prompt") is True
    assert "continue_final_message" not in tok.last_kwargs


def test_trailing_tool_message_is_not_prefill():
    tok = _RecordingTokenizer()
    eng = _engine(tok)
    msgs = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "1", "content": "42"},
    ]
    eng._apply_chat_template(msgs)
    # last message is a tool result → normal generation prompt, NOT prefill
    assert tok.last_kwargs.get("add_generation_prompt") is True
    assert "continue_final_message" not in tok.last_kwargs


def test_old_tokenizer_falls_back_gracefully():
    class _OldTok:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(dict(kwargs))
            if kwargs.get("continue_final_message"):
                raise TypeError(
                    "apply_chat_template() got an unexpected keyword argument 'continue_final_message'"
                )
            return "".join(f"<|{m['role']}|>{m.get('content', '')}" for m in messages)

    tok = _OldTok()
    eng = _engine(tok)
    msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "pre"}]
    out = eng._apply_chat_template(msgs)
    # retried without continue_final_message, with add_generation_prompt=False
    assert tok.calls[-1].get("add_generation_prompt") is False
    assert out  # produced structured text, not the plaintext fallback
    assert "Assistant:" not in out
