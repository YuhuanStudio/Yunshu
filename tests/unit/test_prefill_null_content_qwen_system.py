"""two chat-template-boundary bugs (both collapsed the DEFAULT path to the
structureless plaintext fallback via a swallowed template raise).

HIGH (regression): the prefill gate matched ANY trailing assistant message,
including the canonical OpenAI agent-loop shape {"role":"assistant","content":null,
"tool_calls":[...]}. continue_final_message then made the Jinja template raise ValueError
("no content to continue") — caught only TypeError → fell through to the plaintext
fallback. Now prefill requires non-empty STRING content; the ValueError retry path is
guarded too.

MEDIUM: QwenMessageAdapter didn't hoist a mid-conversation system message (Qwen's template
raises "System message must be at the beginning") — an un-propagated sibling gap (Llama/
GLM/DeepSeek/Phi adapters all hoist). Now hoisted."""
from __future__ import annotations

from yunshu_engine.batched_engine import BatchedEngine
from yunshu_engine.message_adapter import QwenMessageAdapter


class _RecordingTokenizer:
    bos_token = None

    def __init__(self):
        self.last_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.last_kwargs = kwargs
        return "<rendered>"


def _engine(tok):
    e = BatchedEngine.__new__(BatchedEngine)
    e._tokenizer = tok
    e.model_name = "test-model"
    e.enable_thinking = None
    return e


def test_trailing_assistant_null_content_is_not_prefill():
    tok = _RecordingTokenizer()
    eng = _engine(tok)
    # canonical agent-loop continuation: content=None + tool_calls
    msgs = [{"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "get_weather", "arguments": "{}"}}]}]
    eng._apply_chat_template(msgs)
    assert "continue_final_message" not in tok.last_kwargs
    assert tok.last_kwargs.get("add_generation_prompt") is True


def test_trailing_assistant_empty_string_is_not_prefill():
    tok = _RecordingTokenizer()
    eng = _engine(tok)
    eng._apply_chat_template([{"role": "user", "content": "x"},
                              {"role": "assistant", "content": ""}])
    assert "continue_final_message" not in tok.last_kwargs


def test_trailing_assistant_real_text_is_prefill():
    tok = _RecordingTokenizer()
    eng = _engine(tok)
    eng._apply_chat_template([{"role": "user", "content": "2+2?"},
                              {"role": "assistant", "content": "The answer is"}])
    assert tok.last_kwargs.get("continue_final_message") is True
    assert "add_generation_prompt" not in tok.last_kwargs


def test_value_error_on_continue_final_message_degrades_not_collapses():
    # a tokenizer that REJECTS continue_final_message with ValueError must retry with
    # add_generation_prompt=False (structured), NOT fall through to the plaintext fallback.
    class _RejectTok:
        bos_token = None
        def __init__(self):
            self.calls = []
        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(dict(kwargs))
            if kwargs.get("continue_final_message"):
                raise ValueError("continue_final_message is set but the final message has no content to continue!")
            return "<structured>"
    tok = _RejectTok()
    eng = _engine(tok)
    out = eng._apply_chat_template([{"role": "user", "content": "x"},
                                    {"role": "assistant", "content": "pre"}])
    assert out == "<structured>"  # structured render, not the "Role: ...\nAssistant:" fallback
    assert tok.calls[-1].get("add_generation_prompt") is False


def test_qwen_adapter_hoists_mid_system():
    out = QwenMessageAdapter().adapt([
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "2+2"},
    ])
    assert [m["role"] for m in out] == ["system", "user", "user"]


def test_qwen_adapter_leaves_leading_system():
    out = QwenMessageAdapter().adapt([
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ])
    assert [m["role"] for m in out] == ["system", "user"]
