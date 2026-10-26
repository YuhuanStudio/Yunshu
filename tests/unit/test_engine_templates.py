"""Behaviour of engine_templates: message normalisation, double-BOS guard and chat-template fallbacks."""

from types import SimpleNamespace

import pytest

from yunshu_engine.engine_templates import EngineTemplatesMixin as T


def test_normalize_closes_dangling_think_before_tool_call():
    msgs = [
        {"role": "assistant", "content": "<think>plan <tool_call>x</tool_call>"},
        {"role": "assistant", "content": "<think>a</think>b <tool_call>x</tool_call>"},
        {"role": "user", "content": "<think><tool_call>"},
    ]
    out = T._normalize_messages_for_chat_template(msgs)
    assert out[0]["content"] == "<think>plan </think><tool_call>x</tool_call>"
    assert out[1]["content"] == msgs[1]["content"]  # already closed
    assert out[2] is msgs[2]  # non-assistant untouched
    assert (
        msgs[0]["content"] == "<think>plan <tool_call>x</tool_call>"
    )  # input not mutated


@pytest.mark.parametrize(
    "arguments,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ("", {}),
        (None, {}),
        ("not json", {"value": "not json"}),
        ("[1, 2]", {"value": [1, 2]}),
        ("null", {}),
        ({"already": "dict"}, {"already": "dict"}),
    ],
)
def test_normalize_tool_call_arguments_become_dicts(arguments, expected):
    call = {"function": {"name": "f", "arguments": arguments}}
    out = T._normalize_messages_for_chat_template(
        [{"role": "assistant", "content": "", "tool_calls": [call]}]
    )
    assert out[0]["tool_calls"][0]["function"]["arguments"] == expected
    assert out[0]["tool_calls"][0]["function"]["name"] == "f"


class _Tok:
    def __init__(self, bos=None, accept_kwarg=True):
        self.bos_token = bos
        self.accept_kwarg = accept_kwarg
        self.calls = []

    def encode(self, text, **kw):
        if kw and not self.accept_kwarg:
            raise TypeError("no kwarg")
        self.calls.append(kw)
        return [1, 2, 3]


def test_encode_prompt_skips_special_tokens_when_text_already_has_bos():
    tok = _Tok(bos="<s>")
    assert T._encode_prompt(tok, "<s>hello") == [1, 2, 3]
    assert tok.calls[-1] == {"add_special_tokens": False}
    T._encode_prompt(tok, "hello")
    assert tok.calls[-1] == {"add_special_tokens": True}


def test_encode_prompt_falls_back_when_kwarg_is_unsupported():
    tok = _Tok(bos="<s>", accept_kwarg=False)
    assert T._encode_prompt(tok, "hello") == [1, 2, 3]
    assert tok.calls == [{}]


class _TemplateTok:
    """Records apply_chat_template kwargs; rejects continue_final_message when asked."""

    def __init__(self, reject_continue=False):
        self.reject_continue = reject_continue
        self.calls = []

    def apply_chat_template(self, messages, **kw):
        self.calls.append((messages, kw))
        if self.reject_continue and kw.get("continue_final_message"):
            raise ValueError("no content to continue (continue_final_message)")
        return "|".join(f"{m['role']}:{m['content']}" for m in messages) + (
            "|GEN" if kw.get("add_generation_prompt") else ""
        )


def _engine(tok, thinking=None):
    return SimpleNamespace(
        _tokenizer=tok,
        enable_thinking=thinking,
        model_name="test/model",
        _normalize_messages_for_chat_template=T._normalize_messages_for_chat_template,
    )


def test_apply_chat_template_maps_roles_and_coerces_null_content():
    tok = _TemplateTok()
    out = T._apply_chat_template(
        _engine(tok),
        [
            {"role": "developer", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "function", "content": "r", "name": "f"},
        ],
    )
    sent, kw = tok.calls[0]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool"]
    assert sent[2]["content"] == "" and sent[2]["tool_calls"] == [{"id": "1"}]
    assert sent[3]["name"] == "f"
    assert kw["tokenize"] is False and kw["add_generation_prompt"] is True
    assert "None" not in out


def test_apply_chat_template_trailing_assistant_text_continues_the_turn():
    tok = _TemplateTok()
    T._apply_chat_template(
        _engine(tok),
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "The an"}],
    )
    kw = tok.calls[0][1]
    assert kw.get("continue_final_message") is True
    assert "add_generation_prompt" not in kw


def test_apply_chat_template_retries_without_continue_when_template_rejects_it():
    tok = _TemplateTok(reject_continue=True)
    out = T._apply_chat_template(
        _engine(tok),
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "x"}],
    )
    assert len(tok.calls) == 2
    retry = tok.calls[1][1]
    assert "continue_final_message" not in retry
    assert retry["add_generation_prompt"] is False
    assert out.startswith("user:q|assistant:x")


def test_apply_chat_template_thinking_flag_precedence():
    tok = _TemplateTok()
    msg = [{"role": "user", "content": "q"}]
    T._apply_chat_template(_engine(tok, thinking=True), msg)
    assert tok.calls[0][1]["enable_thinking"] is True
    T._apply_chat_template(_engine(tok, thinking=True), msg, False)
    assert tok.calls[1][1]["enable_thinking"] is False
    T._apply_chat_template(_engine(tok, thinking=None), msg)
    assert "enable_thinking" not in tok.calls[2][1]


def test_apply_chat_template_plaintext_fallback_without_tokenizer():
    out = T._apply_chat_template(
        _engine(None),
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": None}],
    )
    assert out == "User: hi\nAssistant: \nAssistant:"
