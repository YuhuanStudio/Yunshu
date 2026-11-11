"""/v1/completions string prompts are raw text on VLM engines too.

VLMEngine.generate wrapped every str prompt as a user turn, so a completions string got
the chat template (28 extra tokens on Qwen2.5). The text engines already tokenize it as
raw; the raw flag keeps VLM in line. Chat and Messages prompts keep the template.
"""

from __future__ import annotations

import pytest

from yunshu_engine.batched_engine import BatchedEngine
from yunshu_engine.vlm_engine import VLMEngine
from yunshu_gateway.routers.completions import _raw_prompt_kwargs


class _FakeTok:
    """Byte-level fake: encode is one id per character; the chat template wraps the text."""

    bos_token = None

    def __init__(self):
        self.template_calls = 0

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]

    def apply_chat_template(self, clean, **kwargs):
        self.template_calls += 1
        body = "".join(m["content"] for m in clean)
        return "<|im_start|>user\n" + body + "<|im_end|>\n<|im_start|>assistant\n"


def _engine(tok):
    eng = VLMEngine.__new__(VLMEngine)
    eng._tokenizer = tok
    eng._model_name = "test-model"
    from yunshu_engine.vlm_engine import _VLMTextPromptCache

    eng._text_prompt_cache = _VLMTextPromptCache(max_entries=8)
    return eng


def test_raw_string_prompt_tokenizes_exactly_as_encode():
    tok = _FakeTok()
    text = "The capital of France is"
    msgs = [{"role": "user", "content": text}]
    ids, pkw, salt = _engine(tok)._runner_input(msgs, [], [], None, None, raw_text=text)
    assert ids == tok.encode(text)  # completions token count == raw tokenization
    assert pkw is None and salt is None
    assert tok.template_calls == 0  # no chat template on the completions path


def test_chat_prompt_keeps_the_template():
    tok = _FakeTok()
    msgs = [{"role": "user", "content": "hi"}]
    ids, _, _ = _engine(tok)._runner_input(msgs, [], [], None, None)
    assert tok.template_calls == 1
    assert len(ids) > len(tok.encode("hi"))  # templated: wrapper tokens present


def test_raw_prompt_rejects_multi_turn_messages():
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    with pytest.raises(ValueError):
        VLMEngine._raw_prompt_text(msgs, {"raw_prompt": True})


def test_raw_prompt_text_is_none_without_flag():
    msgs = [{"role": "user", "content": "hi"}]
    assert VLMEngine._raw_prompt_text(msgs, {}) is None
    assert VLMEngine._raw_prompt_text(msgs, {"raw_prompt": True}) == "hi"


def test_router_sends_raw_flag_only_to_templating_engines():
    vlm = VLMEngine.__new__(VLMEngine)
    text = BatchedEngine.__new__(BatchedEngine)
    assert _raw_prompt_kwargs(vlm) == {"raw_prompt": True}
    assert _raw_prompt_kwargs(text) == {}
