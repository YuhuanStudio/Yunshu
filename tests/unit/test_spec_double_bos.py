"""the speculative / MTP / n-gram-spec prompt builders bypassed the fast path's
double-BOS guard.

They did `apply_chat_template(prompt, tokenize=False)` (string already opens with the
literal bos_token for Gemma/Llama-3/Mistral) then `tokenizer.encode(text)` with the default
add_special_tokens=True → BOS prepended a SECOND time → [BOS, BOS, …], corrupting the
first-token distribution. The fix already applied to _generate_gemma4_assistant_spec —
route messages-format prompts through `_apply_chat_template` + `_encode_prompt` (which
strips a duplicate leading BOS) — must be propagated to every sibling.
"""
from __future__ import annotations

import inspect

import pytest

from yunshu_engine.batched_engine import BatchedEngine

_SPEC_METHODS = [
    "_generate_ngram_spec",
    "_generate_mtp",
    "_generate_speculative",
    "_stream_generate_mtp",
    "_stream_generate_ngram_spec",
    "_stream_generate_speculative",
]


@pytest.mark.parametrize("method_name", _SPEC_METHODS)
def test_spec_path_routes_messages_through_bos_guarded_encode(method_name):
    method = getattr(BatchedEngine, method_name)
    src = inspect.getsource(method)
    # the messages-format branch must encode via _encode_prompt (the double-BOS guard),
    # never raw tokenizer.encode right after a tokenize=False apply_chat_template
    assert "_apply_chat_template(prompt" in src, f"{method_name} not routed via _apply_chat_template"
    assert "_encode_prompt(" in src, f"{method_name} not routed via _encode_prompt"
    # the raw tokenize=False template call (which produces the BOS-bearing string) is gone
    assert "apply_chat_template(prompt, **tpl_kwargs)" not in src, (
        f"{method_name} still uses the raw double-BOS apply_chat_template+encode path"
    )
