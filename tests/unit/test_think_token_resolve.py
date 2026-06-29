"""streaming reasoning separation was DISABLED for the canonical thinking models.

The think-token resolution encoded "<think" / "</think" WITHOUT the closing '>'. For
Qwen3/Qwen3.5/DeepSeek-R1 the bare form tokenizes to TWO tokens (e.g. Qwen3.5 `</think` →
[510, 26003]) while the model emits the SINGLE special token `</think>` (with bracket, e.g.
248069). So the `len == 1` guard failed → think_end_token=None → the streaming reasoning
state machine never engaged → the ENTIRE chain-of-thought (plus literal markup) leaked into
delta.content with reasoning_tokens=0, defeating the streaming fixes on the
DEFAULT path. _resolve_think_token_ids encodes the BRACKETED form with
add_special_tokens=False, and the fix is swept to all 8 think-token call sites.
"""

from __future__ import annotations

import inspect

from yunshu_engine import batched_engine
from yunshu_engine.batched_engine import _resolve_think_token_ids


class _FakeTok:
    """Mimics a tokenizer where the bracketed marker is a single special token but the
    unbracketed prefix is multiple tokens (the Qwen3/DeepSeek-R1 shape)."""

    bos_token_id = 1

    _MAP = {
        "<think>": [248068],
        "</think>": [248069],
        "<think": [13314, 741],
        "</think": [510, 26003],
    }

    def encode(self, s, add_special_tokens=True):
        ids = self._MAP.get(s, [9, 9, 9])
        # simulate a BOS-prepending tokenizer when special tokens are on
        if add_special_tokens:
            return [self.bos_token_id, *ids]
        return ids


def test_resolver_uses_bracketed_form_single_token():
    s, e = _resolve_think_token_ids(_FakeTok())
    assert (s, e) == (248068, 248069)


def test_resolver_strips_bos_on_fallback():
    # a tokenizer whose encode() rejects the kwarg → fallback path must strip the BOS
    class _NoKwargTok:
        bos_token_id = 1

        def encode(self, s):
            return {"<think>": [1, 248068], "</think>": [1, 248069]}.get(s, [1, 9, 9])

    s, e = _resolve_think_token_ids(_NoKwargTok())
    assert (s, e) == (248068, 248069)


def test_resolver_returns_none_for_multitoken_marker():
    class _MultiTok:
        def encode(self, s, add_special_tokens=True):
            return [1, 2, 3]  # never a single token

    assert _resolve_think_token_ids(_MultiTok()) == (None, None)


def test_all_think_sites_use_the_helper():
    src = inspect.getsource(batched_engine)
    # every old bare-encode site is gone; the helper is used throughout
    assert 'encode("</think")' not in src
    assert 'encode("<think")' not in src
    assert src.count("_resolve_think_token_ids(") >= 8
