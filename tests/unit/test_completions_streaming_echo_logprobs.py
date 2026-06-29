"""(MED, stream↔non-stream parity): streaming echo+logprobs dropped ALL prompt-token
logprobs.

The non-streaming /v1/completions path prepends the prompt tokens' logprobs when echo=true
, but the streaming path emitted the echoed prompt as a bare text chunk with NO
logprobs and never requested prompt_logprobs — so an identical request returned prompt-token
logprobs non-streaming but not streaming. computes prompt_logprobs up front and emits
them in the echo chunk through the SAME shared helper (_format_prompt_logprob_entries) the
non-streaming path uses, so the prepend + BOS-collapse logic can't drift between
the two paths.
"""

from __future__ import annotations

import inspect

from yunshu_gateway.routers import (
    completions as C,  # noqa: N812  # intentional short module alias
)
from yunshu_gateway.routers.completions import _format_prompt_logprob_entries


class _Tok:
    _M = {1: "Hello", 2: " world", 3: "!"}

    def decode(self, ids):
        return "".join(self._M.get(i, "") for i in ids)


_PROMPT = "Hello world!"  # Hello(5) " world"(6) "!"(1)


def test_helper_no_bos_first_token_null():
    # no-BOS (Qwen): prompt_logprobs=[None, e2, e3]; position 0 IS the first real token.
    pl = [None, {"token_id": 2, "logprob": -0.8}, {"token_id": 3, "logprob": -0.2}]
    out = _format_prompt_logprob_entries(pl, _Tok(), 0, _PROMPT)
    assert [e["token"] for e in out] == ["Hello", " world", "!"]
    assert [e["logprob"] for e in out] == [None, -0.8, -0.2]


def test_helper_bos_collapses_phantom():
    # BOS (Llama/Gemma/Mistral): input_ids=[BOS,1,2,3] → prompt_logprobs has the BOS null slot
    # at index 0 and the first real token (id 1) carrying a logprob given BOS.
    pl = [
        None,
        {
            "token_id": 1,
            "logprob": -1.5,
        },  # first real token — must be nulled, no phantom
        {"token_id": 2, "logprob": -0.8},
        {"token_id": 3, "logprob": -0.2},
    ]
    out = _format_prompt_logprob_entries(pl, _Tok(), 0, _PROMPT)
    assert [e["token"] for e in out] == ["Hello", " world", "!"]
    assert [e["logprob"] for e in out] == [None, -0.8, -0.2]
    assert "" not in [e["token"] for e in out]


def test_helper_empty_on_unusable_input():
    assert _format_prompt_logprob_entries(None, _Tok(), 0, _PROMPT) == []
    assert _format_prompt_logprob_entries([], _Tok(), 0, _PROMPT) == []
    assert _format_prompt_logprob_entries("nope", _Tok(), 0, _PROMPT) == []
    # a single-token prompt ([None]) is valid: one entry covering the whole prompt, null logprob
    out = _format_prompt_logprob_entries([None], _Tok(), 0, _PROMPT)
    assert out == [{"token": _PROMPT, "logprob": None, "top_logprobs": {}}]


def test_streaming_echo_branch_emits_prompt_logprobs_via_shared_helper():
    src = inspect.getsource(C._stream_completion)
    echo = src.index("if req.echo:")
    helper = src.index("_format_prompt_logprob_entries", echo)
    # the echo chunk must compute prompt logprobs, reuse the shared helper, gate on batched,
    # and hand the result to the echo chunk via logprobs=
    assert "_compute_prompt_logprobs_for" in src[echo:helper], (
        "streaming echo must compute prompt logprobs"
    )
    assert "is_batched" in src[echo:helper], (
        "gated to the batched engine (where the forward lives)"
    )
    chunk = src.index("format_openai_completion_chunk", helper)
    assert "logprobs=_echo_lp" in src[chunk : chunk + 400], (
        "echo chunk must carry the logprobs"
    )


def test_nonstream_format_logprobs_uses_same_helper():
    # parity guard: the non-streaming path delegates to the same helper (no duplicated logic)
    src = inspect.getsource(C._format_logprobs)
    assert "_format_prompt_logprob_entries" in src
