"""legacy /v1/completions echo=true + logprobs=N omitted the PROMPT tokens'
logprobs — only the completion's logprobs were returned, with text_offset shifted past a
prompt whose tokens were absent. OpenAI's contract is that echo+logprobs returns logprobs
for the prompt tokens too (the first prompt token's token_logprobs[0]=null, no preceding
context) — perplexity/eval clients rely on exactly this.

Fix: when echo+logprobs (single prompt), force the prompt_logprobs forward (the machinery
already exists) and _format_logprobs PREPENDS the prompt tokens, reconstructing each
token's string from its own token_id (no re-tokenization → robust to a leading BOS), with
the completion tokens continuing from the end of the echoed prompt.
"""
from __future__ import annotations

from types import SimpleNamespace

from yunshu_gateway.routers.completions import _format_logprobs


class _Tok:
    _M = {1: "b", 2: "c", 10: "X", 11: "Y"}

    def decode(self, ids):
        return "".join(self._M.get(i, "?") for i in ids)


def test_echo_prepends_prompt_token_logprobs():
    state = SimpleNamespace(
        # completion logprobs (one generated token "X")
        logprobs=[{"token_id": 10, "logprob": -0.3, "top_logprobs": []}],
        # vLLM-shaped prompt logprobs: [None, tok1, tok2] for prompt "abc"
        prompt_logprobs=[None,
                         {"token_id": 1, "logprob": -0.5},
                         {"token_id": 2, "logprob": -1.0}],
    )
    out = _format_logprobs(state, _Tok(), top_logprobs=0, echo=True, prompt="abc")
    # prompt tokens prepended, then the completion token
    assert out["tokens"] == ["a", "b", "c", "X"]
    # first prompt token has a NULL logprob (no preceding context)
    assert out["token_logprobs"][0] is None
    assert out["token_logprobs"][1:] == [-0.5, -1.0, -0.3]
    # offsets: prompt at 0,1,2 then completion at len("abc")=3
    assert out["text_offset"] == [0, 1, 2, 3]


def test_no_echo_unchanged_completion_only():
    state = SimpleNamespace(
        logprobs=[{"token_id": 10, "logprob": -0.3, "top_logprobs": []}],
        prompt_logprobs=[None, {"token_id": 1, "logprob": -0.5}],
    )
    out = _format_logprobs(state, _Tok(), top_logprobs=0, echo=False, prompt="abc")
    # echo off → prompt tokens NOT prepended; just the completion token at offset 0
    assert out["tokens"] == ["X"]
    assert out["token_logprobs"] == [-0.3]
    assert out["text_offset"] == [0]


def test_echo_without_prompt_logprobs_falls_back_to_offset_shift():
    # if the engine didn't compute prompt_logprobs, echo still shifts completion offset
    # by len(prompt) (legacy behavior preserved, no crash)
    state = SimpleNamespace(logprobs=[{"token_id": 10, "logprob": -0.3, "top_logprobs": []}])
    out = _format_logprobs(state, _Tok(), top_logprobs=0, echo=True, prompt="abc")
    assert out["tokens"] == ["X"]
    assert out["text_offset"] == [3]  # shifted past the (absent-token) prompt
