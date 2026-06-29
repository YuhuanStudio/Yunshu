"""(MED): echo+logprobs on BOS-prepending models (Llama/Gemma/Mistral) emitted a
PHANTOM empty token and mis-assigned the null logprob.

For a raw /v1/completions string, _encode_prompt uses add_special_tokens=True, so a BOS model
encodes the prompt as input_ids=[BOS, t1, t2, ...]. _compute_prompt_logprobs_sync then returns
a vLLM-shaped list aligned to input_ids: [None(BOS slot), entry(t1|BOS), entry(t2), ...] —
position 0 is the BOS (null, no context) and the first REAL token t1 already carries a logprob
(given BOS). The echo-prepend mapped position 0 to a "" head token, so it emitted
tokens=["", "Hello", ...] with token_logprobs=[null, <real>, ...]: a phantom empty token
carrying the null, and the first real token exposing a logprob OpenAI assigns null.

The default Qwen2.5 model has no BOS, so the test never saw this. collapses the BOS
slot when _head=="" (tail already covers the whole prompt → position 0 has no visible text),
listing only real tokens and nulling the first real token's logprob — matching OpenAI and the
no-BOS path exactly.
"""

from __future__ import annotations

from yunshu_gateway.routers.completions import _format_logprobs


class _Tok:
    """Decodes token ids to fixed strings (no BOS text — BOS id is never decoded into output)."""

    _M = {1: "Hello", 2: " world", 3: "!", 9: " Hi"}

    def decode(self, ids):
        return "".join(self._M.get(i, "") for i in ids)


class _State:
    def __init__(self, prompt_logprobs, logprobs):
        self.prompt_logprobs = prompt_logprobs
        self.logprobs = logprobs


_PROMPT = "Hello world!"  # 12 chars; tokens Hello(5) " world"(6) "!"(1)
_COMPLETION = [{"token_id": 9, "logprob": -0.1}]  # one completion token " Hi"

_EXPECT_TOKENS = ["Hello", " world", "!", " Hi"]
_EXPECT_LOGPROBS = [None, -0.8, -0.2, -0.1]
_EXPECT_OFFSETS = [0, 5, 11, 12]


def test_bos_model_collapses_phantom_and_nulls_first_real_token():
    # BOS model: input_ids=[BOS,1,2,3] → prompt_logprobs has the BOS null slot at index 0
    # and the first real token (id 1) carrying a real logprob given BOS.
    pl = [
        None,
        {
            "token_id": 1,
            "logprob": -1.5,
        },  # first REAL token, logprob given BOS — must be nulled
        {"token_id": 2, "logprob": -0.8},
        {"token_id": 3, "logprob": -0.2},
    ]
    out = _format_logprobs(
        _State(pl, _COMPLETION), _Tok(), top_logprobs=0, echo=True, prompt=_PROMPT
    )
    assert out["tokens"] == _EXPECT_TOKENS, out["tokens"]
    assert out["token_logprobs"] == _EXPECT_LOGPROBS, out["token_logprobs"]
    assert out["text_offset"] == _EXPECT_OFFSETS, out["text_offset"]
    assert "" not in out["tokens"], "no phantom empty BOS token"


def test_no_bos_model_unchanged_parity():
    # No-BOS model (Qwen, the default): input_ids=[1,2,3], prompt_logprobs=[None, e2, e3].
    # Position 0 (None) IS the first real token. Output must match the BOS case byte-for-byte.
    pl = [
        None,
        {"token_id": 2, "logprob": -0.8},
        {"token_id": 3, "logprob": -0.2},
    ]
    out = _format_logprobs(
        _State(pl, _COMPLETION), _Tok(), top_logprobs=0, echo=True, prompt=_PROMPT
    )
    assert out["tokens"] == _EXPECT_TOKENS
    assert out["token_logprobs"] == _EXPECT_LOGPROBS
    assert out["text_offset"] == _EXPECT_OFFSETS


def test_bos_collapse_suppresses_first_token_top_logprobs():
    # The nulled first real token must also carry an EMPTY top_logprobs (OpenAI: the first
    # prompt token has no preceding context, so no alternatives).
    pl = [
        None,
        {
            "token_id": 1,
            "logprob": -1.5,
            "top_logprobs": [
                {"token_id": 1, "logprob": -1.5},
                {"token_id": 2, "logprob": -2.0},
            ],
        },
        {
            "token_id": 2,
            "logprob": -0.8,
            "top_logprobs": [{"token_id": 2, "logprob": -0.8}],
        },
        {"token_id": 3, "logprob": -0.2},
    ]
    out = _format_logprobs(
        _State(pl, _COMPLETION), _Tok(), top_logprobs=2, echo=True, prompt=_PROMPT
    )
    # first real token "Hello": null logprob + empty alternatives
    assert out["tokens"][0] == "Hello"
    assert out["token_logprobs"][0] is None
    assert out["top_logprobs"][0] == {}
    # the second token DOES carry its alternatives
    assert out["top_logprobs"][1] == {" world": -0.8}
