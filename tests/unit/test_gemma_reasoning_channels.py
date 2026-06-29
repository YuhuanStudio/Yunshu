"""Gemma 4 reasoning-channel handling: the <|channel>thought…<channel|> special
tokens (which vanish under skip_special_tokens) are segmented at the TOKEN level
and normalized to standard <think>…</think> so the downstream reasoning split
works. Order-agnostic; no-op for models without the channel tokens."""

from yunshu_engine.vlm_engine import VLMEngine

_VOCAB = {
    1: "The",
    2: " sea",
    3: " is",
    4: " blue.",
    50: "thought",
    51: "\n",
    52: "Let",
    53: " me",
    54: " think.",
    100: "<|channel>",  # open (special — would be '' under skip in real tok)
    101: "<channel|>",  # close
}


class _Tok:
    def decode(self, ids, skip_special_tokens=True):
        # mimic skip_special_tokens stripping the channel markers to ''
        skip = {100, 101} if skip_special_tokens else set()
        return "".join(_VOCAB[i] for i in ids if i not in skip)


def _eng():
    eng = VLMEngine.__new__(VLMEngine)
    eng._tokenizer = _Tok()
    eng._reasoning_channel_ids = (100, 101)
    return eng


def test_reasoning_first_normalized_to_think():
    eng = _eng()
    toks = [100, 50, 51, 52, 53, 54, 101, 1, 2, 3, 4]
    assert eng._decode_with_reasoning_channels(toks) == (
        "<think>Let me think.</think>The sea is blue."
    )


def test_content_first_reordered_to_canonical():
    # order-agnostic: even if content precedes the channel, output is canonical
    eng = _eng()
    toks = [1, 2, 3, 4, 100, 50, 51, 52, 53, 54, 101]
    assert eng._decode_with_reasoning_channels(toks) == (
        "<think>Let me think.</think>The sea is blue."
    )


def test_no_channel_tokens_is_plain_decode():
    eng = _eng()
    assert eng._decode_with_reasoning_channels([1, 2, 3, 4]) == "The sea is blue."


def test_disabled_when_ids_unresolved():
    # No channel ids → plain decode (skip_special strips 100/101 like a real tok)
    eng = _eng()
    eng._reasoning_channel_ids = (None, None)
    assert eng._decode_with_reasoning_channels([1, 2, 100, 50, 101]) == "The seathought"


def test_empty_reasoning_returns_content_only():
    eng = _eng()
    # channel with only the "thought" label and nothing else → no <think> wrapper
    assert (
        eng._decode_with_reasoning_channels([100, 50, 101, 1, 2, 3, 4])
        == "The sea is blue."
    )
