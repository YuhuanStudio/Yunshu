"""BatchedEngine handling of Gemma-4 channel reasoning + multi-eos.

Gemma-4 emits reasoning inside <|channel>thought…<channel|> blocks whose markers
are SINGLE special tokens that decode to '' under skip_special_tokens — so they
vanish before any <think>-based splitter and the chain-of-thought leaks into
content. It also declares eos_token_id [1, 106, 50] in its config while the
tokenizer exposes only 1, so without reading the config the turn-end token 106
never stops generation and the model rambles. These guard both fixes.
"""

import json

from yunshu_engine.batched_engine import (
    _read_config_eos_ids,
    _recover_channel_reasoning,
)

# token vocab: 100=<|channel> (open), 101=<channel|> (close); both '' under skip.
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
    100: "<|channel>",
    101: "<channel|>",
}


class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        skip = {100, 101} if skip_special_tokens else set()
        return "".join(_VOCAB[i] for i in ids if i not in skip)

    def encode(self, s, add_special_tokens=False):
        rev = {v: k for k, v in _VOCAB.items()}
        if s in rev:
            return [rev[s]]
        if s == "<think>":  # not a single token in gemma → forces channel fallback
            return [9, 9, 9]
        if s == "</think>":
            return [8, 8, 8]
        return [0]


def test_recover_channel_reasoning_reasoning_first():
    tok = _Tok()
    # <|channel> thought\n Let me think. <channel|> The sea is blue.
    tokens = [100, 50, 51, 52, 53, 54, 101, 1, 2, 3, 4]
    out, reason_ids = _recover_channel_reasoning(tokens, tok, "ignored")
    assert out == "<think>Let me think.</think>The sea is blue."
    assert reason_ids  # non-empty → counted as reasoning


def test_recover_channel_reasoning_content_first():
    tok = _Tok()
    # content BEFORE the channel (gemma often answers then thinks)
    tokens = [1, 2, 3, 4, 100, 50, 51, 52, 53, 54, 101]
    out, reason_ids = _recover_channel_reasoning(tokens, tok, "ignored")
    assert out == "<think>Let me think.</think>The sea is blue."
    assert reason_ids


def test_recover_no_channel_is_noop():
    tok = _Tok()
    tokens = [1, 2, 3, 4]
    out, reason_ids = _recover_channel_reasoning(tokens, tok, "The sea is blue.")
    assert out == "The sea is blue."
    assert reason_ids == []


def test_read_config_eos_ids_list(tmp_path):
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [1, 106, 50]})
    )
    assert _read_config_eos_ids(str(tmp_path)) == frozenset({1, 106, 50})


def test_read_config_eos_ids_single(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": 2}))
    assert _read_config_eos_ids(str(tmp_path)) == frozenset({2})


def test_read_config_eos_ids_missing_dir():
    assert _read_config_eos_ids("/nonexistent/path/xyz") == frozenset()
