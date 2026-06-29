"""(HIGH): token_id_to_bytes corrupted logprobs.bytes for accented/Latin-1
output on sentencepiece tokenizers (Llama-2, Mistral, Gemma).

The GPT-2 reverse-byte-map branch fired whenever every surface char was in the map.
A sentencepiece surface 'é' (the literal char U+00E9) is all-in-the-map, so it
returned [233] (the codepoint) instead of the real UTF-8 [195,169]. Byte-level BPE
(Qwen) and sentencepiece produce the IDENTICAL surface 'é' but mean different bytes,
so the branch must be gated on an affirmative byte-level detection (a ByteLevel
component in the fast tokenizer). This test uses fakes that model both families.
"""

from __future__ import annotations

from yunshu_engine.text_utils import _is_byte_level_tokenizer, token_id_to_bytes


class _FakeByteLevelTokenizer:
    """Models Qwen-style byte-level BPE: surface chars are GPT-2 byte proxies."""

    class _Backend:
        class _decoder:
            @staticmethod
            def __repr__():
                return "ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)"

        decoder = _decoder()
        pre_tokenizer = None

    backend_tokenizer = _Backend()

    def __init__(self, surface):
        self._surface = surface

    def convert_ids_to_tokens(self, tid):
        return self._surface

    def decode(self, ids):
        return "?"


class _FakeSentencePieceTokenizer:
    """Models Llama-2/Mistral/Gemma: Metaspace decoder, NEVER ByteLevel."""

    class _Backend:
        class _decoder:
            @staticmethod
            def __repr__():
                return "Metaspace(replacement='▁', add_prefix_space=True)"

        decoder = _decoder()
        pre_tokenizer = None

    backend_tokenizer = _Backend()

    def __init__(self, surface, decoded):
        self._surface = surface
        self._decoded = decoded

    def convert_ids_to_tokens(self, tid):
        return self._surface

    def decode(self, ids):
        return self._decoded


def test_detector_distinguishes_families():
    assert _is_byte_level_tokenizer(_FakeByteLevelTokenizer("Ã")) is True
    assert _is_byte_level_tokenizer(_FakeSentencePieceTokenizer("é", "é")) is False


def test_bytelevel_fragment_recovers_raw_byte():
    # Qwen split-CJK lead fragment: surface 'é' = raw byte 233 (0xE9), a multi-byte fragment.
    tok = _FakeByteLevelTokenizer("é")
    assert token_id_to_bytes(tok, 0) == [233]
    # full-char surrogate surface 'Ã©' → [195,169]
    tok2 = _FakeByteLevelTokenizer("Ã©")
    assert token_id_to_bytes(tok2, 0) == [195, 169]


def test_sentencepiece_accented_uses_real_utf8_not_codepoint():
    # The bug: surface 'é' (the literal char) was returned as [233]; correct is UTF-8 [195,169].
    tok = _FakeSentencePieceTokenizer("é", "é")
    assert token_id_to_bytes(tok, 0) == [195, 169]
    # multi-char accented word
    tok2 = _FakeSentencePieceTokenizer("rés", "rés")
    assert token_id_to_bytes(tok2, 0) == list("rés".encode())


def test_sentencepiece_byte_fallback_token_still_first():
    # '<0xE8>' must still be handled before any family branch.
    tok = _FakeSentencePieceTokenizer("<0xE8>", "?")
    assert token_id_to_bytes(tok, 0) == [0xE8]


def test_detector_memoized():
    tok = _FakeByteLevelTokenizer("Ã")
    _is_byte_level_tokenizer(tok)
    assert tok._yunshu_byte_level is True
