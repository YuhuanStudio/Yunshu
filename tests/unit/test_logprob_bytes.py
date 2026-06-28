"""the OpenAI logprobs `bytes` field must be a token's RAW UTF-8 bytes so a
client can reassemble a multi-byte char (CJK/emoji) that byte-level BPE split across
tokens. The old `decode([tid]).encode("utf-8")` lost the raw bytes — a lone byte-fragment
decodes to U+FFFD (�), so its bytes became the replacement char [239,191,189]."""
from __future__ import annotations

from yunshu_engine.text_utils import _GPT2_BYTE_DECODER, token_id_to_bytes


class _ByteLevelBackend:
    """Minimal stand-in for a fast tokenizer's backend whose decoder is ByteLevel —
    this is what real byte-level BPE tokenizers (Qwen/Llama-3/GPT) expose and what
    _is_byte_level_tokenizer (W1004) keys on to enable raw-byte recovery."""
    class _Decoder:
        def __repr__(self):
            return "ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)"
    decoder = _Decoder()
    pre_tokenizer = None


class _FakeGPT2Tokenizer:
    """Mimics a byte-level BPE tokenizer: convert_ids_to_tokens returns GPT-2 surface
    forms (raw bytes mapped through bytes_to_unicode); decode([fragment]) returns U+FFFD
    for an incomplete multi-byte sequence (like real HF tokenizers). It exposes a
    ByteLevel backend_tokenizer so _is_byte_level_tokenizer (W1004) recognises it as
    byte-level — exactly as a real Qwen/GPT fast tokenizer does."""
    backend_tokenizer = _ByteLevelBackend()

    def __init__(self, id_to_surface):
        self._m = id_to_surface

    def convert_ids_to_tokens(self, tid):
        return self._m.get(tid)

    def decode(self, ids):
        # naive: try to decode the concatenated raw bytes; a lone fragment → U+FFFD
        raw = bytearray()
        for tid in ids:
            for c in self._m.get(tid, ""):
                raw.append(_GPT2_BYTE_DECODER.get(c, 0))
        return bytes(raw).decode("utf-8", errors="replace")


def _surface(bs):
    # encode raw bytes into the GPT-2 surface string
    inv = {b: c for c, b in _GPT2_BYTE_DECODER.items()}
    return "".join(inv[b] for b in bs)


def test_split_cjk_char_bytes_reassemble():
    # 聾 (U+807E) = bytes E8 81 BE, split across two tokens [E8 81] and [BE].
    tok = _FakeGPT2Tokenizer({1: _surface([0xE8, 0x81]), 2: _surface([0xBE])})
    b1 = token_id_to_bytes(tok, 1)
    b2 = token_id_to_bytes(tok, 2)
    assert b1 == [0xE8, 0x81]
    assert b2 == [0xBE]
    assert bytes(b1 + b2).decode("utf-8") == "聾"  # client can reconstruct
    # the OLD broken approach would have given the replacement char bytes:
    assert b1 != [239, 191, 189]


def test_ascii_and_space_tokens():
    tok = _FakeGPT2Tokenizer({1: _surface(list(b" hello"))})
    assert bytes(token_id_to_bytes(tok, 1)).decode("utf-8") == " hello"


def test_sentencepiece_byte_token():
    class _SP:
        def convert_ids_to_tokens(self, tid):
            return "<0xE8>"
        def decode(self, ids):
            return "�"
    assert token_id_to_bytes(_SP(), 5) == [0xE8]


def test_lp_bytes_prefers_engine_provided():
    from yunshu_gateway.routers.chat import _lp_bytes
    # engine already put correct raw bytes on the entry → use them as-is.
    entry = {"token": "�", "token_id": 1, "bytes": [0xE8, 0x81]}
    assert _lp_bytes(entry, "�", None) == [0xE8, 0x81]
    # no engine bytes, no tokenizer → fall back to the decoded string's utf-8.
    assert _lp_bytes({"token": "hi"}, "hi", None) == [104, 105]
