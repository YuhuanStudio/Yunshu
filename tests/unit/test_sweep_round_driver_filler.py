"""The round-driver sweep's ``--context`` means prompt tokens, not words: a
32768 context once produced ~76K-token prompts (numbered notes are several
tokens each), turning the long-context parity check into hours of prefill."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))

from sweep_round_driver import filler_text  # noqa: E402


class CharTokenizer:
    """One token per character: every "note N." costs 7+ tokens."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_filler_is_exactly_the_requested_tokens():
    tok = CharTokenizer()
    for n in (1, 100, 4096, 32768):
        assert len(tok.encode(filler_text(tok, n))) == n
    assert filler_text(tok, 0) == ""
