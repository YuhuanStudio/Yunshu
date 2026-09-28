"""String-state constraint uses a cached vocab split and never admits raw control chars."""

from yunshu_engine.json_schema import JsonSchemaConstraint


class _Tok:
    def __init__(self, pieces):
        self._pieces = pieces
        self.get_vocab_calls = 0

    def get_vocab(self):
        self.get_vocab_calls += 1
        return {p: i for i, p in enumerate(self._pieces)}

    def decode(self, ids):
        return "".join(self._pieces[i] for i in ids)


PIECES = ["{", "}", '"', ":", ",", " ", "\n", "\t", "a", "b", "ab\tc", 'x"', "\r", "1", "city"]
SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}


def _texts(tok, ids):
    return {tok.decode([i]) for i in ids}


def test_string_state_excludes_control_chars_and_caches_vocab():
    tok = _Tok(PIECES)
    c = JsonSchemaConstraint(SCHEMA)
    c.advance('{"city": "a')
    first = _texts(tok, c.get_allowed_tokens(tok, []))
    calls = tok.get_vocab_calls
    again = _texts(tok, c.get_allowed_tokens(tok, []))
    assert first == again
    assert tok.get_vocab_calls == calls  # vocab split is cached per tokenizer
    assert {"a", "b"} <= first
    assert not any(ch in t for t in first for ch in "\t\r\n")


def test_structural_whitespace_is_spaces_and_newlines_only():
    tok = _Tok(PIECES)
    c = JsonSchemaConstraint(SCHEMA)
    c.advance("{")
    allowed = _texts(tok, c.get_allowed_tokens(tok, []))
    assert "\t" not in allowed and "\r" not in allowed
    assert " " in allowed and '"' in allowed
