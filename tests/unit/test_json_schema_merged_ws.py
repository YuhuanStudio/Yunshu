"""Constrained JSON allows BPE tokens that merge whitespace with content, and keeps
structural whitespace to one space / a newline plus indentation."""

from yunshu_engine.json_schema import JsonSchemaConstraint


class _Tok:
    def __init__(self, pieces):
        self._pieces = pieces

    def get_vocab(self):
        return {p: i for i, p in enumerate(self._pieces)}

    def decode(self, ids):
        return "".join(self._pieces[i] for i in ids)


PIECES = [
    '{"',
    "city",
    '":',
    ' "',
    '"',
    "Paris",
    '"}',
    "{",
    "}",
    ":",
    " ",
    "  ",
    " \n",
    "\n",
    "\n  ",
    " The",
    ' "Paris',
    " x",
    "Lyon",
]
SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


def _allowed(tok, c):
    return {tok.decode([i]) for i in c.get_allowed_tokens(tok, [])}


def test_merged_space_quote_token_allowed_after_colon():
    tok = _Tok(PIECES)
    c = JsonSchemaConstraint(SCHEMA)
    for t in ('{"', "city", '":'):
        c.advance(t)
    allowed = _allowed(tok, c)
    assert ' "' in allowed  # merged whitespace + quote
    assert ' "Paris' in allowed  # merged whitespace + quote + content
    assert " The" not in allowed and " x" not in allowed  # prose stays out
    assert "  " not in allowed  # two spaces is not a compact run


def test_whitespace_runs_are_compact():
    tok = _Tok(PIECES)
    c = JsonSchemaConstraint(SCHEMA)
    c.advance("{")
    allowed = _allowed(tok, c)
    assert {"\n", "\n  ", " "} <= allowed
    assert " \n" not in allowed  # space then newline: odd run
    c.advance(" ")
    allowed = _allowed(tok, c)
    assert " " not in allowed and "\n" not in allowed and '"' in allowed


def test_natural_tokenization_is_walkable():
    tok = _Tok(PIECES)
    c = JsonSchemaConstraint(SCHEMA)
    for piece in ['{"', "city", '":', ' "', "Paris", '"}']:
        assert piece in _allowed(tok, c), piece
        c.advance(piece)
    assert c.is_done
