"""a TOP-LEVEL scalar number/integer JSON schema (guided_json={"type":"integer"},
response_format json_schema with a scalar root — both allowed by OpenAI/vLLM structured
outputs) could never terminate. Unlike a string (closing `"`), object (`}`), or array
(`]`), a number has no terminating structural char, so the FSM stayed in the NUMBER state,
is_done stayed False, and get_allowed_tokens never offered EOS → the model was masked into
emitting digits until max_tokens (runaway number, finish_reason=length). can_terminate()
now reports a complete top-level number and get_allowed_tokens offers EOS there. A NESTED
number must still end via the enclosing comma/brace and must NOT stop early."""
from __future__ import annotations

from yunshu_engine.json_schema import JsonSchemaConstraint, JsonState


def _advance(schema, text):
    c = JsonSchemaConstraint(schema)
    for ch in text:
        c.advance(ch)
    return c


def test_toplevel_integer_can_terminate_after_digit():
    c = _advance({"type": "integer"}, "4")
    assert c.can_terminate() is True


def test_toplevel_integer_not_complete_at_minus():
    c = _advance({"type": "number"}, "-")
    assert c.can_terminate() is False


def test_toplevel_number_fraction_incomplete():
    c = _advance({"type": "number"}, "3.")
    assert c.can_terminate() is False  # needs a digit after the dot


def test_toplevel_number_can_continue_or_stop():
    c = _advance({"type": "integer"}, "42")
    assert c.can_terminate() is True  # complete; the model may also emit more digits


def test_nested_number_does_not_terminate_early():
    c = _advance({"type": "object", "properties": {"x": {"type": "integer"}},
                  "required": ["x"]}, '{"x":4')
    assert c.can_terminate() is False


def test_get_allowed_tokens_offers_eos_iff_can_terminate():
    # A real-enough tokenizer: distinct eos id well outside the digit-token range.
    class _Tok:
        eos_token_id = 4242
        def get_vocab(self):
            # map single chars to ids so _find_tokens_for_chars finds digit tokens
            return {ch: i for i, ch in enumerate("0123456789-.eE ")}
        def convert_ids_to_tokens(self, tid):
            inv = {i: ch for ch, i in self.get_vocab().items()}
            return inv.get(tid, "")
        def decode(self, ids):
            inv = {i: ch for ch, i in self.get_vocab().items()}
            return "".join(inv.get(i, "") for i in ids)
    tok = _Tok()
    done = _advance({"type": "integer"}, "4")
    nested = _advance({"type": "object", "properties": {"x": {"type": "integer"}},
                       "required": ["x"]}, '{"x":4')
    assert 4242 in done.get_allowed_tokens(tok, []), "top-level complete number must offer EOS"
    assert 4242 not in nested.get_allowed_tokens(tok, []), "nested number must NOT offer EOS"


def test_bare_zero_is_complete():
    c = _advance({"type": "integer"}, "0")
    assert c.state == JsonState.NUMBER_ZERO
    assert c.can_terminate() is True


def test_string_doc_terminates_only_at_done():
    c = _advance({"type": "string"}, '"hi"')
    assert c.state == JsonState.DONE
    assert c.can_terminate() is True
