"""guided decoding under OPT-IN speculative decoding.

The default guided-decoding path (fast path: json_schema / json_object / regex / choice /
cfg) was verified CLEAN. Two real bugs lived only in the spec-decode paths:

HIGH — the cross-model SpeculativeDecoder wired the constraint via
JsonSchemaConstraint(json_schema) UNCONDITIONALLY, so a {"type":"regex"/"choice"/"cfg"}
spec was silently turned into a JSON-object constraint (the type falls through to the
default object branch) and the user's grammar was dropped with no error. Fixed with a
shared _build_grammar_constraint router (mirrors _build_constrained_sampler).

MEDIUM — verify_draft / batch_verify_drafts roll the constraint back to pre-draft before
re-masking the bonus token. JsonSchemaConstraint.rollback() is no-arg; Regex/Choice/Lark
rollback(saved) REQUIRES a dict checkpoint not held in those methods, so the no-arg call
raised TypeError, was suppressed, and the rollback was SKIPPED — then accepted_ids were
re-advanced on top of the K-draft state and the bonus mask was computed from a corrupted
buffer (could mask the correct token). Fixed: only re-mask when the rollback genuinely
succeeded; skip (don't corrupt) for grammar constraints.
"""
from __future__ import annotations

import inspect

from yunshu_engine import speculative_decoder
from yunshu_engine.batched_engine import _build_grammar_constraint
from yunshu_engine.grammar_constraint import (
    ChoiceConstraint,
    LarkGrammarConstraint,
    RegexConstraint,
)
from yunshu_engine.json_schema import JsonSchemaConstraint


class _Tok:
    def get_vocab(self):
        return {"a": 0, "b": 1, "<eos>": 2}

    def decode(self, ids):
        return "".join("ab"[i] if i < 2 else "" for i in ids)

    @property
    def eos_token_id(self):
        return 2

    def encode(self, s):
        return [0]


def test_build_grammar_constraint_routes_each_type():
    tok = _Tok()
    assert isinstance(_build_grammar_constraint({"type": "regex", "pattern": "[ab]+"}, tok), RegexConstraint)
    assert isinstance(_build_grammar_constraint({"type": "choice", "choices": ["a", "b"]}, tok), ChoiceConstraint)
    assert isinstance(_build_grammar_constraint({"type": "cfg", "grammar": 'start: "a"'}, tok), LarkGrammarConstraint)
    # JSON schema / json_object still go to JsonSchemaConstraint
    assert isinstance(_build_grammar_constraint({"type": "object", "properties": {}}, tok), JsonSchemaConstraint)
    assert isinstance(_build_grammar_constraint("json_object", tok), JsonSchemaConstraint)


def test_rollback_api_mismatch_premise_holds():
    """The MEDIUM fix relies on grammar rollback() being no-arg-incompatible while
    JSON rollback() works no-arg. Lock that invariant so the guard stays valid."""
    tok = _Tok()
    rc = _build_grammar_constraint({"type": "regex", "pattern": "[ab]+"}, tok)
    raised = False
    try:
        rc.rollback()  # grammar requires a saved dict → no-arg must fail
    except TypeError:
        raised = True
    assert raised, "RegexConstraint.rollback() unexpectedly accepts no arg"

    js = _build_grammar_constraint({"type": "object", "properties": {}}, tok)
    js.checkpoint()
    js.rollback()  # JSON no-arg rollback must NOT raise


def test_both_verify_sites_guard_rollback_success():
    src = inspect.getsource(speculative_decoder)
    # both bonus-mask sites gate the re-mask on a successful rollback (_rolled_back),
    # and no longer silently suppress the TypeError while still re-advancing
    assert src.count("_rolled_back = True") == 2
    assert src.count("_rolled_back = False") == 2
    assert "if _rolled_back:" in src
