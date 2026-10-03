"""CPU compile reuse, fresh matcher state, and async admission regressions."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from yunshu_engine import grammar_compile as gc
from yunshu_engine.grammar_constraint import (
    CfgGrammarConstraint,
    LlgJsonSchemaConstraint,
    RegexConstraint,
    UnsupportedGrammarError,
    validate_llg_cfg,
    validate_llg_json_schema,
)


@pytest.fixture(autouse=True)
def clean_cache():
    gc.ARTIFACTS.clear()
    yield
    gc.ARTIFACTS.clear()


@pytest.fixture
def tok():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {
        c: i
        for i, c in enumerate(
            ["[UNK]", "[EOS]"] + sorted(pre_tokenizers.ByteLevel.alphabet())
        )
    }
    backend = Tokenizer(models.BPE(vocab, [], unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]"
    )


def test_schema_conversion_is_reused_from_validation_to_binding(monkeypatch, tok):
    from yunshu_engine import grammar_constraint as constraints

    original = constraints._llg_schema_json
    convert = Mock(wraps=original)
    monkeypatch.setattr(constraints, "_llg_schema_json", convert)
    schema = {"type": "string", "enum": ["yes", "no"]}
    validate_llg_json_schema(schema)
    one = LlgJsonSchemaConstraint(schema, tok)
    two = LlgJsonSchemaConstraint(json.loads(json.dumps(schema)), tok)
    assert convert.call_count == 1
    assert one._matcher is not two._matcher
    initial = two.get_allowed_tokens(tok, [])
    one.advance('"yes"')
    assert two._text_buffer == ""
    assert two.get_allowed_tokens(tok, []) == initial
    one.reset()
    assert one.get_allowed_tokens(tok, []) == initial


def test_cfg_templates_reused_with_independent_matchers(tok):
    grammar = 'start: "yes" | "no"'
    validate_llg_cfg(grammar)
    one = CfgGrammarConstraint(grammar, tokenizer=tok)
    two = CfgGrammarConstraint(grammar, tokenizer=tok)
    assert one._matcher is not two._matcher
    assert one._llt is two._llt
    saved = one.checkpoint()
    before = two.get_allowed_tokens(tok, [])
    one.advance("yes")
    assert one.is_done and not two.is_done
    assert two.get_allowed_tokens(tok, []) == before
    one.rollback(saved)
    assert one.get_allowed_tokens(tok, []) == before
    assert len([k for k in gc.ARTIFACTS._entries if k[0] == "matcher"]) == 1


def test_tokenizer_identity_and_compact_options_are_separate(tok):
    from copy import deepcopy

    schema = {"type": "string", "enum": ["yes", "no"]}
    one = LlgJsonSchemaConstraint(schema, tok)
    two = LlgJsonSchemaConstraint(schema, deepcopy(tok))
    compact = LlgJsonSchemaConstraint(schema, tok, compact=True)
    assert one._llt is not two._llt
    assert one._make_grammar() != compact._make_grammar()
    assert len([k for k in gc.ARTIFACTS._entries if k[0] == "matcher"]) == 3


def test_schema_property_order_is_part_of_key():
    a = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
    }
    b = {"type": "object", "properties": dict(reversed(list(a["properties"].items())))}
    assert gc.schema_source(a) != gc.schema_source(b)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize(
    "text", ['{"description":"yes","n":1}', '{"description":"no","n":2}']
)
def test_annotation_variants_reuse_artifacts_with_original_masks(
    monkeypatch, tok, compact, text
):
    schema = {
        "type": "object",
        "title": "A response",
        "description": "Documentation only",
        "$comment": "Version one",
        "properties": {
            "description": {
                "type": "string",
                "enum": ["yes", "no"],
                "description": "Choice",
            },
            "n": {"type": "integer", "minimum": 1, "maximum": 2, "title": "Count"},
        },
        "required": ["description", "n"],
    }
    # Compile the original annotated schema before changing any cache key.
    with monkeypatch.context() as patch:
        patch.setattr(gc, "annotation_free_source", lambda source: source)
        original = LlgJsonSchemaConstraint(schema, tok, compact=compact)
    gc.ARTIFACTS.clear()
    normalized = LlgJsonSchemaConstraint(schema, tok, compact=compact)
    variant = json.loads(json.dumps(schema))
    variant["description"] = "Different documentation"
    variant["properties"]["n"]["title"] = "Different count"
    reused = LlgJsonSchemaConstraint(variant, tok, compact=compact)
    assert normalized._schema_json == reused._schema_json
    assert len([k for k in gc.ARTIFACTS._entries if k[0] == "matcher"]) == 1
    assert normalized._matcher is not reused._matcher
    for tid in tok.encode(text):
        masks = [c.get_allowed_tokens(tok, []) for c in (original, normalized, reused)]
        assert masks[0] == masks[1] == masks[2] and tid in masks[0]
        for constraint in (original, normalized, reused):
            constraint.advance(tok.decode([tid]))
    assert (
        original.get_allowed_tokens(tok, [])
        == normalized.get_allowed_tokens(tok, [])
        == reused.get_allowed_tokens(tok, [])
    )


def test_annotation_cleanup_preserves_data_references_and_normalization():
    from yunshu_engine.grammar_constraint import _llg_schema_json

    schema = {
        "type": "object",
        "properties": {"title": {"const": {"description": "literal", "title": "data"}}},
        "examples": [{"description": "example data"}],
    }
    cleaned = json.loads(gc.schema_source(schema))
    assert (
        cleaned["properties"]["title"]["const"]
        == schema["properties"]["title"]["const"]
    )
    assert cleaned["examples"] == schema["examples"]
    reference = {"$ref": "#/description", "description": {"type": "string"}}
    assert gc.schema_source(reference) == _llg_schema_json(reference)
    # These branches do not merge under the existing normalization policy.
    branches = {
        "allOf": [
            {"properties": {"x": {"type": "string", "description": "one"}}},
            {"properties": {"x": {"type": "string", "description": "two"}}},
        ]
    }
    assert "allOf" in json.loads(gc.schema_source(branches))
    assert gc.schema_source({"type": "string", "minLength": 1}) != gc.schema_source(
        {"type": "string", "minLength": 2}
    )


def test_regex_build_reused_but_lazy_caches_are_private(monkeypatch):
    from yunshu_engine import grammar_constraint as constraints

    constructor = Mock(wraps=constraints._RegexDFA)
    monkeypatch.setattr(constraints, "_RegexDFA", constructor)
    one = RegexConstraint("(a|b){1,4}")
    two = RegexConstraint("(a|b){1,4}")
    assert constructor.call_count == 1
    assert one._dfa is not two._dfa
    assert one._dfa._trans is not two._dfa._trans
    assert one._dfa._closure_cache is not two._dfa._closure_cache
    one.advance("ab")
    assert two._text_buffer == ""
    assert two._dfa_state == two._dfa._dfa_start


def test_single_flight_and_retry_after_error():
    cache = gc.ArtifactCache(max_entries=2)
    entered, release = threading.Event(), threading.Event()
    count = []

    def build():
        count.append(1)
        entered.set()
        assert release.wait(2)
        return object()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(cache.get, ("same",), build) for _ in range(4)]
        assert entered.wait(2)
        release.set()
        results = [f.result() for f in futures]
    assert len(count) == 1 and all(x is results[0] for x in results)
    with pytest.raises(ValueError):
        cache.get(("bad",), lambda: (_ for _ in ()).throw(ValueError("bad")))
    assert cache.get(("bad",), lambda: "retry") == "retry"
    cache.get(("third",), lambda: "third")
    assert len(cache._entries) == 2
    assert ("same",) not in cache._entries


def test_key_byte_bound_and_oversized_bypass():
    cache = gc.ArtifactCache(max_entries=10, max_key_bytes=100)
    for i in range(10):
        cache.get((str(i) * 15,), lambda: object())
    assert cache._bytes <= 100
    build = Mock(side_effect=lambda: object())
    assert cache.get(("x" * 100,), build) is not cache.get(("x" * 100,), build)
    assert build.call_count == 2


@pytest.mark.asyncio
async def test_async_compile_releases_event_loop_and_cancel_is_safe(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    seen = []

    def slow(spec):
        seen.append(threading.current_thread().name)
        entered.set()
        assert release.wait(2)

    monkeypatch.setattr(gc, "_prepare_spec", slow)
    task = asyncio.create_task(gc.prepare_constraint({"type": "string"}))
    try:
        for _ in range(100):
            await asyncio.sleep(0.001)
            if entered.is_set():
                break
        assert entered.is_set() and seen[0].startswith("yunshu-grammar")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()


@pytest.mark.asyncio
async def test_async_errors_prevent_admission_and_good_retry_works():
    admitted = []

    async def request(spec):
        await gc.prepare_constraint(spec)
        admitted.append(True)

    with pytest.raises(UnsupportedGrammarError):
        await request({"type": "cfg", "grammar": "start: ???"})
    assert admitted == []
    await request({"type": "cfg", "grammar": 'start: "yes"'})
    assert admitted == [True]


@pytest.mark.asyncio
async def test_hot_preparation_has_no_worker_round_trip(monkeypatch):
    original = gc._prepare_spec
    prepare = Mock(wraps=original)
    monkeypatch.setattr(gc, "_prepare_spec", prepare)
    spec = {"type": "string"}
    await gc.prepare_constraint(spec)
    await gc.prepare_constraint(json.loads(json.dumps(spec)))
    assert prepare.call_count == 1


@pytest.mark.parametrize("kind", ["cfg", "schema", "compact"])
def test_cold_and_cached_token_masks_match_every_position(tok, kind):
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "string", "enum": ["yes", "no"]},
        },
        "required": ["a", "b"],
    }
    if kind == "cfg":

        def build():
            return CfgGrammarConstraint('start: "yes" | "no"', tokenizer=tok)

        text = "yes"
    else:

        def build():
            return LlgJsonSchemaConstraint(schema, tok, compact=kind == "compact")

        text = '{"a":1,"b":"yes"}'
    cold = build()
    hot = build()
    for tid in tok.encode(text):
        a, b = cold.get_allowed_tokens(tok, []), hot.get_allowed_tokens(tok, [])
        assert a == b and tid in a
        cold.advance(tok.decode([tid]))
        hot.advance(tok.decode([tid]))
    assert cold.get_allowed_tokens(tok, []) == hot.get_allowed_tokens(tok, [])
    assert cold._text_buffer == hot._text_buffer == text
