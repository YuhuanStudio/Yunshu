"""Behaviour of the n-gram and speculative routes on the tiny CPU engine (tiny_text_engine).

engine_ngram / engine_speculative were only covered by import-order tests: here the n-gram
route is compared with the plain greedy fast path (repetitive prompt, stop strings,
suppress_tokens), the json_schema + open ``<think>`` redirect to the gated plain path is
checked, and the speculative-route eligibility gates are tabulated.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx_lm")

from yunshu_engine.ngram_proposer import NgramConfig, NgramProposer  # noqa: E402
from yunshu_engine.spec_draft_verifier import SpecDraftVerifier  # noqa: E402

from .tiny_text_engine import FakeTokenizer, make_engine  # noqa: E402

PROMPT = "abcabcabcabc"
N = 40
OPEN, CLOSE = 60, 61


def _spec_engine(seed: int = 0):
    eng = make_engine(seed)
    eng._ngram_proposer = NgramProposer(NgramConfig(max_n=3, k=4, mode="lps"))
    eng._ngram_stats = {"proposals": 0, "accepted": 0, "total_draft": 0}
    eng._spec_draft_verifier = SpecDraftVerifier(track_stats=True)
    eng._adaptive_spec = None
    eng._kv_manager = None
    eng._kv_prefix_cache = None
    eng._kv_quant_bits = None
    eng._kv_quant_group_size = 64
    eng._kv_quant_start = 0
    eng._lora_manager = None
    eng._mem_pressure_threshold = None
    eng.model_name = "tiny"
    return eng


async def _plain(eng, **kw):
    return await eng._generate_fast(prompt=PROMPT, max_tokens=N, temperature=0.0, **kw)


async def _spec(eng, **kw):
    return await eng._generate_ngram_spec(
        prompt=PROMPT, max_tokens=N, temperature=0.0, **kw
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
async def test_ngram_route_equals_plain_greedy(seed):
    eng = _spec_engine(seed)
    ref = await _plain(eng)
    out = await _spec(eng)
    assert out.error is None and ref.error is None
    assert out.text == ref.text and out.text
    assert out.finish_reason == ref.finish_reason
    assert out.completion_tokens == ref.completion_tokens
    assert eng._ngram_stats["proposals"] > 0  # it really speculated


async def test_ngram_route_with_stop_string_equals_plain():
    eng = _spec_engine(0)
    base = await _plain(eng)
    # a 2-char stop taken from the middle of the plain output (encodes to two ids: a suffix stop)
    stop = [base.text[10:12]]
    ref = await _plain(eng, stop=stop)
    out = await _spec(eng, stop=stop)
    assert ref.finish_reason == "stop" and len(ref.text) < len(base.text)
    assert out.text == ref.text
    assert out.finish_reason == "stop"
    assert stop[0] not in out.text


async def test_ngram_route_with_suppress_tokens_equals_plain():
    from yunshu_engine.text_logit_processors import TokenBan

    eng = _spec_engine(0)
    base = await _plain(eng)
    ids = [2 + (ord(c) % 60) for c in base.text[:1]]  # an id the continuation prints
    ban = sorted(
        {t for t in range(2, 64) if FakeTokenizer().decode([t]) == base.text[0]}
    )
    assert ids and ban
    ref = await _plain(eng, logits_processors=[TokenBan(ban)])
    out = await _spec(eng, suppress_tokens=ban)
    assert ref.text != base.text
    assert out.text == ref.text


async def test_json_schema_with_open_think_redirects_to_the_gated_plain_path():
    class ThinkTok(FakeTokenizer):
        think_start_id = OPEN

        def encode(self, text, add_special_tokens=True, **kw):
            if text == "<think>":
                return [OPEN]
            if text == "</think>":
                return [CLOSE]
            return super().encode(text, add_special_tokens=add_special_tokens)

    eng = _spec_engine(0)
    eng._tokenizer = ThinkTok()
    seen = {}

    async def fake_fast(**kw):
        seen.update(kw)
        from yunshu_engine.batched_engine import GenerationOutput

        return GenerationOutput(text="plain", finished=True)

    eng._generate_fast = fake_fast
    schema = {"type": "choice", "choices": ["ab"]}
    # prompt ends with the think-start token (chr(...) maps to OPEN): "hello:" in the
    # reasoning-gate tests; encode() of ThinkTok maps ':' to 2 + ord(':') % 60 == 60
    out = await eng._generate_ngram_spec(
        prompt="hello:",
        max_tokens=8,
        temperature=0.0,
        json_schema=schema,
        stop=["zz"],
    )
    assert out.text == "plain"  # served by the plain path, not the n-gram loop
    assert seen["json_schema"] == schema and seen["stop"] == ["zz"]
    assert eng._ngram_stats["proposals"] == 0


async def test_json_schema_without_open_think_stays_on_the_ngram_route(monkeypatch):
    from yunshu_engine.grammar_constraint import ConstraintFactory

    from .test_reasoning_gate import SeqConstraint

    # the tiny tokenizer has no vocabulary: a stand-in grammar that allows exactly "ab" then EOS
    monkeypatch.setattr(
        ConstraintFactory, "create", staticmethod(lambda *a, **k: SeqConstraint())
    )
    eng = _spec_engine(0)

    async def fake_fast(**kw):
        raise AssertionError("must not redirect when the prompt does not open <think>")

    eng._generate_fast = fake_fast
    out = await eng._generate_ngram_spec(
        prompt="hello",
        max_tokens=6,
        temperature=0.0,
        json_schema={"type": "choice", "choices": ["ab"]},
    )
    assert out.error is None and out.text == "ab"


# ── speculative eligibility gates ──


def _gemma_eng(proposer=object()):
    eng = make_engine()
    eng._gemma4_assistant_proposer = proposer
    return eng


_BASE = dict(
    logprobs=False,
    json_schema=None,
    logits_processors=None,
    logit_bias=None,
    top_p=1.0,
    top_k=0,
    min_p=0.0,
    repetition_penalty=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    xtc_probability=0.0,
)


def test_gemma4_spec_eligible_plain_greedy_request():
    assert _gemma_eng()._gemma4_spec_eligible(**_BASE) is True


def test_gemma4_spec_requires_the_loaded_drafter():
    assert _gemma_eng(proposer=None)._gemma4_spec_eligible(**_BASE) is False


@pytest.mark.parametrize(
    "override",
    [
        dict(logprobs=True),
        dict(json_schema={"type": "object"}),
        dict(logits_processors=[object()]),
        dict(logit_bias={3: 1.0}),
        dict(lora_adapter="a"),
        dict(top_p=0.9),
        dict(top_k=5),
        dict(min_p=0.1),
        dict(repetition_penalty=1.1),
        dict(frequency_penalty=0.2),
        dict(presence_penalty=0.2),
        dict(xtc_probability=0.5),
    ],
)
def test_gemma4_spec_excludes_every_request_the_primitive_cannot_reproduce(override):
    assert _gemma_eng()._gemma4_spec_eligible(**{**_BASE, **override}) is False


@pytest.mark.parametrize(
    "kw, expected",
    [
        (dict(use_engine_loop=True), None),
        (dict(logprobs=True), None),
        (dict(plain_only=True), None),
        (dict(stream=True), None),  # no streaming spec route
        (dict(temperature=0.7), None),  # lossless only at temperature 0
        (dict(), "ngram"),
        (dict(gemma4_eligible=lambda: True), "gemma4_assistant"),
        (dict(gemma4_eligible=lambda: False), "ngram"),
        (dict(spec_decode=False), None),  # neither per-request nor YUNSHU_NGRAM_DEFAULT
    ],
)
def test_spec_route_table(kw, expected):
    eng = make_engine()
    eng._ngram_proposer = object()
    eng._ngram_greedy_default = False
    args = dict(
        spec_decode=True,
        stream=False,
        temperature=0.0,
        logprobs=False,
        use_engine_loop=False,
    )
    args.update(kw)
    assert eng._spec_route(**args) == expected
