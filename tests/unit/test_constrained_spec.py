"""Exact constrained verification, forced windows and accepted target logprobs."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from yunshu_engine import mtp_lane
from yunshu_engine.vlm_batch_runner import _DONE, VLMBatchRunner


@pytest.mark.parametrize("kind", ["mtp", "dflash"])
@pytest.mark.parametrize("constrained", [False, True])
def test_guided_logprob_request_keeps_spec(monkeypatch, kind, constrained):
    monkeypatch.setitem(mtp_lane._STATE, "installed", True)
    monkeypatch.setitem(mtp_lane._STATE, "enabled", True)
    runner = VLMBatchRunner(
        SimpleNamespace(
            language_model=SimpleNamespace(
                speculative_verify_dflash_hidden=lambda: None
            )
        ),
        None,
        drafter=SimpleNamespace(supports_greedy_draft_argmax=True),
        draft_kind=kind,
    )
    captured = []

    def submit(job):
        captured.append(job)
        job.out.put(_DONE)

    monkeypatch.setattr(runner, "_submit", submit)
    guide = SimpleNamespace(constrained=constrained)
    list(runner.iter_tokens([1, 2], max_tokens=3, guide=guide, logprobs=True))
    assert captured[0].use_draft


from tests.unit.test_mtp_lane_tool_guide import (  # noqa: E402
    ToyTarget,
    ids_of,
    reference,
    run_lane,
)
from tests.unit.test_mtp_lane_tool_guide import toy as toy
from yunshu_engine.constrained_spec import (  # noqa: E402
    ConstraintGuide,
    SpecRequest,
    forced_tokens,
    normalize_rows,
    set_request,
)
from yunshu_engine.grammar_constraint import ConstraintFactory  # noqa: E402
from yunshu_engine.vlm_batch_runner import ConstraintProcessor  # noqa: E402


def make_guide(hf, kind):
    spec = {
        "cfg": 'start: "abc" | "xyz"',
        "regex": "(abc|xyz)",
        "choice": ["abc", "xyz"],
        "json_schema": {
            "type": "object",
            "properties": {"k": {"enum": ["abc", "xyz"]}},
            "required": ["k"],
            "additionalProperties": False,
        },
    }[kind]
    constraint = ConstraintFactory.create(kind, spec, hf)
    return ConstraintGuide(ConstraintProcessor(constraint, hf), len(hf) + 4)


@pytest.mark.parametrize("kind", ["cfg", "regex", "choice", "json_schema"])
@pytest.mark.parametrize("accuracy", [0.0, 0.7, 1.0])
def test_generic_masks_match_serial_and_rollback(toy, monkeypatch, kind, accuracy):
    hf = toy
    text = '{"k":"abc"}' if kind == "json_schema" else "abc"
    script = ids_of(hf, text) + [hf.eos_token_id]
    target = ToyTarget(script, len(hf) + 4, hf.eos_token_id)
    ref = reference(target, make_guide(hf, kind), script[0], hf.eos_token_id, 40)
    got = run_lane(
        monkeypatch,
        target,
        make_guide(hf, kind),
        script[0],
        hf.eos_token_id,
        40,
        accuracy,
        3,
        7,
    )
    assert got == ref == script


def test_forced_window_preserves_committed_state(toy):
    guide = make_guide(toy, "cfg")
    assert forced_tokens(guide, 8) == []  # a or x is a real model choice
    guide.feed(ids_of(toy, "a")[0])
    assert forced_tokens(guide, 2) == ids_of(toy, "bc")
    assert guide.processor._generated == ids_of(toy, "a")
    # Planning an illegal draft must leave the live native matcher usable.
    guide.plan(ids_of(toy, "zbc"), 4)
    assert forced_tokens(guide, 2) == ids_of(toy, "bc")
    assert not guide.processor._constraint._ckpt_stack


@pytest.mark.parametrize("k", [0, 5, 20])
@pytest.mark.parametrize("accuracy", [0.0, 1.0])
def test_accepted_logprobs_match_serial_rows(toy, monkeypatch, k, accuracy):
    hf = toy
    script = ids_of(hf, "abc") + [hf.eos_token_id]
    target = ToyTarget(script, len(hf) + 4, hf.eos_token_id)
    request = SpecRequest(True, k)
    set_request(request)
    try:
        got = run_lane(
            monkeypatch,
            target,
            make_guide(hf, "cfg"),
            script[0],
            hf.eos_token_id,
            40,
            accuracy,
            3,
            7,
        )
    finally:
        set_request(None)
    assert got == script
    refguide = make_guide(hf, "cfg")
    refguide.feed(got[0])
    from yunshu_engine.tool_call_grammar import apply_bitmask

    for i, token in enumerate(got[1:]):
        logits = apply_bitmask(mx.array(target.logits(i))[None, None], refguide.mask())
        probs = normalize_rows(logits.astype(mx.float32))
        expected = probs[0, token].item()
        lp, top = request.take(token)
        assert lp == expected
        idx = mx.argsort(probs, axis=-1)[0, -k:][::-1].tolist() if k else []
        assert [t for t, _ in top] == idx
        assert [v for _, v in top] == [probs[0, t].item() for t in idx]
        refguide.feed(token)
    assert not request.pending


def test_first_logprobs_are_not_dummy_zero():
    request = SpecRequest(True, 2)
    logits = mx.array([[24, 19.75, 19.6, 10]], dtype=mx.bfloat16)
    from mlx_vlm.generate import ar

    from yunshu_engine.constrained_spec import install

    install()
    probs = normalize_rows(request(mx.array([1]), logits)[None])
    set_request(request)
    try:
        ar._sample_with_positions(
            lambda x: mx.argmax(x, axis=-1), probs, row_ids=[0], positions=[0]
        )
    finally:
        set_request(None)
    lp, top = request.take(0)
    probs = normalize_rows(logits.astype(mx.float32)[None])
    assert lp == probs[0, 0].item() < 0
    assert top[0] == (0, lp)


def test_native_mask_pads_model_vocab_with_forbidden_tokens(toy):
    c = ConstraintFactory.create("cfg", 'start: "abc"', toy)
    guide = ConstraintGuide(ConstraintProcessor(c, toy), len(toy) + 128)
    mask = guide.mask()
    import numpy as np

    bits = np.unpackbits(mask.view(np.uint8), bitorder="little")
    assert not bits[len(toy) :].any()
    assert bits[ids_of(toy, "a")[0]]


@pytest.mark.parametrize("lane", ["mtp", "dflash"])
@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("accuracy", [0.0, 0.6, 1.0])
def test_both_lanes_keyed_tokens_and_logprobs(
    toy, monkeypatch, lane, sampled, accuracy
):
    import numpy as np
    from mlx_vlm.speculative import cache_state

    from yunshu_engine.constrained_spec import dflash_rounds, target_rows
    from yunshu_engine.keyed_sampling import KeyedSampler
    from yunshu_engine.vlm_batch_runner import RowParams

    hf = toy
    vocab = len(hf) + 4
    script = ids_of(hf, "abx") + [hf.eos_token_id]
    target = ToyTarget(script, vocab, hf.eos_token_id)

    def guide():
        c = ConstraintFactory.create("cfg", 'start: "a" ("b" | "c") ("x" | "y")', hf)
        return ConstraintGuide(ConstraintProcessor(c, hf), vocab)

    keyed = KeyedSampler(RowParams(15, 0.95, 4, 0.01, 123), 123) if sampled else None
    refguide = guide()
    ref = [script[0]]
    refguide.feed(ref[0])
    expected = []
    for i in range(3):
        row = mx.array(target.logits(i))[None, None]
        toks, probs = target_rows(
            row, refguide.mask(), i + 1, keyed, SpecRequest(True, 5)
        )
        token = toks.item()
        expected.append((token, probs[0, token].item()))
        ref.append(token)
        refguide.feed(token)
    request = SpecRequest(True, 5)
    request.guide = guide()
    set_request(request)
    try:
        if lane == "mtp":
            got = run_lane(
                monkeypatch,
                target,
                request.guide,
                script[0],
                hf.eos_token_id,
                4,
                accuracy,
                3,
                4,
                keyed=keyed,
            )
        else:
            state = {"j": 0}
            rng = np.random.default_rng(42)

            def verify(inputs, cache, layers):
                n = inputs.shape[1]
                final = mx.array(
                    np.stack([target.logits(state["j"] + i) for i in range(n)])
                )[None]
                return [final], final, None

            def commit(lm, cache, states, accepted, bs):
                state["j"] += accepted + 1

            def draft(b, hidden, cache, bs, sampler, dtype, **kw):
                proposals = [
                    int(np.argmax(target.logits(state["j"] + i)))
                    if rng.random() < accuracy
                    else int(rng.integers(vocab))
                    for i in range(bs - 1)
                ]
                return mx.array([proposals], dtype=dtype)

            monkeypatch.setattr(cache_state, "commit_speculative_round", commit)
            monkeypatch.setitem(mtp_lane._STATE, "copy_rows", 0)
            lm = SimpleNamespace(
                speculative_verify_dflash_hidden=verify,
                speculative_logits_from_hidden=lambda h: h,
            )
            head = SimpleNamespace(
                config=SimpleNamespace(target_layer_ids=[0], block_size=4),
                reset=lambda m: [],
                draft_block=draft,
                accept_lens=[],
                draft_lens=[],
            )
            got = [script[0]]
            for tokens, _ in dflash_rounds(
                lm,
                head,
                [],
                mx.zeros((1, 1, 1)),
                request=request,
                first_bonus=mx.array([script[0]]),
                max_tokens=4,
                sampler=keyed,
                draft_block_size=4,
            ):
                got.extend(tokens)
    finally:
        set_request(None)
    assert got == ref
    for token, lp in expected:
        actual, _ = request.take(token)
        assert actual == lp
    assert not request.pending


@pytest.mark.parametrize(
    "kind", ["cfg", "regex", "choice", "json_schema", "inhouse", "bitmask"]
)
def test_guide_masks_are_exactly_the_existing_serial_processor(toy, kind):
    from yunshu_engine.grammar_bitmask import GrammarBitmaskEngine
    from yunshu_engine.json_schema import JsonSchemaConstraint
    from yunshu_engine.tool_call_grammar import apply_bitmask

    def make():
        if kind == "inhouse":
            return JsonSchemaConstraint(
                {
                    "type": "object",
                    "properties": {"k": {"type": "string"}},
                    "required": ["k"],
                    "additionalProperties": False,
                }
            )
        if kind == "bitmask":
            return GrammarBitmaskEngine(
                ConstraintFactory.create("choice", ["abc", "xyz"], toy)
            )
        return make_guide(toy, kind).processor._constraint

    serial = ConstraintProcessor(make(), toy)
    guide = ConstraintGuide(ConstraintProcessor(make(), toy), len(toy) + 128)
    script = ids_of(
        toy, '{"k":"abc"}' if kind in ("json_schema", "inhouse") else "abc"
    ) + [toy.eos_token_id]
    logits = mx.arange(len(toy) + 128, dtype=mx.float32)[None]
    out = serial(mx.array([0]), logits)
    for token in script:
        native = apply_bitmask(logits, guide.mask())
        assert mx.array_equal(out, native).item()
        guide.feed(token)
        out = serial.process_last_token(token, logits)


def test_probe_does_not_close_a_live_tool_call(toy, caplog):
    from tests.unit.test_mtp_lane_tool_guide import make_grammar

    grammar = make_grammar(toy)
    guide = grammar.guide()
    guide.feed(grammar.start_id)
    # Probe the whole call through plan (its close must not affect counters/logs).
    body = ids_of(toy, "<function=ab>\n</function>") + [grammar.end_id]
    before = guide.checkpoint()
    guide.plan(body, len(body) + 1)
    assert guide.checkpoint() == before
    assert guide._planning == 0 and guide.calls == 0
    assert not any("tool call closed" in r.message for r in caplog.records)


def test_dflash_forced_windows_retain_unabsorbed_draft_context(toy, monkeypatch):
    import numpy as np
    from mlx_vlm.speculative import cache_state

    from yunshu_engine.constrained_spec import dflash_rounds

    script = ids_of(toy, "abcxuv") + [toy.eos_token_id]
    vocab = len(toy) + 4
    target = ToyTarget(script, vocab, toy.eos_token_id)
    c = ConstraintFactory.create(
        "cfg", 'start: "a" "bc" ("x" | "y") ("uv" | "pq")', toy
    )
    request = SpecRequest()
    request.guide = ConstraintGuide(ConstraintProcessor(c, toy), vocab)
    state = {"j": 0}
    seen = []

    def verify(inputs, cache, layers):
        rows = mx.array(
            np.stack([target.logits(state["j"] + i) for i in range(inputs.shape[1])])
        )[None]
        return [rows], rows, None

    def commit(lm, cache, states, accepted, bs):
        state["j"] += accepted + 1

    def draft(bonus, hidden, cache, bs, sampler, dtype):
        seen.append(hidden.shape[1])
        return mx.array(
            [[int(np.argmax(target.logits(state["j"] + i))) for i in range(bs - 1)]],
            dtype=dtype,
        )

    monkeypatch.setattr(cache_state, "commit_speculative_round", commit)
    monkeypatch.setitem(mtp_lane._STATE, "copy_rows", 0)
    lm = SimpleNamespace(
        speculative_verify_dflash_hidden=verify,
        speculative_logits_from_hidden=lambda x: x,
    )
    head = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[0], block_size=4),
        reset=lambda m: [],
        draft_block=draft,
        accept_lens=[],
        draft_lens=[],
    )
    got = [script[0]]
    for tokens, _ in dflash_rounds(
        lm,
        head,
        [],
        mx.zeros((1, 2, vocab)),
        request=request,
        first_bonus=mx.array([script[0]]),
        max_tokens=len(script),
        sampler=None,
        draft_block_size=4,
    ):
        got.extend(tokens)
    assert got == script
    assert seen[0] == 5  # prompt's two rows plus the forced verify's three rows


def test_first_report_uses_the_distribution_already_sampled():
    request = SpecRequest(True, 2)
    request.first_probs = mx.array([[-0.125, -2.75]])
    lp, top = request.take(0)
    assert lp == -0.125 and top == [(0, -0.125), (1, -2.75)]
    assert request.first_probs is None


def test_serial_verify_routing_restored_on_error(monkeypatch):
    from yunshu_engine.constrained_spec import exact_verify
    from yunshu_engine.kernels import batch_invariant, omlx
    from yunshu_engine.kernels.omlx import qwen35_verify_qmm as qmm

    monkeypatch.setitem(batch_invariant._STATE, "installed", True)
    monkeypatch.setitem(batch_invariant._STATE, "active", True)
    monkeypatch.setitem(omlx._STATE, "row_exact", False)
    before = qmm._is_armed(), qmm.is_row_exact_armed()
    with pytest.raises(RuntimeError, match="abort"):
        with exact_verify(SpecRequest(True)):
            assert not batch_invariant._STATE["active"]
            assert omlx._STATE["row_exact"] and qmm.is_row_exact_armed()
            raise RuntimeError("abort")
    assert batch_invariant._STATE["active"]
    assert not omlx._STATE["row_exact"]
    assert (qmm._is_armed(), qmm.is_row_exact_armed()) == before


@pytest.mark.parametrize("feature", ["guide", "logprobs", "plain"])
@pytest.mark.parametrize("spec", [False, True])
def test_serial_requests_preserve_ar_attention_and_projections(
    monkeypatch, feature, spec
):
    from yunshu_engine import vlm_batch_runner as vbr
    from yunshu_engine.kernels import batch_invariant, ragged_kv

    monkeypatch.setitem(batch_invariant._STATE, "installed", True)
    monkeypatch.setitem(batch_invariant._STATE, "active", False)
    monkeypatch.setitem(ragged_kv._STATE, "dense_lane", False)
    monkeypatch.setitem(ragged_kv._STATE, "format", None)
    runner = VLMBatchRunner(SimpleNamespace(language_model=object()), None)
    runner.ragged_kv = "bf16"
    runner.prefix_invariant = True
    job = SimpleNamespace(
        abandoned=False,
        cancel_event=None,
        guide=object() if feature == "guide" else None,
        logprobs=feature == "logprobs",
    )
    group = vbr._Group(gen=None, spec=spec, jobs={1: job})
    seen = []
    monkeypatch.setattr(
        runner,
        "_step_generator",
        lambda group: seen.append(
            (batch_invariant._STATE["active"], ragged_kv._STATE["dense_lane"])
        ),
    )
    runner._step_group(group)
    assert seen == [(feature == "plain", feature == "plain")]
    assert not batch_invariant._STATE["active"] and not ragged_kv._STATE["dense_lane"]


def test_qualified_ar_and_spec_use_same_canonical_cache_salt(monkeypatch):

    monkeypatch.setitem(mtp_lane._STATE, "installed", True)
    monkeypatch.setitem(mtp_lane._STATE, "enabled", True)
    gen = SimpleNamespace(
        insert=lambda *args, **kw: (
            saved.append(kw["prompt_kwargs"][0]["_apc_semantic_hash"]) or [0]
        )
    )
    runner = VLMBatchRunner(
        SimpleNamespace(language_model=object()),
        None,
        drafter=SimpleNamespace(supports_greedy_draft_argmax=True),
        draft_kind="mtp",
    )
    saved = []
    jobs = []

    def submit(job):
        jobs.append(job)
        job.out.put(_DONE)

    monkeypatch.setattr(runner, "_submit", submit)
    monkeypatch.setattr(runner, "_new_generator", lambda **kw: gen)
    for guide, lp, draft in [
        (None, False, False),
        (object(), False, False),
        (object(), False, True),
        (None, True, False),
        (None, True, True),
    ]:
        list(
            runner.iter_tokens(
                [1, 2],
                max_tokens=3,
                prompt_kwargs={},
                apc_semantic_hash=123,
                guide=guide,
                logprobs=lp,
                allow_draft=draft,
            )
        )
        runner._spec = None
        runner._batches.clear()
        runner._admit(jobs[-1], alone=True)
    assert saved[0] == 123
    assert len(set(saved[1:])) == 1 and saved[1] != saved[0]


def test_greedy_uses_serial_normalized_tie_breaking():
    from yunshu_engine.constrained_spec import target_rows

    logits = mx.array([[[0.0001, 0.00010002, 0.0]]], dtype=mx.float32)
    serial = normalize_rows(logits)
    assert mx.argmax(logits).item() == 1
    assert mx.argmax(serial).item() == 0  # subtraction rounded the two maxima to a tie
    target, _ = target_rows(logits, None, 1, None, SpecRequest(True))
    assert target.item() == mx.argmax(serial).item()


def test_stock_singleton_prefill_then_dense_verify_representation():
    from mlx_vlm.models.cache import BatchKVCache, KVCache

    from yunshu_engine.constrained_spec import (
        finish_serial_prefill,
        prepare_serial_prefill,
    )

    gen = SimpleNamespace(
        _prompt_batch=SimpleNamespace(prompt_cache=[BatchKVCache([0])])
    )
    prepare_serial_prefill(gen)
    cache = gen._prompt_batch.prompt_cache[0]
    assert isinstance(cache, KVCache)
    cache.keys = mx.ones((1, 1, 2, 2), dtype=mx.bfloat16)
    cache.values = mx.full((1, 1, 2, 2), 2, dtype=mx.bfloat16)
    cache.offset = 2
    gen._generation_batch = SimpleNamespace(prompt_cache=[cache])
    gen._prompt_batch = None
    finish_serial_prefill(gen)
    dense = gen._generation_batch.prompt_cache[0]
    assert isinstance(dense, BatchKVCache)
    assert dense.left_padding is None and dense.offset == 2
    assert mx.array_equal(dense.keys[..., :2, :], cache.keys).item()
    assert mx.array_equal(dense.values[..., :2, :], cache.values).item()


def test_constructor_canonicalizes_before_immediate_prefill():
    from mlx_vlm.generate import ar
    from mlx_vlm.models.cache import ArraysCache, KVCache

    from yunshu_engine.constrained_spec import install

    install()
    model = SimpleNamespace(
        make_cache=lambda: [KVCache(), ArraysCache(size=2)],
        supports_chunked_prefill=lambda **kw: True,
    )
    request = SpecRequest()
    request.guide = object()
    set_request(request)
    try:
        prompt = ar.PromptProcessingBatch(
            model,
            [0],
            [[1, 2, 3]],
            [4],
            mx.zeros((1, 3, 2)),
            {},
            draft_model=object(),
            draft_kind="mtp",
        )
    finally:
        set_request(None)
    assert isinstance(prompt.prompt_cache[0], KVCache)
    assert prompt.prompt_cache[1].left_padding is None
    assert prompt.prompt_cache[1].lengths is None


def test_plain_and_exact_ar_requests_do_not_share_arithmetic_group(monkeypatch):
    runner = VLMBatchRunner(SimpleNamespace(language_model=object()), None)
    jobs = []
    saved = []
    gen = SimpleNamespace(insert=lambda *a, **kw: [0])

    def submit(job):
        jobs.append(job)
        job.out.put(_DONE)

    monkeypatch.setattr(runner, "_submit", submit)
    monkeypatch.setattr(runner, "_new_generator", lambda **kw: saved.append(kw) or gen)
    for guide in (None, object()):
        list(
            runner.iter_tokens(
                [1, 2], max_tokens=3, prompt_kwargs={}, guide=guide, allow_draft=False
            )
        )
        runner._admit(jobs[-1], alone=True)
    assert len(runner._batches) == 2
    assert len(saved) == 2
