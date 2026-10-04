"""Fast tree admission and transaction helpers preserve fallback semantics."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")
from yunshu_engine import dflash_fast as fast  # noqa: E402
from yunshu_engine import mtp_lane
from yunshu_engine import tree_verify as tv


@pytest.fixture
def admission(monkeypatch):
    monkeypatch.setattr(fast, "supported", lambda *_: True)
    monkeypatch.setattr(tv, "lane_ready", lambda *_: True)
    monkeypatch.setattr(
        fast, "context_length", lambda *_: len(mtp_lane._STATE["context"])
    )
    monkeypatch.setattr(mtp_lane, "copy_rows_for_model", lambda *_: 16)
    monkeypatch.setitem(mtp_lane._STATE, "guide", None)
    monkeypatch.setitem(mtp_lane._STATE, "context", list(range(1034)))
    return dict(max_tokens=256, greedy_sampling=True, sampler=None, draft_block_size=8)


def test_measured_short_request_is_admitted(admission):
    assert fast.eligible(None, None, [], admission)


@pytest.mark.parametrize(
    "damage",
    [
        "long",
        "short",
        "guide",
        "copy_off",
        "no_context",
        "sampled",
        "block",
        "backend",
        "cache",
    ],
)
def test_other_requests_keep_the_original_rounds(monkeypatch, admission, damage):
    if damage == "long":
        monkeypatch.setitem(mtp_lane._STATE, "context", list(range(20000)))
    elif damage == "short":
        monkeypatch.setitem(mtp_lane._STATE, "context", [1] * 40)
    elif damage == "guide":
        monkeypatch.setitem(mtp_lane._STATE, "guide", object())
    elif damage == "copy_off":
        monkeypatch.setattr(mtp_lane, "copy_rows_for_model", lambda *_: 0)
    elif damage == "no_context":
        monkeypatch.setitem(mtp_lane._STATE, "context", None)
    elif damage == "sampled":
        admission["greedy_sampling"] = False
    elif damage == "block":
        admission["draft_block_size"] = 4
    elif damage == "backend":
        monkeypatch.setattr(fast, "supported", lambda *_: False)
    else:
        monkeypatch.setattr(tv, "lane_ready", lambda *_: False)
    assert not fast.eligible(None, None, [], admission)


@pytest.mark.parametrize("maximum", [None, 1, 32, 256, 4096, 32768])
def test_admission_does_not_depend_on_request_budget(admission, maximum):
    if maximum is None:
        admission.pop("max_tokens")
    else:
        admission["max_tokens"] = maximum
    assert fast.eligible(None, None, [], admission)


def test_live_context_bound(monkeypatch, admission):
    monkeypatch.setitem(mtp_lane._STATE, "context", [1] * (fast.CONTEXT_LIMIT - 2))
    assert fast.eligible(None, None, [], admission)
    monkeypatch.setitem(mtp_lane._STATE, "context", [1] * (fast.CONTEXT_LIMIT - 1))
    assert not fast.eligible(None, None, [], admission)
    assert fast.live_eligible(
        fast.CONTEXT_MIN, fast.CONTEXT_LIMIT - fast.CONTEXT_MIN - 1
    )
    assert not fast.live_eligible(
        fast.CONTEXT_MIN, fast.CONTEXT_LIMIT - fast.CONTEXT_MIN
    )


def test_chain_handoff_preserves_cache_copy_history_and_bonus():
    from yunshu_engine.copy_drafter import CopyDrafter
    from yunshu_engine.dflash_copy import resume_chain

    draft = SimpleNamespace(accept_lens=[3, 4], config=SimpleNamespace())
    copy = CopyDrafter(max_draft=15)
    copy.extend([1, 2, 3, 4])
    draft_cache = [object()]
    target_cache = [object()]
    hidden = object()
    seen = []

    def original(model, proxy, cache, taps, **kw):
        assert proxy.reset(model) is draft_cache
        assert proxy._copy is copy
        assert cache is target_cache and taps is hidden
        assert proxy._seen_rounds == 2
        assert kw == dict(first_bonus=4, max_tokens=17, sampler=None)
        try:
            yield 5, None
            yield 6, None
        finally:
            seen.append("closed")

    iterator = resume_chain(
        original,
        None,
        draft,
        target_cache,
        hidden,
        copy=copy,
        draft_cache=draft_cache,
        first_bonus=4,
        max_tokens=17,
        sampler=None,
    )
    assert next(iterator) == (5, None)
    iterator.close()
    assert seen == ["closed"]


def test_verifier_delegates_feed_forward_without_global_mutation(monkeypatch):
    from yunshu_engine.kernels import lane_virtual

    class Base:
        def _linears(self, members, x):
            return ("original", x)

        def _feed_forward(self, members, x):
            return self._linears(members, x)

    original = Base()
    method = original._linears.__func__
    proxy = fast.Verifier(original)
    monkeypatch.setattr(lane_virtual, "grouped_linears", lambda *_: ("virtual", 17))
    assert proxy._feed_forward((1, 2), 3) == ("virtual", 17)
    assert original._linears.__func__ is method
    monkeypatch.setattr(lane_virtual, "grouped_linears", lambda *_: None)
    assert proxy._feed_forward((1, 2), 3) == ("original", 3)


def test_cached_gate_record_commits_once_with_the_same_conv_path(monkeypatch):
    class Rows(tuple):
        def replay(self, layer, state, path, count):
            calls.append((layer, state, path.copy(), count.copy()))
            return np.full_like(state, 7)

    class Cache(list):
        pass

    monkeypatch.setattr(
        tv,
        "mx",
        SimpleNamespace(
            array=lambda v, **_: np.array(v),
            int32=np.int32,
            concatenate=np.concatenate,
            take=np.take,
        ),
    )
    calls = []
    state = np.ones((1, 1))
    conv = np.zeros((1, 3, 1))
    mixed = np.arange(3).reshape(1, 3, 1)
    rows = Rows((1, 2, 3, 4))
    layer = object()
    cache = [Cache([conv, state])]
    result = SimpleNamespace(
        shape=SimpleNamespace(width=3),
        n0=42,
        records={0: ("gdn", layer, state, conv, mixed, rows)},
    )
    tv.tree_commit(None, cache, result, [0, 2])
    assert len(calls) == 1 and calls[0][0] is layer and calls[0][1] is state
    assert calls[0][2].tolist() == [0, 2, 0] and calls[0][3].tolist() == [2]
    assert cache[0][1].tolist() == [[7.0]]
    assert cache[0][0].reshape(-1).tolist() == [0, 0, 2]
    assert cache[0]._omlx_gdn_pending is None


def test_prior_force_tree_install_delegates_after_mode_change(monkeypatch):
    from mlx_vlm.speculative import utils

    from yunshu_engine import dflash_copy, dflash_tree, settings

    mode = ["tree"]
    monkeypatch.setattr(settings, "get", lambda *_: mode[0])

    def original(*_, **__):
        yield 11, None

    original._yunshu_copy = True
    monkeypatch.setattr(utils, "_dflash_rounds", original)
    monkeypatch.setattr(
        dflash_tree, "dflash_tree_rounds", lambda *_, **__: iter([(17, None)])
    )
    assert dflash_tree.install()
    assert list(utils._dflash_rounds()) == [(17, None)]
    for selected in ("off", "auto"):
        mode[0] = selected
        assert list(utils._dflash_rounds()) == [(11, None)]
        assert dflash_copy.install()


def test_private_projection_bundle_does_not_keep_the_model_alive(monkeypatch):
    import gc
    import weakref

    import mlx.nn as nn

    from yunshu_engine.kernels import lane_linear

    class Quantized:
        pass

    class Converted:
        weight = object()
        sbt = object()

    class Parent:
        def __init__(self):
            self.child = Quantized()

        def named_modules(self):
            return [("", self)]

        def children(self):
            return {"child": self.child}

        def update_modules(self, values):
            self.__dict__.update(values)

    monkeypatch.setattr(nn, "QuantizedLinear", Quantized)
    monkeypatch.setattr(lane_linear, "eligible", lambda *_: True)
    monkeypatch.setattr(
        lane_linear.LaneLinear, "from_quantized", lambda *_: Converted()
    )
    monkeypatch.setattr(fast.mx, "eval", lambda *_: None)
    model = Parent()
    ref = weakref.ref(model)
    original = model.child
    bundle = fast.PrivateProjections(model)
    bundle.bind(True)
    assert isinstance(model.child, Converted)
    bundle.bind(False)
    assert model.child is original
    del model
    gc.collect()
    assert ref() is None
    bundle.bind(False)  # no parent retention or stale-model lookup


@pytest.mark.parametrize("head_error", [False, True])
def test_single_remaining_token_and_failure_restore_the_sum_scope(
    monkeypatch, head_error
):
    from yunshu_engine import dflash_plan
    from yunshu_engine.kernels import lane_linear

    bindings = []
    aborted = []
    commits = []
    monkeypatch.setattr(
        fast,
        "prepare",
        lambda *_: SimpleNamespace(bind=lambda enabled: bindings.append(enabled)),
    )
    monkeypatch.setattr(mtp_lane, "copy_rows_for_model", lambda *_: 0)
    monkeypatch.setitem(mtp_lane._STATE, "context", [1] * 1034)
    monkeypatch.setattr(lane_linear, "_SUM_REUSE", False)
    monkeypatch.setattr(
        fast,
        "mx",
        SimpleNamespace(
            array=lambda v, **_: np.array(v),
            int32=np.int32,
            concatenate=np.concatenate,
            async_eval=lambda *_: None,
        ),
    )
    monkeypatch.setattr(
        dflash_plan,
        "FastShape",
        lambda *_args, **_kw: SimpleNamespace(width=1, original_ranks=None),
    )

    def forward(*args, **kwargs):
        assert lane_linear.sum_reuse_enabled()
        return SimpleNamespace(hidden=np.ones((1, 1, 1)), captured=[np.ones((1, 1, 1))])

    monkeypatch.setattr(tv, "tree_forward", forward)
    monkeypatch.setattr(tv, "tree_commit", lambda *args: commits.append(args[-1]))
    monkeypatch.setattr(tv, "tree_abort", lambda *_: aborted.append(True))

    def head(*_):
        if head_error:
            raise RuntimeError("head failed")
        return np.array([17])

    model = SimpleNamespace(speculative_argmax_from_hidden=head)
    draft = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[0]), reset=lambda *_: []
    )
    iterator = fast.rounds(
        model, draft, [], np.ones((1, 1, 1)), first_bonus=11, max_tokens=2, sampler=None
    )
    if head_error:
        with pytest.raises(RuntimeError, match="head failed"):
            next(iterator)
        assert aborted == [True] and not commits
    else:
        assert next(iterator) == (17, None)
        iterator.close()
        assert commits == [[0]] and not aborted
    assert not lane_linear.sum_reuse_enabled()
    assert bindings[-1] is False
    assert True not in bindings  # no drafter when only one token remains


def test_live_boundary_hands_off_without_rebuilding_draft_kv(monkeypatch):
    from mlx_vlm.models.qwen3_5 import language as q35

    events = []
    private = SimpleNamespace(bind=lambda enabled: events.append(enabled))
    draft_cache = [SimpleNamespace(offset=7)]
    hidden = object()
    draft = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[]),
        accept_lens=[],
        reset=lambda model: draft_cache,
    )
    monkeypatch.setattr(fast, "prepare", lambda _: private)
    monkeypatch.setattr(fast, "context_length", lambda *_: fast.CONTEXT_LIMIT - 1)
    monkeypatch.setattr(mtp_lane, "copy_rows_for_model", lambda _: 16)
    monkeypatch.setitem(mtp_lane._STATE, "context", [1] * (fast.CONTEXT_LIMIT - 1))
    monkeypatch.setattr(q35, "_EXACT_SPECULATIVE_VERIFIER", object(), raising=False)

    def chain(model, proxy, cache, taps, **kw):
        assert proxy.reset(model) is draft_cache
        assert draft_cache[0].offset == 7
        assert taps is hidden
        assert proxy._copy.ctx[-1] == 9
        assert len(proxy._copy.ctx) == fast.CONTEXT_LIMIT
        assert kw["first_bonus"] == 9 and kw["max_tokens"] == 4096
        assert kw["draft_block_size"] == 8
        yield 10, None

    assert list(
        fast.rounds(
            None,
            draft,
            [],
            hidden,
            first_bonus=9,
            max_tokens=4096,
            sampler=None,
            draft_block_size=8,
            _chain=chain,
        )
    ) == [(10, None)]
    assert events == [False, False]


@pytest.mark.parametrize("remaining", [1, 2, 8, 16])
def test_fast_round_cannot_publish_past_live_crossover(remaining):
    current = fast.CONTEXT_LIMIT - remaining
    assert fast.round_room(current, 4096) == remaining
    assert fast.round_room(current, 1) == 1
    assert fast.round_room(fast.CONTEXT_LIMIT, 4096) == 0


def test_admission_reads_live_kv_instead_of_copy_history(monkeypatch, admission):
    monkeypatch.setattr(fast, "context_length", lambda *_: fast.CONTEXT_LIMIT)
    assert not fast.eligible(None, None, [], admission)
    monkeypatch.setitem(mtp_lane._STATE, "context", [1] * (fast.CONTEXT_LIMIT + 100))
    monkeypatch.setattr(fast, "context_length", lambda *_: 1024)
    assert fast.eligible(None, None, [], admission)


def test_supported_rejects_a_missing_decoder(monkeypatch):
    from yunshu_engine.kernels.tensorfold import lane_qmm

    monkeypatch.setattr(lane_qmm, "_resolve_variant", lambda: "m5")
    monkeypatch.setattr(tv, "supported", lambda *_: True)
    monkeypatch.setattr(tv, "lane_projections", lambda *_: True)
    draft = SimpleNamespace(
        candidate_selector=object(),
        config=SimpleNamespace(
            block_size=8, layer_types=["sliding_attention"], sliding_window=16
        ),
    )
    assert not fast.supported(None, draft)
