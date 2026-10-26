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
        "empty_kv",
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
        monkeypatch.setitem(mtp_lane._STATE, "context", [1] * (fast.CONTEXT_LIMIT + 10))
    elif damage == "empty_kv":
        monkeypatch.setattr(fast, "context_length", lambda *_: -1)
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


@pytest.mark.parametrize("log_level", ["WARNING", "INFO"])
@pytest.mark.parametrize("head_error", [False, True])
def test_single_remaining_token_and_failure_restore_the_sum_scope(
    monkeypatch, caplog, head_error, log_level
):
    # Unknown target KV length must be loggable, including with INFO enabled by
    # an earlier engine/server setup in the full suite.
    caplog.set_level(log_level, logger=fast.logger.name)
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
    assert (
        bindings == []
    )  # the fast tree reads a private view; nothing is bound or unbound


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
    assert events == []


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


def test_round_room_stops_at_generated_limit(monkeypatch):
    monkeypatch.setattr(fast, "GENERATED_LIMIT", 256)
    assert fast.round_room(1000, 4096, 0) == fast.GENERATED_LIMIT
    assert fast.round_room(1000, 4096, fast.GENERATED_LIMIT - 3) == 3
    assert fast.round_room(1000, 4096, fast.GENERATED_LIMIT) == 0
    assert fast.round_room(1000, 2, 0) == 2


def test_generated_limit_hands_the_same_request_to_chain(monkeypatch):
    from mlx_vlm.models.qwen3_5 import language as q35

    private = SimpleNamespace(bind=lambda enabled: None)
    draft_cache = [SimpleNamespace(offset=3)]
    draft = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[]),
        accept_lens=[],
        reset=lambda model: draft_cache,
    )
    monkeypatch.setattr(fast, "prepare", lambda _: private)
    monkeypatch.setattr(fast, "context_length", lambda *_: 1000)
    monkeypatch.setattr(fast, "GENERATED_LIMIT", 1)
    monkeypatch.setattr(mtp_lane, "copy_rows_for_model", lambda _: 16)
    monkeypatch.setitem(mtp_lane._STATE, "context", [1] * 1000)
    monkeypatch.setattr(q35, "_EXACT_SPECULATIVE_VERIFIER", object(), raising=False)
    seen = {}

    def chain(model, proxy, cache, taps, **kw):
        seen.update(kw)
        yield 5, None

    out = list(
        fast.rounds(
            None,
            draft,
            [],
            object(),
            first_bonus=9,
            max_tokens=4096,
            sampler=None,
            draft_block_size=8,
            _chain=chain,
        )
    )
    assert (
        out == [(5, None)] and seen["first_bonus"] == 9 and seen["max_tokens"] == 4096
    )


def test_gathered_prework_reads_the_window_in_kernel():
    source = fast.prework_source()
    assert "conv_idx[" in source and "conv_prev[" in source
    assert "conv_state" not in source and "conv_out" not in source
    # the conv taps, sums and activation lines are the omlx kernel's own
    from yunshu_engine.kernels.omlx import qwen35_gdn_prework as gp

    for line in (
        "acc += float(xv) * float(conv_w[channel * 4 + tap]);",
        "const T act = conv * ((conv < T(0)) ? sy : T(1) - sy);",
    ):
        assert line in gp._SOURCE and line in source


def test_gathered_prework_carries_the_rendered_qk_epsilon():
    # the raw omlx source holds a marker Metal cannot compile; the 27B server's tree verify failed on it
    from yunshu_engine.kernels import gdn_qk_eps

    source = fast.prework_source()
    assert gdn_qk_eps.MARKER not in source
    assert gdn_qk_eps.expression("float(DK)") in source


def test_drafter_bits_setting_defaults_to_the_measured_eight_bits(monkeypatch):
    from yunshu_engine import settings

    monkeypatch.delenv("YUNSHU_DRAFT_BITS", raising=False)
    assert settings.get("YUNSHU_DRAFT_BITS") == 8
    monkeypatch.setenv("YUNSHU_DRAFT_BITS", "4")
    assert settings.get("YUNSHU_DRAFT_BITS") == 4
    monkeypatch.setenv("YUNSHU_DRAFT_BITS", "0")
    assert settings.get("YUNSHU_DRAFT_BITS") == 0


def test_copy_rounds_do_not_feed_the_node_budget():
    from yunshu_engine import dflash_fast
    from yunshu_engine.spec_schedule import NodeBudget

    budget = NodeBudget(15)
    assert dflash_fast.observe_budget(budget, [7, 8, 9], 3, [0, 1, 2], 41.0) is False
    assert budget.rounds == 0 and budget.samples == {}
    assert dflash_fast.observe_budget(budget, [], 4, [0], 52.0) is True
    assert budget.rounds == 1 and budget.samples[4] == [52.0]


def test_private_view_swaps_in_lane_projections_without_touching_the_shared_drafter():
    import mlx.nn as nn

    class Quant(nn.Module):
        pass

    class Lane(nn.Module):
        pass

    class Attn(nn.Module):
        def __init__(self, q):
            super().__init__()
            self.q_proj = q
            self.scale = 3

    class Layer(nn.Module):
        def __init__(self, q):
            super().__init__()
            self.self_attn = Attn(q)

    q = Quant()
    lane = Lane()
    other = Quant()  # not eligible: a plain wrapper copy, same type
    layer = Layer(q)
    layer.self_attn.k_proj = other
    draft = nn.Module()
    draft.layers = [layer]
    draft.embed_tokens = "embed-1"
    bundle = fast.PrivateProjections.__new__(fast.PrivateProjections)
    bundle.bindings = [(None, "q_proj", q, lane)]
    view = bundle.view_of(draft)
    assert view.layers[0].self_attn.q_proj is lane
    assert (
        type(view.layers[0].self_attn.k_proj) is Quant
        and view.layers[0].self_attn.scale == 3
    )
    assert (
        layer.self_attn.q_proj is q and draft.layers[0] is layer
    )  # shared drafter untouched
    assert view is not draft and view.layers[0] is not layer
    draft.embed_tokens = "embed-2"  # a later bind() to a (re)loaded target
    assert bundle.view_of(draft) is view and view.embed_tokens == "embed-2"


def test_packed_readback_returns_the_same_four_lists():
    import mlx.core as mlx

    target = mlx.array([[5, 6, 7, 248000]], dtype=mlx.uint32)
    window = mlx.array([[1, 2, 3, 4]], dtype=mlx.int32)
    parents = mlx.array([-1, 0, 0, 1], dtype=mlx.int32)
    ranks = mlx.array([0, 2, 1, 3], dtype=mlx.int32)
    for r in (ranks, None):
        packed = fast.pack_readback(target, window, parents, r)
        t, w, p, rk = fast.unpack_readback(
            packed.tolist(), target.size, window.size, parents.size, r is not None
        )
        assert t == target.reshape(-1).tolist() and w == window.reshape(-1).tolist()
        assert p == parents.tolist() and rk == (None if r is None else r.tolist())


def test_graft_width_keeps_default_fast_dflash_route(monkeypatch, admission):
    monkeypatch.setattr(mtp_lane, "copy_rows_for_model", lambda *_: 32)
    assert fast.eligible(None, None, [], admission)


def _fake_rounds(monkeypatch, nodes):
    """rounds() on fakes: ``nodes`` lists the node count chosen for each round (0 = plain row,
    >0 = model tree). Returns the event log and a factory for the round iterator."""
    from yunshu_engine import dflash_plan, dflash_tree, spec_metrics, spec_schedule
    from yunshu_engine.kernels import lane_linear

    events = []
    chosen = iter(nodes)
    monkeypatch.setattr(spec_metrics, "record", lambda *_a, **_k: None)
    monkeypatch.setattr(fast, "prepare", lambda *_: SimpleNamespace())
    monkeypatch.setattr(mtp_lane, "copy_rows_for_model", lambda *_: 0)
    monkeypatch.setitem(mtp_lane._STATE, "context", [1] * 1034)
    monkeypatch.setattr(lane_linear, "_SUM_REUSE", False)
    monkeypatch.setattr(fast, "observe_budget", lambda *_a, **_k: False)

    class Budget:
        p, samples = [], {}

        def __init__(self, *_a, **_k):
            pass

        def choose(self, _room):
            return next(chosen)

        def _median(self, v):
            return 0

    monkeypatch.setattr(spec_schedule, "NodeBudget", Budget)
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

    def lattice(*_a, **_k):
        events.append("draft")

    monkeypatch.setattr(dflash_tree, "compute_lattice_gpu", lattice)
    monkeypatch.setattr(
        dflash_plan,
        "search_tree",
        lambda _l, n: (np.arange(n, dtype=np.int32), np.arange(-1, n, dtype=np.int32)),
    )
    monkeypatch.setattr(dflash_plan, "reorder", lambda t, p: (t, p, None))
    monkeypatch.setattr(dflash_tree, "walk", lambda *_: [0])
    monkeypatch.setattr(dflash_tree, "PRIOR", None, raising=False)

    def forward(*args, **kwargs):
        events.append("verify")
        return SimpleNamespace(
            hidden=np.ones((1, 1, 1)), captured=[np.ones((1, 1, 1))], n0=0
        )

    monkeypatch.setattr(tv, "tree_forward", forward)
    monkeypatch.setattr(tv, "tree_commit", lambda *a: events.append("commit"))
    monkeypatch.setattr(tv, "tree_abort", lambda *_: events.append("abort"))
    made = iter(range(20, 40))
    model = SimpleNamespace(
        speculative_argmax_from_hidden=lambda *_: np.array([next(made)])
    )
    draft = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[0]), reset=lambda *_: []
    )

    def start(max_tokens):
        return fast.rounds(
            model,
            draft,
            [],
            np.ones((1, 1, 1)),
            first_bonus=11,
            max_tokens=max_tokens,
            sampler=None,
        )

    return events, start


@pytest.mark.parametrize("stop_early", [False, True])
def test_round_tokens_reach_the_consumer_after_the_next_verify_is_submitted(
    monkeypatch, stop_early
):
    # The consumer's host time overlaps the next round's verify; a consumer that stops
    # while a window is in flight drops that window (it is never committed).
    from yunshu_engine.kernels import lane_linear

    events, start = _fake_rounds(monkeypatch, [0, 0, 0, 0])
    iterator = start(5)
    # the reply's first round is not held (its first visible text is the TTFT)
    assert next(iterator) == (20, None)
    assert events == ["verify"]
    # later rounds reach the consumer once the next verify is on the GPU
    assert next(iterator) == (21, None)
    assert events == ["verify", "commit", "verify", "commit", "verify"]
    assert (
        not lane_linear.sum_reuse_enabled()
    )  # consumer runs outside the round's scope
    if stop_early:
        iterator.close()
        assert events == ["verify", "commit", "verify", "commit", "verify", "abort"]
    else:
        assert list(iterator) == [(22, None), (23, None)]
        assert events == ["verify", "commit"] * 4
    assert not lane_linear.sum_reuse_enabled()


def test_commit_follows_the_next_draft_and_precedes_its_verify(monkeypatch):
    events, start = _fake_rounds(monkeypatch, [0, 2, 2])
    assert list(start(4)) == [(20, None), (21, None), (22, None)]
    assert events == [
        "verify",
        "draft",
        "commit",
        "verify",
        "draft",
        "commit",
        "verify",
        "commit",  # the last round, before the loop ends
    ]


def test_debug_log_traces_every_round(monkeypatch, caplog):
    import logging

    events, start = _fake_rounds(monkeypatch, [0, 2, 2])
    with caplog.at_level(logging.DEBUG, logger="yunshu_engine.dflash_fast"):
        list(start(4))
    lines = [
        r.getMessage() for r in caplog.records if "DFlash round:" in r.getMessage()
    ]
    assert len(lines) == 3
    assert all("kind=" in line and "accepted=" in line for line in lines)


@pytest.mark.parametrize("context", [1, 40, 85, 511, 512])
def test_subtile_context_uses_fast_tree(monkeypatch, admission, context):
    # Kapitsa greedy code starts at 85 tokens: the old 512 floor silently used
    # the chain for the whole request, even after it grew past the threshold.
    monkeypatch.setitem(mtp_lane._STATE, "context", [1] * context)
    assert fast.eligible(None, None, [], admission)
    assert fast.live_eligible(context)
    assert not fast.live_eligible(0)
