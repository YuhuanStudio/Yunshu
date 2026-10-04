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
        "budget",
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
        monkeypatch.setitem(mtp_lane._STATE, "context", list(range(8203)))
    elif damage == "budget":
        admission["max_tokens"] = 512
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


def test_total_context_bound_covers_the_entire_request(monkeypatch, admission):
    monkeypatch.setitem(mtp_lane._STATE, "context", [1] * 1280)
    assert fast.eligible(None, None, [], admission)
    monkeypatch.setitem(mtp_lane._STATE, "context", [1] * 1281)
    assert not fast.eligible(None, None, [], admission)


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
