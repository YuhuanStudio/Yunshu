"""Request-local DFlash copy islands preserve cache catch-up and counters."""

from types import SimpleNamespace

import numpy as np
import pytest

from yunshu_engine import dflash_copy


def draft(window=5):
    calls = []

    def propose(bonus, hidden, cache, block, sampler, dtype):
        calls.append((hidden.copy(), [c.offset for c in cache]))
        return np.zeros((1, block - 1), dtype=np.int32)

    return SimpleNamespace(
        accept_lens=[],
        draft_lens=[],
        config=SimpleNamespace(
            layer_types=["sliding_attention"], sliding_window=window
        ),
        draft_block_greedy=propose,
        calls=calls,
    )


@pytest.fixture(autouse=True)
def cpu_arrays(monkeypatch):
    monkeypatch.setattr(
        dflash_copy,
        "mx",
        SimpleNamespace(
            concatenate=np.concatenate,
            array=lambda value, dtype=None: np.array(value, dtype=np.int32),
        ),
    )


def test_copy_uses_known_context_and_does_not_dispatch_drafter():
    raw = draft()
    proxy = dflash_copy.CopyDraft(
        raw, [10, 11, 12, 13, 14, 15, 16, 17, 10, 11, 12, 13, 14], 15, 16
    )
    assert proxy.choose(8, 100) == 5  # first-width proposal is four tokens
    hidden = np.arange(8).reshape(1, 4, 2)
    assert proxy.draft_block_greedy(15, hidden, [], 5, None).tolist() == [
        [16, 17, 10, 11]
    ]
    assert not raw.calls
    assert proxy._copy.ctx[-1] == 15


def test_bounded_pending_taps_advance_offsets_once_at_catch_up():
    raw = draft(window=5)
    proxy = dflash_copy.CopyDraft(raw, [], 1, 16)
    cache = [SimpleNamespace(offset=9), SimpleNamespace(offset=9)]
    first = np.arange(12).reshape(1, 6, 2)
    second = np.arange(12, 18).reshape(1, 3, 2)
    proxy._next_copy = [5, 6]
    proxy.draft_block_greedy(1, first, cache, 3, None)
    proxy.draft_block_greedy(1, second, cache, 3, None)
    assert proxy._pending_hidden.shape[1] == 4
    assert proxy._pending_skipped == 5
    assert [c.offset for c in cache] == [9, 9]
    proxy._next_copy = []
    third = np.arange(18, 22).reshape(1, 2, 2)
    proxy.draft_block_greedy(1, third, cache, 8, None)
    np.testing.assert_array_equal(
        raw.calls[0][0], np.concatenate([first, second, third], axis=1)[:, -6:]
    )
    assert raw.calls[0][1] == [14, 14]
    assert proxy._pending_hidden is None and proxy._pending_skipped == 0
    proxy.draft_block_greedy(1, third, cache, 8, None)
    assert raw.calls[1][1] == [14, 14]


def test_copy_accounting_counts_published_budget_not_last_bonus():
    raw = draft()
    proxy = dflash_copy.CopyDraft(raw, [], 1, 16)
    proxy._previous = (True, 4)
    raw.accept_lens.append(4)
    for token in (2, 3):
        proxy.emitted(token)
    assert raw.copy_total_rounds == 1
    assert raw.copy_total_tokens == 2
    proxy.observe()
    proxy.observe()
    assert raw.copy_total_rounds == 1
    assert raw.copy_total_tokens == 2
    assert proxy._copy.ctx == [1, 2, 3]


def test_proxy_forwards_lifetime_counters_and_context_is_request_local():
    raw = draft()
    a = dflash_copy.CopyDraft(raw, [3, 4], 5, 16)
    b = dflash_copy.CopyDraft(raw, [6, 7], 8, 16)
    a.speculative_total_rounds = 12
    assert raw.speculative_total_rounds == 12
    a.emitted(9)
    assert b._copy.ctx == [6, 7, 8]
    a._pending_hidden = np.zeros((1, 4, 2))
    a._pending_skipped = 10
    a.close()
    assert a._pending_hidden is None and a._pending_skipped == 0


def test_copy_observation_does_not_repeat_previous_round():
    raw = draft()
    proxy = dflash_copy.CopyDraft(raw, [], 1, 16)
    proxy.choose(8, 100)
    raw.accept_lens.append(3)
    proxy.emitted(2)
    proxy.emitted(3)
    proxy.emitted(4)
    proxy.emitted(5)
    proxy.choose(8, 100)
    expected = proxy._copy._model_tpr
    proxy.choose(8, 100)
    assert proxy._copy._model_tpr == expected


def install_fixture(monkeypatch):
    from mlx_vlm.speculative import dflash, utils

    from yunshu_engine import mtp_lane, tree_verify

    received = []
    closed = []

    def original(model, dm, cache, hidden, **kw):
        received.append(dm)
        if isinstance(dm, dflash_copy.CopyDraft):
            dflash._dflash_next_block_size(dm, 8, 100)
        try:
            dm.accept_lens.append(1)
            yield 2, None
            yield 3, None
        finally:
            closed.append(True)

    monkeypatch.setattr(utils, "_dflash_rounds", original)
    monkeypatch.setattr(
        dflash, "_dflash_next_block_size", lambda dm, b, r, i=None: min(b, r)
    )
    monkeypatch.setitem(mtp_lane._STATE, "context", [11, 12, 13])
    monkeypatch.setitem(mtp_lane._STATE, "guide", None)
    monkeypatch.setattr(mtp_lane, "copy_rows_for_model", lambda lm: 16)
    monkeypatch.setattr(mtp_lane, "verify_max_rows", lambda *args: 32)
    monkeypatch.setattr(tree_verify, "supported", lambda lm: True)
    monkeypatch.setattr(tree_verify, "lane_ready", lambda lm, cache: True)
    monkeypatch.setattr(tree_verify, "lane_projections", lambda lm: True)
    raw = draft()
    raw.config.block_size = 8
    raw.candidate_selector = object()
    dflash_copy.install()
    return utils, raw, received, closed


def test_installer_uses_proxy_without_mutating_draft_methods(monkeypatch):
    utils, raw, received, closed = install_fixture(monkeypatch)
    fn = raw.draft_block_greedy
    assert list(
        utils._dflash_rounds(None, raw, [], None, first_bonus=1, greedy_sampling=True)
    ) == [(2, None), (3, None)]
    assert isinstance(received[0], dflash_copy.CopyDraft)
    assert received[0]._copy.ctx == [11, 12, 13, 1, 2, 3]
    assert raw.draft_block_greedy is fn
    assert closed == [True]
    installed = utils._dflash_rounds
    dflash_copy.install()
    assert utils._dflash_rounds is installed


@pytest.mark.parametrize(
    "case",
    [
        "sampled",
        "no_context",
        "guide",
        "narrow",
        "unsupported",
        "prepared",
        "no_window",
        "oversized",
    ],
)
def test_unsupported_requests_keep_original_loop(monkeypatch, case):
    from yunshu_engine import mtp_lane, tree_verify

    utils, raw, received, _ = install_fixture(monkeypatch)
    kwargs = dict(first_bonus=1, greedy_sampling=True)
    if case == "sampled":
        kwargs["greedy_sampling"] = False
    elif case == "no_context":
        monkeypatch.setitem(mtp_lane._STATE, "context", None)
    elif case == "guide":
        monkeypatch.setitem(mtp_lane._STATE, "guide", object())
    elif case == "narrow":
        monkeypatch.setattr(mtp_lane, "copy_rows_for_model", lambda lm: 0)
    elif case == "unsupported":
        monkeypatch.setattr(tree_verify, "supported", lambda lm: False)
    elif case == "prepared":
        raw.prepare_target_hidden = lambda h: h
    elif case == "no_window":
        raw.config.layer_types = ["full_attention"]
    elif case == "oversized":
        kwargs["draft_block_size"] = 64
    list(utils._dflash_rounds(None, raw, [], None, **kwargs))
    assert received == [raw]


def test_cancel_closes_upstream_and_drops_pending_taps(monkeypatch):
    utils, raw, received, closed = install_fixture(monkeypatch)
    iterator = utils._dflash_rounds(
        None, raw, [], None, first_bonus=1, greedy_sampling=True
    )
    assert next(iterator) == (2, None)
    proxy = received[0]
    proxy._pending_hidden = np.zeros((1, 4, 2))
    proxy._pending_skipped = 10
    iterator.close()
    assert closed == [True]
    assert proxy._pending_hidden is None and proxy._pending_skipped == 0
    assert proxy._copy.ctx[-1] == 2


def test_engine_reload_keeps_copy_choice_hook(monkeypatch):
    from mlx_vlm.speculative import dflash

    from yunshu_engine.spec_schedule import install_chain_budget

    install_fixture(monkeypatch)
    chooser = dflash._dflash_next_block_size
    install_chain_budget()
    assert dflash._dflash_next_block_size is chooser
    dflash_copy.install()
    assert dflash._dflash_next_block_size is chooser


def test_existing_tree_loop_is_not_wrapped_twice(monkeypatch):
    from mlx_vlm.speculative import utils

    def tree(*args, **kwargs):
        return iter(())

    tree._yunshu_tree = True
    monkeypatch.setattr(utils, "_dflash_rounds", tree)
    assert dflash_copy.install() is False
    assert utils._dflash_rounds is tree
