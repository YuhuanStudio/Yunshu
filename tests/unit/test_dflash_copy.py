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


def test_sampled_copy_uses_proposal_and_catches_up_with_sampled_method():
    raw = draft()
    raw.draft_block = raw.draft_block_greedy
    raw.draft_block_greedy = lambda *args: pytest.fail("sampled fallback used greedy")
    proxy = dflash_copy.CopyDraft(raw, [], 1, 16)
    proxy._next_copy = [5, 6]
    hidden = np.arange(8).reshape(1, 4, 2)
    assert proxy.draft_block(1, hidden, [], 3, None).tolist() == [[5, 6]]
    assert not raw.calls
    proxy._next_copy = []
    proxy.draft_block(6, hidden, [], 8, None)
    assert len(raw.calls) == 1


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


@pytest.mark.parametrize(
    "rows,invariant,lanes,expected,enabled",
    [
        (0, True, True, 0, False),
        (16, True, True, 16, True),
        (100, True, True, 32, True),
        (16, False, False, 8, False),
        (16, True, False, 8, False),
    ],
)
def test_configuration_honors_copy_setting_and_certified_backend(
    monkeypatch, rows, invariant, lanes, expected, enabled
):
    from yunshu_engine import mtp_lane

    monkeypatch.setitem(mtp_lane._STATE, "copy_rows", 16)
    monkeypatch.setattr(mtp_lane, "verify_max_rows", lambda wide, lm: 32 if wide else 8)
    assert dflash_copy.configure(
        object(), rows, invariant=invariant, lane_projections=lanes
    ) == (expected, enabled)
    assert mtp_lane._STATE["copy_rows"] == expected


def test_configuration_is_off_on_eight_row_geometry(monkeypatch):
    from yunshu_engine import mtp_lane

    monkeypatch.setitem(mtp_lane._STATE, "copy_rows", 16)
    monkeypatch.setattr(mtp_lane, "verify_max_rows", lambda *args: 8)
    assert dflash_copy.configure(
        object(), 16, invariant=True, lane_projections=True
    ) == (8, False)
