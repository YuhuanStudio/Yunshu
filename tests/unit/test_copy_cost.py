import math

import pytest

from yunshu_engine.copy_cost import CopyCosts
from yunshu_engine.copy_drafter import CopyDrafter


def test_cost_cliff_can_make_a_shorter_copy_faster():
    costs = CopyCosts({8: 42, 16: 44, 24: 90, 32: 150}, model_ms=43)
    assert costs.choose(31, model_tpr=3) == 15
    assert costs.cost(17) == 90
    assert math.isinf(costs.cost(33))


def test_whole_round_rate_can_prefer_copy_with_fewer_tokens():
    costs = CopyCosts({8: 40}, model_ms=80)
    for _ in range(3):
        costs.observe_model(4, 80)
        costs.observe_copy(7, 2, 40)
    assert costs.choose(7, model_tpr=4) == 7


def test_whole_round_rate_rejects_a_slow_high_yield_copy():
    costs = CopyCosts({8: 40}, model_ms=40)
    for _ in range(3):
        costs.observe_model(3, 40)
        costs.observe_copy(7, 4, 100)
    assert costs.choose(7, model_tpr=3) == 0


def test_truncated_copy_does_not_label_unobserved_tail_as_rejected():
    costs = CopyCosts({4: 40, 8: 42}, model_ms=43)
    for _ in range(3):
        costs.observe_copy(3, 3, 40)
    assert costs.choose(7, model_tpr=3) == 7


def test_invalid_costs_are_not_learned_and_history_is_bounded():
    costs = CopyCosts({8: 42}, model_ms=43)
    for _ in range(100):
        costs.observe_model(3, 43)
        costs.observe_copy(7, 7, 42)
    costs.observe_model(1, float("nan"))
    costs.observe_copy(7, 0, -1)
    assert len(costs.model_samples) == len(costs.copy_samples[8]) == 32
    with pytest.raises(ValueError):
        costs.observe_copy(7, 8, 42)
    with pytest.raises(ValueError):
        CopyCosts({8: 0}, model_ms=43)


def test_copy_drafter_prices_even_a_confident_match():
    costs = CopyCosts({8: 40, 16: 44, 24: 90, 32: 150}, model_ms=43)
    source = list(range(100, 180))
    draft = CopyDrafter(max_draft=31, costs=costs)
    draft.extend(source + [999] + source[:40])
    assert draft.draft() == source[40:55]
    for _ in range(3):
        draft.observe_copy(15, 0, 100)
        draft.observe_model(4, 40)
    assert costs.choose(15, draft._model_tpr, confident=True) == 0


def test_short_match_misses_do_not_label_a_new_confident_island():
    costs = CopyCosts({8: 42, 16: 44}, model_ms=43)
    for _ in range(3):
        costs.observe_copy(7, 0, 42)
    assert costs.choose(7, model_tpr=3) == 0
    assert costs.choose(15, model_tpr=3, confident=True) == 15


def test_context_table_prices_short_and_long_context_differently():
    from yunshu_engine.copy_cost import MAX_PRICED_ROWS, costs_for_context

    short, long_ = costs_for_context(900), costs_for_context(40_000)
    assert short.cost(16) < short.cost(8) * 1.05  # flat to 16 rows
    assert long_.cost(16) > long_.cost(8) * 1.15  # attention term at 32K+
    assert max(short.row_ms) == MAX_PRICED_ROWS
    # between buckets a request is priced by the next larger context
    assert costs_for_context(5000).row_ms == costs_for_context(8192).row_ms


def test_long_context_prices_extra_width_out_of_a_mediocre_copy():
    from yunshu_engine.copy_cost import costs_for_context

    for ctx, expect in ((1000, 11), (50_000, 7)):
        costs = costs_for_context(ctx)
        for _ in range(4):  # every copy accepts exactly eight drafts
            costs.observe_copy(15, 8, costs.cost(16))
        assert costs.choose(15, model_tpr=3) == expect


def test_copy_cost_setting_and_lane_state():
    from yunshu_engine import mtp_lane, settings

    assert settings.get("YUNSHU_SPEC_COPY_COST") is False
    assert mtp_lane.set_copy_cost(True) is True
    assert mtp_lane.set_copy_cost(False) is False


def test_mtp_cost_prior_does_not_charge_the_dflash_drafter():
    from yunshu_engine.copy_cost import costs_for_context

    costs = costs_for_context(1024)
    assert costs.model_ms == pytest.approx(costs.row_ms[6] + 2.7)


@pytest.mark.parametrize("requested", [0, 2, 3, 8, 12, 16, 32])
def test_cost_policy_never_raises_the_requested_copy_width(monkeypatch, requested):
    from yunshu_engine import mtp_lane

    monkeypatch.setitem(mtp_lane._STATE, "copy_rows", requested)
    monkeypatch.setitem(mtp_lane._STATE, "copy_cost", True)
    # Unknown backend additionally enforces its conservative eight-row cap.
    assert mtp_lane.copy_rows_for_request(object()) == min(requested, 8)


def test_cost_policy_caps_a_wide_backend_without_overriding_the_request(monkeypatch):
    from yunshu_engine import mtp_lane

    monkeypatch.setitem(mtp_lane._STATE, "copy_cost", True)
    monkeypatch.setattr(mtp_lane, "copy_rows_for_model", lambda model: 32)
    assert mtp_lane.copy_rows_for_request(object()) == 16
    monkeypatch.setattr(mtp_lane, "copy_rows_for_model", lambda model: 12)
    assert mtp_lane.copy_rows_for_request(object()) == 12


def test_consumer_stalls_cannot_teach_a_slow_model_or_copy_price():
    from yunshu_engine.copy_cost import RoundCostClock

    clock = RoundCostClock()
    assert clock.readback(1.0) is None
    clock.published(0.1)
    assert clock.readback(1.05) == pytest.approx(50.0)
    clock.published(5000.0)
    # Do not subtract the stall: speculative GPU work ran during it.
    assert clock.readback(6.06) is None
    clock.published(0.1)
    assert clock.readback(6.11) == pytest.approx(50.0)


def test_publication_stalls_accumulate_across_the_whole_token_window():
    from yunshu_engine.copy_cost import RoundCostClock

    clock = RoundCostClock()
    clock.readback(0.0)
    for _ in range(6):
        clock.published(0.2)
    assert clock.readback(0.05) is None
