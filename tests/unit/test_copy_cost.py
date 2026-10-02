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
