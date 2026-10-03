"""A reachable ordinary-LM draft must use exact greedy acceptance."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from yunshu_engine.speculative_decoder import DraftResult, SpeculativeDecoder


@pytest.mark.parametrize("draft_lp", [-20.0, -0.001])
def test_greedy_rejects_non_argmax_even_when_ratio_would_accept(draft_lp):
    with mx.stream(mx.cpu):

        def target(ids, cache=None):
            return mx.array([[[0.0, 10.0, 0.0], [0.0, 10.0, 0.0]]])

        decoder = SpeculativeDecoder(target, object(), SimpleNamespace(eos_token_id=99))
        decoder.rng = SimpleNamespace(random=lambda: 0.0)
        result = decoder.verify_draft(
            DraftResult([2], [draft_lp]), mx.array([[0]]), cache=[], temperature=0.0
        )
        assert result.accepted_ids == []
        assert result.bonus_token_id == 1


def test_greedy_accepts_argmax_regardless_of_draft_log_probability():
    with mx.stream(mx.cpu):

        def target(ids, cache=None):
            return mx.array([[[0.0, 10.0, 0.0], [0.0, 10.0, 0.0]]])

        decoder = SpeculativeDecoder(target, object(), SimpleNamespace(eos_token_id=99))
        decoder.rng = SimpleNamespace(random=lambda: 0.99999999)
        result = decoder.verify_draft(
            DraftResult([1], [0.0]), mx.array([[0]]), cache=[], temperature=0.0
        )
        assert result.accepted_ids == [1]
        assert result.bonus_token_id == 1
