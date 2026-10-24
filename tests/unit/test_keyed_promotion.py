"""Promotion keeps absolute keyed positions for verification and draft coupling."""

from itertools import product

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine.keyed_sampling import KeyedSampler, coupled_scores, seed_base
from yunshu_engine.vlm_batch_runner import RowParams, RowSampler


@pytest.mark.parametrize(
    "temperature,top_p,top_k,min_p",
    list(product((0.7, 1.0), (1.0, 0.95), (0, 20), (0.0, 0.1))),
)
def test_promoted_local_positions_equal_absolute_serial_draws(
    temperature, top_p, top_k, min_p
):
    params = RowParams(temperature, top_p, top_k, min_p)
    logits = mx.array(np.random.default_rng(44).normal(size=(5, 32)).astype("float32"))
    probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    local = mx.array([1, 2, 2, 3, 4])  # siblings share a generation index
    for seed in range(12):
        serial = KeyedSampler(params, seed)
        promoted = KeyedSampler(params, seed)
        promoted.position_base = 40
        expected = serial.sample_positions(probs, local + 40)
        assert promoted.sample_positions(probs, local).tolist() == expected.tolist()
        # Proposal rankings must use the same absolute positions as verification.
        assert (
            coupled_scores(probs, promoted, local).tolist()
            == coupled_scores(probs, serial, local + 40).tolist()
        )
        assert promoted._next == 0


@pytest.mark.parametrize("seed", [None, 17])
def test_batch_and_lane_share_the_seed_for_an_unseeded_eligible_request(seed):
    params = RowParams(0.7, 0.95, 20, 0.0, seed)
    lane = KeyedSampler(params, seed if seed is not None else 829107)
    batch = RowSampler()
    batch.add(7, params, keyed=lane)
    entry = batch._rows[7]
    assert entry[2] == seed_base(lane.seed)
    assert params.seed == seed  # request metadata is not rewritten


def test_default_position_base_keeps_the_existing_stream():
    params = RowParams(0.7, 1.0, 0, 0.0)
    row = mx.array([[-1.0, -2.0, -3.0]])
    sampler = KeyedSampler(params, 17)
    assert sampler.position_base == 0
    assert sampler(row).tolist() == sampler.sample_positions(row, [0]).tolist()


def test_virtual_chunking_applies_the_position_base_exactly_once():
    params = RowParams(0.7, 0.95, 20, 0.0)
    sampler = KeyedSampler(params, 17)
    sampler.position_base = 129
    serial = KeyedSampler(params, 17)
    rows = mx.array(np.random.default_rng(12).normal(size=(65, 32)).astype("float32"))
    local = mx.arange(65) + 1
    assert sampler._sample_positions_virtual(rows, local).tolist() == (
        serial.sample_positions(rows, local + 129).tolist()
    )
