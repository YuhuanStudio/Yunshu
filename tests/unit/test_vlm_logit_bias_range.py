"""VLM-path logit_bias ignores ids outside the vocabulary (OpenAI; the text path does too).

mlx-lm's closure scattered every key: -1 wrapped to the last vocabulary entry."""

import mlx.core as mx
import pytest

from yunshu_engine.vlm_batch_runner import (
    LogitBiasProcessor,
    build_penalty_processors,
    spec_safe_processors,
)


def test_out_of_range_ids_are_ignored():
    procs = build_penalty_processors(
        1.0, 0.0, 0.0, {-1: 5.0, 100: 5.0, 3: 1.5, 0: -2.0}
    )
    out = procs[0](None, mx.zeros((2, 10)))
    want = [0.0] * 10
    want[3], want[0] = 1.5, -2.0
    assert out.tolist() == [want, want]


def test_mlx_lm_reference_wraps_negative_ids():
    """The bug this replaces: pins that stock mlx-lm biases the wrong token for id -1."""
    from mlx_lm.sample_utils import make_logits_processors

    stock = make_logits_processors(logit_bias={-1: 5.0})[0]
    assert stock(None, mx.zeros((1, 10)))[0, 9].item() == 5.0


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_in_range_ids_are_bit_identical_to_stock(dtype):
    from mlx_lm.sample_utils import make_logits_processors

    bias = {7: 2.5, 31: -100.0, 2: 0.1}
    stock = make_logits_processors(logit_bias=bias)[0]
    mine = LogitBiasProcessor(bias)
    mx.random.seed(4)
    logits = mx.random.normal((3, 64)).astype(dtype)
    assert mx.array_equal(stock(None, logits), mine(None, logits)).item()
    # a different vocabulary width rebuilds the scatter
    logits = mx.random.normal((1, 20)).astype(dtype)
    assert mx.array_equal(stock(None, logits), mine(None, logits)).item()


def test_bias_stays_spec_safe_and_first():
    procs = build_penalty_processors(1.2, 0.5, 0.5, {1: 1.0})
    assert isinstance(procs[0], LogitBiasProcessor)
    assert spec_safe_processors(procs)
