"""TokenMaskProcessor builds its suppress / EOS block masks once, not per decoded token.

The cached masks must give logits bit-identical to the old per-token rebuild, in float32
and bfloat16, through the min_tokens boundary, ignore_eos and top-nsigma.
"""

import mlx.core as mx
import pytest

from yunshu_engine.vlm_batch_runner import TokenMaskProcessor


def _reference(proc, logits, generated):
    """The pre-cache arithmetic: zeros + scatter on every call."""
    neg = mx.array(float("-inf"), logits.dtype)
    vocab = logits.shape[-1]

    def block(ids):
        return mx.zeros((vocab,), dtype=mx.bool_).at[ids].add(True)

    if proc._suppress.size:
        logits = mx.where(block(proc._suppress), neg, logits)
    if proc._eos.size and (proc._ignore_eos or generated < proc._min_tokens):
        logits = mx.where(block(proc._eos), neg, logits)
    if proc._n_sigma > 0:
        top = mx.max(logits, axis=-1, keepdims=True)
        finite = mx.where(mx.isinf(logits), top, logits)
        sigma = mx.sqrt(mx.var(finite, axis=-1, keepdims=True))
        logits = mx.where(logits < top - proc._n_sigma * sigma, neg, logits)
    return logits


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_cached_masks_are_bit_identical(dtype):
    mx.random.seed(3)
    proc = TokenMaskProcessor(
        suppress=[3, 7, 11], eos_ids=[2, 5], min_tokens=4, top_n_sigma=1.5
    )
    for step in range(8):
        logits = mx.random.normal((2, 64)).astype(dtype)
        got = proc.process_last_token(1, logits) if step else proc(None, logits)
        want = _reference(proc, logits, proc._generated)
        assert mx.array_equal(got, want, equal_nan=True).item()


def test_masks_built_once_and_vocab_change_rebuilds():
    proc = TokenMaskProcessor(suppress=[1, 2], eos_ids=[4], ignore_eos=True)
    a = proc._block(proc._suppress, 32)
    assert proc._block(proc._suppress, 32) is a
    assert proc._block(proc._suppress, 40).shape == (40,)
    assert proc._block(proc._eos, 32) is not a
    out = proc(None, mx.zeros((1, 32)))
    assert out[0, 1].item() == float("-inf") and out[0, 4].item() == float("-inf")
    assert out[0, 0].item() == 0.0


def test_out_of_range_suppress_and_eos_ids_are_ignored():
    proc = TokenMaskProcessor(suppress=[-1, 2, 99], eos_ids=[-3, 4], ignore_eos=True)
    out = proc(None, mx.zeros((1, 8)))
    assert [i for i in range(8) if out[0, i].item() == float("-inf")] == [2, 4]
