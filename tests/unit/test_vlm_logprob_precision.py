"""Chosen-token logprobs from the VLM runner must not collapse to 0.0 in bf16.

Upstream computes ``logits - logsumexp(logits)`` in the model dtype. With bf16 logits the
logsumexp rounds coarsely, so a near-certain token read exactly 0.0 while its
alternatives read about -4.25 in the same row. The runner appends an fp32 upcast last
when logprobs are requested.
"""

from __future__ import annotations

import mlx.core as mx

from yunshu_engine.vlm_batch_runner import Fp32LogitsProcessor


def _logprobs(logits):
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def test_fp32_processor_gives_exact_logprobs():
    base = mx.array([[24.0, 19.75, 19.6, 10.0] + [0.0] * 60], dtype=mx.float32)
    exact = _logprobs(base)[0, 0].item()
    assert -0.05 < exact < 0.0  # near certain, but strictly negative

    proc = Fp32LogitsProcessor()
    out = proc(mx.array([1]), base.astype(mx.bfloat16))
    assert out.dtype == mx.float32
    got = _logprobs(out)[0, 0].item()
    assert abs(got - exact) < 0.02
    assert got <= 0.0
    # the chosen token's own entry equals what top-k would report for it
    top = mx.take_along_axis(_logprobs(out), mx.argmax(out, axis=-1, keepdims=True), -1)
    assert abs(top.item() - got) < 1e-6


def test_process_last_token_upcasts_too():
    x = mx.zeros((1, 8), dtype=mx.bfloat16)
    assert Fp32LogitsProcessor().process_last_token(3, x).dtype == mx.float32
