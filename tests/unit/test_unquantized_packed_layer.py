"""The packed Qwen3.5 layer on an unquantized checkpoint: row-invariant projections and head (CPU)."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from yunshu_engine import tree_verify  # noqa: E402
from yunshu_engine.round_driver import forward  # noqa: E402


def _lm(proj, skipped=0):
    layer = SimpleNamespace(
        is_linear=True, linear_attn=SimpleNamespace(in_proj_qkv=proj)
    )
    return SimpleNamespace(
        model=SimpleNamespace(layers=[layer]), _yunshu_lane_skipped=skipped
    )


def test_float_projections_count_as_row_invariant_projections():
    assert tree_verify.lane_projections(_lm(nn.Linear(8, 8)))
    # a quantized projection the lane could not read keeps the 8-row limit
    assert not tree_verify.lane_projections(_lm(nn.Linear(8, 8), skipped=1))
    assert not tree_verify.lane_projections(_lm(nn.QuantizedLinear(64, 64, bits=4)))


def test_dense_lm_head_rows_equal_the_single_row_call():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(1)
        for tied in (True, False):
            emb = nn.Embedding(512, 64)
            head = nn.Linear(64, 512, bias=False)
            lm = SimpleNamespace(
                args=SimpleNamespace(tie_word_embeddings=tied),
                model=SimpleNamespace(embed_tokens=emb),
                lm_head=head,
            )
            h = mx.random.normal((7, 64)).astype(mx.bfloat16)
            emb.set_dtype(mx.bfloat16)
            head.set_dtype(mx.bfloat16)
            got = forward.logits(lm, h)
            one = mx.concatenate([forward.logits(lm, h[i : i + 1]) for i in range(7)])
            assert mx.array_equal(got, one).item()
            assert tuple(got.shape) == (7, 512)
    finally:
        mx.set_default_device(previous)


def test_quantized_head_keeps_one_call_over_the_rows():
    """2026-11-08 yv bisect: 355610644 sent the 27B's quantized LM head through the per-row loop (one
    GEMV per verify row), costing ~11 ms per round (-28% decode at 1K). Only a float head loops."""
    import mlx.core as mx
    import mlx.nn as nn

    from yunshu_engine.round_driver import forward as fwd

    calls = []

    class Head(nn.QuantizedLinear):
        def __call__(self, x):
            calls.append(int(x.shape[0]))
            return super().__call__(x)

    head = Head(64, 32, bias=False, group_size=32, bits=4)

    class Args:
        tie_word_embeddings = False

    class LM:
        args = Args()
        lm_head = head

    fwd.logits(LM(), mx.zeros((5, 64)))
    assert calls == [5]
