"""MoE per-token call equality on the tiny Qwen3.5-MoE sparse block, on the default (Metal) device (M4).

The flat call (the whole segment in one call) must equal the segment calls the packed form makes, bit for bit;
and a token's output must not depend on the tokens of another segment (solo == beside others)."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm")
q35m = pytest.importorskip("mlx_vlm.models.qwen3_5_moe.language")

from yunshu_engine.blocks import mlp as blocks  # noqa: E402

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the MoE call equality is a Metal test"
)


def _tiny_moe():
    args = SimpleNamespace(
        hidden_size=64,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
    )
    mx.random.seed(0)
    block = q35m.Qwen3_5MoeSparseMoeBlock(args)
    mx.eval(block.parameters())
    return block


def test_flat_call_equals_segment_calls_bit_for_bit():
    block = _tiny_moe()
    x = mx.random.normal((1, 12, 64)).astype(mx.bfloat16)
    spans = [(0, 5), (5, 6), (6, 12)]
    flat = block(x)  # the whole segment in one call
    packed_single = blocks.RoutedMoE(block).packed(x, [(0, 12)], [False])
    packed = blocks.RoutedMoE(block).packed(x, spans, [False, False, False])
    per_segment = mx.concatenate([block(x[:, s:e]) for s, e in spans], axis=1)
    mx.eval(flat, packed_single, packed, per_segment)
    assert mx.array_equal(packed_single, flat).item()
    assert mx.array_equal(packed, per_segment).item()


def test_a_tokens_output_does_not_depend_on_its_neighbour_segment():
    block = _tiny_moe()
    x = mx.random.normal((1, 8, 64)).astype(mx.bfloat16)
    other = mx.random.normal((1, 8, 64)).astype(mx.bfloat16)
    both = mx.concatenate([x, other], axis=1)
    beside = blocks.RoutedMoE(block).packed(both, [(0, 8), (8, 16)], [False, False])
    alone = blocks.RoutedMoE(block).packed(x, [(0, 8)], [False])
    mx.eval(beside, alone)
    assert mx.array_equal(beside[:, :8], alone).item()
