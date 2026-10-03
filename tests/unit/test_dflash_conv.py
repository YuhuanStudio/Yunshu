from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.speculative.drafters.dflash2 import dflash2 as upstream

from yunshu_engine.dflash_conv import grouped_conv, install


def test_strided_convolution_preserves_upstream_rounding():
    for dtype in (mx.bfloat16, mx.float32):
        hidden = mx.random.normal((2, 16, 64)).astype(dtype)[:, ::2]
        dynamic = mx.random.normal((2, 8, 2, 2, 4)).astype(dtype)[:, :, 1]
        base = mx.random.normal((2, 64)).astype(mx.bfloat16)
        expected = upstream._grouped_dynamic_convolve(hidden, dynamic, base, 16)
        actual = grouped_conv(hidden, dynamic, base, 16)
        mx.eval(expected, actual)
        assert bool(mx.array_equal(expected, actual))


def test_instance_install_is_private_idempotent_and_keeps_upstream_global():
    convs = [upstream.GroupedDynamicCausalConv(64, 2, 16) for _ in range(2)]
    for conv in convs:
        conv.set_dtype(mx.bfloat16)
        nn.quantize(conv, group_size=64, bits=8)
    hidden = mx.random.normal((1, 8, 64)).astype(mx.bfloat16)
    expected = [conv.prepare(hidden) for conv in convs]
    finish_hidden = mx.random.normal(hidden.shape).astype(mx.bfloat16)
    finished = [
        conv.finish(finish_hidden, dyn)
        for conv, (_, dyn) in zip(convs, expected, strict=True)
    ]
    mx.eval(expected, finished)
    head = object()
    draft = SimpleNamespace(
        layers=[SimpleNamespace(attention_conv=convs[0], mlp_conv=convs[1])],
        lm_head=head,
    )
    original = upstream._grouped_dynamic_convolve
    assert install(draft) == 2
    for conv, (h, dyn), end in zip(convs, expected, finished, strict=True):
        actual_h, actual_dyn = conv.prepare(hidden)
        actual_end = conv.finish(finish_hidden, actual_dyn)
        mx.eval(actual_h, actual_dyn, actual_end)
        assert bool(mx.array_equal(h, actual_h))
        assert bool(mx.array_equal(dyn, actual_dyn))
        assert bool(mx.array_equal(end, actual_end))
    methods = [conv.prepare for conv in convs]
    assert install(draft) == 0
    assert [conv.prepare for conv in convs] == methods
    assert upstream._grouped_dynamic_convolve is original
    assert draft.lm_head is head


def test_half_precision_uses_original_proposal_path():
    conv = upstream.GroupedDynamicCausalConv(64, 2, 16)
    conv.set_dtype(mx.float16)
    hidden = mx.random.normal((1, 4, 64)).astype(mx.float16)
    expected_h, expected_dyn = conv.prepare(hidden)
    expected_end = conv.finish(hidden, expected_dyn)
    mx.eval(expected_h, expected_dyn, expected_end)
    draft = SimpleNamespace(layers=[SimpleNamespace(attention_conv=conv)])
    assert install(draft) == 1
    actual_h, actual_dyn = conv.prepare(hidden)
    actual_end = conv.finish(hidden, actual_dyn)
    mx.eval(actual_h, actual_dyn, actual_end)
    assert bool(mx.array_equal(expected_h, actual_h))
    assert bool(mx.array_equal(expected_end, actual_end))
