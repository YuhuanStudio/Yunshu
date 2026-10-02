"""The shared batch's GDN step must use the singleton's arithmetic."""

from types import SimpleNamespace

import mlx.core as mx
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen3_5.language import Qwen3_5RMSNormGated

from yunshu_engine.cache_decode import fused_gdn_step


def test_fused_gdn_batch_equals_individual_rows_bitwise():
    mx.random.seed(923)
    heads, key_heads, dim = 48, 16, 128
    channels = (2 * key_heads + heads) * dim
    vectors = [
        mx.random.normal((1, 1, n)).astype(mx.bfloat16) * 0.1
        for n in (channels, heads * dim, heads, heads)
    ]

    def projection(base):
        return lambda x: mx.broadcast_to(base, (x.shape[0], 1, base.shape[-1]))

    norm = Qwen3_5RMSNormGated(dim, eps=1e-6)
    norm.weight = norm.weight.astype(mx.bfloat16)
    layer = SimpleNamespace(
        in_proj_qkv=projection(vectors[0]),
        in_proj_z=projection(vectors[1]),
        _project_gates=lambda x: (projection(vectors[2])(x), projection(vectors[3])(x)),
        out_proj=lambda x: x,
        norm=norm,
        head_k_dim=dim,
        head_v_dim=dim,
        num_k_heads=key_heads,
        num_v_heads=heads,
        conv1d=SimpleNamespace(
            weight=mx.random.normal((channels, 4, 1)).astype(mx.bfloat16) * 0.1
        ),
        A_log=mx.zeros((heads,)),
        dt_bias=mx.zeros((heads,)),
    )
    conv = mx.random.normal((1, 3, channels)).astype(mx.bfloat16) * 0.1
    state = mx.random.normal((1, heads, dim, dim)).astype(mx.float32) * 0.1

    def cache(rows):
        c = ArraysCache(2)
        c.cache = [mx.repeat(conv, rows, axis=0), mx.repeat(state, rows, axis=0)]
        return c

    one, many = cache(1), cache(4)
    ref = fused_gdn_step(layer, mx.zeros((1, 1, 32), mx.bfloat16), one)
    out = fused_gdn_step(layer, mx.zeros((4, 1, 32), mx.bfloat16), many)
    mx.eval(ref, out, one.state, many.state)
    for row in range(4):
        assert mx.array_equal(out[row : row + 1], ref).item()
        assert mx.array_equal(many[0][row : row + 1], one[0]).item()
        assert mx.array_equal(many[1][row : row + 1], one[1]).item()


def test_singleton_batch_cache_keeps_the_same_decode_dispatch(monkeypatch):
    from mlx_vlm.models.qwen3_5 import language as q35

    from yunshu_engine import cache_decode

    cls = q35.Qwen3_5GatedDeltaNet
    monkeypatch.setattr(cls, "_yunshu_prefix_decode", False, raising=False)
    monkeypatch.setattr(cls, "__call__", lambda *a, **k: "fallback")
    monkeypatch.setattr(cache_decode, "fused_gdn_step", lambda *a: "canonical")
    cache_decode.install()
    layer = SimpleNamespace(
        training=False,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        conv_dim=384,
        num_v_heads=1,
        conv1d=SimpleNamespace(weight=mx.ones((384, 4, 1), mx.bfloat16)),
    )
    c = ArraysCache(2)
    c.cache = [
        mx.zeros((1, 3, 384), mx.bfloat16),
        mx.zeros((1, 1, 128, 128), mx.float32),
    ]
    cache_decode.set_active(True)
    try:
        assert (
            cls.__call__(layer, mx.zeros((1, 1, 32), mx.bfloat16), cache=c)
            == "canonical"
        )
    finally:
        cache_decode.set_active(False)
