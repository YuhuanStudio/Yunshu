"""The round driver's GatedDeltaNet mix takes the fused conv+silu and Q/K norm-scale kernels and stays
bit-identical to the unfused expression (concat + conv1d + silu + rms_norm). Needs Metal (gpuq)."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_vlm.models.qwen3_5.language")

from mlx_vlm.models.cache import ArraysCache  # noqa: E402
from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_update  # noqa: E402

from yunshu_engine.kernels import gdn_prefill  # noqa: E402
from yunshu_engine.round_driver import forward as fw  # noqa: E402

from .test_gdn_conv_fused import _layer  # noqa: E402

pytestmark = pytest.mark.skipif(
    mx.default_device() != mx.gpu, reason="Metal kernel needs the GPU"
)


class _Cache(ArraysCache):
    def __init__(self):
        super().__init__(2)
        self.cache = [
            (mx.random.normal((1, 3, 1024)) * 0.5).astype(mx.bfloat16),
            (mx.random.normal((1, 4, 128, 128)) * 0.1).astype(mx.float32),
        ]

    def update_window(self, index, values, n):
        self.cache[index] = values[:, -n:]

    def advance(self, n):
        pass


def _unfused(layer, qkv, z, b, a, cache):
    S = qkv.shape[1]
    z = z.reshape(1, S, -1, layer.head_v_dim)
    conv_input = mx.concatenate([cache[0], qkv], axis=1)
    cache.update_window(0, conv_input, 3)
    conv_out = nn.silu(layer.conv1d(conv_input))
    q, k, v = [
        t.reshape(1, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [layer.key_dim, 2 * layer.key_dim], -1),
            [layer.num_k_heads, layer.num_k_heads, layer.num_v_heads],
            [layer.head_k_dim, layer.head_k_dim, layer.head_v_dim],
            strict=True,
        )
    ]
    inv = k.shape[-1] ** -0.5
    q = (inv**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv * mx.fast.rms_norm(k, None, 1e-6)
    out, _ = gated_delta_update(
        q, k, v, a, b, layer.A_log, layer.dt_bias, use_kernel=True, cache=cache
    )
    return layer.norm(out, z).reshape(1, S, -1)


@pytest.mark.parametrize("rows", [3, 64, 130, 1024])
def test_gdn_mix_fused_equals_unfused(rows, monkeypatch):
    layer = _layer()
    layer.eval()
    qkv = (mx.random.normal((1, rows, 1024)) * 2).astype(mx.bfloat16)
    z = mx.random.normal((1, rows, 512)).astype(mx.bfloat16)
    b = mx.random.normal((1, rows, 4)).astype(mx.bfloat16)
    a = mx.random.normal((1, rows, 4)).astype(mx.bfloat16)
    base = _Cache()
    ref_cache, new_cache = _Cache(), _Cache()
    ref_cache.cache, new_cache.cache = list(base.cache), list(base.cache)
    before = gdn_prefill.fused_conv_calls()
    ref = _unfused(layer, qkv, z, b, a, ref_cache)
    new = fw._gdn_mix(layer, qkv, z, b, a, new_cache)
    mx.eval(ref, new, *ref_cache.cache, *new_cache.cache)
    assert gdn_prefill.fused_conv_calls() == before + 1
    assert mx.array_equal(new.view(mx.uint16), ref.view(mx.uint16)).item()
    assert mx.array_equal(new_cache.cache[0], ref_cache.cache[0]).item()
    assert mx.array_equal(new_cache.cache[1], ref_cache.cache[1]).item()
