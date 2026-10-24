"""Exact BF16 rounding parity on Qwen GDN prefill shapes (GPU through gpuq)."""

import pytest

mx = pytest.importorskip("mlx.core")
from yunshu_engine.kernels import gdn_prefill  # noqa: E402


@pytest.mark.parametrize("tokens", [1, 64, 128, 512, 1984, 2048])
@pytest.mark.parametrize("heads", [16, 48])
@pytest.mark.parametrize("inverse", [True, False])
def test_rms_scale_exact(tokens, heads, inverse):
    mx.random.seed(tokens + heads)
    x = mx.random.normal((1, tokens, heads, 128)).astype(mx.bfloat16)
    # Input slices in the actual convolution result are strided across heads.
    x = mx.concatenate([x, x], axis=2)[:, :, :heads]
    scale = 128 ** (-1 if inverse else -0.5)
    ref = scale * mx.fast.rms_norm(x, None, 1e-6)
    got = gdn_prefill.fused_norm_scale(x, inverse=inverse)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert mx.array_equal(got.view(mx.uint16), ref.view(mx.uint16)).item()


def test_dispatch_requires_split_prefill(monkeypatch):
    from types import SimpleNamespace

    class Layer:
        def _normalize_qk(self, q, k):
            return "upstream"

    calls = []
    monkeypatch.setattr(
        gdn_prefill,
        "fused_norm_scale",
        lambda x, *, inverse: calls.append(inverse) or x,
    )
    monkeypatch.setitem(gdn_prefill._STATE, "enabled", True)
    monkeypatch.setitem(gdn_prefill._STATE, "split", True)
    gdn_prefill.install_norm_scale(Layer)
    x = SimpleNamespace(dtype=mx.bfloat16, shape=(1, 64, 16, 128), ndim=4)
    assert Layer()._normalize_qk(x, x) == "upstream"
    with gdn_prefill.prefill_scope():
        assert Layer()._normalize_qk(x, x) == (x, x)
    assert calls == [True, False]
    monkeypatch.setitem(gdn_prefill._STATE, "split", False)
    with gdn_prefill.prefill_scope():
        assert Layer()._normalize_qk(x, x) == "upstream"
