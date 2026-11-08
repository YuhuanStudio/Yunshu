"""mlx-vlm 0.7.8's packed Dk=128 GDN kernel against the step kernel the engine keeps (device probe).

Upstream states the two are bitwise equal on the Apple GPUs it tested. The engine disables the packed
kernel (``gdn_prefill.install``) until this passes on the target device (the M5): a prompt token's bits
must not depend on how the prompt was cut into spans."""

import pytest

mx = pytest.importorskip("mlx.core")
gd = pytest.importorskip("mlx_vlm.models.qwen3_5.gated_delta")


def _inputs(T, seed):
    mx.random.seed(seed)
    B, Hk, Hv, Dk, Dv = 1, 2, 4, 128, 128
    q = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    k = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    v = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
    g = mx.sigmoid(mx.random.normal((B, T, Hv))).astype(mx.float32)
    beta = mx.sigmoid(mx.random.normal((B, T, Hv))).astype(mx.float32)
    state = mx.random.normal((B, Hv, Dv, Dk)).astype(mx.float32)
    return q, k, v, g, beta, state


def test_engine_install_disables_the_packed_kernel(monkeypatch):
    from yunshu_engine.kernels import gdn_prefill

    monkeypatch.setattr(
        gdn_prefill, "_STATE", {**gdn_prefill._STATE, "installed": False}
    )
    original = gd.gated_delta_kernel
    packed = getattr(gd, "_gated_delta_kernel_packed", None)
    try:
        if not hasattr(mx.fast, "gated_delta_update"):
            pytest.skip("no mx.fast.gated_delta_update")
        gdn_prefill.install()
        if hasattr(gd, "_gated_delta_kernel_packed"):
            assert gd._gated_delta_kernel_packed is None
    finally:
        gd.gated_delta_kernel = original
        if hasattr(gd, "_gated_delta_kernel_packed"):
            gd._gated_delta_kernel_packed = packed


@pytest.mark.parametrize("T", [2, 16, 48])
def test_packed_kernel_is_bitwise_equal_to_the_step_kernel_here(T):
    packed = getattr(gd, "_gated_delta_kernel_packed", None)
    if packed is None or not mx.metal.is_available():
        pytest.skip("no packed kernel / no Metal")
    args = _inputs(T, T)
    fast = gd.gated_delta_kernel(*args)
    gd._gated_delta_kernel_packed = None
    try:
        step = gd.gated_delta_kernel(*args)
    finally:
        gd._gated_delta_kernel_packed = packed
    mx.eval(fast, step)
    for a, b in zip(fast, step, strict=True):
        assert mx.array_equal(a.view(mx.uint8), b.view(mx.uint8)).item()
