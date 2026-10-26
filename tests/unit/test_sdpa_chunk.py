"""Query-chunked head-dim-256 SDPA on pre-NAX GPUs (omlx sdpa256_attention, bounded memory).

CPU-only: the chunking math is compared with stock ``mx.fast.scaled_dot_product_attention`` (the
unfused fallback is what CPU always runs) on small shapes with a tiny score budget. Metal bit
parity of the unfused fallback on M1-M4 is on the Mac verification list.
"""

import mlx.core as mx
import pytest

from yunshu_engine.kernels import sdpa_chunk as sc

H, HKV, D = 8, 2, 256


def stock(q, k, v, cache, scale, mask, sinks=None):
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=mask, sinks=sinks
    )


def _qkv(q_len, kv_len, dtype, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((1, H, q_len, D)).astype(dtype)
    k = mx.random.normal((1, HKV, kv_len, D)).astype(dtype)
    v = mx.random.normal((1, HKV, kv_len, D)).astype(dtype)
    return q, k, v


@pytest.fixture(autouse=True)
def pre_nax():
    # The chunking is only installed on M1-M4, where head_dim-256 SDPA runs the unfused fallback. On an M5 the
    # stock call takes the fused kernel (different bits), so compare against the unfused arithmetic on the CPU.
    sc._STATE["force_device"] = True
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)
    sc._STATE["force_device"] = None


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize(
    "q_len, kv_len",
    [
        (40, 100),
        pytest.param(
            33,
            33,
            marks=pytest.mark.xfail(
                strict=False,
                reason="Apple CPU float32 GEMM blocks by row count: 1 ulp off at 33x33 (bf16 exact); the gated "
                "M1-M4 GPU path needs its own bitwise A/B (handoff 'Verify on the Mac' step 9)",
            ),
        ),
        (20, 200),
    ],
)
def test_causal_chunks_equal_stock(dtype, q_len, kv_len):
    q, k, v = _qkv(q_len, kv_len, dtype)
    scale = D**-0.5
    want = stock(q, k, v, None, scale, "causal")
    budget = H * kv_len * dtype.size * 16  # ~16 rows per chunk
    assert sc.rows_per_chunk(q, k, budget) < q_len
    got = sc.chunked_sdpa(stock, q, k, v, None, scale, "causal", None, budget)
    assert got.shape == want.shape
    assert mx.array_equal(got, want).item()


@pytest.mark.parametrize("mask_kind", ["bool", "additive", "broadcast_row"])
def test_array_masks_equal_stock(mask_kind):
    q_len, kv_len = 40, 90
    q, k, v = _qkv(q_len, kv_len, mx.bfloat16, seed=1)
    keep = mx.random.uniform(shape=(q_len, kv_len)) > 0.2
    keep = mx.logical_or(keep, mx.arange(kv_len)[None, :] == 0)  # no fully masked row
    if mask_kind == "bool":
        mask = keep
    elif mask_kind == "additive":
        mask = mx.where(keep, 0.0, -1e4).astype(mx.bfloat16)[None, None]
    else:  # one row broadcast over queries
        mask = keep[:1][None, None]
    want = stock(q, k, v, None, D**-0.5, mask)
    got = sc.chunked_sdpa(
        stock, q, k, v, None, D**-0.5, mask, None, H * kv_len * 2 * 16
    )
    assert mx.array_equal(got, want).item()


def test_sinks_equal_stock():
    q, k, v = _qkv(40, 90, mx.bfloat16, seed=2)
    sinks = mx.random.normal((H,)).astype(mx.bfloat16)
    want = stock(q, k, v, None, D**-0.5, "causal", sinks)
    got = sc.chunked_sdpa(
        stock, q, k, v, None, D**-0.5, "causal", sinks, H * 90 * 2 * 16
    )
    assert mx.array_equal(got, want).item()


def test_single_chunk_when_it_fits():
    q, k, v = _qkv(20, 20, mx.bfloat16)
    assert not sc.should_chunk(q, k, None, "causal")  # scores tiny vs the real budget
    assert sc.should_chunk(q, k, None, "causal", budget=1)
    out = sc.chunked_sdpa(stock, q, k, v, None, D**-0.5, "causal", None, 1 << 40)
    assert mx.array_equal(out, stock(q, k, v, None, D**-0.5, "causal")).item()


def test_gate():
    q, k, v = _qkv(40, 100, mx.bfloat16)
    assert sc.should_chunk(q, k, None, "causal", budget=1)
    # decode / verify rows keep the vector kernel
    assert not sc.should_chunk(q[..., :8, :], k, None, "causal", budget=1)
    # other head dims
    assert not sc.should_chunk(q[..., :128], k[..., :128], None, "causal", budget=1)

    class Quant:
        bits = 4

    assert not sc.should_chunk(q, k, Quant(), "causal", budget=1)
    # square causal bf16 >= 2048 rows is fused by MLX itself
    q2 = mx.zeros((1, H, 2048, D), mx.bfloat16)
    k2 = mx.zeros((1, HKV, 2048, D), mx.bfloat16)
    assert not sc.should_chunk(q2, k2, None, "causal", budget=1)
    # not pre-NAX hardware
    sc._STATE["force_device"] = False
    assert not sc.should_chunk(q, k, None, "causal", budget=1)


def test_install_is_noop_off_pre_nax():
    sc._STATE["force_device"] = False
    sc._STATE["installed"] = False
    assert sc.install() is False


def test_install_is_off_by_default_even_on_pre_nax(monkeypatch):
    # experimental until the M1-M4 bitwise A/B (YUNSHU_SDPA_CHUNK decide=)
    monkeypatch.delenv("YUNSHU_SDPA_CHUNK", raising=False)
    sc._STATE["installed"] = False
    assert sc.install() is False


def test_install_wraps_and_rebinds(monkeypatch):
    import mlx_vlm.models.base as base

    monkeypatch.setenv("YUNSHU_SDPA_CHUNK", "1")
    original = base.scaled_dot_product_attention
    sc._STATE["installed"] = False
    try:
        assert sc.install() is True
        assert base.scaled_dot_product_attention is not original
        assert base.scaled_dot_product_attention._yunshu_sdpa_chunk
        q, k, v = _qkv(40, 100, mx.bfloat16)
        monkeypatch.setattr(sc, "SCORE_BUDGET", 1)
        assert sc.should_chunk(q, k, None, "causal")
        out = base.scaled_dot_product_attention(q, k, v, None, D**-0.5, "causal")
        assert mx.array_equal(out, stock(q, k, v, None, D**-0.5, "causal")).item()
    finally:
        base.scaled_dot_product_attention = original
        import sys

        for name, mod in list(sys.modules.items()):
            if (
                name.startswith("mlx_vlm.models.")
                and getattr(mod, "scaled_dot_product_attention", None) is not None
            ):
                if getattr(
                    mod.scaled_dot_product_attention, "_yunshu_sdpa_chunk", False
                ):
                    mod.scaled_dot_product_attention = original
        sc._STATE["installed"] = False
