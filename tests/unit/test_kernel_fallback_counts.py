"""Kernel gates that send a call to the stock/upstream path are counted once (fallbacks.record).

One test per touched kernel module: each triggers exactly one request-time fallback through the
module's own gate and asserts the counter. GPU-only entry points are faked with monkeypatch.
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from yunshu_engine import fallbacks


@pytest.fixture(autouse=True)
def _fresh_counters():
    fallbacks.reset()
    yield
    fallbacks.reset()


def _count(kind: str) -> int:
    return fallbacks.counts().get(kind, {}).get("count", 0)


def test_lane_linear_nax_prefill_shape_is_counted(monkeypatch):
    from yunshu_engine.kernels import gdn_prefill, lane_linear, nax_prefill

    monkeypatch.setattr(gdn_prefill, "split_enabled", lambda: False)
    monkeypatch.setattr(nax_prefill, "_enabled", True)
    monkeypatch.setattr(lane_linear, "STOCK_ROWS", 512)
    layer = lane_linear.LaneLinear.__new__(lane_linear.LaneLinear)
    nn.Module.__init__(layer)
    layer.bits, layer.group_size = 4, 64
    layer.input_dims, layer.output_dims, layer.tiled = 128, 256, True
    layer.weight = mx.zeros((1, 1), dtype=mx.uint32)
    layer.sbt = mx.zeros((1, 1, 2), dtype=mx.bfloat16)
    dummy = (mx.zeros((1, 1), dtype=mx.uint32), mx.zeros((1, 1)), mx.zeros((1, 1)))
    monkeypatch.setattr(lane_linear.LaneLinear, "stock", lambda self: dummy)
    monkeypatch.setattr(
        mx,
        "quantized_matmul",
        lambda x, *a, **k: mx.zeros((x.shape[0], 256), dtype=x.dtype),
    )
    # 2051 rows: above the 2048-row tail bound and not a multiple of 128, so no NAX atom or tail takes it
    layer(mx.zeros((2051, 128), dtype=mx.bfloat16))
    assert _count("nax_prefill_shape") == 1


def test_attn_prefill_shape_skip_is_counted(monkeypatch):
    from mlx_vlm.models.qwen3_5 import language as lang

    from yunshu_engine.kernels import attn_prefill

    class KVCache:  # the cache kind attn_prefill's skip_reason accepts
        pass

    def stock(*args, **kwargs):
        return "stock"

    monkeypatch.setattr(
        lang, "scaled_dot_product_attention", stock
    )  # restored at teardown
    monkeypatch.setitem(attn_prefill._STATE, "installed", False)
    attn_prefill._install()  # wraps the fake stock; the wrapper is what the test calls
    monkeypatch.setitem(attn_prefill._STATE, "enabled", True)
    # head width 128 is not the 256 the kernel takes: a 1024-row prefill skip
    q = mx.zeros((1, 24, 1024, 128), dtype=mx.bfloat16)
    k = mx.zeros((1, 4, 1024, 128), dtype=mx.bfloat16)
    out = lang.scaled_dot_product_attention(
        q, k, k, KVCache(), attn_prefill._SCALE, "causal", None
    )
    assert out == "stock"
    assert _count("attn_prefill_shape") == 1


def test_sdpa_chunk_head_dim_over_budget_is_counted(monkeypatch):
    from yunshu_engine.kernels import sdpa_chunk

    monkeypatch.setitem(sdpa_chunk._STATE, "force_device", True)  # pre-NAX device
    q = mx.zeros((1, 8, 2048, 128), dtype=mx.bfloat16)  # head width 128: not chunked
    k = mx.zeros((1, 2, 2048, 128), dtype=mx.bfloat16)
    assert sdpa_chunk.should_chunk(q, k, None, "causal", budget=1024) is False
    assert _count("sdpa_chunk_head_dim") == 1


def test_gdn_prefill_norm_scale_width_is_counted(monkeypatch):
    from yunshu_engine.kernels import gdn_prefill, gdn_qk_eps

    monkeypatch.setattr(gdn_qk_eps, "install_lossless_default", lambda: None)
    monkeypatch.setattr(gdn_prefill, "split_enabled", lambda: True)
    monkeypatch.setattr(gdn_prefill, "in_prefill_math", lambda: True)

    class Cls:
        def _normalize_qk(self, q, k):
            return "stock"

    gdn_prefill.install_norm_scale(Cls)
    q = mx.zeros((1, 2, 4, 64), dtype=mx.bfloat16)  # width 64, not the fused 128
    assert Cls()._normalize_qk(q, q) == "stock"
    assert _count("gdn_norm_scale_head_dim") == 1


@pytest.mark.parametrize("variant", ["omlx", "omlx_q4"])
def test_row_exact_qmv_per_row_fallback_is_counted(variant):
    import importlib

    mod = importlib.import_module(f"yunshu_engine.kernels.{variant}.row_exact_qmv")
    n, k = 16, 64

    class FakeLinear:
        bits, group_size, mode = 4, 64, "affine"
        weight = mx.zeros((n, 8), dtype=mx.uint32)

        def __call__(self, x):
            return mx.zeros((x.shape[0], x.shape[1], n), dtype=x.dtype)

    rows = mod.MAX_ROWS + 1  # past the kernel's row cap: one launch per row
    out = mod.quantized_linear(FakeLinear(), mx.zeros((rows, k), dtype=mx.bfloat16))
    assert out.shape == (rows, n)
    assert _count("row_exact_qmv_rows") == 1


@pytest.mark.parametrize("variant", ["omlx", "omlx_q4"])
def test_verify_qmm_no_route_is_counted(monkeypatch, variant):
    import importlib

    mod = importlib.import_module(f"yunshu_engine.kernels.{variant}.qwen35_verify_qmm")
    orig = nn.QuantizedLinear.__call__
    monkeypatch.setattr(nn.QuantizedLinear, "__call__", orig)
    monkeypatch.setattr(
        nn.QuantizedLinear, "_omlx_verify_qmm_patched", False, raising=False
    )
    monkeypatch.setattr(mod, "_QL_PATCHED", False)
    monkeypatch.setattr(mod, "_is_armed", lambda: True)
    monkeypatch.setattr(mod, "is_row_exact_armed", lambda: False)
    for name in ("sg8_eligible", "mma_eligible", "vk_eligible"):
        monkeypatch.setattr(mod, name, lambda *a, **k: False)
    mod.apply_verify_qmm_patch()
    ql = nn.QuantizedLinear(64, 64, bias=False, group_size=64, bits=4)
    out = ql(
        mx.zeros((1, 4, 64), dtype=mx.bfloat16)
    )  # 4 verify rows, no kernel takes them
    assert out.shape == (1, 4, 64)
    assert _count("verify_qmm_no_route") == 1


def test_qwen4_gdn_prework_fallback_is_counted():
    from yunshu_engine.kernels.omlx import qwen35_gdn_prework as prework

    prework._note_qwen4_fallback(
        SimpleNamespace(), mx.zeros((1, 1, 2560), dtype=mx.bfloat16)
    )  # helper-level: the decode closure is only installed by apply (GPU and mlx-vlm patch)
    assert _count("qwen4_gdn_prework") == 1


def test_qwen4_gdn_verify_fallback_is_counted():
    from yunshu_engine.kernels import qwen4_gdn

    inputs = mx.zeros((1, 2, 2560), dtype=mx.bfloat16)
    cache = [
        None,
        None,
    ]  # no canonical conv/recurrent state: the fused verify does not take it
    got = qwen4_gdn.gated_delta(
        None, None, inputs, None, cache, lambda *a, **k: "stock"
    )
    assert got == "stock"
    assert _count("qwen4_gdn_verify") == 1


@pytest.mark.parametrize("variant", ["omlx", "omlx_q4"])
def test_moe_verify_gather_unsupported_is_counted(variant):
    import importlib

    mod = importlib.import_module(f"yunshu_engine.kernels.{variant}.moe_verify_gather")
    switch = SimpleNamespace(
        gate_up_proj=SimpleNamespace(bits=3, group_size=64),
        down_proj=SimpleNamespace(bits=3, group_size=64),
    )
    x = mx.zeros((1, 2, 64), dtype=mx.bfloat16)
    indices = mx.zeros((1, 2, 1), dtype=mx.int32)
    assert (
        mod.fused_switch(switch, x, indices) is None
    )  # 3-bit experts: composed gather
    assert _count("moe_verify_gather_unsupported") == 1


def test_verify_sdpa_split_unsupported_is_counted(monkeypatch):
    from mlx_vlm.models.qwen3_5 import language as q35_lang

    from yunshu_engine.kernels.omlx import qwen35_verify_sdpa_split as split

    def stock(*args, **kwargs):
        return "stock"

    monkeypatch.setattr(q35_lang, "_qwen3_5_left_padded_attention", stock)
    monkeypatch.setattr(split, "_PATCHED", False)
    if not split.apply_qwen35_verify_sdpa_split_patch():
        pytest.skip("verify-split seam needs the Metal device")

    class DenseCache:
        pass

    q = mx.zeros(
        (1, 8, 4, 128), dtype=mx.bfloat16
    )  # head width 128: the split kernel is 256-only
    k = mx.zeros((1, 2, 16, 128), dtype=mx.bfloat16)
    got = q35_lang._qwen3_5_left_padded_attention(
        q, k, k, cache=DenseCache(), scale=1.0, mask="causal"
    )
    assert got == "stock"
    assert _count("verify_sdpa_split_unsupported") == 1


def test_batch_invariant_sg8_miss_is_counted(monkeypatch):
    from yunshu_engine.kernels import batch_invariant
    from yunshu_engine.kernels.omlx import qwen35_verify_qmm

    monkeypatch.setitem(batch_invariant._STATE, "active", True)
    monkeypatch.setattr(qwen35_verify_qmm, "sg8_eligible", lambda *a, **k: False)
    linear = nn.QuantizedLinear(64, 64, bias=False, group_size=64, bits=4)
    linear._yunshu_invariant = True
    x = mx.zeros((1, 2, 64), dtype=mx.bfloat16)
    assert batch_invariant.invariant_linear(linear, x, lambda l, x: "exact") == "exact"
    assert _count("batch_invariant_sg8") == 1


def test_ragged_kv_lane_window_miss_is_counted(monkeypatch):
    from yunshu_engine.kernels import ragged_kv

    monkeypatch.setitem(ragged_kv._STATE, "dense_lane", True)
    monkeypatch.setattr(ragged_kv, "_lane_length", lambda cache: 8)
    monkeypatch.setattr(ragged_kv, "_window_limit", lambda: 8)
    cache = SimpleNamespace(
        keys=mx.zeros((1, 2, 16, 64), dtype=mx.bfloat16),
        values=mx.zeros((1, 2, 16, 64), dtype=mx.bfloat16),
    )
    q = mx.zeros(
        (1, 4, 9, 64), dtype=mx.bfloat16
    )  # 9-token window: past the lane's limit
    assert ragged_kv.dense_lane_attention(q, cache, 1.0) is None
    assert _count("dense_lane_attention") == 1


def test_tf_hc_forward_model_fallback_is_counted(monkeypatch):
    from yunshu_engine.kernels import tf_hc_forward

    monkeypatch.setitem(tf_hc_forward._STATE, "installed", True)
    monkeypatch.setitem(tf_hc_forward._STATE, "original", lambda *a, **k: "stock")
    monkeypatch.setattr(tf_hc_forward, "eligible", lambda model: False)
    model = SimpleNamespace(layers=[object(), object()])
    embeds = mx.zeros((1, 2, 64), dtype=mx.bfloat16)
    got = tf_hc_forward._tf_model(None, model, None, None, embeds, None)
    assert got == "stock"
    assert _count("tf_hc_forward_model") == 1


def test_flashnext_unfused_pack_is_counted(monkeypatch):
    from yunshu_engine.kernels import tf_hc_forward
    from yunshu_engine.round_driver import flashnext, verify

    monkeypatch.setitem(tf_hc_forward._STATE, "installed", True)
    monkeypatch.setattr(tf_hc_forward, "eligible", lambda model: True)
    monkeypatch.setattr(flashnext, "_fused_ready", lambda lm, dtype: False)
    monkeypatch.setattr(flashnext, "_compute_dtype", lambda lm: mx.bfloat16)
    monkeypatch.setattr(
        verify, "native_chain_forward", lambda lm, tokens, caches: "upstream"
    )
    lm = SimpleNamespace(model=object())
    tokens = [mx.zeros((1, 2), dtype=mx.int32)]
    got = flashnext.chain_forward(lm, tokens, [object()])
    assert got == "upstream"
    assert _count("flashnext_fused_pack") == 1


def test_fallback_detail_is_built_only_for_the_first_event():
    calls = []

    def detail():
        calls.append(1)
        return "shape"

    assert fallbacks.record("lazy_detail_probe", detail) is True
    assert fallbacks.record("lazy_detail_probe", detail) is False
    assert calls == [1]
    assert fallbacks.counts()["lazy_detail_probe"] == {"count": 2, "first": "shape"}
