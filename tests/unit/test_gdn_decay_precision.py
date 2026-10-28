"""GDN decay stays FP32 (vllm-metal b9a5167 audit): stock keeps it FP32, so must every Yunshu kernel.

CPU-only, source-level for the Metal kernels (they cannot run here); the numeric half runs stock
``compute_g`` and the fused helpers' formula in float32 against a bf16 round of the same decay.
"""

import re
from pathlib import Path

import mlx.core as mx

from yunshu_engine.kernels.omlx import qwen35_gdn_prework as prework
from yunshu_engine.kernels.omlx import qwen35_gdn_verify_fused as fused

ROOT = Path(__file__).resolve().parents[2] / "python" / "yunshu_engine"


def test_stock_decay_is_fp32_and_a_bf16_decay_would_lose_it():
    from mlx_vlm.models.gated_delta import compute_g

    a = mx.full((1, 1, 4), -0.5, dtype=mx.bfloat16)
    dt_bias = mx.full((4,), -9.0, dtype=mx.bfloat16)
    A_log = mx.full((4,), -6.0, dtype=mx.bfloat16)
    g = compute_g(A_log, a, dt_bias)
    assert g.dtype == mx.float32
    # near-one decay (the b9a5167 probe): the bf16 round of it is exactly 1.0
    assert float(g[0, 0, 0]) < 1.0
    assert float(g.astype(mx.bfloat16)[0, 0, 0]) == 1.0


def test_prework_decay_outputs_and_scratch_are_float32():
    src = prework._QWEN4_DECODE_SOURCE
    assert "g_out[head] = metal::precise::exp(-(ea * float(sp)))" in src
    text = (ROOT / "kernels" / "omlx" / "qwen35_gdn_prework.py").read_text()
    # the decode prework returns g as float32 between qkv-dtype outputs and the bf16 beta
    assert re.search(r"qkv\.dtype,\s*\n\s*mx\.float32,\s*\n\s*qkv\.dtype,", text)
    assert "threadgroup float tg_g[1];" in text and "threadgroup float tg_g[S];" in text
    assert "T tg_g" not in text and "threadgroup T tg_g" not in text


def test_fused_verify_decay_is_computed_and_carried_in_float():
    helpers = fused._HELPERS
    assert "inline float gdn_decay(InT a, InT dt, float neg_a)" in helpers
    assert "inline float gdn_decay(half a, half dt, float neg_a)" in helpers
    assert "return metal::precise::exp(neg_a * static_cast<float>(sp));" in helpers
    text = (ROOT / "kernels" / "omlx" / "qwen35_gdn_verify_fused.py").read_text()
    assert "float lane_g" in text and "float gt = simd_shuffle(lane_g" in text
    assert not re.search(r"\b(?:InT|T|half|bfloat)\s+lane_g\b", text)
    for rel in ("tree_verify.py", "dflash_fast.py"):
        body = (ROOT / rel).read_text()
        assert not re.search(r"\b(?:InT|T|half|bfloat)\s+(?:lane_g|G)\b", body), rel
