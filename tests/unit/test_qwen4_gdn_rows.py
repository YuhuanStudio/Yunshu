"""Packed multi-row Flash-Next GDN (``qwen4_gdn.gated_delta_rows``) equals each row's own call, bit for bit."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
ql = pytest.importorskip("mlx_vlm.models.qwen4_exp.language")

from mlx_vlm.models.cache import ArraysCache  # noqa: E402
from mlx_vlm.models.qwen4_exp.config import TextConfig  # noqa: E402
from mlx_vlm.speculative.cache_state import start_speculative_cache  # noqa: E402

from yunshu_engine.kernels import qwen4_gdn  # noqa: E402


def _layer(in_bits: int, out_bits: int, out_group: int):
    cfg = TextConfig(
        model_type="qwen4_exp",
        hidden_size=2560,
        num_hidden_layers=1,
        num_attention_heads=16,
        linear_num_value_heads=48,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=64,
        moe_intermediate_size=64,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=4096,
    )
    mx.random.seed(3)
    layer = ql.Qwen4ExpGatedDeltaNet(cfg)
    layer.set_dtype(mx.bfloat16)
    nn.quantize(
        layer,
        group_size=64,
        bits=in_bits,
        class_predicate=lambda p, m: isinstance(m, nn.Linear) and p != "out_proj",
    )
    nn.quantize(
        layer,
        group_size=out_group,
        bits=out_bits,
        class_predicate=lambda p, m: isinstance(m, nn.Linear) and p == "out_proj",
    )
    layer.eval()
    mx.eval(layer.parameters())
    return layer


def _cache(seed: int):
    k = mx.random.key(seed)
    k1, k2 = mx.random.split(k)
    c = ArraysCache(size=2)
    c[0] = (mx.random.normal((1, 3, 10240), key=k1) * 0.5).astype(mx.bfloat16)
    c[1] = mx.random.normal((1, 48, 128, 128), key=k2) * 0.05
    mx.eval(c[0], c[1])
    return c


def _clone(c):
    d = ArraysCache(size=2)
    d[0], d[1] = mx.array(c[0]), mx.array(c[1])
    return d


def _never(*_a, **_k):
    raise AssertionError("the composed layer must not run")


# Flash-Next oQ2.5e: in-projections 8-bit g64, out-projection 5-bit g128
@pytest.mark.parametrize(
    ("widths", "keeps", "bits"),
    [
        ([2, 2], [1, 2], (8, 5, 128)),
        ([2, 1, 2], [2, 1, 1], (8, 5, 128)),
        ([2, 2, 2, 2, 2, 2, 2, 2], [1, 2, 2, 1, 2, 1, 1, 2], (8, 5, 128)),
        ([2, 2, 1], [2, 1, 1], (4, 4, 64)),
    ],
)
def test_packed_rows_equal_each_row_alone(widths, keeps, bits):
    assert qwen4_gdn.install_cache_patch()
    layer = _layer(*bits)
    if qwen4_gdn.plan_for(layer) is None:
        pytest.skip("no fused verify plan on this device")
    verifier = ql._QWEN4_BATCH_INVARIANT_FORWARD
    total = sum(widths)
    x = (mx.random.normal((1, total, 2560), key=mx.random.key(7)) * 0.5).astype(
        mx.bfloat16
    )
    spans, at = [], 0
    for w in widths:
        spans.append((at, at + w))
        at += w
    base = [_cache(11 + b) for b in range(len(widths))]

    solo_caches = [_clone(c) for c in base]
    solo_out = []
    for (s, e), c, keep in zip(spans, solo_caches, keeps, strict=True):
        txn = start_speculative_cache([c], e - s)
        solo_out.append(
            qwen4_gdn.gated_delta(verifier, layer, x[:, s:e], None, c, _never)
        )
        mx.eval(solo_out[-1])
        txn.commit(keep)
    solo = mx.concatenate(solo_out, axis=1)

    packed_caches = [_clone(c) for c in base]
    txns = [
        start_speculative_cache([c], e - s)
        for (s, e), c in zip(spans, packed_caches, strict=True)
    ]
    before = qwen4_gdn.counters()["packed"]
    packed = qwen4_gdn.gated_delta_rows(
        verifier, layer, x, spans, [None] * len(widths), packed_caches, _never
    )
    mx.eval(packed)
    assert qwen4_gdn.counters()["packed"] == before + 1
    for txn, keep in zip(txns, keeps, strict=True):
        txn.commit(keep)

    assert packed.shape == solo.shape
    assert mx.array_equal(packed, solo).item()
    for a, b in zip(solo_caches, packed_caches, strict=True):
        mx.eval(a[0], a[1], b[0], b[1])
        assert mx.array_equal(a[0], b[0]).item()
        assert mx.array_equal(a[1], b[1]).item()


def test_one_row_or_masked_rows_keep_the_per_row_call(monkeypatch):
    calls = []

    def fake(verifier, layer, inputs, mask, cache, original):
        calls.append(int(inputs.shape[1]))
        return inputs

    monkeypatch.setattr(qwen4_gdn, "gated_delta", fake)
    x = mx.zeros((1, 3, 2560), dtype=mx.bfloat16)
    out = qwen4_gdn.gated_delta_rows(
        None, object(), x, [(0, 1), (1, 3)], [None, "mask"], [None, None], _never
    )
    assert calls == [1, 2]
    assert out.shape == (1, 3, 2560)
