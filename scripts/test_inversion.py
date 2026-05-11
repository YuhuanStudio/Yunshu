"""Verify DeltaNet inversion correctness by round-tripping forward → invert.

Tests that invert_state(forward_state) ≈ original_state for both
scalar and vectorized gating.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import mlx.core as mx
from yunshu_engine.deltanet_inversion import DeltaNetInverter, DeltaNetInversionEntry


def test_scalar_g():
    """Test inversion with scalar g [B, Hv]."""
    B, Hv, Dv, Dk = 1, 2, 3, 4

    mx.random.seed(42)
    state_old = mx.random.normal((B, Hv, Dv, Dk))
    k = mx.random.normal((B, Hv, Dk)) * 0.1
    v = mx.random.normal((B, Hv, Dv))
    g = mx.ones((B, Hv)) * 0.9
    beta = mx.ones((B, Hv)) * 0.5

    # Forward pass (replicate _gated_delta_step_ops)
    g_exp = g[..., None, None]
    state_decayed = state_old * g_exp
    kv_mem = (state_decayed * k[:, :, None, :]).sum(axis=-1)  # [B, Hv, Dv]
    delta = (v - kv_mem) * beta[..., None]  # [B, Hv, Dv]
    state_new = state_decayed + k[:, :, None, :] * delta[..., None]  # [B, Hv, Dv, Dk]

    # Invert
    inverter = DeltaNetInverter()
    entry = DeltaNetInversionEntry(
        gate=g, beta=beta, key=k, value=v, state_after=state_new,
    )
    recovered = inverter.invert_state(entry)

    err = mx.abs(recovered - state_old).max().item()
    rel_err = (mx.abs(recovered - state_old) / (mx.abs(state_old) + 1e-8)).max().item()
    print(f"Scalar g:   max_abs_err={err:.6e}  max_rel_err={rel_err:.6e}", end="")
    if err < 1e-4:
        print("  PASS")
    else:
        print(f"  FAIL (err > 1e-4)")
    return err < 1e-4


def test_vectorized_g():
    """Test inversion with vectorized g [B, Hv, Dk]."""
    B, Hv, Dv, Dk = 1, 2, 3, 4

    mx.random.seed(123)
    state_old = mx.random.normal((B, Hv, Dv, Dk))
    k = mx.random.normal((B, Hv, Dk)) * 0.1
    v = mx.random.normal((B, Hv, Dv))
    g = mx.random.uniform(shape=(B, Hv, Dk), low=0.5, high=1.0)
    beta = mx.random.uniform(shape=(B, Hv), low=0.1, high=0.8)

    # Forward pass with vectorized g
    g_exp = g[..., None, :]  # [B, Hv, 1, Dk]
    state_decayed = state_old * g_exp
    kv_mem = (state_decayed * k[:, :, None, :]).sum(axis=-1)
    delta = (v - kv_mem) * beta[..., None]
    state_new = state_decayed + k[:, :, None, :] * delta[..., None]

    # Invert
    inverter = DeltaNetInverter()
    entry = DeltaNetInversionEntry(
        gate=g, beta=beta, key=k, value=v, state_after=state_new,
    )
    recovered = inverter.invert_state(entry)

    err = mx.abs(recovered - state_old).max().item()
    rel_err = (mx.abs(recovered - state_old) / (mx.abs(state_old) + 1e-8)).max().item()
    print(f"Vectorized: max_abs_err={err:.6e}  max_rel_err={rel_err:.6e}", end="")
    if err < 1e-4:
        print("  PASS")
    else:
        print(f"  FAIL (err > 1e-4)")
    return err < 1e-4


def test_high_beta():
    """Test near-degenerate case where beta * ||k||² → 1."""
    B, Hv, Dv, Dk = 1, 2, 3, 4

    mx.random.seed(99)
    state_old = mx.random.normal((B, Hv, Dv, Dk)) * 0.1  # small state
    k = mx.random.normal((B, Hv, Dk))
    # Normalize k so ||k||² ≈ 1 (like rms_norm + inv_scale)
    k = k / mx.sqrt((k * k).sum(axis=-1, keepdims=True))
    v = mx.random.normal((B, Hv, Dv))
    g = mx.ones((B, Hv)) * 0.95
    beta = mx.ones((B, Hv)) * 0.95  # beta * ||k||² ≈ 0.95, denom = 0.05

    g_exp = g[..., None, None]
    state_decayed = state_old * g_exp
    kv_mem = (state_decayed * k[:, :, None, :]).sum(axis=-1)
    delta = (v - kv_mem) * beta[..., None]
    state_new = state_decayed + k[:, :, None, :] * delta[..., None]

    inverter = DeltaNetInverter()
    entry = DeltaNetInversionEntry(
        gate=g, beta=beta, key=k, value=v, state_after=state_new,
    )
    recovered = inverter.invert_state(entry)

    err = mx.abs(recovered - state_old).max().item()
    rel_err = (mx.abs(recovered - state_old) / (mx.abs(state_old) + 1e-8)).max().item()
    print(f"High beta:  max_abs_err={err:.6e}  max_rel_err={rel_err:.6e}", end="")
    if err < 1e-2:
        print("  PASS (relaxed)")
    else:
        print(f"  WARN (degenerate case, expected precision loss)")
    return err < 1e-2


def test_bf16():
    """Test in bfloat16 — the actual dtype used in inference."""
    B, Hv, Dv, Dk = 1, 4, 8, 16

    mx.random.seed(77)
    state_old = mx.random.normal((B, Hv, Dv, Dk)).astype(mx.bfloat16)
    k = (mx.random.normal((B, Hv, Dk)) * 0.1).astype(mx.bfloat16)
    v = mx.random.normal((B, Hv, Dv)).astype(mx.bfloat16)
    g = mx.full((B, Hv), 0.9).astype(mx.bfloat16)
    beta = mx.full((B, Hv), 0.5).astype(mx.bfloat16)

    # Forward in bf16
    g_exp = g[..., None, None]
    state_decayed = state_old * g_exp
    kv_mem = (state_decayed * k[:, :, None, :]).sum(axis=-1)
    delta = (v - kv_mem) * beta[..., None]
    state_new = state_decayed + k[:, :, None, :] * delta[..., None]

    # Invert (upcasts to f32 internally)
    inverter = DeltaNetInverter()
    entry = DeltaNetInversionEntry(
        gate=g, beta=beta, key=k, value=v, state_after=state_new,
    )
    recovered = inverter.invert_state(entry)

    # Direct comparison: some BF16 forward error is expected
    direct_err = mx.abs(recovered - state_old).max().item()

    # Round-trip consistency: forward(recovered) should == state_new
    # This is what matters for speculative decoding — the inverted state
    # must reproduce the same state_new when re-running forward.
    state_decayed2 = recovered * g_exp
    kv_mem2 = (state_decayed2 * k[:, :, None, :]).sum(axis=-1)
    delta2 = (v - kv_mem2) * beta[..., None]
    state_new2 = state_decayed2 + k[:, :, None, :] * delta2[..., None]

    roundtrip_err = mx.abs(state_new2 - state_new).max().item()

    print(f"BF16:       direct_err={direct_err:.6e}  roundtrip_err={roundtrip_err:.6e}", end="")
    # BF16 has 7-bit mantissa. Roundtrip error of ~0.016 (=2^-6) is the
    # BF16 precision floor. This is too high for speculative decoding
    # where exact token reproduction is required. The inversion is
    # mathematically correct (proven in float32) but unsuitable for BF16.
    if roundtrip_err < 0.02:
        print("  OK (BF16 precision floor — unsuitable for spec decode)")
    else:
        print(f"  FAIL")
    return roundtrip_err < 0.02


if __name__ == "__main__":
    print("=== DeltaNet Inversion Round-Trip Tests ===\n")
    results = []
    results.append(("Scalar g (float32)", test_scalar_g()))
    results.append(("Vectorized g", test_vectorized_g()))
    results.append(("High beta", test_high_beta()))
    results.append(("BF16", test_bf16()))
    print()
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    for name, r in results:
        print(f"  {'PASS' if r else 'FAIL'}: {name}")
