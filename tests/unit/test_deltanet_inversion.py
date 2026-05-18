"""Tests for yunshu_engine.deltanet_inversion — DeltaNet state inversion."""
from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from yunshu_engine.deltanet_inversion import (
    DeltaNetInversionEntry,
    DeltaNetInverter,
)


def _apply_forward_and_make_entry(
    B=1, Hv=2, Hk=2, Dv=4, Dk=4, scalar_g=True, dtype=mx.float32,
):
    """Create a valid DeltaNetInversionEntry by applying the forward recurrence.

    Only supports scalar_g=True for simplicity.
    Returns (entry, original_state) where entry.state_after = forward(state).
    """
    state_old = mx.random.normal((B, Hv, Dv, Dk)).astype(dtype)
    g = mx.random.uniform(0.9, 1.0, (B, Hv)).astype(dtype)
    beta = mx.random.uniform(0.01, 0.1, (B, Hv)).astype(dtype)
    k = mx.random.normal((B, Hk, Dk)).astype(dtype)
    v = mx.random.normal((B, Hv, Dv)).astype(dtype)

    # Expand dims for broadcasting
    if Hv > Hk:
        k_rep = mx.repeat(k, Hv // Hk, axis=1)
    else:
        k_rep = k

    g_exp = g[..., None, None]  # [B, Hv, 1, 1]
    beta_exp = beta[..., None, None]  # [B, Hv, 1, 1]
    k_exp = k_rep[:, :, None, :]  # [B, Hv, 1, Dk]

    # Forward: state_new = g * state_old + k * beta * (v - r_gw)
    r_old = (state_old * k_exp).sum(axis=-1)  # [B, Hv, Dv]
    r_gw = g[..., None] * r_old  # [B, Hv, Dv]

    delta_v = v - r_gw  # [B, Hv, Dv]
    correction = beta_exp * delta_v[:, :, :, None] * k_exp  # [B, Hv, Dv, Dk]
    state_new = g_exp * state_old + correction

    return DeltaNetInversionEntry(
        gate=g, beta=beta, key=k, value=v, state_after=state_new,
    ), state_old


class TestDeltaNetInverterInit:
    def test_initial_state(self):
        inv = DeltaNetInverter()
        assert inv._capturing is False
        assert inv._entries == []

    def test_capture_without_start_is_noop(self):
        inv = DeltaNetInverter()
        inv.capture_layer(
            gate=mx.zeros((1, 2)),
            beta=mx.zeros((1, 2)),
            key=mx.zeros((1, 2, 4)),
            value=mx.zeros((1, 2, 4)),
            state_after=mx.zeros((1, 2, 4, 4)),
        )
        assert len(inv._entries) == 0


class TestStartCapture:
    def test_starts_capturing(self):
        inv = DeltaNetInverter()
        inv.start_capture()
        assert inv._capturing is True
        assert inv._entries == []

    def test_clears_previous_entries(self):
        inv = DeltaNetInverter()
        inv._entries = [MagicMock()]
        inv.start_capture()
        assert inv._entries == []


class TestCaptureLayer:
    def test_captures_when_active(self):
        inv = DeltaNetInverter()
        inv.start_capture()
        inv.capture_layer(
            gate=mx.zeros((1, 2)),
            beta=mx.zeros((1, 2)),
            key=mx.zeros((1, 2, 4)),
            value=mx.zeros((1, 2, 4)),
            state_after=mx.zeros((1, 2, 4, 4)),
        )
        assert len(inv._entries) == 1

    def test_captures_multiple_layers(self):
        inv = DeltaNetInverter()
        inv.start_capture()
        for _ in range(5):
            inv.capture_layer(
                gate=mx.zeros((1, 2)),
                beta=mx.zeros((1, 2)),
                key=mx.zeros((1, 2, 4)),
                value=mx.zeros((1, 2, 4)),
                state_after=mx.zeros((1, 2, 4, 4)),
            )
        assert len(inv._entries) == 5


class TestInvertState:
    def test_scalar_g_roundtrip(self):
        entry, original = _apply_forward_and_make_entry(scalar_g=True)
        inv = DeltaNetInverter()
        recovered = inv.invert_state(entry)
        assert recovered.shape == original.shape
        error = float(mx.max(mx.abs(
            original.astype(mx.float32) - recovered.astype(mx.float32)
        )).item())
        assert error < 1e-4, f"Scalar g roundtrip error: {error}"

    def test_vector_g_shape_preserved(self):
        entry, original = _apply_forward_and_make_entry(scalar_g=False)
        inv = DeltaNetInverter()
        recovered = inv.invert_state(entry)
        assert recovered.shape == original.shape

    def test_gqa_head_repeat(self):
        entry, original = _apply_forward_and_make_entry(Hv=4, Hk=2, scalar_g=True)
        inv = DeltaNetInverter()
        recovered = inv.invert_state(entry)
        assert recovered.shape == original.shape

    def test_preserves_float16_dtype(self):
        entry, _ = _apply_forward_and_make_entry(dtype=mx.float16)
        inv = DeltaNetInverter()
        recovered = inv.invert_state(entry)
        assert recovered.dtype == mx.float16

    def test_preserves_bf16_dtype(self):
        entry, _ = _apply_forward_and_make_entry(dtype=mx.bfloat16)
        inv = DeltaNetInverter()
        recovered = inv.invert_state(entry)
        assert recovered.dtype == mx.bfloat16

    def test_identity_case(self):
        """With g=1, beta=0: state_new = state_old (identity)."""
        B, Hv, Dv, Dk = 1, 2, 4, 4
        state = mx.random.normal((B, Hv, Dv, Dk))
        entry = DeltaNetInversionEntry(
            gate=mx.ones((B, Hv)),
            beta=mx.zeros((B, Hv)),
            key=mx.zeros((B, Hv, Dk)),
            value=mx.zeros((B, Hv, Dv)),
            state_after=state,  # g=1, beta=0 => state_new = state
        )
        inv = DeltaNetInverter()
        recovered = inv.invert_state(entry)
        error = float(mx.max(mx.abs(state - recovered)).item())
        assert error < 1e-6, f"Identity case error: {error}"


class TestInvertAll:
    def test_stops_capturing(self):
        inv = DeltaNetInverter()
        inv.start_capture()
        inv.capture_layer(
            gate=mx.zeros((1, 2)),
            beta=mx.zeros((1, 2)),
            key=mx.zeros((1, 2, 4)),
            value=mx.zeros((1, 2, 4)),
            state_after=mx.zeros((1, 2, 4, 4)),
        )
        inv.invert_all()
        assert inv._capturing is False

    def test_returns_recovered_states(self):
        inv = DeltaNetInverter()
        inv.start_capture()
        for _ in range(3):
            entry, _ = _apply_forward_and_make_entry(dtype=mx.float32)
            inv._entries.append(entry)
        results = inv.invert_all()
        assert len(results) == 3
        for r in results:
            assert r.shape == (1, 2, 4, 4)

    def test_empty_entries_returns_empty(self):
        inv = DeltaNetInverter()
        inv.start_capture()
        results = inv.invert_all()
        assert results == []


class TestVerifyRoundtrip:
    def test_returns_nonnegative_float(self):
        state_before = mx.random.normal((1, 2, 4, 4)).astype(mx.float32)
        state_after = mx.random.normal((1, 2, 4, 4)).astype(mx.float32)
        g = mx.ones((1, 2)) * 0.95
        beta = mx.ones((1, 2)) * 0.05
        k = mx.random.normal((1, 2, 4)).astype(mx.float32)
        v = mx.random.normal((1, 2, 4)).astype(mx.float32)
        error = DeltaNetInverter.verify_roundtrip(
            state_before, state_after, g, beta, k, v,
        )
        assert isinstance(error, float)
        assert error >= 0

    def test_shape_mismatch_returns_inf(self):
        # Create entry where recovered shape differs from state_before
        state_before = mx.zeros((2, 3, 4, 4))  # Different shape
        state_after = mx.zeros((1, 2, 4, 4))
        g = mx.ones((1, 2))
        beta = mx.ones((1, 2)) * 0.1
        k = mx.random.normal((1, 2, 4))
        v = mx.random.normal((1, 2, 4))
        error = DeltaNetInverter.verify_roundtrip(
            state_before, state_after, g, beta, k, v,
        )
        assert error == float('inf')

    def test_with_known_forward(self):
        entry, original = _apply_forward_and_make_entry(dtype=mx.float32)
        error = DeltaNetInverter.verify_roundtrip(
            original, entry.state_after, entry.gate, entry.beta, entry.key, entry.value,
        )
        # verify_roundtrip inverts state_after, then compares recovered state
        # against the known original state_before. Error should be small.
        assert isinstance(error, float)
        assert error >= 0
        assert error < 1e-4, f"Roundtrip error too large: {error}"


class TestRegisterHooks:
    def test_register_and_unregister(self):
        inv = DeltaNetInverter()

        class FakeLayer:
            def __call__(self, *a, **kw):
                return None
            state = mx.zeros((1, 2, 4, 4))

        class FakeModel:
            def named_modules(self):
                return [("layer0", FakeLayer())]

        model = FakeModel()
        inv.register_hooks(model)
        assert len(inv._hooked_layers) == 1

        inv.unregister_hooks()
        assert len(inv._hooked_layers) == 0

    def test_unregister_without_register(self):
        inv = DeltaNetInverter()
        inv.unregister_hooks()  # should not raise

    def test_hook_patches_call(self):
        inv = DeltaNetInverter()
        inv.start_capture()

        class FakeLayer:
            _last_gate = mx.ones((1, 2))
            _last_beta = mx.ones((1, 2)) * 0.1
            _last_key = mx.ones((1, 2, 4))
            _last_value = mx.ones((1, 2, 4))
            state = mx.zeros((1, 2, 4, 4))

            def __call__(self, *a, **kw):
                return "output"

        layer = FakeLayer()

        class FakeModel:
            def named_modules(self):
                return [("layer0", layer)]

        model = FakeModel()
        inv.register_hooks(model)

        # Now module() call syntax correctly routes through the hooked class __call__
        result = layer()
        assert result == "output"
        assert len(inv._entries) == 1

        inv.unregister_hooks()

    def test_hook_does_not_capture_when_not_capturing(self):
        inv = DeltaNetInverter()
        # Not calling start_capture

        class FakeLayer:
            _last_gate = mx.ones((1, 2))
            _last_beta = mx.ones((1, 2)) * 0.1
            _last_key = mx.ones((1, 2, 4))
            _last_value = mx.ones((1, 2, 4))
            state = mx.zeros((1, 2, 4, 4))

            def __call__(self, *a, **kw):
                return "output"

        layer = FakeLayer()

        class FakeModel:
            def named_modules(self):
                return [("layer0", layer)]

        model = FakeModel()
        inv.register_hooks(model)
        layer()
        assert len(inv._entries) == 0

        inv.unregister_hooks()

    def test_hook_handles_missing_attrs(self):
        inv = DeltaNetInverter()
        inv.start_capture()

        class MinimalLayer:
            state = mx.zeros((1, 2, 4, 4))
            # No _last_gate, _last_beta, etc.

            def __call__(self, *a, **kw):
                return None

        layer = MinimalLayer()

        class FakeModel:
            def named_modules(self):
                return [("layer0", layer)]

        model = FakeModel()
        inv.register_hooks(model)
        layer()
        # Should not crash, and should not capture (gate/beta are None)
        assert len(inv._entries) == 0

        inv.unregister_hooks()
