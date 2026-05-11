"""DeltaNet state inversion — analytically invert the SSM recurrence on rejection.

The GatedDeltaNet recurrence (ICLR 2025, Songlin Yang et al.):

    state'  = g * state_old           (decay)
    kv_mem  = state' · k              (retrieve, dot along Dk)
    delta   = beta * (v - kv_mem)     (error correction)
    state_new = state' + k ⊗ delta   (write)

This is analytically invertible. Let r_gw = sum_dk(g * state_old * k) —
the "g-weighted retrieval". For scalar g, r_gw = g * r_old where
r_old = state_old · k.

Deriving the inversion:
    r_new = state_new · k
          = r_gw + beta * (v - r_gw) * ||k||²
          = r_gw * (1 - beta * ||k||²) + beta * v * ||k||²

    r_gw = (r_new - beta * v * ||k||²) / (1 - beta * ||k||²)

    state_old = (state_new - k * beta * (v - r_gw)) / g

For scalar g [B, Hv]: r_gw = g * r_old (straightforward).
For vectorized g [B, Hv, Dk]: r_gw = sum_dk(g_dk * state_old_dv_dk * k_dk)
  — same inversion formula applies.

Benefits over checkpoint/restore:
  - No per-cycle snapshot of full SSM state (~56MB on 4B)
  - Only save k, v, g, beta per SSM layer (~41KB — 75x less)
  - On acceptance: zero overhead
  - On rejection: cheap matrix ops

Limitations:
  - BF16 precision: roundtrip error ~0.016 per inversion step. Too high
    for speculative decoding where exact token reproduction is required.
    Checkpoint/restore is exact (zero error) and should be preferred.
  - Conv state: still needs snapshot (small, rolling buffer)
  - Multi-token verify (K>1): invert K updates sequentially

Verdict: Mathematically correct (float32 roundtrip < 1e-7 error) but
NOT practical for BF16 speculative decoding. Kept for reference — may
be useful if float32 states or mixed-precision inference becomes viable.
Monkey-patch capture mechanism incomplete. Not integrated with mtp_decoder.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class DeltaNetInversionEntry:
    """Intermediate values saved during verify for one SSM layer."""
    gate: mx.array        # g: [B, Hv] or [B, Hv, Dk]
    beta: mx.array        # write gate: [B, Hv]
    key: mx.array         # k: [B, Hk, Dk]
    value: mx.array       # v: [B, Hv, Dv]
    state_after: mx.array # state_new: [B, Hv, Dv, Dk]


class DeltaNetInverter:
    """Manages DeltaNet state inversion for speculative decoding.

    Usage (in always-advance verify loop):
        inverter = DeltaNetInverter()
        inverter.start_capture()
        # ... forward pass captures intermediates ...
        if rejected:
            states = inverter.invert_all()
    """

    def __init__(self):
        self._entries: list[DeltaNetInversionEntry] = []
        self._capturing = False

    def start_capture(self):
        self._entries.clear()
        self._capturing = True

    def capture_layer(
        self,
        gate: mx.array,
        beta: mx.array,
        key: mx.array,
        value: mx.array,
        state_after: mx.array,
    ):
        if not self._capturing:
            return
        self._entries.append(DeltaNetInversionEntry(
            gate=gate,
            beta=beta,
            key=key,
            value=value,
            state_after=state_after,
        ))

    def invert_state(self, entry: DeltaNetInversionEntry) -> mx.array:
        """Invert the DeltaNet recurrence for one layer.

        Given state_new = g * state_old + k * beta * (v - r_gw)
        where r_gw = sum_dk(g_dk * state_old_dv_dk * k_dk)

        Recovery:
          r_gw = (r_new - beta * v * ||k||²) / (1 - beta * ||k||²)
          state_old = (state_new - k * beta * (v - r_gw)) / g

        Returns the recovered state [B, Hv, Dv, Dk].
        """
        # Upcast to float32 for numerical stability — the subtraction
        # in (1 - beta*k_sq) and (state_new - correction) loses too
        # many bits in float16/bfloat16.
        orig_dtype = entry.state_after.dtype
        state_new = entry.state_after.astype(mx.float32)
        g = entry.gate.astype(mx.float32)
        beta = entry.beta.astype(mx.float32)
        k = entry.key.astype(mx.float32)
        v = entry.value.astype(mx.float32)

        # Broadcast shapes: state_new is [B, Hv, Dv, Dk]
        Hv = state_new.shape[1]
        Hk = k.shape[1]

        # Expand k heads to match v heads if needed (GQA-style repeat)
        if Hv > Hk:
            k = mx.repeat(k, Hv // Hk, axis=1)

        if g.ndim == 2:
            g_expanded = g[..., None, None]  # [B, Hv, 1, 1]
        else:
            g_expanded = g[..., None, :]     # [B, Hv, 1, Dk]

        beta_expanded = beta[..., None, None]  # [B, Hv, 1, 1]
        k_expanded = k[:, :, None, :]          # [B, Hv, 1, Dk]

        # Step 1: r_new = state_new · k (dot along Dk)
        # state_new: [B, Hv, Dv, Dk], k_expanded: [B, Hv, 1, Dk]
        # r_new: [B, Hv, Dv]
        r_new = (state_new * k_expanded).sum(axis=-1)

        # Step 2: k_sq = ||k||² per head
        k_sq = (k * k).sum(axis=-1, keepdims=True)  # [B, Hv, 1]

        # Step 3: r_gw = (r_new - beta * v * ||k||²) / (1 - beta * ||k||²)
        # This is the g-weighted retrieval: r_gw = sum_dk(g * state_old * k)
        # For scalar g: r_gw = g * r_old
        # For vectorized g: r_gw = sum_dk(g_dk * state_old_dv_dk * k_dk)
        beta_kk = (beta * k_sq[:, :, 0])[..., None]  # [B, Hv, 1]
        denom = 1.0 - beta_kk  # [B, Hv, 1]
        r_gw = (r_new - beta_kk * v) / denom  # [B, Hv, Dv]

        # Step 4: state_old = (state_new - k * beta * (v - r_gw)) / g
        delta_v = v - r_gw  # [B, Hv, Dv]
        correction = beta_expanded * delta_v[:, :, :, None] * k_expanded  # [B, Hv, Dv, Dk]
        state_old = (state_new - correction) / g_expanded

        return state_old.astype(orig_dtype)

    def invert_all(self) -> list[mx.array]:
        self._capturing = False
        results = []
        for entry in self._entries:
            results.append(self.invert_state(entry))
        mx.synchronize()
        return results
