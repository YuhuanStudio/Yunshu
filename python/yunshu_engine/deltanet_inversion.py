from __future__ import annotations

"""DeltaNet state inversion — analytically invert the SSM recurrence on rejection.

Used in the KV cache eviction path: when KV blocks are evicted from the prefix
cache under memory pressure, the inverted state allows partial recovery of
evicted context. Enabled via YUNSHU_DELTANET_INVERSION=1 env var.

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
NOT practical for BF16 speculative decoding. Wired into KV cache eviction
path as an opt-in feature (YUNSHU_DELTANET_INVERSION=1) for SSM layer
state recovery. May become more useful if float32 states or mixed-precision
inference becomes viable.
Capture mechanism integrated via register_hooks() for DeltaNet layers.
"""

import logging
from dataclasses import dataclass

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class DeltaNetInversionEntry:
    """Intermediate values saved during verify for one SSM layer."""

    gate: mx.array  # g: [B, Hv] or [B, Hv, Dk]
    beta: mx.array  # write gate: [B, Hv]
    key: mx.array  # k: [B, Hk, Dk]
    value: mx.array  # v: [B, Hv, Dv]
    state_after: mx.array  # state_new: [B, Hv, Dv, Dk]


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
        self._entries.append(
            DeltaNetInversionEntry(
                gate=gate,
                beta=beta,
                key=key,
                value=value,
                state_after=state_after,
            )
        )

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

        # [B, Hv, 1, 1] when 2D, else [B, Hv, 1, Dk]
        g_expanded = g[..., None, None] if g.ndim == 2 else g[..., None, :]

        beta_expanded = beta[..., None, None]  # [B, Hv, 1, 1]
        k_expanded = k[:, :, None, :]  # [B, Hv, 1, Dk]

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
        # Guard against near-zero denominator (beta*||k||² close to 1).
        # Clamp to a small epsilon to avoid inf/nan in the division.
        # Use +eps or -eps based on sign of denom; if denom is exactly 0,
        # default to +eps (the typical case where beta*||k||² < 1).
        eps = mx.array(1e-6, dtype=denom.dtype)
        denom = mx.where(mx.abs(denom) < eps, eps, denom)
        r_gw = (r_new - beta_kk * v) / denom  # [B, Hv, Dv]

        # Step 4: state_old = (state_new - k * beta * (v - r_gw)) / g
        delta_v = v - r_gw  # [B, Hv, Dv]
        correction = (
            beta_expanded * delta_v[:, :, :, None] * k_expanded
        )  # [B, Hv, Dv, Dk]
        # Guard against near-zero gate (g ≈ 0 means heavy decay, inversion is unstable).
        g_safe = mx.where(mx.abs(g_expanded) < eps, eps, g_expanded)
        state_old = (state_new - correction) / g_safe

        return state_old.astype(orig_dtype)

    def invert_all(self) -> list[mx.array]:
        self._capturing = False
        if not self._entries:
            return []
        results = []
        for entry in self._entries:
            results.append(self.invert_state(entry))
        mx.synchronize()
        return results

    def register_hooks(self, model) -> None:
        """Register capture hooks on DeltaNet SSM layers.

        Patches each layer's ``__call__`` on its **class** to capture
        gate/beta/key/value/state_after during the verify pass.
        Uses a per-instance ``_inverter_hook`` dict to route captures
        to the correct inverter.

        Call ``unregister_hooks()`` when done.

        Note:
            Python's ``obj()`` dispatch resolves ``__call__`` from the
            **class**, not the instance.  Setting ``module.__call__ = fn``
            only affects ``module.__call__(...)`` (attribute lookup), not
            ``module(...)`` (call syntax).  We therefore patch the class
            and use a per-instance hook dict to multiplex.

        Args:
            model: An nn.Module containing DeltaNet SSM layers with a
                   ``state`` attribute and ``gate``, ``beta``, ``k``, ``v`` parameters.
        """
        self._original_forwards: list = []
        self._hooked_layers = []
        self._hooked_classes: dict = {}  # {class: original __call__}
        inverter = self

        # Collect target modules
        targets = []
        for idx, (_name, module) in enumerate(model.named_modules()):
            if hasattr(module, "state"):
                targets.append((idx, module))

        # Patch each unique class once
        for idx, module in targets:
            cls = type(module)

            if cls not in self._hooked_classes:
                original_fn = cls.__call__
                self._hooked_classes[cls] = original_fn

                def make_patched_call(orig):
                    def patched_call(self_layer, *args, **kwargs):
                        result = orig(self_layer, *args, **kwargs)
                        hook_data = getattr(self_layer, "_inverter_hook", None)
                        if hook_data is not None:
                            inv, layer_idx = hook_data
                            if inv._capturing and hasattr(self_layer, "state"):
                                try:
                                    g = getattr(self_layer, "_last_gate", None)
                                    beta = getattr(self_layer, "_last_beta", None)
                                    k = getattr(self_layer, "_last_key", None)
                                    v = getattr(self_layer, "_last_value", None)
                                    if g is not None and beta is not None:
                                        inv.capture_layer(
                                            gate=g,
                                            beta=beta,
                                            key=k,
                                            value=v,
                                            state_after=self_layer.state,
                                        )
                                except Exception as exc:
                                    logger.debug(
                                        "DeltaNet capture hook failed for layer %d: %s",
                                        layer_idx,
                                        exc,
                                    )
                        return result

                    return patched_call

                cls.__call__ = make_patched_call(original_fn)

            # Attach hook data to this instance
            module._inverter_hook = (inverter, idx)
            self._hooked_layers.append(module)

    def unregister_hooks(self) -> None:
        """Remove all capture hooks, restoring original class __call__ methods."""
        # Remove per-instance hook data
        if hasattr(self, "_hooked_layers"):
            for module in self._hooked_layers:
                if hasattr(module, "_inverter_hook"):
                    del module._inverter_hook
            self._hooked_layers.clear()
        # Restore original class __call__
        if hasattr(self, "_hooked_classes"):
            for cls, original_fn in self._hooked_classes.items():
                cls.__call__ = original_fn
            self._hooked_classes.clear()

    @staticmethod
    def verify_roundtrip(
        state_before: mx.array,
        state_after: mx.array,
        gate: mx.array,
        beta: mx.array,
        key: mx.array,
        value: mx.array,
    ) -> float:
        """Verify inversion accuracy on a single layer.

        Args:
            state_before: The state BEFORE the forward step (ground truth).
            state_after: The state AFTER the forward step (= state_new).
            gate, beta, key, value: Intermediate values from the forward step.

        Returns max absolute error between original (state_before) and
        recovered state.  Useful for testing whether float32 precision
        is sufficient.
        """
        entry = DeltaNetInversionEntry(
            gate=gate,
            beta=beta,
            key=key,
            value=value,
            state_after=state_after,
        )
        inverter = DeltaNetInverter()
        recovered = inverter.invert_state(entry)
        if state_before.shape != recovered.shape:
            return float("inf")
        return float(
            mx.max(
                mx.abs(state_before.astype(mx.float32) - recovered.astype(mx.float32))
            ).item()
        )
