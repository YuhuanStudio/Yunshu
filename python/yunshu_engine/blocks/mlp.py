"""MLP blocks of the round driver (M4): one interface per structure, chosen per layer by structure.

``RoutedMoE`` for a layer with routed experts (a ``switch_mlp``: router, experts and a shared expert),
``DenseMLP`` for a gated SwiGLU (``gate_proj`` / ``up_proj`` / ``down_proj``). ``block_for`` picks one from the
module's attributes; nothing here names a model family. The bodies are the calls the layers made before M4:
``packed`` is the prefill / verify segment form (``forward._mlp``), ``verified`` the verifier's own
``_feed_forward`` (kept a method call, so the kernels that patch it still apply), and ``fused`` the Flash-Next
fused router + experts (``tf_hc_forward._moe``).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def routed(mlp: Any) -> bool:
    """True for a module with routed experts (the structure ``forward._mlp`` always split on)."""
    return hasattr(mlp, "switch_mlp")


def block_for(mlp: Any) -> DenseMLP | RoutedMoE:
    """The block of one layer's MLP module."""
    return RoutedMoE(mlp) if routed(mlp) else DenseMLP(mlp)


class DenseMLP:
    """Gated SwiGLU over ``gate_proj`` / ``up_proj`` / ``down_proj``."""

    def __init__(self, mlp: Any) -> None:
        self.mlp = mlp

    def packed(self, x: mx.array, spans: list, lane: list) -> mx.array:
        """The packed segment form: each projection per segment (lane projections as their own kernel)."""
        from mlx_vlm.models.qwen3_5.language import swiglu

        from ..round_driver.forward import _project

        mlp = self.mlp
        h = swiglu(
            _project(mlp.gate_proj, x, spans, lane),
            _project(mlp.up_proj, x, spans, lane),
        )
        return _project(mlp.down_proj, h, spans, lane)

    def verified(self, verifier: Any, x: mx.array) -> mx.array:
        """The verifier's own feed-forward (a method call: the verify kernels patch it)."""
        return verifier._feed_forward(self.mlp, x)

    def fused(self, x2d: mx.array) -> Any:
        """The fused router + experts attempt. A dense layer has none: the call still runs (its plan check
        records the ``tf_moe_plan`` fallback, as it always did) and returns None."""
        return _fused(self.mlp, x2d)


class RoutedMoE:
    """Router, routed experts and shared expert of a sparse MoE block."""

    def __init__(self, mlp: Any) -> None:
        self.mlp = mlp

    def packed(self, x: mx.array, spans: list, lane: list) -> mx.array:
        """One call per segment: an expert matmul groups the tokens it is given, so packing segments would
        make a row depend on its neighbours. A single segment is the flat call."""
        from ..round_driver.forward import _cat

        return _cat([self.mlp(x[:, s:e]) for s, e in spans])

    def verified(self, verifier: Any, x: mx.array) -> mx.array:
        return verifier._feed_forward(self.mlp, x)

    def fused(self, x2d: mx.array) -> Any:
        """The fused router + experts (<= MAX_ROWS rows): ``("grouped", ...)`` or None (the caller falls back
        to ``verified``)."""
        return _fused(self.mlp, x2d)


def _fused(mlp: Any, x2d: mx.array) -> Any:
    from ..kernels import tf_hc_forward

    return tf_hc_forward._moe(mlp, x2d)
