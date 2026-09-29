# Upstream (inspired): jundot/omlx (Apache-2.0) omlx/patches/qwen35_verify_qmm.py @ a98d8c8c
# Patches upstream mlx-vlm symbols (see vendor.json kind=patches; `just vendor-check` flags source changes)
"""Fused residual + norm and early submission for the speculative lane's forward.

The lane's target forward (upstream's ``Qwen3_5BatchInvariantForward``) runs each
decoder layer as separate ops: a residual add, an RMSNorm, the mixer, another
add and norm, the MLP. oMLX ships a kernel that does ``(a + b, rms_norm(a + b))``
in one launch, bit-exact to the separate MLX ops (one threadgroup per row, MLX's
per-thread square-sum order); its own use of it is tied to an armed verify mode
Yunshu does not run. Here it is applied whenever the batch-invariant kernels are
active, and a layer's MLP output is added together with the next layer's input
norm, so a layer costs two launches for its adds and norms instead of four.

Every N layers the partial graph is submitted (``async_eval``), so the GPU starts
on the first layers while the host is still building the rest of the forward.

Only rows-per-call up to ``MAX_ROWS`` (speculative windows) take the fused path;
the arithmetic is row-independent, so a row's bits do not depend on the window.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

FLUSH_LAYERS = 8
_STATE: dict = {"installed": False, "next_norm": None, "normed": None, "layers": 0}


def _qmm() -> Any:
    from .omlx import qwen35_verify_qmm

    return qwen35_verify_qmm


def install() -> bool:
    """Patch the verifier's ``_model`` / ``_layer`` (idempotent)."""
    if _STATE["installed"]:
        return True
    try:
        from mlx_vlm.models.qwen3_5.speculative_verifier import (
            Qwen3_5BatchInvariantForward as verifier,
        )
    except ImportError:  # pragma: no cover - older mlx-vlm
        return False
    from . import batch_invariant

    original_layer = verifier._layer
    original_model = verifier._model

    def _model(self, model, *args, **kwargs):
        layers = model.layers
        _STATE["next_norm"] = {
            id(layer): nxt.input_layernorm
            for layer, nxt in zip(layers, layers[1:], strict=False)
        }
        _STATE["normed"] = None
        _STATE["layers"] = 0
        try:
            return original_model(self, model, *args, **kwargs)
        finally:
            _STATE["next_norm"] = None
            _STATE["normed"] = None

    def _layer(self, layer, hidden, mask, cache, position_ids, position_embeddings):
        if not batch_invariant._STATE["active"] or _STATE["next_norm"] is None:
            return original_layer(
                self, layer, hidden, mask, cache, position_ids, position_embeddings
            )
        q = _qmm()
        pending = _STATE["normed"]
        if pending is not None and pending[0] is hidden:
            normed = pending[1]
        else:
            normed = layer.input_layernorm(hidden)
        _STATE["normed"] = None
        if layer.is_linear:
            residual = self._gated_delta(layer.linear_attn, normed, mask, cache)
        else:
            residual = self._attention(
                layer.self_attn, normed, mask, cache, position_ids, position_embeddings
            )
        post = layer.post_attention_layernorm
        if q._add_rms_eligible(hidden, residual, post):
            hidden, normed, _ = q.add_rms_norm(hidden, residual, post)
        else:
            hidden = hidden + residual
            normed = post(hidden)
        out = self._feed_forward(layer.mlp, normed)
        nxt = _STATE["next_norm"].get(id(layer))
        if nxt is not None and q._add_rms_eligible(hidden, out, nxt):
            hidden, normed, _ = q.add_rms_norm(hidden, out, nxt)
            _STATE["normed"] = (hidden, normed)
        else:
            hidden = hidden + out
        _STATE["layers"] += 1
        if _STATE["layers"] % FLUSH_LAYERS == 0:
            mx.async_eval(hidden)
        return hidden

    verifier._model = _model
    verifier._layer = _layer
    _STATE["installed"] = True
    logger.info("lane layers: fused add+norm, flush every %d layers", FLUSH_LAYERS)
    return True


__all__ = ["install"]
