# Upstream (inspired): Blaizzy/mlx-vlm (MIT) speculative/drafters/qwen4_exp_mtp @ 92b31ad3
"""Qwen4 native MTP family: upstream HC/QSA head and invariant verifier.

Do not install the Qwen3.5 lane patches on this family. Its target verifier
captures GDN, QSA indexer and PLE history in upstream speculative transactions.
"""

from typing import Any


def load_native_head(
    config: dict, weights: dict | None, *, direct_norms: bool = False
) -> Any:
    """Strictly load the retained head, respecting mixed per-module quantization."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.config import Qwen4ExpMTPConfig
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.qwen4_exp_mtp import (
        Qwen4ExpMTPDraftModel,
    )

    if not weights:
        raise ValueError("Qwen4 MTP selected without retained native head weights")
    drafter = Qwen4ExpMTPDraftModel(
        Qwen4ExpMTPConfig.from_dict({"text_config": config["text_config"]})
    )
    weights = drafter.sanitize(weights)
    from .qwen4_norms import normalize

    normalize(drafter, weights, direct=direct_norms)
    quant = config.get("mtplx_mtp_quantization") or config.get("quantization") or {}

    def predicate(path, module):
        if f"{path}.scales" not in weights:
            return False
        override = next(
            (
                quant[p + path]
                for p in ("mtp.", "language_model.mtp.", "")
                if p + path in quant
            ),
            None,
        )
        if isinstance(override, dict):
            return override
        # Mixed packs sometimes describe their head incorrectly. Infer affine
        # geometry from the module's unquantized width and stored tensor shape.
        width = module.weight.shape[-1]
        packed = weights[f"{path}.weight"].shape[-1]
        groups = weights[f"{path}.scales"].shape[-1]
        bits, remainder = divmod(packed * 32, width)
        group_size, group_remainder = divmod(width, groups)
        if remainder or group_remainder or bits not in (2, 3, 4, 5, 6, 8):
            raise ValueError(f"Unsupported Qwen4 MTP quantization geometry: {path}")
        return {"bits": bits, "group_size": group_size, "mode": "affine"}

    if any(key.endswith(".scales") for key in weights):
        nn.quantize(drafter, class_predicate=predicate)
    drafter.load_weights(list(weights.items()), strict=True)
    # Allow explicit 2/3/4-depth experiments; upstream otherwise ignores the
    # caller's block and always applies its own adaptive ceiling.
    drafter.prefer_requested_block_size = True
    # The process-global Qwen3.5 early-absorb patch uses this capability to
    # intercept serving. Qwen4 keeps the native loop even after another load.
    drafter.supports_greedy_draft_argmax = False
    mx.eval(drafter.parameters())
    return drafter


def adapt_head(lm: Any) -> bool:
    """Give a quantized LM head the upstream exact kernel cannot read a row-invariant lane matmul.

    mlx-vlm's exact affine head kernel takes 4/5/8-bit heads only. A pack with another head width (the oQ3e
    Flash-Next pack has a 6-bit one) would make ``_supports_batch_invariant_decode`` false, and the server
    refuses to start. The lane matmul reads 2..8-bit affine weights and a row's result does not depend on how
    many rows share the call, so decode, verify and shared-batch rows still agree. True when swapped."""
    import mlx.nn as nn
    from mlx_vlm.models.quantized_verifier import supports_optimized_affine_head

    head = getattr(lm, "lm_head", None)
    if not isinstance(head, nn.QuantizedLinear) or supports_optimized_affine_head(head):
        return False
    from .kernels import lane_linear

    if not lane_linear.eligible(head):
        return False
    lm.lm_head = lane_linear.LaneLinear.from_quantized(head)
    return True


def configure_lane(lm: Any, drafter: Any, block: int | None) -> tuple[dict, int | None]:
    """Keep plain decode and verify on the family-specific upstream arithmetic."""
    adapt_head(lm)
    if getattr(getattr(drafter, "config", None), "target_layer_ids", None):
        from .qwen4_dflash import install

        install(lm)
    required = ("speculative_verify_hidden", "speculative_draft_hidden")
    if not all(callable(getattr(lm, name, None)) for name in required):
        raise RuntimeError("mlx-vlm lacks the Qwen4 hyper-state speculative verifier")
    supports = getattr(lm, "_supports_batch_invariant_decode", None)
    if not callable(supports) or not supports():
        raise RuntimeError("Qwen4 target cannot use batch-invariant plain decode")
    # Block includes the target seed: two rows propose one draft token.  Measured on Flash-Next oQ4e
    # (docs/research/flashnext80/PLAN.md): the first draft is accepted 70-81% of the time, later ones 22-48%,
    # and every extra verify row reads more distinct experts, so block 2 decodes fastest at 1K and 8K.
    return {"qwen4_native": True}, (
        2 if block is None and drafter is not None else block
    )
