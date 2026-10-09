# Upstream (inspired): Blaizzy/mlx-vlm (MIT) speculative/drafters/qwen4_exp_mtp @ 92b31ad3
"""Qwen4 native MTP family: upstream HC/QSA head and invariant verifier.

Do not install the Qwen3.5 lane patches on this family. Its target verifier
captures GDN, QSA indexer and PLE history in upstream speculative transactions.
"""

from typing import Any


def load_native_head(config: dict, weights: dict | None) -> Any:
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


def configure_lane(lm: Any, drafter: Any, block: int | None) -> tuple[dict, int | None]:
    """Keep plain decode and verify on the family-specific upstream arithmetic."""
    required = ("speculative_verify_hidden", "speculative_draft_hidden")
    if not all(callable(getattr(lm, name, None)) for name in required):
        raise RuntimeError("mlx-vlm lacks the Qwen4 hyper-state speculative verifier")
    supports = getattr(lm, "_supports_batch_invariant_decode", None)
    if not callable(supports) or not supports():
        raise RuntimeError("Qwen4 target cannot use batch-invariant plain decode")
    from .qwen4_draft_policy import install

    install()
    # Block includes the target seed: four rows propose three draft tokens.
    return {"qwen4_native": True}, (
        4 if block is None and drafter is not None else block
    )
