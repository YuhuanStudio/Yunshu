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


STOCK_ROWS = 32  # lane projections serve calls up to this many rows (decode, MTP / draft verify); prefill chunks stay stock


def lane_projections(lm: Any, *, head: bool) -> dict:
    """Run the dense quantized projections as the row-invariant lane matmul (``kernels.lane_linear``).

    Upstream's verify forward computes every dense projection of T rows as T singleton ``gather_qmm`` calls so
    that each row gets the one-row decode bits; the weight is read once per row, so T rows cost T-fold dense
    traffic (27B-sized Flash-Next dense side: 1 / 1.4 / 2.4 / 8.1 x the one-row time at 1 / 2 / 4 / 16 rows). The
    lane matmul gives a row the same bits at any row count and reads the weight once. Decode, verify and
    drafting all run through these modules, so speculative output equals plain decode. ``head`` also converts
    the LM head (used when upstream's fused head kernel does not take the checkpoint's head format).
    Calls wider than STOCK_ROWS (prefill chunks) run MLX's quantized matmul on a transient stock-layout copy."""
    from .kernels import lane_linear
    from .kernels.tensorfold import lane_qmm

    if not hasattr(lm, "named_modules") or not lane_qmm.ready():
        return {}
    report = lane_linear.convert(lm, skip=() if head else ("lm_head",))
    lane_linear.set_stock_rows(STOCK_ROWS)
    from .kernels import lane_virtual

    lane_virtual.install_qwen4()
    return {
        "lane_linear": report["converted"],
        "lane_skipped": len(report["skipped"]),
        "prefill_matmul": lane_linear.prefill_kernel_id(),
    }


def configure_lane(lm: Any, drafter: Any, block: int | None) -> tuple[dict, int | None]:
    """Keep plain decode and verify on one family-specific arithmetic.

    Default: upstream's singleton qmv arithmetic. ``YUNSHU_QWEN4_LANE`` (or a head format upstream's fused
    head kernel cannot take, e.g. the 6-bit head of the oQ3e pack) moves the dense projections to the lane
    matmul for decode, verify and drafting alike."""
    from . import settings

    required = ("speculative_verify_hidden", "speculative_draft_hidden")
    if not all(callable(getattr(lm, name, None)) for name in required):
        raise RuntimeError("mlx-vlm lacks the Qwen4 hyper-state speculative verifier")
    supports = getattr(lm, "_supports_batch_invariant_decode", None)
    head_ok = callable(supports) and bool(supports())
    kernels: dict = {"qwen4_native": True}
    if settings.get_bool("YUNSHU_QWEN4_LANE") or not head_ok:
        kernels.update(lane_projections(lm, head=not head_ok))
        if not kernels.get("lane_linear"):
            raise RuntimeError("Qwen4 target cannot use batch-invariant plain decode")
        head_ok = bool(supports())
    if not head_ok:
        raise RuntimeError("Qwen4 target cannot use batch-invariant plain decode")
    # Block includes the target seed: two rows propose one draft token.  Measured on Flash-Next oQ4e
    # (docs/research/flashnext80/PLAN.md): the first draft is accepted 70-81% of the time, later ones 22-48%,
    # and every extra verify row reads more distinct experts, so block 2 decodes fastest at 1K and 8K.
    return kernels, (2 if block is None and drafter is not None else block)
