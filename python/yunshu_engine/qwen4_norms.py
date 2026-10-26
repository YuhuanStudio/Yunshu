# Upstream (derived): jundot/omlx (Apache-2.0) qwen4_exp compatibility sanitizer @ 23b671f6
"""Read both legacy direct-gamma and current residual Qwen4 norm weights.

The checkpoint-wide decision uses base attention HC norms. MTP's trained
norms are not independent anchors and must inherit that decision.
"""

import logging
import math
import re
import statistics

logger = logging.getLogger(__name__)
_ANCHOR = re.compile(
    r"(?:^|\.)model\.layers\.\d+\.attn_hyper_connection\.hc_norm\.weight$"
)


def direct_gamma(means: list[float]) -> bool:
    """Classify a pack from multiple stable anchors; refuse ambiguous mixtures."""
    if len(means) < 8:
        return False
    if not all(math.isfinite(x) for x in means):
        raise ValueError("Qwen4 norm convention has nonfinite anchors")
    median = statistics.median(means)
    vote = sum(x > 0.5 for x in means) / len(means)
    if vote >= 0.9 and 0.75 <= median <= 1.5:
        return True
    if vote <= 0.1 and -0.5 <= median <= 0.25:
        return False
    raise ValueError(
        "Qwen4 norm convention is ambiguous (mixed direct/residual weights)"
    )


def infer_direct_gamma(weights: dict) -> bool:
    import mlx.core as mx

    anchors = [
        mx.mean(v.astype(mx.float32)) for k, v in weights.items() if _ANCHOR.search(k)
    ]
    mx.eval(anchors)
    return direct_gamma([float(x.item()) for x in anchors])


def normalize(model, weights: dict, *, direct: bool) -> int:
    """Recenter only Qwen4ExpRMSNorm; leave GDN/vision ordinary norms untouched."""
    if not direct:
        return 0
    import mlx.core as mx
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNorm

    count = 0
    for path, module in model.named_modules():
        key = f"{path}.weight"
        if not isinstance(module, Qwen4ExpRMSNorm) or key not in weights:
            continue
        value = weights[key]
        residual = value.astype(mx.float32) - 1.0
        narrow = residual.astype(value.dtype)
        # Some BF16 direct gammas below 0.5 have no exact BF16 residual.
        # Preserve FP32 there so runtime 1 + residual reproduces the stored gamma.
        weights[key] = (
            narrow
            if mx.array_equal(narrow.astype(mx.float32), residual).item()
            else residual
        )
        count += 1
    logger.info("Qwen4: normalized %d legacy direct-gamma RMSNorm tensors", count)
    return count
