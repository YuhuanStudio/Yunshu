"""MoE top-k optimization for Mixture-of-Experts models.

Reduces the number of activated experts per token to improve throughput
at the cost of quality. Works by patching the model's SwitchGLU/SwitchLinear
layers to use a lower top_k than the model config default.

Applicable models:
- Qwen3 MoE variants (Qwen3-30B-A3B, etc.)
- Llama 4 Scout/Maverick
- DBRX
- Granite MoE
- DeepSeek V3/R1
- Phi 3.5 MoE

Trade-off:
- Reducing top_k by 1 can yield +7-16% throughput improvement
- Quality degrades proportionally (more experts skipped = more loss)
- Recommended: only reduce top_k for models with top_k >= 4
"""

import logging

logger = logging.getLogger(__name__)


def apply_moe_top_k(model, target_top_k: int) -> dict:
    """Patch MoE layers to use a lower top_k value.

    Finds all SwitchGLU/SwitchLinear layers in the model and patches
    their top_k attribute to the target value.

    Args:
        model: The loaded MLX model
        target_top_k: Target number of experts to activate per token

    Returns:
        Dict with 'patched_layers', 'original_top_k', 'new_top_k'
    """
    patched = []
    original_top_k = None

    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if any(k in cls_name for k in ("SwitchGLU", "SwitchLinear", "TopKRouter")):
            current_top_k = getattr(module, "top_k", None)
            if current_top_k is not None and current_top_k > target_top_k:
                if original_top_k is None:
                    original_top_k = current_top_k
                module.top_k = target_top_k
                patched.append(name)

    if patched:
        logger.info(
            f"MoE top-k optimization: {original_top_k} → {target_top_k} "
            f"({len(patched)} layers patched)"
        )

    return {
        "patched_layers": len(patched),
        "original_top_k": original_top_k,
        "new_top_k": target_top_k if patched else None,
    }


def detect_moe_config(model) -> dict | None:
    """Detect MoE configuration from model.

    Returns dict with num_experts, top_k, num_layers or None if not MoE.
    """
    experts_found = 0
    top_k = None
    moe_layers = 0

    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "SwitchGLU" in cls_name or "SwitchLinear" in cls_name:
            experts_found += 1
            moe_layers += 1
            if top_k is None:
                top_k = getattr(module, "top_k", None)
        elif "TopKRouter" in cls_name:
            top_k = getattr(module, "top_k", None)

    if experts_found == 0:
        return None

    return {
        "moe_layers": moe_layers,
        "top_k": top_k,
        "model_type": "moe",
    }


def restore_moe_top_k(model, original_top_k: int) -> int:
    """Restore original top_k value for all MoE layers."""
    restored = 0
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if any(k in cls_name for k in ("SwitchGLU", "SwitchLinear", "TopKRouter")):
            if hasattr(module, "top_k"):
                module.top_k = original_top_k
                restored += 1
    return restored
