"""MoE top-k optimization for Mixture-of-Experts models.

Reduces the number of activated experts per token to improve throughput at the cost of
quality. Works by patching the MoE GATE/router module's `top_k` to a lower value.

The old code matched the SwitchGLU / SwitchLinear / TopKRouter *expert-weight
container* classes — but on every real mlx-lm MoE model those have NO `top_k` attribute (they
consume a pre-computed `indices` tensor). The routing `top_k` lives on the GATE / sparse-MoE
block (e.g. Qwen3MoeSparseMoeBlock.top_k, DeepSeek MoEGate.top_k). So the old match patched
nothing and the documented `moe_top_k` knob was a complete silent no-op. We now match by the
presence of an integer routing `top_k` and patch the gate directly.

Correctness guard: reducing top_k from k to k' drops the lowest-weighted experts. If the gate
RENORMALIZES the surviving experts' softmax weights (norm_topk_prob-style), their sum stays 1
and the MoE output magnitude is preserved — only quality degrades (the intended trade-off). If
the gate does NOT renormalize, dropping experts removes probability mass and SHRINKS the output
magnitude (a correctness bug that propagates through the residual stream). So we only patch
renormalizing gates; non-renormalizing gates are left at their trained top_k and logged.

Applicable models (renormalizing gates): Qwen3 MoE, DeepSeek V3/R1, GLM-4 MoE, Qwen2 MoE, etc.

Trade-off:
- Reducing top_k by 1 can yield +7-16% throughput improvement
- Quality degrades proportionally (more experts skipped = more loss)
- Recommended: only reduce top_k for models with top_k >= 4
"""

import logging

logger = logging.getLogger(__name__)


def _gate_top_k(module):
    """Return the module's integer routing top_k if it is a MoE gate, else None.

    The routing top_k lives on the gate/block module (not the SwitchGLU expert
    container). Guard against bool (a subclass of int) so a stray boolean attribute
    named top_k isn't treated as a count.
    """
    tk = getattr(module, "top_k", None)
    if isinstance(tk, int) and not isinstance(tk, bool):
        return tk
    return None


def _is_renormalizing_gate(module) -> bool:
    """True iff the gate renormalizes its surviving experts' weights, so reducing
    top_k keeps the MoE output correctly SCALED. Covers the common config flag names
    (norm_topk_prob on Qwen/DeepSeek/GLM, plus a couple of aliases)."""
    return bool(
        getattr(module, "norm_topk_prob", None)
        or getattr(module, "norm_topk_probs", None)
        or getattr(module, "renormalize", None)
    )


def apply_moe_top_k(model, target_top_k: int) -> dict:
    """Patch MoE gate modules to use a lower top_k value.

    Args:
        model: The loaded MLX model
        target_top_k: Target number of experts to activate per token

    Returns:
        Dict with 'patched_layers', 'original_top_k', 'new_top_k', 'skipped_unsafe'.
    """
    patched = []
    skipped_unsafe = 0
    original_top_k = None

    for name, module in model.named_modules():
        cur = _gate_top_k(module)
        if cur is None or cur <= target_top_k:
            continue
        if not _is_renormalizing_gate(module):
            # Reducing k on a non-renormalizing gate would scale the output down —
            # refuse rather than silently degrade magnitude.
            skipped_unsafe += 1
            continue
        if original_top_k is None:
            original_top_k = cur
        module.top_k = target_top_k
        patched.append(name)

    if patched:
        logger.info(
            "MoE top-k optimization: %s -> %d (%d gates patched)",
            original_top_k,
            target_top_k,
            len(patched),
        )
    if skipped_unsafe:
        logger.info(
            "MoE top-k: skipped %d non-renormalizing gate(s) — reducing k there would "
            "scale the MoE output down; left at the trained top_k",
            skipped_unsafe,
        )

    return {
        "patched_layers": len(patched),
        "original_top_k": original_top_k,
        "new_top_k": target_top_k if patched else None,
        "skipped_unsafe": skipped_unsafe,
    }


def detect_moe_config(model) -> dict | None:
    """Detect MoE configuration from model.

    Returns dict with moe_layers, top_k, model_type or None if not MoE. MoE-ness is
    signalled by the SwitchGLU/SwitchLinear expert containers OR a gate carrying an
    integer top_k; the reported top_k is read from the gate (the old code
    read it from SwitchGLU, which never has one).
    """
    container_layers = 0
    gate_top_k = None
    gate_count = 0

    for _, module in model.named_modules():
        cls_name = type(module).__name__
        if "SwitchGLU" in cls_name or "SwitchLinear" in cls_name:
            container_layers += 1
        cur = _gate_top_k(module)
        if cur is not None:
            gate_count += 1
            if gate_top_k is None:
                gate_top_k = cur

    if container_layers == 0 and gate_count == 0:
        return None

    return {
        # prefer the gate count (one gate per MoE layer); fall back to containers.
        "moe_layers": gate_count or container_layers,
        "top_k": gate_top_k,
        "model_type": "moe",
    }


def restore_moe_top_k(model, original_top_k: int) -> int:
    """Restore the original top_k on every renormalizing MoE gate (the set
    apply_moe_top_k could have patched; all share the same trained top_k)."""
    restored = 0
    for _, module in model.named_modules():
        if _gate_top_k(module) is not None and _is_renormalizing_gate(module):
            module.top_k = original_top_k
            restored += 1
    return restored
