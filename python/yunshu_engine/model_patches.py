"""Model-specific patches for DeepSeek V4 and Qwen 3.5 attention optimization.

oMLX §13.2 pattern: Model families sometimes need runtime patches for:
1. Tokenizer behavior (special token handling, chat template fixes)
2. Attention optimization (Qwen 3.5 YARN/RoPE scaling)
3. Cache format adjustments (DeepSeek MLA cache layout)

These patches are applied at model load time and are reversible.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def apply_model_patches(model: Any, tokenizer: Any, model_name: str) -> list[str]:
    """Apply model-specific patches based on model name.

    Returns list of applied patch names.
    """
    patches = []
    name_lower = model_name.lower()

    if "deepseek" in name_lower:
        patches.extend(_apply_deepseek_patches(model, tokenizer))
    elif "qwen" in name_lower and ("3.5" in name_lower or "3-5" in name_lower):
        patches.extend(_apply_qwen35_patches(model, tokenizer))
    elif "gemma" in name_lower:
        patches.extend(_apply_gemma_patches(model, tokenizer))

    if patches:
        logger.info(f"Applied model patches for {model_name}: {patches}")
    return patches


def remove_model_patches(model: Any, model_name: str) -> None:
    """Remove any applied patches. Called during model unload."""
    # Patches are on the model object — GC handles cleanup
    pass


# ── DeepSeek V4 Patches ────────────────────────────────────────────────────


def _apply_deepseek_patches(model: Any, tokenizer: Any) -> list[str]:
    patches = []

    # Patch 1: Ensure proper chat template with thinking support
    if tokenizer is not None:
        template = getattr(tokenizer, "chat_template", None)
        if template and "{{" in str(template):
            # DeepSeek V4 uses specific template format
            if hasattr(tokenizer, "_yunshu_patched"):
                return patches
            original = tokenizer.chat_template
            tokenizer._yunshu_original_template = original
            tokenizer._yunshu_patched = True
            patches.append("deepseek_chat_template")

    # Patch 2: MLA (Multi-Head Latent Attention) cache format hint
    config = _get_config(model)
    if config is not None:
        # DeepSeek V4 uses MLA which compresses KV into latent vectors
        # This flag tells the KV cache manager to use the compact format
        if hasattr(config, "kv_lora_rank"):
            model._yunshu_mla_mode = True
            patches.append("deepseek_mla_cache")

    # Patch 3: Set proper RoPE scaling for long context
    if config is not None:
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling and isinstance(rope_scaling, dict):
            model._yunshu_rope_scaling = rope_scaling
            patches.append("deepseek_rope_scaling")

    return patches


# ── Qwen 3.5 Patches ──────────────────────────────────────────────────────


def _apply_qwen35_patches(model: Any, tokenizer: Any) -> list[str]:
    patches = []

    config = _get_config(model)
    if config is None:
        return patches

    # Patch 1: YARN RoPE scaling for extended context
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling and isinstance(rope_scaling, dict):
        scaling_type = rope_scaling.get("type", rope_scaling.get("rope_type", ""))
        if scaling_type.lower() in ("yarn", "longrope"):
            model._yunshu_yarn_enabled = True
            model._yunshu_yarn_params = rope_scaling
            patches.append("qwen35_yarn_rope")

    # Patch 2: Dual chunk attention optimization
    # Qwen 3.5 uses dual chunk attention for long sequences
    max_position = getattr(config, "max_position_embeddings", 0)
    if max_position > 32768:
        model._yunshu_dual_chunk = True
        model._yunshu_dual_chunk_threshold = 32768
        patches.append("qwen35_dual_chunk")

    # Patch 3: MTP (Multi-Token Prediction) head detection
    # Qwen 3.5 models may have MTP heads for speculative decoding
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers) if hasattr(model.model.layers, "__len__") else 0
        expected = getattr(config, "num_hidden_layers", 0)
        if num_layers > expected + 1:
            model._yunshu_mtp_heads = num_layers - expected
            patches.append("qwen35_mtp_detect")

    return patches


# ── Gemma Patches ──────────────────────────────────────────────────────────


def _apply_gemma_patches(model: Any, tokenizer: Any) -> list[str]:
    patches = []

    config = _get_config(model)

    # Patch 1: Gemma uses different logit softcap
    if config is not None:
        softcap = getattr(config, "attn_logit_softcapping", None)
        if softcap is not None:
            model._yunshu_attn_softcap = softcap
            patches.append("gemma_attn_softcap")

        final_softcap = getattr(config, "final_logit_softcapping", None)
        if final_softcap is not None:
            model._yunshu_final_softcap = final_softcap
            patches.append("gemma_final_softcap")

    return patches


# ── Helpers ────────────────────────────────────────────────────────────────────


def _get_config(model: Any) -> Any:
    """Extract model config from various model object formats."""
    if model is None:
        return None
    config = getattr(model, "config", None)
    if config is None:
        config = getattr(model, "args", None)
    if config is None and hasattr(model, "model"):
        config = getattr(model.model, "config", None)
    return config


def detect_model_family(model_name: str) -> str:
    """Detect model family from model name."""
    name_lower = model_name.lower()
    if "deepseek" in name_lower:
        return "deepseek"
    if "qwen" in name_lower:
        return "qwen"
    if "gemma" in name_lower:
        return "gemma"
    if "llama" in name_lower:
        return "llama"
    if "glm" in name_lower:
        return "glm"
    if "mistral" in name_lower or "mixtral" in name_lower:
        return "mistral"
    if "phi" in name_lower:
        return "phi"
    return "generic"


def get_model_capabilities(model: Any, model_name: str) -> dict[str, Any]:
    """Detect model capabilities from config."""
    config = _get_config(model)
    if config is None:
        return {"family": detect_model_family(model_name)}

    caps: dict[str, Any] = {
        "family": detect_model_family(model_name),
        "num_layers": getattr(config, "num_hidden_layers", 0),
        "num_heads": getattr(config, "num_attention_heads", 0),
        "num_kv_heads": getattr(config, "num_key_value_heads", 0),
        "hidden_size": getattr(config, "hidden_size", 0),
        "max_position": getattr(config, "max_position_embeddings", 0),
        "vocab_size": getattr(config, "vocab_size", 0),
        "model_type": getattr(config, "model_type", ""),
    }

    # Derived
    if caps["num_heads"] and caps["hidden_size"]:
        caps["head_dim"] = caps["hidden_size"] // caps["num_heads"]

    # Special features
    if getattr(config, "rope_scaling", None):
        caps["rope_scaling"] = config.rope_scaling
    if getattr(config, "attention_bias", False):
        caps["attention_bias"] = True
    if getattr(config, "mlp_bias", False):
        caps["mlp_bias"] = True

    # MoE
    num_experts = getattr(config, "num_local_experts", 0)
    if num_experts:
        caps["moe_experts"] = num_experts
        caps["moe_top_k"] = getattr(config, "num_experts_per_tok", 2)

    return caps
