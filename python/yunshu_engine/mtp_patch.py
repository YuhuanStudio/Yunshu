# DEPRECATED : home-grown Qwen3.5 MTP — superseded by mlx-vlm's
# native MTP (see mlxvlm_mtp.py + YUNSHU_MTP=1; ~1.82x in a proof script, not served/gated — experimental, W973).
# This lacked mlx-vlm's GatedDeltaNet intermediate-state capture (garbage on 27B,
# ~0.9x on 9B). Kept only as a legacy escape hatch under YUNSHU_LEGACY_MTP=1.
"""MTP (Multi-Token Prediction) monkey-patch for mlx-lm's Qwen3.5 model.

Ports the oMLX MTP patch (PR 990) to Yunshu. Adds:
- MTPModule + MTPDecoderLayer: prediction head that proposes next token
- Patched TextModel.sanitize: keeps mtp.* weights
- TextModel.mtp_forward / make_mtp_cache: MTP inference methods
- Outer Model return_hidden / mtp_forward passthrough

The MTP head shares the backbone's embedding and norm layers. Only the
fusion projection (mtp.fc) and one decoder layer add overhead (~15%
of backbone cost per oMLX measurement).

Usage:
    from yunshu_engine.mtp_patch import apply_mtp_patch, load_model_with_mtp
    apply_mtp_patch()
    model = load_model_with_mtp("models/Qwen3.5-4B-MLX-bf16")
"""

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_mtp_patch() -> bool:
    """Apply MTP monkey-patches to mlx_lm.models.qwen3_5. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return True

    try:
        from mlx_lm.models import qwen3_5 as q35
    except ImportError:
        logger.debug("mlx_lm.models.qwen3_5 not importable; skipping MTP patch")
        return False

    if hasattr(q35.TextModel, "_yunshu_mtp_patched"):
        _PATCHED = True
        return True

    _register_mtp_classes(q35)
    _patch_text_model(q35)
    _patch_outer_model(q35)

    _PATCHED = True
    logger.info("MTP patch applied to mlx_lm.models.qwen3_5")
    return True


def _register_mtp_classes(q35: Any) -> None:
    if hasattr(q35, "MTPModule"):
        return

    Attention = q35.Attention
    MLP = q35.MLP
    create_attention_mask = q35.create_attention_mask

    class MTPDecoderLayer(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.self_attn = Attention(args)
            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            if getattr(args, "num_experts", 0) > 0:
                self.mlp = q35.SparseMoeBlock(args)
            else:
                self.mlp = MLP(args.hidden_size, args.intermediate_size)

        def __call__(self, x, mask=None, cache=None):
            r = self.self_attn(self.input_layernorm(x), mask, cache)
            h = x + r
            return h + self.mlp(self.post_attention_layernorm(h))

    class MTPModule(nn.Module):
        """Predicts token t+2 from backbone hidden at t + embedding of token t+1."""

        def __init__(self, args):
            super().__init__()
            self.pre_fc_norm_hidden = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.pre_fc_norm_embedding = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            mtp_layers = getattr(args, "mtp_num_hidden_layers", 1) or 1
            self.layers = [MTPDecoderLayer(args) for _ in range(mtp_layers)]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        def __call__(self, hidden_states, next_token_ids, embed_tokens, cache=None):
            embeds = embed_tokens(next_token_ids)
            e = self.pre_fc_norm_embedding(embeds)
            h = self.pre_fc_norm_hidden(hidden_states)
            fused = self.fc(mx.concatenate([e, h], axis=-1))

            if cache is None:
                cache = [None] * len(self.layers)

            mask = create_attention_mask(fused, cache[0] if cache else None)
            for layer, c in zip(self.layers, cache, strict=False):
                fused = layer(fused, mask, c)

            return self.norm(fused)

    q35.MTPDecoderLayer = MTPDecoderLayer
    q35.MTPModule = MTPModule


def _patch_text_model(q35: Any) -> None:
    cls = q35.TextModel
    if "_yunshu_mtp_patched" in cls.__dict__:
        return

    from mlx_lm.models.cache import KVCache

    original_init = cls.__init__

    def __init__(self, args):
        original_init(self, args)
        n_mtp = int(getattr(args, "mtp_num_hidden_layers", 0) or 0)
        if n_mtp > 0:
            self.mtp = q35.MTPModule(args)

    def __call__(self, inputs, cache=None, input_embeddings=None,
                 return_hidden: bool = False, n_confirmed: int = 0):
        # Newer mlx-lm Qwen3_5TextModel.__call__ signature: (inputs, cache,
        # input_embeddings) — does NOT accept n_confirmed. Pass it via
        # kwargs only if the model accepts it (some forks/branches add it).
        try:
            hidden = self.model(inputs, cache, input_embeddings=input_embeddings,
                                n_confirmed=n_confirmed)
        except TypeError:
            hidden = self.model(inputs, cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(hidden)
        else:
            out = self.lm_head(hidden)
        if return_hidden:
            return out, hidden
        return out

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache):
        if hidden_states.shape[1] > 1:
            hidden_states = hidden_states[:, -1:, :]
        mtp_out = self.mtp(
            hidden_states, next_token_ids, self.model.embed_tokens, mtp_cache,
        )
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(mtp_out)
        return self.lm_head(mtp_out)

    def make_mtp_cache(self):
        if hasattr(self, "mtp"):
            return [KVCache() for _ in self.mtp.layers]
        return []

    def sanitize(self, weights):
        has_mtp_weights = any("mtp." in k for k in weights)
        has_unsanitized_conv1d = any(
            "conv1d.weight" in k and v.shape[-1] != 1 for k, v in weights.items()
        )
        should_shift_norm_weights = has_mtp_weights or has_unsanitized_conv1d

        if not hasattr(self, "mtp"):
            weights = {k: v for k, v in weights.items() if "mtp." not in k}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        norm_keys = (
            ".input_layernorm.weight", ".post_attention_layernorm.weight",
            "model.norm.weight", ".q_norm.weight", ".k_norm.weight",
            ".pre_fc_norm_hidden.weight", ".pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        )
        shifted_keys: set[str] = set()
        for k, v in list(weights.items()):
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if should_shift_norm_weights and any(k.endswith(s) for s in norm_keys):
                if v.ndim == 1 and k not in shifted_keys:
                    weights[k] = v + 1.0
                    shifted_keys.add(k)
        return weights

    cls.__init__ = __init__
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.sanitize = sanitize
    cls._yunshu_mtp_patched = True


def _patch_outer_model(q35: Any) -> None:
    """Patch outer Model class for return_hidden and MTP passthrough."""
    cls = q35.Model
    if "_yunshu_mtp_patched" in cls.__dict__:
        return


    def __call__(self, inputs, cache=None, input_embeddings=None,
                 return_hidden: bool = False, n_confirmed: int = 0):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings,
            return_hidden=return_hidden, n_confirmed=n_confirmed,
        )

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache):
        return self.language_model.mtp_forward(hidden_states, next_token_ids, mtp_cache)

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls._yunshu_mtp_patched = True


def load_model_with_mtp(model_path: str):
    """Load a Qwen3.5 model with MTP head enabled.

    Loads the base model, then attaches MTP head and loads its weights
    from mtp-weights.safetensors.
    """
    from pathlib import Path

    from mlx_lm.models import qwen3_5 as q35
    from mlx_lm.utils import load_model

    apply_mtp_patch()

    model_path = Path(model_path)
    model, config = load_model(model_path)

    inner = getattr(model, "language_model", model)

    if not hasattr(inner, "mtp"):
        mtp_path = model_path / "mtp-weights.safetensors"
        if not mtp_path.exists():
            raise FileNotFoundError(
                f"MTP weights not found at {mtp_path}. "
                "Extract from HuggingFace model first."
            )

        mtp_weights = mx.load(str(mtp_path))
        inner.mtp = q35.MTPModule(inner.args)
        inner.load_weights(list(mtp_weights.items()), strict=False)
        logger.info(f"Loaded MTP head with {len(mtp_weights)} weight tensors")

    mtp_cache = inner.make_mtp_cache()
    logger.info(f"MTP head active: {len(mtp_cache)} cache layers")
    return model
