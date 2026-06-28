"""Z-Image-Fun-Controlnet-Union — MLX port.

Architecture (verified against alibaba-pai / DiffSynth-Studio source):
- control input = concat[control_latents(16), inpaint_mask(1), inpaint_latent(16)]
  = 33 ch → 2x2 patchify → 132, fed to `control_all_x_embedder.2-1` [3840,132].
- branch = 2 control_noise_refiner + 15 control_layers, each a DiT block (same as the
  main `_TransformerBlock`) plus a zero-init `after_proj`; `before_proj` only on the
  first block of each stack.
- each control_layer i emits a hint (its after_proj output); the main DiT adds
  `hint[i] * control_scale` to its layer output, 15 hints spread over 30 main layers
  (interval 2 → main layer L uses hint L//2).

This module builds the branch + loads the 295-key safetensors with key remapping
(checkpoint uses Sequential indices `.2-1`, `to_out.0`, `adaLN_modulation.0` that our
flat modules don't). Forward + main-DiT injection are wired in image_engine.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

# Reuse the main DiT's building blocks (one-way import; image_engine lazy-imports us).
from .image_engine import _DiTAttention, _SwiGLUFFN


class _ControlBlock(nn.Module):
    """A main-style DiT block + after_proj (zero-init residual) and an optional
    before_proj (only the first block of each stack has one)."""

    def __init__(self, dim: int, n_heads: int, eps: float = 1e-5,
                 qk_norm: bool = True, has_before: bool = False):
        super().__init__()
        self.attention = _DiTAttention(dim, n_heads, qk_norm, eps)
        self.feed_forward = _SwiGLUFFN(dim, int(dim / 3 * 8))
        self.attention_norm1 = nn.RMSNorm(dim, eps=eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=eps)
        self.adaLN_modulation = nn.Linear(min(dim, 256), 4 * dim, bias=True)
        self.after_proj = nn.Linear(dim, dim, bias=True)
        self.before_proj = nn.Linear(dim, dim, bias=True) if has_before else None

    def _block(self, c, attn_mask, freqs_cis, t_emb):
        """The main-style DiT block computation (adaLN + attn + FFN)."""
        mod = mx.expand_dims(self.adaLN_modulation(t_emb), axis=1)
        s_msa, g_msa, s_mlp, g_mlp = mx.split(mod, 4, axis=2)
        s_msa = 1.0 + s_msa
        s_mlp = 1.0 + s_mlp
        g_msa = mx.tanh(g_msa)
        g_mlp = mx.tanh(g_mlp)
        c = c + g_msa * self.attention_norm2(
            self.attention(self.attention_norm1(c) * s_msa, attn_mask, freqs_cis))
        c = c + g_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(c) * s_mlp))
        return c

    def __call__(self, c, x, attn_mask, freqs_cis, t_emb):
        """ControlNet stack-forward (verbatim port of ZImageControlTransformerBlock):
        c carries a stack [hint_0..hint_{i-1}, hidden] (axis 0). Block 0 seeds it with
        `before_proj(c)+x` (x = the main DiT hidden); each block appends after_proj(c)
        as its hint. Final hidden is the last stack element."""
        if self.before_proj is not None:        # first block of the stack
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = [c[i] for i in range(c.shape[0])]
            c = all_c.pop(-1)                    # current hidden = last element
        c = self._block(c, attn_mask, freqs_cis, t_emb)
        c_skip = self.after_proj(c)              # this block's hint
        all_c = all_c + [c_skip, c]
        return mx.stack(all_c, axis=0)


class ZImageControlNet(nn.Module):
    """Z-Image-Fun-Controlnet-Union branch: embeds the 132-ch control context and
    runs 2 refiner + 15 control blocks, emitting one hint per control layer."""

    def __init__(self, dim: int = 3840, n_heads: int = 30,
                 control_in_dim: int = 132, n_layers: int = 15,
                 n_refiner_layers: int = 2, eps: float = 1e-5, qk_norm: bool = True):
        super().__init__()
        self.dim = dim
        self.control_x_embedder = nn.Linear(control_in_dim, dim, bias=True)
        self.noise_refiner = [
            _ControlBlock(dim, n_heads, eps, qk_norm, has_before=(i == 0))
            for i in range(n_refiner_layers)
        ]
        self.layers = [
            _ControlBlock(dim, n_heads, eps, qk_norm, has_before=(i == 0))
            for i in range(n_layers)
        ]
        # main DiT layer index → control hint index (only even main layers, 0..28)
        self.control_layers_mapping = {2 * i: i for i in range(n_layers)}

    def forward_refiner(self, control_ctx_emb, x_emb_pre, ctrl_freqs, ctrl_mask, t_emb):
        """Run the 2 control_noise_refiner blocks over the embedded control context
        (block 0 seeds via before_proj(c)+x_emb_pre, the PRE-noise_refiner main hidden;
        shares the image RoPE/mask). Returns (refiner_hints[n_refiner], control_context)
        — the hints inject into the main noise_refiner, control_context feeds the
        layer stack."""
        c = control_ctx_emb
        for layer in self.noise_refiner:
            c = layer(c, x_emb_pre, ctrl_mask, ctrl_freqs, t_emb)
        parts = [c[i] for i in range(c.shape[0])]
        return parts[:-1], parts[-1]  # (refiner_hints, control_context)

    def forward_layers(self, control_context, unified_main, cap_emb,
                       unified_freqs, unified_mask, t_emb):
        """Run the 15 control_layers over [control_context ; caption] (block 0 seeds
        via before_proj(c)+unified_main; shares the unified RoPE/mask). Returns
        n_layers hints, each shaped like the main `unified`."""
        cu = mx.concatenate([control_context, cap_emb], axis=1)
        for layer in self.layers:
            cu = layer(cu, unified_main, unified_mask, unified_freqs, t_emb)
        return [cu[i] for i in range(cu.shape[0] - 1)]


def _remap_controlnet_key(k: str) -> str:
    """Checkpoint key → module attribute path."""
    k = k.replace("control_all_x_embedder.2-1.", "control_x_embedder.")
    k = k.replace("control_noise_refiner.", "noise_refiner.")
    k = k.replace("control_layers.", "layers.")
    k = k.replace("attention.to_out.0.", "attention.to_out.")
    k = k.replace("adaLN_modulation.0.", "adaLN_modulation.")
    return k


def load_zimage_controlnet(path: str, dim: int = 3840, n_heads: int = 30) -> tuple:
    """Build + load the ControlNet from a .safetensors. Returns (model, n_loaded,
    n_total) so callers can assert full coverage."""
    model = ZImageControlNet(dim=dim, n_heads=n_heads)
    raw = mx.load(path)  # handles bf16 safetensors natively
    weights = [(_remap_controlnet_key(k), v) for k, v in raw.items()]
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model, len(weights), len(raw)
