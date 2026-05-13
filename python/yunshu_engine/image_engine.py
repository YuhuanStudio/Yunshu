"""Yunshu Image Generation Engine — MLX-native Z-Image diffusion pipeline.

Self-implemented Z-Image/Flux-family diffusion pipeline, studied from:
- mflux: MLX-native Z-Image port (architecture, weight loading, sampling)
- diffusers: Flow-matching Euler scheduler, VAE, transformer architecture

All Metal GPU computation via MLX — no PyTorch or external package dependency.

Architecture (Z-Image-Turbo, 4-bit quantized):
  Text Encoder: Qwen3-based, 2560-dim, 36 layers, GQA (32 Q / 8 KV), head_dim=128
  Transformer:  Z-Image DiT, 3840-dim, 30 main + 2 noise refiner + 2 context refiner
               adaLN-modulated, SwiGLU FFN, 3D RoPE (32+48+48=128=head_dim)
  VAE:         AutoencoderKL, 16 latent channels, 512-ch decoder, 8x spatial upsample
  Scheduler:   Linear flow-matching with resolution-dependent mu-shift

Weight format (andrevp/Z-Image-Turbo-MLX-4bit):
  - Transformer/TextEncoder: 4-bit quantized (.weight + .scales + .biases)
  - VAE: fp16, OIHW conv format (no transpose needed)
  - Text encoder weights prefixed with 'model.' in safetensors
  - Transformer final layer adaLN: weight index is .1. in safetensors, maps to .0. in model
"""

from __future__ import annotations

import asyncio
import gc
import io
import glob
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .engine import EngineConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Building blocks — studied from mflux, rewritten MLX-native
# ═══════════════════════════════════════════════════════════════════════════════


class _RotaryEmbedding:
    """Precomputed RoPE for text encoder (Qwen3-style, rotate_half).

    Not an nn.Module — inv_freq is computed, not a learned parameter.
    """

    def __init__(self, dim: int, base: float = 1000000.0):
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    def __call__(self, x: mx.array, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        seq_len = position_ids.shape[-1]
        freqs = mx.outer(mx.arange(seq_len, dtype=mx.float32), self.inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)[None, :, :]
        sin = mx.sin(emb)[None, :, :]
        return cos.astype(x.dtype), sin.astype(x.dtype)


class _TextMLP(nn.Module):
    """SwiGLU MLP for text encoder (gate * up → down)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _TextAttention(nn.Module):
    """GQA attention for text encoder with QK norm and RoPE."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)

    def __call__(self, x: mx.array, mask: mx.array | None, pos_emb: tuple | None) -> mx.array:
        B, S, _ = x.shape
        q = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if pos_emb is not None:
            q, k = self._apply_rope(q, k, *pos_emb)
        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=2)
            v = mx.repeat(v, self.num_kv_groups, axis=2)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out)

    @staticmethod
    def _apply_rope(q, k, cos, sin):
        cos = mx.expand_dims(cos, axis=2)
        sin = mx.expand_dims(sin, axis=2)
        def rotate_half(x):
            d = x.shape[-1] // 2
            return mx.concatenate([-x[..., d:], x[..., :d]], axis=-1)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class _TextEncoderLayer(nn.Module):
    """Transformer block for text encoder (pre-norm, GQA, SwiGLU)."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, intermediate_size, head_dim, eps):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=eps)
        self.self_attn = _TextAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.mlp = _TextMLP(hidden_size, intermediate_size)

    def __call__(self, x, mask, pos_emb):
        r = x
        x = r + self.self_attn(self.input_layernorm(x), mask, pos_emb)
        r = x
        x = r + self.mlp(self.post_attention_layernorm(x))
        return x


class TextEncoder(nn.Module):
    """Qwen3-based text encoder for Z-Image.

    Returns second-to-last hidden state (like mflux's text_encoder.py).
    """

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2560,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        intermediate_size=9728,
        head_dim=128,
        rope_theta=1000000.0,
        eps=1e-6,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            _TextEncoderLayer(hidden_size, num_heads, num_kv_heads, intermediate_size, head_dim, eps)
            for _ in range(num_layers)
        ]
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.rotary_emb = _RotaryEmbedding(dim=head_dim, base=rope_theta)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        B, S = input_ids.shape
        h = self.embed_tokens(input_ids).astype(mx.float32)
        pos_ids = mx.broadcast_to(mx.arange(S, dtype=mx.int32)[None, :], (B, S))
        pos_emb = self.rotary_emb(h, pos_ids)
        causal = self._causal_mask(S, h.dtype)
        if attention_mask is not None:
            pad = mx.where(
                attention_mask[:, None, None, :] == 1,
                mx.zeros((B, 1, 1, S), dtype=h.dtype),
                mx.full((B, 1, 1, S), float("-inf"), dtype=h.dtype),
            )
            causal = causal + pad
        all_hidden = [h]
        for layer in self.layers:
            h = layer(h, causal, pos_emb)
            all_hidden.append(h)
        # Return second-to-last layer output (mflux pattern)
        return all_hidden[-2]

    @staticmethod
    def _causal_mask(seq_len, dtype):
        idx = mx.arange(seq_len, dtype=mx.int32)
        mask = idx[:, None] >= idx[None, :]
        return mx.where(
            mask,
            mx.zeros((seq_len, seq_len), dtype=dtype),
            mx.full((seq_len, seq_len), float("-inf"), dtype=dtype),
        )[None, None, :, :]


# ── Transformer components ──


class _SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward for transformer blocks (w1*silu*w3 → w2)."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class _DiTAttention(nn.Module):
    """Multi-head attention with QK norm and 3D RoPE for DiT."""

    def __init__(self, dim, n_heads, qk_norm=True, eps=1e-5):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)  # Was [nn.Linear], now flat
        if qk_norm:
            self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
            self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)
        else:
            self.norm_q = None
            self.norm_k = None

    def __call__(self, x, mask=None, freqs_cis=None):
        B, S, _ = x.shape
        q = self.to_q(x).reshape(B, S, self.n_heads, self.head_dim)
        k = self.to_k(x).reshape(B, S, self.n_heads, self.head_dim)
        v = self.to_v(x).reshape(B, S, self.n_heads, self.head_dim)
        if self.norm_q is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)
        if freqs_cis is not None:
            q = self._apply_rotary(q, freqs_cis)
            k = self._apply_rotary(k, freqs_cis)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        attn_mask = None
        if mask is not None:
            attn_mask = mx.where(mask[:, None, None, :], mx.array(0.0), mx.array(float("-inf")))
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=attn_mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.to_out(out)

    @staticmethod
    def _apply_rotary(x, freqs_cis):
        B, S, H, D = x.shape
        x = x.reshape(B, S, H, D // 2, 2)
        fc = mx.expand_dims(mx.expand_dims(freqs_cis, 0), 2)
        xr, xi = x[..., 0], x[..., 1]
        cr, ci = fc[..., 0], fc[..., 1]
        out = mx.stack([xr * cr - xi * ci, xr * ci + xi * cr], axis=-1)
        return out.reshape(B, S, H, D)


class _TransformerBlock(nn.Module):
    """Main DiT block with adaLN modulation."""

    def __init__(self, dim, n_heads, eps=1e-5, qk_norm=True):
        super().__init__()
        self.attention = _DiTAttention(dim, n_heads, qk_norm, eps)
        self.feed_forward = _SwiGLUFFN(dim, int(dim / 3 * 8))
        self.attention_norm1 = nn.RMSNorm(dim, eps=eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=eps)
        self.adaLN_modulation = nn.Linear(min(dim, 256), 4 * dim, bias=True)

    def __call__(self, x, attn_mask, freqs_cis, t_emb):
        mod = mx.expand_dims(self.adaLN_modulation(t_emb), axis=1)
        s_msa, g_msa, s_mlp, g_mlp = mx.split(mod, 4, axis=2)
        s_msa = 1.0 + s_msa
        s_mlp = 1.0 + s_mlp
        g_msa = mx.tanh(g_msa)
        g_mlp = mx.tanh(g_mlp)
        x = x + g_msa * self.attention_norm2(self.attention(self.attention_norm1(x) * s_msa, attn_mask, freqs_cis))
        x = x + g_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * s_mlp))
        return x


class _ContextBlock(nn.Module):
    """Context refiner block (no adaLN, no timestep conditioning)."""

    def __init__(self, dim, n_heads, eps=1e-5, qk_norm=True):
        super().__init__()
        self.attention = _DiTAttention(dim, n_heads, qk_norm, eps)
        self.feed_forward = _SwiGLUFFN(dim, int(dim / 3 * 8))
        self.attention_norm1 = nn.RMSNorm(dim, eps=eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=eps)

    def __call__(self, x, attn_mask, freqs_cis):
        x = x + self.attention_norm2(self.attention(self.attention_norm1(x), attn_mask, freqs_cis))
        x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class _TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding → MLP."""

    def __init__(self, out_size: int, mid_size: int = 1024, freq_size: int = 256):
        super().__init__()
        self.freq_size = freq_size
        self.linear1 = nn.Linear(freq_size, mid_size, bias=True)
        self.linear2 = nn.Linear(mid_size, out_size, bias=True)

    def __call__(self, t: mx.array) -> mx.array:
        t_freq = self._sinusoidal(t, self.freq_size)
        return self.linear2(nn.silu(self.linear1(t_freq)))

    @staticmethod
    def _sinusoidal(t: mx.array, dim: int, max_period: float = 10000.0) -> mx.array:
        half = dim // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half)
        args = t[:, None].astype(mx.float32) * freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
        return emb


class _FinalLayer(nn.Module):
    """adaLN-modulated final projection layer."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Linear(min(hidden_size, 256), hidden_size, bias=True)

    def __call__(self, x, c):
        scale = 1.0 + self.adaLN_modulation(nn.silu(c))
        scale = mx.expand_dims(scale, axis=1)
        x = self.norm(x) * scale
        return self.linear(x)


class _RopeEmbedder:
    """3D RoPE precomputed frequencies for spatial + temporal positions."""

    def __init__(self, theta=256.0, axes_dims=None, axes_lens=None):
        axes_dims = axes_dims or [32, 48, 48]
        axes_lens = axes_lens or [1536, 512, 512]
        self.axes_dims = axes_dims
        self.freqs_cis = []
        for d, e in zip(axes_dims, axes_lens):
            freqs = 1.0 / (theta ** (mx.arange(0, d, 2, dtype=mx.float32) / d))
            t = mx.arange(e, dtype=mx.float32)
            f = mx.outer(t, freqs)
            self.freqs_cis.append(mx.stack([mx.cos(f), mx.sin(f)], axis=-1))

    def __call__(self, ids: mx.array) -> mx.array:
        result = []
        for i in range(len(self.axes_dims)):
            idx = ids[:, i].astype(mx.int32)
            result.append(self.freqs_cis[i][idx])
        return mx.concatenate(result, axis=1)


class ZImageTransformer(nn.Module):
    """Full Z-Image DiT transformer with noise/context refiners."""

    def __init__(
        self,
        patch_size=2,
        f_patch_size=1,
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=None,
        axes_lens=None,
    ):
        super().__init__()
        axes_dims = axes_dims or [32, 48, 48]
        axes_lens = axes_lens or [1536, 512, 512]
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.f_patch_size = f_patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.t_scale = t_scale

        key = f"{patch_size}-{f_patch_size}"
        embed_dim = f_patch_size * patch_size * patch_size * in_channels
        self._embed_key = key
        # Store as direct attribute with the key name (e.g., all_x_embedder_2_1)
        # so MLX load_weights can find them by name pattern "all_x_embedder.2-1.*"
        setattr(self, f"all_x_embedder", nn.Linear(embed_dim, dim, bias=True))
        setattr(self, f"all_final_layer", _FinalLayer(dim, embed_dim))

        self.t_embedder = _TimestepEmbedder(out_size=min(dim, 256), mid_size=1024)
        # Use named sub-modules instead of list for MLX weight loading
        class _CapEmbedder(nn.Module):
            def __init__(self, feat_dim, dim, eps):
                super().__init__()
                self.norm = nn.RMSNorm(feat_dim, eps=eps)
                self.proj = nn.Linear(feat_dim, dim, bias=True)
        self.cap_embedder = _CapEmbedder(cap_feat_dim, dim, norm_eps)
        self.x_pad_token = mx.zeros((1, dim))
        self.cap_pad_token = mx.zeros((1, dim))

        self.noise_refiner = [_TransformerBlock(dim, n_heads, norm_eps, qk_norm) for _ in range(n_refiner_layers)]
        self.context_refiner = [_ContextBlock(dim, n_heads, norm_eps, qk_norm) for _ in range(n_refiner_layers)]
        self.layers = [_TransformerBlock(dim, n_heads, norm_eps, qk_norm) for _ in range(n_layers)]
        self.rope_embedder = _RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

    def __call__(self, x, timestep, sigmas, cap_feats):
        # Time embedding
        if not isinstance(timestep, mx.array):
            if isinstance(timestep, int):
                sigma_t = sigmas[timestep].reshape((1,))
                timestep = mx.ones_like(sigma_t) - sigma_t
            else:
                timestep = mx.array(timestep, dtype=mx.float32)
        if timestep.ndim == 0:
            timestep = timestep.reshape((1,))
        t_emb = self.t_embedder(timestep.astype(mx.float32) * self.t_scale)

        # Patchify
        x_emb, cap_emb, x_size, x_pos, cap_pos, x_pad, cap_pad = self._patchify(x, cap_feats)

        # Image embedding
        x_emb = self.all_x_embedder(x_emb)
        x_emb = mx.where(x_pad[:, None], self.x_pad_token, x_emb)
        x_freqs = self.rope_embedder(x_pos)
        x_mask = mx.ones((1, x_emb.shape[0]), dtype=mx.bool_)
        x_emb = mx.expand_dims(x_emb, axis=0)

        # Noise refiner
        for layer in self.noise_refiner:
            x_emb = layer(x_emb, x_mask, x_freqs, t_emb)

        # Caption embedding
        cap_emb = self.cap_embedder.proj(self.cap_embedder.norm(cap_emb))
        cap_emb = mx.where(cap_pad[:, None], self.cap_pad_token, cap_emb)
        cap_freqs = self.rope_embedder(cap_pos)
        cap_mask = mx.ones((1, cap_emb.shape[0]), dtype=mx.bool_)
        cap_emb = mx.expand_dims(cap_emb, axis=0)

        # Context refiner
        for layer in self.context_refiner:
            cap_emb = layer(cap_emb, cap_mask, cap_freqs)

        # Unify and main layers
        x_len = x_emb.shape[1]
        unified = mx.concatenate([x_emb, cap_emb], axis=1)
        unified_freqs = mx.concatenate([x_freqs, cap_freqs], axis=0)
        unified_mask = mx.ones((1, unified.shape[1]), dtype=mx.bool_)

        for layer in self.layers:
            unified = layer(unified, unified_mask, unified_freqs, t_emb)

        # Final layer and unpatchify
        unified = self.all_final_layer(unified, t_emb)
        output = self._unpatchify(unified[0, :x_len], x_size)
        return -output

    def _patchify(self, image, cap_feats):
        pH = pW = self.patch_size
        pF = self.f_patch_size

        # Caption padding to multiple of 32
        cap_len = cap_feats.shape[0]
        cap_pad_len = (-cap_len) % 32
        cap_pos = self._coord_grid((cap_len + cap_pad_len, 1, 1), (1, 0, 0)).reshape(-1, 3)
        cap_pad_mask = mx.concatenate([mx.zeros((cap_len,), dtype=mx.bool_), mx.ones((cap_pad_len,), dtype=mx.bool_)])
        if cap_pad_len > 0:
            cap_padded = mx.concatenate([cap_feats, mx.repeat(cap_feats[-1:], cap_pad_len, axis=0)], axis=0)
        else:
            cap_padded = cap_feats

        # Image patchification
        C, F, H, W = image.shape
        image_size = (F, H, W)
        Ft, Ht, Wt = F // pF, H // pH, W // pW
        img = image.reshape(C, Ft, pF, Ht, pH, Wt, pW)
        img = img.transpose(1, 3, 5, 2, 4, 6, 0)
        img = img.reshape(Ft * Ht * Wt, pF * pH * pW * C)

        # Image padding to multiple of 32
        img_len = img.shape[0]
        img_pad_len = (-img_len) % 32
        img_pos = self._coord_grid((Ft, Ht, Wt), (cap_len + cap_pad_len + 1, 0, 0)).reshape(-1, 3)
        if img_pad_len > 0:
            img_pos = mx.concatenate([img_pos, mx.zeros((img_pad_len, 3), dtype=mx.int32)], axis=0)
            img = mx.concatenate([img, mx.repeat(img[-1:], img_pad_len, axis=0)], axis=0)
        img_pad_mask = mx.concatenate([mx.zeros((img_len,), dtype=mx.bool_), mx.ones((img_pad_len,), dtype=mx.bool_)])

        return img, cap_padded, image_size, img_pos, cap_pos, img_pad_mask, cap_pad_mask

    def _unpatchify(self, x, size):
        pH = pW = self.patch_size
        pF = self.f_patch_size
        F, H, W = size
        ori_len = (F // pF) * (H // pH) * (W // pW)
        x = x[:ori_len].reshape(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
        x = x.transpose(6, 0, 3, 1, 4, 2, 5)
        return x.reshape(self.out_channels, F, H, W)

    @staticmethod
    def _coord_grid(size, start=None):
        start = start or tuple(0 for _ in size)
        axes = [mx.arange(x0, x0 + span, dtype=mx.int32) for x0, span in zip(start, size)]
        grids = mx.meshgrid(*axes, indexing="ij")
        return mx.stack(grids, axis=-1)


# ── VAE Decoder ──


class _ResnetBlock2D(nn.Module):
    """ResNet block: GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv + shortcut."""

    def __init__(self, in_ch, out_ch, use_conv_shortcut=False):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch, eps=1e-6, affine=True, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch, eps=1e-6, affine=True, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        if use_conv_shortcut or in_ch != out_ch:
            self.conv_shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
        else:
            self.conv_shortcut = None

    def __call__(self, x):
        # x is (B, C, H, W) in NCHW
        x_nhwc = x.transpose(0, 2, 3, 1)
        h = self.norm1(x_nhwc)
        h = nn.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nn.silu(h)
        h = self.conv2(h)
        shortcut = x_nhwc
        if self.conv_shortcut is not None:
            shortcut = self.conv_shortcut(x_nhwc)
        return (shortcut + h).transpose(0, 3, 1, 2)


class _VAEAttention(nn.Module):
    """Self-attention in VAE mid-block."""

    def __init__(self, channels=512):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels, eps=1e-6, affine=True, pytorch_compatible=True)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Linear(channels, channels)

    def __call__(self, x):
        # x: (B, C, H, W)
        xw = x.transpose(0, 2, 3, 1)
        B, H, W, C = xw.shape
        h = self.group_norm(xw)
        q = self.to_q(h).reshape(B, H * W, 1, C).transpose(0, 2, 1, 3)
        k = self.to_k(h).reshape(B, H * W, 1, C).transpose(0, 2, 1, 3)
        v = self.to_v(h).reshape(B, H * W, 1, C).transpose(0, 2, 1, 3)
        scale = 1.0 / mx.sqrt(mx.array(C, dtype=mx.float32))
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, H, W, C)
        out = self.to_out(out)
        return (xw + out).transpose(0, 3, 1, 2)


class _UpSampler(nn.Module):
    """Nearest-neighbor 2x upsample + conv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x):
        xw = x.transpose(0, 2, 3, 1)
        B, H, W, C = xw.shape
        xw = mx.broadcast_to(xw[:, :, None, :, None, :], (B, H, 2, W, 2, C))
        xw = xw.reshape(B, H * 2, W * 2, C)
        out = self.conv(xw)
        return out.transpose(0, 3, 1, 2)


class _UpDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_layers=3, add_upsample=True):
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            use_sc = (i == 0) and (in_ch != out_ch)
            self.resnets.append(_ResnetBlock2D(
                in_ch if i == 0 else out_ch, out_ch, use_conv_shortcut=use_sc
            ))
        # Must be named "upsamplers" as list to match safetensors key structure
        self.upsamplers = [_UpSampler(out_ch, out_ch)] if add_upsample else None

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = upsampler(x)
        return x


class _UNetMidBlock(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.attentions = [_VAEAttention(channels)]
        self.resnets = [
            _ResnetBlock2D(channels, channels),
            _ResnetBlock2D(channels, channels),
        ]

    def __call__(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class VAEDecoder(nn.Module):
    """Z-Image VAE decoder: ConvIn → MidBlock → 4 UpBlocks → ConvOut."""

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(16, 512, kernel_size=3, stride=1, padding=1)
        self.mid_block = _UNetMidBlock(512)
        self.up_blocks = [
            _UpDecoderBlock(512, 512, 3, add_upsample=True),
            _UpDecoderBlock(512, 512, 3, add_upsample=True),
            _UpDecoderBlock(512, 256, 3, add_upsample=True),
            _UpDecoderBlock(256, 128, 3, add_upsample=False),
        ]
        self.conv_norm_out = nn.GroupNorm(32, 128, eps=1e-6, affine=True, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)

    def __call__(self, latents):
        # latents: (B, 16, H, W) in NCHW — conv weights are already OIHW
        h = self.conv_in(latents.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        h = self.mid_block(h)
        for block in self.up_blocks:
            h = block(h)
        h = self.conv_norm_out(h.transpose(0, 2, 3, 1))
        h = nn.silu(h)
        h = self.conv_out(h)
        return h.transpose(0, 3, 1, 2)


class VAE:
    """VAE wrapper with scaling constants."""

    scaling_factor = 0.3611
    shift_factor = 0.1159

    def __init__(self, decoder: VAEDecoder):
        self.decoder = decoder

    def decode(self, latents: mx.array) -> mx.array:
        # latents shape: (C, F, H, W) or (C, H, W)
        # Squeeze the F dimension if present
        if latents.ndim == 4:
            # (C, F, H, W) → (C, H, W)
            latents = latents[:, 0, :, :]
        elif latents.ndim == 5:
            # (B, C, F, H, W) → (B, C, H, W)
            latents = latents[:, :, 0, :, :]
        scaled = (latents / self.scaling_factor) + self.shift_factor
        # Add batch dim: (C, H, W) → (1, C, H, W)
        if scaled.ndim == 3:
            scaled = mx.expand_dims(scaled, axis=0)
        decoded = self.decoder(scaled)
        return decoded


# ═══════════════════════════════════════════════════════════════════════════════
# Weight loading — maps safetensors keys to our nn.Module structure
# ═══════════════════════════════════════════════════════════════════════════════


def _load_component_weights(model_path: Path, subdir: str) -> dict:
    """Load all safetensors from a subdirectory into a flat dict."""
    weights = {}
    for f in sorted(glob.glob(str(model_path / subdir / "*.safetensors"))):
        weights.update(mx.load(f))
    return weights


def _dequantize_weights(pairs: list[tuple[str, mx.array]], group_size: int = 64, bits: int = 4) -> list[tuple[str, mx.array]]:
    """Dequantize any quantized weight triplets (weight+scales+biases) to float16.

    Takes a list of (key, value) pairs. For any .weight key that is uint32
    (quantized), finds the corresponding .scales and .biases, dequantizes,
    and returns only float16 weights (drops .scales/.biases keys).
    """
    # Index by key for fast lookup
    by_key = {}
    for k, v in pairs:
        by_key[k] = v

    result = []
    processed = set()
    for k, v in pairs:
        if k in processed:
            continue
        if k.endswith(".scales") or k.endswith(".biases"):
            processed.add(k)
            continue
        if k.endswith(".weight") and v.dtype == mx.uint32:
            # Found quantized weight — dequantize
            prefix = k[:-len(".weight")]
            scales = by_key.get(f"{prefix}.scales")
            biases = by_key.get(f"{prefix}.biases")
            if scales is not None:
                deq = mx.dequantize(v, scales, biases, group_size, bits)
                bias_key = f"{prefix}.bias"
                bias_val = by_key.get(bias_key)
                result.append((k, deq.astype(mx.float16)))
                if bias_val is not None:
                    result.append((bias_key, bias_val))
                processed.add(k)
                processed.add(f"{prefix}.scales")
                processed.add(f"{prefix}.biases")
                if bias_val is not None:
                    processed.add(bias_key)
                continue
        # Regular (non-quantized) weight — pass through
        result.append((k, v))

    return result


def _load_weights_into(model: nn.Module, weight_pairs: list[tuple[str, mx.array]]) -> None:
    """Load weight pairs into a model by building nested dict and calling update().

    Works around MLX tree_unflatten recursion limit for large models.
    """
    tree = _build_nested_dict(weight_pairs)
    model.update(tree, strict=False)
    mx.eval(model.parameters())


def _build_nested_dict(pairs: list[tuple[str, mx.array]]) -> dict:
    """Build a nested dict from flat (key, value) pairs.

    Handles list-indexed keys (e.g., "layers.0." → list[0]).
    """
    root: dict = {}
    for key, val in pairs:
        parts = key.split(".")
        d = root
        for i, part in enumerate(parts[:-1]):
            next_d = d.get(part)
            if next_d is None:
                next_d = {}
                d[part] = next_d
            d = next_d
        d[parts[-1]] = val

    # Convert dict-indexed lists to actual lists
    _convert_to_lists(root)
    return root


def _convert_to_lists(d: dict) -> None:
    """Recursively convert dict keys that are all numeric strings to lists."""
    for key, val in list(d.items()):
        if isinstance(val, dict):
            # Check if all keys are numeric (list indices)
            if val and all(k.isdigit() for k in val.keys()):
                indices = sorted((int(k), k) for k in val.keys())
                lst = [None] * (indices[-1][0] + 1)
                for idx, str_idx in indices:
                    item = val[str_idx]
                    if isinstance(item, dict):
                        _convert_to_lists(item)
                    lst[idx] = item
                d[key] = lst
            else:
                _convert_to_lists(val)


def _remap_text_encoder_weights(raw: dict) -> list[tuple[str, mx.array]]:
    """Remap text encoder weights: 'model.X' → 'X'."""
    pairs = []
    for key, val in raw.items():
        new_key = key
        if key.startswith("model."):
            new_key = key[len("model."):]
        pairs.append((new_key, val))
    return pairs


def _remap_transformer_weights(raw: dict) -> list[tuple[str, mx.array]]:
    """Remap transformer weights.

    Key differences from safetensors → our model:
    - t_embedder.mlp.{0,2} → t_embedder.linear{1,2}
    - cap_embedder.{0,1} stays same
    - all_final_layer.2-1.adaLN_modulation.{1 → 0} (index shift)
    - all_x_embedder.2-1.* → all_x_embedder.*
    - all_final_layer.2-1.* → all_final_layer.*
    - x_pad_token / cap_pad_token — skipped (handled separately)
    """
    pairs = []
    for key, val in raw.items():
        # Skip pad tokens — handled separately after load
        if key.startswith("x_pad_token") or key.startswith("cap_pad_token"):
            continue
        new_key = key
        # Timestep embedder: mlp indices → linear indices
        if key.startswith("t_embedder.mlp.0."):
            new_key = key.replace("t_embedder.mlp.0.", "t_embedder.linear1.")
        elif key.startswith("t_embedder.mlp.2."):
            new_key = key.replace("t_embedder.mlp.2.", "t_embedder.linear2.")
        # Cap embedder: list indices → named modules
        elif key.startswith("cap_embedder.0."):
            new_key = key.replace("cap_embedder.0.", "cap_embedder.norm.")
        elif key.startswith("cap_embedder.1."):
            new_key = key.replace("cap_embedder.1.", "cap_embedder.proj.")
        # Flatten to_out.0.* → to_out.*
        elif ".to_out.0." in key:
            new_key = key.replace(".to_out.0.", ".to_out.")
        # Flatten adaLN_modulation.0.* → adaLN_modulation.*
        elif ".adaLN_modulation.0." in key:
            new_key = key.replace(".adaLN_modulation.0.", ".adaLN_modulation.")
        # Flatten all_x_embedder.2-1.* → all_x_embedder.*
        elif key.startswith("all_x_embedder.2-1."):
            new_key = key.replace("all_x_embedder.2-1.", "all_x_embedder.")
        # Flatten all_final_layer.2-1.* → all_final_layer.*
        elif key.startswith("all_final_layer.2-1."):
            new_key = key.replace("all_final_layer.2-1.", "all_final_layer.")
            # adaLN index .1. → flatten (model has flat nn.Linear, not list)
            if ".adaLN_modulation.1." in new_key:
                new_key = new_key.replace(".adaLN_modulation.1.", ".adaLN_modulation.")
        pairs.append((new_key, val))
    return pairs


def _remap_vae_weights(raw: dict) -> list[tuple[str, mx.array]]:
    """Remap VAE weights.

    VAE safetensors weights are in PyTorch OIHW format: (out, in, kH, kW)
    MLX Conv2d expects OHWI format: (out, kH, kW, in)
    So conv weights need transpose: (0, 2, 3, 1)
    """
    decoder_weights = []
    for key, val in raw.items():
        if not key.startswith("decoder."):
            continue
        local_key = key[len("decoder."):]
        # Flatten to_out.0.* → to_out.*
        if ".to_out.0." in local_key:
            local_key = local_key.replace(".to_out.0.", ".to_out.")
        # Transpose conv weights from OIHW → OHWI (MLX format)
        is_conv_weight = (
            local_key.endswith(".weight") and val.ndim == 4 and (
                ".conv" in local_key or
                local_key.startswith("conv_") or
                local_key.startswith("conv.") or
                "/conv" in local_key
            )
        )
        if is_conv_weight:
            val = val.transpose(0, 2, 3, 1)
        decoder_weights.append((local_key, val))

    return decoder_weights


def _quantize_model(model, quant_config: dict, weight_keys: set[str] | None = None):
    """Apply 4-bit quantization to compatible layers.

    Only quantizes layers that have corresponding .scales keys in the weight file.
    """
    group_size = quant_config.get("group_size", 64)
    bits = quant_config.get("bits", 4)

    def predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        # Skip norm layers and embeddings
        if isinstance(module, (nn.RMSNorm, nn.LayerNorm, nn.Embedding)):
            return False
        # Skip if weight size not divisible by group_size
        if hasattr(module, "weight") and module.weight.size % group_size != 0:
            return False
        # Only quantize if .scales exists for this key in the weight file
        if weight_keys is not None:
            return f"{path}.scales" in weight_keys
        return True

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)


# ═══════════════════════════════════════════════════════════════════════════════
# Scheduler — Linear flow-matching with mu-shift
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_sigmas(
    num_steps: int,
    width: int = 1024,
    height: int = 1024,
    sigma_base_shift: float = 0.5,
    sigma_max_shift: float = 1.15,
    sigma_base_seq_len: int = 256,
    sigma_max_seq_len: int = 4096,
    requires_sigma_shift: bool = True,
) -> mx.array:
    """Compute sigma schedule for flow-matching Euler.

    For Turbo models: linear spacing with resolution-dependent mu-shift.
    """
    sigmas = mx.linspace(1.0, 1.0 / num_steps, num_steps).astype(mx.float32)

    if requires_sigma_shift:
        m = (sigma_max_shift - sigma_base_shift) / (sigma_max_seq_len - sigma_base_seq_len)
        b = sigma_base_shift - m * sigma_base_seq_len
        mu = m * width * height / 256 + b
        mu = mx.array(mu)
        shifted = mx.exp(mu) / (mx.exp(mu) + (1.0 / sigmas - 1.0))
        sigmas = shifted

    return mx.concatenate([sigmas, mx.zeros(1)])


# ═══════════════════════════════════════════════════════════════════════════════
# Image Engine — full pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class ImageGenEngine:
    """MLX-native image generation engine.

    Pipeline stages (all MLX-native):
    1. Tokenize prompt → token IDs (transformers tokenizer)
    2. Text encoder → cap_feats embedding (Qwen3, second-to-last layer)
    3. Flow-matching noise schedule → latent timesteps
    4. Transformer denoising loop → clean latents
    5. VAE decode → pixel image
    """

    def __init__(self, model_path: str, config: EngineConfig | None = None) -> None:
        self._model_path = model_path
        self._text_encoder: TextEncoder | None = None
        self._transformer: ZImageTransformer | None = None
        self._vae: VAE | None = None
        self._tokenizer = None
        self._tf_config: dict = {}
        self._running = False
        from .mlx_executor import get_mlx_executor
        self._executor = get_mlx_executor()

        # DFlash Block Diffusion (opt-in via YUNSHU_DFLASH=1)
        self._dflash = None
        import os
        if os.environ.get("YUNSHU_DFLASH", "").strip() in ("1", "true", "yes"):
            from .dflash import DFlashEngine, DFlashConfig
            self._dflash = DFlashEngine(DFlashConfig.from_env())
            logger.info("DFlash Block Diffusion enabled")

    @property
    def model_name(self) -> str:
        return self._model_path.rsplit("/", 1)[-1] if "/" in self._model_path else self._model_path

    @property
    def is_loaded(self) -> bool:
        return self._transformer is not None

    def load(self) -> None:
        """Load all pipeline components."""
        model_path = Path(self._model_path)

        # Load configs
        tf_config = json.loads((model_path / "transformer" / "config.json").read_text())
        te_config = json.loads((model_path / "text_encoder" / "config.json").read_text())
        vae_config = json.loads((model_path / "vae" / "config.json").read_text())
        self._tf_config = tf_config

        # 1. Text Encoder — quantize model, then load pre-quantized weights
        logger.info("Loading text encoder...")
        text_encoder = TextEncoder(
            vocab_size=te_config.get("vocab_size", 151936),
            hidden_size=te_config.get("hidden_size", 2560),
            num_layers=te_config.get("num_hidden_layers", 36),
            num_heads=te_config.get("num_attention_heads", 32),
            num_kv_heads=te_config.get("num_key_value_heads", 8),
            intermediate_size=te_config.get("intermediate_size", 9728),
            head_dim=te_config.get("head_dim", 128),
            rope_theta=te_config.get("rope_theta", 1000000.0),
            eps=te_config.get("rms_norm_eps", 1e-6),
        )
        te_weights = _load_component_weights(model_path, "text_encoder")
        te_quant = te_config.get("quantization", {})
        if te_quant:
            _quantize_model(text_encoder, te_quant, set(dict(_remap_text_encoder_weights(te_weights)).keys()))
        text_encoder.load_weights(_remap_text_encoder_weights(te_weights), strict=False)
        mx.eval(text_encoder.parameters())
        text_encoder.eval()
        self._text_encoder = text_encoder
        logger.info(f"Text encoder loaded: {len(te_weights)} tensors")

        # 2. Transformer — dequantize quantized weights, load as float16
        logger.info("Loading transformer...")
        transformer = ZImageTransformer(
            patch_size=tf_config.get("all_patch_size", [2])[0],
            f_patch_size=tf_config.get("all_f_patch_size", [1])[0],
            in_channels=tf_config.get("in_channels", 16),
            dim=tf_config.get("dim", 3840),
            n_layers=tf_config.get("n_layers", 30),
            n_refiner_layers=tf_config.get("n_refiner_layers", 2),
            n_heads=tf_config.get("n_heads", 30),
            norm_eps=tf_config.get("norm_eps", 1e-5),
            qk_norm=tf_config.get("qk_norm", True),
            cap_feat_dim=tf_config.get("cap_feat_dim", 2560),
            rope_theta=tf_config.get("rope_theta", 256.0),
            t_scale=tf_config.get("t_scale", 1000.0),
            axes_dims=tf_config.get("axes_dims"),
            axes_lens=tf_config.get("axes_lens"),
        )
        tf_weights = _load_component_weights(model_path, "transformer")
        remapped = _remap_transformer_weights(tf_weights)
        # Dequantize quantized weights to float16 (avoids QuantizedLinear list issues)
        tf_pairs = _dequantize_weights(remapped)
        _load_weights_into(transformer, tf_pairs)
        # Handle pad tokens
        for token_name in ("x_pad_token", "cap_pad_token"):
            if token_name in tf_weights:
                packed = tf_weights[token_name]
                scales = tf_weights.get(f"{token_name}.scales")
                biases = tf_weights.get(f"{token_name}.biases")
                if scales is not None and packed.dtype == mx.uint32:
                    deq = mx.dequantize(packed, scales, biases, 64, 4)
                    setattr(transformer, token_name, deq.reshape(1, -1))
                else:
                    setattr(transformer, token_name, packed)
        mx.eval(transformer.parameters())
        transformer.eval()
        self._transformer = transformer
        logger.info(f"Transformer loaded: {len(tf_weights)} tensors")

        # 3. VAE Decoder (no quantization per quantize_config.json)
        logger.info("Loading VAE decoder...")
        vae_decoder = VAEDecoder()
        vae_weights = _load_component_weights(model_path, "vae")
        _load_weights_into(vae_decoder, _remap_vae_weights(vae_weights))
        vae_decoder.eval()
        self._vae = VAE(vae_decoder)
        logger.info(f"VAE loaded: {len(vae_weights)} tensors")

        # 4. Tokenizer
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path / "tokenizer"), trust_remote_code=True
        )

        logger.info("Image gen pipeline fully loaded")

    async def start(self) -> None:
        if self._transformer is not None:
            self._running = True
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self.load)
        self._running = True

    async def stop(self) -> None:
        self._text_encoder = None
        self._transformer = None
        self._vae = None
        self._tokenizer = None
        self._running = False
        gc.collect()
        loop = asyncio.get_running_loop()
        from .mlx_executor import sync_and_clear_cache
        await loop.run_in_executor(self._executor, sync_and_clear_cache)

    def resolve_model_id(self, model_id: str) -> bool:
        return model_id in {
            self.model_name, self._model_path,
            self.model_name.lower(), self._model_path.lower(),
        }

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,  # Turbo: no guidance
        seed: int | None = None,
        **kwargs,
    ) -> bytes:
        """Generate an image from a text prompt, returns PNG bytes."""
        if self._transformer is None:
            raise RuntimeError("Engine not started")

        # Memory estimate: latent 16×H/8×W/8×2 bytes + transformer activations
        est_bytes = width * height * 2 * 16  # conservative estimate
        try:
            import mlx.core as mx
            active = mx.get_active_memory()
            total_uma = 0
            import subprocess
            r = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
            )
            total_uma = int(r.stdout.strip())
            if total_uma > 0 and (active + est_bytes) > total_uma * 0.9:
                raise MemoryError(
                    f"Insufficient GPU memory for {width}x{height} image "
                    f"(need ~{est_bytes // 1024 // 1024}MB, "
                    f"available ~{(total_uma - active) // 1024 // 1024}MB)"
                )
        except (OSError, ValueError):
            pass

        t0 = time.monotonic()

        def _generate_sync() -> bytes:
            return self._run_pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_steps=num_inference_steps,
                seed=seed if seed is not None else 42,
            )

        try:
            loop = asyncio.get_running_loop()
            png_bytes = await loop.run_in_executor(self._executor, _generate_sync)
        except MemoryError as e:
            raise MemoryError(f"GPU OOM during image generation: {e}") from e

        elapsed = time.monotonic() - t0
        logger.info(f"Image gen: {elapsed:.2f}s, {len(png_bytes)} bytes, prompt='{prompt[:50]}...'")
        return png_bytes

    async def generate_image_stream(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: int | None = None,
        **kwargs,
    ):
        """Stream image generation progress, yielding step-by-step updates.

        Each yielded dict has:
        - "step": Current step number
        - "total_steps": Total number of steps
        - "progress": 0.0 to 1.0
        - "image": PNG bytes of current latent (decoded preview) or None
        - "is_final": True for the last chunk
        """
        if self._transformer is None:
            raise RuntimeError("Engine not started")

        queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=64)

        def _stream_sync():
            try:
                tokenizer = self._tokenizer
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=True,
                )
                tokens = tokenizer(
                    [formatted], padding="max_length", max_length=512,
                    truncation=True, return_tensors="np",
                )
                input_ids = mx.array(tokens["input_ids"])
                attention_mask = mx.array(tokens["attention_mask"])

                cap_feats = self._text_encoder(input_ids, attention_mask)
                num_valid = int(mx.sum(attention_mask[0]).item())
                cap_feats = cap_feats[0, :num_valid, :]
                mx.eval(cap_feats)

                latent_h = height // 8
                latent_w = width // 8
                latents = mx.random.normal(
                    shape=[16, 1, latent_h, latent_w],
                    key=mx.random.key(seed or 42),
                ).astype(mx.float16)

                sigmas = _compute_sigmas(num_inference_steps, width, height)

                for t in range(num_inference_steps):
                    sigma_t = sigmas[t].reshape((1,))
                    timestep = mx.ones_like(sigma_t) - sigma_t
                    noise_pred = self._transformer(
                        x=latents, timestep=timestep, sigmas=sigmas, cap_feats=cap_feats,
                    )
                    dt = sigmas[t + 1] - sigmas[t]
                    latents = latents + noise_pred * dt
                    mx.eval(latents)

                    progress = (t + 1) / num_inference_steps
                    queue.put_nowait({
                        "step": t + 1,
                        "total_steps": num_inference_steps,
                        "progress": progress,
                        "image": None,
                        "is_final": False,
                    })

                # Final decode
                image = self._vae.decode(latents)
                mx.eval(image)
                png = self._to_png(image)
                queue.put_nowait({
                    "step": num_inference_steps,
                    "total_steps": num_inference_steps,
                    "progress": 1.0,
                    "image": png,
                    "is_final": True,
                })
            except Exception as e:
                logger.error(f"Image stream error: {e}")
                queue.put_nowait(None)

        loop = asyncio.get_running_loop()
        loop.run_in_executor(self._executor, _stream_sync)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
            if chunk.get("is_final"):
                break

    def _run_pipeline(self, prompt, width, height, num_steps, seed) -> bytes:
        """Run the full diffusion pipeline synchronously."""
        # 1. Tokenize with chat template (mflux pattern: enable_thinking + add_generation_prompt)
        tokenizer = self._tokenizer
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        tokens = tokenizer(
            [formatted],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="np",
        )
        input_ids = mx.array(tokens["input_ids"])
        attention_mask = mx.array(tokens["attention_mask"])

        # 2. Text encoding
        cap_feats = self._text_encoder(input_ids, attention_mask)
        # Extract only valid (non-padding) tokens (mflux pattern)
        num_valid = int(mx.sum(attention_mask[0]).item())
        cap_feats = cap_feats[0, :num_valid, :]
        mx.eval(cap_feats)

        # 3. Prepare noise latents (MLX-native random, matching mflux)
        latent_h = height // 8
        latent_w = width // 8
        latents = mx.random.normal(
            shape=[16, 1, latent_h, latent_w],
            key=mx.random.key(seed),
        ).astype(mx.float16)

        # 4. Compute sigma schedule
        sigmas = _compute_sigmas(num_steps, width, height)

        # 5. Denoising loop
        for t in range(num_steps):
            sigma_t = sigmas[t].reshape((1,))
            timestep = mx.ones_like(sigma_t) - sigma_t

            noise_pred = self._transformer(
                x=latents,
                timestep=timestep,
                sigmas=sigmas,
                cap_feats=cap_feats,
            )

            # Euler step: x_{t+1} = x_t + (sigma_{t+1} - sigma_t) * noise
            dt = sigmas[t + 1] - sigmas[t]
            latents = latents + noise_pred * dt
            mx.eval(latents)
            logger.debug(f"Step {t+1}/{num_steps}: sigma={float(sigmas[t]):.4f}")

        # 6. VAE decode
        image = self._vae.decode(latents)
        mx.eval(image)

        # 7. Convert to PNG
        return self._to_png(image)

    @staticmethod
    def _to_png(image: mx.array) -> bytes:
        """Convert decoded image tensor to PNG bytes."""
        from PIL import Image as PILImage

        # image: (1, 3, H, W) or (3, H, W), float
        if image.ndim == 4:
            arr = np.array(image[0]).transpose(1, 2, 0)  # (H, W, 3)
        else:
            arr = np.array(image).transpose(1, 2, 0)
        # Denormalize: VAE outputs [-1, 1] → [0, 1] (mflux pattern)
        arr = np.clip(arr / 2.0 + 0.5, 0, 1)
        arr = (arr * 255).astype(np.uint8)
        pil = PILImage.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()

    def get_stats(self) -> dict:
        return {
            "model": self._model_path,
            "loaded": self.is_loaded,
            "running": self._running,
        }
