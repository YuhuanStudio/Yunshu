from __future__ import annotations
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


import asyncio
import base64
import gc
import io
import glob
import json
import logging
import math
import threading
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .types import EngineConfig

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


def _cosine_ramp(n: int) -> np.ndarray:
    """Cosine blending ramp for tiled VAE overlap regions."""
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    t = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
    return 0.5 - 0.5 * np.cos(t * np.pi)


class VAE:
    """VAE wrapper with scaling constants and optional encoder."""

    scaling_factor = 0.3611
    shift_factor = 0.1159

    def __init__(self, decoder: VAEDecoder, encoder: VAEEncoder | None = None):
        self.decoder = decoder
        self.encoder = encoder

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

    def encode(self, image: mx.array) -> mx.array:
        """Encode image to latent space using the VAE encoder.

        Args:
            image: (1, 3, H, W) float image in [-1, 1].

        Returns:
            latents: (16, H/8, W/8) scaled latent representation.
        """
        if self.encoder is None:
            raise RuntimeError("VAE encoder not loaded")
        mean, logvar = self.encoder(image)
        # Sample from the posterior: z = mean + std * noise
        std = mx.exp(0.5 * logvar)
        noise = mx.random.normal(shape=mean.shape, dtype=mean.dtype)
        z = mean + std * noise
        # Scale latents: (scaling_factor * (z - shift_factor))
        z = self.scaling_factor * (z - self.shift_factor)
        # Remove batch dim → (16, H/8, W/8)
        return z[0]

    def encode_deterministic(self, image: mx.array) -> mx.array:
        """Encode image to latents deterministically (use mean only, no sampling).

        Args:
            image: (1, 3, H, W) float image in [-1, 1].

        Returns:
            latents: (16, H/8, W/8) scaled latent representation.
        """
        if self.encoder is None:
            raise RuntimeError("VAE encoder not loaded")
        mean, _logvar = self.encoder(image)
        z = self.scaling_factor * (mean - self.shift_factor)
        return z[0]

    def decode_tiled(
        self,
        latents: mx.array,
        tile_size_px: int = 512,
        overlap_px: int = 64,
    ) -> mx.array:
        """Decode latents in tiles to reduce peak memory for large images.

        Splits the latent spatial grid into overlapping tiles, decodes each
        tile independently, then blends the outputs with cosine weighting in
        the overlap regions (mflux pattern).

        Args:
            latents: (C, F, H, W) or (C, H, W) latent tensor.
            tile_size_px: Maximum output pixel dimension per tile.
            overlap_px: Overlap in output pixels between adjacent tiles.

        Returns:
            (1, 3, H_out, W_out) decoded image.
        """
        # Normalize to (C, F, H_lat, W_lat)
        if latents.ndim == 3:
            latents = latents[:, np.newaxis, :, :]
        C, F, H_lat, W_lat = latents.shape
        scale = 8
        H_out = H_lat * scale
        W_out = W_lat * scale

        tile_lat_h = max(1, tile_size_px // scale)
        tile_lat_w = max(1, tile_size_px // scale)
        ov_lat_h = max(0, min(overlap_px // scale, tile_lat_h - 1))
        ov_lat_w = max(0, min(overlap_px // scale, tile_lat_w - 1))

        # If the image fits in one tile, just decode normally
        if H_lat <= tile_lat_h and W_lat <= tile_lat_w:
            return self.decode(latents)

        stride_h = max(1, tile_lat_h - ov_lat_h)
        stride_w = max(1, tile_lat_w - ov_lat_w)

        ramp_h = _cosine_ramp(overlap_px)
        ramp_w = _cosine_ramp(overlap_px)

        out_np = np.zeros((H_out, W_out, 3), dtype=np.float32)
        count_np = np.zeros((H_out, W_out, 1), dtype=np.float32)

        for y_lat in range(0, H_lat, stride_h):
            y_lat_end = min(y_lat + tile_lat_h, H_lat)
            for x_lat in range(0, W_lat, stride_w):
                x_lat_end = min(x_lat + tile_lat_w, W_lat)

                # Skip sliver tiles
                if (y_lat > 0 and (y_lat_end - y_lat) <= ov_lat_h) or \
                   (x_lat > 0 and (x_lat_end - x_lat) <= ov_lat_w):
                    continue

                tile_lat = latents[:, :, y_lat:y_lat_end, x_lat:x_lat_end]
                decoded_tile = self.decode(tile_lat)
                tile_np = np.array(decoded_tile.astype(mx.float32))[0].transpose(1, 2, 0)

                y_out = y_lat * scale
                x_out = x_lat * scale
                y_out_end = y_lat_end * scale
                x_out_end = x_lat_end * scale

                eff_h = min(y_out_end - y_out, tile_np.shape[0], H_out - y_out)
                eff_w = min(x_out_end - x_out, tile_np.shape[1], W_out - x_out)
                tile_np = tile_np[:eff_h, :eff_w, :]

                ov_h_out = max(0, min(overlap_px, eff_h - 1))
                ov_w_out = max(0, min(overlap_px, eff_w - 1))

                wh = np.ones((eff_h,), dtype=np.float32)
                ww = np.ones((eff_w,), dtype=np.float32)

                if ov_h_out > 0:
                    if y_lat > 0:
                        wh[:ov_h_out] = ramp_h[:ov_h_out]
                    if y_lat_end < H_lat:
                        wh[-ov_h_out:] = 1.0 - ramp_h[:ov_h_out]
                if ov_w_out > 0:
                    if x_lat > 0:
                        ww[:ov_w_out] = ramp_w[:ov_w_out]
                    if x_lat_end < W_lat:
                        ww[-ov_w_out:] = 1.0 - ramp_w[:ov_w_out]

                w2d = wh[:, None] * ww[None, :]
                out_np[y_out:y_out + eff_h, x_out:x_out + eff_w, :] += tile_np * w2d[:, :, None]
                count_np[y_out:y_out + eff_h, x_out:x_out + eff_w, :] += w2d[:, :, None]

        out_np = out_np / np.clip(count_np, 1e-6, None)
        out_chw = out_np.transpose(2, 0, 1)
        return mx.array(out_chw[None, ...])

    def encode_tiled(
        self,
        image: mx.array,
        tile_size_px: int = 512,
        overlap_px: int = 64,
    ) -> mx.array:
        """Encode image in tiles to reduce peak memory.

        Args:
            image: (1, 3, H, W) float image in [-1, 1].
            tile_size_px: Maximum input pixel dimension per tile.
            overlap_px: Overlap in input pixels between adjacent tiles.

        Returns:
            (16, H/8, W/8) scaled latent representation.
        """
        if self.encoder is None:
            raise RuntimeError("VAE encoder not loaded")

        B, C_in, H, W = image.shape
        scale = 8
        H_lat = (H + scale - 1) // scale
        W_lat = (W + scale - 1) // scale

        tile_lat_h = max(1, tile_size_px // scale)
        tile_lat_w = max(1, tile_size_px // scale)
        ov_lat_h = max(0, min(overlap_px // scale, tile_lat_h - 1))
        ov_lat_w = max(0, min(overlap_px // scale, tile_lat_w - 1))

        if H <= tile_size_px and W <= tile_size_px:
            return self.encode_deterministic(image)

        stride_h = max(1, tile_lat_h - ov_lat_h)
        stride_w = max(1, tile_lat_w - ov_lat_w)

        ramp_h = _cosine_ramp(ov_lat_h)
        ramp_w = _cosine_ramp(ov_lat_w)

        out_np = np.zeros((H_lat, W_lat, 16), dtype=np.float32)
        count_np = np.zeros((H_lat, W_lat, 1), dtype=np.float32)

        for y_lat in range(0, H_lat, stride_h):
            y_lat_end = min(y_lat + tile_lat_h, H_lat)
            for x_lat in range(0, W_lat, stride_w):
                x_lat_end = min(x_lat + tile_lat_w, W_lat)

                if (y_lat > 0 and (y_lat_end - y_lat) <= ov_lat_h) or \
                   (x_lat > 0 and (x_lat_end - x_lat) <= ov_lat_w):
                    continue

                y_in = y_lat * scale
                x_in = x_lat * scale
                y_in_end = min(y_lat_end * scale, H)
                x_in_end = min(x_lat_end * scale, W)

                tile_img = image[:, :, y_in:y_in_end, x_in:x_in_end]
                mean, _logvar = self.encoder(tile_img)
                z = self.scaling_factor * (mean - self.shift_factor)
                enc_np = np.array(z.astype(mx.float32))[0].transpose(1, 2, 0)

                eff_h = min(y_lat_end - y_lat, enc_np.shape[0], H_lat - y_lat)
                eff_w = min(x_lat_end - x_lat, enc_np.shape[1], W_lat - x_lat)
                enc_np = enc_np[:eff_h, :eff_w, :]

                ov_h = max(0, min(ov_lat_h, eff_h - 1))
                ov_w = max(0, min(ov_lat_w, eff_w - 1))

                wh = np.ones((eff_h,), dtype=np.float32)
                ww = np.ones((eff_w,), dtype=np.float32)

                if ov_h > 0:
                    if y_lat > 0:
                        wh[:ov_h] = ramp_h[:ov_h]
                    if y_lat_end < H_lat:
                        wh[-ov_h:] = 1.0 - ramp_h[:ov_h]
                if ov_w > 0:
                    if x_lat > 0:
                        ww[:ov_w] = ramp_w[:ov_w]
                    if x_lat_end < W_lat:
                        ww[-ov_w:] = 1.0 - ramp_w[:ov_w]

                w2d = wh[:, None] * ww[None, :]
                out_np[y_lat:y_lat + eff_h, x_lat:x_lat + eff_w, :] += enc_np * w2d[:, :, None]
                count_np[y_lat:y_lat + eff_h, x_lat:x_lat + eff_w, :] += w2d[:, :, None]

        out_np = out_np / np.clip(count_np, 1e-6, None)
        out_chw = out_np.transpose(2, 0, 1)
        return mx.array(out_chw)


# ── VAE Encoder ──


class _DownSampler(nn.Module):
    """Strided 2x downsampler: pad + Conv2d(stride=2)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def __call__(self, x):
        x = mx.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1)))
        xw = x.transpose(0, 2, 3, 1)
        out = self.conv(xw)
        return out.transpose(0, 3, 1, 2)


class _DownEncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_layers: int = 2, add_downsample: bool = True):
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            use_sc = (i == 0) and (in_ch != out_ch)
            self.resnets.append(_ResnetBlock2D(
                in_ch if i == 0 else out_ch, out_ch, use_conv_shortcut=use_sc,
            ))
        self.downsamplers = [_DownSampler(out_ch)] if add_downsample else None

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsamplers is not None:
            for ds in self.downsamplers:
                x = ds(x)
        return x


class VAEEncoder(nn.Module):
    """Z-Image VAE encoder: ConvIn → 4 DownBlocks → MidBlock → ConvOut.

    Outputs 32 channels (2 × 16 latent), split into mean and logvar for
    the KL posterior.  8x spatial downsample (3 stride-2 convs).
    """

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.down_blocks = [
            _DownEncoderBlock(128, 128, 2, add_downsample=True),
            _DownEncoderBlock(128, 256, 2, add_downsample=True),
            _DownEncoderBlock(256, 512, 2, add_downsample=True),
            _DownEncoderBlock(512, 512, 2, add_downsample=False),
        ]
        self.mid_block = _UNetMidBlock(512)
        self.conv_norm_out = nn.GroupNorm(32, 512, eps=1e-6, affine=True, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Encode image → (mean, logvar) in latent space.

        Args:
            x: (B, 3, H, W) float image in [-1, 1].

        Returns:
            mean: (B, 16, H/8, W/8)
            logvar: (B, 16, H/8, W/8)
        """
        x = x.transpose(0, 2, 3, 1)
        h = self.conv_in(x).transpose(0, 3, 1, 2)
        for block in self.down_blocks:
            h = block(h)
        h = self.mid_block(h)
        h = self.conv_norm_out(h.transpose(0, 2, 3, 1))
        h = nn.silu(h)
        h = self.conv_out(h).transpose(0, 3, 1, 2)
        # Split 32-ch output into mean + logvar (each 16 ch)
        mean, logvar = mx.split(h, 2, axis=1)
        return mean, logvar


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


def _remap_vae_weights(raw: dict, component: str = "decoder") -> list[tuple[str, mx.array]]:
    """Remap VAE weights for decoder or encoder.

    VAE safetensors weights are in PyTorch OIHW format: (out, in, kH, kW)
    MLX Conv2d expects OHWI format: (out, kH, kW, in)
    So conv weights need transpose: (0, 2, 3, 1)
    """
    result = []
    prefix = f"{component}."
    for key, val in raw.items():
        if not key.startswith(prefix):
            continue
        local_key = key[len(prefix):]
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
        result.append((local_key, val))

    return result


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
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
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

        # Wave 43: Diffusion scheduler + pipeline registry
        from .diffusion_infra import DiffusionScheduler, SchedulerType
        self._diffusion_scheduler_type: SchedulerType | None = None
        scheduler_env = os.environ.get("YUNSHU_DIFFUSION_SCHEDULER", "").strip().lower()
        if scheduler_env:
            scheduler_map = {s.value: s for s in SchedulerType}
            if scheduler_env in scheduler_map:
                self._diffusion_scheduler_type = scheduler_map[scheduler_env]
                logger.info(f"DiffusionScheduler override: {self._diffusion_scheduler_type.value}")
            else:
                logger.warning(
                    f"Unknown YUNSHU_DIFFUSION_SCHEDULER={scheduler_env!r}, "
                    f"expected one of {list(scheduler_map.keys())}"
                )
        # Default scheduler instance (always available, used only when env var is set)
        self._diffusion_scheduler = DiffusionScheduler()
        from .image_pipeline import PipelineType
        self._pipeline_type = PipelineType

        # DiffusionLoRAOffloader — priority-based LoRA adapter memory management.
        # Opt-in via YUNSHU_LORA_BUDGET_MB env var (GPU memory budget in MB).
        # When set, LoRA adapters are swapped in/out per diffusion step based on
        # priority and assigned step ranges.
        self._lora_offloader = None
        self._original_modules: dict[str, object] = {}  # name→original Linear before LoRA wrap
        lora_budget_mb = os.environ.get("YUNSHU_LORA_BUDGET_MB", "").strip()
        if lora_budget_mb:
            try:
                budget_bytes = int(float(lora_budget_mb) * 1024 * 1024)
                from .diffusion_infra import DiffusionLoRAOffloader
                self._lora_offloader = DiffusionLoRAOffloader(memory_budget_bytes=budget_bytes)
                logger.info(f"DiffusionLoRAOffloader enabled ({lora_budget_mb} MB budget)")
            except ValueError:
                logger.warning(f"Invalid YUNSHU_LORA_BUDGET_MB={lora_budget_mb!r}, expected number in MB")

        # DistributedDiffusionCoordinator — splits diffusion steps across mesh nodes.
        # Opt-in via YUNSHU_DIFFUSION_NODES env var (number of nodes >= 2).
        # Uses contiguous step assignment by default.
        self._diffusion_coordinator = None
        diffusion_nodes = os.environ.get("YUNSHU_DIFFUSION_NODES", "").strip()
        if diffusion_nodes:
            try:
                n_nodes = int(diffusion_nodes)
                if n_nodes >= 2:
                    from .diffusion_infra import DistributedDiffusionCoordinator
                    self._diffusion_coordinator = DistributedDiffusionCoordinator(num_nodes=n_nodes)
                    logger.info(f"DistributedDiffusionCoordinator enabled ({n_nodes} nodes)")
                else:
                    logger.warning(f"YUNSHU_DIFFUSION_NODES must be >= 2 for distributed diffusion, got {n_nodes}")
            except ValueError:
                logger.warning(f"Invalid YUNSHU_DIFFUSION_NODES={diffusion_nodes!r}, expected integer >= 2")

        # TeaCache (opt-in via YUNSHU_TEACACHE=1 or threshold value)
        self._teacache = None
        self._teacache_lock = threading.Lock()
        teacache_env = os.environ.get("YUNSHU_TEACACHE", "").strip()
        if teacache_env in ("1", "true", "yes"):
            from .teacache import TeaCacheConfig, TeaCacheHook
            self._teacache = TeaCacheHook(TeaCacheConfig(rel_l1_thresh=0.2))
            logger.info("TeaCache enabled (threshold=0.2)")
        elif teacache_env and teacache_env not in ("0", "false", "no"):
            try:
                thresh = float(teacache_env)
                from .teacache import TeaCacheConfig, TeaCacheHook
                self._teacache = TeaCacheHook(TeaCacheConfig(rel_l1_thresh=thresh))
                logger.info(f"TeaCache enabled (threshold={thresh})")
            except ValueError:
                pass

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

        # 3. VAE Decoder + Encoder (no quantization per quantize_config.json)
        logger.info("Loading VAE decoder + encoder...")
        vae_decoder = VAEDecoder()
        vae_weights = _load_component_weights(model_path, "vae")
        _load_weights_into(vae_decoder, _remap_vae_weights(vae_weights, "decoder"))
        vae_decoder.eval()

        # Load VAE encoder (present in Z-Image safetensors as encoder.* keys)
        vae_encoder = VAEEncoder()
        _load_weights_into(vae_encoder, _remap_vae_weights(vae_weights, "encoder"))
        vae_encoder.eval()

        self._vae = VAE(vae_decoder, vae_encoder)
        logger.info(f"VAE loaded (decoder + encoder): {len(vae_weights)} tensors")

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
        """Stop and release resources.

        Idempotent: safe to call multiple times.
        """
        if not self._running and self._transformer is None:
            return
        self._text_encoder = None
        self._transformer = None
        self._vae = None
        self._tokenizer = None
        self._running = False
        self._teacache = None
        if self._lora_offloader is not None:
            try:
                self._lora_offloader.unload_all()
            except Exception:
                logger.debug("LoRA offloader unload failed during stop", exc_info=True)
        self._lora_offloader = None
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

        # Use DFlash block diffusion if enabled and compatible
        use_dflash = (
            self._dflash is not None
            and self._dflash.is_enabled
            and self._dflash.is_compatible(self._transformer)
        )

        if use_dflash:
            def _generate_sync() -> bytes:
                return self._run_dflash_pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_steps=num_inference_steps,
                    seed=seed if seed is not None else 42,
                )
        else:
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
            # Clean up GPU memory before re-raising
            gc.collect()
            from .mlx_executor import sync_and_clear_cache
            await loop.run_in_executor(self._executor, sync_and_clear_cache)
            raise MemoryError(f"GPU OOM during image generation: {e}") from e

        elapsed = time.monotonic() - t0
        logger.info(f"Image gen: {elapsed:.2f}s, {len(png_bytes)} bytes, prompt='{prompt[:50]}...'")
        return png_bytes

    async def generate(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: int | None = None,
        image: bytes | None = None,
        denoise_strength: float = 0.8,
        **kwargs,
    ) -> list[bytes]:
        """Unified generate interface for text-to-image and image conditioning.

        Returns a list of PNG byte strings.

        For img2img: the `image` parameter provides a source image.
        The source image is encoded to latents, partially noised, and
        partially denoised to produce an output that preserves the source
        structure. `denoise_strength` controls how much the output deviates
        from the source (0.0=identical, 1.0=ignore source).
        """
        if self._transformer is None:
            raise RuntimeError("Engine not started")

        if image is not None:
            return await self._generate_variation(
                source_image=image,
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                seed=seed,
                denoise_strength=denoise_strength,
            )

        png = await self.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [png]

    async def _generate_variation(
        self,
        source_image: bytes,
        prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        seed: int | None = None,
        denoise_strength: float = 0.8,
    ) -> list[bytes]:
        """Generate a variation of a source image using img2img pipeline.

        Encodes the source image to latent space via the VAE encoder,
        adds partial noise (controlled by denoise_strength), then runs
        partial denoising to produce a variation that preserves the
        source image structure.

        Args:
            source_image: Source image bytes (PNG/JPEG).
            prompt: Optional text prompt for conditioning.
            width: Output width.
            height: Output height.
            num_inference_steps: Number of denoising steps.
            seed: Random seed.
            denoise_strength: How much to re-denoise (0.0=keep source, 1.0=full noise).

        Returns:
            List of PNG byte strings.
        """
        if self._transformer is None:
            raise RuntimeError("Engine not started")

        if prompt:
            gen_prompt = prompt
        else:
            gen_prompt = "A variation of the provided image, high quality, detailed"

        def _variation_sync() -> bytes:
            return self._run_img2img_pipeline(
                prompt=gen_prompt,
                image_data=source_image,
                width=width,
                height=height,
                num_steps=num_inference_steps,
                seed=seed if seed is not None else 42,
                denoise_strength=denoise_strength,
            )

        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        png_bytes = await loop.run_in_executor(self._executor, _variation_sync)
        elapsed = time.monotonic() - t0
        logger.info(f"Image variation: {elapsed:.2f}s, prompt='{gen_prompt[:50]}...'")
        return [png_bytes]

    async def generate_image_stream(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: int | None = None,
        preview_interval: int = 0,
        cancel_event: Any = None,
        **kwargs,
    ):
        """Stream image generation progress, yielding step-by-step updates.

        Each yielded dict has:
        - "step": Current step number
        - "total_steps": Total number of steps
        - "progress": 0.0 to 1.0
        - "image": PNG bytes of current latent (decoded preview) or None
        - "is_final": True for the last chunk

        preview_interval: decode and emit intermediate preview images every N steps.
            0 = no intermediate previews (default), final image only.
            1 = preview at every step, 2 = every other step, etc.

        cancel_event: Optional asyncio.Event — when set, aborts the diffusion loop.
        """
        if self._transformer is None:
            raise RuntimeError("Engine not started")

        import queue as _queue_mod
        _thread_queue: _queue_mod.Queue[dict | None] = _queue_mod.Queue(maxsize=64)

        # Thread-safe cancel flag for the executor thread.
        # asyncio.Event.is_set() is not safe to call from non-event-loop
        # threads; mirror into a threading.Event instead.
        _cancel = threading.Event()
        if cancel_event is not None and cancel_event.is_set():
            _cancel.set()

        def _should_preview(step: int, total: int) -> bool:
            if preview_interval <= 0:
                return False
            if step == total:
                return True
            return step % preview_interval == 0

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

                sigmas = self._resolve_sigmas(num_inference_steps, width, height)

                for t in range(num_inference_steps):
                    # Check cancel before each expensive diffusion step (thread-safe)
                    if _cancel.is_set():
                        logger.info("Image stream cancelled at step %d/%d", t + 1, num_inference_steps)
                        try:
                            _thread_queue.put_nowait(None)
                        except _queue_mod.Full:
                            pass
                        return

                    sigma_t = sigmas[t].reshape((1,))
                    timestep = mx.ones_like(sigma_t) - sigma_t
                    noise_pred = self._transformer(
                        x=latents, timestep=timestep, sigmas=sigmas, cap_feats=cap_feats,
                    )
                    dt = sigmas[t + 1] - sigmas[t]
                    latents = latents + noise_pred * dt
                    mx.eval(latents)

                    progress = (t + 1) / num_inference_steps
                    step_num = t + 1

                    preview_png = None
                    if _should_preview(step_num, num_inference_steps):
                        image = self._vae.decode(latents)
                        mx.eval(image)
                        preview_png = self._to_png(image)

                    try:
                        _thread_queue.put_nowait({
                            "step": step_num,
                            "total_steps": num_inference_steps,
                            "progress": progress,
                            "image": preview_png,
                            "is_final": False,
                        })
                    except _queue_mod.Full:
                        logger.warning("Image stream queue full — dropping preview chunk")
                        continue

                # Final decode
                image = self._vae.decode(latents)
                mx.eval(image)
                png = self._to_png(image)
                try:
                    _thread_queue.put_nowait({
                        "step": num_inference_steps,
                        "total_steps": num_inference_steps,
                        "progress": 1.0,
                        "image": png,
                        "is_final": True,
                    })
                except _queue_mod.Full:
                    pass
            except Exception as e:
                logger.error(f"Image stream error: {e}", exc_info=True)
                try:
                    _thread_queue.put_nowait(None)
                except _queue_mod.Full:
                    pass

        loop = asyncio.get_running_loop()
        stream_task = loop.run_in_executor(self._executor, _stream_sync)

        try:
            while True:
                # Propagate cancel_event to the thread-safe flag
                if cancel_event is not None and cancel_event.is_set():
                    _cancel.set()
                try:
                    chunk = _thread_queue.get_nowait()
                except _queue_mod.Empty:
                    await asyncio.sleep(0.01)
                    continue
                if chunk is None:
                    break
                yield chunk
                if chunk.get("is_final"):
                    break
        finally:
            # Signal the executor thread to stop
            _cancel.set()
            if not stream_task.done():
                stream_task.cancel()
                try:
                    await stream_task
                except (asyncio.CancelledError, Exception):
                    pass
            # Drain remaining queue items to unblock the executor thread
            while True:
                try:
                    _thread_queue.get_nowait()
                except _queue_mod.Empty:
                    break

    def _run_dflash_pipeline(
        self,
        prompt: str,
        width: int,
        height: int,
        num_steps: int,
        seed: int,
    ) -> bytes:
        """Run block diffusion pipeline using DFlash.

        Implements the 2-stage block diffusion protocol:
        1. Coarse stage: Generate a low-resolution block plan with fewer steps
        2. Refinement stage: Refine each block with full steps and warm-start
           from L1 cached coarse latents

        The key acceleration comes from:
        - Coarse blocks run at reduced resolution (block_size x block_size)
        - L1 cache reuses coarse latents as warm-starts for refinement
        - Block-level parallelism where memory allows
        """
        from .dflash import BlockPlan

        dflash = self._dflash
        config = dflash.config
        overlap = config.overlap_margin if config.overlap_blocks else 0

        # Create block plan for the image
        block_plan = BlockPlan.create(
            width, height, config.block_size, overlap=overlap,
        )
        logger.info(
            f"DFlash pipeline: {width}x{height} → {block_plan.num_blocks} blocks "
            f"(coarse={config.coarse_steps}, refine={config.refine_steps})"
        )

        # 1. Tokenize prompt (shared across all blocks)
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

        # 2. Text encoding (shared across all blocks)
        cap_feats = self._text_encoder(input_ids, attention_mask)
        num_valid = int(mx.sum(attention_mask[0]).item())
        cap_feats = cap_feats[0, :num_valid, :]
        mx.eval(cap_feats)

        # 3. Compute sigma schedules for coarse and refine stages
        sigmas_coarse = _compute_sigmas(config.coarse_steps, width, height)
        sigmas_refine = _compute_sigmas(config.refine_steps, width, height)

        # 4. Stage 1: Coarse generation per block
        coarse_latents = {}
        for i, (bx, by, bx_end, by_end) in enumerate(block_plan.blocks):
            block_key = f"coarse_{bx}_{by}"

            # Check L1 cache for reusable coarse latent
            cached = dflash._l1_cache.get(block_key)
            if cached is not None:
                coarse_latents[block_key] = cached
                logger.debug(f"DFlash block ({bx},{by}): L1 cache hit (coarse)")
                continue

            # Generate initial noise at block granularity
            block_latent_h = (by_end - by) // 8
            block_latent_w = (bx_end - bx) // 8
            block_latents = mx.random.normal(
                shape=[16, 1, block_latent_h, block_latent_w],
                key=mx.random.key(seed + i),
            ).astype(mx.float16)

            # Coarse denoising: fewer steps on full latent space
            for t in range(config.coarse_steps):
                sigma_t = sigmas_coarse[t].reshape((1,))
                timestep = mx.ones_like(sigma_t) - sigma_t
                noise_pred = self._transformer(
                    x=block_latents,
                    timestep=timestep,
                    sigmas=sigmas_coarse,
                    cap_feats=cap_feats,
                )
                dt = sigmas_coarse[t + 1] - sigmas_coarse[t]
                block_latents = block_latents + noise_pred * dt
                mx.eval(block_latents)

            coarse_latents[block_key] = block_latents
            dflash._l1_cache.put(block_key, block_latents)
            logger.debug(f"DFlash block ({bx},{by}): coarse done ({config.coarse_steps} steps)")

        # 5. Stage 2: Refinement using coarse latents as warm start
        # Use the coarse latent as the initial state and refine with full steps
        # Pick the best coarse latent (last block covers the most context)
        best_key = f"coarse_{block_plan.blocks[-1][0]}_{block_plan.blocks[-1][1]}"
        latents = coarse_latents.get(best_key)
        if latents is None:
            # Fallback to first block's latent
            first_key = f"coarse_{block_plan.blocks[0][0]}_{block_plan.blocks[0][1]}"
            latents = coarse_latents[first_key]

        # TeaCache: hold lock for the entire denoising loop to prevent
        # concurrent requests from corrupting shared cache state.
        tc_lock = self._teacache_lock if self._teacache is not None else None
        if tc_lock is not None:
            tc_lock.acquire()
        try:
            if self._teacache is not None:
                self._teacache.reset()

            # Refinement denoising loop with full steps
            for t in range(config.refine_steps):
                sigma_t = sigmas_refine[t].reshape((1,))
                timestep = mx.ones_like(sigma_t) - sigma_t

                if self._teacache is not None:
                    noise_pred = self._teacache.forward(
                        self._transformer, latents, timestep, sigmas_refine, cap_feats,
                    )
                else:
                    noise_pred = self._transformer(
                        x=latents,
                        timestep=timestep,
                        sigmas=sigmas_refine,
                        cap_feats=cap_feats,
                    )

                dt = sigmas_refine[t + 1] - sigmas_refine[t]
                latents = latents + noise_pred * dt
                mx.eval(latents)
                logger.debug(f"DFlash refine step {t+1}/{config.refine_steps}")

            if self._teacache is not None:
                tc_stats = self._teacache.get_stats()
                logger.info(f"TeaCache: {tc_stats['cache_hits']} hits, "
                            f"{tc_stats['cache_misses']} misses, "
                            f"hit_rate={tc_stats['hit_rate']:.1%}")
        finally:
            if tc_lock is not None:
                tc_lock.release()

        # 6. VAE decode
        if width * height > 1024 * 1024:
            image = self._vae.decode_tiled(latents, tile_size_px=512, overlap_px=64)
        else:
            image = self._vae.decode(latents)
        mx.eval(image)

        # 7. Update DFlash stats
        dflash._stats["total_generations"] += 1
        dflash._stats["total_blocks_processed"] += block_plan.num_blocks
        dflash._stats["l1_cache_saved_steps"] += sum(
            1 for k in coarse_latents
            if dflash._l1_cache.get(f"_saved_{k}") is not None
        )

        logger.info(
            f"DFlash pipeline complete: {block_plan.num_blocks} blocks, "
            f"coarse={config.coarse_steps} + refine={config.refine_steps} steps"
        )

        # 8. Convert to PNG
        return self._to_png(image)

    def _resolve_sigmas(
        self,
        num_steps: int,
        width: int,
        height: int,
    ) -> mx.array:
        """Resolve the sigma schedule, using DiffusionScheduler when opt-in env var is set.

        Default path: _compute_sigmas() with resolution-dependent mu-shift for FLUX models.
        Opt-in path (YUNSHU_DIFFUSION_SCHEDULER set): DiffusionScheduler.sigmas converted
        to mx.array.  The scheduler is re-created with the request's num_steps so the
        sigma count matches the denoising loop.
        """
        if self._diffusion_scheduler_type is not None:
            from .diffusion_infra import DiffusionScheduler, SchedulerType
            scheduler = DiffusionScheduler(
                num_inference_steps=num_steps,
                scheduler_type=self._diffusion_scheduler_type,
            )
            sigma_list = scheduler.sigmas  # list[float]
            # Append a 0 sentinel (the default _compute_sigmas always appends a trailing 0)
            sigma_list.append(0.0)
            sigmas = mx.array(sigma_list, dtype=mx.float32)
            logger.info(
                f"Using DiffusionScheduler ({self._diffusion_scheduler_type.value}): "
                f"{num_steps} steps, sigmas range [{sigma_list[0]:.4f}, {sigma_list[-2]:.4f}]"
            )
            return sigmas

        # Default: resolution-dependent mu-shift for FLUX/Z-Image models
        return _compute_sigmas(num_steps, width, height)

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
        sigmas = self._resolve_sigmas(num_steps, width, height)

        # 4b. Initialize distributed coordinator step assignment if active
        if self._diffusion_coordinator is not None:
            self._diffusion_coordinator.assign_steps(
                num_nodes=self._diffusion_coordinator.num_nodes,
                total_steps=num_steps,
            )
            # Log step distribution across nodes
            for nid, assign in self._diffusion_coordinator.assignments.items():
                logger.info(
                    f"DistributedDiffusion: node {nid} -> "
                    f"steps {assign.first_step}..{assign.last_step} "
                    f"({assign.step_count} steps)"
                )

        # 5. Denoising loop
        # TeaCache: hold lock for the entire denoising loop to prevent
        # concurrent requests from corrupting shared cache state.
        tc_lock = self._teacache_lock if self._teacache is not None else None
        if tc_lock is not None:
            tc_lock.acquire()
        try:
            if self._teacache is not None:
                self._teacache.reset()

            # 5b. LoRA offloader: load adapters needed for step 0
            if self._lora_offloader is not None:
                self._lora_offloader.load_for_step(0)

            for t in range(num_steps):
                sigma_t = sigmas[t].reshape((1,))
                timestep = mx.ones_like(sigma_t) - sigma_t

                if self._teacache is not None:
                    noise_pred = self._teacache.forward(
                        self._transformer, latents, timestep, sigmas, cap_feats,
                    )
                else:
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

                # LoRA offloader: swap adapters for next step
                if self._lora_offloader is not None:
                    self._lora_offloader.unload_after_step(t)
                    if t + 1 < num_steps:
                        self._lora_offloader.load_for_step(t + 1)

                # Distributed coordinator: record sync checkpoint at interval boundaries
                if self._diffusion_coordinator is not None:
                    self._diffusion_coordinator.sync_latents(
                        source_node=0, target_node=0, step=t, latent_data=latents,
                    )

            if self._teacache is not None:
                tc_stats = self._teacache.get_stats()
                logger.info(f"TeaCache: {tc_stats['cache_hits']} hits, "
                            f"{tc_stats['cache_misses']} misses, "
                            f"hit_rate={tc_stats['hit_rate']:.1%}")
        finally:
            if tc_lock is not None:
                tc_lock.release()

        # LoRA offloader cleanup after denoising
        if self._lora_offloader is not None:
            unloaded = self._lora_offloader.unload_all()
            if unloaded:
                logger.info(f"LoRA offloader: unloaded {len(unloaded)} adapters after denoising")

        # Distributed coordinator progress report
        if self._diffusion_coordinator is not None:
            progress = self._diffusion_coordinator.get_progress()
            logger.info(
                f"DistributedDiffusion: {progress['progress_pct']:.0f}% complete, "
                f"{progress['checkpoints_count']} checkpoints, "
                f"{progress['sync_points_count']} sync points"
            )

        # 6. VAE decode (auto-tile for large images to reduce peak memory)
        if width * height > 1024 * 1024:
            image = self._vae.decode_tiled(latents, tile_size_px=512, overlap_px=64)
        else:
            image = self._vae.decode(latents)
        mx.eval(image)

        # 7. Convert to PNG
        return self._to_png(image)

    def _load_image_to_tensor(self, image_data: bytes) -> tuple[mx.array, int, int]:
        """Load image bytes → (1, 3, H, W) float tensor in [-1, 1].

        Accepts PNG or JPEG bytes.
        """
        from PIL import Image as PILImage
        pil = PILImage.open(io.BytesIO(image_data)).convert("RGB")
        w, h = pil.size
        arr = np.array(pil, dtype=np.float32) / 255.0  # (H, W, 3)
        arr = (arr - 0.5) / 0.5  # normalize to [-1, 1]
        arr = arr.transpose(2, 0, 1)  # (3, H, W)
        tensor = mx.array(arr[np.newaxis, :, :, :])  # (1, 3, H, W)
        return tensor, h, w

    def _run_img2img_pipeline(
        self,
        prompt: str,
        image_data: bytes,
        width: int,
        height: int,
        num_steps: int,
        seed: int,
        denoise_strength: float = 0.8,
    ) -> bytes:
        """Run image-to-image pipeline: encode source, add noise, partial denoise.

        Pipeline:
        1. Load source image → pixel tensor → encode to latents (VAE encoder)
        2. Add noise to latents at the appropriate timestep determined by denoise_strength
        3. Run partial denoising from the noised latents
        4. VAE decode → image

        Args:
            prompt: Text prompt for conditioning.
            image_data: Source image bytes (PNG/JPEG).
            width: Output width.
            height: Output height.
            num_steps: Denoising steps.
            seed: Random seed.
            denoise_strength: How much to re-denoise (0.0=keep source, 1.0=full noise).
        """
        # 1. Load and encode source image
        image_tensor, img_h, img_w = self._load_image_to_tensor(image_data)
        # Resize to target dimensions if needed
        if img_h != height or img_w != width:
            from PIL import Image as PILImage
            pil = PILImage.open(io.BytesIO(image_data)).convert("RGB").resize(
                (width, height), PILImage.LANCZOS
            )
            arr = np.array(pil, dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            arr = arr.transpose(2, 0, 1)[np.newaxis, :, :, :]
            image_tensor = mx.array(arr)

        source_latents = self._vae.encode_deterministic(image_tensor)
        mx.eval(source_latents)

        # Add frame dimension for transformer: (16, 1, H/8, W/8)
        source_latents_4d = source_latents[:, np.newaxis, :, :]

        # 2. Tokenize prompt
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

        # 3. Text encoding
        cap_feats = self._text_encoder(input_ids, attention_mask)
        num_valid = int(mx.sum(attention_mask[0]).item())
        cap_feats = cap_feats[0, :num_valid, :]
        mx.eval(cap_feats)

        # 4. Compute sigma schedule (respects YUNSHU_DIFFUSION_SCHEDULER opt-in)
        sigmas = self._resolve_sigmas(num_steps, width, height)

        # 5. Add noise to source latents based on denoise_strength
        # Higher denoise_strength → start from later timestep (more noise)
        # denoise_strength=1.0: start from pure noise (timestep 0)
        # denoise_strength=0.0: keep source latents exactly (timestep = num_steps)
        latent_h = height // 8
        latent_w = width // 8
        noise = mx.random.normal(
            shape=[16, 1, latent_h, latent_w],
            key=mx.random.key(seed),
        ).astype(mx.float16)

        # Determine the starting step based on denoise_strength
        # Start from step at index: num_steps * (1 - denoise_strength)
        start_step = int(num_steps * (1.0 - denoise_strength))
        start_step = max(0, min(start_step, num_steps - 1))

        if start_step == 0 and denoise_strength >= 1.0:
            # Full denoising from pure noise (same as text2img)
            latents = noise
        else:
            # Blend source latents with noise at the starting timestep's sigma
            sigma_start = sigmas[start_step]
            # Flow-matching interpolation: latents = (1 - sigma) * source + sigma * noise
            latents = (1.0 - sigma_start) * source_latents_4d.astype(mx.float16) + sigma_start * noise

        mx.eval(latents)

        # 6. Partial denoising loop (from start_step to num_steps)
        # TeaCache: hold lock for the entire denoising loop to prevent
        # concurrent requests from corrupting shared cache state.
        tc_lock = self._teacache_lock if self._teacache is not None else None
        if tc_lock is not None:
            tc_lock.acquire()
        try:
            if self._teacache is not None:
                self._teacache.reset()

            for t in range(start_step, num_steps):
                sigma_t = sigmas[t].reshape((1,))
                timestep = mx.ones_like(sigma_t) - sigma_t

                if self._teacache is not None:
                    noise_pred = self._teacache.forward(
                        self._transformer, latents, timestep, sigmas, cap_feats,
                    )
                else:
                    noise_pred = self._transformer(
                        x=latents,
                        timestep=timestep,
                        sigmas=sigmas,
                        cap_feats=cap_feats,
                    )

                # Euler step
                dt = sigmas[t + 1] - sigmas[t]
                latents = latents + noise_pred * dt
                mx.eval(latents)
                logger.debug(f"img2img step {t + 1}/{num_steps}: sigma={float(sigmas[t]):.4f}")
        finally:
            if tc_lock is not None:
                tc_lock.release()

        # 7. VAE decode (auto-tile for large images)
        if width * height > 1024 * 1024:
            image = self._vae.decode_tiled(latents, tile_size_px=512, overlap_px=64)
        else:
            image = self._vae.decode(latents)
        mx.eval(image)

        return self._to_png(image)

    def _run_inpaint_pipeline(
        self,
        prompt: str,
        image_data: bytes,
        mask_data: bytes | None,
        mask_base64: str | None,
        width: int,
        height: int,
        num_steps: int,
        seed: int,
        denoise_strength: float = 1.0,
    ) -> bytes:
        """Run inpainting: fill masked regions of an image guided by prompt.

        Pipeline:
        1. Load image → pixel tensor → encode to latents (VAE encoder)
        2. Load/create mask → downsample to latent resolution
        3. Create noise latents; blend: masked_latents = (1-mask)*known_latents + mask*noise
        4. Denoise with mask-aware blending at each step
        5. VAE decode → image

        Args:
            prompt: Text prompt for inpainting.
            image_data: Source image bytes (PNG/JPEG).
            mask_data: Mask image bytes (white=inpainted, black=kept). Optional.
            mask_base64: Base64-encoded mask. Used if mask_data is None.
            width: Output width.
            height: Output height.
            num_steps: Denoising steps.
            seed: Random seed.
            denoise_strength: How much to re-denoise masked region (1.0=full).
        """
        # 1. Load and encode source image
        image_tensor, img_h, img_w = self._load_image_to_tensor(image_data)
        # Resize to target dimensions if needed
        if img_h != height or img_w != width:
            from PIL import Image as PILImage
            pil = PILImage.open(io.BytesIO(image_data)).convert("RGB").resize(
                (width, height), PILImage.LANCZOS
            )
            arr = np.array(pil, dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            arr = arr.transpose(2, 0, 1)[np.newaxis, :, :, :]
            image_tensor = mx.array(arr)

        known_latents = self._vae.encode_deterministic(image_tensor)
        mx.eval(known_latents)

        latent_h = height // 8
        latent_w = width // 8

        # 2. Create mask tensor
        if mask_data is not None:
            mask_tensor = self._load_mask(mask_data, latent_h, latent_w)
        elif mask_base64 is not None:
            mask_bytes = base64.b64decode(mask_base64)
            mask_tensor = self._load_mask(mask_bytes, latent_h, latent_w)
        else:
            # No mask → full image inpainting (entire canvas)
            mask_tensor = mx.ones((1, 1, latent_h, latent_w), dtype=mx.float16)

        # Add frame dimension for transformer: (16, 1, H/8, W/8)
        known_latents_4d = known_latents[:, np.newaxis, :, :]  # (16, 1, H/8, W/8)
        mask_4d = mask_tensor  # (1, 1, H/8, W/8)

        # 3. Prepare noise latents
        noise = mx.random.normal(
            shape=[16, 1, latent_h, latent_w],
            key=mx.random.key(seed),
        ).astype(mx.float16)

        # Apply denoise strength: interpolate between known latents and noise
        # At strength=1.0, masked region starts from pure noise
        # At strength<1.0, masked region starts from partially noised known latents
        if denoise_strength < 1.0:
            # Blend noise and known latents for masked region
            init_noise = (1 - denoise_strength) * known_latents_4d + denoise_strength * noise
            latents = (1 - mask_4d) * known_latents_4d + mask_4d * init_noise
        else:
            latents = (1 - mask_4d) * known_latents_4d + mask_4d * noise

        mx.eval(latents)

        # 4. Tokenize prompt
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

        # 5. Text encoding
        cap_feats = self._text_encoder(input_ids, attention_mask)
        num_valid = int(mx.sum(attention_mask[0]).item())
        cap_feats = cap_feats[0, :num_valid, :]
        mx.eval(cap_feats)

        # 6. Compute sigma schedule (respects YUNSHU_DIFFUSION_SCHEDULER opt-in)
        sigmas = self._resolve_sigmas(num_steps, width, height)

        # 7. Masked denoising loop
        # TeaCache: hold lock for the entire denoising loop to prevent
        # concurrent requests from corrupting shared cache state.
        tc_lock = self._teacache_lock if self._teacache is not None else None
        if tc_lock is not None:
            tc_lock.acquire()
        try:
            if self._teacache is not None:
                self._teacache.reset()

            for t in range(num_steps):
                sigma_t = sigmas[t].reshape((1,))
                timestep = mx.ones_like(sigma_t) - sigma_t

                if self._teacache is not None:
                    noise_pred = self._teacache.forward(
                        self._transformer, latents, timestep, sigmas, cap_feats,
                    )
                else:
                    noise_pred = self._transformer(
                        x=latents,
                        timestep=timestep,
                        sigmas=sigmas,
                        cap_feats=cap_feats,
                    )

                # Euler step
                dt = sigmas[t + 1] - sigmas[t]
                denoised = latents + noise_pred * dt

                # Blend: keep known regions from the *previous step's latents*
                # (not the original VAE encoding, which is at a different noise
                # level and would cause visible seams).  Take denoised output for
                # the masked (inpainted) regions.
                latents = (1 - mask_4d) * latents + mask_4d * denoised
                mx.eval(latents)

            if self._teacache is not None:
                tc_stats = self._teacache.get_stats()
                logger.info(f"TeaCache (inpaint): {tc_stats['cache_hits']} hits, "
                            f"{tc_stats['cache_misses']} misses, "
                            f"hit_rate={tc_stats['hit_rate']:.1%}")
        finally:
            if tc_lock is not None:
                tc_lock.release()

        # 8. VAE decode (auto-tile for large images)
        if width * height > 1024 * 1024:
            image = self._vae.decode_tiled(latents, tile_size_px=512, overlap_px=64)
        else:
            image = self._vae.decode(latents)
        mx.eval(image)

        return self._to_png(image)

    def _load_mask(self, mask_data: bytes, latent_h: int, latent_w: int) -> mx.array:
        """Load mask image → (1, 1, H/8, W/8) float tensor.

        White pixels (value > 127) → 1.0 (inpainted region)
        Black pixels → 0.0 (preserved region)
        """
        from PIL import Image as PILImage
        pil = PILImage.open(io.BytesIO(mask_data)).convert("L")
        pil = pil.resize((latent_w, latent_h), PILImage.NEAREST)
        arr = np.array(pil, dtype=np.float32) / 255.0
        # Threshold: any pixel > 0.5 is masked (to inpaint)
        arr = (arr > 0.5).astype(np.float32)
        return mx.array(arr[np.newaxis, np.newaxis, :, :]).astype(mx.float16)

    async def inpaint(
        self,
        prompt: str,
        image: bytes,
        mask: bytes | None = None,
        mask_base64: str | None = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        seed: int | None = None,
        denoise_strength: float = 1.0,
    ) -> bytes:
        """Inpaint masked regions of an image guided by a text prompt.

        Args:
            prompt: Text description of what to fill in.
            image: Source image bytes (PNG/JPEG).
            mask: Mask image bytes (white=fill, black=keep). Optional.
            mask_base64: Base64-encoded mask. Optional, used if mask bytes not provided.
            width: Output width.
            height: Output height.
            num_inference_steps: Number of denoising steps.
            seed: Random seed.
            denoise_strength: How much to re-denoise (1.0=full, 0.5=partial).

        Returns:
            PNG bytes of the inpainted image.
        """
        if self._transformer is None:
            raise RuntimeError("Engine not started")

        def _inpaint_sync() -> bytes:
            return self._run_inpaint_pipeline(
                prompt=prompt,
                image_data=image,
                mask_data=mask,
                mask_base64=mask_base64,
                width=width,
                height=height,
                num_steps=num_inference_steps,
                seed=seed if seed is not None else 42,
                denoise_strength=denoise_strength,
            )

        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        png_bytes = await loop.run_in_executor(self._executor, _inpaint_sync)
        elapsed = time.monotonic() - t0
        logger.info(f"Inpaint: {elapsed:.2f}s, prompt='{prompt[:50]}...'")
        return png_bytes

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
        stats = {
            "model": self._model_path,
            "loaded": self.is_loaded,
            "running": self._running,
        }
        if self._dflash is not None:
            stats["dflash"] = self._dflash.get_stats()
        return stats

    def _run_controlled_pipeline(
        self,
        prompt: str,
        condition_image_data: bytes,
        condition_type: str,
        width: int,
        height: int,
        num_steps: int,
        seed: int,
        controlnet_strength: float,
        canny_low: int,
        canny_high: int,
    ) -> bytes:
        """Run ControlNet-conditioned generation.

        Applies spatial conditioning (canny edges, depth map, etc.) during
        the denoising loop using a ControlNetBlock to inject conditioning signals.
        """
        from .controlnet_engine import (
            ConditioningPreprocessor,
            ControlNetBlock,
            ControlNetConfig,
        )

        # 1. Pre-process conditioning image → latent conditioning
        condition_result = ConditioningPreprocessor.image_to_condition_latents(
            image_data=condition_image_data,
            vae=self._vae,
            width=width,
            height=height,
            condition_type=condition_type,
            canny_low=canny_low,
            canny_high=canny_high,
        )
        condition_latents = condition_result.condition_latents

        # 2. Create ControlNet block
        cn_config = ControlNetConfig(
            condition_type=condition_type,
            controlnet_strength=controlnet_strength,
        )
        cn_block = ControlNetBlock(cn_config)

        # 3. Tokenize prompt
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

        # 4. Text encoding
        cap_feats = self._text_encoder(input_ids, attention_mask)
        num_valid = int(mx.sum(attention_mask[0]).item())
        cap_feats = cap_feats[0, :num_valid, :]
        mx.eval(cap_feats)

        # 5. Prepare noise latents
        latent_h = height // 8
        latent_w = width // 8
        latents = mx.random.normal(
            shape=[16, 1, latent_h, latent_w],
            key=mx.random.key(seed),
        ).astype(mx.float16)

        # 6. Compute sigma schedule (respects YUNSHU_DIFFUSION_SCHEDULER opt-in)
        sigmas = self._resolve_sigmas(num_steps, width, height)

        # 7. Conditioned denoising loop
        for t in range(num_steps):
            sigma_t = sigmas[t].reshape((1,))
            timestep = mx.ones_like(sigma_t) - sigma_t

            # Compute the conditioning signal for this step without
            # mutating the running latent state.  The conditioning bias
            # is applied only to the transformer input so the Euler step
            # operates on the original latents.  Previously inject_condition
            # overwrote `latents` before the Euler step, causing the
            # conditioning signal to be double-counted.
            conditioned_input = cn_block.inject_condition(
                latents, condition_latents, t, num_steps,
            )

            noise_pred = self._transformer(
                x=conditioned_input,
                timestep=timestep,
                sigmas=sigmas,
                cap_feats=cap_feats,
            )

            dt = sigmas[t + 1] - sigmas[t]
            latents = latents + noise_pred * dt
            mx.eval(latents)

        # 8. VAE decode
        if width * height > 1024 * 1024:
            image = self._vae.decode_tiled(latents, tile_size_px=512, overlap_px=64)
        else:
            image = self._vae.decode(latents)
        mx.eval(image)

        return self._to_png(image)

    async def generate_controlled(
        self,
        prompt: str,
        condition_image: bytes,
        condition_type: str = "canny",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        seed: int | None = None,
        controlnet_strength: float = 1.0,
        canny_low: int = 100,
        canny_high: int = 200,
    ) -> bytes:
        """Generate an image with ControlNet spatial conditioning.

        Args:
            prompt: Text prompt.
            condition_image: Conditioning image bytes (edges, depth map, etc.).
            condition_type: Type of conditioning (canny, depth, raw).
            width: Output width.
            height: Output height.
            num_inference_steps: Denoising steps.
            seed: Random seed.
            controlnet_strength: Conditioning strength (0.0–1.0).
            canny_low: Canny lower threshold.
            canny_high: Canny upper threshold.

        Returns:
            PNG bytes of the generated image.
        """
        if self._transformer is None:
            raise RuntimeError("Engine not started")

        def _controlled_sync() -> bytes:
            return self._run_controlled_pipeline(
                prompt=prompt,
                condition_image_data=condition_image,
                condition_type=condition_type,
                width=width,
                height=height,
                num_steps=num_inference_steps,
                seed=seed if seed is not None else 42,
                controlnet_strength=controlnet_strength,
                canny_low=canny_low,
                canny_high=canny_high,
            )

        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        png_bytes = await loop.run_in_executor(self._executor, _controlled_sync)
        elapsed = time.monotonic() - t0
        logger.info(f"ControlNet gen: {elapsed:.2f}s, type={condition_type}, prompt='{prompt[:50]}...'")
        return png_bytes

    def _run_depth_guided_pipeline(
        self,
        prompt: str,
        depth_image_data: bytes,
        width: int,
        height: int,
        num_steps: int,
        seed: int,
        depth_strength: float,
    ) -> bytes:
        """Run depth-guided generation.

        Encodes depth map as additional latent channels and uses it
        to guide the denoising process.
        """
        from .controlnet_engine import DepthGuider

        # 1. Prepare depth latents
        depth_latents = DepthGuider.prepare_depth_latents(
            depth_image=depth_image_data,
            vae=self._vae,
            width=width,
            height=height,
        )

        # 2. Scale depth by strength
        if depth_strength != 1.0:
            depth_latents = depth_latents * depth_strength

        # 3. Tokenize prompt
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

        # 4. Text encoding
        cap_feats = self._text_encoder(input_ids, attention_mask)
        num_valid = int(mx.sum(attention_mask[0]).item())
        cap_feats = cap_feats[0, :num_valid, :]
        mx.eval(cap_feats)

        # 5. Prepare noise latents
        latent_h = height // 8
        latent_w = width // 8
        latents = mx.random.normal(
            shape=[16, 1, latent_h, latent_w],
            key=mx.random.key(seed),
        ).astype(mx.float16)

        # 6. Add depth conditioning as a bias to latents
        if depth_latents.ndim == 3:
            depth_4d = depth_latents[:, np.newaxis, :, :]
        else:
            depth_4d = depth_latents

        # 7. Compute sigma schedule (respects YUNSHU_DIFFUSION_SCHEDULER opt-in)
        sigmas = self._resolve_sigmas(num_steps, width, height)

        # 8. Depth-conditioned denoising loop
        for t in range(num_steps):
            # Blend depth conditioning into latents at each step
            step_frac = t / max(num_steps, 1)
            depth_weight = depth_strength * (1.0 - step_frac) * 0.1
            conditioned = latents + depth_4d * depth_weight

            sigma_t = sigmas[t].reshape((1,))
            timestep = mx.ones_like(sigma_t) - sigma_t

            noise_pred = self._transformer(
                x=conditioned,
                timestep=timestep,
                sigmas=sigmas,
                cap_feats=cap_feats,
            )

            dt = sigmas[t + 1] - sigmas[t]
            latents = latents + noise_pred * dt
            mx.eval(latents)

        # 9. VAE decode
        if width * height > 1024 * 1024:
            image = self._vae.decode_tiled(latents, tile_size_px=512, overlap_px=64)
        else:
            image = self._vae.decode(latents)
        mx.eval(image)

        return self._to_png(image)

    async def generate_depth_guided(
        self,
        prompt: str,
        depth_image: bytes,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        seed: int | None = None,
        depth_strength: float = 1.0,
    ) -> bytes:
        """Generate a depth-guided image.

        Args:
            prompt: Text prompt.
            depth_image: Depth visualization image bytes.
            width: Output width.
            height: Output height.
            num_inference_steps: Denoising steps.
            seed: Random seed.
            depth_strength: Depth conditioning strength (0.0–1.0).

        Returns:
            PNG bytes of the generated image.
        """
        if self._transformer is None:
            raise RuntimeError("Engine not started")

        def _depth_sync() -> bytes:
            return self._run_depth_guided_pipeline(
                prompt=prompt,
                depth_image_data=depth_image,
                width=width,
                height=height,
                num_steps=num_inference_steps,
                seed=seed if seed is not None else 42,
                depth_strength=depth_strength,
            )

        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        png_bytes = await loop.run_in_executor(self._executor, _depth_sync)
        elapsed = time.monotonic() - t0
        logger.info(f"Depth-guided gen: {elapsed:.2f}s, prompt='{prompt[:50]}...'")
        return png_bytes

    def load_lora_adapter(self, adapter_path: str, rank: int = 8, scale: float = 20.0) -> bool:
        """Load a LoRA adapter into the image transformer.

        The adapter's config should specify which layers to apply LoRA to.
        Typical targets: attention Q/V projections in the DiT transformer.
        """
        if self._transformer is None:
            logger.error("Cannot load LoRA: transformer not loaded")
            return False

        import json
        from pathlib import Path

        adapter_dir = Path(adapter_path)
        config_path = adapter_dir / "adapter_config.json"

        if not config_path.exists():
            logger.error(f"No adapter_config.json in {adapter_path}")
            return False

        with open(config_path) as f:
            config = json.load(f)

        lora_params = config.get("lora_parameters", {})
        _rank = lora_params.get("rank", rank)
        _scale = lora_params.get("scale", scale)
        num_layers = config.get("num_layers", 16)

        try:
            from mlx_lm.tuner.lora import LoRALinear
            import mlx.nn as nn

            applied = 0
            for name, module in self._transformer.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                if applied >= num_layers:
                    break
                # Apply to Q and V projections (standard LoRA targets)
                if any(k in name for k in ("q_proj", "v_proj", "qkv")):
                    self._original_modules[name] = module
                    lora_layer = LoRALinear(
                        module.in_features,
                        module.out_features,
                        rank=_rank,
                        scale=_scale,
                    )
                    lora_layer.linear = module
                    # Set the LoRA layer on the parent module
                    parts = name.rsplit(".", 1)
                    if len(parts) == 2:
                        parent = self._transformer
                        for part in parts[0].split("."):
                            parent = getattr(parent, part)
                        setattr(parent, parts[1], lora_layer)
                    applied += 1

            # Load adapter weights
            weights_path = adapter_dir / "adapters.safetensors"
            if weights_path.exists():
                self._transformer.load_weights(str(weights_path), strict=False)

            mx.eval(self._transformer.parameters())
            logger.info(f"LoRA adapter loaded: {adapter_path}, {applied} layers")

            # Register with LoRA offloader if active
            if self._lora_offloader is not None:
                adapter_id = adapter_path.rsplit("/", 1)[-1] if "/" in adapter_path else adapter_path
                # Estimate memory: rank * (in + out) * 4 bytes per layer
                est_bytes = applied * _rank * (256 + 256) * 4  # rough estimate
                self._lora_offloader.register_adapter(
                    lora_id=adapter_id,
                    memory_bytes=est_bytes,
                    priority=int(_scale),
                )
                logger.info(f"LoRA adapter registered with offloader: {adapter_id}")

            return True
        except Exception as e:
            logger.error(f"Failed to load LoRA adapter: {e}", exc_info=True)
            return False

    def unload_lora_adapter(self) -> bool:
        """Restore original Linear modules, removing LoRA wrappers."""
        if self._transformer is None or not self._original_modules:
            return False
        try:
            for name, orig_module in self._original_modules.items():
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = self._transformer
                    for part in parts[0].split("."):
                        parent = getattr(parent, part)
                    setattr(parent, parts[1], orig_module)
            self._original_modules.clear()
            mx.eval(self._transformer.parameters())
            logger.info("LoRA adapter unloaded, original modules restored")
            return True
        except Exception as e:
            logger.error(f"Failed to unload LoRA adapter: {e}", exc_info=True)
            return False
