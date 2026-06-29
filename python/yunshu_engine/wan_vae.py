"""Wan2.2 48-channel 3D causal VAE — MLX port (Lance-3B image/video generation).

Port of ByteDance/Lance ``modeling/vae/wan/vae2_2.py`` to MLX. This is the
latent<->pixel codec the Lance dual-expert tower uses for image_gen (t2i) and
video (t2v); the Qwen2.5-VL backbone itself is already supported by mlx-vlm
(Lance-3B config is ``model_type: qwen2_5_vl``), so the VAE + the gen loop are
the genuinely-new work.

LAYOUT NOTE: PyTorch convs are channels-first ``(N, C, D, H, W)``; MLX convs are
channels-last ``(N, D, H, W, C)``. This port keeps tensors channels-last
throughout and translates the reference's NCDHW ops accordingly.

STATUS: foundational blocks (RMS norm, causal 3D conv) are ported + structurally
tested here. The full Encoder3d/Decoder3d + the ``feat_cache`` temporal-streaming
mechanism + numerical validation against the real Wan2.2 weights are the
remaining (multi-day) work.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class WanRMSNorm(nn.Module):
    """Channels-last RMS norm matching the reference ``RMS_norm``:
    ``F.normalize(x, dim=channel) * sqrt(dim) * gamma + bias``.

    The reference normalizes over the channel dim (L2-normalize, not the usual
    mean-square RMS) then scales by ``sqrt(dim)`` and a per-channel ``gamma``.
    """

    def __init__(self, dim: int, bias: bool = False) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = mx.ones((dim,))
        self.bias = mx.zeros((dim,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        # x: (..., C); L2-normalize over the last (channel) axis.
        f = x.astype(mx.float32)
        normed = f * mx.rsqrt((f * f).sum(axis=-1, keepdims=True) + 1e-12)
        out = normed * self.scale * self.gamma.astype(mx.float32)
        if self.bias is not None:
            out = out + self.bias.astype(mx.float32)
        return out.astype(x.dtype)


class WanCausalConv3d(nn.Module):
    """Causal 3D convolution (channels-last).

    Reference ``CausalConv3d`` pads the temporal axis causally (front by
    ``2*pad_t``, back by 0) and the spatial axes symmetrically, then applies a
    zero-pad conv. With an optional ``cache_x`` (the last frames of the previous
    chunk) the front padding is reduced by the cached length so chunked decoding
    is seamless.

    kernel/stride/padding are given as ``(T, H, W)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride)
        self.pad_t, self.pad_h, self.pad_w = padding

    def __call__(self, x: mx.array, cache_x: mx.array | None = None) -> mx.array:
        # x: (N, D, H, W, C)
        front = 2 * self.pad_t
        if cache_x is not None and front > 0:
            x = mx.concatenate([cache_x, x], axis=1)  # prepend along temporal
            front = max(0, front - cache_x.shape[1])
        x = mx.pad(
            x,
            [
                (0, 0),
                (front, 0),
                (self.pad_h, self.pad_h),
                (self.pad_w, self.pad_w),
                (0, 0),
            ],
        )
        return self.conv(x)


class WanResidualBlock(nn.Module):
    """RMS→SiLU→conv→RMS→SiLU→conv with a 1x1 causal-conv shortcut when the
    channel count changes. (Non-streaming path: full-tensor, sufficient for
    single-image t2i; the feat_cache streaming variant is for long video.)"""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm1 = WanRMSNorm(in_dim, bias=False)
        self.conv1 = WanCausalConv3d(in_dim, out_dim, (3, 3, 3), padding=(1, 1, 1))
        self.norm2 = WanRMSNorm(out_dim, bias=False)
        self.conv2 = WanCausalConv3d(out_dim, out_dim, (3, 3, 3), padding=(1, 1, 1))
        self.shortcut = (
            WanCausalConv3d(in_dim, out_dim, (1, 1, 1)) if in_dim != out_dim else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = x if self.shortcut is None else self.shortcut(x)
        y = self.conv1(nn.silu(self.norm1(x)))
        y = self.conv2(nn.silu(self.norm2(y)))
        return y + h


class WanAttentionBlock(nn.Module):
    """Single-head spatial self-attention applied per frame (channels-last)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.norm = WanRMSNorm(dim, bias=False)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, D, H, W, C) — flatten frames into the batch for per-frame attn.
        n, d, h, w, c = x.shape
        identity = x
        xf = x.reshape(n * d, h, w, c)
        xf = self.norm(xf)
        qkv = self.to_qkv(xf).reshape(n * d, h * w, 3, c)
        q = qkv[:, :, 0][:, None]  # (N*D, 1head, HW, C)
        k = qkv[:, :, 1][:, None]
        v = qkv[:, :, 2][:, None]
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=c**-0.5)
        out = out[:, 0].reshape(n * d, h, w, c)
        out = self.proj(out)
        return out.reshape(n, d, h, w, c) + identity


class WanAvgDown3D(nn.Module):
    """Space-to-channel + grouped-average downsample (channels-last).

    Folds (factor_t × factor_s × factor_s) neighbourhood into the channel axis
    with C-outermost ordering (matching the reference permute), then averages
    over the resulting groups to land at ``out_channels``.
    """

    def __init__(
        self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, H, W, C)
        ft, fs = self.factor_t, self.factor_s
        b, t, h, w, c = x.shape
        pad_t = (ft - t % ft) % ft
        if pad_t:
            x = mx.pad(x, [(0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)])
            t = t + pad_t
        tp, hp, wp = t // ft, h // fs, w // fs
        x = x.reshape(b, tp, ft, hp, fs, wp, fs, c)
        # -> (B, Tp, Hp, Wp, C, ft, fs, fs): C outermost in the folded channel.
        x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)
        x = x.reshape(b, tp, hp, wp, c * self.factor)
        x = x.reshape(b, tp, hp, wp, self.out_channels, self.group_size)
        return x.mean(axis=-1)


class WanDupUp3D(nn.Module):
    """Channel-to-space duplicate-upsample (inverse of WanAvgDown3D)."""

    def __init__(
        self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def __call__(self, x: mx.array, first_chunk: bool = False) -> mx.array:
        # x: (B, T, H, W, C_in)
        ft, fs = self.factor_t, self.factor_s
        b, t, h, w, _ = x.shape
        x = mx.repeat(x, self.repeats, axis=-1)  # (B,T,H,W, C_in*repeats)
        x = x.reshape(b, t, h, w, self.out_channels, ft, fs, fs)
        # -> (B, T, ft, H, fs, W, fs, out) then merge factors into T/H/W.
        x = x.transpose(0, 4, 1, 5, 2, 6, 3, 7)  # (B, out, T, ft, H, fs, W, fs)
        x = x.reshape(b, self.out_channels, t * ft, h * fs, w * fs)
        x = x.transpose(0, 2, 3, 4, 1)  # back to channels-last (B, T', H', W', out)
        if first_chunk:
            x = x[:, ft - 1 :, :, :, :]
        return x


class WanResample(nn.Module):
    """Spatial 2x up/down sampling (channels-last). Image / first-chunk path:
    the reference skips the temporal ``time_conv`` on the first chunk (its
    "Rep" semantics), so single-image decode only needs the spatial conv. The
    ``time_conv`` weights are still constructed so video checkpoints load.
    """

    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        if mode in ("upsample2d", "upsample3d"):
            self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        elif mode in ("downsample2d", "downsample3d"):
            self.conv = nn.Conv2d(dim, dim, 3, stride=2)  # ZeroPad applied below
        else:
            self.conv = None
        if mode == "upsample3d":
            self.time_conv = WanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample3d":
            self.time_conv = WanCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1))
        else:
            self.time_conv = None

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, H, W, C). First-chunk image path → spatial only.
        b, t, h, w, c = x.shape
        xf = x.reshape(b * t, h, w, c)
        if self.mode.startswith("upsample"):
            # nearest-exact 2x == per-pixel duplicate for integer scale.
            xf = mx.broadcast_to(
                xf[:, :, None, :, None, :], (b * t, h, 2, w, 2, c)
            ).reshape(b * t, 2 * h, 2 * w, c)
            xf = self.conv(xf)
        else:
            # ZeroPad2d((left=0,right=1,top=0,bottom=1)) then stride-2 conv.
            xf = mx.pad(xf, [(0, 0), (0, 1), (0, 1), (0, 0)])
            xf = self.conv(xf)
        return xf.reshape(b, t, xf.shape[1], xf.shape[2], xf.shape[3])


class WanUpResidualBlock(nn.Module):
    """``mult`` residual blocks + optional spatial upsample, with a DupUp3D
    shortcut. Image first-chunk path (T preserved at 1)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        mult: int,
        temperal_upsample: bool = False,
        up_flag: bool = False,
    ) -> None:
        super().__init__()
        self.avg_shortcut = (
            WanDupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
            if up_flag
            else None
        )
        ups: list = []
        d = in_dim
        for _ in range(mult):
            ups.append(WanResidualBlock(d, out_dim))
            d = out_dim
        if up_flag:
            ups.append(
                WanResample(
                    out_dim, "upsample3d" if temperal_upsample else "upsample2d"
                )
            )
        self.upsamples = ups

    def __call__(self, x: mx.array, first_chunk: bool = True) -> mx.array:
        h = x
        for m in self.upsamples:
            h = m(h)
        if self.avg_shortcut is not None:
            return h + self.avg_shortcut(x, first_chunk)
        return h


class WanDownResidualBlock(nn.Module):
    """``mult`` residual blocks + optional spatial downsample, with an AvgDown3D
    shortcut (encoder counterpart of WanUpResidualBlock)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        mult: int,
        temperal_downsample: bool = False,
        down_flag: bool = False,
    ) -> None:
        super().__init__()
        self.avg_shortcut = WanAvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        downs: list = []
        d = in_dim
        for _ in range(mult):
            downs.append(WanResidualBlock(d, out_dim))
            d = out_dim
        if down_flag:
            downs.append(
                WanResample(
                    out_dim, "downsample3d" if temperal_downsample else "downsample2d"
                )
            )
        self.downsamples = downs

    def __call__(self, x: mx.array) -> mx.array:
        h = x
        for m in self.downsamples:
            h = m(h)
        return h + self.avg_shortcut(x)


class WanEncoder3d(nn.Module):
    """Wan2.2 VAE encoder (image first-chunk path): conv1(12→dim) → 4×
    Down_ResidualBlock → middle(Res,Attn,Res) → head(RMS,SiLU,conv→z_dim*2)."""

    def __init__(
        self,
        dim: int = 160,
        z_dim: int = 96,  # encoder outputs z_dim*2 (mu+logvar); pass 2*z here
        dim_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temperal_downsample: tuple[bool, ...] = (False, True, True),
    ) -> None:
        super().__init__()
        dims = [dim * u for u in [1, *list(dim_mult)]]
        self.conv1 = WanCausalConv3d(12, dims[0], (3, 3, 3), padding=(1, 1, 1))
        downs = []
        n = len(dim_mult)
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
            t_down = temperal_downsample[i] if i < len(temperal_downsample) else False
            downs.append(
                WanDownResidualBlock(
                    in_dim,
                    out_dim,
                    mult=num_res_blocks,
                    temperal_downsample=t_down,
                    down_flag=i != n - 1,
                )
            )
        self.downsamples = downs
        self.middle = [
            WanResidualBlock(out_dim, out_dim),
            WanAttentionBlock(out_dim),
            WanResidualBlock(out_dim, out_dim),
        ]
        self.head_norm = WanRMSNorm(out_dim, bias=False)
        self.head_conv = WanCausalConv3d(out_dim, z_dim, (3, 3, 3), padding=(1, 1, 1))

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(x)
        for m in self.downsamples:
            h = m(h)
        for m in self.middle:
            h = m(h)
        return self.head_conv(nn.silu(self.head_norm(h)))


def patchify_chw(x: mx.array, patch_size: int = 2) -> mx.array:
    """Channel-patchify (channels-last 5D): ``(B,T,H,W,C) ->
    (B,T,H/p,W/p, C*p*p)`` with C-outermost channel order (inverse of
    unpatchify_chw)."""
    if patch_size == 1:
        return x
    b, t, h, w, c = x.shape
    p = patch_size
    x = x.reshape(b, t, h // p, p, w // p, p, c)
    x = x.transpose(0, 1, 2, 4, 6, 5, 3)  # (B,T,h/p,w/p, C,r,q)
    return x.reshape(b, t, h // p, w // p, c * p * p)


class WanDecoder3d(nn.Module):
    """Wan2.2 VAE decoder (image first-chunk path): conv1 → middle(Res,Attn,Res)
    → 4× Up_ResidualBlock → head(RMS,SiLU,conv→12) → (unpatchify done by caller).
    """

    def __init__(
        self,
        dim: int = 256,
        z_dim: int = 48,
        dim_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temperal_upsample: tuple[bool, ...] = (False, True, True),
    ) -> None:
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1], *list(dim_mult)[::-1]]]
        self.conv1 = WanCausalConv3d(z_dim, dims[0], (3, 3, 3), padding=(1, 1, 1))
        self.middle = [
            WanResidualBlock(dims[0], dims[0]),
            WanAttentionBlock(dims[0]),
            WanResidualBlock(dims[0], dims[0]),
        ]
        ups = []
        n = len(dim_mult)
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
            t_up = temperal_upsample[i] if i < len(temperal_upsample) else False
            ups.append(
                WanUpResidualBlock(
                    in_dim,
                    out_dim,
                    mult=num_res_blocks + 1,
                    temperal_upsample=t_up,
                    up_flag=i != n - 1,
                )
            )
        self.upsamples = ups
        self.head_norm = WanRMSNorm(out_dim, bias=False)
        self.head_conv = WanCausalConv3d(out_dim, 12, (3, 3, 3), padding=(1, 1, 1))

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(x)
        for m in self.middle:
            h = m(h)
        for m in self.upsamples:
            h = m(h, first_chunk=True)
        h = self.head_conv(nn.silu(self.head_norm(h)))
        return h


def _map_vae_key(k: str, prefix: str) -> str:
    """Translate a converted ``{prefix}*`` safetensors key (channels-last,
    layer_N / *_weight naming) to this module tree's parameter path. Works for
    both ``decoder.`` and ``encoder.`` (same block naming)."""
    import re

    k = k[len(prefix) :]
    for a, b in (
        ("residual.layer_0.", "norm1."),
        ("residual.layer_2.", "conv1.conv."),
        ("residual.layer_3.", "norm2."),
        ("residual.layer_6.", "conv2.conv."),
        ("head.layer_0.", "head_norm."),
        ("head.layer_2.", "head_conv.conv."),
        ("to_qkv_weight", "to_qkv.weight"),
        ("to_qkv_bias", "to_qkv.bias"),
        ("proj_weight", "proj.weight"),
        ("proj_bias", "proj.bias"),
        ("resample_weight", "conv.weight"),
        ("resample_bias", "conv.bias"),
        ("shortcut.", "shortcut.conv."),
        ("time_conv.weight", "time_conv.conv.weight"),
        ("time_conv.bias", "time_conv.conv.bias"),
    ):
        k = k.replace(a, b)
    if re.match(r"conv1\.(weight|bias)$", k):  # top-level decoder.conv1
        k = "conv1.conv." + k.split(".")[-1]
    return k


def load_wan_decoder(path: str | Path) -> WanDecoder3d:
    """Load the Wan2.2 VAE *decoder* from a converted (channels-last) MLX
    ``vae.safetensors`` (Lance-3B image/video gen). Values are already
    channels-last so no transpose is needed — only key renaming.
    """
    from mlx.utils import tree_unflatten

    raw = mx.load(str(path))
    mapped = [
        (_map_vae_key(k, "decoder."), v.astype(mx.float32))
        for k, v in raw.items()
        if k.startswith("decoder.")
    ]
    dec = WanDecoder3d(
        dim=256,
        z_dim=48,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        temperal_upsample=(True, True, False),
    )
    dec.update(tree_unflatten(mapped))
    mx.eval(dec.parameters())
    return dec


def load_wan_encoder(path: str | Path) -> WanEncoder3d:
    """Load the Wan2.2 VAE *encoder* from a converted (channels-last) MLX
    ``vae.safetensors`` (image_edit / x2t). Pure key-renaming, no transpose."""
    from mlx.utils import tree_unflatten

    raw = mx.load(str(path))
    mapped = [
        (_map_vae_key(k, "encoder."), v.astype(mx.float32))
        for k, v in raw.items()
        if k.startswith("encoder.")
    ]
    enc = WanEncoder3d(
        dim=160,
        z_dim=96,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        temperal_downsample=(False, True, True),
    )
    enc.update(tree_unflatten(mapped))
    mx.eval(enc.parameters())
    return enc


# Latent normalization (Wan2.2 VAE scale = [mean, 1/std]), 48 channels.
_WAN_MEAN = [
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.1570,
    -0.0098,
    0.0375,
    -0.1825,
    -0.2246,
    -0.1207,
    -0.0698,
    0.5109,
    0.2665,
    -0.2108,
    -0.2158,
    0.2502,
    -0.2055,
    -0.0322,
    0.1109,
    0.1567,
    -0.0729,
    0.0899,
    -0.2799,
    -0.1230,
    -0.0313,
    -0.1649,
    0.0117,
    0.0723,
    -0.2839,
    -0.2083,
    -0.0520,
    0.3748,
    0.0152,
    0.1957,
    0.1433,
    -0.2944,
    0.3573,
    -0.0548,
    -0.1681,
    -0.0667,
]
_WAN_STD = [
    0.4765,
    1.0364,
    0.4514,
    1.1677,
    0.5313,
    0.4990,
    0.4818,
    0.5013,
    0.8158,
    1.0344,
    0.5894,
    1.0901,
    0.6885,
    0.6165,
    0.8454,
    0.4978,
    0.5759,
    0.3523,
    0.7135,
    0.6804,
    0.5833,
    1.4146,
    0.8986,
    0.5659,
    0.7069,
    0.5338,
    0.4889,
    0.4917,
    0.4069,
    0.4999,
    0.6866,
    0.4093,
    0.5709,
    0.6065,
    0.6415,
    0.4944,
    0.5726,
    1.2042,
    0.5458,
    1.6887,
    0.3971,
    1.0600,
    0.3943,
    0.5537,
    0.5444,
    0.4089,
    0.7468,
    0.7744,
]


class WanVAE:
    """Usable Wan2.2 VAE (image path): encode(img)->latent, decode(latent)->img,
    with the conv1/conv2 projections + latent-scale normalization, matching the
    reference ``Wan2_2_VAE``. Single-image (T=1) first-chunk path.
    """

    def __init__(
        self,
        encoder: WanEncoder3d,
        decoder: WanDecoder3d,
        conv1: WanCausalConv3d,
        conv2: WanCausalConv3d,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.conv1 = conv1  # post-encoder, 96->96
        self.conv2 = conv2  # pre-decoder, 48->48
        self.mean = mx.array(_WAN_MEAN)
        self.inv_std = mx.array([1.0 / s for s in _WAN_STD])

    def encode(self, img: mx.array) -> mx.array:
        """img (B,T,H,W,3) in [-1,1] -> normalized mean latent (B,T,h,w,48)."""
        out = self.encoder(patchify_chw(img, 2))
        params = self.conv1(out)
        mu = params[..., :48]
        return (mu - self.mean) * self.inv_std

    def decode(self, z: mx.array) -> mx.array:
        """normalized latent (B,T,h,w,48) -> img (B,T,H,W,3) clamped to [-1,1]."""
        z = z / self.inv_std + self.mean
        x = self.conv2(z)
        out = self.decoder(x)
        img = unpatchify_chw(out, 2)
        return mx.clip(img, -1.0, 1.0)


def load_wan_vae(path: str | Path) -> WanVAE:
    """Load the full image-path Wan2.2 VAE (encoder+decoder+conv1+conv2) from a
    converted channels-last ``vae.safetensors``."""
    from mlx.utils import tree_unflatten

    raw = mx.load(str(path))
    enc = load_wan_encoder(path)
    dec = load_wan_decoder(path)
    conv1 = WanCausalConv3d(96, 96, (1, 1, 1))
    conv2 = WanCausalConv3d(48, 48, (1, 1, 1))
    conv1.update(
        tree_unflatten(
            [
                ("conv.weight", raw["conv1.weight"].astype(mx.float32)),
                ("conv.bias", raw["conv1.bias"].astype(mx.float32)),
            ]
        )
    )
    conv2.update(
        tree_unflatten(
            [
                ("conv.weight", raw["conv2.weight"].astype(mx.float32)),
                ("conv.bias", raw["conv2.bias"].astype(mx.float32)),
            ]
        )
    )
    mx.eval(conv1.parameters(), conv2.parameters())
    return WanVAE(enc, dec, conv1, conv2)


def unpatchify_chw(x: mx.array, patch_size: int = 2) -> mx.array:
    """Inverse of the VAE's channel-patchify (channels-last 5D):
    ``(B,T,H,W, C*p*p) -> (B,T, H*p, W*p, C)`` with C-outermost channel order."""
    if patch_size == 1:
        return x
    b, t, h, w, cpp = x.shape
    p = patch_size
    c = cpp // (p * p)
    x = x.reshape(b, t, h, w, c, p, p)  # (c, r, q) with c outermost
    x = x.transpose(0, 1, 2, 6, 3, 5, 4)  # (B,T,h,q,w,r,c)
    return x.reshape(b, t, h * p, w * p, c)
