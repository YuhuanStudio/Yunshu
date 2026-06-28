"""Structural tests for the Wan2.2 VAE foundational blocks (MLX port)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from yunshu_engine.wan_vae import WanCausalConv3d, WanRMSNorm


def test_rmsnorm_l2_normalizes_channel():
    norm = WanRMSNorm(8)
    x = mx.array(np.random.default_rng(0).standard_normal((2, 3, 4, 4, 8)).astype(np.float32))
    out = np.asarray(norm(x))
    assert out.shape == (2, 3, 4, 4, 8)
    # default gamma=1, bias=None: output = unit-L2(x) * sqrt(8) along channel.
    chan_norm = np.linalg.norm(out, axis=-1)
    np.testing.assert_allclose(chan_norm, np.full((2, 3, 4, 4), 8**0.5), rtol=1e-3, atol=1e-3)


def test_rmsnorm_gamma_scales():
    norm = WanRMSNorm(4)
    norm.gamma = mx.full((4,), 2.0)
    x = mx.array(np.ones((1, 1, 1, 1, 4), dtype=np.float32))
    out = np.asarray(norm(x))
    # unit-L2 of [1,1,1,1] = 0.5 each; *sqrt(4)=2; *gamma 2 = 2.0
    np.testing.assert_allclose(out[0, 0, 0, 0], np.full(4, 2.0), rtol=1e-4, atol=1e-4)


def test_causal_conv3d_shape_and_causal_pad():
    # kernel (3,3,3), pad (1,1,1): temporal causal (front 2, back 0),
    # spatial symmetric. Output temporal length == input (front pad compensates).
    conv = WanCausalConv3d(4, 6, (3, 3, 3), padding=(1, 1, 1))
    x = mx.array(np.random.default_rng(1).standard_normal((1, 5, 8, 8, 4)).astype(np.float32))
    out = conv(x)
    mx.eval(out)
    assert out.shape[0] == 1 and out.shape[-1] == 6  # batch, out-channels
    # temporal: in 5 + front-pad 2 - (kernel 3 - 1) = 5 (causal, length preserved)
    assert out.shape[1] == 5
    assert not np.any(np.isnan(np.asarray(out)))


def test_causal_conv3d_cache_continuity():
    # With cache_x (prev chunk's last frames), the front pad is reduced so the
    # chunked result has the expected temporal length.
    conv = WanCausalConv3d(4, 4, (3, 1, 1), padding=(1, 0, 0))
    x = mx.array(np.random.default_rng(2).standard_normal((1, 3, 2, 2, 4)).astype(np.float32))
    cache = mx.array(np.random.default_rng(3).standard_normal((1, 2, 2, 2, 4)).astype(np.float32))
    out = conv(x, cache_x=cache)
    mx.eval(out)
    # x(3) + cache(2) prepended, front pad 2-2=0; temporal: 5 - (3-1) = 3
    assert out.shape[1] == 3
    assert out.shape[-1] == 4


def test_residual_block_shape_in_eq_out():
    from yunshu_engine.wan_vae import WanResidualBlock
    blk = WanResidualBlock(8, 8)  # no shortcut
    x = mx.array(np.random.default_rng(4).standard_normal((1, 3, 6, 6, 8)).astype(np.float32))
    out = blk(x); mx.eval(out)
    assert out.shape == (1, 3, 6, 6, 8)
    assert not np.any(np.isnan(np.asarray(out)))


def test_residual_block_channel_change_uses_shortcut():
    from yunshu_engine.wan_vae import WanResidualBlock
    blk = WanResidualBlock(8, 16)
    assert blk.shortcut is not None
    x = mx.array(np.random.default_rng(5).standard_normal((1, 2, 4, 4, 8)).astype(np.float32))
    out = blk(x); mx.eval(out)
    assert out.shape == (1, 2, 4, 4, 16)


def test_attention_block_shape_preserved():
    from yunshu_engine.wan_vae import WanAttentionBlock
    blk = WanAttentionBlock(16)
    x = mx.array(np.random.default_rng(6).standard_normal((1, 2, 5, 5, 16)).astype(np.float32))
    out = blk(x); mx.eval(out)
    assert out.shape == (1, 2, 5, 5, 16)
    assert not np.any(np.isnan(np.asarray(out)))


def test_avgdown3d_downsamples_t_and_channels():
    from yunshu_engine.wan_vae import WanAvgDown3D
    # in=8, out=16, factor_t=2, factor_s=2 → factor=8, group=8*8/16=4
    blk = WanAvgDown3D(8, 16, factor_t=2, factor_s=2)
    x = mx.array(np.random.default_rng(7).standard_normal((1, 4, 8, 8, 8)).astype(np.float32))
    out = blk(x); mx.eval(out)
    assert out.shape == (1, 2, 4, 4, 16)  # T/2, H/2, W/2, out
    assert not np.any(np.isnan(np.asarray(out)))


def test_dupup3d_upsamples_inverse_shape():
    from yunshu_engine.wan_vae import WanDupUp3D
    blk = WanDupUp3D(16, 8, factor_t=2, factor_s=2)  # factor=8, repeats=8*8/16=4
    x = mx.array(np.random.default_rng(8).standard_normal((1, 2, 4, 4, 16)).astype(np.float32))
    out = blk(x); mx.eval(out)
    assert out.shape == (1, 4, 8, 8, 8)  # T*2, H*2, W*2, out
    assert not np.any(np.isnan(np.asarray(out)))


def test_dupup3d_first_chunk_trims_temporal():
    from yunshu_engine.wan_vae import WanDupUp3D
    blk = WanDupUp3D(16, 8, factor_t=2, factor_s=1)
    x = mx.array(np.random.default_rng(9).standard_normal((1, 2, 3, 3, 16)).astype(np.float32))
    out = blk(x, first_chunk=True); mx.eval(out)
    # T*2=4, trim (ft-1)=1 → 3
    assert out.shape[1] == 3


# --- Numerical parity vs the PyTorch reference (torch+einops+repo gated) ---

import os  # noqa: E402

import pytest  # noqa: E402

_REF = "reference/Lance"
_skip_no_ref = pytest.mark.skipif(
    not os.path.isdir(_REF), reason="Lance reference repo not present"
)


@_skip_no_ref
def test_numerical_parity_vs_torch_reference():
    torch = pytest.importorskip("torch")
    pytest.importorskip("einops")
    import sys
    sys.path.insert(0, _REF)
    from modeling.vae.wan.vae2_2 import AvgDown3D, DupUp3D

    from yunshu_engine.wan_vae import WanAvgDown3D, WanDupUp3D

    rng = np.random.default_rng(0)
    # AvgDown3D — no weights: pure permute logic, must match EXACTLY.
    x = rng.standard_normal((1, 8, 4, 8, 8)).astype(np.float32)
    t = AvgDown3D(8, 16, factor_t=2, factor_s=2)(torch.from_numpy(x)).detach().numpy()
    m = np.asarray(WanAvgDown3D(8, 16, 2, 2)(mx.array(x.transpose(0, 2, 3, 4, 1)))).transpose(0, 4, 1, 2, 3)
    assert np.allclose(t, m, atol=1e-5), float(np.abs(t - m).max())
    # DupUp3D — inverse, also weightless.
    xi = rng.standard_normal((1, 16, 2, 4, 4)).astype(np.float32)
    t2 = DupUp3D(16, 8, factor_t=2, factor_s=2)(torch.from_numpy(xi)).detach().numpy()
    m2 = np.asarray(WanDupUp3D(16, 8, 2, 2)(mx.array(xi.transpose(0, 2, 3, 4, 1)))).transpose(0, 4, 1, 2, 3)
    assert np.allclose(t2, m2, atol=1e-5), float(np.abs(t2 - m2).max())


def test_resample_upsample_downsample_shape():
    from yunshu_engine.wan_vae import WanResample
    up = WanResample(8, "upsample2d")
    x = mx.array(np.random.default_rng(11).standard_normal((1, 1, 8, 8, 8)).astype(np.float32))
    out = up(x); mx.eval(out)
    assert out.shape == (1, 1, 16, 16, 8)
    down = WanResample(8, "downsample2d")
    out2 = down(x); mx.eval(out2)
    assert out2.shape == (1, 1, 4, 4, 8)


def test_decoder3d_full_decode_path_shape():
    from yunshu_engine.wan_vae import WanDecoder3d, unpatchify_chw
    dec = WanDecoder3d(dim=8, z_dim=4, dim_mult=(1, 2), num_res_blocks=1, temperal_upsample=(True,))
    z = mx.array(np.random.default_rng(0).standard_normal((1, 1, 8, 8, 4)).astype(np.float32))
    out = dec(z); mx.eval(out)
    assert out.shape == (1, 1, 16, 16, 12)  # 1 upsample block 2x; head -> 12 ch
    assert not np.any(np.isnan(np.asarray(out)))
    img = unpatchify_chw(out, 2); mx.eval(img)
    assert img.shape == (1, 1, 32, 32, 3)  # patchify-2 inverse: 2x spatial, 3 RGB


def test_unpatchify_chw_inverse_shape():
    from yunshu_engine.wan_vae import unpatchify_chw
    x = mx.array(np.zeros((1, 1, 2, 2, 12), dtype=np.float32))
    assert unpatchify_chw(x, 2).shape == (1, 1, 4, 4, 3)
    assert unpatchify_chw(x, 1).shape == (1, 1, 2, 2, 12)  # patch_size 1 = identity


_VAE = "/Volumes/P5Plus/models/Wan2.2-VAE-Lance-bf16/vae.safetensors"


@pytest.mark.skipif(not os.path.exists(_VAE), reason="Wan VAE weights not mounted")
def test_real_wan_decoder_loads_and_decodes():
    from yunshu_engine.wan_vae import load_wan_decoder, unpatchify_chw

    dec = load_wan_decoder(_VAE)  # 108 keys, exact map (no missing/extra)
    z = mx.array(np.random.default_rng(0).standard_normal((1, 1, 16, 16, 48)).astype(np.float32))
    out = dec(z)
    img = unpatchify_chw(out, 2)
    mx.eval(img)
    a = np.asarray(img)
    assert img.shape == (1, 1, 256, 256, 3)  # 16-latent * 8(VAE) * 2(patch) = 256, RGB
    assert not np.isnan(a).any()
    assert float(a.min()) > -3.0 and float(a.max()) < 3.0  # VAE output ~[-1,1]


@pytest.mark.skipif(not os.path.exists(_VAE), reason="Wan VAE weights not mounted")
def test_real_wan_encoder_loads_and_encodes():
    from yunshu_engine.wan_vae import load_wan_encoder, patchify_chw

    enc = load_wan_encoder(_VAE)  # 84 keys, exact map
    img = mx.array(np.random.default_rng(1).standard_normal((1, 1, 256, 256, 3)).astype(np.float32))
    xp = patchify_chw(img, 2)
    assert xp.shape == (1, 1, 128, 128, 12)
    params = enc(xp)
    mx.eval(params)
    # 256 -> 16 (8x VAE * 2x patch), 96 = z_dim*2 (mu+logvar)
    assert params.shape == (1, 1, 16, 16, 96)
    assert not np.isnan(np.asarray(params)).any()


@pytest.mark.skipif(not os.path.exists(_VAE), reason="Wan VAE weights not mounted")
def test_real_wan_vae_encode_decode_roundtrip():
    from yunshu_engine.wan_vae import (
        load_wan_decoder,
        load_wan_encoder,
        patchify_chw,
        unpatchify_chw,
    )

    enc = load_wan_encoder(_VAE)
    dec = load_wan_decoder(_VAE)
    img = mx.array(np.random.default_rng(2).standard_normal((1, 1, 256, 256, 3)).astype(np.float32))
    params = enc(patchify_chw(img, 2))
    mu = params[..., :48]  # take the mean latent (drop logvar)
    recon = unpatchify_chw(dec(mu), 2)
    mx.eval(recon)
    assert recon.shape == (1, 1, 256, 256, 3)  # full VAE roundtrip shape preserved
    assert not np.isnan(np.asarray(recon)).any()


@pytest.mark.skipif(not os.path.exists(_VAE), reason="Wan VAE weights not mounted")
def test_real_wan_vae_roundtrip_preserves_sharp_structure():
    # The VAE must reconstruct a HIGH-FREQUENCY image (sharp stripes) with low
    # error — proving the port preserves spatial detail, not just shape/finiteness.
    # This isolates the VAE from the LLM denoiser: if t2i output is blurry but this
    # passes, the blur is in the flow-matching velocity, NOT the codec.
    from yunshu_engine.wan_vae import load_wan_vae

    vae = load_wan_vae(_VAE)
    h = w = 256
    a = np.zeros((h, w, 3), np.float32)
    for i in range(0, h, 16):
        a[i:i + 8] += [1, 0, 0]
    for j in range(0, w, 32):
        a[:, j:j + 16] += [0, 0, 1]
    a = np.clip(a, 0, 1) * 2 - 1
    img = mx.array(a)[None, None]
    recon = vae.decode(vae.encode(img))
    mx.eval(recon)
    err = float(mx.abs(recon - img).mean())
    assert err < 0.15, f"VAE roundtrip not sharp (L1 {err:.3f}) — codec regression"


def test_patchify_unpatchify_inverse():
    from yunshu_engine.wan_vae import patchify_chw, unpatchify_chw
    x = mx.array(np.random.default_rng(3).standard_normal((1, 1, 8, 8, 3)).astype(np.float32))
    rt = unpatchify_chw(patchify_chw(x, 2), 2)
    mx.eval(rt)
    assert rt.shape == x.shape
    assert np.allclose(np.asarray(x), np.asarray(rt), atol=1e-6)  # exact inverse


@pytest.mark.skipif(not os.path.exists(_VAE), reason="Wan VAE weights not mounted")
def test_real_wan_vae_full_module():
    from yunshu_engine.wan_vae import load_wan_vae

    vae = load_wan_vae(_VAE)
    img = mx.array(np.random.default_rng(0).random((1, 1, 256, 256, 3)).astype(np.float32) * 2 - 1)
    z = vae.encode(img)
    recon = vae.decode(z)
    mx.eval(z, recon)
    assert z.shape == (1, 1, 16, 16, 48)  # normalized mean latent
    zz = np.asarray(z)
    assert abs(float(zz.mean())) < 0.5 and 0.3 < float(zz.std()) < 2.0  # ~normalized
    a = np.asarray(recon)
    assert recon.shape == (1, 1, 256, 256, 3)
    assert float(a.min()) >= -1.001 and float(a.max()) <= 1.001  # clamped [-1,1]
    assert not np.isnan(a).any()
